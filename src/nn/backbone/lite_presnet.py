"""Lightweight RT-DETR backbone enhancements for small UAV objects."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvNormLayer, FrozenBatchNorm2d, get_activation
from .presnet import PResNet

from src.core import register


def channel_shuffle(x, groups):
    b, c, h, w = x.shape
    assert c % groups == 0, "channels must be divisible by groups"
    x = x.reshape(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.reshape(b, c, h, w)


class DSConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, act="silu"):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(
            ch_in, ch_in, kernel_size, stride, padding=padding, groups=ch_in, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(ch_in)
        self.pointwise = nn.Conv2d(ch_in, ch_out, 1, 1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)

    def forward(self, x):
        x = self.act(self.depthwise_bn(self.depthwise(x)))
        x = self.act(self.pointwise_bn(self.pointwise(x)))
        return x


class LinearContextAttention(nn.Module):
    """A lightweight linear-complexity spatial attention block."""

    def __init__(self, channels, attn_ratio=0.5, act="silu"):
        super().__init__()
        inner_dim = max(32, int(channels * attn_ratio))
        self.norm = nn.BatchNorm2d(channels)
        self.query = nn.Conv2d(channels, 1, 1, bias=False)
        self.key = nn.Conv2d(channels, inner_dim, 1, bias=False)
        self.value = nn.Conv2d(channels, inner_dim, 1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(inner_dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = get_activation(act)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        b, _, h, w = x.shape

        query = self.query(x).flatten(2)
        attn = torch.softmax(query, dim=-1)
        key = self.key(x).flatten(2)
        value = self.value(x).flatten(2)

        context = torch.sum(key * attn, dim=-1, keepdim=True)
        out = self.act(value) * context
        out = out.reshape(b, -1, h, w)
        out = self.proj(out)
        return residual + out


class HighFrequencyEdgeEnhancer(nn.Module):
    def __init__(self, channels, act="silu"):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            get_activation(act),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            get_activation(act),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        low_freq = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = x - low_freq
        edge = self.refine(high_freq)
        gate = self.gate(torch.cat([x, high_freq], dim=1))
        return x + edge * gate


class GatedShuffleBlock(nn.Module):
    def __init__(self, channels, expansion=1.0, act="silu"):
        super().__init__()
        hidden_channels = max(32, int(channels * expansion))
        self.expand = ConvNormLayer(channels, hidden_channels * 2, 1, 1, act=act)
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            get_activation(act),
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        residual = x
        x = self.expand(x)
        value, gate = x.chunk(2, dim=1)
        value = self.depthwise(value)
        x = self.project(value * torch.sigmoid(gate))
        return residual + x


class LiteFeatureEnhanceBlock(nn.Module):
    def __init__(self, channels, use_high_freq=True, use_linear_attn=False, expansion=1.0,
                 attn_ratio=0.5, act="silu"):
        super().__init__()
        self.local = DSConvNormLayer(channels, channels, 3, 1, act=act)
        self.hf = HighFrequencyEdgeEnhancer(channels, act=act) if use_high_freq else nn.Identity()
        self.attn = LinearContextAttention(channels, attn_ratio=attn_ratio, act=act) \
            if use_linear_attn else nn.Identity()
        self.glu = GatedShuffleBlock(channels, expansion=expansion, act=act)
        self.fuse = ConvNormLayer(channels * 2, channels, 1, 1, act=act)

    def forward(self, x):
        residual = x
        x = self.local(x)
        x = self.hf(x)
        x = self.attn(x)
        x = self.glu(x)
        x = torch.cat([residual, x], dim=1)
        x = channel_shuffle(x, groups=2)
        return self.fuse(x)


@register
class LitePResNet(nn.Module):
    def __init__(self,
                 depth=18,
                 variant='d',
                 num_stages=4,
                 return_idx=[0, 1, 2, 3],
                 act='relu',
                 freeze_at=-1,
                 freeze_norm=False,
                 pretrained=True,
                 enhance_act='silu',
                 enhance_expansion=1.0,
                 enhance_stages=None,
                 attn_ratio=0.25,
                 high_freq_stages=None,
                 linear_attention_stages=None):
        super().__init__()
        self.backbone = PResNet(
            depth=depth,
            variant=variant,
            num_stages=num_stages,
            return_idx=return_idx,
            act=act,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            pretrained=pretrained,
        )

        self.return_idx = self.backbone.return_idx
        self.out_channels = self.backbone.out_channels
        self.out_strides = self.backbone.out_strides

        enhance_stages = set(enhance_stages or [0, 1, 2])
        high_freq_stages = set(high_freq_stages or [0, 1])
        linear_attention_stages = set(linear_attention_stages or [2])
        self.enhance_blocks = nn.ModuleList([
            LiteFeatureEnhanceBlock(
                channels=channels,
                use_high_freq=feat_idx in high_freq_stages,
                use_linear_attn=feat_idx in linear_attention_stages,
                expansion=enhance_expansion,
                attn_ratio=attn_ratio,
                act=enhance_act,
            )
            if feat_idx in enhance_stages else nn.Identity()
            for feat_idx, channels in enumerate(self.out_channels)
        ])

        if freeze_norm:
            self._freeze_norm(self.enhance_blocks)

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        feats = self.backbone(x)
        return [block(feat) for block, feat in zip(self.enhance_blocks, feats)]
