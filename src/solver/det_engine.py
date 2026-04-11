"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
import time
from typing import Iterable

import torch
import torch.amp

from src.data import (
    CocoEvaluator,
    compute_ultralytics_prf1,
    mscoco_label2category,
    process_ultralytics_batch,
)
import src.misc.dist as dist_utils
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _sync_device(device):
    if torch.cuda.is_available() and str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


@torch.no_grad()
def profile_model(model: torch.nn.Module, postprocessors, data_loader, device, num_warmup=3, num_runs=10):
    """Profile Params(M), FLOPs(G), and single-image FPS once using the validation pipeline."""
    default_stats = {
        'metrics/FPS': 0.0,
        'model/FLOPs(G)': 0.0,
        'model/Params(M)': sum(p.numel() for p in dist_utils.de_parallel(model).parameters()) / 1e6,
    }

    try:
        samples, targets = next(iter(data_loader))
    except StopIteration:
        return default_stats

    samples = samples[:1].to(device)
    orig_target_sizes = torch.stack([targets[0]["orig_size"]], dim=0).to(device)

    model_was_training = model.training
    post_was_training = getattr(postprocessors, 'training', False)
    model.eval()
    if hasattr(postprocessors, 'eval'):
        postprocessors.eval()

    def forward_once():
        outputs = model(samples)
        return postprocessors(outputs, orig_target_sizes)

    try:
        for _ in range(max(num_warmup, 0)):
            forward_once()

        _sync_device(device)
        start = time.perf_counter()
        for _ in range(max(num_runs, 1)):
            forward_once()
        _sync_device(device)
        elapsed = time.perf_counter() - start
        fps = max(num_runs, 1) / elapsed if elapsed > 0 else 0.0

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available() and str(device).startswith('cuda'):
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            forward_once()
        flops = prof.key_averages().total_average().flops / 1e9

        profile_stats = {
            'metrics/FPS': float(fps),
            'model/FLOPs(G)': float(flops),
            'model/Params(M)': default_stats['model/Params(M)'],
        }
    except Exception as e:
        print(f'Warning: profiling failed, fallback to params only. {e}')
        profile_stats = default_stats
    finally:
        if model_was_training:
            model.train()
        if hasattr(postprocessors, 'train') and post_was_training:
            postprocessors.train()

    gathered = dist_utils.all_gather(profile_stats)
    for stats in gathered:
        if stats is not None:
            return stats
    return profile_stats



# @torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir, profile_stats=None):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    ultralytics_stats = []
    iouv = torch.linspace(0.5, 0.95, 10, device=device)

    panoptic_evaluator = None

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        # 这里恢复 val loss 统计
        loss_dict = criterion(outputs, targets)
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        metric_logger.update(loss=loss_value, **loss_dict_reduced)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)

        if 'bbox' in iou_types:
            for result, target in zip(results, targets):
                labels = target['labels']
                boxes = target['boxes']
                orig_size = target['orig_size'].to(boxes.device, dtype=boxes.dtype)

                # Validation targets are normalized cxcywh after transforms.
                # Convert them back to original-image xyxy so they share the same space as postprocessed predictions.
                boxes = boxes * orig_size.repeat(2)
                cx, cy, w, h = boxes.unbind(dim=1)
                boxes = torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=1)

                if getattr(postprocessors, 'remap_mscoco_category', False) and labels.numel():
                    labels = torch.tensor(
                        [mscoco_label2category[int(x)] for x in labels.detach().cpu().tolist()],
                        device=labels.device,
                        dtype=labels.dtype,
                    )

                detections = torch.cat(
                    (
                        result['boxes'],
                        result['scores'].unsqueeze(1),
                        result['labels'].unsqueeze(1).float(),
                    ),
                    dim=1,
                ) if result['boxes'].numel() else result['boxes'].new_zeros((0, 6))

                gt = torch.cat(
                    (labels.unsqueeze(1).float(), boxes),
                    dim=1,
                ) if boxes.numel() else boxes.new_zeros((0, 5))

                correct = process_ultralytics_batch(detections, gt, iouv)
                ultralytics_stats.append((
                    correct.cpu().numpy(),
                    result['scores'].detach().cpu().numpy(),
                    result['labels'].detach().cpu().numpy(),
                    labels.detach().cpu().numpy(),
                ))

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if profile_stats is None:
        profile_stats = profile_model(model, postprocessors, data_loader, device)

    ultralytics_prf1 = compute_ultralytics_prf1(ultralytics_stats, num_iou=iouv.numel())
    if coco_evaluator is not None:
        coco_evaluator.custom_prf1 = ultralytics_prf1

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # 先把 val loss 全部记下来
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats.update(profile_stats)

    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            bbox_eval = coco_evaluator.coco_eval['bbox']
            stats['coco_eval_bbox'] = bbox_eval.stats.tolist()

            # COCO 常规指标
            stats['metrics/mAP50-95(B)'] = float(bbox_eval.stats[0])
            stats['metrics/mAP50(B)'] = float(bbox_eval.stats[1])

            stats['metrics/precision(B)'] = ultralytics_prf1['precision']
            stats['metrics/recall(B)'] = ultralytics_prf1['recall']
            stats['metrics/F1(B)'] = ultralytics_prf1['f1']

        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    print(
        " Model profile: Params(M) = {:.3f}, FLOPs(G) = {:.3f}, FPS = {:.2f}".format(
            stats.get('model/Params(M)', 0.0),
            stats.get('model/FLOPs(G)', 0.0),
            stats.get('metrics/FPS', 0.0),
        )
    )

    return stats, coco_evaluator
