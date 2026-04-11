import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOT_KEYS = [
    'train/giou_loss',
    'train/cls_loss',
    'train/l1_loss',
    'metrics/precision(B)',
    'metrics/recall(B)',
    'metrics/FPS',
    'val/giou_loss',
    'val/cls_loss',
    'val/l1_loss',
    'metrics/mAP50(B)',
    'metrics/mAP50-95(B)',
    'model/FLOPs(G)',
    'model/Params(M)',
]


def smooth(y, window=5):
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    y_pad = np.pad(y, (pad_left, pad_right), mode='edge')
    return np.convolve(y_pad, kernel, mode='valid')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='output dir')
    args = parser.parse_args()

    out_dir = Path(args.dir)
    log_file = out_dir / 'log.txt'
    if not log_file.exists():
        raise FileNotFoundError(f'log file not found: {log_file}')

    rows = []
    with log_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    epochs = [row['epoch'] for row in rows]

    n_plots = len(PLOT_KEYS)
    ncols = 5
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, key in zip(axes, PLOT_KEYS):
        y = np.array([row.get(key, np.nan) for row in rows], dtype=float)
        valid = np.isfinite(y)

        ax.plot(epochs, y, 'o-', linewidth=1.5, markersize=3, label='results')

        if valid.any():
            y_fill = y.copy()
            if not valid.all():
                valid_idx = np.where(valid)[0]
                invalid_idx = np.where(~valid)[0]
                y_fill[invalid_idx] = np.interp(invalid_idx, valid_idx, y[valid])
            ax.plot(epochs, smooth(y_fill, window=5), ':', linewidth=2, label='smooth')

        ax.set_title(key)
        ax.grid(True, linestyle='--', alpha=0.4)

    for ax in axes[len(PLOT_KEYS):]:
        ax.axis('off')

    axes[1].legend()

    plt.tight_layout()
    save_path = out_dir / 'results.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'saved to: {save_path}')


if __name__ == '__main__':
    main()
