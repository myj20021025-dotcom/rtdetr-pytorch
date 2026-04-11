import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='output dir')
    args = parser.parse_args()

    out_dir = Path(args.dir)
    pr_file = out_dir / 'pr_curve.npz'
    if not pr_file.exists():
        raise FileNotFoundError(f'pr file not found: {pr_file}')

    data = np.load(pr_file)
    recall = data['recall']
    precision = data['precision']

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label='all classes')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()

    save_path = out_dir / 'PR_curve.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'saved to: {save_path}')


if __name__ == '__main__':
    main()
