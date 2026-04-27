"""
Plot IoU vs VGGT layer index from an ablation results JSON.

Works with both output types from train_floor_seg_ablation.py:
  - eval_only_results.json   (key: "test_iou")
  - ablation_results.json    (key: "best_val_iou")

Usage:
    python3 training/plot_results.py
    python3 training/plot_results.py --results out/floor_seg_ablation/eval_only_results.json
    python3 training/plot_results.py --results results.json --output my_plot.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

mpl.rcParams["font.family"] = "Times New Roman"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=str,
        default="./out/floor_seg_ablation/eval_only_results.json",
        help="Path to results JSON produced by train_floor_seg_ablation.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (default: same dir as results JSON)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = Path(args.results)

    with open(results_path) as f:
        data = json.load(f)

    if "test_iou" in data:
        raw_iou  = data["test_iou"]
        iou_label = "Test IoU"
        split_tag = data.get("test_split", "test")
    elif "best_val_iou" in data:
        raw_iou   = data["best_val_iou"]
        iou_label = "Best Val IoU"
        split_tag = "val"
    else:
        raise KeyError(f"No 'test_iou' or 'best_val_iou' key found. Keys: {list(data.keys())}")

    iou_map     = {int(k): v for k, v in raw_iou.items() if k != "24"} # skip the "24" key which is really the direct output from the DINO backbone before alternating attention
    layers      = sorted(iou_map.keys())
    iou_values  = [iou_map[l] for l in layers]
    best_layer  = max(iou_map, key=iou_map.get)

    FONT_SIZE = 22
    TICK_SIZE = 20

    bar_w  = max(1, layers[1] - layers[0]) * 0.8 if len(layers) > 1 else 1
    colors = ["tomato" if l == best_layer else "steelblue" for l in layers]

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.45 + 2), 5))
    ax.bar(layers, iou_values, color=colors, width=bar_w)

    ax.legend(
        handles=[Patch(color="tomato", label=f"Best layer (layer {best_layer})")],
        fontsize=FONT_SIZE - 2,
        loc="lower right",
    )

    iou_min = max(0.0, min(iou_values) - 0.05)
    iou_min = 0.3
    ax.set_ylim(iou_min, 1.0)

    ax.set_xlabel("ViT Layer Index", fontsize=FONT_SIZE)
    ax.set_ylabel(iou_label, fontsize=FONT_SIZE)
    ax.set_xticks(layers[::max(1, len(layers) // 20)])
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(axis="y", alpha=0.95)

    plt.tight_layout()

    output_path = args.output or str(results_path.parent / "test_iou_vs_layer.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
