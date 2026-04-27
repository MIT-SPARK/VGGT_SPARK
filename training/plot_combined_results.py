"""
Plot floor-seg IoU vs layer index for VGGT and Depth Anything 3 on the same axes.
Bars are overlaid (same x positions) with alpha transparency.
A star marks the best layer for each model.
Both models use 1-based layer indexing for display.

VGGT backbone token (key 24) is excluded.

Usage:
    python3 training/plot_combined_results.py \\
        --vggt_results  out/floor_seg_ablation/eval_only_results.json \\
        --da_results    /path/to/DA3/ablation_floor_seg/eval_only_results.json

    python3 training/plot_combined_results.py ... --output combined.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from scipy.interpolate import make_interp_spline

mpl.rcParams["font.family"] = "Times New Roman"

FONT_SIZE = 26
TICK_SIZE = 24
BACKBONE_IDX = 24  # VGGT pre-AA DINOv2 backbone — excluded from plot


def load_iou(path: Path):
    """Return (iou_map {int->float}, iou_label, split_tag) from a results JSON."""
    with open(path) as f:
        data = json.load(f)
    if "test_iou" in data:
        raw = data["test_iou"]
        return {int(k): v for k, v in raw.items()}, "Test IoU", data.get("test_split", "test")
    elif "best_val_iou" in data:
        raw = data["best_val_iou"]
        return {int(k): v for k, v in raw.items()}, "Best Val IoU", "val"
    raise KeyError(f"No 'test_iou' or 'best_val_iou' in {path}. Keys: {list(data.keys())}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare VGGT and DA3 floor-seg layer ablation on one plot")
    parser.add_argument("--vggt_results", type=str,
                        default="./out/floor_seg_ablation/eval_only_results.json")
    parser.add_argument("--da_results",   type=str,
                        default="/home/dominic/Documents/vggt/Depth-Anything-3/"
                                "ablation_floor_seg/eval_only_results.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG (default: combined_iou_vs_layer.png next to --vggt_results)")
    return parser.parse_args()


def main():
    args = parse_args()

    vggt_map, vggt_label, vggt_split = load_iou(Path(args.vggt_results))
    da_map,   da_label,   da_split   = load_iou(Path(args.da_results))

    # Strip backbone from VGGT
    vggt_map = {k: v for k, v in vggt_map.items() if k != BACKBONE_IDX}

    n = max(max(vggt_map), max(da_map)) + 1  # number of layers (0-based internally)

    vggt_best = max(vggt_map, key=vggt_map.get)
    da_best   = max(da_map,   key=da_map.get)

    # 1-based display positions
    display = np.arange(1, n + 1)
    vggt_vals = np.array([vggt_map.get(l, float("nan")) for l in range(n)])
    da_vals   = np.array([da_map.get(l,   float("nan")) for l in range(n)])

    all_iou = [v for v in np.concatenate([vggt_vals, da_vals]) if not np.isnan(v)]
    ylim = (min(0.4, max(0.0, min(all_iou) - 0.05)), 1.0)
    star_offset = (ylim[1] - ylim[0]) * 0.025

    bar_w = 0.8
    fig_w = max(10, n * 0.55 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    VGGT_COLOR = "#2166AC"   # deep cobalt blue
    DA_COLOR   = "#D6604D"   # warm coral red

    ax.bar(display, vggt_vals, bar_w, color=VGGT_COLOR, alpha=0.6, label="VGGT")
    ax.bar(display, da_vals,   bar_w, color=DA_COLOR,   alpha=0.6, label="DA3")

    # Spline through bar tops (drop NaNs before fitting)
    def fit_spline(xs, ys):
        mask = ~np.isnan(ys)
        x, y = xs[mask], ys[mask]
        spl  = make_interp_spline(x, y, k=3)
        x_dense = np.linspace(x[0], x[-1], 400)
        return x_dense, spl(x_dense)

    vx, vy = fit_spline(display, vggt_vals)
    dx, dy = fit_spline(display, da_vals)
    ax.plot(vx, vy, color=VGGT_COLOR, linewidth=2.5, zorder=4)
    ax.plot(dx, dy, color=DA_COLOR,   linewidth=2.5, zorder=4)

    # Stars above best bars
    vggt_star_x = vggt_best + 1
    da_star_x   = da_best   + 1
    ax.plot(vggt_star_x, vggt_map[vggt_best] + star_offset, "*",
            color=VGGT_COLOR, markersize=18, zorder=5)
    ax.plot(da_star_x,   da_map[da_best]     + star_offset, "*",
            color=DA_COLOR, markersize=18, zorder=5)

    ax.legend(fontsize=FONT_SIZE - 4, loc="lower right")
    ax.set_ylim(*ylim)
    iou_label = vggt_label if vggt_label == da_label else "IoU"
    ax.set_ylabel(iou_label, fontsize=FONT_SIZE)
    ax.set_xlabel("ViT Layer Index", fontsize=FONT_SIZE)

    tick_step = max(1, n // 20)
    ax.set_xticks(display[::tick_step])
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    # ax.grid(axis="y", alpha=0.95)
    # Piecewise-linear y scale: 25% of height for [ymin, 0.8], 75% for [0.8, 1.0]
    SPLIT       = 0.8
    EXPAND_FRAC = 0.7
    ymin_val, ymax_val = ylim

    def _fwd(y):
        cf  = 1.0 - EXPAND_FRAC
        slo = cf            / (SPLIT     - ymin_val)
        shi = EXPAND_FRAC   / (ymax_val  - SPLIT)
        return np.where(y <= SPLIT,
                        slo * (y - ymin_val),
                        cf  + shi * (y - SPLIT))

    def _inv(z):
        cf  = 1.0 - EXPAND_FRAC
        slo = cf            / (SPLIT     - ymin_val)
        shi = EXPAND_FRAC   / (ymax_val  - SPLIT)
        return np.where(z <= cf,
                        ymin_val + z / slo,
                        SPLIT    + (z - cf) / shi)

    ax.set_yscale("function", functions=(_fwd, _inv))
    ax.set_ylim(*ylim)

    # Denser ticks in the upper (expanded) region
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])

    ax.spines["bottom"].set_position(("data", ylim[0]))
    ax.grid(axis="y", color="black", alpha=0.3)

    fig.tight_layout()

    output_path = args.output or str(Path(args.vggt_results).parent / "combined_iou_vs_layer.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
