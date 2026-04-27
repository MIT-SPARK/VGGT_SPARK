"""
Layer ablation for FloorSegHead.

Trains one FloorSegHead per LAYER_IDX (0..23) on a frozen VGGT backbone,
saves the best val-IoU checkpoint per layer, evaluates all on the test split,
logs a JSON summary, and saves a plot of test IoU vs layer index.

The VGGT backbone is loaded ONCE and shared across all 24 training runs.

Usage:
    python3 training/ablation_layer_idx.py \
        --dataset_root /home/dominic/Documents/floor_label/floor_dataset

    # Only train/evaluate specific layers (e.g. for partial / parallel runs):
    python3 training/ablation_layer_idx.py --layers 0 4 8 12 16 20

    # Skip training, only evaluate existing checkpoints + plot:
    python3 training/ablation_layer_idx.py --eval_only

    # Re-train layers even if their checkpoint already exists:
    python3 training/ablation_layer_idx.py --force_retrain
"""

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vggt.models.vggt import VGGT
from vggt.heads.floor_seg_head import FloorSegHead
from training.data.datasets.floor_dataset import YOLOFloorDataset


# ---------------------------------------------------------------------------
# Utilities (adapted from train_floor_seg.py)
# ---------------------------------------------------------------------------

def extract_needed_tokens(agg_tokens_list, layer_idx: int):
    """Return a sparse list with only layer_idx populated; free everything else."""
    needed = agg_tokens_list[layer_idx]
    sparse = [None] * len(agg_tokens_list)
    sparse[layer_idx] = needed
    return sparse


def bce_dice_loss(logits, targets, bce_weight=0.5, pos_weight=3.0):
    pw = torch.tensor([pos_weight], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
    probs = torch.sigmoid(logits)
    smooth = 1e-6
    intersection = (probs * targets).sum()
    dice = 1 - (2 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    return bce_weight * bce + (1 - bce_weight) * dice


def compute_iou_dice(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) >= threshold).float()
    targets = targets.float()
    smooth = 1e-6
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return iou.item(), dice.item()


def build_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_best(path, seg_head, iou):
    torch.save({"seg_head_state_dict": seg_head.state_dict(), "best_iou": iou}, path)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, seg_head, loader, device, layer_idx):
    """Return (mean_iou, mean_dice) over the given loader."""
    seg_head.eval()
    total_iou = total_dice = n = 0.0
    for batch in loader:
        images_bf16 = batch["images"].to(device, dtype=torch.bfloat16)
        masks = batch["masks"].to(device)
        agg_tokens, patch_start_idx, _, _ = model.aggregator(images_bf16)
        agg_tokens = extract_needed_tokens(agg_tokens, layer_idx)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = seg_head(agg_tokens, images_bf16, patch_start_idx)
        iou, dice = compute_iou_dice(logits.float(), masks)
        total_iou += iou
        total_dice += dice
        n += 1
    seg_head.train()
    return (total_iou / n, total_dice / n) if n > 0 else (0.0, 0.0)


# ---------------------------------------------------------------------------
# Train one layer
# ---------------------------------------------------------------------------

def train_one_layer(model, layer_idx, args, device, embed_dim):
    """
    Train a FloorSegHead for `layer_idx`, save best checkpoint, return best val IoU.
    Skips training if checkpoint already exists and --force_retrain is not set.
    """
    ckpt_dir = Path(args.checkpoint_dir) / f"layer_{layer_idx:02d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    if best_path.exists() and not args.force_retrain:
        ckpt = torch.load(best_path, map_location="cpu")
        best_iou = ckpt.get("best_iou", 0.0)
        print(f"[Layer {layer_idx:2d}] Checkpoint exists (val IoU={best_iou:.4f}), skipping.")
        return best_iou

    print(f"\n{'='*64}")
    print(f"  Training  LAYER_IDX = {layer_idx}  ({layer_idx+1}/24)")
    print(f"{'='*64}")

    seg_head = FloorSegHead(dim_in=2 * embed_dim, patch_size=14, layer_idx=layer_idx).to(device)
    optimizer = torch.optim.AdamW(seg_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dataset = YOLOFloorDataset(
        args.dataset_root, split="train", seq_len=args.seq_len, img_size=args.img_size
    )
    val_dataset = YOLOFloorDataset(
        args.dataset_root, split="val", seq_len=args.seq_len, img_size=args.img_size
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)

    best_iou = 0.0

    for epoch in range(args.epochs):
        seg_head.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            images_bf16 = batch["images"].to(device, dtype=torch.bfloat16)
            masks = batch["masks"].to(device)

            with torch.no_grad():
                agg_tokens, patch_start_idx, _, _ = model.aggregator(images_bf16)
                agg_tokens = extract_needed_tokens(agg_tokens, layer_idx)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = seg_head(agg_tokens, images_bf16, patch_start_idx)

            loss = bce_dice_loss(logits.float(), masks, args.bce_weight, args.pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seg_head.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, steps_per_epoch)
        val_iou, val_dice = evaluate(model, seg_head, val_loader, device, layer_idx)
        elapsed = time.time() - t0

        print(
            f"  [Layer {layer_idx:2d}] Epoch {epoch+1:3d}/{args.epochs}  "
            f"loss={avg_loss:.4f}  val_IoU={val_iou:.4f}  val_Dice={val_dice:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            save_best(str(best_path), seg_head, best_iou)
            print(f"    -> New best val IoU: {best_iou:.4f}  (saved {best_path})")

    print(f"[Layer {layer_idx:2d}] Done. Best val IoU: {best_iou:.4f}")

    del seg_head, optimizer, scheduler, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return best_iou


# ---------------------------------------------------------------------------
# Evaluate all saved checkpoints on the test split
# ---------------------------------------------------------------------------

def evaluate_all(model, layers, args, device, embed_dim):
    print(f"\n{'='*64}")
    print(f"  Evaluating {len(layers)} layer(s) on '{args.test_split}' split")
    print(f"{'='*64}")

    test_dataset = YOLOFloorDataset(
        args.dataset_root, split=args.test_split, seq_len=args.seq_len, img_size=args.img_size
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    results = {}
    for layer_idx in layers:
        best_path = Path(args.checkpoint_dir) / f"layer_{layer_idx:02d}" / "best.pt"
        if not best_path.exists():
            print(f"  [Layer {layer_idx:2d}] No checkpoint at {best_path}, skipping.")
            continue

        seg_head = FloorSegHead(dim_in=2 * embed_dim, patch_size=14, layer_idx=layer_idx).to(device)
        ckpt = torch.load(best_path, map_location=device)
        seg_head.load_state_dict(ckpt["seg_head_state_dict"])
        val_iou_saved = ckpt.get("best_iou", 0.0)

        test_iou, test_dice = evaluate(model, seg_head, test_loader, device, layer_idx)
        results[layer_idx] = {
            "test_iou": test_iou,
            "test_dice": test_dice,
            "val_iou": val_iou_saved,
        }
        print(
            f"  [Layer {layer_idx:2d}]  test_IoU={test_iou:.4f}  "
            f"test_Dice={test_dice:.4f}  (best_val_IoU={val_iou_saved:.4f})"
        )

        del seg_head
        gc.collect()
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(results, output_path):
    layers = sorted(results.keys())
    test_ious = [results[l]["test_iou"] for l in layers]
    val_ious = [results[l]["val_iou"] for l in layers]

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(layers, test_ious, "o-", color="steelblue", linewidth=2, markersize=7, label="Test IoU")
    ax.plot(layers, val_ious, "s--", color="coral", linewidth=1.5, markersize=5, alpha=0.8, label="Best Val IoU")

    best_idx = test_ious.index(max(test_ious))
    best_layer = layers[best_idx]
    best_val = test_ious[best_idx]
    ax.axvline(best_layer, color="steelblue", linestyle=":", alpha=0.4)
    ax.annotate(
        f"Best: layer {best_layer}\nIoU = {best_val:.4f}",
        xy=(best_layer, best_val),
        xytext=(best_layer + 0.8, best_val - 0.04),
        fontsize=9,
        color="steelblue",
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.2),
    )

    ax.set_xlabel("Layer Index (aggregated_tokens_list[i])", fontsize=12)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_title("FloorSegHead Ablation: Test IoU vs. VGGT Layer Index", fontsize=14)
    ax.set_xticks(layers)
    ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Layer index ablation for FloorSegHead")
    parser.add_argument("--dataset_root", type=str,
                        default="/home/dominic/Documents/floor_label/floor_dataset")
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(24)),
                        help="Layer indices to run (default: 0-23)")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4, help="Per-run batch size")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/ablation_layer")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--pos_weight", type=float, default=3.0)
    parser.add_argument("--results_json", type=str,
                        default="./checkpoints/ablation_layer/results.json")
    parser.add_argument("--plot_path", type=str,
                        default="./checkpoints/ablation_layer/test_iou_vs_layer.png")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; only evaluate existing checkpoints and plot")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Retrain even if a checkpoint already exists for that layer")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    layers = sorted(args.layers)

    # ------------------------------------------------------------------
    # Load backbone ONCE — shared across all training and eval runs
    # ------------------------------------------------------------------
    print("Loading pretrained VGGT backbone (shared across all layers)...")
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for attr in ("camera_head", "depth_head", "point_head", "track_head"):
        setattr(model, attr, None)
    gc.collect()
    torch.cuda.empty_cache()

    embed_dim = model.aggregator.camera_token.shape[-1]  # 1024 for VGGT-1B
    print(f"Backbone loaded. embed_dim={embed_dim}, FloorSegHead dim_in={2*embed_dim}\n")
    print(f"Layers to process: {layers}")

    # ------------------------------------------------------------------
    # Training phase
    # ------------------------------------------------------------------
    val_ious = {}
    if not args.eval_only:
        for layer_idx in layers:
            val_ious[layer_idx] = train_one_layer(model, layer_idx, args, device, embed_dim)

    # ------------------------------------------------------------------
    # Evaluation phase
    # ------------------------------------------------------------------
    results = evaluate_all(model, layers, args, device, embed_dim)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    results_out = {str(k): v for k, v in results.items()}
    with open(args.results_json, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults saved to {args.results_json}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if results:
        make_plot(results, args.plot_path)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n--- Ablation Summary ---")
    print(f"{'Layer':>6}  {'Test IoU':>9}  {'Test Dice':>10}  {'Val IoU':>9}")
    print("-" * 42)
    for layer_idx in sorted(results.keys()):
        r = results[layer_idx]
        print(f"{layer_idx:6d}  {r['test_iou']:9.4f}  {r['test_dice']:10.4f}  {r['val_iou']:9.4f}")

    if results:
        best = max(results.items(), key=lambda x: x[1]["test_iou"])
        print(f"\nBest layer: {best[0]}  (test IoU = {best[1]['test_iou']:.4f})")


if __name__ == "__main__":
    main()
