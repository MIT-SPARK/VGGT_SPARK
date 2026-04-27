"""
Floor segmentation deployment script.

Loads a trained FloorSegHead checkpoint on top of the frozen VGGT backbone,
processes a folder of images in sequential windows of n frames, and writes
visualizations of the predicted floor mask to an output directory.

Usage:
    python3 deploy_floor_seg.py \\
        --image_dir examples/room/images \\
        --checkpoint checkpoints/floor_seg/best.pt \\
        --out_dir out/floor_seg \\
        --n 4

Output per batch:
    <out_dir>/batch_0000.png               — side-by-side grid: original | overlay | heatmap
    <out_dir>/segmented/<stem>_seg.png     — per-image floor overlay (always saved)
    <out_dir>/individual/<stem>_overlay.png — per-image overlay (with --save_individual)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms as TF

from vggt.models.vggt import VGGT
from vggt.heads.floor_seg_head import FloorSegHead


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def collect_images(image_dir: str) -> list:
    p = Path(image_dir)
    if not p.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")
    return paths


def load_images_for_model(image_paths: list, img_size: int = 518) -> torch.Tensor:
    """Load images and resize to (img_size, img_size), return [N, 3, H, W] float32 in [0,1]."""
    to_tensor = TF.ToTensor()
    resize = TF.Resize((img_size, img_size), antialias=True)
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        t = resize(to_tensor(img))
        imgs.append(t)
    return torch.stack(imgs)   # [N, 3, H, W]


def load_image_original(path) -> np.ndarray:
    """Load an image at its native resolution as HxWx3 uint8 numpy array."""
    return np.array(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(checkpoint_path: str, device: torch.device):
    """Load VGGT backbone (frozen) + FloorSegHead from checkpoint."""
    print("Loading VGGT backbone...")
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device)
    model.to(torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for attr in ("camera_head", "depth_head", "point_head", "track_head"):
        setattr(model, attr, None)
    torch.cuda.empty_cache()

    embed_dim = model.aggregator.camera_token.shape[-1]  # 1024

    print(f"Loading seg head checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    layer_idx = ckpt.get("layer_idx", 10)
    dim_in    = ckpt.get("dim_in", 2 * embed_dim)
    _default_source = "both" if dim_in == 2 * embed_dim else "frame"
    token_source = ckpt.get("token_source", _default_source)
    print(f"  embed_dim={embed_dim}  dim_in={dim_in}  layer_idx={layer_idx}  "
          f"token_source={token_source}")

    seg_head = FloorSegHead(dim_in=dim_in, patch_size=14, layer_idx=layer_idx,
                            token_source=token_source).to(device)
    seg_head.load_state_dict(ckpt["seg_head_state_dict"])
    seg_head.eval()

    return model, seg_head, layer_idx


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_batch(model, seg_head, images_tensor: torch.Tensor,
              layer_idx: int, device: torch.device) -> np.ndarray:
    """
    Run inference on a batch of images.

    Args:
        images_tensor: [N, 3, H, W] float32 in [0, 1]

    Returns:
        probs: [N, H, W] float32 probability maps in [0, 1]
    """
    # VGGT expects [B, S, 3, H, W]; here B=1, S=N
    x = images_tensor.unsqueeze(0).to(device, dtype=torch.bfloat16)  # [1, N, 3, H, W]

    agg_tokens, patch_start_idx, _, _ = model.aggregator(x)

    # Build sparse list — FloorSegHead handles token_source slicing internally
    sparse = [None] * len(agg_tokens)
    sparse[layer_idx] = agg_tokens[layer_idx]

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits = seg_head(sparse, x, patch_start_idx)  # [1, N, H, W]

    probs = torch.sigmoid(logits.float()).squeeze(0)  # [N, H, W]
    return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

OVERLAY_COLOR = np.array([0, 200, 80], dtype=np.float32)   # green


def overlay_mask(image_np: np.ndarray, prob_map: np.ndarray,
                 threshold: float = 0.5, alpha: float = 0.45) -> np.ndarray:
    """Blend a probability map onto an RGB image."""
    H, W = image_np.shape[:2]
    prob_up = np.array(
        Image.fromarray((prob_map * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    mask = prob_up >= threshold
    out = image_np.astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * OVERLAY_COLOR
    return np.clip(out, 0, 255).astype(np.uint8)


def save_segmented_image(blended: np.ndarray, out_path):
    """Save a single overlay image with a 'Ground' legend."""
    fig, ax = plt.subplots(figsize=(blended.shape[1] / 100, blended.shape[0] / 100))
    ax.imshow(blended)
    patch = mpatches.Patch(color=OVERLAY_COLOR / 255, label="Ground")
    ax.legend(handles=[patch], loc="lower right", fontsize=50,
              handlelength=2.5, handleheight=2.0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def make_batch_figure(
    image_paths: list,
    orig_images: list,
    prob_maps: np.ndarray,
    threshold: float,
    alpha: float,
    batch_idx: int,
) -> plt.Figure:
    """Create a matplotlib figure with one row per image: original | overlay | heatmap."""
    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.5 * n))

    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Floor Overlay", "Probability"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=8)

    for i, (path, orig, prob) in enumerate(zip(image_paths, orig_images, prob_maps)):
        blended = overlay_mask(orig, prob, threshold, alpha)

        axes[i, 0].imshow(orig)
        axes[i, 0].set_ylabel(Path(path).name, fontsize=8, rotation=0,
                               labelpad=120, va="center")

        axes[i, 1].imshow(blended)
        patch = mpatches.Patch(color=OVERLAY_COLOR / 255, alpha=0.7, label="floor")
        axes[i, 1].legend(handles=[patch], loc="lower right", fontsize=8)

        H, W = orig.shape[:2]
        prob_up = np.array(
            Image.fromarray((prob * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        im = axes[i, 2].imshow(prob_up, cmap="plasma", vmin=0, vmax=1)
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle(f"Batch {batch_idx:04d}  —  frames {batch_idx * len(prob_maps)} "
                 f"to {batch_idx * len(prob_maps) + n - 1}",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Floor segmentation inference & visualization")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/floor_seg/best.pt",
                        help="Path to trained seg_head checkpoint")
    parser.add_argument("--out_dir", type=str, default="./out/floor_seg",
                        help="Output directory for visualizations")
    parser.add_argument("--n", type=int, default=4,
                        help="Number of images per VGGT forward pass (sequence length)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Frames to advance between batches (default: same as --n, non-overlapping)")
    parser.add_argument("--img_size", type=int, default=518,
                        help="Image size fed to the model")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Sigmoid threshold for binary floor mask")
    parser.add_argument("--alpha", type=float, default=0.45,
                        help="Overlay opacity (0=transparent, 1=solid)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda / cpu). Defaults to cuda if available.")
    parser.add_argument("--save_individual", action="store_true",
                        help="Also save each frame's plain overlay as a separate PNG")
    return parser.parse_args()


def main():
    args = parse_args()
    stride = args.stride if args.stride is not None else args.n

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = out_dir / "segmented"
    seg_dir.mkdir(exist_ok=True)
    if args.save_individual:
        (out_dir / "individual").mkdir(exist_ok=True)

    all_paths = collect_images(args.image_dir)
    print(f"Found {len(all_paths)} images in {args.image_dir}")

    model, seg_head, layer_idx = build_model(args.checkpoint, device)

    starts = list(range(0, len(all_paths), stride))
    print(f"Processing {len(starts)} batch(es) of {args.n} frames (stride={stride})\n")

    for batch_idx, start in enumerate(starts):
        batch_paths = all_paths[start: start + args.n]

        if len(batch_paths) < args.n:
            batch_paths = batch_paths + [batch_paths[-1]] * (args.n - len(batch_paths))
            padded = True
        else:
            padded = False

        names = [p.name for p in batch_paths]
        print(f"  Batch {batch_idx:04d}: {names}" + (" [padded]" if padded else ""))

        images_tensor = load_images_for_model(batch_paths, args.img_size)
        prob_maps = run_batch(model, seg_head, images_tensor, layer_idx, device)
        orig_images = [load_image_original(p) for p in batch_paths]

        fig = make_batch_figure(
            batch_paths, orig_images, prob_maps,
            threshold=args.threshold,
            alpha=args.alpha,
            batch_idx=batch_idx,
        )
        batch_path = out_dir / f"batch_{batch_idx:04d}.png"
        fig.savefig(batch_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {batch_path}")

        real_count = len(all_paths[start: start + args.n])
        for i in range(real_count):
            blended = overlay_mask(orig_images[i], prob_maps[i], args.threshold, args.alpha)
            seg_path = seg_dir / f"{batch_paths[i].stem}_seg.png"
            save_segmented_image(blended, seg_path)
        print(f"    Saved {real_count} segmented image(s) to {seg_dir}/")

        if args.save_individual:
            for i in range(real_count):
                blended = overlay_mask(orig_images[i], prob_maps[i], args.threshold, args.alpha)
                out_path = out_dir / "individual" / f"{batch_paths[i].stem}_overlay.png"
                Image.fromarray(blended).save(out_path)

    print(f"\nDone. Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
