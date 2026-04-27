"""
Standalone DDP training script for the FloorSegHead on a frozen VGGT backbone.

Usage (single GPU):
    python3 training/train_floor_seg.py \\
        --dataset_root /home/dominic/Documents/floor_label/floor_dataset

Usage (2-GPU — plain python3, no torchrun needed):
    python3 training/train_floor_seg.py --gpus 2 \\
        --dataset_root /home/dominic/Documents/floor_label/floor_dataset
"""

import argparse
import gc
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# Ensure the project root (parent of this file's directory) is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    _TB_AVAILABLE = False

from vggt.models.vggt import VGGT
from vggt.heads.floor_seg_head import FloorSegHead
from training.data.datasets.floor_dataset import YOLOFloorDataset

LAYER_IDX = 10

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def find_free_port() -> int:
    """Pick an ephemeral port that is free at the time of the call."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def extract_needed_tokens(agg_tokens_list, layer_idx: int):
    """
    Return a sparse list where only `layer_idx` is populated.
    All other tensors are freed so the ~3 GB of unused activations
    can be reclaimed before the seg head forward pass.
    """
    needed = agg_tokens_list[layer_idx]
    sparse = [None] * len(agg_tokens_list)
    sparse[layer_idx] = needed
    return sparse


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
    pos_weight: float = 3.0,
) -> torch.Tensor:
    """Combined BCE + Dice loss for binary segmentation."""
    pw = torch.tensor([pos_weight], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)

    probs = torch.sigmoid(logits)
    smooth = 1e-6
    intersection = (probs * targets).sum()
    dice = 1 - (2 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

    return bce_weight * bce + (1 - bce_weight) * dice


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_iou_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.85):
    """Compute IoU and Dice coefficient at a given threshold."""
    preds = (torch.sigmoid(logits) >= threshold).float()
    targets = targets.float()

    smooth = 1e-6
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return iou.item(), dice.item()


# ---------------------------------------------------------------------------
# LR scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_main(rank: int) -> bool:
    return rank == 0


def maybe_print(rank: int, *args, **kwargs):
    if is_main(rank):
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, seg_head: nn.Module, optimizer, scheduler, epoch: int, best_iou: float):
    torch.save(
        {
            "epoch": epoch,
            "best_iou": best_iou,
            "seg_head_state_dict": seg_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str, seg_head: nn.Module, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    seg_head.load_state_dict(ckpt["seg_head_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["best_iou"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, seg_head, val_loader, device, rank, world_size):
    seg_head.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    n_batches = 0

    for batch in val_loader:
        images_bf16 = batch["images"].to(device, dtype=torch.bfloat16)
        masks = batch["masks"].to(device)

        agg_tokens, patch_start_idx, _, _ = model.aggregator(images_bf16)
        agg_tokens = extract_needed_tokens(agg_tokens, layer_idx=LAYER_IDX)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = seg_head(agg_tokens, images_bf16, patch_start_idx)

        logits_f = logits.float()
        loss = bce_dice_loss(logits_f, masks)
        iou, dice = compute_iou_dice(logits_f, masks)

        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        n_batches += 1

    # Aggregate across GPUs
    stats = torch.tensor([total_loss, total_iou, total_dice, n_batches], device=device)
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    total_loss, total_iou, total_dice, n_batches = stats.tolist()
    if n_batches == 0:
        return 0.0, 0.0, 0.0

    seg_head.train()
    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches


# ---------------------------------------------------------------------------
# Training worker (one process per GPU)
# ---------------------------------------------------------------------------

def train_worker(rank: int, world_size: int, args):
    # ---- Process group ----
    import datetime
    use_ddp = world_size > 1
    if use_ddp:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=300),
        )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    maybe_print(rank, f"Using {world_size} GPU(s)  |  rank {rank} → {device}")

    # ---- Checkpoint dir + TensorBoard (rank 0 only) ----
    ckpt_dir = Path(args.checkpoint_dir)
    if is_main(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if _TB_AVAILABLE:
            tb_writer = SummaryWriter(log_dir=args.tb_dir)
            print(f"TensorBoard logs → {args.tb_dir}")
            print(f"  Run: tensorboard --logdir {args.tb_dir}")
        else:
            tb_writer = None
            print("tensorboard not installed — skipping TB logging. Run: pip install tensorboard")
    else:
        tb_writer = None

    # ---- Load pretrained VGGT backbone ----
    # Stagger loading so processes don't all hit RAM simultaneously.
    # Each rank waits rank*15 s before loading; the barrier afterwards
    # ensures all ranks have finished before any rank continues.
    if use_ddp and rank > 0:
        time.sleep(rank * 15)
    maybe_print(rank, "Loading pretrained VGGT backbone...")
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device)
    model.to(torch.bfloat16) 
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Free heads we will never use — saves ~1.4 GB VRAM per GPU.
    for attr in ("camera_head", "depth_head", "point_head", "track_head"):
        setattr(model, attr, None)
    gc.collect()
    torch.cuda.empty_cache()

    if use_ddp:
        dist.barrier()  # wait for all ranks to finish loading

    # ---- Build seg head ----
    embed_dim = model.aggregator.camera_token.shape[-1]  # 1024 for VGGT-1B
    seg_head = FloorSegHead(dim_in=2 * embed_dim, patch_size=14, layer_idx=LAYER_IDX).to(device)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        seg_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ---- Datasets ----
    train_dataset = YOLOFloorDataset(
        dataset_root=args.dataset_root,
        split="train",
        seq_len=args.seq_len,
        img_size=args.img_size,
    )

    val_split_path = Path(args.dataset_root) / args.val_split
    if val_split_path.exists():
        val_dataset = YOLOFloorDataset(
            dataset_root=args.dataset_root,
            split=args.val_split,
            seq_len=args.seq_len,
            img_size=args.img_size,
        )
    else:
        maybe_print(rank, f"Warning: val split not found at {val_split_path}, skipping validation.")
        val_dataset = None

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if val_dataset else None
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    ) if val_dataset else None

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)

    # ---- Resume ----
    start_epoch = 0
    best_iou = 0.0
    if args.resume and Path(args.resume).exists():
        maybe_print(rank, f"Resuming from {args.resume}")
        start_epoch, best_iou = load_checkpoint(args.resume, seg_head, optimizer, scheduler)
        start_epoch += 1

    # ---- DDP-wrap seg head only ----
    if use_ddp:
        seg_head_ddp = DDP(seg_head, device_ids=[rank], output_device=rank)
    else:
        seg_head_ddp = seg_head

    maybe_print(rank, f"Starting training: {args.epochs} epochs, {steps_per_epoch} steps/epoch")

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs):
        seg_head.train()
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            images_bf16 = batch["images"].to(device, dtype=torch.bfloat16)
            masks = batch["masks"].to(device)

            with torch.no_grad():
                agg_tokens, patch_start_idx, _, _ = model.aggregator(images_bf16)
                # Keep only the token layer the seg head needs; free the rest.
                agg_tokens = extract_needed_tokens(agg_tokens, layer_idx=LAYER_IDX)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = seg_head_ddp(agg_tokens, images_bf16, patch_start_idx)

            loss = bce_dice_loss(logits.float(), masks, args.bce_weight, args.pos_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seg_head.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            global_step = epoch * steps_per_epoch + step + 1
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss_step", loss.item(), global_step)
                tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            if is_main(rank) and (step + 1) % 50 == 0:
                avg = epoch_loss / (step + 1)
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch+1}/{args.epochs}  "
                    f"Step {step+1}/{steps_per_epoch}  "
                    f"Loss: {avg:.4f}  LR: {lr_now:.2e}  "
                    f"Time: {elapsed:.1f}s"
                )

        # ---- End of epoch ----
        avg_train_loss = epoch_loss / max(1, steps_per_epoch)
        maybe_print(rank, f"Epoch {epoch+1} train loss: {avg_train_loss:.4f}")
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss_epoch", avg_train_loss, epoch + 1)

        if val_loader is not None:
            val_loss, val_iou, val_dice = validate(
                model, seg_head, val_loader, device, rank, world_size
            )
            maybe_print(rank, f"  Val  loss={val_loss:.4f}  IoU={val_iou:.4f}  Dice={val_dice:.4f}")
            if tb_writer is not None:
                tb_writer.add_scalar("val/loss", val_loss, epoch + 1)
                tb_writer.add_scalar("val/iou", val_iou, epoch + 1)
                tb_writer.add_scalar("val/dice", val_dice, epoch + 1)
                tb_writer.flush()

            if is_main(rank) and val_iou > best_iou:
                best_iou = val_iou
                save_checkpoint(str(ckpt_dir / "best.pt"), seg_head, optimizer, scheduler, epoch, best_iou)
                print(f"  New best IoU: {best_iou:.4f}  (saved best.pt)")

        if is_main(rank) and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                str(ckpt_dir / f"epoch_{epoch+1:04d}.pt"),
                seg_head, optimizer, scheduler, epoch, best_iou,
            )
            print(f"  Saved checkpoint: epoch_{epoch+1:04d}.pt")

    # ---- Final save ----
    if is_main(rank):
        save_checkpoint(str(ckpt_dir / "last.pt"), seg_head, optimizer, scheduler, args.epochs - 1, best_iou)
        print(f"Training complete. Best val IoU: {best_iou:.4f}")
        if tb_writer is not None:
            tb_writer.close()

    if use_ddp:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train FloorSegHead on frozen VGGT backbone")
    parser.add_argument("--dataset_root", type=str,
                        default="/home/dominic/Documents/floor_label/floor_dataset")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs to use. Defaults to all available GPUs.")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/floor_seg")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--pos_weight", type=float, default=3.0)
    parser.add_argument("--tb_dir", type=str, default="./runs/floor_seg")
    return parser.parse_args()


def main():
    args = parse_args()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Already inside a torchrun/torch.distributed.run worker — run directly.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        train_worker(rank, world_size, args)
        return

    # Not a distributed worker yet.
    n_gpus = args.gpus if args.gpus is not None else torch.cuda.device_count()
    n_gpus = max(1, n_gpus)

    if n_gpus == 1:
        train_worker(0, 1, args)
    else:
        # Re-launch this script under torch.distributed.run (same as torchrun).
        # This is far more stable than mp.spawn for CUDA multi-GPU work.
        port = find_free_port()
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={n_gpus}",
            "--master_addr=localhost",
            f"--master_port={port}",
            os.path.abspath(__file__),
        ] + sys.argv[1:]  # forward all original arguments
        print(f"Launching {n_gpus}-GPU training via torch.distributed.run (port {port})")
        result = subprocess.run(cmd)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()