import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms


def polygons_to_mask(label_path: str, w: int, h: int) -> torch.Tensor:
    """
    Convert a YOLO-format polygon label file to a binary mask tensor.

    Each line in the file has the format:
        class_id  x1 y1 x2 y2 ... xn yn   (normalized [0,1] coordinates)

    Args:
        label_path: Path to the YOLO label .txt file.
        w: Mask width in pixels.
        h: Mask height in pixels.

    Returns:
        Binary float32 mask tensor of shape [H, W] with values in {0.0, 1.0}.
    """
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                # Need at least class_id + 3 coordinate pairs (6 values)
                continue
            coords = [float(v) for v in parts[1:]]
            polygon = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
            if len(polygon) >= 3:
                draw.polygon(polygon, fill=1)

    return torch.from_numpy(np.array(mask, dtype=np.float32))


class YOLOFloorDataset(Dataset):
    """
    Dataset that loads images and YOLO-format polygon segmentation labels,
    rasterizes polygons to binary floor/ground masks.

    Expected directory layout:
        dataset_root/
          {split}/
            images/  (*.jpg, *.png, *.jpeg, *.bmp, *.webp)
            labels/  (*.txt, one per image, same stem)

    Images with no corresponding label file produce an all-zero mask
    (valid: means no floor visible in that frame).

    Args:
        dataset_root: Root directory of the dataset.
        split: Dataset split, e.g. 'train' or 'val'.
        seq_len: Number of frames randomly sampled per __getitem__ call.
        img_size: Resize images and masks to (img_size, img_size).
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        seq_len: int = 4,
        img_size: int = 518,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.img_size = img_size

        images_dir = Path(dataset_root) / split / "images"
        labels_dir = Path(dataset_root) / split / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Collect all image paths and their (possibly missing) label paths
        self.samples = []
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in self.IMAGE_EXTS:
                continue
            label_path = labels_dir / (img_path.stem + ".txt")
            self.samples.append((str(img_path), str(label_path) if label_path.exists() else None))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {images_dir}")

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # -> [3, H, W] float32 in [0, 1]
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample(self, idx: int):
        img_path, label_path = self.samples[idx]

        # Load and transform image
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.img_transform(img)  # [3, H, W]

        # Load or create mask
        if label_path is not None:
            mask = polygons_to_mask(label_path, self.img_size, self.img_size)
        else:
            mask = torch.zeros(self.img_size, self.img_size, dtype=torch.float32)

        return img_tensor, mask

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with:
            images: [seq_len, 3, H, W] float32 in [0, 1]
            masks:  [seq_len, H, W]    float32 binary
        """
        # Randomly sample seq_len frames (with replacement if dataset is small)
        n = len(self.samples)
        if n >= self.seq_len:
            indices = random.sample(range(n), self.seq_len)
        else:
            indices = random.choices(range(n), k=self.seq_len)

        images = []
        masks = []
        for i in indices:
            img_tensor, mask = self._load_sample(i)
            images.append(img_tensor)
            masks.append(mask)

        return {
            "images": torch.stack(images),  # [S, 3, H, W]
            "masks": torch.stack(masks),    # [S, H, W]
        }
