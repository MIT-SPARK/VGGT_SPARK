# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FloorSegHead(nn.Module):
    """
    Lightweight single-scale floor segmentation head.

    Reads only from aggregated_tokens_list[11] (the 12th alternating attention
    block output, which corresponds to scale-1 in the DPT head). Predicts a
    single-channel binary mask (floor vs. not-floor) as raw logits.

    Args:
        dim_in (int): Input token dimension. Default 2048 (2 * embed_dim for VGGT-1B).
        patch_size (int): Patch size used by the ViT backbone. Default 14.
        layer_idx (int): Which layer in aggregated_tokens_list to read from. Default 11.
    """

    def __init__(self, dim_in: int = 2048, patch_size: int = 14, layer_idx: int = 11):
        super().__init__()
        self.patch_size = patch_size
        self.layer_idx = layer_idx

        self.norm = nn.LayerNorm(dim_in)

        self.decoder = nn.Sequential(
            # 1x1 bottleneck
            nn.Conv2d(dim_in, 256, kernel_size=1),
            nn.GELU(),
            # 3x3 refinement
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GELU(),
            # 2x upsample: ph -> 2ph
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            # 2x upsample: 2ph -> 4ph
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)

    def _apply(self, fn):
        # PyTorch uses _apply() (not to()) when recursing through child modules.
        # Intercept here so that dtype conversions (e.g. model.to(bfloat16)) leave
        # this module's parameters in float32 while still allowing device moves.
        probe = fn(torch.empty(1, dtype=torch.float32))
        if probe.dtype == torch.float32:
            # Not a dtype change (e.g. device move only) — apply normally.
            return super()._apply(fn)
        # fn would change dtype. Wrap it to keep float32 params in float32.
        def _keep_float32(t):
            result = fn(t)
            return result.float() if t.dtype == torch.float32 else result
        return super()._apply(_keep_float32)

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
    ) -> torch.Tensor:
        """
        Args:
            aggregated_tokens_list: List of [B, S, P, dim] tensors from aggregator.
            images: [B, S, 3, H, W] input images (used for spatial dims and upsample target).
            patch_start_idx: index at which patch tokens start in the P dimension.

        Returns:
            logits: [B, S, H, W] raw (pre-sigmoid) floor segmentation logits.
        """
        B, S, _, H, W = images.shape
        ph, pw = H // self.patch_size, W // self.patch_size

        # Extract tokens from the chosen layer: [B, S, N, dim]
        x = aggregated_tokens_list[self.layer_idx][:, :, patch_start_idx:]

        # Flatten batch and sequence dims
        x = x.reshape(B * S, -1, x.shape[-1])  # [B*S, N, dim]

        # Cast tokens to float32. The seg head's weights always stay float32
        # (see to() override), so everything is consistent from here on.
        x = x.float()
        x = self.norm(x)

        # Reshape to spatial feature map
        x = x.permute(0, 2, 1).reshape(B * S, x.shape[-1], ph, pw)  # [B*S, dim, ph, pw]

        # Decode with 4x upsample (ph -> 4ph, pw -> 4pw)
        x = self.decoder(x)  # [B*S, 64, 4ph, 4pw]

        # Upsample to original image resolution
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)  # [B*S, 64, H, W]

        # Final 1x1 conv to logits
        x = self.output_conv(x)  # [B*S, 1, H, W]

        # Remove channel dim and restore batch/sequence dims
        x = x.squeeze(1).reshape(B, S, H, W)  # [B, S, H, W]

        return x
