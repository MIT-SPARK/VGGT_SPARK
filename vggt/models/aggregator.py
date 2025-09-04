# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any
from matplotlib import pyplot as plt

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

from image_retrieval import run_asmk

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

import torch
import matplotlib.pyplot as plt

class CosineAttentionVisualizer:
    def __init__(self, image_height=21, token_offset=5):
        self.image_width = 37
        self.image_height= image_height
        self.token_offset = token_offset  # number of special tokens to skip (e.g., CLS, etc.)
        self.layers = []  # each: {"name": str, "x_norm": tensor[B,N,C], "attn": tensor[B,N,N] or None}
        self.current_layer = 0
        self.mode = "cosine"  # "cosine" | "attention"
        self.last_clicked = None  # (frame_idx, row, col, token_idx)
        self.fig = None
        self.axes = []
        self.ims = []
        self.colorbar = None

    def add_layer(self, name, x, k, q):
        """
        x: (B, N, C) token embeddings (torch Tensor, CPU or CUDA)
        attn: (B, H, N, N) or None. If given, heads will be averaged to (B, N, N).
        """
        eps = 1e-8
        x_norm = x / (x.norm(dim=-1, keepdim=True) + eps)

        self.layers.append({"name": name, "x_norm": x_norm, "k": k, "q": q})

    # ---------- plotting & interaction ----------

    def show(self):
        if not self.layers:
            print("No layers added.")
            return

        # Default reference: center-ish pixel in frame 0
        if self.last_clicked is None:
            r = min(self.image_height // 2, self.image_width - 1)
            c = min(self.image_width // 2, self.image_width - 1)
            token_idx = self.token_offset + r * self.image_width + c
            self.last_clicked = (0, r, c, token_idx)

        self._init_figure()
        self._redraw()
        plt.show()

    def _init_figure(self):
        layer = self.layers[self.current_layer]
        B = layer["x_norm"].shape[0]

        frames_per_row = 5
        ncols = min(B, frames_per_row)
        nrows = (B + ncols - 1) // ncols

        self.fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5*ncols, 4.5*nrows))
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]

        self.axes = [ax for row in axes for ax in row]
        self.ims = []

        # connect events once
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _compute_maps(self):
        """Return (maps: B x (W*W), vmin, vmax) for current mode using current ref token."""
        layer = self.layers[self.current_layer]
        x_norm = layer["x_norm"]      # (B, N, C)
        print("B, N, C", x_norm.shape)
        k = layer["k"]
        q = layer["q"]
        B, N, C = x_norm.shape

        frame_idx, row, col, token_idx = self.last_clicked

        tokens_per_img = self.token_offset + self.image_height*self.image_width

        # guard if click outside valid token area
        if token_idx < self.token_offset or token_idx >= N:
            raise ValueError("Clicked token index out of range.")

        if self.mode == "cosine":
            # Use the *vector from the clicked frame* as the query, compare to ALL tokens in EVERY frame
            ref_vec = x_norm[frame_idx, token_idx, :]                     # (C,)
            # (C) x (B,N,C) -> (B,N)
            maps_full = torch.einsum("c,bnc->bn", ref_vec, x_norm)
            # display only the image tokens (skip special tokens)
            shown = maps_full[:, self.token_offset:tokens_per_img]
            # run_asmk(x_norm)
            # np.save("/home/dominic/vggt/tokens2.npy", x_norm.detach().cpu().numpy())
            vmin, vmax = 0.0, 1.0
            return shown, vmin, vmax

        else:  # attention
            k_selected = k[:, :, token_idx+frame_idx*tokens_per_img:token_idx+frame_idx*tokens_per_img+1, :] 
            attn = q @ k_selected.transpose(-2, -1) 
            attn = attn.transpose(-2, -1) 
            attn = attn.softmax(dim=-1)    
            attn_mean = attn.mean(dim=1)
            attn_mean = attn_mean.view(B, N)
            shown = attn_mean[:, self.token_offset:tokens_per_img]
            # dynamic range across all shown frames for this selection
            vmin = float(shown.min().detach().cpu())
            vmax = float(shown.max().detach().cpu())
            # protect against degenerate ranges
            print(vmin, vmax)
            if vmin == vmax:
                vmax = vmin + 1e-8
            return shown, vmin, vmax

    def _redraw(self):
        maps, vmin, vmax = self._compute_maps()  # (B, W*W)
        B = maps.shape[0]
        W = self.image_width
        H = self.image_height

        # draw/update images
        if not self.ims:
            for i in range(len(self.axes)):
                ax = self.axes[i]
                if i < B:
                    frame = maps[i].float().detach().cpu().reshape(H, W).numpy()
                    im = ax.imshow(frame, cmap="viridis", vmin=vmin, vmax=vmax)
                    ax.set_title(f"{self.layers[self.current_layer]['name']} • Frame {i}")
                    # mark clicked pixel (same row/col) so you can see it
                    if i == self.last_clicked[0]:
                        ax.plot(self.last_clicked[2], self.last_clicked[1], marker='x', markersize=8, mew=2)
                    else:
                        ax.plot(self.last_clicked[2], self.last_clicked[1], marker='x', markersize=6, alpha=0.4)
                    self.ims.append(im)
                else:
                    ax.axis("off")

            # one shared colorbar
            self.colorbar = self.fig.colorbar(self.ims[0], ax=self.axes[:B],
                                              orientation="horizontal", fraction=0.03, pad=0.08)
        else:
            # update existing
            for i in range(B):
                frame = maps[i].float().detach().cpu().reshape(H, W).numpy()
                self.ims[i].set_data(frame)
                self.ims[i].set_clim(vmin, vmax)
                ax = self.axes[i]
                # clear old markers and replot marker
                # (remove all lines in this axes; cheap if there are only markers)
                for ln in list(ax.lines):
                    ln.remove()
                if i == self.last_clicked[0]:
                    ax.plot(self.last_clicked[2], self.last_clicked[1], marker='x', markersize=8, mew=2)
                else:
                    ax.plot(self.last_clicked[2], self.last_clicked[1], marker='x', markersize=6, alpha=0.4)

            if self.colorbar is not None:
                self.colorbar.update_normal(self.ims[0])

        self.fig.suptitle(
            f"Mode: {self.mode} • Layer: {self.layers[self.current_layer]['name']} • "
            f"Ref: frame={self.last_clicked[0]}, token_idx={self.last_clicked[3]} (row={self.last_clicked[1]}, col={self.last_clicked[2]})",
            fontsize=12
        )
        self.fig.canvas.draw_idle()

    # ---------- events ----------

    def _on_click(self, event):
        if event.inaxes not in self.axes:
            return
        ax_idx = self.axes.index(event.inaxes)

        # guard against clicks outside image
        if event.xdata is None or event.ydata is None:
            return

        col = int(event.xdata)
        row = int(event.ydata)

        # map (row, col) -> token index
        token_idx = self.token_offset + row * self.image_width + col

        self.last_clicked = (ax_idx, row, col, token_idx)
        self._redraw()

    def _on_key(self, event):
        if event.key in ("right", "n"):
            self.current_layer = (self.current_layer + 1) % len(self.layers)
            self._redraw()
        elif event.key in ("left", "p"):
            self.current_layer = (self.current_layer - 1) % len(self.layers)
            self._redraw()
        elif event.key in ("m", "a"):  # toggle mode
            self.mode = "attention" if self.mode == "cosine" else "cosine"
            self._redraw()

class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()
        self.viz = CosineAttentionVisualizer()
        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        print("Shape of images after normalization and reshaping:", images.shape)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        print("Shape of tokens after concatenation:")
        print(camera_token.shape, register_token.shape, patch_tokens.shape, tokens.shape)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        target_layer = 20
        target_tokens = None

        # eps = 1e-8
        # patch_tokens_save = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + eps)
        # np.save("/home/dominic/vggt/tokens2_decoder.npy", patch_tokens_save.detach().cpu().numpy())

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    print("----- Processing frame attention block -----", frame_idx)
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    print("----- Processing global attention block -----", global_idx)
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                    if global_idx == target_layer:
                        target_tokens = tokens[:,5:, :].clone()  # exclude camera and register tokens
                        eps = 1e-8
                        target_tokens = target_tokens / (target_tokens.norm(dim=-1, keepdim=True) + eps)

                        ###### For testing loading a token #####
                        # loaded_tokens = np.load("/home/dominic/vggt/tokens1.npy")
                        # loaded_tokens = torch.from_numpy(loaded_tokens).to(target_tokens.device)
                        # tokens = torch.cat([tokens, loaded_tokens[10:11,:,:]], dim=0)
                        # S += 1
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        self.viz.show()

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx, target_tokens

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos, viz_attention=False)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens, k, q = self.global_blocks[global_idx](tokens, pos=pos, viz_attention=True)
                print(k.shape, q.shape)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        self.viz.add_layer("Layer " +str(global_idx), tokens, k, q)

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
