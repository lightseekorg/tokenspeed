# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Hy4-flash AnyRes SigLIP2 vision encoder and 2x2 perceive patch merger."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from tokenspeed.runtime.configs.hy4_flash_config import HYV4FlashVisionConfig
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem


def compute_hy4_vision_frame_lengths(grid_thw: torch.Tensor) -> list[int]:
    """Return per-frame patch counts in packed order for ``grid_thw [M, 3]``."""
    return [int(h) * int(w) for t, h, w in grid_thw.tolist() for _ in range(int(t))]


class HYV4FlashVisionPatchEmbed(nn.Module):
    """Patch Conv2d plus learned 2D positional grid with ``(size + 0.1) / edge`` interpolation."""

    def __init__(self, config: HYV4FlashVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.interpolate_mode = config.interpolate_mode

        self.patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.position_edge = config.max_image_size // self.patch_size
        self.max_num_patches = self.position_edge * self.position_edge
        # Row 0 is the legacy CLS slot; rows 1..N form the 2D patch grid.
        self.position_embedding = nn.Embedding(self.max_num_patches + 1, self.embed_dim)

    def interpolate_pos_encoding(
        self,
        dtype: torch.dtype,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Interpolate the 2D patch positional table to ``(height, width)`` patches."""
        grid = self.position_embedding.weight[1:, :].reshape(
            1, self.position_edge, self.position_edge, self.embed_dim
        )
        grid = grid.permute(0, 3, 1, 2).float()
        scale_h = (int(height) + 0.1) / self.position_edge
        scale_w = (int(width) + 0.1) / self.position_edge
        interp = F.interpolate(
            grid,
            scale_factor=(scale_h, scale_w),
            mode=self.interpolate_mode,
            align_corners=False,
        )
        return interp.permute(0, 2, 3, 1).reshape(-1, self.embed_dim).to(dtype)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patches = pixel_values.reshape(
            -1, self.num_channels, self.patch_size, self.patch_size
        ).to(dtype=target_dtype)
        embeddings = self.patch_embedding(patches).flatten(-2).squeeze(-1)

        pos_parts = [
            self.interpolate_pos_encoding(embeddings.dtype, int(h), int(w)).repeat(
                int(t), 1
            )
            for t, h, w in grid_thw.tolist()
        ]
        position_embeddings = torch.cat(pos_parts, dim=0)
        return embeddings + position_embeddings


class HYV4FlashVisionAttention(nn.Module):
    """Bidirectional multi-head self-attention for packed vision frames."""

    def __init__(self, config: HYV4FlashVisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.embed_dim, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.embed_dim, self.num_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.embed_dim, self.num_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.embed_dim, bias=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        frame_lengths: list[int],
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(num_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(num_tokens, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(num_tokens, self.num_heads, self.head_dim)

        q_splits = torch.split(q, frame_lengths, dim=0)
        k_splits = torch.split(k, frame_lengths, dim=0)
        v_splits = torch.split(v, frame_lengths, dim=0)

        outputs = []
        for qs, ks, vs in zip(q_splits, k_splits, v_splits, strict=True):
            # [1, H, S, D] for PyTorch SDPA (uses Flash/MemoryEfficient on CUDA SM90+).
            out = F.scaled_dot_product_attention(
                qs.transpose(0, 1).unsqueeze(0),
                ks.transpose(0, 1).unsqueeze(0),
                vs.transpose(0, 1).unsqueeze(0),
                is_causal=False,
                scale=self.scaling,
            )
            outputs.append(out.squeeze(0).transpose(0, 1).reshape(-1, self.embed_dim))
        attn_output = torch.cat(outputs, dim=0)
        return self.o_proj(attn_output)


class HYV4FlashVisionMLP(nn.Module):
    """Two-layer feed-forward block matching ``vit.layers.{i}.mlp.dense_*`` keys."""

    def __init__(self, config: HYV4FlashVisionConfig) -> None:
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.dense_4h_to_h = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(self.activation_fn(self.dense_h_to_4h(hidden_states)))


class HYV4FlashVisionBlock(nn.Module):
    """Pre-LayerNorm vision transformer block matching ``vit.layers.{i}.*`` keys."""

    def __init__(self, config: HYV4FlashVisionConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = HYV4FlashVisionAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = HYV4FlashVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        frame_lengths: list[int],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            frame_lengths=frame_lengths,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class HYV4FlashVisionPatchMerger(nn.Module):
    """RMSNorm + 2x2 Conv2d -> act -> 1x1 Conv2d + row-end ``image_newline``."""

    def __init__(self, config: HYV4FlashVisionConfig) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        merged_dim = config.hidden_size * (self.spatial_merge_size**2)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.proj = nn.Sequential(
            nn.Conv2d(
                config.hidden_size,
                merged_dim,
                kernel_size=self.spatial_merge_size,
                stride=self.spatial_merge_size,
                bias=True,
            ),
            ACT2FN[config.hidden_act],
            nn.Conv2d(
                merged_dim,
                config.out_hidden_size,
                kernel_size=1,
                bias=True,
            ),
        )
        self.image_newline = nn.Parameter(torch.empty(config.out_hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        dtype = hidden_states.dtype
        # Each video frame is spatially merged independently.
        hidden_states = hidden_states.reshape(
            -1, size[0], size[1], hidden_states.shape[-1]
        ).permute(0, 3, 1, 2)
        hidden_states = self.proj(hidden_states)
        num_frames, channels, height, _ = hidden_states.shape
        newline = (
            self.image_newline.reshape(1, channels, 1, 1)
            .expand(num_frames, channels, height, 1)
            .to(dtype)
        )
        hidden_states = torch.cat((hidden_states, newline), dim=-1)
        hidden_states = hidden_states.reshape(num_frames, channels, -1).permute(0, 2, 1)
        return hidden_states.reshape(-1, channels)


class HYV4FlashVisionTower(nn.Module):
    """Complete Hy4-flash vision tower matching checkpoint ``vit.*`` structure."""

    def __init__(self, config: HYV4FlashVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = HYV4FlashVisionPatchEmbed(config)
        self.layers = nn.ModuleList(
            [HYV4FlashVisionBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.perceive = HYV4FlashVisionPatchMerger(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embeddings.patch_embedding.weight.device

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Encode packed patches ``[N, C*P*P]`` with ``grid_thw [M, 3]`` -> ``[L, out_dim]``."""
        grid_cpu = grid_thw.to(device="cpu", dtype=torch.int64).reshape(-1, 3)
        hidden_states = self.embeddings(pixel_values, grid_cpu)
        frame_lengths = compute_hy4_vision_frame_lengths(grid_cpu)

        for layer in self.layers:
            hidden_states = layer(hidden_states, frame_lengths=frame_lengths)

        split_sizes = grid_cpu.prod(dim=-1).tolist()
        split_items = torch.split(hidden_states, split_sizes, dim=0)
        merged = [
            self.perceive(item, size=(int(h), int(w)))
            for (_, h, w), item in zip(grid_cpu.tolist(), split_items, strict=True)
        ]
        return torch.cat(merged, dim=0)

    def embed_media(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        """Encode image or video ``MultimodalDataItem`` list in input order."""
        if not items:
            return torch.empty(
                (0, self.config.out_hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
        pixel_values = torch.cat(
            [
                item.feature.to(
                    device=self.device,
                    dtype=self.dtype,
                    non_blocking=self.device.type == "cuda",
                )
                for item in items
            ],
            dim=0,
        )
        grids = []
        for item in items:
            if item.modality == Modality.VIDEO:
                grid = item.video_grid_thw
            else:
                grid = item.image_grid_thw
            if grid is None:
                raise ValueError(
                    f"Hy4-flash {item.modality} item is missing grid_thw metadata"
                )
            grids.append(torch.as_tensor(grid, dtype=torch.int64).reshape(-1, 3))
        return self(pixel_values, torch.cat(grids, dim=0))

    def _make_warmup_items(self, modality: Modality) -> list[MultimodalDataItem]:
        merge = self.config.spatial_merge_size
        patch_size = self.config.patch_size
        num_patches = merge * merge
        # After 2x2 merge + 1 newline per merged row: 1 * (1 + 1) = 2 tokens.
        out_tokens = 2
        patches = torch.zeros(
            (num_patches, self.config.num_channels * patch_size * patch_size),
            dtype=self.dtype,
        )
        grid = torch.tensor([[1, merge, merge]], dtype=torch.int64)
        grid_key = "video_grid_thw" if modality == Modality.VIDEO else "image_grid_thw"
        return [
            MultimodalDataItem(
                modality=modality,
                offsets=[(0, out_tokens - 1)],
                feature=patches,
                model_specific_data={grid_key: grid},
            )
        ]

    def make_image_warmup_items(self) -> list[MultimodalDataItem]:
        return self._make_warmup_items(Modality.IMAGE)

    def make_video_warmup_items(self) -> list[MultimodalDataItem]:
        return self._make_warmup_items(Modality.VIDEO)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load ``vit.*`` (or ``model.visual.*``) weights into this vision tower."""
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for raw_name, loaded_weight in weights:
            name = raw_name
            if name.startswith("vit."):
                name = name[len("vit.") :]
            elif name.startswith("model.visual."):
                name = name[len("model.visual.") :]
            if name not in params_dict:
                raise KeyError(
                    f"Unexpected Hy4-flash vision weight {raw_name!r} (mapped to {name!r})"
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(f"vit.{name}")
        return loaded_params


__all__ = [
    "HYV4FlashVisionAttention",
    "HYV4FlashVisionBlock",
    "HYV4FlashVisionMLP",
    "HYV4FlashVisionPatchEmbed",
    "HYV4FlashVisionPatchMerger",
    "HYV4FlashVisionTower",
    "compute_hy4_vision_frame_lengths",
]
