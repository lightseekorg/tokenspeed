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

"""Portable image and video encoder for the Muse Spark vision tower."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.attention.mm_encoder_attention import VisionAttention
from tokenspeed.runtime.layers.linear import ColumnParallelLinear, RowParallelLinear
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem
from tokenspeed.runtime.utils import add_prefix

_GRID_KEYS = {Modality.IMAGE: "image_grid_thw", Modality.VIDEO: "video_grid_thw"}


PATCH_CHANNELS = 3
GRID_SAMPLE_MODE = "bilinear"
WINDOW_ATTENTION = "window_attention"
FULL_ATTENTION = "full_attention"


@dataclass(frozen=True)
class MuseSparkVisionConfig:
    """Validated view of a Muse Spark vision tower."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    layer_types: tuple[str, ...]
    layer_norm_eps: float
    merge_size: int
    patch_size: int
    patch_temporal: int
    pos_emb_height: int
    pos_emb_width: int
    max_position_embeddings: int
    rope_theta: float
    hidden_act: str
    out_hidden_size: int
    projector_hidden_size: int

    @classmethod
    def from_dict(
        cls, raw: dict, out_hidden_size: int, projector_hidden_size: int
    ) -> MuseSparkVisionConfig:
        """Build the tower's settings from the checkpoint's ``vision_config``."""
        rope_parameters = dict(raw["rope_parameters"])
        if rope_parameters["rope_type"] != "axial":
            raise ValueError("Muse Spark vision attention rotates axially")
        config = cls(
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            intermediate_size=int(raw["intermediate_size"]),
            layer_types=tuple(raw["layer_types"]),
            layer_norm_eps=float(raw["layer_norm_eps"]),
            merge_size=int(raw["merge_size"]),
            patch_size=int(raw["patch_size"]),
            patch_temporal=int(raw["patch_temporal"]),
            pos_emb_height=int(raw["pos_emb_height"]),
            pos_emb_width=int(raw["pos_emb_width"]),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            rope_theta=float(rope_parameters["rope_theta"]),
            hidden_act=raw["hidden_act"],
            out_hidden_size=int(out_hidden_size),
            projector_hidden_size=int(projector_hidden_size),
        )
        if config.hidden_size % config.num_attention_heads:
            raise ValueError(
                "Vision tower hidden size must be divisible by its head count"
            )
        if config.hidden_size * config.merge_size**2 != config.out_hidden_size:
            raise ValueError("Vision tower pixel-shuffle dimensions are inconsistent")
        if len(config.layer_types) != config.num_hidden_layers:
            raise ValueError("Vision tower layer_types must cover every layer")
        if set(config.layer_types) - {WINDOW_ATTENTION, FULL_ATTENTION}:
            raise ValueError("The vision tower attends over full or windowed patches")
        if config.pos_emb_height * config.pos_emb_width != (
            config.max_position_embeddings
        ):
            raise ValueError("Vision tower position grid must span its embedding table")
        if config.hidden_act != "gelu":
            raise ValueError("The Muse Spark vision tower activates through GELU")
        return config


def rope_frequencies(
    head_dim: int,
    grid_h: int,
    grid_w: int,
    theta: float,
    device: torch.device,
) -> torch.Tensor:
    """Build width-then-height complex RoPE frequencies for one patch grid."""
    if head_dim % 4:
        raise ValueError("Vision tower attention head size must be divisible by four")
    half_dim = head_dim // 2
    exponents = torch.arange(0, half_dim, 2, dtype=torch.float32, device=device)
    inverse = theta ** (-exponents / half_dim)
    pos_h = torch.arange(1, grid_h + 1, dtype=torch.float32, device=device)
    pos_w = torch.arange(1, grid_w + 1, dtype=torch.float32, device=device)
    mesh_h, mesh_w = torch.meshgrid(pos_h, pos_w, indexing="ij")
    freq_h = torch.einsum("hw,d->hwd", mesh_h, inverse)
    freq_w = torch.einsum("hw,d->hwd", mesh_w, inverse)
    angles = torch.cat((freq_w, freq_h), dim=-1)
    return torch.stack((angles.cos(), angles.sin()), dim=-1)


def apply_complex_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    frequencies: torch.Tensor,
    _input_shape: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate each head's first half against its second, the vision tower's convention."""
    cosine = frequencies[..., 0].unsqueeze(1)
    sine = frequencies[..., 1].unsqueeze(1)

    def rotate(value: torch.Tensor) -> torch.Tensor:
        first, second = value.float().chunk(2, dim=-1)
        rotated = torch.cat(
            (first * cosine - second * sine, first * sine + second * cosine),
            dim=-1,
        )
        return rotated.to(value.dtype)

    return rotate(query), rotate(key)


def cumulative_lengths(lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    result = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    if lengths:
        result[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return result


def pixel_shuffle_permutation(
    offset: int,
    grid_h: int,
    grid_w: int,
    factor: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Order a patch grid so each downsample block's factor-by-factor patches are adjacent.

    Args:
        offset: Global patch id of the grid's first row-major patch.
        grid_h: Patch rows; must be a multiple of ``factor``.
        grid_w: Patch columns; must be a multiple of ``factor``.
        factor: Downsample factor; ``factor**2`` patches merge into one output.
        device: Device for the returned permutation.

    Returns:
        The ``(order, next offset)`` tuple: global patch ids permuted from
        row-major ``(h, w)`` into block-major ``(h // factor, w // factor,
        in-block h, in-block w)`` order, and the first global id past this grid.
    """
    if grid_h % factor or grid_w % factor:
        raise ValueError("Vision tower patch grid must align to its downsample factor")
    order = torch.arange(offset, offset + grid_h * grid_w, device=device)
    order = (
        order.reshape(grid_h // factor, factor, grid_w // factor, factor)
        .permute(0, 2, 1, 3)
        .reshape(-1)
    )
    return order, offset + grid_h * grid_w


def downsample_pixel_shuffle(
    value: torch.Tensor,
    permutation: torch.Tensor,
    factor: int,
) -> torch.Tensor:
    """Merge each downsample block's patches into one wider row.

    Args:
        value: ``[patches, channels]`` features to downsample.
        permutation: The pixel-shuffle order from
            :func:`pixel_shuffle_permutation`; each run of ``factor**2``
            entries covers one output block's patches.
        factor: Downsample factor; ``factor**2`` rows merge into one.

    Returns:
        The ``[patches // factor**2, channels * factor**2]`` tensor gathered
        by ``permutation`` with every block's patch rows concatenated along
        the channel dimension.
    """
    pixels_per_output = factor**2
    shuffled = value[permutation]
    return (
        shuffled.reshape(-1, pixels_per_output, value.shape[-1])
        .transpose(1, 2)
        .reshape(-1, value.shape[-1] * pixels_per_output)
    )


def sparse_permutation(
    offset: int,
    grid_h: int,
    grid_w: int,
    block_h: int,
    block_w: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], int]:
    """Order a patch grid block by block for window attention, padding ragged blocks.

    The grid is padded with ``-1`` sentinels up to whole ``block_h``-by-
    ``block_w`` blocks, the row-major patch ids are regrouped so each block's
    patches are consecutive, and the sentinels are dropped, leaving blocks of
    unequal valid length along the tail of the grid.

    Args:
        offset: Global patch id of the grid's first row-major patch.
        grid_h: Patch rows.
        grid_w: Patch columns.
        block_h: Window rows.
        block_w: Window columns.
        device: Device for the returned permutation.

    Returns:
        The ``(order, lengths, next offset)`` tuple: the compacted global
        patch ids in block-major order, each block's valid (non-padded)
        patch count, and the first global id past this grid.
    """
    padded_h = math.ceil(grid_h / block_h) * block_h
    padded_w = math.ceil(grid_w / block_w) * block_w
    order = torch.arange(offset, offset + grid_h * grid_w, device=device).reshape(
        grid_h, grid_w
    )
    order = F.pad(order, (0, padded_w - grid_w, 0, padded_h - grid_h), value=-1)
    rows, cols = padded_h // block_h, padded_w // block_w
    order = (
        order.reshape(rows, block_h, cols, block_w)
        .permute(0, 2, 1, 3)
        .reshape(rows * cols, block_h * block_w)
    )
    lengths = (order != -1).sum(dim=1).tolist()
    return order[order != -1], lengths, offset + grid_h * grid_w


class MuseSparkVisionMLP(nn.Module):
    """Tensor-parallel ``fc2(gelu(fc1(x)))``, shared by the blocks and the adapter."""

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        bias: bool,
        mapping: Mapping,
        prefix: str,
    ) -> None:
        super().__init__()
        vision = mapping.vision
        self.fc1 = ColumnParallelLinear(
            input_size=input_size,
            output_size=intermediate_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=False,
            params_dtype=None,
            quant_config=None,
            output_sizes=None,
            prefix=add_prefix("fc1", prefix),
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
        )
        self.fc2 = RowParallelLinear(
            input_size=intermediate_size,
            output_size=output_size,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=False,
            params_dtype=None,
            quant_config=None,
            prefix=add_prefix("fc2", prefix),
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            reduce_results=True,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value, _ = self.fc1(value)
        value, _ = self.fc2(F.gelu(value))
        return value


class MuseSparkVisionBlock(nn.Module):
    """One pre-norm attention-plus-MLP transformer block of the vision tower."""

    def __init__(
        self,
        config: MuseSparkVisionConfig,
        mapping: Mapping,
        prefix: str,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            mapping=mapping,
            head_size=config.hidden_size // config.num_attention_heads,
            quant_config=None,
            prefix=add_prefix("attn", prefix),
            proj_bias=True,
            qkv_bias=True,
            customized_position_embedding_applier=apply_complex_rope,
            position_embedding_mode=None,
            workspace_buffer=None,
            mm_attention_backend=mm_attention_backend,
        )
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MuseSparkVisionMLP(
            input_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=True,
            mapping=mapping,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        value: torch.Tensor,
        frequencies: torch.Tensor,
        lengths: torch.Tensor,
        max_sequence_length: int,
    ) -> torch.Tensor:
        attention = self.attn(
            self.norm1(value),
            cu_seqlens=lengths,
            position_embeddings=frequencies,
            max_seqlen=max_sequence_length,
        )
        value = value + attention.squeeze(0)
        return value + self.mlp(self.norm2(value))


class MuseSparkVisionPatchEmbedder(nn.Module):
    """Project flattened patches and interpolate the learned position grid."""

    def __init__(
        self,
        patch_input_size: int,
        mapping: Mapping,
        config: MuseSparkVisionConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        vision = mapping.vision
        self.config = config
        self.patch_embedding = ColumnParallelLinear(
            input_size=patch_input_size,
            output_size=config.hidden_size,
            bias=False,
            gather_output=True,
            skip_bias_add=False,
            params_dtype=None,
            quant_config=None,
            output_sizes=None,
            prefix=add_prefix("patch_embedding", prefix),
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
        )
        self.position_embedding_table = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

    def forward(self, patches: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Embed ``patches`` and add the position grid sampled at ``positions``."""
        value, _ = self.patch_embedding(patches)
        grid = (
            self.position_embedding_table.weight.reshape(
                self.config.pos_emb_height,
                self.config.pos_emb_width,
                self.config.hidden_size,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        interpolated = F.grid_sample(
            grid,
            positions[None, None],
            mode=GRID_SAMPLE_MODE,
            align_corners=False,
        )[0, :, 0].transpose(0, 1)
        return value + interpolated


class MuseSparkVisionTower(nn.Module):
    """Encode SMG-patchified image/video items at their native resolutions."""

    def __init__(
        self,
        raw_config: dict,
        out_hidden_size: int,
        projector_hidden_size: int,
        mapping: Mapping,
        prefix: str,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        config = MuseSparkVisionConfig.from_dict(
            raw_config, out_hidden_size, projector_hidden_size
        )
        self.config = config
        self.merge_size = config.merge_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.pos_grid_h = config.pos_emb_height
        self.pos_grid_w = config.pos_emb_width
        self.adapter_output_size = config.projector_hidden_size
        patch_input_size = PATCH_CHANNELS * config.patch_size**2 * config.patch_temporal
        self.patch_embedder = MuseSparkVisionPatchEmbedder(
            patch_input_size=patch_input_size,
            mapping=mapping,
            config=config,
            prefix=add_prefix("patch_embedder", prefix),
        )
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(
            MuseSparkVisionBlock(
                config=config,
                mapping=mapping,
                prefix=add_prefix(f"layers.{index}", prefix),
                mm_attention_backend=mm_attention_backend,
            )
            for index in range(config.num_hidden_layers)
        )
        self.adapter = MuseSparkVisionMLP(
            input_size=config.out_hidden_size,
            intermediate_size=config.projector_hidden_size,
            output_size=config.projector_hidden_size,
            bias=False,
            mapping=mapping,
            prefix=add_prefix("adapter", prefix),
        )

    @staticmethod
    def _item_grid(item: MultimodalDataItem) -> tuple[int, int, int]:
        key = _GRID_KEYS.get(item.modality)
        if key is None:
            raise ValueError(
                f"Muse Spark does not support {item.modality.name.lower()} inputs"
            )
        grid = item.model_specific_data.get(key)
        if grid is None or grid.numel() != 3:
            raise ValueError(f"Muse Spark item requires one {key} row")
        clips, grid_h, grid_w = (int(value) for value in grid.reshape(-1).tolist())
        return clips, grid_h, grid_w

    def make_warmup_items(self, modality: Modality) -> list[MultimodalDataItem]:
        """Create one native-shape item for startup kernel compilation."""
        grid_h = self.pos_grid_h
        grid_w = self.pos_grid_w
        patch_features = (
            PATCH_CHANNELS * self.config.patch_temporal * self.config.patch_size**2
        )
        feature = self.patch_embedder.position_embedding_table.weight.new_zeros(
            (grid_h * grid_w, patch_features)
        )
        return [
            MultimodalDataItem(
                modality=modality,
                feature=feature,
                model_specific_data={
                    _GRID_KEYS[modality]: torch.tensor(
                        [[1, grid_h, grid_w]], dtype=torch.int64, device=feature.device
                    )
                },
            )
        ]

    def forward(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        if not items:
            return self.patch_embedder.position_embedding_table.weight.new_empty(
                (0, self.adapter_output_size)
            )

        patches_list: list[torch.Tensor] = []
        positions_list: list[torch.Tensor] = []
        frequencies_list: list[torch.Tensor] = []
        full_sequence_lengths: list[int] = []
        sparse_sequence_lengths: list[int] = []
        downsample_orders: list[torch.Tensor] = []
        sparse_orders: list[torch.Tensor] = []
        downsample_offset = 0
        sparse_offset = 0
        device = self.patch_embedder.position_embedding_table.weight.device

        for item in items:
            if item.feature is None or item.feature.ndim != 2:
                raise ValueError(
                    "Muse Spark vision features must be flattened 2-D patches"
                )
            clips, grid_h, grid_w = self._item_grid(item)
            sequence_length = grid_h * grid_w
            expected = clips * sequence_length
            if item.feature.shape[0] != expected:
                raise ValueError(
                    f"Muse Spark grid describes {expected} patches, got "
                    f"{item.feature.shape[0]}"
                )
            for clip_index in range(clips):
                patches = item.feature[
                    clip_index * sequence_length : (clip_index + 1) * sequence_length
                ]
                patches_list.append(patches)
                pos_h, pos_w = (
                    torch.linspace(
                        -1 + 1 / grid,
                        1 - 1 / grid,
                        grid,
                        device=device,
                        dtype=patches.dtype,
                    )
                    for grid in (grid_h, grid_w)
                )
                positions_list.append(
                    torch.stack(
                        torch.meshgrid(pos_h, pos_w, indexing="xy"), dim=-1
                    ).reshape(-1, 2)
                )
                frequencies_list.append(
                    rope_frequencies(
                        head_dim=self.head_dim,
                        grid_h=grid_h,
                        grid_w=grid_w,
                        theta=self.config.rope_theta,
                        device=device,
                    ).reshape(-1, self.head_dim // 2, 2)
                )
                downsample_order, downsample_offset = pixel_shuffle_permutation(
                    offset=downsample_offset,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    factor=self.merge_size,
                    device=device,
                )
                downsample_orders.append(downsample_order)
                sparse_order, sparse_lengths, sparse_offset = sparse_permutation(
                    offset=sparse_offset,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    block_h=self.pos_grid_h,
                    block_w=self.pos_grid_w,
                    device=device,
                )
                sparse_orders.append(sparse_order)
                sparse_sequence_lengths.extend(sparse_lengths)
                full_sequence_lengths.append(sequence_length)

        patch_weight = self.patch_embedder.patch_embedding.weight
        patches = torch.cat(patches_list, dim=0).to(
            device=device, dtype=patch_weight.dtype
        )
        positions = torch.cat(positions_list, dim=0)
        frequencies = torch.cat(frequencies_list, dim=0)
        downsample_order = torch.cat(downsample_orders, dim=0)
        value = self.ln_pre(self.patch_embedder(patches, positions))

        full_lengths = cumulative_lengths(full_sequence_lengths, device)
        sparse_order = torch.cat(sparse_orders, dim=0)
        value = value[sparse_order]
        frequencies = frequencies[sparse_order]
        sparse_lengths = cumulative_lengths(sparse_sequence_lengths, device)
        full_max = max(full_sequence_lengths)
        sparse_max = max(sparse_sequence_lengths)
        for layer_type, layer in zip(self.config.layer_types, self.layers):
            windowed = layer_type == WINDOW_ATTENTION
            value = layer(
                value,
                frequencies,
                sparse_lengths if windowed else full_lengths,
                sparse_max if windowed else full_max,
            )
        restored = torch.empty_like(value)
        restored.index_copy_(0, sparse_order, value)
        value = self.ln_post(restored)
        value = downsample_pixel_shuffle(value, downsample_order, self.merge_size)
        return F.gelu(self.adapter(value))


__all__ = [
    "MuseSparkVisionTower",
    "MuseSparkVisionConfig",
    "downsample_pixel_shuffle",
    "pixel_shuffle_permutation",
    "sparse_permutation",
]
