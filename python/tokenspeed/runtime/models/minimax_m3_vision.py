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

"""MiniMax-M3 vision tower, 3D RoPE, and multimodal projectors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from tokenspeed.runtime.configs.minimax_m3_config import (
    MiniMaxM3VisionConfig,
    MiniMaxM3VLConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.attention.mm_encoder_attention import (
    VIT_CUDNN_BATCH_BUCKETS,
    VIT_CUDNN_SEQLEN_BUCKETS,
    VIT_CUDNN_WORKSPACE_BYTES,
    VisionAttention,
    round_up_to_bucket,
)
from tokenspeed.runtime.layers.conv import Conv3dLayer
from tokenspeed.runtime.layers.linear import ColumnParallelLinear, RowParallelLinear
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.utils import add_prefix

GridTHW = torch.Tensor | np.ndarray | Sequence[Sequence[int]]


def _normalize_grid_thw(
    grid_thw: GridTHW,
    *,
    spatial_merge_size: int,
) -> np.ndarray:
    """Return validated ``[num_grids, 3]`` THW metadata on the host."""
    if isinstance(grid_thw, torch.Tensor):
        raw_grid = grid_thw.detach().cpu().numpy()
    else:
        raw_grid = np.asarray(grid_thw)

    if raw_grid.ndim != 2 or raw_grid.shape[1] != 3:
        raise ValueError(
            "grid_thw must have shape [num_grids, 3], " f"got {tuple(raw_grid.shape)}."
        )
    if raw_grid.shape[0] == 0:
        raise ValueError("grid_thw must contain at least one image or video grid.")
    if not np.issubdtype(raw_grid.dtype, np.number):
        raise TypeError("grid_thw entries must be integers.")
    if not np.isfinite(raw_grid).all():
        raise ValueError("grid_thw entries must be finite integers.")

    normalized = raw_grid.astype(np.int64, copy=False)
    if not np.equal(raw_grid, normalized).all():
        raise ValueError("grid_thw entries must be integers.")
    if (normalized <= 0).any():
        raise ValueError("grid_thw entries must all be positive.")
    if (normalized[:, 1] % spatial_merge_size != 0).any() or (
        normalized[:, 2] % spatial_merge_size != 0
    ).any():
        raise ValueError(
            "Each grid height and width must be divisible by spatial_merge_size "
            f"({spatial_merge_size})."
        )
    return normalized


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_minimax_m3_vision_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    x_shape: torch.Size | tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MiniMax-M3's partial 3D RoPE to flattened vision Q/K tensors.

    Args:
        q: Query tensor shaped ``[tokens, heads, head_dim]``.
        k: Key tensor shaped ``[tokens, heads, head_dim]``.
        position_embeddings: Cosine and sine tensors shaped
            ``[tokens, rotary_dim]``.
        x_shape: Original attention input shape. It is accepted for the custom
            :class:`VisionAttention` callback interface and is not otherwise used.

    Returns:
        The rotated query and key. Dimensions after ``rotary_dim`` pass through
        unchanged (two dimensions for MiniMax-M3's 80-dimensional heads).
    """
    del x_shape
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(
            "MiniMax-M3 vision rotary expects Q/K shaped "
            f"[tokens, heads, head_dim], got {tuple(q.shape)} and {tuple(k.shape)}."
        )
    if q.shape != k.shape:
        raise ValueError(
            f"MiniMax-M3 vision Q/K shapes must match, got {q.shape} and {k.shape}."
        )
    if q.device != k.device or q.dtype != k.dtype:
        raise ValueError("MiniMax-M3 vision Q/K must share a device and dtype.")
    if not isinstance(position_embeddings, tuple) or len(position_embeddings) != 2:
        raise TypeError("position_embeddings must be a (cos, sin) tuple.")

    cos, sin = position_embeddings
    if cos.ndim != 2 or sin.shape != cos.shape:
        raise ValueError(
            "Vision rotary cos/sin must share shape [tokens, rotary_dim], got "
            f"{tuple(cos.shape)} and {tuple(sin.shape)}."
        )
    if cos.shape[0] != q.shape[0]:
        raise ValueError(
            "Vision rotary token count does not match Q/K: "
            f"{cos.shape[0]} versus {q.shape[0]}."
        )

    rotary_dim = cos.shape[-1]
    if rotary_dim == 0 or rotary_dim % 2 != 0:
        raise ValueError(
            f"Vision rotary_dim must be a positive even value, got {rotary_dim}."
        )
    if rotary_dim > q.shape[-1]:
        raise ValueError(
            f"Vision rotary_dim ({rotary_dim}) exceeds head_dim ({q.shape[-1]})."
        )

    cos = cos.to(device=q.device, dtype=q.dtype).unsqueeze(1)
    sin = sin.to(device=q.device, dtype=q.dtype).unsqueeze(1)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = q_rot * cos + _rotate_half(q_rot) * sin
    k_rot = k_rot * cos + _rotate_half(k_rot) * sin
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


class _MiniMaxM3Vision3DRotaryEmbedding(nn.Module):
    """Generate T/H/W rotary bands in MiniMax-M3 patch-token order."""

    def __init__(
        self,
        head_dim: int,
        *,
        theta: float,
        spatial_merge_size: int,
    ) -> None:
        super().__init__()
        rope_dims = 2 * (head_dim // 2)
        self.axis_dim = 2 * ((rope_dims // 3) // 2)
        if self.axis_dim == 0:
            raise ValueError(f"Vision head_dim is too small for 3D RoPE: {head_dim}.")
        self.rotary_dim = 3 * self.axis_dim
        self.head_dim = head_dim
        self.theta = float(theta)
        self.spatial_merge_size = spatial_merge_size

    def forward(
        self,
        grid_thw: GridTHW,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build cosine/sine embeddings for a batch of image or video grids."""
        grids = _normalize_grid_thw(
            grid_thw,
            spatial_merge_size=self.spatial_merge_size,
        )
        merge_size = self.spatial_merge_size
        coordinates: list[torch.Tensor] = []

        for t_value, h_value, w_value in grids:
            t, h, w = int(t_value), int(h_value), int(w_value)
            height = torch.arange(h, dtype=torch.int64).view(h, 1).expand(h, w)
            height = (
                height.reshape(h // merge_size, merge_size, w // merge_size, merge_size)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            width = torch.arange(w, dtype=torch.int64).view(1, w).expand(h, w)
            width = (
                width.reshape(h // merge_size, merge_size, w // merge_size, merge_size)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            time = torch.arange(t, dtype=torch.int64).repeat_interleave(h * w)
            coordinates.append(
                torch.stack((time, height.repeat(t), width.repeat(t)), dim=-1)
            )

        coords = torch.cat(coordinates, dim=0).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        frequencies = torch.arange(
            0,
            self.axis_dim,
            2,
            dtype=torch.float32,
            device=device,
        )
        inv_freq = 1.0 / (self.theta ** (frequencies / self.axis_dim))
        axis_frequencies = torch.cat(
            [coords[:, axis : axis + 1] * inv_freq for axis in range(3)],
            dim=-1,
        )
        embeddings = torch.cat((axis_frequencies, axis_frequencies), dim=-1)
        return embeddings.cos().to(dtype=dtype), embeddings.sin().to(dtype=dtype)


class _MiniMaxM3VisionEmbeddings(nn.Module):
    """Bias-free Conv3d patch embedding used by MiniMax-M3."""

    def __init__(self, config: MiniMaxM3VisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.num_channels
        self.embed_dim = config.hidden_size
        kernel_size = (
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        # Keep the checkpoint-facing attribute name. The released checkpoint
        # stores ``embeddings.patch_embedding.weight`` directly.
        self.patch_embedding = Conv3dLayer(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
            # Keep the reference Conv3d accumulation order. The generic
            # unfold+linear fast path introduces BF16-visible differences
            # after the pre-LayerNorm that amplify across 32 vision blocks.
            disable_linear=True,
        ).to(dtype=torch.float32)

    @property
    def flattened_patch_size(self) -> int:
        return (
            self.in_channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 2:
            raise ValueError(
                "MiniMax-M3 pixel_values must have shape [patches, flattened_patch], "
                f"got {tuple(pixel_values.shape)}."
            )
        if pixel_values.shape[1] != self.flattened_patch_size:
            raise ValueError(
                "MiniMax-M3 flattened patch width must be "
                f"{self.flattened_patch_size}, got {pixel_values.shape[1]}."
            )

        weight = self.patch_embedding.weight
        pixel_values = pixel_values.to(device=weight.device, dtype=weight.dtype)
        pixel_values = pixel_values.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.patch_embedding(pixel_values).reshape(-1, self.embed_dim)


class _MiniMaxM3VisionMLP(nn.Module):
    """Tensor-parallel CLIP MLP with checkpoint-compatible names."""

    def __init__(
        self,
        config: MiniMaxM3VisionConfig,
        mapping: Mapping,
        *,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        vision = mapping.vision
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
            tp_size=vision.tp_size,
            tp_rank=vision.tp_rank,
            tp_group=vision.tp_group,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
            tp_size=vision.tp_size,
            tp_rank=vision.tp_rank,
            tp_group=vision.tp_group,
            reduce_results=True,
        )
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class _MiniMaxM3VisionEncoderLayer(nn.Module):
    """CLIP pre-norm attention/MLP block with 3D rotary attention."""

    def __init__(
        self,
        config: MiniMaxM3VisionConfig,
        mapping: Mapping,
        *,
        quant_config: QuantizationConfig | None,
        prefix: str,
        workspace_buffer: torch.Tensor | None,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.self_attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_size=config.head_dim,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            proj_bias=True,
            qkv_bias=True,
            customized_position_embedding_applier=apply_minimax_m3_vision_rotary,
            workspace_buffer=workspace_buffer,
            mm_attention_backend=mm_attention_backend,
        )
        self.layer_norm2 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.mlp = _MiniMaxM3VisionMLP(
            config,
            mapping,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        attention_input = self.layer_norm1(hidden_states).transpose(0, 1)
        attention_output = self.self_attn(
            attention_input,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        ).transpose(0, 1)
        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


def _cudnn_sequence_lengths(token_cu_seqlens: np.ndarray) -> np.ndarray:
    batch_size = token_cu_seqlens.size - 1
    padded_batch_size = round_up_to_bucket(batch_size, VIT_CUDNN_BATCH_BUCKETS)
    sequence_lengths = np.diff(token_cu_seqlens).astype(np.int32, copy=False)
    if padded_batch_size > batch_size:
        sequence_lengths = np.pad(
            sequence_lengths,
            (0, padded_batch_size - batch_size),
        )
    return sequence_lengths


def _cudnn_packed_offsets(
    token_cu_seqlens: np.ndarray,
    *,
    elements_per_token: int,
) -> np.ndarray:
    batch_size = token_cu_seqlens.size - 1
    padded_batch_size = round_up_to_bucket(batch_size, VIT_CUDNN_BATCH_BUCKETS)
    token_offsets = token_cu_seqlens.astype(np.int64, copy=False)
    if padded_batch_size > batch_size:
        token_offsets = np.pad(
            token_offsets,
            (0, padded_batch_size - batch_size),
            constant_values=int(token_offsets[-1]),
        )
    element_offsets = (token_offsets * elements_per_token).astype(np.int32)
    return np.concatenate((element_offsets, element_offsets, element_offsets))


class MiniMaxM3VisionTower(nn.Module):
    """MiniMax-M3 CLIP-style vision tower with dynamic-grid 3D RoPE.

    The eager preparation methods perform host-dependent grid validation and
    metadata construction. :meth:`forward_blocks` contains only the stable
    tensor program used by the runtime's vision CUDA-graph wrapper.
    """

    def __init__(
        self,
        config: MiniMaxM3VisionConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        if config.position_embedding_type != "rope" or config.rope_mode != "3d":
            raise ValueError(
                "MiniMax-M3 vision requires position_embedding_type='rope' "
                "and rope_mode='3d'."
            )
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "Vision hidden_size must be divisible by num_attention_heads."
            )
        if config.head_dim != config.hidden_size // config.num_attention_heads:
            raise ValueError("Vision config head_dim is inconsistent with hidden_size.")

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.spatial_merge_size = config.spatial_merge_size
        self.temporal_patch_size = config.temporal_patch_size
        self.mm_attention_backend = mm_attention_backend
        self.tp_size = mapping.vision.tp_size

        self.embeddings = _MiniMaxM3VisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.rotary_emb = _MiniMaxM3Vision3DRotaryEmbedding(
            config.head_dim,
            theta=config.rope_theta,
            spatial_merge_size=config.spatial_merge_size,
        )

        workspace_buffer = None
        if mm_attention_backend == "flashinfer_cudnn":
            workspace_device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else self.device
            )
            workspace_buffer = torch.empty(
                VIT_CUDNN_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=workspace_device,
            )

        self.layers = nn.ModuleList(
            [
                _MiniMaxM3VisionEncoderLayer(
                    config,
                    mapping,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_index}", prefix),
                    workspace_buffer=workspace_buffer,
                    mm_attention_backend=mm_attention_backend,
                )
                for layer_index in range(config.num_hidden_layers)
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        """Working dtype of the pre-norm and transformer blocks."""
        return self.pre_layrnorm.weight.dtype

    @property
    def device(self) -> torch.device:
        """Device holding the vision transformer blocks."""
        return self.pre_layrnorm.weight.device

    def prepare_patch_embed(
        self,
        pixel_values: torch.Tensor,
        grid_thw: GridTHW,
    ) -> torch.Tensor:
        """Patchify pixels and return pre-normalized states shaped ``[S, 1, H]``.

        The Conv3d uses its own parameter dtype, then its output is converted to
        the transformer dtype before LayerNorm, matching the reference model.
        """
        grids = _normalize_grid_thw(
            grid_thw,
            spatial_merge_size=self.spatial_merge_size,
        )
        expected_tokens = int(np.prod(grids, axis=1, dtype=np.int64).sum())
        if pixel_values.ndim < 1 or pixel_values.shape[0] != expected_tokens:
            actual_tokens = pixel_values.shape[0] if pixel_values.ndim > 0 else 0
            raise ValueError(
                "pixel_values patch count must equal sum(t * h * w) from grid_thw: "
                f"expected {expected_tokens}, got {actual_tokens}."
            )

        hidden_states = self.embeddings(pixel_values)
        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        return self.pre_layrnorm(hidden_states).unsqueeze(1)

    def prepare_metadata(self, grid_thw: GridTHW) -> dict[str, Any]:
        """Build 3D RoPE and attention metadata, isolating every THW grid.

        TokenSpeed packs encoder misses from multiple requests into one call, so
        every grid must remain an independent attention sequence. This is
        equivalent to invoking the reference vision tower once per image and
        keeps image embeddings content-addressable for prefix-cache reuse.
        """
        grids = _normalize_grid_thw(
            grid_thw,
            spatial_merge_size=self.spatial_merge_size,
        )
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rotary_emb(
            grids,
            device=self.device,
            dtype=self.dtype,
        )

        sequence_sizes = np.prod(grids, axis=1, dtype=np.int64)
        token_cu_seqlens_i64 = np.concatenate(
            (np.zeros(1, dtype=np.int64), sequence_sizes.cumsum(dtype=np.int64))
        )
        if token_cu_seqlens_i64[-1] > np.iinfo(np.int32).max:
            raise ValueError(
                "Vision request contains too many patches for int32 offsets."
            )
        token_cu_seqlens = token_cu_seqlens_i64.astype(np.int32)
        max_seqlen = int(sequence_sizes.max())

        if self.mm_attention_backend == "flashinfer_cudnn":
            padded_lengths = _cudnn_sequence_lengths(token_cu_seqlens)
            packed_offsets = _cudnn_packed_offsets(
                token_cu_seqlens,
                elements_per_token=self.hidden_size // self.tp_size,
            )
            sequence_lengths: torch.Tensor | None = (
                torch.from_numpy(padded_lengths)
                .to(device=self.device, dtype=torch.int32, non_blocking=True)
                .view(-1, 1, 1, 1)
            )
            cu_seqlens = torch.from_numpy(packed_offsets).to(
                device=self.device,
                dtype=torch.int32,
                non_blocking=True,
            )
            max_seqlen = round_up_to_bucket(
                max_seqlen,
                VIT_CUDNN_SEQLEN_BUCKETS,
            )
        else:
            sequence_lengths = None
            cu_seqlens = torch.from_numpy(token_cu_seqlens).to(
                device=self.device,
                dtype=torch.int32,
                non_blocking=True,
            )

        return {
            # Keep graph-varying tensors at the top level. The generic encoder
            # CUDA-graph wrapper owns and refreshes top-level tensor buffers on
            # replay; nesting these in a tuple would bake the synthetic capture
            # grid's 3D RoPE into every real image.
            "rotary_pos_emb_cos": rotary_pos_emb_cos,
            "rotary_pos_emb_sin": rotary_pos_emb_sin,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
            "sequence_lengths": sequence_lengths,
        }

    def forward_blocks(
        self,
        hidden_states: torch.Tensor,
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        """Run the capture-safe transformer block loop."""
        if hidden_states.ndim != 3 or hidden_states.shape[1:] != (1, self.hidden_size):
            raise ValueError(
                "Vision block input must have shape [tokens, 1, hidden_size], got "
                f"{tuple(hidden_states.shape)}."
            )

        position_embeddings = (
            metadata["rotary_pos_emb_cos"],
            metadata["rotary_pos_emb_sin"],
        )
        cu_seqlens = metadata["cu_seqlens"]
        max_seqlen = metadata["max_seqlen"]
        sequence_lengths = metadata["sequence_lengths"]
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                sequence_lengths=sequence_lengths,
            )
        return hidden_states

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: GridTHW,
    ) -> torch.Tensor:
        """Run eager patch preparation, metadata construction, and all blocks."""
        hidden_states = self.prepare_patch_embed(pixel_values, grid_thw)
        return self.forward_blocks(hidden_states, self.prepare_metadata(grid_thw))


class MiniMaxM3MultiModalProjector(nn.Module):
    """Project each 1280-wide vision patch into the text hidden space."""

    def __init__(
        self,
        config: MiniMaxM3VLConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        vision = mapping.vision
        self.input_size = config.vision_config.hidden_size
        self.output_size = config.text_config.hidden_size
        self.linear_1 = ColumnParallelLinear(
            self.input_size,
            config.projector_hidden_size,
            bias=config.multimodal_projector_bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_1", prefix),
            tp_size=vision.tp_size,
            tp_rank=vision.tp_rank,
            tp_group=vision.tp_group,
        )
        self.linear_2 = RowParallelLinear(
            config.projector_hidden_size,
            self.output_size,
            bias=config.multimodal_projector_bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_2", prefix),
            tp_size=vision.tp_size,
            tp_rank=vision.tp_rank,
            tp_group=vision.tp_group,
            reduce_results=True,
        )
        self.activation_fn = ACT2FN[config.projector_hidden_act]

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project ``[patches, vision_hidden]`` to ``[patches, text_hidden]``."""
        if image_features.ndim != 2 or image_features.shape[-1] != self.input_size:
            raise ValueError(
                "MiniMax-M3 multimodal projector expects "
                f"[patches, {self.input_size}], got {tuple(image_features.shape)}."
            )
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class MiniMaxM3PatchMergeMLP(nn.Module):
    """Merge each spatial patch group into one text-width visual token."""

    def __init__(
        self,
        config: MiniMaxM3VLConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        vision = mapping.vision
        self.input_size = config.text_config.hidden_size
        self.output_size = config.text_config.hidden_size
        self.group_size = config.vision_config.spatial_merge_size**2
        merged_size = self.input_size * self.group_size
        if config.merged_hidden_size != merged_size:
            raise ValueError(
                "merged_hidden_size must equal text hidden size times the spatial "
                f"merge area ({merged_size}), got {config.merged_hidden_size}."
            )

        self.linear_1 = ColumnParallelLinear(
            merged_size,
            config.projector_hidden_size,
            bias=config.multimodal_projector_bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_1", prefix),
            tp_size=vision.tp_size,
            tp_rank=vision.tp_rank,
            tp_group=vision.tp_group,
        )
        self.linear_2 = RowParallelLinear(
            config.projector_hidden_size,
            self.output_size,
            bias=config.multimodal_projector_bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_2", prefix),
            tp_size=vision.tp_size,
            tp_rank=vision.tp_rank,
            tp_group=vision.tp_group,
            reduce_results=True,
        )
        self.activation_fn = ACT2FN[config.projector_hidden_act]

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Reshape patch groups and fuse them with the checkpoint merge MLP."""
        if image_features.ndim != 2 or image_features.shape[-1] != self.input_size:
            raise ValueError(
                "MiniMax-M3 patch merge expects "
                f"[patches, {self.input_size}], got {tuple(image_features.shape)}."
            )
        if image_features.shape[0] % self.group_size != 0:
            raise ValueError(
                "Patch count must be divisible by the spatial merge area "
                f"({self.group_size}), got {image_features.shape[0]}."
            )

        hidden_states = image_features.reshape(
            image_features.shape[0] // self.group_size,
            self.input_size * self.group_size,
        )
        hidden_states, _ = self.linear_1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


__all__ = [
    "MiniMaxM3MultiModalProjector",
    "MiniMaxM3PatchMergeMLP",
    "MiniMaxM3VisionTower",
    "apply_minimax_m3_vision_rotary",
]
