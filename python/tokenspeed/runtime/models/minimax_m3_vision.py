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

"""MiniMax-M3 vision tower and multimodal projectors."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from tokenspeed.runtime.configs.minimax_m3_config import MiniMaxM3VisionConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.attention.mm_encoder_attention import VisionAttention
from tokenspeed.runtime.layers.conv import Conv3dLayer
from tokenspeed.runtime.layers.linear import ColumnParallelLinear, RowParallelLinear
from tokenspeed.runtime.utils import add_prefix


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_vision_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    _input_shape,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MiniMax-M3's partial 3D RoPE to packed vision queries and keys."""

    cos, sin = position_embeddings
    rotary_dim = cos.shape[-1]
    query_rot, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    key_rot, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]

    query_rot = query_rot.float()
    key_rot = key_rot.float()
    cos = cos.float()
    sin = sin.float()
    query_rot = query_rot * cos + _rotate_half(query_rot) * sin
    key_rot = key_rot * cos + _rotate_half(key_rot) * sin
    query = torch.cat((query_rot.to(query_pass.dtype), query_pass), dim=-1)
    key = torch.cat((key_rot.to(key_pass.dtype), key_pass), dim=-1)
    return query, key


class MiniMaxM3VisionEmbeddings(nn.Module):
    """Convert flattened image/video patches into vision hidden states."""

    def __init__(self, config: MiniMaxM3VisionConfig) -> None:
        super().__init__()
        self.num_channels = config.num_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        kernel_size = (
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        self.patch_embedding = Conv3dLayer(
            self.num_channels,
            self.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() != 2:
            raise ValueError(
                f"MiniMax-M3 pixel_values must be 2D, got {pixel_values.dim()}D."
            )
        expected_patch_width = (
            self.num_channels * self.temporal_patch_size * self.patch_size**2
        )
        if pixel_values.shape[1] != expected_patch_width:
            raise ValueError(
                "MiniMax-M3 flattened patch width must be "
                f"{expected_patch_width}, got {pixel_values.shape[1]}."
            )
        pixel_values = pixel_values.reshape(
            -1,
            self.num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.patch_embedding(pixel_values)
        return hidden_states.reshape(-1, self.hidden_size)


class MiniMaxM3VisionMLP(nn.Module):
    """Tensor-parallel GELU feed-forward block used by the vision tower."""

    def __init__(
        self,
        config: MiniMaxM3VisionConfig,
        mapping: Mapping,
        prefix: str,
    ) -> None:
        super().__init__()
        vision = mapping.vision
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=None,
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            prefix=add_prefix("fc1", prefix),
        )
        self.activation = nn.GELU()
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=None,
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            prefix=add_prefix("fc2", prefix),
            reduce_results=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class MiniMaxM3VisionEncoderLayer(nn.Module):
    """One CLIP-style vision layer with MiniMax 3D rotary attention."""

    def __init__(
        self,
        config: MiniMaxM3VisionConfig,
        mapping: Mapping,
        prefix: str,
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
            mapping=mapping,
            head_size=config.hidden_size // config.num_attention_heads,
            quant_config=None,
            prefix=add_prefix("self_attn", prefix),
            proj_bias=True,
            qkv_bias=True,
            customized_position_embedding_applier=_apply_vision_rotary,
            mm_attention_backend=mm_attention_backend,
        )
        self.layer_norm2 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.mlp = MiniMaxM3VisionMLP(
            config,
            mapping,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int,
    ) -> torch.Tensor:
        residual = hidden_states
        attention_output = self.self_attn(
            self.layer_norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
        )
        if attention_output.dim() == 3:
            attention_output = attention_output.squeeze(0)
        hidden_states = residual + attention_output
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class MiniMaxM3VisionEncoder(nn.Module):
    """Stack of MiniMax-M3 vision encoder layers."""

    def __init__(
        self,
        config: MiniMaxM3VisionConfig,
        mapping: Mapping,
        prefix: str,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MiniMaxM3VisionEncoderLayer(
                    config,
                    mapping,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    mm_attention_backend=mm_attention_backend,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cu_seqlens,
                position_embeddings,
                max_seqlen,
            )
        return hidden_states


class MiniMaxM3VisionTransformer(nn.Module):
    """MiniMax-M3 Conv3d vision transformer with packed variable-length attention."""

    def __init__(
        self,
        config: MiniMaxM3VisionConfig,
        mapping: Mapping,
        prefix: str,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError(
                "MiniMax-M3 vision hidden_size must be divisible by "
                "num_attention_heads."
            )

        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.vision_segment_max_frames = getattr(
            config, "vision_segment_max_frames", None
        )
        self.embeddings = MiniMaxM3VisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.encoder = MiniMaxM3VisionEncoder(
            config,
            mapping,
            prefix=add_prefix("encoder", prefix),
            mm_attention_backend=mm_attention_backend,
        )

        head_dim = config.hidden_size // config.num_attention_heads
        rotary_dims = 2 * (head_dim // 2)
        self.axis_dim = 2 * ((rotary_dims // 3) // 2)
        inv_freq = 1.0 / (
            config.rope_parameters["rope_theta"]
            ** (torch.arange(0, self.axis_dim, 2, dtype=torch.float32) / self.axis_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embeddings.patch_embedding.weight.device

    def _split_video_segments(
        self,
        grid_thw: Sequence[Sequence[int]],
    ) -> list[tuple[int, int, int]]:
        segments: list[tuple[int, int, int]] = []
        max_frames = self.vision_segment_max_frames
        for raw_t, raw_h, raw_w in grid_thw:
            grid_t, grid_h, grid_w = int(raw_t), int(raw_h), int(raw_w)
            if min(grid_t, grid_h, grid_w) <= 0:
                raise ValueError(
                    f"Invalid MiniMax-M3 vision grid {(grid_t, grid_h, grid_w)}."
                )
            if grid_h % self.spatial_merge_size or grid_w % self.spatial_merge_size:
                raise ValueError(
                    "MiniMax-M3 vision grid height and width must be divisible "
                    f"by {self.spatial_merge_size}."
                )
            if max_frames is None:
                segments.append((grid_t, grid_h, grid_w))
                continue
            for start in range(0, grid_t, max_frames):
                segments.append((min(max_frames, grid_t - start), grid_h, grid_w))
        return segments

    def _rope_for_grid(self, grid_t: int, grid_h: int, grid_w: int) -> torch.Tensor:
        merge = self.spatial_merge_size
        tokens_per_frame = grid_h * grid_w
        temporal = (
            torch.arange(grid_t, device=self.device)
            .unsqueeze(1)
            .expand(-1, tokens_per_frame)
            .reshape(-1)
        )

        height = (
            torch.arange(grid_h, device=self.device).unsqueeze(1).expand(-1, grid_w)
        )
        width = torch.arange(grid_w, device=self.device).unsqueeze(0).expand(grid_h, -1)
        reordered_shape = (
            grid_h // merge,
            merge,
            grid_w // merge,
            merge,
        )
        height = height.reshape(reordered_shape).permute(0, 2, 1, 3)
        width = width.reshape(reordered_shape).permute(0, 2, 1, 3)
        height = height.unsqueeze(0).expand(grid_t, -1, -1, -1, -1).reshape(-1)
        width = width.unsqueeze(0).expand(grid_t, -1, -1, -1, -1).reshape(-1)

        coordinates = torch.stack((temporal, height, width), dim=-1).float()
        frequencies = (coordinates.unsqueeze(-1) * self.inv_freq).reshape(
            coordinates.shape[0],
            -1,
        )
        frequencies = torch.cat((frequencies, frequencies), dim=-1)
        return frequencies

    def _prepare_metadata(
        self,
        grid_thw: torch.Tensor | Sequence[Sequence[int]],
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        int,
        list[tuple[int, int, int]],
    ]:
        raw_grid = grid_thw.tolist() if isinstance(grid_thw, torch.Tensor) else grid_thw
        segments = self._split_video_segments(raw_grid)
        sequence_lengths = [
            grid_t * grid_h * grid_w for grid_t, grid_h, grid_w in segments
        ]
        cu_seqlens = torch.tensor(
            [0, *sequence_lengths],
            dtype=torch.int32,
            device=self.device,
        ).cumsum(0)
        frequencies = torch.cat(
            [self._rope_for_grid(*segment) for segment in segments],
            dim=0,
        )
        cos = frequencies.cos().unsqueeze(-2).to(self.dtype)
        sin = frequencies.sin().unsqueeze(-2).to(self.dtype)
        return cu_seqlens, (cos, sin), max(sequence_lengths), segments

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | Sequence[Sequence[int]],
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        hidden_states = self.pre_layrnorm(self.embeddings(pixel_values))
        cu_seqlens, position_embeddings, max_seqlen, segments = self._prepare_metadata(
            grid_thw
        )
        expected_tokens = sum(t * h * w for t, h, w in segments)
        if hidden_states.shape[0] != expected_tokens:
            raise ValueError(
                "MiniMax-M3 vision grid describes "
                f"{expected_tokens} patches, but received {hidden_states.shape[0]}."
            )
        return self.encoder(
            hidden_states,
            cu_seqlens,
            position_embeddings,
            max_seqlen,
        )


class MiniMaxM3VisionTower(nn.Module):
    """Checkpoint-compatible wrapper around the MiniMax-M3 vision transformer."""

    def __init__(
        self,
        config: MiniMaxM3VisionConfig,
        mapping: Mapping,
        prefix: str,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        self.vision_model = MiniMaxM3VisionTransformer(
            config,
            mapping,
            prefix=add_prefix("vision_model", prefix),
            mm_attention_backend=mm_attention_backend,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_model.dtype

    @property
    def device(self) -> torch.device:
        return self.vision_model.device

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Encode flattened patches and return one hidden state per input patch."""

        return self.vision_model(pixel_values, grid_thw)


class MiniMaxM3MultiModalProjector(nn.Module):
    """Project per-patch vision states into the language hidden size."""

    def __init__(
        self,
        vision_hidden_size: int,
        projector_hidden_size: int,
        text_hidden_size: int,
        mapping: Mapping,
        prefix: str,
        bias: bool = True,
    ) -> None:
        super().__init__()
        vision = mapping.vision
        self.linear_1 = ColumnParallelLinear(
            vision_hidden_size,
            projector_hidden_size,
            bias=bias,
            quant_config=None,
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            prefix=add_prefix("linear_1", prefix),
        )
        self.activation = nn.GELU()
        self.linear_2 = RowParallelLinear(
            projector_hidden_size,
            text_hidden_size,
            bias=bias,
            quant_config=None,
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            prefix=add_prefix("linear_2", prefix),
            reduce_results=True,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.activation(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class MiniMaxM3PatchMergeMLP(nn.Module):
    """Merge each spatial patch group into one language-model token."""

    def __init__(
        self,
        spatial_merge_size: int,
        text_hidden_size: int,
        projector_hidden_size: int,
        mapping: Mapping,
        prefix: str,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.group_size = spatial_merge_size**2
        vision = mapping.vision
        self.linear_1 = ColumnParallelLinear(
            text_hidden_size * self.group_size,
            projector_hidden_size,
            bias=bias,
            quant_config=None,
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            prefix=add_prefix("linear_1", prefix),
        )
        self.activation = nn.GELU()
        self.linear_2 = RowParallelLinear(
            projector_hidden_size,
            text_hidden_size,
            bias=bias,
            quant_config=None,
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            prefix=add_prefix("linear_2", prefix),
            reduce_results=True,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        if image_features.shape[0] % self.group_size:
            raise ValueError(
                "MiniMax-M3 projected patch count must be divisible by "
                f"{self.group_size}, got {image_features.shape[0]}."
            )
        image_features = image_features.reshape(
            image_features.shape[0] // self.group_size,
            -1,
        )
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.activation(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


__all__ = [
    "MiniMaxM3MultiModalProjector",
    "MiniMaxM3PatchMergeMLP",
    "MiniMaxM3VisionTower",
]
