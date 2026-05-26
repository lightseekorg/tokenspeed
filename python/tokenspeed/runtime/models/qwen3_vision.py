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

"""Qwen3 visual tower base reused by the Qwen3.5 target models."""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import ACT2FN

from tokenspeed.runtime.configs.qwen3_vision_config import Qwen3VLVisionConfig
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
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import get_global_server_args


@lru_cache(maxsize=1024)
def _rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
    if isinstance(h, torch.Tensor):
        h = int(h.item())
    if isinstance(w, torch.Tensor):
        w = int(w.item())
    if isinstance(spatial_merge_size, torch.Tensor):
        spatial_merge_size = int(spatial_merge_size.item())
    hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
    h_div = h // spatial_merge_size
    w_div = w // spatial_merge_size
    hpos_ids = hpos_ids.reshape(h_div, spatial_merge_size, w_div, spatial_merge_size)
    hpos_ids = hpos_ids.transpose(0, 2, 1, 3).flatten()

    wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
    wpos_ids = wpos_ids.reshape(h_div, spatial_merge_size, w_div, spatial_merge_size)
    wpos_ids = wpos_ids.transpose(0, 2, 1, 3).flatten()
    return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))


class Qwen3VLVisionMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        mapping: Mapping,
        bias: bool = True,
        hidden_act="silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        vision = mapping.vision
        self.linear_fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
            tp_size=vision.tp_size,
            tp_rank=vision.tp_rank,
            tp_group=vision.tp_group,
        )
        self.linear_fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
            tp_size=vision.tp_size,
            tp_rank=vision.tp_rank,
            tp_group=vision.tp_group,
            reduce_results=True,
        )
        self.act = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor):
        x_fc1, _ = self.linear_fc1(x)
        x_act = self.act(x_fc1)
        mlp_output, _ = self.linear_fc2(x_act)
        return mlp_output


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = Conv3dLayer(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class Qwen3VLVisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        intermediate_dim: int,
        mapping: Mapping,
        head_size: Optional[int] = None,
        hidden_act="silu",
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        workspace_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            head_size=head_size,
            proj_bias=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            workspace_buffer=workspace_buffer,
            mapping=mapping,
        )
        self.mlp = Qwen3VLVisionMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            mapping=mapping,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: Optional[int] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.norm1(x)
        hidden_states = rearrange(hidden_states, "s b ... -> b s ...")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        attn = rearrange(attn, "b s ... -> s b ...")
        x += attn
        norm2 = self.norm2(x)
        mlp = self.mlp(norm2)
        x += mlp
        return x


class Qwen3VLMoeVisionPatchMerger(nn.Module):

    def __init__(
        self,
        dim: int,
        context_dim: int,
        padded_context_dim: int,
        mapping: Mapping,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.padded_context_dim = padded_context_dim * (spatial_merge_size**2)

        self.use_postshuffle_norm = use_postshuffle_norm

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(
            self.hidden_size if use_postshuffle_norm else context_dim
        )
        vision = mapping.vision
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.padded_context_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
            tp_size=vision.tp_size,
            tp_rank=vision.tp_rank,
            tp_group=vision.tp_group,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(
            self.padded_context_dim,
            dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
            tp_size=vision.tp_size,
            tp_rank=vision.tp_rank,
            tp_group=vision.tp_group,
            reduce_results=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)

        x_parallel, _ = self.linear_fc1(x)
        x_parallel = self.act_fn(x_parallel)
        out, _ = self.linear_fc2(x_parallel)
        return out


class Qwen3VLMoeVisionModel(nn.Module):

    def __init__(
        self,
        vision_config: Qwen3VLVisionConfig,
        mapping: Mapping,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        vision = mapping.vision
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)
        self.spatial_merge_size = vision_config.spatial_merge_size
        # layer indices whose outputs feed the deepstack mergers
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self.patch_embed = Qwen3VLVisionPatchEmbed(config=vision_config)
        self.pos_embed = VocabParallelEmbedding(
            self.num_position_embeddings,
            self.hidden_size,
            quant_config=quant_config,
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            prefix=add_prefix("pos_embed", prefix),
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim // 2,
            max_position=8192,
            base=10000.0,
            is_neox_style=True,
        )

        workspace_buffer = None
        if get_global_server_args().mm_attention_backend == "flashinfer_cudnn":
            if torch.cuda.is_available():
                ws_device = torch.device("cuda", torch.cuda.current_device())
            else:
                ws_device = self.device
            workspace_buffer = torch.empty(
                VIT_CUDNN_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=ws_device,
            )

        self.blocks = nn.ModuleList(
            [
                Qwen3VLVisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    intermediate_dim=vision_config.intermediate_size,
                    head_size=head_dim,
                    hidden_act=vision_config.hidden_act,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),
                    workspace_buffer=workspace_buffer,
                    mapping=mapping,
                )
                for layer_idx in range(vision_config.depth)
            ]
        )
        self.merger = Qwen3VLMoeVisionPatchMerger(
            dim=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            padded_context_dim=self.num_heads * head_dim,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=add_prefix("merger", prefix),
            mapping=mapping,
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLMoeVisionPatchMerger(
                    dim=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    padded_context_dim=self.num_heads * head_dim,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"deepstack_merger_list.{layer_idx}", prefix),
                    mapping=mapping,
                )
                for layer_idx in range(len(self.deepstack_visual_indexes))
            ]
        )

        self.tp_size = vision.tp_size

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(
        self, grid_thw: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_ids = []
        for t, h, w in grid_thw:
            base = _rot_pos_ids(h, w, self.spatial_merge_size)
            pos_ids.append(base if t == 1 else base.repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)

        cos, sin = self._get_rotary_cos_sin(max_grid_size)

        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)

        return cos_combined, sin_combined

    def _get_rotary_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.rotary_pos_emb.cos_sin_cache[:seqlen].to(self.device)
        return cos_sin.chunk(2, dim=-1)

    def fast_pos_embed_interpolate_from_list(self, grid_thw):
        num_grid_per_side = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim

        outputs = []
        for t, h, w in grid_thw:
            h_idxs = torch.linspace(
                0, num_grid_per_side - 1, h, dtype=torch.float32, device=self.device
            )
            w_idxs = torch.linspace(
                0, num_grid_per_side - 1, w, dtype=torch.float32, device=self.device
            )

            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # Create meshgrid view for all h, w vars
            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            # original computation of weights
            # w00 = (1 - dh_grid) * (1 - dw_grid)
            # w01 = (1 - dh_grid) * dw_grid
            # w10 = dh_grid * (1 - dw_grid)
            # w11 = dh_grid * dw_grid
            # we reuse w11 here to avoid duplicate
            # dh_grid * dw_grid computation
            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            h_grid_idx = h_grid * num_grid_per_side

            indices = (h_grid_idx + w_grid).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
            weights = weights.to(dtype=self.dtype)

            embeds = self.pos_embed(indices)
            embeds *= weights
            combined = embeds.sum(dim=0)

            combined = combined.reshape(
                h // m_size, m_size, w // m_size, m_size, hidden_dim
            )
            combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    def compute_cudnn_batch_offsets_packed(
        self,
        token_cu_seqlens: np.ndarray,
        *,
        elem_per_token: int,
    ) -> np.ndarray:
        """
        Build packed *element* indptrs for cuDNN prefill.

        Input:
        token_cu_seqlens: (B+1,) token indptr
        elem_per_token: per-token element width on THIS TP rank
                        (usually hidden_size / attn_tp_size)

        Output:
        packed_offsets: (3 * (B_padded + 1),) int32
            [qk_indptr, v_indptr, o_indptr] concatenated,
            each indptr is (B_padded + 1,) in element units.
        """
        assert token_cu_seqlens.ndim == 1 and token_cu_seqlens.size >= 2
        B = int(token_cu_seqlens.size - 1)
        B_padded = round_up_to_bucket(B, VIT_CUDNN_BATCH_BUCKETS)

        # token indptr -> pad to (B_padded+1,) by appending total_tokens for extra empty sequences
        token_indptr = token_cu_seqlens.astype(np.int64, copy=False)  # (B+1,)
        if B_padded != B:
            pad = np.full((B_padded - B,), token_indptr[-1], dtype=token_indptr.dtype)
            token_indptr = np.concatenate([token_indptr, pad], axis=0)  # (B_padded+1,)

        # convert token indptr -> element indptr
        elem_indptr = (token_indptr * int(elem_per_token)).astype(
            np.int32
        )  # (B_padded+1,)

        # q/k/v/o in this vision encoder path share the same indptr
        return np.concatenate([elem_indptr, elem_indptr, elem_indptr], axis=0)

    def compute_cudnn_sequence_lengths_padded(
        self,
        token_cu_seqlens: np.ndarray,
    ) -> np.ndarray:
        """
        token_cu_seqlens: (B+1,) token indptr
        return: (B_padded,) token lengths (padded with 0)
        """
        assert token_cu_seqlens.ndim == 1 and token_cu_seqlens.size >= 2
        B = int(token_cu_seqlens.size - 1)

        seq_lens = (token_cu_seqlens[1:] - token_cu_seqlens[:-1]).astype(
            np.int32
        )  # (B,)

        B_padded = round_up_to_bucket(B, VIT_CUDNN_BATCH_BUCKETS)
        if B_padded != B:
            pad = np.zeros((B_padded - B,), dtype=np.int32)
            seq_lens = np.concatenate([seq_lens, pad], axis=0)  # (B_padded,)
        return seq_lens

    def prepare_patch_embed(
        self, x: torch.Tensor, grid_thw: torch.Tensor | list
    ) -> torch.Tensor:
        """Eager patch-embed (runs before the captured region): Conv patch embed
        + interpolated position embedding + the ``[s, 1, h]`` reshape the block
        loop expects.

        Kept eager (outside the capture-safe region) -- the interpolation does
        host/numpy work.
        """
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)
        grid_thw_list = grid_thw if isinstance(grid_thw, list) else grid_thw.tolist()
        x = x + self.fast_pos_embed_interpolate_from_list(grid_thw_list)
        return x.unsqueeze(1)

    def prepare_metadata(self, grid_thw: torch.Tensor | list) -> dict:
        """Eager metadata pass: rotary embeddings, cu_seqlens, sequence lengths,
        and ``max_seqlen`` as a Python int.

        Everything here involves a host sync or a data-dependent shape, so it
        lives outside the capture-safe block loop. ``max_seqlen`` is
        materialized as a plain int (CPU/numpy, no GPU sync) so the captured
        block loop never hits the attention backend's ``.item()`` fallback.
        """
        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw_np = np.array(grid_thw, dtype=np.int32)
        else:
            grid_thw_list = grid_thw.tolist()
            grid_thw_np = grid_thw.cpu().numpy()

        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        # ---- build token indptr (B+1,) ----
        token_cu_seqlens = np.concatenate(
            [
                np.zeros(1, dtype=np.int32),
                np.repeat(
                    grid_thw_np[:, 1] * grid_thw_np[:, 2], grid_thw_np[:, 0]
                ).cumsum(axis=0, dtype=np.int32),
            ]
        )
        real_seq_lens = token_cu_seqlens[1:] - token_cu_seqlens[:-1]
        real_max_seqlen = int(real_seq_lens.max()) if real_seq_lens.size > 0 else 0

        if get_global_server_args().mm_attention_backend == "flashinfer_cudnn":
            # (B_padded,) token lengths
            seq_lens_padded = self.compute_cudnn_sequence_lengths_padded(
                token_cu_seqlens
            )
            # element-per-token width on this vision TP rank
            elem_per_token = (
                self.hidden_size // self.tp_size
            )  # == heads_per_rank * head_dim
            # (3*(B_padded+1),) packed element indptrs
            offsets_packed = self.compute_cudnn_batch_offsets_packed(
                token_cu_seqlens,
                elem_per_token=elem_per_token,
            )
            sequence_lengths = (
                torch.from_numpy(seq_lens_padded)
                .to(device=self.device, dtype=torch.int32, non_blocking=True)
                .view(-1, 1, 1, 1)
            )  # match cuDNN test style
            cu_seqlens = torch.from_numpy(offsets_packed).to(
                device=self.device, dtype=torch.int32, non_blocking=True
            )
            max_seqlen = round_up_to_bucket(real_max_seqlen, VIT_CUDNN_SEQLEN_BUCKETS)
        else:
            sequence_lengths = None
            cu_seqlens = torch.from_numpy(token_cu_seqlens).to(
                device=self.device, dtype=torch.int32, non_blocking=True
            )
            max_seqlen = real_max_seqlen

        return {
            "cu_seqlens": cu_seqlens,
            "rotary_pos_emb_cos": rotary_pos_emb_cos,
            "rotary_pos_emb_sin": rotary_pos_emb_sin,
            "max_seqlen": max_seqlen,
            "sequence_lengths": sequence_lengths,
        }

    def forward_blocks(self, x: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Capture-safe encoder body: block loop + deepstack mergers + merger.

        No host syncs and no data-dependent control flow, so this region is
        safe to record into a CUDA graph. ``metadata`` comes from
        :meth:`prepare_metadata`; ``x`` from :meth:`prepare_patch_embed`.
        """
        cu_seqlens = metadata["cu_seqlens"]
        rotary_pos_emb_cos = metadata["rotary_pos_emb_cos"]
        rotary_pos_emb_sin = metadata["rotary_pos_emb_sin"]
        max_seqlen = metadata["max_seqlen"]
        sequence_lengths = metadata["sequence_lengths"]

        deepstack_feature_lists = []
        num_deepstack_captured = 0
        for layer_num, blk in enumerate(self.blocks):
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,
                sequence_lengths=sequence_lengths,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[num_deepstack_captured](
                    x
                )
                deepstack_feature_lists.append(deepstack_feature)
                num_deepstack_captured += 1
        x = self.merger(x)
        # [seq_len, out_hidden_size * (1 + depth_of_deepstack)]
        return torch.cat([x] + deepstack_feature_lists, dim=1)
