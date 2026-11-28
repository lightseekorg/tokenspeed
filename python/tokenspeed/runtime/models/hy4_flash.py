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

"""Inference-only Hy4-flash multimodal and text backbone modeling.

Architecture overview:
* 4-stream Independent Hyper-Connections (iHC) around every attention and FFN
  sublayer, collapsed before the final norm via ``hc_head``.
* Hybrid 42-layer MLA stack:
  - 12 Compressed-DSA layers (32 heads, 4:1 gated key pooling, top-2048 recall,
    interleaved 3D M-RoPE on the trailing 64 channels).
  - 30 Sliding-Window MLA layers (64 heads, sliding window of 512 tokens,
    window_left=511).
  - Every attention layer carries a head-sharded FP32 ``learnable_sink_param``,
    output sigmoid gate (``linear_gate``), and post-projection RMSNorm.
* Layer 0 Dense SwiGLU MLP with ``post_fc2_norm``; Layers 1..41 LatentMoE
  (3072 -> 1536 latent projection -> 512 routed experts top-16 with
  ``swiglu_limit=10.0`` -> 1536 -> 3072 + shared expert -> ``combine_output_norm``).
* Optional 3-gram Engram conditional memory layers (layers 1 and 13, 1-based ids
  2 and 14) backed by ``RequestTokenHistoryView``.
* Optional FP32 LM head (``enable_lm_head_fp32=True``).
* SigLIP2-So400m AnyRes vision tower (``vit``) with 2x2 ``perceive`` merger.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Iterable
from dataclasses import replace

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention import (
    apply_attention_sink,
    attn_merge_state,
)
from tokenspeed_kernel.ops.attention.kpool import (
    kpool_decode_topk,
    kpool_prefill_topk,
)
from tokenspeed_kernel.ops.attention.kpool.hy4_compress import (
    apply_interleaved_rope_last_dim,
    compute_hy4_mrope_cos_sin,
    hy4_pool_group_keys,
    hy4_quantize_and_store_index_k,
)
from tokenspeed_kernel.ops.attention.mla import mla_project_value
from tokenspeed_kernel.ops.attention.prologue import MLAExpandedKV
from tokenspeed_kernel.ops.gemm import bmm
from tokenspeed_kernel.ops.residual import ihc_head, ihc_post, ihc_pre
from torch import nn

from tokenspeed.runtime.configs.hy4_flash_config import (
    INDEXED_ATTENTION,
    SLIDING_ATTENTION,
    HYV4FlashConfig,
    HYV4FlashDFlyConfig,
    HYV4FlashTextConfig,
)
from tokenspeed.runtime.configs.model_config import configure_dsa_attention
from tokenspeed.runtime.configs.model_profile import ModelProfile
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.hy4_flash import (
    Hy4FlashCacheGroupRouter,
    Hy4FlashDSABackend,
    Hy4FlashSWABackend,
    Hy4FlashSWAPrefillMetadata,
)
from tokenspeed.runtime.layers.attention.configs.hy4_flash import (
    Hy4FlashAttentionConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_hy4_flash import (
    HybridHy4FlashTokenToKVPool,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.mlp import gated_activation
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    MoELayer,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
from tokenspeed.runtime.layers.paged_attention import (
    PagedAttention,
    hf_sliding_window_to_window_left,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.models.deepseek_v3 import absorb_mla_kv_b_proj
from tokenspeed.runtime.models.hy4_flash_engram import (
    HYV4FlashEngramLayer,
    build_hy4_engram_inputs,
)
from tokenspeed.runtime.models.hy4_flash_vision import HYV4FlashVisionTower
from tokenspeed.runtime.multimodal.embedder import EncoderSpec, VisionEmbedder
from tokenspeed.runtime.multimodal.inputs import Modality
from tokenspeed.runtime.utils import add_prefix

logger = logging.getLogger(__name__)

_HY4_QUANT_REPLACEMENTS: tuple[tuple[str, str], ...] = (("language_model.model.", ""),)


def _apply_hy4_rmsnorm(x: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
    """Apply RMSNorm with FP32 variance and a single cast back to ``x.dtype``."""
    if x.is_cuda:
        return norm(x)
    orig_dtype = x.dtype
    xf = x.float()
    normed = (
        xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + norm.variance_epsilon)
    ).to(orig_dtype)
    return norm.weight.to(orig_dtype) * normed


def _normalize_positions_3d(positions: torch.Tensor) -> torch.Tensor:
    """Normalize ``[T]`` or ``[3, T]`` positions to ``[3, T]`` int64."""
    if positions.ndim == 1:
        return positions.to(torch.int64).unsqueeze(0).expand(3, -1)
    if positions.ndim == 2 and positions.shape[0] == 3:
        return positions.to(torch.int64)
    raise ValueError(
        f"Hy4-flash expects 1D [T] or 2D [3, T] positions, got {tuple(positions.shape)}"
    )


def _dequantize_fp8_blockwise(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Reference block-wise FP8 E4M3 dequantization for CPU/eager execution."""
    bn, bk = block_size
    wf = weight.float()
    sf = scale_inv.float()
    if wf.ndim == 2:
        n, k = wf.shape
        expanded = sf.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)[:n, :k]
        return wf * expanded
    if wf.ndim == 3:
        e, n, k = wf.shape
        expanded = sf.repeat_interleave(bn, dim=1).repeat_interleave(bk, dim=2)[
            :, :n, :k
        ]
        return wf * expanded
    raise ValueError(f"Unsupported FP8 weight rank {wf.ndim}")


def _linear_cpu_or_cuda(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Execute a TokenSpeed linear layer on CUDA or via exact CPU reference."""
    if x.is_cuda:
        return layer(x)[0]
    weight = layer.weight
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        scale_inv = layer.weight_scale_inv
        block_size = tuple(layer.quant_method.quant_config.weight_block_size)
        w = _dequantize_fp8_blockwise(weight, scale_inv, block_size).to(x.dtype)
    else:
        w = weight.to(x.dtype)
    bias = getattr(layer, "bias", None)
    if bias is not None:
        bias = bias.to(x.dtype)
    return F.linear(x, w, bias)


# ===----------------------------------------------------------------------=== #
# Independent Hyper-Connections (iHC)
# ===----------------------------------------------------------------------=== #


class HYV4FlashHyperPre(nn.Module):
    """Parameter container matching ``hc_{attn,mlp}_layer.hc_pre.{hc_fn,hc_scale,hc_base}``."""

    def __init__(self, hc_mult: int, hidden_size: int) -> None:
        super().__init__()
        self.hc_fn = nn.Parameter(
            torch.empty(2 * hc_mult, hc_mult * hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_scale = nn.Parameter(
            torch.full((2,), 0.01, dtype=torch.float32),
            requires_grad=False,
        )
        base_init = torch.zeros(2 * hc_mult, dtype=torch.float32)
        base_init[:hc_mult] = -math.log(max(hc_mult - 1, 1))
        self.hc_base = nn.Parameter(base_init, requires_grad=False)


class HYV4FlashHyperConnection(nn.Module):
    """4-stream Independent Hyper-Connection wrapping attention or FFN."""

    def __init__(self, config: HYV4FlashTextConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.hc_magnitude = config.hc_magnitude
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_pre = HYV4FlashHyperPre(config.hc_mult, config.hidden_size)

    def pre(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse ``[T, M, H]`` streams into ``(post_coeff [T, M], branch_in [T, H])``."""
        return ihc_pre(
            hidden_streams,
            self.hc_pre.hc_fn,
            self.hc_pre.hc_scale,
            self.hc_pre.hc_base,
            hc_eps=self.hc_eps,
            hc_magnitude=self.hc_magnitude,
            rms_norm_eps=self.rms_norm_eps,
        )

    def post(
        self,
        hidden_streams: torch.Tensor,
        branch_output: torch.Tensor,
        post_coeff: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast-add ``branch_output [T, H]`` back into ``hidden_streams [T, M, H]``."""
        return ihc_post(hidden_streams, branch_output, post_coeff)


class HYV4FlashHyperHead(nn.Module):
    """Final iHC stream collapse matching ``model.hc_head.hc_head_{fn,scale,base}``."""

    def __init__(self, config: HYV4FlashTextConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_head_fn = nn.Parameter(
            torch.empty(
                config.hc_mult, config.hc_mult * config.hidden_size, dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.full((1,), 0.01, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.full(
                (config.hc_mult,),
                -math.log(max(config.hc_mult - 1, 1)),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return ihc_head(
            hidden_streams,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            hc_eps=self.hc_eps,
            rms_norm_eps=self.rms_norm_eps,
        )


# ===----------------------------------------------------------------------=== #
# Compressed Lightning Indexer (4:1 gated key pooling + 3D M-RoPE)
# ===----------------------------------------------------------------------=== #


class HYV4FlashCompressedIndexer(nn.Module):
    """4:1 gated-pooling compressed indexer for Hy4-flash DSA layers."""

    def __init__(
        self,
        config: HYV4FlashTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.index_kpool = config.index_kpool
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.group_topk = config.index_topk // config.index_kpool
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.nope_dim = self.head_dim - self.qk_rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.weights_scale = (self.n_heads**-0.5) * (self.head_dim**-0.5)

        rope_params = config.rope_parameters[INDEXED_ATTENTION]
        self.rope_theta = float(rope_params["rope_theta"])
        self.mrope_section = tuple(int(s) for s in rope_params["mrope_section"])
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.qk_rope_head_dim, 2, dtype=torch.float32)
                / self.qk_rope_head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.query_projection = ReplicatedLinear(
            config.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("query_projection", prefix),
        )
        self.head_weight_projection = ReplicatedLinear(
            config.hidden_size,
            self.n_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("head_weight_projection", prefix),
        )
        self.token_key_projection = ReplicatedLinear(
            config.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("token_key_projection", prefix),
        )
        self.token_key_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # Pooling projections and intra-group APE stay in FP32 to avoid softmax ties.
        self.key_value_projection = ReplicatedLinear(
            self.head_dim,
            self.head_dim,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=add_prefix("key_value_projection", prefix),
        )
        self.key_gate_projection = ReplicatedLinear(
            self.head_dim,
            self.head_dim,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=add_prefix("key_gate_projection", prefix),
        )
        self.key_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.intra_group_ape = nn.Parameter(
            torch.empty(self.index_kpool, self.head_dim, dtype=torch.float32),
            requires_grad=False,
        )

        # Reusable decode top-k workspaces for CUDA Graph capture stability.
        self._decode_topk_workspace: torch.Tensor | None = None
        self._decode_topk_lens_workspace: torch.Tensor | None = None
        self._retired_decode_workspaces: list[torch.Tensor] = []

    def _rotate_trailing(
        self,
        tensor: torch.Tensor,
        positions_3d: torch.Tensor,
    ) -> torch.Tensor:
        """Apply interleaved 3D M-RoPE to the trailing ``qk_rope_head_dim`` channels."""
        cos, sin = compute_hy4_mrope_cos_sin(
            positions_3d, self.inv_freq, self.mrope_section
        )
        # Match HF rotary embedding dtype rounding before FP32 rotation.
        cos_cast = cos.to(tensor.dtype)
        sin_cast = sin.to(tensor.dtype)
        if tensor.ndim == 3:
            cos_cast = cos_cast.unsqueeze(1)
            sin_cast = sin_cast.unsqueeze(1)
        nope = tensor[..., : self.nope_dim]
        rot = apply_interleaved_rope_last_dim(
            tensor[..., self.nope_dim :], cos_cast, sin_cast
        )
        return torch.cat((nope, rot), dim=-1)

    def pool_groups(self, grouped_token_keys: torch.Tensor) -> torch.Tensor:
        """Pool ``[G, index_kpool, head_dim]`` normalized token keys into ``[G, head_dim]``."""
        return hy4_pool_group_keys(
            grouped_token_keys,
            self.key_value_projection.weight,
            self.key_gate_projection.weight,
            self.intra_group_ape,
            self.key_norm.weight,
            eps=self.rms_norm_eps,
        )

    def _project_token_keys(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raw_k = _linear_cpu_or_cuda(self.token_key_projection, hidden_states)
        return _apply_hy4_rmsnorm(raw_k, self.token_key_norm)

    def _project_queries_and_weights(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        positions_3d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.shape[0]
        query = _linear_cpu_or_cuda(self.query_projection, q_resid).view(
            num_tokens, self.n_heads, self.head_dim
        )
        query = self._rotate_trailing(query, positions_3d)
        weights = (
            _linear_cpu_or_cuda(self.head_weight_projection, hidden_states).float()
            * self.weights_scale
        )
        return query, weights

    def _get_decode_workspaces(
        self,
        num_tokens: int,
        width: int,
        device: torch.device,
        *,
        memory_saver_adapter=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots_buf = self._decode_topk_workspace
        lens_buf = self._decode_topk_lens_workspace
        if (
            slots_buf is None
            or lens_buf is None
            or slots_buf.device != device
            or slots_buf.shape[0] < num_tokens
            or slots_buf.shape[1] != width
        ):
            # A captured CUDA graph may still reference the old buffer; keep it
            # alive so growing never frees memory a captured graph replays into.
            if slots_buf is not None:
                self._retired_decode_workspaces.append(slots_buf)
            if lens_buf is not None:
                self._retired_decode_workspaces.append(lens_buf)
            if memory_saver_adapter is not None:
                with memory_saver_adapter.named_region("workspace"):
                    slots_buf = torch.full(
                        (num_tokens, width), -1, dtype=torch.int32, device=device
                    )
                    lens_buf = torch.zeros(
                        (num_tokens,), dtype=torch.int32, device=device
                    )
            else:
                slots_buf = torch.full(
                    (num_tokens, width), -1, dtype=torch.int32, device=device
                )
                lens_buf = torch.zeros((num_tokens,), dtype=torch.int32, device=device)
            self._decode_topk_workspace = slots_buf
            self._decode_topk_lens_workspace = lens_buf
        return slots_buf[:num_tokens], lens_buf[:num_tokens]

    def write_prefill_cache(
        self,
        *,
        token_keys: torch.Tensor,
        positions_3d: torch.Tensor,
        ctx: ForwardContext,
        dsa_leaf: Hy4FlashDSABackend,
    ) -> None:
        """Pool completed prefill 4-token groups into ``index_k`` and update tail state."""
        pool = ctx.token_to_kv_pool
        if not isinstance(pool, HybridHy4FlashTokenToKVPool):
            raise TypeError(
                "HYV4FlashCompressedIndexer requires HybridHy4FlashTokenToKVPool"
            )
        kpool_rt = dsa_leaf.require_kpool_runtime()
        kpool_rt.ensure_prefill_plan(
            ctx,
            dsa_leaf,
            self.layer_id,
            token_capacity=token_keys.shape[0],
        )
        shared_plan = kpool_rt.prefill_plan
        if shared_plan is None:
            raise RuntimeError("Hy4-flash KPool prefill plan was not initialized")
        plan = shared_plan.write

        index_k_buf, tail_token_keys, group_pos0 = pool.get_hy4_kpool_buffers(
            self.layer_id
        )
        index_values, index_scales = pool.index_k_block_views(index_k_buf)
        kpool = self.index_kpool
        tail_width = tail_token_keys.shape[1]

        num_pools = int(plan.pool_write_slots.numel())
        if num_pools > 0:
            offsets = torch.arange(
                kpool, dtype=torch.int64, device=token_keys.device
            ).unsqueeze(0)
            n_from_tail = plan.pool_n_from_tail.to(torch.int64).unsqueeze(1)
            from_tail = offsets < n_from_tail
            tail_rows = torch.remainder(
                plan.pool_tail_logical_base.unsqueeze(1) + offsets,
                tail_width,
            )
            pool_req_slots = plan.request_slots.index_select(
                0, plan.pool_req_ids.to(torch.int64)
            )
            tail_keys = tail_token_keys[pool_req_slots.unsqueeze(1), tail_rows]
            chunk_rows = plan.pool_chunk_src.unsqueeze(1) + torch.clamp(
                offsets - n_from_tail, min=0
            )
            chunk_keys = token_keys.index_select(0, chunk_rows.reshape(-1)).view(
                num_pools, kpool, self.head_dim
            )
            grouped = torch.where(from_tail.unsqueeze(2), tail_keys, chunk_keys)

            # Group RoPE uses the 3D coordinates of slot 0 in each group.
            tail_pos0 = group_pos0.index_select(0, pool_req_slots)  # [G, 3]
            chunk_pos0 = positions_3d.index_select(1, plan.pool_chunk_src).transpose(
                0, 1
            )  # [G, 3]
            use_tail_pos = (plan.pool_n_from_tail > 0).unsqueeze(1)
            group_pos = torch.where(use_tail_pos, tail_pos0, chunk_pos0).transpose(
                0, 1
            )  # [3, G]

            pooled = self.pool_groups(grouped)
            pooled = self._rotate_trailing(pooled, group_pos)
            hy4_quantize_and_store_index_k(
                pooled,
                plan.pool_write_slots,
                index_values,
                index_scales,
            )

        num_tail_writes = int(plan.tail_write_counts.numel())
        if num_tail_writes > 0:
            dst_slots = plan.request_slots.index_select(
                0, plan.tail_req_ids.to(torch.int64)
            )
            # Bounded by kpool - 1 = 3 steps per request; keep fully vectorized on device.
            counts = plan.tail_write_counts.to(torch.int64)
            src_starts = plan.tail_chunk_src.to(torch.int64)
            dst_pos_starts = plan.tail_dst_positions.to(torch.int64)
            for step in range(kpool - 1):
                active = counts > step
                safe_src = torch.where(active, src_starts + step, 0)
                dst_pos = dst_pos_starts + step
                tail_idx = torch.remainder(dst_pos, tail_width)
                incoming_keys = token_keys.index_select(0, safe_src).to(
                    tail_token_keys.dtype
                )
                tail_token_keys[dst_slots, tail_idx] = torch.where(
                    active.unsqueeze(-1),
                    incoming_keys,
                    tail_token_keys[dst_slots, tail_idx],
                )
                starts_group = active & (torch.remainder(dst_pos, kpool) == 0)
                incoming_pos = positions_3d.index_select(1, safe_src).transpose(0, 1)
                group_pos0[dst_slots] = torch.where(
                    starts_group.unsqueeze(-1),
                    incoming_pos,
                    group_pos0[dst_slots],
                )

    def write_decode_cache(
        self,
        *,
        token_keys: torch.Tensor,
        positions_3d: torch.Tensor,
        ctx: ForwardContext,
        dsa_leaf: Hy4FlashDSABackend,
        num_reqs: int,
        q_len_per_req: int,
    ) -> None:
        """Append decode/verify token keys into tail ring and flush closed 4-token groups.

        Uses static shapes and masked scatter writes (``write_slots=-1`` skips
        writes inside ``hy4_quantize_and_store_index_k``) so decode CUDA graphs
        capture cleanly with zero host synchronization.
        """
        pool = ctx.token_to_kv_pool
        if not isinstance(pool, HybridHy4FlashTokenToKVPool):
            raise TypeError(
                "HYV4FlashCompressedIndexer requires HybridHy4FlashTokenToKVPool"
            )
        kpool_rt = dsa_leaf.require_kpool_runtime()
        metadata = dsa_leaf.forward_decode_metadata
        row_start = int(metadata.num_extends or 0)
        seq_lens = metadata.seq_lens_k[row_start : row_start + num_reqs].to(torch.int64)
        index_table = dsa_leaf.kpool_decode_page_table(row_start, num_reqs).to(
            torch.int64
        )
        request_slots = kpool_rt.req_pool_indices
        if request_slots is None:
            raise RuntimeError("Hy4-flash KPool decode requires request-pool indices")
        req_slots = request_slots[row_start : row_start + num_reqs].to(torch.int64)

        index_k_buf, tail_token_keys, group_pos0 = pool.get_hy4_kpool_buffers(
            self.layer_id
        )
        index_values, index_scales = pool.index_k_block_views(index_k_buf)
        index_rows_per_page = index_values.shape[1]
        kpool = self.index_kpool
        tail_width = tail_token_keys.shape[1]

        keys_view = token_keys.view(num_reqs, q_len_per_req, self.head_dim)
        pos_view = positions_3d.view(3, num_reqs, q_len_per_req)
        group_offsets = torch.arange(kpool, dtype=torch.int64, device=token_keys.device)
        req_arange = torch.arange(num_reqs, dtype=torch.int64, device=token_keys.device)

        for step in range(q_len_per_req):
            tok_pos = (seq_lens - q_len_per_req + step).clamp_min(0)
            valid_tok = (seq_lens - q_len_per_req + step) >= 0
            tail_col = torch.remainder(tok_pos, tail_width)
            step_keys = keys_view[:, step, :].to(tail_token_keys.dtype)
            # Only overwrite valid slots; padded graph rows have seq_len <= 0.
            existing = tail_token_keys[req_slots, tail_col]
            tail_token_keys[req_slots, tail_col] = torch.where(
                valid_tok.unsqueeze(1), step_keys, existing
            )

            is_group_start = valid_tok & (torch.remainder(tok_pos, kpool) == 0)
            step_pos = pos_view[:, :, step].transpose(0, 1)  # [num_reqs, 3]
            group_pos0[req_slots] = torch.where(
                is_group_start.unsqueeze(1), step_pos, group_pos0[req_slots]
            )

            is_group_end = valid_tok & (torch.remainder(tok_pos, kpool) == (kpool - 1))
            group_base = tok_pos - (kpool - 1)
            gather_cols = torch.remainder(
                group_base.unsqueeze(1) + group_offsets.unsqueeze(0), tail_width
            )
            grouped = tail_token_keys[req_slots.unsqueeze(1), gather_cols]  # [R, 4, D]
            pooled = self.pool_groups(grouped)
            pooled = self._rotate_trailing(
                pooled, group_pos0[req_slots].transpose(0, 1)
            )

            pool_ids = torch.div(tok_pos, kpool, rounding_mode="floor")
            table_cols = torch.div(
                pool_ids, index_rows_per_page, rounding_mode="floor"
            ).clamp(min=0, max=index_table.shape[1] - 1)
            write_pages = index_table[req_arange, table_cols]
            raw_slots = write_pages * index_rows_per_page + torch.remainder(
                pool_ids, index_rows_per_page
            )
            masked_slots = torch.where(
                is_group_end, raw_slots, torch.full_like(raw_slots, -1)
            )
            hy4_quantize_and_store_index_k(
                pooled,
                masked_slots,
                index_values,
                index_scales,
            )

    def select_prefill(
        self,
        *,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        positions_3d: torch.Tensor,
        ctx: ForwardContext,
        dsa_leaf: Hy4FlashDSABackend,
        num_prefill_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
        """Update prefill index cache and select top-k slots when context > ``index_topk``."""
        del num_prefill_tokens
        token_keys = self._project_token_keys(hidden_states)
        self.write_prefill_cache(
            token_keys=token_keys,
            positions_3d=positions_3d,
            ctx=ctx,
            dsa_leaf=dsa_leaf,
        )
        metadata = dsa_leaf.chunked_prefill_metadata
        prefix_cpu = metadata.extend_prefix_lens_cpu[: ctx.num_extends]
        extend_cpu = metadata.extend_seq_lens_cpu[: ctx.num_extends]
        max_seq_len = max(
            (
                int(p) + int(e)
                for p, e in zip(prefix_cpu.tolist(), extend_cpu.tolist(), strict=True)
            ),
            default=0,
        )
        if max_seq_len <= self.index_topk:
            return None

        query, weights = self._project_queries_and_weights(
            hidden_states, q_resid, positions_3d
        )
        kpool_rt = dsa_leaf.require_kpool_runtime()
        shared_plan = kpool_rt.prefill_plan
        if shared_plan is None:
            raise RuntimeError("Hy4-flash KPool prefill plan was not initialized")
        history_table = dsa_leaf.kpool_prefill_page_table(ctx.num_extends)
        pool = ctx.token_to_kv_pool
        index_cache = pool.get_index_k_buffer(self.layer_id)

        selected_slots, selected_lens = kpool_prefill_topk(
            query.to(torch.bfloat16).contiguous(),
            index_cache,
            weights.contiguous(),
            shared_plan.positions,
            shared_plan.query_start_loc,
            history_table,
            history_table,
            pool_size=self.index_kpool,
            page_size=index_cache.shape[1],
            kv_page_size=pool.arena.kv_page_size,
            topk_pools=self.group_topk,
            softmax_scale=1.0,
            apply_relu=True,
            append_tail=True,
            req_ids=shared_plan.req_ids,
            causal_lens=shared_plan.causal_lens,
            pool_workspace_slots=shared_plan.pool_workspace_slots,
            row_starts=shared_plan.row_starts,
            row_ends=shared_plan.row_ends,
            max_num_pools=shared_plan.max_num_pools,
        )
        return (
            selected_slots,
            selected_lens,
            shared_plan.causal_lens,
            max_seq_len,
        )

    def select_decode(
        self,
        *,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        positions_3d: torch.Tensor,
        ctx: ForwardContext,
        dsa_leaf: Hy4FlashDSABackend,
        num_reqs: int,
        q_len_per_req: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update decode index cache and return ``(topk_slots, topk_lens)``."""
        token_keys = self._project_token_keys(hidden_states)
        self.write_decode_cache(
            token_keys=token_keys,
            positions_3d=positions_3d,
            ctx=ctx,
            dsa_leaf=dsa_leaf,
            num_reqs=num_reqs,
            q_len_per_req=q_len_per_req,
        )
        query, weights = self._project_queries_and_weights(
            hidden_states, q_resid, positions_3d
        )
        metadata = dsa_leaf.forward_decode_metadata
        row_start = int(metadata.num_extends or 0)
        seq_lens = metadata.seq_lens_k[row_start : row_start + num_reqs].to(torch.int32)
        history_table = dsa_leaf.kpool_decode_page_table(row_start, num_reqs)
        pool = ctx.token_to_kv_pool
        index_cache = pool.get_index_k_buffer(self.layer_id)

        num_tokens = hidden_states.shape[0]
        out_width = self.index_topk + self.index_kpool - 1
        out_slots, out_lens = self._get_decode_workspaces(
            num_tokens,
            out_width,
            hidden_states.device,
            memory_saver_adapter=pool.arena.memory_saver_adapter,
        )
        kpool_decode_topk(
            query.to(torch.bfloat16).contiguous(),
            index_cache,
            weights.contiguous(),
            seq_lens,
            history_table,
            history_table,
            pool_size=self.index_kpool,
            page_size=index_cache.shape[1],
            kv_page_size=pool.arena.kv_page_size,
            topk_pools=self.group_topk,
            softmax_scale=1.0,
            q_len_per_req=q_len_per_req,
            apply_relu=True,
            append_tail=True,
            max_seq_len=dsa_leaf.max_context_len,
            out=out_slots,
            lens_out=out_lens,
        )
        return out_slots, out_lens

    def forward_eager_mask(
        self,
        *,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        positions_3d: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Exact single-sequence eager reference matching ``HYV4FlashCompressedIndexer``."""
        seq_length = hidden_states.shape[0]
        if seq_length <= self.index_topk:
            return attention_mask
        token_keys = self._project_token_keys(hidden_states)
        ratio = self.index_kpool
        num_groups = seq_length // ratio
        grouped = token_keys[: num_groups * ratio].view(
            num_groups, ratio, self.head_dim
        )
        group_keys = self.pool_groups(grouped)
        group_pos = positions_3d[:, : num_groups * ratio : ratio]
        keys = self._rotate_trailing(group_keys, group_pos)

        query, weights = self._project_queries_and_weights(
            hidden_states, q_resid, positions_3d
        )
        # query: [T, H, D], keys: [G, D] -> scores: [T, H, G] -> logits: [T, G]
        scores = F.relu(torch.matmul(query.float(), keys.float().transpose(0, 1)))
        logits = torch.einsum("th,thg->tg", weights, scores)

        query_group_index = (
            torch.arange(seq_length, device=hidden_states.device) // ratio
        )
        group_ids = torch.arange(num_groups, device=hidden_states.device)
        logits = logits.masked_fill(
            group_ids.unsqueeze(0) >= query_group_index.unsqueeze(1), float("-inf")
        )
        complete_topk = min(self.group_topk - 1, num_groups)
        recalled_groups = torch.cat(
            (
                logits.topk(complete_topk, dim=-1).indices,
                query_group_index.unsqueeze(-1),
            ),
            dim=-1,
        )
        offsets = torch.arange(ratio, device=recalled_groups.device)
        recalled = (
            (recalled_groups.unsqueeze(-1) * ratio + offsets)
            .flatten(-2)
            .clamp(max=seq_length - 1)
        )
        unrecalled = torch.ones(
            (seq_length, seq_length), dtype=torch.bool, device=hidden_states.device
        )
        unrecalled.scatter_(-1, recalled, False)
        return attention_mask.masked_fill(
            unrecalled.unsqueeze(0), torch.finfo(attention_mask.dtype).min
        )


# ===----------------------------------------------------------------------=== #
# Hybrid Compressed-DSA / SWA-MLA Gated Attention
# ===----------------------------------------------------------------------=== #


class HYV4FlashAttention(nn.Module):
    """Gated MLA supporting both 32-head Compressed-DSA and 64-head SWA-MLA layers."""

    def __init__(
        self,
        config: HYV4FlashTextConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.mapping = mapping
        self.attention_type = config.layer_types[layer_id]
        self.is_indexed = self.attention_type == INDEXED_ATTENTION
        self.is_sliding = self.attention_type == SLIDING_ATTENTION
        if not (self.is_indexed or self.is_sliding):
            raise ValueError(
                f"Layer {layer_id}: unsupported Hy4-flash attention_type={self.attention_type!r}"
            )

        self.num_heads = int(config.layer_num_attention_heads(layer_id))
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = int(config.v_head_dim)
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.scaling = self.qk_head_dim ** (-0.5)

        tp_rank = mapping.attn.tp_rank
        tp_size = mapping.attn.tp_size
        tp_group = mapping.attn.tp_group
        if self.num_heads % tp_size != 0:
            raise ValueError(
                f"Layer {layer_id}: num_heads={self.num_heads} not divisible by attn tp_size={tp_size}"
            )
        self.num_local_heads = self.num_heads // tp_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        self.q_a_proj = ReplicatedLinear(
            config.hidden_size,
            self.q_lora_rank,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("q_a_proj", prefix),
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("q_b_proj", prefix),
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("kv_a_proj_with_mqa", prefix),
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("kv_b_proj", prefix),
        )
        self.linear_gate = ColumnParallelLinear(
            config.hidden_size,
            self.num_heads * self.v_head_dim,
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("linear_gate", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            reduce_results=True,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("o_proj", prefix),
        )
        self.post_proj_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.learnable_sink_param = nn.Parameter(
            torch.full(
                (self.num_local_heads,),
                config.learnable_sink_init,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.learnable_sink_param.weight_loader = sharded_weight_loader(0, tp_rank)

        rope_params = config.rope_parameters[self.attention_type]
        self.rope_theta = float(rope_params["rope_theta"])
        self.mrope_section = tuple(int(s) for s in rope_params["mrope_section"])
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.qk_rope_head_dim, 2, dtype=torch.float32)
                / self.qk_rope_head_dim
            )
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        window_left = (
            hf_sliding_window_to_window_left(config.sliding_window)
            if self.is_sliding
            else -1
        )
        self.window_left = window_left
        # Hy4-flash uses 3D interleaved M-RoPE; rotate Q/K before latent_prologue
        # because the fused MLA prologue kernel rejects multimodal RoPE.
        self.attn_mha = PagedAttention(
            self.num_local_heads,
            self.qk_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            sliding_window_size=window_left,
            rotary_emb=None,
            qk_norm=None,
        )
        self.attn_mqa = PagedAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            sliding_window_size=window_left,
            rotary_emb=None,
            qk_norm=None,
        )

        self.compressed_indexer: HYV4FlashCompressedIndexer | None = (
            HYV4FlashCompressedIndexer(
                config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("compressed_indexer", prefix),
            )
            if self.is_indexed
            else None
        )

        self.w_kc: torch.Tensor | None = None
        self.w_vc: torch.Tensor | None = None

    def prepare_absorbed_mla_weights(self) -> None:
        """Derive ``w_kc`` and ``w_vc`` from ``kv_b_proj`` for absorbed MLA decode/DSA."""
        absorb_mla_kv_b_proj(self)

    def _ensure_absorbed_weights(self) -> None:
        if self.w_kc is None or self.w_vc is None:
            self.prepare_absorbed_mla_weights()

    def _resolve_leaf(
        self, ctx: ForwardContext
    ) -> Hy4FlashDSABackend | Hy4FlashSWABackend:
        backend = ctx.attn_backend
        if isinstance(backend, Hy4FlashCacheGroupRouter):
            return backend.leaf_for(self.attn_mha)
        return backend

    def _absorb_query(
        self,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_absorbed_weights()
        q_view = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q_view.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        absorbed = torch.empty(
            (
                q_nope.shape[0],
                self.num_local_heads,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
        if q_nope.shape[0] > 0:
            bmm(
                q_nope.transpose(0, 1),
                self.w_kc.transpose(1, 2),
                out=absorbed[..., : self.kv_lora_rank].transpose(0, 1),
            )
        return absorbed, q_pe

    def _forward_dense_prefill(
        self,
        *,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        leaf: Hy4FlashDSABackend | Hy4FlashSWABackend,
        slots: torch.Tensor,
        window_left: int,
    ) -> torch.Tensor:
        """Run dense expanded MLA prefill with chunked prefix replay and a single post-merge sink."""
        num_tokens = q.shape[0]
        q_view = q.view(num_tokens, self.num_local_heads, self.qk_head_dim)
        kv = self.kv_b_proj(latent_cache[..., : self.kv_lora_rank].contiguous())[0]
        kv = kv.view(
            num_tokens, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        expanded = self.attn_mha.latent_prologue(
            q_view,
            q_view[..., self.qk_nope_head_dim :],
            latent_cache,
            positions,
            ctx,
            slots=slots,
            expanded=MLAExpandedKV(k_nope=k_nope, value=v),
            key_rows=None,
        )

        chunk_meta = leaf.chunked_prefill_metadata
        assert chunk_meta is not None
        sinks = self.learnable_sink_param
        output = torch.empty(
            (num_tokens, self.num_local_heads * self.v_head_dim),
            dtype=expanded.query.dtype,
            device=expanded.query.device,
        )
        output_view = output.view(num_tokens, self.num_local_heads, self.v_head_dim)
        _, accum_lse = leaf.forward_extend_chunked(
            expanded.query,
            expanded.key,
            expanded.value,
            self.scaling,
            self.attn_mha.logit_cap,
            cum_seq_lens_q=chunk_meta.cum_extend_seq_lens,
            cum_seq_lens_kv=chunk_meta.cum_extend_seq_lens,
            max_q_len=chunk_meta.max_extend_seq_len,
            max_kv_len=chunk_meta.max_extend_seq_len,
            seq_lens=chunk_meta.extend_seq_lens,
            batch_size=chunk_meta.extend_seq_lens.shape[0],
            causal=True,
            out=output_view,
            window_left=window_left,
        )
        if chunk_meta.max_extend_prefix_len:
            prefix_offsets = (
                chunk_meta.prefix_query_start_offsets
                if isinstance(chunk_meta, Hy4FlashSWAPrefillMetadata)
                else None
            )
            if window_left >= 0 and prefix_offsets is None:
                raise RuntimeError(
                    "Hy4-flash SWA prefix extend requires Hy4FlashSWAPrefillMetadata"
                )
            read_dtype = (
                expanded.query.dtype
                if expanded.query.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
                else torch.bfloat16
            )
            history = leaf.read_prefix_history(
                ctx.token_to_kv_pool,
                self.attn_mha,
                chunk_meta.chunk_kv_indices_list,
                dst_dtype=read_dtype,
            )
            for loop_idx, (kv_a_normed, k_pe) in enumerate(history):
                kv_prefix = self.kv_b_proj(kv_a_normed.squeeze(1))[0].view(
                    -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
                )
                k_prefix_nope, v_prefix = kv_prefix.split(
                    [self.qk_nope_head_dim, self.v_head_dim], dim=-1
                )
                k_prefix = torch.cat(
                    (k_prefix_nope, k_pe.expand(-1, self.num_local_heads, -1)), dim=-1
                )
                chunk_out, chunk_lse = leaf.forward_extend_chunked(
                    expanded.query,
                    k_prefix,
                    v_prefix,
                    self.scaling,
                    self.attn_mha.logit_cap,
                    cum_seq_lens_q=chunk_meta.cum_extend_seq_lens,
                    cum_seq_lens_kv=chunk_meta.cu_chunked_seq_len[loop_idx],
                    max_q_len=chunk_meta.max_extend_seq_len,
                    max_kv_len=chunk_meta.max_chunk_len_per_loop[loop_idx],
                    seq_lens=chunk_meta.chunked_seq_len[loop_idx],
                    batch_size=chunk_meta.extend_seq_lens.shape[0],
                    causal=False,
                    window_left=window_left,
                    query_start_offsets=(
                        prefix_offsets[loop_idx] if prefix_offsets is not None else None
                    ),
                )
                attn_merge_state(
                    output_view,
                    accum_lse,
                    chunk_out,
                    chunk_lse,
                    inplace=True,
                )
        apply_attention_sink(output_view, accum_lse, sinks, inplace=True)
        return output

    def _forward_dsa_prefill(
        self,
        *,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        positions: torch.Tensor,
        positions_3d: torch.Tensor,
        ctx: ForwardContext,
        leaf: Hy4FlashDSABackend,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """Run DSA prefill: dense MLA when context <= ``index_topk``, sparse otherwise."""
        assert self.compressed_indexer is not None
        selected = self.compressed_indexer.select_prefill(
            hidden_states=hidden_states,
            q_resid=q_resid,
            positions_3d=positions_3d,
            ctx=ctx,
            dsa_leaf=leaf,
            num_prefill_tokens=hidden_states.shape[0],
        )
        num_tokens = q.shape[0]
        sinks = self.learnable_sink_param
        if selected is None:
            if isinstance(ctx.attn_backend, Hy4FlashCacheGroupRouter):
                ctx.attn_backend.clear_shared_topk_slots_for_layer(self.layer_id)
            return self._forward_dense_prefill(
                q=q,
                latent_cache=latent_cache,
                positions=positions,
                ctx=ctx,
                leaf=leaf,
                slots=slots,
                window_left=-1,
            )

        selected_slots, topk_lens, kv_seq_lens, max_seq_len = selected
        if isinstance(ctx.attn_backend, Hy4FlashCacheGroupRouter):
            ctx.attn_backend.record_shared_topk_slots(self.layer_id, selected_slots)

        absorbed_q, q_pe = self._absorb_query(q)
        rotated_q = self.attn_mqa.latent_prologue(
            absorbed_q,
            q_pe,
            latent_cache,
            positions,
            ctx,
            slots=slots,
            expanded=None,
            key_rows=None,
        ).query
        attn_latent = leaf.forward_sparse_prefill(
            q=rotated_q,
            layer=self.attn_mqa,
            token_to_kv_pool=ctx.token_to_kv_pool,
            kv_seq_lens=kv_seq_lens,
            topk_slots=selected_slots,
            topk_lens=topk_lens,
            max_seq_len=max_seq_len,
            sinks=sinks,
        )
        output = torch.empty(
            (num_tokens, self.num_local_heads * self.v_head_dim),
            dtype=rotated_q.dtype,
            device=rotated_q.device,
        )
        return mla_project_value(
            attn_latent.view(num_tokens, self.num_local_heads, self.kv_lora_rank),
            self.w_vc,
            gate=None,
            out=output,
        )

    def _forward_decode(
        self,
        *,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        positions: torch.Tensor,
        positions_3d: torch.Tensor,
        ctx: ForwardContext,
        leaf: Hy4FlashDSABackend | Hy4FlashSWABackend,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """Run absorbed MLA decode (and multi-token verify) for DSA or SWA."""
        num_tokens = q.shape[0]
        num_reqs = ctx.bs
        q_len_per_req = num_tokens // max(num_reqs, 1)
        absorbed_q, q_pe = self._absorb_query(q)
        rotated_q = self.attn_mqa.latent_prologue(
            absorbed_q,
            q_pe,
            latent_cache,
            positions,
            ctx,
            slots=slots,
            expanded=None,
            key_rows=None,
        ).query

        decode_kwargs: dict[str, object] = {
            "sinks": self.learnable_sink_param,
        }
        if self.is_indexed:
            assert self.compressed_indexer is not None
            assert isinstance(leaf, Hy4FlashDSABackend)
            topk_slots, topk_lens = self.compressed_indexer.select_decode(
                hidden_states=hidden_states,
                q_resid=q_resid,
                positions_3d=positions_3d,
                ctx=ctx,
                dsa_leaf=leaf,
                num_reqs=num_reqs,
                q_len_per_req=q_len_per_req,
            )
            if isinstance(ctx.attn_backend, Hy4FlashCacheGroupRouter):
                ctx.attn_backend.record_shared_topk_slots(self.layer_id, topk_slots)
            decode_kwargs["topk_indices"] = topk_slots
            decode_kwargs["topk_lens"] = topk_lens

        attn_latent = leaf.forward_decode(
            rotated_q,
            None,
            None,
            self.attn_mqa,
            slots,
            ctx.token_to_kv_pool,
            num_reqs,
            **decode_kwargs,
        )
        output = torch.empty(
            (num_tokens, self.num_local_heads * self.v_head_dim),
            dtype=rotated_q.dtype,
            device=rotated_q.device,
        )
        return mla_project_value(
            attn_latent.view(num_tokens, self.num_local_heads, self.kv_lora_rank),
            self.w_vc,
            gate=None,
            out=output,
        )

    def forward_eager_single_seq(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Exact single-sequence eager forward for CPU golden parity verification."""
        seq_len = hidden_states.shape[0]
        positions_3d = _normalize_positions_3d(positions)
        gate_states = _linear_cpu_or_cuda(self.linear_gate, hidden_states).view(
            seq_len, self.num_local_heads, self.v_head_dim
        )
        q_resid = _apply_hy4_rmsnorm(
            _linear_cpu_or_cuda(self.q_a_proj, hidden_states),
            self.q_a_layernorm,
        )
        q_states = _linear_cpu_or_cuda(self.q_b_proj, q_resid).view(
            seq_len, self.num_local_heads, self.qk_head_dim
        )
        q_nope, q_rot = q_states.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = _linear_cpu_or_cuda(self.kv_a_proj_with_mqa, hidden_states)
        kv_pass, k_rot = compressed_kv.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pass = _apply_hy4_rmsnorm(kv_pass, self.kv_a_layernorm)

        cos, sin = compute_hy4_mrope_cos_sin(
            positions_3d, self.rope_inv_freq, self.mrope_section
        )
        cos_cast = cos.to(q_rot.dtype)
        sin_cast = sin.to(q_rot.dtype)
        q_rot = apply_interleaved_rope_last_dim(
            q_rot, cos_cast.unsqueeze(1), sin_cast.unsqueeze(1)
        )
        k_rot = apply_interleaved_rope_last_dim(k_rot, cos_cast, sin_cast)

        query = torch.cat((q_nope, q_rot), dim=-1).transpose(0, 1)  # [H, T, D]
        kv_exp = _linear_cpu_or_cuda(self.kv_b_proj, k_pass).view(
            seq_len, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v_states = kv_exp.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_rot_exp = k_rot.unsqueeze(1).expand(-1, self.num_local_heads, -1)
        key = torch.cat((k_nope, k_rot_exp), dim=-1).transpose(0, 1)  # [H, T, D]
        value = v_states.transpose(0, 1)  # [H, T, Dv]

        # Build causal (and optional sliding-window) additive mask.
        q_idx = torch.arange(seq_len, device=hidden_states.device).unsqueeze(1)
        k_idx = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        allowed = k_idx <= q_idx
        if self.is_sliding:
            allowed = allowed & ((q_idx - k_idx) <= self.window_left)
        mask = torch.zeros(
            (1, seq_len, seq_len), dtype=query.dtype, device=hidden_states.device
        )
        mask = mask.masked_fill(~allowed.unsqueeze(0), torch.finfo(query.dtype).min)

        if self.compressed_indexer is not None:
            mask = self.compressed_indexer.forward_eager_mask(
                hidden_states=hidden_states,
                q_resid=q_resid,
                positions_3d=positions_3d,
                attention_mask=mask,
            )

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scaling + mask
        sinks = (
            self.learnable_sink_param.to(attn_weights.dtype)
            .view(self.num_local_heads, 1, 1)
            .expand(-1, seq_len, 1)
        )
        attn_weights = torch.cat((attn_weights, sinks), dim=-1)
        attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
        probs = F.softmax(attn_weights, dim=-1, dtype=attn_weights.dtype)[..., :-1]
        attn_output = torch.matmul(probs.to(value.dtype), value).transpose(
            0, 1
        )  # [T, H, Dv]
        attn_output = (attn_output * torch.sigmoid(gate_states)).reshape(
            seq_len, self.num_local_heads * self.v_head_dim
        )
        out = _linear_cpu_or_cuda(self.o_proj, attn_output)
        return _apply_hy4_rmsnorm(out, self.post_proj_norm)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext | None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        if ctx is None or not hidden_states.is_cuda:
            return self.forward_eager_single_seq(hidden_states, positions)

        positions_3d = _normalize_positions_3d(positions)
        gate_states = self.linear_gate(hidden_states)[0]
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states)[0])
        q = self.q_b_proj(q_resid)[0]
        raw_latent = self.kv_a_proj_with_mqa(hidden_states)[0]
        cos, sin = compute_hy4_mrope_cos_sin(
            positions_3d, self.rope_inv_freq, self.mrope_section
        )
        cos_cast = cos.to(q.dtype)
        sin_cast = sin.to(q.dtype)
        q_view = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_view[..., self.qk_nope_head_dim :].copy_(
            apply_interleaved_rope_last_dim(
                q_view[..., self.qk_nope_head_dim :],
                cos_cast.unsqueeze(1),
                sin_cast.unsqueeze(1),
            )
        )
        k_rot = apply_interleaved_rope_last_dim(
            raw_latent[..., self.kv_lora_rank :],
            cos_cast,
            sin_cast,
        )
        latent_cache = torch.cat(
            (
                self.kv_a_layernorm(raw_latent[..., : self.kv_lora_rank]),
                k_rot,
            ),
            dim=-1,
        ).contiguous()

        leaf = self._resolve_leaf(ctx)
        router = ctx.attn_backend
        total_tokens = hidden_states.shape[0]
        if ctx.forward_mode.is_mixed():
            prefill_meta = leaf.chunked_prefill_metadata
            assert prefill_meta is not None
            num_prefill_tokens = int(
                prefill_meta.extend_seq_lens_cpu[: ctx.num_extends].sum().item()
            )
            num_decode_tokens = total_tokens - num_prefill_tokens
            num_decodes = ctx.bs - ctx.num_extends
            attn_output = torch.empty(
                (total_tokens, self.num_local_heads * self.v_head_dim),
                dtype=q.dtype,
                device=q.device,
            )
            if num_prefill_tokens > 0:
                prefill_ctx = replace(
                    ctx,
                    bs=ctx.num_extends,
                    num_extends=ctx.num_extends,
                    input_num_tokens=num_prefill_tokens,
                    forward_mode=ForwardMode.EXTEND,
                )
                prefill_slots = router.write_locations(
                    self.attn_mha, ForwardMode.EXTEND
                )
                if self.is_indexed:
                    assert isinstance(leaf, Hy4FlashDSABackend)
                    attn_output[:num_prefill_tokens] = self._forward_dsa_prefill(
                        hidden_states=hidden_states[:num_prefill_tokens],
                        q_resid=q_resid[:num_prefill_tokens],
                        q=q[:num_prefill_tokens],
                        latent_cache=latent_cache[:num_prefill_tokens],
                        positions=positions[..., :num_prefill_tokens],
                        positions_3d=positions_3d[:, :num_prefill_tokens],
                        ctx=prefill_ctx,
                        leaf=leaf,
                        slots=prefill_slots,
                    )
                else:
                    assert isinstance(leaf, Hy4FlashSWABackend)
                    attn_output[:num_prefill_tokens] = self._forward_dense_prefill(
                        q=q[:num_prefill_tokens],
                        latent_cache=latent_cache[:num_prefill_tokens],
                        positions=positions[..., :num_prefill_tokens],
                        ctx=prefill_ctx,
                        leaf=leaf,
                        slots=prefill_slots,
                        window_left=self.window_left,
                    )
            if num_decode_tokens > 0:
                decode_ctx = replace(
                    ctx,
                    bs=num_decodes,
                    num_extends=0,
                    input_num_tokens=num_decode_tokens,
                    forward_mode=ForwardMode.DECODE,
                )
                decode_slots = router.write_locations(self.attn_mqa, ForwardMode.DECODE)
                attn_output[num_prefill_tokens:] = self._forward_decode(
                    hidden_states=hidden_states[num_prefill_tokens:],
                    q_resid=q_resid[num_prefill_tokens:],
                    q=q[num_prefill_tokens:],
                    latent_cache=latent_cache[num_prefill_tokens:],
                    positions=positions[..., num_prefill_tokens:],
                    positions_3d=positions_3d[:, num_prefill_tokens:],
                    ctx=decode_ctx,
                    leaf=leaf,
                    slots=decode_slots,
                )
        elif ctx.forward_mode.is_decode():
            decode_slots = router.write_locations(self.attn_mqa, ForwardMode.DECODE)
            attn_output = self._forward_decode(
                hidden_states=hidden_states,
                q_resid=q_resid,
                q=q,
                latent_cache=latent_cache,
                positions=positions,
                positions_3d=positions_3d,
                ctx=ctx,
                leaf=leaf,
                slots=decode_slots,
            )
        else:
            extend_slots = router.write_locations(self.attn_mha, ForwardMode.EXTEND)
            if self.is_indexed:
                assert isinstance(leaf, Hy4FlashDSABackend)
                attn_output = self._forward_dsa_prefill(
                    hidden_states=hidden_states,
                    q_resid=q_resid,
                    q=q,
                    latent_cache=latent_cache,
                    positions=positions,
                    positions_3d=positions_3d,
                    ctx=ctx,
                    leaf=leaf,
                    slots=extend_slots,
                )
            else:
                assert isinstance(leaf, Hy4FlashSWABackend)
                attn_output = self._forward_dense_prefill(
                    q=q,
                    latent_cache=latent_cache,
                    positions=positions,
                    ctx=ctx,
                    leaf=leaf,
                    slots=extend_slots,
                    window_left=self.window_left,
                )

        attn_output = attn_output * torch.sigmoid(gate_states)
        out = self.o_proj(attn_output)[0]
        return self.post_proj_norm(out)


# ===----------------------------------------------------------------------=== #
# Dense MLP (Layer 0 & Shared Experts) and LatentMoE (Layers 1..41)
# ===----------------------------------------------------------------------=== #


class HYV4FlashMLP(nn.Module):
    """Plain SwiGLU MLP (no ``swiglu_limit``) with optional ``post_fc2_norm``."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mapping: Mapping,
        *,
        is_shared_expert: bool,
        has_post_norm: bool,
        rms_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if is_shared_expert:
            tp_rank = mapping.moe.tp_ep_rank
            tp_size = mapping.moe.tp_ep_size
            tp_group = mapping.moe.tp_ep_group
        else:
            tp_rank = mapping.dense.tp_rank
            tp_size = mapping.dense.tp_size
            tp_group = mapping.dense.tp_group

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=True,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = gated_activation(hidden_act, swiglu_limit=None)
        self.post_fc2_norm: RMSNorm | None = (
            RMSNorm(hidden_size, eps=rms_norm_eps) if has_post_norm else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.is_cuda:
            gate_up = self.gate_up_proj(hidden_states)[0]
            activated = self.act_fn(gate_up)
            out = self.down_proj(activated)[0]
        else:
            gate_up = _linear_cpu_or_cuda(self.gate_up_proj, hidden_states)
            gate, up = gate_up.chunk(2, dim=-1)
            out = _linear_cpu_or_cuda(self.down_proj, F.silu(gate) * up)
        if self.post_fc2_norm is not None:
            out = _apply_hy4_rmsnorm(out, self.post_fc2_norm)
        return out


class HYV4FlashTopkRouter(nn.Module):
    """Sigmoid + correction-bias grouped top-k router matching ``HYV4FlashTopkRouter``."""

    def __init__(self, config: HYV4FlashTextConfig) -> None:
        super().__init__()
        self.top_k = int(config.num_experts_per_tok)
        self.num_experts = int(config.num_local_experts)
        self.hidden_size = int(config.hidden_size)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.num_group = int(config.n_group)
        self.topk_group = int(config.topk_group)
        self.norm_topk_prob = bool(config.norm_topk_prob)

        self.weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size),
            requires_grad=False,
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = hidden_states.view(-1, self.hidden_size)
        router_logits = F.linear(flat.float(), self.weight.float())
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.float()
        if self.num_group > 1 and self.topk_group < self.num_group:
            experts_per_group = self.num_experts // self.num_group
            group_scores = (
                scores_for_choice.view(-1, self.num_group, experts_per_group)
                .topk(min(2, experts_per_group), dim=-1)
                .values.sum(dim=-1)
            )
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            ).indices
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1.0)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(-1, self.num_group, experts_per_group)
                .reshape(-1, self.num_experts)
            )
            scores_for_choice = scores_for_choice.masked_fill(
                ~score_mask.bool(), float("-inf")
            )
        topk_ids = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        ).indices
        topk_weights = scores.gather(1, topk_ids)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


class HYV4FlashLatentMoE(nn.Module):
    """Latent-space MoE block (ℓ-MoE) for Hy4-flash layers 1..41."""

    def __init__(
        self,
        config: HYV4FlashTextConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.mapping = mapping
        self.swiglu_limit = float(config.swiglu_limit)

        self.gate = HYV4FlashTopkRouter(config)
        self.routed_expert_down_proj = ReplicatedLinear(
            config.hidden_size,
            config.routed_expert_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("routed_expert_down_proj", prefix),
        )
        use_clamped_swiglu = (
            config.hidden_act == "silu" and config.swiglu_limit is not None
        )
        self.experts = MoELayer(
            top_k=config.num_experts_per_tok,
            num_experts=config.num_local_experts,
            hidden_size=config.routed_expert_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_id,
            prefix=add_prefix("experts", prefix),
            tp_rank=mapping.moe.tp_rank,
            tp_size=mapping.moe.tp_size,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
            activation="swiglu" if use_clamped_swiglu else config.hidden_act,
            activation_alpha=1.0 if use_clamped_swiglu else None,
            swiglu_limit=config.swiglu_limit if use_clamped_swiglu else None,
            routing_mode="precomputed_topk",
        )
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=True,
            prev_is_moe=False,
            dense_batch_invariant=False,
            query_sharded=False,
        )
        self.routed_expert_up_proj = ReplicatedLinear(
            config.routed_expert_hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("routed_expert_up_proj", prefix),
        )
        self.shared_experts = HYV4FlashMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            hidden_act=config.hidden_act,
            mapping=mapping,
            is_shared_expert=True,
            has_post_norm=False,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
            prefix=add_prefix("shared_experts", prefix),
        )
        self.combine_output_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _forward_experts_cpu(
        self,
        routed_in: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        w13 = self.experts.w13_weight
        w2 = self.experts.w2_weight
        if w13.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            block_size = tuple(self.experts.quant_config.weight_block_size)
            w13 = _dequantize_fp8_blockwise(
                w13, self.experts.w13_weight_scale_inv, block_size
            ).to(routed_in.dtype)
            w2 = _dequantize_fp8_blockwise(
                w2, self.experts.w2_weight_scale_inv, block_size
            ).to(routed_in.dtype)
        else:
            w13 = w13.to(routed_in.dtype)
            w2 = w2.to(routed_in.dtype)

        num_local = w13.shape[0]
        final = torch.zeros_like(routed_in)
        topk_ids_long = topk_ids.to(torch.int64)
        mask = F.one_hot(topk_ids_long, num_classes=num_local + 1).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_tensor in hit:
            expert_idx = int(expert_tensor[0].item())
            if expert_idx >= num_local:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            gate_up = F.linear(routed_in[token_idx], w13[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            current = F.linear(F.silu(gate) * up, w2[expert_idx])
            current = current * topk_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: ForwardContext | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        topk_weights, topk_ids = self.gate(hidden_states)
        routed_in = _linear_cpu_or_cuda(self.routed_expert_down_proj, hidden_states)
        if hidden_states.is_cuda:
            topk_output = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=None,
            )
            if ctx is not None:
                num_global_tokens, max_num_tokens_per_gpu = (
                    self.comm_manager.get_num_tokens(ctx)
                )
            else:
                num_tokens = int(routed_in.shape[0])
                num_global_tokens = num_tokens
                max_num_tokens_per_gpu = num_tokens
            routed_latent = self.experts(
                hidden_states=routed_in,
                topk_output=topk_output,
                num_global_tokens=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            )
            if (
                self.mapping.moe.tp_ep_size > 1
                and not self.experts.supports_all_to_all_ep
            ):
                routed_latent = all_reduce(
                    routed_latent, group=self.mapping.moe.tp_ep_group
                )
        else:
            routed_latent = self._forward_experts_cpu(routed_in, topk_ids, topk_weights)
        routed_out = _linear_cpu_or_cuda(self.routed_expert_up_proj, routed_latent)
        shared_out = self.shared_experts(hidden_states)
        return _apply_hy4_rmsnorm(routed_out + shared_out, self.combine_output_norm)


# ===----------------------------------------------------------------------=== #
# Decoder Layer & Text Backbone
# ===----------------------------------------------------------------------=== #


class HYV4FlashDecoderLayer(nn.Module):
    """Single Hy4-flash decoder layer: optional Engram + iHC-Attention + iHC-MLP/MoE."""

    def __init__(
        self,
        config: HYV4FlashTextConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.is_moe_layer = config.mlp_layer_types[layer_id] == "sparse"

        self.engram_layer: HYV4FlashEngramLayer | None = None
        if (
            config.engram_layer_ids is not None
            and (layer_id + 1) in config.engram_layer_ids
        ):
            self.engram_layer = HYV4FlashEngramLayer(
                config=config,
                layer_id=layer_id + 1,
                mapping=mapping,
            )

        self.hc_attn_layer = HYV4FlashHyperConnection(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = HYV4FlashAttention(
            config=config,
            layer_id=layer_id,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        self.hc_mlp_layer = HYV4FlashHyperConnection(config)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if self.is_moe_layer:
            self.mlp: nn.Module = HYV4FlashLatentMoE(
                config=config,
                layer_id=layer_id,
                mapping=mapping,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = HYV4FlashMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                mapping=mapping,
                is_shared_expert=False,
                has_post_norm=True,
                rms_norm_eps=config.rms_norm_eps,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_streams: torch.Tensor,
        ctx: ForwardContext | None,
        *,
        ngram_tokens: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.engram_layer is not None and ngram_tokens is not None:
            hidden_streams = hidden_streams + self.engram_layer(
                hidden_streams, ngram_tokens, engram_mask
            )

        attn_post, attn_in = self.hc_attn_layer.pre(hidden_streams)
        attn_in = _apply_hy4_rmsnorm(attn_in, self.input_layernorm)
        attn_out = self.self_attn(positions, attn_in, ctx)
        hidden_streams = self.hc_attn_layer.post(hidden_streams, attn_out, attn_post)

        mlp_post, mlp_in = self.hc_mlp_layer.pre(hidden_streams)
        mlp_in = _apply_hy4_rmsnorm(mlp_in, self.post_attention_layernorm)
        mlp_out = self.mlp(mlp_in, ctx=ctx) if self.is_moe_layer else self.mlp(mlp_in)
        hidden_streams = self.hc_mlp_layer.post(hidden_streams, mlp_out, mlp_post)
        return hidden_streams


class HYV4FlashTextModel(nn.Module):
    """Hy4-flash language model backbone (embedding + 42 iHC layers + hc_head + norm)."""

    def __init__(
        self,
        config: HYV4FlashTextConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.hc_mult = config.hc_mult
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                HYV4FlashDecoderLayer(
                    config=config,
                    layer_id=layer_id,
                    mapping=mapping,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.hc_head = HYV4FlashHyperHead(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext | None,
        input_embeds: torch.Tensor | None = None,
        request_token_history=None,
        ngram_tokens: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if ngram_tokens is None and self.config.engram_layer_ids:
            history = (
                request_token_history
                if request_token_history is not None
                else (ctx.request_token_history if ctx is not None else None)
            )
            ngram_tokens, engram_mask = build_hy4_engram_inputs(
                input_ids,
                positions,
                config=self.config,
                request_token_history=history,
            )

        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        hidden_streams = (
            hidden_states.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
        )
        for layer in self.layers:
            hidden_streams = layer(
                positions,
                hidden_streams,
                ctx,
                ngram_tokens=ngram_tokens,
                engram_mask=engram_mask,
            )
        collapsed = self.hc_head(hidden_streams)
        hidden_states = _apply_hy4_rmsnorm(collapsed, self.norm)

        aux_hidden_states: list[torch.Tensor] | None = None
        if (
            ctx is not None
            and ctx.capture_hidden_mode is not None
            and ctx.capture_hidden_mode.need_capture()
        ):
            aux_hidden_states = [hidden_states]
        return hidden_states, aux_hidden_states


# ===----------------------------------------------------------------------=== #
# Weight Loading & CausalLM / ConditionalGeneration Entry Points
# ===----------------------------------------------------------------------=== #


def _canonicalize_hy4_text_weight_name(name: str) -> str:
    """Normalize checkpoint prefixes and HF PR #50903 alias names to runtime layout."""
    if name.startswith("language_model.model."):
        name = name[len("language_model.model.") :]
    elif name.startswith("model.language_model."):
        name = name[len("model.language_model.") :]
        if not name.startswith("model.") and not name.startswith("lm_head."):
            name = f"model.{name}"
    elif name.startswith("language_model."):
        name = name[len("language_model.") :]

    # HF PR #50903 alias normalization (real released checkpoint uses canonical names).
    if ".attn_hc." in name:
        name = (
            name.replace(".attn_hc.fn", ".hc_attn_layer.hc_pre.hc_fn")
            .replace(".attn_hc.scale", ".hc_attn_layer.hc_pre.hc_scale")
            .replace(".attn_hc.base", ".hc_attn_layer.hc_pre.hc_base")
        )
    elif ".ffn_hc." in name:
        name = (
            name.replace(".ffn_hc.fn", ".hc_mlp_layer.hc_pre.hc_fn")
            .replace(".ffn_hc.scale", ".hc_mlp_layer.hc_pre.hc_scale")
            .replace(".ffn_hc.base", ".hc_mlp_layer.hc_pre.hc_base")
        )
    if name == "model.hc_head.hc_fn":
        name = "model.hc_head.hc_head_fn"
    elif name == "model.hc_head.hc_scale":
        name = "model.hc_head.hc_head_scale"
    elif name == "model.hc_head.hc_base":
        name = "model.hc_head.hc_head_base"
    if ".post_mlp_layernorm." in name:
        name = name.replace(".post_mlp_layernorm.", ".mlp.post_fc2_norm.")
    if ".self_attn.gate_proj." in name:
        name = name.replace(".self_attn.gate_proj.", ".self_attn.linear_gate.")
    if name.endswith(".self_attn.sinks"):
        name = name[: -len(".sinks")] + ".learnable_sink_param"
    if ".mlp.pre_expert_proj." in name:
        name = name.replace(".mlp.pre_expert_proj.", ".mlp.routed_expert_down_proj.")
    if ".mlp.post_expert_proj." in name:
        name = name.replace(".mlp.post_expert_proj.", ".mlp.routed_expert_up_proj.")
    return name


def load_hy4_flash_text_weights(
    owner: HYV4FlashForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load Hy4-flash text weights (BF16 / FP8 128x128 + BF16 scale_inv / Engram / 3D MoE)."""
    params_dict = dict(owner.named_parameters(remove_duplicate=False))
    buffers_dict = dict(owner.named_buffers(remove_duplicate=False))
    modules_dict = dict(owner.named_modules())

    expert_schema = ExpertCheckpointSchema(
        gate_proj_name="gate_proj",
        up_proj_name="up_proj",
        down_proj_name="down_proj",
        gate_up_fused_name="gate_up_proj",
    )
    moe_loader = build_moe_checkpoint_loader(
        params_dict=params_dict,
        expert_schema=expert_schema,
        fused_schema=expert_schema,
        num_experts=owner.config.num_local_experts,
        ep_rank=owner.mapping.moe.ep_rank,
        ep_size=owner.mapping.moe.ep_size,
        fused_gate_up_as_w13=True,
        fused_load_style="per_expert",
    )

    loaded_params: set[str] = set()
    engram_tables_updated: set[str] = set()

    for raw_name, loaded_weight in weights:
        name = _canonicalize_hy4_text_weight_name(raw_name)
        if (
            name.startswith(("model.dfly_layers.", "model.mtp_layers.", "vit."))
            or "rotary_emb.inv_freq" in name
            or "rotary_emb.original_inv_freq" in name
        ):
            continue

        # Released FP8 checkpoint stores blockwise scale_inv in BF16; kernels expect FP32.
        if name.endswith("_scale_inv") and loaded_weight.dtype != torch.float32:
            loaded_weight = loaded_weight.to(torch.float32)

        # Engram giant embedding table: supports TP row sharding and --engram-host-table.
        if name.endswith(".engram_layer.multi_head_embedding.embedding.weight"):
            engram_mod_name = name[: -len(".multi_head_embedding.embedding.weight")]
            engram_mod = modules_dict.get(engram_mod_name)
            if not isinstance(engram_mod, HYV4FlashEngramLayer):
                raise KeyError(f"Unexpected Engram embedding tensor {name!r}")
            embed_mod = engram_mod.multi_head_embedding
            if isinstance(loaded_weight, torch.Tensor):
                embed_mod.weight_loader(embed_mod.embedding.weight, loaded_weight)
            else:
                embed_mod.load_sharded(loaded_weight)
            loaded_params.add(name)
            continue

        # Engram lookup_table buffer: copy and refresh deterministic hash multipliers.
        if name.endswith(".engram_layer.lookup_table"):
            engram_mod_name = name[: -len(".lookup_table")]
            engram_mod = modules_dict.get(engram_mod_name)
            if not isinstance(engram_mod, HYV4FlashEngramLayer):
                raise KeyError(f"Unexpected Engram lookup_table tensor {name!r}")
            engram_mod.lookup_table.copy_(
                loaded_weight.to(
                    device=engram_mod.lookup_table.device,
                    dtype=engram_mod.lookup_table.dtype,
                )
            )
            engram_tables_updated.add(engram_mod_name)
            loaded_params.add(name)
            continue

        # 3D fused routed experts (gate_up_proj / down_proj and their _scale_inv).
        if ".mlp.experts." in name and moe_loader.is_expert_checkpoint_weight(name):
            mapped = moe_loader.load(name, loaded_weight)
            loaded_params.add(mapped)
            continue

        # Dense Layer 0 MLP and MoE shared_experts pack gate_proj + up_proj into gate_up_proj.
        packed_hit = False
        for src_proj, shard_id in ((".gate_proj.", 0), (".up_proj.", 1)):
            if src_proj in name and (".mlp." in name):
                mapped_name = name.replace(src_proj, ".gate_up_proj.")
                param = params_dict.get(mapped_name)
                if param is not None:
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(mapped_name)
                    packed_hit = True
                    break
        if packed_hit:
            continue

        param = params_dict.get(name)
        if param is not None:
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
            continue

        buf = buffers_dict.get(name)
        if buf is not None:
            buf.copy_(loaded_weight.to(device=buf.device, dtype=buf.dtype))
            loaded_params.add(name)
            continue

        raise KeyError(
            f"Unrecognized Hy4-flash text checkpoint tensor: {raw_name!r} -> {name!r}"
        )

    for engram_mod_name in engram_tables_updated:
        engram_mod = modules_dict[engram_mod_name]
        assert isinstance(engram_mod, HYV4FlashEngramLayer)
        engram_mod.refresh_hash_multipliers()

    _load_missing_hy4_engram_tables(owner, modules_dict, loaded_params)
    return loaded_params


_VISION_WEIGHT_PREFIXES = ("vit.", "model.visual.")
_TEXT_FILTERED_WEIGHT_PREFIXES = (
    "model.dfly_layers.",
    "model.mtp_layers.",
    *_VISION_WEIGHT_PREFIXES,
)


def build_hy4_lm_head(
    *,
    config: HYV4FlashTextConfig | HYV4FlashDFlyConfig,
    mapping: Mapping,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> nn.Module:
    """Build the Hy4-flash LM head (FP32 unquantized when ``enable_lm_head_fp32``)."""
    head_dtype = torch.float32 if config.enable_lm_head_fp32 else None
    head_quant = None if config.enable_lm_head_fp32 else quant_config
    if mapping.attn.has_dp and not mapping.lm_head.has_tp:
        return ReplicatedLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            params_dtype=head_dtype,
            quant_config=head_quant,
            prefix=add_prefix("lm_head", prefix),
        )
    return ParallelLMHead(
        config.vocab_size,
        config.hidden_size,
        params_dtype=head_dtype,
        quant_config=head_quant,
        prefix=add_prefix("lm_head", prefix),
        tp_rank=mapping.lm_head.tp_rank,
        tp_size=mapping.lm_head.tp_size,
        tp_group=mapping.lm_head.tp_group,
    )


def _load_missing_hy4_engram_tables(
    owner: HYV4FlashForCausalLM,
    modules_dict: dict[str, nn.Module],
    loaded_params: set[str],
) -> None:
    """Stream any filtered-out giant Engram embedding tables on CPU via safetensors get_slice."""
    checkpoint_dir = owner._checkpoint_dir
    if checkpoint_dir is None:
        return
    pending: list[tuple[str, HYV4FlashEngramLayer]] = [
        (f"{mod_name}.multi_head_embedding.embedding.weight", mod)
        for mod_name, mod in modules_dict.items()
        if isinstance(mod, HYV4FlashEngramLayer)
        and f"{mod_name}.multi_head_embedding.embedding.weight" not in loaded_params
    ]
    if not pending:
        return

    from safetensors import safe_open
    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

    index_path = os.path.join(checkpoint_dir, SAFE_WEIGHTS_INDEX_NAME)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"Hy4-flash Engram slice load requires {SAFE_WEIGHTS_INDEX_NAME} in "
            f"{checkpoint_dir}"
        )
    with open(index_path) as handle:
        weight_map: dict[str, str] = json.load(handle)["weight_map"]

    for canonical_name, engram_mod in pending:
        mapped_key = None
        for candidate in (
            canonical_name,
            f"language_model.model.{canonical_name}",
            f"model.language_model.{canonical_name.removeprefix('model.')}",
        ):
            if candidate in weight_map:
                mapped_key = candidate
                break
        if mapped_key is None:
            raise KeyError(
                f"Missing Engram embedding tensor {canonical_name!r} in {index_path}"
            )
        shard_path = os.path.join(checkpoint_dir, weight_map[mapped_key])
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            engram_mod.multi_head_embedding.load_sharded(handle.get_slice(mapped_key))
        loaded_params.add(canonical_name)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _hy4_model_profile(hf_config, *, is_multimodal: bool) -> ModelProfile:
    text_config = (
        hf_config.get_text_config()
        if hasattr(hf_config, "get_text_config")
        else hf_config
    )
    engram_ids = getattr(text_config, "engram_layer_ids", None)
    has_engram = bool(engram_ids)
    max_ngram = int(getattr(text_config, "engram_max_ngram_size", 3))
    return ModelProfile(
        configure_attention=configure_dsa_attention,
        cache_family="hy4_flash",
        linear_attention=None,
        default_attention_backend="dsa",
        request_token_history=has_engram,
        request_token_history_lookback=max(0, max_ngram - 1) if has_engram else None,
        is_multimodal=is_multimodal,
        tokenizer_kwargs={},
        attention_instances_per_layer=1,
        numerics_envelopes=frozenset({"auto"}),
        create_attention_config=Hy4FlashAttentionConfig.generate,
    )


class HYV4FlashForCausalLM(BaseCausalLM):
    """Hy4-flash text-only CausalLM with optional FP32 LM head."""

    model_cls = HYV4FlashTextModel
    quant_module_name_replacements = _HY4_QUANT_REPLACEMENTS

    def __init__(
        self,
        config: HYV4FlashTextConfig | HYV4FlashConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        encoder_only: bool = False,
    ) -> None:
        text_config = (
            config.text_config if isinstance(config, HYV4FlashConfig) else config
        )
        if quant_config is not None:
            quant_config.apply_checkpoint_name_replacements(
                self.quant_module_name_replacements
            )
        super().__init__(
            text_config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
            encoder_only=encoder_only,
        )
        self._checkpoint_dir: str | None = None

    def bind_checkpoint_dir(self, checkpoint_dir: str) -> None:
        self._checkpoint_dir = checkpoint_dir

    @staticmethod
    def checkpoint_weight_name_filter(name: str) -> bool:
        """Skip embedded DFly, ViT, and giant Engram table tensors from the GPU iterator."""
        canonical = _canonicalize_hy4_text_weight_name(name)
        return not (
            canonical.startswith(_TEXT_FILTERED_WEIGHT_PREFIXES)
            or canonical.endswith(".engram_layer.multi_head_embedding.embedding.weight")
        )

    @classmethod
    def model_profile(cls, hf_config) -> ModelProfile:
        return _hy4_model_profile(hf_config, is_multimodal=False)

    def resolve_lm_head(
        self,
        config: HYV4FlashTextConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> nn.Module:
        if getattr(config, "tie_word_embeddings", False):
            return self.model.embed_tokens
        return build_hy4_lm_head(
            config=config,
            mapping=self.mapping,
            quant_config=quant_config,
            prefix=prefix,
        )

    def prepare_model_kwargs(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        kwargs: dict,
    ) -> dict:
        del ctx, input_ids
        model_kwargs = {}
        for key in (
            "input_embeds",
            "request_token_history",
            "ngram_tokens",
            "engram_mask",
        ):
            if key in kwargs:
                model_kwargs[key] = kwargs[key]
        return model_kwargs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = load_hy4_flash_text_weights(self, weights)
        self.post_load_weights()
        return loaded

    def post_load_weights(self) -> None:
        if self.model is None:
            return
        for layer in self.model.layers:
            layer.self_attn.prepare_absorbed_mla_weights()
            if (
                isinstance(layer.mlp, HYV4FlashLatentMoE)
                and layer.mlp.experts.w13_weight.is_cuda
            ):
                layer.mlp.experts.process_weights_after_loading(layer.mlp.experts)


class HYV4FlashForConditionalGeneration(nn.Module):
    """Top-level multimodal Hy4-flash entry class (SigLIP2 AnyRes ViT + Text Backbone)."""

    quant_module_name_replacements = _HY4_QUANT_REPLACEMENTS

    def __init__(
        self,
        config: HYV4FlashConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.is_multimodal_active = is_multimodal_active
        if quant_config is not None:
            quant_config.apply_checkpoint_name_replacements(
                self.quant_module_name_replacements
            )

        encoder_only = bool(getattr(config, "encoder_only", False))
        self.language_model: HYV4FlashForCausalLM | None = (
            None
            if encoder_only
            else HYV4FlashForCausalLM(
                config.text_config,
                mapping=mapping,
                quant_config=quant_config,
                prefix=prefix,
            )
        )
        if is_multimodal_active:
            del mm_attention_backend
            self.vit: HYV4FlashVisionTower | None = HYV4FlashVisionTower(
                config.vision_config
            )
            if self.language_model is not None:
                target_dtype = self.get_input_embeddings().weight.dtype
                self.vit = self.vit.to(dtype=target_dtype)
            self.vision_embedder: VisionEmbedder | None = VisionEmbedder(
                encoder_mapping=mapping.vision
            )
        else:
            self.vit = None
            self.vision_embedder = None

    @classmethod
    def model_profile(cls, hf_config) -> ModelProfile:
        return _hy4_model_profile(hf_config, is_multimodal=True)

    def get_multimodal_encoder_specs(self) -> dict[Modality, EncoderSpec]:
        if self.vit is None:
            return {}
        return {
            Modality.IMAGE: EncoderSpec(
                self.vit.embed_media,
                make_warmup_items=self.vit.make_image_warmup_items,
            ),
            Modality.VIDEO: EncoderSpec(
                self.vit.embed_media,
                make_warmup_items=self.vit.make_video_warmup_items,
            ),
        }

    def get_input_embeddings(self) -> nn.Module:
        if self.language_model is None:
            raise AttributeError(
                "HYV4FlashForConditionalGeneration encoder-only mode exposes no text embeddings"
            )
        return self.language_model.model.get_input_embeddings()

    def get_embed_and_head(self):
        if self.language_model is None:
            raise AttributeError(
                "HYV4FlashForConditionalGeneration encoder-only mode exposes no LM head"
            )
        return self.language_model.get_embed_and_head()

    @property
    def logits_processor(self):
        if self.language_model is None:
            raise AttributeError(
                "HYV4FlashForConditionalGeneration encoder-only mode exposes no logits processor"
            )
        return self.language_model.logits_processor

    @property
    def lm_head(self):
        if self.language_model is None:
            raise AttributeError(
                "HYV4FlashForConditionalGeneration encoder-only mode exposes no LM head"
            )
        return self.language_model.lm_head

    @torch.no_grad()
    def multimodal_input_embeds(
        self,
        input_ids: torch.Tensor,
        ctx: ForwardContext,
        multimodal_context,
    ) -> torch.Tensor | None:
        if (
            multimodal_context is None
            or self.vision_embedder is None
            or self.vit is None
            or not multimodal_context.has_extend_inputs()
            or ctx.forward_mode.is_decode_or_idle()
        ):
            return None
        input_embeds, model_kwargs = self.vision_embedder.apply(
            input_ids=input_ids,
            text_embedding=self.get_input_embeddings(),
            ctx=multimodal_context,
            encoders=self.get_multimodal_encoder_specs(),
            multimodal_model=self,
        )
        if model_kwargs:
            raise RuntimeError(
                "HYV4FlashForConditionalGeneration multimodal path must remain embedding-only"
            )
        return input_embeds

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.language_model is None:
            raise RuntimeError(
                "HYV4FlashForConditionalGeneration encoder-only mode cannot run language forward"
            )
        multimodal_context = kwargs.pop("multimodal_context", None)
        input_embeds = self.multimodal_input_embeds(input_ids, ctx, multimodal_context)
        if input_embeds is not None:
            kwargs["input_embeds"] = input_embeds
        return self.language_model.forward(
            ctx,
            input_ids,
            positions,
            **kwargs,
        )

    def bind_checkpoint_dir(self, checkpoint_dir: str) -> None:
        if self.language_model is not None:
            self.language_model.bind_checkpoint_dir(checkpoint_dir)

    def checkpoint_weight_name_filter(self, name: str) -> bool:
        """Keep active text/vision weights while streaming giant Engram tables on CPU."""
        if name.startswith(_VISION_WEIGHT_PREFIXES):
            return self.vit is not None
        if self.language_model is None:
            return False
        return self.language_model.checkpoint_weight_name_filter(name)

    def post_load_weights(self) -> None:
        if self.language_model is not None:
            self.language_model.post_load_weights()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Route ``vit.*`` weights to the vision tower and stream text weights."""
        vit_weights: list[tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()

        def text_stream() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if name.startswith(_VISION_WEIGHT_PREFIXES):
                    if self.vit is not None:
                        vit_weights.append((name, weight))
                    continue
                yield name, weight

        if self.language_model is not None:
            loaded.update(self.language_model.load_weights(text_stream()))
        else:
            for _ in text_stream():
                pass

        if self.vit is not None and vit_weights:
            loaded.update(self.vit.load_weights(vit_weights))
        return loaded


EntryClass = [HYV4FlashForConditionalGeneration, HYV4FlashForCausalLM]

__all__ = [
    "EntryClass",
    "HYV4FlashAttention",
    "HYV4FlashCompressedIndexer",
    "HYV4FlashDecoderLayer",
    "HYV4FlashForCausalLM",
    "HYV4FlashForConditionalGeneration",
    "HYV4FlashHyperConnection",
    "HYV4FlashHyperHead",
    "HYV4FlashHyperPre",
    "HYV4FlashLatentMoE",
    "HYV4FlashMLP",
    "HYV4FlashTextModel",
    "HYV4FlashTopkRouter",
    "build_hy4_lm_head",
    "load_hy4_flash_text_weights",
]
