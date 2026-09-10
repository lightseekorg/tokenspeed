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

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import torch
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry, Priority
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    select_kernel,
    spec_matches_traits,
)
from tokenspeed_kernel.signature import (
    MXFP8_BLOCK_SCALE,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

AttentionResult = torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]


# One UE8M0 scale per 32 consecutive head_dim elements (MXFP8).
MXFP8_ATTENTION_BLOCK_SCALE = MXFP8_BLOCK_SCALE


def _attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{role: dense_tensor_format(tensor.dtype) for role, tensor in roles.items()}
    )


def _mxfp8_attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{
            role: tensor_format(
                "mxfp8", tensor.dtype, scale=MXFP8_ATTENTION_BLOCK_SCALE
            )
            for role, tensor in roles.items()
        }
    )


def _blockscaled_signature_and_scales(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_scale: torch.Tensor | None,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
):
    """Pick dense vs MXFP8 signature and build the scale kwargs splat.

    q_scale selects the block-scaled path; k_scale/v_scale must accompany it.
    Returns (signature, scale_kwargs) for the paged-KV-cache entry points.
    """
    if q_scale is not None:
        assert (
            k_scale is not None and v_scale is not None
        ), "MXFP8 attention requires q_scale, k_scale, and v_scale together"
        signature = _mxfp8_attention_format_signature(
            q=q, k_cache=k_cache, v_cache=v_cache
        )
    else:
        signature = _attention_format_signature(q=q, k_cache=k_cache, v_cache=v_cache)
    return signature, dict(q_scale=q_scale, k_scale=k_scale, v_scale=v_scale)


LSE_LN = math.log2(math.e)


# ===-----------------------------------------------------------------------===#
# MSA Kernels
# ===-----------------------------------------------------------------------===#


def msa_decode_with_kvcache(
    q: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    index_k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    topk: int,
    page_size: int,
    index_scale: float,
    attention_scale: float,
    init_blocks: int,
    local_blocks: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    k_scale: float | torch.Tensor | None = None,
    v_scale: float | torch.Tensor | None = None,
    score_out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Run MSA decode against paged K/V and index-key caches.

    Args:
        q: Main queries shaped ``[tokens, local_heads, head_dim]``.
        index_q: Index queries shaped ``[tokens, local_groups, index_dim]``.
        index_k: Index keys for the current tokens shaped
            ``[tokens, index_dim]``.
        k_cache: Paged key cache shaped
            ``[pages, local_kv_heads, page_size, head_dim]``.
        v_cache: Paged value cache with the same shape as ``k_cache``.
        index_k_cache: Per-layer index-key cache shaped
            ``[slots, index_dim]``.
        slot_mapping: Cache slot for each current token.
        page_table: Logical-to-physical page table.
        cache_seqlens: Visible sequence lengths after the current tokens.
        topk: Number of sparse blocks selected for each index query.
        page_size: Number of cache tokens in each indexed block.
        index_scale: Scale applied to index scores.
        attention_scale: Scale applied to main attention scores.
        init_blocks: Leading blocks forced into the selected set.
        local_blocks: Recent blocks forced into the selected set.
        max_seqlen_q: Uniform query-token count per request.
        max_seqlen_k: Maximum KV length addressable through ``page_table``.
        k_scale: Optional scalar descale for an FP8 ``k_cache``; keys were
            divided by this scale before quantization. None means 1.0.
        v_scale: Optional scalar descale for an FP8 ``v_cache``, with the
            same convention as ``k_scale``.
        score_out: Optional caller-owned index-score buffer, pre-filled with
            ``-inf`` and reused across layers; forwarded to the kernel to avoid
            a per-layer allocation + fill. Ignored by kernels that do not
            accept it or when its shape does not match.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Attention output with the same shape and dtype as ``q``. The indexer
        stage also writes ``index_k`` into ``index_k_cache`` at
        ``slot_mapping``.
    """
    traits = {
        "head_dim": q.shape[-1],
        "index_head_dim": index_q.shape[-1],
        "page_size": page_size,
        "topk": topk,
    }
    signature = _attention_format_signature(
        q=q,
        index_q=index_q,
        index_k=index_k,
        k_cache=k_cache,
        v_cache=v_cache,
        index_k_cache=index_k_cache,
    )
    kernel = select_kernel(
        "attention",
        "msa_decode_with_kvcache",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": cache_seqlens.shape[0],
        "total_q": q.shape[0],
        "num_pages": k_cache.shape[0],
        "page_size": page_size,
        "max_pages_per_seq": page_table.shape[1],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k_cache.shape[1],
        "head_dim": q.shape[-1],
        "index_head_dim": index_q.shape[-1],
        "topk": topk,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "msa_decode_with_kvcache",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "msa_decode_with_kvcache",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            index_q=index_q,
            index_k=index_k,
            k_cache=k_cache,
            v_cache=v_cache,
            index_k_cache=index_k_cache,
            slot_mapping=slot_mapping,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            topk=topk,
            page_size=page_size,
            index_scale=index_scale,
            attention_scale=attention_scale,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            k_scale=k_scale,
            v_scale=v_scale,
            score_out=score_out,
            enable_pdl=pdl_enabled(),
        )


def msa_extend_with_kvcache(
    q: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    index_k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    topk: int,
    page_size: int,
    index_scale: float,
    attention_scale: float,
    init_blocks: int,
    local_blocks: int,
    seq_lens_cpu: Sequence[int],
    k_scale: float | torch.Tensor | None = None,
    v_scale: float | torch.Tensor | None = None,
    query_lens_cpu: Sequence[int] | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Run MSA extend against paged K/V and index-key caches.

    Args:
        q: Main queries shaped ``[total_q, local_heads, head_dim]``.
        index_q: Index queries shaped
            ``[total_q, local_groups, index_dim]``.
        index_k: Index keys for the current tokens shaped
            ``[total_q, index_dim]``.
        k_cache: Paged key cache shaped
            ``[pages, local_kv_heads, page_size, head_dim]``.
        v_cache: Paged value cache with the same shape as ``k_cache``.
        index_k_cache: Per-layer index-key cache shaped
            ``[slots, index_dim]``.
        slot_mapping: Cache slot for each current token.
        page_table: Logical-to-physical page table.
        cache_seqlens: Visible sequence lengths after the current tokens.
        cu_seqlens_q: Cumulative query lengths shaped ``[batch + 1]``.
        prefix_lens: Cached prefix length for each request.
        max_seqlen_q: Maximum query length in the batch.
        max_seqlen_k: Maximum visible KV length in the batch.
        topk: Number of sparse blocks selected for each index query.
        page_size: Number of cache tokens in each indexed block.
        index_scale: Scale applied to index scores.
        attention_scale: Scale applied to main attention scores.
        init_blocks: Leading blocks forced into the selected set.
        local_blocks: Recent blocks forced into the selected set.
        k_scale: Optional scalar descale for an FP8 ``k_cache``; keys were
            divided by this scale before quantization. None means 1.0.
        v_scale: Optional scalar descale for an FP8 ``v_cache``, with the
            same convention as ``k_scale``.
        query_lens_cpu: Optional host-side per-request new-token counts;
            with ``seq_lens_cpu`` this lets the indexer plan its fmha
            OnlyScore path without a device sync.
        seq_lens_cpu: Host-side per-request total sequence lengths.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Attention output with the same shape and dtype as ``q``. The indexer
        stage also writes ``index_k`` into ``index_k_cache`` at
        ``slot_mapping``.
    """
    traits = {
        "head_dim": q.shape[-1],
        "index_head_dim": index_q.shape[-1],
        "page_size": page_size,
        "topk": topk,
    }
    signature = _attention_format_signature(
        q=q,
        index_q=index_q,
        index_k=index_k,
        k_cache=k_cache,
        v_cache=v_cache,
        index_k_cache=index_k_cache,
    )
    kernel = select_kernel(
        "attention",
        "msa_extend_with_kvcache",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": cache_seqlens.shape[0],
        "total_q": q.shape[0],
        "num_pages": k_cache.shape[0],
        "page_size": page_size,
        "max_pages_per_seq": page_table.shape[1],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k_cache.shape[1],
        "head_dim": q.shape[-1],
        "index_head_dim": index_q.shape[-1],
        "topk": topk,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "msa_extend_with_kvcache",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "msa_extend_with_kvcache",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            index_q=index_q,
            index_k=index_k,
            k_cache=k_cache,
            v_cache=v_cache,
            index_k_cache=index_k_cache,
            slot_mapping=slot_mapping,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            prefix_lens=prefix_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            topk=topk,
            page_size=page_size,
            index_scale=index_scale,
            attention_scale=attention_scale,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            k_scale=k_scale,
            v_scale=v_scale,
            query_lens_cpu=query_lens_cpu,
            seq_lens_cpu=seq_lens_cpu,
        )


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.msa.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.msa.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.msa.triton  # noqa: E402,F401

# isort: on

__all__ = [
    "msa_decode_with_kvcache",
    "msa_extend_with_kvcache",
]
