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
# KPool Kernels
# ===-----------------------------------------------------------------------===#


def _validate_kpool_topk_inputs(
    q: torch.Tensor,
    pooled_k_cache: torch.Tensor,
    weights: torch.Tensor,
    pool_size: int,
    topk_pools: int,
    page_size: int,
) -> None:
    if q.dim() != 3 or q.dtype != torch.bfloat16:
        raise ValueError("KPool queries must be [tokens, heads, dim] in bfloat16")
    if weights.dim() != 2 or weights.shape != q.shape[:2]:
        raise ValueError("KPool weights must match the query token and head axes")
    head_dim = q.shape[2]
    if head_dim % 128 or pool_size <= 1 or topk_pools <= 0 or page_size <= 0:
        raise ValueError("invalid KPool head, pool, top-k, or page geometry")
    row_bytes = head_dim + head_dim // 128 * 4
    expected = (int(page_size), row_bytes)
    flat = pooled_k_cache.dim() == 2 and pooled_k_cache.shape[1] == math.prod(expected)
    rows = pooled_k_cache.dim() == 3 and pooled_k_cache.shape[1:] == expected
    if (
        not (flat or rows)
        or pooled_k_cache.stride(-1) != 1
        or pooled_k_cache.stride(0) % 4
    ):
        raise ValueError("invalid packed KPool cache geometry")


def kpool_prefill_write(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    write_slots: torch.Tensor,
    index_values: torch.Tensor,
    index_scales: torch.Tensor,
    ape: torch.Tensor,
) -> None:
    """Compress completed prefill pools directly into the index cache.

    Args:
        slot_k: Raw index keys shaped ``[rows, pool_size, head_dim]``.
        slot_score: Per-channel pool scores with the same shape as ``slot_k``.
        write_slots: Flattened physical index-cache slots.
        index_values: Paged FP8 pool values, updated in place.
        index_scales: Paged FP32 pool scales, updated in place.
        ape: Learned intra-pool bias shaped ``[pool_size, head_dim]``.
    Returns:
        None. Cache tensors are updated in place.
    """
    rows, pool_size, head_dim = slot_k.shape
    traits = {
        "head_dim": int(head_dim),
        "pool_size": int(pool_size),
        "index_k_format": "fp8_scaled",
        "rotate": True,
    }
    signature = _attention_format_signature(slot_k=slot_k)
    kernel = select_kernel(
        "attention",
        "kpool_prefill_write",
        signature,
        traits=traits,
    )
    shape_params = {
        "rows": int(rows),
        "pool_size": int(pool_size),
        "head_dim": int(head_dim),
        "index_rows_per_page": int(index_values.shape[1]),
    }
    ShapeCapture.get().record(
        "attention",
        "kpool_prefill_write",
        kernel.name,
        slot_k.dtype,
        shape_params,
    )
    with kernel_scope(
        "attention",
        "kpool_prefill_write",
        slot_k.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel(
            slot_k=slot_k,
            slot_score=slot_score,
            write_slots=write_slots,
            index_values=index_values,
            index_scales=index_scales,
            ape=ape,
        )


def kpool_prefill_tail_write(
    k: torch.Tensor,
    gate: torch.Tensor,
    tail_k: torch.Tensor,
    tail_gate: torch.Tensor,
    source_starts: torch.Tensor,
    destination_slots: torch.Tensor,
    destination_positions: torch.Tensor,
    valid_counts: torch.Tensor,
    *,
    pool_size: int,
) -> None:
    """Copy incomplete prefill pools into fixed request-local tail buffers.

    Args:
        k: Full index-key tensor shaped ``[tokens, head_dim]``.
        gate: Per-channel scores with the same shape as ``k``.
        tail_k: Request-local key ring, updated in place.
        tail_gate: Request-local score ring, updated in place.
        source_starts: First source token for each fixed metadata row.
        destination_slots: Stable request-tail slot for each metadata row.
        destination_positions: First logical destination position per row.
        valid_counts: Live token count per row. Zero marks graph padding.
        pool_size: Maximum number of tokens copied by one metadata row.

    Returns:
        None. Cache tensors are updated in place.
    """
    head_dim = k.shape[1]
    traits = {
        "head_dim": int(head_dim),
        "pool_size": int(pool_size),
    }
    signature = _attention_format_signature(k=k)
    kernel = select_kernel(
        "attention",
        "kpool_prefill_tail_write",
        signature,
        traits=traits,
    )
    shape_params = {
        "tokens": int(k.shape[0]),
        "rows": int(source_starts.numel()),
        "pool_size": int(pool_size),
        "tail_size": int(tail_k.shape[1]),
        "head_dim": int(head_dim),
    }
    ShapeCapture.get().record(
        "attention",
        "kpool_prefill_tail_write",
        kernel.name,
        k.dtype,
        shape_params,
    )
    with kernel_scope(
        "attention",
        "kpool_prefill_tail_write",
        k.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel(
            k=k,
            gate=gate,
            tail_k=tail_k,
            tail_gate=tail_gate,
            source_starts=source_starts,
            destination_slots=destination_slots,
            destination_positions=destination_positions,
            valid_counts=valid_counts,
            pool_size=pool_size,
        )


def kpool_decode_append(
    k: torch.Tensor,
    gate: torch.Tensor,
    tail_k: torch.Tensor,
    tail_gate: torch.Tensor,
    seq_lens: torch.Tensor,
    request_slots: torch.Tensor,
    index_block_table: torch.Tensor,
    index_values: torch.Tensor,
    index_scales: torch.Tensor,
    ape: torch.Tensor,
) -> None:
    """Append a decode/verify window to request-local tails and paged indices.

    Args:
        k: Index keys shaped ``[requests, steps, head_dim]``.
        gate: Per-channel pool scores with the same shape as ``k``.
        tail_k: Request-local raw KPool key tails, updated in place.
        tail_gate: Request-local raw KPool score tails, updated in place.
        seq_lens: Final sequence lengths after the decode window.
        request_slots: Stable request-pool row for each batch request.
        index_block_table: Logical-pool-page to physical-index-page table.
        index_values: Paged FP8 pooled values, updated in place.
        index_scales: Paged FP32 pooled-row scales, updated in place.
        ape: Learned intra-pool position bias.

    Returns:
        None. All cache outputs are written in place.
    """
    requests, steps, head_dim = k.shape
    pool_size = ape.shape[0]
    tail_size = tail_k.shape[1]
    traits = {
        "head_dim": int(head_dim),
        "pool_size": int(pool_size),
        "index_k_format": "fp8_scaled",
        "rotate": True,
    }
    signature = _attention_format_signature(k=k)
    kernel = select_kernel(
        "attention",
        "kpool_decode_append",
        signature,
        traits=traits,
    )
    shape_params = {
        "requests": int(requests),
        "steps": int(steps),
        "pool_size": int(pool_size),
        "tail_size": int(tail_size),
        "head_dim": int(head_dim),
        "index_rows_per_page": int(index_values.shape[1]),
    }
    ShapeCapture.get().record(
        "attention",
        "kpool_decode_append",
        kernel.name,
        k.dtype,
        shape_params,
    )
    with kernel_scope(
        "attention",
        "kpool_decode_append",
        k.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            k=k,
            gate=gate,
            tail_k=tail_k,
            tail_gate=tail_gate,
            seq_lens=seq_lens,
            request_slots=request_slots,
            index_block_table=index_block_table,
            index_values=index_values,
            index_scales=index_scales,
            ape=ape,
        )


def kpool_decode_topk(
    q: torch.Tensor,
    pooled_k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    index_block_table: torch.Tensor,
    kv_block_table: torch.Tensor,
    *,
    pool_size: int,
    page_size: int,
    kv_page_size: int,
    topk_pools: int,
    softmax_scale: float,
    q_len_per_req: int = 1,
    apply_relu: bool = True,
    append_tail: bool = True,
    chunk_pools: int = 8192,
    max_seq_len: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select pooled decode candidates and map them to a FlatKV layout.

    Args:
        q: Indexer queries shaped ``[tokens, heads, head_dim]``.
        pooled_k_cache: Paged compressed index-key cache.
        weights: Per-token indexer head weights.
        seq_lens: Raw-token history length for each request.
        index_block_table: Page table for ``pooled_k_cache``.
        kv_block_table: Page table for the raw FlatKV cache.
        pool_size: Number of raw tokens represented by a pooled row.
        page_size: Number of pooled rows per index-cache page.
        kv_page_size: Number of raw tokens per FlatKV page.
        topk_pools: Number of completed pools to select.
        softmax_scale: Scale applied to indexer scores.
        q_len_per_req: Query rows per request.
        apply_relu: Whether to apply ReLU before top-k selection.
        append_tail: Whether to append visible partial-pool tokens.
        chunk_pools: Pools scored per bounded portable reduction window.
        max_seq_len: Optional static context bound for CUDA Graph replay.
        out: Optional global-slot output buffer.
        lens_out: Optional selected-length output buffer.

    Returns:
        Expanded FlatKV indices and raw-token counts.
    """
    _validate_kpool_topk_inputs(
        q, pooled_k_cache, weights, pool_size, topk_pools, page_size
    )
    tokens, num_heads, head_dim = q.shape
    traits = {
        "head_dim": int(head_dim),
        "pool_size": int(pool_size),
        "page_size": int(page_size),
        "index_k_format": "fp8_scaled",
        "score_activation": "relu" if apply_relu else "none",
        "topk_layout": "global_slots",
        "topk_pools": int(topk_pools),
        "q_len_per_req": int(q_len_per_req),
    }
    signature = _attention_format_signature(q=q)
    kernel = select_kernel(
        "attention",
        "kpool_decode_topk",
        signature,
        traits=traits,
    )
    shape_params = {
        "tokens": int(tokens),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "pool_size": int(pool_size),
        "topk_pools": int(topk_pools),
        "page_size": int(page_size),
        "kv_page_size": int(kv_page_size),
        "q_len_per_req": int(q_len_per_req),
    }
    ShapeCapture.get().record(
        "attention", "kpool_decode_topk", kernel.name, q.dtype, shape_params
    )
    with kernel_scope(
        "attention",
        "kpool_decode_topk",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            pooled_k_cache=pooled_k_cache,
            weights=weights,
            seq_lens=seq_lens,
            index_block_table=index_block_table,
            kv_block_table=kv_block_table,
            pool_size=pool_size,
            page_size=page_size,
            kv_page_size=kv_page_size,
            topk_pools=topk_pools,
            softmax_scale=softmax_scale,
            q_len_per_req=q_len_per_req,
            apply_relu=apply_relu,
            append_tail=append_tail,
            chunk_pools=chunk_pools,
            max_seq_len=max_seq_len,
            out=out,
            lens_out=lens_out,
        )


def kpool_prefill_topk(
    q: torch.Tensor,
    pooled_k_cache: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    index_block_table: torch.Tensor,
    kv_block_table: torch.Tensor,
    *,
    pool_size: int,
    page_size: int,
    kv_page_size: int,
    topk_pools: int,
    softmax_scale: float,
    apply_relu: bool = True,
    append_tail: bool = True,
    chunk_pools: int = 8192,
    req_ids: torch.Tensor | None = None,
    causal_lens: torch.Tensor | None = None,
    pool_workspace_slots: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
    max_num_pools: int | None = None,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select ragged-prefill pools and expand them to FlatKV slots.

    Args:
        q: Packed indexer queries shaped ``[tokens, heads, head_dim]``.
        pooled_k_cache: Paged compressed index-key cache.
        weights: Per-token indexer head weights.
        positions: Absolute position for every packed query token.
        query_start_loc: Packed query boundaries for each request.
        index_block_table: Page table for ``pooled_k_cache``.
        kv_block_table: Page table for the raw FlatKV cache.
        pool_size: Number of raw tokens represented by a pooled row.
        page_size: Number of pooled rows per index-cache page.
        kv_page_size: Number of raw tokens per FlatKV page.
        topk_pools: Number of completed pools to select.
        softmax_scale: Scale applied to indexer scores.
        apply_relu: Whether to apply ReLU before top-k selection.
        append_tail: Whether to append visible partial-pool tokens.
        chunk_pools: Number of pools scored per reduction window.
        req_ids: Optional precomputed request id for every query token.
        causal_lens: Optional precomputed visible raw-token length per query.
        pool_workspace_slots: Optional physical pooled-cache slots concatenated
            in request-major logical-pool order.
        row_starts: Optional inclusive workspace start per query token.
        row_ends: Optional exclusive workspace end per query token.
        max_num_pools: Host-known maximum completed-pool count, required with
            the optional prefill plan.
        max_logits_bytes: Optional temporary-logits memory cap for performant
            implementations.
        out: Optional expanded global-slot output buffer.
        lens_out: Optional selected-length output buffer.

    Returns:
        Expanded FlatKV indices and raw-token counts.
    """
    _validate_kpool_topk_inputs(
        q, pooled_k_cache, weights, pool_size, topk_pools, page_size
    )
    tokens, num_heads, head_dim = q.shape
    plan_parts = (
        req_ids,
        causal_lens,
        pool_workspace_slots,
        row_starts,
        row_ends,
        max_num_pools,
    )
    has_prefill_plan = all(part is not None for part in plan_parts)
    if any(part is not None for part in plan_parts) and not has_prefill_plan:
        raise ValueError(
            "KPool prefill plan requires req_ids, causal_lens, "
            "pool_workspace_slots, row_starts, row_ends, and max_num_pools together"
        )
    traits = {
        "index_heads": int(num_heads),
        "head_dim": int(head_dim),
        "pool_size": int(pool_size),
        "page_size": int(page_size),
        "index_k_format": "fp8_scaled",
        "score_activation": "relu" if apply_relu else "none",
        "topk_layout": "global_slots",
        "topk_pools": int(topk_pools),
        "prefill_plan": has_prefill_plan,
    }
    signature = _attention_format_signature(q=q)
    kernel = select_kernel(
        "attention",
        "kpool_prefill_topk",
        signature,
        traits=traits,
    )
    shape_params = {
        "tokens": int(tokens),
        "requests": int(query_start_loc.numel() - 1),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "pool_size": int(pool_size),
        "topk_pools": int(topk_pools),
        "page_size": int(page_size),
        "kv_page_size": int(kv_page_size),
        "workspace_rows": (
            0 if pool_workspace_slots is None else int(pool_workspace_slots.numel())
        ),
    }
    ShapeCapture.get().record(
        "attention", "kpool_prefill_topk", kernel.name, q.dtype, shape_params
    )
    with kernel_scope(
        "attention",
        "kpool_prefill_topk",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            pooled_k_cache=pooled_k_cache,
            weights=weights,
            positions=positions,
            query_start_loc=query_start_loc,
            index_block_table=index_block_table,
            kv_block_table=kv_block_table,
            pool_size=pool_size,
            page_size=page_size,
            kv_page_size=kv_page_size,
            topk_pools=topk_pools,
            softmax_scale=softmax_scale,
            apply_relu=apply_relu,
            append_tail=append_tail,
            chunk_pools=chunk_pools,
            req_ids=req_ids,
            causal_lens=causal_lens,
            pool_workspace_slots=pool_workspace_slots,
            row_starts=row_starts,
            row_ends=row_ends,
            max_num_pools=max_num_pools,
            max_logits_bytes=max_logits_bytes,
            out=out,
            lens_out=lens_out,
        )


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.kpool.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.kpool.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.kpool.deep_gemm  # noqa: E402,F401

# isort: on


__all__ = [
    "kpool_prefill_write",
    "kpool_prefill_tail_write",
    "kpool_decode_append",
    "kpool_decode_topk",
    "kpool_prefill_topk",
]
