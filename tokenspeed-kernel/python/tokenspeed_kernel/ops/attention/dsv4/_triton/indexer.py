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

"""Portable MXFP4 indexer over the existing DSV4 page-planar cache."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsa._triton.topk import triton_topk_from_logits
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@triton.jit
def _unpack_mxfp4(packed, dims, scales):
    nibble = (packed >> ((dims % 2) * 4)) & 15
    magnitude = nibble & 7
    bits = (magnitude.to(tl.uint32) << 22) + (126 << 23)
    values = tl.where(
        magnitude < 2, magnitude.to(tl.float32) * 0.5, bits.to(tl.float32, bitcast=True)
    )
    values = tl.where((nibble & 8) != 0, -values, values)
    return values * tl.exp2(scales.to(tl.float32) - 127.0)


@triton.jit
def _mxfp4_logits_kernel(
    q_ptr,
    q_scales_ptr,
    weights_ptr,
    cache_ptr,
    lengths_ptr,
    table_ptr,
    requests_ptr,
    starts_ptr,
    cu_ptr,
    logits_ptr,
    table_stride,
    table_cols,
    page_stride,
    num_pages,
    num_requests,
    max_candidates,
    HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PREFILL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    length = tl.load(lengths_ptr + token)
    request = token
    local = offsets
    valid = (offsets < max_candidates) & (offsets < length)
    if PREFILL:
        request = tl.load(requests_ptr + token)
        valid_request = (request >= 0) & (request < num_requests)
        begin = tl.load(cu_ptr + request, mask=valid_request, other=0)
        end = tl.load(cu_ptr + request + 1, mask=valid_request, other=0)
        local = tl.load(starts_ptr + token) - begin + offsets
        valid &= valid_request & (local >= 0) & (local < end - begin)
    page_column = local // PAGE_SIZE
    valid &= (page_column >= 0) & (page_column < table_cols)
    page = tl.load(
        table_ptr + request * table_stride + page_column, mask=valid, other=0
    ).to(tl.int64)
    valid &= (page >= 0) & (page < num_pages)
    row = local % PAGE_SIZE

    heads = tl.arange(0, HEADS)
    dims = tl.arange(0, 128)
    q_packed = tl.load(
        q_ptr + (token * HEADS + heads[:, None]) * 64 + dims[None, :] // 2
    )
    q_scales = tl.load(
        q_scales_ptr + (token * HEADS + heads[:, None]) * 4 + dims[None, :] // 32
    )
    query = _unpack_mxfp4(q_packed, dims[None, :], q_scales).to(tl.bfloat16)
    k_packed = tl.load(
        cache_ptr
        + page[None, :] * page_stride
        + row[None, :] * 64
        + dims[:, None] // 2,
        mask=valid[None, :],
        other=0,
    )
    k_scales = tl.load(
        cache_ptr
        + page[None, :] * page_stride
        + PAGE_SIZE * 64
        + row[None, :] * 4
        + dims[:, None] // 32,
        mask=valid[None, :],
        other=127,
    )
    keys = _unpack_mxfp4(k_packed, dims[:, None], k_scales).to(tl.bfloat16)
    products = tl.dot(query, keys)
    weights = tl.load(weights_ptr + token * HEADS + heads)
    scores = tl.sum(tl.maximum(products, 0.0) * weights[:, None], axis=0)
    tl.store(
        logits_ptr + token * max_candidates + offsets,
        tl.where(valid, scores, -float("inf")),
        mask=offsets < max_candidates,
    )


def _indexer_logits(
    index_q: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cache: torch.Tensor,
    lengths: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    max_candidates: int,
    cu_seq_lens: torch.Tensor | None,
    starts: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    q, scales = index_q
    if q.ndim != 3 or q.shape[1] not in (32, 64) or q.shape[2] != 64:
        raise ValueError("MXFP4 index_q must have shape [tokens, 32|64, 64]")
    if not q.is_cuda or q.dtype != torch.uint8 or not q.is_contiguous():
        raise ValueError("MXFP4 index_q must be contiguous uint8 on a GPU")
    if scales.shape != q.shape[:2] or scales.dtype != torch.int32:
        raise ValueError("MXFP4 query scales must be int32 [tokens, heads]")
    if weights.shape != q.shape[:2] or weights.dtype != torch.float32:
        raise ValueError("indexer weights must be float32 [tokens, heads]")
    if page_size != 64:
        raise ValueError("DSV4 MXFP4 indexer requires page_size=64")
    if (
        cache.ndim != 2
        or cache.dtype != torch.uint8
        or cache.shape[1] < page_size * 68
        or cache.stride(1) != 1
        or cache.stride(0) < page_size * 68
    ):
        raise ValueError("index_k_cache must be a uint8 page-planar MXFP4 cache")
    if (
        block_table.ndim != 2
        or block_table.dtype != torch.int32
        or block_table.stride(1) != 1
    ):
        raise ValueError(
            "block_table must be int32 [requests, pages] with contiguous rows"
        )
    if lengths.shape != (q.shape[0],) or lengths.dtype != torch.int32:
        raise ValueError("candidate lengths must be int32 [tokens]")
    tensors = [scales, weights, cache, lengths, block_table]
    requests = None
    if cu_seq_lens is not None:
        if (
            starts is None
            or starts.shape != (q.shape[0],)
            or starts.dtype != torch.int32
            or cu_seq_lens.dtype != torch.int32
            or cu_seq_lens.shape != (block_table.shape[0] + 1,)
        ):
            raise ValueError(
                "prefill requires int32 query starts and request boundaries"
            )
        tensors.extend((cu_seq_lens, starts))
    elif block_table.shape[0] < q.shape[0]:
        raise ValueError("decode block_table needs one row per query")
    if any(t.device != q.device for t in tensors):
        raise ValueError("all indexer tensors must share a device")
    if any(
        not t.is_contiguous()
        for t in tensors
        if t is not cache and t is not block_table
    ):
        raise ValueError("indexer queries and metadata must be contiguous")
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative")
    if cu_seq_lens is not None:
        requests = torch.searchsorted(cu_seq_lens[1:], starts, right=True).to(
            torch.int32
        )
    logits = torch.empty(
        (q.shape[0], max_candidates), device=q.device, dtype=torch.float32
    )
    if q.shape[0] and max_candidates:
        _mxfp4_logits_kernel[(q.shape[0], triton.cdiv(max_candidates, 32))](
            q,
            scales.view(torch.uint8),
            weights,
            cache,
            lengths,
            block_table,
            requests,
            starts,
            cu_seq_lens,
            logits,
            block_table.stride(0),
            block_table.shape[1],
            cache.stride(0),
            cache.shape[0],
            block_table.shape[0],
            max_candidates,
            HEADS=q.shape[1],
            PAGE_SIZE=page_size,
            PREFILL=cu_seq_lens is not None,
            BLOCK_N=32,
            num_warps=4,
            num_stages=1,
        )
    return logits, requests


def _select_topk(
    logits: torch.Tensor,
    topk: int,
    out: torch.Tensor | None,
    base_rows: torch.Tensor | None,
) -> torch.Tensor:
    selected = triton_topk_from_logits(logits, topk, enable_pdl=False)
    if base_rows is not None:
        selected = torch.where(selected >= 0, selected + base_rows[:, None], -1).to(
            torch.int32
        )
    if out is None:
        return selected
    if (
        out.ndim != 2
        or out.shape[0] < logits.shape[0]
        or out.shape[1] != topk
        or out.dtype != torch.int32
        or out.device != logits.device
    ):
        raise ValueError(
            "out must be int32 [at least tokens, topk] on the query device"
        )
    out = out[: logits.shape[0]]
    out.copy_(selected)
    return out


def _check_base_offsets(
    base_offsets: torch.Tensor | None, block_table: torch.Tensor
) -> None:
    if base_offsets is not None and (
        base_offsets.ndim != 1
        or base_offsets.numel() < block_table.shape[0]
        or base_offsets.dtype not in (torch.int32, torch.int64)
        or base_offsets.device != block_table.device
    ):
        raise ValueError(
            "block_table_base_offsets must contain one integer per table row on the same device"
        )


_SIGNATURE = format_signature(
    q=dense_tensor_format(torch.uint8),
    weights=dense_tensor_format(torch.float32),
    index_k_cache=dense_tensor_format(torch.uint8),
)
_TRAITS = {
    "index_heads": frozenset({32, 64}),
    "head_dim": frozenset({128}),
    "topk": frozenset({512, 1024, 2048}),
    "page_size": frozenset({64}),
    "index_k_format": frozenset({"mxfp4"}),
}


@register_kernel(
    "attention",
    "dsv4_prefill_topk",
    name="triton_dsv4_prefill_topk_mxfp4",
    solution="triton",
    signatures=frozenset({_SIGNATURE}),
    traits=_TRAITS,
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    priority=Priority.PORTABLE,
    tags={"portability", "mxfp4", "sparse"},
)
def triton_dsv4_prefill_topk_mxfp4(
    index_q: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    cu_seqlen_k_start: torch.Tensor,
    cu_seqlen_k_end: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    max_seqlen_k: int,
    index_k_format: str,
    block_table_base_offsets: torch.Tensor | None,
    gathered_k: tuple[torch.Tensor, torch.Tensor] | None,
    gather_workspace: tuple[torch.Tensor, torch.Tensor] | None,
    out: torch.Tensor | None,
) -> tuple[torch.Tensor, None]:
    """Implement the public dsv4_prefill_topk contract over cache pages.

    Args:
        index_q: Packed uint8 [tokens, heads, 64] queries and int32 [tokens,
            heads] scales (four E8M0 bytes per head).
        weights: FP32 [tokens, heads] weights with query scaling already folded in.
        index_k_cache: Uint8 page-planar MXFP4 key cache.
        block_table: Int32 request-to-physical-page table.
        cu_seq_lens: Cumulative retained key counts, one boundary per request.
        cu_seqlen_k_start: Inclusive packed candidate start for each query.
        cu_seqlen_k_end: Exclusive packed candidate end for each query.
        seq_lens: Candidate counts for each query.
        page_size: Indexer rows per page; must be 64.
        topk: Number of selected offsets; 512, 1024, or 2048.
        max_seqlen_k: Candidate bound used to size the score workspace.
        index_k_format: Must be "mxfp4".
        block_table_base_offsets: Optional logical base page per compact table row.
        gathered_k: Unused; this implementation reads the cache directly.
        gather_workspace: Unused; this implementation needs no gathered cache.
        out: Optional int32 [at least tokens, topk] destination.

    Returns:
        Selected offsets and None (no gathered cache to reuse). Entries without
        a candidate are -1; compact tables return absolute logical offsets.
    """
    del gathered_k, gather_workspace
    if index_k_format != "mxfp4":
        raise ValueError("Triton DSV4 indexer requires index_k_format='mxfp4'")
    _check_base_offsets(block_table_base_offsets, block_table)
    if (
        cu_seqlen_k_end.shape != seq_lens.shape
        or cu_seqlen_k_start.shape != seq_lens.shape
    ):
        raise ValueError(
            "prefill candidate starts, ends and lengths must have the same shape"
        )
    lengths = torch.minimum(seq_lens, cu_seqlen_k_end - cu_seqlen_k_start).clamp_min(0)
    logits, requests = _indexer_logits(
        index_q,
        weights,
        index_k_cache,
        lengths,
        block_table,
        page_size=page_size,
        max_candidates=max_seqlen_k,
        cu_seq_lens=cu_seq_lens,
        starts=cu_seqlen_k_start,
    )
    base_rows = None
    if block_table_base_offsets is not None:
        requests = requests.long().clamp_max(block_table.shape[0] - 1)
        base_rows = (
            block_table_base_offsets[requests] * page_size
            + cu_seqlen_k_start
            - cu_seq_lens[requests]
        )
    return _select_topk(logits, topk, out, base_rows), None


@register_kernel(
    "attention",
    "dsv4_decode_topk",
    name="triton_dsv4_decode_topk_mxfp4",
    solution="triton",
    signatures=frozenset({_SIGNATURE}),
    traits=_TRAITS,
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    priority=Priority.PORTABLE,
    tags={"portability", "mxfp4", "sparse"},
)
def triton_dsv4_decode_topk_mxfp4(
    index_q: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    max_context_len: int,
    plan: object,
    index_k_format: str,
    block_table_base_offsets: torch.Tensor | None,
    out: torch.Tensor | None,
    persistent_topk_workspace: torch.Tensor | None,
) -> torch.Tensor:
    """Implement the public dsv4_decode_topk contract with portable dot products.

    Args:
        index_q: Packed uint8 queries and int32 packed E8M0 query scales.
        weights: FP32 [tokens, heads] weights with query scaling folded in.
        index_k_cache: Uint8 page-planar MXFP4 key cache.
        context_lens: Int32 [tokens, 1] retained candidate counts.
        block_table: Int32 [tokens, pages] physical page table.
        page_size: Indexer rows per page; must be 64.
        topk: Number of selected offsets; 512, 1024, or 2048.
        max_context_len: Candidate bound used to size the score workspace.
        plan: Unused schedule tensor; this implementation needs no schedule.
        index_k_format: Must be "mxfp4".
        block_table_base_offsets: Optional logical base page per compact table row.
        out: Optional int32 [at least tokens, topk] destination.
        persistent_topk_workspace: Unused CUDA-specific workspace.

    Returns:
        Int32 selected logical offsets, padded with -1, aliasing out if supplied.
    """
    del plan, persistent_topk_workspace
    if index_k_format != "mxfp4":
        raise ValueError("Triton DSV4 indexer requires index_k_format='mxfp4'")
    _check_base_offsets(block_table_base_offsets, block_table)
    logits, _ = _indexer_logits(
        index_q,
        weights,
        index_k_cache,
        context_lens.reshape(-1),
        block_table,
        page_size=page_size,
        max_candidates=max_context_len,
        cu_seq_lens=None,
        starts=None,
    )
    base_rows = (
        None
        if block_table_base_offsets is None
        else block_table_base_offsets[: logits.shape[0]] * page_size
    )
    return _select_topk(logits, topk, out, base_rows)


@register_kernel(
    "attention",
    "dsv4_plan",
    name="triton_dsv4_plan",
    solution="triton",
    signatures=frozenset({format_signature()}),
    traits={"page_size": frozenset({64})},
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    priority=Priority.PORTABLE,
    tags={"portability", "cuda_graph"},
)
def triton_dsv4_plan(
    *, page_size: int, seq_lens_2d: torch.Tensor, out: object | None
) -> torch.Tensor:
    """Keep a stable plan tensor for the runtime's in-place metadata refresh.

    Args:
        page_size: Indexer rows per page; must be 64.
        seq_lens_2d: Int32 [tokens, 1] candidate lengths.
        out: Optional plan tensor to refresh in place.

    Returns:
        A copy of the lengths, aliasing out when supplied. The scorer itself
        reads the live lengths and does not require a separate schedule.
    """
    if page_size != 64:
        raise ValueError("DSV4 indexer requires page_size=64")
    if out is None:
        return seq_lens_2d.clone()
    if (
        not isinstance(out, torch.Tensor)
        or out.shape != seq_lens_2d.shape
        or out.dtype != seq_lens_2d.dtype
        or out.device != seq_lens_2d.device
    ):
        raise ValueError("DSV4 plan output must match seq_lens_2d")
    # Metadata refresh may run outside the inference context that allocated out.
    with torch.inference_mode():
        out.copy_(seq_lens_2d)
    return out
