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

"""MXFP4 DeepSeek V4 sparse-indexer kernels for AMD GFX950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
    _dsa_topk_indices,
)

_HEAD_DIM = 128
_PACKED_DIM = gl.constexpr(_HEAD_DIM // 2)
_SCALE_DIM = gl.constexpr(_HEAD_DIM // 32)
_PAGE_SIZE = 64
_ROW_BYTES = 68
_BLOCK_N = gl.constexpr(32)
_HEADS_PER_MFMA = gl.constexpr(16)
_CHUNK_N = 256
_SUPPORTED_TOPK = (512, 1024, 2048)

__all__ = [
    "gluon_dsv4_indexer_decode_topk_mxfp4_gfx950",
    "gluon_dsv4_indexer_prefill_topk_mxfp4_gfx950",
    "gluon_dsv4_plan_gfx950",
]


@gluon.constexpr_function
def _indexer_mfma_layouts(NUM_WARPS: gl.constexpr):
    mfma = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[1, NUM_WARPS],
    )
    dot_a = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=16)
    dot_b = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=16)
    a_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_a, [_HEADS_PER_MFMA, _SCALE_DIM])
    b_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_b, [_BLOCK_N, _SCALE_DIM])
    return mfma, dot_a, dot_b, a_scale, b_scale


@gluon.jit
def _load_query_group(
    q,
    q_scales,
    weights,
    token,
    head_base: gl.constexpr,
    stride_q_token,
    stride_q_head,
    stride_q_scale_token,
    stride_q_scale_head,
    stride_w_token,
    stride_w_head,
    mfma_layout: gl.constexpr,
    dot_a_layout: gl.constexpr,
    a_scale_layout: gl.constexpr,
):
    heads = gl.arange(0, _HEADS_PER_MFMA, layout=gl.SliceLayout(1, dot_a_layout))[
        :, None
    ]
    packed_dims = gl.arange(0, _PACKED_DIM, layout=gl.SliceLayout(0, dot_a_layout))[
        None, :
    ]
    q_values = gl.amd.cdna4.buffer_load(
        ptr=q,
        offsets=(
            token * stride_q_token + (head_base + heads) * stride_q_head + packed_dims
        ).to(gl.int32),
        cache=".cg",
    )

    scale_heads = gl.arange(
        0, _HEADS_PER_MFMA, layout=gl.SliceLayout(1, a_scale_layout)
    )[:, None]
    scale_groups = gl.arange(0, _SCALE_DIM, layout=gl.SliceLayout(0, a_scale_layout))[
        None, :
    ]
    scales = gl.amd.cdna4.buffer_load(
        ptr=q_scales,
        offsets=(
            token * stride_q_scale_token
            + (head_base + scale_heads) * stride_q_scale_head
            + scale_groups
        ).to(gl.int32),
        cache=".cg",
    )
    weight_heads = gl.arange(0, _HEADS_PER_MFMA, layout=gl.SliceLayout(1, mfma_layout))
    head_weights = gl.amd.cdna4.buffer_load(
        ptr=weights,
        offsets=(
            token * stride_w_token + (head_base + weight_heads) * stride_w_head
        ).to(gl.int32),
        cache=".cg",
    ).to(gl.float32)
    return q_values, scales, head_weights


@gluon.jit
def _candidate_page_rows(
    positions,
    valid,
    query_requests,
    query_starts,
    cu_seq_lens,
    block_table_base_offsets,
    block_table,
    token,
    block_table_stride,
    PAGE_SIZE: gl.constexpr,
    BLOCK_TABLE_COLS: gl.constexpr,
    IS_PREFILL: gl.constexpr,
    HAS_BASE_OFFSETS: gl.constexpr,
):
    if IS_PREFILL:
        request = gl.load(query_requests + token).to(gl.int32)
        packed_position = gl.load(query_starts + token).to(gl.int32) + positions
        logical_position = packed_position - gl.load(cu_seq_lens + request).to(gl.int32)
    else:
        request = token
        logical_position = positions
    logical_page = logical_position // PAGE_SIZE
    if IS_PREFILL and HAS_BASE_OFFSETS:
        logical_page -= gl.load(block_table_base_offsets + request).to(gl.int32)
    safe_logical_page = gl.minimum(gl.maximum(logical_page, 0), BLOCK_TABLE_COLS - 1)
    physical_page = gl.amd.cdna4.buffer_load(
        ptr=block_table,
        offsets=(request * block_table_stride + safe_logical_page).to(gl.int32),
        mask=valid,
        other=0,
    ).to(gl.int64)
    return physical_page, logical_position % PAGE_SIZE


@gluon.jit
def _score_query_group(
    query,
    query_scales,
    head_weights,
    index_k_cache,
    query_requests,
    query_starts,
    cu_seq_lens,
    block_table_base_offsets,
    block_table,
    token,
    tile_start,
    candidate_end,
    block_table_stride,
    page_stride_bytes,
    mfma_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    BLOCK_TABLE_COLS: gl.constexpr,
    IS_PREFILL: gl.constexpr,
    HAS_BASE_OFFSETS: gl.constexpr,
):
    packed_dims = gl.arange(0, _PACKED_DIM, layout=gl.SliceLayout(1, dot_b_layout))[
        :, None
    ]
    columns = gl.arange(0, _BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))[None, :]
    positions = tile_start + columns
    valid = positions < candidate_end
    pages, page_rows = _candidate_page_rows(
        positions,
        valid,
        query_requests,
        query_starts,
        cu_seq_lens,
        block_table_base_offsets,
        block_table,
        token,
        block_table_stride,
        PAGE_SIZE,
        BLOCK_TABLE_COLS,
        IS_PREFILL,
        HAS_BASE_OFFSETS,
    )
    key_offsets = (
        pages * page_stride_bytes + page_rows.to(gl.int64) * _PACKED_DIM + packed_dims
    )
    key = gl.load(index_k_cache + key_offsets, mask=valid, other=0)

    scale_columns = gl.arange(0, _BLOCK_N, layout=gl.SliceLayout(1, b_scale_layout))[
        :, None
    ]
    scale_groups = gl.arange(0, _SCALE_DIM, layout=gl.SliceLayout(0, b_scale_layout))[
        None, :
    ]
    scale_positions = tile_start + scale_columns
    scale_valid = scale_positions < candidate_end
    scale_pages, scale_page_rows = _candidate_page_rows(
        scale_positions,
        scale_valid,
        query_requests,
        query_starts,
        cu_seq_lens,
        block_table_base_offsets,
        block_table,
        token,
        block_table_stride,
        PAGE_SIZE,
        BLOCK_TABLE_COLS,
        IS_PREFILL,
        HAS_BASE_OFFSETS,
    )
    key_scale_offsets = (
        scale_pages * page_stride_bytes
        + PAGE_SIZE * _PACKED_DIM
        + scale_page_rows.to(gl.int64) * _SCALE_DIM
        + scale_groups
    )
    key_scales = gl.load(
        index_k_cache + key_scale_offsets,
        mask=scale_valid,
        other=127,
    )

    accumulator = gl.zeros(
        [_HEADS_PER_MFMA, _BLOCK_N], dtype=gl.float32, layout=mfma_layout
    )
    head_scores = gl.amd.cdna4.mfma_scaled(
        a=query,
        a_scale=query_scales,
        a_format="e2m1",
        b=key,
        b_scale=key_scales,
        b_format="e2m1",
        acc=accumulator,
    )
    head_scores = gl.maximum(
        head_scores,
        0.0,
        propagate_nan=tl.PropagateNan.ALL,
    )
    return gl.sum(head_scores * head_weights[:, None], axis=0)


@gluon.jit(
    do_not_specialize=(
        "stride_q_token",
        "stride_q_head",
        "stride_q_scale_token",
        "stride_q_scale_head",
        "stride_w_token",
        "stride_w_head",
        "block_table_stride",
        "logits_stride",
        "page_stride_bytes",
    )
)
def _dsv4_mxfp4_logits_kernel(
    q,
    q_scales,
    weights,
    index_k_cache,
    lengths,
    query_requests,
    query_starts,
    cu_seq_lens,
    block_table_base_offsets,
    block_table,
    logits,
    stride_q_token,
    stride_q_head,
    stride_q_scale_token,
    stride_q_scale_head,
    stride_w_token,
    stride_w_head,
    block_table_stride,
    logits_stride,
    page_stride_bytes,
    max_candidates,
    NUM_HEADS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    BLOCK_TABLE_COLS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    IS_PREFILL: gl.constexpr,
    HAS_BASE_OFFSETS: gl.constexpr,
):
    token = gl.program_id(0)
    split = gl.program_id(1)
    candidate_start = split * CHUNK_N
    candidate_end = gl.minimum(gl.load(lengths + token), max_candidates)
    candidate_end = gl.minimum(candidate_end, candidate_start + CHUNK_N)
    if candidate_start >= candidate_end:
        return

    layouts: gl.constexpr = _indexer_mfma_layouts(NUM_WARPS)
    mfma_layout: gl.constexpr = layouts[0]
    dot_a_layout: gl.constexpr = layouts[1]
    dot_b_layout: gl.constexpr = layouts[2]
    a_scale_layout: gl.constexpr = layouts[3]
    b_scale_layout: gl.constexpr = layouts[4]
    query_0, query_scale_0, weight_0 = _load_query_group(
        q,
        q_scales,
        weights,
        token,
        0,
        stride_q_token,
        stride_q_head,
        stride_q_scale_token,
        stride_q_scale_head,
        stride_w_token,
        stride_w_head,
        mfma_layout,
        dot_a_layout,
        a_scale_layout,
    )
    query_1, query_scale_1, weight_1 = _load_query_group(
        q,
        q_scales,
        weights,
        token,
        16,
        stride_q_token,
        stride_q_head,
        stride_q_scale_token,
        stride_q_scale_head,
        stride_w_token,
        stride_w_head,
        mfma_layout,
        dot_a_layout,
        a_scale_layout,
    )
    if NUM_HEADS == 64:
        query_2, query_scale_2, weight_2 = _load_query_group(
            q,
            q_scales,
            weights,
            token,
            32,
            stride_q_token,
            stride_q_head,
            stride_q_scale_token,
            stride_q_scale_head,
            stride_w_token,
            stride_w_head,
            mfma_layout,
            dot_a_layout,
            a_scale_layout,
        )
        query_3, query_scale_3, weight_3 = _load_query_group(
            q,
            q_scales,
            weights,
            token,
            48,
            stride_q_token,
            stride_q_head,
            stride_q_scale_token,
            stride_q_scale_head,
            stride_w_token,
            stride_w_head,
            mfma_layout,
            dot_a_layout,
            a_scale_layout,
        )
    else:
        query_2 = query_0
        query_scale_2 = query_scale_0
        weight_2 = weight_0
        query_3 = query_0
        query_scale_3 = query_scale_0
        weight_3 = weight_0

    output_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    output_columns = gl.arange(0, BLOCK_N, layout=output_layout)
    for tile_offset in range(0, CHUNK_N, BLOCK_N):
        tile_start = candidate_start + tile_offset
        scores = _score_query_group(
            query_0,
            query_scale_0,
            weight_0,
            index_k_cache,
            query_requests,
            query_starts,
            cu_seq_lens,
            block_table_base_offsets,
            block_table,
            token,
            tile_start,
            candidate_end,
            block_table_stride,
            page_stride_bytes,
            mfma_layout,
            dot_b_layout,
            b_scale_layout,
            PAGE_SIZE,
            BLOCK_TABLE_COLS,
            IS_PREFILL,
            HAS_BASE_OFFSETS,
        )
        scores += _score_query_group(
            query_1,
            query_scale_1,
            weight_1,
            index_k_cache,
            query_requests,
            query_starts,
            cu_seq_lens,
            block_table_base_offsets,
            block_table,
            token,
            tile_start,
            candidate_end,
            block_table_stride,
            page_stride_bytes,
            mfma_layout,
            dot_b_layout,
            b_scale_layout,
            PAGE_SIZE,
            BLOCK_TABLE_COLS,
            IS_PREFILL,
            HAS_BASE_OFFSETS,
        )
        if NUM_HEADS == 64:
            scores += _score_query_group(
                query_2,
                query_scale_2,
                weight_2,
                index_k_cache,
                query_requests,
                query_starts,
                cu_seq_lens,
                block_table_base_offsets,
                block_table,
                token,
                tile_start,
                candidate_end,
                block_table_stride,
                page_stride_bytes,
                mfma_layout,
                dot_b_layout,
                b_scale_layout,
                PAGE_SIZE,
                BLOCK_TABLE_COLS,
                IS_PREFILL,
                HAS_BASE_OFFSETS,
            )
            scores += _score_query_group(
                query_3,
                query_scale_3,
                weight_3,
                index_k_cache,
                query_requests,
                query_starts,
                cu_seq_lens,
                block_table_base_offsets,
                block_table,
                token,
                tile_start,
                candidate_end,
                block_table_stride,
                page_stride_bytes,
                mfma_layout,
                dot_b_layout,
                b_scale_layout,
                PAGE_SIZE,
                BLOCK_TABLE_COLS,
                IS_PREFILL,
                HAS_BASE_OFFSETS,
            )
        positions = tile_start + output_columns
        valid = positions < candidate_end
        gl.store(
            logits + token * logits_stride + positions,
            gl.where(valid, scores, -float("inf")),
            mask=positions < max_candidates,
        )


def _check_mxfp4_inputs(
    index_q: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    q, q_scales = index_q
    if q.dtype != torch.uint8 or q.dim() != 3:
        raise TypeError("MXFP4 index_q values must be rank-3 uint8")
    if q.shape[1] not in (32, 64) or q.shape[2] != _PACKED_DIM:
        raise ValueError(
            "GFX950 DSV4 index_q values must have shape "
            f"[tokens, 32|64, {_PACKED_DIM}], got {tuple(q.shape)}"
        )
    if not q.is_cuda or not q.is_contiguous():
        raise ValueError("index_q values must be contiguous on an AMD GPU")
    if q_scales.dtype != torch.int32 or q_scales.shape != q.shape[:2]:
        raise ValueError(
            f"index_q scales must be int32 with shape {tuple(q.shape[:2])}"
        )
    if q_scales.device != q.device or not q_scales.is_contiguous():
        raise ValueError("index_q scales must be contiguous and colocated with values")
    if weights.dtype != torch.float32 or weights.shape != q.shape[:2]:
        raise ValueError(f"weights must be float32 with shape {tuple(q.shape[:2])}")
    if weights.device != q.device or not weights.is_contiguous():
        raise ValueError("weights must be contiguous and colocated with index_q")
    if page_size != _PAGE_SIZE:
        raise ValueError(
            f"GFX950 DSV4 MXFP4 indexer requires page_size=64, got {page_size}"
        )
    if index_k_cache.dtype != torch.uint8 or index_k_cache.dim() != 2:
        raise TypeError("index_k_cache must be a rank-2 uint8 page matrix")
    page_bytes = page_size * _ROW_BYTES
    if (
        index_k_cache.device != q.device
        or index_k_cache.shape[1] < page_bytes
        or index_k_cache.stride(1) != 1
        or index_k_cache.stride(0) < page_bytes
    ):
        raise ValueError(
            "index_k_cache must be a colocated page-planar view with at least "
            f"{page_bytes} contiguous bytes per page"
        )
    return q, q_scales, int(index_k_cache.stride(0))


def _check_topk_output(
    out: torch.Tensor | None,
    *,
    tokens: int,
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    if topk not in _SUPPORTED_TOPK:
        raise ValueError(
            f"GFX950 DSV4 indexer supports topk={_SUPPORTED_TOPK}, got {topk}"
        )
    if out is None:
        return torch.empty((tokens, topk), dtype=torch.int32, device=device)
    if (
        out.dtype != torch.int32
        or out.device != device
        or out.dim() != 2
        or out.shape[0] < tokens
        or out.shape[1] != topk
        or out.stride(1) != 1
    ):
        raise ValueError(
            f"out must be int32 with at least shape ({tokens}, {topk}) on {device}"
        )
    return out[:tokens]


def _check_metadata_tensor(
    name: str,
    value: torch.Tensor,
    *,
    device: torch.device,
    ndim: int,
) -> None:
    if value.dtype != torch.int32 or value.device != device or value.dim() != ndim:
        raise ValueError(f"{name} must be a rank-{ndim} int32 tensor on {device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _dsv4_mxfp4_logits(
    index_q: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    lengths: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    max_candidates: int,
    block_table_base_offsets: torch.Tensor | None = None,
    cu_seq_lens: torch.Tensor | None = None,
    cu_seqlen_k_start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize exact DSV4 ``sum_h weight[h] * relu(q[h] @ k)`` logits."""

    q, q_scales, page_stride_bytes = _check_mxfp4_inputs(
        index_q, weights, index_k_cache, page_size
    )
    _check_metadata_tensor("lengths", lengths, device=q.device, ndim=1)
    _check_metadata_tensor("block_table", block_table, device=q.device, ndim=2)
    if lengths.shape != (q.shape[0],):
        raise ValueError(f"lengths must have shape ({q.shape[0]},)")
    if block_table.shape[0] == 0:
        raise ValueError("block_table must contain at least one request row")
    if block_table_base_offsets is not None:
        _check_metadata_tensor(
            "block_table_base_offsets",
            block_table_base_offsets,
            device=q.device,
            ndim=1,
        )
        if block_table_base_offsets.numel() < block_table.shape[0]:
            raise ValueError(
                "block_table_base_offsets must contain one entry per block-table row"
            )
    max_candidates = int(max_candidates)
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative")
    logits = torch.empty(
        (q.shape[0], max_candidates), dtype=torch.float32, device=q.device
    )
    if q.shape[0] == 0 or max_candidates == 0:
        return logits

    is_prefill = cu_seq_lens is not None or cu_seqlen_k_start is not None
    if is_prefill:
        if cu_seq_lens is None or cu_seqlen_k_start is None:
            raise ValueError(
                "prefill scoring requires cu_seq_lens and cu_seqlen_k_start"
            )
        _check_metadata_tensor("cu_seq_lens", cu_seq_lens, device=q.device, ndim=1)
        _check_metadata_tensor(
            "cu_seqlen_k_start", cu_seqlen_k_start, device=q.device, ndim=1
        )
        if cu_seq_lens.numel() != block_table.shape[0] + 1:
            raise ValueError("cu_seq_lens must contain one boundary per request")
        if cu_seqlen_k_start.shape != (q.shape[0],):
            raise ValueError("cu_seqlen_k_start must contain one offset per query")
        query_requests = torch.searchsorted(
            cu_seq_lens[1:].contiguous(),
            cu_seqlen_k_start,
            right=True,
        ).to(torch.int32)
        query_starts = cu_seqlen_k_start
        cu_arg = cu_seq_lens
    else:
        if block_table.shape[0] < q.shape[0]:
            raise ValueError("decode block_table must contain one row per query")
        query_requests = lengths
        query_starts = lengths
        cu_arg = lengths
    base_offsets_arg = (
        block_table if block_table_base_offsets is None else block_table_base_offsets
    )
    _dsv4_mxfp4_logits_kernel[(q.shape[0], triton.cdiv(max_candidates, _CHUNK_N))](
        q,
        q_scales.view(torch.uint8).reshape(q.shape[0], q.shape[1], _SCALE_DIM),
        weights,
        index_k_cache,
        lengths,
        query_requests,
        query_starts,
        cu_arg,
        base_offsets_arg,
        block_table,
        logits,
        q.stride(0),
        q.stride(1),
        _SCALE_DIM * q.shape[1],
        _SCALE_DIM,
        weights.stride(0),
        weights.stride(1),
        block_table.stride(0),
        logits.stride(0),
        page_stride_bytes,
        max_candidates,
        NUM_HEADS=q.shape[1],
        PAGE_SIZE=page_size,
        BLOCK_TABLE_COLS=block_table.shape[1],
        BLOCK_N=_BLOCK_N,
        CHUNK_N=_CHUNK_N,
        NUM_WARPS=2,
        IS_PREFILL=is_prefill,
        HAS_BASE_OFFSETS=block_table_base_offsets is not None,
        num_warps=2,
        waves_per_eu=2,
    )
    return logits


def gluon_dsv4_indexer_prefill_topk_mxfp4_gfx950(
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
    block_table_base_offsets: torch.Tensor | None = None,
    gathered_k: tuple[torch.Tensor, torch.Tensor] | None = None,
    gather_workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Score page-planar MXFP4 keys and return logical prefill offsets."""

    del cu_seqlen_k_end, gathered_k, gather_workspace
    if index_k_format != "mxfp4":
        raise ValueError("GFX950 DSV4 indexer only supports index_k_format='mxfp4'")
    q = index_q[0]
    result = _check_topk_output(out, tokens=q.shape[0], topk=int(topk), device=q.device)
    if q.shape[0] == 0 or int(max_seqlen_k) <= 0:
        result.fill_(-1)
        return result, None
    logits = _dsv4_mxfp4_logits(
        index_q,
        weights,
        index_k_cache,
        seq_lens,
        block_table,
        page_size=int(page_size),
        max_candidates=int(max_seqlen_k),
        block_table_base_offsets=block_table_base_offsets,
        cu_seq_lens=cu_seq_lens,
        cu_seqlen_k_start=cu_seqlen_k_start,
    )
    row_starts = torch.zeros_like(seq_lens)
    lens_out = torch.empty_like(seq_lens)
    _dsa_topk_indices(
        logits,
        row_starts,
        seq_lens,
        topk=int(topk),
        out=result,
        lens_out=lens_out,
    )
    if block_table_base_offsets is not None:
        query_requests = torch.searchsorted(
            cu_seq_lens[1:].contiguous(),
            cu_seqlen_k_start,
            right=True,
        ).to(torch.int64)
        query_requests.clamp_max_(block_table_base_offsets.numel() - 1)
        base_rows = block_table_base_offsets[query_requests].to(torch.int64) * int(
            page_size
        )
        selected = result.to(torch.int64)
        result.copy_(
            torch.where(selected >= 0, selected + base_rows[:, None], selected).to(
                torch.int32
            )
        )
    return result, None


def gluon_dsv4_indexer_decode_topk_mxfp4_gfx950(
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
    block_table_base_offsets: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    persistent_topk_workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score page-planar MXFP4 keys and return logical decode offsets."""

    del plan, persistent_topk_workspace
    if index_k_format != "mxfp4":
        raise ValueError("GFX950 DSV4 indexer only supports index_k_format='mxfp4'")
    q = index_q[0]
    result = _check_topk_output(out, tokens=q.shape[0], topk=int(topk), device=q.device)
    lengths = context_lens.reshape(-1).contiguous()
    if q.shape[0] == 0 or int(max_context_len) <= 0:
        result.fill_(-1)
        return result
    logits = _dsv4_mxfp4_logits(
        index_q,
        weights,
        index_k_cache,
        lengths,
        block_table,
        page_size=int(page_size),
        max_candidates=int(max_context_len),
        block_table_base_offsets=block_table_base_offsets,
    )
    row_starts = torch.zeros_like(lengths)
    lens_out = torch.empty_like(lengths)
    _dsa_topk_indices(
        logits,
        row_starts,
        lengths,
        topk=int(topk),
        out=result,
        lens_out=lens_out,
    )
    if block_table_base_offsets is not None:
        base_rows = block_table_base_offsets[: q.shape[0]].to(torch.int64) * int(
            page_size
        )
        selected = result.to(torch.int64)
        result.copy_(
            torch.where(selected >= 0, selected + base_rows[:, None], selected).to(
                torch.int32
            )
        )
    return result


def gluon_dsv4_plan_gfx950(
    *,
    page_size: int,
    seq_lens_2d: torch.Tensor,
    out: object | None = None,
) -> torch.Tensor:
    """Return graph-stable opaque metadata; the GFX950 scorer needs no schedule."""

    if int(page_size) != _PAGE_SIZE:
        raise ValueError(f"GFX950 DSV4 indexer requires page_size=64, got {page_size}")
    plan = seq_lens_2d.contiguous()
    if out is None:
        return plan
    if not isinstance(out, torch.Tensor):
        raise TypeError("GFX950 DSV4 plan output must be a tensor")
    if out.shape != plan.shape or out.dtype != plan.dtype or out.device != plan.device:
        raise ValueError("GFX950 DSV4 plan output must match seq_lens_2d")
    with torch.inference_mode():
        out.copy_(plan)
    return out
