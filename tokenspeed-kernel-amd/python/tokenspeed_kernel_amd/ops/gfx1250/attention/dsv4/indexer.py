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

"""MXFP4 DeepSeek V4 sparse-indexer kernels for AMD GFX1250."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.sparse_mla import (
    _dsa_topk_indices,
)
from tokenspeed_kernel_amd.ops.gfx1250.attention.mla._common import make_kernel_repr

_HEAD_DIM = 128
_PACKED_DIM_VALUE = _HEAD_DIM // 2
_SCALE_DIM_VALUE = _HEAD_DIM // 32
_PACKED_DIM = gl.constexpr(_PACKED_DIM_VALUE)
_SCALE_DIM = gl.constexpr(_SCALE_DIM_VALUE)
_PAGE_SIZE = 64
_ROW_BYTES = 68
_BLOCK_N = gl.constexpr(64)
_HEADS_PER_WMMA = gl.constexpr(32)
_TDM_KEY_ROWS = gl.constexpr(16)
_TDM_ROW_BYTES = gl.constexpr(256)
_BUFFER_CHUNK_N = 1024
_TDM_CHUNK_N = 512
_NUM_WARPS = 4
_TDM_MIN_CANDIDATES = 1 << 20
_WAVES_PER_EU = 4
_SUPPORTED_TOPK = (512, 1024, 2048)

__all__ = [
    "launch_gluon_dsv4_decode_topk_mxfp4_gfx1250",
    "launch_gluon_dsv4_plan_gfx1250",
    "launch_gluon_dsv4_prefill_topk_mxfp4_gfx1250",
]


def _logits_launch_metadata(grid, kernel, args):
    """Describe the useful index-scoring work for Proton roofline reports."""
    tokens, heads, _ = args["q"].shape
    max_candidates = args["max_candidates"]
    candidates = tokens * max_candidates
    splits = grid[1]
    query_bytes = tokens * splits * heads * (_PACKED_DIM_VALUE + _SCALE_DIM_VALUE + 4)
    cache_bytes = candidates * _ROW_BYTES
    metadata_bytes = candidates * 4 + tokens * splits * 4
    output_bytes = candidates * args["logits"].element_size()
    return {
        "name": kernel.name,
        "flops8": 2 * candidates * heads * _HEAD_DIM,
        "flops32": 2 * candidates * heads,
        "bytes": query_bytes + cache_bytes + metadata_bytes + output_bytes,
    }


@gluon.constexpr_function
def _indexer_wmma_layouts(NUM_WARPS: gl.constexpr):
    assert NUM_WARPS == 4
    wmma = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [0, 2]],
        reg_bases=[],
        instr_shape=[32, 16, 128],
    )
    packed_wmma = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [0, 2]],
        reg_bases=[],
        instr_shape=[32, 16, 64],
    )
    dot_a = gl.DotOperandLayout(operand_index=0, parent=packed_wmma, k_width=16)
    dot_b = gl.DotOperandLayout(operand_index=1, parent=packed_wmma, k_width=16)
    a_scale = gl.amd.cdna5.get_wmma_scale_layout(dot_a, [_HEADS_PER_WMMA, _SCALE_DIM])
    b_scale = gl.amd.cdna5.get_wmma_scale_layout(dot_b, [_BLOCK_N, _SCALE_DIM])
    return wmma, dot_a, dot_b, a_scale, b_scale


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
    wmma_layout: gl.constexpr,
    dot_a_layout: gl.constexpr,
    a_scale_layout: gl.constexpr,
):
    heads = gl.arange(0, _HEADS_PER_WMMA, layout=gl.SliceLayout(1, dot_a_layout))[
        :, None
    ]
    packed_dims = gl.arange(0, _PACKED_DIM, layout=gl.SliceLayout(0, dot_a_layout))[
        None, :
    ]
    query = gl.amd.cdna5.buffer_load(
        q,
        (token * stride_q_token + (head_base + heads) * stride_q_head + packed_dims).to(
            gl.int32
        ),
    )

    scale_heads = gl.arange(
        0, _HEADS_PER_WMMA, layout=gl.SliceLayout(1, a_scale_layout)
    )[:, None]
    scale_groups = gl.arange(0, _SCALE_DIM, layout=gl.SliceLayout(0, a_scale_layout))[
        None, :
    ]
    scales = gl.amd.cdna5.buffer_load(
        q_scales,
        (
            token * stride_q_scale_token
            + (head_base + scale_heads) * stride_q_scale_head
            + scale_groups
        ).to(gl.int32),
    )
    weight_heads = gl.arange(0, _HEADS_PER_WMMA, layout=gl.SliceLayout(1, wmma_layout))
    head_weights = gl.amd.cdna5.buffer_load(
        weights,
        (token * stride_w_token + (head_base + weight_heads) * stride_w_head).to(
            gl.int32
        ),
    ).to(gl.float32)
    return query, scales, head_weights


@gluon.jit
def _candidate_page_rows(
    positions,
    valid,
    query_requests,
    query_starts,
    cu_seq_lens,
    block_table,
    token,
    block_table_stride,
    NUM_PAGES: gl.constexpr,
    NUM_REQUESTS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    BLOCK_TABLE_COLS: gl.constexpr,
    IS_PREFILL: gl.constexpr,
):
    if IS_PREFILL:
        request = gl.load(query_requests + token).to(gl.int32)
        request_valid = (request >= 0) & (request < NUM_REQUESTS)
        request = gl.minimum(gl.maximum(request, 0), NUM_REQUESTS - 1)
        packed_position = gl.load(query_starts + token).to(gl.int32) + positions
        request_begin = gl.load(cu_seq_lens + request).to(gl.int32)
        request_end = gl.load(cu_seq_lens + request + 1).to(gl.int32)
        logical_position = packed_position - request_begin
        valid &= request_valid & (logical_position < request_end - request_begin)
    else:
        request = token
        logical_position = positions
    logical_page = logical_position // PAGE_SIZE
    valid &= (logical_position >= 0) & (logical_page < BLOCK_TABLE_COLS)
    safe_logical_page = gl.minimum(gl.maximum(logical_page, 0), BLOCK_TABLE_COLS - 1)
    physical_page = gl.amd.cdna5.buffer_load(
        block_table,
        (request * block_table_stride + safe_logical_page).to(gl.int32),
        mask=valid,
        other=0,
    ).to(gl.int64)
    valid &= (physical_page >= 0) & (physical_page < NUM_PAGES)
    safe_physical_page = gl.where(valid, physical_page, 0)
    return safe_physical_page, logical_position % PAGE_SIZE, valid


@gluon.jit
def _cache_load(pointer, offsets, mask, other, USE_BUFFER: gl.constexpr):
    if USE_BUFFER:
        return gl.amd.cdna5.buffer_load(
            pointer,
            offsets.to(gl.int32),
            mask=mask,
            other=other,
        )
    return gl.load(pointer + offsets, mask=mask, other=other)


@gluon.jit
def _load_key_tile(
    index_k_cache,
    query_requests,
    query_starts,
    cu_seq_lens,
    block_table,
    token,
    tile_start,
    candidate_end,
    block_table_stride,
    page_stride_bytes,
    dot_b_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
    NUM_PAGES: gl.constexpr,
    NUM_REQUESTS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    BLOCK_TABLE_COLS: gl.constexpr,
    IS_PREFILL: gl.constexpr,
    USE_BUFFER: gl.constexpr,
):
    packed_dims = gl.arange(0, _PACKED_DIM, layout=gl.SliceLayout(1, dot_b_layout))[
        :, None
    ]
    columns = gl.arange(0, _BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))[None, :]
    positions = tile_start + columns
    valid = positions < candidate_end
    pages, page_rows, valid = _candidate_page_rows(
        positions,
        valid,
        query_requests,
        query_starts,
        cu_seq_lens,
        block_table,
        token,
        block_table_stride,
        NUM_PAGES,
        NUM_REQUESTS,
        PAGE_SIZE,
        BLOCK_TABLE_COLS,
        IS_PREFILL,
    )
    key_offsets = (
        pages * page_stride_bytes + page_rows.to(gl.int64) * _PACKED_DIM + packed_dims
    )
    key = _cache_load(
        index_k_cache,
        key_offsets,
        valid,
        0,
        USE_BUFFER,
    )

    scale_columns = gl.arange(0, _BLOCK_N, layout=gl.SliceLayout(1, b_scale_layout))[
        :, None
    ]
    scale_groups = gl.arange(0, _SCALE_DIM, layout=gl.SliceLayout(0, b_scale_layout))[
        None, :
    ]
    scale_positions = tile_start + scale_columns
    scale_valid = scale_positions < candidate_end
    scale_pages, scale_page_rows, scale_valid = _candidate_page_rows(
        scale_positions,
        scale_valid,
        query_requests,
        query_starts,
        cu_seq_lens,
        block_table,
        token,
        block_table_stride,
        NUM_PAGES,
        NUM_REQUESTS,
        PAGE_SIZE,
        BLOCK_TABLE_COLS,
        IS_PREFILL,
    )
    key_scale_offsets = (
        scale_pages * page_stride_bytes
        + PAGE_SIZE * _PACKED_DIM
        + scale_page_rows.to(gl.int64) * _SCALE_DIM
        + scale_groups
    )
    key_scales = _cache_load(
        index_k_cache,
        key_scale_offsets,
        scale_valid,
        127,
        USE_BUFFER,
    )

    return key, key_scales, valid.reshape([_BLOCK_N])


@gluon.jit
def _score_query_group(
    query,
    query_scales,
    head_weights,
    key,
    key_scales,
    wmma_layout: gl.constexpr,
):
    accumulator = gl.zeros(
        [_HEADS_PER_WMMA, _BLOCK_N], dtype=gl.float32, layout=wmma_layout
    )
    head_scores = gl.amd.cdna5.wmma_scaled(
        query,
        query_scales,
        "e2m1",
        key,
        key_scales,
        "e2m1",
        accumulator,
    )
    head_scores = gl.maximum(
        head_scores,
        0.0,
        propagate_nan=tl.PropagateNan.ALL,
    )
    return gl.sum(head_scores * head_weights[:, None], axis=0)


@gluon.jit
def _issue_decode_page_tdm(
    index_k_cache,
    block_table,
    key_buffer,
    scale_buffer,
    token,
    tile_start,
    candidate_end,
    block_table_stride,
    page_stride_bytes,
    key_shared_layout: gl.constexpr,
    scale_shared_layout: gl.constexpr,
    NUM_PAGES: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    BLOCK_TABLE_COLS: gl.constexpr,
):
    logical_page = tile_start // PAGE_SIZE
    page_valid = (tile_start < candidate_end) & (logical_page < BLOCK_TABLE_COLS)
    physical_page = gl.load(
        block_table + token * block_table_stride + logical_page,
        mask=page_valid,
        other=0,
    ).to(gl.int64)
    page_valid &= (physical_page >= 0) & (physical_page < NUM_PAGES)
    physical_page = gl.where(page_valid, physical_page, 0)
    key_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=index_k_cache + physical_page * page_stride_bytes,
        shape=[_TDM_KEY_ROWS, _TDM_ROW_BYTES],
        strides=[_TDM_ROW_BYTES, 1],
        block_shape=[_TDM_KEY_ROWS, _TDM_ROW_BYTES],
        layout=key_shared_layout,
    )
    scale_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=(
            index_k_cache + physical_page * page_stride_bytes + PAGE_SIZE * _PACKED_DIM
        ),
        shape=[1, _TDM_ROW_BYTES],
        strides=[_TDM_ROW_BYTES, 1],
        block_shape=[1, _TDM_ROW_BYTES],
        layout=scale_shared_layout,
    )
    gl.amd.cdna5.tdm.async_load(key_desc, [0, 0], key_buffer)
    gl.amd.cdna5.tdm.async_load(scale_desc, [0, 0], scale_buffer)
    return page_valid


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
    ),
    launch_metadata=_logits_launch_metadata,
    repr=make_kernel_repr(
        "_dsv4_mxfp4_logits_gfx1250",
        [
            "NUM_HEADS",
            "BLOCK_N",
            "CHUNK_N",
            "IS_PREFILL",
            "USE_TDM",
            "USE_BUFFER",
        ],
    ),
)
def _dsv4_mxfp4_logits_gfx1250(
    q,
    q_scales,
    weights,
    index_k_cache,
    lengths,
    query_requests,
    query_starts,
    cu_seq_lens,
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
    NUM_PAGES: gl.constexpr,
    NUM_REQUESTS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    BLOCK_TABLE_COLS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    IS_PREFILL: gl.constexpr,
    USE_TDM: gl.constexpr,
    USE_BUFFER: gl.constexpr,
):
    token = gl.program_id(0)
    split = gl.program_id(1)
    candidate_start = split * CHUNK_N
    candidate_end = gl.minimum(gl.load(lengths + token), max_candidates)
    candidate_end = gl.minimum(candidate_end, candidate_start + CHUNK_N)

    layouts: gl.constexpr = _indexer_wmma_layouts(NUM_WARPS)
    wmma_layout: gl.constexpr = layouts[0]
    dot_a_layout: gl.constexpr = layouts[1]
    dot_b_layout: gl.constexpr = layouts[2]
    a_scale_layout: gl.constexpr = layouts[3]
    b_scale_layout: gl.constexpr = layouts[4]
    output_layout: gl.constexpr = gl.SliceLayout(0, wmma_layout)
    output_columns = gl.arange(0, BLOCK_N, layout=output_layout)
    if candidate_start >= candidate_end:
        for tile_offset in range(0, CHUNK_N, BLOCK_N):
            positions = candidate_start + tile_offset + output_columns
            gl.store(
                logits + token * logits_stride + positions,
                -float("inf"),
                mask=positions < max_candidates,
            )
        return

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
        wmma_layout,
        dot_a_layout,
        a_scale_layout,
    )
    if NUM_HEADS == 64:
        query_1, query_scale_1, weight_1 = _load_query_group(
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
            wmma_layout,
            dot_a_layout,
            a_scale_layout,
        )
    else:
        query_1 = query_0
        query_scale_1 = query_scale_0
        weight_1 = weight_0

    if USE_TDM:
        key_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[_TDM_ROW_BYTES, 16]],
            [_TDM_KEY_ROWS, _TDM_ROW_BYTES],
            [1, 0],
        )
        scale_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[_TDM_ROW_BYTES, 16]],
            [1, _TDM_ROW_BYTES],
            [1, 0],
        )
        key_buffer = gl.allocate_shared_memory(
            gl.uint8,
            [2, _TDM_KEY_ROWS, _TDM_ROW_BYTES],
            key_shared_layout,
        )
        scale_buffer = gl.allocate_shared_memory(
            gl.uint8,
            [2, 1, _TDM_ROW_BYTES],
            scale_shared_layout,
        )
        page_valid = _issue_decode_page_tdm(
            index_k_cache,
            block_table,
            key_buffer.index(0),
            scale_buffer.index(0),
            token,
            candidate_start,
            candidate_end,
            block_table_stride,
            page_stride_bytes,
            key_shared_layout,
            scale_shared_layout,
            NUM_PAGES,
            PAGE_SIZE,
            BLOCK_TABLE_COLS,
        )
        for tile_offset in range(0, CHUNK_N - BLOCK_N, BLOCK_N):
            tile_start = candidate_start + tile_offset
            buffer_index: gl.constexpr = (tile_offset // BLOCK_N) % 2
            next_buffer_index: gl.constexpr = 1 - buffer_index
            next_page_valid = _issue_decode_page_tdm(
                index_k_cache,
                block_table,
                key_buffer.index(next_buffer_index),
                scale_buffer.index(next_buffer_index),
                token,
                tile_start + BLOCK_N,
                candidate_end,
                block_table_stride,
                page_stride_bytes,
                key_shared_layout,
                scale_shared_layout,
                NUM_PAGES,
                PAGE_SIZE,
                BLOCK_TABLE_COLS,
            )
            gl.amd.cdna5.tdm.async_wait(2)
            key = (
                key_buffer.index(buffer_index)
                .reshape([_BLOCK_N, _PACKED_DIM])
                .permute([1, 0])
                .load(dot_b_layout)
            )
            key_scales = (
                scale_buffer.index(buffer_index)
                .reshape([_BLOCK_N, _SCALE_DIM])
                .load(b_scale_layout)
            )
            scores = _score_query_group(
                query_0,
                query_scale_0,
                weight_0,
                key,
                key_scales,
                wmma_layout,
            )
            if NUM_HEADS == 64:
                scores += _score_query_group(
                    query_1,
                    query_scale_1,
                    weight_1,
                    key,
                    key_scales,
                    wmma_layout,
                )
            positions = tile_start + output_columns
            valid = (positions < candidate_end) & page_valid
            gl.store(
                logits + token * logits_stride + positions,
                gl.where(valid, scores, -float("inf")),
                mask=positions < max_candidates,
            )
            page_valid = next_page_valid

        gl.amd.cdna5.tdm.async_wait(0)
        tile_start = candidate_start + CHUNK_N - BLOCK_N
        buffer_index: gl.constexpr = (CHUNK_N // BLOCK_N - 1) % 2
        key = (
            key_buffer.index(buffer_index)
            .reshape([_BLOCK_N, _PACKED_DIM])
            .permute([1, 0])
            .load(dot_b_layout)
        )
        key_scales = (
            scale_buffer.index(buffer_index)
            .reshape([_BLOCK_N, _SCALE_DIM])
            .load(b_scale_layout)
        )
        scores = _score_query_group(
            query_0,
            query_scale_0,
            weight_0,
            key,
            key_scales,
            wmma_layout,
        )
        if NUM_HEADS == 64:
            scores += _score_query_group(
                query_1,
                query_scale_1,
                weight_1,
                key,
                key_scales,
                wmma_layout,
            )
        positions = tile_start + output_columns
        valid = (positions < candidate_end) & page_valid
        gl.store(
            logits + token * logits_stride + positions,
            gl.where(valid, scores, -float("inf")),
            mask=positions < max_candidates,
        )
        return

    for tile_offset in range(0, CHUNK_N, BLOCK_N):
        tile_start = candidate_start + tile_offset
        key, key_scales, valid = _load_key_tile(
            index_k_cache,
            query_requests,
            query_starts,
            cu_seq_lens,
            block_table,
            token,
            tile_start,
            candidate_end,
            block_table_stride,
            page_stride_bytes,
            dot_b_layout,
            b_scale_layout,
            NUM_PAGES,
            NUM_REQUESTS,
            PAGE_SIZE,
            BLOCK_TABLE_COLS,
            IS_PREFILL,
            USE_BUFFER,
        )
        scores = _score_query_group(
            query_0,
            query_scale_0,
            weight_0,
            key,
            key_scales,
            wmma_layout,
        )
        valid = gl.convert_layout(valid, output_layout)
        if NUM_HEADS == 64:
            scores_1 = _score_query_group(
                query_1,
                query_scale_1,
                weight_1,
                key,
                key_scales,
                wmma_layout,
            )
            scores += scores_1
        positions = tile_start + output_columns
        gl.store(
            logits + token * logits_stride + positions,
            gl.where(valid, scores, -float("inf")),
            mask=positions < max_candidates,
        )


@gluon.jit
def _mask_invalid_topk_kernel(
    logits,
    selected,
    logits_stride: gl.constexpr,
    selected_stride: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK: gl.constexpr,
):
    row = gl.program_id(0)
    offsets = gl.arange(
        0,
        BLOCK,
        layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0]),
    )
    indices = gl.load(
        selected + row * selected_stride + offsets,
        mask=offsets < TOPK,
        other=-1,
    ).to(gl.int32)
    valid = indices >= 0
    values = gl.load(
        logits + row * logits_stride + gl.maximum(indices, 0),
        mask=valid,
        other=-float("inf"),
    )
    gl.store(
        selected + row * selected_stride + offsets,
        gl.where(valid & (values != -float("inf")), indices, -1),
        mask=offsets < TOPK,
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
    if q.shape[1] not in (32, 64) or q.shape[2] != _PACKED_DIM_VALUE:
        raise ValueError(
            "GFX1250 DSV4 index_q values must have shape "
            f"[tokens, 32|64, {_PACKED_DIM_VALUE}], got {tuple(q.shape)}"
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
            f"GFX1250 DSV4 MXFP4 indexer requires page_size=64, got {page_size}"
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
            f"GFX1250 DSV4 indexer supports topk={_SUPPORTED_TOPK}, got {topk}"
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


def _check_block_table(block_table: torch.Tensor, device: torch.device) -> None:
    if (
        block_table.dtype != torch.int32
        or block_table.device != device
        or block_table.dim() != 2
        or block_table.stride(1) != 1
    ):
        raise ValueError(
            "block_table must be a rank-2 int32 tensor with contiguous rows "
            f"on {device}"
        )


def _check_base_offsets(
    block_table_base_offsets: torch.Tensor | None,
    block_table: torch.Tensor,
) -> None:
    if block_table_base_offsets is None:
        return
    if (
        block_table_base_offsets.dtype not in (torch.int32, torch.int64)
        or block_table_base_offsets.device != block_table.device
        or block_table_base_offsets.dim() != 1
        or not block_table_base_offsets.is_contiguous()
        or block_table_base_offsets.numel() < block_table.shape[0]
    ):
        raise ValueError(
            "block_table_base_offsets must contain one contiguous integer entry "
            "per block-table row on the same device"
        )


def _dsv4_mxfp4_logits(
    index_q: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    lengths: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    max_candidates: int,
    cu_seq_lens: torch.Tensor | None,
    cu_seqlen_k_start: torch.Tensor | None,
) -> torch.Tensor:
    """Materialize exact DSV4 ``sum_h weight[h] * relu(q[h] @ k)`` logits."""

    q, q_scales, page_stride_bytes = _check_mxfp4_inputs(
        index_q, weights, index_k_cache, page_size
    )
    _check_metadata_tensor("lengths", lengths, device=q.device, ndim=1)
    _check_block_table(block_table, q.device)
    if lengths.shape != (q.shape[0],):
        raise ValueError(f"lengths must have shape ({q.shape[0]},)")
    max_candidates = int(max_candidates)
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative")
    logits = torch.empty(
        (q.shape[0], max_candidates), dtype=torch.float32, device=q.device
    )
    if q.shape[0] == 0 or max_candidates == 0:
        return logits
    if block_table.shape[0] == 0:
        raise ValueError("block_table must contain at least one request row")
    if block_table.shape[1] == 0:
        logits.fill_(-float("inf"))
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
    use_buffer = (
        index_k_cache.storage_offset() == 0
        and index_k_cache.stride(0) * index_k_cache.shape[0]
        <= torch.iinfo(torch.int32).max
    )
    use_tdm = (
        not is_prefill
        and index_k_cache.shape[0] > 0
        and q.shape[0] * max_candidates >= _TDM_MIN_CANDIDATES
    )
    chunk_n = _TDM_CHUNK_N if use_tdm else _BUFFER_CHUNK_N
    _dsv4_mxfp4_logits_gfx1250[
        (
            q.shape[0],
            triton.cdiv(max_candidates, chunk_n),
        )
    ](
        q,
        q_scales.view(torch.uint8).reshape(q.shape[0], q.shape[1], _SCALE_DIM),
        weights,
        index_k_cache,
        lengths,
        query_requests,
        query_starts,
        cu_arg,
        block_table,
        logits,
        q.stride(0),
        q.stride(1),
        _SCALE_DIM_VALUE * q.shape[1],
        _SCALE_DIM_VALUE,
        weights.stride(0),
        weights.stride(1),
        block_table.stride(0),
        logits.stride(0),
        page_stride_bytes,
        max_candidates,
        NUM_HEADS=q.shape[1],
        NUM_PAGES=index_k_cache.shape[0],
        NUM_REQUESTS=block_table.shape[0],
        PAGE_SIZE=page_size,
        BLOCK_TABLE_COLS=block_table.shape[1],
        BLOCK_N=_BLOCK_N,
        CHUNK_N=chunk_n,
        NUM_WARPS=_NUM_WARPS,
        IS_PREFILL=is_prefill,
        USE_TDM=use_tdm,
        USE_BUFFER=use_buffer,
        num_warps=_NUM_WARPS,
        waves_per_eu=_WAVES_PER_EU,
    )
    return logits


def _select_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    result: torch.Tensor,
) -> None:
    row_starts = torch.zeros_like(lengths)
    lens_out = torch.empty_like(lengths)
    _dsa_topk_indices(
        logits,
        row_starts,
        lengths,
        topk=topk,
        out=result,
        lens_out=lens_out,
        block_table=None,
        page_size=1,
        q_len_per_req=1,
    )
    _mask_invalid_topk_kernel[(logits.shape[0],)](
        logits,
        result,
        logits.stride(0),
        result.stride(0),
        TOPK=topk,
        BLOCK=triton.next_power_of_2(topk),
        num_warps=8,
        waves_per_eu=1,
    )


def launch_gluon_dsv4_prefill_topk_mxfp4_gfx1250(
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
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Score page-planar MXFP4 keys and return logical prefill offsets."""

    del gathered_k, gather_workspace
    if index_k_format != "mxfp4":
        raise ValueError("GFX1250 DSV4 indexer only supports index_k_format='mxfp4'")
    q = index_q[0]
    result = _check_topk_output(out, tokens=q.shape[0], topk=int(topk), device=q.device)
    if (
        cu_seqlen_k_end.shape != seq_lens.shape
        or cu_seqlen_k_start.shape != seq_lens.shape
    ):
        raise ValueError(
            "prefill candidate starts, ends and lengths must have the same shape"
        )
    _check_base_offsets(block_table_base_offsets, block_table)
    max_seqlen_k = int(max_seqlen_k)
    if q.shape[0] == 0 or max_seqlen_k <= 0:
        result.fill_(-1)
        return result, None
    lengths = torch.minimum(seq_lens, cu_seqlen_k_end - cu_seqlen_k_start).clamp(
        min=0, max=max_seqlen_k
    )
    logits = _dsv4_mxfp4_logits(
        index_q,
        weights,
        index_k_cache,
        lengths,
        block_table,
        page_size=int(page_size),
        max_candidates=max_seqlen_k,
        cu_seq_lens=cu_seq_lens,
        cu_seqlen_k_start=cu_seqlen_k_start,
    )
    _select_topk(logits, lengths, int(topk), result)
    if block_table_base_offsets is not None:
        query_requests = torch.searchsorted(
            cu_seq_lens[1:].contiguous(),
            cu_seqlen_k_start,
            right=True,
        ).to(torch.int64)
        query_requests.clamp_max_(block_table_base_offsets.numel() - 1)
        base_rows = (
            block_table_base_offsets[query_requests].to(torch.int64) * int(page_size)
            + cu_seqlen_k_start
            - cu_seq_lens[query_requests]
        )
        selected = result.to(torch.int64)
        result.copy_(
            torch.where(selected >= 0, selected + base_rows[:, None], selected).to(
                torch.int32
            )
        )
    return result, None


def launch_gluon_dsv4_decode_topk_mxfp4_gfx1250(
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
    """Score page-planar MXFP4 keys and return logical decode offsets."""

    del plan, persistent_topk_workspace
    if index_k_format != "mxfp4":
        raise ValueError("GFX1250 DSV4 indexer only supports index_k_format='mxfp4'")
    q = index_q[0]
    result = _check_topk_output(out, tokens=q.shape[0], topk=int(topk), device=q.device)
    _check_base_offsets(block_table_base_offsets, block_table)
    max_context_len = int(max_context_len)
    lengths = (
        context_lens.reshape(-1).contiguous().clamp(min=0, max=max(max_context_len, 0))
    )
    if q.shape[0] == 0 or max_context_len <= 0:
        result.fill_(-1)
        return result
    logits = _dsv4_mxfp4_logits(
        index_q,
        weights,
        index_k_cache,
        lengths,
        block_table,
        page_size=int(page_size),
        max_candidates=max_context_len,
        cu_seq_lens=None,
        cu_seqlen_k_start=None,
    )
    _select_topk(logits, lengths, int(topk), result)
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


def launch_gluon_dsv4_plan_gfx1250(
    *,
    page_size: int,
    seq_lens_2d: torch.Tensor,
    out: object | None,
) -> torch.Tensor:
    """Return graph-stable opaque metadata; the GFX1250 scorer needs no schedule."""

    if int(page_size) != _PAGE_SIZE:
        raise ValueError(f"GFX1250 DSV4 indexer requires page_size=64, got {page_size}")
    if out is None:
        return seq_lens_2d.clone(memory_format=torch.contiguous_format)
    if not isinstance(out, torch.Tensor):
        raise TypeError("GFX1250 DSV4 plan output must be a tensor")
    if (
        out.shape != seq_lens_2d.shape
        or out.dtype != seq_lens_2d.dtype
        or out.device != seq_lens_2d.device
    ):
        raise ValueError("GFX1250 DSV4 plan output must match seq_lens_2d")
    with torch.inference_mode():
        out.copy_(seq_lens_2d)
    return out
