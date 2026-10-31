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

"""GFX1250 DeepSeek V4.1 CSA2 indexer.

Scores page-planar MXFP4 index-K with a ``wmma_scaled`` e2m1 tile.
The tokenspeed-kernel adapter owns query packing, bounded query tiling, and
selection.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

__all__ = ["dsv41_index_logits_gfx1250"]

_HEAD_DIM = 128
_PACKED_DIM_VALUE = _HEAD_DIM // 2
_SCALE_DIM_VALUE = _HEAD_DIM // 32
_PACKED_DIM = gl.constexpr(_PACKED_DIM_VALUE)
_SCALE_DIM = gl.constexpr(_SCALE_DIM_VALUE)
_PAGE_SIZE = 64
_BLOCK_N = gl.constexpr(64)
_HEADS_PER_WMMA = gl.constexpr(32)


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


_WMMA_HEADS = 32
_CHUNK_N = 256
_NUM_WARPS = 4


@gluon.jit
def _csa2_page_rows(
    positions,
    valid,
    page_table,
    candidates,
    query,
    table_stride,
    cand_stride,
    num_pages,
    visible,
    PAGE_SIZE: gl.constexpr,
    table_width,
    CANDIDATES: gl.constexpr,
):
    if CANDIDATES >= 0:
        block = gl.amd.cdna5.buffer_load(
            candidates,
            (query * cand_stride + positions // 8).to(gl.int32),
            mask=valid,
            other=-1,
        ).to(gl.int64)
        logical = gl.where(block >= 0, block * 8 + positions % 8, -1)
        valid = valid & (logical >= 0) & (logical < visible)
    else:
        logical = positions.to(gl.int64)
        valid = valid & (logical < visible)
    logical_page = logical // PAGE_SIZE
    safe_page = gl.minimum(gl.maximum(logical_page, 0), table_width - 1)
    physical = gl.amd.cdna5.buffer_load(
        page_table,
        (query * table_stride + safe_page).to(gl.int32),
        mask=valid,
        other=-1,
    ).to(gl.int64)
    valid = valid & (physical >= 0) & (physical < num_pages)
    return gl.where(valid, physical, 0), logical % PAGE_SIZE, valid


@gluon.jit
def _load_csa2_key_tile(
    index_k_cache,
    page_table,
    candidates,
    token,
    tile_start,
    candidate_end,
    table_stride,
    cand_stride,
    page_stride_bytes,
    num_pages,
    visible,
    dot_b_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    table_width,
    CANDIDATES: gl.constexpr,
):
    packed_dims = gl.arange(0, _PACKED_DIM, layout=gl.SliceLayout(1, dot_b_layout))[
        :, None
    ]
    columns = gl.arange(0, _BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))[None, :]
    positions = tile_start + columns
    pages, page_rows, valid = _csa2_page_rows(
        positions,
        positions < candidate_end,
        page_table,
        candidates,
        token,
        table_stride,
        cand_stride,
        num_pages,
        visible,
        PAGE_SIZE,
        table_width,
        CANDIDATES,
    )
    # The launcher requires 16-byte page strides, so packed key bytes load as
    # 128-bit vectors.
    key = gl.load(
        index_k_cache
        + gl.multiple_of(pages * page_stride_bytes, [16, 16])
        + page_rows.to(gl.int64) * _PACKED_DIM
        + packed_dims,
        mask=valid,
        other=0,
    )

    scale_columns = gl.arange(0, _BLOCK_N, layout=gl.SliceLayout(1, b_scale_layout))[
        :, None
    ]
    scale_groups = gl.arange(0, _SCALE_DIM, layout=gl.SliceLayout(0, b_scale_layout))[
        None, :
    ]
    scale_positions = tile_start + scale_columns
    scale_pages, scale_page_rows, scale_valid = _csa2_page_rows(
        scale_positions,
        scale_positions < candidate_end,
        page_table,
        candidates,
        token,
        table_stride,
        cand_stride,
        num_pages,
        visible,
        PAGE_SIZE,
        table_width,
        CANDIDATES,
    )
    key_scales = gl.load(
        index_k_cache
        + gl.multiple_of(scale_pages * page_stride_bytes, [16, 16])
        + PAGE_SIZE * _PACKED_DIM
        + scale_page_rows.to(gl.int64) * _SCALE_DIM
        + scale_groups,
        mask=scale_valid,
        other=127,
    )
    return key, key_scales


def _index_launch_metadata(grid, kernel, args):
    """Describe the score capacity without reading device-resident lengths."""
    queries, width = args["logits"].shape
    heads = args["q"].shape[1]
    return {
        "name": kernel.name,
        "flops8": 2 * queries * heads * width * 128,
        "bytes": queries * width * 68
        + args["q"].numel() * args["q"].element_size() * grid[1]
        + args["logits"].numel() * args["logits"].element_size(),
    }


@gluon.jit(
    launch_metadata=_index_launch_metadata,
    do_not_specialize=(
        "stride_q_token",
        "stride_q_head",
        "stride_q_scale_token",
        "stride_q_scale_head",
        "stride_w_token",
        "stride_w_head",
        "table_stride",
        "table_width",
        "cand_stride",
        "logits_stride",
        "page_stride_bytes",
        "num_pages",
    ),
)
def gluon_dsv41_index_topk_gfx1250(
    q,
    q_scales,
    weights,
    index_k_cache,
    visible,
    page_table,
    candidates,
    logits,
    stride_q_token,
    stride_q_head,
    stride_q_scale_token,
    stride_q_scale_head,
    stride_w_token,
    stride_w_head,
    table_stride,
    cand_stride,
    logits_stride,
    page_stride_bytes,
    num_pages,
    max_candidates,
    PAGE_SIZE: gl.constexpr,
    table_width,
    CANDIDATES: gl.constexpr,
    SCORE_CHUNK: gl.constexpr,
    CHUNK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token = gl.program_id(0)
    split = gl.program_id(1)
    # Clamp before narrowing so large int64 lengths cannot wrap negative.
    vis = gl.minimum(
        gl.maximum(gl.load(visible + token).to(gl.int64), 0),
        table_width * PAGE_SIZE,
    ).to(gl.int32)
    if CANDIDATES >= 0:
        width = gl.where(vis > 0, CANDIDATES * 8, 0)
    else:
        width = vis
    candidate_start = split * SCORE_CHUNK
    candidate_end = gl.minimum(width, candidate_start + SCORE_CHUNK)
    candidate_end = gl.minimum(candidate_end, max_candidates)
    if candidate_start >= candidate_end:
        return

    layouts: gl.constexpr = _indexer_wmma_layouts(NUM_WARPS)
    wmma_layout: gl.constexpr = layouts[0]
    dot_a_layout: gl.constexpr = layouts[1]
    dot_b_layout: gl.constexpr = layouts[2]
    a_scale_layout: gl.constexpr = layouts[3]
    b_scale_layout: gl.constexpr = layouts[4]
    query, query_scales, head_weights = _load_query_group(
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
    output_layout: gl.constexpr = gl.SliceLayout(0, wmma_layout)
    output_columns = gl.arange(0, _BLOCK_N, layout=output_layout)
    for tile_offset in range(0, CHUNK_N, _BLOCK_N):
        tile_start = candidate_start + tile_offset
        key, key_scales = _load_csa2_key_tile(
            index_k_cache,
            page_table,
            candidates,
            token,
            tile_start,
            candidate_end,
            table_stride,
            cand_stride,
            page_stride_bytes,
            num_pages,
            vis,
            dot_b_layout,
            b_scale_layout,
            PAGE_SIZE,
            table_width,
            CANDIDATES,
        )
        scores = _score_query_group(
            query,
            query_scales,
            head_weights,
            key,
            key_scales,
            wmma_layout,
        )
        positions = tile_start + output_columns
        _, _, live = _csa2_page_rows(
            positions,
            positions < candidate_end,
            page_table,
            candidates,
            token,
            table_stride,
            cand_stride,
            num_pages,
            vis,
            PAGE_SIZE,
            table_width,
            CANDIDATES,
        )
        gl.store(
            logits + token * logits_stride + positions,
            scores,
            mask=(positions < max_candidates) & live,
        )


def dsv41_index_logits_gfx1250(
    values,
    scales,
    w,
    cache_2d,
    table,
    visible,
    candidates,
    logits,
    score_chunk_size,
):
    """Score prepared 32-head MXFP4 queries into caller-owned CSA2 logits.

    Args:
        values: Packed E2M1 query values shaped [T, 32, 64].
        scales: E8M0 query scales as int32 words shaped [T, 32].
        w: FP32 head weights shaped [T, 32].
        cache_2d: Page-planar MXFP4 bytes shaped [pages, 64 * 68] with unit
            byte stride and a page stride that is a multiple of 16 bytes.
        table: Physical page IDs shaped [T, logical_pages].
        visible: Visible logical row counts shaped [T].
        candidates: Optional candidate block IDs shaped [T, blocks].
        logits: FP32 destination shaped [T, scored_rows], initialized to -inf.
        score_chunk_size: Positive multiple-of-eight upper bound on rows per CTA.

    Returns:
        None. ``logits`` is mutated in place.
    """
    score_chunk_size = int(score_chunk_size)
    if score_chunk_size < 8 or score_chunk_size % 8:
        raise ValueError("score_chunk_size must be a positive multiple of 8")
    if cache_2d.stride(1) != 1 or cache_2d.stride(0) % 16:
        raise ValueError(
            "cache_2d needs unit byte stride and a page stride that is a "
            "multiple of 16 bytes"
        )
    score_chunk_size = min(score_chunk_size, _CHUNK_N)
    block_n = _BLOCK_N.value
    chunk_n = triton.cdiv(score_chunk_size, block_n) * block_n
    queries, width = logits.shape
    cand = table if candidates is None else candidates
    scale_bytes = scales.view(torch.uint8).reshape(
        queries, _WMMA_HEADS, _SCALE_DIM.value
    )
    gluon_dsv41_index_topk_gfx1250[(queries, triton.cdiv(width, score_chunk_size))](
        values,
        scale_bytes,
        w,
        cache_2d,
        visible,
        table,
        cand,
        logits,
        values.stride(0),
        values.stride(1),
        scale_bytes.stride(0),
        scale_bytes.stride(1),
        w.stride(0),
        w.stride(1),
        table.stride(0),
        cand.stride(0),
        logits.stride(0),
        int(cache_2d.stride(0)),
        int(cache_2d.shape[0]),
        width,
        PAGE_SIZE=_PAGE_SIZE,
        table_width=int(table.shape[1]),
        CANDIDATES=-1 if candidates is None else int(candidates.shape[1]),
        SCORE_CHUNK=score_chunk_size,
        CHUNK_N=chunk_n,
        NUM_WARPS=_NUM_WARPS,
        num_warps=_NUM_WARPS,
    )
