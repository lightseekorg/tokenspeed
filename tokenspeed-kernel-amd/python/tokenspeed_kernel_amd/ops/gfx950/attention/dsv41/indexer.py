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

"""GFX950 DeepSeek V4.1 CSA2 indexer.

Scores page-planar MXFP4 index-K with an ``mfma_scaled`` e2m1 tile.
The tokenspeed-kernel adapter owns query packing, bounded query tiling, and
selection.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

__all__ = [
    "dsv41_index_logits_gfx950",
    "launch_gluon_dsv41_index_block_select_gfx950",
    "launch_gluon_dsv41_index_topk_select_gfx950",
]

_HEAD_DIM = 128
_PACKED_DIM = gl.constexpr(_HEAD_DIM // 2)
_SCALE_DIM = gl.constexpr(_HEAD_DIM // 32)
_PAGE_SIZE = 64
_BLOCK_N = gl.constexpr(32)
_HEADS_PER_MFMA = gl.constexpr(16)
_CHUNK_N = 256


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


_MFMA_HEADS = 32


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
        block = gl.amd.cdna4.buffer_load(
            ptr=candidates,
            offsets=(query * cand_stride + positions // 8).to(gl.int32),
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
    physical = gl.amd.cdna4.buffer_load(
        ptr=page_table,
        offsets=(query * table_stride + safe_page).to(gl.int32),
        mask=valid,
        other=-1,
    ).to(gl.int64)
    valid = valid & (physical >= 0) & (physical < num_pages)
    return gl.where(valid, physical, 0), logical % PAGE_SIZE, valid


def _index_launch_metadata(grid, kernel, args):
    """Describe the score capacity without reading device-resident lengths."""
    queries, width = args["logits"].shape
    heads = args["NUM_HEADS"]
    return {
        "name": kernel.name,
        "flops4": 2 * queries * heads * width * 128,
        "bytes": queries * width * 68
        + args["q"].numel() * args["q"].element_size() * grid[1]
        + args["logits"].numel() * args["logits"].element_size(),
    }


@gluon.jit(
    launch_metadata=_index_launch_metadata,
    do_not_specialize=(
        "table_stride",
        "table_width",
        "cand_stride",
        "logits_stride",
        "num_pages",
    ),
)
def gluon_dsv41_index_topk_gfx950(
    q,
    q_scales,
    weights,
    index_k_cache,
    visible,
    page_table: tl.const,
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
    NUM_HEADS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    table_width,
    CANDIDATES: gl.constexpr,
    SCORE_CHUNK: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token = gl.program_id(0)
    split = gl.program_id(1)
    # Raw lengths may be int64 and exceed int32. Clamp before narrowing so
    # an oversized length cannot wrap negative and hide valid cache rows.
    vis = gl.minimum(
        gl.maximum(gl.load(visible + token).to(gl.int64), 0),
        table_width * PAGE_SIZE,
    ).to(gl.int32)
    if CANDIDATES >= 0:
        # Every candidate column is written, -inf when it maps to no row.
        width = CANDIDATES * 8
    else:
        # Block selection reads whole 8-row blocks: write -inf up to the end
        # of the last visible block.
        width = (vis + 7) // 8 * 8
    scored_end = gl.minimum(width, max_candidates)
    if split * SCORE_CHUNK >= scored_end:
        return

    layouts: gl.constexpr = _indexer_mfma_layouts(NUM_WARPS)
    mfma_layout: gl.constexpr = layouts[0]
    dot_a_layout: gl.constexpr = layouts[1]
    dot_b_layout: gl.constexpr = layouts[2]
    a_scale_layout: gl.constexpr = layouts[3]
    b_scale_layout: gl.constexpr = layouts[4]
    query_0, query_scale_0, weight_0, query_1, query_scale_1, weight_1 = (
        _qtile_load_queries(
            q,
            q_scales,
            weights,
            token,
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
    )
    output_columns = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    table_row = page_table + token * table_stride
    # The grid is bounded independently of the row capacity: CTA ``split``
    # scores chunks split, split + splits, ... below the query's scored end.
    for candidate_start in range(
        split * SCORE_CHUNK, scored_end, gl.num_programs(1) * SCORE_CHUNK
    ):
        candidate_end = gl.minimum(scored_end, candidate_start + SCORE_CHUNK)
        _score_token_chunk(
            index_k_cache,
            table_row,
            candidates,
            logits,
            token,
            candidate_start,
            candidate_end,
            cand_stride,
            logits_stride,
            page_stride_bytes,
            num_pages,
            vis,
            table_width,
            query_0,
            query_scale_0,
            weight_0,
            query_1,
            query_scale_1,
            weight_1,
            output_columns,
            CANDIDATES,
            BLOCK_N,
            CHUNK_N,
            mfma_layout,
            dot_b_layout,
            b_scale_layout,
        )


@gluon.jit
def _score_token_chunk(
    index_k_cache,
    table_row,
    candidates,
    logits,
    token,
    candidate_start,
    candidate_end,
    cand_stride,
    logits_stride,
    page_stride_bytes,
    num_pages,
    vis,
    table_width,
    query_0,
    query_scale_0,
    weight_0,
    query_1,
    query_scale_1,
    weight_1,
    output_columns,
    CANDIDATES: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    mfma_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
):
    key, key_scales, live = _token_tile_keys(
        index_k_cache,
        table_row,
        candidates,
        token,
        candidate_start,
        candidate_end,
        cand_stride,
        page_stride_bytes,
        num_pages,
        vis,
        table_width,
        CANDIDATES,
        mfma_layout,
        dot_b_layout,
        b_scale_layout,
    )
    key_1, key_scales_1, live_1 = _token_tile_keys(
        index_k_cache,
        table_row,
        candidates,
        token,
        candidate_start + BLOCK_N,
        candidate_end,
        cand_stride,
        page_stride_bytes,
        num_pages,
        vis,
        table_width,
        CANDIDATES,
        mfma_layout,
        dot_b_layout,
        b_scale_layout,
    )
    for tile_offset in range(0, CHUNK_N, BLOCK_N):
        tile_start = candidate_start + tile_offset
        # Keep two key tiles in flight: each tile's loads wait on its page
        # (and candidate) lookups. Both head groups share one key tile.
        key_2, key_scales_2, live_2 = _token_tile_keys(
            index_k_cache,
            table_row,
            candidates,
            token,
            tile_start + 2 * BLOCK_N,
            candidate_end,
            cand_stride,
            page_stride_bytes,
            num_pages,
            vis,
            table_width,
            CANDIDATES,
            mfma_layout,
            dot_b_layout,
            b_scale_layout,
        )
        scores = _qtile_score(
            query_0,
            query_scale_0,
            weight_0,
            query_1,
            query_scale_1,
            weight_1,
            key,
            key_scales,
            mfma_layout,
        )
        positions = tile_start + output_columns
        # Columns below the scored width that map to no row are written -inf,
        # so callers need not pre-fill the logits.
        gl.store(
            logits + token * logits_stride + positions,
            gl.where(live, scores, -float("inf")),
            mask=positions < candidate_end,
        )
        key = key_1
        key_scales = key_scales_1
        live = live_1
        key_1 = key_2
        key_scales_1 = key_scales_2
        live_1 = live_2


@gluon.jit
def _qtile_load_keys(
    cache,
    table,
    tile_start,
    end,
    page_stride_bytes,
    num_pages,
    table_width,
    dot_b_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
):
    # The live columns [tile_start, end) stay within one 64-row page, so one
    # scalar page lookup suffices: tiles are 32-aligned (shared-table scorer,
    # chunks that are a multiple of 32), or the chunk size divides 64 and
    # ``end`` is clamped to the chunk's end within that page. Do not mask by
    # anything wider than the chunk end.
    page = gl.load(table + gl.minimum(tile_start // 64, table_width - 1)).to(gl.int64)
    page_ok = (page >= 0) & (page < num_pages)
    base = cache + gl.where(page_ok, page, 0) * page_stride_bytes
    row0 = tile_start % 64
    packed_dims = gl.arange(0, _PACKED_DIM, layout=gl.SliceLayout(1, dot_b_layout))[
        :, None
    ]
    columns = gl.arange(0, _BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))[None, :]
    key = gl.load(
        base + ((row0 + columns) * _PACKED_DIM + packed_dims),
        mask=(tile_start + columns) < end,
        other=0,
    )
    scale_columns = gl.arange(0, _BLOCK_N, layout=gl.SliceLayout(1, b_scale_layout))[
        :, None
    ]
    scale_groups = gl.arange(0, _SCALE_DIM, layout=gl.SliceLayout(0, b_scale_layout))[
        None, :
    ]
    key_scales = gl.load(
        base + (64 * _PACKED_DIM + (row0 + scale_columns) * _SCALE_DIM + scale_groups),
        mask=(tile_start + scale_columns) < end,
        other=127,
    )
    return key, key_scales, page_ok


@gluon.jit
def _qtile_load_queries(
    q,
    q_scales,
    weights,
    token,
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
    return query_0, query_scale_0, weight_0, query_1, query_scale_1, weight_1


@gluon.jit
def _qtile_score(
    query_0,
    query_scale_0,
    weight_0,
    query_1,
    query_scale_1,
    weight_1,
    key,
    key_scales,
    mfma_layout: gl.constexpr,
):
    zero = gl.zeros([_HEADS_PER_MFMA, _BLOCK_N], dtype=gl.float32, layout=mfma_layout)
    heads_0 = gl.amd.cdna4.mfma_scaled(
        a=query_0,
        a_scale=query_scale_0,
        a_format="e2m1",
        b=key,
        b_scale=key_scales,
        b_format="e2m1",
        acc=zero,
    )
    heads_0 = gl.maximum(heads_0, 0.0, propagate_nan=tl.PropagateNan.ALL)
    scores = gl.sum(heads_0 * weight_0[:, None], axis=0)
    heads_1 = gl.amd.cdna4.mfma_scaled(
        a=query_1,
        a_scale=query_scale_1,
        a_format="e2m1",
        b=key,
        b_scale=key_scales,
        b_format="e2m1",
        acc=zero,
    )
    heads_1 = gl.maximum(heads_1, 0.0, propagate_nan=tl.PropagateNan.ALL)
    return scores + gl.sum(heads_1 * weight_1[:, None], axis=0)


@gluon.jit
def _qtile_store(
    logits, token, logits_stride, positions, bound, limit, page_ok, scores
):
    # Write -inf up to the end of the last visible 8-row block, which block
    # selection reads whole.
    gl.store(
        logits + token * logits_stride + positions,
        gl.where(page_ok & (positions < bound), scores, -float("inf")),
        mask=(positions < (bound + 7) // 8 * 8) & (positions < limit),
    )


@gluon.jit
def _token_tile_keys(
    cache,
    table_row,
    candidates,
    token,
    tile_start,
    end,
    cand_stride,
    page_stride_bytes,
    num_pages,
    visible,
    table_width,
    CANDIDATES: gl.constexpr,
    mfma_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
):
    """One query's 32-column key tile, its scales, and its live columns."""
    if CANDIDATES < 0:
        key, key_scales, page_ok = _qtile_load_keys(
            cache,
            table_row,
            tile_start,
            end,
            page_stride_bytes,
            num_pages,
            table_width,
            dot_b_layout,
            b_scale_layout,
        )
        columns = tile_start + gl.arange(
            0, _BLOCK_N, layout=gl.SliceLayout(0, mfma_layout)
        )
        live = (columns < end) & (columns < visible) & page_ok
    else:
        packed_dims = gl.arange(0, _PACKED_DIM, layout=gl.SliceLayout(1, dot_b_layout))[
            :, None
        ]
        columns = (
            tile_start
            + gl.arange(0, _BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))[None, :]
        )
        pages, page_rows, valid = _csa2_page_rows(
            columns,
            columns < end,
            table_row,
            candidates,
            token,
            0,
            cand_stride,
            num_pages,
            visible,
            64,
            table_width,
            CANDIDATES,
        )
        key = gl.load(
            cache
            + (pages * page_stride_bytes + page_rows.to(gl.int64) * _PACKED_DIM)
            + packed_dims,
            mask=valid,
            other=0,
        )
        scale_columns = (
            tile_start
            + gl.arange(0, _BLOCK_N, layout=gl.SliceLayout(1, b_scale_layout))[:, None]
        )
        scale_groups = gl.arange(
            0, _SCALE_DIM, layout=gl.SliceLayout(0, b_scale_layout)
        )[None, :]
        scale_pages, scale_rows, scale_valid = _csa2_page_rows(
            scale_columns,
            scale_columns < end,
            table_row,
            candidates,
            token,
            0,
            cand_stride,
            num_pages,
            visible,
            64,
            table_width,
            CANDIDATES,
        )
        key_scales = gl.load(
            cache
            + (
                scale_pages * page_stride_bytes
                + 64 * _PACKED_DIM
                + scale_rows.to(gl.int64) * _SCALE_DIM
            )
            + scale_groups,
            mask=scale_valid,
            other=127,
        )
        out_columns = tile_start + gl.arange(
            0, _BLOCK_N, layout=gl.SliceLayout(0, mfma_layout)
        )
        _, _, live = _csa2_page_rows(
            out_columns,
            out_columns < end,
            table_row,
            candidates,
            token,
            0,
            cand_stride,
            num_pages,
            visible,
            64,
            table_width,
            CANDIDATES,
        )
    return key, key_scales, live


def _qtile_launch_metadata(grid, kernel, args):
    queries, width = args["logits"].shape
    return {
        "name": kernel.name,
        "flops4": 2 * queries * 32 * width * 128,
        "bytes": grid[0] * width * 68 + queries * width * 4,
    }


@gluon.jit(
    launch_metadata=_qtile_launch_metadata,
    do_not_specialize=("queries", "table_width", "max_candidates"),
)
def gluon_dsv41_index_qtile_gfx950(
    q,
    q_scales,
    weights,
    index_k_cache,
    visible,
    page_table: tl.const,
    logits,
    queries,
    stride_q_token,
    stride_q_head,
    stride_q_scale_token,
    stride_q_scale_head,
    stride_w_token,
    stride_w_head,
    logits_stride,
    page_stride_bytes,
    num_pages,
    table_width,
    max_candidates,
    CHUNK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    # Four consecutive queries sharing one page table row score each key tile.
    first = gl.program_id(0) * 4
    candidate_start = gl.program_id(1) * CHUNK_N
    layouts: gl.constexpr = _indexer_mfma_layouts(NUM_WARPS)
    mfma_layout: gl.constexpr = layouts[0]
    dot_a_layout: gl.constexpr = layouts[1]
    dot_b_layout: gl.constexpr = layouts[2]
    a_scale_layout: gl.constexpr = layouts[3]
    b_scale_layout: gl.constexpr = layouts[4]
    token_0 = first
    token_1 = gl.minimum(first + 1, queries - 1)
    token_2 = gl.minimum(first + 2, queries - 1)
    token_3 = gl.minimum(first + 3, queries - 1)
    capacity = gl.minimum(table_width * 64, max_candidates)
    vis_0 = gl.minimum(
        gl.maximum(gl.load(visible + token_0).to(gl.int64), 0), capacity
    ).to(gl.int32)
    vis_1 = gl.where(
        first + 1 < queries,
        gl.minimum(gl.maximum(gl.load(visible + token_1).to(gl.int64), 0), capacity).to(
            gl.int32
        ),
        0,
    )
    vis_2 = gl.where(
        first + 2 < queries,
        gl.minimum(gl.maximum(gl.load(visible + token_2).to(gl.int64), 0), capacity).to(
            gl.int32
        ),
        0,
    )
    vis_3 = gl.where(
        first + 3 < queries,
        gl.minimum(gl.maximum(gl.load(visible + token_3).to(gl.int64), 0), capacity).to(
            gl.int32
        ),
        0,
    )
    candidate_end = gl.maximum(gl.maximum(vis_0, vis_1), gl.maximum(vis_2, vis_3))
    candidate_end = gl.minimum(candidate_end, candidate_start + CHUNK_N)
    if candidate_start >= candidate_end:
        return

    qa_0, qsa_0, wa_0, qb_0, qsb_0, wb_0 = _qtile_load_queries(
        q,
        q_scales,
        weights,
        token_0,
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
    qa_1, qsa_1, wa_1, qb_1, qsb_1, wb_1 = _qtile_load_queries(
        q,
        q_scales,
        weights,
        token_1,
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
    qa_2, qsa_2, wa_2, qb_2, qsb_2, wb_2 = _qtile_load_queries(
        q,
        q_scales,
        weights,
        token_2,
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
    qa_3, qsa_3, wa_3, qb_3, qsb_3, wb_3 = _qtile_load_queries(
        q,
        q_scales,
        weights,
        token_3,
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
    output_columns = gl.arange(0, _BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    key, key_scales, page_ok = _qtile_load_keys(
        index_k_cache,
        page_table,
        candidate_start,
        candidate_end,
        page_stride_bytes,
        num_pages,
        table_width,
        dot_b_layout,
        b_scale_layout,
    )
    for tile_offset in range(0, CHUNK_N, _BLOCK_N):
        tile_start = candidate_start + tile_offset
        # Prefetch the next tile before this tile's MFMAs.
        next_key, next_key_scales, next_page_ok = _qtile_load_keys(
            index_k_cache,
            page_table,
            tile_start + _BLOCK_N,
            candidate_end,
            page_stride_bytes,
            num_pages,
            table_width,
            dot_b_layout,
            b_scale_layout,
        )
        positions = tile_start + output_columns
        _qtile_store(
            logits,
            token_0,
            logits_stride,
            positions,
            vis_0,
            max_candidates,
            page_ok,
            _qtile_score(
                qa_0, qsa_0, wa_0, qb_0, qsb_0, wb_0, key, key_scales, mfma_layout
            ),
        )
        _qtile_store(
            logits,
            token_1,
            logits_stride,
            positions,
            vis_1,
            max_candidates,
            page_ok,
            _qtile_score(
                qa_1, qsa_1, wa_1, qb_1, qsb_1, wb_1, key, key_scales, mfma_layout
            ),
        )
        _qtile_store(
            logits,
            token_2,
            logits_stride,
            positions,
            vis_2,
            max_candidates,
            page_ok,
            _qtile_score(
                qa_2, qsa_2, wa_2, qb_2, qsb_2, wb_2, key, key_scales, mfma_layout
            ),
        )
        _qtile_store(
            logits,
            token_3,
            logits_stride,
            positions,
            vis_3,
            max_candidates,
            page_ok,
            _qtile_score(
                qa_3, qsa_3, wa_3, qb_3, qsb_3, wb_3, key, key_scales, mfma_layout
            ),
        )
        key = next_key
        key_scales = next_key_scales
        page_ok = next_page_ok


def dsv41_index_logits_gfx950(
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
        cache_2d: Page-planar MXFP4 bytes shaped [pages, 64 * 68].
        table: Physical page IDs shaped [T, logical_pages].
        visible: Visible logical row counts shaped [T].
        candidates: Optional candidate block IDs shaped [T, blocks].
        logits: FP32 destination shaped [T, scored_rows]. Every column below
            a query's visible count (all columns with candidates) is written,
            -inf where no cache row is scored; other columns are untouched.
        score_chunk_size: Positive multiple-of-eight upper bound on rows per CTA.

    Returns:
        None. ``logits`` is mutated in place.
    """
    score_chunk_size = int(score_chunk_size)
    if score_chunk_size < 8 or score_chunk_size % 8:
        raise ValueError("score_chunk_size must be a positive multiple of 8")
    score_chunk_size = min(score_chunk_size, _CHUNK_N)
    chunk_n = triton.cdiv(score_chunk_size, _BLOCK_N) * _BLOCK_N
    queries, width = logits.shape
    cand = table if candidates is None else candidates
    scale_dim = 4
    if candidates is None and table.stride(0) == 0:
        # Every query reads the same page table row: score four queries per CTA.
        grid = (triton.cdiv(queries, 4), triton.cdiv(width, chunk_n))
        gluon_dsv41_index_qtile_gfx950[grid](
            values,
            scales.view(torch.uint8).reshape(queries, _MFMA_HEADS, scale_dim),
            w,
            cache_2d,
            visible,
            table,
            logits,
            queries,
            values.stride(0),
            values.stride(1),
            scale_dim * _MFMA_HEADS,
            scale_dim,
            w.stride(0),
            w.stride(1),
            logits.stride(0),
            int(cache_2d.stride(0)),
            int(cache_2d.shape[0]),
            int(table.shape[1]),
            width,
            CHUNK_N=chunk_n,
            NUM_WARPS=2,
            num_warps=2,
            waves_per_eu=4,
            # Unfused multiply-adds keep the score bits independent of how a
            # variant's code contracts; both scorers produce the same bits.
            enable_fp_fusion=False,
        )
        return
    # A per-query CTA walks its chunk one key tile at a time; shorter chunks
    # spread the dependent page and key loads over more CTAs.
    score_chunk_size = min(score_chunk_size, 64 if queries <= 64 else 128)
    if candidates is None and score_chunk_size % 32 and 64 % score_chunk_size:
        # A Full key tile reads one 64-row page, so a chunk must not straddle
        # pages: use a multiple of 32 or a divisor of 64 within the bound.
        score_chunk_size = (
            score_chunk_size // 32 * 32
            if score_chunk_size > 32
            else 16 if score_chunk_size > 16 else 8
        )
    chunk_n = triton.cdiv(score_chunk_size, _BLOCK_N) * _BLOCK_N
    # Capacity-wide rows (decode tables) would launch mostly idle CTAs; cap
    # the grid and let each CTA stride over chunks.
    splits = min(triton.cdiv(width, score_chunk_size), max(16, 16384 // queries))
    grid = (queries, splits)
    gluon_dsv41_index_topk_gfx950[grid](
        values,
        scales.view(torch.uint8).reshape(queries, _MFMA_HEADS, scale_dim),
        w,
        cache_2d,
        visible,
        table,
        cand,
        logits,
        values.stride(0),
        values.stride(1),
        scale_dim * _MFMA_HEADS,
        scale_dim,
        w.stride(0),
        w.stride(1),
        table.stride(0),
        cand.stride(0),
        logits.stride(0),
        int(cache_2d.stride(0)),
        int(cache_2d.shape[0]),
        width,
        NUM_HEADS=_MFMA_HEADS,
        PAGE_SIZE=_PAGE_SIZE,
        table_width=int(table.shape[1]),
        CANDIDATES=-1 if candidates is None else int(candidates.shape[1]),
        SCORE_CHUNK=score_chunk_size,
        BLOCK_N=32,
        CHUNK_N=chunk_n,
        NUM_WARPS=2,
        num_warps=2,
        waves_per_eu=4,
        enable_fp_fusion=False,
        # With the current LLVM, SLP vectorization slows this kernel.
        llvm_cl_opts="vectorize-slp=false",
    )


# Radix-select digit split of the 32-bit order key: 11 + 11 + 10 bits.
_SELECT_RADIX_BITS = 11
_SELECT_WARPS = 16


def _select_launch_metadata(grid, kernel, args):
    """Each pass re-reads the logits row; candidates add one id load per pick."""
    queries = grid[0]
    width = args["width"]
    topk = args["topk"]
    return {
        "name": kernel.name,
        "bytes": queries * (4 * width * 4 + topk * (4 + 8)),
    }


@gluon.jit
def _select_order_key(values):
    # Monotone map from FP32 to uint32: larger value -> larger key.
    bits = values.to(gl.uint32, bitcast=True)
    flip = gl.where(
        (bits >> 31) != 0,
        gl.full(bits.shape, 0xFFFFFFFF, gl.uint32, bits.type.layout),
        gl.full(bits.shape, 0x80000000, gl.uint32, bits.type.layout),
    )
    return bits ^ flip


@gluon.jit
def _select_radix_pass(
    row,
    width,
    take,
    prefix,
    above,
    remaining,
    hist_smem,
    ones,
    d,
    SHIFT: gl.constexpr,
    DIGIT_BITS: gl.constexpr,
    FIRST: gl.constexpr,
    BLOCK: gl.constexpr,
    BINS: gl.constexpr,
    L: gl.constexpr,
    LBIN: gl.constexpr,
):
    """Histogram one key digit of the live keys matching ``prefix``."""
    # Earlier readers of the histogram finish before it is cleared, and the
    # clear lands before any warp adds to it.
    gl.barrier()
    hist_smem.store(gl.zeros([BINS], gl.int32, layout=LBIN))
    gl.barrier()
    for c0 in range(0, width, BLOCK):
        col = c0 + gl.arange(0, BLOCK, layout=L)
        values = gl.load(row + col, mask=col < width, other=-float("inf"))
        key = _select_order_key(values)
        live = values > -float("inf")
        if not FIRST:
            live = live & ((key >> (SHIFT + DIGIT_BITS)) == prefix)
        digit = ((key >> SHIFT) & ((1 << DIGIT_BITS) - 1)).to(gl.int32)
        hist_smem.atomic_scatter_add(ones, digit, axis=0, mask=live)
    gl.barrier()
    hist = hist_smem.load(LBIN)
    if FIRST:
        remaining = gl.minimum(take, gl.sum(hist, 0))
    # at_or_above[d] counts keys whose digit is >= d; the K-th key's digit is
    # the largest d with at_or_above[d] >= remaining.
    at_or_above = gl.cumsum(hist, 0, reverse=True)
    pick = gl.maximum(gl.sum((at_or_above >= remaining).to(gl.int32), 0) - 1, 0)
    higher = gl.sum(gl.where(d > pick, hist, 0), 0)
    if FIRST:
        prefix = pick.to(gl.uint32)
    else:
        prefix = (prefix << DIGIT_BITS) | pick.to(gl.uint32)
    return prefix, above + higher, remaining - higher


@gluon.jit
def _select_top_digit_pass(
    row,
    width,
    take,
    hist_smem,
    ones,
    d,
    SHIFT: gl.constexpr,
    BLOCK: gl.constexpr,
    BINS: gl.constexpr,
    L: gl.constexpr,
    LBIN: gl.constexpr,
    VEC: gl.constexpr,
):
    """Histogram the top key digit of the live columns; return the pivot digit."""
    hist_smem.store(gl.zeros([BINS], gl.int32, layout=LBIN))
    gl.barrier()
    lane = gl.arange(0, BLOCK, layout=L)
    for c0 in range(0, width, BLOCK):
        col = c0 + lane
        # Masking whole 4-column groups keeps the load vectorized when the row
        # capacity is a multiple of four; the tail is masked by value.
        group = (c0 + (lane // 4) * 4) if VEC else col
        values = gl.load(row + col, mask=group < width, other=-float("inf"))
        values = gl.where(col < width, values, -float("inf"))
        key = _select_order_key(values)
        hist_smem.atomic_scatter_add(
            ones,
            (key >> SHIFT).to(gl.int32),
            axis=0,
            mask=values > -float("inf"),
        )
    gl.barrier()
    hist = hist_smem.load(LBIN)
    remaining = gl.minimum(take, gl.sum(hist, 0))
    at_or_above = gl.cumsum(hist, 0, reverse=True)
    pick = gl.maximum(gl.sum((at_or_above >= remaining).to(gl.int32), 0) - 1, 0)
    higher = gl.sum(gl.where(d > pick, hist, 0), 0)
    pivot_size = gl.sum(gl.where(d == pick, hist, 0), 0)
    return pick.to(gl.uint32), higher, remaining - higher, pivot_size


@gluon.jit
def _select_lds_digit(
    key,
    live,
    prefix,
    above,
    remaining,
    hist_smem,
    ones,
    d,
    SHIFT: gl.constexpr,
    DIGIT_BITS: gl.constexpr,
    BINS: gl.constexpr,
    LBIN: gl.constexpr,
):
    """One MSD digit over register keys whose higher bits equal ``prefix``."""
    gl.barrier()
    hist_smem.store(gl.zeros([BINS], gl.int32, layout=LBIN))
    gl.barrier()
    match = live & ((key >> (SHIFT + DIGIT_BITS)) == prefix)
    digit = ((key >> SHIFT) & ((1 << DIGIT_BITS) - 1)).to(gl.int32)
    hist_smem.atomic_scatter_add(ones, digit, axis=0, mask=match)
    gl.barrier()
    hist = hist_smem.load(LBIN)
    at_or_above = gl.cumsum(hist, 0, reverse=True)
    pick = gl.maximum(gl.sum((at_or_above >= remaining).to(gl.int32), 0) - 1, 0)
    higher = gl.sum(gl.where(d > pick, hist, 0), 0)
    return (
        (prefix << DIGIT_BITS) | pick.to(gl.uint32),
        above + higher,
        remaining - higher,
    )


@gluon.jit(
    launch_metadata=_select_launch_metadata,
    do_not_specialize=(
        "cand_stride",
        "out_stride",
        "row_limits_stride",
        "width",
        "topk",
        "take",
    ),
)
def gluon_dsv41_index_topk_select_gfx950(
    logits,  # [Q, width] fp32, -inf = unscored
    candidates,  # [Q, blocks] int32/int64 block ids, or logits when unused
    row_limits,  # [Q] live widths, or logits when unused
    row_out,  # [Q, TOPK] int32
    row_lens,  # [Q] int32
    logits_stride,
    cand_stride,
    out_stride,
    row_limits_stride,
    width,
    topk,  # row_out width
    take,  # min(topk, width)
    TOPK: gl.constexpr,  # power-of-two capacity >= topk, at least 512
    HAS_CANDIDATES: gl.constexpr,
    HAS_LIMITS: gl.constexpr,
    RADIX_BITS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    REGION: gl.constexpr,  # LDS entries per warp, a power of two > SEGMENT
    SEGMENT: gl.constexpr,  # columns per warp per step
    VEC: gl.constexpr,  # the row capacity is a multiple of four columns
):
    """Top-``take`` finite logits of one query row as position-sorted ids.

    One pass histograms the top key digit and finds the pivot digit of the
    K-th largest finite logit (K = min(take, finite count)). A second pass
    lets each warp scan its own contiguous columns and append, in column
    order, every key whose top digit is at least the pivot to its LDS region.
    The regions in warp order are in column order, so the exact threshold,
    the lowest-column boundary ties, and the output order follow from LDS
    alone. A region overflow falls back to a three-pass MSD radix select
    over the row. Positions >= K are written as -1.
    """
    BLOCK: gl.constexpr = 4 * 64 * NUM_WARPS
    BINS: gl.constexpr = 1 << RADIX_BITS
    L: gl.constexpr = gl.BlockedLayout([4], [64], [NUM_WARPS], [0])
    LBIN: gl.constexpr = gl.BlockedLayout(
        [BINS // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
    )
    SL: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    TOP: gl.constexpr = 32 - RADIX_BITS
    LOW_BITS: gl.constexpr = 32 - 2 * RADIX_BITS
    # Free picked-id slots hold the int64 maximum, so they rank after all ids.
    EMPTY_ID: gl.constexpr = 0x7FFFFFFFFFFFFFFF

    q = gl.program_id(0).to(gl.int64)
    row = logits + q * logits_stride
    if HAS_LIMITS:
        # Columns at or past the row's live width are -inf: skip them.
        width = gl.minimum(
            width,
            gl.maximum(gl.load(row_limits + q * row_limits_stride).to(gl.int64), 0),
        ).to(gl.int32)
    hist_smem = gl.allocate_shared_memory(gl.int32, [BINS], SL)
    ones = gl.full([BLOCK], 1, gl.int32, layout=L)
    d = gl.arange(0, BINS, layout=LBIN)
    pivot, above, remaining, pivot_size = _select_top_digit_pass(
        row, width, take, hist_smem, ones, d, TOP, BLOCK, BINS, L, LBIN, VEC
    )
    count = above + remaining
    FLAT: gl.constexpr = NUM_WARPS * REGION
    FLAT_CAPACITY: gl.constexpr = FLAT - NUM_WARPS * SEGMENT
    # Collect the keys whose bits from ``shift`` up are >= ``prefix``. When the
    # pivot digit is too large to collect, refine it with the next digit.
    prefix = pivot
    shift = TOP
    bin_size = pivot_size
    above_bin = above
    keep = remaining
    if above + pivot_size > FLAT_CAPACITY:
        prefix, above_bin, keep = _select_radix_pass(
            row,
            width,
            take,
            pivot,
            above,
            remaining,
            hist_smem,
            ones,
            d,
            LOW_BITS,
            RADIX_BITS,
            False,
            BLOCK,
            BINS,
            L,
            LBIN,
        )
        hist = hist_smem.load(LBIN)
        bin_size = gl.sum(gl.where(d == (prefix & (BINS - 1)).to(gl.int32), hist, 0), 0)
        shift = LOW_BITS

    # Collect: warp w scans columns [w * seg, (w + 1) * seg) in column order.
    # Lanes without a hit write the region's last SEGMENT slots.
    L2: gl.constexpr = gl.BlockedLayout(
        [1, SEGMENT // 64], [1, 64], [NUM_WARPS, 1], [1, 0]
    )
    LI: gl.constexpr = gl.BlockedLayout(
        [1, REGION // 64], [1, 64], [NUM_WARPS, 1], [1, 0]
    )
    SL2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    CAPACITY: gl.constexpr = REGION - SEGMENT
    key_smem = gl.allocate_shared_memory(gl.uint32, [NUM_WARPS, REGION], SL2)
    col_smem = gl.allocate_shared_memory(gl.int32, [NUM_WARPS, REGION], SL2)
    col_smem.store(gl.full([NUM_WARPS, REGION], -1, gl.int32, layout=LI))
    gl.barrier()
    seg = gl.cdiv(width, NUM_WARPS * SEGMENT) * SEGMENT
    warp = gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(1, L2))[:, None]
    lane = gl.arange(0, SEGMENT, layout=gl.SliceLayout(0, L2))[None, :]
    start = warp * seg
    end = gl.minimum(start + seg, width)
    found = gl.zeros([NUM_WARPS], gl.int32, layout=gl.SliceLayout(1, L2))
    for c0 in range(0, seg, SEGMENT):
        col = start + c0 + lane
        group = (start + c0 + (lane // 4) * 4) if VEC else col
        values = gl.load(row + col, mask=group < end, other=-float("inf"))
        values = gl.where(col < end, values, -float("inf"))
        key = _select_order_key(values)
        hit = (values > -float("inf")) & ((key >> shift) >= prefix)
        h = hit.to(gl.int32)
        slot = found[:, None] + gl.cumsum(h, 1) - h
        hit = hit & (slot < CAPACITY)
        slot = gl.where(hit, slot, CAPACITY + lane)
        key_smem.scatter(key, slot, axis=1)
        col_smem.scatter(gl.where(hit, col, -1), slot, axis=1)
        found += gl.sum(h, 1)
    total = gl.sum(found, 0)
    overflow = gl.max(found, 0) > CAPACITY
    # A warp whose region overflowed: rewrite all collected keys contiguously
    # at exact column-ordered offsets, unless they do not fit.
    exact = overflow & (total <= FLAT_CAPACITY)
    key_flat = key_smem.reshape([FLAT])
    col_flat = col_smem.reshape([FLAT])
    if exact:
        written = gl.cumsum(found[:, None], 0) - found[:, None]
        dump = FLAT - NUM_WARPS * SEGMENT + warp * SEGMENT + lane
        gl.barrier()
        for c0 in range(0, seg, SEGMENT):
            col = start + c0 + lane
            group = (start + c0 + (lane // 4) * 4) if VEC else col
            values = gl.load(row + col, mask=group < end, other=-float("inf"))
            values = gl.where(col < end, values, -float("inf"))
            key = _select_order_key(values)
            hit = (values > -float("inf")) & ((key >> shift) >= prefix)
            h = hit.to(gl.int32)
            slot = gl.reshape(
                gl.where(hit, written + gl.cumsum(h, 1) - h, dump),
                [NUM_WARPS * SEGMENT],
            )
            key_flat.scatter(gl.reshape(key, [NUM_WARPS * SEGMENT]), slot, axis=0)
            col_flat.scatter(gl.reshape(col, [NUM_WARPS * SEGMENT]), slot, axis=0)
            written += gl.sum(h, 1)[:, None]
    gl.barrier()

    picked_smem = gl.allocate_shared_memory(gl.int64, [2 * TOPK], SL)
    LP: gl.constexpr = gl.BlockedLayout(
        [2 * TOPK // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
    )
    out = row_out + q * out_stride
    in_lds = (overflow == 0) | exact
    if in_lds:
        LF: gl.constexpr = gl.BlockedLayout([REGION // 64], [64], [NUM_WARPS], [0])
        key = key_flat.load(LF)
        col = col_flat.load(LF)
        # Regions mark free slots with column -1; the exact rewrite is dense.
        live = gl.where(exact, gl.arange(0, FLAT, layout=LF) < total, col >= 0)
        threshold = prefix << shift
        if keep < bin_size:
            # Resolve the K-th key inside the collected boundary bin.
            ones_f = gl.full([NUM_WARPS * REGION], 1, gl.int32, layout=LF)
            middle = prefix
            if shift == TOP:
                middle, _, keep = _select_lds_digit(
                    key,
                    live,
                    prefix,
                    0,
                    keep,
                    hist_smem,
                    ones_f,
                    d,
                    LOW_BITS,
                    RADIX_BITS,
                    BINS,
                    LBIN,
                )
            threshold, _, keep = _select_lds_digit(
                key,
                live,
                middle,
                0,
                keep,
                hist_smem,
                ones_f,
                d,
                0,
                LOW_BITS,
                BINS,
                LBIN,
            )
            is_above = live & (key > threshold)
            is_tie = live & (key == threshold)
        else:
            # The whole boundary bin is picked.
            is_above = live
            is_tie = col < -1
        tie = is_tie.to(gl.int32)
        pick = is_above | (is_tie & ((gl.cumsum(tie, 0) - tie) < keep))
        picked = pick.to(gl.int32)
        rank = gl.cumsum(picked, 0) - picked
        if HAS_CANDIDATES:
            block = gl.load(
                candidates + q * cand_stride + col // 8, mask=pick, other=-1
            ).to(gl.int64)
            logical = block * 8 + (col % 8).to(gl.int64)
            picked_smem.store(gl.full([2 * TOPK], EMPTY_ID, gl.int64, layout=LP))
            gl.barrier()
            picked_smem.scatter(
                logical,
                gl.where(
                    pick,
                    rank,
                    TOPK + gl.arange(0, NUM_WARPS * REGION, layout=LF) % TOPK,
                ),
                axis=0,
            )
        else:
            gl.store(out + rank, col, mask=pick)
    else:
        # Fallback: three-pass MSD radix select over the row. After each pass,
        # ``prefix`` holds the fixed high bits of the K-th key, ``remaining``
        # its rank among keys sharing them and ``above`` the count of larger keys.
        prefix = pivot
        prefix, above, remaining = _select_radix_pass(
            row,
            width,
            take,
            prefix,
            above,
            remaining,
            hist_smem,
            ones,
            d,
            LOW_BITS,
            RADIX_BITS,
            False,
            BLOCK,
            BINS,
            L,
            LBIN,
        )
        prefix, above, remaining = _select_radix_pass(
            row,
            width,
            take,
            prefix,
            above,
            remaining,
            hist_smem,
            ones,
            d,
            0,
            LOW_BITS,
            False,
            BLOCK,
            BINS,
            L,
            LBIN,
        )
        threshold = prefix
        # Column-order compaction: keys above the threshold plus the first
        # ``remaining`` ties (equal-score ties may pick any equivalent subset).
        picked_smem.store(gl.full([2 * TOPK], EMPTY_ID, gl.int64, layout=LP))
        gl.barrier()
        # One packed scan per chunk counts both: low 16 bits the keys above the
        # threshold, high 16 bits the ties (a chunk holds fewer than 2**16).
        n_above = 0
        n_ties = 0
        for c0 in range(0, width, BLOCK):
            col = c0 + gl.arange(0, BLOCK, layout=L)
            values = gl.load(row + col, mask=col < width, other=-float("inf"))
            key = _select_order_key(values)
            live = values > -float("inf")
            is_above = live & (key > threshold)
            is_tie = live & (key == threshold)
            packed = is_above.to(gl.int32) | (is_tie.to(gl.int32) << 16)
            before = gl.cumsum(packed, 0) - packed
            ties_before = n_ties + (before >> 16)
            take_it = is_above | (is_tie & (ties_before < remaining))
            slot = n_above + (before & 0xFFFF) + gl.minimum(ties_before, remaining)
            if HAS_CANDIDATES:
                block = gl.load(
                    candidates + q * cand_stride + col // 8, mask=take_it, other=-1
                ).to(gl.int64)
                logical = block * 8 + (col % 8).to(gl.int64)
            else:
                logical = col.to(gl.int64)
            # Unpicked lanes write the scratch half [TOPK, 2*TOPK).
            picked_smem.scatter(
                logical, gl.where(take_it, slot, TOPK + col % TOPK), axis=0
            )
            total = gl.sum(packed, 0)
            n_above += total & 0xFFFF
            n_ties += total >> 16

    LT: gl.constexpr = gl.BlockedLayout([1, 64], [64, 1], [NUM_WARPS, 1], [1, 0])
    LR: gl.constexpr = gl.SliceLayout(1, LT)
    LC: gl.constexpr = gl.SliceLayout(0, LT)
    i = gl.arange(0, TOPK, layout=LR)
    if HAS_CANDIDATES or not in_lds:
        # Rank sort of the picked ids: ids are unique, unpicked slots hold the
        # int64 max and rank after them.
        gl.barrier()
        mine = picked_smem.slice(0, TOPK).load(LR)
        rank = i
        # Column order already is position order without candidates, and with
        # candidate blocks listed in ascending order; sort only otherwise.
        previous = gl.gather(mine, gl.maximum(i - 1, 0), axis=0)
        descents = gl.sum(((i > 0) & (i < count) & (mine < previous)).to(gl.int32), 0)
        if descents > 0:
            rank = gl.zeros([TOPK], gl.int32, layout=LR)
            # Slots past ``count`` hold EMPTY_ID and rank no picked id, so skip
            # them.
            chunks = picked_smem.reinterpret(shape=[2 * TOPK // 64, 64], layout=SL)
            for j in range(0, gl.cdiv(count, 64)):
                other = chunks.index(j).load(LC)
                rank += gl.sum(
                    (gl.expand_dims(other, 0) < gl.expand_dims(mine, 1)).to(gl.int32),
                    axis=1,
                )
        gl.store(out + rank, mine.to(gl.int32), mask=i < count)
    gl.store(
        out + i,
        gl.full([TOPK], -1, gl.int32, layout=LR),
        mask=(i >= count) & (i < topk),
    )
    gl.store(row_lens + q, count)


def launch_gluon_dsv41_index_topk_select_gfx950(
    logits, candidates, live_widths, topk, row_out, row_lens, take=None
):
    """Select CSA2 rows from scored logits, position-sorted.

    Args:
        logits: FP32 [Q, width] scores; -inf marks unscored rows.
        candidates: None, or int32/int64 [Q, blocks] block ids whose 8-row
            blocks map logits column c to logical row
            ``candidates[q, c // 8] * 8 + c % 8``.
        live_widths: None, or strided int32/int64 [Q] counts of leading columns
            that may be finite; row q's columns at or past ``live_widths[q]`` must be
            -inf and are not read. Full selection passes the visible lengths, so
            a page-table-capacity row costs its context, not its capacity.
        topk: Selection capacity in [1, 2048]; V4.1 uses 512.
        row_out: Int32 [Q, topk] destination. Row q receives the logical ids of
            its ``min(take, width)`` largest finite logits in ascending order,
            then -1 padding. Equal-score boundary ties keep the lowest columns.
        row_lens: Int32 [Q] destination for the number of ids written.
        take: None for ``topk``, or the number of ids to select, <= topk.

    Returns:
        None.
    """
    queries, width = logits.shape
    if not queries:
        return
    topk = int(topk)
    if not 1 <= topk <= 2048:
        raise ValueError(f"select topk must be in [1, 2048], got {topk}")
    take = topk if take is None else int(take)
    if not 0 <= take <= topk:
        raise ValueError(f"select take must be in [0, {topk}], got {take}")
    if (
        logits.stride(1) != 1
        or row_out.stride(1) != 1
        or row_lens.stride(0) != 1
        or (candidates is not None and candidates.stride(1) != 1)
    ):
        raise ValueError("select requires unit inner strides")
    # The kernel writes row_out[q, :topk] and row_lens[q], and reads
    # candidates[q, col // 8] for every logits column.
    if row_out.shape[0] != queries or row_out.shape[1] < topk:
        raise ValueError(f"row_out must be [{queries}, >= {topk}]")
    if row_lens.shape != (queries,):
        raise ValueError(f"row_lens must be [{queries}]")
    if candidates is not None and (
        candidates.shape[0] != queries or candidates.shape[1] * 8 < width
    ):
        raise ValueError(f"candidates must be [{queries}, >= {-(-width // 8)}]")
    if live_widths is not None and (
        live_widths.shape != (queries,)
        or live_widths.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(f"live_widths must be int32/int64 [{queries}]")
    cand = logits if candidates is None else candidates
    grid = (queries,)
    gluon_dsv41_index_topk_select_gfx950[grid](
        logits,
        cand,
        logits if live_widths is None else live_widths,
        row_out,
        row_lens,
        logits.stride(0),
        0 if candidates is None else candidates.stride(0),
        row_out.stride(0),
        0 if live_widths is None else live_widths.stride(0),
        width,
        topk,
        min(take, width),
        TOPK=max(512, triton.next_power_of_2(topk)),
        HAS_CANDIDATES=candidates is not None,
        HAS_LIMITS=live_widths is not None,
        RADIX_BITS=_SELECT_RADIX_BITS,
        NUM_WARPS=_SELECT_WARPS,
        REGION=512,
        SEGMENT=256,
        VEC=width % 4 == 0,
        num_warps=_SELECT_WARPS,
    )


@gluon.jit(do_not_specialize=("out_stride", "n_cap", "visible_capacity"))
def gluon_dsv41_index_block_max_gfx950(
    logits,
    visible,
    block_max,
    block_count,
    logits_stride,
    out_stride,
    n_cap,
    visible_capacity,
    BLOCKS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Maxima of a row's visible 8-row blocks; the newest finite one is +inf."""
    q = gl.program_id(0).to(gl.int64)
    LB: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [NUM_WARPS, 1], [1, 0])
    LO: gl.constexpr = gl.SliceLayout(1, LB)
    seen = gl.minimum(
        gl.maximum(gl.load(visible + q).to(gl.int64), 0), visible_capacity
    ).to(gl.int32)
    n = gl.minimum(n_cap, (seen + 7) // 8)
    latest = gl.where(seen > 0, (seen - 1) // 8, -1)
    if gl.program_id(1) == 0:
        gl.store(block_count + q, n)
    row = logits + q * logits_stride
    lane = gl.arange(0, 8, layout=gl.SliceLayout(0, LB))[None, :]
    for j0 in range(gl.program_id(1) * BLOCKS, n, gl.num_programs(1) * BLOCKS):
        block = j0 + gl.arange(0, BLOCKS, layout=LO)
        column = block[:, None] * 8 + lane
        # Whole blocks load as vectors; columns past the visible rows are -inf.
        x = gl.load(row + column, mask=block[:, None] < n, other=-float("inf"))
        x = gl.max(gl.where(column < seen, x, -float("inf")), axis=1)
        x = gl.where((block == latest) & (x > -float("inf")), float("inf"), x)
        gl.store(block_max + q * out_stride + block, x, mask=block < n)


def launch_gluon_dsv41_index_block_select_gfx950(
    logits, visible, candidate_topk, block_out, block_lens, visible_capacity
):
    """Select each query's best 8-row blocks, position-sorted.

    Has the ``index_block_select`` contract: block scores are 8-row block
    maxima of the logits below ``visible[q]`` (clamped to
    ``visible_capacity``), the block holding row ``visible[q] - 1`` is forced
    in when it has a finite score, row q receives its
    ``min(candidate_topk, width // 8)`` best block ids in ascending order and
    then -1, and equal-score boundary ties keep the lowest blocks.
    """
    queries, width = logits.shape
    if not queries:
        return
    k = min(int(candidate_topk), width // 8)
    if k < 1:
        block_out.fill_(-1)
        block_lens.zero_()
        return
    if logits.stride(1) != 1 or block_out.stride(1) != 1:
        raise ValueError("block select requires unit inner strides")
    if queries > 1 and visible.stride(0) != 1:
        raise ValueError("block select requires contiguous visible row counts")
    blocks = width // 8
    block_max = torch.empty(
        (queries, blocks), dtype=torch.float32, device=logits.device
    )
    block_count = torch.empty(queries, dtype=torch.int32, device=logits.device)
    per_cta = 256
    grid = (queries, min(triton.cdiv(blocks, per_cta), max(1, 4096 // queries)))
    gluon_dsv41_index_block_max_gfx950[grid](
        logits,
        visible,
        block_max,
        block_count,
        logits.stride(0),
        block_max.stride(0),
        blocks,
        int(visible_capacity),
        BLOCKS=per_cta,
        NUM_WARPS=4,
        num_warps=4,
    )
    launch_gluon_dsv41_index_topk_select_gfx950(
        block_max,
        None,
        block_count,
        block_out.shape[1],
        block_out,
        block_lens,
        take=k,
    )
