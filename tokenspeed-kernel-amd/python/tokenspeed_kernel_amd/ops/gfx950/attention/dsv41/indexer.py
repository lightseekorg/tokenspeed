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

Scores page-planar MXFP4 index-K with the V4 ``mfma_scaled`` e2m1 tile.
The tokenspeed-kernel adapter owns query packing, bounded query tiling, and
selection.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4.indexer import (
    _BLOCK_N,
    _CHUNK_N,
    _HEADS_PER_MFMA,
    _PACKED_DIM,
    _PAGE_SIZE,
    _SCALE_DIM,
    _indexer_mfma_layouts,
    _load_query_group,
)

__all__ = [
    "dsv41_index_logits_gfx950",
    "launch_gluon_dsv41_index_topk_select_gfx950",
]

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


@gluon.jit
def _score_csa2_group(
    query,
    query_scales,
    head_weights,
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
    mfma_layout: gl.constexpr,
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
    valid = positions < candidate_end
    pages, page_rows, valid = _csa2_page_rows(
        positions,
        valid,
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
    scale_pages, scale_page_rows, scale_valid = _csa2_page_rows(
        scale_positions,
        scale_valid,
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
def gluon_dsv41_index_topk_gfx950(
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
    vis = gl.minimum(
        gl.maximum(gl.load(visible + token).to(gl.int32), 0),
        table_width * PAGE_SIZE,
    )
    if CANDIDATES >= 0:
        width = gl.where(vis > 0, CANDIDATES * 8, 0)
    else:
        width = vis
    candidate_start = split * SCORE_CHUNK
    candidate_end = gl.minimum(width, candidate_start + SCORE_CHUNK)
    candidate_end = gl.minimum(candidate_end, max_candidates)
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
    output_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    output_columns = gl.arange(0, BLOCK_N, layout=output_layout)
    for tile_offset in range(0, CHUNK_N, BLOCK_N):
        tile_start = candidate_start + tile_offset
        scores = _score_csa2_group(
            query_0,
            query_scale_0,
            weight_0,
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
            mfma_layout,
            dot_b_layout,
            b_scale_layout,
            PAGE_SIZE,
            table_width,
            CANDIDATES,
        )
        scores += _score_csa2_group(
            query_1,
            query_scale_1,
            weight_1,
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
            mfma_layout,
            dot_b_layout,
            b_scale_layout,
            PAGE_SIZE,
            table_width,
            CANDIDATES,
        )
        positions = tile_start + output_columns
        live = positions < candidate_end
        _, _, live = _csa2_page_rows(
            positions,
            live,
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
        logits: FP32 destination shaped [T, scored_rows], initialized to -inf.
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
    gluon_dsv41_index_topk_gfx950[(queries, triton.cdiv(width, score_chunk_size))](
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
        waves_per_eu=2,
    )


# Radix-select digit split of the 32-bit order key: 11 + 11 + 10 bits.
_SELECT_RADIX_BITS = 11
_SELECT_WARPS = 8


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
    hist_smem.store(gl.zeros([BINS], gl.int32, layout=LBIN))
    for c0 in range(0, width, BLOCK):
        col = c0 + gl.arange(0, BLOCK, layout=L)
        values = gl.load(row + col, mask=col < width, other=-float("inf"))
        key = _select_order_key(values)
        live = values > -float("inf")
        if not FIRST:
            live = live & ((key >> (SHIFT + DIGIT_BITS)) == prefix)
        digit = ((key >> SHIFT) & ((1 << DIGIT_BITS) - 1)).to(gl.int32)
        hist_smem.atomic_scatter_add(ones, digit, axis=0, mask=live)
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


@gluon.jit(
    launch_metadata=_select_launch_metadata,
    do_not_specialize=(
        "logits_stride",
        "cand_stride",
        "out_stride",
        "width",
        "topk",
        "take",
    ),
)
def gluon_dsv41_index_topk_select_gfx950(
    logits,  # [Q, width] fp32, -inf = unscored
    candidates,  # [Q, blocks] int32/int64 block ids, or logits when unused
    row_out,  # [Q, TOPK] int32
    row_lens,  # [Q] int32
    logits_stride,
    cand_stride,
    out_stride,
    width,
    topk,  # row_out width
    take,  # min(topk, width)
    TOPK: gl.constexpr,  # power-of-two capacity >= topk, at least 256
    HAS_CANDIDATES: gl.constexpr,
    RADIX_BITS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Top-``take`` finite logits of one query row as position-sorted ids.

    A three-pass MSD radix select finds the order key of the K-th largest
    finite logit (K = min(take, finite count)); a column-order compaction then
    keeps every larger key plus the lowest-column ties, and an all-pairs rank
    sorts the picked logical ids. Positions >= K are written as -1.
    """
    BLOCK: gl.constexpr = 4 * 64 * NUM_WARPS
    BINS: gl.constexpr = 1 << RADIX_BITS
    L: gl.constexpr = gl.BlockedLayout([4], [64], [NUM_WARPS], [0])
    LBIN: gl.constexpr = gl.BlockedLayout(
        [BINS // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
    )
    SL: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    q = gl.program_id(0).to(gl.int64)
    row = logits + q * logits_stride
    hist_smem = gl.allocate_shared_memory(gl.int32, [BINS], SL)
    ones = gl.full([BLOCK], 1, gl.int32, layout=L)
    d = gl.arange(0, BINS, layout=LBIN)

    # MSD radix select: after each pass, ``prefix`` holds the fixed high bits
    # of the K-th key, ``remaining`` its rank among keys sharing them and
    # ``above`` the count of larger keys.
    LOW_BITS: gl.constexpr = 32 - 2 * RADIX_BITS
    prefix, above, remaining = _select_radix_pass(
        row,
        width,
        take,
        0,
        0,
        0,
        hist_smem,
        ones,
        d,
        32 - RADIX_BITS,
        RADIX_BITS,
        True,
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
    count = above + remaining

    # Column-order compaction: keys above the threshold plus the first
    # ``remaining`` ties (equal-score ties may pick any equivalent subset).
    picked_smem = gl.allocate_shared_memory(gl.int64, [2 * TOPK], SL)
    LP: gl.constexpr = gl.BlockedLayout(
        [2 * TOPK // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
    )
    picked_smem.store(gl.full([2 * TOPK], 0x7FFFFFFFFFFFFFFF, gl.int64, layout=LP))
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
        picked_smem.scatter(logical, gl.where(take_it, slot, TOPK + col % TOPK), axis=0)
        total = gl.sum(packed, 0)
        n_above += total & 0xFFFF
        n_ties += total >> 16

    # Rank sort of the picked ids: ids are unique, unpicked slots hold the
    # int64 max and rank after them.
    LT: gl.constexpr = gl.BlockedLayout([1, 64], [64, 1], [NUM_WARPS, 1], [1, 0])
    LR: gl.constexpr = gl.SliceLayout(1, LT)
    LC: gl.constexpr = gl.SliceLayout(0, LT)
    mine = picked_smem.slice(0, TOPK).load(LR)
    i = gl.arange(0, TOPK, layout=LR)
    rank = i
    # Column order already is position order without candidates, and with
    # candidate blocks listed in ascending order; sort only otherwise.
    previous = gl.gather(mine, gl.maximum(i - 1, 0), axis=0)
    descents = gl.sum(((i > 0) & (i < count) & (mine < previous)).to(gl.int32), 0)
    if descents > 0:
        rank = gl.zeros([TOPK], gl.int32, layout=LR)
        for j0 in gl.static_range(0, TOPK, 64):
            other = picked_smem.slice(j0, 64).load(LC)
            rank += gl.sum(
                (gl.expand_dims(other, 0) < gl.expand_dims(mine, 1)).to(gl.int32),
                axis=1,
            )
    out = row_out + q * out_stride
    gl.store(out + rank, mine.to(gl.int32), mask=i < count)
    gl.store(
        out + i,
        gl.full([TOPK], -1, gl.int32, layout=LR),
        mask=(i >= count) & (i < topk),
    )
    gl.store(row_lens + q, count)


def launch_gluon_dsv41_index_topk_select_gfx950(
    logits, candidates, topk, row_out, row_lens
):
    """Select CSA2 rows from scored logits, position-sorted.

    Args:
        logits: FP32 [Q, width] scores; -inf marks unscored rows.
        candidates: None, or int32/int64 [Q, blocks] block ids whose 8-row
            blocks map logits column c to logical row
            ``candidates[q, c // 8] * 8 + c % 8``.
        topk: Selection capacity in [1, 1024]; V4.1 uses 512.
        row_out: Int32 [Q, topk] destination. Row q receives the logical ids of
            its ``min(topk, width)`` largest finite logits in ascending order,
            then -1 padding. Equal-score boundary ties keep the lowest columns.
        row_lens: Int32 [Q] destination for the number of ids written.

    Returns:
        None.
    """
    queries, width = logits.shape
    if not queries:
        return
    topk = int(topk)
    if not 1 <= topk <= 1024:
        raise ValueError(f"select topk must be in [1, 1024], got {topk}")
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
    cand = logits if candidates is None else candidates
    gluon_dsv41_index_topk_select_gfx950[(queries,)](
        logits,
        cand,
        row_out,
        row_lens,
        logits.stride(0),
        0 if candidates is None else candidates.stride(0),
        row_out.stride(0),
        width,
        topk,
        min(topk, width),
        TOPK=max(256, triton.next_power_of_2(topk)),
        HAS_CANDIDATES=candidates is not None,
        RADIX_BITS=_SELECT_RADIX_BITS,
        NUM_WARPS=_SELECT_WARPS,
        num_warps=_SELECT_WARPS,
    )
