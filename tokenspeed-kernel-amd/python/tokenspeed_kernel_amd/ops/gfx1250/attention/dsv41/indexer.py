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

Wave32 WMMA port of the GFX950 CSA2 scorer. MXFP4 index-K is dequantized to
BF16; 32 padded heads score a 32-wide history tile. The tokenspeed-kernel
adapter owns query preparation, bounded query tiling, and selection.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

__all__ = ["dsv41_index_logits_gfx1250"]

_HEAD_DIM = 128
_PACKED_DIM = gl.constexpr(_HEAD_DIM // 2)
_SCALE_DIM = gl.constexpr(_HEAD_DIM // 32)
_BLOCK_N = 32
_CHUNK_N = 256
_NUM_WARPS = 2


@gluon.jit
def _e2m1_decode(code):
    a = code & 7
    value = gl.where(
        a < 4,
        a.to(gl.float32) * 0.5,
        gl.where(a == 4, 2.0, gl.where(a == 5, 3.0, gl.where(a == 6, 4.0, 6.0))),
    )
    return gl.where((code & 8) != 0, -value, value)


@gluon.constexpr_function
def _wmma_layout(NUM_WARPS: gl.constexpr):
    return gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[1, 0]] if NUM_WARPS == 2 else [[1, 0], [0, 1]],
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )


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
    TABLE_WIDTH: gl.constexpr,
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
    safe_page = gl.minimum(gl.maximum(logical_page, 0), TABLE_WIDTH - 1)
    physical = gl.amd.cdna5.buffer_load(
        page_table,
        (query * table_stride + safe_page).to(gl.int32),
        mask=valid,
        other=-1,
    ).to(gl.int64)
    valid = valid & (physical >= 0) & (physical < num_pages)
    return gl.where(valid, physical, 0), logical % PAGE_SIZE, valid


@gluon.jit
def _load_query(
    q,
    weights,
    token,
    stride_q_token,
    stride_q_head,
    stride_w_token,
    stride_w_head,
    q_load_layout: gl.constexpr,
    q_dot_layout: gl.constexpr,
    wmma_layout: gl.constexpr,
    HEAD_DIM: gl.constexpr,
):
    heads = gl.arange(0, 32, layout=gl.SliceLayout(1, q_load_layout))[:, None]
    dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, q_load_layout))[None, :]
    query = gl.amd.cdna5.buffer_load(
        q,
        (token * stride_q_token + heads * stride_q_head + dims).to(gl.int32),
    )
    weight_heads = gl.arange(0, 32, layout=gl.SliceLayout(1, wmma_layout))
    head_weights = gl.amd.cdna5.buffer_load(
        weights,
        (token * stride_w_token + weight_heads * stride_w_head).to(gl.int32),
    ).to(gl.float32)
    return gl.convert_layout(query, q_dot_layout), head_weights


def _index_launch_metadata(grid, kernel, args):
    """Describe the score capacity without reading device-resident lengths."""
    queries, width = args["logits"].shape
    heads = args["q"].shape[1]
    return {
        "name": kernel.name,
        "flops16": 2 * queries * heads * width * 128,
        "bytes": queries * width * 68
        + args["q"].numel() * args["q"].element_size() * grid[1]
        + args["logits"].numel() * args["logits"].element_size(),
    }


@gluon.jit(
    launch_metadata=_index_launch_metadata,
    do_not_specialize=(
        "stride_q_token",
        "stride_q_head",
        "stride_w_token",
        "stride_w_head",
        "table_stride",
        "cand_stride",
        "logits_stride",
        "page_stride_bytes",
        "num_pages",
    ),
)
def gluon_dsv41_index_topk_gfx1250(
    q,
    weights,
    index_k_cache,
    visible,
    page_table,
    candidates,
    logits,
    stride_q_token,
    stride_q_head,
    stride_w_token,
    stride_w_head,
    table_stride,
    cand_stride,
    logits_stride,
    page_stride_bytes,
    num_pages,
    max_candidates,
    PAGE_SIZE: gl.constexpr,
    TABLE_WIDTH: gl.constexpr,
    CANDIDATES: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    SCORE_CHUNK: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token = gl.program_id(0)
    split = gl.program_id(1)
    vis = gl.minimum(
        gl.maximum(gl.load(visible + token).to(gl.int32), 0),
        TABLE_WIDTH * PAGE_SIZE,
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

    wmma_layout: gl.constexpr = _wmma_layout(NUM_WARPS)
    k_width: gl.constexpr = 8
    q_dot_layout: gl.constexpr = gl.DotOperandLayout(0, wmma_layout, k_width=k_width)
    k_dot_layout: gl.constexpr = gl.DotOperandLayout(1, wmma_layout, k_width=k_width)
    q_load_layout: gl.constexpr = gl.BlockedLayout(
        [1, k_width],
        [4, 8],
        [NUM_WARPS, 1],
        [1, 0],
    )
    query, head_weights = _load_query(
        q,
        weights,
        token,
        stride_q_token,
        stride_q_head,
        stride_w_token,
        stride_w_head,
        q_load_layout,
        q_dot_layout,
        wmma_layout,
        HEAD_DIM,
    )
    output_layout: gl.constexpr = gl.SliceLayout(0, wmma_layout)
    output_columns = gl.arange(0, BLOCK_N, layout=output_layout)
    dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(1, k_dot_layout))[:, None]
    columns = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, k_dot_layout))[None, :]

    for tile_offset in range(0, CHUNK_N, BLOCK_N):
        tile_start = candidate_start + tile_offset
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
            vis,
            PAGE_SIZE,
            TABLE_WIDTH,
            CANDIDATES,
        )
        packed = gl.load(
            index_k_cache
            + pages * page_stride_bytes
            + page_rows.to(gl.int64) * _PACKED_DIM
            + dims // 2,
            mask=valid,
            other=0,
        )
        value = _e2m1_decode((packed.to(gl.int32) >> ((dims % 2) * 4)) & 15)
        scale_u8 = gl.load(
            index_k_cache
            + pages * page_stride_bytes
            + PAGE_SIZE * _PACKED_DIM
            + page_rows.to(gl.int64) * _SCALE_DIM
            + dims // 32,
            mask=valid,
            other=0,
        )
        scale = gl.where(
            scale_u8 == 0,
            2.0**-127,
            (scale_u8.to(gl.int32) << 23).to(gl.float32, bitcast=True),
        )
        key = gl.where(valid, (value * scale).to(gl.bfloat16), 0.0)
        acc = gl.zeros([32, BLOCK_N], gl.float32, layout=wmma_layout)
        head_scores = gl.amd.cdna5.wmma(query, key, acc)
        head_scores = gl.maximum(head_scores, 0.0, propagate_nan=tl.PropagateNan.ALL)
        scores = gl.sum(head_scores * head_weights[:, None], axis=0)
        store_pos = tile_start + output_columns
        live = store_pos < candidate_end
        _, _, live = _csa2_page_rows(
            store_pos,
            live,
            page_table,
            candidates,
            token,
            table_stride,
            cand_stride,
            num_pages,
            vis,
            PAGE_SIZE,
            TABLE_WIDTH,
            CANDIDATES,
        )
        gl.store(
            logits + token * logits_stride + store_pos,
            scores,
            mask=(store_pos < max_candidates) & live,
        )


def dsv41_index_logits_gfx1250(
    q,
    w,
    cache_2d,
    table,
    visible,
    candidates,
    logits,
    score_chunk_size,
):
    """Score prepared 32-head BF16 queries into caller-owned CSA2 logits.

    Args:
        q: Quantized/dequantized BF16 queries shaped [T, 32, 128].
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
    gluon_dsv41_index_topk_gfx1250[(queries, triton.cdiv(width, score_chunk_size))](
        q,
        w,
        cache_2d,
        visible,
        table,
        cand,
        logits,
        q.stride(0),
        q.stride(1),
        w.stride(0),
        w.stride(1),
        table.stride(0),
        cand.stride(0),
        logits.stride(0),
        int(cache_2d.stride(0)),
        int(cache_2d.shape[0]),
        width,
        PAGE_SIZE=64,
        TABLE_WIDTH=int(table.shape[1]),
        CANDIDATES=-1 if candidates is None else int(candidates.shape[1]),
        HEAD_DIM=_HEAD_DIM,
        SCORE_CHUNK=score_chunk_size,
        BLOCK_N=_BLOCK_N,
        CHUNK_N=chunk_n,
        NUM_WARPS=_NUM_WARPS,
        num_warps=_NUM_WARPS,
    )
