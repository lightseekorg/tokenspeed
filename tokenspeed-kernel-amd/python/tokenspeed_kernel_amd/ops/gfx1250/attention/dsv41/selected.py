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

"""Fused two-reader DeepSeek V4.1 selected attention for GFX1250.

Wave32 WMMA port of the GFX950 kernel. Reads page-planar SWA (E4M3/E8M0,
528B) and global (E2M1/E4M3, 288B) caches directly. One online softmax
includes both segments and a single sink.

A workgroup owns one query row, up to 64 heads and a contiguous share of the
row's selected tiles, so each selected row is loaded and dequantized once per
16-head group that needs it. Head groups whose queries are all zero (the
64/128-head padding of the decode layout) skip the score WMMAs: their scores
are exactly zero, so their output is the V row sum scaled per head by the
sink, computed once for all such groups. Small batches split the selected
tiles over workgroups and merge the partial softmax states in a reduce kernel.
See the "DeepSeek V4.1 selected attention" section of ``ops/README.md``.
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

__all__ = ["launch_gluon_dsv41_selected_attention_gfx1250"]

_SWA_ROW_BYTES = 528
_GLOBAL_ROW_BYTES = 288
_PAGE_ROWS = 64
_HEAD_DIM = 512
_BLOCK_H = 16
# Four waves split a tile's rows for QK, 16 rows each.
_TILE_K = 64
# Rows are split over workgroups only while the grid still fits the 256 CUs
# once.
_TARGET_CTAS = 256
_MAX_SPLITS = 16
# The reduce splits head dims over workgroups too: one workgroup per token and
# 16-head group alone is too few to stream a small batch's partials.
_REDUCE_DIM_BLOCK = 64
# One workgroup covers at most 64 query heads (four 16-head groups).
_HEAD_GROUPS_PER_CTA = 4
# Fewer head groups per workgroup re-read each selected row once more per head
# block. That pays off while rows x head blocks x tiles stays within this
# budget (about 2.5 tiles per CU).
_HEAD_BLOCK_TILES = 640


@gluon.jit
def _tile_slots(slots, slot_row, tile_idx, length, width, capacity, TILE_K, LAYOUT):
    """Return (clamped slot, valid) for one tile of a query row's selections."""
    positions = tile_idx * TILE_K + gl.arange(0, TILE_K, layout=LAYOUT)
    in_range = (positions < length) & (positions < width)
    slot = gl.amd.cdna5.buffer_load(
        slots + slot_row, positions, mask=in_range, other=-1
    )
    valid = in_range & (slot >= 0) & (slot < capacity)
    return gl.where(valid, slot, 0), valid


@gluon.jit
def _load_swa_rows(cache, slot, valid, page_stride, RAW_LAYOUT: gl.constexpr):
    """Load SWA rows: E4M3 bytes [TILE_K, 32, 16] and E8M0 scales [TILE_K, 32].

    A lane owns 16 contiguous value bytes of a row (one dwordx4). Chunk ``c``
    covers dims ``16c..16c+15``, whose E8M0 scale is byte ``c // 2``. Invalid
    rows (clamped to slot 0) load zero values and zero scales: an unwritten
    0xFF scale would dequantize to Inf and turn their zeros into NaN.
    """
    scale_layout: gl.constexpr = gl.SliceLayout(2, RAW_LAYOUT)
    page_base = (slot // 64).to(tl.int64) * page_stride
    row = slot % 64
    chunks = gl.arange(0, 32, layout=gl.SliceLayout(0, scale_layout))
    lanes = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, RAW_LAYOUT)))
    value_offsets = (
        (page_base + row * 512)[:, None, None]
        + (chunks * 16)[None, :, None]
        + lanes[None, None, :]
    )
    scale_offsets = (page_base + 64 * 512 + row * 16)[:, None] + (chunks // 2)[None, :]
    values = gl.load(cache + value_offsets, mask=valid[:, None, None], other=0)
    scales = gl.load(cache + scale_offsets, mask=valid[:, None], other=0)
    return values, scales


@gluon.jit
def _dequant_swa_rows(values, scales, RAW_LAYOUT: gl.constexpr):
    """E4M3 times E8M0 scale into BF16 [TILE_K, 512].

    The upcast runs at a uniform unit scale: a per-lane E8M0 scale reaches the
    wrong lanes on GFX1250. The power-of-two scale then multiplies exactly.
    """
    unit = gl.full(values.shape, 127, dtype=gl.uint8, layout=RAW_LAYOUT)
    kv = gl.amd.cdna5.scaled_upcast(
        values.to(gl.float8e4nv, bitcast=True), unit, gl.bfloat16
    )
    # E8M0 byte s is 2^(s - 127): BF16 exponent bits, with s = 0 as 2^-127.
    bits = gl.where(scales == 0, 0x40, scales.to(gl.int16) << 7)
    kv = kv * bits.to(gl.bfloat16, bitcast=True)[:, :, None]
    return kv.reshape([kv.shape[0], 512])


@gluon.jit
def _load_global_rows(cache, slot, valid, page_stride, RAW_LAYOUT: gl.constexpr):
    """Load global rows: E2M1 bytes [TILE_K, 16, 2, 8], E4M3 scales [TILE_K, 16, 2].

    A lane owns 16 contiguous packed bytes (32 dims) of a row: chunk ``c`` and
    scale group ``g`` of 16 dims (scale byte ``2c + g``). Invalid rows
    (clamped to slot 0) get zero bytes and zero scales.
    """
    scale_layout: gl.constexpr = gl.SliceLayout(3, RAW_LAYOUT)
    chunk_layout: gl.constexpr = gl.SliceLayout(0, scale_layout)
    page_base = (slot // 64).to(tl.int64) * page_stride
    row = slot % 64
    chunks = gl.arange(0, 16, layout=gl.SliceLayout(1, chunk_layout))
    groups = gl.arange(0, 2, layout=gl.SliceLayout(0, chunk_layout))
    scale_idx = chunks[:, None] * 2 + groups[None, :]
    lanes = gl.arange(
        0,
        8,
        layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, RAW_LAYOUT))),
    )
    value_offsets = (
        (page_base + row * 256)[:, None, None, None]
        + (scale_idx * 8)[None, :, :, None]
        + lanes[None, None, None, :]
    )
    scale_offsets = (page_base + 64 * 256 + row * 32)[:, None, None] + scale_idx[
        None, :, :
    ]
    values = gl.load(cache + value_offsets, mask=valid[:, None, None, None], other=0)
    scales = gl.load(cache + scale_offsets, mask=valid[:, None, None], other=0)
    return values, scales


@gluon.jit
def _dequant_global_rows(values, scales, OUT_LAYOUT: gl.constexpr):
    """E2M1 (even dim in the low nibble) times E4M3 into BF16 [TILE_K, 512].

    The upcast runs at unit E8M0 scale. E4M3 converts exactly to BF16, and an
    E2M1 value times an E4M3 scale needs at most five significant bits, so the
    product is exact before its single BF16 rounding.
    """
    unit = gl.full([values.shape[0], 16, 2, 16], 127, dtype=gl.uint8, layout=OUT_LAYOUT)
    kv = gl.convert_layout(
        gl.amd.cdna5.scaled_upcast(values, unit, gl.bfloat16, axis=3), OUT_LAYOUT
    )
    scale = gl.convert_layout(
        scales.to(gl.float8e4nv, bitcast=True).to(gl.bfloat16),
        gl.SliceLayout(3, OUT_LAYOUT),
    )
    kv = kv * scale[:, :, :, None]
    return kv.reshape([kv.shape[0], 512])


@gluon.jit
def _attend_tile(
    kv,
    kv_shared,
    valid_rows,
    q_dot,
    valid_heads,
    softmax_scale,
    max_value,
    denominator,
    accumulator,
    row_sum,
    row_count,
    active,
    with_row_sum,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
):
    """Fold one dequantized BF16 tile [TILE_K, 512] into the softmax."""
    k_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=8
    )
    p_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=WMMA_LAYOUT, k_width=8
    )
    kv_shared.store(kv)
    # V is read from LDS only after QK, so the K and V operands are never live
    # together; that register saving outweighs the second LDS read.
    if active:
        k_dot = kv_shared.permute([1, 0]).load(k_dot_layout)
        scores = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=WMMA_LAYOUT)
        scores = gl.amd.cdna5.wmma(q_dot, k_dot, scores) * softmax_scale
        scores = gl.where(
            valid_heads[:, None] & valid_rows[None, :], scores, -float("inf")
        )
        tile_max = gl.max(scores, axis=1)
        next_max = gl.maximum(max_value, tile_max)
        safe_next_max = gl.where(next_max > -float("inf"), next_max, 0.0)
        previous_scale = gl.exp(max_value - safe_next_max)
        probabilities = gl.exp(scores - safe_next_max[:, None])
        denominator = previous_scale * denominator + gl.sum(probabilities, axis=1)
        accumulator *= previous_scale[:, None]
        p_dot = gl.convert_layout(probabilities.to(gl.bfloat16), p_dot_layout)
        v_dot = kv_shared.load(k_dot_layout)
        accumulator = gl.amd.cdna5.wmma(p_dot, v_dot, accumulator)
        max_value = next_max
    if with_row_sum:
        # Invalid rows dequantize to zero, so all-one weights give the V row sum.
        ones = gl.full([BLOCK_H, TILE_K], 1.0, dtype=gl.bfloat16, layout=p_dot_layout)
        row_sum = gl.amd.cdna5.wmma(ones, kv_shared.load(k_dot_layout), row_sum)
        row_count += gl.sum(valid_rows.to(gl.int32), axis=0)
    return max_value, denominator, accumulator, row_sum, row_count


@gluon.jit
def _inverse_rope(values, dims, rows, rope_row, dim_base, ROPE_DIM: gl.constexpr):
    """Rotate the interleaved RoPE tail of BF16 [rows, dims] output tiles.

    dims are absolute head dims; dim_base localizes the gather to the tile.
    Only the last ROPE_DIM of 512 rotate. rope_row points at the token's
    inverse cos/sin table row.
    """
    start: gl.constexpr = 512 - ROPE_DIM
    rope = dims >= start
    pair = gl.where(rope, (dims - start) // 2, 0)
    cosine = gl.load(rope_row + pair, mask=rope, other=1.0)
    sine = gl.load(rope_row + ROPE_DIM // 2 + pair, mask=rope, other=0.0)
    value = values.to(gl.float32)
    partner = gl.gather(
        value, ((dims - dim_base) ^ 1)[None, :] + rows[:, None] * 0, axis=1
    )
    first = value * cosine[None, :]
    second = partner * sine[None, :]
    rotated = gl.where(((dims & 1) != 0)[None, :], first + second, first - second)
    return gl.where(rope[None, :], rotated, value).to(gl.bfloat16)


def _selected_attention_metadata(grid, kernel, args):
    """Report the attention FLOPs and DRAM bytes of one launch.

    Device lengths are not read, so the counts are a capacity bound: every
    selected slot is valid and every head is computed. Split launches store
    FP32 ``[heads, 512]`` partials instead of the BF16 output.
    """
    tokens = args["q"].shape[0]
    heads = args["num_heads"]
    splits = grid[1]
    swa_width = args["swa_width"]
    global_width = args["global_width"] if args["HAS_GLOBAL"] else 0
    output_bytes = heads * _HEAD_DIM * (2 if splits == 1 else 4 * splits)
    return {
        "name": kernel.name,
        "flops16": 4 * tokens * heads * (swa_width + global_width) * _HEAD_DIM,
        "bytes": tokens
        * (
            swa_width * (_SWA_ROW_BYTES + 4)
            + global_width * (_GLOBAL_ROW_BYTES + 4)
            + heads * _HEAD_DIM * 2
            + output_bytes
        ),
    }


def _selected_attention_reduce_metadata(grid, kernel, args):
    """Report the split reduce's DRAM bytes: per token and head, it reads
    ``num_splits`` FP32 partial rows and writes one BF16 output row."""
    tokens, heads = grid[0], args["num_heads"]
    return {
        "name": kernel.name,
        "bytes": tokens * heads * _HEAD_DIM * (4 * args["num_splits"] + 2),
    }


@gluon.jit(launch_metadata=_selected_attention_metadata)
def gluon_dsv41_selected_attention_gfx1250(
    q,
    swa_cache,
    swa_slots,
    swa_lens,
    global_cache,
    global_slots,
    global_lens,
    attn_sink,
    out,
    part_stats,
    part_acc,
    part_row_sum,
    part_count,
    part_active,
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    swa_page_stride: tl.int64,
    swa_slot_stride: tl.int64,
    swa_width,
    global_page_stride: tl.int64,
    global_slot_stride: tl.int64,
    global_width,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    softmax_scale: tl.float32,
    swa_capacity,
    global_capacity,
    num_heads,
    HAS_GLOBAL: gl.constexpr,
    HEAD_GROUPS: gl.constexpr,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
    HEAD_DIM: gl.constexpr,
):
    # Program (t, s, b) covers query row t, the s-th contiguous share of its
    # selected tiles (SWA tiles first, then global tiles) and head block b.
    # With one split it adds the sink and writes the output; otherwise it
    # writes partial softmax state for the reduce kernel. Four waves split a
    # tile's rows for QK and the head dims for PV.
    wmma: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [0, 2]],
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    q_layout: gl.constexpr = gl.BlockedLayout([1, 16], [1, 32], [4, 1], [1, 0])
    swa_raw_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 16], [1, 32, 1], [4, 1, 1], [2, 1, 0]
    )
    global_raw_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 2, 8], [2, 16, 1, 1], [4, 1, 1, 1], [3, 2, 1, 0]
    )
    global_kv_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 2, 16], [2, 16, 1, 1], [4, 1, 1, 1], [3, 2, 1, 0]
    )
    swa_row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, swa_raw_layout))
    global_row_layout: gl.constexpr = gl.SliceLayout(
        1, gl.SliceLayout(2, gl.SliceLayout(3, global_raw_layout))
    )
    score_row_layout: gl.constexpr = gl.SliceLayout(0, wmma)
    head_layout: gl.constexpr = gl.SliceLayout(1, wmma)
    kv_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [TILE_K, HEAD_DIM], [1, 0]
    )
    q_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma, k_width=8
    )

    token_idx = gl.program_id(axis=0)
    split_idx = gl.program_id(axis=1)
    num_splits = gl.num_programs(axis=1)
    head_block = gl.program_id(axis=2)
    head_base = head_block * (HEAD_GROUPS * BLOCK_H)
    single = num_splits == 1
    swa_len = gl.minimum(gl.maximum(gl.load(swa_lens + token_idx), 0), swa_width)
    swa_tiles = gl.cdiv(swa_len, TILE_K)
    swa_row = token_idx.to(tl.int64) * swa_slot_stride
    if HAS_GLOBAL:
        global_len = gl.minimum(
            gl.maximum(gl.load(global_lens + token_idx), 0), global_width
        )
        global_row = token_idx.to(tl.int64) * global_slot_stride
    else:
        global_len = 0
        global_row = token_idx.to(tl.int64) * 0
    total_tiles = swa_tiles + gl.cdiv(global_len, TILE_K)
    tile_begin = split_idx * total_tiles // num_splits
    tile_end = (split_idx + 1) * total_tiles // num_splits
    # This split's SWA tiles [swa_first, swa_stop) and global tiles
    # [global_first, global_stop); rows past its share read as invalid.
    swa_first = gl.minimum(tile_begin, swa_tiles)
    swa_stop = gl.minimum(tile_end, swa_tiles)
    swa_bound = gl.minimum(swa_len, swa_stop * TILE_K)
    global_first = gl.maximum(tile_begin, swa_tiles) - swa_tiles
    global_stop = gl.maximum(tile_end, swa_tiles) - swa_tiles
    global_bound = gl.minimum(global_len, global_stop * TILE_K)

    q_heads = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, q_layout))
    q_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, q_layout))
    q_base = q + token_idx.to(tl.int64) * stride_q_t + q_dims[None, :]
    # Bit g of active_groups marks a head group with a nonzero query, which
    # needs its own pass over the selected rows.
    active_groups = 0
    for group in gl.static_range(HEAD_GROUPS):
        heads = head_base + group * BLOCK_H + q_heads
        q_group = gl.load(
            q_base + heads[:, None].to(tl.int64) * stride_q_h,
            mask=(heads < num_heads)[:, None],
            other=0.0,
        )
        nonzero = gl.sum(gl.sum((q_group != 0.0).to(gl.int32), axis=1), axis=0)
        active_groups += gl.where(nonzero > 0, 1 << group, 0)
    need_row_sum = active_groups != (1 << HEAD_GROUPS) - 1
    kv_shared = gl.allocate_shared_memory(
        gl.bfloat16, [TILE_K, HEAD_DIM], layout=kv_shared_layout
    )

    out_base = out + token_idx.to(tl.int64) * stride_o_t + q_dims[None, :]
    # Partial rows are [token, split, head] ([token, split] for row sums).
    part_row = (token_idx * num_splits + split_idx).to(tl.int64)
    row_sum = gl.zeros([BLOCK_H, HEAD_DIM], dtype=gl.float32, layout=wmma)
    row_count = 0

    for group in range(HEAD_GROUPS):
        active = ((active_groups >> group) & 1) != 0
        # Group 0 runs even when inactive: with with_row_sum, its pass also
        # accumulates the row_sum and row_count that the zero-query groups'
        # closed-form output below needs.
        if active or group == 0:
            with_row_sum = need_row_sum and group == 0
            heads = head_base + group * BLOCK_H + q_heads
            dot_heads = (
                head_base
                + group * BLOCK_H
                + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, q_dot_layout))
            )
            dot_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, q_dot_layout))
            q_dot = gl.load(
                q
                + token_idx.to(tl.int64) * stride_q_t
                + dot_heads[:, None].to(tl.int64) * stride_q_h
                + dot_dims[None, :],
                mask=(dot_heads < num_heads)[:, None],
                other=0.0,
            )
            score_heads = (
                head_base + group * BLOCK_H + gl.arange(0, BLOCK_H, layout=head_layout)
            )
            valid_heads = score_heads < num_heads
            # One split starts from the sink (logit sink, weight 1); split
            # partials start empty and the reduce kernel adds the sink once.
            sink = gl.load(attn_sink + score_heads, mask=valid_heads, other=0.0).to(
                gl.float32
            )
            max_value = gl.where(single, sink, -float("inf"))
            denominator = gl.where(single, 1.0, 0.0) + gl.zeros(
                [BLOCK_H], dtype=gl.float32, layout=head_layout
            )
            accumulator = gl.zeros([BLOCK_H, HEAD_DIM], dtype=gl.float32, layout=wmma)

            # Rows are loaded in the tile that uses them: a software pipeline
            # holding the next tile's raw rows costs more occupancy than its
            # overlap saves.
            for tile in range(swa_first, swa_stop):
                slot, valid = _tile_slots(
                    swa_slots,
                    swa_row,
                    tile,
                    swa_bound,
                    swa_width,
                    swa_capacity,
                    TILE_K,
                    swa_row_layout,
                )
                # The softmax needs the mask in the score layout; reloading the
                # cached slots avoids a cross-lane layout conversion.
                _, score_valid = _tile_slots(
                    swa_slots,
                    swa_row,
                    tile,
                    swa_bound,
                    swa_width,
                    swa_capacity,
                    TILE_K,
                    score_row_layout,
                )
                values, scales = _load_swa_rows(
                    swa_cache, slot, valid, swa_page_stride, swa_raw_layout
                )
                kv = _dequant_swa_rows(values, scales, swa_raw_layout)
                max_value, denominator, accumulator, row_sum, row_count = _attend_tile(
                    kv,
                    kv_shared,
                    score_valid,
                    q_dot,
                    valid_heads,
                    softmax_scale,
                    max_value,
                    denominator,
                    accumulator,
                    row_sum,
                    row_count,
                    active,
                    with_row_sum,
                    BLOCK_H,
                    TILE_K,
                    wmma,
                )
            if HAS_GLOBAL:
                for tile in range(global_first, global_stop):
                    slot, valid = _tile_slots(
                        global_slots,
                        global_row,
                        tile,
                        global_bound,
                        global_width,
                        global_capacity,
                        TILE_K,
                        global_row_layout,
                    )
                    # Score-layout mask, as in the SWA loop.
                    _, score_valid = _tile_slots(
                        global_slots,
                        global_row,
                        tile,
                        global_bound,
                        global_width,
                        global_capacity,
                        TILE_K,
                        score_row_layout,
                    )
                    values, scales = _load_global_rows(
                        global_cache,
                        slot,
                        valid,
                        global_page_stride,
                        global_raw_layout,
                    )
                    kv = _dequant_global_rows(values, scales, global_kv_layout)
                    max_value, denominator, accumulator, row_sum, row_count = (
                        _attend_tile(
                            kv,
                            kv_shared,
                            score_valid,
                            q_dot,
                            valid_heads,
                            softmax_scale,
                            max_value,
                            denominator,
                            accumulator,
                            row_sum,
                            row_count,
                            active,
                            with_row_sum,
                            BLOCK_H,
                            TILE_K,
                            wmma,
                        )
                    )

            if active:
                if single:
                    scale = gl.where(denominator > 0.0, 1.0 / denominator, 0.0)
                    gl.store(
                        out_base + heads[:, None].to(tl.int64) * stride_o_h,
                        gl.convert_layout(
                            (accumulator * scale[:, None]).to(gl.bfloat16), q_layout
                        ),
                        mask=(heads < num_heads)[:, None],
                    )
                else:
                    stats = part_stats + (part_row * num_heads + score_heads) * 2
                    gl.store(stats, max_value, mask=valid_heads)
                    gl.store(stats + 1, denominator, mask=valid_heads)
                    gl.store(
                        part_acc
                        + (part_row * num_heads + heads[:, None]) * HEAD_DIM
                        + q_dims[None, :],
                        gl.convert_layout(accumulator, q_layout),
                        mask=(heads < num_heads)[:, None],
                    )

    if need_row_sum:
        if single:
            # A zero query scores exactly zero on every valid row, so its output
            # is row_sum * e^-m / (count * e^-m + e^(sink - m)), m = max(sink, 0).
            count = row_count.to(gl.float32)
            for group in range(HEAD_GROUPS):
                if ((active_groups >> group) & 1) == 0:
                    score_heads = (
                        head_base
                        + group * BLOCK_H
                        + gl.arange(0, BLOCK_H, layout=head_layout)
                    )
                    sink = gl.load(
                        attn_sink + score_heads,
                        mask=score_heads < num_heads,
                        other=0.0,
                    ).to(gl.float32)
                    frame = gl.maximum(sink, 0.0)
                    weight = gl.exp(-frame)
                    total = count * weight + gl.exp(sink - frame)
                    coefficient = gl.where(total > 0.0, weight / total, 0.0)
                    heads = head_base + group * BLOCK_H + q_heads
                    gl.store(
                        out_base + heads[:, None].to(tl.int64) * stride_o_h,
                        gl.convert_layout(
                            (row_sum * coefficient[:, None]).to(gl.bfloat16), q_layout
                        ),
                        mask=(heads < num_heads)[:, None],
                    )
        else:
            # Every row of row_sum holds the same sum; head row 0 is stored.
            gl.store(
                part_row_sum
                + part_row * HEAD_DIM
                + q_dims[None, :]
                + q_heads[:, None] * 0,
                gl.convert_layout(row_sum, q_layout),
                mask=(q_heads == 0)[:, None],
            )
            gl.store(part_count + part_row, row_count)
    if not single:
        if split_idx == 0:
            gl.store(
                part_active + token_idx * gl.num_programs(axis=2) + head_block,
                active_groups,
            )


@gluon.jit
def _split_state(part_stats, part_count, rows, heads, num_heads, mask, active):
    """Per-split (max, sum) [splits, heads]; zero-query heads use frame 0, count."""
    if active:
        stats = (
            part_stats + (rows[:, None].to(tl.int64) * num_heads + heads[None, :]) * 2
        )
        split_max = gl.load(stats, mask=mask, other=-float("inf"))
        split_sum = gl.load(stats + 1, mask=mask, other=0.0)
    else:
        count = gl.load(
            part_count + rows[:, None] + heads[None, :] * 0, mask=mask, other=0
        )
        split_max = gl.where(count > 0, 0.0, -float("inf"))
        split_sum = count.to(gl.float32)
    return split_max, split_sum


@gluon.jit(launch_metadata=_selected_attention_reduce_metadata)
def gluon_dsv41_selected_attention_reduce_gfx1250(
    part_stats,
    part_acc,
    part_row_sum,
    part_count,
    part_active,
    attn_sink,
    out,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    num_heads,
    num_splits,
    head_blocks,
    rope_positions,
    rope_cos_sin,
    stride_rope: tl.int64,
    BLOCK_H: gl.constexpr,
    HEAD_GROUPS: gl.constexpr,
    MAX_SPLITS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    DIM_BLOCK: gl.constexpr,
    ROPE_DIM: gl.constexpr,
):
    """Merge one token's split partials, adding the sink once.

    Each program covers 16 heads and ``DIM_BLOCK`` dims. Zero-query heads use
    frame 0, weight ``count`` and value ``row_sum`` for every split, which is
    exactly their all-zero-score softmax state. Splits are merged four at a
    time so their loads are in flight together.
    """
    CHUNK: gl.constexpr = 4
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4], [1, 32 // (DIM_BLOCK // 4), DIM_BLOCK // 4], [1, 4, 1], [2, 1, 0]
    )
    stat_layout: gl.constexpr = gl.SliceLayout(2, layout)
    out_layout: gl.constexpr = gl.SliceLayout(0, layout)
    token_idx = gl.program_id(axis=0)
    group_idx = gl.program_id(axis=1)
    dim_base = gl.program_id(axis=2) * DIM_BLOCK
    heads = group_idx * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(0, stat_layout)
    )
    valid_head = heads < num_heads
    active_bits = gl.load(
        part_active + token_idx * head_blocks + group_idx // HEAD_GROUPS
    )
    active = ((active_bits >> (group_idx % HEAD_GROUPS)) & 1) != 0
    sink = gl.load(attn_sink + heads, mask=valid_head, other=0.0).to(gl.float32)
    splits = gl.arange(0, MAX_SPLITS, layout=gl.SliceLayout(1, stat_layout))
    split_max, _ = _split_state(
        part_stats,
        part_count,
        token_idx * num_splits + splits,
        heads,
        num_heads,
        (splits < num_splits)[:, None] & valid_head[None, :],
        active,
    )
    frame = gl.maximum(gl.max(split_max, axis=0), sink)
    frame = gl.where(frame > -float("inf"), frame, 0.0)
    total = gl.exp(sink - frame)
    merged = gl.zeros([BLOCK_H, DIM_BLOCK], dtype=gl.float32, layout=out_layout)
    out_heads = group_idx * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, out_layout)
    )
    dims = dim_base + gl.arange(0, DIM_BLOCK, layout=gl.SliceLayout(0, out_layout))
    chunk = gl.arange(0, CHUNK, layout=gl.SliceLayout(1, stat_layout))
    chunk_dims = dim_base + gl.arange(
        0, DIM_BLOCK, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout))
    )
    for start in gl.static_range(0, MAX_SPLITS, CHUNK):
        if start < num_splits:
            split = start + chunk
            valid = (split < num_splits)[:, None] & valid_head[None, :]
            rows = token_idx * num_splits + split
            chunk_max, chunk_sum = _split_state(
                part_stats, part_count, rows, heads, num_heads, valid, active
            )
            weight = gl.exp(chunk_max - frame[None, :])
            total += gl.sum(chunk_sum * weight, axis=0)
            if active:
                values = gl.load(
                    part_acc
                    + (
                        rows[:, None, None].to(tl.int64) * num_heads
                        + heads[None, :, None]
                    )
                    * HEAD_DIM
                    + chunk_dims[None, None, :],
                    mask=valid[:, :, None],
                    other=0.0,
                )
            else:
                values = gl.load(
                    part_row_sum
                    + rows[:, None, None].to(tl.int64) * HEAD_DIM
                    + chunk_dims[None, None, :]
                    + heads[None, :, None] * 0,
                    mask=valid[:, :, None],
                    other=0.0,
                )
            merged += gl.sum(values * weight[:, :, None], axis=0)
    scale = gl.convert_layout(
        gl.where(total > 0.0, 1.0 / total, 0.0), gl.SliceLayout(1, out_layout)
    )
    result = (merged * scale[:, None]).to(gl.bfloat16)
    if ROPE_DIM > 0:
        if dim_base + DIM_BLOCK > HEAD_DIM - ROPE_DIM:
            rope_row = (
                rope_cos_sin
                + gl.maximum(gl.load(rope_positions + token_idx), 0).to(tl.int64)
                * stride_rope
            )
            result = _inverse_rope(
                result, dims, out_heads, rope_row, dim_base, ROPE_DIM
            )
    gl.store(
        out
        + token_idx.to(tl.int64) * stride_o_t
        + out_heads[:, None].to(tl.int64) * stride_o_h
        + dims[None, :],
        result,
        mask=(out_heads < num_heads)[:, None],
    )


def _output(out, shape, dtype, device):
    if out is None:
        return torch.empty(shape, dtype=dtype, device=device)
    if (
        out.shape != shape
        or out.dtype != dtype
        or out.device != device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must have the specified shape, dtype, device and be contiguous"
        )
    return out


def _cache(cache, row_bytes, name):
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[1:] != (64, row_bytes)
    ):
        raise ValueError(f"{name} cache must be uint8 [pages, 64, {row_bytes}]")
    # Native reads need contiguous page bytes and 16-byte aligned pages.
    if (
        cache.stride(2) != 1
        or cache.stride(1) != row_bytes
        or cache.stride(0) % 16 != 0
        or cache.data_ptr() % 16 != 0
    ):
        raise ValueError(f"{name} cache pages must be contiguous and 16-byte aligned")


def _integers(x, shape, name):
    if x.shape != shape or x.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be int32/int64 with shape {shape}")


@gluon.jit
def gluon_dsv41_output_rope_gfx1250(
    out,
    rope_positions,
    rope_cos_sin,
    stride_o_t,
    stride_o_h,
    stride_rope: tl.int64,
    num_heads,
    BLOCK_H: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    ROPE_DIM: gl.constexpr,
):
    """Rotate the RoPE tail of BLOCK_H heads of one token's output in place.

    Even and odd dims load separately, so the rotation needs no lane swap.
    """
    gl.static_assert(ROPE_DIM >= 2 and ROPE_DIM <= 64, "RoPE tail of 2 to 64")
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [64 // ROPE_DIM, ROPE_DIM // 2], [4, 1], [1, 0]
    )
    token_idx = gl.program_id(axis=0)
    heads = gl.program_id(axis=1) * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, layout)
    )
    pairs = gl.arange(0, ROPE_DIM // 2, layout=gl.SliceLayout(0, layout))
    token_out = out + token_idx.to(tl.int64) * stride_o_t + (HEAD_DIM - ROPE_DIM)
    even_offsets = heads[:, None] * stride_o_h + 2 * pairs[None, :]
    mask = (heads < num_heads)[:, None]
    rope_row = (
        rope_cos_sin
        + gl.maximum(gl.load(rope_positions + token_idx), 0).to(tl.int64) * stride_rope
    )
    cosine = gl.load(rope_row + pairs)[None, :]
    sine = gl.load(rope_row + ROPE_DIM // 2 + pairs)[None, :]
    even = gl.load(token_out + even_offsets, mask=mask, other=0.0).to(gl.float32)
    odd = gl.load(token_out + even_offsets + 1, mask=mask, other=0.0).to(gl.float32)
    gl.store(
        token_out + even_offsets,
        (even * cosine - odd * sine).to(gl.bfloat16),
        mask=mask,
    )
    gl.store(
        token_out + even_offsets + 1,
        (odd * cosine + even * sine).to(gl.bfloat16),
        mask=mask,
    )


def launch_gluon_dsv41_selected_attention_gfx1250(
    q,
    swa_cache,
    swa_slots,
    swa_lens,
    global_cache,
    global_slots,
    global_lens,
    attn_sink,
    softmax_scale,
    out,
    query_chunk_size,
    schedule,
    prefill_kv,
    prefill_indices,
    rope_positions=None,
    rope_cos_sin=None,
):
    """Fused SWA+global selected attention over V4.1 page-planar caches.

    With ``rope_positions`` [tokens] and an FP32 ``rope_cos_sin`` table
    [positions, rope_dim] (inverse signs), the output tail is also rotated, as
    ``rope_inplace`` would do afterwards: in the split reduction, or in a
    per-token pass when the rows do not split. Paged decode inputs only.
    """
    attn_sink = attn_sink[: q.shape[1]]
    if prefill_kv is not None:
        if prefill_indices is None:
            raise ValueError("prefill_kv requires prefill_indices")
        if rope_positions is not None or rope_cos_sin is not None:
            raise ValueError("output RoPE needs paged decode inputs")
        from tokenspeed_kernel.ops.attention.dsv4 import dsv4_prefill

        out = _output(out, q.shape, q.dtype, q.device)
        lengths = torch.full(
            (q.shape[0],),
            prefill_indices.shape[-1],
            dtype=torch.int32,
            device=q.device,
        )
        dsv4_prefill(
            q=q.contiguous(),
            kv=prefill_kv,
            indices=prefill_indices,
            lens=lengths,
            attn_sink=attn_sink.contiguous(),
            softmax_scale=softmax_scale,
            out=out,
            override=None,
            solution=None,
        )
        return out
    if prefill_indices is not None:
        raise ValueError("prefill_indices requires prefill_kv")
    rope = rope_cos_sin is not None
    if rope:
        if (
            rope_cos_sin.ndim != 2
            or rope_cos_sin.dtype != torch.float32
            or rope_cos_sin.stride(1) != 1
            or rope_cos_sin.shape[1] not in (2, 4, 8, 16, 32, 64)
        ):
            raise ValueError("rope_cos_sin must be FP32 [positions, rope_dim]")
        _integers(rope_positions, (q.shape[0],), "rope_positions")
    elif rope_positions is not None:
        raise ValueError("rope_positions requires rope_cos_sin")
    if q.ndim != 3 or q.shape[-1] != 512 or q.dtype != torch.bfloat16 or q.shape[1] < 1:
        raise ValueError("q must be BF16 [tokens, heads, 512]")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    extras = (
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        attn_sink,
        out,
    )
    if not q.is_cuda or any(t.device != q.device for t in extras if t is not None):
        raise ValueError("all tensors must share a CUDA/ROCm device")
    if attn_sink.shape != (q.shape[1],) or attn_sink.dtype != torch.float32:
        raise ValueError("attn_sink must be FP32 [heads]")
    _cache(swa_cache, _SWA_ROW_BYTES, "swa")
    if swa_slots.ndim != 2 or swa_slots.shape[0] != q.shape[0]:
        raise ValueError("slots must have one row per query")
    _integers(swa_slots, swa_slots.shape, "slots")
    _integers(swa_lens, (q.shape[0],), "lens")
    has_global = global_cache is not None
    if not has_global:
        if global_slots is not None or global_lens is not None:
            raise ValueError(
                "absent global cache requires None global slots and lengths"
            )
    else:
        if global_slots is None or global_lens is None:
            raise ValueError("global cache requires global slots and lengths")
        _cache(global_cache, _GLOBAL_ROW_BYTES, "global")
        if global_slots.ndim != 2 or global_slots.shape[0] != q.shape[0]:
            raise ValueError("slots must have one row per query")
        _integers(global_slots, global_slots.shape, "slots")
        _integers(global_lens, (q.shape[0],), "lens")
    out = _output(out, q.shape, q.dtype, q.device)
    width = swa_slots.shape[1] + (global_slots.shape[1] if has_global else 0)
    if q.shape[0] == 0 or width == 0:
        return out.zero_()

    scale = float(softmax_scale)
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")

    q = q.contiguous()
    sink = attn_sink.contiguous()
    swa_slots_i = swa_slots.to(torch.int32).contiguous()
    swa_lens_i = swa_lens.to(torch.int32).contiguous()
    if has_global:
        global_slots_i = global_slots.to(torch.int32).contiguous()
        global_lens_i = global_lens.to(torch.int32).contiguous()
        global_cache_u8 = global_cache
        global_width = global_slots_i.shape[1]
        global_capacity = global_cache.shape[0] * _PAGE_ROWS
        global_page_stride = global_cache.stride(0)
        global_slot_stride = global_slots_i.stride(0)
    else:
        global_slots_i = swa_slots_i
        global_lens_i = swa_lens_i
        global_cache_u8 = swa_cache
        global_width = 0
        global_capacity = 0
        global_page_stride = 0
        global_slot_stride = 0

    tokens = q.shape[0]
    max_tiles = triton.cdiv(swa_slots_i.shape[1], _TILE_K) + triton.cdiv(
        global_width, _TILE_K
    )
    # The token count only picks one of three head-group widths.
    head_groups = 1
    while (
        head_groups < _HEAD_GROUPS_PER_CTA
        and tokens * triton.cdiv(q.shape[1], head_groups * _BLOCK_H) * max_tiles
        > _HEAD_BLOCK_TILES
    ):
        head_groups *= 2
    head_groups = min(head_groups, triton.cdiv(q.shape[1], _BLOCK_H))
    head_blocks = triton.cdiv(q.shape[1], head_groups * _BLOCK_H)
    # Split each row's selected tiles over workgroups so small batches still
    # fill the GPU, at least two tiles per split: a one-tile split costs more
    # in the reduce than it saves. The split count depends only on shapes,
    # never on lengths.
    splits = max(
        1,
        min(
            _MAX_SPLITS,
            max_tiles // 2,
            _TARGET_CTAS // (tokens * head_blocks),
        ),
    )
    if splits > 1:
        heads = q.shape[1]
        part_stats = torch.empty(
            (tokens, splits, heads, 2), dtype=torch.float32, device=q.device
        )
        part_acc = torch.empty(
            (tokens, splits, heads, _HEAD_DIM), dtype=torch.float32, device=q.device
        )
        part_row_sum = torch.empty(
            (tokens, splits, _HEAD_DIM), dtype=torch.float32, device=q.device
        )
        part_count = torch.empty((tokens, splits), dtype=torch.int32, device=q.device)
        part_active = torch.empty(
            (tokens, head_blocks), dtype=torch.int32, device=q.device
        )
    else:
        # Placeholders: a single split never touches the partials.
        part_stats = part_acc = part_row_sum = sink
        part_count = part_active = swa_lens_i
    gluon_dsv41_selected_attention_gfx1250[(tokens, splits, head_blocks)](
        q,
        swa_cache,
        swa_slots_i,
        swa_lens_i,
        global_cache_u8,
        global_slots_i,
        global_lens_i,
        sink,
        out,
        part_stats,
        part_acc,
        part_row_sum,
        part_count,
        part_active,
        q.stride(0),
        q.stride(1),
        swa_cache.stride(0),
        swa_slots_i.stride(0),
        swa_slots_i.shape[1],
        global_page_stride,
        global_slot_stride,
        global_width,
        out.stride(0),
        out.stride(1),
        scale,
        swa_cache.shape[0] * _PAGE_ROWS,
        global_capacity,
        q.shape[1],
        HAS_GLOBAL=has_global,
        HEAD_GROUPS=head_groups,
        BLOCK_H=_BLOCK_H,
        TILE_K=_TILE_K,
        HEAD_DIM=_HEAD_DIM,
        num_warps=4,
        num_stages=1,
    )
    if splits > 1:
        gluon_dsv41_selected_attention_reduce_gfx1250[
            (tokens, triton.cdiv(q.shape[1], _BLOCK_H), _HEAD_DIM // _REDUCE_DIM_BLOCK)
        ](
            part_stats,
            part_acc,
            part_row_sum,
            part_count,
            part_active,
            sink,
            out,
            out.stride(0),
            out.stride(1),
            q.shape[1],
            splits,
            head_blocks,
            rope_positions if rope else swa_lens_i,
            rope_cos_sin if rope else sink,
            rope_cos_sin.stride(0) if rope else 0,
            BLOCK_H=_BLOCK_H,
            HEAD_GROUPS=head_groups,
            MAX_SPLITS=_MAX_SPLITS,
            HEAD_DIM=_HEAD_DIM,
            DIM_BLOCK=_REDUCE_DIM_BLOCK,
            ROPE_DIM=rope_cos_sin.shape[1] if rope else 0,
            num_warps=4,
        )
    elif rope:
        # Rotating in the single-split epilogue costs this register-heavy
        # kernel more than a separate pass.
        gluon_dsv41_output_rope_gfx1250[(tokens, triton.cdiv(q.shape[1], _BLOCK_H))](
            out,
            rope_positions,
            rope_cos_sin,
            out.stride(0),
            out.stride(1),
            rope_cos_sin.stride(0),
            q.shape[1],
            BLOCK_H=_BLOCK_H,
            HEAD_DIM=_HEAD_DIM,
            ROPE_DIM=rope_cos_sin.shape[1],
            num_warps=4,
        )
    return out
