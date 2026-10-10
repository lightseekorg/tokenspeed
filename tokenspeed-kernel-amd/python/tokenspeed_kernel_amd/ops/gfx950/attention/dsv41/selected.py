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

"""Fused two-reader DeepSeek V4.1 selected attention for GFX950.

Reads page-planar SWA (E4M3/E8M0, 528B) and global (E2M1/E4M3, 288B) caches
directly. One online softmax includes both segments and a single sink. This
replaces the portable gather-then-``dsv4_prefill`` path.

A workgroup owns one query row, up to 64 heads and a contiguous share of the
row's selected tiles, so each selected row is loaded and dequantized once per
16-head group that needs it. Head groups whose queries are all zero (the
64/128-head padding of the decode layout) skip the score MFMAs: their scores
are exactly zero, so their output is the V row sum scaled per head by the
sink, computed once for all such groups. Small batches split the selected
tiles over workgroups and merge the partial softmax states in a reduce kernel.
See the "DeepSeek V4.1 selected attention" section of ``ops/README.md``.
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

__all__ = ["launch_gluon_dsv41_selected_attention_gfx950"]

_SWA_ROW_BYTES = 528
_GLOBAL_ROW_BYTES = 288
_PAGE_ROWS = 64
_HEAD_DIM = 512
_BLOCK_H = 16
_TILE_K = 64
# The main kernel keeps one workgroup per CU (VGPR-bound occupancy), so rows
# are split over workgroups only while the grid still fits the 256 CUs once.
_TARGET_CTAS = 256
_MAX_SPLITS = 8
# One workgroup covers at most 64 query heads (four 16-head groups).
_HEAD_GROUPS_PER_CTA = 4


@gluon.jit
def _tile_slots(slots, slot_row, tile_idx, length, width, capacity, TILE_K, LAYOUT):
    """Return (clamped slot, valid) for one tile of a query row's selections."""
    positions = tile_idx * TILE_K + gl.arange(0, TILE_K, layout=LAYOUT)
    in_range = (positions < length) & (positions < width)
    slot = gl.amd.cdna4.buffer_load(
        slots + slot_row, positions, mask=in_range, other=-1
    )
    valid = in_range & (slot >= 0) & (slot < capacity)
    return gl.where(valid, slot, 0), valid


@gluon.jit
def _load_swa_rows(cache, slot, page_stride, RAW_LAYOUT: gl.constexpr):
    """Load SWA rows: E4M3 words [TILE_K, 32, 4] and E8M0 scales [TILE_K, 32].

    A lane owns 16 contiguous value bytes of a row (one dwordx4). Chunk ``c``
    covers dims ``16c..16c+15``, whose E8M0 scale is byte ``c // 2``. Invalid
    rows were clamped to slot 0, so loads stay unmasked and in bounds.
    """
    scale_layout: gl.constexpr = gl.SliceLayout(2, RAW_LAYOUT)
    page_base = (slot // 64).to(tl.int64) * page_stride
    row = slot % 64
    chunks = gl.arange(0, 32, layout=gl.SliceLayout(0, scale_layout))
    words = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, RAW_LAYOUT)))
    word_offsets = (
        ((page_base + row * 512) // 4)[:, None, None]
        + (chunks * 4)[None, :, None]
        + words[None, None, :]
    )
    scale_offsets = (page_base + 64 * 512 + row * 16)[:, None] + (chunks // 2)[None, :]
    values = gl.load(cache.to(gl.pointer_type(gl.int32)) + word_offsets)
    return values, gl.load(cache + scale_offsets)


@gluon.jit
def _dequant_swa_rows(values, scales, valid):
    """E4M3 times E8M0 scale into BF16 pairs, as int32 words [TILE_K, 256].

    The hardware scaled conversion multiplies by 2**(code - 127) (code 0 is
    2**-127) and rounds once to BF16. Invalid rows convert zero words.
    """
    values = values & gl.where(valid, -1, 0)[:, None, None]
    scale = (scales.to(gl.int32) << 23).to(gl.float32, bitcast=True)
    values, scale = gl.broadcast(values, scale[:, :, None])
    low, high = gl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_bf16_fp8 $0, $2, $3\n"
        "v_cvt_scalef32_pk_bf16_fp8 $1, $2, $3 op_sel:[1,0,0]",
        "=&v,=&v,v,v",
        [values, scale],
        dtype=(gl.int32, gl.int32),
        is_pure=True,
        pack=1,
    )
    pairs = gl.join(low, high)
    return pairs.reshape([pairs.shape[0], 256])


@gluon.jit
def _load_global_rows(cache, slot, page_stride, RAW_LAYOUT: gl.constexpr):
    """Load global rows: E2M1 words [TILE_K, 16, 2, 2], E4M3 scales [TILE_K, 16, 2].

    A lane owns 16 contiguous packed bytes (32 dims) of a row: chunk ``c``,
    scale group ``g`` of 16 dims (scale byte ``2c + g``), and word ``w``.
    """
    scale_layout: gl.constexpr = gl.SliceLayout(3, RAW_LAYOUT)
    chunk_layout: gl.constexpr = gl.SliceLayout(0, scale_layout)
    page_base = (slot // 64).to(tl.int64) * page_stride
    row = slot % 64
    chunks = gl.arange(0, 16, layout=gl.SliceLayout(1, chunk_layout))
    groups = gl.arange(0, 2, layout=gl.SliceLayout(0, chunk_layout))
    scale_idx = chunks[:, None] * 2 + groups[None, :]
    words = gl.arange(
        0,
        2,
        layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, RAW_LAYOUT))),
    )
    word_offsets = (
        ((page_base + row * 256) // 4)[:, None, None, None]
        + (scale_idx * 2)[None, :, :, None]
        + words[None, None, None, :]
    )
    scale_offsets = (page_base + 64 * 256 + row * 32)[:, None, None] + scale_idx[
        None, :, :
    ]
    values = gl.load(cache.to(gl.pointer_type(gl.int32)) + word_offsets)
    return values, gl.load(cache + scale_offsets)


@gluon.jit
def _e2m1_pair_scaled(pair, scale):
    """Scale one byte's two E2M1 values (FP32 pair in an int64) into a BF16 pair."""
    low = (pair & 0xFFFFFFFF).to(gl.int32).to(gl.float32, bitcast=True) * scale
    high = (pair >> 32).to(gl.int32).to(gl.float32, bitcast=True) * scale
    return gl.inline_asm_elementwise(
        "v_cvt_pk_bf16_f32 $0, $1, $2",
        "=v,v,v",
        [low, high],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _dequant_global_rows(values, scales, valid):
    """E2M1 (even dim in the low nibble) times E4M3 into BF16 pairs [TILE_K, 256].

    Both factors convert exactly to FP32, so their product is exact before the
    single BF16 rounding. Invalid rows convert zero words with a zero scale.
    """
    keep = gl.where(valid, -1, 0)
    values = values & keep[:, None, None, None]
    scale = scales.to(tl.float8e4nv, bitcast=True).to(gl.float32)
    scale = (scale.to(gl.int32, bitcast=True) & keep[:, None, None]).to(
        gl.float32, bitcast=True
    )
    values, scale = gl.broadcast(values, scale[:, :, :, None])
    # One conversion per byte; op_sel picks the byte of the 32-bit word.
    byte0, byte1, byte2, byte3 = gl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_f32_fp4 $0, $4, 1.0\n"
        "v_cvt_scalef32_pk_f32_fp4 $1, $4, 1.0 op_sel:[1,0,0]\n"
        "v_cvt_scalef32_pk_f32_fp4 $2, $4, 1.0 op_sel:[0,1,0]\n"
        "v_cvt_scalef32_pk_f32_fp4 $3, $4, 1.0 op_sel:[1,1,0]",
        "=&v,=&v,=&v,=&v,v",
        [values],
        dtype=(gl.int64, gl.int64, gl.int64, gl.int64),
        is_pure=True,
        pack=1,
    )
    pairs = gl.join(
        gl.join(_e2m1_pair_scaled(byte0, scale), _e2m1_pair_scaled(byte2, scale)),
        gl.join(_e2m1_pair_scaled(byte1, scale), _e2m1_pair_scaled(byte3, scale)),
    )
    return pairs.reshape([pairs.shape[0], 256])


@gluon.jit
def _attend_tile(
    kv,
    kv_words,
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
    MFMA_LAYOUT: gl.constexpr,
):
    """Fold one dequantized tile (BF16 pairs, int32 [TILE_K, 256]) into the softmax."""
    k_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=MFMA_LAYOUT, k_width=8
    )
    p_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=MFMA_LAYOUT, k_width=8
    )
    kv_words.store(kv)
    v_dot = kv_shared.load(k_dot_layout)
    if active:
        k_dot = kv_shared.permute([1, 0]).load(k_dot_layout)
        scores = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=MFMA_LAYOUT)
        scores = gl.amd.cdna4.mfma(q_dot, k_dot, scores) * softmax_scale
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
        accumulator = gl.amd.cdna4.mfma(p_dot, v_dot, accumulator)
        max_value = next_max
    if with_row_sum:
        # Invalid rows dequantize to zero, so all-one weights give the V row sum.
        ones = gl.full([BLOCK_H, TILE_K], 1.0, dtype=gl.bfloat16, layout=p_dot_layout)
        row_sum = gl.amd.cdna4.mfma(ones, v_dot, row_sum)
        row_count += gl.sum(valid_rows.to(gl.int32), axis=0)
    return max_value, denominator, accumulator, row_sum, row_count


def _selected_attention_metadata(grid, kernel, args):
    # Capacity bound without reading device lengths: every selected slot is
    # valid and every head is computed. Split partials are FP32 [heads, 512].
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
    tokens, heads = grid[0], args["num_heads"]
    return {
        "name": kernel.name,
        "bytes": tokens * heads * _HEAD_DIM * (4 * args["num_splits"] + 2),
    }


@gluon.jit(launch_metadata=_selected_attention_metadata)
def gluon_dsv41_selected_attention_gfx950(
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
    mfma: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    q_layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [4, 1], [1, 0])
    q_block_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 8], [1, 1, 64], [1, 4, 1], [2, 1, 0]
    )
    swa_raw_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4], [2, 32, 1], [4, 1, 1], [2, 1, 0]
    )
    global_raw_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 2, 2], [4, 16, 1, 1], [4, 1, 1, 1], [3, 2, 1, 0]
    )
    swa_row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, swa_raw_layout))
    global_row_layout: gl.constexpr = gl.SliceLayout(
        1, gl.SliceLayout(2, gl.SliceLayout(3, global_raw_layout))
    )
    score_row_layout: gl.constexpr = gl.SliceLayout(0, mfma)
    head_layout: gl.constexpr = gl.SliceLayout(1, mfma)
    kv_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [TILE_K, HEAD_DIM], [1, 0]
    )
    kv_word_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [TILE_K, HEAD_DIM // 2], [1, 0]
    )
    q_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma, k_width=8
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

    # Bit g of active_groups marks a head group with a nonzero query, which
    # needs its own pass over the selected rows.
    block_groups = gl.arange(
        0, HEAD_GROUPS, layout=gl.SliceLayout(1, gl.SliceLayout(2, q_block_layout))
    )
    block_heads = (
        head_base
        + block_groups[:, None] * BLOCK_H
        + gl.arange(
            0, BLOCK_H, layout=gl.SliceLayout(0, gl.SliceLayout(2, q_block_layout))
        )[None, :]
    )
    block_dims = gl.arange(
        0, HEAD_DIM, layout=gl.SliceLayout(0, gl.SliceLayout(1, q_block_layout))
    )
    q_block = gl.load(
        q
        + token_idx.to(tl.int64) * stride_q_t
        + block_heads[:, :, None].to(tl.int64) * stride_q_h
        + block_dims[None, None, :],
        mask=(block_heads < num_heads)[:, :, None],
        other=0.0,
    )
    nonzero = gl.sum(gl.sum((q_block != 0.0).to(gl.int32), axis=2), axis=1)
    active_groups = gl.sum(gl.where(nonzero > 0, 1 << block_groups, 0), axis=0)
    need_row_sum = active_groups != (1 << HEAD_GROUPS) - 1
    kv_shared = gl.allocate_shared_memory(
        gl.bfloat16, [TILE_K, HEAD_DIM], layout=kv_shared_layout
    )
    # Dequantization emits packed BF16 pairs; the same bytes as int32 words.
    kv_words = kv_shared.reinterpret(gl.int32, [TILE_K, HEAD_DIM // 2], kv_word_layout)

    q_heads = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, q_layout))
    q_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, q_layout))
    out_base = out + token_idx.to(tl.int64) * stride_o_t + q_dims[None, :]
    # Partial rows are [token, split, head] ([token, split] for row sums).
    part_row = (token_idx * num_splits + split_idx).to(tl.int64)
    row_sum = gl.zeros([BLOCK_H, HEAD_DIM], dtype=gl.float32, layout=mfma)
    row_count = 0

    for group in range(HEAD_GROUPS):
        active = ((active_groups >> group) & 1) != 0
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
            accumulator = gl.zeros([BLOCK_H, HEAD_DIM], dtype=gl.float32, layout=mfma)

            # Software pipeline: slots are fetched two tiles ahead and rows one
            # tile ahead of the tile being dequantized and attended. Both
            # segments' first tiles are issued before the SWA loop.
            if HAS_GLOBAL:
                g_slot, g_valid = _tile_slots(
                    global_slots,
                    global_row,
                    global_first,
                    global_bound,
                    global_width,
                    global_capacity,
                    TILE_K,
                    global_row_layout,
                )
                _, g_score = _tile_slots(
                    global_slots,
                    global_row,
                    global_first,
                    global_bound,
                    global_width,
                    global_capacity,
                    TILE_K,
                    score_row_layout,
                )
                g_values, g_scales = _load_global_rows(
                    global_cache, g_slot, global_page_stride, global_raw_layout
                )
            s_slot, s_valid = _tile_slots(
                swa_slots,
                swa_row,
                swa_first,
                swa_bound,
                swa_width,
                swa_capacity,
                TILE_K,
                swa_row_layout,
            )
            _, s_score = _tile_slots(
                swa_slots,
                swa_row,
                swa_first,
                swa_bound,
                swa_width,
                swa_capacity,
                TILE_K,
                score_row_layout,
            )
            s_values, s_scales = _load_swa_rows(
                swa_cache, s_slot, swa_page_stride, swa_raw_layout
            )
            n_slot, n_valid = _tile_slots(
                swa_slots,
                swa_row,
                swa_first + 1,
                swa_bound,
                swa_width,
                swa_capacity,
                TILE_K,
                swa_row_layout,
            )
            _, n_score = _tile_slots(
                swa_slots,
                swa_row,
                swa_first + 1,
                swa_bound,
                swa_width,
                swa_capacity,
                TILE_K,
                score_row_layout,
            )
            for tile in range(swa_first, swa_stop):
                nn_slot, nn_valid = _tile_slots(
                    swa_slots,
                    swa_row,
                    tile + 2,
                    swa_bound,
                    swa_width,
                    swa_capacity,
                    TILE_K,
                    swa_row_layout,
                )
                _, nn_score = _tile_slots(
                    swa_slots,
                    swa_row,
                    tile + 2,
                    swa_bound,
                    swa_width,
                    swa_capacity,
                    TILE_K,
                    score_row_layout,
                )
                n_values, n_scales = _load_swa_rows(
                    swa_cache, n_slot, swa_page_stride, swa_raw_layout
                )
                kv = _dequant_swa_rows(s_values, s_scales, s_valid)
                max_value, denominator, accumulator, row_sum, row_count = _attend_tile(
                    kv,
                    kv_words,
                    kv_shared,
                    s_score,
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
                    mfma,
                )
                s_values, s_scales, s_valid, s_score = (
                    n_values,
                    n_scales,
                    n_valid,
                    n_score,
                )
                n_slot, n_valid, n_score = nn_slot, nn_valid, nn_score
            if HAS_GLOBAL:
                n_slot, n_valid = _tile_slots(
                    global_slots,
                    global_row,
                    global_first + 1,
                    global_bound,
                    global_width,
                    global_capacity,
                    TILE_K,
                    global_row_layout,
                )
                _, n_score = _tile_slots(
                    global_slots,
                    global_row,
                    global_first + 1,
                    global_bound,
                    global_width,
                    global_capacity,
                    TILE_K,
                    score_row_layout,
                )
                for tile in range(global_first, global_stop):
                    nn_slot, nn_valid = _tile_slots(
                        global_slots,
                        global_row,
                        tile + 2,
                        global_bound,
                        global_width,
                        global_capacity,
                        TILE_K,
                        global_row_layout,
                    )
                    _, nn_score = _tile_slots(
                        global_slots,
                        global_row,
                        tile + 2,
                        global_bound,
                        global_width,
                        global_capacity,
                        TILE_K,
                        score_row_layout,
                    )
                    n_values, n_scales = _load_global_rows(
                        global_cache, n_slot, global_page_stride, global_raw_layout
                    )
                    kv = _dequant_global_rows(g_values, g_scales, g_valid)
                    max_value, denominator, accumulator, row_sum, row_count = (
                        _attend_tile(
                            kv,
                            kv_words,
                            kv_shared,
                            g_score,
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
                            mfma,
                        )
                    )
                    g_values, g_scales, g_valid, g_score = (
                        n_values,
                        n_scales,
                        n_valid,
                        n_score,
                    )
                    n_slot, n_valid, n_score = nn_slot, nn_valid, nn_score

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
def gluon_dsv41_selected_attention_reduce_gfx950(
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
    BLOCK_H: gl.constexpr,
    HEAD_GROUPS: gl.constexpr,
    MAX_SPLITS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
):
    """Merge one token's split partials for 16 heads and add the sink once.

    Zero-query heads use frame 0, weight ``count`` and value ``row_sum`` for
    every split, which is exactly their all-zero-score softmax state. Splits
    are merged four at a time so their loads are in flight together.
    """
    CHUNK: gl.constexpr = 4
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 1, 64], [1, 4, 1], [2, 1, 0])
    stat_layout: gl.constexpr = gl.SliceLayout(2, layout)
    out_layout: gl.constexpr = gl.SliceLayout(0, layout)
    token_idx = gl.program_id(axis=0)
    group_idx = gl.program_id(axis=1)
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
    merged = gl.zeros([BLOCK_H, HEAD_DIM], dtype=gl.float32, layout=out_layout)
    out_heads = group_idx * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, out_layout)
    )
    dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, out_layout))
    chunk = gl.arange(0, CHUNK, layout=gl.SliceLayout(1, stat_layout))
    chunk_dims = gl.arange(
        0, HEAD_DIM, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout))
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
    gl.store(
        out
        + token_idx.to(tl.int64) * stride_o_t
        + out_heads[:, None].to(tl.int64) * stride_o_h
        + dims[None, :],
        (merged * scale[:, None]).to(gl.bfloat16),
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


def launch_gluon_dsv41_selected_attention_gfx950(
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
):
    """Fused SWA+global selected attention over V4.1 page-planar caches."""
    attn_sink = attn_sink[: q.shape[1]]
    if prefill_kv is not None:
        if prefill_indices is None:
            raise ValueError("prefill_kv requires prefill_indices")
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

    head_groups = min(_HEAD_GROUPS_PER_CTA, triton.cdiv(q.shape[1], _BLOCK_H))
    head_blocks = triton.cdiv(q.shape[1], head_groups * _BLOCK_H)
    tokens = q.shape[0]
    # Split each row's selected tiles over workgroups so small batches still
    # fill the GPU. The split count depends only on shapes, never on lengths.
    max_tiles = triton.cdiv(swa_slots_i.shape[1], _TILE_K) + triton.cdiv(
        global_width, _TILE_K
    )
    splits = max(
        1,
        min(
            _MAX_SPLITS,
            max_tiles,
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
        part_stats = part_acc = part_row_sum = sink
        part_count = part_active = swa_lens_i
    gluon_dsv41_selected_attention_gfx950[(tokens, splits, head_blocks)](
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
        gluon_dsv41_selected_attention_reduce_gfx950[
            (tokens, triton.cdiv(q.shape[1], _BLOCK_H))
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
            BLOCK_H=_BLOCK_H,
            HEAD_GROUPS=head_groups,
            MAX_SPLITS=_MAX_SPLITS,
            HEAD_DIM=_HEAD_DIM,
            num_warps=4,
        )
    return out
