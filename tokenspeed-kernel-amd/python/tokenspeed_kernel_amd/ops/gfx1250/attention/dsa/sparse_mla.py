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

"""DSA indexer logits and top-k Gluon kernels for AMD GFX1250."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.indexing import (
    _check_packed_fp8_inputs,
    gluon_dsa_decode_topk_fp8_gfx1250,
    gluon_dsa_prefill_topk_fp8_gfx1250,
)
from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.standard_cache_logits import (
    gluon_dsa_decode_topk_standard_gfx1250,
    gluon_dsa_prefill_topk_standard_gfx1250,
    gluon_kpool_prefill_topk_fp8_gfx1250,
    gluon_kpool_prefill_topk_fp8_plan_gfx1250,
)

__all__ = [
    "launch_gluon_dsa_decode_topk_fp8_gfx1250",
    "launch_gluon_dsa_decode_topk_standard_gfx1250",
    "gluon_dsa_kpool_prefill_logits_gfx1250",
    "gluon_dsa_kpool_prefill_plan_logits_gfx1250",
    "gluon_dsa_logical_topk_gfx1250",
    "launch_gluon_dsa_prefill_topk_fp8_gfx1250",
    "launch_gluon_dsa_prefill_topk_standard_gfx1250",
]

_RADIX_BITS = (12, 12, 8)
_RADIX0_BITS = gl.constexpr(_RADIX_BITS[0])
_RADIX1_BITS = gl.constexpr(_RADIX_BITS[1])
_RADIX2_BITS = gl.constexpr(_RADIX_BITS[2])
_MAX_BUCKETS = gl.constexpr(1 << max(_RADIX_BITS))
_TOPK_BLOCK_N = 4096
_TOPK_NUM_WARPS = 16
_TOPK_WAVES_PER_EU = 1
_TOPK_HIST_REPLICAS = 4
_TOPK_SHORT_HIST_REPLICAS = 2
_TOPK_SHORT_MAX_COLS = 8192
_TOPK_COMPACT_MIN_COLS = 32768
_DECODE_TOPK_BLOCK_N = 8192
_DECODE_TOPK_NUM_WARPS = 32
_DECODE_TOPK_WAVES_PER_EU = 1
_DECODE_TOPK_HIST_REPLICAS = 8
_STANDARD_DECODE_BLOCK_N = 32
_STANDARD_DECODE_CHUNK_N = 64
_STANDARD_DECODE_NUM_WARPS = 1
_STANDARD_DECODE_WAVES_PER_EU = 4
_STANDARD_PREFILL_BLOCK_N = 128
_STANDARD_PREFILL_NUM_WARPS = 8
_STANDARD_PREFILL_WAVES_PER_EU = 1
_STANDARD_PREFILL_MANY_ROWS = 1024
_STANDARD_PREFILL_MANY_ROWS_BLOCK_N = 32
_STANDARD_PREFILL_MANY_ROWS_NUM_WARPS = 1
_KPOOL_SCORE_BLOCK_N = 128
_KPOOL_SCORE_NUM_WARPS = 4
_KPOOL_SCORE_WAVES_PER_EU = 4


@gluon.constexpr_function
def _vector_layout(
    NUM_WARPS: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    return gl.BlockedLayout([LOAD_ELEMS], [32], [NUM_WARPS], [0])


@gluon.jit
def _fp32_to_topk_key(x):
    """Map descending FP32 order to ascending unsigned integer order."""
    bits = x.to(gl.uint32, bitcast=True)
    sign = bits & 0x80000000
    return bits ^ gl.where(sign != 0, 0, 0x7FFFFFFF)


@gluon.jit
def _topk_add(a, b):
    return a + b


@gluon.jit
def _load_topk_tile(
    candidate_logits,
    tile_start,
    candidate_len,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    """Load one tile; only the tail is masked, so full tiles keep 128-bit loads.

    These loads run inside the tile loops, where gfx1250 buffer loads lower to
    32-bit accesses, so they use global loads.
    """
    offsets = tile_start + gl.arange(0, BLOCK_N, layout=value_layout)
    offsets = gl.max_contiguous(gl.multiple_of(offsets.to(gl.int32), 4), 4)
    if IS_TAIL:
        valid = offsets < candidate_len
        values = gl.load(candidate_logits + offsets, mask=valid, other=-float("inf"))
    else:
        valid = gl.full([BLOCK_N], True, gl.int1, layout=value_layout)
        values = gl.load(candidate_logits + offsets)
    return offsets, values, valid


@gluon.jit
def _add_histogram_tile(
    values,
    valid,
    prefix,
    shared_histogram,
    shift: gl.constexpr,
    radix_bits: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
    HIST_REPLICAS: gl.constexpr,
    FIRST_PASS: gl.constexpr,
):
    keys = _fp32_to_topk_key(values)
    if FIRST_PASS:
        prefix_match = valid
    else:
        prefix_match = valid & ((keys >> (shift + radix_bits)) == prefix)
    slots = ((keys >> shift) & ((1 << radix_bits) - 1)).to(gl.int32)
    if FIRST_PASS and HIST_REPLICAS > 1:
        # Lanes of one wave that share a bucket would serialize on one LDS word;
        # spreading them over adjacent replicas puts them in different banks.
        lanes = gl.arange(0, BLOCK_N, layout=value_layout) // LOAD_ELEMS
        slots = slots * HIST_REPLICAS + (lanes & (HIST_REPLICAS - 1))
    shared_histogram.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        slots,
        axis=0,
        mask=prefix_match,
    )


@gluon.jit
def _accumulate_histogram_tile(
    candidate_logits,
    tile_start,
    candidate_len,
    prefix,
    shared_histogram,
    shift: gl.constexpr,
    radix_bits: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
    HIST_REPLICAS: gl.constexpr,
    FIRST_PASS: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    _, values, valid = _load_topk_tile(
        candidate_logits, tile_start, candidate_len, value_layout, BLOCK_N, IS_TAIL
    )
    _add_histogram_tile(
        values,
        valid,
        prefix,
        shared_histogram,
        shift,
        radix_bits,
        value_layout,
        BLOCK_N,
        LOAD_ELEMS,
        HIST_REPLICAS,
        FIRST_PASS,
    )


@gluon.jit
def _accumulate_histogram_tile_pair(
    candidate_logits,
    tile_start,
    candidate_len,
    prefix,
    shared_histogram,
    shift: gl.constexpr,
    radix_bits: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
    HIST_REPLICAS: gl.constexpr,
    FIRST_PASS: gl.constexpr,
):
    """Issue two full tiles' loads before either tile's LDS atomics."""
    _, first_values, first_valid = _load_topk_tile(
        candidate_logits, tile_start, candidate_len, value_layout, BLOCK_N, False
    )
    _, second_values, second_valid = _load_topk_tile(
        candidate_logits,
        tile_start + BLOCK_N,
        candidate_len,
        value_layout,
        BLOCK_N,
        False,
    )
    _add_histogram_tile(
        first_values,
        first_valid,
        prefix,
        shared_histogram,
        shift,
        radix_bits,
        value_layout,
        BLOCK_N,
        LOAD_ELEMS,
        HIST_REPLICAS,
        FIRST_PASS,
    )
    _add_histogram_tile(
        second_values,
        second_valid,
        prefix,
        shared_histogram,
        shift,
        radix_bits,
        value_layout,
        BLOCK_N,
        LOAD_ELEMS,
        HIST_REPLICAS,
        FIRST_PASS,
    )


@gluon.jit
def _emit_topk_tile(
    candidate_logits,
    block_table,
    tile_start,
    candidate_len,
    candidate_start,
    threshold,
    count_greater,
    remaining,
    written_greater,
    written_equal,
    shared_output_counters,
    out,
    row,
    req,
    block_table_cols: gl.constexpr,
    page_size: gl.constexpr,
    out_stride: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PREFIX_SHIFT: gl.constexpr,
    IS_DECODE: gl.constexpr,
    SCAN_EMIT: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    """Write one tile's selected keys and return the running output counts.

    SCAN_EMIT ranks keys with an exclusive scan of the greater and equal flags
    plus the running counts, which the block carries because it walks tiles in
    order; ties keep the lowest offsets. Otherwise keys reserve positions with
    LDS atomics on two counters: lanes then serialize on one word, but each
    tile skips the block-wide scan, which suits latency-bound decode rows.
    """
    offsets, values, valid = _load_topk_tile(
        candidate_logits, tile_start, candidate_len, value_layout, BLOCK_N, IS_TAIL
    )
    keys = _fp32_to_topk_key(values)
    compared_keys = keys if PREFIX_SHIFT == 0 else keys >> PREFIX_SHIFT
    greater = valid & (compared_keys < threshold)
    equal = valid & (compared_keys == threshold)
    if SCAN_EMIT:
        # A tile holds at most BLOCK_N <= 65535 keys, so both counts pack in 32 bits.
        flags = greater.to(gl.int32) | (equal.to(gl.int32) << 16)
        ranks = gl.associative_scan(flags, 0, _topk_add) - flags
        greater_position = written_greater + (ranks & 0xFFFF)
        equal_rank = written_equal + (ranks >> 16)
        tile_counts = gl.sum(flags, axis=0)
        written_greater += tile_counts & 0xFFFF
        written_equal += tile_counts >> 16
    else:
        reservation = shared_output_counters.atomic_scatter_add(
            gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
            gl.where(greater, 0, 1).to(gl.int32),
            axis=0,
            mask=greater | equal,
        )
        greater_position = reservation
        equal_rank = reservation
    greater_write = greater & (greater_position < count_greater)
    equal_write = equal & (equal_rank < remaining)
    logical_offsets = candidate_start + offsets.to(gl.int32)
    if IS_DECODE:
        block_idx = logical_offsets // page_size
        block_offset = logical_offsets - block_idx * page_size
        page = gl.load(
            block_table + req * block_table_cols + block_idx,
            mask=(greater_write | equal_write) & (block_idx < block_table_cols),
            other=0,
        ).to(gl.int32)
        output_values = page * page_size + block_offset
    else:
        output_values = logical_offsets
    gl.store(
        out + row * out_stride + greater_position,
        output_values,
        mask=greater_write,
    )
    gl.store(
        out + row * out_stride + count_greater + equal_rank,
        output_values,
        mask=equal_write,
    )
    return written_greater, written_equal


@gluon.jit
def _emit_topk(
    candidate_logits,
    block_table,
    candidate_len,
    candidate_start,
    threshold,
    count_greater,
    remaining,
    shared_output_counters,
    out,
    row,
    req,
    block_table_cols: gl.constexpr,
    page_size: gl.constexpr,
    out_stride: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PREFIX_SHIFT: gl.constexpr,
    IS_DECODE: gl.constexpr,
    SCAN_EMIT: gl.constexpr,
):
    written_greater = gl.full([], 0, gl.int32)
    written_equal = gl.full([], 0, gl.int32)
    full_end = candidate_len & -BLOCK_N
    for tile_start in range(0, full_end, BLOCK_N):
        written_greater, written_equal = _emit_topk_tile(
            candidate_logits,
            block_table,
            tile_start,
            candidate_len,
            candidate_start,
            threshold,
            count_greater,
            remaining,
            written_greater,
            written_equal,
            shared_output_counters,
            out,
            row,
            req,
            block_table_cols,
            page_size,
            out_stride,
            value_layout,
            BLOCK_N,
            PREFIX_SHIFT,
            IS_DECODE,
            SCAN_EMIT,
            False,
        )
    if full_end < candidate_len:
        _emit_topk_tile(
            candidate_logits,
            block_table,
            full_end,
            candidate_len,
            candidate_start,
            threshold,
            count_greater,
            remaining,
            written_greater,
            written_equal,
            shared_output_counters,
            out,
            row,
            req,
            block_table_cols,
            page_size,
            out_stride,
            value_layout,
            BLOCK_N,
            PREFIX_SHIFT,
            IS_DECODE,
            SCAN_EMIT,
            True,
        )


@gluon.jit
def _select_bucket(
    counts,
    remaining,
    bucket_offsets,
    group_layout: gl.constexpr,
):
    """Return the bucket holding the remaining-th key, the count before it, and its count."""
    count_pairs = counts.reshape([_MAX_BUCKETS // 2, 2])
    count_low, count_high = gl.split(count_pairs)
    count_low = gl.convert_layout(count_low, group_layout)
    count_high = gl.convert_layout(count_high, group_layout)
    group_counts = count_low + count_high
    cumulative = gl.associative_scan(group_counts, 0, _topk_add)
    before_group = cumulative - group_counts
    selected_group = (before_group < remaining) & (cumulative >= remaining)

    bucket_pairs = bucket_offsets.reshape([_MAX_BUCKETS // 2, 2])
    bucket_low, bucket_high = gl.split(bucket_pairs)
    bucket_low = gl.convert_layout(bucket_low, group_layout)
    bucket_high = gl.convert_layout(bucket_high, group_layout)
    select_low = before_group + count_low >= remaining
    selected_bucket = gl.where(select_low, bucket_low, bucket_high)
    selected_greater = before_group + gl.where(select_low, 0, count_low)
    selected_count = gl.where(select_low, count_low, count_high)
    # Only one group is selected, so sums publish its fields; the greater count
    # is below topk <= 2048 and fits beside the 12-bit bucket.
    packed = selected_bucket.to(gl.uint32) | (selected_greater.to(gl.uint32) << 12)
    packed = gl.sum(gl.where(selected_group, packed, 0), axis=0)
    bucket_count = gl.sum(gl.where(selected_group, selected_count, 0), axis=0)
    return packed & 0xFFF, ((packed >> 12) & 0x7FF).to(gl.int32), bucket_count


@gluon.jit
def _compact_topk_tile(
    candidate_logits,
    tile_start,
    candidate_len,
    candidate_start,
    threshold,
    written_winners,
    written_candidates,
    candidate_keys,
    candidate_offsets,
    out,
    row,
    out_stride: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    """Write keys above the first-pass bucket and stash the bucket's keys in LDS."""
    offsets, values, valid = _load_topk_tile(
        candidate_logits, tile_start, candidate_len, value_layout, BLOCK_N, IS_TAIL
    )
    keys = _fp32_to_topk_key(values)
    high = keys >> (32 - _RADIX0_BITS)
    winner = valid & (high < threshold)
    candidate = valid & (high == threshold)
    flags = winner.to(gl.int32) | (candidate.to(gl.int32) << 16)
    ranks = gl.associative_scan(flags, 0, _topk_add) - flags
    candidate_position = written_candidates + (ranks >> 16)
    logical_offsets = candidate_start + offsets.to(gl.int32)
    gl.store(
        out + row * out_stride + written_winners + (ranks & 0xFFFF),
        logical_offsets,
        mask=winner,
    )
    candidate_keys.atomic_scatter_xchg(
        keys.to(gl.int32, bitcast=True), candidate_position, axis=0, mask=candidate
    )
    candidate_offsets.atomic_scatter_xchg(
        logical_offsets, candidate_position, axis=0, mask=candidate
    )
    tile_counts = gl.sum(flags, axis=0)
    return written_winners + (tile_counts & 0xFFFF), written_candidates + (
        tile_counts >> 16
    )


@gluon.jit
def _finish_from_candidates(
    candidate_logits,
    candidate_len,
    candidate_start,
    prefix,
    remaining,
    candidate_count,
    shared_histogram,
    out,
    row,
    topk: gl.constexpr,
    out_stride: gl.constexpr,
    value_layout: gl.constexpr,
    histogram_layout: gl.constexpr,
    group_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    COMPACT_CAP: gl.constexpr,
):
    """Finish a row whose first-pass bucket holds at most COMPACT_CAP keys.

    One more sweep writes the keys above that bucket and compacts the bucket
    into LDS, so the last two radix passes and the emit never reread the row.
    The candidates live in histogram slots past the first-pass bucket counts,
    which only the replicated first pass uses.
    """
    candidate_keys = shared_histogram.slice(_MAX_BUCKETS, COMPACT_CAP)
    candidate_offsets = shared_histogram.slice(_MAX_BUCKETS + COMPACT_CAP, COMPACT_CAP)
    written_winners = gl.full([], 0, gl.int32)
    written_candidates = gl.full([], 0, gl.int32)
    full_end = candidate_len & -BLOCK_N
    for tile_start in range(0, full_end, BLOCK_N):
        written_winners, written_candidates = _compact_topk_tile(
            candidate_logits,
            tile_start,
            candidate_len,
            candidate_start,
            prefix,
            written_winners,
            written_candidates,
            candidate_keys,
            candidate_offsets,
            out,
            row,
            out_stride,
            value_layout,
            BLOCK_N,
            False,
        )
    if full_end < candidate_len:
        _compact_topk_tile(
            candidate_logits,
            full_end,
            candidate_len,
            candidate_start,
            prefix,
            written_winners,
            written_candidates,
            candidate_keys,
            candidate_offsets,
            out,
            row,
            out_stride,
            value_layout,
            BLOCK_N,
            True,
        )

    candidate_layout: gl.constexpr = _vector_layout(
        gl.num_warps(), COMPACT_CAP // (32 * gl.num_warps())
    )
    positions = gl.arange(0, COMPACT_CAP, layout=candidate_layout)
    valid = positions < candidate_count
    keys = candidate_keys.load(candidate_layout).to(gl.uint32, bitcast=True)
    histogram = shared_histogram.slice(0, _MAX_BUCKETS)
    bucket_offsets = gl.arange(0, _MAX_BUCKETS, layout=histogram_layout)
    count_greater = topk - remaining
    for pass_index in gl.static_range(1, 3):
        radix_bits = _RADIX1_BITS
        shift = 32 - _RADIX0_BITS - _RADIX1_BITS
        if pass_index == 2:
            radix_bits = _RADIX2_BITS
            shift = 0
        histogram.store(gl.zeros([_MAX_BUCKETS], gl.int32, layout=histogram_layout))
        histogram.atomic_scatter_add(
            gl.full([COMPACT_CAP], 1, gl.int32, layout=candidate_layout),
            ((keys >> shift) & ((1 << radix_bits) - 1)).to(gl.int32),
            axis=0,
            mask=valid & ((keys >> (shift + radix_bits)) == prefix),
        )
        bucket, greater, _ = _select_bucket(
            histogram.load(histogram_layout), remaining, bucket_offsets, group_layout
        )
        prefix = (prefix << radix_bits) | bucket
        remaining -= greater

    logical_offsets = candidate_offsets.load(candidate_layout)
    greater = valid & (keys < prefix)
    equal = valid & (keys == prefix)
    flags = greater.to(gl.int32) | (equal.to(gl.int32) << 16)
    ranks = gl.associative_scan(flags, 0, _topk_add) - flags
    equal_rank = ranks >> 16
    gl.store(
        out + row * out_stride + count_greater + (ranks & 0xFFFF),
        logical_offsets,
        mask=greater,
    )
    gl.store(
        out + row * out_stride + topk - remaining + equal_rank,
        logical_offsets,
        mask=equal & (equal_rank < remaining),
    )


@gluon.jit
def _dsa_wave32_radix_topk_kernel(
    logits,
    block_table,
    row_starts,
    row_ends,
    out,
    lens_out,
    logits_stride: gl.constexpr,
    out_stride: gl.constexpr,
    block_table_cols: gl.constexpr,
    page_size: gl.constexpr,
    topk: gl.constexpr,
    q_len_per_req: gl.constexpr,
    IS_DECODE: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
    HIST_REPLICAS: gl.constexpr,
    SCAN_EMIT: gl.constexpr,
    COMPACT_CAP: gl.constexpr,
):
    row = gl.program_id(0)
    value_layout: gl.constexpr = _vector_layout(gl.num_warps(), LOAD_ELEMS)
    histogram_layout: gl.constexpr = _vector_layout(
        gl.num_warps(), _MAX_BUCKETS // (32 * gl.num_warps())
    )
    histogram_slots: gl.constexpr = _MAX_BUCKETS * HIST_REPLICAS
    slot_layout: gl.constexpr = _vector_layout(
        gl.num_warps(), histogram_slots // (32 * gl.num_warps())
    )
    group_layout: gl.constexpr = _vector_layout(
        gl.num_warps(), (_MAX_BUCKETS // 2) // (32 * gl.num_warps())
    )
    output_layout: gl.constexpr = _vector_layout(
        gl.num_warps(), topk // (32 * gl.num_warps())
    )
    histogram_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[histogram_slots, 1]],
        [histogram_slots],
        [0],
    )
    shared_histogram = gl.allocate_shared_memory(
        gl.int32,
        [histogram_slots],
        histogram_shared_layout,
    )
    if SCAN_EMIT:
        shared_output_counters = shared_histogram
    else:
        counter_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[2, 1]], [2], [0]
        )
        shared_output_counters = gl.allocate_shared_memory(
            gl.int32, [2], counter_shared_layout
        )
    histogram_zeros = gl.zeros(
        [_MAX_BUCKETS],
        gl.int32,
        layout=histogram_layout,
    )
    slot_zeros = gl.zeros(
        [histogram_slots],
        gl.int32,
        layout=slot_layout,
    )

    if IS_DECODE:
        req = row // q_len_per_req
        q_offset = row - req * q_len_per_req
        candidate_start = gl.full([], 0, gl.int32)
        candidate_end = gl.load(row_ends + req).to(gl.int32)
        if q_len_per_req != 1:
            candidate_end = candidate_end - (q_len_per_req - 1) + q_offset
    else:
        req = row
        candidate_start = gl.load(row_starts + row).to(gl.int32)
        candidate_end = gl.load(row_ends + row).to(gl.int32)

    candidate_len = gl.maximum(candidate_end - candidate_start, 0)
    selected_count = gl.minimum(candidate_len, topk).to(gl.int32)
    output_offsets = gl.arange(0, topk, layout=output_layout)
    gl.store(lens_out + row, selected_count)

    if candidate_len <= topk:
        valid = output_offsets < candidate_len
        logical_offsets = candidate_start + output_offsets.to(gl.int32)
        if IS_DECODE:
            block_idx = logical_offsets // page_size
            block_offset = logical_offsets - block_idx * page_size
            page = gl.load(
                block_table + req * block_table_cols + block_idx,
                mask=valid & (block_idx < block_table_cols),
                other=0,
            ).to(gl.int32)
            indices = page * page_size + block_offset
        else:
            indices = logical_offsets
        gl.store(
            out + row * out_stride + output_offsets,
            gl.where(valid, indices, -1),
        )
        return

    candidate_logits = logits + row * logits_stride + candidate_start
    bucket_offsets = gl.arange(0, _MAX_BUCKETS, layout=histogram_layout)
    prefix = gl.full([], 0, gl.uint32)
    remaining = gl.full([], topk, gl.int32)
    if not SCAN_EMIT:
        shared_output_counters.store(
            gl.zeros([2], gl.int32, layout=_vector_layout(gl.num_warps(), 1))
        )

    # The three-pass schedule resolves the full ordered FP32 key. Only the first
    # pass replicates the histogram: its buckets are the exponent bits, where
    # nearby scores collide; later passes bucket mantissa bits.
    for pass_index in gl.static_range(3):
        radix_bits = _RADIX0_BITS
        shift = 32 - _RADIX0_BITS
        histogram = shared_histogram
        if pass_index == 0:
            histogram.store(slot_zeros)
        else:
            histogram = shared_histogram.slice(0, _MAX_BUCKETS)
            histogram.store(histogram_zeros)
        if pass_index == 1:
            radix_bits = _RADIX1_BITS
            shift = 32 - _RADIX0_BITS - _RADIX1_BITS
        elif pass_index == 2:
            radix_bits = _RADIX2_BITS
            shift = 0
        full_end = candidate_len & -BLOCK_N
        paired_end = candidate_len & -(2 * BLOCK_N)
        for tile_start in range(0, paired_end, 2 * BLOCK_N):
            _accumulate_histogram_tile_pair(
                candidate_logits,
                tile_start,
                candidate_len,
                prefix,
                histogram,
                shift,
                radix_bits,
                value_layout,
                BLOCK_N,
                LOAD_ELEMS,
                HIST_REPLICAS,
                pass_index == 0,
            )
        for tile_start in range(paired_end, full_end, BLOCK_N):
            _accumulate_histogram_tile(
                candidate_logits,
                tile_start,
                candidate_len,
                prefix,
                histogram,
                shift,
                radix_bits,
                value_layout,
                BLOCK_N,
                LOAD_ELEMS,
                HIST_REPLICAS,
                pass_index == 0,
                False,
            )
        if full_end < candidate_len:
            _accumulate_histogram_tile(
                candidate_logits,
                full_end,
                candidate_len,
                prefix,
                histogram,
                shift,
                radix_bits,
                value_layout,
                BLOCK_N,
                LOAD_ELEMS,
                HIST_REPLICAS,
                pass_index == 0,
                True,
            )

        if pass_index == 0 and HIST_REPLICAS > 1:
            counts = histogram.load(slot_layout)
            counts = gl.sum(counts.reshape([_MAX_BUCKETS, HIST_REPLICAS]), axis=1)
            counts = gl.convert_layout(counts, histogram_layout)
        else:
            counts = histogram.load(histogram_layout)
        bucket, greater, bucket_count = _select_bucket(
            counts, remaining, bucket_offsets, group_layout
        )
        prefix = (prefix << radix_bits) | bucket
        remaining -= greater
        if pass_index == 0 and COMPACT_CAP > 0:
            if bucket_count <= COMPACT_CAP:
                _finish_from_candidates(
                    candidate_logits,
                    candidate_len,
                    candidate_start,
                    prefix,
                    remaining,
                    bucket_count,
                    shared_histogram,
                    out,
                    row,
                    topk,
                    out_stride,
                    value_layout,
                    histogram_layout,
                    group_layout,
                    BLOCK_N,
                    COMPACT_CAP,
                )
                return
        if pass_index == 1:
            if bucket_count == remaining:
                _emit_topk(
                    candidate_logits,
                    block_table,
                    candidate_len,
                    candidate_start,
                    prefix,
                    topk - remaining,
                    remaining,
                    shared_output_counters,
                    out,
                    row,
                    req,
                    block_table_cols,
                    page_size,
                    out_stride,
                    value_layout,
                    BLOCK_N,
                    shift,
                    IS_DECODE,
                    SCAN_EMIT,
                )
                return

    _emit_topk(
        candidate_logits,
        block_table,
        candidate_len,
        candidate_start,
        prefix,
        topk - remaining,
        remaining,
        shared_output_counters,
        out,
        row,
        req,
        block_table_cols,
        page_size,
        out_stride,
        value_layout,
        BLOCK_N,
        0,
        IS_DECODE,
        SCAN_EMIT,
    )


def _check_score_input_contract(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
) -> None:
    if weights.device != q.device or index_k_cache.device != q.device:
        raise ValueError("q, weights, and index_k_cache must be on the same device")
    if q.stride(-1) != 1 or weights.stride(-1) != 1:
        raise ValueError("q and weights must have contiguous innermost dimensions")
    if not index_k_cache.is_contiguous():
        raise ValueError("index_k_cache must be contiguous")


def _check_standard_scorer_inputs(
    q: torch.Tensor,
    q_scales: torch.Tensor | None,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_size: int,
) -> tuple[int, int, bool]:
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError(
            f"standard-cache DSA scorer expects BF16 or FP8 q, got {q.dtype}"
        )
    if weights.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"DSA weights must be BF16 or FP32, got {weights.dtype}")
    if q.dim() != 3 or q.shape[1] not in (32, 64) or q.shape[2] != 128:
        raise ValueError(
            "standard-cache DSA scorer requires q=[tokens, 32|64, 128], got "
            f"{tuple(q.shape)}"
        )
    if weights.shape != q.shape[:2]:
        raise ValueError(
            f"weights must have shape {tuple(q.shape[:2])}, got {tuple(weights.shape)}"
        )
    if q.stride(-1) != 1 or weights.stride(-1) != 1:
        raise ValueError("q and weights must have contiguous innermost dimensions")
    if page_size != 64:
        raise ValueError(
            f"standard-cache DSA scorer requires page_size=64, got {page_size}"
        )
    if index_k_cache.dtype != torch.uint8:
        raise TypeError("index_k_cache must be a uint8 tensor")
    row_bytes = 128 + 4
    if index_k_cache.dim() != 2:
        raise ValueError(
            "index_k_cache must be a packed slot matrix or page-planar matrix, "
            f"got shape {tuple(index_k_cache.shape)}"
        )
    page_bytes = page_size * row_bytes
    if index_k_cache.shape[1] == row_bytes:
        if not index_k_cache.is_contiguous():
            raise ValueError("packed index_k_cache must be contiguous")
        if index_k_cache.shape[0] % page_size:
            raise ValueError("index_k_cache slot count must be page aligned")
        page_stride_bytes = page_bytes
    elif (
        index_k_cache.shape[1] >= page_bytes
        and index_k_cache.stride(1) == 1
        and index_k_cache.stride(0) >= page_bytes
    ):
        page_stride_bytes = index_k_cache.stride(0)
    else:
        raise ValueError(
            "index_k_cache must be contiguous [slots, row_bytes] or page-planar "
            f"[pages, at least {page_bytes} bytes], got "
            f"shape={tuple(index_k_cache.shape)}, stride={index_k_cache.stride()}"
        )
    if index_k_cache.storage_offset() % 4 or page_stride_bytes % 4:
        raise ValueError("index_k_cache page storage must be float32 aligned")
    if weights.device != q.device or index_k_cache.device != q.device:
        raise ValueError("q, weights, and index_k_cache must be on the same device")
    q_is_fp8 = q.dtype == torch.float8_e4m3fn
    if q_is_fp8:
        if q_scales is None:
            raise ValueError("FP8 q requires per-token/head q_scales")
        if (
            q_scales.dtype != torch.float32
            or q_scales.shape != q.shape[:2]
            or q_scales.device != q.device
            or q_scales.stride(-1) != 1
        ):
            raise ValueError(
                "q_scales must be FP32 [tokens, heads] with a contiguous head axis"
            )
    elif q_scales is not None:
        raise ValueError("q_scales is only valid with FP8 q")
    return row_bytes, page_stride_bytes, q_is_fp8


def _compact_cap(hist_replicas: int) -> int:
    """Return the most first-pass bucket keys that fit in the spare replica slots.

    Each candidate takes a key and an offset slot past the first
    ``1 << max(_RADIX_BITS)`` bucket counts; the cap is a power of two.
    """
    spare = (hist_replicas - 1) * (1 << max(_RADIX_BITS)) // 2
    return 1 << (spare.bit_length() - 1) if spare else 0


def _check_topk_contract(topk: int) -> None:
    if topk not in (512, 1024, 2048):
        raise ValueError(
            f"DSA Gluon top-k supports topk=512, 1024, or 2048, got {topk}"
        )


def _dsa_topk_indices(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    out: torch.Tensor,
    lens_out: torch.Tensor,
    block_table: torch.Tensor | None = None,
    page_size: int = 1,
    q_len_per_req: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_table is None:
        is_decode = False
        block_table = row_starts
        block_table_cols = 0
    else:
        is_decode = True
        block_table_cols = block_table.shape[1]
    if is_decode:
        block_n = _DECODE_TOPK_BLOCK_N
        num_warps = min(_DECODE_TOPK_NUM_WARPS, topk // 32)
        waves_per_eu = _DECODE_TOPK_WAVES_PER_EU
        hist_replicas = _DECODE_TOPK_HIST_REPLICAS
        compact_cap = 0
    else:
        block_n = _TOPK_BLOCK_N
        num_warps = _TOPK_NUM_WARPS
        waves_per_eu = _TOPK_WAVES_PER_EU
        hist_replicas = (
            _TOPK_SHORT_HIST_REPLICAS
            if logits.shape[1] <= _TOPK_SHORT_MAX_COLS
            else _TOPK_HIST_REPLICAS
        )
        compact_cap = (
            _compact_cap(hist_replicas)
            if logits.shape[1] >= _TOPK_COMPACT_MIN_COLS
            else 0
        )
    rows = logits.shape[0]
    _dsa_wave32_radix_topk_kernel[(rows,)](
        logits,
        block_table,
        row_starts,
        row_ends,
        out,
        lens_out,
        logits.stride(0),
        out.stride(0),
        block_table_cols,
        page_size=int(page_size),
        topk=topk,
        q_len_per_req=q_len_per_req,
        IS_DECODE=is_decode,
        BLOCK_N=block_n,
        LOAD_ELEMS=block_n // (32 * num_warps),
        HIST_REPLICAS=hist_replicas,
        SCAN_EMIT=not is_decode,
        COMPACT_CAP=compact_cap,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    return out, lens_out


def gluon_dsa_logical_topk_gfx1250(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    out: torch.Tensor | None,
    lens_out: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select logical columns from bounded FP32 rows with Wave32 radix top-k.

    Args:
        logits: Contiguous FP32 scores shaped ``[rows, columns]``.
        row_starts: Contiguous int32 inclusive bounds shaped ``[rows]``.
        row_ends: Contiguous int32 exclusive bounds shaped ``[rows]``.
        topk: Number of logical columns to select.
        out: Optional contiguous int32 output shaped ``[rows, topk]``.
        lens_out: Optional contiguous int32 valid-count output shaped ``[rows]``.

    Returns:
        Selected logical columns padded with ``-1`` and their valid counts.
    """
    topk = int(topk)
    _check_topk_contract(topk)
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D, got shape={tuple(logits.shape)}")
    if logits.dtype != torch.float32:
        raise TypeError(f"logits must be float32, got {logits.dtype}")
    if not logits.is_cuda:
        raise RuntimeError("Gluon logical top-k requires CUDA tensors")
    if not logits.is_contiguous():
        raise ValueError("logits must be contiguous")

    rows = int(logits.shape[0])
    for name, bounds in (("row_starts", row_starts), ("row_ends", row_ends)):
        if bounds.shape != (rows,):
            raise ValueError(
                f"{name} must have shape {(rows,)}, got {tuple(bounds.shape)}"
            )
        if bounds.dtype != torch.int32:
            raise TypeError(f"{name} must be int32, got {bounds.dtype}")
        if bounds.device != logits.device:
            raise ValueError(f"{name} must be on the same device as logits")
        if not bounds.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    expected_out_shape = (rows, topk)
    if out is None:
        out = torch.empty(expected_out_shape, dtype=torch.int32, device=logits.device)
    elif (
        out.shape != expected_out_shape
        or out.dtype != torch.int32
        or out.device != logits.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous int32 on logits.device with shape "
            f"{expected_out_shape}"
        )

    if lens_out is None:
        lens_out = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    elif (
        lens_out.shape != (rows,)
        or lens_out.dtype != torch.int32
        or lens_out.device != logits.device
        or not lens_out.is_contiguous()
    ):
        raise ValueError(
            "lens_out must be contiguous int32 on logits.device with shape "
            f"{(rows,)}"
        )

    if rows == 0:
        return out, lens_out
    return _dsa_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )


def _check_kpool_score_inputs(
    q: torch.Tensor,
    pooled_k_cache: torch.Tensor,
    weights: torch.Tensor,
    *,
    pool_size: int,
    page_size: int,
    ordered_head_fold: bool,
) -> int:
    if not isinstance(ordered_head_fold, bool):
        raise TypeError(
            f"ordered_head_fold must be bool, got {type(ordered_head_fold).__name__}"
        )
    if q.dtype != torch.bfloat16:
        raise TypeError(f"KPool Gluon scorer requires BF16 q, got {q.dtype}")
    if q.dim() != 3 or tuple(q.shape[1:]) != (32, 128):
        raise ValueError(
            f"KPool Gluon scorer requires q=[tokens, 32, 128], got {tuple(q.shape)}"
        )
    if q.stride(-1) != 1:
        raise ValueError("q must have a contiguous head-dimension axis")
    if not q.is_cuda:
        raise RuntimeError("KPool Gluon scorer requires CUDA tensors")
    if weights.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"KPool weights must be BF16 or FP32, got {weights.dtype}")
    if weights.shape != q.shape[:2] or weights.stride(-1) != 1:
        raise ValueError("weights must match q and have a contiguous head axis")
    if pool_size != 4 or page_size != 16:
        raise ValueError(
            "KPool Gluon scorer requires pool_size=4 and page_size=16, got "
            f"pool_size={pool_size}, page_size={page_size}"
        )

    row_bytes = 128 + 4
    compact_page_bytes = page_size * row_bytes
    cache_shape_ok = (
        pooled_k_cache.dim() == 2 and pooled_k_cache.shape[1] == compact_page_bytes
    ) or (
        pooled_k_cache.dim() == 3
        and tuple(pooled_k_cache.shape[1:]) == (page_size, row_bytes)
    )
    cache_page_stride_bytes = int(pooled_k_cache.stride(0))
    packed_within_page = (
        pooled_k_cache.dim() == 2 or pooled_k_cache.stride(1) == row_bytes
    )
    if pooled_k_cache.dtype != torch.uint8:
        raise TypeError(f"pooled_k_cache must be uint8, got {pooled_k_cache.dtype}")
    if not cache_shape_ok or (
        pooled_k_cache.stride(-1) != 1
        or not packed_within_page
        or cache_page_stride_bytes < compact_page_bytes
        or cache_page_stride_bytes % 4
        or pooled_k_cache.storage_offset() % 4
    ):
        raise ValueError(
            "pooled_k_cache requires packed rows and a nonoverlapping, "
            "4-byte-aligned page stride"
        )
    if weights.device != q.device or pooled_k_cache.device != q.device:
        raise ValueError("weights and pooled_k_cache must be on the same device as q")
    return cache_page_stride_bytes


def _prepare_kpool_score_outputs(
    q: torch.Tensor,
    *,
    window_cols: int,
    out: torch.Tensor | None,
    row_ends_out: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_logits = (int(q.shape[0]), window_cols)
    if out is None:
        out = torch.empty(expected_logits, dtype=torch.float32, device=q.device)
    elif (
        out.shape != expected_logits
        or out.dtype != torch.float32
        or out.device != q.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous FP32 on q.device with shape "
            f"{expected_logits}, got shape={tuple(out.shape)}, dtype={out.dtype}, "
            f"device={out.device}, contiguous={out.is_contiguous()}"
        )

    expected_ends = (int(q.shape[0]),)
    if row_ends_out is None:
        row_ends_out = torch.empty(expected_ends, dtype=torch.int32, device=q.device)
    elif (
        row_ends_out.shape != expected_ends
        or row_ends_out.dtype != torch.int32
        or row_ends_out.device != q.device
        or not row_ends_out.is_contiguous()
    ):
        raise ValueError(
            "row_ends_out must be contiguous int32 on q.device with shape "
            f"{expected_ends}"
        )
    return out, row_ends_out


def gluon_dsa_kpool_prefill_logits_gfx1250(
    q: torch.Tensor,
    pooled_k_cache: torch.Tensor,
    weights: torch.Tensor,
    causal_lens: torch.Tensor,
    req_ids: torch.Tensor,
    index_block_table: torch.Tensor,
    *,
    pool_size: int,
    page_size: int,
    pool_offset: int,
    window_cols: int,
    softmax_scale: float,
    ordered_head_fold: bool,
    out: torch.Tensor | None,
    row_ends_out: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a request-table-addressed GLM KPool window with Wave32 WMMA.

    Args:
        q: BF16 queries shaped ``[tokens, 32, 128]``.
        pooled_k_cache: Page-major uint8 FP8 keys and FP32 row scales.
        weights: BF16 or FP32 signed head weights shaped ``[tokens, 32]``.
        causal_lens: Visible raw-token counts shaped ``[tokens]``.
        req_ids: Request-table row for each query token.
        index_block_table: Logical-pool-page to physical-page mapping.
        pool_size: Raw tokens represented by each pooled key.
        page_size: Pooled keys per physical page.
        pool_offset: First global pool represented by output column zero.
        window_cols: Output columns in the local scoring window.
        softmax_scale: Scale applied after weighted per-head ReLU reduction.
        ordered_head_fold: Whether to fold heads in logical order.
        out: Optional contiguous FP32 scoring workspace.
        row_ends_out: Optional contiguous int32 local exclusive bounds.

    Returns:
        Local-window scores and exclusive valid bounds.
    """
    pool_size = int(pool_size)
    page_size = int(page_size)
    pool_offset = int(pool_offset)
    window_cols = int(window_cols)
    cache_page_stride_bytes = _check_kpool_score_inputs(
        q,
        pooled_k_cache,
        weights,
        pool_size=pool_size,
        page_size=page_size,
        ordered_head_fold=ordered_head_fold,
    )
    if pool_offset < 0:
        raise ValueError(f"pool_offset must be nonnegative, got {pool_offset}")
    if window_cols <= 0:
        raise ValueError(f"window_cols must be positive, got {window_cols}")

    tokens = int(q.shape[0])
    for name, value in (("causal_lens", causal_lens), ("req_ids", req_ids)):
        if value.shape != (tokens,):
            raise ValueError(f"{name} must have shape {(tokens,)}, got {value.shape}")
        if value.dtype != torch.int32:
            raise TypeError(f"{name} must be int32, got {value.dtype}")
        if value.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if index_block_table.dim() != 2:
        raise ValueError(
            "index_block_table must be 2-D, got "
            f"shape={tuple(index_block_table.shape)}"
        )
    if index_block_table.dtype != torch.int32:
        raise TypeError(
            f"index_block_table must be int32, got {index_block_table.dtype}"
        )
    if index_block_table.device != q.device:
        raise ValueError("index_block_table must be on the same device as q")
    if index_block_table.stride(-1) != 1:
        raise ValueError("index_block_table must have contiguous page columns")
    if tokens and index_block_table.shape[0] == 0:
        raise ValueError("index_block_table must have at least one request row")

    out, row_ends_out = _prepare_kpool_score_outputs(
        q,
        window_cols=window_cols,
        out=out,
        row_ends_out=row_ends_out,
    )
    if tokens == 0:
        return out, row_ends_out

    max_candidates = int(index_block_table.shape[1]) * page_size
    if max_candidates and pooled_k_cache.shape[0] == 0:
        raise ValueError("a nonempty index_block_table requires pooled cache pages")
    if max_candidates == 0 or pool_offset >= max_candidates:
        row_ends_out.zero_()
        return out, row_ends_out

    num_warps = _KPOOL_SCORE_NUM_WARPS
    gluon_kpool_prefill_topk_fp8_gfx1250[(tokens, 1)](
        q,
        weights,
        pooled_k_cache.view(torch.float8_e4m3fn),
        pooled_k_cache.view(torch.float32),
        weights,
        causal_lens,
        req_ids,
        index_block_table,
        out,
        row_ends_out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        weights.stride(0),
        weights.stride(1),
        weights.stride(0),
        weights.stride(1),
        index_block_table.stride(0),
        out.stride(0),
        float(softmax_scale),
        max_candidates,
        pool_offset,
        PAGE_SIZE=page_size,
        PAGE_STRIDE_BYTES=cache_page_stride_bytes,
        POOL_SIZE=pool_size,
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=_KPOOL_SCORE_BLOCK_N,
        WINDOW_COLS=window_cols,
        NUM_WARPS=num_warps,
        ORDERED_HEAD_FOLD=ordered_head_fold,
        USE_BUFFER_LOAD=pooled_k_cache.untyped_storage().nbytes() < 2**31,
        USE_BUFFER_STORE=out.nbytes < 2**31,
        num_warps=num_warps,
        waves_per_eu=_KPOOL_SCORE_WAVES_PER_EU,
    )
    return out, row_ends_out


def gluon_dsa_kpool_prefill_plan_logits_gfx1250(
    q: torch.Tensor,
    pooled_k_cache: torch.Tensor,
    weights: torch.Tensor,
    pool_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    pool_size: int,
    page_size: int,
    pool_offset: int,
    window_cols: int,
    softmax_scale: float,
    ordered_head_fold: bool,
    out: torch.Tensor | None,
    row_ends_out: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a physical-slot GLM KPool prefill window with Wave32 WMMA.

    Args:
        q: BF16 queries shaped ``[tokens, 32, 128]``.
        pooled_k_cache: Page-major uint8 FP8 keys and FP32 row scales.
        weights: BF16 or FP32 signed head weights shaped ``[tokens, 32]``.
        pool_workspace_slots: Physical pooled-cache slots in logical order.
        row_starts: Inclusive workspace start for each query token.
        row_ends: Exclusive workspace end for each query token.
        pool_size: Raw tokens represented by each pooled key.
        page_size: Pooled keys per physical page.
        pool_offset: First request-local pool represented by output column zero.
        window_cols: Output columns in the local scoring window.
        softmax_scale: Scale applied after weighted per-head ReLU reduction.
        ordered_head_fold: Whether to fold heads in logical order.
        out: Optional contiguous FP32 scoring workspace.
        row_ends_out: Optional contiguous int32 local exclusive bounds.

    Returns:
        Local-window scores and exclusive valid bounds.
    """
    pool_size = int(pool_size)
    page_size = int(page_size)
    pool_offset = int(pool_offset)
    window_cols = int(window_cols)
    cache_page_stride_bytes = _check_kpool_score_inputs(
        q,
        pooled_k_cache,
        weights,
        pool_size=pool_size,
        page_size=page_size,
        ordered_head_fold=ordered_head_fold,
    )
    if pool_offset < 0:
        raise ValueError(f"pool_offset must be nonnegative, got {pool_offset}")
    if window_cols <= 0:
        raise ValueError(f"window_cols must be positive, got {window_cols}")

    tokens = int(q.shape[0])
    if pool_workspace_slots.dim() != 1:
        raise ValueError("pool_workspace_slots must be one-dimensional")
    if pool_workspace_slots.dtype != torch.int64:
        raise TypeError(
            f"pool_workspace_slots must be int64, got {pool_workspace_slots.dtype}"
        )
    for name, value in (("row_starts", row_starts), ("row_ends", row_ends)):
        if value.shape != (tokens,):
            raise ValueError(f"{name} must have shape {(tokens,)}, got {value.shape}")
        if value.dtype != torch.int32:
            raise TypeError(f"{name} must be int32, got {value.dtype}")
    for name, value in (
        ("pool_workspace_slots", pool_workspace_slots),
        ("row_starts", row_starts),
        ("row_ends", row_ends),
    ):
        if value.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    out, row_ends_out = _prepare_kpool_score_outputs(
        q,
        window_cols=window_cols,
        out=out,
        row_ends_out=row_ends_out,
    )
    if tokens == 0:
        return out, row_ends_out
    if pool_workspace_slots.numel() and pooled_k_cache.shape[0] == 0:
        raise ValueError("a nonempty prefill plan requires pooled cache pages")

    num_warps = _KPOOL_SCORE_NUM_WARPS
    gluon_kpool_prefill_topk_fp8_plan_gfx1250[(tokens, 1)](
        q,
        weights,
        pooled_k_cache.view(torch.float8_e4m3fn),
        pooled_k_cache.view(torch.float32),
        weights,
        pool_workspace_slots,
        row_starts,
        row_ends,
        out,
        row_ends_out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        weights.stride(0),
        weights.stride(1),
        weights.stride(0),
        weights.stride(1),
        out.stride(0),
        float(softmax_scale),
        int(pool_workspace_slots.numel()),
        pool_offset,
        PAGE_SIZE=page_size,
        PAGE_STRIDE_BYTES=cache_page_stride_bytes,
        POOL_SIZE=pool_size,
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=_KPOOL_SCORE_BLOCK_N,
        WINDOW_COLS=window_cols,
        NUM_WARPS=num_warps,
        ORDERED_HEAD_FOLD=ordered_head_fold,
        USE_BUFFER_LOAD=pooled_k_cache.untyped_storage().nbytes() < 2**31,
        USE_BUFFER_STORE=out.nbytes < 2**31,
        num_warps=num_warps,
        waves_per_eu=_KPOOL_SCORE_WAVES_PER_EU,
    )
    return out, row_ends_out


def launch_gluon_dsa_decode_topk_fp8_gfx1250(
    q: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    q_len_per_req: int = 1,
    index_k_cache: torch.Tensor | None = None,
    seq_lens_2d: torch.Tensor | None = None,
    plan: object | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del plan, seq_lens_2d
    topk = int(topk)
    q_len_per_req = int(q_len_per_req)
    _check_topk_contract(topk)
    if q_len_per_req not in (1, 2, 3, 4, 5, 6):
        raise ValueError(
            f"DSA Gluon top-k supports q_len_per_req=1..6, got {q_len_per_req}"
        )
    if index_k_cache is None:
        raise RuntimeError("Gluon DSA paged top-k requires packed FP8 index_k_cache")
    row_bytes = _check_packed_fp8_inputs(q, index_k_cache, weights, int(page_size))
    _check_score_input_contract(q, weights, index_k_cache)
    if seq_lens.dim() != 1:
        raise ValueError(
            f"seq_lens must be 1-D, got {tuple(seq_lens.shape)} for q={tuple(q.shape)}"
        )
    expected_tokens = int(seq_lens.numel()) * q_len_per_req
    if expected_tokens != q.shape[0]:
        raise ValueError(
            "q rows must equal seq_lens rows times q_len_per_req, got "
            f"q={tuple(q.shape)}, seq_lens={tuple(seq_lens.shape)}, "
            f"q_len_per_req={q_len_per_req}"
        )
    if block_table.dim() != 2 or block_table.shape[0] < seq_lens.numel():
        raise ValueError(
            "block_table must have at least one row per request, got "
            f"block_table={tuple(block_table.shape)}, q={tuple(q.shape)}"
        )
    if seq_lens.dtype != torch.int32 or block_table.dtype != torch.int32:
        raise TypeError("seq_lens and block_table must be int32")
    if seq_lens.device != q.device or block_table.device != q.device:
        raise ValueError("decode metadata must be on the same device as q")
    if not seq_lens.is_contiguous() or not block_table.is_contiguous():
        raise ValueError("seq_lens and block_table must be contiguous")
    if q.shape[0] == 0:
        empty_out = (
            torch.empty((0, topk), dtype=torch.int32, device=q.device)
            if out is None
            else out
        )
        empty_lens = (
            torch.empty((0,), dtype=torch.int32, device=q.device)
            if lens_out is None
            else lens_out
        )
        return empty_out, empty_lens

    max_seq_len = int(block_table.shape[1]) * int(page_size)
    if out is None:
        out = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    logits = torch.empty(
        (q.shape[0], max_seq_len),
        dtype=torch.float32,
        device=q.device,
    )
    block_n = 32
    gluon_dsa_decode_topk_fp8_gfx1250[(q.shape[0], triton.cdiv(max_seq_len, block_n))](
        q,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        seq_lens,
        block_table,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        weights.stride(0),
        weights.stride(1),
        block_table.stride(0),
        logits.stride(0),
        page_size=int(page_size),
        row_bytes=row_bytes,
        max_seq_len=max_seq_len,
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        num_groups=q.shape[2] // 128,
        softmax_scale=float(softmax_scale),
        q_len_per_req=q_len_per_req,
        BLOCK_N=block_n,
        BLOCK_D=128,
        num_warps=4,
        waves_per_eu=1,
    )
    return _dsa_topk_indices(
        logits,
        seq_lens,
        seq_lens,
        block_table=block_table,
        page_size=int(page_size),
        topk=topk,
        q_len_per_req=q_len_per_req,
        out=out,
        lens_out=lens_out,
    )


def launch_gluon_dsa_prefill_topk_fp8_gfx1250(
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    softmax_scale: float,
    index_k_cache: torch.Tensor | None = None,
    page_size: int | None = None,
    index_k_fp8: torch.Tensor | None = None,
    index_k_scale: torch.Tensor | None = None,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del index_k_fp8, index_k_scale
    topk = int(topk)
    _check_topk_contract(topk)
    if index_k_cache is None or page_size is None:
        raise RuntimeError(
            "Gluon DSA top-k requires packed FP8 index_k_cache and page_size"
        )
    row_bytes = _check_packed_fp8_inputs(q, index_k_cache, weights, int(page_size))
    _check_score_input_contract(q, weights, index_k_cache)
    if kv_workspace_slots.dim() != 1:
        raise ValueError(
            f"kv_workspace_slots must be 1-D, got {tuple(kv_workspace_slots.shape)}"
        )
    if row_starts.shape != (q.shape[0],) or row_ends.shape != (q.shape[0],):
        raise ValueError(
            "row_starts/row_ends must be [tokens], got "
            f"row_starts={tuple(row_starts.shape)}, row_ends={tuple(row_ends.shape)}, "
            f"q={tuple(q.shape)}"
        )
    if (
        kv_workspace_slots.dtype != torch.int64
        or row_starts.dtype != torch.int32
        or row_ends.dtype != torch.int32
    ):
        raise TypeError(
            "kv_workspace_slots must be int64 and row_starts/row_ends must be int32"
        )
    if (
        kv_workspace_slots.device != q.device
        or row_starts.device != q.device
        or row_ends.device != q.device
    ):
        raise ValueError("prefill metadata must be on the same device as q")
    if not (
        kv_workspace_slots.is_contiguous()
        and row_starts.is_contiguous()
        and row_ends.is_contiguous()
    ):
        raise ValueError("prefill metadata must be contiguous")
    if out is None:
        out = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    if q.shape[0] == 0:
        return out, lens_out

    seq_len_sum = int(kv_workspace_slots.numel())
    if seq_len_sum == 0:
        out.fill_(-1)
        lens_out.zero_()
        return out, lens_out
    if max_logits_bytes is None:
        max_query_rows = q.shape[0]
    else:
        max_query_rows = max(1, int(max_logits_bytes) // (max(seq_len_sum, 1) * 4))

    block_n = 32
    for start in range(0, q.shape[0], max_query_rows):
        end = min(start + max_query_rows, q.shape[0])
        logits = torch.empty(
            (end - start, seq_len_sum),
            dtype=torch.float32,
            device=q.device,
        )
        gluon_dsa_prefill_topk_fp8_gfx1250[
            (end - start, triton.cdiv(seq_len_sum, block_n))
        ](
            q[start:end],
            index_k_cache.view(torch.float8_e4m3fn),
            index_k_cache.view(torch.float32),
            weights[start:end],
            kv_workspace_slots,
            row_starts[start:end],
            row_ends[start:end],
            logits,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            weights.stride(0),
            weights.stride(1),
            logits.stride(0),
            seq_len_sum=seq_len_sum,
            page_size=int(page_size),
            row_bytes=row_bytes,
            num_heads=q.shape[1],
            head_dim=q.shape[2],
            num_groups=q.shape[2] // 128,
            softmax_scale=float(softmax_scale),
            BLOCK_N=block_n,
            BLOCK_D=128,
            num_warps=4,
            waves_per_eu=1,
        )
        _dsa_topk_indices(
            logits,
            row_starts[start:end],
            row_ends[start:end],
            topk=topk,
            out=out[start:end],
            lens_out=lens_out[start:end],
        )
    return out, lens_out


def launch_gluon_dsa_decode_topk_standard_gfx1250(
    q: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    q_len_per_req: int = 1,
    index_k_cache: torch.Tensor | None = None,
    q_scales: torch.Tensor | None = None,
    seq_lens_2d: torch.Tensor | None = None,
    plan: object | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score standard FP8 index keys with Wave32 WMMA and return global slots."""
    del plan, seq_lens_2d
    topk = int(topk)
    q_len_per_req = int(q_len_per_req)
    _check_topk_contract(topk)
    if q_len_per_req not in (1, 2, 3, 4, 5, 6):
        raise ValueError(
            f"DSA Gluon top-k supports q_len_per_req=1..6, got {q_len_per_req}"
        )
    if index_k_cache is None:
        raise RuntimeError("standard-cache DSA scorer requires index_k_cache")
    _, page_stride_bytes, q_is_fp8 = _check_standard_scorer_inputs(
        q,
        q_scales,
        weights,
        index_k_cache,
        int(page_size),
    )
    if seq_lens.dim() != 1:
        raise ValueError(f"seq_lens must be 1-D, got {tuple(seq_lens.shape)}")
    expected_tokens = int(seq_lens.numel()) * q_len_per_req
    if expected_tokens != q.shape[0]:
        raise ValueError(
            "q rows must equal seq_lens rows times q_len_per_req, got "
            f"q={tuple(q.shape)}, seq_lens={tuple(seq_lens.shape)}, "
            f"q_len_per_req={q_len_per_req}"
        )
    if block_table.dim() != 2 or block_table.shape[0] < seq_lens.numel():
        raise ValueError(
            "block_table must have at least one row per request, got "
            f"{tuple(block_table.shape)}"
        )
    if seq_lens.dtype != torch.int32 or block_table.dtype != torch.int32:
        raise TypeError("seq_lens and block_table must be int32")
    if seq_lens.device != q.device or block_table.device != q.device:
        raise ValueError("decode metadata must be on the same device as q")
    if not seq_lens.is_contiguous() or not block_table.is_contiguous():
        raise ValueError("seq_lens and block_table must be contiguous")
    if out is None:
        out = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    if q.shape[0] == 0:
        return out, lens_out

    max_candidates = int(block_table.shape[1]) * int(page_size)
    if max_candidates == 0:
        out.fill_(-1)
        lens_out.zero_()
        return out, lens_out
    logits = torch.empty(
        (q.shape[0], max_candidates),
        dtype=torch.float32,
        device=q.device,
    )
    block_n = _STANDARD_DECODE_BLOCK_N
    chunk_n = _STANDARD_DECODE_CHUNK_N
    num_warps = _STANDARD_DECODE_NUM_WARPS
    q_scale_arg = q_scales if q_scales is not None else weights
    cache_span = (
        0
        if index_k_cache.shape[0] == 0
        else (index_k_cache.shape[0] - 1) * index_k_cache.stride(0)
        + index_k_cache.shape[1]
    )
    grid = (q.shape[0], triton.cdiv(max_candidates, chunk_n))
    gluon_dsa_decode_topk_standard_gfx1250[grid](
        q,
        q_scale_arg,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        seq_lens,
        block_table,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q_scale_arg.stride(0),
        q_scale_arg.stride(1),
        weights.stride(0),
        weights.stride(1),
        block_table.stride(0),
        logits.stride(0),
        float(softmax_scale),
        max_candidates,
        q_len_per_req,
        PAGE_SIZE=int(page_size),
        PAGE_STRIDE_BYTES=page_stride_bytes,
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=block_n,
        CHUNK_N=chunk_n,
        NUM_WARPS=num_warps,
        Q_IS_FP8=q_is_fp8,
        USE_BUFFER_LOAD=cache_span < 2**31,
        USE_BUFFER_STORE=logits.nbytes < 2**31,
        num_warps=num_warps,
        waves_per_eu=_STANDARD_DECODE_WAVES_PER_EU,
    )
    return _dsa_topk_indices(
        logits,
        seq_lens,
        seq_lens,
        block_table=block_table,
        page_size=int(page_size),
        topk=topk,
        q_len_per_req=q_len_per_req,
        out=out,
        lens_out=lens_out,
    )


def launch_gluon_dsa_prefill_topk_standard_gfx1250(
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    softmax_scale: float,
    index_k_cache: torch.Tensor | None = None,
    page_size: int | None = None,
    index_k_fp8: torch.Tensor | None = None,
    index_k_scale: torch.Tensor | None = None,
    q_scales: torch.Tensor | None = None,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score standard FP8 workspace keys with Wave32 WMMA and return row ids."""
    del index_k_fp8, index_k_scale
    topk = int(topk)
    _check_topk_contract(topk)
    if index_k_cache is None or page_size is None:
        raise RuntimeError("standard-cache DSA scorer requires cache and page_size")
    _, page_stride_bytes, q_is_fp8 = _check_standard_scorer_inputs(
        q,
        q_scales,
        weights,
        index_k_cache,
        int(page_size),
    )
    if kv_workspace_slots.dim() != 1:
        raise ValueError(
            f"kv_workspace_slots must be 1-D, got {tuple(kv_workspace_slots.shape)}"
        )
    if row_starts.shape != (q.shape[0],) or row_ends.shape != (q.shape[0],):
        raise ValueError("row_starts and row_ends must have one element per q row")
    if (
        kv_workspace_slots.dtype != torch.int64
        or row_starts.dtype != torch.int32
        or row_ends.dtype != torch.int32
    ):
        raise TypeError(
            "kv_workspace_slots must be int64 and row_starts/row_ends must be int32"
        )
    if (
        kv_workspace_slots.device != q.device
        or row_starts.device != q.device
        or row_ends.device != q.device
    ):
        raise ValueError("prefill metadata must be on the same device as q")
    if not (
        kv_workspace_slots.is_contiguous()
        and row_starts.is_contiguous()
        and row_ends.is_contiguous()
    ):
        raise ValueError("prefill metadata must be contiguous")
    if out is None:
        out = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    if q.shape[0] == 0:
        return out, lens_out

    workspace_rows = int(kv_workspace_slots.numel())
    if workspace_rows == 0:
        out.fill_(-1)
        lens_out.zero_()
        return out, lens_out
    if max_logits_bytes is None:
        max_query_rows = q.shape[0]
    else:
        max_query_rows = max(1, int(max_logits_bytes) // (max(workspace_rows, 1) * 4))

    q_scale_arg = q_scales if q_scales is not None else weights
    cache_span = (
        0
        if index_k_cache.shape[0] == 0
        else (index_k_cache.shape[0] - 1) * index_k_cache.stride(0)
        + index_k_cache.shape[1]
    )
    dummy_table = row_starts
    for start in range(0, q.shape[0], max_query_rows):
        end = min(start + max_query_rows, q.shape[0])
        if end - start > _STANDARD_PREFILL_MANY_ROWS:
            block_n = _STANDARD_PREFILL_MANY_ROWS_BLOCK_N
            num_warps = _STANDARD_PREFILL_MANY_ROWS_NUM_WARPS
            # A 64-head query is two register tiles and spills at four waves.
            waves_per_eu = 4 if q.shape[1] == 32 else 2
        else:
            block_n = _STANDARD_PREFILL_BLOCK_N
            num_warps = _STANDARD_PREFILL_NUM_WARPS
            waves_per_eu = _STANDARD_PREFILL_WAVES_PER_EU
        logits = torch.empty(
            (end - start, workspace_rows),
            dtype=torch.float32,
            device=q.device,
        )
        gluon_dsa_prefill_topk_standard_gfx1250[(end - start, 1)](
            q[start:end],
            q_scale_arg[start:end],
            index_k_cache.view(torch.float8_e4m3fn),
            index_k_cache.view(torch.float32),
            weights[start:end],
            kv_workspace_slots,
            row_starts[start:end],
            row_ends[start:end],
            dummy_table,
            logits,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q_scale_arg.stride(0),
            q_scale_arg.stride(1),
            weights.stride(0),
            weights.stride(1),
            logits.stride(0),
            float(softmax_scale),
            workspace_rows,
            PAGE_SIZE=int(page_size),
            PAGE_STRIDE_BYTES=page_stride_bytes,
            NUM_HEADS=q.shape[1],
            HEAD_DIM=q.shape[2],
            BLOCK_N=block_n,
            NUM_WARPS=num_warps,
            Q_IS_FP8=q_is_fp8,
            USE_BUFFER_LOAD=cache_span < 2**31,
            USE_BUFFER_STORE=logits.nbytes < 2**31,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
        )
        _dsa_topk_indices(
            logits,
            row_starts[start:end],
            row_ends[start:end],
            topk=topk,
            out=out[start:end],
            lens_out=lens_out[start:end],
        )
    return out, lens_out
