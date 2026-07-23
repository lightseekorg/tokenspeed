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

"""DSA top-k Gluon kernels for AMD GFX950."""

from __future__ import annotations

from collections import OrderedDict
from functools import cache
from threading import Lock
from typing import NamedTuple

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.attention.gluon.dsa_score_gfx950 import (
    _check_packed_fp8_inputs,
    _dsa_decode_logits_fp8_kernel,
    _dsa_prefill_logits_fp8_kernel,
)

_ONEBLOCK_RADIX_SCHEDULE = (12, 12, 8)
_ONEBLOCK_RADIX_BUCKETS = 1 << max(_ONEBLOCK_RADIX_SCHEDULE)
_ONEBLOCK_DECODE_SHORT_MANUAL_CONFIG = (8192, 8)
_ONEBLOCK_DECODE_LONG_MANUAL_CONFIG = (8192, 4)
_ONEBLOCK_DECODE_SHORT_MANUAL_MAX_COLS = 65536
_ONEBLOCK_PREFILL_RADIX_BLOCK_N = 4096
_ONEBLOCK_PREFILL_WIDE_SHORT_BLOCK_N = 8192
_ONEBLOCK_PREFILL_WIDE_LONG_BLOCK_N = 16384
_ONEBLOCK_PREFILL_WIDE_LONG_MIN_COLS = 512 * 1024
_ONEBLOCK_COMPACT_FINAL_BLOCK_N = 4096
_TOPK2048_MIDRANGE_MAX_COLS = 256 * 1024
_ONEBLOCK_RADIX_MAX_COLS = 90000
_PREFILL_ONEBLOCK_RADIX_MIN_COLS = 98304
_PREFILL_ONEBLOCK_RADIX_MAX_COLS = 196608
_PERSISTENT_PREFILL_MIN_COLS = 128 * 1024
_PERSISTENT_PREFILL_FOUR_GROUP_MIN_COLS = 256 * 1024
_PERSISTENT_PREFILL_FILL_RESIDENCY_MIN_COLS = 1536 * 1024
_PERSISTENT_PREFILL_MIN_ROWS = 32
_TOPK2048_MIDRANGE_MAX_ROWS = 32
_PERSISTENT_GROUP_CANDIDATES = (2, 4, 8, 16, 32, 64)
_TUNED_PERSISTENT_TOPK = 2048
_PERSISTENT_BLOCK_N = 8192
_PERSISTENT_NUM_WARPS = 16
_PERSISTENT_NUM_BUCKETS = gl.constexpr(1 << 11)
_PERSISTENT_NUM_PASSES = gl.constexpr(3)
_PERSISTENT_COUNTER_STRIDE = gl.constexpr(32)
_PERSISTENT_WORKSPACE_CACHE_MAXSIZE = 16

_persistent_topk_workspace_cache: OrderedDict[
    tuple[int, int, int], tuple[torch.Tensor, ...]
] = OrderedDict()
_persistent_topk_graph_workspace_keys: set[tuple[int, int, int]] = set()
_persistent_topk_workspace_lock = Lock()

__all__ = [
    "gluon_dsa_decode_topk_fp8_gfx950",
    "gluon_dsa_prefill_topk_fp8_gfx950",
]


@gluon.constexpr_function
def _vector_layout(
    NUM_WARPS: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    return gl.BlockedLayout([LOAD_ELEMS], [64], [NUM_WARPS], [0])


@gluon.jit
def _fp32_to_topk_key(x):
    bits = x.to(gl.uint32, bitcast=True)
    sign = bits & 0x80000000
    return bits ^ gl.where(sign != 0, 0, 0x7FFFFFFF)


@gluon.jit
def _topk_add(a, b):
    return a + b


@gluon.jit(noinline=True)
def _persistent_histogram_tail(
    row_logits,
    shared_histogram,
    row_start,
    row_len,
    full_tiles,
    pass_index,
    threshold_shift,
    threshold,
    shift,
    bucket_mask,
    BLOCK_N: gl.constexpr,
    value_layout: gl.constexpr,
):
    local_offsets = full_tiles * BLOCK_N + gl.arange(
        0,
        BLOCK_N,
        layout=value_layout,
    )
    local_offsets = gl.max_contiguous(
        gl.multiple_of(local_offsets.to(gl.int32), 4),
        4,
    )
    offsets = row_start + local_offsets
    valid = local_offsets < row_len
    values = gl.amd.cdna4.buffer_load(
        ptr=row_logits,
        offsets=offsets,
        mask=valid,
        other=-float("inf"),
    )
    keys = _fp32_to_topk_key(values)
    if pass_index == 0:
        prefix_match = valid
    else:
        prefix_match = valid & (
            ((keys >> threshold_shift) << threshold_shift) == threshold
        )
    buckets = (keys >> shift) & bucket_mask
    shared_histogram.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        buckets.to(gl.int32),
        axis=0,
        mask=prefix_match,
    )


@gluon.jit(noinline=True)
def _persistent_emit_tail(
    row_logits,
    shared_output_counters,
    shared_greater_offsets,
    shared_equal_offsets,
    row_start,
    row_len,
    full_tiles,
    threshold,
    TOPK: gl.constexpr,
    BLOCK_N: gl.constexpr,
    value_layout: gl.constexpr,
):
    local_offsets = full_tiles * BLOCK_N + gl.arange(
        0,
        BLOCK_N,
        layout=value_layout,
    )
    local_offsets = gl.max_contiguous(
        gl.multiple_of(local_offsets.to(gl.int32), 4),
        4,
    )
    offsets = row_start + local_offsets
    valid = local_offsets < row_len
    values = gl.amd.cdna4.buffer_load(
        ptr=row_logits,
        offsets=offsets,
        mask=valid,
        other=-float("inf"),
    )
    keys = _fp32_to_topk_key(values)
    greater = valid & (keys < threshold)
    equal = valid & (keys == threshold)
    reservation_mask = greater | equal
    reservation_counter = gl.where(greater, 0, 1).to(gl.int32)
    reservation = shared_output_counters.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        reservation_counter,
        axis=0,
        mask=reservation_mask,
    )
    shared_greater_offsets.atomic_scatter_xchg(
        offsets.to(gl.int32),
        reservation,
        axis=0,
        mask=greater & (reservation < TOPK),
    )
    shared_equal_offsets.atomic_scatter_xchg(
        offsets.to(gl.int32),
        reservation,
        axis=0,
        mask=equal & (reservation < TOPK),
    )


@gluon.jit
def _persistent_row_metadata(
    row,
    row_starts,
    row_ends,
    q_len_per_req: gl.constexpr,
    IS_DECODE: gl.constexpr,
):
    if IS_DECODE:
        req = row // q_len_per_req
        q_offset = row - req * q_len_per_req
        row_start = gl.full([], 0, gl.int32)
        row_end = gl.load(row_ends + req).to(gl.int32) - (q_len_per_req - 1) + q_offset
    else:
        req = row
        row_start = gl.load(row_starts + row).to(gl.int32)
        row_end = gl.load(row_ends + row).to(gl.int32)
    row_end = gl.maximum(row_end, row_start)
    return req, row_start, row_end


@gluon.jit(
    do_not_specialize=(
        "logits_stride",
        "block_table_cols",
    ),
)
def _dsa_persistent_radix_topk_kernel(
    logits,
    histograms,
    pass_arrivals,
    pass_done,
    reset_arrivals,
    output_counters,
    block_table,
    row_starts,
    row_ends,
    out,
    lens_out,
    logits_stride,
    out_stride: gl.constexpr,
    block_table_cols,
    page_size: gl.constexpr,
    q_len_per_req: gl.constexpr,
    IS_DECODE: gl.constexpr,
    GROUPS_PER_ROW: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    row = gl.program_id(0)
    group = gl.program_id(1)
    value_layout: gl.constexpr = _vector_layout(
        gl.num_warps(),
        BLOCK_N // (64 * gl.num_warps()),
    )
    output_layout: gl.constexpr = _vector_layout(
        gl.num_warps(),
        max(1, TOPK // (64 * gl.num_warps())),
    )
    req, row_start, row_end = _persistent_row_metadata(
        row,
        row_starts,
        row_ends,
        q_len_per_req,
        IS_DECODE,
    )
    row_len = row_end - row_start
    selected_count = gl.minimum(row_len, TOPK)
    if group == 0:
        gl.store(lens_out + row, selected_count)

    if row_len <= TOPK:
        if group == 0:
            output_offsets = gl.arange(0, TOPK, layout=output_layout)
            valid = output_offsets < row_len
            output_values = row_start + output_offsets
            if IS_DECODE:
                block_idx = output_values // page_size
                physical_page = gl.load(
                    block_table + req * block_table_cols + block_idx,
                    mask=valid & (block_idx < block_table_cols),
                    other=0,
                )
                output_values = physical_page * page_size + output_values % page_size
            gl.store(
                out + row * out_stride + output_offsets,
                gl.where(valid, output_values, -1),
            )
        return

    hist_layout: gl.constexpr = _vector_layout(
        gl.num_warps(),
        _PERSISTENT_NUM_BUCKETS // (64 * gl.num_warps()),
    )
    group_layout: gl.constexpr = _vector_layout(
        gl.num_warps(),
        (_PERSISTENT_NUM_BUCKETS // 2) // (64 * gl.num_warps()),
    )
    hist_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[_PERSISTENT_NUM_BUCKETS, 1]],
        [_PERSISTENT_NUM_BUCKETS],
        [0],
    )
    histogram_zeros = gl.zeros(
        [_PERSISTENT_NUM_BUCKETS],
        gl.int32,
        layout=hist_layout,
    )
    shared_histogram = gl.allocate_shared_memory(
        gl.int32,
        [_PERSISTENT_NUM_BUCKETS],
        hist_shared_layout,
    )
    compact_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[TOPK, 1]],
        [TOPK],
        [0],
    )
    shared_greater_offsets = gl.allocate_shared_memory(
        gl.int32,
        [TOPK],
        compact_shared_layout,
    )
    shared_equal_offsets = gl.allocate_shared_memory(
        gl.int32,
        [TOPK],
        compact_shared_layout,
    )
    output_counter_layout: gl.constexpr = _vector_layout(
        gl.num_warps(),
        1,
    )
    output_counter_shared_layout: gl.constexpr = (
        gl.PaddedSharedLayout.with_identity_for(
            [[2, 1]],
            [2],
            [0],
        )
    )
    output_counter_zeros = gl.zeros([2], gl.int32, layout=output_counter_layout)
    shared_output_counters = gl.allocate_shared_memory(
        gl.int32,
        [2],
        output_counter_shared_layout,
    )
    # Initialize local state before this row's radix passes.
    shared_histogram.store(histogram_zeros)
    shared_output_counters.store(output_counter_zeros)
    gl.barrier()

    bucket_offsets = gl.arange(
        0,
        _PERSISTENT_NUM_BUCKETS,
        layout=hist_layout,
    )
    row_logits = logits + row * logits_stride
    full_tiles = row_len // BLOCK_N
    tail_size = row_len - full_tiles * BLOCK_N
    tail_owner = full_tiles % GROUPS_PER_ROW
    threshold = gl.full([], 0, gl.uint32)
    threshold_shift = gl.full([], 32, gl.int32)
    remaining = gl.full([], TOPK, gl.int32)

    # Static passes avoid keeping the scan's lane masks live across the
    # radix loop.
    for pass_index in gl.static_range(_PERSISTENT_NUM_PASSES):
        if pass_index != 0:
            gl.barrier()
            shared_histogram.store(histogram_zeros)
            gl.barrier()

        shift = gl.maximum(21 - pass_index * 11, 0)
        bucket_mask = gl.where(pass_index == 2, 0x3FF, 0x7FF)

        for tile in range(group, full_tiles, GROUPS_PER_ROW):
            offsets = tile * BLOCK_N + gl.arange(
                0,
                BLOCK_N,
                layout=value_layout,
            )
            offsets = gl.max_contiguous(
                gl.multiple_of(offsets.to(gl.int32), 4),
                4,
            )
            offsets = row_start + offsets
            values = gl.amd.cdna4.buffer_load(
                ptr=row_logits,
                offsets=offsets,
            )
            keys = _fp32_to_topk_key(values)
            if pass_index == 0:
                prefix_match = gl.full([BLOCK_N], True, gl.int1, layout=value_layout)
            else:
                prefix_match = (
                    (keys >> threshold_shift) << threshold_shift
                ) == threshold
            buckets = (keys >> shift) & bucket_mask
            shared_histogram.atomic_scatter_add(
                gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
                buckets.to(gl.int32),
                axis=0,
                mask=prefix_match,
            )

        if (tail_size != 0) & (group == tail_owner):
            _persistent_histogram_tail(
                row_logits,
                shared_histogram,
                row_start,
                row_len,
                full_tiles,
                pass_index,
                threshold_shift,
                threshold,
                shift,
                bucket_mask,
                BLOCK_N,
                value_layout,
            )

        gl.barrier()
        local_counts = shared_histogram.load(hist_layout)
        row_histogram = (
            histograms
            + (row * _PERSISTENT_NUM_PASSES + pass_index) * _PERSISTENT_NUM_BUCKETS
        )
        gl.atomic_add(
            row_histogram + bucket_offsets,
            local_counts,
            mask=local_counts != 0,
            sem="relaxed",
            scope="gpu",
        )
        gl.barrier()

        row_pass_arrival = pass_arrivals + row * _PERSISTENT_COUNTER_STRIDE
        old = gl.atomic_add(
            row_pass_arrival,
            1,
            # Match AITER's fence-free gfx950 arrival protocol after the
            # workgroup barrier. Only pass_done publishes GPU-wide completion.
            sem="relaxed",
            scope="gpu",
        )
        generation_last_arrival = (pass_index + 1) * GROUPS_PER_ROW - 1
        if old == generation_last_arrival:
            # pass_done gates each generation; the counter cannot exceed
            # NUM_PASSES * GROUPS_PER_ROW before the final reset.
            gl.atomic_add(
                pass_done + row * _PERSISTENT_COUNTER_STRIDE,
                1,
                sem="release",
                scope="gpu",
            )
            gl.barrier()
        else:
            gl.atomic_poll(
                pass_done + row * _PERSISTENT_COUNTER_STRIDE,
                pass_index + 1,
                sem="acquire",
                scope="gpu",
            )

        total_counts = gl.load(
            row_histogram + bucket_offsets,
            volatile=True,
        )
        count_pairs = total_counts.reshape([_PERSISTENT_NUM_BUCKETS // 2, 2])
        count_low, count_high = gl.split(count_pairs)
        count_low = gl.convert_layout(count_low, group_layout)
        count_high = gl.convert_layout(count_high, group_layout)
        group_counts = count_low + count_high
        cumulative = gl.associative_scan(group_counts, 0, _topk_add)
        before_group = cumulative - group_counts
        selected_group = (before_group < remaining) & (cumulative >= remaining)
        bucket_pairs = bucket_offsets.reshape([_PERSISTENT_NUM_BUCKETS // 2, 2])
        bucket_low, bucket_high = gl.split(bucket_pairs)
        bucket_low = gl.convert_layout(bucket_low, group_layout)
        bucket_high = gl.convert_layout(bucket_high, group_layout)
        select_low = before_group + count_low >= remaining
        group_bucket = gl.where(select_low, bucket_low, bucket_high)
        group_selected_before = before_group + gl.where(select_low, 0, count_low)
        # Exactly one pair is selected, and both fields fit in 11 bits.
        packed_selection = group_bucket.to(gl.uint32) | (
            group_selected_before.to(gl.uint32) << 11
        )
        packed_selection = gl.sum(
            gl.where(selected_group, packed_selection, 0),
            axis=0,
        )
        selected_bucket = (packed_selection & 0x7FF).to(gl.int32)
        selected_greater = ((packed_selection >> 11) & 0x7FF).to(gl.int32)
        threshold |= selected_bucket.to(gl.uint32) << shift
        threshold_shift = shift
        remaining -= selected_greater

    for tile in range(group, full_tiles, GROUPS_PER_ROW):
        offsets = tile * BLOCK_N + gl.arange(
            0,
            BLOCK_N,
            layout=value_layout,
        )
        offsets = gl.max_contiguous(
            gl.multiple_of(offsets.to(gl.int32), 4),
            4,
        )
        offsets = row_start + offsets
        values = gl.amd.cdna4.buffer_load(
            ptr=row_logits,
            offsets=offsets,
        )
        keys = _fp32_to_topk_key(values)
        greater = keys < threshold
        equal = keys == threshold
        reservation_mask = greater | equal
        reservation_counter = gl.where(greater, 0, 1).to(gl.int32)
        reservation = shared_output_counters.atomic_scatter_add(
            gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
            reservation_counter,
            axis=0,
            mask=reservation_mask,
        )
        shared_greater_offsets.atomic_scatter_xchg(
            offsets.to(gl.int32),
            reservation,
            axis=0,
            mask=greater & (reservation < TOPK),
        )
        shared_equal_offsets.atomic_scatter_xchg(
            offsets.to(gl.int32),
            reservation,
            axis=0,
            mask=equal & (reservation < TOPK),
        )

    if (tail_size != 0) & (group == tail_owner):
        _persistent_emit_tail(
            row_logits,
            shared_output_counters,
            shared_greater_offsets,
            shared_equal_offsets,
            row_start,
            row_len,
            full_tiles,
            threshold,
            TOPK,
            BLOCK_N,
            value_layout,
        )

    gl.barrier()
    output_counter_offsets = gl.arange(0, 2, layout=output_counter_layout)
    local_output_counts = shared_output_counters.load(output_counter_layout)
    local_greater = gl.sum(
        gl.where(output_counter_offsets == 0, local_output_counts, 0),
        axis=0,
    ).to(gl.int32)
    local_equal = gl.sum(
        gl.where(output_counter_offsets == 1, local_output_counts, 0),
        axis=0,
    ).to(gl.int32)
    greater_start = gl.atomic_add(
        output_counters + (row * 2) * _PERSISTENT_COUNTER_STRIDE,
        local_greater,
        sem="relaxed",
        scope="gpu",
    )
    equal_start = gl.atomic_add(
        output_counters + (row * 2 + 1) * _PERSISTENT_COUNTER_STRIDE,
        local_equal,
        sem="relaxed",
        scope="gpu",
    )
    copy_offsets = gl.arange(0, TOPK, layout=output_layout)
    greater_values = shared_greater_offsets.load(output_layout)
    equal_values = shared_equal_offsets.load(output_layout)
    greater_positions = greater_start + copy_offsets
    equal_positions = equal_start + copy_offsets
    greater_write = (copy_offsets < local_greater) & (greater_positions < TOPK)
    equal_write = (copy_offsets < local_equal) & (equal_positions < remaining)
    if IS_DECODE:
        greater_block_idx = greater_values // page_size
        greater_page = gl.load(
            block_table + req * block_table_cols + greater_block_idx,
            mask=(greater_block_idx >= 0)
            & (greater_block_idx < block_table_cols)
            & greater_write,
            other=0,
        )
        greater_values = greater_page * page_size + greater_values % page_size
        equal_block_idx = equal_values // page_size
        equal_page = gl.load(
            block_table + req * block_table_cols + equal_block_idx,
            mask=(equal_block_idx >= 0)
            & (equal_block_idx < block_table_cols)
            & equal_write,
            other=0,
        )
        equal_values = equal_page * page_size + equal_values % page_size
    gl.store(
        out + row * out_stride + greater_positions,
        greater_values,
        mask=greater_write,
    )
    gl.store(
        out + row * out_stride + (TOPK - 1 - equal_positions),
        equal_values,
        mask=equal_write,
    )

    gl.barrier()
    reset_old = gl.atomic_add(
        reset_arrivals + row * _PERSISTENT_COUNTER_STRIDE,
        1,
        sem="relaxed",
        scope="gpu",
    )
    if reset_old == GROUPS_PER_ROW - 1:
        for reset_pass in gl.static_range(_PERSISTENT_NUM_PASSES):
            gl.store(
                histograms
                + (row * _PERSISTENT_NUM_PASSES + reset_pass) * _PERSISTENT_NUM_BUCKETS
                + bucket_offsets,
                histogram_zeros,
            )
        # This is the only reset of the monotonic pass-arrival counter.
        gl.store(
            pass_arrivals + row * _PERSISTENT_COUNTER_STRIDE,
            0,
        )
        gl.store(
            pass_done + row * _PERSISTENT_COUNTER_STRIDE,
            0,
        )
        gl.store(
            output_counters + (row * 2) * _PERSISTENT_COUNTER_STRIDE,
            0,
        )
        gl.store(
            output_counters + (row * 2 + 1) * _PERSISTENT_COUNTER_STRIDE,
            0,
        )
        gl.store(
            reset_arrivals + row * _PERSISTENT_COUNTER_STRIDE,
            0,
        )


@gluon.jit
def _dsa_trivial_topk_kernel(
    block_table,
    row_starts,
    row_ends,
    out,
    lens_out,
    out_stride: gl.constexpr,
    block_table_cols: gl.constexpr,
    page_size: gl.constexpr,
    topk: gl.constexpr,
    q_len_per_req: gl.constexpr,
    IS_DECODE: gl.constexpr,
    TOPK_LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = _vector_layout(gl.num_warps(), TOPK_LOAD_ELEMS)
    offsets = gl.arange(0, topk, layout=layout)

    if IS_DECODE:
        req = row // q_len_per_req
        q_offset = row - req * q_len_per_req
        candidate_len = gl.load(row_ends + req).to(gl.int32)
        if q_len_per_req != 1:
            candidate_len = candidate_len - (q_len_per_req - 1) + q_offset
        valid = offsets < candidate_len
        block_idx = offsets.to(gl.int32) // page_size
        block_offset = offsets.to(gl.int32) - block_idx * page_size
        page = gl.load(
            block_table + req * block_table_cols + block_idx,
            mask=valid & (block_idx < block_table_cols),
            other=0,
        ).to(gl.int32)
        indices = page * page_size + block_offset
    else:
        row_start = gl.load(row_starts + row).to(gl.int32)
        row_end = gl.load(row_ends + row).to(gl.int32)
        candidate_len = gl.maximum(row_end - row_start, 0)
        valid = offsets < candidate_len
        indices = row_start + offsets.to(gl.int32)

    gl.store(out + row * out_stride + offsets, gl.where(valid, indices, -1))
    gl.store(lens_out + row, gl.minimum(candidate_len, topk).to(gl.int32))


@gluon.jit
def _load_oneblock_tile(
    candidate_logits,
    tile_start,
    candidate_len,
    vector_end,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    offsets = tile_start + gl.arange(0, BLOCK_N, layout=value_layout)
    offsets = gl.max_contiguous(gl.multiple_of(offsets.to(gl.int32), 4), 4)
    if IS_TAIL:
        valid = offsets < candidate_len
        vector_mask = gl.max_constancy(offsets < vector_end, 4)
        vector_values = gl.amd.cdna4.buffer_load(
            ptr=candidate_logits,
            offsets=offsets,
            mask=vector_mask,
            other=-float("inf"),
        )
        tail_mask = (offsets >= vector_end) & valid
        tail_values = gl.load(
            candidate_logits + offsets,
            mask=tail_mask,
            other=-float("inf"),
        )
        values = gl.where(tail_mask, tail_values, vector_values)
    else:
        valid = gl.full([BLOCK_N], True, gl.int1, layout=value_layout)
        values = gl.amd.cdna4.buffer_load(
            ptr=candidate_logits,
            offsets=offsets,
        )
    return offsets, values, valid


@gluon.jit
def _accumulate_oneblock_histogram_tile(
    candidate_logits,
    tile_start,
    candidate_len,
    vector_end,
    prefix,
    shared_histogram,
    shift: gl.constexpr,
    radix_bits: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    FIRST_PASS: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    _, values, valid = _load_oneblock_tile(
        candidate_logits,
        tile_start,
        candidate_len,
        vector_end,
        value_layout,
        BLOCK_N,
        IS_TAIL,
    )
    keys = _fp32_to_topk_key(values)
    if FIRST_PASS:
        prefix_match = valid
    else:
        prefix_match = valid & ((keys >> (shift + radix_bits)) == prefix)
    buckets = (keys >> shift) & ((1 << radix_bits) - 1)
    shared_histogram.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        buckets.to(gl.int32),
        axis=0,
        mask=prefix_match,
    )


@gluon.jit
def _accumulate_oneblock_histogram_tile_pair(
    candidate_logits,
    tile_start,
    candidate_len,
    vector_end,
    prefix,
    shared_histogram,
    shift: gl.constexpr,
    radix_bits: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    FIRST_PASS: gl.constexpr,
):
    _, first_values, first_valid = _load_oneblock_tile(
        candidate_logits,
        tile_start,
        candidate_len,
        vector_end,
        value_layout,
        BLOCK_N,
        False,
    )
    _, second_values, second_valid = _load_oneblock_tile(
        candidate_logits,
        tile_start + BLOCK_N,
        candidate_len,
        vector_end,
        value_layout,
        BLOCK_N,
        False,
    )
    first_keys = _fp32_to_topk_key(first_values)
    second_keys = _fp32_to_topk_key(second_values)
    if FIRST_PASS:
        first_match = first_valid
        second_match = second_valid
    else:
        prefix_shift = shift + radix_bits
        first_match = first_valid & ((first_keys >> prefix_shift) == prefix)
        second_match = second_valid & ((second_keys >> prefix_shift) == prefix)
    bucket_mask: gl.constexpr = (1 << radix_bits) - 1
    first_buckets = (first_keys >> shift) & bucket_mask
    second_buckets = (second_keys >> shift) & bucket_mask
    pair_buckets = gl.join(first_buckets, second_buckets).reshape([2 * BLOCK_N])
    pair_match = gl.join(first_match, second_match).reshape([2 * BLOCK_N])
    pair_updates = gl.join(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
    ).reshape([2 * BLOCK_N])
    shared_histogram.atomic_scatter_add(
        pair_updates,
        pair_buckets.to(gl.int32),
        axis=0,
        mask=pair_match,
    )


@gluon.jit
def _emit_oneblock_topk_tile(
    candidate_logits,
    tile_start,
    candidate_len,
    vector_end,
    candidate_start,
    prefix,
    count_greater,
    remaining,
    shared_output_counters,
    out,
    row,
    out_stride: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PREFIX_SHIFT: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    offsets, values, valid = _load_oneblock_tile(
        candidate_logits,
        tile_start,
        candidate_len,
        vector_end,
        value_layout,
        BLOCK_N,
        IS_TAIL,
    )
    keys = _fp32_to_topk_key(values)
    compared_keys = keys if PREFIX_SHIFT == 0 else keys >> PREFIX_SHIFT
    greater_mask = valid & (compared_keys < prefix)
    equal_mask = valid & (compared_keys == prefix)
    reservation_mask = greater_mask | equal_mask
    reservation_counter = gl.where(greater_mask, 0, 1).to(gl.int32)
    reservation = shared_output_counters.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        reservation_counter,
        axis=0,
        mask=reservation_mask,
    )
    greater_position = reservation
    equal_rank = reservation
    equal_position = count_greater + equal_rank
    greater_write = greater_mask
    equal_write = equal_mask & (equal_rank < remaining)
    logical_offsets = candidate_start + offsets.to(gl.int32)

    gl.store(
        out + row * out_stride + greater_position,
        logical_offsets,
        mask=greater_write,
    )
    gl.store(
        out + row * out_stride + equal_position,
        logical_offsets,
        mask=equal_write,
    )


@gluon.jit
def _accumulate_compact_final_histogram_tile(
    candidate_logits,
    tile_start,
    candidate_len,
    vector_end,
    candidate_start,
    prefix,
    shared_histogram,
    shared_output_counters,
    shared_compact_keys,
    shared_compact_offsets,
    out,
    row,
    out_stride: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    RADIX_BITS: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    offsets, values, valid = _load_oneblock_tile(
        candidate_logits,
        tile_start,
        candidate_len,
        vector_end,
        value_layout,
        BLOCK_N,
        IS_TAIL,
    )
    keys = _fp32_to_topk_key(values)
    high_prefix = keys >> RADIX_BITS
    prefix_match = valid & (high_prefix == prefix)
    buckets = keys & ((1 << RADIX_BITS) - 1)
    shared_histogram.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        buckets.to(gl.int32),
        axis=0,
        mask=prefix_match,
    )

    definite_winner = valid & (high_prefix < prefix)
    reservation_mask = definite_winner | prefix_match
    reservation_counter = gl.where(definite_winner, 0, 2).to(gl.int32)
    reservation = shared_output_counters.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        reservation_counter,
        axis=0,
        mask=reservation_mask,
    )
    logical_offsets = candidate_start + offsets.to(gl.int32)
    shared_compact_keys.atomic_scatter_xchg(
        keys,
        reservation,
        axis=0,
        mask=prefix_match,
    )
    shared_compact_offsets.atomic_scatter_xchg(
        logical_offsets,
        reservation,
        axis=0,
        mask=prefix_match,
    )

    gl.store(
        out + row * out_stride + reservation,
        logical_offsets,
        mask=definite_winner,
    )


@gluon.jit
def _emit_compact_final_topk(
    shared_compact_keys,
    shared_compact_offsets,
    compact_count,
    prefix,
    count_greater,
    remaining,
    shared_output_counters,
    out,
    row,
    out_stride: gl.constexpr,
    topk: gl.constexpr,
    output_layout: gl.constexpr,
):
    compact_positions = gl.arange(0, topk, layout=output_layout)
    valid = compact_positions < compact_count
    keys = shared_compact_keys.load(output_layout)
    logical_offsets = shared_compact_offsets.load(output_layout)
    greater_mask = valid & (keys < prefix)
    equal_mask = valid & (keys == prefix)
    reservation_mask = greater_mask | equal_mask
    reservation_counter = gl.where(greater_mask, 0, 1).to(gl.int32)
    reservation = shared_output_counters.atomic_scatter_add(
        gl.full([topk], 1, gl.int32, layout=output_layout),
        reservation_counter,
        axis=0,
        mask=reservation_mask,
    )
    greater_position = reservation
    equal_rank = reservation
    equal_position = count_greater + equal_rank
    greater_write = greater_mask
    equal_write = equal_mask & (equal_rank < remaining)

    gl.store(
        out + row * out_stride + greater_position,
        logical_offsets,
        mask=greater_write,
    )
    gl.store(
        out + row * out_stride + equal_position,
        logical_offsets,
        mask=equal_write,
    )


@gluon.jit
def _dsa_oneblock_manual_radix_topk_kernel(
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
    RADIX0_BITS: gl.constexpr,
    RADIX1_BITS: gl.constexpr,
    RADIX2_BITS: gl.constexpr,
    MAX_BUCKETS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
    COMPACT_FINAL_BLOCK_N: gl.constexpr,
    USE_COMPACT_FINAL: gl.constexpr,
):
    row = gl.program_id(0)
    value_layout: gl.constexpr = _vector_layout(
        gl.num_warps(),
        LOAD_ELEMS,
    )
    histogram_layout: gl.constexpr = _vector_layout(
        gl.num_warps(),
        triton.cdiv(MAX_BUCKETS, 64 * gl.num_warps()),
    )
    group_layout: gl.constexpr = _vector_layout(
        gl.num_warps(),
        triton.cdiv(MAX_BUCKETS // 2, 64 * gl.num_warps()),
    )
    output_layout: gl.constexpr = _vector_layout(
        gl.num_warps(),
        triton.cdiv(topk, 64 * gl.num_warps()),
    )
    if USE_COMPACT_FINAL:
        compact_final_layout: gl.constexpr = _vector_layout(
            gl.num_warps(),
            triton.cdiv(COMPACT_FINAL_BLOCK_N, 64 * gl.num_warps()),
        )
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[MAX_BUCKETS, 1]],
        [MAX_BUCKETS],
        [0],
    )
    histogram_zeros = gl.zeros([MAX_BUCKETS], gl.int32, layout=histogram_layout)
    shared_histogram = gl.allocate_shared_memory(
        gl.int32,
        [MAX_BUCKETS],
        shared_layout,
    )
    output_counter_count: gl.constexpr = 4 if USE_COMPACT_FINAL else 2
    output_counter_layout: gl.constexpr = _vector_layout(gl.num_warps(), 1)
    output_counter_shared_layout: gl.constexpr = (
        gl.PaddedSharedLayout.with_identity_for(
            [[output_counter_count, 1]], [output_counter_count], [0]
        )
    )
    output_counter_zeros = gl.zeros(
        [output_counter_count], gl.int32, layout=output_counter_layout
    )
    shared_output_counters = gl.allocate_shared_memory(
        gl.int32,
        [output_counter_count],
        output_counter_shared_layout,
    )
    if USE_COMPACT_FINAL:
        compact_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[topk, 1]],
            [topk],
            [0],
        )
        shared_compact_keys = gl.allocate_shared_memory(
            gl.uint32,
            [topk],
            compact_shared_layout,
        )
        shared_compact_offsets = gl.allocate_shared_memory(
            gl.int32,
            [topk],
            compact_shared_layout,
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

    shared_histogram.store(histogram_zeros)
    shared_output_counters.store(output_counter_zeros)
    gl.barrier()

    candidate_logits = logits + row * logits_stride + candidate_start
    vector_end = candidate_len & -4
    prefix = gl.full([], 0, gl.uint32)
    remaining = gl.full([], topk, gl.int32)
    if USE_COMPACT_FINAL:
        compact_count = candidate_len
    bucket_offsets = gl.arange(0, MAX_BUCKETS, layout=histogram_layout)

    for pass_index in gl.static_range(3):
        radix_bits = RADIX0_BITS
        shift = 32 - RADIX0_BITS
        if pass_index == 1:
            radix_bits = RADIX1_BITS
            shift = 32 - RADIX0_BITS - RADIX1_BITS
        elif pass_index == 2:
            radix_bits = RADIX2_BITS
            shift = 0

        if pass_index != 0:
            gl.barrier()
            shared_histogram.store(histogram_zeros)
            gl.barrier()

        full_end = candidate_len & -BLOCK_N
        if USE_COMPACT_FINAL and pass_index == 2:
            if compact_count <= topk:
                compact_full_end = candidate_len & -COMPACT_FINAL_BLOCK_N
                for tile_start in range(0, compact_full_end, COMPACT_FINAL_BLOCK_N):
                    _accumulate_compact_final_histogram_tile(
                        candidate_logits,
                        tile_start,
                        candidate_len,
                        vector_end,
                        candidate_start,
                        prefix,
                        shared_histogram,
                        shared_output_counters,
                        shared_compact_keys,
                        shared_compact_offsets,
                        out,
                        row,
                        out_stride,
                        compact_final_layout,
                        COMPACT_FINAL_BLOCK_N,
                        radix_bits,
                        False,
                    )
                if compact_full_end < candidate_len:
                    _accumulate_compact_final_histogram_tile(
                        candidate_logits,
                        compact_full_end,
                        candidate_len,
                        vector_end,
                        candidate_start,
                        prefix,
                        shared_histogram,
                        shared_output_counters,
                        shared_compact_keys,
                        shared_compact_offsets,
                        out,
                        row,
                        out_stride,
                        compact_final_layout,
                        COMPACT_FINAL_BLOCK_N,
                        radix_bits,
                        True,
                    )
            else:
                for tile_start in range(0, full_end, BLOCK_N):
                    _accumulate_oneblock_histogram_tile(
                        candidate_logits,
                        tile_start,
                        candidate_len,
                        vector_end,
                        prefix,
                        shared_histogram,
                        shift,
                        radix_bits,
                        value_layout,
                        BLOCK_N,
                        False,
                        False,
                    )
                if full_end < candidate_len:
                    _accumulate_oneblock_histogram_tile(
                        candidate_logits,
                        full_end,
                        candidate_len,
                        vector_end,
                        prefix,
                        shared_histogram,
                        shift,
                        radix_bits,
                        value_layout,
                        BLOCK_N,
                        False,
                        True,
                    )
        else:
            if not USE_COMPACT_FINAL:
                paired_end = candidate_len & -(2 * BLOCK_N)
                for tile_start in range(0, paired_end, 2 * BLOCK_N):
                    _accumulate_oneblock_histogram_tile_pair(
                        candidate_logits,
                        tile_start,
                        candidate_len,
                        vector_end,
                        prefix,
                        shared_histogram,
                        shift,
                        radix_bits,
                        value_layout,
                        BLOCK_N,
                        pass_index == 0,
                    )
                for tile_start in range(paired_end, full_end, BLOCK_N):
                    _accumulate_oneblock_histogram_tile(
                        candidate_logits,
                        tile_start,
                        candidate_len,
                        vector_end,
                        prefix,
                        shared_histogram,
                        shift,
                        radix_bits,
                        value_layout,
                        BLOCK_N,
                        pass_index == 0,
                        False,
                    )
            else:
                for tile_start in range(0, full_end, BLOCK_N):
                    _accumulate_oneblock_histogram_tile(
                        candidate_logits,
                        tile_start,
                        candidate_len,
                        vector_end,
                        prefix,
                        shared_histogram,
                        shift,
                        radix_bits,
                        value_layout,
                        BLOCK_N,
                        pass_index == 0,
                        False,
                    )
            if full_end < candidate_len:
                _accumulate_oneblock_histogram_tile(
                    candidate_logits,
                    full_end,
                    candidate_len,
                    vector_end,
                    prefix,
                    shared_histogram,
                    shift,
                    radix_bits,
                    value_layout,
                    BLOCK_N,
                    pass_index == 0,
                    True,
                )

        gl.barrier()
        counts = shared_histogram.load(histogram_layout)
        count_pairs = counts.reshape([MAX_BUCKETS // 2, 2])
        count_low, count_high = gl.split(count_pairs)
        count_low = gl.convert_layout(count_low, group_layout)
        count_high = gl.convert_layout(count_high, group_layout)
        group_counts = count_low + count_high
        group_cumulative = gl.associative_scan(group_counts, 0, _topk_add)
        group_greater = group_cumulative - group_counts
        selected_group = (group_greater < remaining) & (group_cumulative >= remaining)
        bucket_pairs = bucket_offsets.reshape([MAX_BUCKETS // 2, 2])
        bucket_low, bucket_high = gl.split(bucket_pairs)
        bucket_low = gl.convert_layout(bucket_low, group_layout)
        bucket_high = gl.convert_layout(bucket_high, group_layout)
        select_low = group_greater + count_low >= remaining
        group_bucket = gl.where(select_low, bucket_low, bucket_high)
        group_selected_greater = group_greater + gl.where(select_low, 0, count_low)
        if pass_index == 1:
            group_selected_count = gl.where(select_low, count_low, count_high)
        # Only one group is selected, so packed sums publish all fields in one reduction.
        if pass_index == 1 and USE_COMPACT_FINAL:
            packed_selection = (
                group_bucket.to(gl.uint64)
                | (group_selected_greater.to(gl.uint64) << 12)
                | (group_selected_count.to(gl.uint64) << 23)
            )
            selected = gl.sum(gl.where(selected_group, packed_selection, 0), axis=0)
            selected_bucket_count = (selected >> 23).to(gl.int32)
        else:
            packed_selection = group_bucket.to(gl.uint32) | (
                group_selected_greater.to(gl.uint32) << 12
            )
            if pass_index == 1 and not USE_COMPACT_FINAL:
                selected_bucket_complete = group_selected_count == (
                    remaining - group_selected_greater
                )
                packed_selection |= selected_bucket_complete.to(gl.uint32) << 23
            selected = gl.sum(gl.where(selected_group, packed_selection, 0), axis=0)
        selected_bucket = (selected & 0xFFF).to(gl.int32)
        selected_greater = ((selected >> 12) & 0x7FF).to(gl.int32)
        prefix = (prefix << radix_bits) | selected_bucket.to(gl.uint32)
        remaining -= selected_greater
        if not USE_COMPACT_FINAL and pass_index == 1:
            if ((selected >> 23) & 1) != 0:
                count_greater = topk - remaining
                emit_full_end = candidate_len & -BLOCK_N
                for tile_start in range(0, emit_full_end, BLOCK_N):
                    _emit_oneblock_topk_tile(
                        candidate_logits,
                        tile_start,
                        candidate_len,
                        vector_end,
                        candidate_start,
                        prefix,
                        count_greater,
                        remaining,
                        shared_output_counters,
                        out,
                        row,
                        out_stride,
                        value_layout,
                        BLOCK_N,
                        shift,
                        False,
                    )
                if emit_full_end < candidate_len:
                    _emit_oneblock_topk_tile(
                        candidate_logits,
                        emit_full_end,
                        candidate_len,
                        vector_end,
                        candidate_start,
                        prefix,
                        count_greater,
                        remaining,
                        shared_output_counters,
                        out,
                        row,
                        out_stride,
                        value_layout,
                        BLOCK_N,
                        shift,
                        True,
                    )

                if IS_DECODE:
                    gl.barrier()
                    logical_offsets = gl.load(
                        out + row * out_stride + output_offsets,
                    ).to(gl.int32)
                    block_idx = logical_offsets // page_size
                    block_offset = logical_offsets - block_idx * page_size
                    page = gl.load(
                        block_table + req * block_table_cols + block_idx,
                        mask=block_idx < block_table_cols,
                        other=0,
                    ).to(gl.int32)
                    gl.store(
                        out + row * out_stride + output_offsets,
                        page * page_size + block_offset,
                    )
                return
        if USE_COMPACT_FINAL and pass_index == 1:
            compact_count = selected_bucket_count

    count_greater = topk - remaining
    if USE_COMPACT_FINAL:
        if compact_count <= topk:
            _emit_compact_final_topk(
                shared_compact_keys,
                shared_compact_offsets,
                compact_count,
                prefix,
                count_greater,
                remaining,
                shared_output_counters,
                out,
                row,
                out_stride,
                topk,
                output_layout,
            )
            return

    full_end = candidate_len & -BLOCK_N
    for tile_start in range(0, full_end, BLOCK_N):
        _emit_oneblock_topk_tile(
            candidate_logits,
            tile_start,
            candidate_len,
            vector_end,
            candidate_start,
            prefix,
            count_greater,
            remaining,
            shared_output_counters,
            out,
            row,
            out_stride,
            value_layout,
            BLOCK_N,
            0,
            False,
        )
    if full_end < candidate_len:
        _emit_oneblock_topk_tile(
            candidate_logits,
            full_end,
            candidate_len,
            vector_end,
            candidate_start,
            prefix,
            count_greater,
            remaining,
            shared_output_counters,
            out,
            row,
            out_stride,
            value_layout,
            BLOCK_N,
            0,
            True,
        )

    if IS_DECODE:
        gl.barrier()
        logical_offsets = gl.load(
            out + row * out_stride + output_offsets,
        ).to(gl.int32)
        block_idx = logical_offsets // page_size
        block_offset = logical_offsets - block_idx * page_size
        page = gl.load(
            block_table + req * block_table_cols + block_idx,
            mask=block_idx < block_table_cols,
            other=0,
        ).to(gl.int32)
        gl.store(
            out + row * out_stride + output_offsets,
            page * page_size + block_offset,
        )


def _load_elems(block: int, num_warps: int) -> int:
    return max(1, triton.cdiv(int(block), 64 * int(num_warps)))


def _check_score_input_contract(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
) -> None:
    if weights.device != q.device or index_k_cache.device != q.device:
        raise ValueError("q, weights, and index_k_cache must be on the same device")
    if not (
        q.is_contiguous() and weights.is_contiguous() and index_k_cache.is_contiguous()
    ):
        raise ValueError("q, weights, and index_k_cache must be contiguous")


def _check_topk_contract(topk: int) -> None:
    if topk not in (512, 1024, 2048):
        raise ValueError(
            f"DSA Gluon top-k supports topk=512, 1024, or 2048, got {topk}"
        )


@cache
def _device_compute_units(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


class _TopKLaunchPlan(NamedTuple):
    kind: str
    groups_per_row: int = 0
    block_n: int = 0
    load_elems: int = 0
    use_compact_final: bool = False


def _tuned_prefill_topk2048_groups(
    rows: int,
    cols: int,
    topk: int,
    device: torch.device,
) -> int | None:
    if (
        rows < _PERSISTENT_PREFILL_MIN_ROWS
        or topk != _TUNED_PERSISTENT_TOPK
        or cols < _PERSISTENT_PREFILL_MIN_COLS
    ):
        return None
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    max_groups = _device_compute_units(device_index) // rows
    if cols >= _PERSISTENT_PREFILL_FILL_RESIDENCY_MIN_COLS:
        return max_groups if max_groups >= 2 else None
    target_groups = 2 if cols < _PERSISTENT_PREFILL_FOUR_GROUP_MIN_COLS else 4
    if max_groups >= target_groups:
        return target_groups
    if target_groups == 4 and max_groups >= 2:
        return 2
    return None


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _oneblock_launch_plan(cols: int, *, is_decode: bool) -> _TopKLaunchPlan:
    if is_decode:
        block_n, load_elems = (
            _ONEBLOCK_DECODE_SHORT_MANUAL_CONFIG
            if cols <= _ONEBLOCK_DECODE_SHORT_MANUAL_MAX_COLS
            else _ONEBLOCK_DECODE_LONG_MANUAL_CONFIG
        )
        return _TopKLaunchPlan("oneblock", block_n=block_n, load_elems=load_elems)

    if cols <= _ONEBLOCK_RADIX_MAX_COLS:
        block_n = _ONEBLOCK_PREFILL_RADIX_BLOCK_N
        use_compact_final = False
    elif _PREFILL_ONEBLOCK_RADIX_MIN_COLS <= cols <= _PREFILL_ONEBLOCK_RADIX_MAX_COLS:
        block_n = _ONEBLOCK_PREFILL_RADIX_BLOCK_N
        use_compact_final = True
    elif cols < _ONEBLOCK_PREFILL_WIDE_LONG_MIN_COLS:
        block_n = _ONEBLOCK_PREFILL_WIDE_SHORT_BLOCK_N
        use_compact_final = True
    else:
        block_n = _ONEBLOCK_PREFILL_WIDE_LONG_BLOCK_N
        use_compact_final = True
    return _TopKLaunchPlan(
        "oneblock",
        block_n=block_n,
        load_elems=_load_elems(block_n, 16),
        use_compact_final=use_compact_final,
    )


def _residency_safe_homogeneous_groups(
    rows: int,
    cols: int,
    device: torch.device,
) -> int | None:
    if rows <= 0:
        return None
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    resident_groups = _device_compute_units(device_index) // rows
    width_groups = _next_power_of_two(triton.cdiv(int(cols), _PERSISTENT_BLOCK_N))
    viable = tuple(
        groups
        for groups in _PERSISTENT_GROUP_CANDIDATES
        if groups <= resident_groups and groups <= width_groups
    )
    return max(viable) if viable else None


def _dsa_topk_plan(
    rows: int,
    cols: int,
    topk: int,
    device: torch.device,
    *,
    is_decode: bool,
) -> _TopKLaunchPlan:
    if cols <= topk:
        return _TopKLaunchPlan("trivial")

    if cols <= _ONEBLOCK_RADIX_MAX_COLS:
        return _oneblock_launch_plan(cols, is_decode=is_decode)

    if (
        topk == _TUNED_PERSISTENT_TOPK
        and cols <= _TOPK2048_MIDRANGE_MAX_COLS
        and rows > _TOPK2048_MIDRANGE_MAX_ROWS
    ):
        return _oneblock_launch_plan(cols, is_decode=is_decode)

    if not is_decode:
        groups = _tuned_prefill_topk2048_groups(rows, cols, topk, device)
        if groups is not None:
            return _TopKLaunchPlan(
                "persistent-homogeneous",
                groups_per_row=groups,
                block_n=_PERSISTENT_BLOCK_N,
            )

    groups = _residency_safe_homogeneous_groups(rows, cols, device)
    if groups is not None:
        return _TopKLaunchPlan(
            "persistent-homogeneous",
            groups_per_row=groups,
            block_n=_PERSISTENT_BLOCK_N,
        )
    return _oneblock_launch_plan(cols, is_decode=is_decode)


def _persistent_topk_workspace(
    rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Return zero-once scratch owned by the current stream and size bucket."""
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream_id = int(torch.cuda.current_stream(device_index).cuda_stream)
    row_bucket = _next_power_of_two(rows)
    key = (device_index, stream_id, row_bucket)
    capturing = torch.cuda.is_current_stream_capturing()
    with _persistent_topk_workspace_lock:
        workspace = _persistent_topk_workspace_cache.get(key)
        if workspace is not None:
            _persistent_topk_workspace_cache.move_to_end(key)
            if capturing:
                _persistent_topk_graph_workspace_keys.add(key)
    if workspace is not None:
        return workspace

    # Retaining a stream-local allocation keeps graph pointers alive and avoids
    # cross-stream races. The kernel restores every touched word to zero.
    workspace = (
        torch.zeros(
            (
                row_bucket,
                _PERSISTENT_NUM_PASSES,
                _PERSISTENT_NUM_BUCKETS,
            ),
            dtype=torch.int32,
            device=device,
        ),
        torch.zeros(
            (row_bucket, _PERSISTENT_COUNTER_STRIDE),
            dtype=torch.int32,
            device=device,
        ),
        torch.zeros(
            (row_bucket, _PERSISTENT_COUNTER_STRIDE),
            dtype=torch.int32,
            device=device,
        ),
        torch.zeros(
            (row_bucket, _PERSISTENT_COUNTER_STRIDE),
            dtype=torch.int32,
            device=device,
        ),
        torch.zeros(
            (row_bucket, 2, _PERSISTENT_COUNTER_STRIDE),
            dtype=torch.int32,
            device=device,
        ),
    )
    with _persistent_topk_workspace_lock:
        existing = _persistent_topk_workspace_cache.get(key)
        if existing is not None:
            _persistent_topk_workspace_cache.move_to_end(key)
            if capturing:
                _persistent_topk_graph_workspace_keys.add(key)
            return existing

        if len(_persistent_topk_workspace_cache) >= (
            _PERSISTENT_WORKSPACE_CACHE_MAXSIZE
        ):
            evict_key = next(
                (
                    cached_key
                    for cached_key in _persistent_topk_workspace_cache
                    if cached_key not in _persistent_topk_graph_workspace_keys
                ),
                None,
            )
            if evict_key is None:
                # Allocations made during capture belong to the graph's memory
                # pool, so an uncached workspace remains valid for replay.
                return workspace
            del _persistent_topk_workspace_cache[evict_key]

        _persistent_topk_workspace_cache[key] = workspace
        if capturing:
            _persistent_topk_graph_workspace_keys.add(key)
        return workspace


def _dsa_persistent_radix_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    groups: int,
    block_n: int,
    workspace: tuple[torch.Tensor, ...],
    out: torch.Tensor,
    lens_out: torch.Tensor,
    block_table: torch.Tensor | None = None,
    page_size: int = 1,
    q_len_per_req: int = 1,
    is_decode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = logits.shape[0]
    histograms, pass_arrivals, pass_done, reset_arrivals, output_counters = workspace
    if block_table is None:
        block_table = row_starts
        block_table_cols = 0
    else:
        block_table_cols = block_table.shape[1]
    _dsa_persistent_radix_topk_kernel[(rows, groups)](
        logits,
        histograms,
        pass_arrivals,
        pass_done,
        reset_arrivals,
        output_counters,
        block_table,
        row_starts,
        row_ends,
        out,
        lens_out,
        logits.stride(0),
        out.stride(0),
        block_table_cols=block_table_cols,
        page_size=page_size,
        q_len_per_req=q_len_per_req,
        IS_DECODE=is_decode,
        GROUPS_PER_ROW=groups,
        TOPK=topk,
        BLOCK_N=block_n,
        num_warps=_PERSISTENT_NUM_WARPS,
    )
    return out, lens_out


def _dsa_oneblock_topk_indices(
    logits: torch.Tensor,
    block_table: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    block_table_cols: int,
    page_size: int,
    topk: int,
    q_len_per_req: int,
    is_decode: bool,
    plan: _TopKLaunchPlan,
    out: torch.Tensor,
    lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device_index = logits.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    waves_per_eu = 8 if logits.shape[0] > _device_compute_units(device_index) else 0
    _dsa_oneblock_manual_radix_topk_kernel[(logits.shape[0],)](
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
        RADIX0_BITS=_ONEBLOCK_RADIX_SCHEDULE[0],
        RADIX1_BITS=_ONEBLOCK_RADIX_SCHEDULE[1],
        RADIX2_BITS=_ONEBLOCK_RADIX_SCHEDULE[2],
        MAX_BUCKETS=_ONEBLOCK_RADIX_BUCKETS,
        BLOCK_N=plan.block_n,
        LOAD_ELEMS=plan.load_elems,
        COMPACT_FINAL_BLOCK_N=_ONEBLOCK_COMPACT_FINAL_BLOCK_N,
        USE_COMPACT_FINAL=plan.use_compact_final,
        num_warps=16,
        # Preserve two resident workgroups only when the grid can use them.
        waves_per_eu=waves_per_eu,
    )
    return out, lens_out


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

    rows, cols = logits.shape
    plan = _dsa_topk_plan(
        rows,
        cols,
        topk,
        logits.device,
        is_decode=is_decode,
    )
    if plan.kind == "trivial":
        _dsa_trivial_topk_kernel[(rows,)](
            block_table,
            row_starts,
            row_ends,
            out,
            lens_out,
            out.stride(0),
            block_table_cols,
            page_size=int(page_size),
            topk=topk,
            q_len_per_req=q_len_per_req,
            IS_DECODE=is_decode,
            TOPK_LOAD_ELEMS=_load_elems(topk, 8),
            num_warps=8,
        )
        return out, lens_out

    if plan.kind == "persistent-homogeneous":
        workspace = _persistent_topk_workspace(rows, logits.device)
        return _dsa_persistent_radix_topk(
            logits,
            row_starts,
            row_ends,
            topk=topk,
            groups=plan.groups_per_row,
            block_n=plan.block_n,
            workspace=workspace,
            out=out,
            lens_out=lens_out,
            block_table=block_table if is_decode else None,
            page_size=page_size,
            q_len_per_req=q_len_per_req,
            is_decode=is_decode,
        )

    return _dsa_oneblock_topk_indices(
        logits,
        block_table,
        row_starts,
        row_ends,
        block_table_cols=block_table_cols,
        page_size=page_size,
        topk=topk,
        q_len_per_req=q_len_per_req,
        is_decode=is_decode,
        plan=plan,
        out=out,
        lens_out=lens_out,
    )


def gluon_dsa_decode_topk_fp8_gfx950(
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
            torch.empty((0, int(topk)), dtype=torch.int32, device=q.device)
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
        (q.shape[0], max_seq_len), dtype=torch.float32, device=q.device
    )
    block_n = 32
    _dsa_decode_logits_fp8_kernel[(q.shape[0], triton.cdiv(max_seq_len, block_n))](
        q,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        seq_lens,
        block_table,
        logits,
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


def gluon_dsa_prefill_topk_fp8_gfx950(
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
            (end - start, seq_len_sum), dtype=torch.float32, device=q.device
        )
        _dsa_prefill_logits_fp8_kernel[
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
