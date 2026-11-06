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
from tokenspeed_kernel_amd._extern import extern_compile_options, release_drain
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.indexing import (
    _check_packed_fp8_inputs,
    gluon_dsa_decode_topk_fp8_gfx950,
    gluon_dsa_prefill_topk_fp8_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.standard_cache_logits import (
    gluon_dsa_decode_topk_bf16_gfx950,
    gluon_dsa_decode_topk_fp8_rows_gfx950,
    gluon_dsa_decode_topk_standard_gfx950,
    gluon_dsa_prefill_topk_bf16_gfx950,
    gluon_dsa_prefill_topk_fp8_rows_gfx950,
    gluon_dsa_prefill_topk_standard_gfx950,
    gluon_kpool_prefill_topk_fp8_gfx950,
    gluon_kpool_prefill_topk_fp8_plan_gfx950,
)

_ONEBLOCK_RADIX_SCHEDULE = (12, 12, 8)
_ONEBLOCK_RADIX_BUCKETS = 1 << max(_ONEBLOCK_RADIX_SCHEDULE)
_ONEBLOCK_DECODE_SHORT_MANUAL_CONFIG = (8192, 8)
_ONEBLOCK_DECODE_LONG_MANUAL_CONFIG = (8192, 4)
_ONEBLOCK_DECODE_SHORT_MANUAL_MAX_COLS = 65536
# Logits crowd a few exponent prefixes, so pass one spreads each bucket over 4 counters.
_ONEBLOCK_DECODE_HIST_COPIES = 4
# 24-bit threshold buckets this small rank in LDS during emit, skipping a third pass.
_ONEBLOCK_DECODE_TIE_CAP = 64
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
# Decode batches up to this many rows may split each row over CTAs (histogram + select).
_MULTI_CTA_MAX_ROWS = 32
# The extra launch (~1.8 us) loses at or below this width; wider logits go multi-CTA.
_MULTI_CTA_ONEBLOCK_MAX_COLS = 16384
_MULTI_CTA_WIDE_MIN_ROWS = 9
_MULTI_CTA_BLOCK_N = 2048
_MULTI_CTA_WIDE_BLOCK_N = 4096
# A CTA takes every n-th run of _MULTI_CTA_GRANULE columns of its row.
_MULTI_CTA_GRANULE = 64
_MULTI_CTA_NUM_WARPS = 8
_MULTI_CTA_NUM_BUCKETS = 4096
_MULTI_CTA_ARRIVAL_STRIDE = 64
# The last CTA gathers these slots over these columns; the rest stream from overflow.
_MULTI_CTA_GATHER = 4096
_MULTI_CTA_GATHER_COLS = 65536
# Threshold prefixes shared by at most this many listed keys rank them in LDS.
_MULTI_CTA_TIE_CAP = 64
# BF16 prefill query rows per program, by head count; 64 KiB LDS fits two per CU.
_BF16_PREFILL_ROWS = {32: 8, 64: 4}
_BF16_PREFILL_BLOCK_N = 64
_BF16_PREFILL_CHUNK_N = 4096
_BF16_PREFILL_NUM_WARPS = 4
_BF16_PREFILL_WAVES_PER_EU = 2
_BF16_PREFILL_NUM_STAGES = 3
_BF16_DECODE_BLOCK_N = 64
# Short chunks keep enough programs in flight for one or a few requests.
_BF16_DECODE_CHUNK_N = 256
_BF16_DECODE_NUM_STAGES = 3
_BF16_DECODE_NUM_WARPS = 4
# FP8-query scorers: small chunks even the last wave; 4 waves/SIMD suit 32 heads only.
_FP8_ROWS_PREFILL_TILES = {32: (128, 2048)}
_FP8_ROWS_DECODE_NUM_STAGES = 2
_FP8_ROWS_DECODE_WAVES_PER_EU = 4
_FP8_ROWS_DECODE_PROGRAMS_PER_CU = 16

_persistent_topk_workspace_cache: OrderedDict[
    tuple[str, int, int, int], tuple[torch.Tensor, ...]
] = OrderedDict()
_persistent_topk_graph_workspace_keys: set[tuple[str, int, int, int]] = set()
_persistent_topk_workspace_lock = Lock()

__all__ = [
    "launch_gluon_dsa_decode_topk_bf16_gfx950",
    "launch_gluon_dsa_decode_topk_fp8_gfx950",
    "launch_gluon_dsa_decode_topk_fp8_rows_gfx950",
    "launch_gluon_dsa_decode_topk_standard_gfx950",
    "gluon_dsa_kpool_prefill_logits_gfx950",
    "gluon_dsa_kpool_prefill_plan_logits_gfx950",
    "gluon_dsa_logical_topk_gfx950",
    "gluon_dsa_row_bounded_logical_topk_gfx950",
    "launch_gluon_dsa_prefill_topk_bf16_gfx950",
    "launch_gluon_dsa_prefill_topk_fp8_gfx950",
    "launch_gluon_dsa_prefill_topk_fp8_rows_gfx950",
    "launch_gluon_dsa_prefill_topk_standard_gfx950",
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
            shared_histogram.store(histogram_zeros)

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
        # Every wave added to the row histogram, but one thread arrives for
        # the whole program, and s_barrier does not wait for memory. So each
        # wave first waits until its no-return atomics reach L2 (vmcnt counts
        # them like stores).
        release_drain()
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
        cumulative = gl.cumsum(group_counts, 0)
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


@gluon.jit(do_not_specialize=("block_table_cols",))
def _dsa_trivial_topk_kernel(
    block_table,
    row_starts,
    row_ends,
    out,
    lens_out,
    out_stride: gl.constexpr,
    block_table_cols,
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
def _multi_cta_decode_row(row, row_ends, q_len_per_req: gl.constexpr):
    req = row // q_len_per_req
    candidate_len = gl.load(row_ends + req).to(gl.int32)
    if q_len_per_req != 1:
        candidate_len = (
            candidate_len - (q_len_per_req - 1) + (row - req * q_len_per_req)
        )
    return req, gl.maximum(candidate_len, 0)


@gluon.jit
def _load_multi_cta_tile(
    row_logits,
    tile,
    tiles,
    candidate_len,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    GRANULE: gl.constexpr,
):
    # Interleave GRANULE-column runs over tiles so crowded keys spread over every CTA.
    gl.static_assert(BLOCK_N % GRANULE == 0, "a tile holds whole granules")
    local = gl.arange(0, BLOCK_N, layout=value_layout)
    offsets = (local // GRANULE) * (tiles * GRANULE) + tile * GRANULE + local % GRANULE
    offsets = gl.max_contiguous(gl.multiple_of(offsets.to(gl.int32), 4), 4)
    # Rows hold whole pages, so rounding up to a 4-vector stays in the row.
    vector_end = gl.cdiv(candidate_len, 4) * 4
    values = gl.amd.cdna4.buffer_load(
        ptr=row_logits,
        offsets=offsets,
        mask=gl.max_constancy(offsets < vector_end, 4),
        other=-float("inf"),
    )
    return offsets, _fp32_to_topk_key(values), offsets < candidate_len


@gluon.jit
def _multi_cta_tiles(
    candidate_len,
    BLOCK_N: gl.constexpr,
    GRANULE: gl.constexpr,
    TILE_GROUP: gl.constexpr,
):
    # Interleave over the tiles the row fills (at least a group), not the logits width.
    tiles = gl.minimum(
        gl.maximum(gl.cdiv(candidate_len, BLOCK_N), TILE_GROUP), gl.num_programs(1)
    )
    return tiles, gl.minimum(gl.cdiv(candidate_len, GRANULE), tiles)


@gluon.jit
def _physical_slots(block_table, req, block_table_cols, logical, mask, page_size):
    # Buffer ops mask by offset; masked global ops branch and wait on earlier accesses.
    block_idx = logical // page_size
    page = gl.amd.cdna4.buffer_load(
        ptr=block_table + req * block_table_cols,
        offsets=block_idx,
        mask=mask & (block_idx < block_table_cols),
        other=0,
    ).to(gl.int32)
    return page * page_size + (logical - block_idx * page_size)


# Inclusive int32 scan over a wave's 64 lanes via DPP row shifts and row broadcasts.
_WAVE_SCAN_ASM = gl.constexpr(
    "s_nop 1\n"
    "v_mov_b32 $0, $1\n"
    "s_nop 1\n"
    "v_add_u32_dpp $0, $0, $0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
    "s_nop 1\n"
    "v_add_u32_dpp $0, $0, $0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
    "s_nop 1\n"
    "v_add_u32_dpp $0, $0, $0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
    "s_nop 1\n"
    "v_add_u32_dpp $0, $0, $0 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
    "s_nop 1\n"
    "v_add_u32_dpp $0, $0, $0 row_bcast:15 row_mask:0xa bank_mask:0xf\n"
    "s_nop 1\n"
    "v_add_u32_dpp $0, $0, $0 row_bcast:31 row_mask:0xc bank_mask:0xf\n"
    "s_nop 1"
)


@gluon.jit
def _wave_inclusive_scan(x):
    # Every wave scans its own lanes; all 64 lanes must be active.
    return gl.inline_asm_elementwise(
        _WAVE_SCAN_ASM, "=&v,v", [x], dtype=gl.int32, is_pure=True, pack=1
    )


@gluon.constexpr_function
def _wave_bins_layout(num_warps, bins_per_lane):
    # Lane i holds bins [i * bins_per_lane, ...); every wave holds a full copy.
    return gl.BlockedLayout([1, bins_per_lane], [64, 1], [num_warps, 1], [1, 0])


@gluon.jit
def _select_bins_in_wave(counts, remaining):
    # Finds the remaining-th key's bin, keys below and in it; each wave on its own copy.
    V: gl.constexpr = counts.shape[1]
    layout: gl.constexpr = counts.type.layout
    lane_total = gl.sum(counts, axis=1)
    lane_below = _wave_inclusive_scan(lane_total) - lane_total
    inclusive = gl.expand_dims(lane_below, 1) + gl.cumsum(counts, 1)
    below = inclusive - counts
    selected = (below < remaining) & (inclusive >= remaining)
    bins = gl.expand_dims(
        gl.arange(0, 64, layout=gl.SliceLayout(1, layout)) * V, 1
    ) + gl.expand_dims(gl.arange(0, V, layout=gl.SliceLayout(0, layout)), 0)
    return (
        gl.sum(gl.sum(gl.where(selected, bins, 0), axis=1), axis=0),
        gl.sum(gl.sum(gl.where(selected, below, 0), axis=1), axis=0),
        gl.sum(gl.sum(gl.where(selected, counts, 0), axis=1), axis=0),
    )


@gluon.jit
def _load_wave_bins(shared_bins, V: gl.constexpr):
    # Every wave loads all 64 * V bins, lane i holding [i * V, (i + 1) * V).
    W: gl.constexpr = gl.num_warps()
    bins = shared_bins.load(gl.BlockedLayout([V], [64], [W], [0]))
    return gl.convert_layout(bins.reshape([64, V]), _wave_bins_layout(W, V))


@gluon.jit
def _select_shared_digit(shared_histogram, remaining, NUM_BUCKETS: gl.constexpr):
    # Threshold bin, keys below and in it: the 64-bin run from run sums, then the bin.
    gl.static_assert(NUM_BUCKETS == 4096, "64 runs of 64 bins")
    W: gl.constexpr = gl.num_warps()
    sum_layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [W, 1], [1, 0])
    run_layout: gl.constexpr = _wave_bins_layout(W, 1)
    counts = shared_histogram.slice(0, NUM_BUCKETS).load(
        gl.BlockedLayout([16], [64], [W], [0])
    )
    run_sums = gl.sum(gl.convert_layout(counts.reshape([64, 64]), sum_layout), axis=1)
    run, run_greater, _ = _select_bins_in_wave(
        gl.expand_dims(gl.convert_layout(run_sums, gl.SliceLayout(1, run_layout)), 1),
        remaining,
    )
    runs = gl.arange(0, 64, layout=gl.SliceLayout(1, run_layout))
    digit, digit_greater, digit_count = _select_bins_in_wave(
        gl.expand_dims(shared_histogram.gather(run * 64 + runs, axis=0), 1),
        remaining - run_greater,
    )
    return run * 64 + digit, run_greater + digit_greater, digit_count


@gluon.jit
def _block_exclusive_ranks(x, V: gl.constexpr):
    # x in BlockedLayout([V], [64], [W], [0]); ranks go (wave, lane, repeat, vector).
    W: gl.constexpr = gl.num_warps()
    R: gl.constexpr = x.shape[0] // (W * 64 * V)
    L4: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1, V], [1, 1, 64, 1], [1, W, 1, 1], [3, 2, 1, 0]
    )
    L3: gl.constexpr = gl.BlockedLayout([1, 1, R * V], [1, 64, 1], [W, 1, 1], [2, 1, 0])
    LWW: gl.constexpr = gl.BlockedLayout([1, W], [1, 64], [W, 1], [1, 0])
    x4 = gl.convert_layout(x.reshape([R, W, 64, V]), L4)
    x3 = gl.convert_layout(gl.permute(x4, [1, 2, 0, 3]).reshape([W, 64, R * V]), L3)
    thread_total = gl.sum(x3, axis=2)
    lane_inclusive = _wave_inclusive_scan(thread_total)
    wave_total = gl.max(lane_inclusive, axis=1)
    totals = gl.convert_layout(wave_total, gl.SliceLayout(0, LWW))
    waves = gl.arange(0, W, layout=gl.SliceLayout(1, LWW))
    lower = gl.arange(0, W, layout=gl.SliceLayout(0, LWW))
    wave_below = gl.sum(
        gl.where(lower[None, :] < waves[:, None], totals[None, :], 0), axis=1
    )
    wave_below = gl.convert_layout(wave_below, wave_total.type.layout)
    thread_below = gl.expand_dims(wave_below, 1) + lane_inclusive - thread_total
    ranks3 = gl.expand_dims(thread_below, 2) + gl.cumsum(x3, 2) - x3
    ranks4 = gl.permute(ranks3.reshape([W, 64, R, V]), [2, 0, 1, 3])
    return (
        gl.convert_layout(ranks4.reshape([x.shape[0]]), x.type.layout),
        gl.sum(totals, axis=0),
    )


@gluon.jit
def _add_histogram_to_global(
    row_histogram, shared_histogram, NUM_BUCKETS: gl.constexpr
):
    # Adds bins, then run sums; lanes take adjacent bins (atomics serialize per line).
    lane_layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    counts = shared_histogram.load(lane_layout)
    gl.atomic_add(
        row_histogram + gl.arange(0, NUM_BUCKETS, layout=lane_layout),
        counts,
        mask=counts != 0,
        sem="relaxed",
        scope="gpu",
    )
    run_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 64], [gl.num_warps(), 1], [1, 0]
    )
    run_sums = gl.sum(
        gl.convert_layout(counts.reshape([NUM_BUCKETS // 64, 64]), run_layout), axis=1
    )
    gl.atomic_add(
        row_histogram
        + NUM_BUCKETS
        + gl.arange(0, NUM_BUCKETS // 64, layout=gl.SliceLayout(1, run_layout)),
        run_sums,
        mask=run_sums != 0,
        sem="relaxed",
        scope="gpu",
    )


@gluon.constexpr_function
def _flat_shared_layout(size):
    return gl.PaddedSharedLayout.with_identity_for([[size, 1]], [size], [0])


@gluon.jit(do_not_specialize=("logits_stride",))
def _dsa_multi_cta_histogram_kernel(
    logits,
    row_ends,
    histograms,
    logits_stride,
    topk: gl.constexpr,
    q_len_per_req: gl.constexpr,
    NUM_BUCKETS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    GRANULE: gl.constexpr,
    TILE_GROUP: gl.constexpr,
):
    # Radix level 1: each CTA counts a tile's 12-bit prefixes into the row histogram.
    row = gl.program_id(0)
    tile = gl.program_id(1)
    _, candidate_len = _multi_cta_decode_row(row, row_ends, q_len_per_req)
    tiles, num_tiles = _multi_cta_tiles(candidate_len, BLOCK_N, GRANULE, TILE_GROUP)
    if (candidate_len <= topk) | (tile >= num_tiles):
        return
    value_layout: gl.constexpr = _vector_layout(
        gl.num_warps(), BLOCK_N // (64 * gl.num_warps())
    )
    histogram_layout: gl.constexpr = _vector_layout(
        gl.num_warps(), NUM_BUCKETS // (64 * gl.num_warps())
    )
    shared_histogram = gl.allocate_shared_memory(
        gl.int32, [NUM_BUCKETS], _flat_shared_layout(NUM_BUCKETS)
    )
    shared_histogram.store(gl.zeros([NUM_BUCKETS], gl.int32, layout=histogram_layout))
    _, keys, valid = _load_multi_cta_tile(
        logits + gl.multiple_of(row * logits_stride, 4),
        tile,
        tiles,
        candidate_len,
        value_layout,
        BLOCK_N,
        GRANULE,
    )
    shared_histogram.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        (keys >> 20).to(gl.int32),
        axis=0,
        mask=valid,
    )
    _add_histogram_to_global(
        histograms + row * (NUM_BUCKETS + 64), shared_histogram, NUM_BUCKETS
    )


@gluon.jit
def _wait_vector_memory():
    gl.inline_asm("s_waitcnt vmcnt(0)", "", [], (), is_pure=False)


@gluon.jit
def _count_multi_cta_digits(
    shared_histogram,
    keys,
    listed,
    prefix,
    PREFIX_SHIFT: gl.constexpr,
    DIGIT_SHIFT: gl.constexpr,
    DIGIT_MASK: gl.constexpr,
    SPILL: gl.constexpr,
):
    # Unmatched lanes hit distinct spill bins, so the LDS atomics need no branches.
    spill = SPILL + gl.arange(0, keys.shape[0], layout=keys.type.layout) % SPILL
    shared_histogram.atomic_scatter_add(
        gl.full(keys.shape, 1, gl.int32, layout=keys.type.layout),
        gl.where(
            listed & ((keys >> PREFIX_SHIFT) == prefix),
            ((keys >> DIGIT_SHIFT) & DIGIT_MASK).to(gl.int32),
            spill,
        ),
        axis=0,
    )


@gluon.jit
def _emit_multi_cta_listed(
    keys,
    slots,
    listed,
    threshold,
    below_start,
    equal_start,
    equal_needed,
    shared_tie_keys,
    shared_tie_slots,
    row_out,
    SHIFT: gl.constexpr,
    TIES: gl.constexpr,
):
    # Below-threshold keys emit at below_start; ties buffer or fill up to equal_needed.
    compared = keys >> SHIFT
    below = listed & (compared < threshold)
    equal = listed & (compared == threshold)
    ranks, totals = _block_exclusive_ranks(
        below.to(gl.int32) | (equal.to(gl.int32) << 16), 4
    )
    below_ranks = ranks & 0xFFFF
    equal_ranks = equal_start + (ranks >> 16)
    if TIES:
        gl.amd.cdna4.buffer_store(slots, row_out, below_start + below_ranks, mask=below)
        shared_tie_keys.atomic_scatter_xchg(keys, equal_ranks, axis=0, mask=equal)
        shared_tie_slots.atomic_scatter_xchg(slots, equal_ranks, axis=0, mask=equal)
    else:
        gl.amd.cdna4.buffer_store(
            slots,
            row_out,
            gl.where(below, below_start + below_ranks, equal_ranks),
            mask=below | (equal & (equal_ranks < equal_needed)),
        )
    return totals & 0xFFFF, totals >> 16


@gluon.jit
def _visit_multi_cta_listed(
    keys,
    slots,
    listed,
    shared_histogram,
    shared_tie_keys,
    shared_tie_slots,
    prefix,
    below_start,
    equal_start,
    equal_needed,
    row_out,
    PHASE: gl.constexpr,
    NUM_BUCKETS: gl.constexpr,
):
    # PHASE 0/1 count digits 2/3; 2 emits on the 32-bit prefix, 3 on 24-bit with ties.
    if PHASE == 0:
        _count_multi_cta_digits(
            shared_histogram, keys, listed, prefix, 20, 8, 0xFFF, NUM_BUCKETS
        )
        below_count = gl.full([], 0, gl.int32)
        equal_count = gl.full([], 0, gl.int32)
    elif PHASE == 1:
        _count_multi_cta_digits(
            shared_histogram, keys, listed, prefix, 8, 0, 0xFF, NUM_BUCKETS
        )
        below_count = gl.full([], 0, gl.int32)
        equal_count = gl.full([], 0, gl.int32)
    else:
        below_count, equal_count = _emit_multi_cta_listed(
            keys,
            slots,
            listed,
            prefix,
            below_start,
            equal_start,
            equal_needed,
            shared_tie_keys,
            shared_tie_slots,
            row_out,
            8 if PHASE == 3 else 0,
            PHASE == 3,
        )
    return below_count, equal_count


@gluon.jit
def _load_listed_group(
    row_counts,
    row_keys,
    row_slots,
    group_start,
    num_tiles,
    gather_layout: gl.constexpr,
    TILE_GROUP: gl.constexpr,
    CHUNK: gl.constexpr,
):
    # Each tile pads to CHUNK slots; adjacent 4-key lane vectors read whole cache lines.
    flat = gl.arange(0, TILE_GROUP * CHUNK, layout=gather_layout)
    tiles = group_start + flat // CHUNK
    positions = flat % CHUNK
    in_row = gl.max_constancy(tiles < num_tiles, 4)
    counts = gl.amd.cdna4.buffer_load(
        ptr=row_counts, offsets=tiles, mask=in_row, other=0, cache=".cv"
    )
    offsets = gl.max_contiguous(gl.multiple_of(tiles * CHUNK + positions, 4), 4)
    keys = gl.amd.cdna4.buffer_load(
        ptr=row_keys, offsets=offsets, mask=in_row, other=0, cache=".cv"
    ).to(gl.uint32, bitcast=True)
    slots = gl.amd.cdna4.buffer_load(
        ptr=row_slots, offsets=offsets, mask=in_row, other=0, cache=".cv"
    )
    return keys, slots, positions < counts


@gluon.jit
def _load_listed_overflow(
    row_overflow_keys,
    row_overflow_slots,
    start,
    overflow_count,
    gather_layout: gl.constexpr,
    SIZE: gl.constexpr,
):
    offsets = gl.max_contiguous(
        gl.multiple_of(start + gl.arange(0, SIZE, layout=gather_layout), 4), 4
    )
    listed = offsets < overflow_count
    keys = gl.amd.cdna4.buffer_load(
        ptr=row_overflow_keys, offsets=offsets, mask=listed, other=0, cache=".cv"
    ).to(gl.uint32, bitcast=True)
    slots = gl.amd.cdna4.buffer_load(
        ptr=row_overflow_slots, offsets=offsets, mask=listed, other=0, cache=".cv"
    )
    return keys, slots, listed


@gluon.jit
def _load_spilled_listed(
    source,
    extra_groups,
    row_counts,
    row_keys,
    row_slots,
    row_overflow_keys,
    row_overflow_slots,
    num_tiles,
    overflow_count,
    gather_layout: gl.constexpr,
    TILE_GROUP: gl.constexpr,
    CHUNK: gl.constexpr,
):
    # Source i: tile group i + 1 below extra_groups, then overflow chunks, then nothing.
    if source < extra_groups:
        keys, slots, listed = _load_listed_group(
            row_counts,
            row_keys,
            row_slots,
            (source + 1) * TILE_GROUP,
            num_tiles,
            gather_layout,
            TILE_GROUP,
            CHUNK,
        )
    else:
        keys, slots, listed = _load_listed_overflow(
            row_overflow_keys,
            row_overflow_slots,
            (source - extra_groups) * (TILE_GROUP * CHUNK),
            overflow_count,
            gather_layout,
            TILE_GROUP * CHUNK,
        )
    return keys, slots, listed


@gluon.jit
def _visit_spilled_listed(
    row_counts,
    row_keys,
    row_slots,
    row_overflow_keys,
    row_overflow_slots,
    num_tiles,
    overflow_count,
    shared_histogram,
    shared_tie_keys,
    shared_tie_slots,
    prefix,
    below_start,
    equal_start,
    equal_needed,
    row_out,
    gather_layout: gl.constexpr,
    PHASE: gl.constexpr,
    TILE_GROUP: gl.constexpr,
    CHUNK: gl.constexpr,
    NUM_BUCKETS: gl.constexpr,
):
    # Later groups then overflow, each source loading while the previous one is visited.
    extra_groups = gl.cdiv(num_tiles, TILE_GROUP) - 1
    sources = extra_groups + gl.cdiv(overflow_count, TILE_GROUP * CHUNK)
    below_start += gl.full([], 0, gl.int32)
    equal_start += gl.full([], 0, gl.int32)
    if sources > 0:
        keys, slots, listed = _load_spilled_listed(
            0,
            extra_groups,
            row_counts,
            row_keys,
            row_slots,
            row_overflow_keys,
            row_overflow_slots,
            num_tiles,
            overflow_count,
            gather_layout,
            TILE_GROUP,
            CHUNK,
        )
        for source in range(sources):
            next_keys, next_slots, next_listed = _load_spilled_listed(
                source + 1,
                extra_groups,
                row_counts,
                row_keys,
                row_slots,
                row_overflow_keys,
                row_overflow_slots,
                num_tiles,
                overflow_count,
                gather_layout,
                TILE_GROUP,
                CHUNK,
            )
            below_count, equal_count = _visit_multi_cta_listed(
                keys,
                slots,
                listed,
                shared_histogram,
                shared_tie_keys,
                shared_tie_slots,
                prefix,
                below_start,
                equal_start,
                equal_needed,
                row_out,
                PHASE,
                NUM_BUCKETS,
            )
            below_start += below_count
            equal_start += equal_count
            keys = next_keys
            slots = next_slots
            listed = next_listed


@gluon.jit
def _rank_multi_cta_ties(
    shared_tie_keys,
    shared_tie_slots,
    tie_count,
    out_start,
    needed,
    row_out,
    TIE_CAP: gl.constexpr,
):
    # Rank tied keys by key, then buffer position, and emit the `needed` lowest.
    W: gl.constexpr = gl.num_warps()
    gl.static_assert(TIE_CAP == 64, "the tie ranking layout covers 64 keys")
    tie_layout: gl.constexpr = gl.BlockedLayout(
        [1, triton.cdiv(64, 4 * (W // 4))], [16, 4], [4, W // 4], [1, 0]
    )
    own = gl.arange(0, TIE_CAP, layout=gl.SliceLayout(1, tie_layout))
    other = gl.arange(0, TIE_CAP, layout=gl.SliceLayout(0, tie_layout))
    own_keys = shared_tie_keys.load(gl.SliceLayout(1, tie_layout))
    other_keys = shared_tie_keys.load(gl.SliceLayout(0, tie_layout))
    ranked_before = (other[None, :] < tie_count) & (
        (other_keys[None, :] < own_keys[:, None])
        | ((other_keys[None, :] == own_keys[:, None]) & (other[None, :] < own[:, None]))
    )
    rank = gl.sum(ranked_before.to(gl.int32), axis=1)
    gl.amd.cdna4.buffer_store(
        shared_tie_slots.load(gl.SliceLayout(1, tie_layout)),
        row_out,
        out_start + rank,
        mask=(own < tie_count) & (rank < needed),
    )


@gluon.jit(
    do_not_specialize=(
        "logits_stride",
        "candidate_stride",
        "overflow_stride",
        "counts_stride",
        "block_table_cols",
    ),
)
def _dsa_multi_cta_select_kernel(
    logits,
    block_table,
    row_ends,
    histograms,
    arrivals,
    tile_counts,
    candidate_keys,
    candidate_slots,
    overflow_keys,
    overflow_slots,
    out,
    lens_out,
    logits_stride,
    candidate_stride,
    overflow_stride,
    counts_stride,
    out_stride: gl.constexpr,
    block_table_cols,
    page_size: gl.constexpr,
    topk: gl.constexpr,
    q_len_per_req: gl.constexpr,
    NUM_BUCKETS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    GRANULE: gl.constexpr,
    ARRIVAL_STRIDE: gl.constexpr,
    TILE_GROUP: gl.constexpr,
    CHUNK: gl.constexpr,
    TIE_CAP: gl.constexpr,
):
    # Level 2: CTAs list bucket keys; the last to arrive resolves and re-zeroes the row.
    gl.static_assert(
        (BLOCK_N < 65536) & (TILE_GROUP * CHUNK < 65536),
        "the rank scans pack two 16-bit counts",
    )
    gl.static_assert(topk < 4096, "an arrival packs the winner count above 12 bits")
    row = gl.program_id(0)
    tile = gl.program_id(1)
    req, candidate_len = _multi_cta_decode_row(row, row_ends, q_len_per_req)
    if tile == 0:
        gl.store(lens_out + row, gl.minimum(candidate_len, topk))
    output_layout: gl.constexpr = _vector_layout(
        gl.num_warps(), triton.cdiv(topk, 64 * gl.num_warps())
    )
    row_out = out + row * out_stride
    if candidate_len <= topk:
        if tile == 0:
            output_offsets = gl.arange(0, topk, layout=output_layout)
            valid = output_offsets < candidate_len
            gl.store(
                row_out + output_offsets,
                gl.where(
                    valid,
                    _physical_slots(
                        block_table,
                        req,
                        block_table_cols,
                        output_offsets,
                        valid,
                        page_size,
                    ),
                    -1,
                ),
            )
        return
    tiles, num_tiles = _multi_cta_tiles(candidate_len, BLOCK_N, GRANULE, TILE_GROUP)
    if tile >= num_tiles:
        return

    W: gl.constexpr = gl.num_warps()
    value_layout: gl.constexpr = _vector_layout(W, BLOCK_N // (64 * W))
    histogram_layout: gl.constexpr = _vector_layout(W, NUM_BUCKETS // (64 * W))
    counter_layout: gl.constexpr = _vector_layout(W, 1)
    run_layout: gl.constexpr = _wave_bins_layout(W, 1)

    row_histogram = histograms + row * (NUM_BUCKETS + 64)
    runs = gl.expand_dims(gl.arange(0, 64, layout=gl.SliceLayout(1, run_layout)), 1)
    row_arrival = arrivals + row * ARRIVAL_STRIDE
    row_overflow_cursor = row_arrival + ARRIVAL_STRIDE // 2
    # The listed-key rows hold whole chunks, so they start 16-byte aligned.
    row_keys = candidate_keys + gl.multiple_of(row * candidate_stride, 4)
    row_slots = candidate_slots + gl.multiple_of(row * candidate_stride, 4)
    row_overflow_keys = overflow_keys + gl.multiple_of(row * overflow_stride, 4)
    row_overflow_slots = overflow_slots + gl.multiple_of(row * overflow_stride, 4)
    row_counts = tile_counts + row * counts_stride

    offsets, keys, valid = _load_multi_cta_tile(
        logits + gl.multiple_of(row * logits_stride, 4),
        tile,
        tiles,
        candidate_len,
        value_layout,
        BLOCK_N,
        GRANULE,
    )
    slots = _physical_slots(
        block_table, req, block_table_cols, offsets, valid, page_size
    )
    # Find the threshold's 64-bin run from the run sums, then the bin within it.
    run, run_greater, _ = _select_bins_in_wave(
        gl.load(row_histogram + NUM_BUCKETS + runs), topk
    )
    bucket, greater, _ = _select_bins_in_wave(
        gl.load(row_histogram + run * 64 + runs), topk - run_greater
    )
    bucket += run * 64
    greater += run_greater
    high = keys >> 20
    winner = valid & (high < bucket.to(gl.uint32))
    listed = valid & (high == bucket.to(gl.uint32))
    # Ranks of the tile's winners (low half) and listed keys (high half).
    ranks, totals = _block_exclusive_ranks(
        winner.to(gl.int32) | (listed.to(gl.int32) << 16), BLOCK_N // (64 * W)
    )
    listed_ranks = ranks >> 16
    listed_count = totals >> 16
    in_chunk = listed & (listed_ranks < CHUNK)
    gl.amd.cdna4.buffer_store(
        keys.to(gl.int32, bitcast=True),
        row_keys + tile * CHUNK,
        listed_ranks,
        mask=in_chunk,
        cache=".wt",
    )
    gl.amd.cdna4.buffer_store(
        slots, row_slots + tile * CHUNK, listed_ranks, mask=in_chunk, cache=".wt"
    )
    if listed_count > CHUNK:
        overflow_start = (
            gl.atomic_add(
                row_overflow_cursor, listed_count - CHUNK, sem="relaxed", scope="gpu"
            )
            - CHUNK
        )
        spilled = listed & (listed_ranks >= CHUNK)
        gl.amd.cdna4.buffer_store(
            keys.to(gl.int32, bitcast=True),
            row_overflow_keys,
            overflow_start + listed_ranks,
            mask=spilled,
            cache=".wt",
        )
        gl.amd.cdna4.buffer_store(
            slots,
            row_overflow_slots,
            overflow_start + listed_ranks,
            mask=spilled,
            cache=".wt",
        )
    gl.store(row_counts + tile, gl.minimum(listed_count, CHUNK), cache_modifier=".wt")
    # Write-through stores: arrival needs no writeback, only every wave's stores done.
    _wait_vector_memory()
    gl.barrier()
    arrived = gl.atomic_add(
        row_arrival, 1 + ((totals & 0xFFFF) << 12), sem="relaxed", scope="gpu"
    )
    gl.amd.cdna4.buffer_store(
        slots, row_out, (arrived >> 12) + (ranks & 0xFFFF), mask=winner
    )
    if (arrived & 0xFFF) != num_tiles - 1:
        return

    # Others have read the histogram and arrived; listed keys load past the caches.
    gl.atomic_xchg(row_arrival, 0, sem="relaxed", scope="gpu")
    overflow_count = gl.atomic_xchg(row_overflow_cursor, 0, sem="relaxed", scope="gpu")
    bucket_offsets = gl.arange(0, NUM_BUCKETS, layout=histogram_layout)
    gl.store(row_histogram + bucket_offsets, gl.zeros_like(bucket_offsets))
    gl.store(row_histogram + NUM_BUCKETS + runs, gl.zeros_like(runs))
    shared_histogram = gl.allocate_shared_memory(
        gl.int32, [2 * NUM_BUCKETS], _flat_shared_layout(2 * NUM_BUCKETS)
    )
    shared_tie_keys = gl.allocate_shared_memory(
        gl.uint32, [TIE_CAP], _flat_shared_layout(TIE_CAP)
    )
    shared_tie_slots = gl.allocate_shared_memory(
        gl.int32, [TIE_CAP], _flat_shared_layout(TIE_CAP)
    )
    shared_histogram.slice(0, NUM_BUCKETS).store(
        gl.zeros([NUM_BUCKETS], gl.int32, layout=histogram_layout)
    )
    gather_layout: gl.constexpr = gl.BlockedLayout([4], [64], [W], [0])
    remaining = topk - greater
    # The first tile group's keys stay in registers; the rest stream once per level.
    gathered_keys, gathered_slots, gathered = _load_listed_group(
        row_counts, row_keys, row_slots, 0, num_tiles, gather_layout, TILE_GROUP, CHUNK
    )
    spilled = (num_tiles > TILE_GROUP) | (overflow_count > 0)
    bucket_prefix = bucket.to(gl.uint32)
    _count_multi_cta_digits(
        shared_histogram,
        gathered_keys,
        gathered,
        bucket_prefix,
        20,
        8,
        0xFFF,
        NUM_BUCKETS,
    )
    _visit_spilled_listed(
        row_counts,
        row_keys,
        row_slots,
        row_overflow_keys,
        row_overflow_slots,
        num_tiles,
        overflow_count,
        shared_histogram,
        shared_tie_keys,
        shared_tie_slots,
        bucket_prefix,
        0,
        0,
        0,
        row_out,
        gather_layout,
        0,
        TILE_GROUP,
        CHUNK,
        NUM_BUCKETS,
    )
    digit, digit_greater, tie_count = _select_shared_digit(
        shared_histogram, remaining, NUM_BUCKETS
    )
    prefix = (bucket_prefix << 12) | digit.to(gl.uint32)
    remaining -= digit_greater
    if tie_count <= TIE_CAP:
        # Few ties: buffer them while emitting the keys below, then rank them in LDS.
        below_count, equal_count = _emit_multi_cta_listed(
            gathered_keys,
            gathered_slots,
            gathered,
            prefix,
            greater,
            0,
            0,
            shared_tie_keys,
            shared_tie_slots,
            row_out,
            8,
            True,
        )
        if spilled:
            _visit_spilled_listed(
                row_counts,
                row_keys,
                row_slots,
                row_overflow_keys,
                row_overflow_slots,
                num_tiles,
                overflow_count,
                shared_histogram,
                shared_tie_keys,
                shared_tie_slots,
                prefix,
                greater + below_count,
                equal_count,
                0,
                row_out,
                gather_layout,
                3,
                TILE_GROUP,
                CHUNK,
                NUM_BUCKETS,
            )
        _rank_multi_cta_ties(
            shared_tie_keys,
            shared_tie_slots,
            tie_count,
            topk - remaining,
            remaining,
            row_out,
            TIE_CAP,
        )
    else:
        shared_histogram.slice(0, 256).store(
            gl.zeros([256], gl.int32, layout=counter_layout)
        )
        _count_multi_cta_digits(
            shared_histogram, gathered_keys, gathered, prefix, 8, 0, 0xFF, NUM_BUCKETS
        )
        _visit_spilled_listed(
            row_counts,
            row_keys,
            row_slots,
            row_overflow_keys,
            row_overflow_slots,
            num_tiles,
            overflow_count,
            shared_histogram,
            shared_tie_keys,
            shared_tie_slots,
            prefix,
            0,
            0,
            0,
            row_out,
            gather_layout,
            1,
            TILE_GROUP,
            CHUNK,
            NUM_BUCKETS,
        )
        last, last_greater, _ = _select_bins_in_wave(
            _load_wave_bins(shared_histogram.slice(0, 256), 4), remaining
        )
        threshold = (prefix << 8) | last.to(gl.uint32)
        remaining -= last_greater
        # Keys under the 32-bit threshold win; ties fill the last `remaining` outputs.
        below_count, equal_count = _emit_multi_cta_listed(
            gathered_keys,
            gathered_slots,
            gathered,
            threshold,
            greater,
            topk - remaining,
            topk,
            shared_tie_keys,
            shared_tie_slots,
            row_out,
            0,
            False,
        )
        if spilled:
            _visit_spilled_listed(
                row_counts,
                row_keys,
                row_slots,
                row_overflow_keys,
                row_overflow_slots,
                num_tiles,
                overflow_count,
                shared_histogram,
                shared_tie_keys,
                shared_tie_slots,
                threshold,
                greater + below_count,
                topk - remaining + equal_count,
                topk,
                row_out,
                gather_layout,
                2,
                TILE_GROUP,
                CHUNK,
                NUM_BUCKETS,
            )


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
        values = gl.amd.cdna4.buffer_load(
            ptr=candidate_logits,
            offsets=offsets,
            mask=vector_mask,
            other=-float("inf"),
        )
        if vector_end < candidate_len:
            tail_mask = (offsets >= vector_end) & valid
            tail_values = gl.load(
                candidate_logits + offsets,
                mask=tail_mask,
                other=-float("inf"),
            )
            values = gl.where(tail_mask, tail_values, values)
    else:
        valid = gl.full([BLOCK_N], True, gl.int1, layout=value_layout)
        values = gl.amd.cdna4.buffer_load(
            ptr=candidate_logits,
            offsets=offsets,
        )
    return offsets, values, valid


@gluon.jit
def _histogram_slots(
    buckets,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HIST_COPIES: gl.constexpr,
):
    slots = buckets.to(gl.int32)
    if HIST_COPIES > 1:
        # Per-bucket HIST_COPIES counters stop crowded buckets serializing LDS atomics.
        load_elems: gl.constexpr = BLOCK_N // (64 * gl.num_warps())
        lane = gl.arange(0, BLOCK_N, layout=value_layout) // load_elems
        slots = slots * HIST_COPIES + lane % HIST_COPIES
    return slots


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
    HIST_COPIES: gl.constexpr = 1,
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
        _histogram_slots(buckets, value_layout, BLOCK_N, HIST_COPIES),
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
    HIST_COPIES: gl.constexpr = 1,
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
    first_buckets = _histogram_slots(
        (first_keys >> shift) & bucket_mask, value_layout, BLOCK_N, HIST_COPIES
    )
    second_buckets = _histogram_slots(
        (second_keys >> shift) & bucket_mask, value_layout, BLOCK_N, HIST_COPIES
    )
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
def _emit_oneblock_ties_tile(
    candidate_logits,
    tile_start,
    candidate_len,
    vector_end,
    candidate_start,
    prefix,
    shared_output_counters,
    shared_tie_keys,
    shared_tie_offsets,
    out,
    row,
    out_stride: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PREFIX_SHIFT: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    # _emit_oneblock_topk_tile, but threshold-prefix ties go to the LDS tie buffers.
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
    compared_keys = keys >> PREFIX_SHIFT
    greater_mask = valid & (compared_keys < prefix)
    equal_mask = valid & (compared_keys == prefix)
    reservation = shared_output_counters.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        gl.where(greater_mask, 0, 1).to(gl.int32),
        axis=0,
        mask=greater_mask | equal_mask,
    )
    logical_offsets = candidate_start + offsets.to(gl.int32)
    gl.store(
        out + row * out_stride + reservation,
        logical_offsets,
        mask=greater_mask,
    )
    shared_tie_keys.atomic_scatter_xchg(keys, reservation, axis=0, mask=equal_mask)
    shared_tie_offsets.atomic_scatter_xchg(
        logical_offsets, reservation, axis=0, mask=equal_mask
    )


@gluon.jit
def _resolve_oneblock_ties(
    shared_tie_keys,
    shared_tie_offsets,
    tie_count,
    count_greater,
    remaining,
    out,
    row,
    out_stride: gl.constexpr,
    TIE_CAP: gl.constexpr,
):
    # Rank ties by key, then buffer order; emit `remaining` of them after count_greater.
    gl.static_assert(TIE_CAP == 64, "the tie ranking layout covers 64 candidates")
    tie_layout: gl.constexpr = gl.BlockedLayout(
        [1, triton.cdiv(64, 4 * (gl.num_warps() // 4))],
        [16, 4],
        [4, gl.num_warps() // 4],
        [1, 0],
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, tie_layout)
    col_layout: gl.constexpr = gl.SliceLayout(0, tie_layout)
    own = gl.arange(0, TIE_CAP, layout=row_layout)
    other = gl.arange(0, TIE_CAP, layout=col_layout)
    own_keys = shared_tie_keys.load(row_layout)
    other_keys = shared_tie_keys.load(col_layout)
    ranked_before = (other[None, :] < tie_count) & (
        (other_keys[None, :] < own_keys[:, None])
        | ((other_keys[None, :] == own_keys[:, None]) & (other[None, :] < own[:, None]))
    )
    rank = gl.sum(ranked_before.to(gl.int32), axis=1)
    gl.store(
        out + row * out_stride + count_greater + rank,
        shared_tie_offsets.load(row_layout),
        mask=(own < tie_count) & (rank < remaining),
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


@gluon.jit(
    do_not_specialize=(
        "logits_stride",
        "block_table_cols",
    ),
)
def _dsa_oneblock_manual_radix_topk_kernel(
    logits,
    block_table,
    row_starts,
    row_ends,
    out,
    lens_out,
    logits_stride,
    out_stride: gl.constexpr,
    block_table_cols,
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
    ALIGNED_ROWS: gl.constexpr,
    HIST_COPIES: gl.constexpr = 1,
    TIE_CAP: gl.constexpr = 0,
):
    gl.static_assert(
        HIST_COPIES == 1 or not USE_COMPACT_FINAL,
        "the compact final pass keeps a single histogram copy",
    )
    gl.static_assert(
        TIE_CAP == 0 or not USE_COMPACT_FINAL,
        "the compact final pass resolves its own threshold bucket",
    )
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
    HIST_SLOTS: gl.constexpr = MAX_BUCKETS * HIST_COPIES
    slots_layout: gl.constexpr = _vector_layout(
        gl.num_warps(),
        triton.cdiv(HIST_SLOTS, 64 * gl.num_warps()),
    )
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[HIST_SLOTS, 1]],
        [HIST_SLOTS],
        [0],
    )
    slot_zeros = gl.zeros([HIST_SLOTS], gl.int32, layout=slots_layout)
    histogram_zeros = gl.zeros([MAX_BUCKETS], gl.int32, layout=histogram_layout)
    shared_histogram = gl.allocate_shared_memory(
        gl.int32,
        [HIST_SLOTS],
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
    if TIE_CAP > 0:
        tie_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[TIE_CAP, 1]],
            [TIE_CAP],
            [0],
        )
        shared_tie_keys = gl.allocate_shared_memory(
            gl.uint32,
            [TIE_CAP],
            tie_shared_layout,
        )
        shared_tie_offsets = gl.allocate_shared_memory(
            gl.int32,
            [TIE_CAP],
            tie_shared_layout,
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
        if row_starts is None:
            candidate_start = gl.full([], 0, gl.int32)
        else:
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

    shared_histogram.store(slot_zeros)
    shared_output_counters.store(output_counter_zeros)

    row_offset = row.to(gl.int64) * logits_stride
    if ALIGNED_ROWS:
        # Unspecialized logits_stride hides the 16-byte alignment dwordx4 loads need.
        row_offset = gl.multiple_of(row_offset, 4)
    candidate_logits = logits + row_offset + candidate_start
    if ALIGNED_ROWS:
        # Aligned rows hold whole pages, so rounding up to a 4-vector stays in the row.
        vector_end = gl.cdiv(candidate_len, 4) * 4
    else:
        vector_end = (candidate_len // 4) * 4
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

        # Later passes see one bucket's spread-out digits; only pass one needs copies.
        pass_copies = HIST_COPIES if pass_index == 0 else 1
        if pass_index != 0:
            shared_histogram.slice(0, MAX_BUCKETS).store(histogram_zeros)

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
                        pass_copies,
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
                        pass_copies,
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
                    pass_copies,
                )

        if pass_copies == 1:
            counts = shared_histogram.slice(0, MAX_BUCKETS).load(histogram_layout)
        else:
            copies = shared_histogram.load(slots_layout).reshape(
                [MAX_BUCKETS, HIST_COPIES]
            )
            counts = gl.convert_layout(gl.sum(copies, axis=1), histogram_layout)
        count_pairs = counts.reshape([MAX_BUCKETS // 2, 2])
        count_low, count_high = gl.split(count_pairs)
        count_low = gl.convert_layout(count_low, group_layout)
        count_high = gl.convert_layout(count_high, group_layout)
        group_counts = count_low + count_high
        group_cumulative = gl.cumsum(group_counts, 0)
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
            pass1_complete = ((selected >> 23) & 1) != 0
            if TIE_CAP > 0 and not pass1_complete:
                tie_count = gl.sum(
                    gl.where(bucket_offsets == selected_bucket, counts, 0), axis=0
                )
                if tie_count <= TIE_CAP:
                    # Few ties: rank them in LDS instead of a third pass over the row.
                    count_greater = topk - remaining
                    emit_full_end = candidate_len & -BLOCK_N
                    for tile_start in range(0, emit_full_end, BLOCK_N):
                        _emit_oneblock_ties_tile(
                            candidate_logits,
                            tile_start,
                            candidate_len,
                            vector_end,
                            candidate_start,
                            prefix,
                            shared_output_counters,
                            shared_tie_keys,
                            shared_tie_offsets,
                            out,
                            row,
                            out_stride,
                            value_layout,
                            BLOCK_N,
                            shift,
                            False,
                        )
                    if emit_full_end < candidate_len:
                        _emit_oneblock_ties_tile(
                            candidate_logits,
                            emit_full_end,
                            candidate_len,
                            vector_end,
                            candidate_start,
                            prefix,
                            shared_output_counters,
                            shared_tie_keys,
                            shared_tie_offsets,
                            out,
                            row,
                            out_stride,
                            value_layout,
                            BLOCK_N,
                            shift,
                            True,
                        )
                    _resolve_oneblock_ties(
                        shared_tie_keys,
                        shared_tie_offsets,
                        tie_count,
                        count_greater,
                        remaining,
                        out,
                        row,
                        out_stride,
                        TIE_CAP,
                    )
                    # Other threads wrote `out`; the compiler orders only LDS hazards.
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
            if pass1_complete:
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
                    # ``out`` was just written by other threads; the
                    # compiler only orders shared-memory hazards.
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
        # ``out`` was just written by other threads; the compiler only
        # orders shared-memory hazards.
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
    if not q.is_contiguous() or not weights.is_contiguous():
        raise ValueError("q and weights must be contiguous")
    if index_k_cache.stride(-1) != 1:
        raise ValueError("index_k_cache pages must contain contiguous bytes")


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


def _whole_page_rows(logits: torch.Tensor, page_size: int) -> bool:
    # Aligned whole-page rows let tiles load dwordx4 up to the rounded-up row length.
    return (
        page_size % 4 == 0
        and logits.shape[1] % 4 == 0
        and logits.stride(0) % 4 == 0
        and logits.data_ptr() % 16 == 0
    )


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

    if (
        is_decode
        and rows <= _MULTI_CTA_MAX_ROWS
        and cols > _MULTI_CTA_ONEBLOCK_MAX_COLS
    ):
        return _TopKLaunchPlan(
            "multi-cta",
            block_n=(
                _MULTI_CTA_WIDE_BLOCK_N
                if rows >= _MULTI_CTA_WIDE_MIN_ROWS
                else _MULTI_CTA_BLOCK_N
            ),
        )

    if cols <= _ONEBLOCK_RADIX_MAX_COLS:
        return _oneblock_launch_plan(cols, is_decode=is_decode)

    # One-block decode cost follows context, the persistent kernel's the logits width.
    if (
        topk == _TUNED_PERSISTENT_TOPK
        and (is_decode or cols <= _TOPK2048_MIDRANGE_MAX_COLS)
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


def _allocate_persistent_topk_workspace(
    row_bucket: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return (
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


def _allocate_multi_cta_topk_workspace(
    row_bucket: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    # Per row: histogram + run sums, then arrival and overflow counters on own lines.
    return (
        torch.zeros(
            (row_bucket, _MULTI_CTA_NUM_BUCKETS + 64), dtype=torch.int32, device=device
        ),
        torch.zeros(
            (row_bucket, _MULTI_CTA_ARRIVAL_STRIDE), dtype=torch.int32, device=device
        ),
    )


def _persistent_topk_workspace(
    rows: int,
    device: torch.device,
    allocate=_allocate_persistent_topk_workspace,
) -> tuple[torch.Tensor, ...]:
    """Return zero-once scratch owned by the current stream and size bucket."""
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream_id = int(torch.cuda.current_stream(device_index).cuda_stream)
    row_bucket = _next_power_of_two(rows)
    key = (allocate.__name__, device_index, stream_id, row_bucket)
    capturing = torch.cuda.is_current_stream_capturing()
    with _persistent_topk_workspace_lock:
        workspace = _persistent_topk_workspace_cache.get(key)
        if workspace is not None:
            _persistent_topk_workspace_cache.move_to_end(key)
            if capturing:
                _persistent_topk_graph_workspace_keys.add(key)
    if workspace is not None:
        return workspace

    # Per-stream caching keeps graph pointers alive; the kernel re-zeroes what it uses.
    workspace = allocate(row_bucket, device)
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
        **extern_compile_options(),
    )
    return out, lens_out


def _dsa_multi_cta_topk(
    logits: torch.Tensor,
    block_table: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    q_len_per_req: int,
    block_n: int,
    out: torch.Tensor,
    lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = logits.shape
    histograms, arrivals = _persistent_topk_workspace(
        rows, logits.device, _allocate_multi_cta_topk_workspace
    )
    tiles = triton.cdiv(cols, block_n)
    tile_group = _MULTI_CTA_GATHER_COLS // block_n
    chunk = _MULTI_CTA_GATHER // tile_group
    candidate_keys = torch.empty(
        (rows, tiles * chunk), dtype=torch.int32, device=logits.device
    )
    candidate_slots = torch.empty_like(candidate_keys)
    overflow_keys = torch.empty(
        (rows, triton.cdiv(cols, 4) * 4), dtype=torch.int32, device=logits.device
    )
    overflow_slots = torch.empty_like(overflow_keys)
    tile_counts = torch.empty((rows, tiles), dtype=torch.int32, device=logits.device)
    grid = (rows, tiles)
    _dsa_multi_cta_histogram_kernel[grid](
        logits,
        row_ends,
        histograms,
        logits.stride(0),
        topk=topk,
        q_len_per_req=q_len_per_req,
        NUM_BUCKETS=_MULTI_CTA_NUM_BUCKETS,
        BLOCK_N=block_n,
        GRANULE=_MULTI_CTA_GRANULE,
        TILE_GROUP=tile_group,
        num_warps=_MULTI_CTA_NUM_WARPS,
    )
    _dsa_multi_cta_select_kernel[grid](
        logits,
        block_table,
        row_ends,
        histograms,
        arrivals,
        tile_counts,
        candidate_keys,
        candidate_slots,
        overflow_keys,
        overflow_slots,
        out,
        lens_out,
        logits.stride(0),
        candidate_keys.stride(0),
        overflow_keys.stride(0),
        tile_counts.stride(0),
        out.stride(0),
        block_table.shape[1],
        page_size=int(page_size),
        topk=topk,
        q_len_per_req=q_len_per_req,
        NUM_BUCKETS=_MULTI_CTA_NUM_BUCKETS,
        BLOCK_N=block_n,
        GRANULE=_MULTI_CTA_GRANULE,
        ARRIVAL_STRIDE=_MULTI_CTA_ARRIVAL_STRIDE,
        TILE_GROUP=tile_group,
        CHUNK=chunk,
        TIE_CAP=_MULTI_CTA_TIE_CAP,
        num_warps=_MULTI_CTA_NUM_WARPS,
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
    # Fixed per decode call site; prefill rows start at runtime row_starts, unaligned.
    aligned_rows = (
        is_decode
        and page_size % 4 == 0
        and logits.stride(0) % 4 == 0
        and logits.data_ptr() % 16 == 0
    )
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
        ALIGNED_ROWS=aligned_rows,
        HIST_COPIES=(
            _ONEBLOCK_DECODE_HIST_COPIES
            if is_decode and not plan.use_compact_final
            else 1
        ),
        TIE_CAP=(
            _ONEBLOCK_DECODE_TIE_CAP if is_decode and not plan.use_compact_final else 0
        ),
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
    if plan.kind == "multi-cta" and not _whole_page_rows(logits, page_size):
        plan = _oneblock_launch_plan(cols, is_decode=is_decode)
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

    if plan.kind == "multi-cta":
        return _dsa_multi_cta_topk(
            logits,
            block_table,
            row_ends,
            page_size=page_size,
            topk=topk,
            q_len_per_req=q_len_per_req,
            block_n=plan.block_n,
            out=out,
            lens_out=lens_out,
        )

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


def gluon_dsa_logical_topk_gfx950(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the largest values from independent logical FP32 rows.

    ``row_starts`` and ``row_ends`` describe an inclusive/exclusive candidate
    range in every logits row. Returned indices are logical columns in
    ``logits``; this entry point never performs DSA page-table remapping.
    Callers must provide bounds satisfying ``0 <= start <= end <= cols``.

    Args:
        logits: Contiguous CUDA FP32 scores shaped ``[rows, cols]``.
        row_starts: Contiguous CUDA int32 inclusive bounds shaped ``[rows]``.
        row_ends: Contiguous CUDA int32 exclusive bounds shaped ``[rows]``.
        topk: Number of candidates to select per row.
        out: Optional contiguous CUDA int32 destination shaped ``[rows, topk]``.
        lens_out: Optional contiguous CUDA int32 destination shaped ``[rows]``.

    Returns:
        Logical column indices padded with ``-1`` and valid counts per row.
    """
    topk = int(topk)
    out, lens_out = _check_logical_topk_args(
        logits, row_starts, row_ends, topk=topk, out=out, lens_out=lens_out
    )
    if logits.shape[0] == 0:
        return out, lens_out
    return _dsa_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )


def gluon_dsa_row_bounded_logical_topk_gfx950(
    logits: torch.Tensor,
    row_starts: torch.Tensor | None,
    row_ends: torch.Tensor,
    *,
    topk: int,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`gluon_dsa_logical_topk_gfx950` for rows whose ranges are far
    narrower than ``logits``, such as KVP ranks' owned candidates in a
    max-model-len-wide buffer.

    One workgroup selects each row over its own range, so the cost follows
    the rows' lengths, not the buffer's width; the width-planned kernels
    scan the whole width (MI350X, 64 rows over 1M columns: 25 us at 6K-token
    rows against 86 us, 78 against 99 at 128K). Past about 200K-token rows
    the persistent kernel's split rows are faster.

    Args and returns as :func:`gluon_dsa_logical_topk_gfx950`, except that
    ``row_starts`` may be None: every row then starts at column 0, without
    a launch filling zeros to say so.
    """
    topk = int(topk)
    out, lens_out = _check_logical_topk_args(
        logits,
        row_ends if row_starts is None else row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )
    rows, cols = logits.shape
    if rows == 0:
        return out, lens_out
    if cols <= topk:
        return _dsa_topk_indices(
            logits,
            torch.zeros_like(row_ends) if row_starts is None else row_starts,
            row_ends,
            topk=topk,
            out=out,
            lens_out=lens_out,
        )
    return _dsa_oneblock_topk_indices(
        logits,
        # No block table outside decode; any int32 pointer stands in.
        row_ends,
        row_starts,
        row_ends,
        block_table_cols=0,
        page_size=1,
        topk=topk,
        q_len_per_req=1,
        is_decode=False,
        plan=_oneblock_launch_plan(cols, is_decode=False),
        out=out,
        lens_out=lens_out,
    )


def _check_logical_topk_args(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    out: torch.Tensor | None,
    lens_out: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    else:
        if out.shape != expected_out_shape:
            raise ValueError(
                f"out must have shape {expected_out_shape}, got {tuple(out.shape)}"
            )
        if out.dtype != torch.int32:
            raise TypeError(f"out must be int32, got {out.dtype}")
        if out.device != logits.device:
            raise ValueError("out must be on the same device as logits")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    expected_lens_shape = (rows,)
    if lens_out is None:
        lens_out = torch.empty(
            expected_lens_shape, dtype=torch.int32, device=logits.device
        )
    else:
        if lens_out.shape != expected_lens_shape:
            raise ValueError(
                "lens_out must have shape "
                f"{expected_lens_shape}, got {tuple(lens_out.shape)}"
            )
        if lens_out.dtype != torch.int32:
            raise TypeError(f"lens_out must be int32, got {lens_out.dtype}")
        if lens_out.device != logits.device:
            raise ValueError("lens_out must be on the same device as logits")
        if not lens_out.is_contiguous():
            raise ValueError("lens_out must be contiguous")
    return out, lens_out


def gluon_dsa_kpool_prefill_logits_gfx950(
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
    ordered_head_fold: bool = False,
    out: torch.Tensor | None = None,
    row_ends_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a local GLM KPool prefill window with the GFX950 Gluon scorer.

    Each output column represents the global pool ``pool_offset + column``.
    Only complete pools visible at each query row are written. ``row_ends``
    returns the exclusive local bound for every row, so a length-aware top-k
    can ignore untouched output capacity.

    Args:
        q: BF16 queries shaped ``[tokens, 32, 128]``.
        pooled_k_cache: Page-major scaled-FP8 pooled keys. Each page stores all
            FP8 key rows followed by one FP32 scale per row. Padded page strides
            are supported.
        weights: BF16 or FP32 signed head weights shaped ``[tokens, 32]``.
        causal_lens: Int32 visible raw-token counts shaped ``[tokens]``.
        req_ids: Int32 request row for each query token.
        index_block_table: Int32 logical-pool to physical-page mapping.
        pool_size: Raw tokens represented by a pooled key. Must be ``4``.
        page_size: Pooled keys per physical page. Must be ``16``.
        pool_offset: First global pool represented by output column zero.
        window_cols: Number of local output columns.
        softmax_scale: Model scale applied after the weighted per-head ReLU
            reduction.
        ordered_head_fold: When true, accumulate the 32 signed head
            contributions from logical head zero through 31 instead of using
            the MFMA layout's tree reduction.
        out: Optional contiguous FP32 destination ``[tokens, window_cols]``.
        row_ends_out: Optional contiguous int32 destination ``[tokens]``.

    Returns:
        ``(logits, row_ends)`` with local-window logits and exclusive valid
        bounds. The default valid scores equal
        ``softmax_scale * sum(weights * relu(query @ pooled_key))``.
    """
    pool_size = int(pool_size)
    page_size = int(page_size)
    pool_offset = int(pool_offset)
    window_cols = int(window_cols)
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
    if weights.shape != q.shape[:2]:
        raise ValueError(
            f"weights must have shape {tuple(q.shape[:2])}, got {tuple(weights.shape)}"
        )
    if weights.stride(-1) != 1:
        raise ValueError("weights must have a contiguous head axis")
    if pool_size != 4 or page_size != 16:
        raise ValueError(
            "KPool Gluon scorer requires pool_size=4 and page_size=16, got "
            f"pool_size={pool_size}, page_size={page_size}"
        )
    if pool_offset < 0:
        raise ValueError(f"pool_offset must be nonnegative, got {pool_offset}")
    if window_cols <= 0:
        raise ValueError(f"window_cols must be positive, got {window_cols}")

    row_bytes = 128 + 4
    compact_page_bytes = page_size * row_bytes
    cache_shape_ok = (
        pooled_k_cache.dim() == 2 and pooled_k_cache.shape[1] == compact_page_bytes
    ) or (
        pooled_k_cache.dim() == 3
        and tuple(pooled_k_cache.shape[1:]) == (page_size, row_bytes)
    )
    if not cache_shape_ok:
        raise ValueError(
            "pooled_k_cache must have shape "
            f"[pages, {compact_page_bytes}] or [pages, {page_size}, {row_bytes}], "
            f"got {tuple(pooled_k_cache.shape)}"
        )
    if pooled_k_cache.dtype != torch.uint8:
        raise TypeError(f"pooled_k_cache must be uint8, got {pooled_k_cache.dtype}")
    cache_page_stride_bytes = int(pooled_k_cache.stride(0))
    packed_within_page = (
        pooled_k_cache.dim() == 2 or pooled_k_cache.stride(1) == row_bytes
    )
    if (
        pooled_k_cache.stride(-1) != 1
        or not packed_within_page
        or cache_page_stride_bytes < compact_page_bytes
        or cache_page_stride_bytes % 4
        or pooled_k_cache.storage_offset() % 4
    ):
        raise ValueError(
            "pooled_k_cache requires byte-contiguous rows and a nonoverlapping, "
            "4-byte-aligned page stride"
        )

    tokens = int(q.shape[0])
    for name, value in (("causal_lens", causal_lens), ("req_ids", req_ids)):
        if value.shape != (tokens,):
            raise ValueError(f"{name} must have shape {(tokens,)}, got {value.shape}")
        if value.dtype != torch.int32:
            raise TypeError(f"{name} must be int32, got {value.dtype}")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if index_block_table.dim() != 2:
        raise ValueError(
            f"index_block_table must be 2-D, got shape={tuple(index_block_table.shape)}"
        )
    if index_block_table.dtype != torch.int32:
        raise TypeError(
            f"index_block_table must be int32, got {index_block_table.dtype}"
        )
    if index_block_table.stride(-1) != 1:
        raise ValueError("index_block_table must have contiguous page columns")
    if tokens and index_block_table.shape[0] == 0:
        raise ValueError("index_block_table must have at least one request row")
    for name, value in (
        ("weights", weights),
        ("pooled_k_cache", pooled_k_cache),
        ("causal_lens", causal_lens),
        ("req_ids", req_ids),
        ("index_block_table", index_block_table),
    ):
        if value.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")

    expected_logits = (tokens, window_cols)
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
    if row_ends_out is None:
        row_ends_out = torch.empty((tokens,), dtype=torch.int32, device=q.device)
    elif (
        row_ends_out.shape != (tokens,)
        or row_ends_out.dtype != torch.int32
        or row_ends_out.device != q.device
        or not row_ends_out.is_contiguous()
    ):
        raise ValueError(
            "row_ends_out must be contiguous int32 on q.device with shape "
            f"{(tokens,)}, got shape={tuple(row_ends_out.shape)}, "
            f"dtype={row_ends_out.dtype}, device={row_ends_out.device}, "
            f"contiguous={row_ends_out.is_contiguous()}"
        )
    if tokens == 0:
        return out, row_ends_out

    max_candidates = int(index_block_table.shape[1]) * page_size
    if max_candidates and pooled_k_cache.shape[0] == 0:
        raise ValueError("a nonempty index_block_table requires pooled cache pages")
    if max_candidates == 0 or pool_offset >= max_candidates:
        row_ends_out.zero_()
        return out, row_ends_out

    block_n = 128
    num_warps = 4
    gluon_kpool_prefill_topk_fp8_gfx950[(tokens, 1)](
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
        BLOCK_N=block_n,
        WINDOW_COLS=window_cols,
        NUM_WARPS=num_warps,
        ORDERED_HEAD_FOLD=ordered_head_fold,
        USE_BUFFER_LOAD=pooled_k_cache.untyped_storage().nbytes() < 2**31,
        USE_BUFFER_STORE=out.nbytes < 2**31,
        num_warps=num_warps,
        waves_per_eu=4,
    )
    return out, row_ends_out


def gluon_dsa_kpool_prefill_plan_logits_gfx950(
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
    ordered_head_fold: bool = False,
    out: torch.Tensor | None = None,
    row_ends_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a KPool window through the current physical-slot prefill plan.

    ``pool_workspace_slots`` stores physical pooled-cache rows in request-local
    logical order. ``row_starts`` and ``row_ends`` select the visible prefix for
    every query row. Output columns remain request-local pool ids beginning at
    ``pool_offset``.
    """
    pool_size = int(pool_size)
    page_size = int(page_size)
    pool_offset = int(pool_offset)
    window_cols = int(window_cols)
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
    if pool_offset < 0:
        raise ValueError(f"pool_offset must be nonnegative, got {pool_offset}")
    if window_cols <= 0:
        raise ValueError(f"window_cols must be positive, got {window_cols}")

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
        ("weights", weights),
        ("pooled_k_cache", pooled_k_cache),
        ("pool_workspace_slots", pool_workspace_slots),
        ("row_starts", row_starts),
        ("row_ends", row_ends),
    ):
        if value.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
    for name, value in (
        ("pool_workspace_slots", pool_workspace_slots),
        ("row_starts", row_starts),
        ("row_ends", row_ends),
    ):
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    expected_logits = (tokens, window_cols)
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
    if row_ends_out is None:
        row_ends_out = torch.empty((tokens,), dtype=torch.int32, device=q.device)
    elif (
        row_ends_out.shape != (tokens,)
        or row_ends_out.dtype != torch.int32
        or row_ends_out.device != q.device
        or not row_ends_out.is_contiguous()
    ):
        raise ValueError("row_ends_out must be contiguous int32 on q.device")
    if tokens == 0:
        return out, row_ends_out
    if pool_workspace_slots.numel() and pooled_k_cache.shape[0] == 0:
        raise ValueError("a nonempty prefill plan requires pooled cache pages")

    block_n = 128
    num_warps = 4
    gluon_kpool_prefill_topk_fp8_plan_gfx950[(tokens, 1)](
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
        BLOCK_N=block_n,
        WINDOW_COLS=window_cols,
        NUM_WARPS=num_warps,
        ORDERED_HEAD_FOLD=ordered_head_fold,
        USE_BUFFER_LOAD=pooled_k_cache.untyped_storage().nbytes() < 2**31,
        USE_BUFFER_STORE=out.nbytes < 2**31,
        num_warps=num_warps,
        waves_per_eu=4,
    )
    return out, row_ends_out


def launch_gluon_dsa_decode_topk_fp8_gfx950(
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
        raise RuntimeError("Gluon DSA paged top-k requires an FP8 index_k_cache")
    _, page_stride_bytes = _check_packed_fp8_inputs(
        q, index_k_cache, weights, int(page_size)
    )
    if not weights.is_contiguous():
        weights = weights.contiguous()
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
    score_grid = (q.shape[0], triton.cdiv(max_seq_len, block_n))
    gluon_dsa_decode_topk_fp8_gfx950[score_grid](
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
        page_stride_bytes=page_stride_bytes,
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


def launch_gluon_dsa_prefill_topk_fp8_gfx950(
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
            "Gluon DSA top-k requires an FP8 index_k_cache and page_size"
        )
    _, page_stride_bytes = _check_packed_fp8_inputs(
        q, index_k_cache, weights, int(page_size)
    )
    if not weights.is_contiguous():
        weights = weights.contiguous()
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
        gluon_dsa_prefill_topk_fp8_gfx950[
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
            page_stride_bytes=page_stride_bytes,
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


def _check_standard_decode_metadata(
    q: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    q_len_per_req: int,
) -> None:
    if seq_lens.dtype != torch.int32 or block_table.dtype != torch.int32:
        raise TypeError("seq_lens and block_table must be int32")
    if not seq_lens.is_contiguous() or not block_table.is_contiguous():
        raise ValueError("seq_lens and block_table must be contiguous")
    if seq_lens.device != q.device or block_table.device != q.device:
        raise ValueError("decode metadata must be on the same device as q")
    if q.shape[0] != seq_lens.numel() * q_len_per_req:
        raise ValueError("q rows must equal seq_lens rows times q_len_per_req")
    if block_table.dim() != 2 or block_table.shape[0] < seq_lens.numel():
        raise ValueError("block_table must have at least one row per request")


def _check_standard_prefill_metadata(
    q: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
) -> None:
    if (
        kv_workspace_slots.dtype != torch.int64
        or row_starts.dtype != torch.int32
        or row_ends.dtype != torch.int32
    ):
        raise TypeError("workspace slots must be int64 and row bounds must be int32")
    if (
        kv_workspace_slots.shape != (kv_workspace_slots.numel(),)
        or row_starts.shape != (q.shape[0],)
        or row_ends.shape != (q.shape[0],)
    ):
        raise ValueError("workspace slots must be 1-D and row bounds must be [tokens]")
    if not (
        kv_workspace_slots.is_contiguous()
        and row_starts.is_contiguous()
        and row_ends.is_contiguous()
    ):
        raise ValueError("prefill metadata must be contiguous")
    if not (
        kv_workspace_slots.device == row_starts.device == row_ends.device == q.device
    ):
        raise ValueError("prefill metadata must be on the same device as q")


def _check_bf16_scorer_inputs(
    q: torch.Tensor,
    q_scales: torch.Tensor | None,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_size: int,
) -> None:
    """Validate BF16 queries against a bf16 ``[slots, 128]`` index-K plane."""
    if q.dtype != torch.bfloat16:
        raise TypeError(f"bf16 index-K scorer expects BF16 q, got {q.dtype}")
    if q_scales is not None:
        raise ValueError("bf16 index-K scorer takes no q_scales")
    if weights.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"DSA weights must be BF16 or FP32, got {weights.dtype}")
    if q.dim() != 3 or q.shape[1] not in (32, 64) or q.shape[2] != 128:
        raise ValueError(
            "bf16 index-K scorer requires q=[tokens, 32|64, 128], got "
            f"{tuple(q.shape)}"
        )
    if weights.shape != q.shape[:2]:
        raise ValueError(
            f"weights must have shape {tuple(q.shape[:2])}, got {tuple(weights.shape)}"
        )
    if q.stride(-1) != 1 or weights.stride(-1) != 1:
        raise ValueError("q and weights must have contiguous innermost dimensions")
    if page_size != 64:
        raise ValueError(f"bf16 index-K scorer requires page_size=64, got {page_size}")
    if (
        index_k_cache.dtype != torch.bfloat16
        or index_k_cache.dim() != 2
        or index_k_cache.shape[1] != q.shape[2]
    ):
        raise TypeError(
            f"index_k_cache must be a bf16 [slots, {q.shape[2]}] plane, got "
            f"{index_k_cache.dtype} {tuple(index_k_cache.shape)}"
        )
    if not index_k_cache.is_contiguous():
        raise ValueError("bf16 index_k_cache must be contiguous")
    if weights.device != q.device or index_k_cache.device != q.device:
        raise ValueError("q, weights, and index_k_cache must be on the same device")


def launch_gluon_dsa_decode_topk_standard_gfx950(
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
    """Score the standard paged cache and return global top-k slots."""
    del plan, seq_lens_2d
    topk = int(topk)
    q_len_per_req = int(q_len_per_req)
    _check_topk_contract(topk)
    if q_len_per_req not in (1, 2, 3, 4, 5, 6):
        raise ValueError(f"q_len_per_req must be in 1..6, got {q_len_per_req}")
    if index_k_cache is None:
        raise RuntimeError("standard-cache DSA scorer requires index_k_cache")
    _, page_stride_bytes, q_is_fp8 = _check_standard_scorer_inputs(
        q, q_scales, weights, index_k_cache, int(page_size)
    )
    _check_standard_decode_metadata(q, seq_lens, block_table, q_len_per_req)
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
        (q.shape[0], max_candidates), dtype=torch.float32, device=q.device
    )
    if q.shape[1] == 32 and q_is_fp8:
        block_n = 64
        waves_per_eu = 4
    else:
        block_n = 32
        waves_per_eu = 2
    chunk_n = 256
    num_warps = 1
    q_scale_arg = q_scales if q_scales is not None else weights
    grid = (q.shape[0], triton.cdiv(max_candidates, chunk_n))
    gluon_dsa_decode_topk_standard_gfx950[grid](
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
        USE_BUFFER_LOAD=(
            (index_k_cache.shape[0] - 1) * index_k_cache.stride(0)
            + index_k_cache.shape[1]
            < 2**31
        ),
        USE_BUFFER_STORE=logits.nbytes < 2**31,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
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


def launch_gluon_dsa_prefill_topk_standard_gfx950(
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
    """Score workspace slots in-kernel and return workspace-row top-k ids."""
    del index_k_fp8, index_k_scale
    topk = int(topk)
    _check_topk_contract(topk)
    if index_k_cache is None or page_size is None:
        raise RuntimeError("standard-cache DSA scorer requires cache and page_size")
    _, page_stride_bytes, q_is_fp8 = _check_standard_scorer_inputs(
        q, q_scales, weights, index_k_cache, int(page_size)
    )
    _check_standard_prefill_metadata(q, kv_workspace_slots, row_starts, row_ends)
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

    max_query_rows = q.shape[0]
    if max_logits_bytes is not None:
        max_query_rows = max(1, int(max_logits_bytes) // (max(workspace_rows, 1) * 4))
    q_scale_arg = q_scales if q_scales is not None else weights
    block_n = 128
    num_warps = 4
    waves_per_eu = 4 if q.shape[1] == 32 else 2
    dummy_table = row_starts
    for start in range(0, q.shape[0], max_query_rows):
        end = min(start + max_query_rows, q.shape[0])
        logits = torch.empty(
            (end - start, workspace_rows), dtype=torch.float32, device=q.device
        )
        gluon_dsa_prefill_topk_standard_gfx950[(end - start, 1)](
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
            USE_BUFFER_LOAD=(
                (index_k_cache.shape[0] - 1) * index_k_cache.stride(0)
                + index_k_cache.shape[1]
                < 2**31
            ),
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


def launch_gluon_dsa_decode_topk_bf16_gfx950(
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
    """Score a bf16 paged index-K plane and return global top-k slots.

    Args:
        q: BF16 ``[tokens, 32|64, 128]`` index queries, ``q_len_per_req``
            consecutive rows per request.
        weights: BF16 or FP32 ``[tokens, heads]`` head weights; the row
            stride may be padded.
        seq_lens: int32 ``[requests]`` full KV lengths; row ``j`` of a request
            scores the first ``seq_lens - (q_len_per_req - 1) + j`` positions.
        block_table: int32 ``[requests, max_pages]`` pages of ``page_size``
            rows of the plane.
        page_size: Rows per page; 64.
        topk: Slots to select; 512, 1024, or 2048.
        softmax_scale: Positive scale applied to every score.
        q_len_per_req: Query rows per request, 1..6.
        index_k_cache: Contiguous bf16 ``[slots, 128]`` plane.
        q_scales: Must be ``None``; BF16 queries carry no scales.
        seq_lens_2d: Unused; per-row bounds derive from ``seq_lens``.
        plan: Unused.
        out: Optional int32 ``[tokens, topk]`` output.
        lens_out: Optional int32 ``[tokens]`` output.

    Returns:
        ``(slots, lens)``: selected global slots padded with ``-1`` and the
        count of valid slots per query row.
    """
    del plan, seq_lens_2d
    topk = int(topk)
    q_len_per_req = int(q_len_per_req)
    _check_topk_contract(topk)
    if q_len_per_req not in (1, 2, 3, 4, 5, 6):
        raise ValueError(f"q_len_per_req must be in 1..6, got {q_len_per_req}")
    if index_k_cache is None:
        raise RuntimeError("bf16 index-K scorer requires index_k_cache")
    _check_bf16_scorer_inputs(q, q_scales, weights, index_k_cache, int(page_size))
    _check_standard_decode_metadata(q, seq_lens, block_table, q_len_per_req)
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
        (q.shape[0], max_candidates), dtype=torch.float32, device=q.device
    )
    # One program per request; bucket its rows so q_len_per_req stays runtime.
    rows = _next_power_of_two(q_len_per_req)
    warps_m = min(rows, _BF16_DECODE_NUM_WARPS)
    warps_n = _BF16_DECODE_NUM_WARPS // warps_m
    block_n = max(_BF16_DECODE_BLOCK_N, 32 * warps_n)
    grid = (
        seq_lens.numel(),
        triton.cdiv(max_candidates, _BF16_DECODE_CHUNK_N),
    )
    gluon_dsa_decode_topk_bf16_gfx950[grid](
        q,
        index_k_cache,
        weights,
        seq_lens,
        block_table,
        logits,
        q.shape[0],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        weights.stride(0),
        weights.stride(1),
        block_table.stride(0),
        logits.stride(0),
        float(softmax_scale),
        max_candidates,
        q_len_per_req,
        PAGE_SIZE=int(page_size),
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        ROWS=rows,
        BLOCK_N=block_n,
        CHUNK_N=_BF16_DECODE_CHUNK_N,
        WARPS_M=warps_m,
        WARPS_N=warps_n,
        NUM_STAGES=_BF16_DECODE_NUM_STAGES,
        USE_BUFFER_LOAD=index_k_cache.nbytes < 2**31,
        USE_BUFFER_STORE=logits.nbytes < 2**31,
        num_warps=_BF16_DECODE_NUM_WARPS,
        # Eight rows of 64 heads need the registers of a single wave per SIMD.
        waves_per_eu=1 if rows * q.shape[1] > 256 else 2,
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


def launch_gluon_dsa_prefill_topk_bf16_gfx950(
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
    q_scales: torch.Tensor | None = None,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score workspace rows of a bf16 index-K plane and return top-k row ids.

    Args:
        q: BF16 ``[tokens, 32|64, 128]`` index queries.
        weights: BF16 or FP32 ``[tokens, heads]`` head weights; the row
            stride may be padded.
        kv_workspace_slots: int64 ``[workspace_rows]`` plane slot of every
            workspace row.
        row_starts: int32 ``[tokens]`` inclusive first workspace row.
        row_ends: int32 ``[tokens]`` exclusive last workspace row.
        topk: Rows to select; 512, 1024, or 2048.
        softmax_scale: Positive scale applied to every score.
        index_k_cache: Contiguous bf16 ``[slots, 128]`` plane.
        page_size: The plane's page size; 64. Slots address the plane
            directly, so it only validates the cache geometry.
        q_scales: Must be ``None``; BF16 queries carry no scales.
        max_logits_bytes: Optional cap on the temporary logits; query rows
            are scored in chunks that fit it.
        out: Optional int32 ``[tokens, topk]`` output.
        lens_out: Optional int32 ``[tokens]`` output.

    Returns:
        ``(rows, lens)``: selected absolute workspace row ids padded with
        ``-1`` and the count of valid rows per query.
    """
    topk = int(topk)
    _check_topk_contract(topk)
    if index_k_cache is None or page_size is None:
        raise RuntimeError("bf16 index-K scorer requires cache and page_size")
    _check_bf16_scorer_inputs(q, q_scales, weights, index_k_cache, int(page_size))
    _check_standard_prefill_metadata(q, kv_workspace_slots, row_starts, row_ends)
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

    max_query_rows = q.shape[0]
    if max_logits_bytes is not None:
        max_query_rows = max(1, int(max_logits_bytes) // (workspace_rows * 4))
    rows_per_program = _BF16_PREFILL_ROWS[q.shape[1]]
    for start in range(0, q.shape[0], max_query_rows):
        end = min(start + max_query_rows, q.shape[0])
        logits = torch.empty(
            (end - start, workspace_rows), dtype=torch.float32, device=q.device
        )
        grid = (
            triton.cdiv(end - start, rows_per_program),
            triton.cdiv(workspace_rows, _BF16_PREFILL_CHUNK_N),
        )
        gluon_dsa_prefill_topk_bf16_gfx950[grid](
            q[start:end],
            index_k_cache,
            weights[start:end],
            kv_workspace_slots,
            row_starts[start:end],
            row_ends[start:end],
            logits,
            end - start,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            weights.stride(0),
            weights.stride(1),
            logits.stride(0),
            float(softmax_scale),
            workspace_rows,
            NUM_HEADS=q.shape[1],
            HEAD_DIM=q.shape[2],
            ROWS=rows_per_program,
            BLOCK_N=_BF16_PREFILL_BLOCK_N,
            CHUNK_N=_BF16_PREFILL_CHUNK_N,
            WARPS_M=_BF16_PREFILL_NUM_WARPS,
            WARPS_N=1,
            NUM_STAGES=_BF16_PREFILL_NUM_STAGES,
            USE_BUFFER_LOAD=index_k_cache.nbytes < 2**31,
            USE_BUFFER_STORE=logits.nbytes < 2**31,
            num_warps=_BF16_PREFILL_NUM_WARPS,
            waves_per_eu=_BF16_PREFILL_WAVES_PER_EU,
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


def _check_fp8_rows_scorer_inputs(
    q: torch.Tensor,
    q_scales: torch.Tensor | None,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_size: int,
) -> int:
    """Validate queries against fp8_scaled pages; return the page stride."""
    _, page_stride_bytes, _ = _check_standard_scorer_inputs(
        q, q_scales, weights, index_k_cache, page_size
    )
    return page_stride_bytes


def _fp8_rows_buffer_load(index_k_cache: torch.Tensor) -> bool:
    return (index_k_cache.shape[0] - 1) * index_k_cache.stride(0) + index_k_cache.shape[
        1
    ] < 2**31


def _fp8_rows_decode_splits(
    requests: int, chunks: int, waves_per_eu: int, device: torch.device
) -> int:
    # Programs per request: requests share one residency wave, two past 32 chunks each.
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    units = _device_compute_units(device_index)
    if requests * chunks <= _FP8_ROWS_DECODE_PROGRAMS_PER_CU * units:
        return chunks
    splits = max(1, triton.cdiv(waves_per_eu * units, requests))
    splits = min(max(splits, triton.cdiv(chunks, 32)), 2 * splits)
    return min(splits, chunks)


def launch_gluon_dsa_decode_topk_fp8_rows_gfx950(
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
    """Score fp8_scaled paged index keys and return global top-k slots.

    ``launch_gluon_dsa_decode_topk_bf16_gfx950``'s contract over the
    standard fp8_scaled pages (``launch_gluon_dsa_decode_topk_standard_gfx950``
    ``index_k_cache``) with BF16 queries, or E4M3 queries with FP32
    ``q_scales[token, head]`` on the FP8 MFMA: one program covers a request's
    rows, so each gathered key tile serves all of them.
    """
    del plan, seq_lens_2d
    topk = int(topk)
    q_len_per_req = int(q_len_per_req)
    _check_topk_contract(topk)
    if q_len_per_req not in (1, 2, 3, 4, 5, 6):
        raise ValueError(f"q_len_per_req must be in 1..6, got {q_len_per_req}")
    if index_k_cache is None:
        raise RuntimeError("fp8_scaled multi-row scorer requires index_k_cache")
    page_stride_bytes = _check_fp8_rows_scorer_inputs(
        q, q_scales, weights, index_k_cache, int(page_size)
    )
    _check_standard_decode_metadata(q, seq_lens, block_table, q_len_per_req)
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
        (q.shape[0], max_candidates), dtype=torch.float32, device=q.device
    )
    rows = _next_power_of_two(q_len_per_req)
    warps_m = min(rows, _BF16_DECODE_NUM_WARPS)
    warps_n = _BF16_DECODE_NUM_WARPS // warps_m
    block_n = max(_BF16_DECODE_BLOCK_N, 32 * warps_n)
    num_stages = _BF16_DECODE_NUM_STAGES
    waves_per_eu = 1 if rows * q.shape[1] > 256 else 2
    if q_scales is not None:
        num_stages = _FP8_ROWS_DECODE_NUM_STAGES
        if q.shape[1] == 32 and rows <= 4:
            waves_per_eu = _FP8_ROWS_DECODE_WAVES_PER_EU
    chunks = triton.cdiv(max_candidates, _BF16_DECODE_CHUNK_N)
    grid = (
        seq_lens.numel(),
        _fp8_rows_decode_splits(seq_lens.numel(), chunks, waves_per_eu, q.device),
    )
    # The head weights stand in for a BF16 query's unused scales.
    q_rows_scales = q_scales if q_scales is not None else weights
    gluon_dsa_decode_topk_fp8_rows_gfx950[grid](
        q,
        q_rows_scales,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        seq_lens,
        block_table,
        logits,
        q.shape[0],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        weights.stride(0),
        weights.stride(1),
        q_rows_scales.stride(0),
        q_rows_scales.stride(1),
        block_table.stride(0),
        logits.stride(0),
        float(softmax_scale),
        max_candidates,
        q_len_per_req,
        PAGE_SIZE=int(page_size),
        PAGE_STRIDE_BYTES=page_stride_bytes,
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        ROWS=rows,
        BLOCK_N=block_n,
        CHUNK_N=_BF16_DECODE_CHUNK_N,
        WARPS_M=warps_m,
        WARPS_N=warps_n,
        NUM_STAGES=num_stages,
        Q_FP8=q_scales is not None,
        USE_BUFFER_LOAD=_fp8_rows_buffer_load(index_k_cache),
        USE_BUFFER_STORE=logits.nbytes < 2**31,
        CHUNK_LOOP=grid[1] < chunks,
        num_warps=_BF16_DECODE_NUM_WARPS,
        waves_per_eu=waves_per_eu,
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


def launch_gluon_dsa_prefill_topk_fp8_rows_gfx950(
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
    """Score workspace rows of fp8_scaled pages and return top-k row ids.

    ``launch_gluon_dsa_prefill_topk_bf16_gfx950``'s contract over the
    standard fp8_scaled pages with BF16 queries, or E4M3 queries with FP32
    ``q_scales[token, head]`` on the FP8 MFMA: ``kv_workspace_slots``
    holds plane slots, and a program shares each key tile across its
    block of query rows.
    """
    del index_k_fp8, index_k_scale
    topk = int(topk)
    _check_topk_contract(topk)
    if index_k_cache is None or page_size is None:
        raise RuntimeError("fp8_scaled multi-row scorer requires cache and page_size")
    page_stride_bytes = _check_fp8_rows_scorer_inputs(
        q, q_scales, weights, index_k_cache, int(page_size)
    )
    _check_standard_prefill_metadata(q, kv_workspace_slots, row_starts, row_ends)
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

    max_query_rows = q.shape[0]
    if max_logits_bytes is not None:
        max_query_rows = max(1, int(max_logits_bytes) // (workspace_rows * 4))
    rows_per_program = _BF16_PREFILL_ROWS[q.shape[1]]
    block_n, chunk_n = (_BF16_PREFILL_BLOCK_N, _BF16_PREFILL_CHUNK_N)
    if q_scales is not None:
        block_n, chunk_n = _FP8_ROWS_PREFILL_TILES.get(q.shape[1], (block_n, chunk_n))
    # The head weights stand in for a BF16 query's unused scales.
    q_rows_scales = q_scales if q_scales is not None else weights
    for start in range(0, q.shape[0], max_query_rows):
        end = min(start + max_query_rows, q.shape[0])
        logits = torch.empty(
            (end - start, workspace_rows), dtype=torch.float32, device=q.device
        )
        grid = (
            triton.cdiv(end - start, rows_per_program),
            triton.cdiv(workspace_rows, chunk_n),
        )
        gluon_dsa_prefill_topk_fp8_rows_gfx950[grid](
            q[start:end],
            q_rows_scales[start:end],
            index_k_cache.view(torch.float8_e4m3fn),
            index_k_cache.view(torch.float32),
            weights[start:end],
            kv_workspace_slots,
            row_starts[start:end],
            row_ends[start:end],
            logits,
            end - start,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            weights.stride(0),
            weights.stride(1),
            q_rows_scales.stride(0),
            q_rows_scales.stride(1),
            logits.stride(0),
            float(softmax_scale),
            workspace_rows,
            PAGE_SIZE=int(page_size),
            PAGE_STRIDE_BYTES=page_stride_bytes,
            NUM_HEADS=q.shape[1],
            HEAD_DIM=q.shape[2],
            ROWS=rows_per_program,
            BLOCK_N=block_n,
            CHUNK_N=chunk_n,
            WARPS_M=_BF16_PREFILL_NUM_WARPS,
            WARPS_N=1,
            NUM_STAGES=_BF16_PREFILL_NUM_STAGES,
            Q_FP8=q_scales is not None,
            USE_BUFFER_LOAD=_fp8_rows_buffer_load(index_k_cache),
            USE_BUFFER_STORE=logits.nbytes < 2**31,
            num_warps=_BF16_PREFILL_NUM_WARPS,
            waves_per_eu=_BF16_PREFILL_WAVES_PER_EU,
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
