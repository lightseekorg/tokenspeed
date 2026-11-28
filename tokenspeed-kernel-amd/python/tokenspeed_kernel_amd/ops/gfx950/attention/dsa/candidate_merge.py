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

"""A global top-k merged in one launch from every KVP rank's gathered
(position, score) candidate records."""

import torch
from tokenspeed_kernel_amd._extern import extern_compile_options, wave_inclusive_sum_i32
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
    _fp32_to_topk_key,
    _vector_layout,
)

# Four 8-bit radix passes resolve a 32-bit key.
_PASSES = 4
_BUCKETS = 256
# Copies of every histogram bucket; lane l adds to copy l % _COPIES, so keys
# sharing a digit (the sign and exponent of most of a row's scores, or a run
# of equal scores in every pass) contend _COPIES ways less.
_COPIES = 8
# A row's candidates from every rank stay in registers, 32 per lane: 8 warps
# hold up to 16384 of them and 16 warps up to _MAX_COLUMNS.
_MAX_COLUMNS = 32768


def _launch_metadata(grid, kernel, args):
    rows = grid[0]
    columns = args["DEGREE"] * args["WIDTH"]
    return {
        "name": kernel.name,
        # Every record once, then every output slot and count.
        "bytes": rows * columns * 8 + rows * (args["TOPK"] + 1) * 4,
    }


@gluon.jit
def _count_first_digits(histogram, keys, live, lane, COPIES: gl.constexpr):
    # LDS-atomic histogram of the top 8 bits of the live keys, lane l adding
    # to copy l % COPIES of a bucket. Dead keys add zero to per-lane slots,
    # apart from each other: unmasked LDS atomics need no branch per element.
    slot = gl.where(live, (keys >> 24).to(gl.int32) * COPIES + lane % COPIES, lane)
    histogram.atomic_scatter_add(live.to(gl.int32), slot, axis=0)


@gluon.jit
def _pick_digit(
    histogram,
    k,
    BUCKETS: gl.constexpr,
    COPIES: gl.constexpr,
    num_warps: gl.constexpr,
):
    # Find the digit holding the k-th smallest counted key (k at most the
    # count). Every wave reads the whole histogram, lane i holding buckets
    # [i * V, (i + 1) * V) with their copies, so the search stays within the
    # wave: in-lane sums, a DPP prefix sum over lanes, wave reductions, no
    # LDS exchange. Returns the digit, the keys before it, its own count,
    # and the count.
    V: gl.constexpr = BUCKETS // 64
    layout: gl.constexpr = gl.BlockedLayout([1, V], [64, 1], [num_warps, 1], [1, 0])
    copies = histogram.load(gl.BlockedLayout([V * COPIES], [64], [num_warps], [0]))
    counts = gl.convert_layout(
        gl.sum(gl.reshape(copies, [64, V, COPIES]), axis=2), layout
    )
    lane_total = gl.sum(counts, axis=1)
    lane_inclusive = wave_inclusive_sum_i32(lane_total)
    total = gl.max(lane_inclusive, axis=0)
    k = gl.minimum(k, total)
    inclusive = (lane_inclusive - lane_total)[:, None] + gl.cumsum(counts, 1)
    before = inclusive - counts
    chosen = (before < k) & (inclusive >= k)
    digits = (
        gl.arange(0, 64, layout=gl.SliceLayout(1, layout))[:, None] * V
        + gl.arange(0, V, layout=gl.SliceLayout(0, layout))[None, :]
    )
    # One digit is chosen (none without keys): a sum publishes each field.
    digit = gl.sum(gl.sum(gl.where(chosen, digits, 0), axis=1), axis=0)
    above = gl.sum(gl.sum(gl.where(chosen, before, 0), axis=1), axis=0)
    bucket = gl.sum(gl.sum(gl.where(chosen, counts, 0), axis=1), axis=0)
    return digit, above, bucket, total


@gluon.jit(launch_metadata=_launch_metadata, do_not_specialize=("rows",))
def gluon_dsa_merge_candidates_gfx950(
    records,
    positions,
    counts,
    rows,
    WIDTH: gl.constexpr,
    DEGREE: gl.constexpr,
    TOPK: gl.constexpr,
    COLUMNS_BLOCK: gl.constexpr,
    TOPK_BLOCK: gl.constexpr,
    PASSES: gl.constexpr,
    BUCKETS: gl.constexpr,
    COPIES: gl.constexpr,
    HISTOGRAM_SLOTS: gl.constexpr,
):
    # One program per row holds every rank's WIDTH candidates of the row, in
    # rank-major candidate order. A negative position, a NaN or a -inf score
    # makes a candidate dead, and -0.0 scores +0.0 so both zeros tie as
    # floats do. Keys order larger scores first. Radix passes over the live
    # keys find the bucket of the k-th smallest, k = min(live, TOPK), 8 bits
    # at a time, stopping once that bucket is taken whole. Every live key
    # below the bucket is kept, then its first keys in candidate order up to
    # k, and the kept positions are written in candidate order.
    row = gl.program_id(0)
    NUM_WARPS: gl.constexpr = gl.num_warps()
    ELEMENTS: gl.constexpr = COLUMNS_BLOCK // (64 * NUM_WARPS)
    COLUMNS: gl.constexpr = DEGREE * WIDTH
    SLOTS: gl.constexpr = BUCKETS * COPIES
    all_layout: gl.constexpr = _vector_layout(NUM_WARPS, ELEMENTS)
    topk_layout: gl.constexpr = _vector_layout(
        NUM_WARPS, TOPK_BLOCK // (64 * NUM_WARPS)
    )
    zero_layout: gl.constexpr = _vector_layout(
        NUM_WARPS, (HISTOGRAM_SLOTS + 64 * NUM_WARPS - 1) // (64 * NUM_WARPS)
    )
    # One histogram per pass in a power-of-two allocation, zeroed once.
    histograms = gl.allocate_shared_memory(
        gl.int32,
        [HISTOGRAM_SLOTS],
        gl.PaddedSharedLayout.with_identity_for(
            [[HISTOGRAM_SLOTS, 1]], [HISTOGRAM_SLOTS], [0]
        ),
    )
    histograms.store(gl.zeros([HISTOGRAM_SLOTS], gl.int32, layout=zero_layout))
    column = gl.arange(0, COLUMNS_BLOCK, layout=all_layout)
    lane = (column // ELEMENTS) % 64
    source = column // WIDTH
    # Record (source, row, local) is int64 element ((source * rows + row) *
    # WIDTH + local): the position in its low half, the score bits in its
    # high half. The rank stride grows with rows, so the offset is int64.
    record = gl.load(
        records.to(gl.pointer_type(gl.int64))
        + (source.to(gl.int64) * rows + row) * WIDTH
        + (column - source * WIDTH),
        mask=column < COLUMNS,
        other=-1,
    )
    position = record.to(gl.int32)
    score = (record >> 32).to(gl.int32).to(gl.float32, bitcast=True)
    live = (position >= 0) & (score == score) & (score > -float("inf"))
    keys = _fp32_to_topk_key(gl.where(score == 0.0, 0.0, score))

    first_histogram = histograms.slice(0, SLOTS)
    _count_first_digits(first_histogram, keys, live, lane, COPIES)
    digit, before, bucket, total = _pick_digit(
        first_histogram, TOPK, BUCKETS, COPIES, NUM_WARPS
    )
    k = gl.minimum(total, TOPK)
    prefix = digit.to(gl.uint32)
    remaining = k - before
    shift = gl.to_tensor(32 - 8).to(gl.uint32)
    for pass_index in gl.static_range(1, PASSES):
        if bucket > remaining:
            pass_histogram = histograms.slice(pass_index * SLOTS, SLOTS)
            # Usually few keys still match: masked atomics skip the
            # instructions none of a wave's lanes issue.
            pass_histogram.atomic_scatter_add(
                gl.full([COLUMNS_BLOCK], 1, gl.int32, layout=all_layout),
                ((keys >> (32 - 8 * (pass_index + 1))) & 0xFF).to(gl.int32) * COPIES
                + lane % COPIES,
                axis=0,
                mask=live & ((keys >> shift) == prefix),
            )
            digit, before, bucket, _ = _pick_digit(
                pass_histogram, remaining, BUCKETS, COPIES, NUM_WARPS
            )
            prefix = (prefix << 8) | digit.to(gl.uint32)
            remaining -= before
            shift = gl.to_tensor(32 - 8 * (pass_index + 1)).to(gl.uint32)

    # Keep every live key below the bucket and its first ``remaining`` keys
    # in candidate order. One scan carries both running counts: the keys
    # below in the high half (fewer than k <= 2**15), the bucket's keys in
    # the low half (at most 2**15), so neither carries or overflows.
    resolved = keys >> shift
    below = live & (resolved < prefix)
    tie = live & (resolved == prefix)
    flags = (below.to(gl.int32) << 16) | tie.to(gl.int32)
    before_flags = gl.cumsum(flags, 0) - flags
    ties_before = before_flags & 0xFFFF
    keep = below | (tie & (ties_before < remaining))
    slot = (before_flags >> 16) + gl.minimum(ties_before, remaining)
    row_positions = positions + row.to(gl.int64) * TOPK
    gl.store(row_positions + slot, position, mask=keep)
    tail = gl.arange(0, TOPK_BLOCK, layout=topk_layout)
    gl.store(
        row_positions + tail,
        gl.full([TOPK_BLOCK], -1, gl.int32, layout=topk_layout),
        mask=(tail >= k) & (tail < TOPK),
    )
    gl.store(counts + row, k)


def merge_candidates_supported(*, ranks: int, candidates: int) -> bool:
    """Whether :func:`launch_gluon_dsa_merge_candidates_gfx950` merges
    ``ranks`` ranks' rows of ``candidates`` candidates: every rank's
    candidates of a row fit one program's registers. Host values only."""
    return ranks >= 1 and candidates >= 1 and ranks * candidates <= _MAX_COLUMNS


def launch_gluon_dsa_merge_candidates_gfx950(
    records: torch.Tensor, *, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a global top-``topk`` from every rank's gathered candidates.

    ``records`` are contiguous int32 ``[ranks, rows, candidates, 2]``
    (position, FP32 score bits) pairs that
    :func:`merge_candidates_supported` admits. Returns int32 ``[rows, topk]``
    positions, the kept ones first in candidate order and ``-1`` after, and
    int32 ``[rows]`` counts: bit for bit ``select_through_threshold`` over an
    exact top-k of the rank-major scores.
    """
    if records.dim() != 4 or records.shape[3] != 2:
        raise ValueError(
            "candidate records must be [ranks, rows, candidates, 2], got "
            f"{tuple(records.shape)}"
        )
    if records.dtype != torch.int32 or not records.is_contiguous():
        raise ValueError("candidate records must be contiguous int32")
    degree, rows, width, _ = records.shape
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if not merge_candidates_supported(ranks=degree, candidates=width):
        raise ValueError(
            f"{degree} ranks of {width} candidates exceed one program's "
            f"{_MAX_COLUMNS} register-resident candidates"
        )
    positions = torch.empty((rows, topk), device=records.device, dtype=torch.int32)
    counts = torch.empty((rows,), device=records.device, dtype=torch.int32)
    if rows:
        num_warps = 16 if degree * width > 16384 else 8
        lanes = 64 * num_warps
        gluon_dsa_merge_candidates_gfx950[(rows,)](
            records,
            positions,
            counts,
            rows,
            WIDTH=width,
            DEGREE=degree,
            TOPK=topk,
            COLUMNS_BLOCK=max(triton.next_power_of_2(degree * width), lanes),
            TOPK_BLOCK=max(triton.next_power_of_2(topk), lanes),
            PASSES=_PASSES,
            BUCKETS=_BUCKETS,
            COPIES=_COPIES,
            HISTOGRAM_SLOTS=triton.next_power_of_2(_PASSES * _BUCKETS * _COPIES),
            num_warps=num_warps,
            **extern_compile_options(),
        )
    return positions, counts
