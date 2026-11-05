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

"""A KVP rank's share of a global top-k, selected in one launch straight from
every rank's gathered candidate scores."""

import torch
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
    _fp32_to_topk_key,
    _vector_layout,
)

# Measured on MI350X over 8 x 2048 gathered candidates per row: 13.5 us per
# launch with these digits and warps, 19.3 us with the merge top-k's 12-12-8
# digits (twice the buckets to scan per pass), 18.9 us with 4 warps.
_RADIX_SCHEDULE = (11, 11, 10)
_NUM_WARPS = 8
# A row's candidates from every rank stay in registers: 32 per lane.
_MAX_COLUMNS = 32 * 64 * _NUM_WARPS


@gluon.jit
def _rank_scores(row_scores, rank, rank_stride, local):
    values = gl.amd.cdna4.buffer_load(
        ptr=row_scores + rank * rank_stride, offsets=local
    )
    return gl.where(values == values, values, -float("inf"))


@gluon.jit(do_not_specialize=("rank", "rank_stride", "position_stride", "table_stride"))
def _select_owned_from_gathered_kernel(
    scores,
    positions,
    local_page_table,
    query_requests,
    slots,
    counts,
    rank,
    rank_stride,
    position_stride,
    table_stride,
    WIDTH: gl.constexpr,
    DEGREE: gl.constexpr,
    PAGE: gl.constexpr,
    RADIX0_BITS: gl.constexpr,
    RADIX1_BITS: gl.constexpr,
    RADIX2_BITS: gl.constexpr,
    MAX_BUCKETS: gl.constexpr,
):
    # One program per row holds every rank's WIDTH candidate scores of the
    # row (NaN as -inf). Three radix passes over their keys, which order
    # larger scores first, find the k-th largest live score for k = WIDTH,
    # or the smallest one when fewer are live: the threshold the merge's
    # top-k leaves. The keep then compares scores as floats, so both zeros
    # tie: every live score above the threshold, then the ties at it through
    # the ranks in order and within a rank in candidate order, WIDTH in all.
    row = gl.program_id(0)
    NUM_WARPS: gl.constexpr = gl.num_warps()
    COLUMNS: gl.constexpr = DEGREE * WIDTH
    all_layout: gl.constexpr = _vector_layout(NUM_WARPS, COLUMNS // (64 * NUM_WARPS))
    rank_layout: gl.constexpr = _vector_layout(NUM_WARPS, WIDTH // (64 * NUM_WARPS))
    histogram_layout: gl.constexpr = _vector_layout(
        NUM_WARPS, (MAX_BUCKETS + 64 * NUM_WARPS - 1) // (64 * NUM_WARPS)
    )
    shared_histogram = gl.allocate_shared_memory(
        gl.int32,
        [MAX_BUCKETS],
        gl.PaddedSharedLayout.with_identity_for([[MAX_BUCKETS, 1]], [MAX_BUCKETS], [0]),
    )
    column = gl.arange(0, COLUMNS, layout=all_layout)
    source = column // WIDTH
    values = gl.amd.cdna4.buffer_load(
        ptr=scores, offsets=source * rank_stride + row * WIDTH + column % WIDTH
    )
    values = gl.where(values == values, values, -float("inf"))
    live = values != -float("inf")
    keys = _fp32_to_topk_key(values)
    ones = gl.full([COLUMNS], 1, gl.int32, layout=all_layout)
    buckets = gl.arange(0, MAX_BUCKETS, layout=histogram_layout)
    prefix = gl.full([], 0, gl.uint32)
    remaining = gl.full([], WIDTH, gl.int32)
    live_count = gl.full([], 0, gl.int32)
    for pass_index in gl.static_range(3):
        radix_bits = RADIX0_BITS
        shift = 32 - RADIX0_BITS
        if pass_index == 1:
            radix_bits = RADIX1_BITS
            shift = 32 - RADIX0_BITS - RADIX1_BITS
        elif pass_index == 2:
            radix_bits = RADIX2_BITS
            shift = 0
        shared_histogram.store(
            gl.zeros([MAX_BUCKETS], gl.int32, layout=histogram_layout)
        )
        match = live
        if pass_index != 0:
            match = match & ((keys >> (shift + radix_bits)) == prefix)
        shared_histogram.atomic_scatter_add(
            ones,
            ((keys >> shift) & ((1 << radix_bits) - 1)).to(gl.int32),
            axis=0,
            mask=match,
        )
        counted = shared_histogram.load(histogram_layout)
        if pass_index == 0:
            live_count = gl.sum(counted, axis=0)
            remaining = gl.minimum(live_count, WIDTH)
        cumulative = gl.cumsum(counted, 0)
        before = cumulative - counted
        chosen = (before < remaining) & (cumulative >= remaining)
        # One bucket is chosen: a single sum publishes it and the count of
        # keys before it (under WIDTH).
        picked = gl.sum(
            gl.where(chosen, buckets.to(gl.uint32) | (before.to(gl.uint32) << 12), 0),
            axis=0,
        )
        prefix = (prefix << radix_bits) | (picked & 0xFFF)
        remaining -= (picked >> 12).to(gl.int32)
    threshold = gl.where(prefix < 0x80000000, prefix ^ 0x7FFFFFFF, prefix).to(
        gl.float32, bitcast=True
    )
    threshold = gl.where(live_count > 0, threshold, float("inf"))

    above = gl.sum((values > threshold).to(gl.int32), axis=0)
    tied_before = gl.sum(
        ((values == threshold) & live & (source < rank)).to(gl.int32), axis=0
    )
    quota = WIDTH - above
    local = gl.arange(0, WIDTH, layout=rank_layout)
    own = _rank_scores(scores + row.to(gl.int64) * WIDTH, rank, rank_stride, local)
    own_live = own != -float("inf")
    own_tie = (own == threshold) & own_live
    tie_rank = tied_before + gl.cumsum(own_tie.to(gl.int32), 0) - 1
    keep = ((own > threshold) | (own_tie & (tie_rank < quota))) & own_live
    position = gl.load(positions + row * position_stride + local, mask=keep, other=0)
    request = gl.load(query_requests + row).to(gl.int64)
    page = gl.load(
        local_page_table + request * table_stride + position // PAGE,
        mask=keep,
        other=-1,
    )
    # A kept candidate on a page this rank does not own is dropped, as
    # compacting the whole selection's owned slots would drop it.
    keep = keep & (page >= 0)
    slot = gl.cumsum(keep.to(gl.int32), 0) - 1
    gl.store(slots + row * WIDTH + slot, page * PAGE + position % PAGE, mask=keep)
    count = gl.sum(keep.to(gl.int32), axis=0)
    gl.store(
        slots + row * WIDTH + local,
        gl.full([WIDTH], -1, gl.int32, layout=rank_layout),
        mask=local >= count,
    )
    gl.store(counts + row, count)


def gathered_owned_selection_fits(scores: torch.Tensor, topk: int) -> bool:
    """Whether :func:`gluon_dsa_select_owned_from_gathered_gfx950` takes
    these gathered scores (FP32 ``[ranks, rows, candidates]``): ``topk``
    candidates per rank and row, a power of two that every rank's share of
    a row fills registers with, rows contiguous, and every byte read within
    one buffer's 32-bit range."""
    if scores.dim() != 3 or scores.dtype != torch.float32:
        return False
    degree, rows, width = scores.shape
    return (
        width == topk
        and width & (width - 1) == 0
        and width >= 64 * _NUM_WARPS
        and degree & (degree - 1) == 0
        and degree * width <= _MAX_COLUMNS
        and scores.stride()[1:] == (width, 1)
        and ((degree - 1) * scores.stride(0) + rows * width) * 4 < 2**31
    )


def gluon_dsa_select_owned_from_gathered_gfx950(
    scores: torch.Tensor,
    positions: torch.Tensor,
    local_page_table: torch.Tensor,
    query_requests: torch.Tensor,
    *,
    rank: int,
    topk: int,
    page_size: int,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank ``rank``'s share of the top-``topk`` over every rank's gathered
    candidate scores, as local cache slots, in one launch per call.

    ``scores`` are FP32 ``[ranks, rows, topk]`` (rows contiguous, the rank
    stride possibly wider) that :func:`gathered_owned_selection_fits`;
    ``positions`` the int32 ``[rows, topk]`` logical positions of this
    rank's candidates; ``local_page_table`` its int32 ``[requests, pages]``
    local pages (``-1`` where not owned); ``query_requests`` every row's
    request. Returns int32 ``[rows, topk]`` slots, the kept ones first in
    candidate order and ``-1`` after, and int32 ``[rows]`` counts: bit for
    bit ``select_owned_through_threshold`` over the merge's top-k.
    """
    if not gathered_owned_selection_fits(scores, topk):
        raise ValueError(
            "gathered owned selection needs FP32 [ranks, rows, topk] scores "
            "with contiguous rows, power-of-two ranks and topk, and every "
            f"rank's row in registers; got {tuple(scores.shape)} strided "
            f"{scores.stride()} for top-{topk}"
        )
    degree, rows, width = scores.shape
    if not 0 <= rank < degree:
        raise ValueError(f"rank {rank} is outside {degree} ranks")
    if any(
        value.dtype != torch.int32
        for value in (positions, local_page_table, query_requests)
    ):
        raise TypeError("candidate positions, local pages and requests must be int32")
    if positions.shape != (rows, width) or positions.stride(1) != 1:
        raise ValueError(
            "this rank's candidate positions need contiguous [rows, topk] rows"
        )
    if local_page_table.stride(1) != 1:
        raise ValueError("local pages need contiguous rows")
    if query_requests.shape != (rows,) or not query_requests.is_contiguous():
        raise ValueError("every row needs its request")
    if out is None:
        out = torch.empty((rows, topk), device=scores.device, dtype=torch.int32)
    if lens_out is None:
        lens_out = torch.empty((rows,), device=scores.device, dtype=torch.int32)
    if out.shape != (rows, topk) or not out.is_contiguous():
        raise ValueError("owned slots need a contiguous [rows, topk] output")
    if lens_out.shape != (rows,) or not lens_out.is_contiguous():
        raise ValueError("owned counts need a contiguous [rows] output")
    if rows:
        _select_owned_from_gathered_kernel[(rows,)](
            scores,
            positions,
            local_page_table,
            query_requests,
            out,
            lens_out,
            rank,
            scores.stride(0),
            positions.stride(0),
            local_page_table.stride(0),
            WIDTH=width,
            DEGREE=degree,
            PAGE=page_size,
            RADIX0_BITS=_RADIX_SCHEDULE[0],
            RADIX1_BITS=_RADIX_SCHEDULE[1],
            RADIX2_BITS=_RADIX_SCHEDULE[2],
            MAX_BUCKETS=1 << max(_RADIX_SCHEDULE),
            num_warps=_NUM_WARPS,
        )
    return out, lens_out
