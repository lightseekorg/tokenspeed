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

"""Select a KVP rank's share of a merged top-k in one CTA per row.

The selection reads every rank's gathered candidate scores. Each CTA holds its row's ``ranks * topk`` scores as 32-bit order keys (NaN and
-inf as 0, -0 as +0), finds the k-th largest live key for k = ``topk`` (the
smallest one with fewer live) in three radix digits of 11, 11, and 10 bits, and
keeps this rank's candidates above it, then the ties at it through the ranks in
order and within a rank in candidate order, ``topk`` in all: this rank's share
of the merge's selection. Kept candidates map to local slots through the local
page table in candidate order, ``-1`` after. A row with at most ``topk`` live
keys in all (a history no longer than the top-k) keeps every one without the
digits.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.common.primitives import (
    float_bits,
)
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.dsa.indexer import (
    BINS,
    LOW_BINS,
    SELECT_THREADS,
    SELECT_WARPS,
    block_exclusive,
    order_key,
    shared_add,
    table_entry,
    topdown,
    warp_sum,
)

THREADS = SELECT_THREADS
# Every rank's scores of a row in registers, at most this many per thread.
MAX_KEYS_PER_THREAD = 32
NAN_BITS = 0x7F800000
NEG_INF_BITS = 0xFF800000


@cute.jit
def _live_key(bits: Uint32) -> Uint32:
    """Return the order key of FP32 ``bits``, or 0 for NaN and -inf.

    Every live key is above 0, and -0 shares +0's key.
    """
    key = order_key(bits)
    if (bits & Uint32(0x7FFFFFFF)) > Uint32(NAN_BITS):
        key = Uint32(0)
    if bits == Uint32(NEG_INF_BITS):
        key = Uint32(0)
    return key


class OwnedSelect:
    """Select one rank's share of the merged top-k, one CTA per row.

    Reads ``degree`` ranks' gathered FP32 ``[degree, rows, width]`` scores and
    writes rank ``rank``'s local slots ``[rows, width]`` and counts ``[rows]``.
    """

    def __init__(self, degree: int, width: int, page: int):
        self.degree, self.width, self.page = degree, width, page
        self.keys = degree * width // THREADS
        self.own = max(1, width // THREADS)

    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,
        positions: cute.Tensor,
        table: cute.Tensor,
        requests: cute.Tensor,
        out: cute.Tensor,
        lens: cute.Tensor,
        rank: Int32,
        output_page_stride: Int64,
        stream,
    ):
        self.kernel(
            scores, positions, table, requests, out, lens, rank, output_page_stride
        ).launch(grid=(out.shape[0], 1, 1), block=(THREADS, 1, 1), stream=stream)

    @cute.jit
    def digit(self, hist, info, keys, shift, mask, match_shift, match, tid):
        """Histogram one radix digit of the live keys under a prefix.

        Counts ``(key >> shift) & mask`` over the live keys whose bits above
        ``match_shift`` are ``match`` (every live key at 32).
        """
        cute.arch.sync_threads()
        for b in cutlass.range(tid, BINS, THREADS, unroll=1):
            hist[b] = Uint32(0)
        if tid == 0:
            info[2] = Int32(0)
            info[3] = Int32(0)
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(self.keys):
            key = keys[i]
            hit = Int32(0)
            if key != Uint32(0):
                hit = Int32(1)
            if cutlass.const_expr(match_shift < 32):
                if (key >> Uint32(match_shift)) != match:
                    hit = Int32(0)
            if hit != Int32(0):
                shared_add(hist.iterator + ((key >> Uint32(shift)) & mask), 1)
        cute.arch.sync_threads()

    @cute.jit
    def threshold(self, hist, info, scan, keys, own_keys, rank, tid, warp, lane):
        """Find the row's threshold key and this rank's tie quota at it.

        Three radix digits run over the row's live keys (more than ``W``
        whenever this runs) to find the ``W``-th largest. Returns the threshold,
        the ties to take at it, and this thread's first tie's rank among them
        (lower ranks' ties first, then candidate order).
        """
        W = self.width
        self.digit(hist, info, keys, 21, 0x7FF, 32, Uint32(0), tid)
        need = cutlass.min(Uint32(info[4]), Uint32(W))
        topdown(hist, BINS, need, info, scan, tid, warp, lane)
        c0 = Uint32(info[2])
        need -= Uint32(info[3])
        self.digit(hist, info, keys, 10, 0x7FF, 21, c0, tid)
        topdown(hist, BINS, need, info, scan, tid, warp, lane)
        c1 = Uint32(info[2])
        need -= Uint32(info[3])
        prefix = (c0 << Uint32(11)) | c1
        self.digit(hist, info, keys, 0, 0x3FF, 10, prefix, tid)
        topdown(hist, LOW_BINS, need, info, scan, tid, warp, lane)
        threshold = (prefix << Uint32(10)) | Uint32(info[2])
        # The ties at the threshold to take, through the ranks in order: one
        # scan sums the lower ranks' ties (high half) and ranks this rank's.
        take = need - Uint32(info[3])
        ties = Uint32(0)
        for i in cutlass.range_constexpr(self.keys):
            column = tid + THREADS * i
            if keys[i] == threshold and keys[i] != Uint32(0):
                if column // W < rank:
                    ties += Uint32(1 << 16)
        for e in cutlass.range_constexpr(self.own):
            if own_keys[e] == threshold and own_keys[e] != Uint32(0):
                ties += Uint32(1)
        tie_rank, tie_total = block_exclusive(ties, scan, warp, lane)
        tie_rank = (tie_rank & Uint32(0xFFFF)) + (tie_total >> Uint32(16))
        return threshold, take, tie_rank

    @cute.kernel
    def kernel(
        self, scores, positions, table, requests, out, lens, rank, output_page_stride
    ):
        """Select rank ``rank``'s share of one row's merged top-k.

        Reads the row's scores of every rank, finds the threshold key in three
        radix digits, then keeps this rank's candidates above it and the ties
        at it its quota allows (lower ranks' ties first, then candidate
        order). Writes their local slots through ``positions`` and ``table``
        to ``out`` in candidate order, ``-1`` after, and the count to ``lens``.
        """
        W = self.width
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32

        smem = cutlass.memory.SmemAllocator()
        hist = smem.allocate_tensor(Uint32, cute.make_layout(BINS), 16)
        scan = smem.allocate_tensor(Uint32, cute.make_layout(SELECT_WARPS), 16)
        info = smem.allocate_tensor(Int32, cute.make_layout(8), 16)

        # Column c = tid + THREADS * i of the row: rank c // W, candidate c % W.
        raw = []
        for i in cutlass.range_constexpr(self.keys):
            column = tid + THREADS * i
            raw.append(scores[column // W, row, column % W])
        keys = []
        live = Uint32(0)
        for i in cutlass.range_constexpr(self.keys):
            key = _live_key(Uint32(float_bits(raw[i])))
            if key != Uint32(0):
                live += Uint32(1)
            keys.append(key)
        # This rank's candidates [tid * own, tid * own + own) in candidate
        # order, with their local slots: loads that overlap the radix digits.
        first = tid * self.own
        request = requests[row]
        own_keys = []
        slots = []
        for e in cutlass.range_constexpr(self.own):
            key = Uint32(0)
            slot = Int32(-1)
            if first + e < W:
                key = _live_key(Uint32(float_bits(scores[rank, row, first + e])))
            if key != Uint32(0):
                position = positions[row, first + e]
                if position >= Int32(0):
                    page = table_entry(table, request, position // Int32(self.page))
                    if page >= Int32(0):
                        slot = Int32(
                            Int64(page) * output_page_stride
                            + Int64(position % Int32(self.page))
                        )
            own_keys.append(key)
            slots.append(slot)
        live = warp_sum(live)
        if tid == 0:
            info[4] = Int32(0)
        cute.arch.sync_threads()
        if lane == 0:
            shared_add(info.iterator + 4, live)
        cute.arch.sync_threads()

        # At most ``W`` live keys in all (a row no longer than the top-k): every
        # live key is kept, so threshold 0 with no ties gives the same selection.
        threshold = Uint32(0)
        take = Uint32(0)
        tie_rank = Uint32(0)
        if Uint32(info[4]) > Uint32(W):
            threshold, take, tie_rank = self.threshold(
                hist, info, scan, keys, own_keys, rank, tid, warp, lane
            )
        kept = Uint32(0)
        kept_slots = []
        for e in cutlass.range_constexpr(self.own):
            key = own_keys[e]
            slot = slots[e]
            keep = Int32(0)
            if key > threshold:
                keep = Int32(1)
            if key == threshold and key != Uint32(0):
                if tie_rank < take:
                    keep = Int32(1)
                tie_rank += Uint32(1)
            if keep == Int32(0):
                slot = Int32(-1)
            if slot >= Int32(0):
                kept += Uint32(1)
            kept_slots.append(slot)
        at, count = block_exclusive(kept, scan, warp, lane)
        for e in cutlass.range_constexpr(self.own):
            if kept_slots[e] >= Int32(0):
                out[row, Int32(at)] = kept_slots[e]
                at += Uint32(1)
        for c in cutlass.range(tid, W, THREADS, unroll=1):
            if c >= Int32(count):
                out[row, c] = Int32(-1)
        if tid == 0:
            lens[row] = Int32(count)


def _fake(dtype, shape, stride, align):
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=align
    )


@functools.cache
def compile_owned_select(degree: int, width: int, page: int):
    """Compile the owned selection for a rank count, width, and page size.

    Selects among ``degree`` ranks' ``width`` candidates per row over
    ``page``-token pages. Rows, the rank stride, the table, and the rank are
    runtime values: one compile serves every batch.
    """
    sym = cute.sym_int
    scores = _fake(
        cutlass.Float32, (degree, sym(), width), (cute.sym_int64(), width, 1), 4
    )
    positions = _fake(Int32, (sym(), width), (cute.sym_int64(), 1), 4)
    table = _fake(Int32, (sym(), sym()), (cute.sym_int64(), 1), 4)
    requests = _fake(Int32, (sym(),), (1,), 4)
    out = _fake(Int32, (sym(), width), (width, 1), 4)
    lens = _fake(Int32, (sym(),), (1,), 4)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        OwnedSelect(degree, width, page),
        scores,
        positions,
        table,
        requests,
        out,
        lens,
        Int32(0),
        Int64(0),
        stream,
        options="--enable-tvm-ffi",
    )


def owned_selection_supported(
    *, ranks: int, rows: int, topk: int, rank_stride: int
) -> bool:
    """Return whether the owned selection takes these gathered scores.

    The scores are ``ranks`` ranks' ``rows`` rows of ``topk`` candidates,
    ``rank_stride`` elements apart. It takes a power-of-two ``topk`` of at
    least 512 and a power-of-two rank count whose every-rank row fits the
    CTA's registers. Host values only.
    """
    keys = ranks * topk
    return (
        topk >= 512
        and topk & (topk - 1) == 0
        and ranks >= 2
        and ranks & (ranks - 1) == 0
        and keys % THREADS == 0
        and keys // THREADS <= MAX_KEYS_PER_THREAD
        and rank_stride >= rows * topk
    )


def cute_dsl_select_owned_from_gathered(
    scores: torch.Tensor,
    positions: torch.Tensor,
    local_page_table: torch.Tensor,
    query_requests: torch.Tensor,
    *,
    rank: int,
    topk: int,
    page_size: int,
    output_page_stride: int,
    out: torch.Tensor,
    lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select rank ``rank``'s share of the merged top-k in one launch.

    Reads every rank's gathered FP32 ``[ranks, rows, topk]`` scores (rows
    contiguous) and writes int32 ``out`` ``[rows, topk]``, the kept local slots
    first in candidate order and ``-1`` after, and ``lens_out`` ``[rows]``
    counts.
    ``output_page_stride`` includes padding between physical cache pages.
    """
    degree, rows, width = scores.shape
    if rows:
        compile_owned_select(degree, width, page_size)(
            scores,
            positions,
            local_page_table,
            query_requests,
            out,
            lens_out,
            rank,
            output_page_stride,
        )
    return out, lens_out
