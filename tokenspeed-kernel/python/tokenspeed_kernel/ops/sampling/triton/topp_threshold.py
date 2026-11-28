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

"""Exact top-p nucleus of each row, as a threshold.

The nucleus of a row is the shortest prefix of its tokens, ranked by scaled
logit ``s = logits / T`` descending and then by token id ascending, such that
the mass before each kept token is below ``top_p`` of the row's total mass.
It is returned as its last token: that token's ``s`` (``threshold``) and id
(``limit``); token ``t`` is kept iff ``s_t > threshold`` or ``s_t ==
threshold`` and ``t <= limit``. Rows with ``top_p >= 1`` keep every token
(``threshold = -inf``).

Launches:

1. Per vocabulary block (many programs per row): the block's max, its
   ``sum(exp(s - max))`` and its ``NUM_CANDIDATES`` best tokens as rank keys.
2. Per row: the total mass and the candidates above the highest
   ``NUM_CANDIDATES``-th key of any block. Every token above that key is a
   candidate of its block, so they are an exact prefix of the row's ranking;
   when their mass reaches the target the boundary is among them, found by
   sorting and scanning them. Otherwise the row takes the rare path.
3. Rare path, for rows whose nucleus reaches past every block's candidates
   (very flat rows): a fixed grid of programs builds radix histograms of the
   row over its whole vocabulary, one level per launch. Masses are 2**-40
   fixed point summed with integer atomics, so the histograms do not depend
   on the order the programs add in. Three levels walk the scaled logit's
   order-preserving bits (12 + 12 + 8) down to the boundary's value; two more
   count the token ids holding that value, so ties resolve by lowest id; a
   last launch writes the result. Rows past the histogram slots take a
   single-program search over the row instead.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

_BLOCK_SIZE = 1024
_NUM_CANDIDATES = 32
_FEW_CANDIDATES = 4
_CAP = 4096
_NUM_PIVOTS = 16
# Each bracket round splits the bracket's value range 17 ways: fp32 runs out
# of distinct values long before this bound.
_MAX_ROUNDS = 64
_TOP_P_EPS = 1.0e-6
_KEEP_ALL_LIMIT = 2147483647
# Rare path: histogram bins per level, the grid of each level launch, and the
# rows a call keeps histograms for (more fall back to the per-row search).
_HIST_BINS = 4096
_HIST_PROGRAMS = 64
# Prefix fields a rare-path launch passes to the next: high, mid, value,
# need, id_high, below (int64) and the mass above and the target (fp64).
_RARE_STATE_FIELDS = 8
_MAX_RARE_ROWS = 64
_RARE_HISTOGRAM = 1
_RARE_SEARCH = 2


@triton.jit
def _ordered_bits(s):
    """uint32 whose unsigned order is the fp32 order of ``s``."""
    bits = s.to(tl.uint32, bitcast=True)
    return tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)


@triton.jit
def _from_ordered_bits(ordered):
    bits = tl.where((ordered & 0x80000000) != 0, ordered ^ 0x80000000, ~ordered)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _rank_key(s, ids):
    """uint64 rank key: ordered fp32 bits above the inverted token id."""
    inverted_ids = ~(ids.to(tl.uint32))
    return (_ordered_bits(s).to(tl.uint64) << 32) | inverted_ids.to(tl.uint64)


@triton.jit
def _rank_key_value(key):
    """The scaled logit and token id a rank key holds."""
    ids = (~(key.to(tl.uint32))).to(tl.int32)
    return _from_ordered_bits((key >> 32).to(tl.uint32)), ids


@triton.jit
def _fixed_mass(s, row_max):
    """A token's mass ``exp(s - max)`` in 2**-40 fixed point."""
    return (tl.exp(s - row_max) * 1099511627776.0).to(tl.int64)


@triton.jit
def _top_p_candidates_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    top_p_pool_ptr,
    block_stats_ptr,
    candidates_ptr,
    rare_rows_ptr,
    logits_row_stride,
    stats_row_stride,
    candidates_row_stride,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    MAX_RARE_ROWS: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
    TOP_P_EPS: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    if (row == 0) & (block_idx == 0):
        # The selection that follows counts its rare rows from zero.
        tl.store(rare_rows_ptr + MAX_RARE_ROWS, 0)
    pool_idx = tl.load(req_pool_indices_ptr + row // NUM_TOKENS_PER_REQ).to(tl.int64)
    top_p = tl.load(top_p_pool_ptr + pool_idx).to(tl.float32)
    if top_p < 1.0 - TOP_P_EPS:
        temperature = tl.maximum(
            tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
        )
        cols = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        s = tl.load(
            logits_ptr + row.to(tl.int64) * logits_row_stride + cols,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        s = s / temperature
        block_max = tl.max(s, axis=0)
        safe_max = tl.where(block_max == float("-inf"), 0.0, block_max)
        block_sum = tl.sum(tl.where(mask, tl.exp(s - safe_max), 0.0), axis=0)
        stats = block_stats_ptr + row.to(tl.int64) * stats_row_stride + 2 * block_idx
        tl.store(stats, block_max)
        tl.store(stats + 1, block_sum)
        # Padding keys are 0, below every real token.
        keys = tl.where(mask, _rank_key(s, cols), tl.zeros_like(_rank_key(s, cols)))
        best = tl.topk(keys, NUM_CANDIDATES)
        lanes = tl.arange(0, NUM_CANDIDATES)
        tl.store(
            candidates_ptr
            + row.to(tl.int64) * candidates_row_stride
            + block_idx * NUM_CANDIDATES
            + lanes,
            best.to(tl.int64, bitcast=True),
        )


@triton.jit
def _scan_candidates(
    candidates_row,
    row_max,
    target,
    num_blocks: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    PER_BLOCK: tl.constexpr,
    LANES: tl.constexpr,
):
    """The boundary among each block's ``PER_BLOCK`` best tokens, if there.

    Every token above the highest ``PER_BLOCK``-th key of any block is among
    the candidates read here, so the candidates above that key are the row's
    exact top ranks. Sorting and scanning them finds the boundary when their
    mass reaches the target. Returns ``(found, threshold, limit)``.
    """
    lanes = tl.arange(0, LANES)
    in_range = lanes < num_blocks * PER_BLOCK
    block = lanes // PER_BLOCK
    rank = lanes % PER_BLOCK
    keys = tl.load(
        candidates_row + block * NUM_CANDIDATES + rank, mask=in_range, other=0
    ).to(tl.uint64, bitcast=True)
    floor = tl.max(tl.where(in_range & (rank == PER_BLOCK - 1), keys, 0), axis=0)
    keys = tl.where(in_range & (keys > floor), keys, 0)
    # Sum the candidates' mass first: below the target the boundary is not
    # among them, and the sort would be wasted.
    s, _ = _rank_key_value(keys)
    found = tl.sum(tl.where(keys > 0, tl.exp(s - row_max), 0.0), axis=0) >= target
    threshold = tl.full((), float("-inf"), tl.float32)
    limit = tl.full((), -1, tl.int32)
    if found:
        keys = tl.sort(keys, descending=True)
        s, ids = _rank_key_value(keys)
        valid = keys > 0
        w = tl.where(valid, tl.exp(s - row_max), 0.0)
        before = tl.cumsum(w, axis=0) - w
        last = tl.sum((valid & (before < target)).to(tl.int32), axis=0) - 1
        threshold = tl.max(tl.where(lanes == last, s, float("-inf")), axis=0)
        limit = tl.max(tl.where(lanes == last, ids, -1), axis=0)
    return found, threshold, limit


@triton.jit
def _top_p_select_kernel(
    req_pool_indices_ptr,
    top_p_pool_ptr,
    block_stats_ptr,
    candidates_ptr,
    threshold_ptr,
    limit_ptr,
    rare_ptr,
    rare_rows_ptr,
    rare_row_max_ptr,
    mass_hist_ptr,
    count_hist_ptr,
    stats_row_stride,
    candidates_row_stride,
    num_blocks: tl.constexpr,
    NUM_BLOCKS_PAD: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    FEW_CANDIDATES: tl.constexpr,
    FEW_PAD: tl.constexpr,
    ALL_PAD: tl.constexpr,
    BINS: tl.constexpr,
    MAX_RARE_ROWS: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
    TOP_P_EPS: tl.constexpr,
    RARE_HISTOGRAM: tl.constexpr,
    RARE_SEARCH: tl.constexpr,
):
    row = tl.program_id(0)
    pool_idx = tl.load(req_pool_indices_ptr + row // NUM_TOKENS_PER_REQ).to(tl.int64)
    top_p = tl.load(top_p_pool_ptr + pool_idx).to(tl.float32)
    threshold = tl.full((), float("-inf"), tl.float32)
    limit = tl.full((), 2147483647, tl.int32)
    rare = tl.full((), 0, tl.int32)
    if top_p < 1.0 - TOP_P_EPS:
        blocks = tl.arange(0, NUM_BLOCKS_PAD)
        block_mask = blocks < num_blocks
        stats = block_stats_ptr + row.to(tl.int64) * stats_row_stride + 2 * blocks
        block_max = tl.load(stats, mask=block_mask, other=float("-inf"))
        block_sum = tl.load(stats + 1, mask=block_mask, other=0.0)
        row_max = tl.max(block_max, axis=0)
        total = tl.sum(block_sum * tl.exp(block_max - row_max), axis=0)
        target = top_p * total
        candidates_row = candidates_ptr + row.to(tl.int64) * candidates_row_stride
        # Most nuclei are a few tokens: each block's first few candidates
        # usually hold them, and sort much faster than all of them.
        found, threshold, limit = _scan_candidates(
            candidates_row,
            row_max,
            target,
            num_blocks,
            NUM_CANDIDATES,
            FEW_CANDIDATES,
            FEW_PAD,
        )
        if not found:
            found, threshold, limit = _scan_candidates(
                candidates_row,
                row_max,
                target,
                num_blocks,
                NUM_CANDIDATES,
                NUM_CANDIDATES,
                ALL_PAD,
            )
        if not found:
            # Rare path: a histogram slot (_top_p_rare_level_kernel) while
            # slots last, else the per-row search (_top_p_threshold_exact_kernel).
            slot = tl.atomic_add(rare_rows_ptr + MAX_RARE_ROWS, 1)
            if slot < MAX_RARE_ROWS:
                rare = RARE_HISTOGRAM
                tl.store(rare_rows_ptr + slot, row)
                tl.store(rare_row_max_ptr + slot, row_max)
                bins = tl.arange(0, BINS)
                for level in tl.static_range(3):
                    tl.store(mass_hist_ptr + (slot * 3 + level) * BINS + bins, 0)
                for level in tl.static_range(2):
                    tl.store(count_hist_ptr + (slot * 2 + level) * BINS + bins, 0)
            else:
                rare = RARE_SEARCH
    tl.store(threshold_ptr + row, threshold)
    tl.store(limit_ptr + row, limit)
    tl.store(rare_ptr + row, rare)


@triton.jit
def _top_p_threshold_exact_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    top_p_pool_ptr,
    rare_ptr,
    keys_ptr,
    threshold_ptr,
    limit_ptr,
    logits_row_stride,
    keys_row_stride,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CAP: tl.constexpr,
    NUM_PIVOTS: tl.constexpr,
    MAX_ROUNDS: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
    RARE_SEARCH: tl.constexpr,
):
    """The single-program search over the whole row, for rare rows past the
    histogram slots."""
    row = tl.program_id(0)
    if tl.load(rare_ptr + row) == RARE_SEARCH:
        pool_idx = tl.load(req_pool_indices_ptr + row // NUM_TOKENS_PER_REQ).to(
            tl.int64
        )
        temperature = tl.maximum(
            tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
        )
        top_p = tl.load(top_p_pool_ptr + pool_idx).to(tl.float32)
        logits_row = logits_ptr + row.to(tl.int64) * logits_row_stride
        keys_row = keys_ptr + row.to(tl.int64) * keys_row_stride
        offs = tl.arange(0, BLOCK_SIZE)
        pivot_ids = tl.arange(0, NUM_PIVOTS)

        row_max = tl.full((), float("-inf"), tl.float32)
        row_min = tl.full((), float("inf"), tl.float32)
        for start in tl.range(0, vocab_size, BLOCK_SIZE):
            cols = start + offs
            mask = cols < vocab_size
            s = tl.load(logits_row + cols, mask=mask, other=float("-inf")).to(
                tl.float32
            )
            s = s / temperature
            row_max = tl.maximum(row_max, tl.max(s, axis=0))
            finite = mask & (s > float("-inf"))
            row_min = tl.minimum(row_min, tl.min(tl.where(finite, s, float("inf"))))

        # Bracket (lo, hi] around the boundary's scaled logit, with the mass
        # and count of the tokens above each end. Nothing is above the max.
        hi = row_max
        hi_mass = tl.full((), 0.0, tl.float32)
        hi_count = tl.full((), 0, tl.int32)
        lo = row_min - 1.0
        lo_mass = tl.full((), 0.0, tl.float32)
        lo_count = tl.full((), 0, tl.int32)
        target = tl.full((), 0.0, tl.float32)
        state = tl.full((), 0, tl.int32)  # 0 search, 1 sort, 2 one value left
        rounds = tl.full((), 0, tl.int32)
        while (state == 0) & (rounds < MAX_ROUNDS):
            # First round: fixed distances below the max (most mass sits near
            # it), the last pivot below every token for the total. Later
            # rounds: evenly inside the bracket.
            gap = 0.25 * tl.exp(pivot_ids.to(tl.float32) * 0.3716)
            first_pivots = tl.where(
                pivot_ids == NUM_PIVOTS - 1, row_min - 1.0, row_max - gap
            )
            step = (hi - lo) / (NUM_PIVOTS + 1)
            even_pivots = hi - step * (pivot_ids + 1).to(tl.float32)
            pivots = tl.where(rounds == 0, first_pivots, even_pivots)

            masses = tl.zeros((NUM_PIVOTS,), tl.float32)
            counts = tl.zeros((NUM_PIVOTS,), tl.int32)
            for start in tl.range(0, vocab_size, BLOCK_SIZE):
                cols = start + offs
                mask = cols < vocab_size
                s = tl.load(logits_row + cols, mask=mask, other=float("-inf")).to(
                    tl.float32
                )
                s = s / temperature
                w = tl.where(mask, tl.exp(s - row_max), 0.0)
                above = mask[:, None] & (s[:, None] > pivots[None, :])
                masses += tl.sum(tl.where(above, w[:, None], 0.0), axis=0)
                counts += tl.sum(above.to(tl.int32), axis=0)
            if rounds == 0:
                total = tl.max(tl.where(pivot_ids == NUM_PIVOTS - 1, masses, 0.0))
                target = top_p * total

            # Pivots descend, so masses ascend: the first pivot whose mass
            # above reaches the target is the bracket's new low end.
            j = tl.min(tl.where(masses >= target, pivot_ids, NUM_PIVOTS))
            at_j = pivot_ids == j
            at_prev = pivot_ids == j - 1
            if j < NUM_PIVOTS:
                lo = tl.max(tl.where(at_j, pivots, float("-inf")))
                lo_mass = tl.max(tl.where(at_j, masses, 0.0))
                lo_count = tl.max(tl.where(at_j, counts, 0))
            if j > 0:
                hi = tl.max(tl.where(at_prev, pivots, float("-inf")))
                hi_mass = tl.max(tl.where(at_prev, masses, 0.0))
                hi_count = tl.max(tl.where(at_prev, counts, 0))
            if lo_count - hi_count <= CAP:
                state = 1
            else:
                # Shrink the bracket to the values it holds: nothing lies in
                # (in_max, hi] or (lo, in_min), so the masses and counts
                # above its ends do not change.
                in_min = tl.full((), float("inf"), tl.float32)
                in_max = tl.full((), float("-inf"), tl.float32)
                for start in tl.range(0, vocab_size, BLOCK_SIZE):
                    cols = start + offs
                    mask = cols < vocab_size
                    s = tl.load(logits_row + cols, mask=mask, other=float("-inf")).to(
                        tl.float32
                    )
                    s = s / temperature
                    inside = mask & (s > lo) & (s <= hi)
                    in_min = tl.minimum(
                        in_min, tl.min(tl.where(inside, s, float("inf")))
                    )
                    in_max = tl.maximum(
                        in_max, tl.max(tl.where(inside, s, float("-inf")))
                    )
                hi = in_max
                if in_min == in_max:
                    state = 2
                else:
                    lo = in_min - (in_min - lo) * 0.5
            rounds += 1

        threshold = hi
        limit = tl.full((), 0, tl.int32)
        if state == 1:
            # Compact the bracket's tokens as rank keys, sort and scan them.
            base = tl.full((), 0, tl.int32)
            for start in tl.range(0, vocab_size, BLOCK_SIZE):
                cols = start + offs
                mask = cols < vocab_size
                s = tl.load(logits_row + cols, mask=mask, other=float("-inf")).to(
                    tl.float32
                )
                s = s / temperature
                take = mask & (s > lo) & (s <= hi)
                slots = base + tl.cumsum(take.to(tl.int32), axis=0) - 1
                tl.store(
                    keys_row + slots,
                    _rank_key(s, cols).to(tl.int64, bitcast=True),
                    mask=take,
                )
                base += tl.sum(take.to(tl.int32), axis=0)
            lanes = tl.arange(0, CAP)
            keys = tl.load(keys_row + lanes, mask=lanes < base, other=0).to(
                tl.uint64, bitcast=True
            )
            keys = tl.sort(keys, descending=True)
            s, ids = _rank_key_value(keys)
            valid = lanes < base
            w = tl.where(valid, tl.exp(s - row_max), 0.0)
            before = hi_mass + tl.cumsum(w, axis=0) - w
            last = tl.sum((valid & (before < target)).to(tl.int32), axis=0) - 1
            threshold = tl.max(tl.where(lanes == last, s, float("-inf")))
            limit = tl.max(tl.where(lanes == last, ids, -1))
        else:
            # Every bracket token has scaled logit hi (or the search ran out
            # of rounds): keep the lowest ids among them until the mass
            # reaches the target, bisecting the id of the last one.
            w_tie = tl.exp(hi - row_max)
            need = tl.ceil((target - hi_mass) / w_tie).to(tl.int32)
            need = tl.minimum(tl.maximum(need, 1), lo_count - hi_count)
            id_lo = tl.full((), -1, tl.int32)
            id_hi = tl.full((), vocab_size - 1, tl.int32)
            while id_hi - id_lo > 1:
                id_mid = id_lo + (id_hi - id_lo) // 2
                n = tl.full((), 0, tl.int32)
                for start in tl.range(0, vocab_size, BLOCK_SIZE):
                    cols = start + offs
                    mask = cols < vocab_size
                    s = tl.load(logits_row + cols, mask=mask, other=float("-inf")).to(
                        tl.float32
                    )
                    s = s / temperature
                    n += tl.sum((mask & (s == hi) & (cols <= id_mid)).to(tl.int32))
                take_lower = n >= need
                id_hi = tl.where(take_lower, id_mid, id_hi)
                id_lo = tl.where(take_lower, id_lo, id_mid)
            limit = id_hi
        tl.store(threshold_ptr + row, threshold)
        tl.store(limit_ptr + row, limit)


@triton.jit
def _scan_mass_descending(hist_ptr, above, target, BINS: tl.constexpr):
    """Walk a level's mass bins from the highest key down: the bin where the
    mass reaches ``target``, and the mass above that bin."""
    walk = tl.arange(0, BINS)
    mass = tl.load(hist_ptr + (BINS - 1 - walk))
    reach = (above + tl.cumsum(mass, axis=0)).to(tl.float64) >= target
    first = tl.minimum(tl.min(tl.where(reach, walk, BINS), axis=0), BINS - 1)
    return BINS - 1 - first, above + tl.sum(tl.where(walk < first, mass, 0), axis=0)


@triton.jit
def _scan_count_ascending(hist_ptr, below, need, BINS: tl.constexpr):
    """Walk a level's count bins from the lowest id up: the bin holding the
    ``need``-th token, and the count below that bin."""
    walk = tl.arange(0, BINS)
    count = tl.load(hist_ptr + walk)
    first = tl.min(
        tl.where(below + tl.cumsum(count, axis=0) >= need, walk, BINS), axis=0
    )
    first = tl.minimum(first, BINS - 1)
    return first, below + tl.sum(tl.where(walk < first, count, 0), axis=0)


@triton.jit
def _add_mass_by_bin(hist_ptr, bins, mass, mask, BINS: tl.constexpr):
    """Add each token's mass to its bin with one atomic pair per distinct bin.

    A flat row puts most of a block in one bin of the first level, where
    per-token atomics serialize. Sorting (bin, mass) keys groups each bin's tokens; a bin's
    mass is the prefix sum at its last token minus the one before its first.
    """
    size: tl.constexpr = bins.shape[0]
    # The mass fits the low 41 bits (it is at most 2**40) and the bin the 12
    # above, clear of the sign bit; masked-off tokens sort last under bin BINS
    # and add nothing.
    keys = (tl.where(mask, bins, BINS).to(tl.int64) << 41) | tl.where(mask, mass, 0)
    keys = tl.sort(keys)
    sorted_bins = (keys >> 41).to(tl.int32)
    sorted_mass = keys & 0x1FFFFFFFFFF
    prefix = tl.cumsum(sorted_mass, axis=0)
    lanes = tl.arange(0, size)
    next_bins = tl.gather(sorted_bins, tl.minimum(lanes + 1, size - 1), axis=0)
    prev_bins = tl.gather(sorted_bins, tl.maximum(lanes - 1, 0), axis=0)
    real = sorted_bins < BINS
    last = real & ((lanes == size - 1) | (next_bins != sorted_bins))
    first = real & ((lanes == 0) | (prev_bins != sorted_bins))
    tl.atomic_add(hist_ptr + sorted_bins, prefix, mask=last, sem="relaxed")
    tl.atomic_add(
        hist_ptr + sorted_bins, sorted_mass - prefix, mask=first, sem="relaxed"
    )


@triton.jit
def _top_p_rare_level_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    top_p_pool_ptr,
    rare_rows_ptr,
    rare_row_max_ptr,
    mass_hist_ptr,
    count_hist_ptr,
    state_ptr,
    threshold_ptr,
    limit_ptr,
    logits_row_stride,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BINS: tl.constexpr,
    MAX_RARE_ROWS: tl.constexpr,
    STATE_FIELDS: tl.constexpr,
    ID_SHIFT: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
    LEVEL: tl.constexpr,
):
    """One rare-path launch. Levels 1-3 add each token's fixed-point mass to
    the bin of the next 12, 12 and 8 bits of its ordered scaled logit, among
    the tokens whose higher bits match the boundary's; levels 4-5 count the
    tokens at the boundary's value by the high and low bits of their id;
    level 6 writes the threshold and limit. Every launch extends the
    boundary's prefix by scanning the previous level's histogram; program 0
    hands the prefix to the next launch through the other half of the
    double-buffered state, so no launch reads a field it also writes."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_rows = tl.minimum(tl.load(rare_rows_ptr + MAX_RARE_ROWS), MAX_RARE_ROWS)
    offs = tl.arange(0, BLOCK_SIZE)
    for slot in tl.range(0, num_rows):
        row = tl.load(rare_rows_ptr + slot)
        pool_idx = tl.load(req_pool_indices_ptr + row // NUM_TOKENS_PER_REQ).to(
            tl.int64
        )
        temperature = tl.maximum(
            tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
        )
        top_p = tl.load(top_p_pool_ptr + pool_idx).to(tl.float32)
        row_max = tl.load(rare_row_max_ptr + slot)
        mass_hist = mass_hist_ptr + slot * 3 * BINS
        count_hist = count_hist_ptr + slot * 2 * BINS
        logits_row = logits_ptr + row.to(tl.int64) * logits_row_stride

        # The boundary's prefix: the previous launch's fields, extended by a
        # scan of the level it filled.
        state_in = state_ptr + ((LEVEL + 1) % 2 * MAX_RARE_ROWS + slot) * STATE_FIELDS
        state_out = state_ptr + (LEVEL % 2 * MAX_RARE_ROWS + slot) * STATE_FIELDS
        fields = tl.arange(0, STATE_FIELDS)
        prefix = tl.zeros((STATE_FIELDS,), tl.int64)
        if LEVEL > 2:
            prefix = tl.load(state_in + fields)
        high = tl.sum(tl.where(fields == 0, prefix, 0), axis=0).to(tl.uint32)
        mid = tl.sum(tl.where(fields == 1, prefix, 0), axis=0).to(tl.uint32)
        value = tl.sum(tl.where(fields == 2, prefix, 0), axis=0).to(tl.uint32)
        need = tl.sum(tl.where(fields == 3, prefix, 0), axis=0)
        id_high = tl.sum(tl.where(fields == 4, prefix, 0), axis=0).to(tl.int32)
        below = tl.sum(tl.where(fields == 5, prefix, 0), axis=0)
        above = tl.sum(tl.where(fields == 6, prefix, 0), axis=0)
        target = tl.sum(tl.where(fields == 7, prefix, 0), axis=0).to(
            tl.float64, bitcast=True
        )
        if LEVEL == 2:
            total = tl.sum(tl.load(mass_hist + tl.arange(0, BINS)), axis=0)
            target = top_p.to(tl.float64) * total.to(tl.float64)
            bin1, above = _scan_mass_descending(
                mass_hist, tl.full((), 0, tl.int64), target, BINS
            )
            high = bin1.to(tl.uint32)
        if LEVEL == 3:
            bin2, above = _scan_mass_descending(mass_hist + BINS, above, target, BINS)
            mid = (high << 12) | bin2.to(tl.uint32)
        if LEVEL == 4:
            bin3, above = _scan_mass_descending(
                mass_hist + 2 * BINS, above, target, BINS
            )
            value = (mid << 8) | bin3.to(tl.uint32)
            # Ties share one fixed-point mass: keep the lowest ids among them
            # until the mass reaches the target.
            tie_mass = _fixed_mass(_from_ordered_bits(value), row_max).to(tl.float64)
            need = tl.maximum(tl.ceil((target - above.to(tl.float64)) / tie_mass), 1.0)
            need = need.to(tl.int64)
        if LEVEL == 5:
            bin4, below = _scan_count_ascending(
                count_hist, tl.full((), 0, tl.int64), need, BINS
            )
            id_high = bin4
        if (LEVEL >= 2) & (LEVEL < 6) & (pid == 0):
            out = tl.where(fields == 0, high.to(tl.int64), 0)
            out = tl.where(fields == 1, mid.to(tl.int64), out)
            out = tl.where(fields == 2, value.to(tl.int64), out)
            out = tl.where(fields == 3, need, out)
            out = tl.where(fields == 4, id_high.to(tl.int64), out)
            out = tl.where(fields == 5, below, out)
            out = tl.where(fields == 6, above, out)
            out = tl.where(fields == 7, target.to(tl.int64, bitcast=True), out)
            tl.store(state_out + fields, out)

        if LEVEL == 6:
            if slot % num_programs == pid:
                bin5, _ = _scan_count_ascending(count_hist + BINS, below, need, BINS)
                tl.store(threshold_ptr + row, _from_ordered_bits(value))
                tl.store(limit_ptr + row, (id_high << ID_SHIFT) | bin5)
        else:
            for start in tl.range(
                pid * BLOCK_SIZE, vocab_size, num_programs * BLOCK_SIZE
            ):
                cols = start + offs
                mask = cols < vocab_size
                s = tl.load(logits_row + cols, mask=mask, other=float("-inf")).to(
                    tl.float32
                )
                s = s / temperature
                key = _ordered_bits(s)
                if LEVEL == 1:
                    _add_mass_by_bin(
                        mass_hist,
                        (key >> 20).to(tl.int32),
                        _fixed_mass(s, row_max),
                        mask,
                        BINS,
                    )
                elif LEVEL == 2:
                    # The boundary's first-level bin spreads over these bits:
                    # per-token atomics contend little and skip the sort.
                    tl.atomic_add(
                        mass_hist + BINS + ((key >> 8) & 0xFFF),
                        _fixed_mass(s, row_max),
                        mask=mask & ((key >> 20) == high),
                        sem="relaxed",
                    )
                elif LEVEL == 3:
                    tl.atomic_add(
                        mass_hist + 2 * BINS + (key & 0xFF),
                        _fixed_mass(s, row_max),
                        mask=mask & ((key >> 8) == mid),
                        sem="relaxed",
                    )
                elif LEVEL == 4:
                    tl.atomic_add(
                        count_hist + (cols >> ID_SHIFT),
                        tl.full((BLOCK_SIZE,), 1, tl.int64),
                        mask=mask & (key == value),
                        sem="relaxed",
                    )
                else:
                    tl.atomic_add(
                        count_hist + BINS + (cols & ((1 << ID_SHIFT) - 1)),
                        tl.full((BLOCK_SIZE,), 1, tl.int64),
                        mask=mask & (key == value) & ((cols >> ID_SHIFT) == id_high),
                        sem="relaxed",
                    )


def _num_blocks(vocab_size: int) -> int:
    return triton.cdiv(vocab_size, _BLOCK_SIZE)


def _cap(vocab_size: int) -> int:
    """Most tokens the per-row search sorts at once: a bracket never holds more
    tokens than the row."""
    return min(_CAP, triton.next_power_of_2(vocab_size))


def _id_shift(vocab_size: int) -> int:
    """Low id bits the second id level counts; the first counts the rest."""
    return max(0, (vocab_size - 1).bit_length() - 12)


def top_p_threshold_scratch_shapes(
    max_rows: int, vocab_size: int
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Shape and dtype of each scratch tensor of ``top_p_thresholds_from_pools``.
    Host arithmetic only.

    ``block_stats`` and ``candidates`` are per row; the rare-path tensors are
    per histogram slot: ``rare_rows`` (the slots' rows, then their count),
    ``rare_row_max``, the ``mass_hist`` / ``count_hist`` levels and the
    double-buffered ``rare_state`` the level launches pass on.
    """
    num_blocks = _num_blocks(vocab_size)
    # The per-row search reuses a row's candidates as its sort buffer.
    width = max(num_blocks * _NUM_CANDIDATES, _cap(vocab_size))
    return {
        "block_stats": ((max_rows, 2 * num_blocks), torch.float32),
        "candidates": ((max_rows, width), torch.int64),
        "rare_rows": ((_MAX_RARE_ROWS + 1,), torch.int32),
        "rare_row_max": ((_MAX_RARE_ROWS,), torch.float32),
        "mass_hist": ((_MAX_RARE_ROWS, 3, _HIST_BINS), torch.int64),
        "count_hist": ((_MAX_RARE_ROWS, 2, _HIST_BINS), torch.int64),
        "rare_state": ((2, _MAX_RARE_ROWS, _RARE_STATE_FIELDS), torch.int64),
    }


def top_p_thresholds_from_pools(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    scratch: dict[str, torch.Tensor],
    threshold: torch.Tensor,
    limit: torch.Tensor,
    rare: torch.Tensor,
    *,
    num_tokens_per_req: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's exact top-p nucleus as ``(threshold, limit)``.

    Args:
        logits: ``[rows, V]`` logits, stride-1 vocabulary.
        req_pool_indices: ``[rows / num_tokens_per_req]`` int32 pool slot of
            each request.
        temperature_pool: ``[pool]`` fp32 temperature.
        top_p_pool: ``[pool]`` fp32 top-p.
        scratch: tensors named and shaped by ``top_p_threshold_scratch_shapes``
            (the per-row ones with at least ``rows`` rows).
        threshold: ``[rows]`` fp32 output; scaled logit of the last kept token
            (``-inf`` for rows that keep every token).
        limit: ``[rows]`` int32 output; id of the last kept token.
        rare: ``[rows]`` int32 output; 0 where the candidates held the
            boundary, 1 for a histogram-slot row, 2 for a per-row search.
        num_tokens_per_req: Rows per request.

    Returns:
        ``(threshold, limit)``: token ``t`` of row ``r`` is kept iff
        ``logits[r, t] / T > threshold[r]``, or it equals it and
        ``t <= limit[r]``.
    """
    rows, vocab_size = logits.shape
    if logits.stride(-1) != 1:
        raise ValueError("top_p_thresholds_from_pools needs a stride-1 vocabulary")
    if vocab_size > _HIST_BINS << 12:
        raise ValueError(
            f"top_p_thresholds_from_pools counts token ids in two 12-bit levels; "
            f"got vocab_size={vocab_size}"
        )
    for name, (shape, dtype) in top_p_threshold_scratch_shapes(
        rows, vocab_size
    ).items():
        tensor = scratch[name]
        per_row = name in ("block_stats", "candidates")
        if (
            tensor.dtype != dtype
            or not tensor.is_contiguous()
            or (tensor.shape[0] < rows if per_row else tensor.shape[0] != shape[0])
            or tuple(tensor.shape[1:]) != shape[1:]
        ):
            raise ValueError(
                f"scratch {name} must be contiguous {dtype} {shape} (rows >= {rows}), "
                f"got {tensor.dtype} {tuple(tensor.shape)}"
            )
    if rows == 0:
        return threshold[:0], limit[:0]
    num_blocks = _num_blocks(vocab_size)
    block_stats, candidates = scratch["block_stats"], scratch["candidates"]
    rare_rows, rare_row_max = scratch["rare_rows"], scratch["rare_row_max"]
    mass_hist, count_hist = scratch["mass_hist"], scratch["count_hist"]
    _top_p_candidates_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        temperature_pool,
        top_p_pool,
        block_stats,
        candidates,
        rare_rows,
        logits.stride(0),
        block_stats.stride(0),
        candidates.stride(0),
        vocab_size=vocab_size,
        BLOCK_SIZE=_BLOCK_SIZE,
        NUM_CANDIDATES=_NUM_CANDIDATES,
        MAX_RARE_ROWS=_MAX_RARE_ROWS,
        NUM_TOKENS_PER_REQ=num_tokens_per_req,
        TOP_P_EPS=_TOP_P_EPS,
        num_warps=4,
    )
    _top_p_select_kernel[(rows,)](
        req_pool_indices,
        top_p_pool,
        block_stats,
        candidates,
        threshold,
        limit,
        rare,
        rare_rows,
        rare_row_max,
        mass_hist,
        count_hist,
        block_stats.stride(0),
        candidates.stride(0),
        num_blocks=num_blocks,
        NUM_BLOCKS_PAD=triton.next_power_of_2(num_blocks),
        NUM_CANDIDATES=_NUM_CANDIDATES,
        FEW_CANDIDATES=_FEW_CANDIDATES,
        FEW_PAD=triton.next_power_of_2(num_blocks * _FEW_CANDIDATES),
        ALL_PAD=triton.next_power_of_2(num_blocks * _NUM_CANDIDATES),
        BINS=_HIST_BINS,
        MAX_RARE_ROWS=_MAX_RARE_ROWS,
        NUM_TOKENS_PER_REQ=num_tokens_per_req,
        TOP_P_EPS=_TOP_P_EPS,
        RARE_HISTOGRAM=_RARE_HISTOGRAM,
        RARE_SEARCH=_RARE_SEARCH,
        num_warps=16,
    )
    for level in range(1, 7):
        _top_p_rare_level_kernel[(_HIST_PROGRAMS,)](
            logits,
            req_pool_indices,
            temperature_pool,
            top_p_pool,
            rare_rows,
            rare_row_max,
            mass_hist,
            count_hist,
            scratch["rare_state"],
            threshold,
            limit,
            logits.stride(0),
            vocab_size=vocab_size,
            BLOCK_SIZE=_BLOCK_SIZE,
            BINS=_HIST_BINS,
            MAX_RARE_ROWS=_MAX_RARE_ROWS,
            STATE_FIELDS=_RARE_STATE_FIELDS,
            ID_SHIFT=_id_shift(vocab_size),
            NUM_TOKENS_PER_REQ=num_tokens_per_req,
            LEVEL=level,
            num_warps=8,
        )
    _top_p_threshold_exact_kernel[(rows,)](
        logits,
        req_pool_indices,
        temperature_pool,
        top_p_pool,
        rare,
        candidates,
        threshold,
        limit,
        logits.stride(0),
        candidates.stride(0),
        vocab_size=vocab_size,
        BLOCK_SIZE=_BLOCK_SIZE,
        CAP=_cap(vocab_size),
        NUM_PIVOTS=_NUM_PIVOTS,
        MAX_ROUNDS=_MAX_ROUNDS,
        NUM_TOKENS_PER_REQ=num_tokens_per_req,
        RARE_SEARCH=_RARE_SEARCH,
        num_warps=8,
    )
    return threshold[:rows], limit[:rows]
