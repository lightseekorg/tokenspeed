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

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import tl, triton


@triton.jit
def _load_keys(row, start, n, latest, BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
    # Order-preserving uint32 keys; invalid entries are -inf scores.
    idx = start + tl.arange(0, BLOCK)
    in_range = idx < n
    if BLOCKS:
        # Block score is the max of its 8 rows; the latest visible block with
        # a valid score is forced in.
        x = tl.max(
            tl.load(
                row + idx[:, None] * 8 + tl.arange(0, 8)[None, :],
                mask=in_range[:, None],
                other=-float("inf"),
            ),
            axis=1,
        )
        x = tl.where((idx == latest) & (x > -float("inf")), float("inf"), x)
    else:
        x = tl.load(row + idx, mask=in_range, other=-float("inf"))
    valid = in_range & (x > -float("inf"))
    bits = x.to(tl.uint32, bitcast=True)
    sign = (bits >> 31) != 0
    key = tl.where(sign, ~bits, bits | 0x80000000)
    return idx, key, valid


@triton.jit
def dsv41_index_select_gfx950(
    logits,
    visible,
    candidates,
    row_out,
    row_lens,
    block_out,
    block_lens,
    width,
    stride_logits,
    stride_candidates,
    stride_row_out,
    stride_block_out,
    topk,
    candidate_topk,
    row_cap,
    block_cap,
    HAS_CANDIDATES: tl.constexpr,
    SORT_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
):
    query = tl.program_id(0)
    BLOCKS = tl.program_id(1) == 1
    row = logits + query.to(tl.int64) * stride_logits
    seen = tl.load(visible + query).to(tl.int32)
    latest = tl.where(seen > 0, (seen - 1) // 8, -1)
    if HAS_CANDIDATES:
        # Rows past the last valid (>= 0) candidate block are never scored;
        # padded candidate lists are mostly -1 at short contexts.
        slot = tl.arange(0, CANDIDATE_BLOCK)
        listed = tl.load(
            candidates + query.to(tl.int64) * stride_candidates + slot,
            mask=slot < width // 8,
            other=-1,
        )
        n = tl.minimum(width, (tl.max(tl.where(listed >= 0, slot, -1)) + 1) * 8)
        k = topk
        out = row_out + query.to(tl.int64) * stride_row_out
        lens = row_lens + query
    else:
        # Full rows past the visible length are never scored.
        n = tl.minimum(width, tl.maximum(seen, 0))
        k = topk
        out = row_out + query.to(tl.int64) * stride_row_out
        lens = row_lens + query
        if BLOCKS:
            n = tl.minimum(width // 8, (n + 7) // 8)
            k = candidate_topk
            out = block_out + query.to(tl.int64) * stride_block_out
            lens = block_lens + query

    # MSB-first radix select of the k-th largest key, 8 bits per pass.
    bins = tl.arange(0, 256)
    prefix = tl.full([], 0, tl.uint32)
    high = tl.full([], 0, tl.uint32)
    remaining = k
    total = 0
    for p in tl.static_range(4):
        hist = tl.zeros([256], dtype=tl.int32)
        for start in range(0, n, BLOCK):
            if BLOCKS:
                _, key, valid = _load_keys(row, start, n, latest, True, BLOCK)
            else:
                _, key, valid = _load_keys(row, start, n, latest, False, BLOCK)
            match = valid & ((key & high) == prefix)
            bucket = ((key >> (24 - 8 * p)) & 255).to(tl.int32)
            hist += tl.histogram(bucket, 256, mask=match)
        if p == 0:
            total = tl.sum(hist)
        at_least = tl.sum(hist) - tl.cumsum(hist, 0) + hist
        digit = tl.max(tl.where(at_least >= remaining, bins, 0))
        above = tl.sum(tl.where(bins == digit, at_least - hist, 0))
        remaining -= above
        prefix |= digit.to(tl.uint32) << (24 - 8 * p)
        high |= tl.full([], 255, tl.uint32) << (24 - 8 * p)
    take_all = total <= k

    # Emit in index order: everything above the threshold, then the first
    # remaining ties.
    selected = 0
    ties = 0
    for start in range(0, n, BLOCK):
        if BLOCKS:
            idx, key, valid = _load_keys(row, start, n, latest, True, BLOCK)
        else:
            idx, key, valid = _load_keys(row, start, n, latest, False, BLOCK)
        equal = (valid & (key == prefix)).to(tl.int32)
        tie_rank = ties + tl.cumsum(equal, 0) - equal
        take = valid & ((key > prefix) | ((equal != 0) & (tie_rank < remaining)))
        take = take | (take_all & valid)
        chosen = take.to(tl.int32)
        position = selected + tl.cumsum(chosen, 0) - chosen
        logical = idx
        if HAS_CANDIDATES:
            block = tl.load(
                candidates + query.to(tl.int64) * stride_candidates + idx // 8,
                mask=take,
                other=-1,
            )
            logical = block * 8 + idx % 8
        tl.store(out + position, logical, mask=take)
        selected += tl.sum(chosen)
        ties += tl.sum(equal)
    tl.store(lens, selected)
    # Pad the rest of the destination row, so callers need not pre-fill it.
    cap = tl.where(BLOCKS, block_cap, row_cap)
    for start in range(0, cap, BLOCK):
        pad = start + tl.arange(0, BLOCK)
        tl.store(out + pad, -1, mask=(pad >= selected) & (pad < cap))
    if candidate_topk == 0:
        tl.store(block_lens + query, 0)

    if SORT_ROWS:
        # Candidate blocks need not be ordered: sort the selected logical rows.
        tl.debug_barrier()
        slots = tl.arange(0, 512)
        ids = tl.load(out + slots, mask=slots < selected, other=2147483647)
        ids = tl.sort(ids)
        tl.store(out + slots, tl.where(slots < selected, ids, -1), mask=slots < k)


def launch_dsv41_index_select_gfx950(
    logits,
    visible,
    candidates,
    topk,
    candidate_topk,
    row_out,
    row_lens,
    block_out,
    block_lens,
):
    queries, width = logits.shape
    if not queries:
        return
    has_candidates = candidates is not None
    dsv41_index_select_gfx950[(queries, 2 if candidate_topk else 1)](
        logits,
        visible,
        candidates if has_candidates else visible,
        row_out,
        row_lens,
        block_out,
        block_lens,
        width,
        logits.stride(0),
        candidates.stride(0) if has_candidates else 0,
        row_out.stride(0),
        block_out.stride(0) if block_out.ndim == 2 and block_out.shape[1] else 0,
        min(int(topk), width),
        min(int(candidate_topk), width // 8),
        row_out.shape[1],
        block_out.shape[1] if block_out.ndim == 2 else 0,
        HAS_CANDIDATES=has_candidates,
        SORT_ROWS=has_candidates,
        BLOCK=2048,
        CANDIDATE_BLOCK=triton.next_power_of_2(max(width // 8, 1)),
        num_warps=8,
    )
