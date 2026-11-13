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

from tokenspeed_kernel._triton import tl, triton


@triton.jit(do_not_specialize=["n", "vocab_end", "row_offset"])
def _ple_gather_staged_tail_kernel(
    addr,
    ids,
    tags,
    hot,
    out,
    staged,
    miss_slot,
    winners,
    n,
    vocab_end,
    row_offset,
    scale,
    SLOT_BITS: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    gid = tl.load(ids + row, row < n, other=-1)
    owned = (row < n) & (gid >= 0) & (gid < vocab_end)
    slot = (gid.to(tl.uint32) * 2654435761) >> (32 - SLOT_BITS)
    tag = tl.load(tags + slot, owned, other=-1)
    hit = owned & (tag == gid)
    col = tl.arange(0, C)
    host = addr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    src = tl.where(hit, hot + slot.to(tl.int64) * D, host + tl.where(owned, gid, 0) * D)
    raw = tl.load(
        src[:, None] + col[None, :], owned[:, None] & (col[None, :] < D), other=0.0
    )
    tl.store(
        out + row[:, None].to(tl.int64) * D + col[None, :],
        raw.to(tl.float32) * scale,
        (row[:, None] < n) & (col[None, :] < D),
    )
    miss = owned & ~hit
    tl.store(miss_slot + row, tl.where(miss, slot.to(tl.int32), -1), row < n)
    tl.store(
        staged + row[:, None].to(tl.int64) * D + col[None, :],
        raw,
        miss[:, None] & (col[None, :] < D),
    )
    tl.atomic_max(winners + slot, row + row_offset, mask=miss, sem="relaxed")


@triton.jit(do_not_specialize=["rows"])
def _ple_fill_staged_kernel(
    ids,
    staged,
    miss_slot,
    winners,
    tags,
    hot,
    rows,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    slot = tl.load(miss_slot + row, row < rows, other=-1)
    win = (slot >= 0) & (tl.load(winners + slot, slot >= 0, other=-1) == row)
    if tl.sum(win.to(tl.int32), 0) > 0:
        col = tl.arange(0, BLOCK)
        mask = win[:, None] & (col[None, :] < D)
        raw = tl.load(
            staged + row[:, None].to(tl.int64) * D + col[None, :], mask, other=0.0
        )
        gid = tl.load(ids + row, win, other=0).to(tl.int32)
        tl.store(hot + slot[:, None].to(tl.int64) * D + col[None, :], raw, mask)
        tl.store(tags + slot, gid, win)
