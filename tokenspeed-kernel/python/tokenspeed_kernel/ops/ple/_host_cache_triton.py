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

"""Staged hot-cache gather and fill Triton kernels used by host_cache.py."""

from tokenspeed_kernel._triton import tl, triton


@triton.jit(do_not_specialize=["n", "vocab_start", "vocab_end", "row_offset"])
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
    vocab_start,
    vocab_end,
    row_offset,
    scale,
    SLOT_BITS: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
):
    """Gather the tail rows the CUDA tile gather leaves over.

    The tile gather covers whole 64-token tiles; this launch takes the
    remaining rows. A row whose id falls in this shard's ``vocab_start`` ..
    ``vocab_end`` range probes its direct-mapped slot with a multiplicative
    hash of the local id and reads the row from the hot cache on a tag hit
    or straight from the pinned host table on a miss; ``out`` receives the
    values either way, dequantized by ``scale`` for an FP8 table. Rows
    outside the shard write zeros and take no part in the refill.

    A miss also records its slot in ``miss_slot``, stages the raw
    storage-dtype row in ``staged`` and bids for the slot in ``winners``
    with an atomic max of ``row + row_offset`` -- ``row_offset`` shifts the
    tail's row indices to be global across both gather kernels -- electing
    one row per slot to refill the cache on the fill stream.
    """

    row = tl.program_id(0) * R + tl.arange(0, R)
    gid = tl.load(ids + row, row < n, other=-1)
    owned = (row < n) & (gid >= vocab_start) & (gid < vocab_end)
    local_id = tl.where(owned, gid - vocab_start, -1).to(tl.int32)
    slot = (local_id.to(tl.uint32) * 2654435761) >> (32 - SLOT_BITS)
    tag = tl.load(tags + slot, owned, other=-1)
    hit = owned & (tag == local_id)
    col = tl.arange(0, C)
    host = addr.to(tl.int64).to(hot.dtype)
    host_offset = tl.where(owned, local_id, 0).to(tl.int64) * D
    src = tl.where(hit, hot + slot.to(tl.int64) * D, host + host_offset)
    raw = tl.load(
        src[:, None] + col[None, :], owned[:, None] & (col[None, :] < D), other=0.0
    )
    values = (
        raw.to(tl.float32) * scale if hot.dtype.element_ty == tl.float8e4nv else raw
    )
    tl.store(
        out + row[:, None].to(tl.int64) * D + col[None, :],
        values,
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


@triton.jit(do_not_specialize=["rows", "vocab_start"])
def _ple_fill_staged_kernel(
    ids,
    staged,
    miss_slot,
    winners,
    tags,
    hot,
    rows,
    vocab_start,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Refill the hot cache from staged miss rows on the side fill stream.

    A row copies its staged bytes into ``hot`` and publishes its local id in
    ``tags`` only while ``winners`` still names it the slot's elected row,
    so a superseded bid writes nothing. The moved rows are raw FP8 or BF16
    storage bytes; any dequantization happens on the gather side. Each
    program covers ``R`` rows of ``D`` elements in ``BLOCK``-wide accesses.
    """

    row = tl.program_id(0) * R + tl.arange(0, R)
    slot = tl.load(miss_slot + row, row < rows, other=-1)
    win = (slot >= 0) & (tl.load(winners + slot, slot >= 0, other=-1) == row)
    if tl.sum(win.to(tl.int32), 0) > 0:
        col = tl.arange(0, BLOCK)
        mask = win[:, None] & (col[None, :] < D)
        raw = tl.load(
            staged + row[:, None].to(tl.int64) * D + col[None, :], mask, other=0.0
        )
        gid = tl.load(ids + row, win, other=0)
        local_id = (gid - vocab_start).to(tl.int32)
        tl.store(hot + slot[:, None].to(tl.int64) * D + col[None, :], raw, mask)
        tl.store(tags + slot, local_id, win)
