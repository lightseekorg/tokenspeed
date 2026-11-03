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

"""CSA2 row selection from scored logits for GFX1250."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

__all__ = ["launch_gluon_dsv41_index_topk_select_gfx1250"]

# Radix-select digit split of the 32-bit order key: 11 + 11 + 10 bits.
_SELECT_RADIX_BITS = 11
_SELECT_WARPS = 16


def _select_launch_metadata(grid, kernel, args):
    """Each pass re-reads the logits row; candidates add one id load per pick."""
    queries = grid[0]
    width = args["width"]
    topk = args["topk"]
    return {
        "name": kernel.name,
        "bytes": queries * (4 * width * 4 + topk * (4 + 8)),
    }


@gluon.jit
def _select_order_key(values):
    # Monotone map from FP32 to uint32: larger value -> larger key. -0.0 maps
    # to +0.0 so the two zeros tie and keep the lowest columns.
    bits = values.to(gl.uint32, bitcast=True)
    bits = gl.where(
        bits == 0x80000000, gl.full(bits.shape, 0, gl.uint32, bits.type.layout), bits
    )
    flip = gl.where(
        (bits >> 31) != 0,
        gl.full(bits.shape, 0xFFFFFFFF, gl.uint32, bits.type.layout),
        gl.full(bits.shape, 0x80000000, gl.uint32, bits.type.layout),
    )
    return bits ^ flip


@gluon.jit
def _select_radix_pass(
    row,
    width,
    take,
    prefix,
    above,
    remaining,
    hist_smem,
    ones,
    d,
    SHIFT: gl.constexpr,
    DIGIT_BITS: gl.constexpr,
    FIRST: gl.constexpr,
    BLOCK: gl.constexpr,
    BINS: gl.constexpr,
    L: gl.constexpr,
    LBIN: gl.constexpr,
):
    """Histogram one key digit of the live keys matching ``prefix``."""
    hist_smem.store(gl.zeros([BINS], gl.int32, layout=LBIN))
    for c0 in range(0, width, BLOCK):
        col = c0 + gl.arange(0, BLOCK, layout=L)
        values = gl.load(row + col, mask=col < width, other=-float("inf"))
        key = _select_order_key(values)
        live = values > -float("inf")
        if not FIRST:
            live = live & ((key >> (SHIFT + DIGIT_BITS)) == prefix)
        digit = ((key >> SHIFT) & ((1 << DIGIT_BITS) - 1)).to(gl.int32)
        hist_smem.atomic_scatter_add(ones, digit, axis=0, mask=live)
    hist = hist_smem.load(LBIN)
    if FIRST:
        remaining = gl.minimum(take, gl.sum(hist, 0))
    # at_or_above[d] counts keys whose digit is >= d; the K-th key's digit is
    # the largest d with at_or_above[d] >= remaining.
    at_or_above = gl.cumsum(hist, 0, reverse=True)
    pick = gl.maximum(gl.sum((at_or_above >= remaining).to(gl.int32), 0) - 1, 0)
    higher = gl.sum(gl.where(d > pick, hist, 0), 0)
    if FIRST:
        prefix = pick.to(gl.uint32)
    else:
        prefix = (prefix << DIGIT_BITS) | pick.to(gl.uint32)
    return prefix, above + higher, remaining - higher


@gluon.jit(
    launch_metadata=_select_launch_metadata,
    do_not_specialize=(
        "logits_stride",
        "cand_stride",
        "out_stride",
        "row_limits_stride",
        "width",
        "topk",
        "take",
    ),
)
def gluon_dsv41_index_topk_select_gfx1250(
    logits,  # [Q, width] fp32, -inf = unscored
    candidates,  # [Q, blocks] int32/int64 block ids, or logits when unused
    row_limits,  # [Q] live widths, or logits when unused
    row_out,  # [Q, TOPK] int32
    row_lens,  # [Q] int32
    logits_stride,
    cand_stride,
    out_stride,
    row_limits_stride,
    width,
    topk,  # row_out width
    take,  # min(topk, width)
    TOPK: gl.constexpr,  # power-of-two capacity >= topk, at least 256
    HAS_CANDIDATES: gl.constexpr,
    HAS_LIMITS: gl.constexpr,
    RADIX_BITS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Top-``take`` finite logits of one query row as position-sorted ids.

    A three-pass MSD radix select finds the order key of the K-th largest
    finite logit (K = min(take, finite count)); a column-order compaction then
    keeps every larger key plus the lowest-column ties, and an all-pairs rank
    sorts the picked logical ids. Positions >= K are written as -1.
    """
    BLOCK: gl.constexpr = 4 * 32 * NUM_WARPS
    BINS: gl.constexpr = 1 << RADIX_BITS
    L: gl.constexpr = gl.BlockedLayout([4], [32], [NUM_WARPS], [0])
    LBIN: gl.constexpr = gl.BlockedLayout(
        [BINS // (32 * NUM_WARPS)], [32], [NUM_WARPS], [0]
    )
    SL: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    q = gl.program_id(0).to(gl.int64)
    row = logits + q * logits_stride
    if HAS_LIMITS:
        # Columns at or past the row's live width are -inf: skip them.
        width = gl.minimum(
            width,
            gl.maximum(gl.load(row_limits + q * row_limits_stride).to(gl.int64), 0),
        ).to(gl.int32)
    hist_smem = gl.allocate_shared_memory(gl.int32, [BINS], SL)
    ones = gl.full([BLOCK], 1, gl.int32, layout=L)
    d = gl.arange(0, BINS, layout=LBIN)

    # MSD radix select: after each pass, ``prefix`` holds the fixed high bits
    # of the K-th key, ``remaining`` its rank among keys sharing them and
    # ``above`` the count of larger keys.
    LOW_BITS: gl.constexpr = 32 - 2 * RADIX_BITS
    prefix, above, remaining = _select_radix_pass(
        row,
        width,
        take,
        0,
        0,
        0,
        hist_smem,
        ones,
        d,
        32 - RADIX_BITS,
        RADIX_BITS,
        True,
        BLOCK,
        BINS,
        L,
        LBIN,
    )
    prefix, above, remaining = _select_radix_pass(
        row,
        width,
        take,
        prefix,
        above,
        remaining,
        hist_smem,
        ones,
        d,
        LOW_BITS,
        RADIX_BITS,
        False,
        BLOCK,
        BINS,
        L,
        LBIN,
    )
    prefix, above, remaining = _select_radix_pass(
        row,
        width,
        take,
        prefix,
        above,
        remaining,
        hist_smem,
        ones,
        d,
        0,
        LOW_BITS,
        False,
        BLOCK,
        BINS,
        L,
        LBIN,
    )
    threshold = prefix
    count = above + remaining

    # Column-order compaction: keys above the threshold plus the first
    # ``remaining`` ties (equal-score ties may pick any equivalent subset).
    picked_smem = gl.allocate_shared_memory(gl.int64, [2 * TOPK], SL)
    LP: gl.constexpr = gl.BlockedLayout(
        [2 * TOPK // (32 * NUM_WARPS)], [32], [NUM_WARPS], [0]
    )
    picked_smem.store(gl.full([2 * TOPK], 0x7FFFFFFFFFFFFFFF, gl.int64, layout=LP))
    # One packed scan per chunk counts both: low 16 bits the keys above the
    # threshold, high 16 bits the ties (a chunk holds fewer than 2**16).
    n_above = 0
    n_ties = 0
    for c0 in range(0, width, BLOCK):
        col = c0 + gl.arange(0, BLOCK, layout=L)
        values = gl.load(row + col, mask=col < width, other=-float("inf"))
        key = _select_order_key(values)
        live = values > -float("inf")
        is_above = live & (key > threshold)
        is_tie = live & (key == threshold)
        packed = is_above.to(gl.int32) | (is_tie.to(gl.int32) << 16)
        before = gl.cumsum(packed, 0) - packed
        ties_before = n_ties + (before >> 16)
        take_it = is_above | (is_tie & (ties_before < remaining))
        slot = n_above + (before & 0xFFFF) + gl.minimum(ties_before, remaining)
        if HAS_CANDIDATES:
            block = gl.load(
                candidates + q * cand_stride + col // 8, mask=take_it, other=-1
            ).to(gl.int64)
            logical = block * 8 + (col % 8).to(gl.int64)
        else:
            logical = col.to(gl.int64)
        # Unpicked lanes write the scratch half [TOPK, 2*TOPK).
        picked_smem.scatter(logical, gl.where(take_it, slot, TOPK + col % TOPK), axis=0)
        total = gl.sum(packed, 0)
        n_above += total & 0xFFFF
        n_ties += total >> 16

    # Rank sort of the picked ids: ids are unique, unpicked slots hold the
    # int64 max and rank after them.
    LT: gl.constexpr = gl.BlockedLayout([1, 64], [32, 1], [NUM_WARPS, 1], [1, 0])
    LR: gl.constexpr = gl.SliceLayout(1, LT)
    LC: gl.constexpr = gl.SliceLayout(0, LT)
    mine = picked_smem.slice(0, TOPK).load(LR)
    i = gl.arange(0, TOPK, layout=LR)
    rank = i
    # Column order already is position order without candidates, and with
    # candidate blocks listed in ascending order; sort only otherwise.
    previous = gl.gather(mine, gl.maximum(i - 1, 0), axis=0)
    descents = gl.sum(((i > 0) & (i < count) & (mine < previous)).to(gl.int32), 0)
    if descents > 0:
        rank = gl.zeros([TOPK], gl.int32, layout=LR)
        # Slots past ``count`` hold the int64 max and rank no picked id, so
        # skip them.
        chunks = picked_smem.reinterpret(shape=[2 * TOPK // 64, 64], layout=SL)
        for j in range(0, gl.cdiv(count, 64)):
            other = chunks.index(j).load(LC)
            rank += gl.sum(
                (gl.expand_dims(other, 0) < gl.expand_dims(mine, 1)).to(gl.int32),
                axis=1,
            )
    out = row_out + q * out_stride
    gl.store(out + rank, mine.to(gl.int32), mask=i < count)
    gl.store(
        out + i,
        gl.full([TOPK], -1, gl.int32, layout=LR),
        mask=(i >= count) & (i < topk),
    )
    gl.store(row_lens + q, count)


def launch_gluon_dsv41_index_topk_select_gfx1250(
    logits: torch.Tensor,
    candidates: torch.Tensor | None,
    live_widths: torch.Tensor | None,
    topk: int,
    row_out: torch.Tensor,
    row_lens: torch.Tensor,
) -> None:
    """Select CSA2 rows from scored logits, position-sorted.

    Args:
        logits: FP32 [Q, width] scores; -inf marks unscored rows.
        candidates: None, or int32/int64 [Q, blocks] block ids whose 8-row
            blocks map logits column c to logical row
            ``candidates[q, c // 8] * 8 + c % 8``.
        live_widths: None, or strided int32/int64 [Q] counts of leading columns
            that may be finite; row q's columns at or past ``live_widths[q]`` must be
            -inf and are not read. Full selection passes the visible lengths, so
            a page-table-capacity row costs its context, not its capacity.
        topk: Selection capacity in [1, 1024]; V4.1 uses 512.
        row_out: Int32 [Q, topk] destination. Row q receives the logical ids of
            its ``min(topk, width)`` largest finite logits in ascending order,
            then -1 padding. Equal-score boundary ties keep the lowest columns.
        row_lens: Int32 [Q] destination for the number of ids written.

    Returns:
        None.
    """
    queries, width = logits.shape
    if not queries:
        return
    topk = int(topk)
    if not 1 <= topk <= 1024:
        raise ValueError(f"select topk must be in [1, 1024], got {topk}")
    if (
        logits.stride(1) != 1
        or row_out.stride(1) != 1
        or row_lens.stride(0) != 1
        or (candidates is not None and candidates.stride(1) != 1)
    ):
        raise ValueError("select requires unit inner strides")
    # The kernel writes row_out[q, :topk] and row_lens[q], and reads
    # candidates[q, col // 8] for every logits column.
    if row_out.shape[0] != queries or row_out.shape[1] < topk:
        raise ValueError(f"row_out must be [{queries}, >= {topk}]")
    if row_lens.shape != (queries,):
        raise ValueError(f"row_lens must be [{queries}]")
    if candidates is not None and (
        candidates.shape[0] != queries or candidates.shape[1] * 8 < width
    ):
        raise ValueError(f"candidates must be [{queries}, >= {-(-width // 8)}]")
    if live_widths is not None and (
        live_widths.shape != (queries,)
        or live_widths.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(f"live_widths must be int32/int64 [{queries}]")
    cand = logits if candidates is None else candidates
    gluon_dsv41_index_topk_select_gfx1250[(queries,)](
        logits,
        cand,
        logits if live_widths is None else live_widths,
        row_out,
        row_lens,
        logits.stride(0),
        0 if candidates is None else candidates.stride(0),
        row_out.stride(0),
        0 if live_widths is None else live_widths.stride(0),
        width,
        topk,
        min(topk, width),
        TOPK=max(256, triton.next_power_of_2(topk)),
        HAS_CANDIDATES=candidates is not None,
        HAS_LIMITS=live_widths is not None,
        RADIX_BITS=_SELECT_RADIX_BITS,
        NUM_WARPS=_SELECT_WARPS,
        num_warps=_SELECT_WARPS,
    )
