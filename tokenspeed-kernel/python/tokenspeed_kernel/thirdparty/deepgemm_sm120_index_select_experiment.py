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

"""Length-bounded exact selection from native FP4 logits."""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _index_finish_parts,
    _index_merge,
    _index_unpack,
)


@triton.jit
def _select_scores(
    S,
    T,
    V,
    RS,
    RI,
    BS,
    BI,
    SS,
    TS0,
    TS1,
    VS,
    WIDTH,
    PAGES,
    PARTS: tl.constexpr,
    RK: tl.constexpr,
    BK: tl.constexpr,
    BLOCKS: tl.constexpr,
    TILE: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    visible = tl.minimum(tl.maximum(tl.load(V + row * VS), 0), WIDTH)
    span = tl.cdiv(visible, PARTS * 8) * 8
    begin = part * span
    end = tl.minimum(begin + span, visible)
    ra = tl.full((RK,), 0, tl.uint64)
    if BLOCKS:
        ba = tl.full((BK,), 0, tl.uint64)
    for start in range(begin, end, TILE):
        ids = start + tl.arange(0, TILE)
        valid = ids < end
        pages = tl.load(T + row * TS0 + (ids // 64) * TS1, valid, other=-1)
        valid = valid & (pages >= 0) & (pages < PAGES)
        values = tl.load(S + row * SS + ids, valid, other=-float("inf"))
        ra = _index_merge(ra, values, ids, TILE, RK)
        if BLOCKS:
            maxima = tl.max(values.reshape(TILE // 8, 8), 1)
            bids = start // 8 + tl.arange(0, TILE // 8)
            maxima = tl.where(
                (bids == (visible - 1) // 8) & (maxima > -float("inf")),
                float("inf"),
                maxima,
            )
            ba = _index_merge(ba, maxima, bids, TILE // 8, BK)
    rv, ri = _index_unpack(ra)
    off = (row * PARTS + part) * RK + tl.arange(0, RK)
    tl.store(RS + off, rv)
    tl.store(RI + off, ri)
    if BLOCKS:
        bv, bi = _index_unpack(ba)
        off = (row * PARTS + part) * BK + tl.arange(0, BK)
        tl.store(BS + off, bv)
        tl.store(BI + off, bi)


def select_scores(scores, table, lengths, pages, topk, candidate_topk, out, parts):
    from tokenspeed_kernel.ops.attention.dsv41._cute_dsl.selection import (
        enabled,
        select_full,
    )

    if enabled() and topk <= 2048 and candidate_topk <= 2048:
        return select_full(scores, table, lengths, pages, topk, candidate_topk, out)
    rows = scores.shape[0]
    rk = max(256, triton.next_power_of_2(topk))
    bk = max(32, triton.next_power_of_2(max(1, candidate_topk)))
    rs = torch.empty((rows, parts * rk), device=scores.device, dtype=torch.float32)
    ri = torch.empty_like(rs, dtype=torch.int64)
    bs = torch.empty(
        (rows, parts * bk if candidate_topk else 0),
        device=scores.device,
        dtype=torch.float32,
    )
    bi = torch.empty_like(bs, dtype=torch.int64)
    _select_scores[(rows, parts)](
        scores,
        table,
        lengths,
        rs,
        ri,
        bs,
        bi,
        scores.stride(0),
        *table.stride(),
        lengths.stride(0),
        scores.shape[1],
        pages,
        parts,
        rk,
        bk,
        bool(candidate_topk),
        256,
        enable_fp_fusion=False
    )
    _index_finish_parts(rs, ri, topk, out[0], out[1])
    if candidate_topk:
        _index_finish_parts(bs, bi, candidate_topk, out[2], out[3])
    return out
