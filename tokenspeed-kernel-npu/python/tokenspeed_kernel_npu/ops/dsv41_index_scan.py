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


from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _index_score_scan_kernel(
    Q,
    W,
    Cache,
    Table,
    Visible,
    RowScores,
    QS0,
    QS1,
    QS2,
    WS0,
    WS1,
    CP,
    CR: tl.constexpr,
    CB: tl.constexpr,
    TS0,
    TS1,
    VS,
    OUTPUT_WIDTH,
    PAGES: tl.constexpr,
    TABLE_WIDTH,
    HEADS: tl.constexpr,
    SHARD_HEADS: tl.constexpr,
    SHARDS: tl.constexpr,
    H: tl.constexpr,
    PARTS: tl.constexpr,
    B: tl.constexpr,
    STEP: tl.constexpr,
):
    query, part = (tl.program_id(0), tl.program_id(1))
    visible = tl.minimum(tl.maximum(tl.load(Visible + query * VS), 0), TABLE_WIDTH * 64)
    width = visible
    tl.static_assert(B <= 64)
    tl.static_assert(STEP == B)
    tl.static_assert(64 % B == 0)
    span = tl.cdiv(width, PARTS * 64) * 64
    begin = part * span
    end = tl.minimum(begin + span, width)
    h = tl.arange(0, H)
    d = tl.arange(0, 128)
    q = tl.load(
        Q + query * QS0 + h[:, None] * QS1 + d[None, :] * QS2,
        h[:, None] < HEADS,
        other=0,
    )
    weights = tl.load(W + query * WS0 + h * WS1, h < HEADS, other=0)
    for start in range(begin, end, STEP):
        col = start + tl.arange(0, B)
        logical = col.to(tl.int64)
        valid = (
            (col < tl.minimum(start + STEP, end)) & (logical >= 0) & (logical < visible)
        )
        physical_page = tl.load(Table + query * TS0 + start // 64 * TS1).to(tl.int64)
        page = tl.full((B,), 0, tl.int64) + physical_page
        valid = valid & (page > 0) & (page < PAGES)
        row = logical % 64
        safe_page = tl.where(
            (physical_page > 0) & (physical_page < PAGES), physical_page, 0
        )
        page_base = Cache + safe_page * CP
        key_rows = tl.load(
            page_base + row[:, None] * CR + d[None, :] * CB, valid[:, None], other=0
        )
        key = tl.trans(key_rows)
        dots = tl.dot(q, key).to(tl.bfloat16).to(tl.float32)
        products = (
            (tl.maximum(dots, 0.0) * weights[:, None].to(tl.float32))
            .to(W.dtype.element_ty)
            .to(tl.float32)
        )
        scores = tl.full((B,), 0.0, tl.float32)
        for rank in tl.static_range(SHARDS):
            partial = tl.sum(
                tl.where(
                    (h[:, None] >= rank * SHARD_HEADS)
                    & (h[:, None] < (rank + 1) * SHARD_HEADS),
                    products,
                    0.0,
                ),
                0,
            )
            scores = scores + partial.to(W.dtype.element_ty).to(tl.float32)
        scores = scores.to(W.dtype.element_ty).to(tl.float32)
        scores = tl.where(valid, scores, -float("inf"))
        tl.store(
            RowScores + query * OUTPUT_WIDTH + col,
            scores,
            col < tl.minimum(start + STEP, end),
        )
