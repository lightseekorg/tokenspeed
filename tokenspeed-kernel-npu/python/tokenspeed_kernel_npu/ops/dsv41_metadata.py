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


"""Batch page-table copies with a bounded number of Ascend programs."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _selection_table(
    P,
    Req,
    Table,
    Out,
    Lens,
    TC,
    TR,
    TS0,
    TS1,
    PS,
    RS,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
    PARTS: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    req = tl.load(Req + row * RS)
    pos = tl.load(P + row * PS)
    live = (req >= 0) & (req < TR)
    offset = tl.arange(0, BLOCK)
    for start in range(part * BLOCK, TC, PARTS * BLOCK):
        col = start + offset
        page = tl.load(Table + req * TS0 + col * TS1, live & (col < TC), other=-1)
        tl.store(Out + row * TC + col, tl.where(page > 0, page, -1), col < TC)
    if part == 0:
        tl.store(Lens + row, tl.maximum((pos + 1) // RATIO, 0))


def selection_table(positions, requests, table, ratio):
    """Copy full table rows, including inactive tails, preserving the existing ABI."""
    n = positions.numel()
    out = torch.empty((n, table.shape[1]), dtype=torch.int32, device=table.device)
    lengths = torch.empty(n, dtype=torch.int32, device=table.device)
    if n:
        _selection_table[(n, 4)](
            positions,
            requests,
            table,
            out,
            lengths,
            table.shape[1],
            table.shape[0],
            *table.stride(),
            positions.stride(0),
            requests.stride(0),
            ratio,
            512,
            4,
            multibuffer=False,
        )
    return out, lengths


@triton.jit
def _global_slots(
    Rows,
    P,
    Req,
    Table,
    Out,
    K,
    LS0,
    LS1,
    PS,
    RS,
    TR,
    TC,
    TS0,
    TS1,
    RATIO: tl.constexpr,
    PAGES: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    logical = tl.load(Rows + row.to(tl.int64) * LS0 + col * LS1, col < K, other=-1)
    pos = tl.load(P + row * PS).to(tl.int64)
    req = tl.load(Req + row * RS).to(tl.int64)
    visible = (pos + 1) // RATIO
    lc = logical // 64
    valid = (
        (col < K)
        & (logical >= 0)
        & (logical < visible)
        & (req >= 0)
        & (req < TR)
        & (lc < TC)
    )
    lc32 = tl.where(valid, lc, 0).to(tl.int32)
    base = Table + tl.where((req >= 0) & (req < TR), req, 0) * TS0
    page = tl.load(base + lc32 * TS1, valid, other=-1)
    ok = valid & (page > 0) & (page < PAGES)
    slot = page.to(tl.int32) * 64 + (logical.to(tl.int32) % 64)
    tl.store(Out + row.to(tl.int64) * K + col, tl.where(ok, slot, -1), col < K)


def global_slots(rows, positions, requests, table, ratio, pages):
    """Map logical selections once per query, retaining scalar 64-bit bases.

    Avoid expanding request/position metadata to every selection and avoid
    multiplying logical positions by ratio only to divide them again. All
    bounds and causal masking remain live device values during graph replay.
    """
    if not table.numel():
        return torch.full(rows.shape, -1, device=rows.device, dtype=torch.int32)
    if not 0 < pages < 2**25 or table.stride(1) * table.shape[1] >= 2**31:
        from tokenspeed_kernel.ops.attention.dsv41.triton import (
            global_slots as original,
        )

        return original(rows, positions, requests, table, ratio, pages)
    out = torch.empty(rows.shape, device=rows.device, dtype=torch.int32)
    if rows.numel():
        block = 512
        _global_slots[(rows.shape[0], triton.cdiv(rows.shape[1], block))](
            rows,
            positions,
            requests,
            table,
            out,
            rows.shape[1],
            *rows.stride(),
            positions.stride(0),
            requests.stride(0),
            *table.shape,
            *table.stride(),
            ratio,
            pages,
            block,
            num_warps=1,
            multibuffer=False,
            enable_fp_fusion=False,
        )
    return out
