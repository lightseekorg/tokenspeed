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

"""Stable selection compaction and the existing SFA sink epilogue on Ascend."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _compact(Ids, Out, Count, IS0, IS1, WIDTH, KV_ROWS, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    count = tl.full((), 0, tl.int32)
    for begin in range(0, WIDTH, BLOCK):
        column = begin + lane
        ids = tl.load(Ids + row * IS0 + column * IS1, column < WIDTH, other=-1)
        valid = (column < WIDTH) & (ids >= 0) & (ids < KV_ROWS)
        count += tl.sum(valid.to(tl.int32), 0)
    before = tl.full((), 0, tl.int32)
    for begin in range(0, WIDTH, BLOCK):
        column = begin + lane
        ids = tl.load(Ids + row * IS0 + column * IS1, column < WIDTH, other=-1)
        valid = (column < WIDTH) & (ids >= 0) & (ids < KV_ROWS)
        prefix = before + tl.cumsum(valid.to(tl.int32), 0)
        # Both halves form a permutation. No initialize/scatter race, even
        # with all-masked rows, interior holes or repeated indices.
        destination = tl.where(valid, prefix - 1, count + column - prefix)
        tl.store(
            Out + row * WIDTH + destination, tl.where(valid, ids, -1), column < WIDTH
        )
        before += tl.sum(valid.to(tl.int32), 0)
    tl.store(Count + row, count)


@triton.jit(do_not_specialize=["WIDTH", "KV_ROWS", "IS0", "OS0"])
def _prepare_compact(
    Ids, ValidIds, Keys, Count, IS0, OS0, WIDTH, KV_ROWS, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    # Input columns are contiguous; row bases remain runtime 64-bit offsets.
    source = Ids + row.to(tl.int64) * IS0
    ids = tl.load(source + column, column < WIDTH, other=-1)
    valid = (column < WIDTH) & (ids >= 0) & (ids < KV_ROWS)
    destination = row.to(tl.int64) * OS0 + column
    tl.store(
        ValidIds + destination, tl.where(valid, ids.to(tl.int32), -1), column < WIDTH
    )
    tl.store(
        Keys + destination,
        tl.where(valid, column, column + WIDTH).to(tl.float32),
        column < WIDTH,
    )
    tl.store(Count + row, tl.sum(valid.to(tl.int32), 0))


def compact_indices(indices, kv_rows):
    """Return int32 valid-first IDs and counts, preserving order and duplicates."""
    if indices.ndim != 2 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must be integer [tokens,width]")
    rows, width = indices.shape
    if width > 256:
        # Keep native sort for wide selections; fuse its validity, keys and
        # counts preparation. Ascend lowers strided vector loads into large
        # gather temporaries, so normalize only genuinely strided callers.
        indices = indices.contiguous()
        stride = triton.cdiv(width, 16) * 16
        ids = torch.empty((rows, stride), device=indices.device, dtype=torch.int32)[
            :, :width
        ]
        keys = torch.empty((rows, stride), device=indices.device, dtype=torch.float32)[
            :, :width
        ]
        counts = torch.empty(rows, device=indices.device, dtype=torch.int32)
        if not rows:
            return ids, counts
        _prepare_compact[(rows,)](
            indices,
            ids,
            keys,
            counts,
            indices.stride(0),
            stride,
            width,
            kv_rows,
            BLOCK=triton.next_power_of_2(width),
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
        return ids.gather(1, torch.argsort(keys, dim=-1)), counts
    out = torch.empty((rows, width), device=indices.device, dtype=torch.int32)
    counts = (
        torch.zeros(rows, device=indices.device, dtype=torch.int32)
        if not width
        else torch.empty(rows, device=indices.device, dtype=torch.int32)
    )
    if rows and width:
        _compact[(rows,)](
            indices,
            out,
            counts,
            *indices.stride(),
            width,
            kv_rows,
            BLOCK=min(triton.next_power_of_2(width), 256),
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False
        )
    return out, counts


@triton.jit
def _sink(
    Output,
    Maximum,
    Total,
    Sink,
    Counts,
    Result,
    ROWS,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    begin = tl.program_id(0) * ROW_TILE
    step = tl.num_programs(0) * ROW_TILE
    d = tl.arange(0, D)
    rows = tl.arange(0, ROW_TILE)
    while begin < ROWS:
        row = begin + rows
        m = tl.load(Maximum + row, row < ROWS, other=0)
        t = tl.load(Total + row, row < ROWS, other=1)
        sink = tl.load(Sink + row % HEADS)
        active = tl.load(Counts + row // HEADS, row < ROWS, other=0) > 0
        factor = tl.div_rn(1.0, 1.0 + tl.exp(-(m + tl.log(t) - sink)))
        value = tl.load(
            Output + row[:, None] * D + d[None, :], row[:, None] < ROWS, other=0
        ).to(tl.float32)
        result = tl.where(active[:, None], value * factor[:, None], 0.0)
        tl.store(Result + row[:, None] * D + d[None, :], result, row[:, None] < ROWS)
        begin += step


def apply_sink(output, maximum, total, sink, counts):
    """Rescale the already-rounded BF16 SFA result and zero all-masked rows."""
    if output.ndim != 3 or output.shape[-1] != 512:
        raise ValueError("output must be [tokens,heads,512]")
    tokens, heads, dim = output.shape
    output = output.contiguous()
    maximum, total = maximum.contiguous(), total.contiguous()
    result = torch.empty_like(output)
    if tokens:
        # More rows per program amortize the same pointwise work. Keep
        # large prefill and tiny rows on their original geometry.
        row_tile = 16 if 128 <= tokens * heads <= 4096 else 4
        blocks = 48 if row_tile == 16 else 256
        _sink[(min(triton.cdiv(tokens * heads, row_tile), blocks),)](
            output,
            maximum,
            total,
            sink,
            counts,
            result,
            tokens * heads,
            HEADS=heads,
            D=dim,
            ROW_TILE=row_tile,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return result
