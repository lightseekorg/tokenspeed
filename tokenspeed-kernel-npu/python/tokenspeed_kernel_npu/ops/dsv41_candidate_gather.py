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


"""Gather contiguous BF16 candidate blocks without an intermediate slot tensor."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _candidate_keys_contiguous(
    Candidates,
    Table,
    Visible,
    Cache,
    Output,
    CS0,
    CS1,
    TS0,
    TS1,
    VS,
    COUNT,
    TABLE_WIDTH,
    PAGES,
    PARTS: tl.constexpr,
):
    query, part = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, 1024)
    visible = tl.load(Visible + query.to(tl.int64) * VS)
    for column in range(part, COUNT, PARTS):
        candidate = tl.load(Candidates + query.to(tl.int64) * CS0 + column * CS1)
        logical_page = tl.minimum(tl.maximum(candidate // 8, 0), TABLE_WIDTH - 1)
        page = tl.load(Table + query.to(tl.int64) * TS0 + logical_page * TS1)
        valid = (
            (candidate >= 0) & (candidate * 8 < visible) & (page > 0) & (page < PAGES)
        )
        # Scalar 64-bit base plus a contiguous 8x128 block. Vector page
        # addresses/broadcasts miscompiled in tested Ascend compiler shapes.
        page = tl.where(valid, page, 0).to(tl.int64)
        block = tl.where(valid, candidate, column) % 8
        source = (page * 64 + block * 8) * 128
        values = tl.load(Cache + source + lane)
        destination = (query.to(tl.int64) * COUNT + column) * 1024
        tl.store(Output + destination + lane, values)


def candidate_keys(candidates, table, visible, cache, *, parts=32):
    """Match candidate_slots + IndexSelect, including padding-page contents."""
    n, count = candidates.shape
    if (
        cache.dtype != torch.bfloat16
        or not cache.is_contiguous()
        or tuple(cache.shape[1:]) != (64, 128)
    ):
        raise ValueError("BF16 contiguous [pages,64,128] required")
    if n and count and (not table.shape[1] or not cache.shape[0]):
        raise ValueError("Nonempty page table and cache required")
    if n * count >= 32768:
        from tokenspeed_kernel_npu.ops.dsv41_candidate_gather_native import (
            gather_candidates,
        )

        native = gather_candidates(candidates, table, visible, cache)
        if native is not None:
            return native
    out = torch.empty((n, count * 8, 128), dtype=cache.dtype, device=cache.device)
    if n and count:
        _candidate_keys_contiguous[(n, parts)](
            candidates,
            table,
            visible,
            cache,
            out,
            *candidates.stride(),
            *table.stride(),
            visible.stride(0),
            count,
            table.shape[1],
            cache.shape[0],
            parts,
            multibuffer=False,
        )
    return out
