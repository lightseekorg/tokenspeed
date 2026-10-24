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

"""Byte-preserving preparation of owner-local V4.1 history pages."""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import virtual_block_to_local


@triton.jit
def _pack_decode_history(
    cache, table, visible_rows, output,
    page_stride, row_stride, byte_stride, table_stride, virtual_count,
    WIDTH: tl.constexpr, DEGREE: tl.constexpr, RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page_index = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    virtual_page = tl.load(
        table + (page_index - 1) * table_stride, page_index > 0, other=0
    ).to(tl.int64)
    local_page, owned = virtual_block_to_local(virtual_page, DEGREE, RANK)
    active = (
        (page_index > 0) & owned & (virtual_page < virtual_count)
        & ((page_index - 1) * 64 < tl.load(visible_rows))
    )
    # Copy packed pages, not token rows: scales have page-local offsets.
    values = tl.load(
        cache + local_page * page_stride
        + (offsets // WIDTH) * row_stride + (offsets % WIDTH) * byte_stride,
        active & (offsets < 64 * WIDTH), other=0,
    )
    tl.store(
        output + page_index * (64 * WIDTH) + offsets,
        values, offsets < 64 * WIDTH,
    )


def pack_decode_history(
    cache: torch.Tensor,
    table: torch.Tensor,
    visible_rows: torch.Tensor,
    *,
    virtual_block_count: int,
    degree: int,
    rank: int,
) -> torch.Tensor:
    """Prepare uint8 history for MAX reduction, including a zero null page.

    cache is a possibly strided [local_pages, 64, row_bytes] field; table
    contains virtual pages and may be strided. visible_rows is a device
    scalar. Nonowners, invalid pages and pages beyond visible_rows contribute
    zero. A partially visible page retains all bytes, including its scales.
    The returned contiguous tensor has shape [table.numel() + 1, 64, row_bytes].
    """
    if cache.ndim != 3 or cache.shape[1] != 64 or cache.dtype != torch.uint8:
        raise ValueError("History requires uint8 [pages, 64, row_bytes]")
    if table.ndim != 1 or table.dtype not in (torch.int32, torch.int64):
        raise ValueError("History requires an integer page table")
    if visible_rows.numel() != 1 or visible_rows.dtype not in (torch.int32, torch.int64):
        raise ValueError("History requires an integer visible-row scalar")
    if not cache.is_cuda or table.device != cache.device or visible_rows.device != cache.device:
        raise ValueError("History tensors must be on one GPU")
    if virtual_block_count <= 1 or degree <= 0 or not 0 <= rank < degree:
        raise ValueError("Invalid history placement")
    if cache.shape[0] < max(0, triton.cdiv(virtual_block_count - 1 - rank, degree)) + 1 or cache.shape[2] <= 0:
        raise ValueError("History cache does not cover the placement")
    result = cache.new_empty((table.numel() + 1, 64, cache.shape[2]))
    _pack_decode_history[
        (table.numel() + 1, triton.cdiv(64 * cache.shape[2], 1024))
    ](
        cache, table, visible_rows, result,
        *cache.stride(), table.stride(0), virtual_block_count,
        WIDTH=cache.shape[2], DEGREE=degree, RANK=rank, BLOCK=1024,
    )
    return result
