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

"""Byte-preserving, request-bounded V4.1 KVP history materialization."""

import torch

from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.layers.attention.dcp.cache import (
    gather_history_rows,
    plan_history_gather,
)
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    owned_history_rows,
)


def gather_paged_history(
    cache: torch.Tensor,
    table: torch.Tensor,
    table_cpu: torch.Tensor,
    *,
    rows: int,
    placement: CachePlacement,
) -> torch.Tensor:
    """Gather one request's packed rows, retaining null page and row encoding.

    Tables name virtual pages. Each page stores all value rows followed by
    all scale rows: its apparent [64, row_bytes] axis is NOT a token axis.
    Transport whole pages, including the last page's masked padding, so scales
    retain their offsets. Cache fields can be strided views of the LCM plane.
    Every owner participates, even with no local queries.
    """
    if cache.ndim != 3 or cache.shape[1] != 64 or cache.dtype != torch.uint8:
        raise ValueError("V4.1 history requires uint8 [pages, 64, row_bytes]")
    if rows < 0 or placement.block_granularity != 64:
        raise ValueError("V4.1 history requires nonnegative rows and 64-row pages")
    pages = (rows + 63) // 64
    if table.ndim != 1 or table_cpu.ndim != 1 or table.numel() < pages:
        raise ValueError("V4.1 history needs one complete request page table")
    counts = owned_history_rows(
        table_cpu[None, :],
        torch.tensor([pages * 64], dtype=torch.int64),
        page_size=64,
        placement=placement,
    )[:, 0].tolist()
    logical = torch.arange(pages * 64, device=cache.device, dtype=torch.int64)
    virtual_slots = table[logical // 64].long() * 64 + logical % 64
    plan = plan_history_gather(
        virtual_slots,
        placement=placement,
        owned_rows_per_rank=counts,
    )
    slots = plan.local_fetch_slots
    local = cache[slots // 64, slots % 64].contiguous()
    # Page zero is never selected; padding is masked by visible lengths.
    result = cache.new_zeros((pages + 1, 64, cache.shape[-1]))
    gather_history_rows(plan, local, out=result[1:].view(-1, cache.shape[-1]))
    return result


def gather_decode_history(
    cache: torch.Tensor,
    table: torch.Tensor,
    visible_rows: torch.Tensor,
    *,
    placement: CachePlacement,
) -> torch.Tensor:
    """Reconstruct packed pages with fixed shapes for decode/verify graph replay.

    The device scalar bounds visible pages; the table capacity fixes allocation
    and communication size. Each byte has exactly one owner. Unsigned MAX with
    zero-filled nonowners transports bytes exactly, including page-local scales,
    without floating-point arithmetic or host ownership counts. Page zero stays
    reserved. Callers still mask each query by its own causal row count.
    """
    from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots

    if cache.ndim != 3 or cache.shape[1] != 64 or cache.dtype != torch.uint8:
        raise ValueError("V4.1 history requires uint8 [pages, 64, row_bytes]")
    if (
        table.ndim != 1
        or visible_rows.numel() != 1
        or placement.block_granularity != 64
    ):
        raise ValueError("V4.1 decode history requires a page table and scalar length")
    slots, owned = resolve_cache_slots(table.long() * 64, placement)
    active = owned & (
        torch.arange(table.numel(), device=table.device) * 64 < visible_rows
    )
    pages = cache[(slots // 64).clamp_min(0)]
    pages = pages.masked_fill(~active[:, None, None], 0)
    pages = all_reduce(
        pages.contiguous(), placement.group, op=torch.distributed.ReduceOp.MAX
    )
    result = cache.new_zeros((table.numel() + 1, 64, cache.shape[-1]))
    result[1:].copy_(pages)
    return result
