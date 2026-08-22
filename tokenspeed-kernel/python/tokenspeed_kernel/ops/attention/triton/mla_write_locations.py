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

"""Absolute MLA latent write locations in one launch."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["total", "q_len", "num_pages", "table_stride", "page_size"]
)
def _mla_write_locations_kernel(
    seq_lens,
    table,
    out,
    total,
    q_len,
    num_pages,
    table_stride,
    page_size,
    BLOCK: tl.constexpr,
):
    # Flat over bs*q_len: a per-batch constexpr would JIT just before replay.
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < total
    row = idx // q_len
    step = idx % q_len
    last = tl.maximum(tl.load(seq_lens + row, mask=live, other=0).to(tl.int64) - 1, 0)
    # Verify writes the trailing q_len positions, request-major.
    pos = tl.maximum(last + (step - (q_len - 1)), 0)
    page_index = tl.minimum(pos // page_size, num_pages - 1)
    page = tl.load(table + row * table_stride + page_index, mask=live, other=0).to(
        tl.int64
    )
    tl.store(out + idx, tl.maximum(page, 0) * page_size + pos % page_size, mask=live)


def _torch_write_locations(seq_lens, table, page_size, q_len_per_req, batch_size, out):
    """Portable spelling, kept for non-CUDA callers (the backends' CPU tests)."""
    last = (seq_lens[:batch_size].to(torch.int64) - 1).clamp_min(0)
    steps = torch.arange(
        1 - q_len_per_req, 1, device=seq_lens.device, dtype=torch.int64
    )
    positions = (last.unsqueeze(1) + steps).clamp_min(0)
    pages = table[:batch_size].gather(
        1, torch.div(positions, page_size, rounding_mode="floor")
    )
    locations = (
        pages.clamp_min(0).to(torch.int64) * page_size + (positions % page_size)
    ).reshape(-1)
    if out is None:
        return locations
    out[: batch_size * q_len_per_req].copy_(locations)
    return out[: batch_size * q_len_per_req]


def mla_write_locations(
    seq_lens: torch.Tensor,
    table: torch.Tensor,
    *,
    page_size: int,
    q_len_per_req: int,
    batch_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resolve where each decoded token writes into the paged latent cache.

    The torch spelling of this is a chain of a dozen elementwise launches
    (cast, sub, clamp, arange, add, clamp, div, gather, clamp, cast, mul,
    remainder, add) over ``[batch_size]`` integers, which on a decode step costs
    launch overhead only. Every step is row-local, so one program covers the
    whole batch.

    Args:
        seq_lens: Integer sequence lengths ``[>=batch_size]``.
        table: Page table ``[>=batch_size, num_pages]``; entries below zero are
            table holes and clamp to the null page.
        page_size: Tokens per page in the kernel's paging.
        q_len_per_req: Trailing positions written per request (1 for plain
            decode, the speculative window on target verify).
        batch_size: Live requests; rows past it are ignored.
        out: Optional INT64 destination written in place, so a CUDA-graph
            replay keeps the buffer the capture recorded.

    Returns:
        INT64 absolute write locations ``[batch_size * q_len_per_req]``,
        flattened request-major. With ``out`` this is a view of its prefix.

    A sequence longer than the table can address (``num_pages * page_size``)
    resolves to the last page rather than raising: the kernel cannot fail
    loudly without a host sync, and reading past the table would be worse.
    The debug validators catch that case where they already sync.
    """
    if seq_lens.device != table.device:
        raise ValueError("MLA write locations need colocated seq_lens and table")
    # Unit-stride arithmetic below; the torch spelling honoured any stride.
    if seq_lens.stride(0) != 1 or table.stride(1) != 1:
        raise ValueError("MLA write locations need unit-stride seq_lens and table rows")
    if batch_size < 0 or batch_size > table.shape[0]:
        raise ValueError(
            f"batch_size {batch_size} exceeds the table's {table.shape[0]} rows"
        )
    if q_len_per_req < 1:
        raise ValueError(f"q_len_per_req must be positive, got {q_len_per_req}")
    total = batch_size * q_len_per_req
    if out is None:
        out = torch.empty(total, dtype=torch.int64, device=table.device)
    elif out.dtype != torch.int64:
        raise ValueError(f"out must be INT64, got {out.dtype}")
    elif out.device != table.device:
        raise ValueError("out must live on the table's device")
    elif out.stride(0) != 1:
        raise ValueError("out must be unit-stride")
    elif out.numel() < total:
        raise ValueError(f"out holds {out.numel()} entries, need {total}")
    elif out.data_ptr() == table.data_ptr():
        raise ValueError("out must not alias the page table")
    if total == 0:
        return out[:0]
    if not table.is_cuda:
        return _torch_write_locations(
            seq_lens, table, page_size, q_len_per_req, batch_size, out
        )

    block = 256
    _mla_write_locations_kernel[(triton.cdiv(total, block),)](
        seq_lens,
        table,
        out,
        total,
        q_len_per_req,
        table.shape[1],
        table.stride(0),
        page_size,
        BLOCK=block,
    )
    return out[:total]
