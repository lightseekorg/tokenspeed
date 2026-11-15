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

"""Broadcast each request's block-end length to its decode entries in one launch.

A block drafter's draft decode gives every query row of a request the same
cache length: the block end, clamped to the verify width and the context
limit. The attention backend keeps one row per request and that row repeated
over the request's ``expansion`` entries; both come from one launch on CUDA
(and ROCm) devices and from the same tensor ops elsewhere.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["broadcast_block_lengths"]

_BLOCK = 128


@triton.jit(do_not_specialize=["batch_size", "low", "high"])
def _broadcast_block_lengths_kernel(
    lengths_ptr,
    rows_ptr,
    entries_ptr,
    batch_size,
    low,
    high,
    EXPANSION: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = rows < batch_size
    lengths = tl.load(lengths_ptr + rows, mask=mask, other=0)
    clamped = tl.minimum(tl.maximum(lengths, low), high).to(rows_ptr.dtype.element_ty)
    tl.store(rows_ptr + rows, clamped, mask=mask)
    for entry in tl.static_range(EXPANSION):
        value = clamped - (EXPANSION - 1 - entry) if CAUSAL else clamped
        tl.store(entries_ptr + rows * EXPANSION + entry, value, mask=mask)


def broadcast_block_lengths(
    lengths: torch.Tensor,
    low: int,
    high: int,
    *,
    causal: bool,
    rows_out: torch.Tensor,
    entries_out: torch.Tensor,
) -> None:
    """Clamp every request's block-end length and write it per request and per entry.

    Args:
        lengths: ``[batch]`` integer block-end lengths.
        low: Smallest length written.
        high: Largest length written; at least ``low``.
        causal: Whether entry ``e`` of ``expansion`` sees ``expansion - 1 - e``
            fewer tokens than the row (a block-causal draft) instead of all of them.
        rows_out: ``[batch]`` contiguous lengths, one per request.
        entries_out: ``[batch, expansion]`` contiguous lengths, each request's
            row repeated over its entries; ``rows_out``'s dtype.

    Returns:
        None. ``rows_out`` and ``entries_out`` are written.

    Raises:
        ValueError: A tensor is not the shape, dtype, layout, or device written,
            or ``low > high``.
    """
    batch_size = lengths.shape[0]
    if lengths.ndim != 1 or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must be a 1D int32/int64 tensor")
    if lengths.stride(0) != 1:
        raise ValueError("lengths must be dense")
    if any(t.device != lengths.device for t in (rows_out, entries_out)):
        raise ValueError("broadcast_block_lengths needs colocated tensors")
    if rows_out.shape != (batch_size,) or entries_out.ndim != 2:
        raise ValueError(f"rows_out must be [{batch_size}], entries_out 2D")
    if entries_out.shape[0] != batch_size or entries_out.dtype != rows_out.dtype:
        raise ValueError("entries_out must hold rows_out's dtype for every request")
    if not (rows_out.is_contiguous() and entries_out.is_contiguous()):
        raise ValueError("broadcast_block_lengths writes contiguous outputs")
    if low > high:
        raise ValueError(f"low {low} exceeds high {high}")
    if batch_size == 0:
        return
    if not lengths.is_cuda:
        torch.clamp(lengths, low, high, out=rows_out)
        if causal:
            expansion = entries_out.shape[1]
            backoff = torch.arange(
                expansion - 1, -1, -1, dtype=rows_out.dtype, device=rows_out.device
            )
            torch.sub(rows_out.unsqueeze(1), backoff, out=entries_out)
        else:
            entries_out.copy_(rows_out.unsqueeze(1))
        return
    _broadcast_block_lengths_kernel[(triton.cdiv(batch_size, _BLOCK),)](
        lengths,
        rows_out,
        entries_out,
        batch_size,
        low,
        high,
        EXPANSION=entries_out.shape[1],
        CAUSAL=causal,
        BLOCK=_BLOCK,
        num_warps=4,
    )
