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

"""Gather every request's last accepted row of a verify batch in one launch.

A verify batch packs ``rows_per_request`` rows per request, and a request
that accepts ``n`` of them continues from its row ``n - 1``. A drafter
that seeds its next proposals from that row reads it here; on CUDA (and
ROCm) devices one Triton launch gathers every request's row, other
devices run the same arithmetic as tensor ops.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["gather_last_accepted_rows"]

_BLOCK = 1024


@triton.jit(do_not_specialize=["num_rows"])
def _gather_last_accepted_rows_kernel(
    rows_ptr,
    accept_lengths_ptr,
    out_ptr,
    num_rows,
    width,
    stride_rows,
    stride_out,
    ROWS_PER_REQUEST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    accepted = tl.load(accept_lengths_ptr + request).to(tl.int64)
    accepted = tl.minimum(tl.maximum(accepted, 0), ROWS_PER_REQUEST)
    row = request * ROWS_PER_REQUEST - 1 + accepted
    row = tl.minimum(tl.maximum(row, 0), num_rows - 1)
    mask = cols < width
    values = tl.load(rows_ptr + row * stride_rows + cols, mask=mask)
    tl.store(out_ptr + request * stride_out + cols, values, mask=mask)


def _gather_last_accepted_rows_torch(
    rows: torch.Tensor,
    accept_lengths: torch.Tensor,
    rows_per_request: int,
    out: torch.Tensor,
) -> None:
    requests = accept_lengths.shape[0]
    accepted = accept_lengths.to(torch.int64).clamp(min=0, max=rows_per_request)
    starts = torch.arange(requests, device=rows.device) * rows_per_request - 1
    last = (starts + accepted).clamp(min=0, max=rows.shape[0] - 1)
    out.copy_(rows.index_select(0, last))


def gather_last_accepted_rows(
    rows: torch.Tensor,
    accept_lengths: torch.Tensor,
    *,
    rows_per_request: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Each request's last accepted row of a verify batch.

    Request ``r`` owns rows ``[r * rows_per_request, (r + 1) *
    rows_per_request)`` and accepts ``accept_lengths[r]`` of them (clamped to
    ``[0, rows_per_request]``); its row is ``r * rows_per_request - 1 +
    accepted``, clamped into ``rows``.

    Args:
        rows: ``[num_rows, width]`` verify rows with dense columns.
        accept_lengths: ``[requests]`` contiguous integer accepted rows per
            request.
        rows_per_request: Verify rows per request.
        out: Optional ``[requests, width]`` destination in ``rows``' dtype
            with dense columns; a new tensor when omitted.

    Returns:
        ``[requests, width]`` gathered rows (``out`` when given).

    Raises:
        ValueError: A tensor is not the shape, dtype or layout the gather
            indexes with.
    """
    requests = accept_lengths.shape[0]
    if rows.ndim != 2 or rows.stride(1) != 1:
        raise ValueError("rows must be a 2D tensor with dense columns")
    if (
        accept_lengths.ndim != 1
        or accept_lengths.dtype not in (torch.int32, torch.int64)
        or not accept_lengths.is_contiguous()
    ):
        raise ValueError("accept_lengths must be a contiguous 1D int32/int64 tensor")
    if rows_per_request < 1:
        raise ValueError(f"rows_per_request must be positive, got {rows_per_request}")
    if out is None:
        out = rows.new_empty((requests, rows.shape[1]))
    elif (
        out.shape != (requests, rows.shape[1])
        or out.dtype != rows.dtype
        or out.stride(1) != 1
    ):
        raise ValueError(
            f"out must be [{requests}, {rows.shape[1]}] {rows.dtype} with dense columns"
        )
    if requests == 0 or rows.shape[0] == 0:
        return out
    if not rows.is_cuda:
        _gather_last_accepted_rows_torch(rows, accept_lengths, rows_per_request, out)
        return out
    width = rows.shape[1]
    _gather_last_accepted_rows_kernel[(requests, triton.cdiv(width, _BLOCK))](
        rows,
        accept_lengths,
        out,
        rows.shape[0],
        width,
        rows.stride(0),
        out.stride(0),
        ROWS_PER_REQUEST=rows_per_request,
        BLOCK=_BLOCK,
        num_warps=4,
    )
    return out
