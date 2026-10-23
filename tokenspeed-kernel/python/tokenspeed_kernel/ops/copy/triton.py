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

"""Zero the trailing rows of a tensor past a device-held row count."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

_BLOCK_ROWS = 16
_BLOCK_COLS = 256


# ``live_rows`` is often the last entry of a per-request offset table, so its
# pointer alignment moves with the request count; it is read once, so keying
# the compile on that alignment would only recompile while serving.
@triton.jit(
    do_not_specialize=["rows", "cols", "row_stride", "col_stride"],
    do_not_specialize_on_alignment=["live_rows"],
)
def _zero_rows_after_kernel(
    data,
    live_rows,
    rows,
    cols,
    row_stride,
    col_stride,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    first = tl.program_id(0).to(tl.int64) * BLOCK_ROWS
    live = tl.load(live_rows).to(tl.int64)
    # Blocks wholly inside the live rows touch no memory.
    if first + BLOCK_ROWS > live:
        row = first + tl.arange(0, BLOCK_ROWS)
        row_mask = (row >= live) & (row < rows)
        zero = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=data.dtype.element_ty)
        for start in range(0, cols, BLOCK_COLS):
            # 64-bit like the rows: a strided view's column walk can pass 2**31.
            col = start + tl.arange(0, BLOCK_COLS).to(tl.int64)
            mask = row_mask[:, None] & (col[None, :] < cols)
            offsets = row[:, None] * row_stride + col[None, :] * col_stride
            tl.store(data + offsets, zero, mask=mask)


def triton_zero_rows_after(tensor: torch.Tensor, live_rows: torch.Tensor) -> None:
    """Zero rows ``[live_rows, tensor.shape[0])`` of ``tensor`` in place.

    See :func:`tokenspeed_kernel.ops.copy.zero_rows_after`.
    """
    if tensor.dim() == 0:
        raise ValueError("zero_rows_after needs a tensor with a row dimension")
    if live_rows.numel() != 1 or live_rows.dtype not in (torch.int32, torch.int64):
        raise ValueError("live_rows must be a one-element int32 or int64 tensor")
    if live_rows.device != tensor.device:
        raise ValueError("live_rows must be on the tensor's device")
    rows = tensor.shape[0]
    if rows == 0 or tensor.numel() == 0:
        return
    # Trailing dims must already be one strided dimension: a reshaped copy
    # would zero a temporary instead of the caller's rows.
    if tensor.dim() == 1:
        flat = tensor.view(rows, 1)
    elif tensor.dim() == 2:
        flat = tensor
    else:
        flat = tensor.view(rows, -1)
    _zero_rows_after_kernel[(triton.cdiv(rows, _BLOCK_ROWS),)](
        flat,
        live_rows,
        rows,
        flat.shape[1],
        flat.stride(0),
        flat.stride(1),
        BLOCK_ROWS=_BLOCK_ROWS,
        BLOCK_COLS=_BLOCK_COLS,
    )
