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

"""Gather several row-aligned model tensors and zero query padding together."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit(
    do_not_specialize=["stride0", "stride1", "stride2", "stride3"],
)
def _gather_query_rows_kernel(
    Inputs,
    Indices,
    Valid,
    Outputs,
    stride0,
    stride1,
    stride2,
    stride3,
    WIDTHS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    input_strides = (stride0, stride1, stride2, stride3)
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(Indices + row).to(tl.int64)
    valid = tl.load(Valid + row)
    for i in tl.static_range(len(WIDTHS)):
        values = tl.load(
            Inputs[i] + source * input_strides[i] + columns,
            mask=valid & (columns < WIDTHS[i]),
            other=0,
        )
        tl.store(
            Outputs[i] + row * WIDTHS[i] + columns, values, mask=columns < WIDTHS[i]
        )


def gather_query_rows(
    values: tuple[torch.Tensor, ...],
    indices: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Gather rows of aligned tensors, writing zero for invalid destinations.

    Args:
        values: One to four tensors with a shared first row dimension.
            Inner dimensions must be contiguous; the row stride may differ.
        indices: Contiguous int64 source-row indices, one per output row.
            Indices for valid rows must be in range; invalid rows are not read.
        valid: Contiguous bool mask with the same shape as indices.

    Returns:
        Contiguous tensors with ``indices.numel()`` rows and the input inner
        shapes and dtypes. Values are copied without changing arithmetic.
    """
    if not 1 <= len(values) <= 4 or indices.ndim != 1 or indices.dtype != torch.int64:
        raise ValueError("query gather needs tensors and int64 row indices")
    if valid.shape != indices.shape or valid.dtype != torch.bool:
        raise ValueError("query gather mask must be bool and match row indices")
    if not indices.is_contiguous() or not valid.is_contiguous():
        raise ValueError("query gather maps must be contiguous")
    if values[0].ndim < 1 or values[0].shape[0] == 0:
        raise ValueError("query gather needs nonempty input rows")
    rows = values[0].shape[0]
    widths = tuple(math.prod(value.shape[1:]) for value in values)
    for value in values:
        if value.ndim < 1 or value.shape[0] != rows:
            raise ValueError("query gather tensors must share their row count")
        if value.device != indices.device or valid.device != indices.device:
            raise ValueError("query gather tensors and maps must share a device")
        expected = 1
        for size, stride in zip(
            reversed(value.shape[1:]), reversed(value.stride()[1:])
        ):
            if size > 1 and stride != expected:
                raise ValueError("query gather inner dimensions must be contiguous")
            expected *= size
    if not indices.is_cuda:
        return tuple(
            torch.where(
                valid.view((-1,) + (1,) * (value.ndim - 1)),
                value.index_select(0, indices.clamp(0, max(rows - 1, 0))),
                torch.zeros((), dtype=value.dtype, device=value.device),
            )
            for value in values
        )
    outputs = tuple(
        value.new_empty((indices.numel(), *value.shape[1:])) for value in values
    )
    if indices.numel() and max(widths):
        _gather_query_rows_kernel[(indices.numel(), triton.cdiv(max(widths), 256))](
            values,
            indices,
            valid,
            outputs,
            *(tuple(value.stride(0) for value in values) + (0,) * (4 - len(values))),
            widths,
            256,
        )
    return outputs
