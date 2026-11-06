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

"""Triton UVA gather of uint8 rows from a host-resident table."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_CAPABILITY = CapabilityRequirement(vendors=frozenset({"nvidia", "amd"}))
_SIGNATURES = frozenset({format_signature(x=dense_tensor_format(torch.uint8))})


@triton.jit
def _host_uint8_row_gather_kernel(
    table_ptr,
    indices_ptr,
    out_ptr,
    n_rows,
    width,
    table_stride,
    out_stride,
    INDEX_INT64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(indices_ptr + row)
    if not INDEX_INT64:
        idx = idx.to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    valid = (idx >= 0) & (idx < n_rows)
    host_row = tl.cast(table_ptr + idx * table_stride, tl.pointer_type(tl.uint8))
    values = tl.load(host_row + cols, mask=mask & valid, other=0)
    tl.store(out_ptr + row * out_stride + cols, values, mask=mask)


@triton.jit
def _host_mxfp8_row_gather_kernel(
    table_ptr,
    indices_ptr,
    codes_ptr,
    scales_ptr,
    n_rows,
    n_columns,
    index_stride_token,
    index_stride_column,
    table_stride,
    DIM: tl.constexpr,
):
    # One program per (token, column): load a packed [codes | scales] row,
    # rebuild its FP32 values and quantize them again with the group-32 UE8M0
    # rule (1e-4 amax floor), as quantizing the dequantized BF16 row would.
    row = tl.program_id(0)
    token = row // n_columns
    column = row % n_columns
    idx = tl.load(
        indices_ptr
        + token.to(tl.int64) * index_stride_token
        + column.to(tl.int64) * index_stride_column
    ).to(tl.int64)
    valid = (idx >= 0) & (idx < n_rows)
    host_row = tl.cast(
        table_ptr + tl.where(valid, idx, 0) * table_stride, tl.pointer_type(tl.uint8)
    )
    cols = tl.arange(0, DIM)
    groups = tl.arange(0, DIM // 32)
    codes = tl.load(host_row + cols, mask=valid, other=0)
    exponents = tl.load(host_row + DIM + groups, mask=valid, other=0)
    scale = (exponents.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    values = tl.reshape(
        codes.to(tl.float8e4nv, bitcast=True).to(tl.float32), (DIM // 32, 32)
    ) * tl.expand_dims(scale, 1)
    amax = tl.maximum(tl.max(tl.abs(values), 1), 1.0e-4)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    inverse_scale = 1.0 / (exponent << 23).to(tl.float32, bitcast=True)
    quantized = (values * tl.expand_dims(inverse_scale, 1)).to(tl.float8e4nv)
    out_row = row.to(tl.int64)
    tl.store(
        codes_ptr + out_row * DIM + cols,
        tl.reshape(quantized.to(tl.uint8, bitcast=True), (DIM,)),
    )
    tl.store(scales_ptr + out_row * (DIM // 32) + groups, exponent.to(tl.uint8))


@register_kernel(
    "embedding",
    "host_mxfp8_row_gather",
    name="triton_host_mxfp8_row_gather",
    solution="triton",
    capability=_CAPABILITY,
    signatures=_SIGNATURES,
    priority=Priority.PORTABLE,
)
def triton_host_mxfp8_row_gather(
    table: torch.Tensor,
    indices: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    dim = table.shape[1] // 33 * 32
    rows = indices.shape[0] * indices.shape[1]
    if rows == 0:
        return
    _host_mxfp8_row_gather_kernel[(rows,)](
        current_platform().device_visible_data_ptr(table),
        indices,
        codes.view(torch.uint8),
        scales,
        table.shape[0],
        indices.shape[1],
        indices.stride(0),
        indices.stride(1),
        table.stride(0),
        DIM=dim,
    )


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@register_kernel(
    "embedding",
    "host_uint8_row_gather",
    name="triton_host_uint8_row_gather",
    solution="triton",
    capability=_CAPABILITY,
    signatures=_SIGNATURES,
    priority=Priority.PORTABLE,
)
def triton_host_uint8_row_gather(
    table: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    flat = indices.reshape(-1)
    width = table.shape[1]
    table_bytes = table.view(torch.uint8)
    table_ptr = current_platform().device_visible_data_ptr(table_bytes)
    grid = (flat.numel(),)
    _host_uint8_row_gather_kernel[grid](
        table_ptr,
        flat,
        out.reshape(-1, width),
        table.shape[0],
        width,
        table_bytes.stride(0),
        width,
        INDEX_INT64=flat.dtype == torch.int64,
        BLOCK=_next_power_of_2(width),
    )
    return out
