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
    host_row = tl.cast(
        table_ptr + idx * table_stride, tl.pointer_type(tl.uint8)
    )
    values = tl.load(host_row + cols, mask=mask & valid, other=0)
    tl.store(out_ptr + row * out_stride + cols, values, mask=mask)


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
    tags={"portability"},
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
