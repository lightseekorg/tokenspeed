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

"""Gather uint8 rows from a host-resident table onto the device.

The table stays in pinned / cudaHostRegistered host memory. The kernel reads
it through a device-visible pointer (UVA on NVIDIA, hipHostGetDevicePointer
on AMD). Invalid indices write zeros. Callers own dequantization so FP8/E8M0
rounding stays with the same torch casts as the GPU-resident path.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.selection import SelectionObjective, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["uint8_row_gather"]


def uint8_row_gather(
    table: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Gather ``table[indices]`` as uint8 rows.

    Args:
        table: Host or device uint8 ``[rows, width]``. Host tables must be
            GPU-visible (pinned or ``register_host_tensor_for_gpu_access``).
        indices: Int32/int64 ids with arbitrary shape. Negative or out-of-range
            ids produce a zero row.
        out: Optional contiguous uint8 ``[*indices.shape, width]`` on
            ``indices.device``. Allocated when omitted.

    Returns:
        The destination tensor. Empty ``indices`` returns an empty view.
    """
    if table.dtype != torch.uint8 or table.ndim != 2:
        raise TypeError("host row gather requires a 2D uint8 table")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("host row gather indices must be int32 or int64")
    width = table.shape[1]
    dest_shape = (*indices.shape, width)
    if out is None:
        out = torch.empty(dest_shape, dtype=torch.uint8, device=indices.device)
    elif tuple(out.shape) != dest_shape or out.dtype != torch.uint8:
        raise ValueError("host row gather out must be uint8 [*indices, width]")
    if indices.numel() == 0:
        return out
    if indices.device.type == "cpu":
        rows = table.shape[0]
        flat = indices.reshape(-1)
        valid = (flat >= 0) & (flat < rows)
        gathered = table[flat.clamp(0, max(rows - 1, 0)).long()]
        gathered = gathered.masked_fill(~valid.unsqueeze(-1), 0)
        out.reshape(-1, width).copy_(gathered)
        return out
    kernel = select_kernel(
        "embedding",
        "host_uint8_row_gather",
        format_signature(x=dense_tensor_format(torch.uint8)),
        features=None,
        platform=None,
        objective=SelectionObjective.DEFAULT,
        traits=None,
        solution=None,
        override=None,
    )
    return kernel(table, indices, out)
