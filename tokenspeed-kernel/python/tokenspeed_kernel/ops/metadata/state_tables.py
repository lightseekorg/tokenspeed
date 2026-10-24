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

"""Pad the state rows of stopped requests in a batch's block tables.

A decode dispatched before the round that ended its request landed must not
touch the request's live state block: its state tables' row becomes a
padding row (every slot -1). On CUDA (and ROCm) devices one Triton launch
masks every state group's table in place; other devices run the same as
tensor ops.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["mask_stopped_state_tables_"]

_BLOCK = 64


@triton.jit
def _mask_stopped_state_tables_kernel(
    stopped_ptr,
    req_pool_indices_ptr,
    tables,
    row_strides,
    widths,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    stopped = tl.load(stopped_ptr + tl.load(req_pool_indices_ptr + row)) != 0
    if stopped:
        for g in tl.static_range(len(tables)):
            width = widths[g]
            for start in range(0, width, BLOCK):
                cols = start + tl.arange(0, BLOCK)
                tl.store(
                    tables[g] + row * row_strides[g] + cols,
                    tl.full((BLOCK,), -1, tl.int32),
                    mask=cols < width,
                )


def mask_stopped_state_tables_(
    tables: Sequence[torch.Tensor],
    stopped: torch.Tensor,
    req_pool_indices: torch.Tensor,
) -> None:
    """Pad the rows of stopped requests in the state tables, in place.

    The tables keep their storage, so views of one packed upload stay views.

    Args:
        tables: ``[batch, width]`` int32 block tables (rows may be strided).
        stopped: ``[pool]`` bool, whether each request slot has stopped.
        req_pool_indices: ``[batch]`` int64 request slot of every row.
    """
    if not tables:
        return
    batch = req_pool_indices.shape[0]
    for t in tables:
        if t.dtype != torch.int32 or t.dim() != 2 or t.shape[0] != batch:
            raise ValueError(f"tables must be int32 [{batch}, width], got {t.shape}")
        if t.stride(1) != 1:
            raise ValueError("table rows must be dense")
    if tables[0].device.type != "cuda":
        rows = stopped.index_select(0, req_pool_indices)[:, None]
        for t in tables:
            t.masked_fill_(rows, -1)
        return
    if batch == 0:
        return
    _mask_stopped_state_tables_kernel[(batch,)](
        stopped.view(torch.uint8),
        req_pool_indices,
        tuple(tables),
        tuple(t.stride(0) for t in tables),
        tuple(t.shape[1] for t in tables),
        BLOCK=_BLOCK,
    )
