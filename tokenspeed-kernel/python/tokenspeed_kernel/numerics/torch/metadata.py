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

"""PyTorch references for ``tokenspeed_kernel.ops.metadata``."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "metadata",
    "advance_accepted_frontier",
    name="torch_advance_accepted_frontier",
    solution="torch",
    signatures=format_signatures(
        "req_pool_indices", "dense", {torch.int32, torch.int64}
    ),
    priority=Priority.REFERENCE,
)
def torch_advance_accepted_frontier(
    req_pool_indices: torch.Tensor,
    input_lengths: torch.Tensor,
    accept_lengths: torch.Tensor,
    valid_cache_lengths: torch.Tensor,
    num_extends: int,
    padding_index: int,
) -> None:
    """Advance each request's valid cache length, in place: the first
    ``num_extends`` requests by their input length, the rest by their
    accepted length; padding rows add nothing."""
    deltas = torch.cat([input_lengths[:num_extends], accept_lengths[num_extends:]])
    deltas = torch.where(req_pool_indices != padding_index, deltas, 0).to(torch.int32)
    valid_cache_lengths.index_add_(0, req_pool_indices, deltas)


@register_kernel(
    "metadata",
    "expand_block_table_rows",
    name="torch_expand_block_table_rows",
    solution="torch",
    signatures=format_signatures("rows", "dense", {torch.int32}),
    priority=Priority.REFERENCE,
)
def torch_expand_block_table_rows(
    rows: torch.Tensor, values: torch.Tensor, out: torch.Tensor
) -> None:
    """Fill each block-table row of ``out`` from its span, one row at a time:
    null holes (0) before the span, its ``values``, then padding (-1)."""
    for offset, width, start, end, source in rows.tolist():
        row = out[offset : offset + width]
        row[:start] = 0
        row[start:end] = values[source : source + end - start]
        row[end:] = -1
