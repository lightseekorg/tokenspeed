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

"""Logical selection metadata for grouped QSA prefill implementations."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QSAPrefillMetadata:
    """Logical representation of the same candidates as ``selected_slots``.

    ``selected_blocks`` is int32 [tokens, block_topk]. Each row starts with
    min(block_topk, (query_position + 1) // block_size) distinct completed
    block IDs; unused entries follow that prefix. The incomplete causal tail
    is implicit. ``block_table`` maps [request, logical_page] to physical pages.
    ``token_to_request`` is int32 [tokens], and ``query_positions`` contains
    zero-based causal positions. ``query_start_loc`` is the CPU cumulative
    query length tuple, including both endpoints, so groups never cross requests.
    ``page_size`` is the physical cache page size; ``block_size`` is the
    indexer's compression ratio. Cache padding must contain finite values.
    """

    selected_blocks: torch.Tensor
    block_table: torch.Tensor
    token_to_request: torch.Tensor
    query_positions: torch.Tensor
    query_start_loc: tuple[int, ...]
    page_size: int
    block_size: int

    def validate(self, rows: int, selected_width: int) -> None:
        """Check metadata geometry without reading GPU tensor values."""
        if self.selected_blocks.ndim != 2 or self.selected_blocks.shape[0] != rows:
            raise ValueError("QSA logical blocks must have one row per query")
        if self.block_size < 1 or self.page_size < 1:
            raise ValueError("QSA block and page sizes must be positive")
        if (
            selected_width
            != self.selected_blocks.shape[1] * self.block_size + self.block_size - 1
        ):
            raise ValueError(
                "QSA logical blocks and physical slots have different budgets"
            )
        if self.block_table.ndim != 2:
            raise ValueError("QSA block table must have one row per request")
        if self.token_to_request.shape != (rows,) or self.query_positions.shape != (
            rows,
        ):
            raise ValueError(
                "QSA request IDs and positions must have one entry per query"
            )
        offsets = self.query_start_loc
        if (
            len(offsets) != self.block_table.shape[0] + 1
            or not offsets
            or offsets[0] != 0
            or offsets[-1] != rows
            or any(end < begin for begin, end in zip(offsets, offsets[1:]))
        ):
            raise ValueError("QSA query offsets must cover the requests and query rows")
