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

"""Pool-bound MLA DCP metadata shared by attention leaves."""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.attention.dcp.metadata import (
    CompactDCPLayout,
    CompactDCPMetadata,
    refresh_dcp_page_table_metadata,
)
from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement


class DCPDecodeState:
    """Own pool-bound DCP decode storage and its refresh operations.

    Per-batch CompactDCPMetadata objects are views of this storage. The leaf
    caches those views alongside its ordinary decode views and drops this
    owner when rebinding the cache pool.
    """

    def __init__(self, placement: CachePlacement):
        self.placement = placement
        self.metadata: CompactDCPMetadata | None = None
        self._visible_lens_buf: torch.Tensor | None = None
        self._query_offsets_buf: torch.Tensor | None = None

    def initialize(
        self,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        page_size: int,
        query_count: int,
        causal: bool,
    ) -> None:
        device = page_table.device
        self._visible_lens_buf = torch.zeros(
            (page_table.shape[0], query_count), dtype=torch.int32, device=device
        )
        self._query_offsets_buf = (
            torch.arange(1 - query_count, 1, dtype=torch.int32, device=device)
            if causal
            else torch.zeros(query_count, dtype=torch.int32, device=device)
        )
        self.metadata = refresh_dcp_page_table_metadata(
            page_table=page_table,
            virtual_block_count=self.placement.virtual_block_count,
            degree=len(self.placement.group),
            rank=self.placement.rank,
            layout=CompactDCPLayout(
                seq_lens,
                page_size,
                self.placement.block_granularity,
                self._global_visible_lengths(seq_lens),
            ),
            previous=None,
        )

    def refresh(self, metadata: CompactDCPMetadata, seq_lens: torch.Tensor) -> None:
        """Rebuild local tables and ownership prefixes at the existing addresses."""
        refresh_dcp_page_table_metadata(
            page_table=metadata.virtual_page_table,
            virtual_block_count=self.placement.virtual_block_count,
            degree=len(self.placement.group),
            rank=self.placement.rank,
            layout=CompactDCPLayout(
                seq_lens,
                metadata.page_size,
                self.placement.block_granularity,
                self._global_visible_lengths(seq_lens),
            ),
            previous=metadata,
        )

    def refresh_lengths(
        self, metadata: CompactDCPMetadata, seq_lens: torch.Tensor
    ) -> None:
        """Reuse the ownership prefix when only the draft frontier changes."""
        metadata.refresh_visible_lengths(self._global_visible_lengths(seq_lens))

    def _global_visible_lengths(self, seq_lens: torch.Tensor) -> torch.Tensor:
        visible = self._visible_lens_buf[: seq_lens.shape[0]]
        torch.add(seq_lens[:, None], self._query_offsets_buf, out=visible)
        # Initialization starts with zero lengths; causal offsets can be negative.
        # Prefix-table lookups require nonnegative endpoints even for dummy rows.
        visible.clamp_min_(0)
        return visible
