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

"""Shared DCP page-table construction with explicit consumer layouts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import overload

import torch
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import (
    compact_dcp_pages,
    virtual_slots_to_local,
)


@dataclass(frozen=True)
class PositionPreservingDCPLayout:
    """Scheduler-block table retaining null/foreign entries as -1."""


@dataclass(frozen=True)
class CompactDCPLayout:
    """Dense kernel-page table packed in token order, with local lengths."""

    seq_lens: torch.Tensor
    page_size: int
    block_granularity: int


@dataclass(frozen=True)
class DCPPageTableMetadata:
    """Common placement state; each consumer explicitly selects its layout.

    Storage belongs to the cache pool. These are only address views. Refresh
    reuses compatible output buffers for both eager and CUDA-graph execution.
    """

    virtual_page_table: torch.Tensor
    local_page_table: torch.Tensor
    virtual_block_count: int
    degree: int
    rank: int


@dataclass(frozen=True)
class PositionPreservingDCPMetadata(DCPPageTableMetadata):
    """One entry per input block, with -1 for null/foreign blocks."""

    owner_mask: torch.Tensor

    def slice_requests(self, start: int, end: int) -> PositionPreservingDCPMetadata:
        return replace(
            self,
            virtual_page_table=self.virtual_page_table[start:end],
            local_page_table=self.local_page_table[start:end],
            owner_mask=self.owner_mask[start:end],
        )


@dataclass(frozen=True)
class CompactDCPMetadata(DCPPageTableMetadata):
    """Owned kernel pages in a contiguous prefix, followed by zero padding."""

    local_seq_lens: torch.Tensor
    page_size: int
    block_granularity: int

    def slice_requests(self, start: int, end: int) -> CompactDCPMetadata:
        return replace(
            self,
            virtual_page_table=self.virtual_page_table[start:end],
            local_page_table=self.local_page_table[start:end],
            local_seq_lens=self.local_seq_lens[start:end],
        )


@overload
def refresh_dcp_page_table_metadata(
    *,
    page_table: torch.Tensor,
    virtual_block_count: int,
    degree: int,
    rank: int,
    layout: PositionPreservingDCPLayout,
    previous: PositionPreservingDCPMetadata | None,
) -> PositionPreservingDCPMetadata: ...


@overload
def refresh_dcp_page_table_metadata(
    *,
    page_table: torch.Tensor,
    virtual_block_count: int,
    degree: int,
    rank: int,
    layout: CompactDCPLayout,
    previous: CompactDCPMetadata | None,
) -> CompactDCPMetadata: ...


def refresh_dcp_page_table_metadata(
    *,
    page_table: torch.Tensor,
    virtual_block_count: int,
    degree: int,
    rank: int,
    layout: PositionPreservingDCPLayout | CompactDCPLayout,
    previous: DCPPageTableMetadata | None,
) -> PositionPreservingDCPMetadata | CompactDCPMetadata:
    """Refresh the explicitly selected DCP layout without an intermediate table.

    Position-preserving input uses scheduler block IDs. Compact input uses
    kernel page IDs, with ``block_granularity / page_size`` pages per ownership
    block; its sequence lengths count tokens, not pages. Only compact layouts
    need lengths. Switching layout/topology/geometry requires fresh metadata.
    """
    if page_table.ndim != 2 or page_table.shape[1] == 0:
        raise ValueError("DCP page_table must have shape [batch, nonzero max_pages]")
    if degree < 1 or not 0 <= rank < degree or virtual_block_count <= 1:
        raise ValueError("invalid DCP placement geometry")
    if not isinstance(layout, (PositionPreservingDCPLayout, CompactDCPLayout)):
        raise TypeError("DCP layout must be explicitly selected")
    compact = isinstance(layout, CompactDCPLayout)
    if compact:
        if (
            layout.page_size <= 0
            or layout.block_granularity <= 0
            or layout.block_granularity % layout.page_size
        ):
            raise ValueError("DCP ownership blocks must contain whole kernel pages")
        if (
            layout.seq_lens.shape != page_table.shape[:1]
            or layout.seq_lens.device != page_table.device
        ):
            raise ValueError("DCP sequence lengths must match page-table rows/device")
    if previous is not None:
        expected_type = CompactDCPMetadata if compact else PositionPreservingDCPMetadata
        if not isinstance(previous, expected_type):
            raise ValueError("DCP metadata layout changed during refresh")
        if previous.degree != degree or previous.rank != rank:
            raise ValueError("DCP metadata topology changed during refresh")
        if previous.virtual_block_count != virtual_block_count:
            raise ValueError("DCP virtual block capacity changed during refresh")
        if isinstance(previous, CompactDCPMetadata):
            assert isinstance(layout, CompactDCPLayout)
            if (
                previous.page_size != layout.page_size
                or previous.block_granularity != layout.block_granularity
            ):
                raise ValueError("DCP page geometry changed during refresh")
        local = previous.local_page_table
        if (
            local.shape != page_table.shape
            or local.dtype != page_table.dtype
            or local.device != page_table.device
            or not local.is_contiguous()
        ):
            # Reallocating here would hand back fresh buffers while a captured
            # graph keeps replaying kernels against the previous ones.
            raise ValueError("DCP metadata buffers do not match the refreshed table")

    if previous is None:
        # Replay setup may run outside the warmup inference context.
        with torch.inference_mode(False):
            local = torch.empty_like(page_table, memory_format=torch.contiguous_format)
            common = dict(
                virtual_page_table=page_table,
                local_page_table=local,
                virtual_block_count=virtual_block_count,
                degree=degree,
                rank=rank,
            )
            if isinstance(layout, CompactDCPLayout):
                previous = CompactDCPMetadata(
                    **common,
                    local_seq_lens=torch.empty_like(layout.seq_lens),
                    page_size=layout.page_size,
                    block_granularity=layout.block_granularity,
                )
            else:
                previous = PositionPreservingDCPMetadata(
                    **common,
                    owner_mask=torch.empty(
                        page_table.shape, dtype=torch.bool, device=page_table.device
                    ),
                )

    result = replace(previous, virtual_page_table=page_table)
    if isinstance(layout, CompactDCPLayout):
        assert isinstance(result, CompactDCPMetadata)
        compact_dcp_pages(
            page_table,
            layout.seq_lens,
            page_size=layout.page_size,
            block_granularity=layout.block_granularity,
            virtual_block_count=virtual_block_count,
            degree=degree,
            rank=rank,
            out=result.local_page_table,
            local_lengths=result.local_seq_lens,
        )
    else:
        assert isinstance(result, PositionPreservingDCPMetadata)
        virtual_slots_to_local(
            page_table,
            rows_per_page=1,
            virtual_block_count=virtual_block_count,
            degree=degree,
            rank=rank,
            out=result.local_page_table,
            owner_mask=result.owner_mask,
        )
        result.local_page_table.masked_fill_(~result.owner_mask, -1)
    return result
