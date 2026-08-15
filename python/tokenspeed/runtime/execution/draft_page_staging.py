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

"""Persistent staging of the full-history table for in-graph draft consumers.

The multi-step drafters derive their KV write locations inside the captured
graph (the inputs depend on verify's in-graph ``accept_lengths``), so the
kernels record a fixed table address at capture time. Per-forward group
tables are fresh tensors; this component owns the one address-stable buffer
they are staged into, and the single publish path that fills it.

Invariants:

* The staged table is batch-ordered — row ``i`` IS batch position ``i``.
* Ids are draft-kernel pages; the logical→kernel expansion happens here and
  nowhere else.
* Rows ``[bs, padded_bs)`` are scrubbed as part of every publish: graph
  replay reads ``padded_bs`` rows, and a stale id past ``bs`` aliases
  another request's pages (#955).
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.execution.cache_loc_kernel import (
    compute_out_cache_loc_sliding,
    compute_out_cache_loc_uniform,
)
from tokenspeed.runtime.layers.attention.page_table import expand_page_table


class CacheView:
    """The drafter's window onto the staged page table.

    Wraps the address-stable staged table with the write-location math so
    drafters express intent ("slots for the next k positions of this batch")
    without touching page ids or page-size arithmetic. The kernel is picked
    by the group's retention: ``full_history`` walks columns monotonically,
    ``sliding_window`` maps positions onto the table's ring; ``state``
    groups have no page table and
    no view.

    ``table`` and ``kernel_page_size`` remain readable for consumers that hand the
    staged placeholder to attention-metadata inits.
    """

    def __init__(
        self,
        table: torch.Tensor,
        kernel_page_size: int,
        retention: str = "full_history",
    ) -> None:
        if retention not in ("full_history", "sliding_window"):
            raise ValueError(f"unsupported cache view retention {retention!r}")
        self.table = table
        self.kernel_page_size = int(kernel_page_size)
        self.retention = retention

    @property
    def max_tokens(self) -> int:
        """Token capacity of one table row (width × kernel page size)."""
        return self.table.shape[1] * self.kernel_page_size

    def out_cache_loc_uniform(
        self,
        out: torch.Tensor,
        cache_start: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Resolve KV write slots for ``num_tokens`` positions per request.

        Captured-graph friendly (fixed table address, no host sync). Row
        ``i`` of ``cache_start`` is batch position ``i``'s first write
        position; slots land in ``out`` request-major.

        Args:
            out: Flat int64 output, at least ``bs * num_tokens`` long.
            cache_start: ``[bs]`` int32 first write position per request.
            num_tokens: Positions per request (uniform across the batch).

        Returns:
            ``out`` for chaining.
        """
        if self.retention == "sliding_window":
            compute_out_cache_loc_sliding(
                out_cache_loc_ptr=out,
                uniform_input_length=num_tokens,
                cache_start=cache_start,
                page_table=self.table,
                page_size=self.kernel_page_size,
            )
            return out
        compute_out_cache_loc_uniform(
            out_cache_loc_ptr=out,
            uniform_input_length=num_tokens,
            cache_start=cache_start,
            page_table=self.table,
            page_size=self.kernel_page_size,
        )
        return out


class DraftPageStaging:
    """Owns the batch-ordered draft page table and its only publish path."""

    def __init__(
        self,
        *,
        max_bs: int,
        max_pages_per_req: int,
        block_granularity: int,
        draft_kernel_page_size: int,
        full_history_group_id: str | None,
        enabled: bool,
        device,
    ) -> None:
        """Allocate the address-stable staged table.

        Args:
            max_bs: Widest replayable batch; the table's row count.
            max_pages_per_req: Table width in draft-kernel pages.
            block_granularity: Grain of the incoming scheduler tables in tokens.
            draft_kernel_page_size: Draft backend's kernel page size (the staged
                unit). ``block_granularity`` must be a positive multiple.
            full_history_group_id: Group whose table is staged; None when the
                contract has no full-history group (the table then stays a
                zeros placeholder for idle/warmup consumers).
            enabled: False when the draft path does not read this table at
                all (e.g. DeepSeek-V4 consumes group tables directly);
                publish is then scrub-only.
            device: CUDA device for the persistent buffer.
        """
        if block_granularity % draft_kernel_page_size:
            raise ValueError(
                f"block granularity {block_granularity} is not a multiple "
                f"of the draft kernel page size {draft_kernel_page_size}"
            )
        self.block_granularity = int(block_granularity)
        self.draft_kernel_page_size = int(draft_kernel_page_size)
        self.page_ratio = self.block_granularity // self.draft_kernel_page_size
        self.full_history_group_id = full_history_group_id
        self.enabled = enabled
        self.table = torch.zeros(
            (max_bs, max_pages_per_req), dtype=torch.int32, device=device
        )
        self.view = CacheView(self.table, self.draft_kernel_page_size)

    def publish(self, block_tables, bs: int, padded_bs: int) -> None:
        """Stage this forward's full-history table for the draft's kernels.

        Args:
            block_tables: Per-group tables of the current batch (may be None
                or empty on idle/warmup forwards).
            bs: Number of live requests; rows ``[0, bs)`` are written.
            padded_bs: Rows the upcoming replay reads; ``[bs, padded_bs)``
                are scrubbed to the null page.
        """
        # Replay reads padded_bs rows; stale ids past bs alias another
        # request's pages. Scrub before the copy so an early return below
        # (idle, no table) still leaves the padded rows inert.
        if padded_bs > bs:
            self.table[bs:padded_bs].zero_()
        if (
            not self.enabled
            or bs <= 0
            or not block_tables
            or self.full_history_group_id is None
        ):
            return
        table = block_tables.get(self.full_history_group_id)
        if table is None:
            return
        rows = self.table[:bs]
        max_width = self.table.shape[1]
        if self.page_ratio > 1:
            # -1 pads clamp into table page 0, itself reserved as the null page.
            expand_page_table(
                table,
                block_granularity=self.block_granularity,
                kernel_page_size=self.draft_kernel_page_size,
                max_kernel_pages=max_width,
                out=rows,
            )
            return
        # -1 column pads -> dummy page 0 (negative locs otherwise).
        width = table.shape[1]
        rows[:, :width].copy_(table)
        rows[:, :width].clamp_min_(0)
        if width < max_width:
            rows[:, width:].zero_()
