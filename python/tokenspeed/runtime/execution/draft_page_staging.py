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
they are staged into, the single publish path that fills it, and the
write-location math over it — drafters express intent ("slots for the next
k positions of this batch") without touching page ids or page-size
arithmetic.

Invariants:

* The staged table is batch-ordered — row ``i`` IS batch position ``i``.
* Ids are RAW scheduler pages of ``block_granularity`` tokens; publish is a
  pure copy, no page mapping. Absolute slot math is page-size invariant
  (``table[i, pos // P] * P + pos % P`` equals the same expression over any
  kernel-page expansion), so location consumers use ``block_granularity``
  directly and backends expand into their own kernel pages themselves.
* Rows ``[bs, padded_bs)`` are scrubbed as part of every publish: graph
  replay reads ``padded_bs`` rows, and a stale id past ``bs`` aliases
  another request's pages (#955).
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.execution.cache_loc_kernel import (
    compute_out_cache_loc_uniform,
)


class DraftPageStaging:
    """Owns the batch-ordered draft page table, its only publish path, and
    the write-location math the drafters run over it.

    ``table`` and ``block_granularity`` remain readable for consumers that
    hand the staged table to attention-metadata inits (the table's unit is
    the raw scheduler page).
    """

    def __init__(
        self,
        *,
        max_bs: int,
        max_pages_per_req: int,
        block_granularity: int,
        full_history_group_id: str | None,
        device,
    ) -> None:
        """Allocate the address-stable staged table.

        Args:
            max_bs: Widest replayable batch; the table's row count.
            max_pages_per_req: Table width in raw scheduler pages.
            block_granularity: Page size of the staged ids in tokens.
            full_history_group_id: Group whose table is staged; None when the
                contract has no full-history group (the table then stays a
                zeros placeholder for idle/warmup consumers).
            device: CUDA device for the persistent buffer.
        """
        self.block_granularity = int(block_granularity)
        self.full_history_group_id = full_history_group_id
        self.table = torch.zeros(
            (max_bs, max_pages_per_req), dtype=torch.int32, device=device
        )

    @property
    def max_tokens(self) -> int:
        """Token capacity of one table row (width × page size)."""
        return self.table.shape[1] * self.block_granularity

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
        compute_out_cache_loc_uniform(
            out_cache_loc_ptr=out,
            uniform_input_length=num_tokens,
            cache_start=cache_start,
            page_table=self.table,
            page_size=self.block_granularity,
        )
        return out

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
        if bs <= 0 or not block_tables or self.full_history_group_id is None:
            return
        table = block_tables.get(self.full_history_group_id)
        if table is None:
            return
        rows = self.table[:bs]
        max_width = self.table.shape[1]
        width = min(table.shape[1], max_width)
        # -1 column pads -> dummy page 0 (negative locs otherwise).
        rows[:, :width].copy_(table[:, :width])
        rows[:, :width].clamp_min_(0)
        if width < max_width:
            rows[:, width:].zero_()
