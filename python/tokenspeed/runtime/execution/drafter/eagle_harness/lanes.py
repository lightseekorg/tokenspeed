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

"""EAGLE-4's draft-tree lanes: the draft attention's lane inputs for EAGLE-4
trees, through the ``TreeDraftInputs`` hooks
(docs/design/tree-speculation.md)."""

from __future__ import annotations

import torch
from typing_extensions import override

from tokenspeed.runtime.layers.attention.backends.paged.base import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (
    TreeCascadeRows,
    TreeDraftInputs,
)

__all__ = ["Eagle4TreeLanes"]


class Eagle4TreeLanes(TreeDraftInputs):
    """The lanes of EAGLE-4's chains and trees (static or EAGLE-Harness).

    Beyond the plain lanes:

    * Lane windows up to 128 slots (EAGLE-Harness drafts past the verify
      budget): a row's ancestor mask is ``mask_words`` int64 words, window
      slot ``j`` at bit ``j % 64`` of word ``j // 64``.
    * A sliding-window draft layer (``slides``) sees the window frozen at
      the round's prefix end: each lane row decodes the committed prefix
      alone, over a copy of its request's table row.
    """

    def __init__(
        self,
        topk: int,
        num_steps: int,
        max_bs: int,
        device: torch.device,
        *,
        slides: bool,
    ) -> None:
        super().__init__(topk, num_steps, max_bs, device)
        self.max_bs = max_bs
        # Only a sliding-window draft layer reads the per-row tables and frontiers.
        self.slides = slides
        self.mask_words = (self.num_slots + 63) // 64
        self.lane_mask = torch.zeros(
            (max_bs * topk * self.mask_words,), dtype=torch.int64, device=device
        )
        # The committed keys per lane row (sliding-window rows decode the prefix alone).
        self.frontier_rows = torch.zeros(
            (max_bs * topk,), dtype=torch.int32, device=device
        )
        # Per leaf, its page table per lane row; reallocated when a rebind resizes the leaf's table.
        self._row_tables: dict[PagedAttentionBackend, torch.Tensor] = {}

    @override
    def set_frontier(self, bs: int, frontier: torch.Tensor) -> None:
        super().set_frontier(bs, frontier)
        if not self.slides:
            return
        k = self.topk
        self.frontier_rows[: bs * k].view(bs, k).copy_(
            frontier[:bs, None].expand(-1, k)
        )

    @override
    def cascade_rows(self, leaf: PagedAttentionBackend, bs: int) -> TreeCascadeRows:
        k = self.topk
        table = leaf.page_table_buf[:bs]
        row_table = row_prefix_lens = None
        if self.slides:
            pages = leaf.page_table_buf.shape[1]
            row_tables = self._row_tables.get(leaf)
            if row_tables is None or row_tables.shape[1] != pages:
                row_tables = table.new_zeros((self.max_bs * k, pages))
                self._row_tables[leaf] = row_tables
            row_table = row_tables[: bs * k]
            # The lane attention kernels read dense page-table rows.
            row_table.view(bs, k, -1).copy_(table[:, None].expand(-1, k, -1))
            row_prefix_lens = self.frontier_rows[: bs * k]
        return TreeCascadeRows(
            page_table=table,
            seq_lens=self.window_seq_lens[:bs],
            prefix_lens=self.frontier[:bs],
            mask=self.lane_mask[: bs * k * self.mask_words],
            rows=k,
            window=self.num_slots,
            row_page_table=row_table,
            row_prefix_lens=row_prefix_lens,
        )
