# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
#
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

"""Which rows of the MoE input are real tokens, for the expert load counters.

A padded forward -- a decode graph replayed at a ladder batch size, a prefill
graph replayed at a bucket, the idle replay of an attention-DP rank with no
work -- feeds filler rows through every MoE layer, and under attention DP the
filler of every rank is interleaved with the real rows in the all-gathered
MoE input. The router counts routes per physical expert inside the forward
(graph-captured), so it cannot know which rows are filler from the host
ints that decided the padding. ``ExpertLoadRowMask`` carries that knowledge
onto the device: one ``[max_rows]`` bool buffer, True for real rows, that the
graph owners mark before a padded replay and clear right after it, on the
same stream. Eager forwards are never padded and read the all-True mask;
``record_expert_load`` counts a route iff its row is marked. The buffer is
reserved once before the first forward, so captured graphs hold its address.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tokenspeed.runtime.distributed.comm_manager import moe_input_row_segments
from tokenspeed.runtime.distributed.mapping import Mapping

__all__ = ["ExpertLoadRowMask", "LayerExpertLoad"]


class ExpertLoadRowMask:
    """The live-row mask of the MoE input, shared by every MoE layer.

    Args:
        mapping: The parallel layout; it decides how the ranks' rows are laid
            out in the MoE input (``moe_input_row_segments``).
    """

    def __init__(self, mapping: Mapping) -> None:
        self._mapping = mapping
        # [max_rows] bool on the placement's device; None until reserved.
        self.mask: torch.Tensor | None = None

    @property
    def max_rows(self) -> int:
        if self.mask is None:
            raise RuntimeError("the expert load row mask has not been reserved")
        return self.mask.shape[0]

    def reserve(self, max_rows: int, device: torch.device | str) -> None:
        """Allocate the mask (all rows live) before the first forward.

        Args:
            max_rows: Rows of the largest MoE input any forward can carry
                (the MoE TP-EP group size times the per-rank forward bound).
            device: Where the MoE layers run.
        """
        if self.mask is not None:
            raise RuntimeError("the expert load row mask is already reserved")
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        self.mask = torch.ones(max_rows, dtype=torch.bool, device=device)

    def rows(self, num_rows: int) -> torch.Tensor:
        """The ``[num_rows]`` view the router reads for a forward of that many rows."""
        if self.mask is None:
            raise RuntimeError(
                "the expert load row mask has not been reserved; "
                "ModelRunner.prepare_communication_runtime reserves it"
            )
        if num_rows > self.mask.shape[0]:
            raise ValueError(
                f"a forward with {num_rows} MoE rows exceeds the reserved "
                f"{self.mask.shape[0]}"
            )
        return self.mask[:num_rows]

    def mark_padded(
        self,
        *,
        padded_global_num_tokens: Sequence[int],
        live_global_num_tokens: Sequence[int],
    ) -> None:
        """Mark the filler rows of a padded forward before it is replayed.

        Args:
            padded_global_num_tokens: Rows every rank feeds the model, by
                global rank (uniform under a graph replay).
            live_global_num_tokens: Real token rows per global rank.
        """
        segments = moe_input_row_segments(
            self._mapping,
            padded_global_num_tokens=padded_global_num_tokens,
            live_global_num_tokens=live_global_num_tokens,
        )
        total = sum(rows for rows, _ in segments)
        mask = self.rows(total)
        if all(live == rows for rows, live in segments):
            mask.fill_(True)
            return
        host = torch.zeros(total, dtype=torch.bool, pin_memory=mask.is_cuda)
        start = 0
        for rows, live in segments:
            host[start : start + live] = True
            start += rows
        # A fresh pinned staging per call: a reused one would race the next
        # forward's host writes against this copy (see InputBuffers._bulk_pinned).
        mask.copy_(host, non_blocking=True)

    def clear(self) -> None:
        """Every row counts again: called right after a padded replay."""
        if self.mask is None:
            raise RuntimeError("the expert load row mask has not been reserved")
        self.mask.fill_(True)


class LayerExpertLoad:
    """One MoE layer's view of the load counters.

    Args:
        physical_load: The layer's ``[num_physical_experts]`` int64 row of the
            placement's counters (a view, so captured graphs and an in-place
            table switch keep reading the same storage).
        rows: The model-wide live-row mask.
    """

    def __init__(self, physical_load: torch.Tensor, rows: ExpertLoadRowMask) -> None:
        self.physical_load = physical_load
        self.rows = rows

    def record(self, topk_ids: torch.Tensor) -> None:
        """Count every real route of ``topk_ids`` into the counters.

        ``-1`` marks zero-expert and masked routes; filler rows are not
        routes either. One ``scatter_add_`` with a zero weight for those keeps
        this graph-capturable.

        Args:
            topk_ids: ``[rows, top_k]`` physical ids of this layer's routing.
        """
        valid = (topk_ids >= 0) & self.rows.rows(topk_ids.shape[0]).unsqueeze(1)
        self.physical_load.scatter_add_(
            0,
            topk_ids.masked_fill(~valid, 0).reshape(-1).long(),
            valid.reshape(-1).to(self.physical_load.dtype),
        )
