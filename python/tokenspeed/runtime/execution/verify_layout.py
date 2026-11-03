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

"""Graph-stable selection of contiguous speculative prefixes.

The request window retains its maximum width for cache ownership and state
commit. Query indices describe the subset computed by row-local model work.
They are refreshed on the execution stream; no tensor value reaches Python.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.metadata import gather_query_rows


class VerifyQueryLayout:
    """Persistent query maps shared by eager execution and graph replay.

    Args:
        batch_size: Request capacity, including graph padding.
        query_capacity: Static number of selected model rows.
        width: Maximum input window, including the anchor.
        device: Device of the model inputs.
    """

    def __init__(self, batch_size: int, query_capacity: int, width: int, device):
        if (
            batch_size < 1
            or width < 1
            or not batch_size <= query_capacity <= batch_size * width
        ):
            raise ValueError("query capacity must lie between B and B * width")
        self.batch_size = batch_size
        self.query_capacity = query_capacity
        self.width = width
        self.query_lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.query_offsets = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device
        )
        self.packed_rows = torch.zeros(query_capacity, dtype=torch.int64, device=device)
        self.packed_valid = torch.zeros(query_capacity, dtype=torch.bool, device=device)
        self.request_indices = torch.zeros(
            query_capacity, dtype=torch.int64, device=device
        )
        self.columns = torch.zeros(query_capacity, dtype=torch.int64, device=device)
        self.expanded_rows = torch.zeros(
            batch_size * width, dtype=torch.int64, device=device
        )
        self.expanded_valid = torch.zeros(
            batch_size * width, dtype=torch.bool, device=device
        )
        self._requests = torch.arange(batch_size, device=device)
        self._columns = torch.arange(width, device=device)
        self._queries = torch.arange(query_capacity, device=device)

    def refresh(
        self, rates: torch.Tensor, ready: torch.Tensor, active: torch.Tensor
    ) -> None:
        """Allocate the budget by prefix probability, with stable tie breaking.

        Every active request gets its anchor. Fresh requests have no extras.
        Non-finite probabilities are treated as zero. Selected positions form
        a prefix because cumulative probabilities are monotonically decreasing.
        """
        if rates.shape != (self.batch_size, self.width - 1):
            raise ValueError("confidence shape must match the request window")
        probabilities = torch.nan_to_num(
            rates.float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0, 1)
        scores = probabilities.cumprod(-1)
        eligible = ready & active
        scores = torch.where(eligible[:, None], scores, -torch.ones_like(scores))
        flat = scores.flatten()
        # Ineligible entries sort last, while equal prefix scores keep request
        # then position order. A prefix cannot lose to its own later position.
        order = torch.argsort(flat, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(0, order, torch.arange(order.numel(), device=order.device))
        extra_budget = self.query_capacity - active.to(torch.int32).sum()
        wins = (ranks.view_as(scores) < extra_budget) & eligible[:, None]
        self.query_lengths.copy_(active.to(torch.int32) + wins.sum(-1).to(torch.int32))
        self.query_offsets[0].zero_()
        torch.cumsum(self.query_lengths, 0, out=self.query_offsets[1:])
        requests = torch.searchsorted(self.query_offsets[1:], self._queries, right=True)
        valid = requests < self.batch_size
        safe_requests = requests.clamp_max(self.batch_size - 1)
        columns = self._queries - self.query_offsets[safe_requests]
        self.packed_valid.copy_(valid)
        self.request_indices.copy_(safe_requests)
        self.columns.copy_(torch.where(valid, columns, -1))
        self.packed_rows.copy_(
            torch.where(valid, safe_requests * self.width + columns, -1)
        )
        expanded = self.query_offsets[:-1, None] + self._columns
        self.expanded_rows.copy_(expanded.flatten().clamp_max(self.query_capacity - 1))
        self.expanded_valid.copy_(
            (self._columns < self.query_lengths[:, None]).flatten()
        )

    def pack(self, values: torch.Tensor, *, token_dim: int) -> torch.Tensor:
        """Gather selected query rows; graph filler rows have zero values."""
        if token_dim == 0:
            return self.pack_rows((values,))[0]
        result = values.index_select(token_dim, self.packed_rows.clamp_min(0))
        shape = [1] * result.ndim
        shape[token_dim] = self.query_capacity
        return torch.where(
            self.packed_valid.view(shape), result, torch.zeros_like(result)
        )

    def expand(
        self, values: torch.Tensor, *, token_dim: int, request_count: int | None = None
    ) -> torch.Tensor:
        """Restore maximum-width scratch geometry without exposing filler rows."""
        count = self.batch_size if request_count is None else request_count
        if not 0 <= count <= self.batch_size:
            raise ValueError("expanded requests must fit the query layout")
        rows = self.expanded_rows[: count * self.width]
        valid = self.expanded_valid[: count * self.width]
        if token_dim == 0:
            return gather_query_rows((values,), rows, valid)[0]
        result = values.index_select(token_dim, rows)
        shape = [1] * result.ndim
        shape[token_dim] = count * self.width
        return torch.where(valid.view(shape), result, torch.zeros_like(result))

    def pack_rows(self, values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Gather aligned model inputs in one launch."""
        return gather_query_rows(values, self.packed_rows, self.packed_valid)

    def mask_compact_rows(self, values: torch.Tensor) -> torch.Tensor:
        """Zero bucket filler without reading its unspecified activations."""
        return gather_query_rows((values,), self._queries, self.packed_valid)[0]

    def expand_rows(self, values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Restore aligned model outputs in one launch."""
        return gather_query_rows(values, self.expanded_rows, self.expanded_valid)

    def causal_metadata(
        self, seq_lens: torch.Tensor, page_table: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map full-window metadata to independent causal queries.

        Sequence lengths include the reserved maximum window. Each selected
        column reads through its own input token; filler reads the null page.
        """
        pages = page_table.index_select(0, self.request_indices)
        pages = torch.where(self.packed_valid[:, None], pages, 0)
        lengths = (
            seq_lens.index_select(0, self.request_indices)
            - self.width
            + self.columns.to(seq_lens.dtype)
            + 1
        ).clamp_min(1)
        lengths = torch.where(self.packed_valid, lengths, 1)
        return pages, lengths


class ConfidenceFeedback:
    """Bounded asynchronous timing and older-score feedback for graph selection."""

    def __init__(self, width: int, max_pending: int, max_batch_size: int):
        from tokenspeed_scheduler import ConfidenceBudgetPlanner

        self.planner = ConfidenceBudgetPlanner(width)
        self.width = width
        self.max_pending = max_pending
        self._free_buffers = [
            torch.empty(
                (max_batch_size, width - 1), dtype=torch.float32, pin_memory=True
            )
            for _ in range(max_pending)
        ]
        self.pending: list[
            tuple[
                tuple[str, ...],
                torch.Tensor,
                torch.cuda.Event,
                torch.cuda.Event,
                int,
                int,
            ]
        ] = []
        self.rates: dict[str, list[float]] = {}

    def poll(self, request_ids: list[str]) -> None:
        while self.pending and self.pending[0][3].query():
            ids, rates, start, end, batch, queries = self.pending.pop(0)
            self.rates.update(zip(ids, rates[: len(ids)].tolist(), strict=True))
            self.planner.observe(batch, queries, start.elapsed_time(end))
            self._free_buffers.append(rates)
        # Bound host state to the requests in this round. A returning request
        # can use the startup curve until its next completed feedback arrives.
        active = set(request_ids)
        self.rates = {key: value for key, value in self.rates.items() if key in active}

    def select(self, batch: int, request_ids: list[str]) -> int:
        rates = [self.rates.get(key, [1.0] * (self.width - 1)) for key in request_ids]
        return self.planner.select(batch, rates)

    def can_record(self) -> bool:
        return len(self.pending) < self.max_pending

    def record(
        self,
        request_ids: list[str],
        older_rates: torch.Tensor,
        start: torch.cuda.Event,
        batch: int,
        queries: int,
    ) -> None:
        host = self._free_buffers.pop()
        host[: older_rates.shape[0]].copy_(older_rates, non_blocking=True)
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.pending.append((tuple(request_ids), host, start, end, batch, queries))
