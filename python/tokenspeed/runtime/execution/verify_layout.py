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

import time
from dataclasses import dataclass, field

import torch
from tokenspeed_kernel.ops.metadata import gather_query_rows

from tokenspeed.runtime.execution.confidence_history import AcceptanceHistory
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


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
        self.accept_lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)
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


@dataclass
class _ConfidenceRequest:
    slot: int
    epoch: int
    generation: int
    device_generation_offset: int | None
    history: AcceptanceHistory | None
    rates: dict[int, list[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class _ConfidenceSample:
    # A tag names a request lifetime and this round's published generation.
    tags: tuple[tuple[str, int, int] | None, ...]
    values: torch.Tensor
    generations: torch.Tensor
    start: torch.cuda.Event | None
    end: torch.cuda.Event
    batch: int | None
    queries: int | None


class ConfidenceFeedback:
    """Asynchronous, generation-checked scores and live graph cost feedback.

    A budget for draft generation N uses generation N-2. Missing feedback
    uses observed conditional acceptance, with a Beta(1, 1) cold-start prior.
    Request scores survive batches that do not schedule the request. Slot
    reuse bounds this expendable host cache without changing device state.
    """

    def __init__(
        self, width: int, max_pending: int, max_batch_size: int, *, budget_policy: str
    ):
        from tokenspeed_scheduler import ConfidenceBudgetPlanner

        if budget_policy not in ("confidence", "history"):
            raise ValueError("confidence budget policy must be confidence or history")
        self.planner = ConfidenceBudgetPlanner(width)
        self.budget_policy = budget_policy
        self.width = width
        self.max_pending = max_pending
        self._free_buffers = [
            (
                torch.empty(
                    (max_batch_size, 2 * width),
                    dtype=torch.float32,
                    pin_memory=True,
                ),
                torch.empty(max_batch_size, dtype=torch.int64, pin_memory=True),
            )
            for _ in range(max_pending)
        ]
        self.pending: list[_ConfidenceSample] = []
        self._requests: dict[str, _ConfidenceRequest] = {}
        self._slot_owners: dict[int, str] = {}
        self._next_epoch = 0
        self._round_tags: tuple[tuple[str, int, int] | None, ...] | None = None
        self._successes = [1] * (width - 1)
        self._trials = [2] * (width - 1)
        self.selections = 0
        self.selected_requests = 0
        self.fallback_requests = 0
        self.rejected_generations = 0
        self.completed_rounds = 0
        self.allocated_queries = 0
        self.effective_queries = 0
        self.verified_requests = 0
        self.calibration_requests = 0
        self.predicted_tokens = 0.0
        self.accepted_tokens = 0
        self.confidence_error = 0.0
        self.confidence_observations = 0
        self.history_requests = 0
        self.history_censored_requests = 0
        self.history_generation_lag = 0
        self.history_growth_rounds = 0
        self.history_growth_queries = 0
        self.decision_nanoseconds = 0

    def prepare(
        self,
        request_ids: list[str],
        slots: list[int],
        output_layout: ForwardOutputLayout,
    ) -> None:
        self.poll()
        if len(request_ids) != len(slots) or len(slots) != (
            output_layout.num_extends + output_layout.num_decodes
        ):
            raise ValueError("confidence requests must match the forward layout")
        tags = []
        for row, (key, slot) in enumerate(zip(request_ids, slots, strict=True)):
            owner = self._slot_owners.get(slot)
            if owner is not None and owner != key:
                self._requests.pop(owner, None)
            state = self._requests.get(key)
            if state is not None and state.slot != slot:
                self._slot_owners.pop(state.slot, None)
            if state is None or row < output_layout.num_extends:
                self._next_epoch += 1
                state = _ConfidenceRequest(
                    slot,
                    self._next_epoch,
                    0,
                    0 if row < output_layout.num_extends else None,
                    (
                        AcceptanceHistory(self.width, observation_weight=0.25)
                        if self.budget_policy == "history"
                        else None
                    ),
                )
                self._requests[key] = state
            elif state.slot != slot:
                state.slot = slot
            self._slot_owners[slot] = key
            if output_layout.output_width(row):
                state.generation += 1
                tags.append((key, state.epoch, state.generation))
            else:
                tags.append(None)
        self._round_tags = tuple(tags)

    def poll(self) -> None:
        while self.pending and self.pending[0].end.query():
            sample = self.pending.pop(0)
            values = sample.values[: len(sample.tags)].tolist()
            generations = sample.generations[: len(sample.tags)].tolist()
            if sample.start is not None:
                self.planner.observe(
                    sample.batch, sample.queries, sample.start.elapsed_time(sample.end)
                )
                self.completed_rounds += 1
                self.allocated_queries += sample.queries
                self.effective_queries += sum(int(row[-2]) for row in values)
                self.verified_requests += sum(row[-2] > 0 for row in values)
            for tag, row, generation in zip(
                sample.tags, values, generations, strict=True
            ):
                if tag is None:
                    continue
                key, epoch, expected = tag
                state = self._requests.get(key)
                if state is None or state.epoch != epoch:
                    continue
                if sample.start is not None and row[-2] == 0:
                    # A submitted lookahead can outlive the request. Its null
                    # state destination intentionally publishes no generation.
                    continue
                if state.device_generation_offset is None:
                    # A restored slot may carry an absolute device counter
                    # from before this host cache acquired the request.
                    state.device_generation_offset = generation - expected
                if generation != expected + state.device_generation_offset:
                    self.rejected_generations += 1
                    continue
                rates = row[: self.width - 1]
                state.rates[expected] = rates
                state.rates = {
                    age: scores
                    for age, scores in state.rates.items()
                    if age >= state.generation - 3
                }
                queries, accepted = int(row[-2]), int(row[-1])
                if queries <= 0 or expected < 2:
                    continue
                if state.history is not None:
                    # The newly published scores are generation `expected`;
                    # acceptance describes its predecessor's candidates.
                    state.history.observe(
                        expected - 1, queries=queries, accepted=accepted
                    )
                # These scores accompanied the candidates just verified;
                # newly published scores above are for a later budget.
                rates = row[self.width - 1 : 2 * (self.width - 1)]
                self.calibration_requests += 1
                self.accepted_tokens += accepted
                prefix = 1.0
                self.predicted_tokens += 1.0
                for position in range(queries - 1):
                    probability = rates[position]
                    prefix *= probability
                    self.predicted_tokens += prefix
                    # After the first mismatch, later positions are censored.
                    # Acceptance is recorded before EOS/length truncation.
                    if accepted >= position + 1:
                        success = int(accepted > position + 1)
                        self._successes[position] += success
                        self._trials[position] += 1
                        self.confidence_error += abs(probability - success)
                        self.confidence_observations += 1
            self._free_buffers.append((sample.values, sample.generations))

    def select(self, batch: int, request_ids: list[str]) -> int:
        started = time.perf_counter_ns()
        prior = [
            successes / trials
            for successes, trials in zip(self._successes, self._trials, strict=True)
        ]
        rates = []
        observed_rates = []
        censored = False
        for key in request_ids:
            state = self._requests.get(key)
            scores = (
                state.rates.get(state.generation - 2) if state is not None else None
            )
            if scores is None:
                self.fallback_requests += 1
                scores = prior
            observed = scores
            if (
                self.budget_policy == "history"
                and state is not None
                and state.history is not None
            ):
                forecast = state.history.forecast(
                    scores, maximum_generation=state.generation - 2
                )
                if forecast is not None:
                    scores, observed = forecast.rates, forecast.observed_rates
                    self.history_requests += 1
                    self.history_generation_lag += (
                        state.generation - forecast.generation
                    )
                    self.history_censored_requests += int(forecast.censored)
                    censored |= forecast.censored
            rates.append(scores)
            observed_rates.append(observed)
        self.selections += 1
        self.selected_requests += len(request_ids)
        queries = self.planner.select(batch, rates)
        if censored:
            observed_queries = self.planner.select(batch, observed_rates)
            if queries > observed_queries:
                self.history_growth_rounds += 1
                self.history_growth_queries += queries - observed_queries
        self.decision_nanoseconds += time.perf_counter_ns() - started
        if self.selections % 256 == 0:
            logger.info(f"Confidence verification: {self.summary()}")
        return queries

    def summary(self) -> dict[str, float | int | str]:
        """Return completed, same-round budget and confidence diagnostics."""
        return {
            "budget_policy": self.budget_policy,
            "rounds": self.completed_rounds,
            "mean_q": self.allocated_queries / max(1, self.completed_rounds),
            "queries_per_request": self.effective_queries
            / max(1, self.verified_requests),
            "padding_fraction": 1
            - self.effective_queries / max(1, self.allocated_queries),
            "predicted_length": self.predicted_tokens
            / max(1, self.calibration_requests),
            "accepted_length": self.accepted_tokens / max(1, self.calibration_requests),
            "confidence_mae": self.confidence_error
            / max(1, self.confidence_observations),
            "fallback_fraction": self.fallback_requests
            / max(1, self.selected_requests),
            "rejected_generations": self.rejected_generations,
            "history_fraction": self.history_requests / max(1, self.selected_requests),
            "history_censored_fraction": self.history_censored_requests
            / max(1, self.history_requests),
            "history_generation_lag": self.history_generation_lag
            / max(1, self.history_requests),
            "history_growth_rounds": self.history_growth_rounds,
            "history_growth_queries": self.history_growth_queries,
            "decision_time_us": self.decision_nanoseconds
            / max(1, self.selections)
            / 1000,
        }

    def can_record(self) -> bool:
        return len(self.pending) < self.max_pending

    def record(
        self,
        current_rates: torch.Tensor,
        current_generations: torch.Tensor,
        older_rates: torch.Tensor,
        query_lengths: torch.Tensor,
        accept_lengths: torch.Tensor,
        start: torch.cuda.Event | None,
        batch: int | None,
        queries: int | None,
    ) -> None:
        tags = self._round_tags
        self._round_tags = None
        if tags is None or not self.can_record():
            return
        host, generations = self._free_buffers.pop()
        count = len(tags)
        values = torch.cat(
            (
                current_rates,
                older_rates,
                query_lengths[:, None].to(torch.float32),
                accept_lengths[:, None].to(torch.float32),
            ),
            dim=-1,
        )
        host[:count].copy_(values, non_blocking=True)
        generations[:count].copy_(current_generations, non_blocking=True)
        end = torch.cuda.Event(enable_timing=start is not None)
        end.record()
        self.pending.append(
            _ConfidenceSample(tags, host, generations, start, end, batch, queries)
        )
