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

"""Completed confidence feedback keyed by request lifetime and generation."""

import logging
import math
import time
from collections import deque
from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass, field

import torch

from tokenspeed.runtime.execution.confidence_history import AcceptanceHistory
from tokenspeed.runtime.execution.drafter.verify_budget import VerifyBudgetPlanner
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout

logger = logging.getLogger(__name__)


@dataclass
class _RequestFeedback:
    slot: int
    epoch: int
    generation: int
    completed: dict[int, tuple[float, ...]] = field(default_factory=dict)
    contexts: dict[int, int] = field(default_factory=dict)
    device_generation_offset: int | None = None
    history: AcceptanceHistory | None = None


@dataclass(frozen=True)
class FeedbackSnapshot:
    request_id: Hashable
    epoch: int
    generation: int


@dataclass(frozen=True)
class PendingFeedback:
    snapshot: tuple[FeedbackSnapshot | None, ...]
    values: tuple[torch.Tensor, ...]
    ready: torch.cuda.Event
    timing: tuple[torch.cuda.Event, int, int] | None


class FeedbackQueue:
    """Keep pinned readbacks alive until the DMA and their CPU consumer finish."""

    def __init__(self, max_pending: int):
        self.max_pending = max_pending
        self.pending: deque[PendingFeedback] = deque()

    def can_record(self) -> bool:
        return len(self.pending) < self.max_pending

    def record(
        self,
        snapshot: tuple[FeedbackSnapshot | None, ...],
        sources: tuple[torch.Tensor, ...],
        *,
        buffers: tuple[torch.Tensor, ...] | None,
        timing: tuple[torch.cuda.Event, int, int] | None,
    ) -> None:
        if not self.can_record():
            raise RuntimeError("verification feedback queue is full")
        if buffers is None:
            buffers = tuple(
                torch.empty(x.shape, dtype=x.dtype, device="cpu", pin_memory=True)
                for x in sources
            )
        for source, host in zip(sources, buffers, strict=True):
            host[: len(snapshot)].copy_(source, non_blocking=True)
        ready = torch.cuda.Event(enable_timing=timing is not None)
        ready.record()
        self.pending.append(PendingFeedback(snapshot, buffers, ready, timing))

    def completed(self) -> Iterator[PendingFeedback]:
        while self.pending and self.pending[0].ready.query():
            yield self.pending.popleft()


class VerifyFeedback:
    def __init__(self, *, draft_width: int):
        if type(draft_width) is not int or draft_width < 1:
            raise ValueError("draft_width must be positive")
        self.draft_width = draft_width
        self._requests: dict[Hashable, _RequestFeedback] = {}
        self._slots: dict[int, Hashable] = {}
        self._active: tuple[Hashable, ...] = ()
        self._epoch = 0

    def prepare(
        self, request_ids: Sequence[Hashable], slots: Sequence[int], *, num_extends: int
    ) -> None:
        self._prepare(request_ids, slots, num_extends=num_extends, produced=None)

    def prepare_proposals(
        self,
        request_ids: Sequence[Hashable],
        slots: Sequence[int],
        *,
        num_extends: int,
        produced: Sequence[bool],
    ) -> tuple[FeedbackSnapshot | None, ...]:
        """Advance only rows that will publish proposals, including their first one."""
        if len(produced) != len(request_ids):
            raise ValueError("proposal flags must match the active requests")
        self._prepare(request_ids, slots, num_extends=num_extends, produced=produced)
        return tuple(
            tag if advances else None
            for tag, advances in zip(self.snapshot(), produced, strict=True)
        )

    def _prepare(
        self,
        request_ids: Sequence[Hashable],
        slots: Sequence[int],
        *,
        num_extends: int,
        produced: Sequence[bool] | None,
    ) -> None:
        if (
            len(request_ids) != len(slots)
            or len(set(request_ids)) != len(slots)
            or len(set(slots)) != len(slots)
        ):
            raise ValueError("request identities and unique slots must align")
        if not 0 <= num_extends <= len(slots) or any(slot < 0 for slot in slots):
            raise ValueError("invalid extend count or slot")
        self._active = tuple(request_ids)
        for row, (request_id, slot) in enumerate(zip(request_ids, slots, strict=True)):
            if slot in self._slots and self._slots[slot] != request_id:
                self._requests.pop(self._slots[slot])
            request = self._requests.get(request_id)
            moved = request is not None and request.slot != slot
            if moved:
                self._slots.pop(request.slot)
            restart = (
                request is None or row < num_extends or (produced is None and moved)
            )
            if restart:
                self._epoch += 1
                request = _RequestFeedback(slot, self._epoch, 0)
                if produced is not None and row < num_extends:
                    request.device_generation_offset = 0
                self._requests[request_id] = request
            request.slot = slot
            self._slots[slot] = request_id
            advances = not restart if produced is None else produced[row]
            if advances:
                request.generation += 1
            minimum = request.generation - (2 if produced is None else 3)
            request.completed = {
                g: scores for g, scores in request.completed.items() if g >= minimum
            }
            request.contexts = {
                g: length for g, length in request.contexts.items() if g >= minimum
            }

    def snapshot(self) -> tuple[FeedbackSnapshot, ...]:
        return tuple(
            FeedbackSnapshot(
                key, self._requests[key].epoch, self._requests[key].generation
            )
            for key in self._active
        )

    def request(self, request_id: Hashable) -> _RequestFeedback | None:
        return self._requests.get(request_id)

    def resolve(self, snapshot: FeedbackSnapshot) -> _RequestFeedback | None:
        request = self.request(snapshot.request_id)
        return (
            request if request is not None and request.epoch == snapshot.epoch else None
        )

    def publish(
        self,
        snapshot: Sequence[FeedbackSnapshot | None],
        scores: Sequence[Sequence[float]],
        *,
        context_lengths: Sequence[int] | None = None,
    ) -> None:
        """Publish only after the associated device-to-host copy has completed."""
        if len(snapshot) != len(scores):
            raise ValueError("feedback rows do not match their captured identities")
        if context_lengths is not None and (
            len(context_lengths) != len(scores)
            or any(type(length) is not int or length < 1 for length in context_lengths)
        ):
            raise ValueError("feedback context lengths must be positive and aligned")
        for index, (identity, row) in enumerate(zip(snapshot, scores, strict=True)):
            if identity is None:
                continue
            current = self.resolve(identity)
            if (
                current is None
                or not current.generation - 2
                <= identity.generation
                <= current.generation
                or identity.generation in current.completed
            ):
                continue
            if len(row) != self.draft_width or any(not math.isfinite(v) for v in row):
                continue
            current.completed[identity.generation] = tuple(row)
            if context_lengths is not None:
                current.contexts[identity.generation] = context_lengths[index]

    def scores(self) -> tuple[tuple[float, ...], ...] | None:
        """Use exact N-2 per request; incomplete or stale feedback means fallback."""
        result = []
        for key in self._active:
            current = self._requests[key]
            scores = current.completed.get(current.generation - 2)
            if scores is None:
                return None
            result.append(scores)
        return tuple(result) if result else None

    def context_bounds(self) -> tuple[int, int] | None:
        """Bound current contexts from N-2 lengths without a device readback.

        At most the full verify width can be committed per intervening round.
        Retraction/state import must invalidate this history before reuse.
        """
        lengths = []
        for key in self._active:
            current = self._requests[key]
            length = current.contexts.get(current.generation - 2)
            if length is None:
                return None
            lengths.append(length)
        if not lengths:
            return None
        return min(lengths), max(lengths) + 2 * (self.draft_width + 1)

    def invalidate(self) -> None:
        self._requests.clear()
        self._slots.clear()
        self._active = ()


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
        if budget_policy not in ("confidence", "history"):
            raise ValueError("confidence budget policy must be confidence or history")
        self.planner = VerifyBudgetPlanner(width)
        self.budget_policy = budget_policy
        self.width = width
        self.requests = VerifyFeedback(draft_width=width - 1)
        self.queue = FeedbackQueue(max_pending)
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
        self._round_tags: tuple[FeedbackSnapshot | None, ...] | None = None
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
        request_ids: Sequence[Hashable],
        slots: list[int],
        output_layout: ForwardOutputLayout,
    ) -> None:
        self.poll()
        if len(request_ids) != len(slots) or len(slots) != (
            output_layout.num_extends + output_layout.num_decodes
        ):
            raise ValueError("confidence requests must match the forward layout")
        self._round_tags = self.requests.prepare_proposals(
            request_ids,
            slots,
            num_extends=output_layout.num_extends,
            produced=tuple(
                bool(output_layout.output_width(row)) for row in range(len(slots))
            ),
        )

    def poll(self) -> None:
        for sample in self.queue.completed():
            host, device_generations = sample.values
            values = host[: len(sample.snapshot)].tolist()
            generations = device_generations[: len(sample.snapshot)].tolist()
            if sample.timing is not None:
                start, batch, queries = sample.timing
                self.planner.observe(batch, queries, start.elapsed_time(sample.ready))
                self.completed_rounds += 1
                self.allocated_queries += queries
                self.effective_queries += sum(int(row[-2]) for row in values)
                self.verified_requests += sum(row[-2] > 0 for row in values)
            for tag, row, generation in zip(
                sample.snapshot, values, generations, strict=True
            ):
                if tag is None:
                    continue
                expected = tag.generation
                state = self.requests.resolve(tag)
                if state is None:
                    continue
                if sample.timing is not None and row[-2] == 0:
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
                if expected >= state.generation - 3:
                    state.completed[expected] = tuple(row[: self.width - 1])
                queries, accepted = int(row[-2]), int(row[-1])
                if queries <= 0 or expected < 2:
                    continue
                if self.budget_policy == "history":
                    if state.history is None:
                        state.history = AcceptanceHistory(
                            self.width, observation_weight=0.25
                        )
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
            self._free_buffers.append(sample.values)

    def select(self, batch: int, request_ids: Sequence[Hashable]) -> int:
        started = time.perf_counter_ns()
        prior = [
            successes / trials
            for successes, trials in zip(self._successes, self._trials, strict=True)
        ]
        rates = []
        observed_rates = []
        censored = False
        for key in request_ids:
            state = self.requests.request(key)
            scores = (
                state.completed.get(state.generation - 2) if state is not None else None
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
        return self.queue.can_record()

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
        timing = None
        if start is not None:
            if batch is None or queries is None:
                raise ValueError("timed confidence feedback requires B and Q")
            timing = (start, batch, queries)
        values = torch.cat(
            (
                current_rates,
                older_rates,
                query_lengths[:, None].to(torch.float32),
                accept_lengths[:, None].to(torch.float32),
            ),
            dim=-1,
        )
        self.queue.record(
            tags,
            (values, current_generations),
            buffers=self._free_buffers.pop(),
            timing=timing,
        )
