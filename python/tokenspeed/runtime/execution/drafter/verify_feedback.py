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
from collections import deque
from collections.abc import Collection, Hashable, Iterator, Sequence
from dataclasses import dataclass, field

import torch

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
        ready = torch.cuda.Event()
        ready.record()
        self.pending.append(PendingFeedback(snapshot, buffers, ready))

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
            minimum = request.generation - 2
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
    """Read completed proposal probabilities without blocking decode.

    The host budget uses generation N-2, with a 0.5 conditional prior for
    requests missing that generation. The GPU uses current proposal scores.
    """

    def __init__(self, width: int, max_pending: int, max_batch_size: int):
        self.planner = VerifyBudgetPlanner(width)
        self.requests = VerifyFeedback(draft_width=width - 1)
        self.queue = FeedbackQueue(max_pending)
        self._prior = (0.5,) * (width - 1)
        self._free_buffers = [
            (
                torch.empty(
                    (max_batch_size, width - 1), dtype=torch.float32, pin_memory=True
                ),
                torch.empty(max_batch_size, dtype=torch.int64, pin_memory=True),
            )
            for _ in range(max_pending)
        ]
        self._round_tags: tuple[FeedbackSnapshot | None, ...] | None = None
        self.selections = 0
        self.selected_requests = 0
        self.fallback_requests = 0
        self.rejected_generations = 0

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
            count = len(sample.snapshot)
            for tag, rates, generation in zip(
                sample.snapshot,
                host[:count].tolist(),
                device_generations[:count].tolist(),
                strict=True,
            ):
                if tag is None or generation <= 0:
                    continue
                state = self.requests.resolve(tag)
                if state is None or tag.generation < state.generation - 2:
                    continue
                if state.device_generation_offset is None:
                    # A restored slot can carry an absolute counter from
                    # before the host bound this admission.
                    state.device_generation_offset = generation - tag.generation
                if generation != tag.generation + state.device_generation_offset:
                    self.rejected_generations += 1
                    continue
                self.requests.publish((tag,), (rates,))
            self._free_buffers.append(sample.values)

    def select(
        self,
        batch: int,
        request_ids: Sequence[Hashable],
        *,
        capacities: Collection[int] | None = None,
    ) -> int:
        rates = []
        for key in request_ids:
            state = self.requests.request(key)
            scores = (
                state.completed.get(state.generation - 2) if state is not None else None
            )
            if scores is None:
                self.fallback_requests += 1
                scores = self._prior
            rates.append(scores)
        self.selections += 1
        self.selected_requests += len(request_ids)
        queries = self.planner.select(batch, rates, capacities=capacities)
        if self.selections % 256 == 0:
            logger.info(f"Confidence verification: {self.summary()}")
        return queries

    def summary(self) -> dict[str, float | int]:
        return {
            "fallback_fraction": self.fallback_requests
            / max(1, self.selected_requests),
            "rejected_generations": self.rejected_generations,
        }

    def can_record(self) -> bool:
        return self.queue.can_record()

    def record(self, rates: torch.Tensor, generations: torch.Tensor) -> None:
        tags = self._round_tags
        self._round_tags = None
        if tags is None or not self.can_record():
            return
        self.queue.record(tags, (rates, generations), buffers=self._free_buffers.pop())
