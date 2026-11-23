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

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class _RequestFeedback:
    request_id: object
    epoch: int
    generation: int
    completed: dict[int, tuple[float, ...]] = field(default_factory=dict)
    contexts: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class FeedbackSnapshot:
    slot: int
    epoch: int
    generation: int


class VerifyFeedback:
    def __init__(self, *, draft_width: int):
        if type(draft_width) is not int or draft_width < 1:
            raise ValueError("draft_width must be positive")
        self.draft_width = draft_width
        self.slots: dict[int, _RequestFeedback] = {}
        self.active: tuple[int, ...] = ()
        self.epoch = 0

    def prepare(
        self, request_ids: Sequence[object], slots: Sequence[int], *, num_extends: int
    ) -> None:
        if len(request_ids) != len(slots) or len(set(slots)) != len(slots):
            raise ValueError("request identities and unique slots must align")
        if not 0 <= num_extends <= len(slots) or any(slot < 0 for slot in slots):
            raise ValueError("invalid extend count or slot")
        self.active = tuple(slots)
        for row, (request_id, slot) in enumerate(zip(request_ids, slots, strict=True)):
            previous = self.slots.get(slot)
            if (
                previous is None
                or previous.request_id != request_id
                or row < num_extends
            ):
                self.epoch += 1
                self.slots[slot] = _RequestFeedback(request_id, self.epoch, 0)
            else:
                previous.generation += 1
                previous.completed = {
                    g: scores
                    for g, scores in previous.completed.items()
                    if g >= previous.generation - 2
                }

                previous.contexts = {
                    g: length
                    for g, length in previous.contexts.items()
                    if g >= previous.generation - 2
                }

    def snapshot(self) -> tuple[FeedbackSnapshot, ...]:
        return tuple(
            FeedbackSnapshot(slot, self.slots[slot].epoch, self.slots[slot].generation)
            for slot in self.active
        )

    def publish(
        self,
        snapshot: Sequence[FeedbackSnapshot],
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
            current = self.slots.get(identity.slot)
            if (
                current is None
                or current.epoch != identity.epoch
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
        for slot in self.active:
            current = self.slots[slot]
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
        for slot in self.active:
            current = self.slots[slot]
            length = current.contexts.get(current.generation - 2)
            if length is None:
                return None
            lengths.append(length)
        if not lengths:
            return None
        return min(lengths), max(lengths) + 2 * (self.draft_width + 1)

    def invalidate(self) -> None:
        self.slots.clear()
        self.active = ()
