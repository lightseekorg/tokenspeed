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

"""Validated local replicas of scheduler-published load snapshots."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from tokenspeed.runtime.engine.io_struct import GetLoadReqOutput, LoadSnapshot


@dataclass(frozen=True)
class _StoredSnapshot:
    snapshot: LoadSnapshot
    received_at: float


class LoadSnapshotStore:
    """Keep one ordered, locally-expiring snapshot for each DP rank."""

    def __init__(
        self, dp_size: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if dp_size <= 0:
            raise ValueError("dp_size must be positive")
        self._dp_size = dp_size
        self._clock = clock
        self._snapshots: dict[int, _StoredSnapshot] = {}
        self._retired_epochs: list[set[str]] = [set() for _ in range(dp_size)]

    def accept(self, snapshot: LoadSnapshot) -> bool:
        """Validate and store a newer snapshot, returning whether it was accepted."""
        if not self._is_valid(snapshot):
            return False

        rank = snapshot.dp_rank
        if snapshot.epoch in self._retired_epochs[rank]:
            return False

        previous = self._snapshots.get(rank)
        if previous is not None:
            if snapshot.epoch == previous.snapshot.epoch:
                if snapshot.sequence <= previous.snapshot.sequence:
                    return False
            else:
                self._retired_epochs[rank].add(previous.snapshot.epoch)

        copied = LoadSnapshot(
            snapshot.epoch,
            snapshot.sequence,
            snapshot.dp_rank,
            snapshot.num_running_reqs,
            snapshot.num_waiting_reqs,
            snapshot.num_active_pages,
            snapshot.num_used_pages,
            snapshot.max_total_pages,
            snapshot.valid_for_ms,
        )
        self._snapshots[rank] = _StoredSnapshot(copied, self._clock())
        return True

    def fresh_snapshots(self) -> list[LoadSnapshot]:
        """Return every currently fresh snapshot in rank order."""
        return self._fresh_snapshots(self._clock())

    def _fresh_snapshots(self, now: float) -> list[LoadSnapshot]:
        return [
            stored.snapshot
            for _, stored in sorted(self._snapshots.items())
            if self._is_fresh(stored, now)
        ]

    def project_loads(self) -> list[GetLoadReqOutput]:
        """Return the public load schema only for a complete, fresh replica."""
        snapshots = self._fresh_snapshots(self._clock())
        if len(snapshots) != self._dp_size:
            return []
        return [
            GetLoadReqOutput(
                dp_rank=snapshot.dp_rank,
                num_reqs=snapshot.num_running_reqs + snapshot.num_waiting_reqs,
                num_waiting_reqs=snapshot.num_waiting_reqs,
                num_pages=snapshot.num_used_pages,
            )
            for snapshot in snapshots
        ]

    def is_complete_fresh(self) -> bool:
        """Whether every expected rank has a currently valid snapshot."""
        return len(self._fresh_snapshots(self._clock())) == self._dp_size

    def _is_valid(self, snapshot: LoadSnapshot) -> bool:
        if not isinstance(snapshot, LoadSnapshot):
            return False
        if not 0 <= snapshot.dp_rank < self._dp_size:
            return False
        if snapshot.sequence < 0 or snapshot.valid_for_ms < 0:
            return False
        if any(
            value < 0
            for value in (
                snapshot.num_running_reqs,
                snapshot.num_waiting_reqs,
                snapshot.num_active_pages,
                snapshot.num_used_pages,
            )
        ):
            return False
        if snapshot.max_total_pages <= 0:
            return False
        return (
            snapshot.num_active_pages <= snapshot.max_total_pages
            and snapshot.num_used_pages <= snapshot.max_total_pages
        )

    @staticmethod
    def _is_fresh(stored: _StoredSnapshot, now: float) -> bool:
        return now - stored.received_at <= stored.snapshot.valid_for_ms / 1_000
