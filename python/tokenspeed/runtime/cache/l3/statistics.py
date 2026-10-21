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

"""Bounded, rank-local I/O observations; no scheduling or backend decisions."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    state: str
    epoch: int
    failures: int


@dataclass(frozen=True, slots=True)
class OperationSnapshot:
    kind: str
    events: tuple[tuple[str, int], ...]
    success_keys: int
    failed_keys: int
    success_bytes: int
    queue_seconds: float
    backend_seconds: float
    operations: tuple[tuple[str, int], ...]
    bytes: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class ExecutorSnapshot:
    operations: tuple[OperationSnapshot, ...]
    health: HealthSnapshot
    unsafe_io: bool


@dataclass(slots=True)
class OperationCounters:
    accepted: int = 0
    backend_calls: int = 0
    completed: int = 0
    cancelled_queued: int = 0
    cancelled_running: int = 0
    rejected: int = 0
    unretired: int = 0
    success_keys: int = 0
    failed_keys: int = 0
    success_bytes: int = 0
    queue_seconds: float = 0.0
    backend_seconds: float = 0.0

    def snapshot(self, kind, *, operations, byte_counts):
        return OperationSnapshot(
            kind=kind,
            events=(
                ("accepted", self.accepted),
                ("backend_calls", self.backend_calls),
                ("completed", self.completed),
                ("cancelled_queued", self.cancelled_queued),
                ("cancelled_running", self.cancelled_running),
                ("rejected", self.rejected),
                ("unretired", self.unretired),
            ),
            success_keys=self.success_keys,
            failed_keys=self.failed_keys,
            success_bytes=self.success_bytes,
            queue_seconds=self.queue_seconds,
            backend_seconds=self.backend_seconds,
            operations=tuple(operations.items()),
            bytes=tuple(byte_counts.items()),
        )
