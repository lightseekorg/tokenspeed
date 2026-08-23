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

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import zmq

from tokenspeed.runtime.engine.io_struct import (
    GetLoadReqOutput,
    LoadSnapshot,
    MsgpackEncoder,
)

logger = logging.getLogger(__name__)

_LoadValues = tuple[int, int, int, int, int]
_SocketFactory = Callable[[str], tuple[object, object]]
_SEND_RETRY_INTERVAL_S = 0.01
_CLOSE_TIMEOUT_S = 1.0


def _close_snapshot_socket(context: object | None, socket: object | None) -> None:
    """Close a partially or fully acquired socket before terminating its context."""
    try:
        if socket is not None:
            socket.close(linger=0)
    finally:
        if context is not None:
            context.term()


def _open_snapshot_socket(endpoint: str) -> tuple[zmq.Context, zmq.Socket]:
    """Create the publisher's private context and configured PUSH socket."""
    context = zmq.Context()
    socket = None
    try:
        socket = context.socket(zmq.PUSH)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.connect(endpoint)
        return context, socket
    except Exception:
        try:
            _close_snapshot_socket(context, socket)
        except Exception:
            logger.exception("Failed to clean up load snapshot socket setup")
        raise


class NullLoadSnapshotPublisher:
    """No-op publisher used outside standard-serving attention TP0."""

    @staticmethod
    def observe(values: _LoadValues) -> None:
        return None

    @staticmethod
    def close() -> None:
        return None


class DirectLoadSnapshotSink:
    """Project shared observations onto the next direct-ZMQ output batch."""

    def __init__(self, set_load_snapshot: Callable[[int, int, int, int], None]) -> None:
        self._set_load_snapshot = set_load_snapshot

    def observe(self, values: _LoadValues) -> None:
        num_running, num_waiting, num_active_pages, _, max_total_pages = values
        self._set_load_snapshot(
            num_running,
            num_waiting,
            num_active_pages,
            max_total_pages,
        )

    @staticmethod
    def close() -> None:
        return None


class LoadSnapshotPublisher:
    """Publish the newest scheduler load tuple from a private transport thread.

    ``observe`` is the scheduler-thread boundary: it only compares a tuple,
    replaces one guarded slot, and notifies the publisher. The background
    thread owns all snapshot encoding and ZMQ context/socket lifecycle work.
    """

    def __init__(
        self,
        endpoint: str,
        dp_rank: int,
        heartbeat_interval: float,
        socket_factory: _SocketFactory | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._dp_rank = dp_rank
        self._heartbeat_interval = max(1.0, heartbeat_interval)
        self._valid_for_ms = int(3 * self._heartbeat_interval * 1_000)
        self._socket_factory = socket_factory or _open_snapshot_socket
        self._epoch = uuid.uuid4().hex

        self._condition = threading.Condition()
        self._latest_values: _LoadValues | None = None
        self._generation = 0
        self._closed = False
        self._sequence = 0
        self.socket_owner_thread: int | None = None

        self._thread = threading.Thread(
            target=self._run,
            name=f"load-snapshot-publisher-dp{dp_rank}",
            daemon=True,
        )
        self._thread.start()

    def observe(self, values: _LoadValues) -> None:
        """Replace the mailbox when scalar load state changes."""
        with self._condition:
            if self._closed or values == self._latest_values:
                return
            self._latest_values = values
            self._generation += 1
            self._condition.notify()

    def close(self) -> None:
        """Stop the publisher without waiting indefinitely for transport."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify()
        self._thread.join(timeout=_CLOSE_TIMEOUT_S)
        if self._thread.is_alive():
            logger.warning(
                "Load snapshot publisher did not stop within %.1fs", _CLOSE_TIMEOUT_S
            )

    def _run(self) -> None:
        self.socket_owner_thread = threading.get_ident()
        context = None
        socket = None
        try:
            context, socket = self._socket_factory(self._endpoint)
            encoder = MsgpackEncoder()
            pending: LoadSnapshot | None = None
            pending_generation = -1
            sent_generation = -1
            last_sent_at: float | None = None

            while True:
                with self._condition:
                    while True:
                        if self._closed:
                            return

                        generation = self._generation
                        values = self._latest_values
                        if values is not None and generation != pending_generation:
                            if pending is not None or generation != sent_generation:
                                pending = self._new_snapshot(values)
                                pending_generation = generation
                                break

                        if pending is not None:
                            break

                        if values is None or last_sent_at is None:
                            self._condition.wait()
                            continue

                        heartbeat_remaining = (
                            last_sent_at + self._heartbeat_interval - time.monotonic()
                        )
                        if heartbeat_remaining <= 0:
                            pending = self._new_snapshot(values)
                            pending_generation = generation
                            break
                        self._condition.wait(timeout=heartbeat_remaining)

                frames = encoder.encode(pending)
                if len(frames) != 1:
                    raise RuntimeError("LoadSnapshot must encode as exactly one frame")
                try:
                    socket.send(frames[0], flags=zmq.NOBLOCK)
                except zmq.Again:
                    with self._condition:
                        if not self._closed:
                            self._condition.wait(timeout=_SEND_RETRY_INTERVAL_S)
                    continue

                sent_generation = pending_generation
                last_sent_at = time.monotonic()
                pending = None
        except Exception:
            logger.exception("Load snapshot publisher stopped unexpectedly")
        finally:
            try:
                _close_snapshot_socket(context, socket)
            except Exception:
                logger.exception("Failed to clean up load snapshot publisher")

    def _new_snapshot(self, values: _LoadValues) -> LoadSnapshot:
        self._sequence += 1
        return LoadSnapshot(
            self._epoch,
            self._sequence,
            self._dp_rank,
            *values,
            self._valid_for_ms,
        )


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

        now = self._clock()
        rank = snapshot.dp_rank
        if snapshot.epoch in self._retired_epochs[rank]:
            return False

        previous = self._snapshots.get(rank)
        if previous is not None:
            if snapshot.epoch == previous.snapshot.epoch:
                if snapshot.sequence <= previous.snapshot.sequence:
                    return False
            else:
                if self._is_fresh(previous, now):
                    return False
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
        self._snapshots[rank] = _StoredSnapshot(copied, now)
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
