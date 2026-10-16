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

"""Best-effort raw metrics over trusted, node-local IPC.

Producers never serialize or perform I/O in ``emit``. Each producer's bounded
queue feeds its own transport thread, and one writer thread owns the receiving
socket and all file operations. Full queues drop new events. A stopped writer
does not interrupt serving. Closing is bounded and best-effort: PUSH acceptance
is not a disk acknowledgement, and crashes can lose queued or buffered events.
"""

from __future__ import annotations

import fcntl
import json
import logging
import math
import re
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import zmq

from tokenspeed.runtime.metrics.events import (
    MetricConfig,
    MetricEvent,
    MetricEventSink,
    MetricOperation,
)

logger = logging.getLogger(__name__)
_POLL_INTERVAL_MS = 10
_DRAIN_QUIET_SECONDS = 0.05
_SHARD_FILENAME = re.compile(r"metrics-[0-9a-f]{32}-[0-9]{6,}\.jsonl")


@dataclass(frozen=True)
class _QueuedEvent:
    timestamp_ns: int
    metric: str
    operation: MetricOperation
    value: float
    labels: tuple[tuple[str, str], ...]
    weight: int

    def to_dict(self) -> dict:
        return {
            "timestamp_ns": self.timestamp_ns,
            "metric": self.metric,
            "operation": self.operation,
            "value": self.value,
            "labels": dict(self.labels),
            "weight": self.weight,
        }


def _require_ipc(endpoint: str) -> None:
    if not endpoint.startswith("ipc://"):
        raise ValueError("Raw metric transport requires a node-local ipc:// endpoint")


class JsonlMetricSender:
    """Send metric events to a node's writer from a private transport thread.

    Args:
        endpoint: Trusted local writer's ipc:// endpoint.
        queue_size: Maximum queued events, excluding one event being sent and
            ZMQ's bounded outgoing queue of the same size.

    ``dropped_events`` counts local overflow, failure and close-timeout drops;
    it cannot detect events lost after ZMQ accepts them. ``error_count`` counts
    transport/encoding failures. Both are reported by the background thread.
    """

    def __init__(self, endpoint: str, *, queue_size: int) -> None:
        _require_ipc(endpoint)
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self._endpoint = endpoint
        self._queue_size = queue_size
        self._queue: deque[_QueuedEvent] = deque()
        self._condition = threading.Condition()
        self._closing = False
        self._failed = False
        self._close_deadline: float | None = None
        self._dropped_events = 0
        self._error_count = 0
        self._thread = threading.Thread(
            target=self._run, name="raw-metrics-sender", daemon=True
        )
        self._thread.start()

    @property
    def dropped_events(self) -> int:
        with self._condition:
            return self._dropped_events

    @property
    def error_count(self) -> int:
        with self._condition:
            return self._error_count

    def emit(self, event: MetricEvent) -> None:
        """Freeze labels and enqueue, dropping new events when the queue is full."""
        queued = _QueuedEvent(
            event.timestamp_ns,
            event.metric,
            event.operation,
            event.value,
            tuple(event.labels.items()),
            event.weight,
        )
        with self._condition:
            if self._closing or self._failed or len(self._queue) >= self._queue_size:
                self._dropped_events += 1
                return
            self._queue.append(queued)
            self._condition.notify()

    def close(self, timeout: float) -> None:
        """Stop accepting events and try to drain within timeout seconds.

        No delivery acknowledgement is available. At the deadline the thread
        drops remaining local events and closes its socket without further wait.
        The thread may finish its final cleanup just after this call returns.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            self._closing = True
            if self._close_deadline is None:
                self._close_deadline = deadline
            else:
                self._close_deadline = min(self._close_deadline, deadline)
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, timeout))
        if self._thread.is_alive():
            logger.warning(
                "Raw metrics sender close timed out; pending events may drop"
            )

    def _run(self) -> None:
        context = None
        socket = None
        pending: _QueuedEvent | None = None
        last_reported_drops = 0
        last_report_at = 0.0
        encoding_errors = 0
        try:
            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.setsockopt(zmq.SNDHWM, self._queue_size)
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.connect(self._endpoint)
            while True:
                with self._condition:
                    if self._closing and (
                        time.monotonic() >= self._close_deadline
                        or (pending is None and not self._queue)
                    ):
                        break
                    if pending is None:
                        if not self._queue:
                            self._condition.wait(timeout=0.1)
                            continue
                        pending = self._queue.popleft()
                    dropped = self._dropped_events
                now = time.monotonic()
                if dropped != last_reported_drops and now - last_report_at >= 5:
                    logger.warning(f"Raw metrics sender has dropped {dropped} events")
                    last_reported_drops = dropped
                    last_report_at = now
                # IMMEDIATE prevents an absent writer from absorbing the whole
                # queue into an unconnected ZMQ pipe with no observable drops.
                if not socket.poll(_POLL_INTERVAL_MS, zmq.POLLOUT):
                    continue
                try:
                    payload = json.dumps(pending.to_dict(), allow_nan=False).encode()
                except (TypeError, ValueError):
                    with self._condition:
                        self._error_count += 1
                        self._dropped_events += 1
                    encoding_errors += 1
                    if encoding_errors == 1:
                        logger.exception(
                            "Raw metrics sender dropped an unencodable event"
                        )
                    pending = None
                    continue
                try:
                    socket.send(payload, flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                pending = None
        except Exception:
            with self._condition:
                self._error_count += 1
                self._failed = True
            logger.exception("Raw metrics sender stopped after an error")
        finally:
            with self._condition:
                self._dropped_events += len(self._queue) + (pending is not None)
                self._queue.clear()
                remaining = (
                    max(0.0, self._close_deadline - time.monotonic())
                    if self._close_deadline is not None
                    else 0.0
                )
                dropped = self._dropped_events
            if socket is not None:
                socket.close(linger=int(remaining * 1000))
            if context is not None:
                context.term()
            if dropped:
                logger.warning(
                    f"Raw metrics sender closed with {dropped} dropped events"
                )
            if encoding_errors:
                logger.warning(
                    f"Raw metrics sender closed with {encoding_errors} encoding errors"
                )


class JsonlMetricWriter:
    """Own a receiving socket and one rotating JSONL stream in a single thread.

    Args:
        directory: Exclusive output directory, created if needed. A file lock
            prevents another live writer from using the same directory.
        endpoint: Trusted node-local ipc:// address to bind.
        max_bytes: Rotate before a complete line would exceed this many bytes.
            A single larger record is written intact in its own shard.
        backup_count: Number of closed shards to retain, in addition to the
            current shard. Includes previous writer streams in this directory.
        flush_interval: Maximum ordinary buffering interval, in seconds.

    Construction waits for socket/file setup. Runtime setup, serialization and
    disk failures are logged and disable this writer without raising in serving.
    ``written_events`` counts records passed to the file buffer, not fsynced data;
    ``error_count`` counts failures. Sequence and shard numbers start at zero.
    """

    def __init__(
        self,
        directory: str,
        endpoint: str,
        *,
        max_bytes: int,
        backup_count: int,
        flush_interval: float,
    ) -> None:
        _require_ipc(endpoint)
        if max_bytes <= 0 or backup_count < 0 or flush_interval <= 0:
            raise ValueError("Invalid raw metric rotation or flush configuration")
        self.stream_id = uuid.uuid4().hex
        self.written_events = 0
        self.error_count = 0
        self._directory = Path(directory)
        self._endpoint = endpoint
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._flush_interval = flush_interval
        self._directory_lock: BinaryIO | None = None
        self._file: BinaryIO | None = None
        self._active_path: Path | None = None
        self._active_bytes = 0
        self._closed_paths: deque[Path] = deque()
        self._shard = 0
        self._sequence = 0
        self._ready = threading.Event()
        self._closing = threading.Event()
        self._close_deadline: float | None = None
        self._close_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="raw-metrics-writer", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=5):
            logger.warning("Raw metrics writer setup has not completed after 5 seconds")

    def close(self, timeout: float) -> None:
        """Drain available messages and flush, waiting at most timeout seconds.

        Producers must close first. A short quiet interval allows transport
        messages already in flight to arrive; this is not a delivery guarantee.
        A filesystem operation already in progress cannot be interrupted.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._close_lock:
            if self._close_deadline is None:
                self._close_deadline = deadline
            else:
                self._close_deadline = min(self._close_deadline, deadline)
            self._closing.set()
        self._thread.join(timeout=max(0.0, timeout))
        if self._thread.is_alive():
            logger.warning(
                "Raw metrics writer close timed out; buffered data may be lost"
            )

    def _run(self) -> None:
        context = None
        socket = None
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            self._directory_lock = (self._directory / ".metrics.lock").open("a+b")
            fcntl.flock(self._directory_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # The exclusive directory lock makes all previous streams inactive.
            self._closed_paths.extend(
                sorted(
                    (
                        path
                        for path in self._directory.iterdir()
                        if _SHARD_FILENAME.fullmatch(path.name) and path.is_file()
                    ),
                    key=lambda path: (path.stat().st_mtime_ns, path.name),
                )
            )
            context = zmq.Context()
            socket = context.socket(zmq.PULL)
            socket.setsockopt(zmq.RCVHWM, 1024)
            socket.bind(self._endpoint)
            self._open_shard()
            self._prune_shards()
            self._ready.set()
            last_flush = time.monotonic()
            quiet_since: float | None = None
            while True:
                now = time.monotonic()
                if self._closing.is_set() and now >= self._close_deadline:
                    break
                if socket.poll(_POLL_INTERVAL_MS, zmq.POLLIN):
                    quiet_since = None
                    self._write_event(json.loads(socket.recv()))
                elif self._closing.is_set():
                    if quiet_since is None:
                        quiet_since = time.monotonic()
                    elif time.monotonic() - quiet_since >= _DRAIN_QUIET_SECONDS:
                        break
                if time.monotonic() - last_flush >= self._flush_interval:
                    self._file.flush()
                    last_flush = time.monotonic()
        except Exception:
            self.error_count += 1
            logger.exception("Raw metrics writer stopped after an error")
        finally:
            self._ready.set()
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    self.error_count += 1
                    logger.exception(
                        "Raw metrics writer could not flush its final shard"
                    )
            if socket is not None:
                socket.close(linger=0)
            if context is not None:
                context.term()
            if self._directory_lock is not None:
                # Keep the lock file's inode stable across writer restarts.
                self._directory_lock.close()

    def _open_shard(self) -> None:
        self._active_path = self._directory / (
            f"metrics-{self.stream_id}-{self._shard:06d}.jsonl"
        )
        self._file = self._active_path.open("xb")
        self._active_bytes = 0

    def _write_event(self, event: dict) -> None:
        record = {
            **event,
            "schema_version": 1,
            "stream_id": self.stream_id,
            "sequence": self._sequence,
        }
        line = (
            json.dumps(
                record, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            + "\n"
        ).encode("utf-8")
        if self._active_bytes and self._active_bytes + len(line) > self._max_bytes:
            self._file.close()
            self._closed_paths.append(self._active_path)
            self._shard += 1
            self._open_shard()
            self._prune_shards()
        self._file.write(line)
        self._active_bytes += len(line)
        self._sequence += 1
        self.written_events += 1

    def _prune_shards(self) -> None:
        while len(self._closed_paths) > self._backup_count:
            self._closed_paths.popleft().unlink(missing_ok=True)


def validate_jsonl_config(config: MetricConfig) -> MetricConfig:
    """Validate explicit JSONL node settings without creating resources."""
    required = {
        "directory",
        "max_bytes",
        "backup_count",
        "flush_interval",
        "queue_size",
    }
    if set(config) != required:
        missing = sorted(required - config.keys())
        unknown = sorted(config.keys() - required)
        raise ValueError(
            f"Invalid JSONL metrics fields: missing={missing}, unknown={unknown}"
        )
    directory = config["directory"]
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("JSONL metrics directory must be a nonempty string")
    for name in ("max_bytes", "backup_count", "queue_size"):
        value = config[name]
        minimum = 0 if name == "backup_count" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"JSONL metrics {name} must be an integer >= {minimum}")
    interval = config["flush_interval"]
    if type(interval) not in (int, float):
        raise ValueError("JSONL metrics flush_interval must be a number")
    try:
        interval = float(interval)
    except OverflowError:
        raise ValueError("JSONL metrics flush_interval is too large") from None
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("JSONL metrics flush_interval must be finite and positive")
    return {**config, "flush_interval": interval}


class _JsonlNodeSession:
    def __init__(self, config: MetricConfig) -> None:
        self._directory: tempfile.TemporaryDirectory | None = None
        self._writer: JsonlMetricWriter | None = None
        self._producer_config: MetricConfig = {}
        self._lock = threading.Lock()
        try:
            # Keep Unix socket paths below sockaddr_un's platform-dependent limit.
            self._directory = tempfile.TemporaryDirectory(prefix="tsm-", dir="/tmp")
            endpoint = f"ipc://{self._directory.name}/events.sock"
            self._writer = JsonlMetricWriter(
                config["directory"],
                endpoint,
                max_bytes=config["max_bytes"],
                backup_count=config["backup_count"],
                flush_interval=config["flush_interval"],
            )
            self._producer_config = {
                "endpoint": endpoint,
                "queue_size": config["queue_size"],
            }
        except Exception:
            self.close(timeout=2.0)
            raise

    @property
    def producer_config(self) -> MetricConfig:
        return dict(self._producer_config)

    def close(self, timeout: float) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            writer, self._writer = self._writer, None
            directory, self._directory = self._directory, None
            try:
                if writer is not None:
                    writer.close(timeout=max(0.0, deadline - time.monotonic()))
            finally:
                if directory is not None:
                    directory.cleanup()


def start_jsonl_node(config: MetricConfig) -> _JsonlNodeSession:
    """Create one writer and IPC directory owned by this node's launcher."""
    return _JsonlNodeSession(config)


def create_jsonl_sink(config: MetricConfig) -> JsonlMetricSender:
    """Create one process-local sender from a node session's producer config."""
    return JsonlMetricSender(config["endpoint"], queue_size=config["queue_size"])
