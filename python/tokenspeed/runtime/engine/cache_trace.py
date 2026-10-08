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


"""Bounded, opt-in diagnostic capture. No prompt text or token IDs are written."""

import atexit
import hashlib
import json
import logging
import os
import queue
import threading
import time
import uuid

logger = logging.getLogger(__name__)


class CacheTraceWriter:
    def __init__(self, path: str, metadata: dict):
        self.epoch = uuid.uuid4().hex
        compatibility = metadata.get("compatibility")
        metadata = {
            key: value for key, value in metadata.items() if key != "compatibility"
        }
        self.metadata = {
            "schema": 2,
            "epoch": self.epoch,
            "clock": "unix_ns",
            "compatibility_fingerprint": (
                hashlib.sha256(
                    json.dumps(
                        {
                            "model_config": compatibility,
                            "prefix_granularity": metadata.get("prefix_granularity"),
                            "groups": metadata.get("groups"),
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                if compatibility is not None
                else None
            ),
            "compatibility_scope": "configuration_only",
            **metadata,
        }
        self._history_complete = False
        self._history_lost = False
        self._last_sequence = 0
        self.path = f"{path}.{metadata['global_rank']}.{self.epoch}.jsonl"
        self._queue = queue.Queue(maxsize=16)
        self._dropped = 0
        self._failed = False
        self._closed = False
        self._bytes = 0
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._file = os.fdopen(fd, "w", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def publish(self, events: list[dict]) -> None:
        if not events or self._failed or self._closed:
            return
        batch = events
        if self._dropped:
            batch = [
                {
                    "kind": "gap",
                    "reason": "writer_queue_full",
                    "dropped_events": self._dropped,
                },
                *events,
            ]
        try:
            self._queue.put_nowait(batch)
        except queue.Full:
            self._dropped += len(events)
        else:
            self._dropped = 0

    def _write(self, event: dict) -> None:
        # The native start, not capture_start, certifies an empty index. Gaps
        # permanently invalidate this epoch, even if subsequent events resume.
        sequence = event.get("sequence")
        if sequence is not None:
            if sequence != self._last_sequence + 1:
                self._history_lost = True
            self._last_sequence = sequence
        if event.get("kind") == "gap":
            self._history_lost = True
        if event.get("kind") == "start":
            self._history_complete = event.get("reason") == "empty" and sequence == 1
        self._history_complete = self._history_complete and not self._history_lost
        metadata = (
            self.metadata
            if event.get("kind") == "capture_start"
            else {"schema": self.metadata["schema"], "epoch": self.epoch}
        )
        record = {**metadata, "history_complete": self._history_complete, **event}
        record.setdefault("timestamp_ns", time.time_ns())
        line = json.dumps(record, separators=(",", ":")) + "\n"
        self._file.write(line)
        self._bytes += len(line.encode("utf-8"))

    def _run(self) -> None:
        try:
            self._write({"kind": "capture_start", "history_complete": False})
            while True:
                try:
                    batch = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if not self._closed:
                        continue
                    batch = None
                if batch is None:
                    if self._dropped:
                        self._write(
                            {
                                "kind": "gap",
                                "reason": "writer_queue_full",
                                "dropped_events": self._dropped,
                            }
                        )
                    self._write({"kind": "capture_end"})
                    break
                for event in batch:
                    if self._bytes >= 2 * 1024 * 1024 * 1024:
                        self._write({"kind": "gap", "reason": "retention_limit"})
                        self._failed = True
                        return
                    self._write(event)
                self._file.flush()
        except (OSError, ValueError):
            self._failed = True
            logger.exception("Cache diagnostic capture failed")
        finally:
            self._file.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(None, timeout=1)
        except queue.Full:
            logger.warning("Cache diagnostic capture ended with an undrained queue")
        self._thread.join(timeout=1)
