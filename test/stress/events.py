"""Structured events + JSONL sink.

Every interesting thing that happens during a stress run -- a request starting,
getting its first token, being cancelled, a health-probe status flip -- is
recorded as a single JSON object on its own line. Post-run tooling consumes
this file; nothing else needs a shared in-memory schema.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

# Kinds of events. Keep these stable; the report tooling matches on them.
REQUEST_SUBMITTED = "request_submitted"
REQUEST_FIRST_TOKEN = "request_first_token"
REQUEST_COMPLETED = "request_completed"
REQUEST_CANCELLED = "request_cancelled"
REQUEST_ERROR = "request_error"
# Emitted when a grammar-constrained response fails JSON parsing or schema
# validation. Carries {rid, error_kind, detail, content_preview}.
# DEPRECATED: superseded by AUDIT_FINDING (check="json_schema"); still emitted
# by older event files and aggregated for backward compatibility.
REQUEST_INVALID_SCHEMA = "request_invalid_schema"
# Emitted when a streaming request stops producing tokens for longer than the
# stall timeout (a wedge/hang that the total request timeout would catch only
# much later). Carries {rid, stage, gap_s, workload}.
REQUEST_STALL = "request_stall"
# Emitted per output-quality finding from the auditor pipeline. Carries
# {rid, check, severity, detail, value, workload}.
AUDIT_FINDING = "audit_finding"

HEALTH_PROBE = "health_probe"
HEALTH_TRANSITION = "health_transition"

# Emitted once per RssMonitor poll; carries {root_pid, total_kb, num_pids,
# per_pid: {pid: {rss_kb, comm}}}. On /proc read error: {root_pid, error}.
RSS_PROBE = "rss_probe"

# Emitted once per MetricsMonitor poll; carries scraped server-wide gauges
# (e.g. spec-decode accept_len) plus the raw counters used to derive them.
METRICS_PROBE = "metrics_probe"

RUN_STARTED = "run_started"
RUN_FINISHED = "run_finished"


@dataclass
class Event:
    kind: str
    ts: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        # dataclasses.asdict is fine; `data` is a plain dict so nothing exotic.
        return json.dumps(asdict(self), separators=(",", ":"))


class JsonlSink:
    """Thread-safe append-only JSONL writer. One file per run."""

    def __init__(self, path: Optional[str]):
        self.path = path
        self._fp = None
        self._lock = threading.Lock()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            self._fp = open(path, "a", buffering=1)  # line-buffered

    def emit(self, kind: str, **data: Any) -> None:
        ev = Event(kind=kind, data=data)
        if self._fp is None:
            return
        line = ev.to_json()
        with self._lock:
            self._fp.write(line)
            self._fp.write("\n")

    def close(self) -> None:
        if self._fp is not None:
            with self._lock:
                self._fp.flush()
                self._fp.close()
            self._fp = None


class NullSink(JsonlSink):
    def __init__(self):
        super().__init__(path=None)
