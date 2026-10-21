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

"""Bounded L3 observations for the selected metric outputs."""

from tokenspeed.runtime.cache.l3.statistics import ExecutorSnapshot
from tokenspeed.runtime.metrics.events import MetricEventSink
from tokenspeed.runtime.metrics.reporter import MetricFactory


class L3Metrics:
    def __init__(
        self,
        *,
        labels: dict[str, str],
        enabled: bool,
        reporters: list[str],
        event_sink: MetricEventSink | None,
        registry=None,
    ):
        factory = MetricFactory(
            enabled=enabled,
            reporters=reporters,
            event_sink=event_sink,
            registry=registry,
        )

        self.labels = dict(labels)
        self._exported: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        base = list(labels)
        self._events = factory.counter(
            name="tokenspeed:l3_io_events",
            documentation="Local executor events; backend_calls counts adapter invocations, not native transfers.",
            labelnames=base + ["kind", "event"],
        )
        self._keys = factory.counter(
            name="tokenspeed:l3_io_keys",
            documentation="Reported per-key outcomes, including query misses and rejected/cancelled jobs.",
            labelnames=base + ["kind", "result"],
        )
        self._bytes = factory.counter(
            name="tokenspeed:l3_io_adapter_bytes",
            documentation="Successful adapter bytes, including existing PUT objects; not network bytes.",
            labelnames=base + ["kind"],
        )
        self._seconds = factory.counter(
            name="tokenspeed:l3_io_seconds",
            documentation="Cumulative local queue and adapter call durations.",
            labelnames=base + ["kind", "stage"],
        )
        self._operations = factory.gauge(
            name="tokenspeed:l3_io_operations",
            documentation="Local jobs, including completed facts not yet polled.",
            labelnames=base + ["kind", "state"],
            multiprocess_mode="livemostrecent",
        )
        self._held_bytes = factory.gauge(
            name="tokenspeed:l3_io_buffer_bytes",
            documentation="Job byte ranges; excludes C++ Ready and quarantined ownership after poll.",
            labelnames=base + ["kind", "state"],
            multiprocess_mode="livemostrecent",
        )
        self._health = factory.gauge(
            name="tokenspeed:l3_backend_state",
            documentation="Backend health: healthy=0, open=1, half_open=2. Quarantine is separate.",
            labelnames=base,
            multiprocess_mode="livemostrecent",
        )
        self._epoch = factory.gauge(
            name="tokenspeed:l3_backend_epoch",
            documentation="Local backend health epoch.",
            labelnames=base,
            multiprocess_mode="livemostrecent",
        )
        self._unsafe = factory.gauge(
            name="tokenspeed:l3_io_quarantined",
            documentation="Local native retirement unproven; new I/O disabled until restart.",
            labelnames=base,
            multiprocess_mode="livemostrecent",
        )
        self._resources = factory.gauge(
            name="tokenspeed:l3_scheduler_resources",
            documentation="Mirrored C++ resource counts; do not sum across TP ranks.",
            labelnames=base + ["resource"],
            multiprocess_mode="livemostrecent",
        )
        self._expirations = factory.counter(
            name="tokenspeed:l3_preparation_expirations",
            documentation="Mirrored scheduler deadline expirations; do not sum across TP ranks.",
            labelnames=base + ["phase"],
        )

        self._pressure_bypasses = factory.counter(
            name="tokenspeed:l3_pressure_bypasses",
            documentation="Mirrored preparation attempts bypassed by pressure; do not sum TP ranks.",
            labelnames=base,
        )
        self._pressure_drops = factory.counter(
            name="tokenspeed:l3_pressure_dropped_backup_keys",
            documentation="Mirrored backup candidates dropped by pressure; do not sum TP ranks.",
            labelnames=base,
        )
        self._backup_drops = factory.counter(
            name="tokenspeed:l3_dropped_backup_keys",
            documentation="Mirrored backup candidates discarded by reason; do not sum TP ranks.",
            labelnames=base + ["reason"],
        )

    def _increment(self, counter, name, value, **labels):
        key = (name, tuple(sorted(labels.items())))
        previous = self._exported.get(key, 0)
        delta = value - previous if value >= previous else value
        if delta:
            counter.labels(**self.labels, **labels).inc(delta)
        self._exported[key] = value

    def record(self, io: ExecutorSnapshot, scheduler) -> None:
        """Export snapshots idempotently; no request/key identifiers become labels."""
        for row in io.operations:
            for event, value in row.events:
                self._increment(
                    self._events, "events", value, kind=row.kind, event=event
                )
            self._increment(
                self._keys, "keys", row.success_keys, kind=row.kind, result="success"
            )
            self._increment(
                self._keys, "keys", row.failed_keys, kind=row.kind, result="failure"
            )
            self._increment(self._bytes, "bytes", row.success_bytes, kind=row.kind)
            self._increment(
                self._seconds,
                "seconds",
                row.queue_seconds,
                kind=row.kind,
                stage="queue",
            )
            self._increment(
                self._seconds,
                "seconds",
                row.backend_seconds,
                kind=row.kind,
                stage="backend",
            )
            for state, value in row.operations:
                self._operations.labels(**self.labels, kind=row.kind, state=state).set(
                    value
                )
            for state, value in row.bytes:
                self._held_bytes.labels(**self.labels, kind=row.kind, state=state).set(
                    value
                )
        self._health.labels(**self.labels).set(
            {"healthy": 0, "open": 1, "half_open": 2}[io.health.state]
        )
        self._epoch.labels(**self.labels).set(io.health.epoch)
        self._unsafe.labels(**self.labels).set(io.unsafe_io)
        resources = {
            "operations": scheduler.operations,
            "held_blocks": scheduler.held_blocks,
            "quarantined": scheduler.quarantined,
            "negative_keys": scheduler.negative_keys,
            "backup_candidates": scheduler.backup_candidates,
            "waiting_requests": scheduler.waiting_requests,
            "ready_requests": scheduler.ready_requests,
            "pressure_gate_closed": int(scheduler.pressure_gate_closed),
        }
        for resource, value in resources.items():
            self._resources.labels(**self.labels, resource=resource).set(value)
        self._increment(
            self._expirations, "expirations", scheduler.wait_timeouts, phase="waiting"
        )
        self._increment(
            self._expirations, "expirations", scheduler.ready_expirations, phase="ready"
        )
        self._increment(
            self._pressure_bypasses, "pressure_bypasses", scheduler.pressure_bypasses
        )
        self._increment(
            self._pressure_drops,
            "pressure_drops",
            scheduler.pressure_dropped_backup_keys,
        )
        for reason, count in (
            ("pressure", scheduler.pressure_dropped_backup_keys),
            ("overflow", scheduler.overflow_dropped_backup_keys),
            ("host_unavailable", scheduler.unavailable_dropped_backup_keys),
            ("write_disabled", scheduler.disabled_dropped_backup_keys),
            ("cleanup", scheduler.cleanup_dropped_backup_keys),
        ):
            self._increment(self._backup_drops, "backup_drops", count, reason=reason)
