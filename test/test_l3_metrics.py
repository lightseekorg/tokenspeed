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

from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry
from tokenspeed_scheduler import AttnKind, CacheGroupConfig, Scheduler, SchedulerConfig

from tokenspeed.runtime.cache.l3.metrics import L3Metrics
from tokenspeed.runtime.cache.l3.statistics import (
    ExecutorSnapshot,
    HealthSnapshot,
    OperationCounters,
)


def test_prometheus_snapshots_are_idempotent_and_keep_quarantine_visible():
    registry = CollectorRegistry()
    labels = {"model_name": "model", "dp_rank": "0", "tp_rank": "1"}
    metrics = L3Metrics(
        labels=labels,
        enabled=True,
        reporters=["prometheus"],
        event_sink=None,
        registry=registry,
    )
    counters = OperationCounters(
        accepted=2,
        completed=2,
        backend_calls=1,
        cancelled_queued=1,
        unretired=1,
        failed_keys=2,
    )
    row = counters.snapshot(
        "get",
        operations=dict(queued=0, running=0, completed=0),
        byte_counts=dict(queued=0, running=0, completed=0),
    )
    snapshot = ExecutorSnapshot((row,), HealthSnapshot("open", 3, 1), True)
    scheduler = SimpleNamespace(
        operations=1,
        held_blocks=2,
        quarantined=1,
        negative_keys=1,
        backup_candidates=0,
        waiting_requests=0,
        ready_requests=0,
        wait_timeouts=1,
        ready_expirations=0,
        pressure_gate_closed=True,
        pressure_bypasses=3,
        pressure_dropped_backup_keys=7,
        overflow_dropped_backup_keys=2,
        unavailable_dropped_backup_keys=3,
        disabled_dropped_backup_keys=4,
        cleanup_dropped_backup_keys=5,
    )
    metrics.record(snapshot, scheduler)
    metrics.record(snapshot, scheduler)
    for reason, count in (
        ("pressure", 7),
        ("overflow", 2),
        ("host_unavailable", 3),
        ("write_disabled", 4),
        ("cleanup", 5),
    ):
        assert (
            registry.get_sample_value(
                "tokenspeed:l3_dropped_backup_keys_total", dict(labels, reason=reason)
            )
            == count
        )
    assert (
        registry.get_sample_value(
            "tokenspeed:l3_io_events_total", dict(labels, kind="get", event="completed")
        )
        == 2
    )
    assert (
        registry.get_sample_value(
            "tokenspeed:l3_preparation_expirations_total", dict(labels, phase="waiting")
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            "tokenspeed:l3_scheduler_resources", dict(labels, resource="held_blocks")
        )
        == 2
    )
    assert registry.get_sample_value("tokenspeed:l3_io_quarantined", labels) == 1
    assert registry.get_sample_value("tokenspeed:l3_backend_state", labels) == 1
    assert (
        registry.get_sample_value("tokenspeed:l3_pressure_bypasses_total", labels) == 3
    )
    assert (
        registry.get_sample_value(
            "tokenspeed:l3_pressure_dropped_backup_keys_total", labels
        )
        == 7
    )
    assert (
        registry.get_sample_value(
            "tokenspeed:l3_scheduler_resources",
            dict(labels, resource="pressure_gate_closed"),
        )
        == 1
    )
    assert all(
        set(sample.labels)
        <= set(labels)
        | {"kind", "event", "result", "stage", "state", "resource", "phase", "reason"}
        for metric in registry.collect()
        for sample in metric.samples
    )


class _EventSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


@pytest.mark.parametrize("reporters", [["external"], ["prometheus", "external"]])
def test_external_reporters_receive_l3_deltas_and_gauge_changes(reporters):
    sink = _EventSink()
    registry = CollectorRegistry()
    labels = {"model_name": "model", "dp_rank": "0", "tp_rank": "1"}
    metrics = L3Metrics(
        labels=labels,
        enabled=True,
        reporters=reporters,
        event_sink=sink,
        registry=registry,
    )
    config = SchedulerConfig()
    config.prefix_granularity = 2
    config.num_device_pages = 64
    config.num_host_pages = 32
    config.num_snapshot_pages = 1
    config.max_scheduled_tokens = 64
    config.max_batch_size = 8
    config.cache_groups = [
        CacheGroupConfig(
            group_id="full",
            block_granularity=2,
            total_pages=64,
            attn_kind=AttnKind.Full,
        )
    ]
    scheduler = Scheduler(config)
    counters = OperationCounters(completed=2)
    row = counters.snapshot("get", operations={}, byte_counts={})
    first = ExecutorSnapshot((row,), HealthSnapshot("open", 1, 1), True)
    metrics.record(first, scheduler.storage_stats())
    metrics.record(first, scheduler.storage_stats())
    counters.completed = 3
    row = counters.snapshot("get", operations={}, byte_counts={})
    next_snapshot = ExecutorSnapshot((row,), HealthSnapshot("healthy", 2, 0), False)
    metrics.record(next_snapshot, scheduler.storage_stats())
    health = [
        event for event in sink.events if event.metric == "tokenspeed:l3_backend_state"
    ]
    assert health[0].value == 1 and health[-1].value == 0
    completed = [
        event
        for event in sink.events
        if event.metric == "tokenspeed:l3_io_events_total"
        and event.labels["event"] == "completed"
    ]
    assert [event.value for event in completed] == [2, 1]
    assert all(event.operation == "set" and event.labels == labels for event in health)
    assert bool(list(registry.collect())) == ("prometheus" in reporters)
    if "prometheus" in reporters:
        assert registry.get_sample_value("tokenspeed:l3_backend_state", labels) == 0
