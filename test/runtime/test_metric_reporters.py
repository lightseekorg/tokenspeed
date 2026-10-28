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

"""Reporter selection preserves metric observations and labels."""

import builtins
import os
import sys

import pytest
from prometheus_client import CollectorRegistry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.metrics import reporter  # noqa: E402
from tokenspeed.runtime.metrics.collector import (  # noqa: E402
    EngineMetrics,
    KVTransferMetrics,
)
from tokenspeed.runtime.metrics.reporter import MetricFactory  # noqa: E402

LABELS = {"model_name": "m", "app_key": "k"}


class Clock:
    def __init__(self, *, now: float = 0.0, wall_ns: int = 2_000_000_000):
        self.now = now
        self.wall_ns = wall_ns

    def monotonic(self):
        return self.now

    def time_ns(self):
        return self.wall_ns


class ListSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def test_plugin_only_does_not_load_prometheus(monkeypatch):
    original_import = builtins.__import__

    def reject_prometheus(name, *args, **kwargs):
        if name == "prometheus_client" or name.startswith("prometheus_client."):
            raise AssertionError("Plugin-only metrics imported Prometheus")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_prometheus)
    sink = ListSink()
    metrics = EngineMetrics(
        LABELS,
        enabled=True,
        reporters=["test_reporter"],
        event_sink=sink,
    )
    metrics.observe_request_queue_time(0.25)
    assert [(e.metric, e.operation, e.value) for e in sink.events] == [
        ("tokenspeed:request_queue_time_seconds", "observe", 0.25)
    ]


@pytest.mark.parametrize(
    "reporters", [["test_reporter"], ["prometheus", "test_reporter"]]
)
def test_preemptions_preserve_positive_round_deltas(reporters):
    sink = ListSink()
    registry = CollectorRegistry()
    metrics = EngineMetrics(
        LABELS,
        enabled=True,
        reporters=reporters,
        event_sink=sink,
        registry=registry,
    )
    for count in (0, 2, -1, 1):
        metrics.record_preemptions(count)
    assert [
        (event.metric, event.operation, event.value, event.labels)
        for event in sink.events
    ] == [
        ("tokenspeed:num_preemptions_total", "inc", 2, LABELS),
        ("tokenspeed:num_preemptions_total", "inc", 1, LABELS),
    ]
    if "prometheus" in reporters:
        assert (
            registry.get_sample_value("tokenspeed:num_preemptions_total", LABELS) == 3
        )


def test_dual_scheduler_snapshot_and_compile_deltas():
    from tokenspeed_kernel.compile_monitor import CompileStats

    sink = ListSink()
    registry = CollectorRegistry()
    metrics = EngineMetrics(
        LABELS,
        enabled=True,
        reporters=["prometheus", "test_reporter"],
        event_sink=sink,
        registry=registry,
    )
    metrics.record_scheduler_iteration(
        running=3,
        waiting=2,
        num_active_pages=2,
        num_total_pages=8,
        num_iteration_tokens=12,
    )
    metrics.record_jit_compiles(CompileStats(0, 0.0, 2, 0.25))
    metrics.record_jit_compiles(CompileStats(0, 0.0, 2, 0.25))
    metrics.record_jit_compiles(CompileStats(0, 0.0, 5, 1.0))
    metrics.observe_request_queue_time(0.5)
    assert [(e.metric, e.operation, e.value) for e in sink.events] == [
        ("tokenspeed:num_requests_running", "set", 3),
        ("tokenspeed:num_requests_waiting", "set", 2),
        ("tokenspeed:kv_cache_usage_perc", "set", 0.25),
        ("tokenspeed:iteration_tokens_total", "observe", 12),
        ("tokenspeed:jit_serving_compiles_total", "inc", 2),
        ("tokenspeed:jit_serving_compile_seconds_total", "inc", 0.25),
        ("tokenspeed:jit_serving_compiles_total", "inc", 3),
        ("tokenspeed:jit_serving_compile_seconds_total", "inc", 0.75),
        ("tokenspeed:request_queue_time_seconds", "observe", 0.5),
    ]
    assert registry.get_sample_value("tokenspeed:num_requests_running", LABELS) == 3
    assert (
        registry.get_sample_value("tokenspeed:jit_serving_compiles_total", LABELS) == 5
    )
    assert (
        registry.get_sample_value(
            "tokenspeed:jit_serving_compile_seconds_total", LABELS
        )
        == 1.0
    )


@pytest.mark.parametrize(
    "reporters", [["test_reporter"], ["prometheus", "test_reporter"]]
)
def test_idle_scheduler_gauges_emit_once_per_second(monkeypatch, reporters):
    clock = Clock()
    monkeypatch.setattr(reporter.time, "monotonic", clock.monotonic)
    sink = ListSink()
    registry = CollectorRegistry()
    metrics = EngineMetrics(
        LABELS,
        enabled=True,
        reporters=reporters,
        event_sink=sink,
        registry=registry,
    )
    # Ten seconds at 4,000 idle scheduler rounds per second.
    for iteration in range(40_000):
        clock.now = iteration / 4_000
        metrics.record_scheduler_iteration(
            running=0,
            waiting=0,
            num_active_pages=0,
            num_total_pages=8,
            num_iteration_tokens=0,
        )
    assert [(e.metric, e.operation, e.value) for e in sink.events] == [
        ("tokenspeed:num_requests_running", "set", 0),
        ("tokenspeed:num_requests_waiting", "set", 0),
        ("tokenspeed:kv_cache_usage_perc", "set", 0),
    ] * 10
    if "prometheus" in reporters:
        assert registry.get_sample_value("tokenspeed:num_requests_running", LABELS) == 0
        assert registry.get_sample_value("tokenspeed:num_requests_waiting", LABELS) == 0
        assert registry.get_sample_value("tokenspeed:kv_cache_usage_perc", LABELS) == 0


def test_gauge_changes_are_immediate_and_snapshot_uses_monotonic_time(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(reporter.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(reporter.time, "time_ns", clock.time_ns)
    sink = ListSink()
    registry = CollectorRegistry()
    metrics = EngineMetrics(
        LABELS,
        enabled=True,
        reporters=["prometheus", "test_reporter"],
        event_sink=sink,
        registry=registry,
    )
    gauge = metrics.num_requests_running.labels(**LABELS)
    for now, value in [(0, 0), (0.25, 1), (0.5, 0), (1.0, 0)]:
        clock.now = now
        gauge.set(value)
        assert (
            registry.get_sample_value("tokenspeed:num_requests_running", LABELS)
            == value
        )
    assert [e.value for e in sink.events] == [0, 1, 0]

    clock.now = 1.5
    clock.wall_ns = 1_000_000_000  # A wall-clock adjustment must not delay refresh.
    gauge.set(0)
    assert [e.value for e in sink.events] == [0, 1, 0, 0]
    assert sink.events[-1].timestamp_ns == 1_000_000_000

    clock.now = 100.0
    gauge.set(0)
    assert [e.value for e in sink.events] == [0, 1, 0, 0, 0]


def test_gauge_snapshot_state_is_independent_per_label_set(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(reporter.time, "monotonic", clock.monotonic)
    sink = ListSink()
    gauge = MetricFactory(
        enabled=True, reporters=["test_reporter"], event_sink=sink
    ).gauge(
        name="tokenspeed:test_gauge",
        documentation="Gauge with independently updated labels.",
        labelnames=["worker"],
        multiprocess_mode="livesum",
    )
    for now, worker in [(0, "a"), (0.5, "b"), (1, "a"), (1, "b"), (1.5, "b")]:
        clock.now = now
        gauge.labels(worker=worker).set(0)
    assert [(e.labels, e.value) for e in sink.events] == [
        ({"worker": "a"}, 0),
        ({"worker": "b"}, 0),
        ({"worker": "a"}, 0),
        ({"worker": "b"}, 0),
    ]


def test_repeated_counter_increments_and_histogram_observations_are_preserved():
    sink = ListSink()
    metrics = EngineMetrics(
        LABELS, enabled=True, reporters=["test_reporter"], event_sink=sink
    )
    for _ in range(3):
        metrics.record_nan_abort()
        metrics.observe_request_queue_time(0.25)
    assert [(e.metric, e.operation, e.value) for e in sink.events] == [
        ("tokenspeed:num_nan_aborted_requests_total", "inc", 1),
        ("tokenspeed:request_queue_time_seconds", "observe", 0.25),
    ] * 3


def test_plugin_speculation_and_abort_counters():
    sink = ListSink()
    metrics = EngineMetrics(
        LABELS, enabled=True, reporters=["test_reporter"], event_sink=sink
    )
    metrics.record_spec_decode_step(
        num_decode_slots=2, accepted_draft_tokens=3, draft_width=4
    )
    metrics.record_spec_decode_step(
        num_decode_slots=0, accepted_draft_tokens=0, draft_width=4
    )
    metrics.record_nan_abort()
    metrics.record_capacity_abort()
    assert [(e.metric, e.operation, e.value) for e in sink.events] == [
        ("tokenspeed:spec_decode_num_drafts_total", "inc", 2),
        ("tokenspeed:spec_decode_num_draft_tokens_total", "inc", 8),
        ("tokenspeed:spec_decode_num_accepted_tokens_total", "inc", 3),
        ("tokenspeed:num_nan_aborted_requests_total", "inc", 1),
        ("tokenspeed:num_capacity_aborted_requests_total", "inc", 1),
    ]


def test_transfer_reporters_preserve_dynamic_stage_labels():
    sink = ListSink()
    registry = CollectorRegistry()
    metrics = KVTransferMetrics(
        LABELS,
        enabled=True,
        reporters=["prometheus", "test_reporter"],
        event_sink=sink,
        registry=registry,
    )
    metrics.record_kv_transfer_timeout("bootstrapping")
    metrics.record_kv_transfer_timeout("waiting_for_input")
    metrics.record_kv_transfer_failure()
    metrics.observe_kv_transfer_latency(0.03)
    assert [(e.metric, e.labels, e.value) for e in sink.events] == [
        (
            "tokenspeed:kv_transfer_timeouts_total",
            {**LABELS, "stage": "bootstrapping"},
            1,
        ),
        (
            "tokenspeed:kv_transfer_timeouts_total",
            {**LABELS, "stage": "waiting_for_input"},
            1,
        ),
        ("tokenspeed:kv_transfer_failures_total", LABELS, 1),
        ("tokenspeed:kv_transfer_latency_seconds", LABELS, 0.03),
    ]
    for stage in ("bootstrapping", "waiting_for_input"):
        assert (
            registry.get_sample_value(
                "tokenspeed:kv_transfer_timeouts_total", {**LABELS, "stage": stage}
            )
            == 1
        )


@pytest.mark.parametrize(
    "reporters", [["prometheus"], ["test_reporter"], ["prometheus", "test_reporter"]]
)
def test_disabled_collectors_never_register_or_emit(reporters):
    sink = ListSink()
    registry = CollectorRegistry()
    for cls, call in (
        (EngineMetrics, lambda m: m.record_nan_abort()),
        (KVTransferMetrics, lambda m: m.record_kv_transfer_failure()),
    ):
        metrics = cls(
            LABELS,
            enabled=False,
            reporters=reporters,
            event_sink=sink,
            registry=registry,
        )
        call(metrics)
    assert list(registry.collect()) == []
    assert sink.events == []


def test_enabled_plugin_requires_a_sink():
    with pytest.raises(ValueError, match="sink"):
        EngineMetrics(LABELS, enabled=True, reporters=["test_reporter"])
    EngineMetrics(LABELS, enabled=False, reporters=["test_reporter"])


def test_prometheus_only_ignores_optional_plugin_sink():
    sink = ListSink()
    registry = CollectorRegistry()
    metrics = EngineMetrics(
        LABELS,
        enabled=True,
        reporters=["prometheus"],
        event_sink=sink,
        registry=registry,
    )
    metrics.observe_request_queue_time(0.2)
    assert sink.events == []
    assert (
        registry.get_sample_value("tokenspeed:request_queue_time_seconds_count", LABELS)
        == 1
    )
