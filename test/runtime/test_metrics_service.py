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

"""Process-local ownership and cleanup of registered metric reporters."""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.metrics.events import MetricEvent
from tokenspeed.runtime.metrics.service import MetricsService
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs


def _args(reporters):
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.set_defaults(
        enable_metrics=True,
        metrics_reporter_config=None,
        metrics_reporters=reporters,
    )
    return parser.parse_args(["--model", "test-model"])


def _ports():
    return PortArgs(
        tokenizer_ipc_name="",
        scheduler_input_ipc_name="",
        nccl_port=0,
        dist_init_addr="",
        rpc_ipc_name="",
        metrics_ipc_name="",
        tokenizer_worker_ipc_name=None,
    )


def test_prometheus_only_does_not_create_plugin_sinks():
    service = MetricsService()
    ports = _ports()
    service.start(_args(["prometheus"]), ports)
    assert ports.metrics_producer_configs == {}
    assert service.get_event_sink() is None
    service.close(timeout=1)


def test_services_create_separate_sinks_from_node_configuration():
    created = []

    class Session:
        def __init__(self):
            self.producer_config = {"service": "test-service"}
            self.closed = False

        def close(self, timeout):
            assert all(sink.closed for _, sink in created)
            self.closed = True

    session = Session()

    def create_sink(config):
        sink = RecordingSink(fail_emit=False, fail_close=False)
        created.append((config, sink))
        return sink

    _register("service_producers", create_sink, start_node=lambda _: session)
    owner = MetricsService()
    producer = MetricsService()
    ports = _ports()
    args = _args(["service_producers"])
    owner.start(args, ports)
    producer.connect(ports.metrics_producer_configs)
    for service, metric in ((owner, "frontend"), (producer, "scheduler")):
        sink = service.get_event_sink()
        assert sink is service.get_event_sink()
        sink.emit(MetricEvent(time.time_ns(), metric, "observe", 0.5, {}))
    producer.close(timeout=2)
    assert not session.closed
    owner.close(timeout=2)
    assert [config for config, _ in created] == [
        {"service": "test-service"},
        {"service": "test-service"},
    ]
    assert [sink.events[0].metric for _, sink in created] == ["frontend", "scheduler"]
    assert session.closed
    assert owner.get_event_sink() is None
    owner.close(timeout=1)


class RecordingSink:
    def __init__(self, *, fail_emit, fail_close):
        self.events = []
        self.closed = False
        self.close_timeouts = []
        self.fail_emit = fail_emit
        self.fail_close = fail_close
        self.emit_count = 0

    def emit(self, event):
        assert not self.closed
        self.emit_count += 1
        if self.fail_emit:
            raise RuntimeError("output unavailable")
        self.events.append(event)

    def close(self, timeout):
        self.closed = True
        self.close_timeouts.append(timeout)
        if self.fail_close:
            raise RuntimeError("output close failed")


def _register(name, create_sink, *, start_node):
    from tokenspeed.runtime.metrics.registry import register_metric_reporter

    register_metric_reporter(
        name,
        validate=lambda config: config,
        create_sink=create_sink,
        start_node=start_node,
        override=True,
    )


def test_empty_connection_creates_no_sink():
    service = MetricsService()
    service.connect({})
    assert service.get_event_sink() is None
    service.close(timeout=1)


def test_process_reuses_sink_and_keeps_configuration_snapshot():
    created = []

    def create_sink(config):
        sink = RecordingSink(fail_emit=False, fail_close=False)
        created.append((config, sink))
        return sink

    _register("service_shared", create_sink, start_node=None)
    service = MetricsService()
    configs = {"service_shared": {"labels": {"service": "original"}}}
    service.connect(configs)
    configs["service_shared"]["labels"]["service"] = "mutated"
    assert created == []
    with ThreadPoolExecutor(max_workers=8) as executor:
        sinks = list(executor.map(lambda _: service.get_event_sink(), range(32)))
    assert all(sink is sinks[0] for sink in sinks)
    assert len(created) == 1
    assert created[0][0] == {"labels": {"service": "original"}}
    service.connect({"service_shared": {"labels": {"service": "original"}}})
    with pytest.raises(RuntimeError, match="active"):
        service.connect({"service_shared": {"labels": {"service": "other"}}})
    service.close(timeout=1)
    assert service.get_event_sink() is None
    with pytest.raises(RuntimeError, match="closed"):
        service.connect({"service_shared": {}})


def test_emit_failure_disables_only_failed_sink_and_retains_cleanup(caplog):
    failing = RecordingSink(fail_emit=True, fail_close=True)
    healthy = RecordingSink(fail_emit=False, fail_close=False)
    _register("service_failing", lambda _: failing, start_node=None)
    _register("service_healthy", lambda _: healthy, start_node=None)
    service = MetricsService()
    service.connect({"service_failing": {}, "service_healthy": {}})
    sink = service.get_event_sink()
    event = MetricEvent(1, "test", "observe", 0.5, {}, 2)
    sink.emit(event)
    sink.emit(event)
    assert failing.emit_count == 1
    assert not failing.closed
    assert healthy.events == [event, event]
    service.close(timeout=1)
    sink.emit(event)
    assert healthy.events == [event, event]
    assert failing.closed and healthy.closed
    assert "service_failing" in caplog.text
    service.close(timeout=1)
    assert len(failing.close_timeouts) == len(healthy.close_timeouts) == 1


def test_sink_creation_failure_closes_previous_sinks_without_retry():
    first = RecordingSink(fail_emit=False, fail_close=False)
    attempts = []
    _register("service_first", lambda _: first, start_node=None)

    def create_failing_sink(config):
        attempts.append(config)
        raise RuntimeError("construction failed")

    _register("service_bad_create", create_failing_sink, start_node=None)
    service = MetricsService()
    service.connect({"service_first": {}, "service_bad_create": {}})
    with pytest.raises(RuntimeError, match="construction failed"):
        service.get_event_sink()
    assert first.closed
    assert service.get_event_sink() is None
    assert len(attempts) == 1


def test_partial_node_failure_closes_sessions_in_reverse_order():
    closed = []

    class Session:
        def __init__(self, name):
            self.name = name
            self.producer_config = {"name": name}

        def close(self, timeout):
            closed.append(self.name)

    def fail_start(config):
        raise RuntimeError("node startup failed")

    for name in ("service_node_a", "service_node_b"):
        _register(name, lambda _: None, start_node=lambda _, name=name: Session(name))
    _register("service_node_bad", lambda _: None, start_node=fail_start)
    service = MetricsService()
    args = _args(["service_node_a", "service_node_b", "service_node_bad"])
    with pytest.raises(RuntimeError, match="node startup failed"):
        service.start(args, _ports())
    assert closed == ["service_node_b", "service_node_a"]
    service.close(timeout=1)
    assert closed == ["service_node_b", "service_node_a"]


def test_invalid_node_config_closes_session_before_error():
    closed = []

    class Session:
        producer_config = {"connection": object()}

        def close(self, timeout):
            closed.append(True)

    _register("service_invalid_node", lambda _: None, start_node=lambda _: Session())
    service = MetricsService()
    with pytest.raises((TypeError, ValueError)):
        service.start(
            _args(["service_invalid_node"]),
            _ports(),
        )
    assert closed == [True]


def test_local_sink_close_does_not_close_node_resources():
    order = []
    sink = RecordingSink(fail_emit=False, fail_close=False)

    class Session:
        producer_config = {"endpoint": "test"}

        def close(self, timeout):
            assert sink.closed
            order.append("node")

    _register("service_node", lambda _: sink, start_node=lambda _: Session())
    service = MetricsService()
    ports = _ports()
    service.start(_args(["service_node"]), ports)
    service.get_event_sink()
    service.close_sinks(timeout=1)
    assert sink.closed
    assert order == []
    service.close_nodes(timeout=1)
    assert order == ["node"]


def test_close_uses_one_budget_and_continues_after_failure():
    timeouts = []

    class SlowSink:
        def emit(self, event):
            pass

        def close(self, timeout):
            timeouts.append(timeout)
            time.sleep(0.01)
            raise RuntimeError("close failed")

    _register("service_budget_a", lambda _: SlowSink(), start_node=None)
    _register("service_budget_b", lambda _: SlowSink(), start_node=None)
    service = MetricsService()
    service.connect({"service_budget_a": {}, "service_budget_b": {}})
    service.get_event_sink()
    service.close(timeout=1)
    assert len(timeouts) == 2
    assert 0 <= timeouts[1] < timeouts[0] <= 1.0


def test_explicit_start_can_begin_new_lifecycle_after_close():
    created = []

    def create_sink(config):
        sink = RecordingSink(fail_emit=False, fail_close=False)
        created.append(sink)
        return sink

    _register("service_restart", create_sink, start_node=None)
    service = MetricsService()
    args = _args(["service_restart"])
    service.start(args, _ports())
    first = service.get_event_sink()
    service.close(timeout=1)
    assert service.get_event_sink() is None
    service.prepare(args)
    service.start(args, _ports())
    second = service.get_event_sink()
    assert second is not first
    assert len(created) == 2
    assert created[0].closed
    assert not created[1].closed
    service.close(timeout=1)


def test_prepare_validates_once_and_copies_session_configuration():
    from tokenspeed.runtime.metrics.registry import register_metric_reporter

    validations = []
    sessions = []
    created = []

    class Session:
        def __init__(self):
            self.producer_config = {"nested": {"value": "initial"}}

        def close(self, timeout):
            pass

    def validate(config):
        validations.append(config)
        return {"normalized": True}

    def start_node(config):
        assert config == {"normalized": True}
        session = Session()
        sessions.append(session)
        return session

    def create_sink(config):
        created.append(config)
        return RecordingSink(fail_emit=False, fail_close=False)

    register_metric_reporter(
        "service_prepare",
        validate=validate,
        create_sink=create_sink,
        start_node=start_node,
        override=True,
    )
    service = MetricsService()
    args = _args(["service_prepare"])
    service.prepare(args)
    service.prepare(args)
    assert validations == [{}]
    assert sessions == []
    ports = _ports()
    service.start(args, ports)
    assert validations == [{}]
    assert created == []
    sessions[0].producer_config["nested"]["value"] = "session changed"
    ports.metrics_producer_configs["service_prepare"]["nested"][
        "value"
    ] = "ports changed"
    service.get_event_sink()
    assert created == [{"nested": {"value": "initial"}}]
    service.close(timeout=1)


def test_close_waits_for_inflight_emit_and_rejects_later_emits():
    entered = threading.Event()
    release = threading.Event()
    closing = threading.Event()
    sink = RecordingSink(fail_emit=False, fail_close=False)

    class InflightSink:
        def emit(self, event):
            entered.set()
            assert release.wait(timeout=2)
            sink.emit(event)

        def close(self, timeout):
            sink.close(timeout)

    _register("service_concurrent", lambda _: InflightSink(), start_node=None)
    service = MetricsService()
    service.connect({"service_concurrent": {}})
    shared = service.get_event_sink()
    event = MetricEvent(1, "test", "inc", 1.0, {})

    def close():
        closing.set()
        service.close(timeout=1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        emitting = executor.submit(shared.emit, event)
        try:
            assert entered.wait(timeout=2)
            shutdown = executor.submit(close)
            assert closing.wait(timeout=2)
        finally:
            release.set()
        emitting.result(timeout=2)
        shutdown.result(timeout=2)
    shared.emit(event)
    assert sink.closed
    assert sink.events == [event]
