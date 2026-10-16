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

"""Metrics configuration and shutdown across real producer processes."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.metrics.events import MetricEvent
from tokenspeed.runtime.metrics.registry import register_metric_reporter
from tokenspeed.runtime.metrics.service import MetricsService
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs

_REPORTER_NAME = "runtime_recording"


def _args(config_path):
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return parser.parse_args(
        [
            "--model",
            "test-model",
            "--enable-metrics",
            "--metrics-reporters",
            _REPORTER_NAME,
            "--metrics-reporter-config",
            str(config_path),
        ]
    )


def _ports():
    return PortArgs(
        tokenizer_ipc_name="unused",
        scheduler_input_ipc_name="unused",
        nccl_port=0,
        dist_init_addr="localhost:0",
        rpc_ipc_name="unused",
        metrics_ipc_name="unused",
        tokenizer_worker_ipc_name=None,
    )


class RecordingSink:
    def __init__(self, config):
        self.config = config
        self.events = []
        self.closed = False

    def emit(self, event):
        assert not self.closed
        self.events.append(event)

    def close(self, timeout):
        self.closed = True


def _register(created):
    def create_sink(config):
        sink = RecordingSink(config)
        created.append(sink)
        return sink

    register_metric_reporter(
        _REPORTER_NAME,
        validate=lambda config: config,
        create_sink=create_sink,
        start_node=None,
        override=True,
    )


def test_close_after_prepare_allows_retry_with_new_configuration(tmp_path):
    created = []
    _register(created)
    service = MetricsService()
    abandoned = tmp_path / "abandoned.json"
    abandoned.write_text(json.dumps({_REPORTER_NAME: {"service": "abandoned"}}))
    retry = tmp_path / "retry.json"
    retry.write_text(json.dumps({_REPORTER_NAME: {"service": "retry"}}))
    service.prepare(_args(abandoned))
    # Model preparation can fail before any reporter or scheduler is started.
    service.close(timeout=1)
    assert created == []
    service.start(_args(retry), _ports())
    event = MetricEvent(1, "retry", "inc", 1, {})
    try:
        service.get_event_sink().emit(event)
    finally:
        service.close(timeout=2)
    assert len(created) == 1
    assert created[0].config == {"service": "retry"}
    assert created[0].events == [event]
    assert created[0].closed


def _metric_worker(ports, rank, connection):
    created = []
    _register(created)
    service = MetricsService()
    try:
        try:
            service.connect(ports.metrics_producer_configs)
            service.get_event_sink().emit(
                MetricEvent(
                    time.time_ns(),
                    "worker_event",
                    "observe",
                    0.5,
                    {"rank": str(rank)},
                    2,
                )
            )
        finally:
            service.close(timeout=2)
        sink = created[0]
        connection.send(
            {"config": sink.config, "events": sink.events, "closed": sink.closed}
        )
    finally:
        connection.close()


def test_spawned_producers_receive_config_and_close_their_sinks(tmp_path):
    created = []
    _register(created)
    service = MetricsService()
    ports = _ports()
    config = {"service": "test-service", "labels": {"deployment": "test"}}
    config_path = tmp_path / "reporter.json"
    config_path.write_text(json.dumps({_REPORTER_NAME: config}))
    service.start(_args(config_path), ports)
    context = mp.get_context("spawn")
    workers = []
    results = []
    frontend_event = MetricEvent(time.time_ns(), "frontend_event", "inc", 1, {})
    try:
        for rank in range(2):
            receiver, sender = context.Pipe(duplex=False)
            worker = context.Process(target=_metric_worker, args=(ports, rank, sender))
            worker.start()
            sender.close()
            workers.append((worker, receiver))
        for worker, receiver in workers:
            assert receiver.poll(timeout=30)
            results.append(receiver.recv())
            worker.join(timeout=30)
            assert worker.exitcode == 0
        service.get_event_sink().emit(frontend_event)
    finally:
        for worker, receiver in workers:
            if worker.is_alive():
                worker.kill()
            worker.join(timeout=5)
            receiver.close()
        service.close(timeout=2)

    for rank, result in enumerate(results):
        assert result["config"] == config
        assert result["closed"]
        assert len(result["events"]) == 1
        event = result["events"][0]
        assert event.metric == "worker_event"
        assert event.operation == "observe"
        assert event.value == 0.5
        assert event.labels == {"rank": str(rank)}
        assert event.weight == 2
    assert len(created) == 1
    assert created[0].config == config
    assert created[0].events == [frontend_event]
    assert created[0].closed
