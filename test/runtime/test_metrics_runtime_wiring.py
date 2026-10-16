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
import contextlib
import json
import multiprocessing as mp
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.metrics.events import MetricEvent
from tokenspeed.runtime.metrics.service import MetricsService
from tokenspeed.runtime.utils.process import (
    receive_ready_messages,
    start_process_with_ready_pipe,
    terminate_child_processes,
)
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs


def _args(directory):
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return parser.parse_args(
        [
            "--model",
            "test-model",
            "--enable-metrics",
            "--metrics-reporters",
            "jsonl",
            "--metrics-jsonl-dir",
            str(directory),
            "--metrics-jsonl-flush-interval",
            "0.01",
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


def _rows(directory):
    return [
        json.loads(line)
        for path in directory.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]


def test_close_after_prepare_allows_retry_with_new_configuration(tmp_path):
    service = MetricsService()
    abandoned = tmp_path / "abandoned"
    retry = tmp_path / "retry"
    service.prepare(_args(abandoned))
    # Model preparation can fail before any writer or scheduler is started.
    service.close(timeout=1)
    service.start(_args(retry), _ports())
    try:
        service.get_event_sink().emit(MetricEvent(1, "retry", "inc", 1, {}))
    finally:
        service.close(timeout=2)
    assert not abandoned.exists()
    assert [row["metric"] for row in _rows(retry)] == ["retry"]


def _metric_worker(configs, ready, rank):
    service = MetricsService()
    service.connect(configs)
    sink = service.get_event_sink()

    def stop(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    try:
        ready.set()
        while True:
            time.sleep(0.01)
    finally:
        sink.emit(
            MetricEvent(
                time.time_ns(), "worker_shutdown", "inc", 1, {"rank": str(rank)}
            )
        )
        service.close(timeout=2)


def _metric_controller(ports, writer):
    context = mp.get_context("spawn")
    service = MetricsService()
    service.connect(ports.metrics_producer_configs)
    workers = []
    try:
        for rank in range(2):
            ready = context.Event()
            worker = context.Process(
                target=_metric_worker,
                args=(ports.metrics_producer_configs, ready, rank),
            )
            worker.start()
            workers.append(worker)
            if not ready.wait(timeout=30):
                raise RuntimeError("metric producer failed to start")
        writer.send({"status": "ready"})
        for worker in workers:
            worker.join()
            if worker.exitcode != 0:
                raise RuntimeError(
                    f"metric producer exited with code {worker.exitcode}"
                )
        service.get_event_sink().emit(
            MetricEvent(time.time_ns(), "controller_shutdown", "inc", 1, {})
        )
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
            worker.join(timeout=5)
        service.close(timeout=2)


def test_worker_shutdown_events_reach_node_writer_before_it_closes(tmp_path):
    service = MetricsService()
    ports = _ports()
    service.start(_args(tmp_path), ports)
    controller, reader = start_process_with_ready_pipe(
        _metric_controller, (ports,), contextlib.nullcontext()
    )
    try:
        assert reader.poll(60), "metric controller failed to become ready"
        assert receive_ready_messages([reader], [controller]) == [{"status": "ready"}]
        terminate_child_processes(controller.pid, timeout=5)
        controller.join(timeout=10)
        assert controller.exitcode == 0
    finally:
        if controller.is_alive():
            terminate_child_processes(controller.pid, timeout=1)
            controller.kill()
        controller.join(timeout=5)
        reader.close()
        service.close(timeout=2)

    rows = _rows(tmp_path)
    assert len(rows) == 3
    assert {
        row["labels"]["rank"] for row in rows if row["metric"] == "worker_shutdown"
    } == {"0", "1"}
    assert sum(row["metric"] == "controller_shutdown" for row in rows) == 1
    assert len({row["stream_id"] for row in rows}) == 1
