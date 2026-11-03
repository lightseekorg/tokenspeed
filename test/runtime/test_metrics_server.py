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

"""The engine's Prometheus /metrics endpoint (launch_metrics_server)."""

from __future__ import annotations

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import multiprocessing as mp
import socket
import time
import urllib.request
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.utils.common import launch_metrics_server
from tokenspeed.runtime.utils.network import get_free_port


def _write_sample_counter(multiproc_dir: str) -> None:
    """Write one sample into the multiproc dir from a fresh process.

    prometheus_client picks multiprocess (mmap-file) mode only when
    PROMETHEUS_MULTIPROC_DIR is set before its first import, which a spawned
    child guarantees regardless of what this test process already imported.
    """
    import os

    os.environ["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir
    from prometheus_client import Counter

    Counter("tokenspeed_metrics_server_probe", "probe counter").inc(7)


def _wait_started(server, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not server.started:
        assert time.monotonic() < deadline, "metrics server did not start"
        time.sleep(0.05)


@pytest.fixture
def sample_multiproc_dir(tmp_path, monkeypatch):
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_write_sample_counter, args=(str(tmp_path),))
    proc.start()
    proc.join()
    assert proc.exitcode == 0
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    return tmp_path


def test_metrics_endpoint_serves_aggregated_metrics(sample_multiproc_dir):
    port = get_free_port()
    # Unset reporters resolve to Prometheus (see test_metrics_config.py).
    args = SimpleNamespace(
        enable_metrics=True,
        metrics_reporters=None,
        metrics_host=None,
        metrics_port=port,
        host="127.0.0.1",
    )
    server = launch_metrics_server(args)
    try:
        _wait_started(server)
        # The trailing-slash variant exercises the 307-redirect workaround.
        for path in ("/metrics", "/metrics/"):
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{path}", timeout=5
            ) as resp:
                assert resp.status == 200
                body = resp.read().decode()
            assert "tokenspeed_metrics_server_probe_total 7.0" in body
    finally:
        server.should_exit = True


def test_metrics_server_off_without_enable_metrics():
    args = SimpleNamespace(enable_metrics=False)
    assert launch_metrics_server(args) is None


def test_metrics_server_bind_failure_degrades_to_none(tmp_path, monkeypatch):
    # The bind fails before anything is served, so no sample is needed.
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    port = blocker.getsockname()[1]
    args = SimpleNamespace(
        enable_metrics=True,
        metrics_reporters=["prometheus"],
        metrics_host=None,
        metrics_port=port,
        host="127.0.0.1",
    )
    try:
        # A SystemExit escaping launch_metrics_server would fail here; the
        # contract is an error log plus None — metrics never take down serving.
        assert launch_metrics_server(args) is None
    finally:
        blocker.close()


def test_plugin_only_does_not_start_prometheus_endpoint():
    args = SimpleNamespace(enable_metrics=True, metrics_reporters=["test_reporter"])
    assert launch_metrics_server(args) is None
