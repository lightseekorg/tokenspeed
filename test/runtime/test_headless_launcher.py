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

"""The headless launcher: KV bootstrap hosting and scheduler supervision.

Without an in-process frontend there is no ``AsyncLLM`` to start a prefill
engine's KV bootstrap server, so the launcher does; and with nothing above it
to clean up, the launcher must not wait forever on a scheduler that ignores
its termination.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import sys
import time
import urllib.request

import pytest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.entrypoints.engine import (  # noqa: E402
    _start_kv_bootstrap_server,
    _supervise_schedulers,
    headless_encode_channel,
)
from tokenspeed.runtime.utils.network import get_free_port  # noqa: E402
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402


def _server_args(mode: str, port: int, node_rank: int = 0) -> ServerArgs:
    return ServerArgs(
        model="x",
        disaggregation_mode=mode,
        disaggregation_bootstrap_port=port,
        disaggregation_transfer_backend="mooncake",
        node_rank=node_rank,
    )


def test_prefill_launcher_serves_the_bootstrap_port():
    port = get_free_port()
    server = _start_kv_bootstrap_server(_server_args("prefill", port))
    assert server is not None
    try:
        # The route a decode rank queries first. Before any prefill rank has
        # registered, it answers without a parallel layout.
        url = f"http://127.0.0.1:{port}/route?engine_rank=-1&target_dp_group=-1"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=10) as response:
            assert response.status == 200
            info = json.loads(response.read())
        assert info["prefill_tp_size"] is None
        assert info["prefill_dp_size"] is None
    finally:
        server.close()


@pytest.mark.parametrize(
    ("mode", "node_rank"), [("decode", 0), ("null", 0), ("prefill", 1)]
)
def test_other_launchers_start_none(mode, node_rank):
    assert (
        _start_kv_bootstrap_server(_server_args(mode, get_free_port(), node_rank))
        is None
    )


@pytest.mark.parametrize(
    ("address", "channel"),
    [("10.0.0.2", "tcp://10.0.0.2:31000"), ("::1", "tcp://[::1]:31000")],
)
def test_a_headless_encode_server_takes_requests_at_the_frontend_address(
    address, channel
):
    # The frontend pushes encode requests where an engine rank would dial it.
    server_args = ServerArgs(
        model="x",
        disaggregation_mode="encode",
        data_parallel_address=address,
        data_parallel_rpc_port=31000,
    )
    assert headless_encode_channel(server_args) == channel
    # The encode ranks host their bootstrap server; the launcher starts none.
    assert _start_kv_bootstrap_server(server_args) is None


def _exit_with(code: int) -> None:
    sys.exit(code)


def _ignore_sigterm_and_wait() -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(600)


class _Schedulers:
    """Real child processes standing in for scheduler ranks."""

    def __init__(self, *targets):
        context = mp.get_context("fork")
        self.procs = [context.Process(target=t, args=a) for t, a in targets]
        for proc in self.procs:
            proc.start()
        self.terminate_requested_at: list[float] = []

    def terminate(self) -> None:
        if not self.terminate_requested_at:
            self.terminate_requested_at.append(time.monotonic())
        for proc in self.procs:
            if proc.is_alive():
                proc.terminate()

    def supervise(self, exit_grace_s: float) -> bool:
        return _supervise_schedulers(
            self.procs,
            self.terminate,
            self.terminate_requested_at,
            exit_grace_s=exit_grace_s,
        )

    def kill_all(self) -> None:
        for proc in self.procs:
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=5)


def test_supervision_succeeds_when_every_scheduler_exits_cleanly():
    schedulers = _Schedulers((_exit_with, (0,)), (_exit_with, (0,)))
    try:
        assert schedulers.supervise(exit_grace_s=30.0) is False
    finally:
        schedulers.kill_all()


def test_a_failed_rank_takes_the_others_down():
    schedulers = _Schedulers((_exit_with, (3,)), (time.sleep, (600,)))
    try:
        assert schedulers.supervise(exit_grace_s=30.0) is True
        assert schedulers.procs[1].exitcode == -signal.SIGTERM
    finally:
        schedulers.kill_all()


def test_a_scheduler_ignoring_termination_is_given_up_on_after_the_grace():
    schedulers = _Schedulers(
        (_ignore_sigterm_and_wait, ()),
    )
    try:
        # Let the child install its handler before it is asked to stop.
        time.sleep(0.5)
        schedulers.terminate()
        started = time.monotonic()
        assert schedulers.supervise(exit_grace_s=2.0) is True
        assert time.monotonic() - started < 10
        # Still running: killing it is the caller's job.
        assert schedulers.procs[0].is_alive()
    finally:
        schedulers.kill_all()
