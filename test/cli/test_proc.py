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

"""Tests for subprocess supervision helpers."""

from __future__ import annotations

import asyncio

import pytest

from tokenspeed.cli._proc import (
    engine_argv,
    spawn_gateway,
    terminate_then_kill,
    zmq_handshake_port,
)


class _RecordingProcess:
    """Stands in for an ``asyncio.subprocess.Process``: records the signals it
    receives and drains ``wait`` on a test-controlled delay."""

    def __init__(self, returncode, wait_delay=0.0):
        self.returncode = returncode
        self.wait_delay = wait_delay
        self.signals = []

    def terminate(self):
        self.signals.append("terminate")

    def kill(self):
        self.signals.append("kill")

    async def wait(self):
        await asyncio.sleep(self.wait_delay)
        return self.returncode


async def _gateway_command(monkeypatch, **kwargs) -> tuple:
    captured = {}

    async def fake_exec(*cmd, **exec_kwargs):
        captured["cmd"] = cmd

        class _P:
            async def wait(self):
                return 0

        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    await spawn_gateway(["--model", "/m"], **kwargs)
    return captured["cmd"]


def _value(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


@pytest.mark.asyncio
async def test_spawn_gateway_fronts_one_tokenspeed_zmq_worker(monkeypatch):
    """Single-worker mode: one ipc:// TokenSpeed worker, with retries and the
    circuit breaker off."""
    cmd = await _gateway_command(
        monkeypatch, worker_url="ipc:///tmp/smg-zmq/ts0.ipc", engine_count=1
    )
    assert cmd[1:4] == ("-m", "smg", "launch")
    assert _value(cmd, "--worker-urls") == "ipc:///tmp/smg-zmq/ts0.ipc"
    assert _value(cmd, "--backend") == "tokenspeed"
    assert "--disable-retries" in cmd
    assert "--disable-circuit-breaker" in cmd
    # One engine is an ungrouped worker; the operator's args come last.
    assert "--zmq-engine-count" not in cmd
    assert cmd[-2:] == ("--model", "/m")


@pytest.mark.asyncio
async def test_spawn_gateway_awaits_every_attention_dp_rank(monkeypatch):
    cmd = await _gateway_command(
        monkeypatch, worker_url="ipc:///tmp/smg-zmq/ts0.ipc", engine_count=4
    )
    assert _value(cmd, "--zmq-engine-count") == "4"


def test_handshake_port_matches_the_gateways_derivation():
    # SMG's own pinned vectors for its derive_handshake_port.
    assert zmq_handshake_port("ipc:///tmp/smg-zmq/ts0.ipc") == 25152
    assert zmq_handshake_port("ipc:///tmp/smg-zmq/engine-31000") == 22714


def test_engine_argv_dials_the_handshake_last(monkeypatch):
    """The handshake flags come last, so they win over the operator's; a
    follower node none of whose ranks registers gets none."""
    monkeypatch.delenv("TS_SERVE_ENGINE_MODULE", raising=False)
    argv = engine_argv(
        ["--model", "/m", "--data-parallel-rpc-port", "30500"],
        host="127.0.0.1",
        port=31000,
        handshake=("10.0.0.5", 23455),
    )
    assert argv[1:3] == ["-m", "tokenspeed.runtime.entrypoints.engine"]
    assert argv[-6:] == [
        "--data-parallel-address",
        "10.0.0.5",
        "--data-parallel-rpc-port",
        "23455",
        "--zmq-engine-index",
        "0",
    ]
    follower = engine_argv(["--model", "/m"], host="0.0.0.0", port=8000, handshake=None)
    assert follower[3:] == ["--host", "0.0.0.0", "--port", "8000", "--model", "/m"]


@pytest.mark.asyncio
async def test_terminate_then_kill_escalates_only_when_the_drain_times_out():
    """The signals are the observable effect: noop on a dead child, SIGTERM
    alone when it drains in time, SIGTERM then SIGKILL when it doesn't."""
    dead = _RecordingProcess(returncode=0)
    await terminate_then_kill(dead, drain_timeout=0.5)
    assert dead.signals == []

    draining = _RecordingProcess(returncode=None)
    await terminate_then_kill(draining, drain_timeout=0.5)
    assert draining.signals == ["terminate"]

    stuck = _RecordingProcess(returncode=None, wait_delay=0.1)
    await terminate_then_kill(stuck, drain_timeout=0.05)
    assert stuck.signals == ["terminate", "kill"]
