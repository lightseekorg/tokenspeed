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
    spawn_gateway,
    terminate_then_kill,
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


@pytest.mark.asyncio
async def test_spawn_gateway_disables_retries_and_circuit_breaker(monkeypatch):
    """Single-worker mode: retries and circuit-breaker are disabled by default."""

    captured = {}

    async def fake_exec(*cmd, **kwargs):
        captured["cmd"] = cmd

        class _P:
            async def wait(self):
                return 0

        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    await spawn_gateway([], engine_host="127.0.0.1", engine_port=12345)
    cmd = captured["cmd"]
    assert "--disable-retries" in cmd
    assert "--disable-circuit-breaker" in cmd


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
