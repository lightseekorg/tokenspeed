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

"""Tests for subprocess supervision + signal forwarding helpers."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from tokenspeed.cli._proc import (
    pick_free_port,
    spawn_gateway,
    terminate_then_kill,
)


@pytest.mark.asyncio
async def test_spawn_gateway_disables_retries_and_circuit_breaker(monkeypatch):
    """Single-worker mode: retries (against the same worker) and a tripped
    circuit breaker (which has nowhere to fail over to) are net-negative.
    Spawn the gateway with both disabled by default."""

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


def test_pick_free_port_returns_in_user_range():
    port = pick_free_port()
    assert 1024 < port < 65536


def test_pick_free_port_returns_distinct_ports_on_repeated_calls():
    p1 = pick_free_port()
    p2 = pick_free_port()
    # Not strictly guaranteed but overwhelmingly likely.
    assert p1 != p2


@pytest.mark.asyncio
async def test_terminate_then_kill_uses_sigterm_first_then_sigkill():
    """If the child doesn't exit within drain_timeout, we escalate to SIGKILL."""
    proc = MagicMock()
    proc.returncode = None  # Still running.
    proc.terminate = MagicMock()
    proc.kill = MagicMock()

    async def fake_wait():
        # Simulate "won't exit on SIGTERM": sleep slightly longer than drain.
        await asyncio.sleep(0.1)

    proc.wait = MagicMock(side_effect=fake_wait)
    await terminate_then_kill(proc, drain_timeout=0.05)
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_terminate_then_kill_skips_kill_when_drain_succeeds():
    proc = MagicMock()
    proc.returncode = None

    async def fake_wait():
        # Simulate child exiting promptly on SIGTERM.
        proc.returncode = 0
        return

    proc.wait = MagicMock(side_effect=fake_wait)
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    await terminate_then_kill(proc, drain_timeout=0.5)
    proc.terminate.assert_called_once()
    proc.kill.assert_not_called()


@pytest.mark.asyncio
async def test_terminate_then_kill_noop_on_already_dead():
    proc = MagicMock()
    proc.returncode = 0  # Already exited.
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    await terminate_then_kill(proc, drain_timeout=0.5)
    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()
