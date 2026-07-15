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
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from tokenspeed.cli._proc import (
    spawn_gateway,
    terminate_then_kill,
)


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
async def test_terminate_then_kill_uses_sigterm_first_then_sigkill():
    """If the child doesn't exit within drain_timeout, we escalate to SIGKILL."""
    proc = MagicMock()
    proc.returncode = None  # Still running.
    proc.terminate = MagicMock()
    proc.kill = MagicMock()

    async def fake_wait():
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
    proc.returncode = 0
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    await terminate_then_kill(proc, drain_timeout=0.5)
    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process reaping contract")
def test_kill_process_tree_reaps_owned_nested_children():
    """Killed children and grandchildren must not escape to a non-reaping PID 1."""
    script = """
import subprocess
import sys
import time

import psutil

from tokenspeed.runtime.utils.process import kill_process_tree

parent = subprocess.Popen([
    sys.executable,
    "-c",
    "import subprocess, sys, time; "
    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
    "time.sleep(60)",
])
deadline = time.monotonic() + 5
descendants = []
while time.monotonic() < deadline:
    descendants = psutil.Process().children(recursive=True)
    if len(descendants) == 2:
        break
    time.sleep(0.01)
if len(descendants) != 2:
    raise SystemExit(f"nested process did not start: {descendants}")
owned_pids = {process.pid for process in descendants}
kill_process_tree(None, include_parent=False, wait_timeout=5.0)
remaining = psutil.Process().children(recursive=True)
if remaining:
    raise SystemExit(f"unreaped children: {[(p.pid, p.status()) for p in remaining]}")
escaped = [pid for pid in owned_pids if psutil.pid_exists(pid)]
if escaped:
    raise SystemExit(f"descendants escaped reaping: {escaped}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
