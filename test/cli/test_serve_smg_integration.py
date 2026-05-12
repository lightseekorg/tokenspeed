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

"""Integration: orchestrator + real ``smg launch`` + fake_engine.

Spawns the orchestrator as its own subprocess so SIGTERM doesn't
interfere with pytest. Skipped if ``smg`` is not installed.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time

import pytest

pytest.importorskip("smg")
pytest.importorskip("smg_grpc_proto")
import aiohttp  # noqa: E402,F401  -- imported for skip side-effect


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http_200(url: str, deadline: float) -> bool:
    import urllib.error
    import urllib.request

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    return False


def test_orchestrator_runs_against_fake_engine():
    user_port = _free_port()

    env = os.environ.copy()
    # Point the orchestrator's spawn_engine at our fake module.
    env["TS_SERVE_ENGINE_MODULE"] = "test.cli._fixtures.fake_engine"
    # Make sure the test/ root is importable.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tokenspeed.cli",
            "serve",
            "--model",
            "/fake",
            "--host",
            "127.0.0.1",
            "--port",
            str(user_port),
            "--engine-startup-timeout",
            "30",
            "--gateway-startup-timeout",
            "30",
            "--drain-timeout",
            "5",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        ok = _wait_http_200(
            f"http://127.0.0.1:{user_port}/health", time.monotonic() + 60
        )
        if not ok:
            stdout, stderr = proc.communicate(timeout=5)
            pytest.fail(
                f"gateway never became healthy.\nstdout:\n{stdout.decode()}\n"
                f"stderr:\n{stderr.decode()}"
            )

        proc.send_signal(signal.SIGTERM)
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            stdout, stderr = proc.communicate(timeout=5)
            pytest.fail(
                f"orchestrator did not exit 30s after SIGTERM.\n"
                f"stdout:\n{stdout.decode()}\n"
                f"stderr:\n{stderr.decode()}"
            )
        assert rc == 0, f"non-zero rc={rc}"
    finally:
        if proc.returncode is None:
            proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
