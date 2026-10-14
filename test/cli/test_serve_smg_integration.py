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

"""Integration: orchestrator + real ``smg launch`` + fake_engine."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time

import pytest

pytest.importorskip("smg")
pytest.importorskip("smg_grpc_proto")
import aiohttp  # noqa: E402,F401


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http_200(url: str, deadline: float, proc) -> bool:
    import urllib.error
    import urllib.request

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    return False


@pytest.mark.parametrize("connection_mode", ["grpc", "zmq"])
def test_orchestrator_runs_against_fake_engine(connection_mode):
    if connection_mode == "zmq":
        pytest.importorskip("zmq")
        pytest.importorskip("msgspec")
        from smg.router_args import RouterArgs

        if "worker_specs" not in vars(RouterArgs):
            pytest.skip("ZMQ integration requires the paired SMG source build")
    user_port = _free_port()

    env = os.environ.copy()
    env["TS_SERVE_ENGINE_MODULE"] = "test.cli._fixtures.fake_engine"
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tokenspeed.cli",
            "serve",
            "--connection-mode",
            connection_mode,
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
            # The fake engine serves a placeholder model id (/fake) with no real
            # tokenizer and no GetTokenizer RPC. Newer smg gates gateway
            # readiness on tokenizer registration, so skip autoload here — this
            # test exercises orchestration/teardown, not tokenizer loading.
            "--disable-tokenizer-autoload",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        ok = _wait_http_200(
            f"http://127.0.0.1:{user_port}/readiness", time.monotonic() + 60, proc
        )
        if not ok:
            proc.terminate()
            stdout, stderr = proc.communicate(timeout=5)
            pytest.fail(
                f"gateway never became healthy.\nstdout:\n{stdout.decode()}\n"
                f"stderr:\n{stderr.decode()}"
            )

        # The selected transport reaches readiness, and admin remains connected
        # to the same engine through its native control service.
        import urllib.request

        request = urllib.request.Request(
            f"http://127.0.0.1:{user_port}/flush_cache", method="POST"
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            body = json.load(response)
        assert body["status"] == "success", body
        assert body["workers_flushed"] == 1, body
        if connection_mode == "zmq":
            assert body["total_zmq_workers"] == 1, body
            assert body["total_zmq_workers_skipped"] == 0, body

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
