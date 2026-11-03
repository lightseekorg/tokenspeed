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

"""Integration: orchestrator + real ``smg launch`` + the fake ZMQ engine."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("smg")
pytest.importorskip("zmq")
import aiohttp  # noqa: E402,F401


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http_200(url: str, deadline: float) -> bool:
    import urllib.error
    import urllib.request

    # These readiness endpoints are local, even when CI configures an HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        try:
            with opener.open(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    return False


def _stop_and_collect(proc, *, timeout: float) -> tuple[bytes, bytes]:
    """Stop before reading captured pipes, which stay open while the process runs."""
    if proc.poll() is None:
        proc.terminate()
    try:
        return proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.communicate(timeout=5)


def test_readiness_probe_ignores_environment_proxy(monkeypatch):
    import urllib.request

    class ReadyHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

    with (
        ThreadingHTTPServer(("127.0.0.1", 0), ReadyHandler) as server,
        socket.socket() as unavailable_proxy,
    ):
        unavailable_proxy.bind(("127.0.0.1", 0))
        proxy = f"http://127.0.0.1:{unavailable_proxy.getsockname()[1]}"
        for name in ("http_proxy", "HTTP_PROXY"):
            monkeypatch.setenv(name, proxy)
        for name in ("no_proxy", "NO_PROXY"):
            monkeypatch.setenv(name, "")
        # Use a real proxy-enabled opener even if an earlier test cached one.
        monkeypatch.setattr(urllib.request, "_opener", urllib.request.build_opener())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/readiness"
            assert _wait_http_200(url, time.monotonic() + 2)
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize("ignore_sigterm", [False, True])
def test_failed_startup_stops_the_process_and_preserves_logs(ignore_sigterm):
    script = (
        "import signal, sys, time\n"
        f"signal.signal(signal.SIGTERM, signal.SIG_IGN if {ignore_sigterm} else signal.SIG_DFL)\n"
        # The parent may terminate the child as soon as it reads the ready marker.
        "print('startup details', file=sys.stderr, flush=True)\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdout.readline() == b"ready\n"
        stdout, stderr = _stop_and_collect(proc, timeout=1)
        assert stdout == b""
        assert stderr == b"startup details\n"
        expected_signal = signal.SIGKILL if ignore_sigterm else signal.SIGTERM
        assert proc.returncode == -expected_signal
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=5)


# What ``ts serve`` reads before it starts the engine: the engine's parallel
# layout, resolved from the checkpoint config.
_LLAMA_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "dtype": "bfloat16",
    "vocab_size": 32000,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 512,
}


def test_orchestrator_runs_against_fake_engine(tmp_path):
    user_port = _free_port()
    (tmp_path / "config.json").write_text(json.dumps(_LLAMA_CONFIG))
    # Cold runtime imports are part of engine startup on shared CI filesystems.
    startup_timeout = 120

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
            "--model",
            str(tmp_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(user_port),
            "--engine-startup-timeout",
            str(startup_timeout),
            "--gateway-startup-timeout",
            str(startup_timeout),
            "--drain-timeout",
            "5",
            # The model has no tokenizer. Newer smg gates gateway readiness
            # on tokenizer registration, so skip autoload here — this test
            # exercises orchestration/teardown, not tokenizer loading.
            "--disable-tokenizer-autoload",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        ok = _wait_http_200(
            f"http://127.0.0.1:{user_port}/readiness",
            time.monotonic() + 2 * startup_timeout + 30,
        )
        if not ok:
            stdout, stderr = _stop_and_collect(proc, timeout=5)
            pytest.fail(
                f"gateway never became healthy.\nstdout:\n{stdout.decode()}\n"
                f"stderr:\n{stderr.decode()}"
            )

        proc.send_signal(signal.SIGTERM)
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            stdout, stderr = _stop_and_collect(proc, timeout=5)
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
