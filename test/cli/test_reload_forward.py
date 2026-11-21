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

"""``ts reload-forward`` selects TokenSpeed workers and reports their verdicts."""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import NamedTuple

import pytest

from tokenspeed.cli.reload_forward import add_arguments, run


class _Worker(NamedTuple):
    engine: str
    status: int
    success: bool


class _Gateway(ThreadingHTTPServer):
    """Lists the workers and answers each reload with its configured verdict."""

    def __init__(self, workers: dict[str, _Worker]) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.workers = workers
        self.requests: list[tuple[str, dict]] = []


class _Handler(BaseHTTPRequestHandler):
    server: _Gateway

    def do_GET(self) -> None:
        workers = [
            {"id": worker_id, "engine": worker.engine}
            for worker_id, worker in self.server.workers.items()
        ]
        self._send(200, {"workers": workers})

    def do_POST(self) -> None:
        worker_id = self.path.split("/")[4]
        length = int(self.headers["Content-Length"])
        self.server.requests.append((worker_id, json.loads(self.rfile.read(length))))
        _, status, success = self.server.workers[worker_id]
        body = {"success": success, "message": "reloaded" if success else "undone"}
        self._send(status, {"worker_id": worker_id, "status": status, "body": body})

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        del format, args


@pytest.fixture
def gateway() -> Iterator[Callable[[dict[str, _Worker]], _Gateway]]:
    servers: list[_Gateway] = []

    def start(workers: dict[str, _Worker]) -> _Gateway:
        server = _Gateway(workers)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def _args(server: _Gateway, *flags: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    host, port = server.server_address
    return parser.parse_args(["--url", f"http://{host}:{port}", *flags])


def test_only_tokenspeed_workers_reload_and_share_one_id(gateway):
    server = gateway(
        {
            "a": _Worker("tokenspeed", 200, True),
            "other": _Worker("sglang", 200, True),
            "b": _Worker("tokenspeed", 200, True),
        }
    )
    assert run(_args(server, "--modules", "tokenspeed.runtime.models.qwen3")) == 0
    assert {worker_id for worker_id, _ in server.requests} == {"a", "b"}
    assert [body["modules"] for _, body in server.requests] == [
        ["tokenspeed.runtime.models.qwen3"]
    ] * 2
    assert len({body["id"] for _, body in server.requests}) == 1


@pytest.mark.parametrize(
    "workers",
    [{}, {"other": _Worker("sglang", 200, True)}],
    ids=["empty", "other-engine"],
)
def test_no_tokenspeed_workers_exits_one_without_reloading(gateway, workers, capsys):
    server = gateway(workers)
    assert run(_args(server)) == 1
    assert not server.requests
    assert "No TokenSpeed workers" in capsys.readouterr().err


@pytest.mark.parametrize("status", [200, 503])
def test_a_refused_or_undone_reload_on_any_engine_exits_one(gateway, status):
    server = gateway(
        {
            "a": _Worker("tokenspeed", 200, True),
            "b": _Worker("tokenspeed", status, False),
        }
    )
    assert run(_args(server)) == 1
