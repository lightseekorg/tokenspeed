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

"""``ts reload-forward`` exits by every engine's verdict, against a local gateway."""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tokenspeed.cli.reload_forward import add_arguments, run


class _Gateway(ThreadingHTTPServer):
    """Lists the workers and answers each reload with its (HTTP status, success)."""

    def __init__(self, replies: dict[str, tuple[int, bool]]) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.replies = replies
        self.bodies: list[dict] = []


class _Handler(BaseHTTPRequestHandler):
    server: _Gateway

    def do_GET(self) -> None:
        workers = [{"id": worker} for worker in self.server.replies]
        self._send(200, {"workers": workers})

    def do_POST(self) -> None:
        worker = self.path.split("/")[4]
        length = int(self.headers["Content-Length"])
        self.server.bodies.append(json.loads(self.rfile.read(length)))
        status, success = self.server.replies[worker]
        body = {"success": success, "message": "reloaded" if success else "undone"}
        self._send(status, {"worker_id": worker, "status": status, "body": body})

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
def gateway() -> Iterator[Callable[[dict[str, tuple[int, bool]]], _Gateway]]:
    servers: list[_Gateway] = []

    def start(replies: dict[str, tuple[int, bool]]) -> _Gateway:
        server = _Gateway(replies)
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


def test_every_engine_reloading_exits_zero_and_shares_one_id(gateway):
    server = gateway({"a": (200, True), "b": (200, True)})
    assert run(_args(server, "--modules", "tokenspeed.runtime.models.qwen3")) == 0
    assert [body["modules"] for body in server.bodies] == [
        ["tokenspeed.runtime.models.qwen3"]
    ] * 2
    assert len({body["id"] for body in server.bodies}) == 1


@pytest.mark.parametrize("reply", [(200, False), (503, False)])
def test_a_refused_or_undone_reload_on_any_engine_exits_one(gateway, reply):
    server = gateway({"a": (200, True), "b": reply})
    assert run(_args(server)) == 1
