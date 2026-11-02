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

"""The launcher's bootstrap server (``tokenspeed_engine.BootstrapServer``):
it serves once constructed, a taken port fails its construction, and closing
it releases the port."""

from __future__ import annotations

import http.client
import os
import socket
import sys

import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed_engine import BootstrapServer  # noqa: E402


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get_health(port: int) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_constructor_returns_a_serving_server() -> None:
    port = _unused_tcp_port()
    server = BootstrapServer(port, cache_layouts=False)
    try:
        assert server.port == port
        assert _get_health(port) == (200, b"OK")
    finally:
        server.close()


def test_a_taken_port_fails_the_constructor() -> None:
    first = BootstrapServer(0, cache_layouts=True)
    try:
        with pytest.raises(RuntimeError, match="Bootstrap server failed to bind"):
            BootstrapServer(first.port, cache_layouts=True)
    finally:
        first.close()


def test_close_is_idempotent_and_releases_listening_port() -> None:
    port = _unused_tcp_port()
    first = BootstrapServer(port, cache_layouts=True)

    first.close()
    first.close()

    with pytest.raises(ConnectionRefusedError):
        _get_health(port)
    # Immediate reuse proves the listener closed with the server.
    second = BootstrapServer(port, cache_layouts=True)
    try:
        assert _get_health(port) == (200, b"OK")
    finally:
        second.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
