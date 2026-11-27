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

"""Wire transport unit tests for the shared L2 pool owner client."""

import os
import sys
import threading

import msgspec
import pytest
import zmq

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.cache.shared_l2.wire import OwnerConnection  # noqa: E402


def test_owner_connection_round_trips_ipv6_loopback() -> None:
    """OwnerConnection enables zmq.IPV6 so IPv6 loopback and hostnames connect."""
    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.setsockopt(zmq.IPV6, 1)
    rep.setsockopt(zmq.RCVTIMEO, 2000)
    try:
        try:
            port = rep.bind_to_random_port("tcp://[::1]")
        except zmq.ZMQError as exc:
            pytest.skip(f"IPv6 loopback unavailable in this environment: {exc}")
        received: list[dict[str, object]] = []

        def serve_one() -> None:
            payload = rep.recv()
            req = msgspec.msgpack.decode(payload, type=dict[str, object])
            received.append(req)
            rep.send(msgspec.msgpack.encode({"ok": True, "echo": req["op"]}))

        server = threading.Thread(target=serve_one, daemon=True)
        server.start()
        conn = OwnerConnection(f"tcp://[::1]:{port}", timeout_ms=2000)
        try:
            reply = conn.request("ping", client=7)
        finally:
            conn.close()
        server.join(timeout=2.0)
        assert not server.is_alive()
        assert received == [{"op": "ping", "client": 7}]
        assert reply == {"ok": True, "echo": "ping"}
    finally:
        rep.close(0)
