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

"""The request-reply wire between the owner and its clients and providers."""

from __future__ import annotations

import threading
from typing import Any


class OwnerConnection:
    """One REQ socket per calling thread; every request is a msgpack map."""

    def __init__(self, address: str, *, timeout_ms: int):
        import zmq

        self._zmq = zmq
        self._context = zmq.Context.instance()
        self._address = address
        self._timeout_ms = timeout_ms
        self._local = threading.local()
        self._sockets: list[Any] = []
        self._lock = threading.Lock()

    def _socket(self):
        socket = self._local.__dict__.get("socket")
        if socket is None:
            socket = self._context.socket(self._zmq.REQ)
            socket.setsockopt(self._zmq.LINGER, 0)
            socket.setsockopt(self._zmq.RCVTIMEO, self._timeout_ms)
            socket.setsockopt(self._zmq.SNDTIMEO, self._timeout_ms)
            # A timed-out request leaves the socket usable for the next one.
            socket.setsockopt(self._zmq.REQ_RELAXED, 1)
            socket.setsockopt(self._zmq.REQ_CORRELATE, 1)
            socket.connect(self._address)
            self._local.socket = socket
            with self._lock:
                self._sockets.append(socket)
        return socket

    def request(self, op: str, **fields: Any) -> dict[str, Any]:
        import msgpack

        socket = self._socket()
        try:
            socket.send(msgpack.packb({"op": op, **fields}, use_bin_type=True))
            reply = msgpack.unpackb(socket.recv(), raw=False)
        except self._zmq.Again as exc:
            raise TimeoutError(f"the shared L2 owner did not answer {op!r}") from exc
        if not reply.get("ok", False):
            raise RuntimeError(
                f"the shared L2 owner rejected {op!r}: {reply.get('error')}"
            )
        return reply

    def close(self) -> None:
        with self._lock:
            for socket in self._sockets:
                socket.close(0)
            self._sockets.clear()
