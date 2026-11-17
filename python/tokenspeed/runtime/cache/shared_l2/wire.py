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

"""The request-reply wire between the owner and its clients and providers.

A request is a msgpack map: ``op`` and the operation's fields. A reply is
``{"ok": True, ...}``, or ``{"ok": False, "error": ...}``.

Cache keys are opaque here. A key is a tuple on both sides and an array on the
wire, and every key the owner places travels with its bucket (its shard
number). A prompt's tokens travel as int32 bytes.
"""

from __future__ import annotations

import threading
from typing import Any

import msgspec
import zmq


class OwnerError(RuntimeError):
    """The owner answered a request with an error (an owner that does not
    know the request's client answers so), or with something that is not a
    reply of the wire."""


class _ThreadSocket(threading.local):
    """The calling thread's socket; None before its first request."""

    def __init__(self):
        self.socket: zmq.Socket | None = None


class OwnerConnection:
    """One REQ socket per calling thread; the owner reads a thread's requests
    in the order the thread sent them."""

    def __init__(self, address: str, *, timeout_ms: int):
        self._context = zmq.Context.instance()
        self._address = address
        self._timeout_ms = timeout_ms
        self._local = _ThreadSocket()
        self._sockets: list[zmq.Socket] = []
        self._lock = threading.Lock()

    def request(self, op: str, **fields: Any) -> dict[str, Any]:
        """Send one request and return its reply.

        Args:
            op: The operation.
            **fields: The operation's fields.

        Raises:
            TimeoutError: No reply within the connection's timeout. The owner
                may still handle the request.
            OwnerError: The owner rejected the request, or the reply cannot
                be decoded or lacks a field of the wire's reply.
        """
        socket = self._local.socket
        if socket is None:
            socket = self._context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
            socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            # A timed-out request leaves the socket usable for the next one,
            # and its late reply is not taken for the next one's.
            socket.setsockopt(zmq.REQ_RELAXED, 1)
            socket.setsockopt(zmq.REQ_CORRELATE, 1)
            socket.connect(self._address)
            self._local.socket = socket
            with self._lock:
                self._sockets.append(socket)
        try:
            socket.send(msgspec.msgpack.encode({"op": op, **fields}))
            payload = socket.recv()
        except zmq.Again as exc:
            raise TimeoutError(f"the shared L2 owner did not answer {op!r}") from exc
        try:
            reply = msgspec.msgpack.decode(payload, type=dict[str, Any])
            if reply["ok"]:
                return reply
            reason = reply["error"]
        except (msgspec.DecodeError, KeyError) as exc:
            raise OwnerError(
                f"the shared L2 owner's reply to {op!r} is not a reply: {exc!r}"
            ) from exc
        raise OwnerError(f"the shared L2 owner rejected {op!r}: {reason}")

    def close(self) -> None:
        with self._lock:
            for socket in self._sockets:
                socket.close(0)
            self._sockets.clear()
