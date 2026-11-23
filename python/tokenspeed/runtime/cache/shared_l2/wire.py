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

from typing import Any

import msgspec
import zmq


class OwnerError(RuntimeError):
    """The owner answered a request with an error (an owner that does not
    know the request's client answers so), or with something that is not a
    reply of the wire."""


class OwnerConnection:
    """One REQ socket, for one thread at a time; the owner reads its requests
    in the order they were sent."""

    def __init__(self, address: str, *, timeout_ms: int):
        self._socket = zmq.Context.instance().socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        # A timed-out request leaves the socket usable for the next one,
        # and its late reply is not taken for the next one's.
        self._socket.setsockopt(zmq.REQ_RELAXED, 1)
        self._socket.setsockopt(zmq.REQ_CORRELATE, 1)
        self._socket.connect(address)

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
        try:
            self._socket.send(msgspec.msgpack.encode({"op": op, **fields}))
            payload = self._socket.recv()
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
        self._socket.close(0)
