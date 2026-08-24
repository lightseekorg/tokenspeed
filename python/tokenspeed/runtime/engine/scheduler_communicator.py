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

from __future__ import annotations

import asyncio
import copy
from collections import deque
from typing import Generic, Literal, Protocol, TypeVar

T = TypeVar("T")
_Mode = Literal["queueing", "watching"]


class _Sender(Protocol):
    def send_pyobj(self, obj: object) -> object: ...


class _Communicator(Generic[T]):
    """Note: The communicator now only run up to 1 in-flight request at any time."""

    def __init__(self, sender: _Sender, fan_out: int, mode: _Mode = "queueing") -> None:
        self._sender = sender
        self._fan_out = fan_out
        self._mode = mode
        self._result_event: asyncio.Event | None = None
        self._result_values: list[T] | None = None
        self._ready_queue: deque[asyncio.Event] = deque()

        if mode not in ("queueing", "watching"):
            raise ValueError(f"Invalid communicator mode: {mode}")

    async def queueing_call(self, obj: T) -> list[T]:
        ready_event = asyncio.Event()
        if self._result_event is not None or len(self._ready_queue) > 0:
            self._ready_queue.append(ready_event)
            await ready_event.wait()
            if self._result_event is not None or self._result_values is not None:
                raise RuntimeError("Communicator result state was not reset.")

        if obj:
            self._sender.send_pyobj(obj)

        self._result_event = asyncio.Event()
        self._result_values = []
        await self._result_event.wait()
        result_values = self._result_values
        self._result_event = self._result_values = None

        if len(self._ready_queue) > 0:
            self._ready_queue.popleft().set()

        return result_values

    async def watching_call(self, obj: T) -> list[T]:
        if self._result_event is None:
            if self._result_values is not None:
                raise RuntimeError("Communicator result values were not reset.")
            self._result_values = []
            self._result_event = asyncio.Event()
            if obj:
                self._sender.send_pyobj(obj)

        result_event = self._result_event
        result_values = self._result_values
        if result_event is None or result_values is None:
            raise RuntimeError("Communicator watching state is incomplete.")

        await result_event.wait()
        return copy.deepcopy(result_values)

    async def __call__(self, obj: T) -> list[T]:
        if self._mode == "queueing":
            return await self.queueing_call(obj)
        else:
            return await self.watching_call(obj)

    def handle_recv(self, recv_obj: T) -> None:
        result_event = self._result_event
        result_values = self._result_values
        if result_event is None or result_values is None:
            raise RuntimeError("Communicator result state is incomplete.")

        result_values.append(recv_obj)
        if len(result_values) == self._fan_out:
            if self._mode == "watching":
                self._result_event = self._result_values = None
            result_event.set()
