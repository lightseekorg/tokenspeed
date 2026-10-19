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

"""Sync facade over ``AsyncLLM`` for blocking callers.

A dedicated daemon thread owns a private ``asyncio`` event loop that
drives ``AsyncLLM.generate_request`` coroutines. Sync callers submit
work via ``asyncio.run_coroutine_threadsafe``; streaming callers
consume results through a ``queue.Queue`` with three-valued
semantics (normal item / terminal sentinel / exception) so
exceptions propagate to the caller thread instead of getting
swallowed by the driver loop.

Bridging at this layer — rather than maintaining a separate sync
IPC client — keeps ``AsyncLLM`` single-producer and the scheduler
IPC surface single-caller, while still letting ``Engine`` expose a
blocking API to callers that cannot own an event loop themselves.
"""

import asyncio
import queue
import threading
from collections.abc import Iterator
from typing import Any

from tokenspeed.runtime.engine.async_llm import AsyncLLM
from tokenspeed.runtime.engine.io_struct import GenerateReqInput

# Sentinel marking the end of a streaming bridge. Identity-compared,
# so a dict payload happening to equal this value is impossible.
_STREAM_END = object()


class LLM:
    """Sync adapter around an existing ``AsyncLLM`` instance.

    Owns a daemon thread running its own ``asyncio`` event loop; all
    coroutines are driven on that loop via
    ``asyncio.run_coroutine_threadsafe``. The adapter is cheap to
    construct and safe to share across caller threads — the bg loop
    is single-threaded, so ordering of queued work matches the
    submission order.
    """

    def __init__(self, async_llm: AsyncLLM) -> None:
        self.async_llm = async_llm
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="tokenspeed-llm-sync",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro) -> Any:
        """Submit ``coro`` to the bg loop and block until it completes.

        This is the single sync entry point used by ``Engine`` for every
        blocking coroutine call (``flush_cache``, ``start_profile``,
        weight sync, internal-state queries, …).
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def generate(self, obj: GenerateReqInput) -> dict:
        """Run ``AsyncLLM.generate_request`` to completion and return the
        final dict. Used by non-streaming sync paths
        (``Engine.generate`` with ``stream=False``).
        """

        async def _one() -> dict:
            gen = self.async_llm.generate_request(obj)
            return await gen.__anext__()

        return self.run(_one())

    def generate_stream(self, obj: GenerateReqInput) -> Iterator[dict]:
        """Run ``AsyncLLM.generate_request`` and yield each output dict
        as it lands. Implements the three-valued queue bridge:

          * dicts are produced by the async generator → ``q.put(item)``
          * terminal → ``q.put(_STREAM_END)``
          * exception → ``q.put(exc)`` then ``q.put(_STREAM_END)``

        The drain coroutine is fire-and-forget on the bg loop. If the
        caller abandons the iterator early (e.g. ``break`` out of the
        ``for`` loop) the drain will keep running to completion and
        push items into a queue nobody reads — that is acceptable for
        finite ``generate_request`` streams, which is the only kind
        ``AsyncLLM`` emits today.
        """
        q: queue.Queue[Any] = queue.Queue()

        async def _drain() -> None:
            pending_exc: BaseException | None = None
            try:
                async for item in self.async_llm.generate_request(obj):
                    q.put(item)
            except BaseException as exc:  # noqa: BLE001 — propagate anything
                pending_exc = exc
            finally:
                if pending_exc is not None:
                    q.put(pending_exc)
                q.put(_STREAM_END)

        asyncio.run_coroutine_threadsafe(_drain(), self._loop)

        while True:
            item = q.get()
            if item is _STREAM_END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def shutdown(self) -> None:
        """Stop the bg event loop and join the thread.

        Idempotent. Safe to call before ``AsyncLLM`` teardown. The
        daemon flag means an accidentally-skipped shutdown will not
        hang interpreter exit, but leaving the loop running past
        ``AsyncLLM`` teardown will surface as "Event loop is closed"
        errors on later submissions, so call this before tearing
        down the engine.
        """
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
