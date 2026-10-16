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

Shutdown stops accepting submissions, then the loop's owner thread cancels
and awaits pending tasks, finalizes async generators, and closes the loop.
Normal shutdown waits for that thread to exit. ``Engine`` still releases its
subprocesses if cleanup fails or times out.
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
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self._shutdown_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run_loop,
            name="tokenspeed-llm-sync",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            try:
                self._loop.run_forever()
            finally:
                with self._shutdown_lock:
                    self._shutdown_started = True
                try:
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        self._loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                        for task in pending:
                            if not task.cancelled():
                                error = task.exception()
                                if error is not None and self._shutdown_error is None:
                                    self._shutdown_error = error
                    self._shutdown_asyncgens()
                finally:
                    self._loop.close()
        except BaseException as exc:  # noqa: BLE001 — reported to the caller
            self._shutdown_error = exc

    def _shutdown_asyncgens(self) -> None:
        previous_handler = self._loop.get_exception_handler()

        def record_cleanup_error(loop, context) -> None:
            # shutdown_asyncgens reports aclose failures through the loop's
            # exception handler instead of raising them to its caller.
            error = context.get("exception")
            if (
                "asyncgen" in context
                and isinstance(error, BaseException)
                and self._shutdown_error is None
            ):
                self._shutdown_error = error
            if previous_handler is None:
                loop.default_exception_handler(context)
            else:
                previous_handler(loop, context)

        self._loop.set_exception_handler(record_cleanup_error)
        try:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        finally:
            self._loop.set_exception_handler(previous_handler)

    def _submit(self, coro):
        # Serialize submission with shutdown so no work lands after loop.stop.
        with self._shutdown_lock:
            if self._shutdown_started:
                coro.close()
                raise RuntimeError("LLM has been shut down")
            return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def run(self, coro) -> Any:
        """Submit ``coro`` to the bg loop and block until it completes.

        This is the single sync entry point used by ``Engine`` for every
        blocking coroutine call (``flush_cache``, ``start_profile``,
        weight sync, internal-state queries, …). It replaces the
        deprecated ``asyncio.get_event_loop()`` +
        ``run_until_complete(coro)`` idiom that previously lived inline
        at every call site.
        """
        return self._submit(coro).result()

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

        def _cancelled_before_start(future) -> None:
            # A task cancelled before its first step never enters _drain's
            # finally block, but the blocking iterator still needs a terminal.
            if future.cancelled():
                q.put(asyncio.CancelledError())
                q.put(_STREAM_END)

        future = self._submit(_drain())
        future.add_done_callback(_cancelled_before_start)

        while True:
            item = q.get()
            if item is _STREAM_END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def shutdown(self) -> None:
        """Cancel pending work, close the bg event loop, and join its thread.

        Idempotent, including concurrent callers. Pending sync calls receive
        cancellation; later submissions fail immediately. Call this before
        ``AsyncLLM`` teardown so task and generator cleanup can still use it.
        Raises if the owner thread fails to clean up or cannot stop within
        five seconds.
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError("Cannot shut down LLM from its event loop thread")
        with self._shutdown_lock:
            if not self._shutdown_started:
                self._shutdown_started = True
                self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("LLM event loop did not shut down within 5 seconds")
        if self._shutdown_error is not None:
            raise RuntimeError(
                "LLM event loop shutdown failed"
            ) from self._shutdown_error
