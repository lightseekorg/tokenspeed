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

"""Per-result stream completion for asynchronous NPU output copies."""

from __future__ import annotations

from threading import Lock
from typing import Any


class CopyStreamPool:
    """Lend a stream until its result is committed, never to two pending results.

    The forward thread acquires streams; the commit thread returns them only
    after synchronization. In-flight depth therefore bounds the pool's size.
    """

    def __init__(self, device_module: Any, device: Any) -> None:
        self.device_module = device_module
        self.device = device
        self._available: list[Any] = []
        self._lock = Lock()

    def copy_after(
        self, producer: Any, sources: tuple[Any, ...]
    ) -> CopyStreamCompletion:
        with self._lock:
            stream = self._available.pop() if self._available else None
        if stream is None:
            stream = self.device_module.Stream(device=self.device)
        return CopyStreamCompletion(self, stream, producer, sources)

    def _release(self, stream: Any) -> None:
        with self._lock:
            self._available.append(stream)


class CopyStreamCompletion:
    """Copy scope and host completion boundary for exactly one forward result.

    Device-side waits order production, copying and subsequent scratch reuse.
    Host commit waits only on this copy stream, which cannot contain a later
    forward's collectives. Source references survive until that wait finishes.
    """

    def __init__(
        self,
        pool: CopyStreamPool,
        stream: Any,
        producer: Any,
        sources: tuple[Any, ...],
    ) -> None:
        self._pool = pool
        self.stream = stream
        self._producer = producer
        self._sources = sources
        self._context: Any = None
        self._sealed = False
        self._synced = False

    def __enter__(self) -> CopyStreamCompletion:
        self.stream.wait_stream(self._producer)
        self._context = self._pool.device_module.stream(self.stream)
        self._context.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._context.__exit__(exc_type, exc_value, traceback)
        # The sampler reuses its output pack on the next forward. Do not let
        # that overwrite race this stream's reads, including optional logprobs.
        self._producer.wait_stream(self.stream)
        self._sealed = True
        if exc_type is not None:
            # No result will reach commit on failure. Keep its sources alive
            # through cleanup; a failed synchronization never recycles a stream.
            self.synchronize()

    def synchronize(self) -> None:
        if not self._sealed or self._synced:
            raise RuntimeError(
                "Copy stream completion requires one finished copy scope."
            )
        # An event wait returned before D2H had landed on the installed NPU
        # stack. A stream leased to this result gives a bounded stream wait;
        # synchronizing the producer could wait on future, undispatched peers.
        with self._pool.device_module.device(self._pool.device):
            self.stream.synchronize()
        self._sources = ()
        self._synced = True
        self._pool._release(self.stream)
