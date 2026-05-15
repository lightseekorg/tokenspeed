# Adapted from meituan-longcat/SGLang-FluentLLM.
# This file has been modified for this repository.
# This file may incorporate material from ModelTC/lightllm,
# vllm-project/vllm, and sgl-project/sglang, as identified in
# python/THIRDPARTYNOTICES.

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

import asyncio


class RWLock:
    def __init__(self):
        # Protects internal state
        self._lock = asyncio.Lock()

        # Condition variable used to wait for state changes
        self._cond = asyncio.Condition(self._lock)

        # Number of readers currently holding the lock
        self._readers = 0

        # Whether a writer is currently holding the lock
        self._writer_active = False

        # How many writers are queued waiting for a turn
        self._waiting_writers = 0

    @property
    def reader_lock(self):
        """
        A context manager for acquiring a shared (reader) lock.

        Example:
            async with rwlock.reader_lock:
                # read-only access
        """
        return _ReaderLock(self)

    @property
    def writer_lock(self):
        """
        A context manager for acquiring an exclusive (writer) lock.

        Example:
            async with rwlock.writer_lock:
                # exclusive access
        """
        return _WriterLock(self)

    async def acquire_reader(self):
        async with self._lock:
            # Wait until there is no active writer or waiting writer
            # to ensure fairness.
            while self._writer_active or self._waiting_writers > 0:
                await self._cond.wait()
            self._readers += 1

    async def release_reader(self):
        async with self._lock:
            self._readers -= 1
            # If this was the last reader, wake up anyone waiting
            # (potentially a writer or new readers).
            if self._readers == 0:
                self._cond.notify_all()

    async def acquire_writer(self):
        async with self._lock:
            # Increment the count of writers waiting
            self._waiting_writers += 1
            try:
                # Wait while either a writer is active or readers are present
                while self._writer_active or self._readers > 0:
                    await self._cond.wait()
                self._writer_active = True
            finally:
                # Decrement waiting writers only after we've acquired the writer lock
                self._waiting_writers -= 1

    async def release_writer(self):
        async with self._lock:
            self._writer_active = False
            # Wake up anyone waiting (readers or writers)
            self._cond.notify_all()


class _ReaderLock:
    def __init__(self, rwlock: RWLock):
        self._rwlock = rwlock

    async def __aenter__(self):
        await self._rwlock.acquire_reader()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._rwlock.release_reader()


class _WriterLock:
    def __init__(self, rwlock: RWLock):
        self._rwlock = rwlock

    async def __aenter__(self):
        await self._rwlock.acquire_writer()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._rwlock.release_writer()
