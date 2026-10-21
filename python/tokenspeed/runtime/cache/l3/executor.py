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

"""Bounded Host-only I/O. No scheduler, collective, model, or device work."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokenspeed.runtime.cache.l3.backend import KvStoreStorage, storage_object_key
from tokenspeed.runtime.cache.l3.config import StorageRuntimeConfig
from tokenspeed.runtime.cache.l3.io_types import (
    StorageCompletion,
    StorageIOResult,
    StoragePoll,
)
from tokenspeed.runtime.cache.l3.statistics import (
    ExecutorSnapshot,
    HealthSnapshot,
    OperationCounters,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.host_storage import HostCacheStorage
    from tokenspeed.runtime.cache.transfer.ownership import BlockOwnerTranslation

logger = logging.getLogger(__name__)

# Shutdown errors must not turn into deregistration when the owning runtime is
# garbage-collected. Only process exit releases an unretired native transport.
_unretired_resources: list[tuple[KvStoreStorage, HostCacheStorage]] = []


@dataclass(frozen=True, slots=True)
class _HealthToken:
    epoch: int
    recovery: bool


class _BackendHealth:
    def __init__(
        self,
        *,
        failure_threshold: int,
        recovery_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        if failure_threshold < 1 or recovery_interval_s < 0:
            raise ValueError("Invalid L3 health limits")
        self._threshold = failure_threshold
        self._interval = recovery_interval_s
        self._clock = clock
        self._lock = threading.Lock()
        self._epoch = 0
        self._failures = 0
        self._unavailable_until: float | None = None
        self._probe_inflight = False

    def snapshot(self) -> HealthSnapshot:
        """Observe health without acquiring the half-open probe permission."""
        with self._lock:
            state = (
                "healthy"
                if self._unavailable_until is None
                else "half_open" if self._probe_inflight else "open"
            )
            return HealthSnapshot(state, self._epoch, self._failures)

    def begin_operation(self) -> _HealthToken | None:
        with self._lock:
            if self._unavailable_until is None:
                return _HealthToken(self._epoch, False)
            if self._clock() < self._unavailable_until or self._probe_inflight:
                return None
            self._probe_inflight = True
            return _HealthToken(self._epoch, True)

    def report_failure(self, token: _HealthToken) -> None:
        with self._lock:
            if token.epoch != self._epoch:
                return
            self._failures += 1
            if token.recovery or self._failures >= self._threshold:
                self._epoch += 1
                self._unavailable_until = self._clock() + self._interval
                self._probe_inflight = False

    def report_success(self, token: _HealthToken) -> None:
        with self._lock:
            if token.epoch != self._epoch:
                return
            self._failures = 0
            if token.recovery:
                self._epoch += 1
                self._unavailable_until = None
                self._probe_inflight = False


@dataclass(frozen=True, slots=True)
class StorageJob:
    op_id: int
    kind: str
    keys: tuple[str, ...]
    offsets: tuple[int | None, ...]
    sizes: tuple[int, ...]


@dataclass(slots=True)
class _JobState:
    job: StorageJob
    submitted_at: float
    num_bytes: int
    started_at: float | None = None
    started: bool = False
    cancelled: bool = False
    completed: bool = False


class L3Executor:
    def __init__(
        self,
        backend: KvStoreStorage,
        host_storage: HostCacheStorage,
        *,
        key_prefix: str,
        rank: int,
        ownership: BlockOwnerTranslation | None,
        config: StorageRuntimeConfig,
        prefix_for_weight_version: Callable[[str], str] | None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.backend = backend
        self.host_storage = host_storage
        self._key_prefix = key_prefix
        self._rank = int(rank)
        self._ownership = ownership
        self.config = config
        self._prefix_for_weight_version = prefix_for_weight_version
        self._closed = False
        self._clock = clock
        self._health = _BackendHealth(
            failure_threshold=config.failure_threshold,
            recovery_interval_s=config.recovery_interval_ms / 1000,
            clock=clock,
        )
        self._lock = threading.Lock()
        self._jobs: dict[int, _JobState] = {}
        self._results: queue.SimpleQueue[StorageCompletion] = queue.SimpleQueue()
        self._queues = {
            kind: queue.Queue(config.scheduler.max_operations)
            for kind in ("query", "get", "put")
        }
        self._stopping = threading.Event()
        self._unsafe_io = False
        self._retained_on_shutdown = False
        self._last_submitted = 0
        self._counters = {kind: OperationCounters() for kind in self._queues}
        self._threads: list[threading.Thread] = []
        for kind, count in (
            ("query", config.query_workers),
            ("get", config.read_workers),
            ("put", config.write_workers),
        ):
            for index in range(count):
                thread = threading.Thread(
                    target=self._worker,
                    args=(kind,),
                    name=f"l3-{kind}-{index}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()

    @property
    def key_prefix(self) -> str:
        return self._key_prefix

    def _object_key(self, content_hash: str, group_id: int, page_offset: int) -> str:
        return storage_object_key(
            content_hash,
            group_id,
            page_offset,
            prefix=self._key_prefix,
            rank=(
                self._rank // self._ownership.shard_counts[group_id]
                if self._ownership is not None
                else self._rank
            ),
        )

    def _page_ranges(
        self, pages: Sequence[tuple[int, int, str, int]]
    ) -> tuple[list[str], list[int | None], list[int]]:
        """Freeze keys and local Host ranges in logical operation order.

        Rows are (group, Host block, content hash, logical offset). A None byte
        offset means this rank owns no payload; its completion stays neutral.
        Object keys use replica identity so a later allocation may change owner.
        """
        keys = [self._object_key(h, g, o) for g, _, h, o in pages]
        if self._ownership is None:
            ranges = [self.host_storage.host_block_range(g, b) for g, b, _, _ in pages]
            return keys, [offset for offset, _ in ranges], [size for _, size in ranges]
        offsets: list[int | None] = [None] * len(pages)
        sizes = [self.host_storage.host_cache_block_bytes[g] for g, _, _, _ in pages]
        owned = self._ownership.owned_host_positions([(g, b) for g, b, _, _ in pages])
        for position, (group, local_block) in owned:
            offsets[position], sizes[position] = self.host_storage.host_block_range(
                group, local_block
            )
        return keys, offsets, sizes

    def submit_operations(self, operations) -> None:
        """Snapshot plain storage descriptors and enqueue without waiting."""
        from tokenspeed_scheduler import StorageKind

        if not operations:
            return
        kinds = {
            StorageKind.Query: "query",
            StorageKind.Get: "get",
            StorageKind.Put: "put",
        }
        for operation in operations:
            if operation.kind == StorageKind.Cancel:
                self.cancel(int(operation.op_id))
                continue
            kind = kinds[operation.kind]
            logical_keys = list(operation.keys)
            # Resolve namespace now; later weight-version changes cannot alter
            # a descriptor already owned by a worker.
            if kind == "query":
                keys = [
                    self._object_key(str(h), int(g), int(o)) for g, h, o in logical_keys
                ]
                offsets, sizes = (), ()
            else:
                keys, offsets, sizes = self._page_ranges(
                    [
                        (int(g), int(block), str(h), int(o))
                        for (g, h, o), block in zip(
                            logical_keys, operation.host_blocks, strict=True
                        )
                    ]
                )
            self.submit(
                StorageJob(
                    int(operation.op_id),
                    kind,
                    tuple(keys),
                    tuple(offsets),
                    tuple(sizes),
                )
            )

    def set_weight_version(self, weight_version: str) -> None:
        """Switch the namespace after the caller's replica-agreed cache clear."""
        if self._prefix_for_weight_version is None:
            raise RuntimeError("L3 prefix cannot be rebuilt without a factory")
        if not self.is_idle():
            raise RuntimeError("L3 operations must retire before changing namespace")
        key_prefix = self._prefix_for_weight_version(weight_version)
        if not key_prefix:
            raise ValueError("key_prefix must be non-empty")
        self._key_prefix = key_prefix

    def delete_namespace(self) -> bool:
        """Delete an idle namespace after the caller agrees that caches can clear."""
        if not self.is_idle():
            return False
        try:
            return bool(self.backend.remove_by_prefix(f"{self._key_prefix}_"))
        except Exception:
            logger.exception("L3 namespace delete failed")
            return False

    def close(self, *, timeout_s: float) -> None:
        """Retire native work before deregistration; retain Host storage on failure.

        The engine drains device copies separately before calling this method.
        L2 and L3 each retain the same Host allocation, not each other's executor.
        """
        if self._closed:
            return
        self._shutdown_workers(timeout_s=timeout_s)
        self.backend.close()
        self._closed = True

    def submit(self, job: StorageJob) -> None:
        if (
            job.kind not in self._queues
            or not job.keys
            or len(job.keys) > self.config.scheduler.max_keys_per_operation
        ):
            raise ValueError("Invalid L3 job descriptor")
        if job.kind != "query" and not (
            len(job.keys) == len(job.offsets) == len(job.sizes)
        ):
            raise ValueError("Ragged L3 job ranges")
        if job.kind == "query" and (job.offsets or job.sizes):
            raise ValueError("L3 queries cannot carry Host byte ranges")
        with self._lock:
            if job.op_id <= self._last_submitted:
                raise ValueError("L3 operation ids must increase")
            self._last_submitted = job.op_id
            if len(self._jobs) >= self.config.scheduler.max_operations:
                # The scheduler bounds all accepted operations; exceeding that
                # bound is a protocol fault, not a silent dropped completion.
                raise RuntimeError("L3 scheduler exceeded operation credits")
            state = _JobState(
                job,
                self._clock(),
                sum(
                    size
                    for offset, size in zip(job.offsets, job.sizes)
                    if offset is not None
                ),
            )
            self._counters[job.kind].accepted += 1
            self._jobs[job.op_id] = state
            if self._stopping.is_set() or self._unsafe_io:
                self._counters[job.kind].rejected += 1
                self._finish_locked(
                    state,
                    StorageIOResult(
                        (False,) * len(job.keys), True, (False,) * len(job.keys)
                    ),
                )
                return
            try:
                self._queues[job.kind].put_nowait(state)
            except queue.Full:
                self._counters[job.kind].rejected += 1
                self._finish_locked(
                    state,
                    StorageIOResult(
                        (False,) * len(job.keys), True, (False,) * len(job.keys)
                    ),
                )

    def cancel(self, op_id: int) -> None:
        with self._lock:
            state = self._jobs.get(op_id)
            if state is not None and not state.completed and not state.cancelled:
                state.cancelled = True
                counters = self._counters[state.job.kind]
                if state.started:
                    counters.cancelled_running += 1
                else:
                    counters.cancelled_queued += 1
                if not state.started:
                    # Retire immediately even when every worker is blocked in
                    # native I/O. The inert queue entry cannot start after this
                    # transition, and the worker later discards it exactly once.
                    self._finish_locked(
                        state,
                        StorageIOResult(
                            (False,) * len(state.job.keys),
                            True,
                            (False,) * len(state.job.keys),
                        ),
                    )

    def _finish_locked(self, state: _JobState, result: StorageIOResult) -> None:
        state.completed = True
        counters = self._counters[state.job.kind]
        counters.completed += 1
        positions = (
            range(len(state.job.keys))
            if state.job.kind == "query"
            else [i for i, offset in enumerate(state.job.offsets) if offset is not None]
        )
        counters.success_keys += sum(result.success[i] for i in positions)
        counters.failed_keys += sum(not result.success[i] for i in positions)
        counters.success_bytes += sum(
            size
            for offset, size, success in zip(
                state.job.offsets, state.job.sizes, result.success
            )
            if offset is not None and success
        )
        if state.started_at is None:
            counters.queue_seconds += self._clock() - state.submitted_at
        if not result.retired:
            counters.unretired += 1
            self._unsafe_io = True
        self._results.put(
            StorageCompletion(
                state.job.op_id, result.success, result.retired, result.object_failures
            )
        )

    def _worker(self, kind: str) -> None:
        work_queue = self._queues[kind]
        while not self._stopping.is_set() or not work_queue.empty():
            try:
                state = work_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            job = state.job
            with self._lock:
                if state.completed:
                    work_queue.task_done()
                    continue
                if state.cancelled or self._stopping.is_set() or self._unsafe_io:
                    self._counters[kind].rejected += 1
                    self._finish_locked(
                        state,
                        StorageIOResult(
                            (False,) * len(job.keys), True, (False,) * len(job.keys)
                        ),
                    )
                    work_queue.task_done()
                    continue
                state.started = True
                state.started_at = self._clock()
                self._counters[kind].queue_seconds += (
                    state.started_at - state.submitted_at
                )
            positions = (
                tuple(range(len(job.keys)))
                if kind == "query"
                else tuple(
                    i for i, offset in enumerate(job.offsets) if offset is not None
                )
            )
            if not positions:
                # No native work exists on this owner. It must not act as a
                # health recovery probe or charge bytes it did not transfer.
                with self._lock:
                    self._finish_locked(
                        state,
                        StorageIOResult(
                            (True,) * len(job.keys), True, (False,) * len(job.keys)
                        ),
                    )
                work_queue.task_done()
                continue
            local_keys = tuple(job.keys[i] for i in positions)
            local_offsets = (
                tuple(job.offsets[i] for i in positions) if kind != "query" else ()
            )
            local_sizes = (
                tuple(job.sizes[i] for i in positions) if kind != "query" else ()
            )
            token = self._health.begin_operation()
            if token is None:
                with self._lock:
                    self._counters[kind].rejected += 1
                result = StorageIOResult(
                    (False,) * len(local_keys), True, (False,) * len(local_keys)
                )
            else:
                started_at = self._clock()
                with self._lock:
                    self._counters[kind].backend_calls += 1
                try:
                    backend = self.backend
                    if kind == "query":
                        flags = tuple(backend.batch_exists(local_keys))
                        result = StorageIOResult(
                            flags, True, (False,) * len(local_keys)
                        )
                    elif kind == "get":
                        result = backend.batch_get_into_result(
                            local_keys,
                            self.host_storage.host_buffer,
                            local_offsets,
                            local_sizes,
                        )
                    else:
                        result = backend.batch_put_from_result(
                            local_keys,
                            self.host_storage.host_buffer,
                            local_offsets,
                            local_sizes,
                        )
                    if (
                        type(result.retired) is not bool
                        or len(result.success) != len(local_keys)
                        or any(type(flag) is not bool for flag in result.success)
                        or len(result.object_failures) != len(local_keys)
                        or any(
                            type(flag) is not bool for flag in result.object_failures
                        )
                        or any(
                            ok and failed
                            for ok, failed in zip(
                                result.success, result.object_failures
                            )
                        )
                        or (kind != "get" and any(result.object_failures))
                    ):
                        raise ValueError("Malformed L3 result")
                except Exception:
                    logger.exception(f"L3 {kind} failed for operation {job.op_id}")
                    result = StorageIOResult(
                        (False,) * len(local_keys),
                        kind == "query" or self.backend.failures_retire,
                        (False,) * len(local_keys),
                    )
                    self._health.report_failure(token)
                else:
                    # Retired failures are object or capacity facts, never
                    # backend unavailability: a retired GET may be an
                    # eviction race or an object with incompatible length,
                    # and a retired PUT may be the store's allocation
                    # rejection. Soft timeout also never changes health.
                    if kind == "query" or result.retired:
                        self._health.report_success(token)
                    else:
                        self._health.report_failure(token)
                finally:
                    with self._lock:
                        self._counters[kind].backend_seconds += (
                            self._clock() - started_at
                        )
            success = [True] * len(job.keys)
            failures = [False] * len(job.keys)
            for position, ok, failed in zip(
                positions, result.success, result.object_failures, strict=True
            ):
                success[position], failures[position] = ok, failed
            result = StorageIOResult(tuple(success), result.retired, tuple(failures))
            with self._lock:
                self._finish_locked(state, result)
            work_queue.task_done()

    def poll(self) -> StoragePoll:
        """Drain completions and sample pressure under one lock, without I/O.

        Native work that has retired no longer contributes to pressure. Running
        cancelled work still does; request detachment cannot retire its buffers.
        The scheduler separately charges Ready holds and quarantined references.
        """
        results = []
        with self._lock:
            while True:
                try:
                    result = self._results.get_nowait()
                except queue.Empty:
                    break
                del self._jobs[result.op_id]
                results.append(result)
            now = self._clock()
            queued_operations = 0
            queued_bytes = 0
            oldest = now
            for state in self._jobs.values():
                if state.completed:
                    continue
                oldest = min(oldest, state.submitted_at)
                if not state.started:
                    queued_operations += 1
                    queued_bytes += state.num_bytes
            return StoragePoll(
                tuple(results),
                queued_operations,
                queued_bytes,
                max(0, int((now - oldest) * 1_000_000)),
            )

    def is_idle(self) -> bool:
        with self._lock:
            return not self._jobs and not self._unsafe_io

    def snapshot(self) -> ExecutorSnapshot:
        """Copy bounded counters and current ownership without backend/GPU work.

        Byte counters describe successful adapter objects, including idempotent
        PUT hits. They do not measure network traffic or scheduler publication.
        Completed jobs remain counted until poll delivers their facts.
        """
        with self._lock:
            rows = []
            for kind, counters in self._counters.items():
                operations = dict(queued=0, running=0, completed=0)
                byte_counts = dict(queued=0, running=0, completed=0)
                for state in self._jobs.values():
                    if state.job.kind != kind:
                        continue
                    phase = (
                        "completed"
                        if state.completed
                        else "running" if state.started else "queued"
                    )
                    operations[phase] += 1
                    byte_counts[phase] += state.num_bytes
                rows.append(
                    counters.snapshot(
                        kind, operations=operations, byte_counts=byte_counts
                    )
                )
            return ExecutorSnapshot(
                tuple(rows), self._health.snapshot(), self._unsafe_io
            )

    def _shutdown_workers(self, *, timeout_s: float) -> None:
        self._stopping.set()
        deadline = time.monotonic() + timeout_s
        for thread in self._threads:
            thread.join(max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in self._threads) or self._unsafe_io:
            if not self._retained_on_shutdown:
                _unretired_resources.append((self.backend, self.host_storage))
                self._retained_on_shutdown = True
            raise RuntimeError(
                "L3 I/O retirement is unproven; retaining registered Host storage; restart required"
            )
