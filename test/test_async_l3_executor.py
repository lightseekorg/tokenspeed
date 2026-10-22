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

import gc
import json
import queue
import threading
import time
import weakref
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tokenspeed.runtime.cache.l3.backend import MemoryKvStore
from tokenspeed.runtime.cache.l3.config import StorageRuntimeConfig
from tokenspeed.runtime.cache.l3.executor import (
    L3Executor,
    StorageJob,
    _BackendHealth,
)
from tokenspeed.runtime.cache.l3.io_types import StorageIOResult


def config():
    return StorageRuntimeConfig.from_extra_config(
        None, host_blocks=32, prefix_granularity=2
    )


class _HostAllocation:
    """CPU substitute for the GPU-pinned allocation."""

    def __init__(self, data):
        self.host_buffer = data


def executor(backend, *, cfg, data=None):
    host = _HostAllocation(bytearray(64) if data is None else data)
    return L3Executor(
        backend,
        host,
        key_prefix="test",
        rank=0,
        ownership=None,
        config=cfg,
        prefix_for_weight_version=None,
    )


def wait_results(e, count):
    result = []
    deadline = time.monotonic() + 3
    while len(result) < count and time.monotonic() < deadline:
        result.extend(e.poll().completions)
        if len(result) < count:
            time.sleep(0.001)
    assert len(result) == count
    return result


def test_monitor_stale_completions_cannot_reopen_a_failed_recovery():
    now = [1.0]
    monitor = _BackendHealth(
        failure_threshold=1, recovery_interval_s=10, clock=lambda: now[0]
    )
    old = monitor.begin_operation()
    failing = monitor.begin_operation()
    monitor.report_failure(failing)
    monitor.report_success(old)
    assert monitor.begin_operation() is None
    now[0] = 12.0
    probe = monitor.begin_operation()
    assert probe.recovery
    assert monitor.begin_operation() is None
    monitor.report_failure(probe)
    monitor.report_success(probe)  # This completion belongs to the failed epoch.
    assert monitor.begin_operation() is None
    assert monitor.snapshot().state == "open"
    now[0] = 23.0
    recovered = monitor.begin_operation()
    assert recovered.recovery
    assert monitor.begin_operation() is None
    monitor.report_success(recovered)
    assert not monitor.begin_operation().recovery


def test_slow_query_does_not_block_later_worker_or_submit_thread():
    started, release = threading.Event(), threading.Event()

    class Backend(MemoryKvStore):
        def batch_exists(self, keys):
            if keys == ("slow",):
                started.set()
                assert release.wait(3)
            return [True] * len(keys)

    e = executor(Backend(), cfg=config())
    try:
        e.submit(StorageJob(1, "query", ("slow",), (), ()))
        assert started.wait(1)
        e.submit(StorageJob(2, "query", ("fast",), (), ()))
        result = wait_results(e, 1)[0]
        assert result.op_id == 2 and result.success == (True,)
    finally:
        release.set()
        e.close(timeout_s=2)


def test_cancel_queued_get_prevents_native_access():
    started, release = threading.Event(), threading.Event()
    calls = []

    class Backend(MemoryKvStore):
        def batch_get_into_result(self, keys, host_buffer, offsets, sizes):
            calls.append(keys)
            started.set()
            assert release.wait(3)
            return StorageIOResult((True,), True, (False,))

    e = executor(Backend(), cfg=replace(config(), read_workers=1))
    try:
        e.submit(StorageJob(1, "get", ("a",), (0,), (4,)))
        assert started.wait(1)
        e.submit(StorageJob(2, "get", ("b",), (4,), (4,)))
        e.cancel(2)
        cancelled = wait_results(e, 1)[0]
        assert (
            cancelled.op_id == 2
            and cancelled.retired
            and cancelled.success == (False,)
            and cancelled.object_failures == (False,)
        )
        assert calls == [("a",)]  # the only worker is still inside a's I/O
        release.set()
        assert wait_results(e, 1)[0].op_id == 1
    finally:
        release.set()
        e.close(timeout_s=2)


def test_query_exception_degrades_without_killing_worker():
    class Backend(MemoryKvStore):
        def batch_exists(self, keys):
            raise OSError("injected unavailable store")

    e = executor(Backend(), cfg=config())
    try:
        e.submit(StorageJob(1, "query", ("a",), (), ()))
        result = wait_results(e, 1)[0]
        assert result.retired and result.success == (False,)
        e.submit(StorageJob(2, "query", ("b",), (), ()))
        assert wait_results(e, 1)[0].op_id == 2
    finally:
        e.close(timeout_s=2)


@pytest.mark.parametrize("success", [False, True])
def test_unproven_native_failure_disables_io_and_refuses_buffer_shutdown(success):
    class Backend(MemoryKvStore):
        def batch_get_into_result(self, *args):
            return StorageIOResult((success,), False, (False,))

    e = executor(Backend(), cfg=config())
    e.submit(StorageJob(1, "get", ("a",), (0,), (4,)))
    assert not wait_results(e, 1)[0].retired
    assert not e.is_idle()
    assert e.snapshot().health.state == "open"
    e.submit(StorageJob(2, "get", ("b",), (4,), (4,)))
    result = wait_results(e, 1)[0]
    assert result.retired and result.success == (False,)  # no new I/O submitted
    with pytest.raises(RuntimeError, match="retirement is unproven"):
        e.close(timeout_s=2)
    backend_ref = weakref.ref(e.backend)
    host_ref = weakref.ref(e.host_storage)
    executor_ref = weakref.ref(e)
    del e
    gc.collect()
    assert executor_ref() is None
    assert backend_ref() is not None
    assert host_ref() is not None  # Host registration survives owner teardown


def test_global_store_roundtrip_between_independent_host_executors():
    backend = MemoryKvStore()
    writer = executor(backend, cfg=config(), data=bytearray(b"abcdefgh" + b"0" * 56))
    reader = executor(backend, cfg=config())
    try:
        writer.submit(StorageJob(1, "put", ("shared-key",), (0,), (8,)))
        assert wait_results(writer, 1)[0].success == (True,)
        reader.submit(StorageJob(1, "get", ("shared-key",), (16,), (8,)))
        assert wait_results(reader, 1)[0].success == (True,)
        assert reader.host_storage.host_buffer[16:24] == b"abcdefgh"
        assert reader.host_storage.host_buffer[:8] == bytes(8)
    finally:
        writer.close(timeout_s=2)
        reader.close(timeout_s=2)


def test_operation_credits_bound_queue_and_undelivered_results():
    e = executor(
        MemoryKvStore(),
        cfg=StorageRuntimeConfig.from_extra_config(
            '{"async":{"max_operations":1}}', host_blocks=32, prefix_granularity=2
        ),
    )
    try:
        e.submit(StorageJob(1, "query", ("a",), (), ()))
        with pytest.raises(RuntimeError, match="operation credits"):
            e.submit(StorageJob(2, "query", ("b",), (), ()))
        wait_results(e, 1)
    finally:
        e.close(timeout_s=2)


def test_executor_applies_the_parsed_storage_key_limit():
    cfg = StorageRuntimeConfig.from_extra_config(
        '{"async":{"max_keys_per_operation":2}}',
        host_blocks=32,
        prefix_granularity=2,
    )
    e = executor(MemoryKvStore(), cfg=cfg)
    try:
        with pytest.raises(ValueError, match="job descriptor"):
            e.submit(StorageJob(1, "query", ("a", "b", "c"), (), ()))
        e.submit(StorageJob(1, "query", ("a", "b"), (), ()))
        assert wait_results(e, 1)[0].success == (False, False)
    finally:
        e.close(timeout_s=2)


@pytest.mark.parametrize(
    "name",
    [
        "max_operations",
        "max_keys_per_operation",
        "max_host_blocks",
        "metadata_entries",
        "max_prefetch_tokens",
        "min_prefetch_tokens",
    ],
)
def test_storage_limits_reject_values_outside_the_scheduler_integer_range(name):
    with pytest.raises(ValueError, match="int32 limit"):
        StorageRuntimeConfig.from_extra_config(
            json.dumps({"async": {"metadata_entries": 128, name: 2**31}}),
            host_blocks=2**32,
            prefix_granularity=2,
        )


def test_retired_get_miss_does_not_open_backend_circuit():
    e = executor(MemoryKvStore(), cfg=config())
    try:
        e.submit(StorageJob(1, "get", ("evicted",), (0,), (8,)))
        result = wait_results(e, 1)[0]
        assert result.retired and result.success == (False,)
        assert result.object_failures == (True,)
        assert e.snapshot().health.state == "healthy"
    finally:
        e.close(timeout_s=2)


def test_snapshot_accounts_for_cancelled_queue_and_late_completed_bytes():
    started, release = threading.Event(), threading.Event()

    class Backend(MemoryKvStore):
        def batch_get_into_result(self, keys, host_buffer, offsets, sizes):
            started.set()
            assert release.wait(3)
            return StorageIOResult((True,), True, (False,))

    e = executor(Backend(), cfg=replace(config(), read_workers=1))
    try:
        e.submit(StorageJob(1, "get", ("running",), (0,), (8,)))
        assert started.wait(1)
        e.submit(StorageJob(2, "get", ("queued",), (8,), (16,)))
        snapshot = e.snapshot().operations[1]
        assert dict(snapshot.operations) == dict(queued=1, running=1, completed=0)
        assert dict(snapshot.bytes) == dict(queued=16, running=8, completed=0)
        e.cancel(2)
        e.cancel(2)
        e.cancel(1)
        wait_results(e, 1)
        row = e.snapshot().operations[1]
        assert dict(row.events)["cancelled_queued"] == 1
        assert dict(row.events)["cancelled_running"] == 1
        assert dict(row.events)["backend_calls"] == 1
        assert dict(row.operations) == dict(queued=0, running=1, completed=0)
        release.set()
        deadline = time.monotonic() + 3
        while dict(e.snapshot().operations[1].operations)["completed"] != 1:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        row = e.snapshot().operations[1]
        assert dict(row.bytes)["completed"] == 8
        assert row.success_bytes == 8  # I/O success does not imply publication.
        assert row.success_keys == row.failed_keys == 1
        assert row.queue_seconds >= 0 and row.backend_seconds > 0
        wait_results(e, 1)
        assert not any(dict(e.snapshot().operations[1].operations).values())
    finally:
        release.set()
        e.close(timeout_s=2)


def test_snapshot_cannot_consume_recovery_probe_and_rejections_are_visible():
    now = [0.0]
    monitor = _BackendHealth(
        failure_threshold=1, recovery_interval_s=1, clock=lambda: now[0]
    )
    monitor.report_failure(monitor.begin_operation())
    now[0] = 2.0
    for _ in range(3):
        assert monitor.snapshot().state == "open"
    probe = monitor.begin_operation()
    assert probe.recovery and monitor.snapshot().state == "half_open"
    monitor.report_success(probe)
    assert monitor.snapshot().state == "healthy"

    e = executor(MemoryKvStore(), cfg=config())
    try:
        e._health.report_failure(e._health.begin_operation())
        e.submit(StorageJob(1, "put", ("rejected",), (0,), (8,)))
        assert wait_results(e, 1)[0].success == (False,)
        row = e.snapshot().operations[2]
        assert dict(row.events)["rejected"] == 1
        assert dict(row.events)["backend_calls"] == 0
        assert row.success_bytes == row.backend_seconds == 0
    finally:
        e.close(timeout_s=2)


def test_pressure_snapshot_tracks_queue_and_cancelled_native_work_until_retirement():
    now = [1.0]
    started, release = threading.Event(), threading.Event()

    class Backend(MemoryKvStore):
        def batch_get_into_result(self, keys, host_buffer, offsets, sizes):
            started.set()
            assert release.wait(5)
            return StorageIOResult(
                (True,) * len(keys),
                True,
                tuple(not flag for flag in (True,) * len(keys)),
            )

    host = SimpleNamespace(host_buffer=bytearray(64))
    e = L3Executor(
        Backend(),
        host,
        key_prefix="pressure",
        rank=0,
        ownership=None,
        config=replace(config(), read_workers=1),
        clock=lambda: now[0],
        prefix_for_weight_version=None,
    )
    try:
        e.submit(StorageJob(1, "get", ("active",), (0,), (8,)))
        assert started.wait(1)
        e.submit(StorageJob(2, "get", ("queued",), (8,), (16,)))
        now[0] = 1.75
        poll = e.poll()
        assert not poll.completions
        assert (
            poll.queued_operations,
            poll.queued_bytes,
            poll.oldest_operation_us,
        ) == (1, 16, 750000)
        e.cancel(2)
        poll = e.poll()
        assert [result.op_id for result in poll.completions] == [2]
        assert (
            poll.queued_operations,
            poll.queued_bytes,
            poll.oldest_operation_us,
        ) == (0, 0, 750000)
        e.cancel(1)
        assert e.poll().oldest_operation_us == 750000
        assert e.snapshot().health.state == "healthy"
        release.set()
        assert wait_results(e, 1)[0].retired
        poll = e.poll()
        assert (
            poll.queued_operations,
            poll.queued_bytes,
            poll.oldest_operation_us,
        ) == (0, 0, 0)
        assert e.is_idle()
    finally:
        release.set()
        e.close(timeout_s=2)


def test_query_pressure_counts_metadata_jobs_without_fake_payload_bytes():
    now = [1.0]
    started, release = threading.Event(), threading.Event()

    class Backend(MemoryKvStore):
        def batch_exists(self, keys):
            started.set()
            assert release.wait(5)
            return [False] * len(keys)

    host = SimpleNamespace(host_buffer=bytearray(64))
    e = L3Executor(
        Backend(),
        host,
        key_prefix="query-pressure",
        rank=0,
        ownership=None,
        config=replace(config(), query_workers=1),
        clock=lambda: now[0],
        prefix_for_weight_version=None,
    )
    try:
        e.submit(StorageJob(1, "query", ("a",), (), ()))
        assert started.wait(1)
        e.submit(StorageJob(2, "query", ("b",), (), ()))
        now[0] = 2.0
        poll = e.poll()
        assert (
            poll.queued_operations,
            poll.queued_bytes,
            poll.oldest_operation_us,
        ) == (1, 0, 1000000)
        release.set()
        assert all(result.retired for result in wait_results(e, 2))
        assert e.poll().oldest_operation_us == 0
    finally:
        release.set()
        e.close(timeout_s=2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"queue_low_operations": 8},
        {"queue_high_bytes": 128 * 1024 * 1024},
        {"operation_low_age_ms": 500},
        {"queue_low_operations": -1},
    ],
)
def test_pressure_config_rejects_invalid_hysteresis(overrides):
    import json

    with pytest.raises(ValueError):
        StorageRuntimeConfig.from_extra_config(
            json.dumps({"async": overrides}), host_blocks=32, prefix_granularity=2
        )


def test_pressure_config_one_credit_can_recover_on_an_empty_queue():
    cfg = StorageRuntimeConfig.from_extra_config(
        '{"async":{"max_operations":1}}', host_blocks=32, prefix_granularity=2
    )
    assert (
        cfg.scheduler.pressure_high.queued_operations == 1
        and cfg.scheduler.pressure_low.queued_operations == 0
    )


@pytest.mark.parametrize("retired", ["pending", 1])
def test_malformed_retirement_cannot_publish_or_release_native_buffers(retired):
    class Backend(MemoryKvStore):
        def __init__(self):
            super().__init__()
            self.failures_retire = False

        def batch_get_into_result(self, *args):
            return StorageIOResult((True,), retired, (False,))

    e = executor(Backend(), cfg=config())
    e.submit(StorageJob(1, "get", ("unproven",), (0,), (8,)))
    result = wait_results(e, 1)[0]
    # The error carries no proof that native code stopped accessing Host memory.
    assert result.retired is False
    assert result.success == (False,)
    assert e.snapshot().unsafe_io
    with pytest.raises(RuntimeError, match="retirement is unproven"):
        e.close(timeout_s=2)


@pytest.mark.parametrize("failure", ["health", "queue", "exception"])
def test_get_rejection_or_transport_error_has_no_object_failure(failure):
    backend = MemoryKvStore()
    backend.batch_put_from(("present",), bytearray(b"data"), (0,), (4,))
    e = executor(backend, cfg=config())
    try:
        if failure == "health":
            e._health.report_failure(e._health.begin_operation())
            e.submit(StorageJob(1, "get", ("present",), (0,), (4,)))
        elif failure == "queue":
            with patch.object(e._queues["get"], "put_nowait", side_effect=queue.Full):
                e.submit(StorageJob(1, "get", ("present",), (0,), (4,)))
        else:
            with patch.object(
                backend, "batch_get_into_result", side_effect=OSError("unavailable")
            ):
                e.submit(StorageJob(1, "get", ("present",), (0,), (4,)))
                result = wait_results(e, 1)[0]
        if failure != "exception":
            result = wait_results(e, 1)[0]
            row = next(row for row in e.snapshot().operations if row.kind == "get")
            assert dict(row.events)["backend_calls"] == 0
        assert result.retired and result.success == (False,)
        assert result.object_failures == (False,)
        assert backend.batch_exists(("present",)) == [True]
    finally:
        e.close(timeout_s=2)


@pytest.mark.parametrize("object_failures", [(), (1,), (True,)])
def test_malformed_object_failure_facts_cannot_publish(object_failures):
    class Backend(MemoryKvStore):
        def batch_get_into_result(self, *args):
            return StorageIOResult((True,), True, object_failures)

    e = executor(Backend(), cfg=config())
    try:
        e.submit(StorageJob(1, "get", ("a",), (0,), (4,)))
        result = wait_results(e, 1)[0]
        assert result.success == (False,)
        assert result.object_failures == (False,)
        assert e.snapshot().health.state == "open"
    finally:
        e.close(timeout_s=2)


def test_close_waits_for_native_retirement_and_preserves_host_owner():
    started, release = threading.Event(), threading.Event()

    class Backend(MemoryKvStore):
        def batch_get_into_result(self, *args):
            started.set()
            assert release.wait(3)
            return super().batch_get_into_result(*args)

    backend = Backend()
    host = SimpleNamespace(host_buffer=bytearray(b"abcdefgh"))
    backend.batch_put_from(("a",), host.host_buffer, (0,), (8,))
    e = L3Executor(
        backend,
        host,
        key_prefix="test",
        rank=0,
        ownership=None,
        config=config(),
        prefix_for_weight_version=None,
    )
    del host  # L2's resource reference can disappear independently.
    try:
        with patch.object(backend, "close", wraps=backend.close) as close:
            e.submit(StorageJob(1, "get", ("a",), (0,), (8,)))
            assert started.wait(1)
            # The serving poll is independent of GPU/device submission.
            assert not e.poll().completions
            assert e.host_storage.host_buffer == b"abcdefgh"
            release.set()
            assert wait_results(e, 1)[0].success == (True,)
            e.close(timeout_s=2)
            e.close(timeout_s=2)
            close.assert_called_once_with()
    finally:
        release.set()
        e.close(timeout_s=2)


def test_close_timeout_preserves_backend_and_host_after_worker_exit():
    started, release = threading.Event(), threading.Event()

    class Backend(MemoryKvStore):
        def batch_get_into_result(self, *args):
            started.set()
            assert release.wait(3)
            return super().batch_get_into_result(*args)

    e = executor(Backend(), cfg=config())
    e.backend.batch_put_from(("a",), bytearray(b"data"), (0,), (4,))
    backend_ref = weakref.ref(e.backend)
    host_ref = weakref.ref(e.host_storage)
    executor_ref = weakref.ref(e)
    threads = tuple(e._threads)
    try:
        e.submit(StorageJob(1, "get", ("a",), (0,), (4,)))
        assert started.wait(1)
        with pytest.raises(RuntimeError, match="retirement is unproven"):
            e.close(timeout_s=0)
        assert backend_ref().batch_exists(("a",)) == [True]
        del e
    finally:
        release.set()
        for thread in threads:
            thread.join(3)
    gc.collect()
    assert executor_ref() is None
    assert backend_ref() is not None and host_ref() is not None
    assert host_ref().host_buffer[:4] == b"data"


def test_close_never_deregisters_unknown_native_access():
    class Backend(MemoryKvStore):
        def batch_get_into_result(self, *args):
            return StorageIOResult((False,), False, (False,))

    backend = Backend()
    e = executor(backend, cfg=config())
    with patch.object(backend, "close") as close:
        e.submit(StorageJob(1, "get", ("a",), (0,), (8,)))
        assert not wait_results(e, 1)[0].retired
        with pytest.raises(RuntimeError, match="retirement is unproven"):
            e.close(timeout_s=2)
        close.assert_not_called()


@pytest.mark.parametrize("offsets,sizes", [((0,), ()), ((), (4,)), ((0,), (4,))])
def test_query_rejects_host_ranges_without_consuming_an_operation_id(offsets, sizes):
    e = executor(MemoryKvStore(), cfg=config())
    try:
        with pytest.raises(ValueError, match="queries cannot carry"):
            e.submit(StorageJob(1, "query", ("missing",), offsets, sizes))
        e.submit(StorageJob(1, "query", ("missing",), (), ()))
        result = wait_results(e, 1)[0]
        assert result.op_id == 1 and result.retired and result.success == (False,)
    finally:
        e.close(timeout_s=2)
