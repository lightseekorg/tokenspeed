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

"""Async L3 control feedback and immutable Host operation snapshots."""

import threading
import time
from test.l3_test_utils import storage_ops

import pytest
import tokenspeed_scheduler as ts
from prometheus_client import CollectorRegistry

from tokenspeed.runtime.cache.l3.backend import MemoryKvStore
from tokenspeed.runtime.cache.l3.config import StorageRuntimeConfig
from tokenspeed.runtime.cache.l3.executor import L3Executor, StorageJob
from tokenspeed.runtime.cache.l3.metrics import L3Metrics
from tokenspeed.runtime.engine.l3_cache_hooks import L3CacheHooks


class _Host:
    """CPU allocation in place of the pinned allocation used for GPU copies."""

    def __init__(self):
        self.host_buffer = bytearray(256)
        self.block_bytes = 8

    def host_block_range(self, group, block):
        assert group == 0
        return (block - 1) * self.block_bytes, self.block_bytes


def test_disabled_storage_admits_without_owner_agreement_or_storage_events():
    config = ts.SchedulerConfig()
    config.prefix_granularity = 2
    config.num_snapshot_pages = 1
    config.num_device_pages = 32
    config.num_host_pages = 32
    config.max_batch_size = 4
    config.max_scheduled_tokens = 16
    config.cache_groups = [ts.CacheGroupConfig("kv", 2, 32, ts.AttnKind.Full)]
    scheduler = ts.Scheduler(config)
    hooks = L3CacheHooks(
        scheduler,
        None,
        config=None,
        attn_tp_size=2,
        attn_tp_cpu_group=None,
        pp_size=1,
        pp_cpu_group=None,
        metrics=None,
    )
    request = ts.RequestSpec()
    request.request_id = "plain"
    request.tokens = [1, 2, 3, 4]
    scheduler.submit_requests([request])
    assert hooks.poll_ready_events(allow_forward=True, allow_storage=True) == []
    hooks.record_metrics()
    assert not hooks.cancel_request("plain")
    plan = scheduler.next_execution_plan()
    assert [rid for batch in plan.forward for rid in batch.request_ids] == ["plain"]
    assert list(plan.forward[0].extend_prefix_lens) == [0]
    assert not storage_ops(plan)


class _ControlledMemoryStore(MemoryKvStore):
    """Real memory reads with a controllable transport stall or failure."""

    def __init__(self):
        super().__init__()
        self.read_started = threading.Event()
        self.release_reads = threading.Event()
        self.release_reads.set()
        self.fail_next_read = False

    def batch_get_into_result(self, keys, host_buffer, offsets, sizes):
        self.read_started.set()
        if not self.release_reads.wait(5):
            raise TimeoutError("Test did not release the storage read")
        if self.fail_next_read:
            self.fail_next_read = False
            raise OSError("Storage transport failed before writing Host bytes")
        return super().batch_get_into_result(keys, host_buffer, offsets, sizes)


@pytest.fixture
def runtime():
    config = StorageRuntimeConfig.from_extra_config(
        '{"async":{"read_workers":1,"max_operations":2}}',
        host_blocks=31,
        prefix_granularity=2,
    )
    host = _Host()
    backend = _ControlledMemoryStore()
    now = [time.monotonic()]
    executor = L3Executor(
        backend,
        host,
        key_prefix="old",
        rank=0,
        ownership=None,
        config=config,
        prefix_for_weight_version=lambda version: f"new-{version}",
        clock=lambda: now[0],
    )
    try:
        yield config, host, backend, executor, now
    finally:
        backend.release_reads.set()
        executor.close(timeout_s=2)


def _scheduler(config):
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 2
    cfg.num_snapshot_pages = 1
    cfg.num_device_pages = 32
    cfg.num_host_pages = 32
    cfg.max_batch_size = 4
    cfg.max_scheduled_tokens = 16
    cfg.enable_l3_storage = True
    cfg.storage_config = config.scheduler
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="kv",
            block_granularity=2,
            total_pages=32,
            attn_kind=ts.AttnKind.Full,
        )
    ]
    scheduler = ts.Scheduler(cfg)
    _send(scheduler, ts.Cache.StorageTick(1, True, True, ts.StoragePressure(0, 0, 0)))
    return scheduler


def _hooks(scheduler, executor, config, metrics):
    return L3CacheHooks(
        scheduler,
        executor,
        config=config.scheduler,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pp_size=1,
        pp_cpu_group=None,
        metrics=metrics,
    )


def _send(scheduler, event):
    message = ts.ExecutionEvent()
    message.add_event(event)
    scheduler.advance(message)


def _query(scheduler, request_id):
    request = ts.RequestSpec()
    request.request_id = request_id
    request.tokens = list(range(8))
    request.max_new_tokens = 2
    scheduler.submit_requests([request])
    operation = storage_ops(scheduler.next_execution_plan())[0]
    assert operation.kind == ts.StorageKind.Query
    return operation


def _keys(executor, operation):
    return [executor._object_key(h, g, o) for g, h, o in operation.keys]


def _seed(backend, keys, payload):
    assert backend.batch_put_from(
        keys,
        bytearray(payload * len(keys)),
        tuple(8 * i for i in range(len(keys))),
        (8,) * len(keys),
    ) == [True] * len(keys)


def _wait_until(predicate):
    deadline = time.perf_counter() + 3
    while not predicate():
        assert time.perf_counter() < deadline, "Storage worker did not finish"
        time.sleep(0.001)


def _completions(executor, count):
    results = []

    def finished():
        results.extend(executor.poll().completions)
        return len(results) >= count

    _wait_until(finished)
    assert len(results) == count
    return results


def _finish(scheduler, result):
    _send(
        scheduler,
        ts.Cache.StorageDone(
            result.op_id,
            list(result.success),
            result.retired,
            list(result.object_failures),
        ),
    )


def _get_descriptor(config, backend, executor):
    scheduler = _scheduler(config)
    query = _query(scheduler, "snapshot")
    keys = _keys(executor, query)
    _seed(backend, keys, b"snapshot")
    _send(
        scheduler,
        ts.Cache.StorageDone(
            query.op_id, backend.batch_exists(keys), True, [False] * len(keys)
        ),
    )
    operation = storage_ops(scheduler.next_execution_plan())[0]
    assert operation.kind == ts.StorageKind.Get
    return scheduler, operation, keys


def test_parsed_storage_deadline_controls_scheduler_admission():
    config = StorageRuntimeConfig.from_extra_config(
        '{"async":{"wait_timeout_ms":7,"metadata_entries":124}}',
        host_blocks=31,
        prefix_granularity=2,
    )
    scheduler = _scheduler(config)
    query = _query(scheduler, "deadline")
    _send(
        scheduler,
        ts.Cache.StorageTick(7000, True, True, ts.StoragePressure(0, 0, 0)),
    )
    assert not any(
        batch.request_ids for batch in scheduler.next_execution_plan().forward
    )
    _send(
        scheduler,
        ts.Cache.StorageTick(7002, True, True, ts.StoragePressure(0, 0, 0)),
    )
    plan = scheduler.next_execution_plan()
    assert [rid for batch in plan.forward for rid in batch.request_ids] == ["deadline"]
    assert scheduler.storage_stats().wait_timeouts == 1
    cancellations = storage_ops(plan)
    assert len(cancellations) == 1
    assert cancellations[0].kind == ts.StorageKind.Cancel
    assert cancellations[0].op_id == query.op_id


def test_local_completion_is_reported_once_and_does_not_select_a_prefix(runtime):
    config, host, backend, executor, _ = runtime
    scheduler = _scheduler(config)
    h = _hooks(scheduler, executor, config, None)
    _seed(backend, ["present"], b"restored")
    executor.submit(StorageJob(7, "get", ("present", "missing"), (0, 8), (8, 8)))
    events = []

    def reported():
        events.extend(h.poll_ready_events(allow_forward=True, allow_storage=True)[1:])
        return bool(events)

    _wait_until(reported)
    assert len(events) == 1
    assert events[0].op_id == 7
    assert list(events[0].success) == [1, 0]
    assert list(events[0].object_failures) == [0, 1]
    assert host.host_buffer[:16] == b"restored" + bytes(8)
    assert len(h.poll_ready_events(allow_forward=False, allow_storage=False)) == 1
    assert not scheduler.storage_pending()
    assert all(
        not batch.request_ids for batch in scheduler.next_execution_plan().forward
    )


def test_storage_ranges_and_namespace_are_captured_before_queueing(runtime):
    config, host, backend, executor, _ = runtime
    _, operation, keys = _get_descriptor(config, backend, executor)
    ranges = [
        host.host_block_range(g, b)
        for (g, _, _), b in zip(operation.keys, operation.host_blocks, strict=True)
    ]
    _seed(backend, ["blocker"], b"blocking")
    backend.release_reads.clear()
    executor.submit(StorageJob(1, "get", ("blocker",), (240,), (8,)))
    assert backend.read_started.wait(2)
    executor.submit_operations([operation])
    # Bypass the management guard only to expose descriptor lifetime: workers
    # must consume the namespace and byte ranges captured at submission.
    executor._key_prefix = "changed"
    host.block_bytes = 16
    _seed(backend, _keys(executor, operation), b"wrong!!!")
    backend.release_reads.set()
    results = _completions(executor, 2)
    assert all(result.retired and all(result.success) for result in results)
    for offset, size in ranges:
        assert host.host_buffer[offset : offset + size] == b"snapshot"
    assert backend.batch_exists(keys) == [True] * len(keys)
    executor.set_weight_version("v2")
    assert executor.key_prefix == "new-v2"


def test_cancelled_queued_scheduler_get_retires_without_writing_host(runtime):
    config, host, backend, executor, _ = runtime
    scheduler, operation, _ = _get_descriptor(config, backend, executor)
    _seed(backend, ["blocker"], b"blocking")
    backend.release_reads.clear()
    executor.submit(StorageJob(1, "get", ("blocker",), (240,), (8,)))
    assert backend.read_started.wait(2)
    executor.submit_operations([operation])
    aborted = ts.ForwardEvent.Abort()
    aborted.request_id = "snapshot"
    _send(scheduler, aborted)
    cancellations = storage_ops(scheduler.next_execution_plan())
    assert len(cancellations) == 1 and cancellations[0].kind == ts.StorageKind.Cancel
    executor.submit_operations(cancellations)
    result = _completions(executor, 1)[0]
    assert result.op_id == operation.op_id and result.retired
    assert not any(result.success) and not any(result.object_failures)
    assert host.host_buffer[:240] == bytes(240)
    backend.release_reads.set()
    assert _completions(executor, 1)[0].op_id == 1
    assert host.host_buffer[:240] == bytes(240)


def test_expired_queued_get_retires_while_active_get_keeps_host_references(runtime):
    config, host, backend, executor, _ = runtime
    scheduler, active, _ = _get_descriptor(config, backend, executor)
    backend.release_reads.clear()
    executor.submit_operations([active])
    assert backend.read_started.wait(2)

    query = _query(scheduler, "queued")
    keys = _keys(executor, query)
    _send(
        scheduler,
        ts.Cache.StorageDone(
            query.op_id, backend.batch_exists(keys), True, [False] * len(keys)
        ),
    )
    queued = storage_ops(scheduler.next_execution_plan())[0]
    assert queued.kind == ts.StorageKind.Get
    executor.submit_operations([queued])

    # Both deadlines include queue wait. Expiration permits ordinary admission
    # while the active native call still owns its destinations.
    _send(
        scheduler,
        ts.Cache.StorageTick(
            config.scheduler.wait_timeout_us + 2,
            True,
            True,
            ts.StoragePressure(0, 0, 0),
        ),
    )
    plan = scheduler.next_execution_plan()
    assert {rid for batch in plan.forward for rid in batch.request_ids} == {
        "snapshot",
        "queued",
    }
    assert all(
        prefix == 0 for batch in plan.forward for prefix in batch.extend_prefix_lens
    )
    assert scheduler.storage_stats().wait_timeouts == 2
    cancellations = storage_ops(plan)
    assert {op.op_id for op in cancellations} == {active.op_id, queued.op_id}
    assert all(op.kind == ts.StorageKind.Cancel for op in cancellations)
    executor.submit_operations(cancellations)

    result = _completions(executor, 1)[0]
    assert result.op_id == queued.op_id and result.retired
    assert not any(result.success) and not any(result.object_failures)
    _finish(scheduler, result)
    assert not executor.poll().completions  # Repeated polling cannot replay the ACK.
    assert scheduler.storage_held_blocks() == len(active.host_blocks)
    assert scheduler.storage_pending()
    assert executor.delete_namespace() is False
    with pytest.raises(RuntimeError, match="must retire"):
        executor.set_weight_version("v2")
    assert host.host_buffer == bytes(len(host.host_buffer))

    backend.release_reads.set()
    result = _completions(executor, 1)[0]
    assert result.op_id == active.op_id and result.retired and all(result.success)
    _finish(scheduler, result)
    assert not scheduler.storage_pending()
    assert scheduler.storage_held_blocks() == 0
    assert scheduler.host_pool_cached_blocks() == 0  # Detached reads cannot publish.
    for block in active.host_blocks:
        offset, size = host.host_block_range(0, block)
        assert host.host_buffer[offset : offset + size] == b"snapshot"
    for block in queued.host_blocks:
        offset, size = host.host_block_range(0, block)
        assert host.host_buffer[offset : offset + size] == bytes(size)
    executor.set_weight_version("v2")


@pytest.mark.parametrize("phase", ["running", "ready", "detached"])
def test_reused_warm_host_address_preserves_get_results(phase):
    config = StorageRuntimeConfig.from_extra_config(
        '{"async":{"read_workers":1,"max_operations":2,"max_host_blocks":2}}',
        host_blocks=2,
        prefix_granularity=2,
    )
    host = _Host()
    backend = _ControlledMemoryStore()
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 2
    cfg.num_snapshot_pages = 1
    cfg.num_device_pages = 32
    cfg.num_host_pages = 3  # Two usable Host slots: one warm and one GET destination.
    cfg.max_batch_size = 4
    cfg.max_scheduled_tokens = 8
    cfg.enable_l3_storage = True
    cfg.storage_config = config.scheduler
    cfg.cache_groups = [ts.CacheGroupConfig("kv", 2, 32, ts.AttnKind.Full)]
    scheduler = ts.Scheduler(cfg)
    _send(scheduler, ts.Cache.StorageTick(1, False, True, ts.StoragePressure(0, 0, 0)))
    executor = L3Executor(
        backend,
        host,
        key_prefix="warm-lifetime",
        rank=0,
        ownership=None,
        config=config,
        prefix_for_weight_version=None,
    )

    def prepare_read(request_id, tokens, bound, payload):
        spec = ts.RequestSpec()
        spec.request_id = request_id
        spec.tokens = tokens
        spec.max_cached_prefix_tokens = bound
        scheduler.submit_requests([spec])
        operations = storage_ops(scheduler.next_execution_plan())
        assert len(operations) == 1 and operations[0].kind == ts.StorageKind.Query
        query = operations[0]
        _seed(backend, _keys(executor, query), payload)
        executor.submit_operations([query])
        _finish(scheduler, _completions(executor, 1)[0])
        operations = storage_ops(scheduler.next_execution_plan())
        assert len(operations) == 1 and operations[0].kind == ts.StorageKind.Get
        return operations[0]

    try:
        seed = prepare_read("seed", [0, 1, 2], 2, b"warmdata")
        executor.submit_operations([seed])
        _finish(scheduler, _completions(executor, 1)[0])
        assert not any(
            batch.request_ids for batch in scheduler.next_execution_plan().forward
        )
        abort = ts.ForwardEvent.Abort()
        abort.request_id = "seed"
        _send(scheduler, abort)
        scheduler.next_execution_plan()
        assert scheduler.storage_held_blocks() == 0
        assert scheduler.host_pool_cached_blocks() == 1

        active = prepare_read("active", [0, 1, 2, 3, 4], 4, b"readback")
        backend.read_started.clear()
        backend.release_reads.clear()
        executor.submit_operations([active])
        assert backend.read_started.wait(2)
        assert scheduler.storage_held_blocks() == len(active.host_blocks)
        if phase == "ready":
            backend.release_reads.set()
            _finish(scheduler, _completions(executor, 1)[0])
            scheduler.next_execution_plan()
            assert scheduler.storage_stats().ready_requests == 1

        # A second real reservation must evict the old warm entry and reuse
        # its address while the first GET or Ready hold owns its destination.
        replacement = prepare_read("replacement", [10, 11, 12], 2, b"newowner")
        assert list(replacement.host_blocks) == list(seed.host_blocks)
        warm_offset, size = host.host_block_range(0, replacement.host_blocks[0])
        host.host_buffer[warm_offset : warm_offset + size] = b"newowner"
        if phase == "detached":
            abort = ts.ForwardEvent.Abort()
            abort.request_id = "active"
            _send(scheduler, abort)
            executor.submit_operations(storage_ops(scheduler.next_execution_plan()))

        if phase != "ready":
            backend.release_reads.set()
            result = _completions(executor, 1)[0]
            assert (
                result.op_id == active.op_id and result.retired and all(result.success)
            )
            _finish(scheduler, result)
        assert host.host_buffer[warm_offset : warm_offset + size] == b"newowner"
        offset, size = host.host_block_range(0, active.host_blocks[0])
        assert host.host_buffer[offset : offset + size] == b"readback"
        _send(
            scheduler, ts.Cache.StorageTick(2, True, True, ts.StoragePressure(0, 0, 0))
        )
        plan = scheduler.next_execution_plan()
        if phase == "detached":
            assert not any(batch.request_ids for batch in plan.forward)
        else:
            assert list(plan.forward[0].request_ids) == ["active"]
            assert list(plan.forward[0].extend_prefix_lens) == [0]

        executor.submit_operations([replacement])
        _finish(scheduler, _completions(executor, 1)[0])
        plan = scheduler.next_execution_plan()
        batch = next(
            batch for batch in plan.forward if "replacement" in batch.request_ids
        )
        index = list(batch.request_ids).index("replacement")
        assert batch.extend_prefix_lens[index] == 2
        assert scheduler.storage_held_blocks() == 0
        assert (
            scheduler.host_pool_pinned_blocks() > 0
        )  # Ordinary H2D owns the new prefix.
        for operation in plan.cache:
            if isinstance(operation, ts.Cache.LoadBackOp):
                for op_id in operation.op_ids:
                    _send(scheduler, ts.Cache.LoadBackDoneEvent(op_id))
        assert scheduler.storage_quarantined() == 0
    finally:
        backend.release_reads.set()
        executor.close(timeout_s=2)


def test_namespace_mutation_rejects_inflight_io_without_waiting(runtime):
    _, host, backend, executor, _ = runtime
    old_key = executor._object_key("hash", 0, 0)
    _seed(backend, [old_key, "other_hash"], b"original")
    backend.release_reads.clear()
    executor.submit(StorageJob(1, "get", (old_key,), (0,), (8,)))
    assert backend.read_started.wait(2)
    started = time.perf_counter()
    assert executor.delete_namespace() is False
    with pytest.raises(RuntimeError, match="must retire"):
        executor.set_weight_version("v2")
    assert time.perf_counter() - started < 1
    assert not backend.release_reads.is_set()
    assert executor.key_prefix == "old"
    assert backend.batch_exists([old_key, "other_hash"]) == [True, True]
    assert host.host_buffer[:8] == bytes(8)
    backend.release_reads.set()
    assert _completions(executor, 1)[0].success == (True,)
    assert host.host_buffer[:8] == b"original"
    assert executor.delete_namespace()
    assert backend.batch_exists([old_key, "other_hash"]) == [False, True]
    executor.set_weight_version("v2")
    assert executor.key_prefix == "new-v2"
    new_key = executor._object_key("hash", 0, 0)
    _seed(backend, [new_key], b"newbytes")
    assert executor.delete_namespace()
    assert backend.batch_exists([new_key, "other_hash"]) == [False, True]


def test_metrics_sample_pending_transitions_without_extra_feedback(
    runtime, monkeypatch
):
    import tokenspeed.runtime.engine.l3_cache_hooks as module

    config, _, _, executor, _ = runtime
    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    registry = CollectorRegistry()
    metrics = L3Metrics(
        labels={},
        enabled=True,
        reporters=["prometheus"],
        event_sink=None,
        registry=registry,
    )
    scheduler = _scheduler(config)
    h = _hooks(scheduler, executor, config, metrics)

    def value(metric, **labels):
        return registry.get_sample_value(f"tokenspeed:l3_{metric}", labels)

    def completed(kind):
        row = next(row for row in executor.snapshot().operations if row.kind == kind)
        return dict(row.events)["completed"]

    h.record_metrics()
    assert value("scheduler_resources", resource="operations") == 0
    query = _query(scheduler, "metrics")
    executor.submit_operations([query])
    _wait_until(lambda: completed("query") == 1)
    h.record_metrics()
    assert value("scheduler_resources", resource="waiting_requests") == 1
    assert value("io_events_total", kind="query", event="completed") == 1
    assert scheduler.storage_pending() and not executor.is_idle()

    # An unchanged pending state is throttled even when new I/O has completed.
    executor.submit(StorageJob(2, "put", ("metrics-object",), (0,), (8,)))
    _wait_until(lambda: completed("put") == 1)
    h.record_metrics()
    assert value("io_events_total", kind="put", event="completed") is None
    now[0] += 1
    h.record_metrics()
    assert value("io_events_total", kind="put", event="completed") == 1

    for result in _completions(executor, 2):
        if result.op_id == query.op_id:
            _finish(scheduler, result)
    plan = scheduler.next_execution_plan()
    assert list(plan.forward[0].extend_prefix_lens) == [0]
    assert not scheduler.storage_pending()
    # Returning to idle is visible immediately, without another second passing.
    h.record_metrics()
    assert value("scheduler_resources", resource="operations") == 0
    assert value("io_events_total", kind="query", event="completed") == 1


def test_full_read_queue_rejects_without_invalidating_and_recovers(runtime):
    _, host, backend, executor, _ = runtime
    _seed(backend, ["blocker", "retry"], b"restored")
    backend.release_reads.clear()
    executor.submit(StorageJob(1, "get", ("blocker",), (240,), (8,)))
    assert backend.read_started.wait(2)
    # Cancelled descriptors keep occupying the bounded queue until its worker
    # returns, even after polling has returned their operation credits.
    for op_id in (2, 3):
        executor.submit(StorageJob(op_id, "get", ("retry",), ((op_id - 1) * 8,), (8,)))
        executor.cancel(op_id)
        result = _completions(executor, 1)[0]
        assert result.op_id == op_id and result.retired
        assert result.success == (False,) and result.object_failures == (False,)
    executor.submit(StorageJob(4, "get", ("retry",), (24,), (8,)))
    result = _completions(executor, 1)[0]
    assert result.op_id == 4 and result.retired
    assert result.success == (False,) and result.object_failures == (False,)
    assert host.host_buffer[:32] == bytes(32)
    row = next(row for row in executor.snapshot().operations if row.kind == "get")
    assert dict(row.events)["rejected"] == 1
    assert dict(row.events)["backend_calls"] == 1
    backend.release_reads.set()
    assert _completions(executor, 1)[0].success == (True,)
    _wait_until(lambda: executor._queues["get"].empty())
    executor.submit(StorageJob(5, "get", ("retry",), (32,), (8,)))
    assert _completions(executor, 1)[0].success == (True,)
    assert host.host_buffer[8:32] == bytes(24)
    assert host.host_buffer[32:40] == b"restored"
    assert executor.snapshot().health.state == "healthy"


def test_transient_get_failure_retries_same_remote_keys_after_recovery(runtime):
    config, host, backend, executor, now = runtime
    scheduler = _scheduler(config)
    query = _query(scheduler, "first")
    keys = _keys(executor, query)
    _seed(backend, keys, b"abcdefgh")
    executor.submit_operations([query])
    _finish(scheduler, _completions(executor, 1)[0])
    read = storage_ops(scheduler.next_execution_plan())[0]
    assert read.kind == ts.StorageKind.Get
    backend.fail_next_read = True
    executor.submit_operations([read])
    result = _completions(executor, 1)[0]
    assert result.retired and not any(result.object_failures)
    _finish(scheduler, result)
    assert scheduler.storage_negative_keys() == 0
    assert executor.snapshot().health.state == "open"
    plan = scheduler.next_execution_plan()
    assert list(plan.forward[0].extend_prefix_lens) == [0]
    finished = ts.ForwardEvent.Abort()
    finished.request_id = "first"
    _send(scheduler, finished)
    now[0] += 100  # Permit the monitor's single recovery probe.
    query2 = _query(scheduler, "after-recovery")
    assert _keys(executor, query2) == keys
    executor.submit_operations([query2])
    _finish(scheduler, _completions(executor, 1)[0])
    assert executor.snapshot().health.state == "healthy"
    read2 = storage_ops(scheduler.next_execution_plan())[0]
    assert read2.keys == read.keys
    executor.submit_operations([read2])
    _finish(scheduler, _completions(executor, 1)[0])
    plan = scheduler.next_execution_plan()
    assert list(plan.forward[0].extend_prefix_lens) == [7]
    assert all(
        host.host_buffer[(block - 1) * 8 : block * 8] == b"abcdefgh"
        for block in read2.host_blocks
    )
    assert scheduler.storage_negative_keys() == 0
    assert scheduler.storage_quarantined() == 0
