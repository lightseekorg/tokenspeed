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

import pytest
import tokenspeed_scheduler as ts
from conftest import _advance, _spec, storage_ops


def scheduler(*, role, groups=1, host=32, timeout=100):
    cfg = ts.SchedulerConfig()
    cfg.role = role
    cfg.prefix_granularity = 2
    cfg.num_snapshot_pages = 1
    cfg.num_device_pages = 64
    cfg.num_host_pages = host
    cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteThrough
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.enable_l3_storage = True
    cfg.storage_config = ts.StorageConfig(
        8,
        32,
        max(1, (host - 1) // 2),
        4,
        32,
        2,
        timeout,
        100,
        ts.StoragePressure(8, 1048576, 1000),
        ts.StoragePressure(4, 524288, 500),
    )
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id=f"g{i}",
            block_granularity=2,
            total_pages=64,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
            attn_kind=ts.AttnKind.Full,
        )
        for i in range(groups)
    ]
    cfg.host_group_packing = [1] * groups
    result = ts.Scheduler(cfg)
    tick(result, 1, forward=True, storage=True)
    return result


def event(s, item):
    e = ts.ExecutionEvent()
    e.add_event(item)
    s.advance(e)


def tick(s, now, *, forward, storage):
    event(s, ts.Cache.StorageTick(now, forward, storage, ts.StoragePressure(0, 0, 0)))


def done(s, op, flags, *, retired):
    event(
        s,
        ts.Cache.StorageDone(
            op.op_id,
            flags,
            retired,
            [
                int(not flag and retired and op.kind == ts.StorageKind.Get)
                for flag in flags
            ],
        ),
    )


def requests(plan):
    return [rid for batch in plan.forward for rid in batch.request_ids]


def read_pending(s, rid="r"):
    spec = _spec(rid, list(range(8)))
    spec.max_cached_prefix_tokens = 6
    s.submit_requests([spec])
    q = storage_ops(s.next_execution_plan())[0]
    assert q.kind == ts.StorageKind.Query
    done(s, q, [1] * len(q.keys), retired=True)
    read = storage_ops(s.next_execution_plan())[0]
    assert read.kind == ts.StorageKind.Get
    return read


def test_get_holds_only_host_then_uses_normal_loadback():
    s = scheduler(role=ts.SchedulerConfig.Role.Fused)
    before = s.available_lcm_blocks()
    read = read_pending(s)
    assert s.available_lcm_blocks() == before
    assert s.storage_held_blocks() == len(read.keys)
    done(s, read, [1] * len(read.keys), retired=True)
    plan = s.next_execution_plan()
    assert requests(plan) == ["r"]
    assert list(plan.forward[0].extend_prefix_lens) == [6]
    load = next(op for op in plan.cache if isinstance(op, ts.Cache.LoadBackOp))
    assert sum(len(row) for row in load.src_pages) == len(read.keys)
    assert s.storage_held_blocks() == 0
    assert s.host_pool_pinned_blocks() > 0  # ordinary H2D owns the handoff
    event(s, ts.Cache.LoadBackDoneEvent(load.op_ids[0]))


def test_soft_timeout_does_not_free_native_destination():
    s = scheduler(role=ts.SchedulerConfig.Role.Fused)
    read = read_pending(s)
    held = s.storage_held_blocks()
    tick(s, 102, forward=True, storage=True)
    plan = s.next_execution_plan()
    assert requests(plan) == ["r"]
    assert list(plan.forward[0].extend_prefix_lens) == [0]
    assert any(op.kind == ts.StorageKind.Cancel for op in storage_ops(plan))
    assert s.storage_held_blocks() == held
    done(s, read, [1] * len(read.keys), retired=True)
    assert s.storage_held_blocks() == 0
    assert (
        s.host_pool_cached_blocks() == 0
    )  # detached data never resurrects the request


def test_unknown_retirement_quarantines_instead_of_reusing_blocks():
    s = scheduler(role=ts.SchedulerConfig.Role.Fused)
    read = read_pending(s)
    done(s, read, [0] * len(read.keys), retired=False)
    plan = s.next_execution_plan()
    assert requests(plan) == ["r"]
    assert s.storage_quarantined() == 1
    assert s.storage_held_blocks() == len(read.keys)
    assert not s.can_clear_cache()


def test_short_read_is_not_published_and_negative_metadata_is_bounded():
    s = scheduler(role=ts.SchedulerConfig.Role.Fused, groups=2)
    read = read_pending(s)
    done(s, read, [0] * len(read.keys), retired=True)
    assert s.storage_negative_keys() == 4
    plan = s.next_execution_plan()
    assert requests(plan) == ["r"]
    assert list(plan.forward[0].extend_prefix_lens) == [0]
    assert s.storage_held_blocks() == 0


def test_later_query_can_progress_while_earlier_query_is_pending():
    s = scheduler(role=ts.SchedulerConfig.Role.Fused)
    s.submit_requests([_spec("a", list(range(8))), _spec("b", list(range(10, 18)))])
    queries = storage_ops(s.next_execution_plan())
    assert len(queries) == 2
    done(s, queries[1], [0] * len(queries[1].keys), retired=True)
    assert requests(s.next_execution_plan()) == ["b"]
    assert s.storage_pending()  # a is still querying


@pytest.mark.parametrize("phase", [ts.StorageKind.Query, ts.StorageKind.Get])
def test_repeated_storage_wait_keeps_device_and_all_forward_slots_available(phase):
    s = scheduler(role=ts.SchedulerConfig.Role.Fused, timeout=10000)
    before = s.available_lcm_blocks()
    spec = _spec("waiting", list(range(8)))
    spec.max_cached_prefix_tokens = 6
    s.submit_requests([spec])
    operation = storage_ops(s.next_execution_plan())[0]
    if phase == ts.StorageKind.Get:
        done(s, operation, [1] * len(operation.keys), retired=True)
        operation = storage_ops(s.next_execution_plan())[0]
    assert operation.kind == phase
    held = s.storage_held_blocks()

    for now in range(2, 18):
        tick(s, now, forward=True, storage=True)
        plan = s.next_execution_plan()
        assert not requests(plan) and not plan.cache and not storage_ops(plan)
        assert s.available_lcm_blocks() == before
        assert s.storage_held_blocks() == held
        assert s.storage_stats().operations == 1
        assert s.storage_stats().waiting_requests == 1

    # One-token prompts bypass storage and consume all eight forward slots.
    # A waiting QUERY or GET must own none of them or seal new admission.
    ready = [_spec(f"short-{i}", [100 + i]) for i in range(8)]
    s.submit_requests(ready)
    plan = s.next_execution_plan()
    assert requests(plan) == [spec.request_id for spec in ready]
    assert not storage_ops(plan)
    assert s.storage_held_blocks() == held
    assert s.storage_stats().waiting_requests == 1


def test_zero_logprob_bound_never_queries_storage():
    s = scheduler(role=ts.SchedulerConfig.Role.Fused)
    spec = _spec("r", list(range(8)))
    spec.max_cached_prefix_tokens = 0
    s.submit_requests([spec])
    plan = s.next_execution_plan()
    assert requests(plan) == ["r"] and not storage_ops(plan)


def test_ready_result_is_protected_while_forward_is_frozen():
    s = scheduler(role=ts.SchedulerConfig.Role.Fused)
    read = read_pending(s)
    tick(s, 2, forward=False, storage=False)
    done(s, read, [1] * len(read.keys), retired=True)
    plan = s.next_execution_plan()
    assert not requests(plan)
    assert s.storage_held_blocks() > 0
    tick(s, 103, forward=False, storage=False)
    assert not requests(s.next_execution_plan())
    assert s.storage_held_blocks() == 0
    tick(s, 104, forward=True, storage=True)
    plan = s.next_execution_plan()
    assert requests(plan) == ["r"] and not storage_ops(plan)


def test_d2h_publishes_before_put_and_freeze_drops_unowned_candidates():
    s = scheduler(role=ts.SchedulerConfig.Role.Fused)
    s.submit_requests([_spec("r", list(range(8)))])
    q = storage_ops(s.next_execution_plan())[0]
    done(s, q, [0] * len(q.keys), retired=True)
    s.next_execution_plan()
    _advance(s, "r", [99])
    plan = s.next_execution_plan()
    wb = next(op for op in plan.cache if isinstance(op, ts.Cache.WriteBackOp))
    ack = ts.Cache.WriteBackDoneEvent()
    ack.op_id = wb.op_ids[0]
    event(s, ack)
    assert s.host_pool_cached_blocks() > 0
    assert s.storage_held_blocks() == 0
    candidates = s.storage_stats().backup_candidates
    assert candidates > 0
    tick(s, 3, forward=False, storage=False)
    frozen = s.next_execution_plan()
    assert not requests(frozen) and not storage_ops(frozen)
    assert s.storage_held_blocks() == 0
    assert s.storage_stats().backup_candidates == 0
    assert s.storage_stats().cleanup_dropped_backup_keys == candidates


def test_query_credits_never_split_a_prefix_boundary_across_groups():
    s = scheduler(role=ts.SchedulerConfig.Role.Fused, groups=6)
    s.submit_requests([_spec("bounded", list(range(16)))])
    op = storage_ops(s.next_execution_plan())[0]
    assert op.kind == ts.StorageKind.Query
    # 32 keys fit five whole six-group boundaries. A partial sixth boundary
    # cannot establish a usable prefix and must not consume query credits.
    assert len(op.keys) == 30
    for group in range(6):
        assert sum(key[0] == group for key in op.keys) == 5


def test_storage_statistics_separate_request_expiry_from_native_retirement():
    s = scheduler(role=ts.SchedulerConfig.Role.Fused)
    read = read_pending(s)
    stats = s.storage_stats()
    assert stats.waiting_requests == stats.operations == 1
    assert stats.held_blocks == len(read.keys)
    tick(s, 102, forward=True, storage=True)
    stats = s.storage_stats()
    assert stats.wait_timeouts == 1 and stats.ready_expirations == 0
    assert stats.waiting_requests == 0 and stats.operations == 1
    assert stats.held_blocks == len(read.keys)
    tick(s, 103, forward=True, storage=True)
    assert s.storage_stats().wait_timeouts == 1
    done(s, read, [1] * len(read.keys), retired=True)
    assert s.storage_stats().operations == s.storage_stats().held_blocks == 0


def test_ready_expiration_and_detach_have_distinct_statistics():
    s = scheduler(role=ts.SchedulerConfig.Role.Fused)
    read = read_pending(s)
    tick(s, 2, forward=False, storage=False)
    done(s, read, [1] * len(read.keys), retired=True)
    s.next_execution_plan()
    assert s.storage_stats().ready_requests == 1
    tick(s, 103, forward=False, storage=False)
    stats = s.storage_stats()
    assert stats.wait_timeouts == 0 and stats.ready_expirations == 1
    assert stats.operations == stats.ready_requests == stats.held_blocks == 0

    s = scheduler(role=ts.SchedulerConfig.Role.Fused)
    read = read_pending(s)
    event(s, ts.Cache.StorageDetach("r"))
    done(s, read, [0] * len(read.keys), retired=False)
    stats = s.storage_stats()
    assert stats.wait_timeouts == stats.ready_expirations == 0
    assert stats.quarantined == stats.operations == 1
    assert stats.held_blocks == len(read.keys)


def pressure_tick(s, now, *, queued, byte_count, age):
    event(
        s,
        ts.Cache.StorageTick(
            now, True, True, ts.StoragePressure(queued, byte_count, age)
        ),
    )


def test_pressure_bypasses_new_requests_and_reopens_only_below_low_watermarks():
    s = scheduler(role=ts.SchedulerConfig.Role.Fused, timeout=10000)
    pressure_tick(s, 2, queued=8, byte_count=0, age=0)
    s.submit_requests([_spec("bypass", list(range(8)))])
    plan = s.next_execution_plan()
    assert requests(plan) == ["bypass"]
    assert not storage_ops(plan) and s.storage_stats().pressure_bypasses == 1
    assert s.storage_stats().pressure_gate_closed
    pressure_tick(s, 3, queued=5, byte_count=0, age=0)
    assert s.storage_stats().pressure_gate_closed
    pressure_tick(s, 4, queued=4, byte_count=524288, age=500)
    assert not s.storage_stats().pressure_gate_closed
    s.submit_requests([_spec("recovered", list(range(40, 48)))])
    plan = s.next_execution_plan()
    assert any(op.kind == ts.StorageKind.Query for op in storage_ops(plan))


def test_pressure_detaches_query_without_waiting_for_deadline_or_issuing_get():
    for completed in (False, True):
        s = scheduler(role=ts.SchedulerConfig.Role.Fused, timeout=10000)
        s.submit_requests([_spec("r", list(range(8)))])
        query = storage_ops(s.next_execution_plan())[0]
        pressure_tick(s, 2, queued=0, byte_count=0, age=1000)
        if completed:
            done(s, query, [1] * len(query.keys), retired=True)
        plan = s.next_execution_plan()
        assert requests(plan) == ["r"]
        assert not any(
            op.kind in (ts.StorageKind.Query, ts.StorageKind.Get)
            for op in storage_ops(plan)
        )
        assert s.storage_stats().pressure_bypasses == 1
        assert s.storage_stats().wait_timeouts == 0
        assert s.storage_held_blocks() == 0
        if not completed:
            assert any(op.kind == ts.StorageKind.Cancel for op in storage_ops(plan))
            done(s, query, [1] * len(query.keys), retired=True)
        assert not s.storage_pending()


def test_active_get_and_ready_data_progress_while_pressure_blocks_new_work():
    s = scheduler(role=ts.SchedulerConfig.Role.Fused, timeout=10000)
    read = read_pending(s)
    held = s.storage_held_blocks()
    pressure_tick(s, 2, queued=8, byte_count=0, age=0)
    s.submit_requests([_spec("other", list(range(40, 48)))])
    plan = s.next_execution_plan()
    assert requests(plan) == ["other"]
    assert not storage_ops(plan)
    assert s.storage_held_blocks() == held
    done(s, read, [1] * len(read.keys), retired=True)
    plan = s.next_execution_plan()
    assert "r" in requests(plan)
    restored = next(batch for batch in plan.forward if "r" in batch.request_ids)
    assert restored.extend_prefix_lens[list(restored.request_ids).index("r")] == 6
    assert any(isinstance(op, ts.Cache.LoadBackOp) for op in plan.cache)
    assert s.storage_held_blocks() == 0
    assert s.storage_stats().pressure_gate_closed


def test_pressure_does_not_pause_soft_timeout_or_release_detached_native_get():
    s = scheduler(role=ts.SchedulerConfig.Role.Fused, timeout=100)
    read = read_pending(s)
    held = s.storage_held_blocks()
    pressure_tick(s, 2, queued=8, byte_count=0, age=0)
    pressure_tick(s, 102, queued=8, byte_count=0, age=1000)
    plan = s.next_execution_plan()
    assert requests(plan) == ["r"]
    assert s.storage_held_blocks() == held
    assert s.storage_stats().wait_timeouts == 1
    done(s, read, [1] * len(read.keys), retired=True)
    assert s.storage_held_blocks() == 0
    assert s.host_pool_cached_blocks() == 0


@pytest.mark.parametrize("bootstrap_before_get", [False, True])
def test_prefill_storage_preparation_survives_bootstrap(bootstrap_before_get):
    s = scheduler(role=ts.SchedulerConfig.Role.P)
    before = s.available_lcm_blocks()
    s.submit_requests([_spec("p", list(range(8)))])
    plan = s.next_execution_plan()
    assert not requests(plan)
    assert (
        len(storage_ops(plan)) == 1
        and storage_ops(plan)[0].kind == ts.StorageKind.Query
    )
    assert s.bootstrapping_size() == 1 and s.waiting_size() == 0
    query = storage_ops(plan)[0]
    if bootstrap_before_get:
        event(s, ts.PD.BootstrappedEvent("p"))
        assert s.bootstrapping_size() == 0 and s.waiting_size() == 1
    done(s, query, [1] * len(query.keys), retired=True)
    plan = s.next_execution_plan()
    assert not requests(plan)
    assert (
        len(storage_ops(plan)) == 1 and storage_ops(plan)[0].kind == ts.StorageKind.Get
    )
    read = storage_ops(plan)[0]
    assert s.available_lcm_blocks() == before
    done(s, read, [1] * len(read.keys), retired=True)
    if not bootstrap_before_get:
        plan = s.next_execution_plan()
        assert not requests(plan) and not storage_ops(plan)
        assert s.available_lcm_blocks() == before
        event(s, ts.PD.BootstrappedEvent("p"))
    plan = s.next_execution_plan()
    assert requests(plan) == ["p"]
    assert list(plan.forward[0].extend_prefix_lens) == [7]
    assert any(isinstance(op, ts.Cache.LoadBackOp) for op in plan.cache)
    assert s.storage_held_blocks() == 0


def mixed_replay_scheduler():
    cfg = ts.SchedulerConfig()
    cfg.role = ts.SchedulerConfig.Role.Fused
    cfg.prefix_granularity = 2
    cfg.num_device_pages = 64
    cfg.num_host_pages = 32
    cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteThrough
    cfg.num_snapshot_pages = 1
    cfg.max_scheduled_tokens = 4
    cfg.max_batch_size = 8
    cfg.enable_mixed_prefill_decode = True
    cfg.enable_l3_storage = True
    cfg.storage_config = ts.StorageConfig(
        8,
        32,
        15,
        4,
        32,
        2,
        10000,
        100,
        ts.StoragePressure(8, 1048576, 1000),
        ts.StoragePressure(4, 524288, 500),
    )
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="full",
            block_granularity=2,
            total_pages=64,
            attn_kind=ts.AttnKind.Full,
        ),
        ts.CacheGroupConfig(
            group_id="replayable",
            block_granularity=2,
            total_pages=64,
            attn_kind=ts.AttnKind.SlidingWindow,
            sliding_window_tokens=2,
            replayable=True,
        ),
    ]
    cfg.host_group_packing = [1, 1]
    result = ts.Scheduler(cfg)
    tick(result, 1, forward=True, storage=True)
    return result


@pytest.mark.parametrize("phase", [ts.StorageKind.Query, ts.StorageKind.Get])
def test_storage_wait_never_reserves_mixed_prefill_tokens(phase):
    s = mixed_replay_scheduler()
    s.submit_requests([_spec("decode", [100])])
    assert requests(s.next_execution_plan()) == ["decode"]
    _advance(s, "decode", [101])
    assert requests(s.next_execution_plan()) == ["decode"]
    _advance(s, "decode", [102])
    s.submit_requests([_spec("waiting", list(range(8)))])
    plan = s.next_execution_plan()
    assert requests(plan) == ["decode"]
    _advance(s, "decode", [103])
    operation = storage_ops(plan)[0]
    if phase == ts.StorageKind.Get:
        done(s, operation, [1] * len(operation.keys), retired=True)
        plan = s.next_execution_plan()
        assert requests(plan) == ["decode"]
        _advance(s, "decode", [104])
        operation = storage_ops(plan)[0]
    assert operation.kind == phase
    for now in range(2, 6):
        tick(s, now, forward=True, storage=True)
        assert requests(s.next_execution_plan()) == ["decode"]
        _advance(s, "decode", [104 + now])
        assert s.storage_stats().waiting_requests == 1


@pytest.mark.parametrize("phase", [ts.StorageKind.Query, ts.StorageKind.Get])
def test_prefill_timeout_preserves_bootstrap_and_native_retirement(phase):
    s = scheduler(role=ts.SchedulerConfig.Role.P)
    s.submit_requests([_spec("p", list(range(8)))])
    operation = storage_ops(s.next_execution_plan())[0]
    if phase == ts.StorageKind.Get:
        tick(s, 50, forward=True, storage=True)
        done(s, operation, [1] * len(operation.keys), retired=True)
        operation = storage_ops(s.next_execution_plan())[0]
    assert operation.kind == phase
    held = s.storage_held_blocks()
    tick(s, 102, forward=True, storage=True)
    plan = s.next_execution_plan()
    assert not requests(plan)
    assert any(op.kind == ts.StorageKind.Cancel for op in storage_ops(plan))
    assert s.bootstrapping_size() == 1
    assert s.waiting_size() == 0
    assert s.storage_stats().wait_timeouts == 1
    assert s.storage_held_blocks() == held
    done(s, operation, [1] * len(operation.keys), retired=True)
    assert s.storage_held_blocks() == 0
    assert s.host_pool_cached_blocks() == 0
    event(s, ts.PD.BootstrappedEvent("p"))
    plan = s.next_execution_plan()
    assert requests(plan) == ["p"] and not storage_ops(plan)
    assert list(plan.forward[0].extend_prefix_lens) == [0]


def test_ready_hold_expires_while_prefill_still_waits_for_bootstrap():
    s = scheduler(role=ts.SchedulerConfig.Role.P)
    read = read_pending(s)
    done(s, read, [1] * len(read.keys), retired=True)
    assert not requests(s.next_execution_plan())
    assert s.storage_stats().ready_requests == 1
    assert s.bootstrapping_size() == 1
    assert s.waiting_size() == 0
    assert s.storage_held_blocks() > 0
    tick(s, 102, forward=True, storage=True)
    assert not requests(s.next_execution_plan())
    assert s.storage_stats().ready_expirations == 1
    assert s.storage_held_blocks() == 0
    event(s, ts.PD.BootstrappedEvent("r"))
    plan = s.next_execution_plan()
    assert requests(plan) == ["r"] and not storage_ops(plan)


@pytest.mark.parametrize(
    "role", [ts.SchedulerConfig.Role.P, ts.SchedulerConfig.Role.Fused]
)
def test_abort_prefetching_keeps_native_pins_and_never_restarts(role):
    s = scheduler(role=role)
    read = read_pending(s)
    held = s.storage_held_blocks()
    assert s.can_abort_before_admission("r") == (role == ts.SchedulerConfig.Role.Fused)
    if role == ts.SchedulerConfig.Role.P:
        event(s, ts.PD.FailedEvent("r"))
    else:
        abort = ts.ForwardEvent.Abort()
        abort.request_id = "r"
        event(s, abort)
    plan = s.next_execution_plan()
    assert not requests(plan)
    assert any(op.kind == ts.StorageKind.Cancel for op in storage_ops(plan))
    assert s.storage_held_blocks() == held
    done(s, read, [1] * len(read.keys), retired=True)
    assert s.storage_held_blocks() == 0 and s.host_pool_cached_blocks() == 0
    assert not requests(s.next_execution_plan())
