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

"""Decode residency grows independently of active execution admission."""

import pytest
import tokenspeed_scheduler as ts
from conftest import _advance, _event, _find_forward_op, _finish, _spec


def _config(batch_size: int, residents: int) -> ts.SchedulerConfig:
    cfg = ts.SchedulerConfig()
    cfg.role = ts.SchedulerConfig.Role.D
    cfg.prefix_granularity = 2
    cfg.max_batch_size = batch_size
    cfg.max_resident_requests = residents
    cfg.max_scheduled_tokens = 64
    cfg.num_device_pages = 257
    cfg.num_host_pages = 0
    cfg.num_snapshot_pages = 257
    cfg.max_retracted_requests = 4
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = True
    cfg.overlap_schedule_depth = 1
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="history",
            block_granularity=2,
            total_pages=257,
            attn_kind=ts.AttnKind.Full,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        )
    ]
    cfg.host_group_packing = [1]
    return cfg


def _send(scheduler, event) -> None:
    scheduler.advance(ts.ExecutionEvent().add_event(event))


def _submit(scheduler, *request_ids: str) -> None:
    scheduler.submit_requests(
        [_spec(rid, [100 * i + 1, 100 * i + 2]) for i, rid in enumerate(request_ids)]
    )
    for rid in request_ids:
        _send(scheduler, ts.PD.BootstrappedEvent(rid))


def _ready(scheduler, *request_ids: str) -> None:
    for rid in request_ids:
        _send(scheduler, ts.PD.RemotePrefillDoneEvent(rid, 42))


def _forward_ids(plan) -> list[str]:
    op = _find_forward_op(plan)
    return [] if op is None else list(op.request_ids)


def _admit_all(scheduler, request_ids: tuple[str, ...]) -> list[int]:
    slots = []
    for rid in request_ids:
        plan = scheduler.next_execution_plan()
        assert plan.remote_prefill is not None
        assert list(plan.remote_prefill.request_ids) == [rid]
        slots.extend(plan.remote_prefill.request_pool_indices)
    return slots


def _ack(scheduler, op, kind) -> None:
    for op_id in op.op_ids:
        event = kind()
        event.op_id = int(op_id)
        _send(scheduler, event)


def test_fused_decode_uses_request_capacity_without_decode_permits():
    cfg = _config(2, 2)
    cfg.role = ts.SchedulerConfig.Role.Fused
    scheduler = ts.Scheduler(cfg)
    scheduler.submit_requests(
        [
            _spec(rid, [100 * i + 1, 100 * i + 2])
            for i, rid in enumerate(("a", "b", "c"))
        ]
    )
    assert _forward_ids(scheduler.next_execution_plan()) == ["a", "b"]
    for rid in ("a", "b"):
        _advance(scheduler, rid, [42])
    assert scheduler.allocated_request_slots() == 2
    assert scheduler.ready_waiting_size() == 0

    for token in (43, 44):
        assert _forward_ids(scheduler.next_execution_plan()) == ["a", "b"]
        assert scheduler.decoding_size() == 2
        assert scheduler.active_decode_slots() == 0
        for rid in ("a", "b"):
            _advance(scheduler, rid, [token])

    _finish(scheduler, "a")
    assert _forward_ids(scheduler.next_execution_plan()) == ["c"]
    _advance(scheduler, "c", [42])
    assert _forward_ids(scheduler.next_execution_plan()) == ["b", "c"]
    assert scheduler.allocated_request_slots() == 2
    assert scheduler.active_decode_slots() == 0
    assert scheduler.ready_waiting_size() == 0
    for rid in ("b", "c"):
        _advance(scheduler, rid, [45])
        _finish(scheduler, rid)
    assert _forward_ids(scheduler.next_execution_plan()) == []
    assert scheduler.allocated_request_slots() == 0


def test_fused_decode_restores_without_reserving_a_decode_permit():
    cfg = _config(1, 1)
    cfg.role = ts.SchedulerConfig.Role.Fused
    cfg.debug_force_retraction_interval = 4
    scheduler = ts.Scheduler(cfg)
    scheduler.submit_requests([_spec("running", [1, 2])])
    assert _forward_ids(scheduler.next_execution_plan()) == ["running"]
    _advance(scheduler, "running", [42])
    for token in (43, 44):
        assert _forward_ids(scheduler.next_execution_plan()) == ["running"]
        _advance(scheduler, "running", [token])

    retract = scheduler.next_execution_plan()
    [snapshot] = [op for op in retract.cache if isinstance(op, ts.Cache.SnapshotOp)]
    assert _forward_ids(retract) == []
    assert scheduler.retracted_size() == 1
    assert scheduler.active_decode_slots() == 0
    _ack(scheduler, snapshot, ts.Cache.SnapshotDoneEvent)

    restore = scheduler.next_execution_plan()
    [restore_op] = [op for op in restore.cache if isinstance(op, ts.Cache.RestoreOp)]
    assert _forward_ids(restore) == []
    assert list(restore_op.request_ids) == ["running"]
    assert scheduler.allocated_request_slots() == 1
    assert scheduler.active_decode_slots() == 0
    _ack(scheduler, restore_op, ts.Cache.RestoreDoneEvent)
    assert scheduler.retracted_size() == 0
    assert scheduler.decoding_size() == 1
    assert scheduler.request_token_size("running") == 5

    assert _forward_ids(scheduler.next_execution_plan()) == ["running"]
    assert scheduler.active_decode_slots() == 0
    assert scheduler.ready_waiting_size() == 0
    _advance(scheduler, "running", [45])
    _finish(scheduler, "running")
    assert _forward_ids(scheduler.next_execution_plan()) == []
    assert scheduler.allocated_request_slots() == 0
    assert scheduler.active_decode_slots() == 0


def test_remote_prefill_uses_resident_capacity_beyond_the_decode_batch():
    scheduler = ts.Scheduler(_config(2, 4))
    _submit(scheduler, "a", "b", "c", "d", "e")
    slots = _admit_all(scheduler, ("a", "b", "c", "d"))
    assert len(set(slots)) == 4
    assert max(slots) > 2
    assert scheduler.remote_prefilling_size() == 4
    assert scheduler.allocated_request_slots() == 4
    assert scheduler.active_decode_slots() == 0
    assert scheduler.ready_waiting_size() == 0
    assert scheduler.next_execution_plan().remote_prefill is None


def test_late_older_prefills_do_not_displace_streaming_requests():
    scheduler = ts.Scheduler(_config(2, 4))
    _submit(scheduler, "a", "b", "c", "d")
    _admit_all(scheduler, ("a", "b", "c", "d"))
    _ready(scheduler, "c", "d")
    assert scheduler.ready_waiting_size() == 2
    assert _forward_ids(scheduler.next_execution_plan()) == ["c", "d"]
    _advance(scheduler, "c", [43])
    _advance(scheduler, "d", [43])
    _ready(scheduler, "a", "b")
    assert scheduler.active_decode_slots() == 2
    assert scheduler.ready_waiting_size() == 2
    for token in range(44, 47):
        plan = scheduler.next_execution_plan()
        assert _forward_ids(plan) == ["c", "d"]
        assert not plan.aborts
        assert not any(isinstance(op, ts.Cache.SnapshotOp) for op in plan.cache)
        _advance(scheduler, "c", [token])
        _advance(scheduler, "d", [token])
    _finish(scheduler, "c")
    assert _forward_ids(scheduler.next_execution_plan()) == ["a", "d"]


def test_decode_page_growth_keeps_its_slot_and_hands_capacity_to_a_waiter():
    scheduler = ts.Scheduler(_config(1, 2))
    _submit(scheduler, "running", "waiting")
    slots = _admit_all(scheduler, ("running", "waiting"))
    _ready(scheduler, "running", "waiting")
    first = _find_forward_op(scheduler.next_execution_plan())
    assert list(first.request_ids) == ["running"]
    assert list(first.request_pool_indices) == [slots[0]]
    pages = [int(page) for page in first.block_tables["history"][0] if page > 0]
    _advance(scheduler, "running", [43])

    for token, reserve in enumerate((1, 6, 1, 1), start=44):
        _send(
            scheduler,
            _event(
                "UpdateReserveNumTokens",
                "running",
                reserve_num_tokens_in_next_schedule_event=reserve,
            ),
        )
        plan = scheduler.next_execution_plan()
        op = _find_forward_op(plan)
        assert list(op.request_ids) == ["running"]
        assert list(op.request_pool_indices) == [slots[0]]
        next_pages = [int(page) for page in op.block_tables["history"][0] if page > 0]
        assert next_pages[: len(pages)] == pages
        if reserve == 6:
            assert len(next_pages) > len(pages)
        assert scheduler.active_decode_slots() == 1
        assert scheduler.allocated_request_slots() == 2
        assert scheduler.ready_waiting_size() == 1
        assert not plan.aborts
        pages = next_pages
        _advance(scheduler, "running", [token])

    _finish(scheduler, "running")
    next_op = _find_forward_op(scheduler.next_execution_plan())
    assert list(next_op.request_ids) == ["waiting"]
    assert list(next_op.request_pool_indices) == [slots[1]]
    assert scheduler.active_decode_slots() == 1
    assert scheduler.allocated_request_slots() == 1
    assert scheduler.ready_waiting_size() == 0
    _advance(scheduler, "waiting", [99])
    _finish(scheduler, "waiting")
    assert _forward_ids(scheduler.next_execution_plan()) == []
    assert scheduler.active_decode_slots() == 0
    assert scheduler.allocated_request_slots() == 0


@pytest.mark.parametrize("terminal", ["Finish", "Abort"])
def test_terminal_request_keeps_its_permit_until_overlap_results_land(terminal):
    scheduler = ts.Scheduler(_config(1, 2))
    _submit(scheduler, "running", "waiting")
    _admit_all(scheduler, ("running", "waiting"))
    _ready(scheduler, "running", "waiting")
    assert _forward_ids(scheduler.next_execution_plan()) == ["running"]
    assert _forward_ids(scheduler.next_execution_plan()) == ["running"]
    active_pages = scheduler.active_lcm_blocks()
    _send(scheduler, _event(terminal, "running"))
    assert scheduler.active_lcm_blocks() == active_pages
    assert scheduler.active_decode_slots() == 1
    assert scheduler.allocated_request_slots() == 2
    assert scheduler.ready_waiting_size() == 1
    assert _forward_ids(scheduler.next_execution_plan()) == []
    _advance(scheduler, "running", [43])
    assert _forward_ids(scheduler.next_execution_plan()) == []
    _advance(scheduler, "running", [44])
    assert scheduler.active_decode_slots() == 0
    assert scheduler.allocated_request_slots() == 1
    assert _forward_ids(scheduler.next_execution_plan()) == ["waiting"]


def test_aborted_forward_does_not_publish_a_prefix():
    cfg = _config(1, 1)
    cfg.disable_prefix_cache = False
    scheduler = ts.Scheduler(cfg)
    _submit(scheduler, "aborted")
    _admit_all(scheduler, ("aborted",))
    _ready(scheduler, "aborted")
    assert _forward_ids(scheduler.next_execution_plan()) == ["aborted"]
    _send(scheduler, _event("Abort", "aborted"))
    _advance(scheduler, "aborted", [43])
    scheduler.next_execution_plan()
    assert scheduler.active_lcm_blocks() == 0
    assert scheduler.available_lcm_blocks() == cfg.num_device_pages - 1


@pytest.mark.parametrize("cancel_restore", [False, True])
def test_restore_reserves_a_permit_until_its_ack(cancel_restore):
    cfg = _config(1, 3)
    cfg.debug_force_retraction_interval = 5
    scheduler = ts.Scheduler(cfg)
    _submit(scheduler, "running", "waiting")
    _admit_all(scheduler, ("running", "waiting"))  # plans 1, 2
    _ready(scheduler, "running")
    assert _forward_ids(scheduler.next_execution_plan()) == ["running"]  # 3
    _advance(scheduler, "running", [43])
    assert _forward_ids(scheduler.next_execution_plan()) == ["running"]  # 4
    _advance(scheduler, "running", [44])
    retract = scheduler.next_execution_plan()  # 5
    stores = [op for op in retract.cache if isinstance(op, ts.Cache.SnapshotOp)]
    assert len(stores) == 1
    assert _forward_ids(retract) == []
    for op in stores:
        _ack(scheduler, op, ts.Cache.SnapshotDoneEvent)
    _ready(scheduler, "waiting")
    restore = scheduler.next_execution_plan()  # 6: restore first, then ready work
    restores = [op for op in restore.cache if isinstance(op, ts.Cache.RestoreOp)]
    assert len(restores) == 1
    assert _forward_ids(restore) == []
    assert scheduler.active_decode_slots() == 1
    assert scheduler.allocated_request_slots() == 2
    assert scheduler.ready_waiting_size() == 1
    if cancel_restore:
        _send(scheduler, _event("Abort", "running"))
    _submit(scheduler, "other")
    pending = scheduler.next_execution_plan()  # 7
    assert _forward_ids(pending) == []
    assert list(pending.remote_prefill.request_ids) == ["other"]
    _ready(scheduler, "other")
    assert scheduler.active_decode_slots() == 1
    assert scheduler.allocated_request_slots() == 3
    for op in restores:
        _ack(scheduler, op, ts.Cache.RestoreDoneEvent)
        _ack(
            scheduler, op, ts.Cache.RestoreDoneEvent
        )  # duplicate does not add capacity
    assert scheduler.active_decode_slots() == (0 if cancel_restore else 1)
    assert scheduler.allocated_request_slots() == (2 if cancel_restore else 3)
    expected = ["waiting"] if cancel_restore else ["running"]
    assert _forward_ids(scheduler.next_execution_plan()) == expected
    _advance(scheduler, expected[0], [45])
    assert _forward_ids(scheduler.next_execution_plan()) == expected


@pytest.mark.parametrize("batch_size", [1, 2])
def test_zero_extra_capacity_preserves_the_request_pool_limit(batch_size):
    scheduler = ts.Scheduler(_config(batch_size, batch_size))
    ids = tuple(f"r{i}" for i in range(batch_size + 1))
    _submit(scheduler, *ids)
    _admit_all(scheduler, ids[:-1])
    assert scheduler.next_execution_plan().remote_prefill is None


@pytest.mark.parametrize(
    "role", [ts.SchedulerConfig.Role.P, ts.SchedulerConfig.Role.Fused]
)
def test_non_decode_role_rejects_extra_residency(role):
    cfg = _config(2, 4)
    cfg.role = role
    if role == ts.SchedulerConfig.Role.P:
        cfg.num_snapshot_pages = 1
        cfg.max_retracted_requests = 0
    with pytest.raises(ValueError, match="resident|residen|decode|role"):
        ts.Scheduler(cfg)


def test_resident_capacity_must_cover_the_execution_capacity():
    with pytest.raises(ValueError, match="resident"):
        ts.Scheduler(_config(2, 1))


def test_capacity_model_prices_state_using_resident_requests():
    cfg = _config(2, 4)
    cfg.cache_groups = [
        *cfg.cache_groups,
        ts.CacheGroupConfig(
            group_id="state",
            block_granularity=0,
            total_pages=257,
            attn_kind=ts.AttnKind.State,
            transfer_policy=ts.CacheTransferPolicy.LatestSnapshot,
        ),
    ]
    model = ts.CapacityModel(cfg)
    assert model.concurrent_group_pages(32, 16)[1] == 8


@pytest.mark.parametrize("residents", [-1, 2**31 - 1])
def test_invalid_resident_capacity_is_rejected_before_allocation(residents):
    with pytest.raises(ValueError, match="resident"):
        ts.Scheduler(_config(1, residents))


def test_restore_waits_for_a_running_request_to_release_its_permit():
    cfg = _config(1, 3)
    cfg.debug_force_retraction_interval = 5
    scheduler = ts.Scheduler(cfg)
    _submit(scheduler, "retracted", "running")
    _admit_all(scheduler, ("retracted", "running"))
    _ready(scheduler, "retracted")
    for token in (43, 44):
        assert _forward_ids(scheduler.next_execution_plan()) == ["retracted"]
        _advance(scheduler, "retracted", [token])
    retract = scheduler.next_execution_plan()
    stores = [op for op in retract.cache if isinstance(op, ts.Cache.SnapshotOp)]
    assert len(stores) == 1
    _ready(scheduler, "running")
    assert _forward_ids(scheduler.next_execution_plan()) == ["running"]
    _advance(scheduler, "running", [43])
    for op in stores:
        _ack(scheduler, op, ts.Cache.SnapshotDoneEvent)
    busy = scheduler.next_execution_plan()
    assert _forward_ids(busy) == ["running"]
    assert not any(isinstance(op, ts.Cache.RestoreOp) for op in busy.cache)
    assert not busy.aborts
    _advance(scheduler, "running", [44])
    _finish(scheduler, "running")
    restore = scheduler.next_execution_plan()
    restores = [op for op in restore.cache if isinstance(op, ts.Cache.RestoreOp)]
    assert len(restores) == 1
    assert _forward_ids(restore) == []
    for op in restores:
        _ack(scheduler, op, ts.Cache.RestoreDoneEvent)
    assert _forward_ids(scheduler.next_execution_plan()) == ["retracted"]


def test_extra_request_slots_do_not_bypass_kv_admission():
    cfg = _config(1, 3)
    cfg.num_device_pages = 4
    groups = cfg.cache_groups
    groups[0].total_pages = 4
    cfg.cache_groups = groups
    scheduler = ts.Scheduler(cfg)
    _submit(scheduler, "receiving", "waiting")
    _admit_all(scheduler, ("receiving",))
    plan = scheduler.next_execution_plan()
    assert plan.remote_prefill is None
    assert not plan.aborts
    assert scheduler.remote_prefilling_size() == 1
    assert scheduler.retracted_size() == 0


def _decode_growth_blocked_by_ready_requests(snapshot_rows, snapshot_pages):
    cfg = _config(1, 3)
    cfg.num_device_pages = 17
    cfg.num_snapshot_pages = snapshot_pages
    cfg.max_retracted_requests = snapshot_rows
    groups = cfg.cache_groups
    groups[0].total_pages = 17
    cfg.cache_groups = groups
    scheduler = ts.Scheduler(cfg)
    scheduler.submit_requests(
        [
            _spec("running", [1, 2]),
            _spec("waiting-a", list(range(101, 109))),
            _spec("waiting-b", list(range(201, 209))),
        ]
    )
    for rid in ("running", "waiting-a", "waiting-b"):
        _send(scheduler, ts.PD.BootstrappedEvent(rid))
    _admit_all(scheduler, ("running", "waiting-a", "waiting-b"))
    _ready(scheduler, "running")
    assert _forward_ids(scheduler.next_execution_plan()) == ["running"]
    _advance(scheduler, "running", [43])
    _ready(scheduler, "waiting-a", "waiting-b")
    assert scheduler.ready_waiting_size() == 2
    assert scheduler.available_lcm_blocks() == 4
    # The running request grows from 2 to 15 blocks. Both 5-block waiters
    # must give way: the 4 free blocks plus either one's release cannot fit it.
    _send(
        scheduler,
        _event(
            "UpdateReserveNumTokens",
            "running",
            reserve_num_tokens_in_next_schedule_event=26,
        ),
    )
    return scheduler, scheduler.next_execution_plan()


def test_ready_waiting_requests_retract_beyond_batch_size_and_resume():
    scheduler, pressure = _decode_growth_blocked_by_ready_requests(3, 257)
    [snapshot] = [op for op in pressure.cache if isinstance(op, ts.Cache.SnapshotOp)]
    assert list(snapshot.request_ids) == ["waiting-a", "waiting-b"]
    assert _forward_ids(pressure) == ["running"]
    assert not pressure.aborts
    assert scheduler.retracted_size() == 2  # More suspended requests than B=1.
    assert scheduler.allocated_request_slots() == 1
    assert scheduler.active_decode_slots() == 1
    assert scheduler.ready_waiting_size() == 0
    assert scheduler.snapshot_pool_free_blocks() == 248
    _ack(scheduler, snapshot, ts.Cache.SnapshotDoneEvent)
    _advance(scheduler, "running", [44])
    _send(
        scheduler,
        _event(
            "UpdateReserveNumTokens",
            "running",
            reserve_num_tokens_in_next_schedule_event=1,
        ),
    )
    busy = scheduler.next_execution_plan()
    assert _forward_ids(busy) == ["running"]
    assert not any(isinstance(op, ts.Cache.RestoreOp) for op in busy.cache)
    assert not busy.aborts
    _advance(scheduler, "running", [45])
    _finish(scheduler, "running")

    restore_a = scheduler.next_execution_plan()
    [restore] = [op for op in restore_a.cache if isinstance(op, ts.Cache.RestoreOp)]
    assert list(restore.request_ids) == ["waiting-a"]
    assert _forward_ids(restore_a) == []
    assert scheduler.allocated_request_slots() == 1
    assert scheduler.active_decode_slots() == 0
    _ack(scheduler, restore, ts.Cache.RestoreDoneEvent)
    assert scheduler.ready_waiting_size() == 1
    assert scheduler.request_token_size("waiting-a") == 9

    restore_b = scheduler.next_execution_plan()
    [restore] = [op for op in restore_b.cache if isinstance(op, ts.Cache.RestoreOp)]
    assert list(restore.request_ids) == ["waiting-b"]
    assert _forward_ids(restore_b) == ["waiting-a"]
    assert list(_find_forward_op(restore_b).decode_input_ids) == [42]
    assert scheduler.active_decode_slots() == 1
    assert scheduler.allocated_request_slots() == 2
    _ack(scheduler, restore, ts.Cache.RestoreDoneEvent)
    assert scheduler.ready_waiting_size() == 1
    assert scheduler.retracted_size() == 0
    assert scheduler.snapshot_pool_free_blocks() == 256
    _advance(scheduler, "waiting-a", [46])
    assert scheduler.request_token_size("waiting-a") == 10
    _finish(scheduler, "waiting-a")

    resumed = scheduler.next_execution_plan()
    assert _forward_ids(resumed) == ["waiting-b"]
    assert list(_find_forward_op(resumed).decode_input_ids) == [42]
    assert scheduler.request_token_size("waiting-b") == 9
    assert resumed.remote_prefill is None
    _advance(scheduler, "waiting-b", [47])
    assert scheduler.request_token_size("waiting-b") == 10
    _finish(scheduler, "waiting-b")
    assert _forward_ids(scheduler.next_execution_plan()) == []
    assert scheduler.allocated_request_slots() == 0
    assert scheduler.active_decode_slots() == 0
    assert scheduler.available_lcm_blocks() == 16
    assert scheduler.snapshot_pool_free_blocks() == 256


@pytest.mark.parametrize(
    "snapshot_rows,snapshot_pages,suspended,aborted,limiting_arg",
    [
        (1, 257, ["waiting-a"], ["waiting-b"], "--retraction-snapshot-max-requests"),
        (3, 2, [], ["waiting-b", "waiting-a"], "--retraction-snapshot-host-gb"),
    ],
    ids=["snapshot-rows-exhausted", "snapshot-kv-exhausted"],
)
def test_ready_waiting_capacity_abort_preserves_progress(
    snapshot_rows, snapshot_pages, suspended, aborted, limiting_arg
):
    scheduler, pressure = _decode_growth_blocked_by_ready_requests(
        snapshot_rows, snapshot_pages
    )
    snapshots = [op for op in pressure.cache if isinstance(op, ts.Cache.SnapshotOp)]
    assert [rid for op in snapshots for rid in op.request_ids] == suspended
    assert [abort.request_id for abort in pressure.aborts] == aborted
    for abort in pressure.aborts:
        assert abort.reason == ts.AbortReason.ImageDoesNotFit
        assert limiting_arg in abort.detail
    assert _forward_ids(pressure) == ["running"]
    assert scheduler.retracted_size() == len(suspended)
    assert scheduler.allocated_request_slots() == 1
    assert scheduler.active_decode_slots() == 1
    for snapshot in snapshots:
        _ack(scheduler, snapshot, ts.Cache.SnapshotDoneEvent)
    _advance(scheduler, "running", [44])
    assert scheduler.request_token_size("running") == 5
    _finish(scheduler, "running")

    for rid in suspended:
        plan = scheduler.next_execution_plan()
        [restore] = [op for op in plan.cache if isinstance(op, ts.Cache.RestoreOp)]
        assert list(restore.request_ids) == [rid]
        assert not plan.aborts
        _ack(scheduler, restore, ts.Cache.RestoreDoneEvent)
        resumed = scheduler.next_execution_plan()
        assert _forward_ids(resumed) == [rid]
        assert list(_find_forward_op(resumed).decode_input_ids) == [42]
        _advance(scheduler, rid, [45])
        assert scheduler.request_token_size(rid) == 10
        _finish(scheduler, rid)
    drained = scheduler.next_execution_plan()
    assert _forward_ids(drained) == []
    assert not drained.aborts
    assert scheduler.retracted_size() == 0
    assert scheduler.allocated_request_slots() == 0
    assert scheduler.active_decode_slots() == 0
    assert scheduler.available_lcm_blocks() == 16
    assert scheduler.snapshot_pool_free_blocks() == snapshot_pages - 1


def test_long_decode_retracts_a_ready_request_when_reserved_headroom_runs_out():
    cfg = _config(1, 2)
    cfg.prefix_granularity = 64
    cfg.num_device_pages = 193
    cfg.num_snapshot_pages = 193
    cfg.max_retracted_requests = 2
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="history",
            block_granularity=64,
            total_pages=193,
            attn_kind=ts.AttnKind.Full,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        ),
    ]
    scheduler = ts.Scheduler(cfg)
    requests = [
        _spec("running", list(range(1024))),
        _spec("waiting", list(range(10000, 12048))),
    ]
    for request in requests:
        request.max_new_tokens = 9000
    scheduler.submit_requests(requests)
    for rid in ("running", "waiting"):
        _send(scheduler, ts.PD.BootstrappedEvent(rid))
    _admit_all(scheduler, ("running", "waiting"))
    _ready(scheduler, "running", "waiting")
    # Admission prepays 4096 generation tokens each: 80 + 96 of 192 blocks.
    assert scheduler.available_lcm_blocks() == 16
    victims = []
    max_used_blocks = 0
    for token in range(20000, 28192):
        plan = scheduler.next_execution_plan()
        assert _forward_ids(plan) == ["running"]
        assert not plan.aborts
        assert not any(isinstance(op, ts.Cache.RestoreOp) for op in plan.cache)
        for op in plan.cache:
            if isinstance(op, ts.Cache.SnapshotOp):
                victims.extend(op.request_ids)
                _ack(scheduler, op, ts.Cache.SnapshotDoneEvent)
        max_used_blocks = max(max_used_blocks, scheduler.active_lcm_blocks())
        _advance(scheduler, "running", [token])
    assert victims == ["waiting"]
    assert max_used_blocks == 192
    assert scheduler.request_token_size("running") == 9217
    assert scheduler.retracted_size() == 1
    _finish(scheduler, "running")

    plan = scheduler.next_execution_plan()
    [restore] = [op for op in plan.cache if isinstance(op, ts.Cache.RestoreOp)]
    assert list(restore.request_ids) == ["waiting"]
    assert scheduler.active_decode_slots() == 0
    _ack(scheduler, restore, ts.Cache.RestoreDoneEvent)
    resumed = scheduler.next_execution_plan()
    assert _forward_ids(resumed) == ["waiting"]
    assert list(_find_forward_op(resumed).decode_input_ids) == [42]
    assert scheduler.request_token_size("waiting") == 2049
    _advance(scheduler, "waiting", [77])
    assert scheduler.request_token_size("waiting") == 2050
    _finish(scheduler, "waiting")
    assert _forward_ids(scheduler.next_execution_plan()) == []
    assert scheduler.retracted_size() == 0
    assert scheduler.allocated_request_slots() == 0
    assert scheduler.active_decode_slots() == 0
    assert scheduler.available_lcm_blocks() == 192
    assert scheduler.snapshot_pool_free_blocks() == 192
