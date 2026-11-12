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
