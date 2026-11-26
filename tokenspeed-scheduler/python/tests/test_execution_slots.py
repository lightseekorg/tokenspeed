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

"""Per-row execution slots and fresh flags (docs/design/scheduler.md).

An execution slot lies in 1..max_batch_size on every role: the decode permit
on the D role, the request row elsewhere, -1 for a row the transfer peer runs.
A row is fresh on the request's first forward on its slot, and a retraction
lists the victim's slot with its state folds before releasing it.
"""

import pytest
import tokenspeed_scheduler as ts
from conftest import K3_STATE_GROUPS, _advance, _event, _finish, _make_k3_config

Role = ts.SchedulerConfig.Role


def _history_config(role, batch_size: int, residents: int) -> ts.SchedulerConfig:
    cfg = ts.SchedulerConfig()
    cfg.role = role
    cfg.prefix_granularity = 2
    cfg.max_batch_size = batch_size
    cfg.max_resident_requests = residents
    cfg.max_scheduled_tokens = 64
    cfg.num_device_pages = 257
    cfg.num_host_pages = 0
    cfg.num_snapshot_pages = 257 if role != Role.P else 1
    cfg.max_retracted_requests = 4 if role != Role.P else 0
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
    if role != Role.P:
        cfg.host_group_packing = [1]
    return cfg


def _spec(request_id: str, tokens: list[int]) -> ts.RequestSpec:
    spec = ts.RequestSpec()
    spec.request_id = request_id
    spec.tokens = list(tokens)
    spec.max_new_tokens = 16
    return spec


def _send(scheduler, event) -> None:
    scheduler.advance(ts.ExecutionEvent().add_event(event))


def _submit_pd(scheduler, *request_ids: str) -> None:
    scheduler.submit_requests(
        [_spec(rid, [100 * i + 1, 100 * i + 2]) for i, rid in enumerate(request_ids)]
    )
    for rid in request_ids:
        _send(scheduler, ts.PD.BootstrappedEvent(rid))


def _land(scheduler, *request_ids: str) -> None:
    for rid in request_ids:
        _send(scheduler, ts.PD.RemotePrefillDoneEvent(rid, 42))


def _admit(scheduler, *request_ids: str) -> None:
    """One remote admission per plan; the landing rows hold no execution slot."""
    for rid in request_ids:
        landing = scheduler.next_execution_plan().remote_prefill
        assert list(landing.request_ids) == [rid]
        assert list(landing.execution_slots) == [-1]
        assert list(landing.execution_slot_fresh) == [False]


def _ack(scheduler, op, kind) -> None:
    for op_id in op.op_ids:
        event = kind()
        event.op_id = int(op_id)
        _send(scheduler, event)


def _forward(plan):
    for batch in plan.forward:
        if batch.request_ids:
            return batch
    return None


def _rows(plan) -> dict[str, tuple[int, int, bool]]:
    """request id -> (request row, execution slot, fresh) of the plan's batch."""
    batch = _forward(plan)
    if batch is None:
        return {}
    ids = list(batch.request_ids)
    rows = list(batch.request_pool_indices)
    slots = list(batch.execution_slots)
    fresh = list(batch.execution_slot_fresh)
    assert len(rows) == len(slots) == len(fresh) == len(ids)
    return {rid: (rows[i], slots[i], fresh[i]) for i, rid in enumerate(ids)}


def _only(plan, request_id: str) -> tuple[int, bool]:
    rows = _rows(plan)
    assert list(rows) == [request_id]
    _, slot, fresh = rows[request_id]
    return slot, fresh


def _cache_ops(plan, kind) -> list:
    return [op for op in plan.cache if isinstance(op, kind)]


def test_decode_role_slots_are_permits_distinct_stable_and_fresh_once():
    scheduler = ts.Scheduler(_history_config(Role.D, 2, 4))
    _submit_pd(scheduler, "a", "b", "c", "d")
    _admit(scheduler, "a", "b", "c", "d")
    # The two newest requests hold request rows beyond the batch size.
    _land(scheduler, "c", "d")

    first = _rows(scheduler.next_execution_plan())
    assert set(first) == {"c", "d"}
    assert {row for row, _, _ in first.values()} == {3, 4}
    slots = {rid: slot for rid, (_, slot, _) in first.items()}
    assert sorted(slots.values()) == [1, 2]
    assert all(fresh for _, _, fresh in first.values())

    for token in (43, 44, 45):
        for rid in ("c", "d"):
            _advance(scheduler, rid, [token])
        rows = _rows(scheduler.next_execution_plan())
        assert {rid: slot for rid, (_, slot, _) in rows.items()} == slots
        assert not any(fresh for _, _, fresh in rows.values())

    # c's permit goes to the oldest ready request, fresh there; d keeps its own.
    _land(scheduler, "a", "b")
    for rid in ("c", "d"):
        _advance(scheduler, rid, [46])
    _finish(scheduler, "c")
    rows = _rows(scheduler.next_execution_plan())
    assert rows["a"][1:] == (slots["c"], True)
    assert rows["d"][1:] == (slots["d"], False)


@pytest.mark.parametrize("terminal", ["Finish", "Abort"])
def test_a_released_slot_is_reused_only_after_its_results_land(terminal):
    scheduler = ts.Scheduler(_history_config(Role.D, 1, 2))
    _submit_pd(scheduler, "running", "waiting")
    _admit(scheduler, "running", "waiting")
    _land(scheduler, "running", "waiting")
    assert _only(scheduler.next_execution_plan(), "running") == (1, True)
    # Overlap: the next round is planned before the first one's result.
    assert _only(scheduler.next_execution_plan(), "running") == (1, False)
    _send(scheduler, _event(terminal, "running"))
    assert _rows(scheduler.next_execution_plan()) == {}
    _advance(scheduler, "running", [43])
    assert _rows(scheduler.next_execution_plan()) == {}
    _advance(scheduler, "running", [44])
    assert _only(scheduler.next_execution_plan(), "waiting") == (1, True)


def test_a_restored_decode_is_fresh_on_its_first_decode_only():
    cfg = _history_config(Role.D, 1, 3)
    cfg.debug_force_retraction_interval = 5
    cfg.debug_force_retraction_max_per_request = 1
    scheduler = ts.Scheduler(cfg)
    _submit_pd(scheduler, "retracted", "running")
    _admit(scheduler, "retracted", "running")  # plans 1, 2
    _land(scheduler, "retracted")
    assert _only(scheduler.next_execution_plan(), "retracted") == (1, True)
    _advance(scheduler, "retracted", [43])
    assert _only(scheduler.next_execution_plan(), "retracted") == (1, False)
    _advance(scheduler, "retracted", [44])
    retract = scheduler.next_execution_plan()  # 5
    [snapshot] = _cache_ops(retract, ts.Cache.SnapshotOp)
    assert _rows(retract) == {}

    # The released permit starts over for the next request to decode.
    _land(scheduler, "running")
    assert _only(scheduler.next_execution_plan(), "running") == (1, True)
    _advance(scheduler, "running", [43])
    _ack(scheduler, snapshot, ts.Cache.SnapshotDoneEvent)
    busy = scheduler.next_execution_plan()
    assert _only(busy, "running") == (1, False)
    assert not _cache_ops(busy, ts.Cache.RestoreOp)
    _advance(scheduler, "running", [44])
    _finish(scheduler, "running")
    restore = scheduler.next_execution_plan()
    [restore_op] = _cache_ops(restore, ts.Cache.RestoreOp)
    assert _rows(restore) == {}
    _ack(scheduler, restore_op, ts.Cache.RestoreDoneEvent)

    resumed = scheduler.next_execution_plan()
    assert _only(resumed, "retracted") == (1, True)
    assert list(_forward(resumed).decode_input_ids) == [44]
    _advance(scheduler, "retracted", [45])
    assert _only(scheduler.next_execution_plan(), "retracted") == (1, False)


def _deferred_decode_config() -> ts.SchedulerConfig:
    """D role, one decode permit, two residents, deferred State groups."""
    cfg = _make_k3_config()
    cfg.role = Role.D
    cfg.max_batch_size = 1
    cfg.max_resident_requests = 2
    cfg.num_snapshot_pages = 33
    cfg.max_retracted_requests = 2
    cfg.debug_force_retraction_interval = 5
    cfg.debug_force_retraction_max_per_request = 1
    cfg.host_group_packing = [1] * len(cfg.cache_groups)
    groups = cfg.cache_groups
    for group in groups:
        stateful = group.group_id in K3_STATE_GROUPS
        group.deferred_decode_state = stateful
        group.transfer_policy = (
            ts.CacheTransferPolicy.LatestSnapshot
            if stateful
            else ts.CacheTransferPolicy.FullSuffix
        )
    cfg.cache_groups = groups
    return cfg


def test_a_retraction_folds_from_the_victims_slot_before_the_same_plan_reuses_it():
    scheduler = ts.Scheduler(_deferred_decode_config())
    # next is admitted first, so the victim's request row differs from its permit.
    _submit_pd(scheduler, "next", "victim")
    _admit(scheduler, "next", "victim")  # plans 1, 2
    _land(scheduler, "victim")
    assert _only(scheduler.next_execution_plan(), "victim") == (1, True)
    _advance(scheduler, "victim", [43])
    decode = _forward(scheduler.next_execution_plan())  # 4
    [row] = list(decode.request_pool_indices)
    assert row == 2
    assert row != 1  # the victim's permit
    tables = dict(decode.block_tables)
    live = {group: int(tables[group][0][1]) for group in K3_STATE_GROUPS}
    _advance(scheduler, "victim", [44])
    _land(scheduler, "next")

    retract = scheduler.next_execution_plan()  # 5: forced retraction of victim
    assert scheduler.retracted_size() == 1
    folds = {
        group: folds.tolist() for group, folds in retract.state_folds_arrays().items()
    }
    assert folds == {group: [[row, page, 1]] for group, page in live.items()}
    for group, fold_rows in retract.state_folds.items():
        assert [
            (fold.request_slot, fold.page, fold.execution_slot) for fold in fold_rows
        ] == [(row, live[group], 1)]
    # The permit the retraction released starts over for its next owner.
    assert _only(retract, "next") == (1, True)


def _folds(plan) -> dict[str, list[list[int]]]:
    arrays = {
        group: folds.tolist() for group, folds in plan.state_folds_arrays().items()
    }
    listed = {
        group: [[f.request_slot, f.page, f.execution_slot] for f in fold_rows]
        for group, fold_rows in plan.state_folds.items()
    }
    assert arrays == listed
    return arrays


def _decode_until_retracted(scheduler, tokens) -> tuple:
    """Decodes the victim once per token, then plans the forced retraction.

    Returns the retraction plan and the folds the last decode's records need.
    """
    expected = {}
    for token in tokens:
        decode = _forward(scheduler.next_execution_plan())
        assert list(decode.request_ids) == ["victim"]
        [row] = list(decode.request_pool_indices)
        [slot] = list(decode.execution_slots)
        tables = dict(decode.block_tables)
        expected = {
            group: [[row, int(tables[group][0][1]), slot]] for group in K3_STATE_GROUPS
        }
        _advance(scheduler, "victim", [token])
    retract = scheduler.next_execution_plan()
    assert _rows(retract) == {}
    assert scheduler.retracted_size() == 1
    return retract, expected


def test_a_restored_decode_retracted_before_its_first_forward_folds_nothing():
    cfg = _deferred_decode_config()
    cfg.debug_force_retraction_max_per_request = 3
    scheduler = ts.Scheduler(cfg)
    _submit_pd(scheduler, "victim")
    _admit(scheduler, "victim")  # plan 1
    _land(scheduler, "victim")

    # Plans 2-4 decode, 5 retracts: the folds carry the decodes' records.
    first, expected = _decode_until_retracted(scheduler, [43, 44, 45])
    assert _folds(first) == expected
    assert all(fold[2] == 1 for folds in expected.values() for fold in folds)
    [snapshot] = _cache_ops(first, ts.Cache.SnapshotOp)
    for _ in range(2):  # 6-7: the image has not landed
        idle = scheduler.next_execution_plan()
        assert _rows(idle) == {}
        assert not _cache_ops(idle, ts.Cache.RestoreOp)
    _ack(scheduler, snapshot, ts.Cache.SnapshotDoneEvent)
    restore = scheduler.next_execution_plan()  # 8
    [restore_op] = _cache_ops(restore, ts.Cache.RestoreOp)
    _ack(scheduler, restore_op, ts.Cache.RestoreDoneEvent)
    # Drop the entries the first retraction published. Otherwise the next
    # publication swaps the live blocks for those equal entries, and no fold
    # is listed whatever the slot holds. A successful clear also advances the
    # plan count (9), which the retraction interval counts.
    assert scheduler.clear_cache()

    # 10: the knob retracts the restored decode before its first resumed
    # forward. Its permit has run nothing for it, so nothing is folded, while
    # its image is still taken.
    second = scheduler.next_execution_plan()
    assert _rows(second) == {}
    assert scheduler.retracted_size() == 1
    assert _folds(second) == {}
    [snapshot] = _cache_ops(second, ts.Cache.SnapshotOp)
    _ack(scheduler, snapshot, ts.Cache.SnapshotDoneEvent)
    restore = scheduler.next_execution_plan()  # 11
    [restore_op] = _cache_ops(restore, ts.Cache.RestoreOp)
    _ack(scheduler, restore_op, ts.Cache.RestoreDoneEvent)

    # 12: the resumed decode is fresh and carries the last token it generated.
    resumed = scheduler.next_execution_plan()
    assert _only(resumed, "victim") == (1, True)
    assert list(_forward(resumed).decode_input_ids) == [45]
    _advance(scheduler, "victim", [46])

    # 13, 14 decode, 15 retracts: once it forwarded, its records are folded again.
    third, expected = _decode_until_retracted(scheduler, [47, 48])
    assert expected
    assert _folds(third) == expected


def test_fused_slots_are_request_rows_fresh_on_admission_and_restore():
    cfg = _history_config(Role.Fused, 1, 1)
    cfg.max_scheduled_tokens = 4
    cfg.debug_force_retraction_interval = 5
    cfg.debug_force_retraction_max_per_request = 1
    scheduler = ts.Scheduler(cfg)
    scheduler.submit_requests([_spec("running", list(range(1, 9)))])

    first = _rows(scheduler.next_execution_plan())  # 1: first chunk
    [(row, slot, fresh)] = first.values()
    assert (slot, fresh) == (row, True)
    _advance(scheduler, "running", [])  # an intermediate chunk yields no token
    second = _rows(scheduler.next_execution_plan())  # 2: last chunk
    assert second == {"running": (row, row, False)}
    _advance(scheduler, "running", [42])
    for _ in range(2):  # 3, 4: decode rounds
        assert _rows(scheduler.next_execution_plan()) == {"running": (row, row, False)}
        _advance(scheduler, "running", [43])

    retract = scheduler.next_execution_plan()  # 5
    [snapshot] = _cache_ops(retract, ts.Cache.SnapshotOp)
    assert _rows(retract) == {}
    _ack(scheduler, snapshot, ts.Cache.SnapshotDoneEvent)
    restore = scheduler.next_execution_plan()
    [restore_op] = _cache_ops(restore, ts.Cache.RestoreOp)
    _ack(scheduler, restore_op, ts.Cache.RestoreDoneEvent)

    resumed = _rows(scheduler.next_execution_plan())
    [(row, slot, fresh)] = resumed.values()
    assert (slot, fresh) == (row, True)
    _advance(scheduler, "running", [44])
    assert _rows(scheduler.next_execution_plan()) == {"running": (row, row, False)}


def test_fused_mixed_batch_keeps_slots_aligned_after_prefill_rows_move_first():
    cfg = _history_config(Role.Fused, 4, 4)
    cfg.enable_mixed_prefill_decode = True
    scheduler = ts.Scheduler(cfg)
    scheduler.submit_requests([_spec("decoding", list(range(1, 9)))])
    scheduler.next_execution_plan()
    _advance(scheduler, "decoding", [42])
    scheduler.next_execution_plan()
    _advance(scheduler, "decoding", [43])

    scheduler.submit_requests([_spec("new", list(range(11, 19)))])
    batch = _forward(scheduler.next_execution_plan())
    assert list(batch.request_ids) == ["new", "decoding"]
    assert batch.num_extends() == 1
    assert list(batch.execution_slots) == list(batch.request_pool_indices)
    assert list(batch.execution_slot_fresh) == [True, False]


def test_prefill_role_slots_are_request_rows_and_its_handoff_carries_none():
    scheduler = ts.Scheduler(_history_config(Role.P, 2, 2))
    _submit_pd(scheduler, "a", "b")
    prefill = _rows(scheduler.next_execution_plan())
    assert set(prefill) == {"a", "b"}
    for row, slot, fresh in prefill.values():
        assert (slot, fresh) == (row, True)
    for rid in ("a", "b"):
        _advance(scheduler, rid, [5])

    handoff = scheduler.next_execution_plan().remote_decode
    assert sorted(handoff.request_ids) == ["a", "b"]
    assert list(handoff.execution_slots) == [-1, -1]
    assert list(handoff.execution_slot_fresh) == [False, False]
