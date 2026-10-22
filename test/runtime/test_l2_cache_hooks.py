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

"""CPU-only tests for CacheOpHooks (Host L2 and retraction snapshot ops).

Single-rank cases (every replica group size 1) skip collectives. Cross-rank
WriteBackDone agreement is driven here with mocked all_reduce /
all_gather_object so a finished L3 backup cannot CacheHostBlock on one
mirrored scheduler while a CP/PP peer still has the op pending.
"""

from __future__ import annotations

import ast
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.engine.scheduler_utils import cache_ops_from_plan

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed_scheduler import Cache  # noqa: E402

from tokenspeed.runtime.engine import cache_hooks as cache_hooks_module  # noqa: E402
from tokenspeed.runtime.engine import (  # noqa: E402
    scheduler_utils as scheduler_utils_module,
)
from tokenspeed.runtime.engine.cache_hooks import (  # noqa: E402
    CacheOpHooks,
    cache_hooks_armed,
)


class _FakeWriteBackOp:
    def __init__(self, op_ids) -> None:
        self.op_ids = op_ids


class _FakeSnapshotBatch:
    """The wire face of ``Cache.SnapshotOp``: one row per retracted request."""

    def __init__(self, op_ids) -> None:
        self.op_ids = list(op_ids)
        self.request_ids = [f"r{op_id}" for op_id in op_ids]
        self.request_pool_indices = [1] * len(op_ids)
        self.snapshot_slots = list(range(len(op_ids)))
        self.group_ids = [[0]] * len(op_ids)
        self.src_pages = [[1]] * len(op_ids)
        self.dst_pages = [[1]] * len(op_ids)


class _FakeRestoreBatch(_FakeSnapshotBatch):
    def __init__(self, op_ids) -> None:
        super().__init__(op_ids)
        self.content_hashes = [[""]] * len(op_ids)
        self.page_offsets = [[0]] * len(op_ids)
        self.source_tiers = [[1]] * len(op_ids)


class _Device:
    """The DeviceHandle surface the hooks use. Submission rides
    ``DeviceHandle.execute`` with the rest of the round's plan-derived device
    work; this side only counts and polls."""

    def __init__(self) -> None:
        self.results: list = []
        self.backup_failed = False

    def poll_cache_results(self) -> list:
        results, self.results = self.results, []
        return results

    def consume_l3_backup_poll_failure(self) -> bool:
        failed = self.backup_failed
        self.backup_failed = False
        return failed


def _hooks(
    device,
    speculative_algorithm,
    attn_tp_size,
    attn_tp_cpu_group,
    pp_size,
    pp_cpu_group,
) -> CacheOpHooks:
    return CacheOpHooks(
        device,
        speculative_algorithm=speculative_algorithm,
        attn_tp_rank=0,
        attn_tp_size=attn_tp_size,
        attn_tp_cpu_group=attn_tp_cpu_group,
        pp_size=pp_size,
        pp_cpu_group=pp_cpu_group,
        global_rank=0,
    )


def _single_rank_hooks(device) -> CacheOpHooks:
    return _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pp_size=1,
        pp_cpu_group=None,
    )


def _writeback_done_event(op_id: int):
    return _cache_event("WriteBackDoneEvent", op_id)


def _cache_event(kind: str, op_id: int):
    event = getattr(Cache, kind)()
    event.op_id = op_id
    return event


def _install_collectives(
    monkeypatch: pytest.MonkeyPatch,
    *,
    all_reduce,
    all_gather_object,
) -> None:
    monkeypatch.setattr(cache_hooks_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(cache_hooks_module.dist, "all_gather_object", all_gather_object)


@pytest.fixture()
def fake_cache_ops(monkeypatch: pytest.MonkeyPatch):
    # The C++ op bindings (Cache.WriteBackOp, Cache.SnapshotOp, ...) expose no
    # Python constructor, so substitute the types the plan adapter
    # (scheduler_utils.cache_ops_from_plan) dispatches on.
    monkeypatch.setattr(
        scheduler_utils_module,
        "Cache",
        SimpleNamespace(
            WriteBackOp=_FakeWriteBackOp,
            LoadBackOp=(),
            SnapshotOp=_FakeSnapshotBatch,
            RestoreOp=_FakeRestoreBatch,
            StorageQueryOp=Cache.StorageQueryOp,
            StorageGetOp=Cache.StorageGetOp,
            StoragePutOp=Cache.StoragePutOp,
            StorageCancelOp=Cache.StorageCancelOp,
        ),
    )


def test_disabled_kvstore_is_a_no_op() -> None:
    hooks = _single_rank_hooks(None)
    hooks.count_ops(())
    assert hooks.poll_ready_events() == []


def test_hooks_arm_for_l2_and_for_the_retraction_snapshot_pool() -> None:
    # Armed hooks all-reduce every round, so only an engine that can emit
    # cache ops arms them: the L2 tier, or a pool the scheduler retracts into.
    assert cache_hooks_armed(enable_kvstore=True, max_retracted_requests=0)
    assert cache_hooks_armed(enable_kvstore=True, max_retracted_requests=4)
    assert cache_hooks_armed(enable_kvstore=False, max_retracted_requests=4)
    assert not cache_hooks_armed(enable_kvstore=False, max_retracted_requests=0)


def test_submit_counts_in_flight_and_rejects_unknown_ops(fake_cache_ops) -> None:
    device = _Device()
    hooks = _single_rank_hooks(device)
    plan = SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[1, 2])])

    hooks.count_ops(cache_ops_from_plan(plan).device)

    assert hooks._num_inflight == 2

    with pytest.raises(TypeError, match="unsupported cache op kind"):
        hooks.count_ops(cache_ops_from_plan(SimpleNamespace(cache=[object()])).device)


def test_snapshot_and_restore_batches_count_one_ticket_per_request(
    fake_cache_ops,
) -> None:
    # A snapshot store or restore batch carries one op id per retracted
    # request and each is acknowledged on its own, so each counts once --
    # unlike an L2 batch, whose tickets are its op_ids, these are expanded
    # to per-request ops by the plan adapter first.
    device = _Device()
    hooks = _single_rank_hooks(device)
    plan = SimpleNamespace(
        cache=[
            _FakeWriteBackOp(op_ids=[1]),
            _FakeSnapshotBatch(op_ids=[2, 3]),
            _FakeRestoreBatch(op_ids=[4]),
            _FakeSnapshotBatch(op_ids=[5, 6]),
        ]
    )

    hooks.count_ops(cache_ops_from_plan(plan).device)

    assert hooks._num_inflight == 6
    for event in (
        _cache_event("SnapshotDoneEvent", 2),
        _cache_event("RestoreDoneEvent", 4),
        _cache_event("SnapshotDoneEvent", 5),
    ):
        device.results = [event]
        (ready,) = hooks.poll_ready_events()
        assert type(ready).__name__ == type(event).__name__
        assert ready.op_id == event.op_id
    assert hooks._num_inflight == 3


def test_poll_returns_completed_events_and_settles_inflight(fake_cache_ops) -> None:
    device = _Device()
    hooks = _single_rank_hooks(device)
    hooks.count_ops(
        cache_ops_from_plan(
            SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[7])])
        ).device
    )

    # Nothing completed yet: in flight, but no ready payloads.
    assert hooks.poll_ready_events() == []

    device.results = [_writeback_done_event(7)]
    events = hooks.poll_ready_events()

    assert [type(e).__name__ for e in events] == ["WriteBackDoneEvent"]
    assert events[0].op_id == 7
    assert hooks._num_inflight == 0
    # Settled: the idle short-circuit now skips polling work entirely.
    assert hooks.poll_ready_events() == []
    assert hooks._pending_payloads == {}


def test_pp_peer_missing_writeback_keeps_payload_pending(
    fake_cache_ops, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _Device()
    pp_group = object()
    hooks = _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pp_size=2,
        pp_cpu_group=pp_group,
    )
    hooks.count_ops(
        cache_ops_from_plan(
            SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[7])])
        ).device
    )
    device.results = [_writeback_done_event(7)]

    gather_groups: list = []

    def _all_reduce(tensor, *, op, group) -> None:
        del tensor, op, group

    def _all_gather_object(output, obj, *, group) -> None:
        gather_groups.append(group)
        output[0] = list(obj)
        output[1] = []

    _install_collectives(
        monkeypatch, all_reduce=_all_reduce, all_gather_object=_all_gather_object
    )

    assert hooks.poll_ready_events() == []
    assert gather_groups == [pp_group]
    assert ("WriteBackDoneEvent", 7) in hooks._pending_payloads


def test_pp_agreed_writeback_emits_event(
    fake_cache_ops, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _Device()
    pp_group = object()
    hooks = _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pp_size=2,
        pp_cpu_group=pp_group,
    )
    hooks.count_ops(
        cache_ops_from_plan(
            SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[7])])
        ).device
    )
    device.results = [_writeback_done_event(7)]

    gather_groups: list = []

    def _all_reduce(tensor, *, op, group) -> None:
        del tensor, op, group

    def _all_gather_object(output, obj, *, group) -> None:
        gather_groups.append(group)
        output[0] = list(obj)
        output[1] = list(obj)

    _install_collectives(
        monkeypatch, all_reduce=_all_reduce, all_gather_object=_all_gather_object
    )

    events = hooks.poll_ready_events()
    assert gather_groups == [pp_group]
    assert [type(e).__name__ for e in events] == ["WriteBackDoneEvent"]
    assert events[0].op_id == 7
    assert hooks._pending_payloads == {}


def test_gather_order_is_attention_tp_then_pp(
    fake_cache_ops, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _Device()
    tp_group = object()
    pp_group = object()
    hooks = _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=2,
        attn_tp_cpu_group=tp_group,
        pp_size=2,
        pp_cpu_group=pp_group,
    )
    hooks.count_ops(
        cache_ops_from_plan(
            SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[7])])
        ).device
    )
    device.results = [_writeback_done_event(7)]

    reduce_groups: list = []
    gather_groups: list = []

    def _all_reduce(tensor, *, op, group) -> None:
        del tensor, op
        reduce_groups.append(group)

    def _all_gather_object(output, obj, *, group) -> None:
        gather_groups.append(group)
        for i in range(len(output)):
            output[i] = list(obj)

    _install_collectives(
        monkeypatch, all_reduce=_all_reduce, all_gather_object=_all_gather_object
    )

    events = hooks.poll_ready_events()
    assert reduce_groups == [tp_group, pp_group]
    assert gather_groups == [tp_group, pp_group]
    assert [type(e).__name__ for e in events] == ["WriteBackDoneEvent"]
    assert events[0].op_id == 7


def test_empty_tp_intersection_still_gathers_pp(
    fake_cache_ops, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _Device()
    tp_group = object()
    pp_group = object()
    hooks = _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=2,
        attn_tp_cpu_group=tp_group,
        pp_size=2,
        pp_cpu_group=pp_group,
    )
    hooks.count_ops(
        cache_ops_from_plan(
            SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[7])])
        ).device
    )
    device.results = [_writeback_done_event(7)]

    gather_groups: list = []
    gather_objs: list = []
    peer_ready = [{"kind": "WriteBackDoneEvent", "op_id": 7}]

    def _all_reduce(tensor, *, op, group) -> None:
        del tensor, op, group

    def _all_gather_object(output, obj, *, group) -> None:
        gather_groups.append(group)
        gather_objs.append(list(obj))
        output[0] = list(obj)
        # TP peer has not finished; PP peers whose TP group already
        # agreed still enter the later gathers.
        output[1] = [] if group is tp_group else peer_ready

    _install_collectives(
        monkeypatch, all_reduce=_all_reduce, all_gather_object=_all_gather_object
    )

    assert hooks.poll_ready_events() == []
    assert gather_groups == [tp_group, pp_group]
    assert len(gather_objs[0]) == 1
    assert gather_objs[1] == []
    assert ("WriteBackDoneEvent", 7) in hooks._pending_payloads


def test_replica_runs_collectives_only_while_a_ticket_is_open(
    fake_cache_ops, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an L3 store the open-ticket count mirrors across the replica, so
    an idle replica polls nothing, and a committed op closes its ticket."""
    device = _Device()
    pp_group = object()
    hooks = _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pp_size=2,
        pp_cpu_group=pp_group,
    )
    collectives: list = []

    def _all_reduce(tensor, *, op, group) -> None:
        del op
        collectives.append("reduce")
        tensor[0] = 1

    def _all_gather_object(output, obj, *, group) -> None:
        del group
        collectives.append("gather")
        output[0] = list(obj)
        output[1] = list(obj)

    _install_collectives(
        monkeypatch, all_reduce=_all_reduce, all_gather_object=_all_gather_object
    )

    assert hooks.poll_ready_events() == [] and collectives == []
    hooks.count_ops(
        cache_ops_from_plan(
            SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[7])])
        ).device
    )
    device.results = [_writeback_done_event(7)]
    assert [e.op_id for e in hooks.poll_ready_events()] == [7]
    assert collectives == ["reduce", "gather"]
    assert hooks.poll_ready_events() == [] and collectives == ["reduce", "gather"]


def test_tickets_stay_open_until_the_replica_commits_them(
    fake_cache_ops, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completion seen here keeps its ticket open (and the replica polling)
    until every peer has it; a batch closes ticket by ticket."""
    device = _Device()
    pp_group = object()
    hooks = _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pp_size=2,
        pp_cpu_group=pp_group,
    )
    collectives: list = []
    peer_ready: list = []

    def _all_reduce(tensor, *, op, group) -> None:
        del op, group
        collectives.append("reduce")
        tensor[0] = 1

    def _all_gather_object(output, obj, *, group) -> None:
        del group
        collectives.append("gather")
        output[0] = list(obj)
        output[1] = [p for p in obj if p["op_id"] in peer_ready]

    _install_collectives(
        monkeypatch, all_reduce=_all_reduce, all_gather_object=_all_gather_object
    )
    hooks.count_ops(
        cache_ops_from_plan(
            SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[7, 8])])
        ).device
    )
    device.results = [_writeback_done_event(7), _writeback_done_event(8)]
    assert hooks.poll_ready_events() == []  # the peer has neither yet
    assert hooks.poll_ready_events() == []
    assert collectives == ["reduce", "gather"] * 2
    peer_ready.append(7)
    assert [e.op_id for e in hooks.poll_ready_events()] == [7]
    peer_ready.append(8)
    assert [e.op_id for e in hooks.poll_ready_events()] == [8]
    collectives.clear()
    assert hooks.poll_ready_events() == [] and collectives == []


def test_d2h_ack_is_independent_of_backend_health(fake_cache_ops):
    device = _Device()
    hooks = _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pp_size=1,
        pp_cpu_group=None,
    )
    hooks.count_ops(
        cache_ops_from_plan(
            SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[7])])
        ).device
    )
    event = Cache.WriteBackDoneEvent()
    event.op_id = 7
    device.results = [event]
    device.backup_failed = True
    assert [x.op_id for x in hooks.poll_ready_events()] == [7]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


class _GridGroup:
    """A CPU group of the simulated grid: a barrier-backed rendezvous whose
    collectives time out (and fail the test) if any member skips them."""

    def __init__(self, members: list[int]) -> None:
        self.members = members
        self.slots: dict[int, object] = {}
        self.barrier = threading.Barrier(len(members), timeout=10)


def test_tp_pp_grid_counts_stay_equal_and_all_ranks_stop_polling_together(
    fake_cache_ops, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four independent hooks (attention TP 2 x PP 2) count the same plans while
    completions reach each rank in different rounds; every round commits the
    same ops with the same open count everywhere, and once all tickets close
    no rank enters a collective, so none waits on a peer that skipped."""
    tp_groups = {0: _GridGroup([0, 1]), 1: _GridGroup([2, 3])}
    pp_groups = {0: _GridGroup([0, 2]), 1: _GridGroup([1, 3])}
    local = threading.local()
    collectives = {rank: 0 for rank in range(4)}

    def _rendezvous(group: _GridGroup, value):
        rank = local.rank
        collectives[rank] += 1
        group.slots[rank] = value
        group.barrier.wait()
        gathered = [group.slots[member] for member in group.members]
        group.barrier.wait()
        return gathered

    def _all_reduce(tensor, *, op, group) -> None:
        del op
        gathered = _rendezvous(group, tensor.clone())
        tensor.copy_(torch.stack(gathered).max(dim=0).values)

    def _all_gather_object(output, obj, *, group) -> None:
        output[:] = _rendezvous(group, list(obj))

    _install_collectives(
        monkeypatch, all_reduce=_all_reduce, all_gather_object=_all_gather_object
    )

    # Ticket -> (event kind, round in which each rank sees it complete).
    arrivals = {
        7: ("WriteBackDoneEvent", [1, 3, 2, 1]),
        8: ("WriteBackDoneEvent", [2, 1, 4, 2]),
        20: ("SnapshotDoneEvent", [1, 1, 1, 5]),
        21: ("SnapshotDoneEvent", [3, 2, 1, 1]),
    }
    plan = SimpleNamespace(
        cache=[_FakeWriteBackOp(op_ids=[7, 8]), _FakeSnapshotBatch([20, 21])]
    )
    num_rounds = 9
    committed = {rank: [] for rank in range(4)}
    open_counts = {rank: [] for rank in range(4)}
    collectives_at_close: dict[int, int] = {}
    errors: list = []
    round_barrier = threading.Barrier(4, timeout=30)

    def _run(rank: int) -> None:
        local.rank = rank
        device = _Device()
        hooks = CacheOpHooks(
            device,
            speculative_algorithm=None,
            attn_tp_rank=rank % 2,
            attn_tp_size=2,
            attn_tp_cpu_group=tp_groups[rank // 2],
            pp_size=2,
            pp_cpu_group=pp_groups[rank % 2],
            global_rank=rank,
        )
        try:
            hooks.count_ops(cache_ops_from_plan(plan).device)
            for round_idx in range(num_rounds):
                device.results = [
                    _cache_event(kind, op_id)
                    for op_id, (kind, rounds) in arrivals.items()
                    if rounds[rank] == round_idx
                ]
                committed[rank].append(
                    sorted(event.op_id for event in hooks.poll_ready_events())
                )
                open_counts[rank].append(hooks._num_open)
                if round_idx == 5:
                    collectives_at_close[rank] = collectives[rank]
                round_barrier.wait()
        except Exception as exc:  # BrokenBarrierError on a skipped collective
            errors.append((rank, exc))
            round_barrier.abort()

    threads = [threading.Thread(target=_run, args=(rank,)) for rank in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    for rank in range(1, 4):
        assert committed[rank] == committed[0]
        assert open_counts[rank] == open_counts[0]
    assert sorted(op for ops in committed[0] for op in ops) == [7, 8, 20, 21]
    # Each ticket commits once every rank has it: the latest arrival among ranks.
    assert committed[0][4] == [8] and committed[0][5] == [20]
    assert committed[0][3] == [7, 21]
    assert open_counts[0][5:] == [0] * (num_rounds - 5)
    # The last ticket closed in round 5; rounds 6-8 entered no collective anywhere.
    assert collectives == collectives_at_close
