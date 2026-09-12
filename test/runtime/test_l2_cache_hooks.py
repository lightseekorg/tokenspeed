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

"""CPU-only tests for L2CacheHooks.

Single-rank cases (every replica group size 1) skip collectives. Cross-rank
WriteBackDone agreement is driven here with mocked all_reduce /
all_gather_object so a finished L3 backup cannot CacheHostBlock on one
mirrored scheduler while a CP/PP peer still has the op pending.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed_scheduler import Cache  # noqa: E402

from tokenspeed.runtime.engine import cache_hooks as cache_hooks_module  # noqa: E402
from tokenspeed.runtime.engine.cache_hooks import L2CacheHooks  # noqa: E402


class _FakeWriteBackOp:
    def __init__(self, op_ids) -> None:
        self.op_ids = op_ids


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
    attn_cp_size,
    attn_cp_cpu_group,
    pp_size,
    pp_cpu_group,
) -> L2CacheHooks:
    return L2CacheHooks(
        device,
        speculative_algorithm=speculative_algorithm,
        attn_tp_rank=0,
        attn_tp_size=attn_tp_size,
        attn_tp_cpu_group=attn_tp_cpu_group,
        attn_cp_size=attn_cp_size,
        attn_cp_cpu_group=attn_cp_cpu_group,
        pp_size=pp_size,
        pp_cpu_group=pp_cpu_group,
        global_rank=0,
    )


def _single_rank_hooks(device) -> L2CacheHooks:
    return _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        attn_cp_size=1,
        attn_cp_cpu_group=None,
        pp_size=1,
        pp_cpu_group=None,
    )


def _writeback_done_event(op_id: int):
    event = Cache.WriteBackDoneEvent()
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
    # The C++ op bindings (Cache.WriteBackOp) expose no Python constructor, so
    # substitute the type the isinstance check dispatches on.
    monkeypatch.setattr(
        cache_hooks_module,
        "Cache",
        SimpleNamespace(WriteBackOp=_FakeWriteBackOp, LoadBackOp=()),
    )


def test_disabled_kvstore_is_a_no_op() -> None:
    hooks = _single_rank_hooks(None)
    hooks.count_plan_ops(SimpleNamespace(cache=[SimpleNamespace()]))
    assert hooks.poll_ready_events() == []


def test_submit_counts_in_flight_and_rejects_unknown_ops(fake_cache_ops) -> None:
    device = _Device()
    hooks = _single_rank_hooks(device)
    plan = SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[1, 2])])

    hooks.count_plan_ops(plan)

    assert hooks._num_inflight == 2

    with pytest.raises(TypeError, match="unsupported cache op kind"):
        hooks.count_plan_ops(SimpleNamespace(cache=[object()]))


def test_poll_returns_completed_events_and_settles_inflight(fake_cache_ops) -> None:
    device = _Device()
    hooks = _single_rank_hooks(device)
    hooks.count_plan_ops(SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[7])]))

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


def test_cp_peer_missing_writeback_keeps_payload_pending(
    fake_cache_ops, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _Device()
    cp_group = object()
    hooks = _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        attn_cp_size=2,
        attn_cp_cpu_group=cp_group,
        pp_size=1,
        pp_cpu_group=None,
    )
    hooks.count_plan_ops(SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[7])]))
    device.results = [_writeback_done_event(7)]

    gather_groups: list = []

    def _all_reduce(tensor, op=None, group=None) -> None:
        del tensor, op, group

    def _all_gather_object(output, obj, group=None) -> None:
        gather_groups.append(group)
        output[0] = list(obj)
        output[1] = []

    _install_collectives(
        monkeypatch, all_reduce=_all_reduce, all_gather_object=_all_gather_object
    )

    assert hooks.poll_ready_events() == []
    assert gather_groups == [cp_group]
    assert ("WriteBackDoneEvent", 7) in hooks._pending_payloads


def test_cp_agreed_writeback_emits_event(
    fake_cache_ops, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _Device()
    cp_group = object()
    hooks = _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        attn_cp_size=2,
        attn_cp_cpu_group=cp_group,
        pp_size=1,
        pp_cpu_group=None,
    )
    hooks.count_plan_ops(SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[7])]))
    device.results = [_writeback_done_event(7)]

    gather_groups: list = []

    def _all_reduce(tensor, op=None, group=None) -> None:
        del tensor, op, group

    def _all_gather_object(output, obj, group=None) -> None:
        gather_groups.append(group)
        output[0] = list(obj)
        output[1] = list(obj)

    _install_collectives(
        monkeypatch, all_reduce=_all_reduce, all_gather_object=_all_gather_object
    )

    events = hooks.poll_ready_events()
    assert gather_groups == [cp_group]
    assert [type(e).__name__ for e in events] == ["WriteBackDoneEvent"]
    assert events[0].op_id == 7
    assert hooks._pending_payloads == {}


def test_gather_order_is_attention_tp_then_cp_then_pp(
    fake_cache_ops, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _Device()
    tp_group = object()
    cp_group = object()
    pp_group = object()
    hooks = _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=2,
        attn_tp_cpu_group=tp_group,
        attn_cp_size=2,
        attn_cp_cpu_group=cp_group,
        pp_size=2,
        pp_cpu_group=pp_group,
    )
    hooks.count_plan_ops(SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[7])]))
    device.results = [_writeback_done_event(7)]

    reduce_groups: list = []
    gather_groups: list = []

    def _all_reduce(tensor, op=None, group=None) -> None:
        del tensor, op
        reduce_groups.append(group)

    def _all_gather_object(output, obj, group=None) -> None:
        gather_groups.append(group)
        for i in range(len(output)):
            output[i] = list(obj)

    _install_collectives(
        monkeypatch, all_reduce=_all_reduce, all_gather_object=_all_gather_object
    )

    events = hooks.poll_ready_events()
    assert reduce_groups == [tp_group, cp_group, pp_group]
    assert gather_groups == [tp_group, cp_group, pp_group]
    assert [type(e).__name__ for e in events] == ["WriteBackDoneEvent"]
    assert events[0].op_id == 7


def test_empty_tp_intersection_still_gathers_cp_and_pp(
    fake_cache_ops, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _Device()
    tp_group = object()
    cp_group = object()
    pp_group = object()
    hooks = _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=2,
        attn_tp_cpu_group=tp_group,
        attn_cp_size=2,
        attn_cp_cpu_group=cp_group,
        pp_size=2,
        pp_cpu_group=pp_group,
    )
    hooks.count_plan_ops(SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[7])]))
    device.results = [_writeback_done_event(7)]

    gather_groups: list = []
    gather_objs: list = []
    peer_ready = [{"kind": "WriteBackDoneEvent", "op_id": 7}]

    def _all_reduce(tensor, op=None, group=None) -> None:
        del tensor, op, group

    def _all_gather_object(output, obj, group=None) -> None:
        gather_groups.append(group)
        gather_objs.append(list(obj))
        output[0] = list(obj)
        # TP peer has not finished; CP/PP peers whose TP group already
        # agreed still enter the later gathers.
        output[1] = [] if group is tp_group else peer_ready

    _install_collectives(
        monkeypatch, all_reduce=_all_reduce, all_gather_object=_all_gather_object
    )

    assert hooks.poll_ready_events() == []
    assert gather_groups == [tp_group, cp_group, pp_group]
    assert len(gather_objs[0]) == 1
    assert gather_objs[1] == []
    assert gather_objs[2] == []
    assert ("WriteBackDoneEvent", 7) in hooks._pending_payloads


def test_idle_replica_max_reduces_then_gathers_when_peer_has_work(
    fake_cache_ops, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _Device()
    cp_group = object()
    hooks = _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        attn_cp_size=2,
        attn_cp_cpu_group=cp_group,
        pp_size=1,
        pp_cpu_group=None,
    )

    reduce_groups: list = []
    gather_groups: list = []

    def _all_reduce(tensor, op=None, group=None) -> None:
        del op
        reduce_groups.append(group)
        tensor[0] = 1

    def _all_gather_object(output, obj, group=None) -> None:
        gather_groups.append(group)
        output[0] = list(obj)
        output[1] = [{"kind": "WriteBackDoneEvent", "op_id": 3}]

    _install_collectives(
        monkeypatch, all_reduce=_all_reduce, all_gather_object=_all_gather_object
    )

    assert hooks.poll_ready_events() == []
    assert reduce_groups == [cp_group]
    assert gather_groups == [cp_group]
    assert hooks._pending_payloads == {}


def test_idle_replica_skips_gather_only_after_unanimous_max_reduce(
    fake_cache_ops, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _Device()
    cp_group = object()
    hooks = _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        attn_cp_size=2,
        attn_cp_cpu_group=cp_group,
        pp_size=1,
        pp_cpu_group=None,
    )

    reduce_groups: list = []

    def _all_reduce(tensor, op=None, group=None) -> None:
        del tensor, op
        reduce_groups.append(group)

    def _all_gather_object(output, obj, group=None) -> None:
        del output, obj, group
        raise AssertionError("idle replica must not gather")

    _install_collectives(
        monkeypatch, all_reduce=_all_reduce, all_gather_object=_all_gather_object
    )

    assert hooks.poll_ready_events() == []
    assert reduce_groups == [cp_group]


def test_replica_backup_failure_raises_after_poll_all_reduce(
    fake_cache_ops, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _Device()
    device.backup_failed = True
    cp_group = object()
    hooks = _hooks(
        device,
        speculative_algorithm=None,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        attn_cp_size=2,
        attn_cp_cpu_group=cp_group,
        pp_size=1,
        pp_cpu_group=None,
    )
    hooks.count_plan_ops(SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[7])]))

    reduce_groups: list = []
    gather_groups: list = []

    def _all_reduce(tensor, op=None, group=None) -> None:
        del op
        reduce_groups.append(group)
        tensor[1] = max(int(tensor[1].item()), 1)

    def _all_gather_object(output, obj, group=None) -> None:
        gather_groups.append(group)
        del output, obj

    _install_collectives(
        monkeypatch, all_reduce=_all_reduce, all_gather_object=_all_gather_object
    )

    with pytest.raises(RuntimeError, match="L3 backup failed on a replica rank"):
        hooks.poll_ready_events()
    assert reduce_groups == [cp_group]
    assert gather_groups == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
