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

"""The scheduler wire adapter in ``engine/scheduler_utils``: the plan's cache
ops onto the runtime's per-request ops, the four cache ACK kinds across the
rank-sync payload, the recompute-retract event and the retraction fields of
``make_config``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

ts = pytest.importorskip("tokenspeed_scheduler")

from tokenspeed.runtime.cache.transfer.ops import (  # noqa: E402
    CacheTransfer,
    HostTier,
    RestoreOp,
    SnapshotOp,
)
from tokenspeed.runtime.engine import scheduler_utils  # noqa: E402
from tokenspeed.runtime.engine.scheduler_utils import (  # noqa: E402
    cache_event_from_payload,
    cache_event_key,
    cache_event_to_payload,
    cache_ops_from_plan,
    make_config,
    make_recompute_retract_event,
    pop_common_cache_event_payloads,
)


class _WriteBackOp(SimpleNamespace):
    pass


class _LoadBackOp(SimpleNamespace):
    pass


class _SnapshotBatch(SimpleNamespace):
    pass


class _RestoreBatch(SimpleNamespace):
    pass


@pytest.fixture()
def wire(monkeypatch):
    # The op bindings have no Python constructor; dispatch on these fakes.
    monkeypatch.setattr(
        scheduler_utils,
        "Cache",
        SimpleNamespace(
            WriteBackOp=_WriteBackOp,
            LoadBackOp=_LoadBackOp,
            SnapshotOp=_SnapshotBatch,
            RestoreOp=_RestoreBatch,
        ),
    )


def _snapshot_batch(**overrides):
    fields = dict(
        op_ids=[4, 5],
        request_ids=["a", "b"],
        request_pool_indices=[1, 2],
        snapshot_slots=[0, 1],
        group_ids=[[0, 1], []],
        src_pages=[[3, 1], []],
        dst_pages=[[7, 2], []],
    )
    fields.update(overrides)
    return _SnapshotBatch(**fields)


def _restore_batch(**overrides):
    fields = dict(
        op_ids=[6],
        request_ids=["a"],
        request_pool_indices=[3],
        snapshot_slots=[0],
        group_ids=[[0, 0, 1]],
        src_pages=[[9, 7, 2]],
        dst_pages=[[11, 12, 4]],
        content_hashes=[["h9", "", ""]],
        page_offsets=[[2, 0, 0]],
        source_tiers=[[0, 1, 1]],
    )
    fields.update(overrides)
    return _RestoreBatch(**fields)


def test_plan_adapter_passes_l2_batches_and_expands_snapshot_batches(wire):
    write_back, load_back = _WriteBackOp(op_ids=[1]), _LoadBackOp(op_ids=[2, 3])
    plan = SimpleNamespace(
        cache=[write_back, _snapshot_batch(), load_back, _restore_batch()]
    )

    ops = cache_ops_from_plan(plan)

    assert ops[0] is write_back and ops[3] is load_back
    assert ops[1] == SnapshotOp(
        op_id=4,
        request_id="a",
        request_pool_index=1,
        snapshot_slot=0,
        transfers=(
            CacheTransfer(0, 3, 7, "", 0),
            CacheTransfer(1, 1, 2, "", 0),
        ),
    )
    # A victim whose every page went to L2 images its slot state alone.
    assert ops[2] == SnapshotOp(
        op_id=5, request_id="b", request_pool_index=2, snapshot_slot=1, transfers=()
    )
    assert ops[4] == RestoreOp(
        op_id=6,
        request_id="a",
        request_pool_index=3,
        snapshot_slot=0,
        transfers=(
            CacheTransfer(0, 9, 11, "h9", 2),
            CacheTransfer(0, 7, 12, "", 0),
            CacheTransfer(1, 2, 4, "", 0),
        ),
        source_tier=(HostTier.L2, HostTier.SNAPSHOT_POOL, HostTier.SNAPSHOT_POOL),
    )
    assert cache_ops_from_plan(SimpleNamespace(cache=[])) == []


def test_plan_adapter_rejects_ragged_batches_and_unknown_kinds(wire):
    with pytest.raises(ValueError, match="ragged _SnapshotBatch batch"):
        cache_ops_from_plan(
            SimpleNamespace(cache=[_snapshot_batch(snapshot_slots=[0])])
        )
    with pytest.raises(ValueError, match="ragged cache operation 4"):
        cache_ops_from_plan(
            SimpleNamespace(cache=[_snapshot_batch(dst_pages=[[7], []])])
        )
    with pytest.raises(ValueError, match="ragged cache operation 6"):
        cache_ops_from_plan(
            SimpleNamespace(cache=[_restore_batch(source_tiers=[[0, 1]])])
        )
    with pytest.raises(ValueError):
        cache_ops_from_plan(
            SimpleNamespace(cache=[_restore_batch(source_tiers=[[0, 1, 7]])])
        )
    with pytest.raises(TypeError, match="unsupported cache op kind: str"):
        cache_ops_from_plan(SimpleNamespace(cache=["op"]))


@pytest.mark.parametrize(
    "kind", ["WriteBackDoneEvent", "SnapshotDoneEvent", "RestoreDoneEvent"]
)
def test_cache_acks_round_trip_the_rank_sync_payload(kind):
    event = getattr(ts.Cache, kind)()
    event.op_id = 42
    payload = cache_event_to_payload(event)
    assert payload == {"kind": kind, "op_id": 42}
    assert cache_event_key(payload) == (kind, 42)
    rebuilt = cache_event_from_payload(payload)
    assert type(rebuilt).__name__ == kind and rebuilt.op_id == 42
    # The rebuilt event is what advance_scheduler hands the binding.
    ts.ExecutionEvent().add_event(rebuilt)


def test_load_back_ack_carries_its_outcome():
    payload = cache_event_to_payload(ts.Cache.LoadBackDoneEvent(7, False))
    assert payload == {"kind": "LoadBackDoneEvent", "op_id": 7, "success": False}
    rebuilt = cache_event_from_payload(payload)
    assert (rebuilt.op_id, rebuilt.success) == (7, False)
    with pytest.raises(ValueError, match="Unsupported cache event type"):
        cache_event_from_payload({"kind": "RetractDoneEvent", "op_id": 1})


def test_replica_intersection_is_per_kind_and_op_id():
    snapshot = {"kind": "SnapshotDoneEvent", "op_id": 3}
    restore = {"kind": "RestoreDoneEvent", "op_id": 3}
    write = {"kind": "WriteBackDoneEvent", "op_id": 3}
    assert pop_common_cache_event_payloads([[snapshot, restore], [restore, write]]) == [
        restore
    ]


def test_recompute_retract_event_is_the_l3_miss_path():
    event = make_recompute_retract_event("r1")
    assert isinstance(event, ts.ForwardEvent.RecomputeRetract)
    assert event.request_id == "r1"
    assert not hasattr(ts.ForwardEvent, "Retract")


def _config(**overrides):
    fields = dict(
        num_device_pages=32,
        max_scheduled_tokens=64,
        max_batch_size=8,
        prefix_granularity=2,
        num_host_pages=0,
        disable_l2_cache=True,
        enable_l3_storage=False,
        role="fused",
        num_snapshot_pages=1,
        max_retracted_requests=0,
    )
    fields.update(overrides)
    return make_config(**fields)


def test_make_config_states_the_snapshot_pool_on_every_role():
    config = _config()
    assert config.num_snapshot_pages == 1
    assert config.max_retracted_requests == 0
    assert config.debug_force_retraction_interval == 0
    config = _config(
        num_snapshot_pages=9,
        max_retracted_requests=4,
        debug_force_retraction_interval=-2,
    )
    assert (config.num_snapshot_pages, config.max_retracted_requests) == (9, 4)
    assert config.debug_force_retraction_interval == -2
    with pytest.raises(ValueError, match="null page"):
        _config(num_snapshot_pages=0)
    with pytest.raises(ValueError, match="go together"):
        _config(num_snapshot_pages=9)
    with pytest.raises(ValueError, match="go together"):
        _config(max_retracted_requests=4)
    with pytest.raises(TypeError):
        # The retraction fields are keyword-only and required.
        make_config(32, 64, 8, 2, 0, True, False, "fused")
