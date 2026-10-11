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

"""The retraction image through the Host cache executor's second tier.

CPU tests drive ``HostCacheExecutor`` with mocked storage, streams and lanes
and assert the stream contract: a retraction's two store legs (the ordered
L2 write-back and the snapshot store with its slot-state export) ride the
write stream under ONE fence on the default stream; a restore orders behind
the zeroing, copies its rows from both Host tiers, imports the slot state and
is acknowledged once. The owner translation keeps a KVP rank's rows on both
ends of every row of every op. One GPU test (skipped without CUDA) round-trips
real bytes through two ranks' executors, both tiers, over one sharded group.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import nullcontext
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import tokenspeed.runtime.cache.l2.executor as executor_module  # noqa: E402
import tokenspeed.runtime.cache.transfer.lanes as lanes_module  # noqa: E402
from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor  # noqa: E402
from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest  # noqa: E402
from tokenspeed.runtime.cache.transfer.lanes import (  # noqa: E402
    CompletionQueue,
    HostTransferLane,
)
from tokenspeed.runtime.cache.transfer.layout import (  # noqa: E402
    CacheField,
    CacheGroupLayout,
    CacheTransferLayout,
)
from tokenspeed.runtime.cache.transfer.ops import (  # noqa: E402
    CacheTransfer,
    HostTier,
    PrefetchOp,
    PrefetchRow,
    RestoreOp,
    SnapshotOp,
)
from tokenspeed.runtime.cache.transfer.ownership import (  # noqa: E402
    BlockOwnerTranslation,
)
from tokenspeed.runtime.engine import (  # noqa: E402
    scheduler_utils as scheduler_utils_module,
)
from tokenspeed.runtime.engine.cache_hooks import CacheOpHooks  # noqa: E402
from tokenspeed.runtime.engine.scheduler_utils import (  # noqa: E402
    cache_event_from_payload,
    cache_event_to_payload,
)
from tokenspeed.runtime.execution.slot_state import (  # noqa: E402
    pack_slot_rows,
    slot_state_image_bytes,
    unpack_slot_rows,
)

# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


def _acks(events) -> list[tuple[str, int]]:
    """The binding ACK events as comparable ``(kind, op_id)`` pairs."""
    return [(type(event).__name__, int(event.op_id)) for event in events]


def _snapshot_done(*op_ids):
    return [("SnapshotDoneEvent", op_id) for op_id in op_ids]


def _restore_done(*op_ids):
    return [("RestoreDoneEvent", op_id) for op_id in op_ids]


def _layout(num_lcm_blocks, groups):
    """``groups`` is ``[(group_id, packing)]``; fields are dummies."""
    return SimpleNamespace(
        num_lcm_blocks=num_lcm_blocks,
        groups=tuple(
            SimpleNamespace(
                group_id=group_id,
                cache_blocks_per_lcm_block=packing,
                fields=(SimpleNamespace(field_id=f"{group_id}.k", payload_bytes=4),),
            )
            for group_id, packing in groups
        ),
        buffers=(SimpleNamespace(device=SimpleNamespace(type="cuda")),),
        consumers=(tuple(f"{group_id}.k" for group_id, _ in groups),),
    )


def _contract(layout, shard_counts):
    specs = tuple(
        SimpleNamespace(group_id=group.group_id, shard_count=shard)
        for group, shard in zip(layout.groups, shard_counts)
    )
    return SimpleNamespace(
        group_specs=specs,
        virtual_block_counts={
            group.group_id: 1
            + layout.num_lcm_blocks * group.cache_blocks_per_lcm_block * shard
            for group, shard in zip(layout.groups, shard_counts)
        },
        num_lcm_blocks=layout.num_lcm_blocks,
        token_capacity=layout.num_lcm_blocks * 16,
    )


class _Pool:
    def __init__(self, layout, shard_counts):
        self._layout = layout
        contract = _contract(layout, shard_counts)
        self.arena = SimpleNamespace(
            cache_group_specs=contract.group_specs, runtime_contract=contract
        )
        self.load_tracker = None

    def cache_transfer_layout(self):
        return self._layout

    def register_layerwise_load_tracker(self, tracker):
        self.load_tracker = tracker


class _SlotState:
    """A slot-state owner that records the executor's calls."""

    def __init__(self, nbytes=32):
        self.nbytes = nbytes
        self.exports: list = []
        self.imports: list = []

    def slot_state_bytes(self):
        return self.nbytes

    def export_slot_state(self, slot, out, stream, *, request_id):
        self.exports.append((slot, out, stream, request_id))

    def import_slot_state(self, slot, src, stream, *, request_id):
        self.imports.append((slot, src, stream, request_id))


class _WriteBackOp:
    """The L2 op's wire face (the C++ type has no Python constructor)."""

    def __init__(self, op_ids, group_ids, src_pages, dst_pages, source_pinned):
        self.op_ids = op_ids
        self.group_ids = group_ids
        self.src_pages = src_pages
        self.dst_pages = dst_pages
        self.source_pinned = source_pinned
        # Unkeyed rows unless a test sets the keys: nothing for L3 to back up.
        self.content_hashes = [[""] * len(groups) for groups in group_ids]
        self.page_offsets = [[0] * len(groups) for groups in group_ids]


class _LoadBackOp:
    def __init__(self, op_ids, group_ids, src_pages, dst_pages):
        self.op_ids = op_ids
        self.group_ids = group_ids
        self.src_pages = src_pages
        self.dst_pages = dst_pages


def _transfer(group, source, destination, key=""):
    return CacheTransfer(
        group_id=group,
        source_page=source,
        destination_page=destination,
        content_hash=key,
        page_offset=0,
    )


def _snapshot_op(op_id, slot, transfers, *, pool_index=1):
    return SnapshotOp(
        op_id=op_id,
        request_id=f"r{op_id}",
        request_pool_index=pool_index,
        snapshot_slot=slot,
        transfers=tuple(transfers),
    )


def _restore_op(op_id, slot, rows, *, pool_index=1):
    """``rows`` is ``[(tier, transfer)]``."""
    return RestoreOp(
        op_id=op_id,
        request_id=f"r{op_id}",
        request_pool_index=pool_index,
        snapshot_slot=slot,
        transfers=tuple(transfer for _, transfer in rows),
        source_tier=tuple(tier for tier, _ in rows),
    )


def _pool(host_gb, max_retracted, *, ratio=None, tail=0):
    """Pool knobs as ServerArgs resolves them: 0/None unset, ratio 0 = no pool."""
    return RetractionPoolRequest(
        host_gb=host_gb,
        ratio=ratio,
        max_retracted_requests=max_retracted,
        tail_lcm_blocks_per_request=tail,
    )


NO_POOL = _pool(0.0, 0, ratio=0.0)


def _build(
    *,
    layout,
    shard_counts,
    rank=0,
    l2_tier=True,
    snapshot_host_gb=3 / 1000,  # 3 pool LCM blocks of 1 MB
    max_retracted=4,
    slot_state=None,
):
    """A two-tier executor over fakes: mocked storage, geometry, streams, lanes.

    Returns ``(executor, slot_state, lanes)`` with ``lanes`` keyed by role.
    ``snapshot_host_gb`` 0 builds no pool.
    """
    if slot_state is None and snapshot_host_gb > 0:
        slot_state = _SlotState()
    write_stream = Mock(name="write_stream")
    load_stream = Mock(name="load_stream")
    lane_names = ("ordered", "pinned", "snapshot", "restore_l2", "restore_pool")
    lanes = {name: Mock(name=f"{name}_lane") for name in lane_names}
    storages = iter(
        [
            SimpleNamespace(host_buffer="l2-host"),
            SimpleNamespace(host_buffer="pool-host"),
        ]
        if l2_tier
        else [SimpleNamespace(host_buffer="pool-host")]
    )
    geometries = iter(
        ["l2-geometry", "pool-geometry"] if l2_tier else ["pool-geometry"]
    )
    tracker = Mock()
    tracker.event_sets = [object()]
    with (
        patch.object(
            executor_module, "compute_host_lcm_block_bytes", return_value=1_000_000
        ),
        patch.object(executor_module, "check_host_memory"),
        patch.object(
            executor_module,
            "HostCacheStorage",
            side_effect=lambda *a, **k: next(storages),
        ),
        patch.object(
            executor_module,
            "allocate_blob_arena",
            side_effect=lambda rows, nbytes: torch.zeros(
                rows, nbytes, dtype=torch.uint8
            ),
        ),
        patch.object(
            executor_module,
            "build_transfer_geometry",
            side_effect=lambda *a, **k: next(geometries),
        ),
        patch.object(
            executor_module, "new_cache_stream", side_effect=[write_stream, load_stream]
        ),
        patch.object(
            executor_module, "HostTransferLane", side_effect=list(lanes.values())
        ),
        patch.object(executor_module, "HostTransferWorkspace", side_effect=Mock),
        patch.object(executor_module, "LayerwiseLoadTracker", return_value=tracker),
    ):
        executor = HostCacheExecutor(
            _Pool(layout, shard_counts),
            draft_pool=None,
            l2_tier=l2_tier,
            host_ratio=1.5,  # 3 L2 LCM blocks over a 2-block device layout
            host_size_gb=0,
            snapshot_pool=(
                _pool(snapshot_host_gb, max_retracted)
                if snapshot_host_gb > 0
                else NO_POOL
            ),
            slot_state_exporters=(slot_state,) if slot_state is not None else None,
            io_backend="direct",
            attn_tp_rank=0,
            kvp_rank=rank,
        )
    return executor, slot_state, lanes


# ----------------------------------------------------------------------
# Wire types and sizing
# ----------------------------------------------------------------------


def test_op_dataclasses_carry_the_wire_fields():
    wire = ("op_id", "request_id", "request_pool_index", "snapshot_slot", "transfers")
    assert tuple(f.name for f in fields(SnapshotOp)) == wire
    assert tuple(f.name for f in fields(RestoreOp)) == wire + ("source_tier",)
    assert tuple(f.name for f in fields(CacheTransfer)) == (
        "group_id",
        "source_page",
        "destination_page",
        "content_hash",
        "page_offset",
    )
    assert (int(HostTier.L2), int(HostTier.SNAPSHOT_POOL)) == (0, 1)
    assert (
        int(executor_module.Cache.HostTier.L2),
        int(executor_module.Cache.HostTier.SnapshotPool),
    ) == (0, 1)
    for kind, op_id in (("SnapshotDoneEvent", 3), ("RestoreDoneEvent", 4)):
        event = getattr(executor_module.Cache, kind)()
        event.op_id = op_id
        payload = cache_event_to_payload(event)
        assert payload == {"kind": kind, "op_id": op_id}
        assert _acks([cache_event_from_payload(payload)]) == [(kind, op_id)]


def test_constructor_requires_a_tier_and_resolves_the_pool_against_the_layout():
    layout = _layout(2, [("full", 2)])
    with pytest.raises(ValueError, match="L2 tier, a snapshot pool or both"):
        _build(layout=layout, shard_counts=[1], l2_tier=False, snapshot_host_gb=0)
    with patch.object(executor_module, "compute_host_lcm_block_bytes", return_value=1):
        # A pool with no slot-state rows, or no exporters to fill them, is refused.
        with pytest.raises(ValueError, match="slot-state exporters"):
            HostCacheExecutor(
                _Pool(layout, [1]),
                l2_tier=True,
                host_ratio=1.0,
                host_size_gb=0,
                snapshot_pool=_pool(1e-9, 3),
                slot_state_exporters=None,
                io_backend="direct",
                attn_tp_rank=0,
                kvp_rank=0,
            )
        with pytest.raises(ValueError, match="slot-state rows"):
            HostCacheExecutor(
                _Pool(layout, [1]),
                l2_tier=True,
                host_ratio=1.0,
                host_size_gb=0,
                snapshot_pool=_pool(1e-9, 0),
                slot_state_exporters=(_SlotState(),),
                io_backend="direct",
                attn_tp_rank=0,
                kvp_rank=0,
            )
    # Too small a size holds no whole block and is refused, not rounded to none.
    with pytest.raises(ValueError, match="no whole LCM block"):
        _build(layout=layout, shard_counts=[1], snapshot_host_gb=0.0005)
    executor, _, _ = _build(layout=layout, shard_counts=[1], snapshot_host_gb=0)
    assert (executor.num_host_pages, executor.num_snapshot_pages) == (4, 1)
    assert executor.max_retracted_requests == 0 and executor.blob_arena is None
    executor, _, _ = _build(layout=layout, shard_counts=[1], l2_tier=False)
    assert (executor.num_host_pages, executor.num_snapshot_pages) == (0, 4)
    assert executor.host_storage is None and executor._load_trackers == []


def test_count_plan_ops_counts_each_request_of_a_snapshot_or_restore_batch():
    hooks = CacheOpHooks(
        SimpleNamespace(),
        speculative_algorithm=None,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pp_size=1,
        pp_cpu_group=None,
        global_rank=0,
    )

    class _SnapshotBatch:
        op_ids = [1, 2]
        request_ids = ["a", "b"]
        request_pool_indices = [1, 2]
        snapshot_slots = [0, 1]
        group_ids = [[0], []]
        src_pages = [[1], []]
        dst_pages = [[1], []]

    class _RestoreBatch:
        op_ids = [3]
        request_ids = ["c"]
        request_pool_indices = [4]
        snapshot_slots = [2]
        group_ids = [[0]]
        src_pages = [[1]]
        dst_pages = [[1]]
        content_hashes = [["h"]]
        page_offsets = [[0]]
        source_tiers = [[0]]

    plan = SimpleNamespace(cache=[_SnapshotBatch(), _RestoreBatch()])
    with patch.object(
        scheduler_utils_module,
        "Cache",
        SimpleNamespace(
            WriteBackOp=(),
            LoadBackOp=(),
            SnapshotOp=_SnapshotBatch,
            RestoreOp=_RestoreBatch,
        ),
    ):
        hooks.count_plan_ops(plan)
    assert hooks._num_inflight == 3


# ----------------------------------------------------------------------
# Owner translation
# ----------------------------------------------------------------------


def test_replicated_groups_translate_to_the_identity_and_reject_null():
    owners = BlockOwnerTranslation(
        shard_counts=[1, 1],
        device_virtual_counts=[9, 5],
        host_virtual_counts=[17, 3],
        rank=0,
    )
    rows = [(1, 4, 2), (0, 8, 16), (0, 1, 1)]
    # Every row is owned and keeps its position; grouped by group index,
    # input order kept within a group.
    assert owners.owned_positions(rows) == [
        (1, (0, 8, 16)),
        (2, (0, 1, 1)),
        (0, (1, 4, 2)),
    ]
    with pytest.raises(ValueError, match="null block"):
        owners.owned_positions([(0, 0, 1)])
    with pytest.raises(IndexError, match="unknown group"):
        owners.owned_positions([(2, 1, 1)])
    with pytest.raises(IndexError):
        owners.owned_positions([(0, 9, 1)])
    with pytest.raises(IndexError):
        owners.owned_positions([(1, 1, 3)])


def test_sharded_group_keeps_each_ranks_rows_on_both_ends():
    # Device: 2 LCM blocks x packing 2 x 2 shards -> virtual 1..8; Host: 3 LCM
    # blocks -> virtual 1..12. Owner of v is (v - 1) % 2, local (v - 1) // 2 + 1.
    layout = _layout(2, [("full", 2)])
    contract = _contract(layout, [2])
    owners = {
        rank: BlockOwnerTranslation.for_host_pool(
            layout, contract, num_host_lcm_blocks=3, rank=rank
        )
        for rank in (0, 1)
    }
    rows = [(0, 1, 3), (0, 2, 6), (0, 5, 11), (0, 8, 12)]
    assert owners[0].owned_positions(rows) == [(0, (0, 1, 2)), (2, (0, 3, 6))]
    assert owners[1].owned_positions(rows) == [(1, (0, 1, 3)), (3, (0, 4, 6))]
    with pytest.raises(ValueError, match="residue class"):
        owners[0].owned_positions([(0, 1, 2)])


# ----------------------------------------------------------------------
# The executor over fakes
# ----------------------------------------------------------------------


def test_both_store_legs_ride_the_write_stream_under_one_fence():
    layout = _layout(4, [("full", 4), ("state", 1)])
    executor, slot_state, lanes = _build(layout=layout, shard_counts=[1, 1])
    cache_ops = [
        # One L2 batch: the victim's hash-complete pages (stream-ordered)
        # and another request's ordinary publication (pinned).
        _WriteBackOp([11, 12], [[0], [0]], [[1], [3]], [[5], [7]], [False, True]),
        # The same victim's tail pages and slot state.
        _snapshot_op(7, 1, [_transfer(0, 2, 2), _transfer(1, 1, 1)], pool_index=3),
        _snapshot_op(8, 2, [_transfer(0, 4, 6)], pool_index=6),
    ]
    ordered_finish, snapshot_finish, pinned_finish = (
        Mock(name="ordered_finish"),
        Mock(name="snapshot_finish"),
        Mock(name="pinned_finish"),
    )
    for event in (ordered_finish, snapshot_finish, pinned_finish):
        event.query.return_value = False
    events = iter([ordered_finish, snapshot_finish, pinned_finish])
    fence_stream = Mock(name="fence_stream")
    order = Mock()
    order.attach_mock(executor.write_stream.wait_stream, "wait")
    order.attach_mock(lanes["ordered"].start_d2h, "l2_ordered")
    order.attach_mock(lanes["snapshot"].start_d2h, "pool")
    order.attach_mock(snapshot_finish.record, "record")
    order.attach_mock(fence_stream.wait_event, "fence")
    order.attach_mock(lanes["pinned"].start_d2h, "l2_pinned")
    lanes["ordered"].start_d2h.return_value = ordered_finish
    lanes["pinned"].start_d2h.return_value = pinned_finish
    with (
        patch.object(executor_module.Cache, "WriteBackOp", _WriteBackOp, create=True),
        patch.object(
            executor_module.device_module, "Event", return_value=snapshot_finish
        ),
    ):
        executor.submit_write_backs(
            cache_ops, prerequisite_stream="execution-stream", fence_stream=fence_stream
        )

    # The ordered L2 rows, then the tail rows on the same stream, then the
    # slot exports, then ONE event the default stream waits on -- before the
    # pinned publication, which fences nothing.
    assert [c[0] for c in order.mock_calls] == [
        "l2_ordered",
        "wait",
        "pool",
        "record",
        "fence",
        "l2_pinned",
    ]
    assert lanes["ordered"].start_d2h.call_args.args[0] == [(0, 1, 5)]
    assert lanes["ordered"].start_d2h.call_args.kwargs["host_buffer"] == "l2-host"
    assert lanes["ordered"].start_d2h.call_args.kwargs["geometry"] == "l2-geometry"
    assert lanes["ordered"].start_d2h.call_args.kwargs["prerequisite_stream"] == (
        "execution-stream"
    )
    assert lanes["snapshot"].start_d2h.call_args.args[0] == [
        (0, 2, 2),
        (1, 1, 1),
        (0, 4, 6),
    ]
    assert lanes["snapshot"].start_d2h.call_args.kwargs["host_buffer"] == "pool-host"
    assert lanes["snapshot"].start_d2h.call_args.kwargs["geometry"] == "pool-geometry"
    assert (
        lanes["snapshot"].start_d2h.call_args.kwargs["stream"] is executor.write_stream
    )
    assert lanes["pinned"].start_d2h.call_args.args[0] == [(0, 3, 7)]
    fence_stream.wait_event.assert_called_once_with(snapshot_finish)
    # Each victim's slot is exported into its arena row, on the write stream,
    # as the victim's (a request-keyed owner images whether it prepared it).
    assert [(slot, stream, rid) for slot, _, stream, rid in slot_state.exports] == [
        (3, executor.write_stream, "r7"),
        (6, executor.write_stream, "r8"),
    ]
    assert slot_state.exports[0][1].data_ptr() == executor.blob_arena[1].data_ptr()
    assert slot_state.exports[1][1].data_ptr() == executor.blob_arena[2].data_ptr()
    # Three ACK kinds, each released by its own event.
    assert executor.poll_results() == []
    snapshot_finish.query.return_value = True
    assert _acks(executor.poll_results()) == _snapshot_done(7, 8)
    ordered_finish.query.return_value = True
    pinned_finish.query.return_value = True
    assert sorted(int(e.op_id) for e in executor.poll_results()) == [11, 12]
    assert executor.poll_results() == []
    del events


def test_slot_state_only_image_round_trips_through_empty_ops():
    """A victim whose every page went to L2 has an image of its slot state
    alone: the store and the restore both carry no transfers, touch no lane,
    and still export, import and acknowledge."""
    layout = _layout(2, [("full", 2)])
    executor, slot_state, lanes = _build(layout=layout, shard_counts=[1])
    finish = Mock()
    finish.query.return_value = True
    fence_stream = Mock()
    with patch.object(executor_module.device_module, "Event", return_value=finish):
        executor.submit_write_backs(
            [_snapshot_op(1, 0, [], pool_index=2)],
            prerequisite_stream="x",
            fence_stream=fence_stream,
        )
    lanes["snapshot"].start_d2h.assert_not_called()
    lanes["ordered"].start_d2h.assert_not_called()
    fence_stream.wait_event.assert_called_once_with(finish)
    assert [slot for slot, _, _, _ in slot_state.exports] == [2]
    assert slot_state.exports[0][1].data_ptr() == executor.blob_arena[0].data_ptr()
    assert _acks(executor.poll_results()) == _snapshot_done(1)

    tracker = executor._load_trackers[0][0]
    with (
        patch.object(executor_module, "get_is_capture_mode", return_value=False),
        patch.object(executor_module.device_module, "Event", return_value=finish),
    ):
        executor.submit_load_backs(
            [_restore_op(2, 0, [], pool_index=5)],
            prerequisite_stream="default-stream",
        )
    lanes["restore_l2"].start_h2d.assert_not_called()
    lanes["restore_pool"].start_h2d.assert_not_called()
    executor.load_stream.wait_stream.assert_called_once_with("default-stream")
    assert [(slot, rid) for slot, _, _, rid in slot_state.imports] == [(5, "r2")]
    assert slot_state.imports[0][1].data_ptr() == executor.blob_arena[0].data_ptr()
    tracker.begin_load.assert_not_called()
    assert _acks(executor.poll_results()) == _restore_done(2)


def test_a_failed_snapshot_leg_still_fences_the_launched_ordered_rows():
    """The ordered L2 rows launch before the snapshot leg; if an exporter raises
    mid-export, the rows (and whatever of the leg launched) are in flight with
    no event recorded. The fence is issued on the write stream's tail anyway
    before the error propagates, so the plan's zeroing cannot run over copy
    sources still being read."""
    layout = _layout(4, [("full", 4)])
    executor, slot_state, lanes = _build(layout=layout, shard_counts=[1])
    slot_state.export_slot_state = Mock(side_effect=RuntimeError("exporter broke"))
    ordered_finish, tail = Mock(name="ordered_finish"), Mock(name="tail")
    lanes["ordered"].start_d2h.return_value = ordered_finish
    fence_stream = Mock(name="fence_stream")
    order = Mock()
    order.attach_mock(lanes["ordered"].start_d2h, "l2_ordered")
    order.attach_mock(lanes["snapshot"].start_d2h, "pool")
    order.attach_mock(tail.record, "record")
    order.attach_mock(fence_stream.wait_event, "fence")
    with (
        patch.object(executor_module.Cache, "WriteBackOp", _WriteBackOp, create=True),
        patch.object(executor_module.device_module, "Event", return_value=tail),
        pytest.raises(RuntimeError, match="exporter broke"),
    ):
        executor.submit_write_backs(
            [
                _WriteBackOp([11], [[0]], [[1]], [[5]], [False]),
                _snapshot_op(7, 1, [_transfer(0, 2, 2)], pool_index=3),
            ],
            prerequisite_stream="execution-stream",
            fence_stream=fence_stream,
        )
    # Both legs launched, then the tail event was recorded on the write
    # stream and fenced -- not the ordered leg's own event.
    assert [c[0] for c in order.mock_calls] == ["l2_ordered", "pool", "record", "fence"]
    tail.record.assert_called_once_with(executor.write_stream)
    fence_stream.wait_event.assert_called_once_with(tail)
    # The pinned publication never started: the plan failed before it.
    lanes["pinned"].start_d2h.assert_not_called()


def test_restore_reads_both_tiers_imports_the_slot_and_acks_once():
    layout = _layout(4, [("full", 4), ("state", 1)])
    executor, slot_state, lanes = _build(layout=layout, shard_counts=[1, 1])
    op = _restore_op(
        9,
        3,
        [
            (HostTier.L2, _transfer(0, 9, 6, key="h9")),
            (HostTier.SNAPSHOT_POOL, _transfer(0, 2, 7)),
            (HostTier.L2, _transfer(0, 10, 8, key="h10")),
            (HostTier.SNAPSHOT_POOL, _transfer(1, 1, 3)),
        ],
        pool_index=4,
    )
    finish = Mock()
    finish.query.return_value = False
    order = Mock()
    order.attach_mock(executor.load_stream.wait_stream, "wait")
    order.attach_mock(lanes["restore_l2"].start_h2d, "l2")
    order.attach_mock(lanes["restore_pool"].start_h2d, "pool")
    order.attach_mock(finish.record, "record")
    tracker = executor._load_trackers[0][0]
    with (
        patch.object(executor_module, "get_is_capture_mode", return_value=False),
        patch.object(executor_module.device_module, "Event", return_value=finish),
    ):
        executor.submit_load_backs(
            [op],
            prerequisite_stream="default-stream",
        )

    assert [c[0] for c in order.mock_calls] == ["wait", "l2", "pool", "record"]
    executor.load_stream.wait_stream.assert_called_once_with("default-stream")
    lanes["restore_l2"].start_h2d.assert_called_once_with(
        [(0, 6, 9), (0, 8, 10)],
        device_buffers=layout.buffers,
        host_buffer="l2-host",
        geometry="l2-geometry",
        stream=executor.load_stream,
        prerequisite_stream=None,
        backend="dma",
    )
    lanes["restore_pool"].start_h2d.assert_called_once_with(
        [(0, 7, 2), (1, 3, 1)],
        device_buffers=layout.buffers,
        host_buffer="pool-host",
        geometry="pool-geometry",
        stream=executor.load_stream,
        prerequisite_stream=None,
        backend="dma",
    )
    assert slot_state.imports == [
        (4, slot_state.imports[0][1], executor.load_stream, "r9")
    ]
    assert slot_state.imports[0][1].data_ptr() == executor.blob_arena[3].data_ptr()
    finish.record.assert_called_once_with(executor.load_stream)
    # A restore arms no layerwise fence: the round's load trackers see no load.
    tracker.begin_load.assert_not_called()
    tracker.set_consumers.assert_called_once_with(-1)
    assert executor.poll_results() == []
    finish.query.return_value = True
    assert _acks(executor.poll_results()) == _restore_done(9)


def test_load_backs_start_before_restores_on_the_load_stream():
    """The forward reads a load-back layer by layer behind the layerwise
    fences; a restore is read by nothing this round, so it queues behind the
    loads on the shared stream instead of delaying them."""
    layout = _layout(4, [("full", 4)])
    executor, _, _ = _build(layout=layout, shard_counts=[1])
    order = Mock()
    load = Mock(return_value=0)
    restore = Mock()
    order.attach_mock(load, "load")
    order.attach_mock(restore, "restore")
    with (
        patch.object(executor_module.Cache, "LoadBackOp", _LoadBackOp, create=True),
        patch.object(executor, "_start_loading", load),
        patch.object(executor, "_start_restore", restore),
    ):
        executor.submit_load_backs(
            [
                _restore_op(3, 0, [(HostTier.SNAPSHOT_POOL, _transfer(0, 1, 2))]),
                _LoadBackOp([4], [[0]], [[5]], [[3]]),
            ],
            prerequisite_stream="default-stream",
        )
    assert [c[0] for c in order.mock_calls] == ["load", "restore"]
    assert load.call_args.args[0] == [4]
    assert restore.call_args.kwargs["prerequisite_stream"] == "default-stream"


def test_restore_rows_need_their_tier_and_run_outside_capture():
    layout = _layout(2, [("full", 2)])
    # Without an L2 tier a restore naming L2 rows is a configuration error.
    executor, _, _ = _build(layout=layout, shard_counts=[1], l2_tier=False)
    l2_row = _restore_op(1, 0, [(HostTier.L2, _transfer(0, 1, 2, key="h"))])
    with patch.object(executor_module, "get_is_capture_mode", return_value=False):
        with pytest.raises(RuntimeError, match="no L2 tier"):
            executor.submit_load_backs(
                [l2_row],
                prerequisite_stream="s",
            )
    pool_row = _restore_op(2, 0, [(HostTier.SNAPSHOT_POOL, _transfer(0, 1, 2))])
    with patch.object(executor_module, "get_is_capture_mode", return_value=True):
        with pytest.raises(RuntimeError, match="graph capture"):
            executor.submit_load_backs(
                [pool_row],
                prerequisite_stream="s",
            )
    # Without a pool, snapshot ops are refused outright.
    executor, _, _ = _build(layout=layout, shard_counts=[1], snapshot_host_gb=0)
    with pytest.raises(RuntimeError, match="retraction snapshot pool"):
        executor.submit_write_backs(
            [_snapshot_op(3, 0, [])],
            prerequisite_stream="s",
            fence_stream=Mock(),
        )


def test_plan_level_checks_refuse_duplicates_bad_slots_and_ragged_tiers():
    """Every op of a plan is validated before its first copy is launched: a
    bad snapshot op beside a stream-ordered L2 write-back must not leave that
    write-back in flight without the fence that was to follow it."""
    layout = _layout(2, [("full", 2)])
    executor, slot_state, lanes = _build(
        layout=layout, shard_counts=[1], max_retracted=2
    )
    fence = Mock()
    # The victim's L2 leg precedes the bad snapshot leg in every plan below.
    l2_leg = _WriteBackOp([11], [[0]], [[1]], [[5]], [False])
    with patch.object(executor_module.Cache, "WriteBackOp", _WriteBackOp, create=True):
        with pytest.raises(ValueError, match="duplicate snapshot op id"):
            executor.submit_write_backs(
                [l2_leg, _snapshot_op(1, 0, []), _snapshot_op(1, 1, [])],
                prerequisite_stream="s",
                fence_stream=fence,
            )
        with pytest.raises(ValueError, match="duplicate snapshot slot"):
            executor.submit_write_backs(
                [l2_leg, _snapshot_op(1, 0, []), _snapshot_op(2, 0, [])],
                prerequisite_stream="s",
                fence_stream=fence,
            )
        with pytest.raises(IndexError, match="snapshot slot 2"):
            executor.submit_write_backs(
                [l2_leg, _snapshot_op(1, 2, [])],
                prerequisite_stream="s",
                fence_stream=fence,
            )
    ragged = RestoreOp(
        op_id=5,
        request_id="r",
        request_pool_index=1,
        snapshot_slot=0,
        transfers=(_transfer(0, 1, 1),),
        source_tier=(),
    )
    duplicate_restore = _restore_op(
        6, 1, [(HostTier.SNAPSHOT_POOL, _transfer(0, 1, 2))]
    )
    with patch.object(executor_module.Cache, "LoadBackOp", _LoadBackOp, create=True):
        with pytest.raises(ValueError, match="ragged cache operation 5"):
            executor.submit_load_backs(
                [_LoadBackOp([21], [[0]], [[5]], [[1]]), ragged],
                prerequisite_stream="s",
            )
        with pytest.raises(ValueError, match="duplicate snapshot slot"):
            executor.submit_load_backs(
                [
                    _LoadBackOp([21], [[0]], [[5]], [[1]]),
                    duplicate_restore,
                    _restore_op(7, 1, []),
                ],
                prerequisite_stream="s",
            )
    # Nothing was launched, exported, imported or queued for an ACK.
    executor.write_stream.wait_stream.assert_not_called()
    executor.load_stream.wait_stream.assert_not_called()
    for lane in lanes.values():
        lane.start_d2h.assert_not_called()
        lane.start_h2d.assert_not_called()
    fence.wait_event.assert_not_called()
    assert slot_state.exports == [] and slot_state.imports == []
    assert executor._completions._pending == []
    executor._load_trackers[0][0].begin_load.assert_not_called()


def test_kvp_ranks_store_their_owned_subsets_on_both_legs():
    # One sharded group (D = 2): rank r keeps rows whose Device block has
    # (v - 1) % 2 == r; the scheduler placed every Host block in the same class.
    layout = _layout(2, [("full", 2)])
    l2_rows = [11, 12]
    seen = {}
    for rank in (0, 1):
        executor, slot_state, lanes = _build(layout=layout, shard_counts=[2], rank=rank)
        cache_ops = [
            _WriteBackOp(l2_rows, [[0], [0]], [[1], [2]], [[3], [6]], [False, False]),
            _snapshot_op(
                5,
                0,
                [
                    _transfer(0, 5, 1),
                    _transfer(0, 6, 4),
                    _transfer(0, 7, 3),
                    _transfer(0, 8, 10),
                ],
            ),
        ]
        finish = Mock()
        finish.query.return_value = True
        lanes["ordered"].start_d2h.return_value = finish
        with (
            patch.object(
                executor_module.Cache, "WriteBackOp", _WriteBackOp, create=True
            ),
            patch.object(executor_module.device_module, "Event", return_value=finish),
        ):
            executor.submit_write_backs(
                cache_ops, prerequisite_stream="s", fence_stream=Mock()
            )
        seen[rank] = (
            lanes["ordered"].start_d2h.call_args.args[0],
            lanes["snapshot"].start_d2h.call_args.args[0],
        )
        # Slot state is per rank: every rank exports its own and ACKs every op.
        assert [slot for slot, _, _, _ in slot_state.exports] == [1]
        acks = _acks(executor.poll_results())
        assert ("SnapshotDoneEvent", 5) in acks
        assert sorted(
            op_id for kind, op_id in acks if kind == "WriteBackDoneEvent"
        ) == (l2_rows)
    assert seen == {
        0: ([(0, 1, 2)], [(0, 3, 1), (0, 4, 2)]),
        1: ([(0, 1, 3)], [(0, 3, 2), (0, 4, 5)]),
    }
    bad = _snapshot_op(6, 1, [_transfer(0, 1, 2)])
    with pytest.raises(ValueError, match="residue class"):
        executor.submit_write_backs([bad], prerequisite_stream="s", fence_stream=Mock())


def test_kvp_rank_backs_up_and_prefetches_only_the_host_pages_it_owns():
    """L3 sees the rows this rank owns, by local Host id: the L2 write's
    backup list and a prefetch op's pages both pass the owner filter, so a
    peer's page is neither put nor fetched from here."""
    layout = _layout(2, [("full", 2)])
    # Device 1 -> Host 3 is rank 0's (local Host 2); Device 2 -> Host 6 is
    # rank 1's (local Host 3).
    for rank, owned_page in ((0, (0, 2, "h3", 0)), (1, (0, 3, "h6", 1))):
        executor, _, lanes = _build(layout=layout, shard_counts=[2], rank=rank)
        store = _WriteBackOp([11], [[0, 0]], [[1, 2]], [[3, 6]], [True])
        store.content_hashes = [["h3", "h6"]]
        store.page_offsets = [[0, 1]]
        finish = Mock()
        finish.query.return_value = False
        lanes["pinned"].start_d2h.return_value = finish
        with patch.object(
            executor_module.Cache, "WriteBackOp", _WriteBackOp, create=True
        ):
            executor.submit_write_backs(
                [store],
                prerequisite_stream="s",
                fence_stream=Mock(),
            )
        ((pending_finish, ack),) = executor._completions._pending
        assert pending_finish is finish
        assert (ack.op_ids, ack.backup_pages) == ([11], [owned_page])
        # The prefetch of the same two pages fetches only this rank's one; the
        # peer's page counts as landed here so the replica MIN is the owners'.
        executor.l3_store = Mock()
        executor.l3_store.prefetch.return_value = [True]
        executor._l3_prefetch_timeout_base_s = 10.0
        executor._l3_prefetch_timeout_per_page_s = 0.0
        executor._l3_prefetch_batch_pages = 4
        executor.submit_prefetches(
            [
                PrefetchOp(
                    op_id=12,
                    request_id="r12",
                    first_page=0,
                    num_pages=2,
                    rows=(
                        PrefetchRow(0, 3, "h3", 0, 0),
                        PrefetchRow(0, 6, "h6", 1, 1),
                    ),
                )
            ]
        )
        for _ in range(200):
            progress = executor.l3_prefetch_progress()
            if progress[12][0]:
                break
            time.sleep(0.01)
        assert progress[12] == (True, 2)
        executor.l3_store.prefetch.assert_called_once_with([owned_page])
        executor._l3_prefetch_lane.shutdown(wait=True)


@pytest.mark.parametrize("backend", ["dma", "auto"])
def test_a_lane_completes_a_zero_row_copy_without_touching_the_transport(backend):
    """An op whose every row belongs to other KVP ranks is an empty copy on
    this rank: both directions record the completion event behind the
    prerequisite stream and never stage or launch a zero-row transfer (the
    kernel backend's table upload refuses one), under either io backend."""

    def fake_load(transfers, *, geometry):
        assert transfers, "the workspace was asked to stage zero rows"
        return len(transfers), (0, len(transfers))

    def fake_commit(num_blocks, device, non_blocking):
        assert num_blocks > 0, "the Device table upload was asked for zero rows"

    def fake_transfer(direction, *args, num_blocks, **kwargs):
        assert num_blocks > 0, "the transport was asked for a zero-row transfer"

    with patch.object(lanes_module, "HostTransferWorkspace", Mock):
        lane = HostTransferLane()
    lane.workspace.load_block_transfers.side_effect = fake_load
    lane.workspace.commit_block_transfers.side_effect = fake_commit
    lane.workspace.prepare_backend.return_value = SimpleNamespace(
        uses_device_tables=backend == "auto"
    )
    stream = Mock(name="stream")
    events = []
    launch = dict(
        device_buffers=(SimpleNamespace(device="cuda"),),
        host_buffer="host",
        geometry=SimpleNamespace(num_field_rows=3),
        stream=stream,
        prerequisite_stream="prerequisite",
        backend=backend,
    )
    with (
        patch.object(
            lanes_module, "transfer_cache_blocks", side_effect=fake_transfer
        ) as transfer,
        patch.object(lanes_module.device_module, "stream", return_value=nullcontext()),
        patch.object(
            lanes_module.device_module,
            "Event",
            side_effect=lambda: events.append(Mock(name="event")) or events[-1],
        ),
    ):
        d2h = lane.start_d2h([], **launch)
        h2d = lane.start_h2d([], **launch)
        # A non-empty copy still goes through the transport.
        lane.start_d2h([(0, 1, 1)], **launch)
    assert (d2h, h2d) == (events[0], events[1])
    for finish in (d2h, h2d):
        finish.record.assert_called_once_with(stream)
    assert stream.wait_stream.call_args_list == [call("prerequisite")] * 3
    assert transfer.call_count == 1
    assert transfer.call_args.args[0] == "d2h"
    assert transfer.call_args.kwargs["num_blocks"] == 1
    assert lane.workspace.load_block_transfers.call_count == 1
    assert lane.workspace.commit_block_transfers.call_count == (
        1 if backend == "auto" else 0
    )
    assert lane.metadata_done is (events[2] if backend == "auto" else None)


def test_completion_queue_releases_in_order_and_drops_on_reset():
    queue = CompletionQueue()
    first, second = Mock(), Mock()
    first.query.return_value = False
    second.query.return_value = True
    queue.push(first, "first")
    queue.push(second, "second")
    assert queue.pop_ready() == ["second"]
    first.query.return_value = True
    assert queue.pop_ready() == ["first"]
    assert queue.pop_ready() == []
    queue.push(Mock(), "stale")
    assert queue.drop_all() == ["stale"]
    assert queue.pop_ready() == []


# ----------------------------------------------------------------------
# Real bytes through two KVP ranks and both tiers (GPU)
# ----------------------------------------------------------------------


class _CudaSlotState:
    """A two-row owner whose slot state lives on the device."""

    def __init__(self, rows: int, width: int):
        self.table = torch.zeros((rows, width), dtype=torch.int32, device="cuda")
        self.flag = torch.zeros(rows, dtype=torch.bool, device="cuda")

    def slot_state_rows(self, slot):
        return [self.table[slot], self.flag[slot]]

    def slot_state_bytes(self):
        return slot_state_image_bytes(self.slot_state_rows(0))

    def export_slot_state(self, slot, out, stream, *, request_id):
        pack_slot_rows(self.slot_state_rows(slot), out, stream)

    def import_slot_state(self, slot, src, stream, *, request_id):
        unpack_slot_rows(self.slot_state_rows(slot), src, stream)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("io_backend", ["direct", "kernel"])
def test_two_ranks_round_trip_a_sharded_image_through_both_tiers(io_backend):
    """Two ranks (two device buffers, two executors) image their owned
    subsets of one sharded group -- the lower half through the stream-ordered
    L2 write-back, the upper half through the snapshot store -- plus their
    slot state; wiping both devices and restoring through one two-tier op
    brings every rank's bytes back, so the owner filters cover each op
    exactly once and address the right local blocks in the right buffer. A
    second victim whose image is its slot state alone rides empty ops."""
    executors = {}
    try:
        _round_trip_two_ranks(io_backend, executors)
    finally:
        for executor in executors.values():
            executor.shutdown()


def _round_trip_two_ranks(io_backend, executors):
    torch.manual_seed(0)
    # Per rank: 2 local LCM blocks x packing 2 = 4 local blocks of 8 B at
    # stride 16 from offset 8. The sharded group has 2 x 2 x 2 = 8 virtual
    # blocks; rank r owns v with (v - 1) % 2 == r at local (v - 1) // 2 + 1.
    # Both Host pools hold 3 LCM blocks -> 12 virtual blocks each.
    contract = SimpleNamespace(
        group_specs=(SimpleNamespace(group_id="full", shard_count=2),),
        virtual_block_counts={"full": 1 + 2 * 2 * 2},
        num_lcm_blocks=2,
        token_capacity=32,
    )
    buffers = {
        rank: torch.full((128,), 0xCC, dtype=torch.uint8, device="cuda")
        for rank in (0, 1)
    }
    slot_states = {rank: _CudaSlotState(rows=4, width=5) for rank in (0, 1)}
    with patch.object(executor_module, "_HOST_MEM_HEADROOM_BYTES", 0):
        for rank in (0, 1):
            layout = CacheTransferLayout(
                num_lcm_blocks=2,
                groups=(
                    CacheGroupLayout(
                        "full", 2, (CacheField("layer.0.k", 0, 8, 16, 8),)
                    ),
                ),
                buffers=(buffers[rank],),
                consumers=(("layer.0.k",),),
            )
            pool = SimpleNamespace(
                cache_transfer_layout=lambda layout=layout: layout,
                register_layerwise_load_tracker=lambda tracker: None,
                arena=SimpleNamespace(
                    cache_group_specs=contract.group_specs, runtime_contract=contract
                ),
            )
            executors[rank] = HostCacheExecutor(
                pool,
                draft_pool=None,
                l2_tier=True,
                host_ratio=1.5,  # 3 L2 LCM blocks
                host_size_gb=0,
                snapshot_pool=_pool(3 * 16 / 1e9, 2),  # 3 pool LCM blocks of 2 x 8 B
                slot_state_exporters=(slot_states[rank],),
                io_backend=io_backend,
                attn_tp_rank=0,
                kvp_rank=rank,
            )
    assert (executors[0].num_host_pages, executors[0].num_snapshot_pages) == (4, 4)

    originals = {
        rank: torch.randint(1, 255, (128,), dtype=torch.uint8, device="cuda")
        for rank in (0, 1)
    }
    for rank in (0, 1):
        buffers[rank].copy_(originals[rank])
        slot_states[rank].table[1].fill_(100 + rank)
        slot_states[rank].flag[1] = True
        # The second victim (slot 2) has nothing in the pool: slot state only.
        slot_states[rank].table[2].fill_(200 + rank)
    torch.cuda.synchronize()
    # Device blocks 1..4 go to L2 blocks 5..8 (stream-ordered), 5..8 to pool
    # blocks 1..4; every pair keeps its parity, i.e. its owner.
    l2_store = _WriteBackOp(
        [21], [[0, 0, 0, 0]], [[1, 2, 3, 4]], [[5, 6, 7, 8]], [False]
    )
    pool_store = _snapshot_op(
        1, 0, [_transfer(0, v, v - 4) for v in range(5, 9)], pool_index=1
    )
    empty_store = _snapshot_op(3, 1, [], pool_index=2)
    stream = torch.cuda.current_stream()
    with patch.object(executor_module.Cache, "WriteBackOp", _WriteBackOp, create=True):
        for executor in executors.values():
            executor.submit_write_backs(
                [l2_store, pool_store, empty_store],
                prerequisite_stream=stream,
                fence_stream=stream,
            )
    torch.cuda.synchronize()
    for executor in executors.values():
        acks = _acks(executor.poll_results())
        assert ("SnapshotDoneEvent", 1) in acks and ("SnapshotDoneEvent", 3) in acks
        assert [op_id for kind, op_id in acks if kind == "WriteBackDoneEvent"] == [21]

    for rank in (0, 1):
        buffers[rank].fill_(0xEE)
        slot_states[rank].table.zero_()
        slot_states[rank].flag.zero_()
    torch.cuda.synchronize()
    restore = _restore_op(
        2,
        0,
        [(HostTier.L2, _transfer(0, v + 4, v, key=f"h{v}")) for v in range(1, 5)]
        + [(HostTier.SNAPSHOT_POOL, _transfer(0, v - 4, v)) for v in range(5, 9)],
        pool_index=3,
    )
    empty_restore = _restore_op(4, 1, [], pool_index=0)
    for executor in executors.values():
        executor.submit_load_backs(
            [restore, empty_restore],
            prerequisite_stream=stream,
        )
    torch.cuda.synchronize()
    assert [_acks(executor.poll_results()) for executor in executors.values()] == [
        _restore_done(2, 4),
        _restore_done(2, 4),
    ]
    for rank in (0, 1):
        expected = torch.full((128,), 0xEE, dtype=torch.uint8)
        for local in range(1, 5):
            lo, hi = 8 + 16 * local, 16 + 16 * local
            expected[lo:hi] = originals[rank][lo:hi].cpu()
        assert torch.equal(buffers[rank].cpu(), expected), f"rank {rank}"
        assert slot_states[rank].table[3].tolist() == [100 + rank] * 5
        assert bool(slot_states[rank].flag[3]) and not bool(slot_states[rank].flag[1])
        assert slot_states[rank].table[0].tolist() == [200 + rank] * 5
        assert not bool(slot_states[rank].flag[0])
