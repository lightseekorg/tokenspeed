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
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import tokenspeed.runtime.cache.l2.executor as executor_module  # noqa: E402
from tokenspeed.runtime.cache.l2.executor import (  # noqa: E402
    HostCacheExecutor,
    num_snapshot_lcm_blocks,
)
from tokenspeed.runtime.cache.transfer.lanes import CompletionQueue  # noqa: E402
from tokenspeed.runtime.cache.transfer.layout import (  # noqa: E402
    CacheField,
    CacheGroupLayout,
    CacheTransferLayout,
)
from tokenspeed.runtime.cache.transfer.ops import (  # noqa: E402
    CacheTransfer,
    HostTier,
    RestoreDoneEvent,
    RestoreOp,
    SnapshotDoneEvent,
    SnapshotOp,
)
from tokenspeed.runtime.cache.transfer.ownership import (  # noqa: E402
    BlockOwnerTranslation,
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

    def export_slot_state(self, slot, out, stream):
        self.exports.append((slot, out, stream))

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
            snapshot_host_gb=snapshot_host_gb,
            max_retracted_requests=max_retracted if snapshot_host_gb > 0 else 0,
            slot_state=slot_state,
            io_backend="direct",
            attn_tp_rank=0,
            dcp_rank=rank,
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
    for event in (SnapshotDoneEvent(op_id=3), RestoreDoneEvent(op_id=4)):
        payload = cache_event_to_payload(event)
        assert payload == {"kind": type(event).__name__, "op_id": event.op_id}
        assert cache_event_from_payload(payload) == event


def test_num_snapshot_lcm_blocks_is_explicit_about_zero():
    assert num_snapshot_lcm_blocks(host_gb=0.0035, host_lcm_block_bytes=1_000_000) == 3
    with pytest.raises(ValueError, match="positive"):
        num_snapshot_lcm_blocks(host_gb=0, host_lcm_block_bytes=1)
    with pytest.raises(ValueError, match="no whole LCM block"):
        num_snapshot_lcm_blocks(host_gb=0.0005, host_lcm_block_bytes=1_000_000)


def test_constructor_requires_a_tier_and_matching_pool_knobs():
    layout = _layout(2, [("full", 2)])
    with pytest.raises(ValueError, match="L2 tier, a snapshot pool or both"):
        _build(layout=layout, shard_counts=[1], l2_tier=False, snapshot_host_gb=0)
    with pytest.raises(ValueError, match="go together"):
        with patch.object(
            executor_module, "compute_host_lcm_block_bytes", return_value=1
        ):
            HostCacheExecutor(
                _Pool(layout, [1]),
                l2_tier=True,
                host_ratio=1.0,
                host_size_gb=0,
                snapshot_host_gb=0.0,
                max_retracted_requests=3,
                slot_state=None,
                io_backend="direct",
                attn_tp_rank=0,
                dcp_rank=0,
            )
    executor, _, _ = _build(layout=layout, shard_counts=[1], snapshot_host_gb=0)
    assert (executor.num_host_pages, executor.num_snapshot_pages) == (4, 1)
    assert executor.max_retracted_requests == 0 and executor.blob_arena is None
    executor, _, _ = _build(layout=layout, shard_counts=[1], l2_tier=False)
    assert (executor.num_host_pages, executor.num_snapshot_pages) == (0, 4)
    assert executor.host_storage is None and executor._load_trackers == []


def test_count_plan_ops_counts_a_snapshot_or_restore_once():
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
    plan = SimpleNamespace(
        cache=[
            _snapshot_op(1, 0, [_transfer(0, 1, 1)]),
            _restore_op(2, 0, [(HostTier.L2, _transfer(0, 1, 1))]),
        ]
    )
    hooks.count_plan_ops(plan)
    assert hooks._num_inflight == 2


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
    # Grouped by group index, input order kept within a group.
    assert owners.owned_rows(rows) == [(0, 8, 16), (0, 1, 1), (1, 4, 2)]
    with pytest.raises(ValueError, match="null block"):
        owners.owned_rows([(0, 0, 1)])
    with pytest.raises(IndexError, match="unknown group"):
        owners.owned_rows([(2, 1, 1)])
    with pytest.raises(IndexError):
        owners.owned_rows([(0, 9, 1)])
    with pytest.raises(IndexError):
        owners.owned_rows([(1, 1, 3)])


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
    assert owners[0].owned_rows(rows) == [(0, 1, 2), (0, 3, 6)]
    assert owners[1].owned_rows(rows) == [(0, 1, 3), (0, 4, 6)]
    assert len(owners[0].owned_rows(rows)) + len(owners[1].owned_rows(rows)) == len(
        rows
    )
    with pytest.raises(ValueError, match="residue class"):
        owners[0].owned_rows([(0, 1, 2)])


# ----------------------------------------------------------------------
# The executor over fakes
# ----------------------------------------------------------------------


def test_both_store_legs_ride_the_write_stream_under_one_fence():
    layout = _layout(4, [("full", 4), ("state", 1)])
    executor, slot_state, lanes = _build(layout=layout, shard_counts=[1, 1])
    plan = SimpleNamespace(
        cache=[
            # One L2 batch: the victim's hash-complete pages (stream-ordered)
            # and another request's ordinary publication (pinned).
            _WriteBackOp([11, 12], [[0], [0]], [[1], [3]], [[5], [7]], [False, True]),
            # The same victim's tail pages and slot state.
            _snapshot_op(7, 1, [_transfer(0, 2, 2), _transfer(1, 1, 1)], pool_index=3),
            _snapshot_op(8, 2, [_transfer(0, 4, 6)], pool_index=6),
        ]
    )
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
            plan, prerequisite_stream="execution-stream", fence_stream=fence_stream
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
    # Each victim's slot is exported into its arena row, on the write stream.
    assert [(slot, stream) for slot, _, stream in slot_state.exports] == [
        (3, executor.write_stream),
        (6, executor.write_stream),
    ]
    assert slot_state.exports[0][1].data_ptr() == executor.blob_arena[1].data_ptr()
    assert slot_state.exports[1][1].data_ptr() == executor.blob_arena[2].data_ptr()
    # Three ACK kinds, each released by its own event.
    assert executor.poll_results() == []
    snapshot_finish.query.return_value = True
    assert executor.poll_results() == [SnapshotDoneEvent(7), SnapshotDoneEvent(8)]
    ordered_finish.query.return_value = True
    pinned_finish.query.return_value = True
    assert sorted(int(e.op_id) for e in executor.poll_results()) == [11, 12]
    assert executor.poll_results() == []
    del events


def test_store_with_every_page_in_l2_images_the_slot_state_alone():
    layout = _layout(2, [("full", 2)])
    executor, slot_state, lanes = _build(layout=layout, shard_counts=[1])
    finish = Mock()
    finish.query.return_value = True
    with patch.object(executor_module.device_module, "Event", return_value=finish):
        executor.submit_write_backs(
            SimpleNamespace(cache=[_snapshot_op(1, 0, [], pool_index=2)]),
            prerequisite_stream="x",
            fence_stream=Mock(),
        )
    lanes["snapshot"].start_d2h.assert_not_called()
    lanes["ordered"].start_d2h.assert_not_called()
    assert [slot for slot, _, _ in slot_state.exports] == [2]
    assert executor.poll_results() == [SnapshotDoneEvent(1)]


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
            SimpleNamespace(cache=[op]),
            prerequisite_stream="default-stream",
            l3_prefetch_ok={},
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
    assert executor.poll_results() == [RestoreDoneEvent(9)]


def test_restore_rows_need_their_tier_and_run_outside_capture():
    layout = _layout(2, [("full", 2)])
    # Without an L2 tier a restore naming L2 rows is a configuration error.
    executor, _, _ = _build(layout=layout, shard_counts=[1], l2_tier=False)
    l2_row = _restore_op(1, 0, [(HostTier.L2, _transfer(0, 1, 2, key="h"))])
    with patch.object(executor_module, "get_is_capture_mode", return_value=False):
        with pytest.raises(RuntimeError, match="no L2 tier"):
            executor.submit_load_backs(
                SimpleNamespace(cache=[l2_row]),
                prerequisite_stream="s",
                l3_prefetch_ok={},
            )
    pool_row = _restore_op(2, 0, [(HostTier.SNAPSHOT_POOL, _transfer(0, 1, 2))])
    with patch.object(executor_module, "get_is_capture_mode", return_value=True):
        with pytest.raises(RuntimeError, match="graph capture"):
            executor.submit_load_backs(
                SimpleNamespace(cache=[pool_row]),
                prerequisite_stream="s",
                l3_prefetch_ok={},
            )
    # Without a pool, snapshot ops are refused outright.
    executor, _, _ = _build(layout=layout, shard_counts=[1], snapshot_host_gb=0)
    with pytest.raises(RuntimeError, match="retraction snapshot pool"):
        executor.submit_write_backs(
            SimpleNamespace(cache=[_snapshot_op(3, 0, [])]),
            prerequisite_stream="s",
            fence_stream=Mock(),
        )


def test_plan_level_checks_refuse_duplicates_bad_slots_and_ragged_tiers():
    layout = _layout(2, [("full", 2)])
    executor, _, _ = _build(layout=layout, shard_counts=[1], max_retracted=2)
    fence = Mock()
    with pytest.raises(ValueError, match="duplicate snapshot op id"):
        executor.submit_write_backs(
            SimpleNamespace(cache=[_snapshot_op(1, 0, []), _snapshot_op(1, 1, [])]),
            prerequisite_stream="s",
            fence_stream=fence,
        )
    with pytest.raises(ValueError, match="duplicate snapshot slot"):
        executor.submit_write_backs(
            SimpleNamespace(cache=[_snapshot_op(1, 0, []), _snapshot_op(2, 0, [])]),
            prerequisite_stream="s",
            fence_stream=fence,
        )
    with pytest.raises(IndexError, match="snapshot slot 2"):
        executor.submit_write_backs(
            SimpleNamespace(cache=[_snapshot_op(1, 2, [])]),
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
    with pytest.raises(ValueError, match="ragged cache operation 5"):
        executor.submit_load_backs(
            SimpleNamespace(cache=[ragged]), prerequisite_stream="s", l3_prefetch_ok={}
        )
    executor.write_stream.wait_stream.assert_not_called()
    executor.load_stream.wait_stream.assert_not_called()


def test_kvp_ranks_store_their_owned_subsets_on_both_legs():
    # One sharded group (D = 2): rank r keeps rows whose Device block has
    # (v - 1) % 2 == r; the scheduler placed every Host block in the same class.
    layout = _layout(2, [("full", 2)])
    l2_rows = [11, 12]
    seen = {}
    for rank in (0, 1):
        executor, slot_state, lanes = _build(layout=layout, shard_counts=[2], rank=rank)
        plan = SimpleNamespace(
            cache=[
                _WriteBackOp(
                    l2_rows, [[0], [0]], [[1], [2]], [[3], [6]], [False, False]
                ),
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
        )
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
                plan, prerequisite_stream="s", fence_stream=Mock()
            )
        seen[rank] = (
            lanes["ordered"].start_d2h.call_args.args[0],
            lanes["snapshot"].start_d2h.call_args.args[0],
        )
        # Slot state is per rank: every rank exports its own and ACKs every op.
        assert [slot for slot, _, _ in slot_state.exports] == [1]
        acks = executor.poll_results()
        assert SnapshotDoneEvent(5) in acks
        assert (
            sorted(int(e.op_id) for e in acks if not isinstance(e, SnapshotDoneEvent))
            == l2_rows
        )
    assert seen == {
        0: ([(0, 1, 2)], [(0, 3, 1), (0, 4, 2)]),
        1: ([(0, 1, 3)], [(0, 3, 2), (0, 4, 5)]),
    }
    bad = _snapshot_op(6, 1, [_transfer(0, 1, 2)])
    with pytest.raises(ValueError, match="residue class"):
        executor.submit_write_backs(
            SimpleNamespace(cache=[bad]), prerequisite_stream="s", fence_stream=Mock()
        )


def test_completion_queue_releases_in_order_and_drops_on_reset():
    queue = CompletionQueue()
    first, second = Mock(), Mock()
    first.query.return_value = False
    second.query.return_value = True
    queue.push(first, "first")
    queue.push(second, "second")
    assert len(queue) == 2
    assert queue.pop_ready() == ["second"]
    first.query.return_value = True
    assert queue.pop_ready() == ["first"]
    queue.push(Mock(), "stale")
    assert queue.drop_all() == ["stale"]
    assert len(queue) == 0


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

    def export_slot_state(self, slot, out, stream):
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
    exactly once and address the right local blocks in the right buffer."""
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
    executors = {}
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
                snapshot_host_gb=3 * 16 / 1e9,  # 3 pool LCM blocks of 2 x 8 B
                max_retracted_requests=2,
                slot_state=slot_states[rank],
                io_backend=io_backend,
                attn_tp_rank=0,
                dcp_rank=rank,
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
    torch.cuda.synchronize()
    # Device blocks 1..4 go to L2 blocks 5..8 (stream-ordered), 5..8 to pool
    # blocks 1..4; every pair keeps its parity, i.e. its owner.
    l2_store = _WriteBackOp(
        [21], [[0, 0, 0, 0]], [[1, 2, 3, 4]], [[5, 6, 7, 8]], [False]
    )
    pool_store = _snapshot_op(
        1, 0, [_transfer(0, v, v - 4) for v in range(5, 9)], pool_index=1
    )
    stream = torch.cuda.current_stream()
    with patch.object(executor_module.Cache, "WriteBackOp", _WriteBackOp, create=True):
        for executor in executors.values():
            executor.submit_write_backs(
                SimpleNamespace(cache=[l2_store, pool_store]),
                prerequisite_stream=stream,
                fence_stream=stream,
            )
    torch.cuda.synchronize()
    for executor in executors.values():
        acks = executor.poll_results()
        assert SnapshotDoneEvent(1) in acks
        assert [int(e.op_id) for e in acks if not isinstance(e, SnapshotDoneEvent)] == [
            21
        ]

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
    for executor in executors.values():
        executor.submit_load_backs(
            SimpleNamespace(cache=[restore]),
            prerequisite_stream=stream,
            l3_prefetch_ok={},
        )
    torch.cuda.synchronize()
    assert [executor.poll_results() for executor in executors.values()] == [
        [RestoreDoneEvent(2)],
        [RestoreDoneEvent(2)],
    ]
    for rank in (0, 1):
        expected = torch.full((128,), 0xEE, dtype=torch.uint8)
        for local in range(1, 5):
            lo, hi = 8 + 16 * local, 16 + 16 * local
            expected[lo:hi] = originals[rank][lo:hi].cpu()
        assert torch.equal(buffers[rank].cpu(), expected), f"rank {rank}"
        assert slot_states[rank].table[3].tolist() == [100 + rank] * 5
        assert bool(slot_states[rank].flag[3]) and not bool(slot_states[rank].flag[1])
    for executor in executors.values():
        executor.shutdown()
