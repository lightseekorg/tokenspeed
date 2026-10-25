"""Compact Host cache executor tests."""

from __future__ import annotations

import os
import sys
import threading
from contextlib import ExitStack, nullcontext
from importlib import import_module, util
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

# Pure Python, no runtime dependencies: the knobs of an L2-only executor.
from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest  # noqa: E402

NO_POOL = RetractionPoolRequest(
    host_gb=0.0, ratio=0.0, max_retracted_requests=0, tail_lcm_blocks_per_request=0
)


class _LoadEvents(SimpleNamespace):
    def set_completion(self, event):
        self.layer_done_events[:] = [event] * len(self.layer_done_events)


def _replicated_contract(layout, *, num_lcm_blocks=None):
    """An arena contract for ``layout`` with every group replicated (shard 1)."""
    if num_lcm_blocks is None:
        num_lcm_blocks = layout.num_lcm_blocks
    specs = tuple(
        SimpleNamespace(group_id=group.group_id, shard_count=1)
        for group in layout.groups
    )
    return SimpleNamespace(
        group_specs=specs,
        virtual_block_counts={
            group.group_id: 1 + num_lcm_blocks * group.cache_blocks_per_lcm_block
            for group in layout.groups
        },
        num_lcm_blocks=num_lcm_blocks,
        token_capacity=num_lcm_blocks,
    )


def _synthetic_arena(layout):
    contract = _replicated_contract(layout)
    return SimpleNamespace(
        cache_group_specs=contract.group_specs, runtime_contract=contract
    )


class _SyntheticPool:
    def __init__(self, layout, arena=None):
        self._layout = layout
        if arena is None:
            arena = _synthetic_arena(layout)
        self.arena = arena

    def cache_transfer_layout(self):
        return self._layout

    def register_layerwise_load_tracker(self, tracker):
        self.load_tracker = tracker


def _load_executor_module_without_triton(*, force_isolated=False):
    """Load executor orchestration when optional Triton is not installed."""

    # Keep real dependencies outside the temporary sys.modules snapshot.
    # Otherwise the first isolated load removes psutil again on exit.
    import_module("psutil")

    if not force_isolated:
        executor_name = "tokenspeed.runtime.cache.l2.executor"
        if executor_name in sys.modules or util.find_spec("tokenspeed_triton"):
            return import_module("tokenspeed.runtime.cache.l2.executor")
    host_transfer = ModuleType("tokenspeed_kernel.ops.kvcache.host_transfer")
    host_transfer.HostTransferWorkspace = Mock
    host_transfer.build_host_transfer_geometry = Mock()
    host_transfer.transfer_cache_blocks = Mock()
    host_transfer.wait_layer_ready = Mock()
    ownership = ModuleType("tokenspeed.runtime.cache.transfer.ownership")
    ownership.BlockOwnerTranslation = Mock
    lanes = ModuleType("tokenspeed.runtime.cache.transfer.lanes")
    lanes.CompletionQueue = Mock
    lanes.HostTransferLane = Mock
    lanes.build_transfer_geometry = Mock()
    lanes.check_host_memory = Mock()
    lanes.host_memory_share = Mock(return_value=10**12)
    lanes.largest_pinned_fit = Mock(side_effect=lambda room_bytes: max(room_bytes, 0))
    lanes.pinned_host_bytes = Mock(side_effect=lambda nbytes: nbytes)
    lanes.load_stream_priority = Mock(return_value=None)
    lanes.new_cache_stream = Mock()
    scheduler = ModuleType("tokenspeed_scheduler")

    class Cache:
        class WriteBackOp:
            pass

        class LoadBackOp:
            pass

        class PrefetchOp:
            pass

        class WriteBackDoneEvent:
            pass

        class LoadBackDoneEvent:
            def __init__(self, op_id):
                self.op_id = op_id

        class PrefetchDoneEvent:
            def __init__(self, op_id, landed_pages):
                self.op_id = op_id
                self.landed_pages = landed_pages

    scheduler.Cache = Cache
    layerwise_load = ModuleType("tokenspeed.runtime.cache.l2.layerwise_load")
    layerwise_load.LayerwiseLoadTracker = Mock
    storage = ModuleType("tokenspeed.runtime.cache.host_storage")
    storage.HostCacheStorage = Mock
    storage.compute_host_lcm_block_bytes = Mock(return_value=1)
    layout = ModuleType("tokenspeed.runtime.cache.transfer.layout")
    layout.combine_cache_transfer_layouts = lambda target, draft, group_ids=None: (
        target if draft is None else draft
    )
    forward_step = ModuleType("tokenspeed.runtime.execution.forward_step")
    forward_step.get_is_capture_mode = Mock(return_value=False)
    runtime_utils = ModuleType("tokenspeed.runtime.utils")
    runtime_utils.get_colorful_logger = Mock(return_value=Mock())
    runtime_utils.get_device_module = Mock(return_value=Mock())
    fake_modules = {
        "tokenspeed_kernel.ops.kvcache.host_transfer": host_transfer,
        "tokenspeed_scheduler": scheduler,
        "tokenspeed.runtime.cache.l2.layerwise_load": layerwise_load,
        "tokenspeed.runtime.cache.host_storage": storage,
        "tokenspeed.runtime.cache.transfer.lanes": lanes,
        "tokenspeed.runtime.cache.transfer.layout": layout,
        "tokenspeed.runtime.cache.transfer.ownership": ownership,
        "tokenspeed.runtime.execution.forward_step": forward_step,
        "tokenspeed.runtime.utils": runtime_utils,
    }
    executor_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "python",
            "tokenspeed",
            "runtime",
            "cache",
            "l2",
            "executor.py",
        )
    )
    spec = util.spec_from_file_location("_isolated_l2_executor", executor_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load isolated executor from {executor_path}")
    executor_module = util.module_from_spec(spec)
    with patch.dict(sys.modules, fake_modules, clear=False):
        spec.loader.exec_module(executor_module)
    return executor_module


def _lanes_module():
    """The shared Host-transfer lane module the L2 write path delegates to."""
    return import_module("tokenspeed.runtime.cache.transfer.lanes")


def _identity_owners(num_groups=1, bound=64):
    """A replicated (shard 1) owner translation: local ids are scheduler ids."""
    from tokenspeed.runtime.cache.transfer.ownership import BlockOwnerTranslation

    return BlockOwnerTranslation(
        shard_counts=[1] * num_groups,
        device_virtual_counts=[bound] * num_groups,
        host_virtual_counts=[bound] * num_groups,
        rank=0,
    )


class TestGroupAwareWireTest:
    def _executor_module(self):
        try:
            return _load_executor_module_without_triton()
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs runtime dependencies: {exc}")

    def _make_load_executor(
        self,
        *,
        consumers,
        layer_slices,
        backend="auto",
        device_rows=None,
        load_stream=None,
    ):
        executor_module = self._executor_module()
        executor = executor_module.HostCacheExecutor.__new__(
            executor_module.HostCacheExecutor
        )
        executor._ack_lock = threading.Lock()
        executor.attn_tp_rank = 0
        executor._ready_load_acks = []
        executor._completions = _lanes_module().CompletionQueue()
        executor._load_poisoned = False
        executor.load_stream = object() if load_stream is None else load_stream
        executor.transfer_backend = backend
        device = SimpleNamespace(type="cuda")
        executor.layout = SimpleNamespace(
            buffers=(SimpleNamespace(device=device),),
            consumers=consumers,
        )
        executor.host_storage = SimpleNamespace(host_buffer="host")
        geometry = SimpleNamespace(
            layer_slices=layer_slices,
            device_rows=device_rows,
            num_field_rows=sum(count for _, count in layer_slices),
        )
        executor._transfer_geometry = geometry
        workspace = MagicMock()
        workspace.load_block_transfers.return_value = (1, (0, 1))
        workspace.prepare_backend.return_value = SimpleNamespace(
            uses_device_tables=device_rows is not None,
            layer_ready=device_rows is not None,
        )
        executor._load_workspaces = (workspace,)
        return executor_module, executor, device, geometry, workspace

    def test_hybrid_state_access_waits_for_layer_load(self, subtests):
        try:
            from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
                HybridKDATokenToKVPool,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
                HybridMHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs runtime dependencies: {exc}")

        for pool_type in (HybridMHATokenToKVPool, HybridKDATokenToKVPool):
            with subtests.test(pool_type=pool_type.__name__):
                tracker = Mock()
                pool = pool_type.__new__(pool_type)
                pool.layerwise_load_tracker = tracker
                pool._state_buffers_by_layer = {3: ("conv", "recurrent")}
                if pool_type is HybridMHATokenToKVPool:
                    pool._state_layer_ids = (3,)

                assert (pool.get_component(3, "conv_state")) == ("conv")
                tracker.wait_for_layer.assert_called_once_with(3)

    def test_pool_transfer_layout_matches_scheduler_group_order(self):
        try:
            from test.runtime.cache_pool_test_utils import MinimalCacheView
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs runtime dependencies: {exc}")

        pool = MinimalCacheView.__new__(MinimalCacheView)
        pool.layer_num = 2
        pool._field_layer_offset = 0
        # The arena owns the buffer, the field views and the published specs;
        # the pool only answers for them.
        pool.arena = SimpleNamespace(
            buffer=object(),
            cache_group_specs=(
                SimpleNamespace(group_id="state"),
                SimpleNamespace(group_id="full"),
            ),
        )
        pool.arena.plan = SimpleNamespace(
            num_lcm_blocks=4,
            planes=(
                SimpleNamespace(
                    plane_id="shared",
                    bytes_per_lcm_block=4096,
                    arena_offset_bytes=0,
                ),
            ),
            groups=(
                SimpleNamespace(
                    group_id="full",
                    cache_blocks_per_lcm_block=32,
                ),
                SimpleNamespace(
                    group_id="state",
                    cache_blocks_per_lcm_block=1,
                ),
            ),
            fields=(
                SimpleNamespace(
                    group_id="full",
                    field_id="layer.1.k",
                    plane_id="shared",
                    field_offset_bytes=0,
                    page_stride_bytes=128,
                    payload_bytes=128,
                ),
                SimpleNamespace(
                    group_id="state",
                    field_id="layer.0.state",
                    plane_id="shared",
                    field_offset_bytes=0,
                    page_stride_bytes=4096,
                    payload_bytes=4096,
                ),
            ),
        )

        layout = pool.cache_transfer_layout()

        assert (tuple(group.group_id for group in layout.groups)) == (("state", "full"))

    def test_submit_load_backs_clears_layerwise_waits_without_load(self):
        HostCacheExecutor = self._executor_module().HostCacheExecutor

        tracker = Mock()
        executor = HostCacheExecutor.__new__(HostCacheExecutor)
        executor._ack_lock = threading.Lock()
        executor._load_trackers = [(tracker, 1)]
        executor._load_poisoned = False
        executor.block_owners = _identity_owners()

        executor.submit_load_backs([], prerequisite_stream=object())

        tracker.set_consumers.assert_called_once_with(-1)

    def _owner_executor(self, *, num_groups=2):
        """An executor with replicated (identity) owner translations only."""
        executor_module = self._executor_module()
        executor = executor_module.HostCacheExecutor.__new__(
            executor_module.HostCacheExecutor
        )
        executor.block_owners = _identity_owners(num_groups)
        executor.snapshot_block_owners = _identity_owners(num_groups)
        return executor_module, executor

    def test_submit_preserves_group_identity(self):
        from tokenspeed.runtime.cache.transfer.ops import HostTier

        _, executor = self._owner_executor()
        op_ids = []
        transfers = []
        executor._append_transfers(
            [7],
            [[0, 1]],
            [[5, 5]],
            [[9, 9]],
            collected_op_ids=op_ids,
            transfers=transfers,
            source_is_device=True,
            tier=HostTier.L2,
        )
        assert (op_ids) == ([7])
        assert (transfers) == ([(0, 5, 9), (1, 5, 9)])

    def test_sharded_rows_keep_only_this_ranks_owned_blocks_on_both_ends(
        self, subtests
    ):
        # A group dealt to two owners: rank r keeps the rows whose Device AND
        # Host block have residue r, in local ids; the op is collected even
        # when every row belongs to the other rank.
        from tokenspeed.runtime.cache.transfer.ops import HostTier
        from tokenspeed.runtime.cache.transfer.ownership import BlockOwnerTranslation

        executor_module = self._executor_module()
        for rank, expected in ((0, [(0, 1, 2), (0, 2, 3)]), (1, [(0, 1, 1)])):
            with subtests.test(rank=rank):
                executor = executor_module.HostCacheExecutor.__new__(
                    executor_module.HostCacheExecutor
                )
                executor.block_owners = BlockOwnerTranslation(
                    shard_counts=[2],
                    device_virtual_counts=[9],
                    host_virtual_counts=[9],
                    rank=rank,
                )
                op_ids: list[int] = []
                transfers: list[tuple[int, int, int]] = []
                executor._append_transfers(
                    [7],
                    [[0, 0, 0]],
                    [[1, 3, 2]],
                    [[3, 5, 2]],
                    collected_op_ids=op_ids,
                    transfers=transfers,
                    source_is_device=True,
                    tier=HostTier.L2,
                )
                assert (op_ids) == ([7])
                assert (transfers) == (expected)
        with pytest.raises((ValueError), match=("residue class")):
            executor._append_transfers(
                [8],
                [[0]],
                [[1]],
                [[2]],
                collected_op_ids=[],
                transfers=[],
                source_is_device=True,
                tier=HostTier.L2,
            )

    def _make_write_executor(self, executor_module):
        HostCacheExecutor = executor_module.HostCacheExecutor
        lanes_module = _lanes_module()
        executor = HostCacheExecutor.__new__(HostCacheExecutor)
        executor._ack_lock = threading.Lock()
        executor.attn_tp_rank = 0
        device = SimpleNamespace(type="cuda")
        executor.layout = SimpleNamespace(buffers=(SimpleNamespace(device=device),))
        executor.host_storage = SimpleNamespace(host_buffer="host")
        executor.transfer_backend = "auto"
        executor._completions = lanes_module.CompletionQueue()
        executor.block_owners = _identity_owners()
        executor.write_stream = Mock(name="write_stream")
        # Real lanes over mocked workspaces: the staging discipline under test
        # is the lane's, the executor only picks which lane an op rides.
        for lane_name in ("_ordered_write_lane", "_pinned_write_lane"):
            with patch.object(lanes_module, "HostTransferWorkspace", Mock):
                lane = lanes_module.HostTransferLane()
            lane.workspace.load_block_transfers.return_value = (1, (0, 1))
            lane.workspace.prepare_backend.return_value = SimpleNamespace(
                uses_device_tables=True,
            )
            setattr(executor, lane_name, lane)
        executor._transfer_geometry = SimpleNamespace(
            device_rows=object(),
            layer_slices=((0, 2), (2, 1)),
            num_field_rows=3,
        )
        return executor, device

    def test_writeback_rides_the_write_stream_ordered_after_the_caller(self, subtests):
        executor_module = self._executor_module()
        lanes_module = _lanes_module()
        executor, device = self._make_write_executor(executor_module)
        lane = executor._pinned_write_lane
        prerequisite_stream = object()
        finish = Mock()
        metadata_done = Mock()
        # Every stream the executor touches is one the caller named; the
        # thread's current stream is never consulted.
        no_current_stream = patch.object(
            lanes_module.device_module,
            "current_stream",
            side_effect=AssertionError("current stream must not be consulted"),
        )

        with (
            no_current_stream,
            patch.object(lanes_module.device_module, "stream") as stream_ctx,
            patch.object(
                lanes_module.device_module,
                "Event",
                side_effect=[metadata_done, finish],
            ),
            patch.object(lanes_module, "transfer_cache_blocks") as transfer,
        ):
            fence = executor._start_writing(
                [7],
                [(0, 5, 9)],
                lane=lane,
                prerequisite_stream=prerequisite_stream,
            )

        # On the write stream, ordered after the prerequisite stream the
        # caller named (the forwards that wrote the pages). The completion
        # event is handed back so a stream-ordered submission can fence the
        # caller's fence stream on it; a pinned one simply drops it.
        assert (fence) is (finish)
        executor.write_stream.wait_stream.assert_called_once_with(prerequisite_stream)
        # The address tables and the metadata H2D are enqueued on the write
        # stream too: the payload kernel reads them from that stream, and a
        # copy left on the caller's stream would land behind the wait above
        # with nothing ordering it ahead of the kernel.
        stream_ctx.assert_called_once_with(executor.write_stream)
        lane.workspace.load_block_transfers.assert_called_once_with(
            [(0, 5, 9)], geometry=executor._transfer_geometry
        )
        lane.workspace.commit_block_transfers.assert_called_once_with(
            1, device, non_blocking=True
        )
        transfer.assert_called_once_with(
            "d2h",
            executor.layout.buffers,
            executor.host_storage.host_buffer,
            executor._transfer_geometry,
            lane.workspace,
            executor.write_stream,
            num_blocks=1,
            geometry_offset=0,
            num_geometry_rows=3,
            backend="auto",
            grid_cap=None,
            layer_ready_flags=None,
        )
        finish.record.assert_called_once_with(executor.write_stream)
        metadata_done.record.assert_called_once_with(executor.write_stream)
        metadata_done.synchronize.assert_not_called()
        assert (executor._ordered_write_lane.metadata_done) is None

        # Refill must wait for metadata, but must never wait for payload ACK;
        # the upload and its retirement event sit inside the write-stream
        # context, the payload launch names the stream explicitly.
        for ready in (False, True):
            with subtests.test(metadata_ready=ready):
                metadata_done.reset_mock()
                metadata_done.query.return_value = ready
                order = Mock()
                order.attach_mock(metadata_done.synchronize, "retire")
                order.attach_mock(lane.workspace.load_block_transfers, "refill")
                order.attach_mock(lane.workspace.commit_block_transfers, "upload")
                order.attach_mock(metadata_done.record, "record")
                with (
                    no_current_stream,
                    patch.object(lanes_module.device_module, "stream") as stream_ctx,
                    patch.object(
                        lanes_module.device_module, "Event", return_value=finish
                    ),
                    patch.object(lanes_module, "transfer_cache_blocks") as transfer,
                ):
                    order.attach_mock(stream_ctx, "stream")
                    order.attach_mock(transfer, "payload")
                    executor._start_writing(
                        [8],
                        [(0, 6, 10)],
                        lane=lane,
                        prerequisite_stream=prerequisite_stream,
                    )
                names = [call[0] for call in order.mock_calls]
                assert (names) == (
                    ([] if ready else ["retire"])
                    + [
                        "refill",
                        "stream",
                        "stream().__enter__",
                        "upload",
                        "record",
                        "stream().__exit__",
                        "payload",
                    ]
                )
                finish.synchronize.assert_not_called()

    def test_submit_write_backs_fences_only_stream_ordered_ops(self):
        executor_module = self._executor_module()
        lanes_module = _lanes_module()
        executor, _ = self._make_write_executor(executor_module)
        fence_stream = Mock(name="fence_stream")
        prerequisite = object()
        ordered_finish = Mock(name="ordered_finish")
        pinned_finish = Mock(name="pinned_finish")
        events = iter(
            [
                Mock(name="ordered_meta"),
                ordered_finish,
                Mock(name="pinned_meta"),
                pinned_finish,
            ]
        )

        class WriteBackOp:
            def __init__(self):
                self.op_ids = [11, 12, 13]
                self.group_ids = [[0], [0], [0]]
                self.src_pages = [[1], [2], [3]]
                self.dst_pages = [[5], [6], [7]]
                self.source_pinned = [True, False, True]
                self.content_hashes = [["pinned-a"], ["ordered"], ["pinned-b"]]
                self.page_offsets = [[0], [1], [2]]

        with (
            patch.object(
                executor_module.Cache, "WriteBackOp", WriteBackOp, create=True
            ),
            patch.object(
                lanes_module.device_module, "stream", return_value=nullcontext()
            ),
            patch.object(
                lanes_module.device_module,
                "Event",
                side_effect=lambda: next(events),
            ),
            patch.object(lanes_module, "transfer_cache_blocks") as transfer,
        ):
            executor.submit_write_backs(
                [WriteBackOp()],
                prerequisite_stream=prerequisite,
                fence_stream=fence_stream,
            )

        # The stream-ordered op (12) launches first and the fence stream the
        # caller named waits on ITS completion only; the pinned ops (11, 13)
        # follow on the write stream and fence nothing -- the scheduler holds
        # their sources.
        ordered_lane = executor._ordered_write_lane
        pinned_lane = executor._pinned_write_lane
        ordered_lane.workspace.load_block_transfers.assert_called_once_with(
            [(0, 2, 6)], geometry=executor._transfer_geometry
        )
        pinned_lane.workspace.load_block_transfers.assert_called_once_with(
            [(0, 1, 5), (0, 3, 7)], geometry=executor._transfer_geometry
        )
        assert ([call.args[4] for call in transfer.call_args_list]) == (
            [ordered_lane.workspace, pinned_lane.workspace]
        )
        fence_stream.wait_event.assert_called_once_with(ordered_finish)
        pending = executor._completions._pending
        assert [(finish, ack.kind, ack.op_ids) for finish, ack in pending] == [
            (ordered_finish, executor_module._AckKind.WRITE_BACK, [12]),
            (pinned_finish, executor_module._AckKind.WRITE_BACK, [11, 13]),
        ]

        assert (executor.write_stream.wait_stream.call_args_list) == (
            [call(prerequisite), call(prerequisite)]
        ), "both launches order behind the prerequisite stream the caller named"

    def test_submit_write_backs_without_stream_ordered_ops_fences_nothing(self):
        executor_module = self._executor_module()
        lanes_module = _lanes_module()
        executor, _ = self._make_write_executor(executor_module)
        fence_stream = Mock(name="fence_stream")
        prerequisite = object()

        class WriteBackOp:
            def __init__(self):
                self.op_ids = [11]
                self.group_ids = [[0]]
                self.src_pages = [[1]]
                self.dst_pages = [[5]]
                self.source_pinned = [True]
                self.content_hashes = [[""]]
                self.page_offsets = [[0]]

        with (
            patch.object(
                executor_module.Cache, "WriteBackOp", WriteBackOp, create=True
            ),
            patch.object(
                lanes_module.device_module, "stream", return_value=nullcontext()
            ),
            patch.object(lanes_module.device_module, "Event", side_effect=Mock),
            patch.object(lanes_module, "transfer_cache_blocks"),
        ):
            executor.submit_write_backs(
                [WriteBackOp()],
                prerequisite_stream=prerequisite,
                fence_stream=fence_stream,
            )

        fence_stream.wait_event.assert_not_called()
        executor._ordered_write_lane.workspace.load_block_transfers.assert_not_called()

    def test_submit_write_backs_rejects_ragged_guard_vector(self):
        executor_module = self._executor_module()
        executor, _ = self._make_write_executor(executor_module)
        prerequisite = object()

        class WriteBackOp:
            def __init__(self):
                self.op_ids = [11, 12]
                self.group_ids = [[0], [0]]
                self.src_pages = [[1], [2]]
                self.dst_pages = [[5], [6]]
                self.source_pinned = [True]

        with patch.object(
            executor_module.Cache, "WriteBackOp", WriteBackOp, create=True
        ):
            with pytest.raises((ValueError)):
                executor.submit_write_backs(
                    [WriteBackOp()],
                    prerequisite_stream=prerequisite,
                    fence_stream=object(),
                )

    def test_cache_operation_without_transfers_is_refused(self, subtests):
        # An op is acknowledged by its copy's completion event, so one with
        # nothing to copy could never be acknowledged; the scheduler never
        # emits one, and the runtime refuses rather than inventing an ack.
        from tokenspeed.runtime.cache.transfer.ops import HostTier

        _, executor = self._owner_executor()
        for source_is_device in (True, False):
            with subtests.test(source_is_device=source_is_device):
                op_ids: list[int] = []
                transfers: list[tuple[int, int, int]] = []
                with pytest.raises((ValueError), match=("operation 11 carries no")):
                    executor._append_transfers(
                        [7, 11],
                        [[0], []],
                        [[1], []],
                        [[5], []],
                        collected_op_ids=op_ids,
                        transfers=transfers,
                        source_is_device=source_is_device,
                        tier=HostTier.L2,
                    )

    def test_loadback_logs_non_empty_batch(self):
        executor_module, executor, _, geometry, workspace = self._make_load_executor(
            consumers=(("field",),),
            layer_slices=((0, 1),),
            backend="dma",
            device_rows=None,
        )
        workspace.load_block_transfers.return_value = (2, (0, 2))
        load_events = SimpleNamespace(start_event=Mock(), layer_done_events=[None])
        tracker = Mock()
        tracker.begin_load.return_value = 0
        tracker.event_sets = [load_events]
        executor._load_trackers = [(tracker, 1)]
        finish = Mock()
        prerequisite_stream = object()

        with (
            patch.object(executor_module, "get_is_capture_mode", return_value=False),
            patch.object(
                executor_module.device_module, "stream", return_value=nullcontext()
            ),
            patch.object(executor_module.device_module, "Event", return_value=finish),
            patch.object(executor_module, "transfer_cache_blocks") as transfer,
            patch.object(executor_module.logger, "info") as log_info,
        ):
            executor._start_loading(
                [9],
                [(0, 2, 1), (0, 5, 4)],
                prerequisite_stream=prerequisite_stream,
            )

        workspace.load_block_transfers.assert_called_once_with(
            [(0, 2, 1), (0, 5, 4)], geometry=geometry
        )
        workspace.commit_block_transfers.assert_not_called()
        transfer.assert_called_once()
        log_info.assert_called_once_with(
            "[L2] load started: operations=1 blocks=2",
        )
        # The load orders after the prerequisite stream the caller named --
        # the one that zeroed its destinations -- not after the current stream.
        load_events.start_event.record.assert_called_once_with(prerequisite_stream)
        load_events.start_event.wait.assert_called_once_with(executor.load_stream)

    def test_resolved_transport_selects_events_even_with_device_geometry(
        self, subtests
    ):
        for uses_device_tables in (False, True):
            with subtests.test(uses_device_tables=uses_device_tables):
                module, executor, _, _, workspace = self._make_load_executor(
                    consumers=(("first",), ("second",)),
                    layer_slices=((0, 1), (1, 1)),
                    device_rows="bound geometry",
                )
                workspace.prepare_backend.return_value = SimpleNamespace(
                    uses_device_tables=uses_device_tables, layer_ready=False
                )
                events = _LoadEvents(start_event=Mock(), layer_done_events=[None, None])
                tracker = Mock()
                tracker.begin_load.return_value = 0
                tracker.event_sets = [events]
                executor._load_trackers = [(tracker, 2)]
                with (
                    patch.object(module, "get_is_capture_mode", return_value=False),
                    patch.object(
                        module.device_module, "stream", return_value=nullcontext()
                    ),
                    patch.object(
                        module.device_module, "Event", side_effect=[Mock(), Mock()]
                    ),
                    patch.object(module, "transfer_cache_blocks") as transfer,
                ):
                    executor._start_loading(
                        [9], [(0, 1, 1)], prerequisite_stream=object()
                    )
                assert (workspace.commit_block_transfers.call_count) == (
                    int(uses_device_tables)
                )
                workspace.prepare_layer_ready.assert_not_called()
                assert (events.layer_ready_flags) is None
                assert (transfer.call_count) == (2)
                assert all(event is not None for event in events.layer_done_events)

    def test_kernel_init_builds_consumer_ordered_static_geometry_once(self):
        executor_module = self._executor_module()
        lanes_module = _lanes_module()
        HostCacheExecutor = executor_module.HostCacheExecutor

        device = SimpleNamespace(type="cuda")
        buffer = SimpleNamespace(device=device)
        fields = {
            "target.0.k": SimpleNamespace(
                field_id="target.0.k",
                device_buffer_index=0,
                device_block_zero_offset_bytes=8,
                block_stride_bytes=16,
                payload_bytes=12,
            ),
            "target.2.state": SimpleNamespace(
                field_id="target.2.state",
                device_buffer_index=1,
                device_block_zero_offset_bytes=32,
                block_stride_bytes=64,
                payload_bytes=20,
            ),
            "draft.0.k": SimpleNamespace(
                field_id="draft.0.k",
                device_buffer_index=0,
                device_block_zero_offset_bytes=48,
                block_stride_bytes=16,
                payload_bytes=12,
            ),
        }
        combined_layout = SimpleNamespace(
            num_lcm_blocks=11,
            buffers=(buffer, SimpleNamespace(device=device)),
            groups=(
                SimpleNamespace(
                    group_id="state",
                    cache_blocks_per_lcm_block=4,
                    fields=(fields["target.2.state"],),
                ),
                SimpleNamespace(
                    group_id="full",
                    cache_blocks_per_lcm_block=8,
                    fields=(fields["target.0.k"], fields["draft.0.k"]),
                ),
            ),
            # Target layers precede draft layers, and empty layers are retained.
            consumers=(
                ("target.0.k",),
                (),
                ("target.2.state",),
                ("draft.0.k",),
            ),
        )
        target_layout = SimpleNamespace(
            consumers=(("target.0.k",), (), ("target.2.state",))
        )
        draft_layout = SimpleNamespace(consumers=(("draft.0.k",),))
        target_pool = Mock()
        target_pool.cache_transfer_layout.return_value = target_layout
        target_pool.arena.runtime_contract = _replicated_contract(
            combined_layout, num_lcm_blocks=11
        )
        draft_pool = Mock()
        draft_pool.cache_transfer_layout.return_value = draft_layout
        storage = SimpleNamespace(
            host_cache_block_bytes=(20, 24),
            host_field_offsets=((0,), (0, 12)),
            host_lcm_block_bytes=192,
            num_host_lcm_blocks=3,
            host_buffer="host",
        )
        unbound_geometry = Mock()
        bound_geometry = object()
        unbound_geometry.bind.return_value = bound_geometry
        trackers = []

        def make_tracker(consumer_count):
            tracker = Mock()
            tracker.event_sets = [object(), object()]
            trackers.append((consumer_count, tracker))
            return tracker

        with (
            patch.object(
                executor_module,
                "combine_cache_transfer_layouts",
                return_value=combined_layout,
            ),
            patch.object(
                executor_module,
                "compute_host_lcm_block_bytes",
                return_value=storage.host_lcm_block_bytes,
            ),
            patch.object(executor_module, "HostCacheStorage", return_value=storage),
            patch.object(
                lanes_module.psutil,
                "virtual_memory",
                return_value=SimpleNamespace(available=10**12),
            ),
            patch.object(
                executor_module, "LayerwiseLoadTracker", side_effect=make_tracker
            ),
            patch.object(
                executor_module,
                "new_cache_stream",
                side_effect=lambda _priority: Mock(),
            ),
            patch.object(
                executor_module.device_module, "stream", return_value=nullcontext()
            ),
            patch.object(executor_module, "HostTransferWorkspace", side_effect=Mock),
            patch.object(lanes_module, "HostTransferWorkspace", side_effect=Mock),
            patch.object(
                lanes_module,
                "build_host_transfer_geometry",
                return_value=unbound_geometry,
            ) as build_geometry,
        ):
            executor = HostCacheExecutor(
                target_pool,
                draft_pool=draft_pool,
                l2_tier=True,
                host_ratio=1.0,
                host_size_gb=0,
                snapshot_pool=NO_POOL,
                slot_state_exporters=None,
                io_backend="kernel",
                attn_tp_rank=0,
                kvp_rank=0,
                nprocs_per_node=1,
                world_cpu_group=None,
            )

        build_geometry.assert_called_once_with(
            rows=(
                (1, 0, 8, 16, 24, 0, 8, 12),
                (0, 1, 32, 64, 20, 0, 4, 20),
                (1, 0, 48, 16, 24, 12, 8, 12),
            ),
            layer_slices=((0, 1), (1, 0), (1, 1), (2, 1)),
            group_packing=(4, 8),
            host_lcm_block_bytes=192,
            num_host_lcm_blocks=3,
            num_device_lcm_blocks=11,
            num_device_buffers=2,
        )
        unbound_geometry.bind.assert_called_once_with(device, non_blocking=False)
        assert (executor._transfer_geometry) is (bound_geometry)
        assert ([count for count, _ in trackers]) == ([3, 1])

    def test_direct_and_npu_init_keep_geometry_on_the_host(self, subtests):
        executor_module = self._executor_module()
        lanes_module = _lanes_module()
        HostCacheExecutor = executor_module.HostCacheExecutor

        pool = Mock()
        pool.arena.runtime_contract = SimpleNamespace(
            group_specs=(SimpleNamespace(group_id="group", shard_count=1),),
            virtual_block_counts={"group": 3},
        )
        storage = SimpleNamespace(
            host_cache_block_bytes=(16,),
            host_field_offsets=((0,),),
            host_lcm_block_bytes=64,
            num_host_lcm_blocks=2,
            host_buffer="host",
        )
        tracker = Mock()
        tracker.event_sets = [object()]

        with (
            patch.object(
                executor_module,
                "compute_host_lcm_block_bytes",
                return_value=storage.host_lcm_block_bytes,
            ),
            patch.object(executor_module, "HostCacheStorage", return_value=storage),
            patch.object(
                lanes_module.psutil,
                "virtual_memory",
                return_value=SimpleNamespace(available=10**12),
            ),
            patch.object(executor_module, "LayerwiseLoadTracker", return_value=tracker),
            patch.object(
                executor_module,
                "new_cache_stream",
                side_effect=lambda _priority: Mock(),
            ),
            patch.object(
                executor_module.device_module, "stream", return_value=nullcontext()
            ),
            patch.object(executor_module, "HostTransferWorkspace", side_effect=Mock),
            patch.object(lanes_module, "HostTransferWorkspace", side_effect=Mock),
            patch.object(
                lanes_module,
                "build_host_transfer_geometry",
                side_effect=lambda **_kwargs: SimpleNamespace(
                    device_rows=None,
                    bind=Mock(),
                ),
            ) as build_geometry,
        ):
            for io_backend, device_type in (("direct", "cuda"), ("kernel", "npu")):
                with subtests.test(io_backend=io_backend, device_type=device_type):
                    field = SimpleNamespace(
                        field_id="field",
                        device_buffer_index=0,
                        device_block_zero_offset_bytes=0,
                        block_stride_bytes=16,
                        payload_bytes=16,
                    )
                    layout = SimpleNamespace(
                        num_lcm_blocks=2,
                        buffers=(
                            SimpleNamespace(device=SimpleNamespace(type=device_type)),
                        ),
                        groups=(
                            SimpleNamespace(
                                group_id="group",
                                cache_blocks_per_lcm_block=1,
                                fields=(field,),
                            ),
                        ),
                        consumers=(("field",),),
                    )
                    pool.cache_transfer_layout.return_value = layout
                    executor = HostCacheExecutor(
                        pool,
                        l2_tier=True,
                        host_ratio=1.0,
                        host_size_gb=0,
                        snapshot_pool=NO_POOL,
                        slot_state_exporters=None,
                        io_backend=io_backend,
                        attn_tp_rank=0,
                        kvp_rank=0,
                        nprocs_per_node=1,
                        world_cpu_group=None,
                    )
                    assert (executor._transfer_geometry.device_rows) is None
                    executor._transfer_geometry.bind.assert_not_called()

        assert (build_geometry.call_count) == (2)

    def test_optional_dependency_shim_restores_existing_modules(self):
        protected_names = (
            "tokenspeed_kernel.ops.kvcache.host_transfer",
            "tokenspeed_scheduler",
            "tokenspeed.runtime.cache.l2.layerwise_load",
            "tokenspeed.runtime.cache.host_storage",
            "tokenspeed.runtime.cache.transfer.lanes",
            "tokenspeed.runtime.cache.transfer.layout",
            "tokenspeed.runtime.cache.transfer.ownership",
            "tokenspeed.runtime.execution.forward_step",
            "tokenspeed.runtime.utils",
            "tokenspeed.runtime.cache.l2.executor",
        )
        sentinels = {name: ModuleType(name) for name in protected_names}

        with patch.dict(sys.modules, sentinels, clear=False):
            isolated = _load_executor_module_without_triton(force_isolated=True)

            assert (isolated) is not (sentinels[protected_names[-1]])
            for name, sentinel in sentinels.items():
                assert (sys.modules[name]) is (sentinel)

    def test_two_isolated_loads_preserve_imported_real_modules(self):
        # psutil is a real dependency of the shared lane module; the isolated
        # loader imports it ahead of its sys.modules snapshot so the snapshot's
        # restore cannot drop it again.
        first = _load_executor_module_without_triton(force_isolated=True)
        first_psutil = sys.modules["psutil"]
        assert (_lanes_module().psutil) is (first_psutil)

        second = _load_executor_module_without_triton(force_isolated=True)

        assert (first) is not (second)
        assert (sys.modules["psutil"]) is (first_psutil)
        assert (_lanes_module().psutil) is (first_psutil)

    def test_loadback_commits_block_ids_once_and_launches_one_flagged_kernel(self):
        executor_module, executor, device, geometry, workspace = (
            self._make_load_executor(
                consumers=(("layer.0",), (), ("layer.2",)),
                layer_slices=((0, 2), (2, 0), (2, 1)),
                device_rows=object(),
            )
        )
        flags = Mock()
        flags.__getitem__ = Mock(return_value=flags)
        workspace.prepare_layer_ready.return_value = flags
        load_events = _LoadEvents(
            start_event=Mock(),
            layer_done_events=[None, None, None],
            layer_ready_flags=None,
            wait_layer_ready=None,
            layer_ready_init_event=Mock(),
        )
        tracker = Mock()
        tracker.begin_load.return_value = 0
        tracker.event_sets = [load_events]
        executor._load_trackers = [(tracker, 3)]
        finish = Mock()

        with (
            patch.object(executor_module, "get_is_capture_mode", return_value=False),
            patch.object(
                executor_module.device_module,
                "stream",
                return_value=nullcontext(),
            ),
            patch.object(executor_module.device_module, "Event", return_value=finish),
            patch.object(executor_module, "transfer_cache_blocks") as transfer,
        ):
            executor._start_loading([9], [(0, 2, 1)], prerequisite_stream=object())

        workspace.load_block_transfers.assert_called_once_with(
            [(0, 2, 1)], geometry=geometry
        )
        workspace.commit_block_transfers.assert_called_once_with(
            1, device, non_blocking=True
        )
        workspace.prepare_layer_ready.assert_called_once_with(3, device)
        load_events.layer_ready_init_event.record.assert_called_once_with(
            executor.load_stream
        )
        transfer.assert_called_once_with(
            "h2d",
            executor.layout.buffers,
            executor.host_storage.host_buffer,
            geometry,
            workspace,
            executor.load_stream,
            num_blocks=1,
            geometry_offset=0,
            num_geometry_rows=3,
            backend="auto",
            layer_ready_flags=flags,
            grid_cap=None,
        )
        finish.record.assert_called_once_with(executor.load_stream)
        assert (load_events.layer_done_events) == ([finish, finish, finish])
        assert (load_events.layer_ready_flags) is (flags)
        assert (load_events.wait_layer_ready) is (executor_module.wait_layer_ready)
        ((pending_finish, ack),) = executor._completions._pending
        assert (pending_finish) is (finish)
        assert ((ack.kind, ack.op_ids)) == ((executor_module._AckKind.LOAD_BACK, [9]))

    def test_loadback_launch_failure_retires_all_target_and_draft_events(self):
        executor_module, executor, _, _, _ = self._make_load_executor(
            consumers=(("target.0",), ("target.1",), ("draft.0",)),
            layer_slices=((0, 1), (1, 1), (2, 1)),
            device_rows=object(),
            load_stream=Mock(),
        )
        target_events = _LoadEvents(
            start_event=Mock(),
            layer_done_events=[Mock(), Mock()],
            layer_ready_init_event=Mock(),
        )
        draft_events = _LoadEvents(
            start_event=Mock(),
            layer_done_events=[Mock()],
            layer_ready_init_event=Mock(),
        )
        target_tracker = Mock()
        target_tracker.begin_load.return_value = 0
        target_tracker.event_sets = [target_events]
        draft_tracker = Mock()
        draft_tracker.begin_load.return_value = 0
        draft_tracker.event_sets = [draft_events]
        executor._load_trackers = [(target_tracker, 2), (draft_tracker, 1)]
        retirement = Mock()

        with (
            patch.object(executor_module, "get_is_capture_mode", return_value=False),
            patch.object(
                executor_module.device_module,
                "stream",
                return_value=nullcontext(),
            ),
            patch.object(
                executor_module.device_module,
                "Event",
                return_value=retirement,
            ),
            patch.object(
                executor_module,
                "transfer_cache_blocks",
                side_effect=RuntimeError("layer launch failed"),
            ) as transfer,
        ):
            with pytest.raises((RuntimeError), match=("layer launch failed")):
                executor._start_loading([9], [(0, 2, 1)], prerequisite_stream=object())

        assert (transfer.call_count) == (1)
        retirement.record.assert_called_once_with(executor.load_stream)
        assert (target_events.layer_done_events) == ([retirement, retirement])
        assert (draft_events.layer_done_events) == ([retirement])
        executor.load_stream.synchronize.assert_not_called()
        assert (executor._completions._pending) == ([])

    def test_failed_retirement_sync_poisons_executor_and_preserves_original_error(self):
        executor_module, executor, _, _, _ = self._make_load_executor(
            consumers=(("target.0",),),
            layer_slices=((0, 1),),
            device_rows=object(),
            load_stream=Mock(),
        )
        load_events = _LoadEvents(
            start_event=Mock(),
            layer_done_events=[Mock()],
            layer_ready_init_event=Mock(),
        )
        tracker = Mock()
        tracker.begin_load.return_value = 0
        tracker.event_sets = [load_events]
        executor._load_trackers = [(tracker, 1)]
        original_error = RuntimeError("original layer launch failed")
        retirement = Mock()
        retirement.record.side_effect = RuntimeError("retirement record failed")
        executor.load_stream.synchronize.side_effect = RuntimeError(
            "retirement sync failed"
        )

        with (
            patch.object(executor_module, "get_is_capture_mode", return_value=False),
            patch.object(
                executor_module.device_module,
                "stream",
                return_value=nullcontext(),
            ),
            patch.object(
                executor_module.device_module,
                "Event",
                return_value=retirement,
            ),
            patch.object(
                executor_module,
                "transfer_cache_blocks",
                side_effect=original_error,
            ),
        ):
            with pytest.raises((RuntimeError)) as raised:
                executor._start_loading([9], [(0, 2, 1)], prerequisite_stream=object())

            assert (raised.value) is (original_error)
            assert (str(raised.value)) == ("original layer launch failed")
            assert executor._load_poisoned
            notes = getattr(raised.value, "__notes__", ())
            assert any("retirement record failed" in note for note in notes)
            assert any("retirement sync failed" in note for note in notes)

            executor.load_stream.synchronize.side_effect = None
            with pytest.raises((RuntimeError), match=("poisoned")):
                executor._start_loading([10], [(0, 3, 2)], prerequisite_stream=object())

        tracker.begin_load.assert_called_once_with()


class TestHostMemoryShareTest:
    """The ranks of a node pin their Host tiers at once: each sizes them
    against an equal share of the Host memory the node has available, every
    buffer at the power of two its pinned allocation rounds up to."""

    GB = 10**9
    BLOCK_BYTES = 10**6  # one Host LCM block
    # --kvstore-ratio 2 over a ~139 GB Device KV: 277.95 GB of L2 per rank.
    LARGE_POOL_BLOCKS = 138_975

    def setup_method(self):
        try:
            self.executor_module = import_module("tokenspeed.runtime.cache.l2.executor")
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs runtime dependencies: {exc}")
        self.headroom = self.executor_module._HOST_MEM_HEADROOM_BYTES

    def _fitting_blocks(self, available_gb, nprocs_per_node):
        """The L2 blocks in the largest power of two of a rank's share."""
        share = int(available_gb * self.GB) // nprocs_per_node - self.headroom
        return (1 << (share.bit_length() - 1)) // self.BLOCK_BYTES

    def _build(
        self,
        *,
        available_gb,
        nprocs_per_node,
        device_lcm_blocks,
        host_size_gb=0,
        world_cpu_group=None,
    ):
        """An L2-only executor at --kvstore-ratio 2 and its mocked logger."""
        executor_module = self.executor_module
        lanes_module = _lanes_module()
        field = SimpleNamespace(
            field_id="field",
            device_buffer_index=0,
            device_block_zero_offset_bytes=0,
            block_stride_bytes=16,
            payload_bytes=16,
        )
        pool = Mock()
        pool.cache_transfer_layout.return_value = SimpleNamespace(
            num_lcm_blocks=device_lcm_blocks,
            buffers=(SimpleNamespace(device=SimpleNamespace(type="cuda")),),
            groups=(
                SimpleNamespace(
                    group_id="group", cache_blocks_per_lcm_block=1, fields=(field,)
                ),
            ),
            consumers=(("field",),),
        )
        pool.arena.runtime_contract = SimpleNamespace(
            group_specs=(SimpleNamespace(group_id="group", shard_count=1),),
            virtual_block_counts={"group": 1 + device_lcm_blocks},
        )
        storage = SimpleNamespace(
            host_cache_block_bytes=(self.BLOCK_BYTES,),
            host_field_offsets=((0,),),
            host_lcm_block_bytes=self.BLOCK_BYTES,
            num_host_lcm_blocks=0,
            host_buffer="host",
        )
        tracker = Mock()
        tracker.event_sets = [object()]
        logger = Mock()
        available = SimpleNamespace(available=int(available_gb * self.GB))
        with (
            patch.object(
                executor_module,
                "compute_host_lcm_block_bytes",
                return_value=self.BLOCK_BYTES,
            ),
            patch.object(executor_module, "HostCacheStorage", return_value=storage),
            patch.object(lanes_module.psutil, "virtual_memory", return_value=available),
            patch.object(executor_module, "LayerwiseLoadTracker", return_value=tracker),
            patch.object(
                executor_module,
                "new_cache_stream",
                side_effect=lambda _priority: Mock(),
            ),
            patch.object(
                executor_module.device_module, "stream", return_value=nullcontext()
            ),
            patch.object(executor_module, "HostTransferWorkspace", side_effect=Mock),
            patch.object(lanes_module, "HostTransferWorkspace", side_effect=Mock),
            patch.object(
                lanes_module,
                "build_host_transfer_geometry",
                return_value=SimpleNamespace(device_rows=None, bind=Mock()),
            ),
            patch.object(executor_module, "logger", logger),
        ):
            executor = executor_module.HostCacheExecutor(
                pool,
                l2_tier=True,
                host_ratio=2.0,
                host_size_gb=host_size_gb,
                snapshot_pool=NO_POOL,
                slot_state_exporters=None,
                io_backend="direct",
                attn_tp_rank=0,
                kvp_rank=0,
                nprocs_per_node=nprocs_per_node,
                world_cpu_group=world_cpu_group,
            )
        return executor, logger

    def test_four_ranks_cap_an_over_committed_ratio_l2_to_the_node(self):
        # 4 x 277.95 GB of L2 does not fit a node with 956 GB available, and
        # neither does an L2 of the whole 228.26 GB share: it pins 274.88 GB.
        executor, logger = self._build(
            available_gb=956,
            nprocs_per_node=4,
            device_lcm_blocks=self.LARGE_POOL_BLOCKS,
        )
        l2_blocks = executor.num_host_pages - 1
        assert (l2_blocks) == (self._fitting_blocks(956, 4))
        assert (l2_blocks) == (2**37 // self.BLOCK_BYTES)
        pinned = _lanes_module().pinned_host_bytes(l2_blocks * self.BLOCK_BYTES)
        assert (4 * (pinned + self.headroom)) <= (956 * self.GB)
        logger.warning.assert_called_once()
        message = logger.warning.call_args.args[0]
        assert ("from 277.95 GB to 137.44 GB") in (message)
        assert ("--kvstore-ratio 2.0") in (message)

    def test_an_explicit_size_that_does_not_fit_the_node_is_refused(self, subtests):
        # 278 GB and 200 GB pin 549.76 GB and 274.88 GB: neither fits the
        # 228.26 GB share of 4 ranks on 956 GB.
        for host_size_gb in (278, 200):
            with (
                subtests.test(host_size_gb=host_size_gb),
                pytest.raises((ValueError), match=("Not enough Host memory")),
            ):
                self._build(
                    available_gb=956,
                    nprocs_per_node=4,
                    device_lcm_blocks=self.LARGE_POOL_BLOCKS,
                    host_size_gb=host_size_gb,
                )
        # One whose pinned size fits the node's share is kept as asked.
        executor, logger = self._build(
            available_gb=956,
            nprocs_per_node=4,
            device_lcm_blocks=self.LARGE_POOL_BLOCKS,
            host_size_gb=128,
        )
        assert (executor.num_host_pages - 1) == (128_000)
        logger.warning.assert_not_called()

    def test_a_single_rank_and_a_small_pool_keep_the_ratio_size(self, subtests):
        for nprocs_per_node, device_lcm_blocks in (
            (1, self.LARGE_POOL_BLOCKS),  # 277.95 GB pins 549.76 GB of 956
            (4, 50_760),  # 4 x 101.52 GB pin 4 x 137.44 GB
        ):
            with subtests.test(nprocs_per_node=nprocs_per_node):
                executor, logger = self._build(
                    available_gb=956,
                    nprocs_per_node=nprocs_per_node,
                    device_lcm_blocks=device_lcm_blocks,
                )
                assert (executor.num_host_pages - 1) == (2 * device_lcm_blocks)
                logger.warning.assert_not_called()

    def test_no_room_for_one_block_is_refused_not_capped_to_nothing(self):
        with pytest.raises((ValueError), match=("Not enough Host memory")):
            self._build(
                available_gb=4 * self.headroom / self.GB,
                nprocs_per_node=4,
                device_lcm_blocks=self.LARGE_POOL_BLOCKS,
            )

    def test_every_rank_takes_the_l2_size_the_smallest_share_holds(self):
        # A peer node has 500 GB available: every rank takes what it holds,
        # and the agreement runs before any rank pins, explicit size or not.
        torch = import_module("torch")
        group = object()
        peer_blocks = self._fitting_blocks(500, 4)
        assert (peer_blocks) < (self._fitting_blocks(956, 4))

        def all_reduce(tensor, *, op, group):
            assert (op) is (torch.distributed.ReduceOp.MIN)
            tensor.fill_(min(int(tensor.item()), peer_blocks))

        with patch.object(
            torch.distributed, "all_reduce", side_effect=all_reduce
        ) as reduce:
            executor, _ = self._build(
                available_gb=956,
                nprocs_per_node=4,
                device_lcm_blocks=self.LARGE_POOL_BLOCKS,
                world_cpu_group=group,
            )
            assert (executor.num_host_pages - 1) == (peer_blocks)
            explicit, _ = self._build(
                available_gb=956,
                nprocs_per_node=4,
                device_lcm_blocks=self.LARGE_POOL_BLOCKS,
                host_size_gb=128,
                world_cpu_group=group,
            )
            assert (explicit.num_host_pages - 1) == (128_000)
        assert (reduce.call_count) == (2)
        assert all(c.kwargs["group"] is group for c in reduce.call_args_list)

    def test_a_pinned_buffer_takes_the_power_of_two_it_rounds_up_to(self):
        lanes_module = _lanes_module()
        pinned = lanes_module.pinned_host_bytes
        fit = lanes_module.largest_pinned_fit
        assert (pinned(0)) == (0)
        assert (pinned(1)) == (1)
        assert (pinned(2**37)) == (2**37)
        assert (pinned(2**37 + 1)) == (2**38)
        assert (pinned(228_260 * self.BLOCK_BYTES)) == (2**38)
        assert (fit(-1)) == (0)
        assert (fit(0)) == (0)
        assert (fit(2**38 - 1)) == (2**37)
        assert (fit(2**38)) == (2**38)
        for room in (1, 3, 2**37 - 1, 228_260 * self.BLOCK_BYTES):
            assert (pinned(fit(room))) <= (room)
            assert (pinned(fit(room) + 1)) > (room)


class TestL3FlatKvExecutorTest:

    def test_load_with_every_row_owned_elsewhere_acks_without_h2d(self):
        try:
            from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
            from tokenspeed.runtime.cache.transfer.lanes import CompletionQueue
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs runtime dependencies: {exc}")

        # A KVP rank that owns none of a load's rows copies nothing, arms no
        # layer fence and still acknowledges the op (the hooks' replica
        # intersection waits for the owners).
        executor = HostCacheExecutor.__new__(HostCacheExecutor)
        executor._ack_lock = threading.Lock()
        executor._ready_load_acks = []
        executor._load_poisoned = False
        executor._completions = CompletionQueue()
        executor._load_trackers = [(Mock(), 1)]
        assert executor._start_loading([9], [], prerequisite_stream=object()) is None
        executor._load_trackers[0][0].begin_load.assert_not_called()
        events = executor.poll_results()
        assert ([(type(e).__name__, int(e.op_id)) for e in events]) == (
            [("LoadBackDoneEvent", 9)]
        )


class TestCompactLayoutRoundTripTest:
    @pytest.fixture(autouse=True)
    def _executor_cleanup(self):
        with ExitStack() as self.executor_cleanup:
            yield

    def setup_method(self):
        try:
            import torch

            import tokenspeed.runtime.cache.l2.executor as executor_module
            from tokenspeed.runtime.cache.transfer.layout import (
                CacheField,
                CacheGroupLayout,
                CacheTransferLayout,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs runtime dependencies: {exc}")
        if not torch.cuda.is_available():
            pytest.skip("needs a CUDA device")
        self.torch = torch
        self.executor_module = executor_module
        self.CacheField = CacheField
        self.CacheGroupLayout = CacheGroupLayout
        self.CacheTransferLayout = CacheTransferLayout

    def _make_executor(self, layout, *, draft_layout=None, io_backend):
        pool = _SyntheticPool(layout)
        draft_pool = (
            _SyntheticPool(draft_layout, pool.arena)
            if draft_layout is not None
            else None
        )
        with patch.object(self.executor_module, "_HOST_MEM_HEADROOM_BYTES", 0):
            executor = self.executor_module.HostCacheExecutor(
                pool,
                draft_pool=draft_pool,
                l2_tier=True,
                host_ratio=1.0,
                host_size_gb=0,
                snapshot_pool=NO_POOL,
                slot_state_exporters=None,
                io_backend=io_backend,
                attn_tp_rank=0,
                kvp_rank=0,
                nprocs_per_node=1,
                world_cpu_group=None,
            )
        self.executor_cleanup.callback(executor.shutdown)
        return executor, pool, draft_pool

    def _single_group_layout(self, buffer, *fields):
        return self.CacheTransferLayout(
            4,
            (self.CacheGroupLayout("full", 1, fields),),
            (buffer,),
            tuple((field.field_id,) for field in fields),
        )

    def test_kernel_executor_round_trip_restores_compact_layout_byte_exactly(self):
        torch = self.torch
        first = torch.full((128,), 0xCC, dtype=torch.uint8, device="cuda")
        second = torch.full((128,), 0xCC, dtype=torch.uint8, device="cuda")
        layout = self.CacheTransferLayout(
            num_lcm_blocks=4,
            groups=(
                self.CacheGroupLayout(
                    group_id="full",
                    cache_blocks_per_lcm_block=2,
                    fields=(
                        self.CacheField("layer.0.k", 0, 8, 8, 4),
                        self.CacheField("layer.0.v", 1, 16, 12, 6),
                    ),
                ),
                self.CacheGroupLayout(
                    group_id="state",
                    cache_blocks_per_lcm_block=1,
                    fields=(self.CacheField("layer.1.state", 0, 64, 10, 5),),
                ),
            ),
            buffers=(first, second),
            consumers=(("layer.0.k", "layer.0.v"), ("layer.1.state",)),
        )

        executor, pool, _ = self._make_executor(layout, io_backend="kernel")

        # Hand-derived Device ranges for blocks (full: 1, 4; state: 3).
        full_k_one = torch.tensor([0x11, 0x12, 0x13, 0x14], dtype=torch.uint8)
        full_v_one = torch.tensor(
            [0x21, 0x22, 0x23, 0x24, 0x25, 0x26], dtype=torch.uint8
        )
        full_k_four = torch.tensor([0x41, 0x42, 0x43, 0x44], dtype=torch.uint8)
        full_v_four = torch.tensor(
            [0x51, 0x52, 0x53, 0x54, 0x55, 0x56], dtype=torch.uint8
        )
        state_three = torch.tensor([0x71, 0x72, 0x73, 0x74, 0x75], dtype=torch.uint8)
        first[16:20].copy_(full_k_one)
        second[28:34].copy_(full_v_one)
        first[40:44].copy_(full_k_four)
        second[64:70].copy_(full_v_four)
        first[94:99].copy_(state_three)
        torch.cuda.synchronize()

        executor._start_writing(  # pylint: disable=protected-access
            [7],
            [(0, 1, 1), (0, 4, 4), (1, 3, 3)],
            lane=executor._pinned_write_lane,  # pylint: disable=protected-access
            prerequisite_stream=torch.cuda.current_stream(),
        )
        executor.write_stream.synchronize()
        write_results = executor.poll_results()
        assert ([int(event.op_id) for event in write_results]) == ([7])

        # Destroy every Device byte so stale cache contents cannot make the
        # H2D half of the round trip pass accidentally.
        first.fill_(0xEE)
        second.fill_(0xEE)
        torch.cuda.synchronize()

        load_index = executor._start_loading(  # pylint: disable=protected-access
            [9],
            [(0, 2, 1), (0, 5, 4), (1, 4, 3)],
            prerequisite_stream=torch.cuda.current_stream(),
        )
        assert (load_index) is not None
        pool.load_tracker.set_consumers(load_index)
        pool.load_tracker.wait_for_layer(0)
        pool.load_tracker.wait_for_layer(1)
        torch.cuda.synchronize()
        load_results = executor.poll_results()
        assert ([int(event.op_id) for event in load_results]) == ([9])
        # Hand-derived destination ranges for blocks (full: 2, 5; state: 4).
        expected_first = torch.full((128,), 0xEE, dtype=torch.uint8)
        expected_second = torch.full((128,), 0xEE, dtype=torch.uint8)
        expected_first[24:28].copy_(full_k_one)
        expected_second[40:46].copy_(full_v_one)
        expected_first[48:52].copy_(full_k_four)
        expected_second[76:82].copy_(full_v_four)
        expected_first[104:109].copy_(state_three)
        assert torch.equal(first.cpu(), expected_first)
        assert torch.equal(second.cpu(), expected_second)

    def test_async_write_metadata_reuse_keeps_batches_distinct(self):
        torch = self.torch
        device = torch.zeros((128,), dtype=torch.uint8, device="cuda")
        layout = self._single_group_layout(
            device, self.CacheField("layer.0.k", 0, 8, 8, 4)
        )
        executor, pool, _ = self._make_executor(layout, io_backend="kernel")
        for generation in range(3):
            # Three back-to-back submissions, each with its own Device source
            # and Host block, with no caller synchronization between them: if
            # a later batch's block table overwrote an earlier one's before its
            # payload ran, the wrong pair would be copied. The sources are
            # distinct on purpose -- a pinned store's source is never rewritten
            # while its copy is in flight, so the test must not rewrite one
            # either.
            for block in range(1, 4):
                offset = 8 + block * 8
                device[offset : offset + 4].fill_(generation * 16 + block)
                executor._start_writing(
                    [block],
                    [(0, block, block)],
                    lane=executor._pinned_write_lane,
                    prerequisite_stream=torch.cuda.current_stream(),
                )
            # The load below has no scheduler ACK to wait for, so order it
            # behind the copies the way a stream-ordered submission would.
            torch.cuda.current_stream().wait_stream(executor.write_stream)
            device.fill_(0xEE)
            load_index = executor._start_loading(
                [9],
                [(0, block, block) for block in range(1, 4)],
                prerequisite_stream=torch.cuda.current_stream(),
            )
            pool.load_tracker.set_consumers(load_index)
            pool.load_tracker.wait_for_layer(0)
            torch.cuda.current_stream().synchronize()
            for block in range(1, 4):
                offset = 8 + block * 8
                assert (device[offset : offset + 4].tolist()) == (
                    [generation * 16 + block] * 4
                )
            assert (sorted(int(event.op_id) for event in executor.poll_results())) == (
                [1, 2, 3, 9]
            )

    def test_real_transfer_restores_merged_owner_draft_subset_once(self):
        torch = self.torch
        device = torch.full((128,), 0xCC, dtype=torch.uint8, device="cuda")
        target_fields = (
            self.CacheField("layer.0.k", 0, 8, 8, 4),
            self.CacheField("layer.1.k", 0, 48, 8, 4),
        )
        target_layout = self._single_group_layout(device, *target_fields)
        draft_layout = self._single_group_layout(device, target_fields[1])
        executor, target_pool, draft_pool = self._make_executor(
            target_layout,
            draft_layout=draft_layout,
            io_backend="kernel",
        )

        device[16:20].fill_(0x11)
        device[56:60].fill_(0x12)
        torch.cuda.synchronize()
        executor._start_writing(  # pylint: disable=protected-access
            [7],
            [(0, 1, 1)],
            lane=executor._pinned_write_lane,
            prerequisite_stream=torch.cuda.current_stream(),
        )
        torch.cuda.synchronize()
        assert ([int(event.op_id) for event in executor.poll_results()]) == ([7])

        device.fill_(0xEE)
        torch.cuda.synchronize()
        load_index = executor._start_loading(  # pylint: disable=protected-access
            [9],
            [(0, 2, 1)],
            prerequisite_stream=torch.cuda.current_stream(),
        )
        assert (load_index) is not None
        target_pool.load_tracker.set_consumers(load_index)
        draft_pool.load_tracker.set_consumers(load_index)
        target_pool.load_tracker.wait_for_layer(0)
        target_pool.load_tracker.wait_for_layer(1)
        draft_pool.load_tracker.wait_for_layer(0)
        torch.cuda.synchronize()
        assert ([int(event.op_id) for event in executor.poll_results()]) == ([9])
        assert (device[24:28].tolist()) == ([0x11] * 4)
        assert (device[64:68].tolist()) == ([0x12] * 4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
