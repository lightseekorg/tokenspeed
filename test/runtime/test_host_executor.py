"""Compact Host cache executor tests."""

from __future__ import annotations

import os
import sys
import threading
import unittest
from contextlib import nullcontext
from importlib import import_module, util
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")


class _SyntheticPool:
    def __init__(self, layout, arena=None):
        self._layout = layout
        if arena is None:
            arena = SimpleNamespace(
                cache_group_specs=tuple(
                    SimpleNamespace(group_id=group.group_id) for group in layout.groups
                )
            )
        self.arena = arena

    def cache_transfer_layout(self):
        return self._layout

    def register_layerwise_load_tracker(self, tracker):
        self.load_tracker = tracker


def _load_executor_module_without_triton(*, force_isolated=False):
    """Load executor orchestration when optional Triton is not installed."""

    if not force_isolated:
        executor_name = "tokenspeed.runtime.cache.l2.executor"
        if executor_name in sys.modules or util.find_spec("tokenspeed_triton"):
            return import_module("tokenspeed.runtime.cache.l2.executor")
    host_transfer = ModuleType("tokenspeed_kernel.ops.kvcache.host_transfer")
    host_transfer.HostTransferWorkspace = Mock
    host_transfer.build_host_transfer_geometry = Mock()
    host_transfer.layer_ready_ptx_supported = Mock(return_value=True)
    host_transfer.transfer_cache_blocks = Mock()
    host_transfer.wait_layer_ready = Mock()
    scheduler = ModuleType("tokenspeed_scheduler")

    class Cache:
        class WriteBackOp:
            pass

        class LoadBackOp:
            pass

        class WriteBackDoneEvent:
            pass

        class LoadBackDoneEvent:
            pass

    scheduler.Cache = Cache
    layerwise_load = ModuleType("tokenspeed.runtime.cache.l2.layerwise_load")
    layerwise_load.LayerwiseLoadTracker = Mock
    storage = ModuleType("tokenspeed.runtime.cache.l2.storage")
    storage.HostCacheStorage = Mock
    storage.compute_host_lcm_block_bytes = Mock(return_value=1)
    layout = ModuleType("tokenspeed.runtime.cache.transfer.layout")
    layout.combine_cache_transfer_layouts = lambda target, draft, group_ids=None: (
        target if draft is None else draft
    )
    graph_wrapper = ModuleType("tokenspeed.runtime.execution.cuda_graph_wrapper")
    graph_wrapper.get_is_capture_mode = Mock(return_value=False)
    runtime_utils = ModuleType("tokenspeed.runtime.utils")
    runtime_utils.get_colorful_logger = Mock(return_value=Mock())
    runtime_utils.get_device_module = Mock(return_value=Mock())
    fake_modules = {
        "tokenspeed_kernel.ops.kvcache.host_transfer": host_transfer,
        "tokenspeed_scheduler": scheduler,
        "tokenspeed.runtime.cache.l2.layerwise_load": layerwise_load,
        "tokenspeed.runtime.cache.l2.storage": storage,
        "tokenspeed.runtime.cache.transfer.layout": layout,
        "tokenspeed.runtime.execution.cuda_graph_wrapper": graph_wrapper,
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


class CacheEventPayloadTest(unittest.TestCase):
    def setUp(self):
        try:
            from tokenspeed_scheduler import Cache

            from tokenspeed.runtime.engine.scheduler_utils import (
                cache_event_from_payload,
                cache_event_to_payload,
                pop_common_cache_event_payloads,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")
        self.Cache = Cache
        self.from_payload = cache_event_from_payload
        self.to_payload = cache_event_to_payload
        self.pop_common = pop_common_cache_event_payloads

    def test_cache_completion_payload_round_trip_has_no_failure_channel(self):
        for event_type in (
            self.Cache.WriteBackDoneEvent,
            self.Cache.LoadBackDoneEvent,
        ):
            with self.subTest(event_type=event_type.__name__):
                event = event_type()
                event.op_id = 7

                payload = self.to_payload(event)

                self.assertEqual(
                    payload,
                    {
                        "kind": event_type.__name__,
                        "op_id": 7,
                    },
                )
                self.assertEqual(
                    self.pop_common([[payload], [dict(payload)]]), [payload]
                )
                restored = self.from_payload(payload)
                self.assertIsInstance(restored, event_type)
                self.assertEqual(int(restored.op_id), 7)


class GroupAwareWireTest(unittest.TestCase):
    def _executor_module(self):
        try:
            return _load_executor_module_without_triton()
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

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
        executor = executor_module.L2CacheExecutor.__new__(
            executor_module.L2CacheExecutor
        )
        executor._ack_lock = threading.Lock()
        executor.attn_tp_rank = 0
        executor._ready_load_op_ids = []
        executor._load_acks = []
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
        )
        executor._transfer_geometry = geometry
        workspace = MagicMock()
        workspace.load_block_transfers.return_value = (1, (0, 1))
        executor._load_workspaces = (workspace,)
        return executor_module, executor, device, geometry, workspace

    def test_hybrid_state_access_waits_for_layer_load(self):
        try:
            from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
                HybridKDATokenToKVPool,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
                HybridMHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

        for pool_type in (HybridMHATokenToKVPool, HybridKDATokenToKVPool):
            with self.subTest(pool_type=pool_type.__name__):
                tracker = Mock()
                pool = pool_type.__new__(pool_type)
                pool.layerwise_load_tracker = tracker
                pool._state_buffers_by_layer = {3: ("conv", "recurrent")}
                if pool_type is HybridMHATokenToKVPool:
                    pool._state_layer_ids = (3,)

                self.assertEqual(pool.get_component(3, "conv_state"), "conv")
                tracker.wait_for_layer.assert_called_once_with(3)

    def test_pool_transfer_layout_matches_scheduler_group_order(self):
        try:
            from cache_pool_test_utils import MinimalCacheView
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

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

        self.assertEqual(
            tuple(group.group_id for group in layout.groups),
            ("state", "full"),
        )

    def test_submit_load_backs_clears_layerwise_waits_without_load(self):
        L2CacheExecutor = self._executor_module().L2CacheExecutor

        tracker = Mock()
        executor = L2CacheExecutor.__new__(L2CacheExecutor)
        executor._ack_lock = threading.Lock()
        executor._load_trackers = [(tracker, 1)]

        executor.submit_load_backs(SimpleNamespace(cache=[]))

        tracker.set_consumers.assert_called_once_with(-1)

    def test_submit_preserves_group_identity(self):
        L2CacheExecutor = self._executor_module().L2CacheExecutor

        op_ids = []
        transfers = []
        L2CacheExecutor._append_transfers(
            [7],
            [[0, 1]],
            [[5, 5]],
            [[9, 9]],
            collected_op_ids=op_ids,
            transfers=transfers,
            source_is_device=True,
        )
        self.assertEqual(op_ids, [7])
        self.assertEqual(transfers, [(0, 5, 9), (1, 5, 9)])

    def test_writeback_reuses_static_geometry_on_the_caller_stream(self):
        executor_module = self._executor_module()
        L2CacheExecutor = executor_module.L2CacheExecutor

        executor = L2CacheExecutor.__new__(L2CacheExecutor)
        executor._ack_lock = threading.Lock()
        executor.attn_tp_rank = 0
        executor._ready_write_op_ids = []
        device = SimpleNamespace(type="cuda")
        executor.layout = SimpleNamespace(buffers=(SimpleNamespace(device=device),))
        executor.host_storage = SimpleNamespace(host_buffer="host")
        executor.transfer_backend = "auto"
        executor._write_acks = []
        executor._write_workspace = Mock()
        executor._write_workspace.load_block_transfers.return_value = (1, (0, 1))
        executor._transfer_geometry = SimpleNamespace(
            device_rows=object(),
            layer_slices=((0, 2), (2, 1)),
            num_field_rows=3,
        )
        stream = object()
        finish = Mock()

        with (
            patch.object(
                executor_module.device_module, "current_stream", return_value=stream
            ),
            patch.object(executor_module.device_module, "Event", return_value=finish),
            patch.object(executor_module, "transfer_cache_blocks") as transfer,
        ):
            executor._start_writing([7], [(0, 5, 9)])

        # On the CALLER's current stream: the copy must read the source pages
        # before anything later in the plan (zeroing, the granted request's
        # writes) can touch them, and the single-stream FIFO is that fence.
        executor._write_workspace.load_block_transfers.assert_called_once_with(
            [(0, 5, 9)], geometry=executor._transfer_geometry
        )
        executor._write_workspace.commit_block_transfers.assert_called_once_with(
            1, device
        )
        transfer.assert_called_once_with(
            "d2h",
            executor.layout.buffers,
            executor.host_storage.host_buffer,
            executor._transfer_geometry,
            executor._write_workspace,
            stream,
            num_blocks=1,
            geometry_offset=0,
            num_geometry_rows=3,
            backend="auto",
        )
        finish.record.assert_called_once_with(stream)

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

        with (
            patch.object(executor_module, "get_is_capture_mode", return_value=False),
            patch.object(executor_module.device_module, "Event", return_value=finish),
            patch.object(executor_module, "transfer_cache_blocks") as transfer,
            patch.object(executor_module.logger, "info") as log_info,
        ):
            executor._start_loading([9], [(0, 2, 1), (0, 5, 4)])

        workspace.load_block_transfers.assert_called_once_with(
            [(0, 2, 1), (0, 5, 4)], geometry=geometry
        )
        workspace.commit_block_transfers.assert_not_called()
        transfer.assert_called_once()
        log_info.assert_called_once_with(
            "[L2] load started: operations=%d blocks=%d", 1, 2
        )

    def test_kernel_init_builds_consumer_ordered_static_geometry_once(self):
        executor_module = self._executor_module()
        L2CacheExecutor = executor_module.L2CacheExecutor

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
                    cache_blocks_per_lcm_block=4,
                    fields=(fields["target.2.state"],),
                ),
                SimpleNamespace(
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
        target_pool.arena.cache_group_specs = (SimpleNamespace(group_id="state"),)
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
                executor_module.psutil,
                "virtual_memory",
                return_value=SimpleNamespace(available=10**12),
            ),
            patch.object(
                executor_module, "LayerwiseLoadTracker", side_effect=make_tracker
            ),
            patch.object(executor_module, "_new_cache_stream", return_value="load"),
            patch.object(executor_module, "HostTransferWorkspace", side_effect=Mock),
            patch.object(
                executor_module,
                "build_host_transfer_geometry",
                return_value=unbound_geometry,
            ) as build_geometry,
        ):
            executor = L2CacheExecutor(
                target_pool,
                draft_pool=draft_pool,
                host_ratio=1.0,
                host_size_gb=0,
                io_backend="kernel",
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
        unbound_geometry.bind.assert_called_once_with(device)
        self.assertIs(executor._transfer_geometry, bound_geometry)
        self.assertEqual([count for count, _ in trackers], [3, 1])

    def test_direct_and_npu_init_keep_geometry_on_the_host(self):
        executor_module = self._executor_module()
        L2CacheExecutor = executor_module.L2CacheExecutor

        pool = Mock()
        pool.arena.cache_group_specs = (SimpleNamespace(group_id="group"),)
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
                executor_module.psutil,
                "virtual_memory",
                return_value=SimpleNamespace(available=10**12),
            ),
            patch.object(executor_module, "LayerwiseLoadTracker", return_value=tracker),
            patch.object(executor_module, "_new_cache_stream", return_value="load"),
            patch.object(executor_module, "HostTransferWorkspace", side_effect=Mock),
            patch.object(
                executor_module,
                "build_host_transfer_geometry",
                side_effect=lambda **_kwargs: SimpleNamespace(
                    device_rows=None,
                    bind=Mock(),
                ),
            ) as build_geometry,
        ):
            for io_backend, device_type in (("direct", "cuda"), ("kernel", "npu")):
                with self.subTest(io_backend=io_backend, device_type=device_type):
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
                                cache_blocks_per_lcm_block=1,
                                fields=(field,),
                            ),
                        ),
                        consumers=(("field",),),
                    )
                    pool.cache_transfer_layout.return_value = layout
                    executor = L2CacheExecutor(
                        pool,
                        host_ratio=1.0,
                        host_size_gb=0,
                        io_backend=io_backend,
                    )
                    self.assertIsNone(executor._transfer_geometry.device_rows)
                    executor._transfer_geometry.bind.assert_not_called()

        self.assertEqual(build_geometry.call_count, 2)

    def test_optional_dependency_shim_restores_existing_modules(self):
        protected_names = (
            "tokenspeed_kernel.ops.kvcache.host_transfer",
            "tokenspeed_scheduler",
            "tokenspeed.runtime.cache.l2.layerwise_load",
            "tokenspeed.runtime.cache.l2.storage",
            "tokenspeed.runtime.cache.transfer.layout",
            "tokenspeed.runtime.execution.cuda_graph_wrapper",
            "tokenspeed.runtime.utils",
            "tokenspeed.runtime.cache.l2.executor",
        )
        sentinels = {name: ModuleType(name) for name in protected_names}

        with patch.dict(sys.modules, sentinels, clear=False):
            isolated = _load_executor_module_without_triton(force_isolated=True)

            self.assertIsNot(isolated, sentinels[protected_names[-1]])
            for name, sentinel in sentinels.items():
                self.assertIs(sys.modules[name], sentinel)

    def test_two_isolated_loads_preserve_imported_torch_modules(self):
        first = _load_executor_module_without_triton(force_isolated=True)
        first_torch = first.torch

        self.assertIs(sys.modules["torch"], first_torch)
        second = _load_executor_module_without_triton(force_isolated=True)

        self.assertIs(second.torch, first_torch)
        self.assertIs(sys.modules["torch"], first_torch)

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
        executor._verifier = None
        load_events = SimpleNamespace(
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
            patch.object(
                executor_module, "layer_ready_ptx_supported", return_value=True
            ),
            patch.object(executor_module, "transfer_cache_blocks") as transfer,
        ):
            executor._start_loading([9], [(0, 2, 1)])

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
        )
        finish.record.assert_called_once_with(executor.load_stream)
        self.assertEqual(load_events.layer_done_events, [finish, finish, finish])
        self.assertIs(load_events.layer_ready_flags, flags)
        self.assertIs(load_events.wait_layer_ready, executor_module.wait_layer_ready)
        self.assertIs(executor._load_acks[0].finish_event, finish)

    def test_loadback_launch_failure_retires_all_target_and_draft_events(self):
        executor_module, executor, _, _, _ = self._make_load_executor(
            consumers=(("target.0",), ("target.1",), ("draft.0",)),
            layer_slices=((0, 1), (1, 1), (2, 1)),
            device_rows=object(),
            load_stream=Mock(),
        )
        target_events = SimpleNamespace(
            start_event=Mock(),
            layer_done_events=[Mock(), Mock()],
            layer_ready_init_event=Mock(),
        )
        draft_events = SimpleNamespace(
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
                executor_module, "layer_ready_ptx_supported", return_value=True
            ),
            patch.object(
                executor_module,
                "transfer_cache_blocks",
                side_effect=RuntimeError("layer launch failed"),
            ) as transfer,
        ):
            with self.assertRaisesRegex(RuntimeError, "layer launch failed"):
                executor._start_loading([9], [(0, 2, 1)])

        self.assertEqual(transfer.call_count, 1)
        retirement.record.assert_called_once_with(executor.load_stream)
        self.assertEqual(
            target_events.layer_done_events,
            [retirement, retirement],
        )
        self.assertEqual(draft_events.layer_done_events, [retirement])
        executor.load_stream.synchronize.assert_not_called()
        self.assertEqual(executor._load_acks, [])

    def test_failed_retirement_sync_poisons_executor_and_preserves_original_error(self):
        executor_module, executor, _, _, _ = self._make_load_executor(
            consumers=(("target.0",),),
            layer_slices=((0, 1),),
            device_rows=object(),
            load_stream=Mock(),
        )
        executor._write_acks = []
        executor._ready_write_op_ids = []
        load_events = SimpleNamespace(
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
                executor_module, "layer_ready_ptx_supported", return_value=True
            ),
            patch.object(
                executor_module,
                "transfer_cache_blocks",
                side_effect=original_error,
            ),
        ):
            with self.assertRaises(RuntimeError) as raised:
                executor._start_loading([9], [(0, 2, 1)])

            self.assertIs(raised.exception, original_error)
            self.assertEqual(str(raised.exception), "original layer launch failed")
            self.assertTrue(executor._load_poisoned)
            notes = getattr(raised.exception, "__notes__", ())
            self.assertTrue(any("retirement record failed" in note for note in notes))
            self.assertTrue(any("retirement sync failed" in note for note in notes))

            executor.load_stream.synchronize.side_effect = None
            executor.shutdown = Mock()
            executor.reset()
            self.assertTrue(executor._load_poisoned)
            with self.assertRaisesRegex(RuntimeError, "poisoned"):
                executor._start_loading([10], [(0, 3, 2)])

        tracker.begin_load.assert_called_once_with()


class CompactLayoutRoundTripTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch

            import tokenspeed.runtime.cache.l2.executor as executor_module
            from tokenspeed.runtime.cache.transfer.layout import (
                CacheField,
                CacheGroupLayout,
                CacheTransferLayout,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
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
            executor = self.executor_module.L2CacheExecutor(
                pool,
                draft_pool=draft_pool,
                host_ratio=1.0,
                host_size_gb=0,
                io_backend=io_backend,
            )
        self.addCleanup(executor.shutdown)
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
        )
        torch.cuda.current_stream().synchronize()
        write_results = executor.poll_results()
        self.assertEqual([int(event.op_id) for event in write_results], [7])

        # Destroy every Device byte so stale cache contents cannot make the
        # H2D half of the round trip pass accidentally.
        first.fill_(0xEE)
        second.fill_(0xEE)
        torch.cuda.synchronize()

        load_index = executor._start_loading(  # pylint: disable=protected-access
            [9],
            [(0, 2, 1), (0, 5, 4), (1, 4, 3)],
        )
        self.assertIsNotNone(load_index)
        pool.load_tracker.set_consumers(load_index)
        pool.load_tracker.wait_for_layer(0)
        pool.load_tracker.wait_for_layer(1)
        torch.cuda.synchronize()
        load_results = executor.poll_results()
        self.assertEqual([int(event.op_id) for event in load_results], [9])
        # Hand-derived destination ranges for blocks (full: 2, 5; state: 4).
        expected_first = torch.full((128,), 0xEE, dtype=torch.uint8)
        expected_second = torch.full((128,), 0xEE, dtype=torch.uint8)
        expected_first[24:28].copy_(full_k_one)
        expected_second[40:46].copy_(full_v_one)
        expected_first[48:52].copy_(full_k_four)
        expected_second[76:82].copy_(full_v_four)
        expected_first[104:109].copy_(state_three)
        self.assertTrue(torch.equal(first.cpu(), expected_first))
        self.assertTrue(torch.equal(second.cpu(), expected_second))

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
            target_layout, draft_layout=draft_layout, io_backend="kernel"
        )

        device[16:20].fill_(0x11)
        device[56:60].fill_(0x12)
        torch.cuda.synchronize()
        executor._start_writing([7], [(0, 1, 1)])  # pylint: disable=protected-access
        torch.cuda.synchronize()
        self.assertEqual([int(event.op_id) for event in executor.poll_results()], [7])

        device.fill_(0xEE)
        torch.cuda.synchronize()
        load_index = executor._start_loading(  # pylint: disable=protected-access
            [9], [(0, 2, 1)]
        )
        self.assertIsNotNone(load_index)
        target_pool.load_tracker.set_consumers(load_index)
        draft_pool.load_tracker.set_consumers(load_index)
        target_pool.load_tracker.wait_for_layer(0)
        target_pool.load_tracker.wait_for_layer(1)
        draft_pool.load_tracker.wait_for_layer(0)
        torch.cuda.synchronize()
        self.assertEqual([int(event.op_id) for event in executor.poll_results()], [9])
        self.assertEqual(device[24:28].tolist(), [0x11] * 4)
        self.assertEqual(device[64:68].tolist(), [0x12] * 4)


if __name__ == "__main__":
    unittest.main()
