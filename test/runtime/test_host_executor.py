"""Compact Host cache executor tests."""

from __future__ import annotations

import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")


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
        try:
            from tokenspeed.runtime.cache.l2.executor import L2CacheExecutor
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

        tracker = Mock()
        executor = L2CacheExecutor.__new__(L2CacheExecutor)
        executor._ack_lock = threading.Lock()
        executor._load_trackers = [(tracker, 1)]

        executor.submit_load_backs(SimpleNamespace(cache=[]))

        tracker.set_consumers.assert_called_once_with(-1)

    def test_submit_preserves_group_identity(self):
        try:
            from tokenspeed.runtime.cache.l2.executor import L2CacheExecutor
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

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

    def test_writeback_calls_transfer_with_compact_layout(self):
        try:
            import tokenspeed.runtime.cache.l2.executor as executor_module

            L2CacheExecutor = executor_module.L2CacheExecutor
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

        executor = L2CacheExecutor.__new__(L2CacheExecutor)
        executor._ack_lock = threading.Lock()
        executor._ready_write_op_ids = []
        executor.layout = SimpleNamespace(buffers=("device",))
        executor.host_storage = SimpleNamespace(host_buffer="host")
        executor.transfer_backend = "dma"
        executor._write_acks = []
        ranges = [(0, 64, 128, 32)]
        executor._transfer_ranges = Mock(return_value=ranges)
        stream = object()
        finish = Mock()

        with (
            patch.object(
                executor_module.torch.cuda, "current_stream", return_value=stream
            ),
            patch.object(executor_module.torch.cuda, "Event", return_value=finish),
            patch.object(executor_module, "transfer_cache_ranges") as transfer,
        ):
            executor._start_writing([7], [(0, 5, 9)])

        # On the CALLER's current stream: the copy must read the source pages
        # before anything later in the plan (zeroing, the granted request's
        # writes) can touch them, and the single-stream FIFO is that fence.
        transfer.assert_called_once_with(
            "d2h",
            executor.layout.buffers,
            executor.host_storage.host_buffer,
            ranges,
            stream,
            backend="dma",
        )
        finish.record.assert_called_once_with(stream)

    def test_loadback_logs_non_empty_batch(self):
        try:
            import tokenspeed.runtime.cache.l2.executor as executor_module

            L2CacheExecutor = executor_module.L2CacheExecutor
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

        executor = L2CacheExecutor.__new__(L2CacheExecutor)
        executor._ack_lock = threading.Lock()
        executor._ready_load_op_ids = []
        executor._load_acks = []
        executor.load_stream = object()
        executor.transfer_backend = "dma"
        executor.layout = SimpleNamespace(buffers=("device",), consumers=(("field",),))
        executor.host_storage = SimpleNamespace(host_buffer="host")
        executor._transfer_ranges = Mock(return_value=[(0, 64, 128, 32)])
        load_events = SimpleNamespace(start_event=Mock(), layer_done_events=[None])
        tracker = Mock()
        tracker.begin_load.return_value = 0
        tracker.event_sets = [load_events]
        executor._load_trackers = [(tracker, 1)]
        finish = Mock()

        with (
            patch.object(executor_module, "get_is_capture_mode", return_value=False),
            patch.object(executor_module.device_module, "Event", return_value=finish),
            patch.object(executor_module, "transfer_cache_ranges"),
            patch.object(executor_module.logger, "info") as log_info,
        ):
            executor._start_loading([9], [(0, 2, 1), (0, 5, 4)])

        log_info.assert_called_once_with(
            "[L2] load started: operations=%d blocks=%d", 1, 2
        )


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

    def test_real_transfer_restores_compact_multigroup_layout_byte_exactly(self):
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

        class SyntheticPool:
            def cache_transfer_layout(self):
                return layout

            def register_layerwise_load_tracker(self, tracker):
                self.load_tracker = tracker

        pool = SyntheticPool()
        pool.arena = SimpleNamespace(
            cache_group_specs=tuple(
                SimpleNamespace(group_id=group.group_id) for group in layout.groups
            ),
        )
        with patch.object(self.executor_module, "_HOST_MEM_HEADROOM_BYTES", 0):
            executor = self.executor_module.L2CacheExecutor(
                pool,
                host_ratio=1.0,
                host_size_gb=0,
                io_backend="direct",
            )

        # Hand-derived Device ranges for blocks (full: 1, 4; state: 3).
        first[16:20].fill_(0x11)
        second[28:34].fill_(0x12)
        first[40:44].fill_(0x41)
        second[64:70].fill_(0x42)
        first[94:99].fill_(0x73)
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
        self.assertTrue(
            torch.equal(first[24:28].cpu(), torch.full((4,), 0x11, dtype=torch.uint8))
        )
        self.assertTrue(
            torch.equal(second[40:46].cpu(), torch.full((6,), 0x12, dtype=torch.uint8))
        )
        self.assertTrue(
            torch.equal(first[48:52].cpu(), torch.full((4,), 0x41, dtype=torch.uint8))
        )
        self.assertTrue(
            torch.equal(second[76:82].cpu(), torch.full((6,), 0x42, dtype=torch.uint8))
        )
        self.assertTrue(
            torch.equal(first[104:109].cpu(), torch.full((5,), 0x73, dtype=torch.uint8))
        )
        executor.shutdown()


if __name__ == "__main__":
    unittest.main()
