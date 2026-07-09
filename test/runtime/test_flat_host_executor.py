"""FlatMemoryExecutor (M15 Phase D2): flat host tier through the kvstore
transport.

Covers the executor roundtrip against a real (tiny) device pool with
WriteBackDone AND LoadBackDone acks, the ack payload shape riding the
TP-synced commit path, the layer -> mirror-tensor fencing mapping, and the
num_host_pages sizing arithmetic (pure CPU).
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")

_PKG_FLAT_PROBE = (
    "tokenspeed.runtime.configs.paged_cache_spec.scheduler_ext_flat_kvcache"
)

LAYER_TYPES = ("sliding_attention", "full_attention") * 2

# GDN hybrid: layers 0/2 are state layers (slab pairs 0/1); the KV side
# stays legacy per-layer (linear_attention disables slab pairing), and the
# LAST layer is an attention layer -- exercising the finish-event pin.
GDN_LAYER_TYPES = ("linear_attention", "full_attention") * 2


class _StubPool:
    """CPU-only device-pool stand-in for sizing arithmetic: 4 layers dedup
    to 2 K + 2 V slabs (paired layers alias the same tensor)."""

    def __init__(self, torch):
        self.page_size = 4
        self.size = 32
        rows = self.size + self.page_size
        k_slabs = [torch.zeros((rows, 1, 8), dtype=torch.bfloat16) for _ in range(2)]
        v_slabs = [torch.zeros((rows, 1, 8), dtype=torch.bfloat16) for _ in range(2)]
        self.k_buffer = [k_slabs[0], k_slabs[0], k_slabs[1], k_slabs[1]]
        self.v_buffer = [v_slabs[0], v_slabs[0], v_slabs[1], v_slabs[1]]


class FlatHostPageSizingTest(unittest.TestCase):
    """num_host_pages budget arithmetic; no CUDA, no scheduler ext."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.cache.flat_host_mirror import (
                FlatHostMirror,
                flat_bytes_per_host_page,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch: {exc}")
        self.torch = torch
        self.FlatHostMirror = FlatHostMirror
        self.flat_bytes_per_host_page = flat_bytes_per_host_page

    def test_bytes_per_host_page_matches_mirror(self):
        stub = _StubPool(self.torch)
        # 4 distinct mirrors x page_size 4 x row 1*8 bf16 (16 B) = 256 B.
        self.assertEqual(self.flat_bytes_per_host_page(stub), 256)
        mirror = self.FlatHostMirror(stub, num_host_pages=2)
        self.assertEqual(mirror.bytes_per_host_page(), 256)

    def test_num_host_pages_formula(self):
        from tokenspeed.runtime.cache.executor.flat_memory_executor import (
            flat_num_host_pages,
        )

        # Ratio sizing mirrors the radix token->page align-up:
        # int(size * ratio) // page_size + 1.
        self.assertEqual(
            flat_num_host_pages(
                bytes_per_host_page=256,
                device_pool_size=32,
                page_size=4,
                host_ratio=2.0,
                host_size_gb=0,
            ),
            int(32 * 2.0) // 4 + 1,
        )
        # Explicit GB budget floors to whole mirror pages (never exceeds).
        self.assertEqual(
            flat_num_host_pages(
                bytes_per_host_page=256,
                device_pool_size=32,
                page_size=4,
                host_ratio=2.0,
                host_size_gb=1,
            ),
            int(1e9) // 256,
        )
        # A budget below one page is a configuration error.
        with self.assertRaises(ValueError):
            flat_num_host_pages(
                bytes_per_host_page=int(1e9),
                device_pool_size=32,
                page_size=4,
                host_ratio=2.0,
                host_size_gb=0.5,
            )


class LoadBackDonePayloadTest(unittest.TestCase):
    """LoadBackDone rides the same TP-synced commit path as WriteBackDone;
    needs the scheduler ext, not a GPU."""

    def setUp(self):
        try:
            from tokenspeed_scheduler import Cache

            from tokenspeed.runtime.engine.scheduler_utils import (
                cache_event_from_payload,
                cache_event_to_payload,
                pop_common_cache_event_payloads,
            )
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            self.skipTest(f"needs tokenspeed_scheduler ext: {exc}")
        if not hasattr(Cache, "LoadBackDoneEvent"):
            self.skipTest("scheduler ext predates the LoadBackDoneEvent binding")
        self.Cache = Cache
        self.to_payload = cache_event_to_payload
        self.from_payload = cache_event_from_payload
        self.pop_common = pop_common_cache_event_payloads

    def _load_done(self, op_id: int):
        evt = self.Cache.LoadBackDoneEvent()
        evt.op_id = op_id
        evt.success = True
        return evt

    def test_payload_shape_and_roundtrip(self):
        payload = self.to_payload(self._load_done(3))
        self.assertEqual(
            payload,
            {
                "kind": "LoadBackDoneEvent",
                "op_id": 3,
                "success": True,
                "request_id": "",
            },
        )
        ready = self.pop_common([[payload]])  # world_size=1 gather
        self.assertEqual(ready, [payload])
        evt = self.from_payload(ready[0])
        self.assertIsInstance(evt, self.Cache.LoadBackDoneEvent)
        self.assertEqual(int(evt.op_id), 3)
        self.assertTrue(evt.success)

    def test_pop_common_requires_all_ranks(self):
        payload = self.to_payload(self._load_done(4))
        self.assertEqual(self.pop_common([[payload], []]), [])
        both = self.pop_common([[payload], [dict(payload)]])
        self.assertEqual(both, [payload])


class FlatMemoryExecutorTest(unittest.TestCase):
    """Real (tiny) MHATokenToKVPool on GPU driving the flat executor."""

    def setUp(self):
        try:
            import torch

            from tokenspeed_scheduler import Cache

            from tokenspeed.runtime.cache.executor.flat_memory_executor import (
                FlatMemoryExecutor,
            )
            from tokenspeed.runtime.cache.transfer.types import CacheKind
            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel + scheduler ext: {exc}")
        if not hasattr(Cache, "LoadBackDoneEvent"):
            self.skipTest("scheduler ext predates the LoadBackDoneEvent binding")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        self.torch = torch
        self.Cache = Cache
        self.CacheKind = CacheKind
        self.FlatMemoryExecutor = FlatMemoryExecutor
        self.MHATokenToKVPool = MHATokenToKVPool

    def _pool(self):
        kwargs = dict(
            size=32,
            dtype=self.torch.bfloat16,
            head_num=1,
            head_dim=8,
            layer_num=4,
            device="cuda",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=64,
            page_size=4,
            rank=0,
            layer_types=LAYER_TYPES,
            sliding_window_tokens=128,
            enable_alt_stream=False,
        )
        with mock.patch(_PKG_FLAT_PROBE, return_value=True):
            return self.MHATokenToKVPool(**kwargs)

    def _executor(self, pool):
        return self.FlatMemoryExecutor(
            device_pool=pool, host_ratio=2.0, host_size_gb=0
        )

    def _fill_device_pages(self, mirror, device_pages):
        p = mirror.page_size
        for tensor_idx, (dev, _) in enumerate(mirror.tensor_pairs):
            for d in device_pages:
                dev[d * p : (d + 1) * p].fill_(tensor_idx * 16 + d + 1)
        self.torch.cuda.synchronize()

    def _snapshot(self, mirror, device_pages):
        p = mirror.page_size
        return [
            {d: dev[d * p : (d + 1) * p].cpu().clone() for d in device_pages}
            for dev, _ in mirror.tensor_pairs
        ]

    def _drain(self, executor, expect: int) -> list:
        results = []
        for _ in range(1000):
            results.extend(executor.poll_results())
            if len(results) >= expect:
                return results
            self.torch.cuda.synchronize()
        self.fail(f"expected {expect} acks, drained {len(results)}")

    def test_roundtrip_with_acks_and_fencing(self):
        torch = self.torch
        pool = self._pool()
        executor = self._executor(pool)
        mirror = executor.mirror
        # Ratio sizing: int(32 * 2.0) // 4 + 1.
        self.assertEqual(executor.num_host_pages, 17)
        self.assertTrue(executor.emits_loadback_acks)
        # The fencing counter is registered where the radix pool would be.
        self.assertIs(pool.layer_transfer_counter, executor._counter)

        device_pages = [1, 2]
        self._fill_device_pages(mirror, device_pages)
        before = self._snapshot(mirror, device_pages)

        # WriteBack: device pages [1, 2] -> host pages [5, 6].
        executor.submit_writeback([7], [[1, 2]], [[5, 6]])
        executor.flush()
        results = self._drain(executor, 1)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], self.Cache.WriteBackDoneEvent)
        self.assertEqual(int(results[0].op_id), 7)
        self.assertTrue(results[0].success)

        for dev, _ in mirror.tensor_pairs:
            p = mirror.page_size
            for d in device_pages:
                dev[d * p : (d + 1) * p].zero_()
        torch.cuda.synchronize()

        # LoadBack: host pages [5, 6] -> device pages [1, 2] (wire order:
        # src=host, dst=device, as C++ FlatLoadBackOperation emits).
        executor.submit_loadback([9], [[5, 6]], [[1, 2]])
        executor.flush()

        # Layerwise fencing: producer registered under the op, consumer waits
        # gate per layer through the pool's registered counter.
        producer_idx = executor.get_producer_index(self.CacheKind.KV, 9)
        self.assertIsNotNone(producer_idx)
        executor.set_consumer(self.CacheKind.KV, [producer_idx])
        for layer_id in range(4):
            pool.layer_transfer_counter.wait_until(layer_id)
        torch.cuda.synchronize()

        results = self._drain(executor, 1)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], self.Cache.LoadBackDoneEvent)
        self.assertEqual(int(results[0].op_id), 9)
        self.assertTrue(results[0].success)

        after = self._snapshot(mirror, device_pages)
        for tensor_idx in range(len(mirror.tensor_pairs)):
            for d in device_pages:
                self.assertTrue(
                    torch.equal(
                        before[tensor_idx][d].view(torch.uint8),
                        after[tensor_idx][d].view(torch.uint8),
                    ),
                    f"tensor {tensor_idx} device page {d} not byte-exact",
                )

        # Producer index is popped exactly once (event_loop consumes it in
        # _setup_layerwise_loadback right after submit).
        self.assertIsNone(executor.get_producer_index(self.CacheKind.KV, 9))

    def test_layer_event_mapping(self):
        pool = self._pool()
        executor = self._executor(pool)
        mirror = executor.mirror
        self._fill_device_pages(mirror, [3])
        executor.submit_writeback([1], [[3]], [[0]])
        executor.flush()
        self._drain(executor, 1)

        executor.submit_loadback([2], [[0]], [[3]])
        executor.flush()
        producer_idx = executor.get_producer_index(self.CacheKind.KV, 2)
        producer_event = executor._counter.events[producer_idx]
        # Paired slab layers share their V slab's event; distinct groups
        # get distinct events; finish_event covers every copy.
        self.assertIs(producer_event.load_events[0], producer_event.load_events[1])
        self.assertIs(producer_event.load_events[2], producer_event.load_events[3])
        self.assertIsNot(producer_event.load_events[0], producer_event.load_events[2])
        self.torch.cuda.synchronize()
        self.assertTrue(producer_event.finish_event.query())
        self._drain(executor, 1)

    def _state_pool(self):
        kwargs = dict(
            size=32,
            dtype=self.torch.bfloat16,
            head_num=1,
            head_dim=8,
            layer_num=4,
            device="cuda",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=64,
            page_size=4,
            rank=0,
            layer_types=GDN_LAYER_TYPES,
            sliding_window_tokens=None,
            enable_alt_stream=False,
            conv_state_shape=(2, 4),
            temporal_state_shape=(2, 8),
        )
        with mock.patch(_PKG_FLAT_PROBE, return_value=True):
            return self.MHATokenToKVPool(**kwargs)

    def _fill_spans(self, mirror, device_pages):
        for tensor_idx, ((dev, _), span) in enumerate(
            zip(mirror.tensor_pairs, mirror.row_spans)
        ):
            for d in device_pages:
                dev[d * span : (d + 1) * span].fill_(tensor_idx * 16 + d + 1)
        self.torch.cuda.synchronize()

    def _snapshot_spans(self, mirror, device_pages):
        return [
            {
                d: dev[d * span : (d + 1) * span].cpu().clone()
                for d in device_pages
            }
            for (dev, _), span in zip(mirror.tensor_pairs, mirror.row_spans)
        ]

    def test_state_layer_event_mapping(self):
        pool = self._state_pool()
        executor = self._executor(pool)
        mirror = executor.mirror
        # Legacy KV (4 K + 4 V) + conv0, ssm0, conv1, ssm1.
        self.assertEqual(len(mirror.tensor_pairs), 12)
        self._fill_spans(mirror, [3])
        executor.submit_writeback([1], [[3]], [[0]])
        executor.flush()
        self._drain(executor, 1)

        # Spy on the per-tensor events to pin the layer -> event mapping.
        captured = {}
        orig = mirror.load_pages_with_events

        def spy(pairs, stream):
            events = orig(pairs, stream)
            captured["events"] = events
            return events

        mirror.load_pages_with_events = spy
        executor.submit_loadback([2], [[0]], [[3]])
        executor.flush()
        events = captured["events"]
        self.assertEqual(len(events), 12)
        producer_idx = executor.get_producer_index(self.CacheKind.KV, 2)
        producer_event = executor._counter.events[producer_idx]
        # State layers 0/2 ack on their ssm event (conv precedes ssm on the
        # serial stream, so it covers the pair); attention layer 1 keeps its
        # V-tensor event; the LAST layer pins events[-1] so finish_event
        # (producer-slot reuse fence) covers the trailing state copies.
        self.assertIs(producer_event.load_events[0], events[9])
        self.assertIs(producer_event.load_events[1], events[5])
        self.assertIs(producer_event.load_events[2], events[11])
        self.assertIs(producer_event.load_events[3], events[-1])
        self.torch.cuda.synchronize()
        self.assertTrue(producer_event.finish_event.query())
        self._drain(executor, 1)

    def test_state_pool_roundtrip_with_fencing(self):
        torch = self.torch
        pool = self._state_pool()
        executor = self._executor(pool)
        mirror = executor.mirror

        device_pages = [1, 2]
        self._fill_spans(mirror, device_pages)
        before = self._snapshot_spans(mirror, device_pages)

        executor.submit_writeback([21], [[1, 2]], [[5, 6]])
        executor.flush()
        self._drain(executor, 1)

        for (dev, _), span in zip(mirror.tensor_pairs, mirror.row_spans):
            for d in device_pages:
                dev[d * span : (d + 1) * span].zero_()
        torch.cuda.synchronize()

        executor.submit_loadback([23], [[5, 6]], [[1, 2]])
        executor.flush()
        producer_idx = executor.get_producer_index(self.CacheKind.KV, 23)
        self.assertIsNotNone(producer_idx)
        executor.set_consumer(self.CacheKind.KV, [producer_idx])
        for layer_id in range(4):
            pool.layer_transfer_counter.wait_until(layer_id)
        torch.cuda.synchronize()
        self._drain(executor, 1)

        after = self._snapshot_spans(mirror, device_pages)
        for tensor_idx in range(len(mirror.tensor_pairs)):
            for d in device_pages:
                self.assertTrue(
                    torch.equal(
                        before[tensor_idx][d].view(torch.uint8),
                        after[tensor_idx][d].view(torch.uint8),
                    ),
                    f"tensor {tensor_idx} device page {d} not byte-exact",
                )

    def test_empty_op_acks_immediately(self):
        pool = self._pool()
        executor = self._executor(pool)
        # C++ dedups transfers across ops of one batched operation, so an op
        # can arrive with empty page lists; it still owes exactly one ack.
        executor.submit_writeback([11], [[]], [[]])
        executor.submit_loadback([12], [[]], [[]])
        executor.flush()
        results = self._drain(executor, 2)
        kinds = {type(r).__name__: int(r.op_id) for r in results}
        self.assertEqual(
            kinds, {"WriteBackDoneEvent": 11, "LoadBackDoneEvent": 12}
        )


if __name__ == "__main__":
    unittest.main()
