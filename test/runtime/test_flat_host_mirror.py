"""FlatHostMirror (M15 Phase D1): byte-blind pinned-CPU slab mirror.

Pins the transport contract only (no engine wiring): one mirror per
distinct device KV tensor, whole-page row-range copies both directions,
per-tensor load events, and the layer -> tensor-index mapping D2 fences on.
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

# GDN hybrid: layers 0/2 are state layers (pairs 0/1); linear_attention
# disables slab pairing, so the KV side stays legacy per-layer.
GDN_LAYER_TYPES = ("linear_attention", "full_attention") * 2


class FlatHostMirrorTest(unittest.TestCase):
    """Real (tiny) MHATokenToKVPool on GPU, slab and legacy layouts."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.cache.flat_host_mirror import (
                FlatHostMirror,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        self.torch = torch
        self.FlatHostMirror = FlatHostMirror
        self.MHATokenToKVPool = MHATokenToKVPool

    def _pool(self, *, flat_ext: bool = True):
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
        with mock.patch(_PKG_FLAT_PROBE, return_value=flat_ext):
            return self.MHATokenToKVPool(**kwargs)

    def _fill_device_pages(self, mirror, device_pages):
        # Sentinels distinct per (tensor, page); bf16-exact small ints.
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

    def _roundtrip_assert(self, mirror, pairs):
        torch = self.torch
        p = mirror.page_size
        device_pages = [d for d, _ in pairs]
        self._fill_device_pages(mirror, device_pages)
        before = self._snapshot(mirror, device_pages)

        stream = torch.cuda.Stream()
        mirror.store_pages(pairs, stream)
        stream.synchronize()
        for dev, _ in mirror.tensor_pairs:
            for d in device_pages:
                dev[d * p : (d + 1) * p].zero_()
        torch.cuda.synchronize()
        mirror.load_pages(pairs, stream)
        stream.synchronize()

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

    def test_slab_roundtrip(self):
        pool = self._pool(flat_ext=True)
        mirror = self.FlatHostMirror(pool, num_host_pages=8)
        # 4 layers dedup to 2 K + 2 V slabs.
        self.assertEqual(len(mirror.tensor_pairs), 4)
        self._roundtrip_assert(mirror, [(1, 5), (2, 6), (3, 7)])
        # 4 mirrors x page_size 4 x row 1*8 bf16 (16 B) = 256 B per page.
        self.assertEqual(mirror.bytes_per_host_page(), 4 * 4 * 16)

    def test_interleaved_groups_roundtrip(self):
        # Pages owned by different groups: byte-blind copies need no
        # group awareness (id-exclusivity keeps rows disjoint).
        pool = self._pool(flat_ext=True)
        mirror = self.FlatHostMirror(pool, num_host_pages=4)
        self._roundtrip_assert(mirror, [(2, 0), (3, 1)])

    def test_legacy_roundtrip(self):
        # Legacy layout: all 4+4 per-layer mirrors carry data; copying
        # rows dead for a page's owner group is harmless (byte-exact).
        pool = self._pool(flat_ext=False)
        mirror = self.FlatHostMirror(pool, num_host_pages=8)
        self.assertEqual(len(mirror.tensor_pairs), 8)
        self._roundtrip_assert(mirror, [(1, 3), (2, 4)])

    def test_events_and_layer_mapping(self):
        torch = self.torch
        pool = self._pool(flat_ext=True)
        mirror = self.FlatHostMirror(pool, num_host_pages=8)
        self._fill_device_pages(mirror, [1])

        stream = torch.cuda.Stream()
        events = mirror.load_pages_with_events([(1, 5)], stream)
        self.assertEqual(len(events), len(mirror.tensor_pairs))
        stream.synchronize()
        self.assertTrue(all(event.query() for event in events))

        # Slab: paired layers map to the same K-tensor index.
        self.assertEqual(mirror.num_k_tensors, 2)
        self.assertEqual(
            mirror.tensor_index_of_layer(0), mirror.tensor_index_of_layer(1)
        )
        self.assertEqual(
            mirror.tensor_index_of_layer(2), mirror.tensor_index_of_layer(3)
        )
        self.assertNotEqual(
            mirror.tensor_index_of_layer(0), mirror.tensor_index_of_layer(2)
        )
        for layer_id in range(4):
            idx = mirror.tensor_index_of_layer(layer_id)
            self.assertIs(mirror.tensor_pairs[idx][0], pool.k_buffer[layer_id])
            self.assertIs(
                mirror.tensor_pairs[idx + mirror.num_k_tensors][0],
                pool.v_buffer[layer_id],
            )

        # Legacy: every layer maps to a distinct index.
        legacy = self.FlatHostMirror(self._pool(flat_ext=False), num_host_pages=2)
        self.assertEqual(
            {legacy.tensor_index_of_layer(i) for i in range(4)}, {0, 1, 2, 3}
        )


class FlatHostMirrorStateSlabTest(unittest.TestCase):
    """State slabs join the mirrored set: tensor_pairs order is K*, V*,
    then (conv, ssm) flattened in slab order; state mirrors use 1-row
    PAGE spans (state slabs are page-indexed) while KV mirrors span
    page_size token rows."""

    CONV_SHAPE = (2, 4)  # 16 B/row bf16
    SSM_SHAPE = (2, 8)  # 32 B/row bf16

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.cache.flat_host_mirror import (
                FlatHostMirror,
                flat_bytes_per_host_page,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        self.torch = torch
        self.FlatHostMirror = FlatHostMirror
        self.flat_bytes_per_host_page = flat_bytes_per_host_page
        self.MHATokenToKVPool = MHATokenToKVPool

    def _pool(self, *, with_state: bool = True):
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
        )
        if with_state:
            kwargs.update(
                conv_state_shape=self.CONV_SHAPE,
                temporal_state_shape=self.SSM_SHAPE,
            )
        with mock.patch(_PKG_FLAT_PROBE, return_value=True):
            return self.MHATokenToKVPool(**kwargs)

    def _fill_device_pages(self, mirror, device_pages):
        # Sentinels distinct per (tensor, page); bf16-exact small ints.
        for tensor_idx, ((dev, _), span) in enumerate(
            zip(mirror.tensor_pairs, mirror.row_spans)
        ):
            for d in device_pages:
                dev[d * span : (d + 1) * span].fill_(tensor_idx * 16 + d + 1)
        self.torch.cuda.synchronize()

    def _snapshot(self, mirror, device_pages):
        return [
            {
                d: dev[d * span : (d + 1) * span].cpu().clone()
                for d in device_pages
            }
            for (dev, _), span in zip(mirror.tensor_pairs, mirror.row_spans)
        ]

    def test_state_tensors_follow_kv_in_slab_order(self):
        pool = self._pool()
        mirror = self.FlatHostMirror(pool, num_host_pages=8)
        # Legacy KV layout (4 K + 4 V), then conv0, ssm0, conv1, ssm1 --
        # PINNED order: K*, V*, state tensors flattened in slab order.
        self.assertEqual(mirror.num_k_tensors, 4)
        self.assertEqual(len(mirror.tensor_pairs), 12)
        self.assertEqual(len(pool.state_slabs), 2)
        for n, (conv, ssm) in enumerate(pool.state_slabs):
            self.assertIs(mirror.tensor_pairs[8 + 2 * n][0], conv)
            self.assertIs(mirror.tensor_pairs[8 + 2 * n + 1][0], ssm)
        # Per-pair row spans: page_size token rows for KV, 1 page row for
        # state (state slabs are page-indexed).
        self.assertEqual(mirror.row_spans, (4,) * 8 + (1,) * 4)
        for (dev, host), span in zip(mirror.tensor_pairs, mirror.row_spans):
            if span == 1:
                self.assertEqual(host.shape, (8, *dev.shape[1:]))
            else:
                self.assertEqual(host.shape, (8 * 4, *dev.shape[1:]))

    def test_bytes_per_host_page_includes_state_rows(self):
        base = self.flat_bytes_per_host_page(self._pool(with_state=False))
        pool = self._pool()
        with_state = self.flat_bytes_per_host_page(pool)
        # 2 state layers x (conv 2*4 + ssm 2*8) bf16 page rows = 2 x 48 B.
        self.assertEqual(with_state - base, 96)
        mirror = self.FlatHostMirror(pool, num_host_pages=2)
        self.assertEqual(mirror.bytes_per_host_page(), with_state)

    def test_state_roundtrip(self):
        torch = self.torch
        pool = self._pool()
        mirror = self.FlatHostMirror(pool, num_host_pages=8)
        pairs = [(1, 5), (2, 6), (3, 7)]
        device_pages = [d for d, _ in pairs]
        self._fill_device_pages(mirror, device_pages)
        before = self._snapshot(mirror, device_pages)

        stream = torch.cuda.Stream()
        mirror.store_pages(pairs, stream)
        stream.synchronize()
        for (dev, _), span in zip(mirror.tensor_pairs, mirror.row_spans):
            for d in device_pages:
                dev[d * span : (d + 1) * span].zero_()
        torch.cuda.synchronize()
        events = mirror.load_pages_with_events(pairs, stream)
        self.assertEqual(len(events), len(mirror.tensor_pairs))
        stream.synchronize()
        self.assertTrue(all(event.query() for event in events))

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

    def test_state_tensor_indices_of_layer(self):
        pool = self._pool()
        mirror = self.FlatHostMirror(pool, num_host_pages=2)
        # State layers 0/2 bind slab pairs 0/1 -> flattened indices after
        # the 8 KV mirrors; conv immediately precedes its ssm.
        self.assertEqual(mirror.state_tensor_indices_of_layer(0), (8, 9))
        self.assertEqual(mirror.state_tensor_indices_of_layer(2), (10, 11))
        self.assertIsNone(mirror.state_tensor_indices_of_layer(1))
        self.assertIsNone(mirror.state_tensor_indices_of_layer(3))
        # Pools without state slabs expose no state indices for any layer.
        kv_only = self.FlatHostMirror(
            self._pool(with_state=False), num_host_pages=2
        )
        for layer_id in range(4):
            self.assertIsNone(kv_only.state_tensor_indices_of_layer(layer_id))


if __name__ == "__main__":
    unittest.main()
