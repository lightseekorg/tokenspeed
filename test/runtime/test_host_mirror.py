"""HostMirror (M15 Phase D1): byte-blind pinned-CPU slab mirror.

Pins the transport contract only (no engine wiring): one mirror per
distinct device KV tensor, whole-page row-range copies both directions,
per-tensor load events, and the layer -> tensor-index mapping D2 fences on.
"""

from __future__ import annotations

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")

LAYER_TYPES = ("sliding_attention", "full_attention") * 2

# GDN hybrid: layers 0/2 are state layers (pairs 0/1); linear_attention
# disables slab pairing, so the KV side stays per-layer -- and under the
# grouped GDN predicate the state layers' k/v slots are None (M18a T4).
GDN_LAYER_TYPES = ("linear_attention", "full_attention") * 2


class HostMirrorTest(unittest.TestCase):
    """Real (tiny) MHATokenToKVPool on GPU."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.cache.host_mirror import (
                HostMirror,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        self.torch = torch
        self.HostMirror = HostMirror
        self.MHATokenToKVPool = MHATokenToKVPool

    def _pool(self):
        from cache_pool_test_utils import make_mha_memory_plan

        kwargs = {
            "size": 32,
            "dtype": self.torch.bfloat16,
            "head_num": 1,
            "head_dim": 8,
            "layer_num": 4,
            "device": "cuda",
            "enable_memory_saver": False,
            "max_batch_size": 2,
            "max_context_len": 64,
            "page_size": 4,
            "rank": 0,
            "layer_types": LAYER_TYPES,
            "sliding_window_tokens": 128,
        }
        from cache_pool_test_utils import make_layer_group_ids

        kwargs["memory_plan"] = make_mha_memory_plan(
            size=kwargs["size"],
            page_size=kwargs["page_size"],
            layer_num=kwargs["layer_num"],
            kv_heads=kwargs["head_num"],
            head_dim=kwargs["head_dim"],
            dtype=kwargs["dtype"],
            layer_types=kwargs["layer_types"],
            sliding_window_tokens=kwargs["sliding_window_tokens"],
        )
        kwargs["layer_group_ids"] = make_layer_group_ids(
            layer_num=kwargs["layer_num"],
            layer_types=kwargs["layer_types"],
            sliding_window_tokens=kwargs["sliding_window_tokens"],
        )
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

    def test_roundtrip(self):
        pool = self._pool()
        mirror = self.HostMirror(pool, num_host_pages=8)
        # The canonical hybrid layout deduplicates paired layers into two K
        # and two V slabs.
        self.assertEqual(len(mirror.tensor_pairs), 4)
        self._roundtrip_assert(mirror, [(1, 5), (2, 6), (3, 7)])
        # 4 mirrors x page_size 4 x row 1*8 bf16 (16 B) = 256 B per page.
        self.assertEqual(mirror.bytes_per_host_page(), 4 * 4 * 16)

    def test_interleaved_groups_roundtrip(self):
        # Pages owned by different groups: byte-blind copies need no
        # group awareness (id-exclusivity keeps rows disjoint).
        pool = self._pool()
        mirror = self.HostMirror(pool, num_host_pages=4)
        self._roundtrip_assert(mirror, [(2, 0), (3, 1)])

    def test_events_and_layer_mapping(self):
        torch = self.torch
        pool = self._pool()
        mirror = self.HostMirror(pool, num_host_pages=8)
        self._fill_device_pages(mirror, [1])

        stream = torch.cuda.Stream()
        events = mirror.load_pages_with_events([(1, 5)], stream)
        self.assertEqual(len(events), len(mirror.tensor_pairs))
        stream.synchronize()
        self.assertTrue(all(event.query() for event in events))

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


class HostMirrorStateSlabTest(unittest.TestCase):
    """State slabs join the mirrored set: tensor_pairs order is K*, V*,
    then (conv, ssm) flattened in slab order; state mirrors use 1-row
    PAGE spans (state slabs are page-indexed) while KV mirrors span
    page_size token rows."""

    CONV_SHAPE = (2, 4)  # 16 B/row bf16
    SSM_SHAPE = (2, 8)  # 32 B/row bf16

    def setUp(self):
        try:
            import torch
            from cache_pool_test_utils import plan_fields

            from tokenspeed.runtime.cache.host_mirror import (
                HostMirror,
                bytes_per_host_page,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
                HybridMHATokenToKVPool,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen35 import (
                qwen_gdn_cache_fields,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        self.torch = torch
        self.HostMirror = HostMirror
        self.bytes_per_host_page = bytes_per_host_page
        self.MHATokenToKVPool = MHATokenToKVPool
        self.HybridMHATokenToKVPool = HybridMHATokenToKVPool
        fields = qwen_gdn_cache_fields(
            layer_types=GDN_LAYER_TYPES,
            layer_group_ids=(
                "linear_attention_0",
                "full_attention",
                "linear_attention_0",
                "full_attention",
            ),
            logical_block_tokens=4,
            kv_shape=(4, 1, 8),
            kv_element_size=2,
            conv_shape=self.CONV_SHAPE,
            conv_element_size=2,
            ssm_shape=self.SSM_SHAPE,
            ssm_element_size=2,
        )
        self.lcm_plan = plan_fields(
            fields,
            logical_block_tokens=4,
            budget_bytes=1280,
            alignment=2,
            max_padding_fraction=0.5,
        )

    def _pool(self, *, with_state: bool = True):
        kwargs = {
            "size": 32,
            "dtype": self.torch.bfloat16,
            "head_num": 1,
            "head_dim": 8,
            "layer_num": 4,
            "device": "cuda",
            "enable_memory_saver": False,
            "max_batch_size": 2,
            "max_context_len": 64,
            "page_size": 4,
            "rank": 0,
            "layer_types": GDN_LAYER_TYPES,
            "sliding_window_tokens": None,
        }
        if with_state:
            kwargs.update(
                state_field_dtypes={
                    f"layer.{layer_id}.{field}": self.torch.bfloat16
                    for layer_id in (0, 2)
                    for field in ("conv", "ssm")
                },
                memory_plan=self.lcm_plan,
                layer_group_ids=(
                    "linear_attention_0",
                    "full_attention",
                    "linear_attention_0",
                    "full_attention",
                ),
            )
        pool_cls = self.HybridMHATokenToKVPool if with_state else self.MHATokenToKVPool
        return pool_cls(**kwargs)

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
            {d: dev[d * span : (d + 1) * span].cpu().clone() for d in device_pages}
            for (dev, _), span in zip(mirror.tensor_pairs, mirror.row_spans)
        ]

    def test_state_tensors_follow_kv_in_slab_order(self):
        pool = self._pool()
        mirror = self.HostMirror(pool, num_host_pages=8)
        # Grouped GDN: state layers carry no KV (k/v slots are None, M18a T4),
        # so only the 2 attention layers mirror KV (2 K + 2 V), then
        # conv0, ssm0, conv1, ssm1 -- PINNED order: K*, V*, state tensors
        # flattened in slab order.
        self.assertEqual(mirror.num_k_tensors, 2)
        self.assertEqual(len(mirror.tensor_pairs), 8)
        self.assertEqual(len(pool.state_slabs), 2)
        for n, (conv, ssm) in enumerate(pool.state_slabs):
            self.assertIs(mirror.tensor_pairs[4 + 2 * n][0], conv)
            self.assertIs(mirror.tensor_pairs[4 + 2 * n + 1][0], ssm)
        # Per-pair row spans: page_size token rows for KV, 1 page row for
        # state (state slabs are page-indexed).
        self.assertEqual(mirror.row_spans, (4,) * 4 + (1,) * 4)
        for (dev, host), span in zip(mirror.tensor_pairs, mirror.row_spans):
            if span == 1:
                self.assertEqual(host.shape, (8, *dev.shape[1:]))
            else:
                self.assertEqual(host.shape, (8 * 4, *dev.shape[1:]))

    def test_bytes_per_host_page_includes_state_rows(self):
        # Without state shapes the grouped GDN predicate is off: all 4 layers
        # keep KV -> 8 mirrors x page_size 4 x 16 B rows = 512 B.
        base = self.bytes_per_host_page(self._pool(with_state=False))
        self.assertEqual(base, 512)
        # Grouped GDN: state layers carry no KV -> 4 KV mirrors (256 B) plus
        # 2 state layers x (conv 2*4 + ssm 2*8) bf16 page rows (2 x 48 B).
        pool = self._pool()
        with_state = self.bytes_per_host_page(pool)
        self.assertEqual(with_state, 4 * 4 * 16 + 96)
        mirror = self.HostMirror(pool, num_host_pages=2)
        self.assertEqual(mirror.bytes_per_host_page(), with_state)

    def test_state_roundtrip(self):
        torch = self.torch
        pool = self._pool()
        mirror = self.HostMirror(pool, num_host_pages=8)
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
        mirror = self.HostMirror(pool, num_host_pages=2)
        # State layers 0/2 bind slab pairs 0/1 -> flattened indices after
        # the 4 KV mirrors; conv immediately precedes its ssm.
        self.assertEqual(mirror.state_tensor_indices_of_layer(0), (4, 5))
        self.assertEqual(mirror.state_tensor_indices_of_layer(2), (6, 7))
        self.assertIsNone(mirror.state_tensor_indices_of_layer(1))
        self.assertIsNone(mirror.state_tensor_indices_of_layer(3))
        # Pools without state slabs expose no state indices for any layer.
        kv_only = self.HostMirror(self._pool(with_state=False), num_host_pages=2)
        for layer_id in range(4):
            self.assertIsNone(kv_only.state_tensor_indices_of_layer(layer_id))


class HostMirrorNoneKVTest(unittest.TestCase):
    """Grouped GDN pools carry None k/v slots on state layers (M18a T4): the
    mirror's identity-dedup walks must skip them and mirror only the real
    slabs. CPU stub pool, no CUDA, no scheduler ext -- state mirroring via
    get_state_buffers is a separate surface and unaffected."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.cache.host_mirror import (
                HostMirror,
                bytes_per_host_page,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch: {exc}")
        self.torch = torch
        self.HostMirror = HostMirror
        self.bytes_per_host_page = bytes_per_host_page

    def _stub_pool(self):
        import types

        torch = self.torch
        rows = 8
        kv = [torch.zeros((rows, 1, 8), dtype=torch.bfloat16) for _ in range(4)]
        return types.SimpleNamespace(
            page_size=4,
            k_buffer=[None, kv[0], None, kv[1]],
            v_buffer=[None, kv[2], None, kv[3]],
        )

    def test_mirror_skips_none_kv_entries(self):
        stub = self._stub_pool()
        # 4 real mirrors x page_size 4 x 16 B rows = 256 B per host page.
        self.assertEqual(self.bytes_per_host_page(stub), 256)
        mirror = self.HostMirror(stub, num_host_pages=2)
        self.assertEqual(mirror.num_k_tensors, 2)
        self.assertEqual(len(mirror.tensor_pairs), 4)
        self.assertIs(mirror.tensor_pairs[0][0], stub.k_buffer[1])
        self.assertIs(mirror.tensor_pairs[1][0], stub.k_buffer[3])
        # KV layers keep their tensor-index mapping; state layers have no
        # KV mirror and must fail loud if D2 fencing ever asks for one.
        self.assertEqual(mirror.tensor_index_of_layer(1), 0)
        self.assertEqual(mirror.tensor_index_of_layer(3), 1)
        with self.assertRaisesRegex(ValueError, r"state layer"):
            mirror.tensor_index_of_layer(0)


if __name__ == "__main__":
    unittest.main()
