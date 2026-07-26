"""FlatHostMirror (M15 Phase D1): byte-blind pinned-CPU slab mirror.

Pins the transport contract only (no engine wiring): one mirror per
distinct device KV tensor, whole-page row-range copies both directions,
per-tensor load events, and the layer -> tensor-index mapping D2 fences on.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from contextlib import nullcontext
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
# disables slab pairing, so the KV side stays per-layer -- and under the
# flat GDN predicate the state layers' k/v slots are None (M18a T4).
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
            from tokenspeed.runtime.configs.lcm_layouts import qwen_gdn_lcm_fields
            from tokenspeed.runtime.configs.lcm_memory_plan import plan_lcm_fields
            from tokenspeed.runtime.layers.attention.kv_cache.lcm_mha import (
                LcmMHATokenToKVPool,
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
        self.LcmMHATokenToKVPool = LcmMHATokenToKVPool
        fields = qwen_gdn_lcm_fields(
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
        self.lcm_plan = plan_lcm_fields(
            fields,
            logical_block_tokens=4,
            budget_bytes=1280,
            alignment=2,
            max_padding_fraction=0.5,
        )

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
        with mock.patch(_PKG_FLAT_PROBE, return_value=True):
            pool_cls = self.LcmMHATokenToKVPool if with_state else self.MHATokenToKVPool
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
        mirror = self.FlatHostMirror(pool, num_host_pages=8)
        # Flat GDN: state layers carry no KV (k/v slots are None, M18a T4),
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
        # Without state shapes the flat GDN predicate is off: all 4 layers
        # keep KV -> 8 mirrors x page_size 4 x 16 B rows = 512 B.
        base = self.flat_bytes_per_host_page(self._pool(with_state=False))
        self.assertEqual(base, 512)
        # Flat GDN: state layers carry no KV -> 4 KV mirrors (256 B) plus
        # 2 state layers x (conv 2*4 + ssm 2*8) bf16 page rows (2 x 48 B).
        pool = self._pool()
        with_state = self.flat_bytes_per_host_page(pool)
        self.assertEqual(with_state, 4 * 4 * 16 + 96)
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
        # the 4 KV mirrors; conv immediately precedes its ssm.
        self.assertEqual(mirror.state_tensor_indices_of_layer(0), (4, 5))
        self.assertEqual(mirror.state_tensor_indices_of_layer(2), (6, 7))
        self.assertEqual(mirror.ready_tensor_index_of_layer(0), 5)
        self.assertEqual(mirror.ready_tensor_index_of_layer(2), 7)
        self.assertIsNone(mirror.state_tensor_indices_of_layer(1))
        self.assertIsNone(mirror.state_tensor_indices_of_layer(3))
        self.assertEqual(mirror.ready_tensor_index_of_layer(1), 2)
        self.assertEqual(mirror.ready_tensor_index_of_layer(3), 3)
        # Pools without state slabs expose no state indices for any layer.
        kv_only = self.FlatHostMirror(self._pool(with_state=False), num_host_pages=2)
        for layer_id in range(4):
            self.assertIsNone(kv_only.state_tensor_indices_of_layer(layer_id))


class FlatHostMirrorMSAIndexKTest(unittest.TestCase):
    """CPU stub for MiniMax MSA's optional index-K side cache.

    Sparse layers are deliberately non-contiguous and share K/V slabs with
    dense layers. This pins both the index-K layer mapping and the fact that a
    sparse layer must fence after index-K rather than after its shared V slab.
    """

    INDEXED_LAYER_IDS = (0, 2)

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

    def _stub_pool(
        self,
        *,
        with_index_k: bool = True,
        alias_index_k: bool = False,
        alias_kv: bool = True,
        with_state: bool = False,
    ):
        torch = self.torch
        page_size = 4
        size = 8
        rows = size + page_size
        kv_tensor_count = 2 if alias_kv else 4
        k_slabs = [
            torch.zeros((rows, 1, 8), dtype=torch.bfloat16)
            for _ in range(kv_tensor_count)
        ]
        v_slabs = [
            torch.zeros((rows, 1, 8), dtype=torch.bfloat16)
            for _ in range(kv_tensor_count)
        ]
        if alias_kv:
            k_buffer = [k_slabs[0], k_slabs[0], k_slabs[1], k_slabs[1]]
            v_buffer = [v_slabs[0], v_slabs[0], v_slabs[1], v_slabs[1]]
        else:
            k_buffer = k_slabs
            v_buffer = v_slabs
        pool = types.SimpleNamespace(
            page_size=page_size,
            size=size,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
        )
        if with_index_k:
            pool.indexed_layer_ids = frozenset(self.INDEXED_LAYER_IDS)
            # Match MSATokenToKVPool: dict insertion order is sorted layer id.
            pool.index_k_buffer = {
                layer_id: torch.zeros((rows, 8), dtype=torch.bfloat16)
                for layer_id in sorted(pool.indexed_layer_ids)
            }
            if alias_index_k:
                pool.index_k_buffer[2] = pool.index_k_buffer[0]
        if with_state:
            conv = torch.zeros((4, 2), dtype=torch.bfloat16)
            ssm = torch.zeros((4, 4), dtype=torch.bfloat16)
            pool.state_slabs = [(conv, ssm)]

            def get_state_buffers(layer_id):
                if layer_id == 0:
                    return conv, ssm
                raise ValueError(f"layer {layer_id} is not a state layer")

            pool.get_state_buffers = get_state_buffers
        return pool

    def _mirror(self, pool, num_host_pages: int = 3):
        # Keep this stub test CPU-only even when it runs on a CUDA worker.
        with mock.patch(
            "tokenspeed.runtime.cache.flat_host_mirror.torch.cuda.is_available",
            return_value=False,
        ):
            return self.FlatHostMirror(pool, num_host_pages=num_host_pages)

    def _fill_page_byte_pattern(self, tensor, rows: slice, *, seed: int) -> None:
        page_bytes = (
            tensor[rows].view(self.torch.uint8).reshape(rows.stop - rows.start, -1)
        )
        row_ids = self.torch.arange(
            rows.start,
            rows.stop,
            dtype=self.torch.int64,
            device=tensor.device,
        ).unsqueeze(1)
        column_ids = self.torch.arange(
            page_bytes.shape[1],
            dtype=self.torch.int64,
            device=tensor.device,
        ).unsqueeze(0)
        pattern = (row_ids * 17 + column_ids * 29 + seed) % 256
        page_bytes.copy_(pattern.to(self.torch.uint8))

    def test_capacity_and_non_contiguous_sparse_layer_mapping(self):
        pool = self._stub_pool()
        # K/V: 4 mirrors * 4 rows * 16 B = 256 B.
        # index-K: 2 sparse layers * 4 rows * 16 B = 128 B.
        self.assertEqual(self.flat_bytes_per_host_page(pool), 384)

        mirror = self._mirror(pool)
        self.assertEqual(mirror.num_k_tensors, 2)
        self.assertEqual(mirror.num_index_k_tensors, 2)
        self.assertEqual(len(mirror.tensor_pairs), 6)
        self.assertEqual(mirror.row_spans, (4,) * 6)
        self.assertEqual(mirror.bytes_per_host_page(), 384)

        # PINNED order: K*, V*, then index-K* sorted by sparse layer id.
        self.assertIs(mirror.tensor_pairs[4][0], pool.index_k_buffer[0])
        self.assertIs(mirror.tensor_pairs[5][0], pool.index_k_buffer[2])
        self.assertEqual(mirror.index_k_tensor_index_of_layer(0), 4)
        self.assertIsNone(mirror.index_k_tensor_index_of_layer(1))
        self.assertEqual(mirror.index_k_tensor_index_of_layer(2), 5)
        self.assertIsNone(mirror.index_k_tensor_index_of_layer(3))

        # Dense layers fence on V; sparse layers fence on their later index-K.
        self.assertEqual(
            [mirror.ready_tensor_index_of_layer(i) for i in range(4)],
            [4, 2, 5, 3],
        )

    def test_per_layer_kv_layout_matches_m3(self):
        pool = self._stub_pool(alias_kv=False)
        mirror = self._mirror(pool)

        # M3 uses per-layer K/V tensors: 8 K/V mirrors plus two sparse index-K
        # mirrors. Each mirror contributes 4 rows * 16 bytes per host page.
        self.assertEqual(self.flat_bytes_per_host_page(pool), 640)
        self.assertEqual(mirror.num_k_tensors, 4)
        self.assertEqual(mirror.num_index_k_tensors, 2)
        self.assertEqual(len(mirror.tensor_pairs), 10)
        self.assertEqual(mirror.row_spans, (4,) * 10)
        self.assertEqual(mirror.bytes_per_host_page(), 640)
        self.assertEqual(
            [mirror.ready_tensor_index_of_layer(i) for i in range(4)],
            [8, 5, 9, 7],
        )

    def test_index_k_identity_alias_is_mirrored_once(self):
        pool = self._stub_pool(alias_index_k=True)
        mirror = self._mirror(pool)

        self.assertEqual(mirror.num_index_k_tensors, 1)
        self.assertEqual(len(mirror.tensor_pairs), 5)
        self.assertEqual(mirror.bytes_per_host_page(), 320)
        self.assertEqual(mirror.index_k_tensor_index_of_layer(0), 4)
        self.assertEqual(mirror.index_k_tensor_index_of_layer(2), 4)
        self.assertEqual(
            [mirror.ready_tensor_index_of_layer(i) for i in range(4)],
            [4, 2, 4, 3],
        )

    def test_state_offsets_include_index_k_and_state_ready_wins(self):
        pool = self._stub_pool(with_state=True)
        mirror = self._mirror(pool)

        # K*/V* occupy [0, 4), index-K occupies [4, 6), so state starts at
        # 6 rather than the old KV-only base 4. Layer 0 deliberately has both
        # state and index-K: its SSM event is the final readiness fence.
        self.assertEqual(mirror.state_tensor_indices_of_layer(0), (6, 7))
        self.assertEqual(mirror.index_k_tensor_index_of_layer(0), 4)
        self.assertEqual(mirror.ready_tensor_index_of_layer(0), 7)
        self.assertEqual(mirror.ready_tensor_index_of_layer(2), 5)

    def test_store_load_roundtrip_preserves_index_k(self):
        # Exercise M3's real per-layer K/V topology here. The aliased-slab
        # layout remains covered by the layout/dedup tests above.
        pool = self._stub_pool(alias_kv=False)
        mirror = self._mirror(pool)
        store_pairs = [(1, 2), (2, 0)]
        source_device_pages = [device_page for device_page, _ in store_pairs]

        before = []
        for tensor_idx, ((dev, _), span) in enumerate(
            zip(mirror.tensor_pairs, mirror.row_spans)
        ):
            pages = {}
            for device_page in source_device_pages:
                rows = slice(device_page * span, (device_page + 1) * span)
                self._fill_page_byte_pattern(
                    dev,
                    rows,
                    seed=tensor_idx * 41 + device_page * 13 + 7,
                )
                pages[device_page] = dev[rows].clone()
            before.append(pages)

        for tensor_idx, ((_, host), span) in enumerate(
            zip(mirror.tensor_pairs, mirror.row_spans)
        ):
            for _, host_page in store_pairs:
                host_rows = host[host_page * span : (host_page + 1) * span]
                host_rows.view(self.torch.uint8).fill_((0xA5 + tensor_idx) % 256)

        # _copy_pages only needs a stream context; CPU copy_ exercises the
        # same row-span/index ordering without requiring a CUDA worker.
        with mock.patch(
            "tokenspeed.runtime.cache.flat_host_mirror.torch.cuda.stream",
            side_effect=lambda _stream: nullcontext(),
        ):
            mirror.store_pages(store_pairs, stream=None)

            # Validate D2H independently. Otherwise matching addressing bugs in
            # store_pages and load_pages could cancel in the final roundtrip.
            for tensor_idx, ((_, host), span) in enumerate(
                zip(mirror.tensor_pairs, mirror.row_spans)
            ):
                for device_page, host_page in store_pairs:
                    self.assertTrue(
                        self.torch.equal(
                            host[host_page * span : (host_page + 1) * span].view(
                                self.torch.uint8
                            ),
                            before[tensor_idx][device_page].view(self.torch.uint8),
                        ),
                        f"tensor {tensor_idx} host page {host_page} D2H mismatch",
                    )

            # Reload the two host pages into reversed device destinations so
            # H2D addressing is tested independently from the D2H mapping.
            load_pairs = [(2, 2), (1, 0)]
            for (dev, _), span in zip(mirror.tensor_pairs, mirror.row_spans):
                for device_page, _ in load_pairs:
                    dev[device_page * span : (device_page + 1) * span].zero_()
            mirror.load_pages(load_pairs, stream=None)

        source_page_by_host_page = {
            host_page: device_page for device_page, host_page in store_pairs
        }
        for tensor_idx, ((dev, _), span) in enumerate(
            zip(mirror.tensor_pairs, mirror.row_spans)
        ):
            for destination_page, host_page in load_pairs:
                source_page = source_page_by_host_page[host_page]
                self.assertTrue(
                    self.torch.equal(
                        dev[
                            destination_page * span : (destination_page + 1) * span
                        ].view(self.torch.uint8),
                        before[tensor_idx][source_page].view(self.torch.uint8),
                    ),
                    f"tensor {tensor_idx} device page {destination_page} "
                    "not byte-exact",
                )

    def test_pool_without_index_k_keeps_original_layout_and_sizing(self):
        pool = self._stub_pool(with_index_k=False)
        mirror = self._mirror(pool)

        self.assertEqual(self.flat_bytes_per_host_page(pool), 256)
        self.assertEqual(mirror.bytes_per_host_page(), 256)
        self.assertEqual(mirror.num_k_tensors, 2)
        self.assertEqual(mirror.num_index_k_tensors, 0)
        self.assertEqual(len(mirror.tensor_pairs), 4)
        self.assertEqual(mirror.row_spans, (4,) * 4)
        self.assertEqual(
            [mirror.ready_tensor_index_of_layer(i) for i in range(4)],
            [2, 2, 3, 3],
        )
        for layer_id in range(4):
            self.assertIsNone(mirror.index_k_tensor_index_of_layer(layer_id))

    def test_non_mapping_index_cache_keeps_original_layout(self):
        pool = self._stub_pool(with_index_k=False)
        pool.indexed_layer_ids = frozenset(range(4))
        pool.index_k_buffer = [
            self.torch.zeros((12, 8), dtype=self.torch.uint8) for _ in range(4)
        ]
        mirror = self._mirror(pool)

        # DSA-style packed/list side caches are outside the MSA mapping
        # contract and keep the pre-existing flat mirror behavior.
        self.assertEqual(self.flat_bytes_per_host_page(pool), 256)
        self.assertEqual(mirror.num_index_k_tensors, 0)
        self.assertEqual(len(mirror.tensor_pairs), 4)


class FlatHostMirrorNoneKVTest(unittest.TestCase):
    """Flat GDN pools carry None k/v slots on state layers (M18a T4): the
    mirror's identity-dedup walks must skip them and mirror only the real
    slabs. CPU stub pool, no CUDA, no scheduler ext -- state mirroring via
    get_state_buffers is a separate surface and unaffected."""

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
        self.assertEqual(self.flat_bytes_per_host_page(stub), 256)
        mirror = self.FlatHostMirror(stub, num_host_pages=2)
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
