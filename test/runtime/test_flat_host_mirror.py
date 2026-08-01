"""FlatHostMirror (M15 Phase D1): byte-blind pinned-CPU slab mirror.

Pins the transport contract only (no engine wiring): one mirror per
distinct device tensor the pool declares in ``host_mirror_families()``,
whole-page row-range copies both directions, per-tensor load events, and the
layer -> fence-tensor mapping D2 fences on.

Covers every declared layout: MHA K/V (slab and legacy), GDN state slabs,
MLA's fused latent (plus its per-token-head quantized triple) and DSA's
latent + packed index-K.
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

    def test_slab_roundtrip(self):
        pool = self._pool(flat_ext=True)
        mirror = self.FlatHostMirror(pool, num_host_pages=8)
        # 4 layers dedup to 2 K + 2 V slabs.
        self.assertEqual(len(mirror.tensor_pairs), 4)
        assert_page_roundtrip(self, mirror, [(1, 5), (2, 6), (3, 7)])
        # 4 mirrors x page_size 4 x row 1*8 bf16 (16 B) = 256 B per page.
        self.assertEqual(mirror.bytes_per_host_page(), 4 * 4 * 16)

    def test_interleaved_groups_roundtrip(self):
        # Pages owned by different groups: byte-blind copies need no
        # group awareness (id-exclusivity keeps rows disjoint).
        pool = self._pool(flat_ext=True)
        mirror = self.FlatHostMirror(pool, num_host_pages=4)
        assert_page_roundtrip(self, mirror, [(2, 0), (3, 1)])

    def test_legacy_roundtrip(self):
        # Legacy layout: all 4+4 per-layer mirrors carry data; copying
        # rows dead for a page's owner group is harmless (byte-exact).
        pool = self._pool(flat_ext=False)
        mirror = self.FlatHostMirror(pool, num_host_pages=8)
        self.assertEqual(len(mirror.tensor_pairs), 8)
        assert_page_roundtrip(self, mirror, [(1, 3), (2, 4)])

    def test_events_and_layer_mapping(self):
        torch = self.torch
        pool = self._pool(flat_ext=True)
        mirror = self.FlatHostMirror(pool, num_host_pages=8)

        stream = torch.cuda.Stream()
        events = mirror.load_pages_with_events([(1, 5)], stream)
        self.assertEqual(len(events), len(mirror.tensor_pairs))
        stream.synchronize()
        self.assertTrue(all(event.query() for event in events))

        # A layer fences on the LAST mirror carrying its bytes: its V slab
        # (mirror order K0, K1, V0, V1). Paired slab layers share it.
        self.assertEqual(mirror.layer_num, 4)
        for layer_id in range(4):
            idx = mirror.fence_tensor_index_of_layer(layer_id)
            self.assertIs(mirror.tensor_pairs[idx][0], pool.v_buffer[layer_id])
        self.assertEqual(
            mirror.fence_tensor_index_of_layer(0),
            mirror.fence_tensor_index_of_layer(1),
        )
        self.assertEqual(
            mirror.fence_tensor_index_of_layer(2),
            mirror.fence_tensor_index_of_layer(3),
        )
        self.assertNotEqual(
            mirror.fence_tensor_index_of_layer(0),
            mirror.fence_tensor_index_of_layer(2),
        )

        # Legacy: every layer fences on its own V mirror (K0..K3, V0..V3).
        legacy = self.FlatHostMirror(self._pool(flat_ext=False), num_host_pages=2)
        self.assertEqual(
            {legacy.fence_tensor_index_of_layer(i) for i in range(4)}, {4, 5, 6, 7}
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

    def test_state_tensors_follow_kv_in_slab_order(self):
        pool = self._pool()
        mirror = self.FlatHostMirror(pool, num_host_pages=8)
        # Flat GDN: state layers carry no KV (k/v slots are None, M18a T4),
        # so only the 2 attention layers mirror KV (2 K + 2 V), then
        # conv0, ssm0, conv1, ssm1 -- declared family order: K*, V*, state
        # tensors flattened in slab order.
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
        # State slabs ride 1-row page spans while KV rides page_size rows;
        # the shared helper is span-aware, so one call covers both.
        mirror = self.FlatHostMirror(self._pool(), num_host_pages=8)
        assert_page_roundtrip(self, mirror, [(1, 5), (2, 6), (3, 7)])

    def test_state_layers_fence_on_their_ssm_slab(self):
        pool = self._pool()
        mirror = self.FlatHostMirror(pool, num_host_pages=2)
        # State layers 0/2 bind slab pairs 0/1, flattened after the 4 KV
        # mirrors as conv0, ssm0, conv1, ssm1; the pair's LAST tensor (ssm)
        # is the fence. Attention layers 1/3 keep their V mirror.
        self.assertEqual(mirror.fence_tensor_index_of_layer(0), 5)
        self.assertEqual(mirror.fence_tensor_index_of_layer(2), 7)
        self.assertEqual(mirror.fence_tensor_index_of_layer(1), 2)
        self.assertEqual(mirror.fence_tensor_index_of_layer(3), 3)
        # Without state slabs every layer keeps KV, so all fence on V.
        kv_only = self.FlatHostMirror(self._pool(with_state=False), num_host_pages=2)
        self.assertEqual(
            [kv_only.fence_tensor_index_of_layer(i) for i in range(4)], [4, 5, 6, 7]
        )


class FlatHostMirrorNoneKVTest(unittest.TestCase):
    """Flat GDN pools carry None k/v slots on state layers (M18a T4): the
    base pool's family derivation must skip them and mirror only the real
    slabs. CPU stub pool, no CUDA, no scheduler ext -- the stub borrows the
    real default derivation, so nothing here re-implements it."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.cache.flat_host_mirror import (
                FlatHostMirror,
                flat_bytes_per_host_page,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.base import (
                BaseTokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch: {exc}")
        self.torch = torch
        self.FlatHostMirror = FlatHostMirror
        self.flat_bytes_per_host_page = flat_bytes_per_host_page
        self.BaseTokenToKVPool = BaseTokenToKVPool

    def _stub_pool(self, *, with_state: bool):
        torch = self.torch
        kv = [torch.zeros((8, 1, 8), dtype=torch.bfloat16) for _ in range(4)]
        slabs = [
            (
                torch.zeros((2, 2, 4), dtype=torch.bfloat16),  # conv, 16 B/row
                torch.zeros((2, 2, 8), dtype=torch.bfloat16),  # ssm, 32 B/row
            )
            for _ in range(2)
        ]
        base = self.BaseTokenToKVPool

        class _StubPool:
            # The real default derivation, on a pool too small to construct.
            host_mirror_families = base.host_mirror_families
            page_size = 4
            k_buffer = [None, kv[0], None, kv[1]]
            v_buffer = [None, kv[2], None, kv[3]]
            state_slabs = slabs if with_state else []

            def get_state_buffers(self, layer_id):
                if layer_id not in (0, 2):
                    raise ValueError(f"layer {layer_id} is not a state layer")
                return slabs[layer_id // 2]

        return _StubPool()

    def test_mirror_skips_none_kv_entries(self):
        stub = self._stub_pool(with_state=True)
        # 4 real KV mirrors x page_size 4 x 16 B rows (256 B), plus 2 state
        # pairs x (conv 16 B + ssm 32 B) page rows.
        self.assertEqual(self.flat_bytes_per_host_page(stub), 256 + 96)
        mirror = self.FlatHostMirror(stub, num_host_pages=2)
        self.assertEqual(len(mirror.tensor_pairs), 8)
        self.assertIs(mirror.tensor_pairs[0][0], stub.k_buffer[1])
        self.assertIs(mirror.tensor_pairs[1][0], stub.k_buffer[3])
        # KV layers fence on their V mirror, state layers on their ssm slab.
        self.assertEqual(
            [mirror.fence_tensor_index_of_layer(i) for i in range(4)], [5, 2, 7, 3]
        )

    def test_layer_in_no_family_is_refused(self):
        # The same None slots WITHOUT the state declaration: layers 0/2 would
        # be mirrored by nothing, so a loadback could not fence them.
        stub = self._stub_pool(with_state=False)
        with self.assertRaisesRegex(ValueError, r"in no declared family"):
            self.FlatHostMirror(stub, num_host_pages=2)


class FlatHostMirrorMLATest(unittest.TestCase):
    """MLA (K2.5) declares ONE fused-latent family: no independent V tensor,
    so a layer fences on its own latent mirror. The per-token-head quantized
    variant splits that entry into a (k_lora, k_scale, k_rope) triple, which
    is the same family with three tensors per layer.

    Also covers the speculative draft pool, which is built from this same
    pool constructor (a second MLA pool with fewer layers)."""

    LAYER_NUM = 3

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.cache.flat_host_mirror import (
                FlatHostMirror,
                flat_bytes_per_host_page,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.mla import (
                MLATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        self.torch = torch
        self.FlatHostMirror = FlatHostMirror
        self.flat_bytes_per_host_page = flat_bytes_per_host_page
        self.MLATokenToKVPool = MLATokenToKVPool

    def _pool(self, **overrides):
        kwargs = dict(
            size=32,
            dtype=self.torch.bfloat16,
            model_dtype=self.torch.bfloat16,
            quant_method="",
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            layer_num=self.LAYER_NUM,
            device="cuda",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=64,
            page_size=4,
            rank=0,
            enable_alt_stream=False,
        )
        kwargs.update(overrides)
        with mock.patch(_PKG_FLAT_PROBE, return_value=True):
            return self.MLATokenToKVPool(**kwargs)

    def test_fused_latent_layout(self):
        pool = self._pool()
        mirror = self.FlatHostMirror(pool, num_host_pages=8)
        # One mirror per layer, spanning page_size token rows.
        self.assertEqual(len(mirror.tensor_pairs), self.LAYER_NUM)
        self.assertEqual(mirror.layer_num, self.LAYER_NUM)
        self.assertEqual(mirror.row_spans, (4,) * self.LAYER_NUM)
        for layer_id in range(self.LAYER_NUM):
            self.assertIs(mirror.tensor_pairs[layer_id][0], pool.kv_buffer[layer_id])
            self.assertEqual(mirror.fence_tensor_index_of_layer(layer_id), layer_id)
        # 3 mirrors x page_size 4 x row 1*(16+8) bf16 (48 B) = 576 B.
        self.assertEqual(mirror.bytes_per_host_page(), 3 * 4 * 48)
        self.assertEqual(self.flat_bytes_per_host_page(pool), 3 * 4 * 48)

    def test_per_token_head_quantization_mirrors_the_triple(self):
        pool = self._pool(quant_method="per_token_head")
        mirror = self.FlatHostMirror(pool, num_host_pages=8)
        # (k_lora, k_scale, k_rope) per layer, in that order.
        self.assertEqual(len(mirror.tensor_pairs), 3 * self.LAYER_NUM)
        for layer_id in range(self.LAYER_NUM):
            for n, buf in enumerate(pool.kv_buffer[layer_id]):
                self.assertIs(mirror.tensor_pairs[3 * layer_id + n][0], buf)
            # Fence on the layer's LAST tensor (k_rope).
            self.assertEqual(
                mirror.fence_tensor_index_of_layer(layer_id), 3 * layer_id + 2
            )
        assert_page_roundtrip(self, mirror, [(1, 5), (2, 6)])

    # Draft pool: the drafter writes at the TARGET's slot ids, so both pools
    # ride one host page. Draft families lead, keeping the target's fences.
    DRAFT_LAYER_NUM = 1

    def test_draft_tensors_lead_the_target_and_keep_its_fences(self):
        target = self._pool()
        draft = self._pool(layer_num=self.DRAFT_LAYER_NUM)
        mirror = self.FlatHostMirror(target, num_host_pages=8, draft_kv_pool=draft)
        self.assertEqual(
            len(mirror.tensor_pairs), self.LAYER_NUM + self.DRAFT_LAYER_NUM
        )
        # Draft first, then the target's layers.
        self.assertIs(mirror.tensor_pairs[0][0], draft.kv_buffer[0])
        for layer_id in range(self.LAYER_NUM):
            self.assertIs(
                mirror.tensor_pairs[self.DRAFT_LAYER_NUM + layer_id][0],
                target.kv_buffer[layer_id],
            )
        # Layer count stays the TARGET's, and every layer still fences on its
        # own target mirror -- the draft copies precede them all.
        self.assertEqual(mirror.layer_num, self.LAYER_NUM)
        for layer_id in range(self.LAYER_NUM):
            self.assertEqual(
                mirror.fence_tensor_index_of_layer(layer_id),
                self.DRAFT_LAYER_NUM + layer_id,
            )
        # Sizing counts the draft pool too (byte-exact roundtrip with a draft
        # pool is covered end-to-end in test_flat_host_executor).
        both = self.flat_bytes_per_host_page(target, draft)
        self.assertEqual(both, (self.LAYER_NUM + self.DRAFT_LAYER_NUM) * 4 * 48)
        self.assertEqual(both, self.flat_bytes_per_host_page(target) + 4 * 48)
        self.assertEqual(mirror.bytes_per_host_page(), both)

    def test_unmirrorable_or_mismatched_draft_pool_is_refused(self):
        import types

        target = self._pool()
        # A draft pool the mirror cannot describe must fail loud rather than
        # leave draft KV silently unmirrored.
        blind = types.SimpleNamespace(
            host_mirror_families=lambda: [], page_size=4, size=32
        )
        with self.assertRaisesRegex(ValueError, r"declares no tensor families"):
            self.FlatHostMirror(target, num_host_pages=2, draft_kv_pool=blind)
        # Page geometry must line up: page ids index both pools.
        with self.assertRaisesRegex(ValueError, r"page geometry"):
            self.FlatHostMirror(
                target,
                num_host_pages=2,
                draft_kv_pool=self._pool(layer_num=self.DRAFT_LAYER_NUM, page_size=8),
            )
        # A draft pool deeper than the target breaks the layer padding.
        with self.assertRaisesRegex(ValueError, r"more layers than the target"):
            self.FlatHostMirror(
                target,
                num_host_pages=2,
                draft_kv_pool=self._pool(layer_num=self.LAYER_NUM + 1),
            )


class FlatHostMirrorDSATest(unittest.TestCase):
    """DSA (GLM-5.2) adds the packed index-K family after MLA's latent.

    Index-K is block-split WITHIN a page, so the byte-exact roundtrip below
    is the load-bearing check: it only holds because the mirror copies whole
    pages (a page's bytes are exactly its page_size rows)."""

    LAYER_NUM = 2
    INDEX_HEAD_DIM = 128

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.cache.flat_host_mirror import (
                FlatHostMirror,
                flat_bytes_per_host_page,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.dsa import (
                DSATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + DSA kernels: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        self.torch = torch
        self.FlatHostMirror = FlatHostMirror
        self.flat_bytes_per_host_page = flat_bytes_per_host_page
        self.DSATokenToKVPool = DSATokenToKVPool

    def _pool(self):
        kwargs = dict(
            size=32,
            dtype=self.torch.bfloat16,
            model_dtype=self.torch.bfloat16,
            quant_method="",
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            layer_num=self.LAYER_NUM,
            device="cuda",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=64,
            page_size=4,
            rank=0,
            enable_alt_stream=False,
            index_head_dim=self.INDEX_HEAD_DIM,
        )
        with mock.patch(_PKG_FLAT_PROBE, return_value=True):
            return self.DSATokenToKVPool(**kwargs)

    def test_index_k_family_follows_the_latent_family(self):
        pool = self._pool()
        mirror = self.FlatHostMirror(pool, num_host_pages=8)
        self.assertEqual(len(mirror.tensor_pairs), 2 * self.LAYER_NUM)
        for layer_id in range(self.LAYER_NUM):
            self.assertIs(mirror.tensor_pairs[layer_id][0], pool.kv_buffer[layer_id])
            self.assertIs(
                mirror.tensor_pairs[self.LAYER_NUM + layer_id][0],
                pool.index_k_buffer[layer_id],
            )
            # Index-K lands last, so it is the layer's fence.
            self.assertEqual(
                mirror.fence_tensor_index_of_layer(layer_id),
                self.LAYER_NUM + layer_id,
            )
        # Latent rows (48 B) plus index-K rows (128 FP8 + 1 group * 4 B).
        row_bytes = 48 + (self.INDEX_HEAD_DIM + 4)
        self.assertEqual(mirror.bytes_per_host_page(), self.LAYER_NUM * 4 * row_bytes)
        self.assertEqual(
            self.flat_bytes_per_host_page(pool), mirror.bytes_per_host_page()
        )

    def test_block_split_index_k_roundtrips_byte_exact(self):
        mirror = self.FlatHostMirror(self._pool(), num_host_pages=8)
        assert_page_roundtrip(self, mirror, [(1, 5), (2, 6), (3, 7)])


def assert_page_roundtrip(case, mirror, pairs):
    """store_pages -> zero the device pages -> load_pages is byte-exact.

    Fills every mirrored device tensor with RANDOM bytes first, so an
    intra-page reordering (e.g. mishandling DSA's block-split index-K)
    fails here, not just a page mixup.

    Args:
        case: The TestCase raising the failure.
        mirror: A built :class:`FlatHostMirror`.
        pairs: (device_page, host_page) pairs to round-trip.
    """
    torch = case.torch
    device_pages = [d for d, _ in pairs]
    for dev, _ in mirror.tensor_pairs:
        dev.view(torch.uint8).random_(0, 256)
    torch.cuda.synchronize()

    def snapshot():
        return [
            {d: dev[d * span : (d + 1) * span].cpu().clone() for d in device_pages}
            for (dev, _), span in zip(mirror.tensor_pairs, mirror.row_spans)
        ]

    before = snapshot()
    stream = torch.cuda.Stream()
    mirror.store_pages(pairs, stream)
    stream.synchronize()
    for (dev, _), span in zip(mirror.tensor_pairs, mirror.row_spans):
        for d in device_pages:
            dev[d * span : (d + 1) * span].zero_()
    torch.cuda.synchronize()
    mirror.load_pages(pairs, stream)
    stream.synchronize()

    after = snapshot()
    for tensor_idx in range(len(mirror.tensor_pairs)):
        for d in device_pages:
            case.assertTrue(
                torch.equal(
                    before[tensor_idx][d].view(torch.uint8),
                    after[tensor_idx][d].view(torch.uint8),
                ),
                f"tensor {tensor_idx} device page {d} not byte-exact",
            )


if __name__ == "__main__":
    unittest.main()
