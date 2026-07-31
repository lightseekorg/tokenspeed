"""KV pool paged-cache group publication vs ext build flavor.

Rule under test (kv_cache/mha.py, kv_cache/mla.py): a pool publishes
paged_cache_group_specs iff the tokenspeed_scheduler ext is flat-built
(TOKENSPEED_FLAT_KVCACHE); radix builds publish nothing. Speculative
decoding does not gate publication (flat+spec is supported); backend
capability is checked separately by validate_flat_scheduler_config.

Two pool families live here because they share that one rule:

- MHA (gpt-oss style): one group per distinct (retention, window), so
  hybrid full+sliding models publish two.
- MLA and its DSA subclass (Kimi-K2.5, GLM-5.2): uniform full-history
  attention, so always exactly ONE "full_attention" group. That single
  group is what makes them servable on a flat ext at all -- the flat
  build's req_to_page fallback is a group-0 sample
  (csrc/fsm/forward_states.h GetOccupiedPages), exact only when one group
  exists, which is why the table-blind tokenspeed_mla / DSABackend need no
  per-group flat tables.

The installed ext's real build flavor must not decide these tests, so the
scheduler_ext_flat_kvcache probe is patched per case; the probe's own
default-False behavior is covered separately.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")

GPT_OSS_LAYER_TYPES = (
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
)

_FLAT_PROBE = "tokenspeed.runtime.configs.paged_cache_spec.scheduler_ext_flat_kvcache"


class MHAPoolGroupPublicationTest(unittest.TestCase):
    """Constructs a real (tiny, CPU) MHATokenToKVPool; skips without deps."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.MHATokenToKVPool = MHATokenToKVPool

    def _pool(self, *, flat_ext: bool = True, **overrides):
        kwargs = dict(
            size=32,
            dtype=self.torch.bfloat16,
            head_num=1,
            head_dim=8,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=64,
            page_size=16,
            rank=0,
            enable_alt_stream=False,
        )
        kwargs.update(overrides)
        # The pool resolves the probe lazily at construction time; patching
        # the module attribute pins the ext flavor regardless of the install.
        with mock.patch(_FLAT_PROBE, return_value=flat_ext):
            return self.MHATokenToKVPool(**kwargs)

    def test_plain_no_spec_publishes_single_full_group(self):
        # The flat scheduler allocates pages only through configured groups,
        # so plain models must keep the single full-history group published.
        pool = self._pool()
        self.assertEqual(len(pool.paged_cache_group_specs), 1)
        spec = pool.paged_cache_group_specs[0]
        self.assertEqual(spec.group_id, "full_attention")
        self.assertEqual(spec.retention, "full_history")
        self.assertIn("full_attention", pool.paged_cache_group_page_counts)

    def test_hybrid_no_spec_publishes_two_groups(self):
        # layer_num must match len(layer_types): the M12 slab layout's
        # pairing-completeness assert cross-checks them.
        pool = self._pool(
            layer_types=GPT_OSS_LAYER_TYPES,
            sliding_window_tokens=128,
            layer_num=len(GPT_OSS_LAYER_TYPES),
        )
        self.assertEqual(
            {s.group_id for s in pool.paged_cache_group_specs},
            {"full_attention", "sliding_attention"},
        )
        self.assertEqual(
            set(pool.paged_cache_group_page_counts),
            {"full_attention", "sliding_attention"},
        )

    def test_radix_ext_plain_publishes_no_groups(self):
        # A radix scheduler never fills flat_block_tables, so publication
        # must stay off or graph capture binds buffers that never refresh.
        pool = self._pool(flat_ext=False)
        self.assertEqual(pool.paged_cache_group_specs, ())
        self.assertEqual(pool.paged_cache_group_page_counts, {})

    def test_radix_ext_hybrid_publishes_no_groups(self):
        pool = self._pool(
            flat_ext=False,
            layer_types=GPT_OSS_LAYER_TYPES,
            sliding_window_tokens=128,
        )
        self.assertEqual(pool.paged_cache_group_specs, ())
        self.assertEqual(pool.paged_cache_group_page_counts, {})


class MLAPoolGroupPublicationTest(unittest.TestCase):
    """Constructs a real (tiny, CPU) MLA/DSA pool; skips without deps."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.kv_cache.mla import (
                MLATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.MLATokenToKVPool = MLATokenToKVPool

    def _pool(self, *, flat_ext: bool = True, cls=None, **overrides):
        kwargs = dict(
            size=64,
            dtype=self.torch.bfloat16,
            model_dtype=self.torch.bfloat16,
            quant_method="",
            kv_lora_rank=32,
            qk_rope_head_dim=8,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=128,
            page_size=16,
            rank=0,
            enable_alt_stream=False,
            max_scheduled_tokens=32,
        )
        kwargs.update(overrides)
        with mock.patch(_FLAT_PROBE, return_value=flat_ext):
            return (cls or self.MLATokenToKVPool)(**kwargs)

    def test_flat_ext_publishes_single_full_history_group(self):
        pool = self._pool(flat_ext=True)
        self.assertEqual(len(pool.paged_cache_group_specs), 1)
        spec = pool.paged_cache_group_specs[0]
        self.assertEqual(spec.group_id, "full_attention")
        self.assertEqual(spec.retention, "full_history")
        self.assertEqual(spec.family, "history")
        # rows_per_page must be the pool's page size: the scheduler derives
        # its block_size from it, and a mismatch would mis-address every read.
        self.assertEqual(spec.rows_per_page, 16)
        self.assertEqual(spec.entry_stride_tokens, 1)
        self.assertIsNone(spec.sliding_window_tokens)
        self.assertGreater(pool.paged_cache_group_page_counts["full_attention"], 0)

    def test_radix_ext_publishes_no_groups(self):
        pool = self._pool(flat_ext=False)
        self.assertEqual(pool.paged_cache_group_specs, ())
        self.assertEqual(pool.paged_cache_group_page_counts, {})

    def test_single_group_admits_table_blind_backend(self):
        # The exact guard that rejected every MLA model before the pool
        # started publishing. test_flat_scheduler_config_guard covers this
        # rule with fakes; here it runs against the REAL pool's real specs,
        # which is the integration point that was broken.
        from tokenspeed.runtime.configs.paged_cache_spec import (
            validate_flat_scheduler_config,
        )

        pool = self._pool(flat_ext=True)

        class TableBlindBackend:
            uses_paged_cache_groups = False
            uses_flat_cache_groups = False
            flat_spec_capable = True

        validate_flat_scheduler_config(
            flat_kvcache_ext=True,
            paged_cache_groups=pool.paged_cache_group_specs,
            attn_backend=TableBlindBackend(),
            kv_pool=pool,
            speculative_algorithm="EAGLE3",
        )

    def test_dsa_subclass_inherits_publication(self):
        # GLM-5.2 rides the same single group; its packed index-K buffer is
        # addressed by the same page*page_size+slot locations.
        try:
            from tokenspeed.runtime.layers.attention.kv_cache.dsa import (
                DSATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs DSA kernels: {exc}")
        pool = self._pool(flat_ext=True, cls=DSATokenToKVPool, index_head_dim=128)
        self.assertEqual(len(pool.paged_cache_group_specs), 1)
        self.assertEqual(pool.paged_cache_group_specs[0].group_id, "full_attention")


class MLAConfigFieldOrderTest(unittest.TestCase):
    """DSAConfig must stay constructible after MLAConfig gained a default.

    MLAConfig is a positional dataclass (only BaseAttnConfig is kw_only), so
    max_scheduled_tokens must be keyword-only -- otherwise DSAConfig's
    required index_* fields sit behind a defaulted one and the whole configs
    package raises "non-default argument follows default argument" at import.
    """

    def test_dsa_config_required_fields_follow_kw_only_default(self):
        try:
            import dataclasses

            from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
            from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch: {exc}")
        field_map = {f.name: f for f in dataclasses.fields(MLAConfig)}
        self.assertTrue(field_map["max_scheduled_tokens"].kw_only)
        dsa_fields = {f.name: f for f in dataclasses.fields(DSAConfig)}
        for name in ("index_topk", "index_head_dim", "index_n_heads"):
            self.assertIs(dsa_fields[name].default, dataclasses.MISSING)


class SchedulerExtFlatKvcacheProbeTest(unittest.TestCase):
    """scheduler_ext_flat_kvcache reads the ext's FLAT_KVCACHE build flag with
    a radix-safe default: no package or no attribute -> False."""

    def setUp(self):
        try:
            # paged_cache_spec itself is torch-free, but the configs package
            # __init__ pulls transformers-backed model configs.
            from tokenspeed.runtime.configs.paged_cache_spec import (
                scheduler_ext_flat_kvcache,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs the tokenspeed runtime deps: {exc}")

        self.probe = scheduler_ext_flat_kvcache

    def test_flat_built_ext_reports_true(self):
        fake = types.ModuleType("tokenspeed_scheduler")
        fake.FLAT_KVCACHE = True
        with mock.patch.dict(sys.modules, {"tokenspeed_scheduler": fake}):
            self.assertTrue(self.probe())

    def test_radix_built_ext_reports_false(self):
        fake = types.ModuleType("tokenspeed_scheduler")
        fake.FLAT_KVCACHE = False
        with mock.patch.dict(sys.modules, {"tokenspeed_scheduler": fake}):
            self.assertFalse(self.probe())

    def test_older_ext_without_attribute_defaults_false(self):
        # Pre-FLAT_KVCACHE extensions lack the attribute entirely.
        fake = types.ModuleType("tokenspeed_scheduler")
        with mock.patch.dict(sys.modules, {"tokenspeed_scheduler": fake}):
            self.assertFalse(self.probe())

    def test_missing_package_defaults_false(self):
        # sys.modules[name] = None makes `import name` raise ImportError.
        with mock.patch.dict(sys.modules, {"tokenspeed_scheduler": None}):
            self.assertFalse(self.probe())


if __name__ == "__main__":
    unittest.main()
