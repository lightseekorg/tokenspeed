"""MHA pool paged-cache group publication vs ext build flavor.

Rule under test (kv_cache/mha.py): the pool publishes
paged_cache_group_specs iff the tokenspeed_scheduler ext is flat-built
(TOKENSPEED_FLAT_KVCACHE); radix builds publish nothing. Speculative
decoding does not gate publication (flat+spec is supported); backend
capability is checked separately by validate_flat_scheduler_config.

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

register_cuda_ci(est_time=10, suite="runtime-1gpu")

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
