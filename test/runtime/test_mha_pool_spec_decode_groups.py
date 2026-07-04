"""MHA pool paged-cache group publication vs ext build flavor + spec decode.

Rule under test (kv_cache/mha.py): the pool publishes paged_cache_group_specs
iff the tokenspeed_scheduler ext is flat-built (TOKENSPEED_FLAT_KVCACHE) AND
speculative decoding is off. A radix-built (default) ext never populates
flat_block_tables, so publication must stay off for it — otherwise CUDA-graph
capture binds flat per-group buffers to tables that never arrive. Spec decode
must publish NOTHING either (no flat capture kwarg, overlap schedule
unaffected). With a flat ext and no spec decode, hybrid models keep their two
groups and plain models keep the single full-history group the flat scheduler
build allocates from.

The installed ext's real build flavor must not decide these tests, so the
scheduler_ext_flat_kvcache probe is patched per case; the probe's own
default-False behavior (older ext without the attribute, missing package) is
covered separately.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from types import SimpleNamespace
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
        # The pool resolves the probe lazily at construction time, so patching
        # the module attribute pins the ext flavor regardless of what is
        # actually installed (the container ships a radix-built ext).
        with mock.patch(_FLAT_PROBE, return_value=flat_ext):
            return self.MHATokenToKVPool(**kwargs)

    def test_plain_no_spec_publishes_single_full_group(self):
        # llama/qwen shape: empty layer_types. The flat scheduler build
        # allocates pages only through configured groups, so the single
        # full-history group must stay published.
        pool = self._pool()
        self.assertEqual(len(pool.paged_cache_group_specs), 1)
        spec = pool.paged_cache_group_specs[0]
        self.assertEqual(spec.group_id, "full_attention")
        self.assertEqual(spec.retention, "full_history")
        self.assertIn("full_attention", pool.paged_cache_group_page_counts)

    def test_hybrid_no_spec_publishes_two_groups(self):
        # layer_num must match len(layer_types): the M12 slab layout's
        # pairing-completeness assert cross-checks them (the old fixture's
        # 4-types/2-layers mismatch was silently tolerated pre-M12).
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

    def test_spec_decode_plain_publishes_no_groups(self):
        # eagle3 on llama/qwen: publishing a group would (a) hand the CUDA
        # graph wrapper flat capture group ids — the backend asserts on the
        # flat_cache_group_ids capture kwarg with spec_num_tokens > 1, (b)
        # silently disable overlap scheduling via should_use_overlap_schedule.
        pool = self._pool(speculative_enabled=True)
        self.assertEqual(pool.paged_cache_group_specs, ())
        self.assertEqual(pool.paged_cache_group_page_counts, {})

    def test_spec_decode_hybrid_publishes_no_groups(self):
        pool = self._pool(
            layer_types=GPT_OSS_LAYER_TYPES,
            sliding_window_tokens=128,
            speculative_enabled=True,
        )
        self.assertEqual(pool.paged_cache_group_specs, ())
        self.assertEqual(pool.paged_cache_group_page_counts, {})

    def test_radix_ext_plain_publishes_no_groups(self):
        # Radix-built ext: the scheduler never fills flat_block_tables
        # (MaybeFillFlatBlockTables is a no-op), so publication must stay off
        # or CUDA-graph capture would bind flat buffers that never get fresh
        # tables and the backend replay guard would raise on the first graph
        # decode.
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

    def test_radix_ext_with_spec_decode_publishes_no_groups(self):
        pool = self._pool(flat_ext=False, speculative_enabled=True)
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
        # Pre-FLAT_KVCACHE extensions (e.g. an already-installed radix build)
        # lack the attribute entirely; the getattr default keeps them on the
        # radix-safe no-publication path.
        fake = types.ModuleType("tokenspeed_scheduler")
        with mock.patch.dict(sys.modules, {"tokenspeed_scheduler": fake}):
            self.assertFalse(self.probe())

    def test_missing_package_defaults_false(self):
        # sys.modules[name] = None makes `import name` raise ImportError.
        with mock.patch.dict(sys.modules, {"tokenspeed_scheduler": None}):
            self.assertFalse(self.probe())


class MHAConfigSpecSignalTest(unittest.TestCase):
    """MHAConfig.generate derives speculative_enabled from
    server_args.speculative_algorithm — the same authoritative signal the
    scheduler config (event_loop) and should_use_overlap_schedule read."""

    def setUp(self):
        try:
            from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch: {exc}")
        self.MHAConfig = MHAConfig

    def _server_args(self, speculative_algorithm):
        return SimpleNamespace(
            device="cpu",
            speculative_algorithm=speculative_algorithm,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            attention_backend="mha",
            drafter_attention_backend="mha",
            attn_tp_size=1,
            mapping=SimpleNamespace(attn=SimpleNamespace(tp_size=1, dp_size=1)),
            kv_cache_dtype="bfloat16",
            block_size=16,
            max_num_seqs=8,
            data_parallel_size=1,
            max_cudagraph_capture_size=4,
            kv_cache_quant_method="",
            chunked_prefill_size=512,
            enable_kvstore=False,
            disaggregation_mode="null",
        )

    def _model_config(self):
        import torch

        return SimpleNamespace(
            hf_config=SimpleNamespace(layer_types=None, sliding_window=None),
            context_len=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            dtype=torch.bfloat16,
        )

    def test_spec_algorithm_sets_speculative_enabled(self):
        cfg = self.MHAConfig.generate(
            self._server_args(speculative_algorithm="EAGLE3"),
            self._model_config(),
        )
        self.assertTrue(cfg.speculative_enabled)

    def test_no_spec_algorithm_leaves_speculative_disabled(self):
        cfg = self.MHAConfig.generate(
            self._server_args(speculative_algorithm=None),
            self._model_config(),
        )
        self.assertFalse(cfg.speculative_enabled)


if __name__ == "__main__":
    unittest.main()
