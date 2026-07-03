"""MHA pool paged-cache group publication vs speculative decoding.

Rule under test (kv_cache/mha.py): the pool publishes paged_cache_group_specs
iff speculative decoding is off. Spec decode must publish NOTHING (no flat
capture kwarg, overlap schedule unaffected); without spec decode hybrid models
keep their two groups and plain models keep the single full-history group the
flat scheduler build allocates from.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

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

    def _pool(self, **overrides):
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
        pool = self._pool(
            layer_types=GPT_OSS_LAYER_TYPES,
            sliding_window_tokens=128,
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
        # graph wrapper a flat_block_tables kwarg the backend asserts on with
        # spec_num_tokens > 1, (b) silently disable overlap scheduling via
        # should_use_overlap_schedule.
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
