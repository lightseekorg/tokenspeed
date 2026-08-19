"""MHA pools always publish their scheduler cache groups."""

from __future__ import annotations

import os
import sys
import unittest

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
        from cache_pool_test_utils import make_mha_memory_plan

        kwargs = {
            "size": 32,
            "dtype": self.torch.bfloat16,
            "head_num": 1,
            "head_dim": 8,
            "layer_num": 2,
            "device": "cpu",
            "enable_memory_saver": False,
            "prefix_granularity": 16,
            "rank": 0,
        }
        kwargs.update(overrides)
        from cache_pool_test_utils import make_layer_group_ids

        plan = make_mha_memory_plan(
            size=kwargs["size"],
            prefix_granularity=kwargs["prefix_granularity"],
            layer_num=kwargs["layer_num"],
            kv_heads=kwargs["head_num"],
            head_dim=kwargs["head_dim"],
            dtype=kwargs["dtype"],
            layer_types=kwargs.get("layer_types", ()),
            sliding_window_tokens=kwargs.get("sliding_window_tokens"),
        )
        kwargs.setdefault(
            "layer_group_ids",
            make_layer_group_ids(
                layer_num=kwargs["layer_num"],
                layer_types=kwargs.get("layer_types", ()),
                sliding_window_tokens=kwargs.get("sliding_window_tokens"),
            ),
        )
        from cache_pool_test_utils import specs_for_layers

        kwargs.setdefault(
            "cache_group_specs",
            specs_for_layers(
                layer_types=kwargs.get("layer_types", ()),
                group_ids=kwargs["layer_group_ids"],
                sliding_window_tokens=kwargs.get("sliding_window_tokens"),
                prefix_granularity=kwargs["prefix_granularity"],
            ),
        )
        kwargs.pop("sliding_window_tokens", None)
        device = kwargs.pop("device")
        for owned_by_arena in ("size", "prefix_granularity", "enable_memory_saver"):
            kwargs.pop(owned_by_arena, None)
        from cache_pool_test_utils import make_pool

        _, pool = make_pool(self.MHATokenToKVPool, plan, device=device, **kwargs)
        return pool

    def test_plain_no_spec_publishes_single_full_group(self):
        # The scheduler allocates pages only through configured groups, so
        # plain models keep one full-history group published.
        pool = self._pool()
        self.assertEqual(len(pool.arena.cache_group_specs), 1)
        spec = pool.arena.cache_group_specs[0]
        self.assertEqual(spec.group_id, "full_attention")
        self.assertEqual(spec.retention, "full_history")
        self.assertIn("full_attention", pool.arena.cache_group_page_counts)
        self.assertIsNotNone(pool.arena.buffer)
        self.assertEqual(
            pool.k_buffer[0].untyped_storage().data_ptr(),
            pool.arena.buffer.untyped_storage().data_ptr(),
        )

    def test_hybrid_no_spec_publishes_two_groups(self):
        # layer_num must match len(layer_types): the M12 slab layout's
        # pairing-completeness assert cross-checks them.
        pool = self._pool(
            layer_types=GPT_OSS_LAYER_TYPES,
            sliding_window_tokens=128,
            layer_num=len(GPT_OSS_LAYER_TYPES),
        )
        self.assertEqual(
            {s.group_id for s in pool.arena.cache_group_specs},
            {"full_attention", "sliding_attention"},
        )
        self.assertEqual(
            set(pool.arena.cache_group_page_counts),
            {"full_attention", "sliding_attention"},
        )


if __name__ == "__main__":
    unittest.main()
