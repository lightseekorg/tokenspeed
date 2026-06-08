# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

_CONFIGS_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "configs"
)


def _load(mod_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _CONFIGS_DIR / file_name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_generic = _load("tokenspeed.runtime.configs.paged_cache_spec", "paged_cache_spec.py")
_v4 = _load(
    "tokenspeed_runtime_configs_deepseek_v4_cache_spec_smoke",
    "deepseek_v4_cache_spec.py",
)

build_v4_cache_specs = _v4.build_v4_cache_specs
compute_paged_cache_group_page_counts = _generic.compute_paged_cache_group_page_counts
PagedCacheGroupSpec = _generic.PagedCacheGroupSpec


class TestV4SlidingWindowGroupsSmoke(unittest.TestCase):
    def test_sliding_window_scheduled_tokens_are_global_and_capped(self):
        specs = [
            PagedCacheGroupSpec(
                group_id="sliding",
                retention="sliding_window",
                rows_per_page=4,
                entry_stride_tokens=1,
                sliding_window_tokens=8,
            )
        ]

        counts = compute_paged_cache_group_page_counts(
            specs,
            max_live_requests=10,
            max_scheduled_tokens=100,
            max_total_tokens=20,
            max_context_len=4096,
        )

        resident_pages = 10 * math.ceil(7 / 4)
        scheduled_pages = math.ceil(20 / 4)
        request_fragment_pages = 10
        dummy_pages = 1
        self.assertEqual(
            counts["sliding"],
            resident_pages + scheduled_pages + request_fragment_pages + dummy_pages,
        )

    def test_page_counts_positive_finite_and_under_total_times_live(self):
        inputs = dict(
            max_live_requests=32,
            max_scheduled_tokens=2048,
            max_total_tokens=64 * 1024,
            max_context_len=64 * 1024,
        )
        specs = build_v4_cache_specs(
            SimpleNamespace(sliding_window=128),
            layer_ratio=(1, 4, 128),
        )
        counts = compute_paged_cache_group_page_counts(specs, **inputs)
        bound = inputs["max_total_tokens"] * inputs["max_live_requests"]
        for spec in specs:
            n = counts[spec.group_id]
            self.assertIsInstance(n, int, spec.group_id)
            self.assertGreater(n, 0, spec.group_id)
            self.assertTrue(math.isfinite(n), spec.group_id)
            self.assertLess(n, bound, spec.group_id)

    def test_deepseek_v4_pool_exposes_scheduler_cache_groups(self):
        from tokenspeed.runtime.layers.attention.kv_cache import (
            deepseek_v4 as deepseek_v4_kv,
        )
        from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
            DeepseekV4TokenToKVPool,
            deepseek_v4_cache_layout_from_config,
        )

        hf_config = SimpleNamespace(
            compress_ratios=(1, 4, 128),
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            sliding_window=128,
        )
        layout = deepseek_v4_cache_layout_from_config(
            hf_config,
            page_size=256,
            use_fp4_indexer_cache=True,
        )
        pool = DeepseekV4TokenToKVPool(
            size=1024,
            model_dtype=torch.bfloat16,
            layout=layout,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=1024,
            page_size=256,
            rank=0,
            hf_config=hf_config,
            max_scheduled_tokens=256,
        )

        group_ids = {spec.group_id for spec in pool.paged_cache_group_specs}
        self.assertIn("v4.swa_kv", group_ids)
        self.assertIn("v4.c4a.compressor_state", group_ids)
        self.assertIn("v4.c128a.compressor_state", group_ids)
        self.assertIn("v4.c4a.compressed_kv", group_ids)
        self.assertIn("v4.c128a.compressed_kv", group_ids)
        self.assertIn("v4.c4a.indexer_compressor_state", group_ids)
        self.assertGreater(
            pool.paged_cache_group_page_counts["v4.c4a.compressed_kv"], 1
        )
        self.assertFalse(hasattr(pool, "prefix_cache_state_policy"))
        self.assertFalse(pool.supports_hierarchical_kv_cache)
        self.assertEqual(
            pool.prefix_cache_required_group_ids,
            (
                "v4.c4a.compressed_kv",
                "v4.c128a.compressed_kv",
            ),
        )

        class FakePagedCacheScheduler:
            @staticmethod
            def paged_cache_group_total_pages(group_id: str) -> int:
                return 11

            @staticmethod
            def paged_cache_group_available_pages(group_id: str) -> int:
                return 4

            @staticmethod
            def paged_cache_group_failed_alloc_count(group_id: str) -> int:
                return 2

        pool.bind_paged_cache_scheduler(FakePagedCacheScheduler())
        with (
            patch.object(deepseek_v4_kv.logger, "isEnabledFor", return_value=True),
            patch.object(deepseek_v4_kv.logger, "debug") as log_debug,
        ):
            pool.maybe_log_paged_cache_group_pages()
        log_debug.assert_called_once()
        logged_groups = log_debug.call_args.args[1]
        self.assertIn("v4.swa_kv: used=7/11", logged_groups)
        self.assertIn("v4.c4a.indexer_compressor_state", logged_groups)
        self.assertIn("failed_alloc=2", logged_groups)

    def test_deepseek_v4_capacity_profile_matches_pool_buffers(self):
        from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
            DeepseekV4TokenToKVPool,
            deepseek_v4_cache_layout_from_config,
            profile_deepseek_v4_max_num_pages,
        )

        hf_config = SimpleNamespace(
            compress_ratios=(1, 4, 128),
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            sliding_window=128,
        )
        layout = deepseek_v4_cache_layout_from_config(
            hf_config,
            page_size=64,
            use_fp4_indexer_cache=True,
        )

        def make_pool(num_pages: int) -> DeepseekV4TokenToKVPool:
            return DeepseekV4TokenToKVPool(
                size=num_pages * layout.page_size,
                model_dtype=torch.bfloat16,
                layout=layout,
                layer_num=3,
                device="cpu",
                enable_memory_saver=False,
                max_batch_size=2,
                max_context_len=1024,
                page_size=layout.page_size,
                rank=0,
                hf_config=hf_config,
                max_scheduled_tokens=1,
            )

        def buffer_bytes(pool: DeepseekV4TokenToKVPool) -> int:
            tensors = []
            tensors.extend(pool.swa_kv_buffer)
            tensors.extend(pool.compressed_kv_buffer)
            tensors.extend(pool.compressor_state_buffer)
            tensors.extend(pool.indexer_kv_buffer)
            tensors.extend(pool.indexer_state_buffer)
            return sum(
                tensor.numel() * tensor.element_size()
                for tensor in tensors
                if tensor is not None
            )

        target_pages = 8
        current_bytes = buffer_bytes(make_pool(target_pages))
        next_bytes = buffer_bytes(make_pool(target_pages + 1))

        self.assertGreater(next_bytes, current_bytes)
        self.assertEqual(
            profile_deepseek_v4_max_num_pages(
                layout=layout,
                hf_config=hf_config,
                layer_num=3,
                max_live_requests=2,
                max_scheduled_tokens=1,
                max_context_len=1024,
                available_cache_memory_bytes=current_bytes,
            ),
            target_pages,
        )

        legacy_pages = current_bytes // (layout.cache_cell_size(3) * layout.page_size)
        self.assertLess(legacy_pages, target_pages)

    def test_deepseek_v4_profile_does_not_multiply_scheduled_tokens_by_requests(self):
        from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
            deepseek_v4_cache_layout_from_config,
            profile_deepseek_v4_max_num_pages,
        )

        hf_config = SimpleNamespace(
            compress_ratios=tuple([1, 1] + [4, 128] * 20 + [4, 1]),
            head_dim=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            sliding_window=128,
        )
        layout = deepseek_v4_cache_layout_from_config(
            hf_config,
            page_size=256,
            use_fp4_indexer_cache=True,
        )

        self.assertGreater(
            profile_deepseek_v4_max_num_pages(
                layout=layout,
                hf_config=hf_config,
                layer_num=43,
                max_live_requests=160,
                max_scheduled_tokens=8192,
                max_context_len=4096,
                available_cache_memory_bytes=80 * (1 << 30),
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
