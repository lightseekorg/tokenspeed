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


class TestV4SlidingWindowGroupsSmoke(unittest.TestCase):
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
        from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
            DeepseekV4TokenToKVPool,
            deepseek_v4_cache_layout_from_config,
        )

        hf_config = SimpleNamespace(
            compress_ratios=(1, 4, 128),
            head_dim=512,
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


if __name__ == "__main__":
    unittest.main()
