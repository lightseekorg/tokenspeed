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
compute_max_logical_pages_for_capture = _generic.compute_max_logical_pages_for_capture
compute_paged_cache_group_page_counts = _generic.compute_paged_cache_group_page_counts
PagedCacheGroupSpec = _generic.PagedCacheGroupSpec

_PAGE_SHAPES = ((4, 1), (4, 2), (16, 4), (2, 128))


class TestV4SlidingWindowGroupsSmoke(unittest.TestCase):
    def test_overlap_page_budget_is_parameterized_by_verify_width_and_depth(self):
        max_live_requests = 3
        for rows_per_page, entry_stride_tokens in _PAGE_SHAPES:
            raw_per_page = rows_per_page * entry_stride_tokens
            specs = [
                PagedCacheGroupSpec(
                    group_id="full",
                    retention="full_history",
                    rows_per_page=rows_per_page,
                    entry_stride_tokens=entry_stride_tokens,
                    sliding_window_tokens=None,
                ),
                PagedCacheGroupSpec(
                    group_id="sliding",
                    retention="sliding_window",
                    rows_per_page=rows_per_page,
                    entry_stride_tokens=entry_stride_tokens,
                    sliding_window_tokens=3 * raw_per_page + 1,
                ),
            ]
            common = dict(
                max_live_requests=max_live_requests,
                max_scheduled_tokens=1024,
                max_total_tokens=4096,
                max_context_len=4096,
            )
            for verify_width in (1, 2, 4, 8):
                baseline = compute_paged_cache_group_page_counts(
                    specs,
                    **common,
                    decode_input_tokens=verify_width,
                    overlap_schedule_depth=0,
                )
                for overlap_depth in (0, 1):
                    with self.subTest(
                        raw_per_page=raw_per_page,
                        verify_width=verify_width,
                        overlap_depth=overlap_depth,
                    ):
                        actual = compute_paged_cache_group_page_counts(
                            specs,
                            **common,
                            decode_input_tokens=verify_width,
                            overlap_schedule_depth=overlap_depth,
                        )
                        protected_pages = max_live_requests * math.ceil(
                            overlap_depth * verify_width / raw_per_page
                        )
                        for group_id in ("full", "sliding"):
                            self.assertEqual(
                                actual[group_id],
                                baseline[group_id] + protected_pages,
                            )

    def test_capture_table_width_is_parameterized_by_verify_width_and_depth(self):
        for rows_per_page, entry_stride_tokens in _PAGE_SHAPES:
            raw_per_page = rows_per_page * entry_stride_tokens
            full = PagedCacheGroupSpec(
                group_id="full",
                retention="full_history",
                rows_per_page=rows_per_page,
                entry_stride_tokens=entry_stride_tokens,
                sliding_window_tokens=None,
            )
            window = 3 * raw_per_page + 1
            sliding = PagedCacheGroupSpec(
                group_id="sliding",
                retention="sliding_window",
                rows_per_page=rows_per_page,
                entry_stride_tokens=entry_stride_tokens,
                sliding_window_tokens=window,
            )
            context_len = 5 * raw_per_page + 1
            for verify_width in (1, 2, 4, 8):
                for overlap_depth in (0, 1):
                    with self.subTest(
                        raw_per_page=raw_per_page,
                        verify_width=verify_width,
                        overlap_depth=overlap_depth,
                    ):
                        full_pages = compute_max_logical_pages_for_capture(
                            full,
                            max_context_len=context_len,
                            max_tokens_per_req=verify_width,
                            overlap_schedule_depth=overlap_depth,
                        )
                        self.assertEqual(
                            full_pages,
                            math.ceil(
                                (context_len + (overlap_depth + 1) * verify_width)
                                / raw_per_page
                            ),
                        )

                        sliding_pages = compute_max_logical_pages_for_capture(
                            sliding,
                            max_context_len=context_len,
                            max_tokens_per_req=verify_width,
                            overlap_schedule_depth=overlap_depth,
                        )
                        self.assertEqual(
                            sliding_pages,
                            math.ceil(
                                (window + (overlap_depth + 1) * verify_width)
                                / raw_per_page
                            )
                            + 1,
                        )

    def test_sliding_capture_width_covers_conservative_reservation_bound(self):
        for rows_per_page, entry_stride_tokens in _PAGE_SHAPES:
            raw_per_page = rows_per_page * entry_stride_tokens
            # Cover a window that is a multiple of raw_per_page and one that is
            # not, exercising both page-alignment relationships between the
            # window and the physical page stride.
            for window in (3 * raw_per_page, 3 * raw_per_page + 1):
                spec = PagedCacheGroupSpec(
                    group_id="sliding",
                    retention="sliding_window",
                    rows_per_page=rows_per_page,
                    entry_stride_tokens=entry_stride_tokens,
                    sliding_window_tokens=window,
                )
                for context_len in (2 * raw_per_page + 1, 5 * raw_per_page + 1):
                    for verify_width in (1, 2, 4, 8):
                        for overlap_depth in (0, 1):
                            reservation_end = (
                                context_len + (overlap_depth + 1) * verify_width
                            )
                            with self.subTest(
                                raw_per_page=raw_per_page,
                                window=window,
                                context_len=context_len,
                                verify_width=verify_width,
                                overlap_depth=overlap_depth,
                            ):
                                capture_pages = compute_max_logical_pages_for_capture(
                                    spec,
                                    max_context_len=context_len,
                                    max_tokens_per_req=verify_width,
                                    overlap_schedule_depth=overlap_depth,
                                )
                                # Exercise the conservative full-window
                                # metadata bound used for capture.
                                retained_begin = max(0, reservation_end - window)
                                conservative_pages = math.ceil(
                                    reservation_end / raw_per_page
                                ) - math.floor(retained_begin / raw_per_page)
                                self.assertGreaterEqual(
                                    capture_pages, conservative_pages
                                )

    def test_overlap_sizing_rejects_invalid_runtime_parameters(self):
        spec = PagedCacheGroupSpec(
            group_id="full",
            retention="full_history",
            rows_per_page=4,
            entry_stride_tokens=1,
            sliding_window_tokens=None,
        )
        count_args = dict(
            max_live_requests=1,
            max_scheduled_tokens=8,
            max_total_tokens=8,
            max_context_len=8,
        )
        for overrides, message in (
            ({"decode_input_tokens": -1}, "decode_input_tokens"),
            ({"overlap_schedule_depth": 2}, "overlap_schedule_depth"),
            (
                {"decode_input_tokens": 0, "overlap_schedule_depth": 1},
                "decode_input_tokens",
            ),
        ):
            with self.subTest(function="page_counts", overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    compute_paged_cache_group_page_counts(
                        [spec], **count_args, **overrides
                    )

        for overrides, message in (
            ({"max_context_len": -1}, "max_context_len"),
            ({"max_tokens_per_req": 0}, "max_tokens_per_req"),
            ({"overlap_schedule_depth": 2}, "overlap_schedule_depth"),
        ):
            with self.subTest(function="capture_width", overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    compute_max_logical_pages_for_capture(
                        spec,
                        **{
                            "max_context_len": 8,
                            "max_tokens_per_req": 1,
                            **overrides,
                        },
                    )

        invalid_specs = (
            (
                PagedCacheGroupSpec("bad-rows", "full_history", 0, 1, None),
                "rows_per_page",
            ),
            (
                PagedCacheGroupSpec("bad-window", "sliding_window", 4, 1, 0),
                "sliding_window_tokens",
            ),
            (
                PagedCacheGroupSpec("bad-retention", "unknown", 4, 1, None),
                "unsupported retention",
            ),
        )
        for invalid_spec, message in invalid_specs:
            with self.subTest(group=invalid_spec.group_id):
                with self.assertRaisesRegex(ValueError, message):
                    compute_max_logical_pages_for_capture(
                        invalid_spec,
                        max_context_len=8,
                    )

    def test_overlap_schedule_enablement_truth_table(self):
        from tokenspeed.runtime.engine.scheduler_utils import (
            should_use_overlap_schedule,
        )

        cases = (
            # disabled, mode, expected
            (True, "fused", False),
            (False, "prefill", False),
            (False, "fused", True),
            (False, "decode", True),
        )
        for disabled, mode, expected in cases:
            with self.subTest(disabled=disabled, mode=mode):
                self.assertEqual(
                    should_use_overlap_schedule(
                        disable_overlap_schedule=disabled,
                        disaggregation_mode=mode,
                    ),
                    expected,
                )

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
        self.assertTrue(pool.supports_hierarchical_kv_cache)
        self.assertEqual(
            pool.prefix_cache_required_group_ids,
            (
                "v4.c4a.compressed_kv",
                "v4.c128a.compressed_kv",
            ),
        )

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
