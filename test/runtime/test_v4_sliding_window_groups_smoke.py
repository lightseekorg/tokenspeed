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

_CONFIGS_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "layers"
    / "attention"
    / "kv_cache"
    / "recipes"
)


def _load(mod_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _CONFIGS_DIR / file_name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Shadow the real module only while the v4 spec module binds its imports,
# then restore: leaving it would fork PagedCacheGroupSpec into two classes
# and fail the contract's isinstance check in later test files.
_orig_generic = sys.modules.get(
    "tokenspeed.runtime.layers.attention.kv_cache.recipes.spec"
)
_generic = _load(
    "tokenspeed.runtime.layers.attention.kv_cache.recipes.spec",
    "spec.py",
)
_v4 = _load(
    "tokenspeed_runtime_configs_deepseek_v4_cache_spec_smoke",
    "deepseek_v4_cache_spec.py",
)
if _orig_generic is not None:
    sys.modules["tokenspeed.runtime.layers.attention.kv_cache.recipes.spec"] = (
        _orig_generic
    )
else:
    del sys.modules["tokenspeed.runtime.layers.attention.kv_cache.recipes.spec"]

build_v4_cache_specs = _v4.build_v4_cache_specs
deepseek_v4_lcm_blocks_needed = _v4.deepseek_v4_lcm_blocks_needed
deepseek_v4_token_capacity_for_cache_pool = (
    _v4.deepseek_v4_token_capacity_for_cache_pool
)
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
            common = {
                "max_live_requests": max_live_requests,
                "max_scheduled_tokens": 1024,
                "max_total_tokens": 4096,
                "max_context_len": 4096,
            }
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
        count_args = {
            "max_live_requests": 1,
            "max_scheduled_tokens": 8,
            "max_total_tokens": 8,
            "max_context_len": 8,
        }
        for overrides, message in (
            ({"decode_input_tokens": -1}, "decode_input_tokens"),
            ({"overlap_schedule_depth": 2}, "overlap_schedule_depth"),
            (
                {"decode_input_tokens": 0, "overlap_schedule_depth": 1},
                "decode_input_tokens",
            ),
        ):
            with (
                self.subTest(function="page_counts", overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                compute_paged_cache_group_page_counts([spec], **count_args, **overrides)

        for overrides, message in (
            ({"max_context_len": -1}, "max_context_len"),
            ({"max_tokens_per_req": 0}, "max_tokens_per_req"),
            ({"overlap_schedule_depth": 2}, "overlap_schedule_depth"),
        ):
            with (
                self.subTest(function="capture_width", overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                compute_max_logical_pages_for_capture(
                    spec,
                    **{
                        "max_context_len": 8,
                        "max_tokens_per_req": 1,
                        **overrides,
                    },
                )

        # Non-positive row geometry is rejected at spec construction now.
        with (
            self.subTest(group="bad-rows"),
            self.assertRaisesRegex(ValueError, "rows_per_page"),
        ):
            PagedCacheGroupSpec("bad-rows", "full_history", 0, 1, None)

        invalid_specs = (
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
            with (
                self.subTest(group=invalid_spec.group_id),
                self.assertRaisesRegex(ValueError, message),
            ):
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
        inputs = {
            "max_live_requests": 32,
            "max_scheduled_tokens": 2048,
            "max_total_tokens": 64 * 1024,
            "max_context_len": 64 * 1024,
        }
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

    def test_lcm_specs_preserve_group_page_sizes_and_publish_packing(self):
        packing = {
            "v4.swa_kv": 1,
            "v4.c4a.compressor_state": 16,
            "v4.c4a.compressed_kv": 2,
            "v4.c128a.compressor_state": 32,
            "v4.c128a.compressed_kv": 8,
            "v4.c4a.indexer_compressor_state": 16,
        }

        specs = build_v4_cache_specs(
            SimpleNamespace(sliding_window=128),
            layer_ratio=(1, 4, 128),
            cache_blocks_per_lcm_block=packing,
        )

        self.assertEqual(
            {spec.group_id: spec.cache_blocks_per_lcm_block for spec in specs},
            packing,
        )
        rows = {spec.group_id: spec.rows_per_page for spec in specs}
        self.assertEqual(rows["v4.swa_kv"], 64)
        self.assertEqual(rows["v4.c4a.compressor_state"], 4)
        self.assertEqual(rows["v4.c128a.compressor_state"], 8)

    def test_c4_state_window_covers_wide_verify_blocks(self):
        base = build_v4_cache_specs(
            SimpleNamespace(sliding_window=128),
            layer_ratio=(4,),
        )
        wide = build_v4_cache_specs(
            SimpleNamespace(sliding_window=128),
            layer_ratio=(4,),
            decode_input_tokens=6,
        )

        base_windows = {spec.group_id: spec.sliding_window_tokens for spec in base}
        wide_windows = {spec.group_id: spec.sliding_window_tokens for spec in wide}
        for group_id in (
            "v4.c4a.compressor_state",
            "v4.c4a.indexer_compressor_state",
        ):
            self.assertEqual(base_windows[group_id], 8)
            self.assertEqual(wide_windows[group_id], 10)

    def test_lcm_capacity_is_the_inverse_of_parent_demand(self):
        packing = {
            "v4.swa_kv": 1,
            "v4.c4a.compressor_state": 4,
            "v4.c4a.compressed_kv": 2,
            "v4.c128a.compressor_state": 1,
            "v4.c128a.compressed_kv": 8,
            "v4.c4a.indexer_compressor_state": 4,
        }
        specs = build_v4_cache_specs(
            SimpleNamespace(sliding_window=128),
            layer_ratio=(1, 4, 128),
            cache_blocks_per_lcm_block=packing,
        )
        sizing = {
            "prefix_granularity": 256,
            "max_live_requests": 1,
            "max_scheduled_tokens": 256,
            "max_context_len": 4096,
        }
        num_lcm_blocks = 100
        capacity = deepseek_v4_token_capacity_for_cache_pool(
            specs,
            num_lcm_blocks=num_lcm_blocks,
            upper_bound_tokens=4096,
            **sizing,
        )

        self.assertLessEqual(
            deepseek_v4_lcm_blocks_needed(
                specs,
                token_capacity=capacity,
                **sizing,
            ),
            num_lcm_blocks,
        )
        self.assertGreater(
            deepseek_v4_lcm_blocks_needed(
                specs,
                token_capacity=capacity + 1,
                **sizing,
            ),
            num_lcm_blocks,
        )


if __name__ == "__main__":
    unittest.main()
