# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Per-window page budgets for multi-window models (full + W=128 + W=4 style):
each sliding group's device budget must follow ITS OWN window, keyed by the
suffixed group ids the spec grouping emits."""

from __future__ import annotations

import importlib.util
import math
import os
import pathlib
import sys
import types
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_CONFIGS_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "configs"
)

# compute_paged_cache_group_page_counts lazily imports ceil_div from
# tokenspeed.runtime.utils.common, whose package pulls torch/psutil. Prefer the
# real module (container runs); register a minimal equivalent only where the
# runtime deps are absent, so the pure math stays testable everywhere.
try:
    from tokenspeed.runtime.utils.common import ceil_div as _real_ceil_div  # noqa: F401
except Exception:
    if "tokenspeed.runtime.utils.common" not in sys.modules:
        _common = types.ModuleType("tokenspeed.runtime.utils.common")
        _common.ceil_div = lambda a, b: -(-a // b)
        sys.modules["tokenspeed.runtime.utils.common"] = _common


def _load(mod_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _CONFIGS_DIR / file_name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_pcs = _load("paged_cache_spec_for_page_counts", "paged_cache_spec.py")
compute_paged_cache_group_page_counts = _pcs.compute_paged_cache_group_page_counts
group_specs_from_layer_types = _pcs.group_specs_from_layer_types
PagedCacheGroupSpec = _pcs.PagedCacheGroupSpec
DUMMY = _pcs._PAGED_CACHE_GROUP_DUMMY_PAGES


def _spec(group_id, retention, window=None, rows_per_page=64):
    return PagedCacheGroupSpec(
        group_id=group_id,
        retention=retention,
        rows_per_page=rows_per_page,
        entry_stride_tokens=1,
        sliding_window_tokens=window,
    )


class MultiWindowPageCountsTest(unittest.TestCase):
    """full + W=128 + W=4 on page 64: three different budgets from one call."""

    def counts(self, specs, **kw):
        defaults = dict(
            max_live_requests=4,
            max_scheduled_tokens=512,
            max_total_tokens=4096,
            max_context_len=4096,
        )
        defaults.update(kw)
        return compute_paged_cache_group_page_counts(specs, **defaults)

    def test_each_window_budgets_independently(self):
        counts = self.counts(
            [
                _spec("full_attention", "full_history"),
                _spec("sliding_attention_128", "sliding_window", window=128),
                _spec("sliding_attention_4", "sliding_window", window=4),
            ]
        )
        # full: ceil(4096/64) + live + dummy
        self.assertEqual(counts["full_attention"], 64 + 4 + DUMMY)
        # W=128: resident ceil(127/64)=2 per request; scheduled ceil(512/64)=8
        self.assertEqual(counts["sliding_attention_128"], 4 * 2 + 8 + 4 + DUMMY)
        # W=4: resident ceil(3/64)=1 per request -- a sub-page window still
        # holds one page while its partial tail is live
        self.assertEqual(counts["sliding_attention_4"], 4 * 1 + 8 + 4 + DUMMY)
        self.assertGreater(
            counts["sliding_attention_128"], counts["sliding_attention_4"]
        )

    def test_window_one_holds_no_resident_pages(self):
        counts = self.counts([_spec("s", "sliding_window", window=1)])
        self.assertEqual(counts["s"], 0 + 8 + 4 + DUMMY)

    def test_resident_window_clamped_by_context_len(self):
        wide = self.counts(
            [_spec("s", "sliding_window", window=128)], max_context_len=32
        )
        # min(127, 32) = 32 -> 1 resident page per request instead of 2
        self.assertEqual(wide["s"], 4 * 1 + 8 + 4 + DUMMY)

    def test_scheduled_tokens_capped_by_total(self):
        counts = self.counts(
            [_spec("s", "sliding_window", window=128)],
            max_scheduled_tokens=10_000,
            max_total_tokens=4096,
        )
        self.assertEqual(counts["s"], 4 * 2 + math.ceil(4096 / 64) + 4 + DUMMY)

    def test_safety_margin_added_per_group(self):
        base = self.counts([_spec("full_attention", "full_history")])
        padded = self.counts([_spec("full_attention", "full_history")], safety_margin=7)
        self.assertEqual(padded["full_attention"], base["full_attention"] + 7)

    def test_sliding_without_window_raises(self):
        with self.assertRaises(ValueError):
            self.counts([_spec("s", "sliding_window", window=None)])


class SuffixedGroupIdFlowTest(unittest.TestCase):
    """Spec grouping and budget computation agree on the suffixed group ids --
    the exact dict the C++ scheduler receives as per-group total_pages."""

    def test_grouping_feeds_counts_end_to_end(self):
        layer_types = [
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ]
        windows = [None, 128, None, 4]
        specs = group_specs_from_layer_types(
            layer_types=layer_types, page_size=64, sliding_window_tokens=windows
        )
        self.assertEqual(
            [s.group_id for s in specs],
            ["full_attention", "sliding_attention_128", "sliding_attention_4"],
        )
        counts = compute_paged_cache_group_page_counts(
            specs,
            max_live_requests=4,
            max_scheduled_tokens=512,
            max_total_tokens=4096,
            max_context_len=4096,
        )
        self.assertEqual(
            set(counts),
            {"full_attention", "sliding_attention_128", "sliding_attention_4"},
        )
        self.assertGreater(counts["full_attention"], counts["sliding_attention_128"])
        self.assertGreater(
            counts["sliding_attention_128"], counts["sliding_attention_4"]
        )


if __name__ == "__main__":
    unittest.main()
