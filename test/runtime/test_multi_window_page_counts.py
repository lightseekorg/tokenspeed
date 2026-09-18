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
suffixed group ids the spec grouping emits. The budget is the scheduler's
capacity model, reached through the recipes' bridge."""

from __future__ import annotations

import math
import os
import sys
import unittest

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

ts = pytest.importorskip("tokenspeed_scheduler")

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (  # noqa: E402
    CacheFieldSpec,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (  # noqa: E402
    SchedulerLimits,
    capacity_model,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (  # noqa: E402
    CacheGroupSpec,
    group,
    layer_group_ids,
)

PAGE = 64


def group_specs_from_layer_types(**kwargs):
    """The specs a layer vocabulary produces, via the one-walk ``group``.

    A group must declare fields, so a one-byte placeholder stands in: these
    tests are about scheduler semantics, not bytes.
    """
    return tuple(
        group_spec
        for group_spec, _ in group(
            fields_for_layer=lambda layer_id, group_id, occurrence: (
                CacheFieldSpec(
                    f"layer.{layer_id}.probe", f"unit.{occurrence}", (1,), "uint8"
                ),
            ),
            **kwargs,
        )
    )


def _spec(group_id, retention, window=None, rows_per_page=PAGE):
    return CacheGroupSpec(
        group_id=group_id,
        retention=retention,
        rows_per_page=rows_per_page,
        entry_stride_tokens=1,
        sliding_window_tokens=window,
        replayable=False,
    )


def _pages(specs, **kw):
    """Child pages per group (null page excluded), keyed by group id."""
    defaults = dict(
        max_live_requests=4,
        max_scheduled_tokens=512,
        max_total_tokens=4096,
        max_context_len=4096,
    )
    defaults.update(kw)
    model = capacity_model(
        specs,
        prefix_granularity=PAGE,
        virtual_packing={spec.group_id: 1 for spec in specs},
        limits=SchedulerLimits(
            role=ts.SchedulerConfig.Role.Fused,
            max_live_requests=defaults["max_live_requests"],
            max_scheduled_tokens=defaults["max_scheduled_tokens"],
            max_context_len=defaults["max_context_len"],
            decode_input_tokens=1,
            overlap_schedule_depth=0,
            disable_prefix_cache=False,
        ),
    )
    pages = model.concurrent_group_pages(
        max_total_tokens=defaults["max_total_tokens"],
        max_context_len=defaults["max_context_len"],
    )
    return dict(zip((spec.group_id for spec in specs), pages))


class MultiWindowPageCountsTest(unittest.TestCase):
    """full + W=128 + W=4 on page 64: three different budgets from one call.

    Per live request a sliding group retains ceil((W - 1 + decode + page - 1)
    / page) pages -- its window, the next decode token, at any alignment --
    and one in-flight prefill chunk adds its lookback plus ceil(chunk / page)
    rows before they slide out.
    """

    def test_each_window_budgets_independently(self):
        pages = _pages(
            [
                _spec("full_attention", "full_history"),
                _spec("sliding_attention_128", "sliding_window", window=128),
                _spec("sliding_attention_4", "sliding_window", window=4),
            ]
        )
        # full: ceil(4096/64) dense + one unaligned tail page per request
        self.assertEqual(pages["full_attention"], 64 + 4)
        # W=128: ceil((127 + 1 + 63)/64) = 3 per request; lookback ceil(127/64)
        # = 2 and ceil(512/64) = 8 rows for the chunk in flight
        self.assertEqual(pages["sliding_attention_128"], 4 * 3 + 2 + 8)
        # W=4: ceil((3 + 1 + 63)/64) = 2 per request -- a sub-page window can
        # still straddle two pages while its partial tail is live; lookback 1
        self.assertEqual(pages["sliding_attention_4"], 4 * 2 + 1 + 8)
        self.assertGreater(pages["sliding_attention_128"], pages["sliding_attention_4"])

    def test_window_one_holds_only_the_token_being_written(self):
        pages = _pages([_spec("s", "sliding_window", window=1)])
        # No resident history and no lookback: one page per request for the
        # decode token, plus the chunk in flight.
        self.assertEqual(pages["s"], 4 * 1 + 0 + 8)

    def test_resident_window_clamped_by_context_len(self):
        wide = _pages([_spec("s", "sliding_window", window=128)], max_context_len=32)
        # min(127, 32) = 32 -> ceil((32 + 1 + 63)/64) = 2 pages per request
        # instead of 3; the resumable-boundary lookback still follows the window.
        self.assertEqual(wide["s"], 4 * 2 + 2 + 8)

    def test_scheduled_tokens_capped_by_total(self):
        pages = _pages(
            [_spec("s", "sliding_window", window=128)],
            max_scheduled_tokens=10_000,
            max_total_tokens=4096,
        )
        self.assertEqual(pages["s"], 4 * 3 + 2 + math.ceil(4096 / PAGE))

    def test_sliding_without_window_raises(self):
        with self.assertRaises(ValueError):
            _pages([_spec("s", "sliding_window", window=None)])


class SuffixedGroupIdFlowTest(unittest.TestCase):
    """Spec grouping and budget computation agree on the suffixed group ids --
    the groups the C++ scheduler sizes and later reads."""

    def test_grouping_feeds_counts_end_to_end(self):
        layer_types = [
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ]
        windows = [None, 128, None, 4]
        specs = group_specs_from_layer_types(
            layer_types=layer_types,
            group_ids=layer_group_ids(
                layer_types=layer_types, sliding_window_tokens=windows
            ),
            prefix_granularity=PAGE,
            sliding_window_tokens=windows,
        )
        self.assertEqual(
            [s.group_id for s in specs],
            ["full_attention", "sliding_attention_128", "sliding_attention_4"],
        )
        pages = _pages(specs)
        self.assertEqual(
            set(pages),
            {"full_attention", "sliding_attention_128", "sliding_attention_4"},
        )
        self.assertGreater(pages["full_attention"], pages["sliding_attention_128"])
        self.assertGreater(pages["sliding_attention_128"], pages["sliding_attention_4"])


if __name__ == "__main__":
    unittest.main()
