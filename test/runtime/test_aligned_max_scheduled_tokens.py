"""aligned_max_scheduled_tokens (engine/scheduler_utils, applied in
engine/event_loop before make_config).

Recurrent-state cache groups (family=State, retention=FullHistory; the
C++ ``final_state_manager`` criterion) register their snapshot only when a
prefill chunk ends page-aligned (RegistersAlignedFinalPageOnly). The helper
floors the scheduler's max_scheduled_tokens to the LCM of those groups' page
grains so state pages can register and prefix-cache reuse stays live; the
admission probe takes the min across groups, so a never-registering state
group silently zeroes reuse for the whole model (observed on Kimi-K3:
chunked_prefill_size=8192 with prefix_granularity=1536 -> #cached-token: 0).
"""

from __future__ import annotations

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed_scheduler import (
    CacheGroupConfig,
    CacheGroupFamily,
    CacheRetention,
)

from tokenspeed.runtime.engine.scheduler_utils import aligned_max_scheduled_tokens


def _group(
    group_id: str,
    family: CacheGroupFamily,
    retention: CacheRetention,
    page_size: int = 64,
    sliding_window_tokens: int | None = None,
) -> CacheGroupConfig:
    kwargs = dict(
        group_id=group_id,
        rows_per_page=page_size,
        entry_stride_tokens=1,
        total_pages=8,
        retention=retention,
        family=family,
    )
    if sliding_window_tokens is not None:
        kwargs["sliding_window_tokens"] = sliding_window_tokens
    return CacheGroupConfig(**kwargs)


def _kimi_k3_groups(page_size: int = 1536) -> list[CacheGroupConfig]:
    groups = [
        _group(
            "full_attention",
            CacheGroupFamily.History,
            CacheRetention.FullHistory,
            page_size=page_size,
        )
    ]
    groups.extend(
        _group(
            f"linear_attention_{i}",
            CacheGroupFamily.State,
            CacheRetention.FullHistory,
            page_size=page_size,
        )
        for i in range(3)
    )
    return groups


class AlignedMaxScheduledTokensTest(unittest.TestCase):
    def test_kimi_k3_shape_floors_to_page_grain(self):
        # The observed production shape: 8192 % 1536 != 0 -> floor to 7680.
        self.assertEqual(aligned_max_scheduled_tokens(8192, _kimi_k3_groups()), 7680)

    def test_aligned_value_unchanged(self):
        self.assertEqual(aligned_max_scheduled_tokens(7680, _kimi_k3_groups()), 7680)

    def test_no_state_group_unchanged(self):
        groups = [
            _group(
                "full_attention",
                CacheGroupFamily.History,
                CacheRetention.FullHistory,
                page_size=1536,
            )
        ]
        self.assertEqual(aligned_max_scheduled_tokens(8192, groups), 8192)

    def test_empty_and_none_groups_unchanged(self):
        self.assertEqual(aligned_max_scheduled_tokens(8192, []), 8192)
        self.assertEqual(aligned_max_scheduled_tokens(8192, None), 8192)

    def test_sliding_window_state_group_is_swa_not_snapshot(self):
        # V4-style SWA KV rides State family with SlidingWindow retention; it
        # is a dense window, not an aligned-final-page snapshot group.
        groups = [
            _group(
                "v4.swa_kv",
                CacheGroupFamily.State,
                CacheRetention.SlidingWindow,
                page_size=1536,
                sliding_window_tokens=1536,
            )
        ]
        self.assertEqual(aligned_max_scheduled_tokens(8192, groups), 8192)

    def test_chunk_below_one_page_is_rejected(self):
        # Increasing the scheduler limit after executor buffers were sized
        # would make the first aligned chunk overflow those buffers.
        with self.assertRaisesRegex(ValueError, "minimum 1536"):
            aligned_max_scheduled_tokens(512, _kimi_k3_groups())

    def test_state_grain_comes_from_page_size(self):
        groups = [
            _group(
                "state",
                CacheGroupFamily.State,
                CacheRetention.FullHistory,
                page_size=96,
            )
        ]
        self.assertEqual(aligned_max_scheduled_tokens(1000, groups), 960)

    def test_mixed_grains_use_lcm(self):
        groups = [
            _group(
                "state_a",
                CacheGroupFamily.State,
                CacheRetention.FullHistory,
                page_size=64,
            ),
            _group(
                "state_b",
                CacheGroupFamily.State,
                CacheRetention.FullHistory,
                page_size=96,
            ),
        ]
        # lcm(64, 96) = 192; floor(1000 / 192) * 192 = 960.
        self.assertEqual(aligned_max_scheduled_tokens(1000, groups), 960)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            aligned_max_scheduled_tokens(0, _kimi_k3_groups())


if __name__ == "__main__":
    unittest.main()
