"""aligned_max_scheduled_tokens (engine/scheduler_utils, applied in
engine/event_loop before make_config).

Recurrent-state paged-cache groups (family=State, retention=FullHistory; the
C++ ``final_state_manager`` criterion) register their snapshot only when a
prefill chunk ends page-aligned (RegistersAlignedFinalPageOnly). The helper
floors the scheduler's max_scheduled_tokens to the LCM of those groups' page
grains so state pages can register and prefix-cache reuse stays live; the
admission probe takes the min across groups, so a never-registering state
group silently zeroes reuse for the whole model (observed on Kimi-K3:
chunked_prefill_size=8192 with block_size=1536 -> #cached-token: 0).
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
    PagedCacheGroupConfig,
    PagedCacheGroupFamily,
    PagedCacheRetention,
)

from tokenspeed.runtime.engine.scheduler_utils import aligned_max_scheduled_tokens


def _group(
    group_id: str,
    family: PagedCacheGroupFamily,
    retention: PagedCacheRetention,
    block_size: int = 0,
    sliding_window_tokens: int | None = None,
) -> PagedCacheGroupConfig:
    kwargs = dict(
        group_id=group_id,
        rows_per_page=64,
        entry_stride_tokens=1,
        total_pages=8,
        retention=retention,
        family=family,
    )
    if sliding_window_tokens is not None:
        kwargs["sliding_window_tokens"] = sliding_window_tokens
    cfg = PagedCacheGroupConfig(**kwargs)
    if block_size:
        cfg.block_size = block_size
    return cfg


def _kimi_k3_groups(block_size: int = 1536) -> list[PagedCacheGroupConfig]:
    groups = [
        _group(
            "full_attention",
            PagedCacheGroupFamily.History,
            PagedCacheRetention.FullHistory,
            block_size=block_size,
        )
    ]
    groups.extend(
        _group(
            f"linear_attention_{i}",
            PagedCacheGroupFamily.State,
            PagedCacheRetention.FullHistory,
            block_size=block_size,
        )
        for i in range(3)
    )
    return groups


class AlignedMaxScheduledTokensTest(unittest.TestCase):
    def test_kimi_k3_shape_floors_to_page_grain(self):
        # The observed production shape: 8192 % 1536 != 0 -> floor to 7680.
        self.assertEqual(
            aligned_max_scheduled_tokens(8192, _kimi_k3_groups(), 1536), 7680
        )

    def test_aligned_value_unchanged(self):
        self.assertEqual(
            aligned_max_scheduled_tokens(7680, _kimi_k3_groups(), 1536), 7680
        )

    def test_no_state_group_unchanged(self):
        groups = [
            _group(
                "full_attention",
                PagedCacheGroupFamily.History,
                PagedCacheRetention.FullHistory,
                block_size=1536,
            )
        ]
        self.assertEqual(aligned_max_scheduled_tokens(8192, groups, 1536), 8192)

    def test_empty_and_none_groups_unchanged(self):
        self.assertEqual(aligned_max_scheduled_tokens(8192, [], 64), 8192)
        self.assertEqual(aligned_max_scheduled_tokens(8192, None, 64), 8192)

    def test_sliding_window_state_group_is_swa_not_snapshot(self):
        # V4-style SWA KV rides State family with SlidingWindow retention; it
        # is a dense window, not an aligned-final-page snapshot group.
        groups = [
            _group(
                "v4.swa_kv",
                PagedCacheGroupFamily.State,
                PagedCacheRetention.SlidingWindow,
                block_size=1536,
                sliding_window_tokens=1536,
            )
        ]
        self.assertEqual(aligned_max_scheduled_tokens(8192, groups, 1536), 8192)

    def test_chunk_below_one_page_clamps_to_one_page(self):
        # A chunk smaller than one page could never register a snapshot.
        self.assertEqual(
            aligned_max_scheduled_tokens(512, _kimi_k3_groups(), 1536), 1536
        )

    def test_zero_block_size_falls_back_to_page_size(self):
        groups = [
            _group(
                "state",
                PagedCacheGroupFamily.State,
                PagedCacheRetention.FullHistory,
                # block_size left 0 = unset -> global page size governs.
            )
        ]
        self.assertEqual(aligned_max_scheduled_tokens(1000, groups, 96), 960)

    def test_mixed_grains_use_lcm(self):
        groups = [
            _group(
                "state_a",
                PagedCacheGroupFamily.State,
                PagedCacheRetention.FullHistory,
                block_size=64,
            ),
            _group(
                "state_b",
                PagedCacheGroupFamily.State,
                PagedCacheRetention.FullHistory,
                block_size=96,
            ),
        ]
        # lcm(64, 96) = 192; floor(1000 / 192) * 192 = 960.
        self.assertEqual(aligned_max_scheduled_tokens(1000, groups, 64), 960)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            aligned_max_scheduled_tokens(0, _kimi_k3_groups(), 1536)
        with self.assertRaises(ValueError):
            aligned_max_scheduled_tokens(8192, _kimi_k3_groups(), 0)


if __name__ == "__main__":
    unittest.main()
