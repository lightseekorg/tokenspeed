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

import math
import sys

import pytest

from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (
    SchedulerLimits,
    capacity_model,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    STATE_TABLE_SLOTS,
    CacheGroupSpec,
    compute_max_logical_pages_for_capture,
)


def compute_cache_group_pages(
    specs,
    *,
    prefix_granularity,
    max_live_requests,
    max_scheduled_tokens,
    max_total_tokens,
    max_context_len,
    decode_input_tokens,
    overlap_schedule_depth,
):
    """Child pages per group (null page excluded) from the scheduler's model."""
    from tokenspeed_scheduler import SchedulerConfig

    model = capacity_model(
        specs,
        prefix_granularity=prefix_granularity,
        virtual_packing={spec.group_id: 1 for spec in specs},
        limits=SchedulerLimits(
            role=SchedulerConfig.Role.Fused,
            max_live_requests=max_live_requests,
            max_scheduled_tokens=max_scheduled_tokens,
            max_context_len=max_context_len,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=overlap_schedule_depth,
            disable_prefix_cache=False,
        ),
    )
    pages = model.concurrent_group_pages(
        max_total_tokens=max_total_tokens, max_context_len=max_context_len
    )
    return dict(zip((spec.group_id for spec in specs), pages))


_PAGE_SHAPES = ((4, 1), (4, 2), (16, 4), (2, 128))


class TestV4SlidingWindowGroupsSmoke:
    def test_overlap_page_budget_is_parameterized_by_verify_width_and_depth(
        self, subtests
    ):
        max_live_requests = 3
        for rows_per_page, entry_stride_tokens in _PAGE_SHAPES:
            raw_per_page = rows_per_page * entry_stride_tokens
            specs = [
                CacheGroupSpec(
                    group_id="full",
                    attn_kind="full_history",
                    rows_per_page=rows_per_page,
                    entry_stride_tokens=entry_stride_tokens,
                    sliding_window_tokens=None,
                    replayable=False,
                ),
                CacheGroupSpec(
                    group_id="sliding",
                    attn_kind="sliding_window",
                    rows_per_page=rows_per_page,
                    entry_stride_tokens=entry_stride_tokens,
                    sliding_window_tokens=3 * raw_per_page + 1,
                    replayable=False,
                ),
            ]
            common = {
                "prefix_granularity": raw_per_page,
                "max_live_requests": max_live_requests,
                "max_scheduled_tokens": 1024,
                "max_total_tokens": 4096,
                "max_context_len": 4096,
            }
            for verify_width in (1, 2, 4, 8):
                baseline = compute_cache_group_pages(
                    specs,
                    **common,
                    decode_input_tokens=verify_width,
                    overlap_schedule_depth=0,
                )
                overlapped = compute_cache_group_pages(
                    specs,
                    **common,
                    decode_input_tokens=verify_width,
                    overlap_schedule_depth=1,
                )
                with subtests.test(
                    raw_per_page=raw_per_page, verify_width=verify_width
                ):
                    # One overlapped step protects one more verify window per
                    # live request: never fewer pages, never more than the
                    # window's own page span. Speculative decode already
                    # reserves a draft window; overlap adds another window
                    # beyond that reservation and the unaligned history tail.
                    for group_id in ("full", "sliding"):
                        delta = overlapped[group_id] - baseline[group_id]
                        _pytest_a_12 = delta
                        _pytest_b_13 = 0
                        _pytest_msg_14 = group_id
                        assert _pytest_a_12 >= _pytest_b_13, _pytest_msg_14
                        _pytest_a_15 = delta
                        _pytest_b_16 = max_live_requests * math.ceil(
                            verify_width / raw_per_page
                        )
                        _pytest_msg_17 = group_id
                        assert _pytest_a_15 <= _pytest_b_16, _pytest_msg_17
                    draft_tokens = verify_width if verify_width > 1 else 0
                    assert (overlapped["full"] - baseline["full"]) == (
                        max_live_requests
                        * (
                            math.ceil(
                                (raw_per_page - 1 + draft_tokens + verify_width)
                                / raw_per_page
                            )
                            - math.ceil(
                                (raw_per_page - 1 + draft_tokens) / raw_per_page
                            )
                        )
                    )

    def test_capture_table_width_is_parameterized_by_verify_width_and_depth(
        self, subtests
    ):
        for rows_per_page, entry_stride_tokens in _PAGE_SHAPES:
            raw_per_page = rows_per_page * entry_stride_tokens
            full = CacheGroupSpec(
                group_id="full",
                attn_kind="full_history",
                rows_per_page=rows_per_page,
                entry_stride_tokens=entry_stride_tokens,
                sliding_window_tokens=None,
                replayable=False,
            )
            window = 3 * raw_per_page + 1
            sliding = CacheGroupSpec(
                group_id="sliding",
                attn_kind="sliding_window",
                rows_per_page=rows_per_page,
                entry_stride_tokens=entry_stride_tokens,
                sliding_window_tokens=window,
                replayable=False,
            )
            context_len = 5 * raw_per_page + 1
            for verify_width in (1, 2, 4, 8):
                for overlap_depth in (0, 1):
                    with subtests.test(
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
                        assert (full_pages) == (
                            math.ceil(
                                (context_len + (overlap_depth + 1) * verify_width)
                                / raw_per_page
                            )
                        )

                        sliding_pages = compute_max_logical_pages_for_capture(
                            sliding,
                            max_context_len=context_len,
                            max_tokens_per_req=verify_width,
                            overlap_schedule_depth=overlap_depth,
                        )
                        assert (sliding_pages) == (
                            math.ceil(
                                (window + (overlap_depth + 1) * verify_width)
                                / raw_per_page
                            )
                            + 1
                        )

    def test_sliding_capture_width_covers_conservative_reservation_bound(
        self, subtests
    ):
        for rows_per_page, entry_stride_tokens in _PAGE_SHAPES:
            raw_per_page = rows_per_page * entry_stride_tokens
            # Cover a window that is a multiple of raw_per_page and one that is
            # not, exercising both page-alignment relationships between the
            # window and the physical page stride.
            for window in (3 * raw_per_page, 3 * raw_per_page + 1):
                spec = CacheGroupSpec(
                    group_id="sliding",
                    attn_kind="sliding_window",
                    rows_per_page=rows_per_page,
                    entry_stride_tokens=entry_stride_tokens,
                    sliding_window_tokens=window,
                    replayable=False,
                )
                for context_len in (2 * raw_per_page + 1, 5 * raw_per_page + 1):
                    for verify_width in (1, 2, 4, 8):
                        for overlap_depth in (0, 1):
                            reservation_end = (
                                context_len + (overlap_depth + 1) * verify_width
                            )
                            with subtests.test(
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
                                assert (capture_pages) >= (conservative_pages)

    def test_state_capture_width_is_the_two_slots_whatever_the_context(self):
        state = CacheGroupSpec(
            group_id="state",
            attn_kind="state",
            sliding_window_tokens=None,
            replayable=False,
        )
        for context_len in (0, 1, 4096, 1 << 20):
            for verify_width in (1, 8):
                for overlap_depth in (0, 1):
                    assert (
                        compute_max_logical_pages_for_capture(
                            state,
                            max_context_len=context_len,
                            max_tokens_per_req=verify_width,
                            overlap_schedule_depth=overlap_depth,
                        )
                    ) == (STATE_TABLE_SLOTS)
        with pytest.raises((ValueError)):
            compute_max_logical_pages_for_capture(
                state, max_context_len=8, max_tokens_per_req=0
            )

    def test_overlap_sizing_rejects_invalid_runtime_parameters(self, subtests):
        spec = CacheGroupSpec(
            group_id="full",
            attn_kind="full_history",
            rows_per_page=4,
            entry_stride_tokens=1,
            sliding_window_tokens=None,
            replayable=False,
        )
        count_args = {
            "prefix_granularity": 4,
            "max_live_requests": 1,
            "max_scheduled_tokens": 8,
            "max_total_tokens": 8,
            "max_context_len": 8,
            "decode_input_tokens": 1,
            "overlap_schedule_depth": 0,
        }
        # The scheduler's own validation, surfaced as ValueError by the binding.
        for overrides, message in (
            ({"decode_input_tokens": -1}, "decode_input_tokens"),
            ({"overlap_schedule_depth": 2}, "overlap_schedule_depth"),
            (
                {"decode_input_tokens": 0, "overlap_schedule_depth": 1},
                "decode_input_tokens",
            ),
            ({"max_total_tokens": -1}, "max_total_tokens"),
        ):
            with (
                subtests.test(function="page_counts", overrides=overrides),
                pytest.raises((ValueError), match=(message)),
            ):
                compute_cache_group_pages([spec], **{**count_args, **overrides})

        for overrides, message in (
            ({"max_context_len": -1}, "max_context_len"),
            ({"max_tokens_per_req": 0}, "max_tokens_per_req"),
            ({"overlap_schedule_depth": 2}, "overlap_schedule_depth"),
        ):
            with (
                subtests.test(function="capture_width", overrides=overrides),
                pytest.raises((ValueError), match=(message)),
            ):
                compute_max_logical_pages_for_capture(
                    spec,
                    **{
                        "max_context_len": 8,
                        "max_tokens_per_req": 1,
                        **overrides,
                    },
                )

        # Bad row geometry, a non-positive window and an unknown kind are all
        # rejected at spec construction.
        invalid_specs = (
            (("bad-rows", "full_history", 0, 1, None), "rows_per_page"),
            (("bad-window", "sliding_window", 4, 1, 0), "sliding_window_tokens"),
            (("bad-kind", "unknown", 4, 1, None), "unknown cache-group kind"),
        )
        for args, message in invalid_specs:
            with (
                subtests.test(group=args[0]),
                pytest.raises((ValueError), match=(message)),
            ):
                CacheGroupSpec(*args, replayable=False)

    def test_overlap_schedule_enablement_truth_table(self, subtests):
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
            with subtests.test(disabled=disabled, mode=mode):
                assert (
                    should_use_overlap_schedule(
                        disable_overlap_schedule=disabled,
                        disaggregation_mode=mode,
                    )
                ) == (expected)

    def test_sliding_window_scheduled_tokens_are_global_and_capped(self):
        specs = [
            CacheGroupSpec(
                group_id="sliding",
                attn_kind="sliding_window",
                rows_per_page=4,
                entry_stride_tokens=1,
                sliding_window_tokens=8,
                replayable=False,
            )
        ]

        pages = compute_cache_group_pages(
            specs,
            prefix_granularity=4,
            max_live_requests=10,
            max_scheduled_tokens=100,
            max_total_tokens=20,
            max_context_len=4096,
            decode_input_tokens=1,
            overlap_schedule_depth=0,
        )

        # Each request retains its 7-token window plus the decode token at any
        # page alignment, and with the prefix cache on hands its partial page
        # to the cache at the prompt end while continuing on a fresh one (one
        # more block per request, ceil((P - 1) / g), within the lookback); one
        # chunk in flight -- capped by the total -- adds its rows behind the
        # resumable-boundary lookback, which spans the most blocks a window can
        # cover at any alignment: ceil((W - 1 + g - 1) / g).
        resident_pages = 10 * math.ceil((7 + 1 + 3) / 4)
        lookback_pages = math.ceil((7 + 3) / 4)
        handover_pages = 10 * min(math.ceil((4 - 1) / 4), lookback_pages)
        scheduled_pages = math.ceil(20 / 4)
        assert (pages["sliding"]) == (
            resident_pages + handover_pages + lookback_pages + scheduled_pages
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
