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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from tokenspeed.runtime.utils.common import ceil_div

Retention = Literal["full_history", "sliding_window"]
Family = Literal["history", "state"]


@dataclass(frozen=True)
class PagedCacheGroupSpec:
    group_id: str
    retention: Retention
    rows_per_page: int
    entry_stride_tokens: int
    sliding_window_tokens: int | None
    # History groups form a chain; State groups only need the trailing window.
    family: Family = "history"


_PAGED_CACHE_GROUP_DUMMY_PAGES = 1


def compute_paged_cache_group_page_counts(
    specs: Sequence[PagedCacheGroupSpec],
    *,
    max_live_requests: int,
    max_scheduled_tokens: int,
    max_total_tokens: int,
    max_context_len: int,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
    safety_margin: int = 0,
) -> dict[str, int]:
    if max_live_requests < 0:
        raise ValueError(f"max_live_requests must be >= 0, got {max_live_requests}")
    if max_scheduled_tokens < 0:
        raise ValueError(
            f"max_scheduled_tokens must be >= 0, got {max_scheduled_tokens}"
        )
    if max_total_tokens < 0:
        raise ValueError(f"max_total_tokens must be >= 0, got {max_total_tokens}")
    if max_context_len < 0:
        raise ValueError(f"max_context_len must be >= 0, got {max_context_len}")
    if decode_input_tokens < 0:
        raise ValueError(f"decode_input_tokens must be >= 0, got {decode_input_tokens}")
    if overlap_schedule_depth not in (0, 1):
        raise ValueError(
            f"overlap_schedule_depth must be 0 or 1, got {overlap_schedule_depth}"
        )
    if overlap_schedule_depth > 0 and decode_input_tokens == 0:
        raise ValueError(
            "overlapped paged-cache sizing requires decode_input_tokens > 0"
        )
    if safety_margin < 0:
        raise ValueError(f"safety_margin must be >= 0, got {safety_margin}")

    counts: dict[str, int] = {}
    for spec in specs:
        raw_per_page = spec.rows_per_page * spec.entry_stride_tokens
        if raw_per_page <= 0:
            raise ValueError(
                f"PagedCacheGroupSpec {spec.group_id}: rows_per_page * "
                "entry_stride_tokens must be > 0"
            )
        protected_pages = max_live_requests * ceil_div(
            overlap_schedule_depth * decode_input_tokens, raw_per_page
        )
        if spec.retention == "full_history":
            full_pages = ceil_div(max_total_tokens, raw_per_page)
            total = (
                full_pages
                + max_live_requests
                + protected_pages
                + _PAGED_CACHE_GROUP_DUMMY_PAGES
                + safety_margin
            )
        elif spec.retention == "sliding_window":
            window = spec.sliding_window_tokens
            if window is None or window <= 0:
                raise ValueError(
                    f"PagedCacheGroupSpec {spec.group_id}: sliding group missing "
                    "positive sliding_window_tokens"
                )
            resident_tokens_per_req = min(max(window - 1, 0), max_context_len)
            resident_pages = max_live_requests * ceil_div(
                resident_tokens_per_req, raw_per_page
            )
            scheduled_tokens = min(max_scheduled_tokens, max_total_tokens)
            scheduled_pages = ceil_div(scheduled_tokens, raw_per_page)
            total = (
                resident_pages
                + scheduled_pages
                + max_live_requests
                + protected_pages
                + _PAGED_CACHE_GROUP_DUMMY_PAGES
                + safety_margin
            )
        else:
            raise ValueError(
                f"PagedCacheGroupSpec {spec.group_id}: unsupported retention "
                f"{spec.retention!r}"
            )
        counts[spec.group_id] = int(total)
    return counts


def compute_max_logical_pages_for_capture(
    spec: PagedCacheGroupSpec,
    *,
    max_context_len: int,
    max_tokens_per_req: int = 1,
    overlap_schedule_depth: int = 0,
) -> int:
    """Return CUDA Graph block-table width for one paged-cache group.

    Decode admission reserves the current verify span plus one span for each
    overlapped schedule.  Include that complete reservation horizon here: a
    request close to the model context limit can still expose the reserved
    pages in its scheduler block-table row before the accepted tokens are
    truncated by the request-length limit.

    Args:
        spec: Paged-cache group layout and retention policy.
        max_context_len: Maximum accepted raw-token context length.
        max_tokens_per_req: Runtime decode/verify width.
        overlap_schedule_depth: Number of additionally in-flight decode steps.

    Returns:
        Required block-table columns for one request.
    """
    if max_context_len < 0:
        raise ValueError(f"max_context_len must be >= 0, got {max_context_len}")
    if max_tokens_per_req <= 0:
        raise ValueError(f"max_tokens_per_req must be > 0, got {max_tokens_per_req}")
    if overlap_schedule_depth not in (0, 1):
        raise ValueError(
            f"overlap_schedule_depth must be 0 or 1, got {overlap_schedule_depth}"
        )
    raw_per_page = spec.rows_per_page * spec.entry_stride_tokens
    if raw_per_page <= 0:
        raise ValueError(
            f"PagedCacheGroupSpec {spec.group_id}: rows_per_page * "
            "entry_stride_tokens must be > 0"
        )
    reservation_horizon = (overlap_schedule_depth + 1) * max_tokens_per_req
    if spec.retention == "sliding_window":
        window = spec.sliding_window_tokens
        if window is None or window <= 0:
            raise ValueError(
                f"PagedCacheGroupSpec {spec.group_id}: sliding group missing "
                "positive sliding_window_tokens"
            )
        retained_history = min(window - 1, max_context_len)
        live_tokens = retained_history + reservation_horizon
        return ceil_div(live_tokens, raw_per_page) + 1
    if spec.retention == "full_history":
        live_tokens = max_context_len + reservation_horizon
        return ceil_div(live_tokens, raw_per_page)
    raise ValueError(
        f"PagedCacheGroupSpec {spec.group_id}: unsupported retention "
        f"{spec.retention!r}"
    )


__all__ = [
    "PagedCacheGroupSpec",
    "Retention",
    "compute_max_logical_pages_for_capture",
    "compute_paged_cache_group_page_counts",
]
