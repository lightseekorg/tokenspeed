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

Retention = Literal["full_history", "sliding_window"]
Family = Literal["history", "state"]
TransferPolicy = Literal["full_suffix", "latest_snapshot"]


@dataclass(frozen=True)
class PagedCacheGroupSpec:
    group_id: str
    retention: Retention
    rows_per_page: int
    entry_stride_tokens: int
    sliding_window_tokens: int | None
    # History groups form a chain; State groups only need the trailing window.
    family: Family = "history"
    # Physical child CacheBlocks packed into one shared LCM parent.
    cache_blocks_per_lcm_block: int = 1
    # None preserves standalone/non-PD behavior; PD plans set this explicitly.
    transfer_policy: TransferPolicy | None = None

    @property
    def cache_block_tokens(self) -> int:
        """Raw-token span represented by one CacheBlock in this group."""
        return self.rows_per_page * self.entry_stride_tokens


_PAGED_CACHE_GROUP_DUMMY_PAGES = 1


def _ceil_div(dividend: int, divisor: int) -> int:
    return (dividend + divisor - 1) // divisor


# Paged-cache label vocabulary (NOT the HF checkpoint's serialized enum:
# Qwen3.5 checkpoints spell full attention "attention").
FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"

# Labels whose group is state-family (recurrent state rows, not KV history).
STATE_LAYER_TYPES = frozenset({LINEAR_ATTENTION})


def _kv_backend(attn_backend: object) -> object:
    """The backend whose KV-table consumption matters for scheduler safety: the
    backend itself, or a composite's full-attention sub-backend (hybrid's
    per-layer KV routing lives there and is user-selectable). The linear
    side consumes only the state group's table through its own explicit
    path and is out of scope here.
    """
    sub = getattr(attn_backend, "full_attn_backend", None)
    if sub is not None:
        return _kv_backend(sub)
    return attn_backend


def validate_scheduler_config(
    *,
    paged_cache_groups: Sequence[object],
    attn_backend: object,
    kv_pool: object,
    speculative_algorithm: str | None = None,
) -> None:
    """Validate the cache-group contract before constructing the scheduler.

    A single-group backend may consume the scheduler's compatibility table.
    Multi-group pools must consume the explicit per-group tables.
    """
    contract = getattr(kv_pool, "runtime_contract", None)
    pool_name = type(kv_pool).__name__
    if speculative_algorithm is not None and not getattr(
        attn_backend, "cache_group_spec_capable", True
    ):
        raise RuntimeError(
            f"attention backend {type(attn_backend).__name__} does not support "
            "cache groups with speculative decoding"
        )
    backend = _kv_backend(attn_backend)
    backend_name = type(backend).__name__
    uses_cache_groups = bool(getattr(backend, "uses_cache_groups", False))
    if speculative_algorithm is not None and not getattr(
        backend, "cache_group_spec_capable", True
    ):
        raise RuntimeError(
            f"attention backend {backend_name} does not support cache groups "
            "with speculative decoding"
        )
    if len(paged_cache_groups) > 1 and not uses_cache_groups and contract is None:
        # A table-blind backend on a multi-group pool would index every
        # layer through the C++ single-table fallback (a first-group
        # sample) — with slab-aliased layouts that silently corrupts KV
        # past the sliding window. Refuse at startup instead.
        #
        # Contract pools are exempt from this flag check: their consumers
        # travel contract-specific batch metadata and are validated by the
        # family-coverage check below.
        # The MLA sub-backend's uses_cache_groups flag deliberately
        # stays False so DeepSeek keeps the single-table CUDA-graph
        # capture path that flag routes.
        raise RuntimeError(
            f"KV pool {pool_name} publishes {len(paged_cache_groups)} cache "
            f"groups but attention backend {backend_name} does not consume "
            "per-group tables (uses_cache_groups=False); the single-"
            "table fallback would serve one group's pages to every layer, "
            "silently corrupting KV"
        )
    if not paged_cache_groups:
        raise RuntimeError(
            "the cache-group scheduler requires at least one paged-cache "
            f"group, but KV pool {pool_name} publishes none"
        )
    if contract is not None:
        required_families = frozenset(spec.family for spec in contract.group_specs)
        supported_families = frozenset(
            getattr(attn_backend, "cache_consumer_families", ())
        )
        missing_families = required_families - supported_families
        if missing_families:
            raise RuntimeError(
                "paged cache pool requires consumer families "
                f"{sorted(required_families)}, but backend "
                f"{type(attn_backend).__name__} is missing "
                f"{sorted(missing_families)}"
            )


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
        cache_block_tokens = spec.cache_block_tokens
        if cache_block_tokens <= 0:
            raise ValueError(
                f"PagedCacheGroupSpec {spec.group_id}: cache_block_tokens "
                "(rows_per_page * entry_stride_tokens) must be > 0"
            )
        protected_pages = max_live_requests * _ceil_div(
            overlap_schedule_depth * decode_input_tokens, cache_block_tokens
        )
        # Mamba-state kind = family "state" AND retention != sliding_window
        # (the C++ side keys it the same way); V4's sliding-window state tail
        # buffers keep the sliding-window formula below.
        if spec.family == "state" and spec.retention == "full_history":
            # State group: 2 live pages/request (the W=2 write window) +
            # floor(T/P) snapshot pages (snapshots are bounded by the shared
            # page-id space), capped at the full-history count.
            full_history_total = (
                _ceil_div(max_total_tokens, cache_block_tokens)
                + max_live_requests
                + protected_pages
                + _PAGED_CACHE_GROUP_DUMMY_PAGES
                + safety_margin
            )
            state_total = (
                max_live_requests * 2
                + max_total_tokens // cache_block_tokens
                + protected_pages
                + _PAGED_CACHE_GROUP_DUMMY_PAGES
                + safety_margin
            )
            total = min(state_total, full_history_total)
        elif spec.retention == "full_history":
            full_pages = _ceil_div(max_total_tokens, cache_block_tokens)
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
            # Capacity tracks resident history before the next token.
            resident_tokens_per_req = min(max(window - 1, 0), max_context_len)
            resident_pages = max_live_requests * _ceil_div(
                resident_tokens_per_req, cache_block_tokens
            )
            scheduled_tokens = min(max_scheduled_tokens, max_total_tokens)
            scheduled_pages = _ceil_div(scheduled_tokens, cache_block_tokens)
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
    cache_block_tokens = spec.cache_block_tokens
    if cache_block_tokens <= 0:
        raise ValueError(
            f"PagedCacheGroupSpec {spec.group_id}: cache_block_tokens "
            "(rows_per_page * entry_stride_tokens) must be > 0"
        )
    reservation_horizon = (overlap_schedule_depth + 1) * max_tokens_per_req
    if spec.retention == "sliding_window":
        window = spec.sliding_window_tokens
        if window is None or window <= 0:
            raise ValueError(
                f"PagedCacheGroupSpec {spec.group_id}: sliding group missing "
                "positive sliding_window_tokens"
            )
        # Capture uses a conservative metadata bound; it does not change the
        # per-token attention history counted as window - 1 above.
        retention_bound = min(window, max_context_len)
        live_tokens = retention_bound + reservation_horizon
        return _ceil_div(live_tokens, cache_block_tokens) + 1
    if spec.retention == "full_history":
        live_tokens = max_context_len + reservation_horizon
        return _ceil_div(live_tokens, cache_block_tokens)
    raise ValueError(
        f"PagedCacheGroupSpec {spec.group_id}: unsupported retention {spec.retention!r}"
    )


__all__ = [
    "FULL_ATTENTION",
    "LINEAR_ATTENTION",
    "STATE_LAYER_TYPES",
    "PagedCacheGroupSpec",
    "Retention",
    "compute_max_logical_pages_for_capture",
    "compute_paged_cache_group_page_counts",
    "validate_scheduler_config",
]
