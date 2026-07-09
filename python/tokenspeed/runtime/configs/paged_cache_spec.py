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
from typing import Dict, Literal, Optional

Retention = Literal["full_history", "sliding_window"]
Family = Literal["history", "state"]


@dataclass(frozen=True)
class PagedCacheGroupSpec:
    group_id: str
    retention: Retention
    rows_per_page: int
    entry_stride_tokens: int
    sliding_window_tokens: Optional[int]
    # History groups form a chain; State groups only need the trailing window.
    family: Family = "history"


_PAGED_CACHE_GROUP_DUMMY_PAGES = 1


def scheduler_ext_flat_kvcache() -> bool:
    """True iff the installed tokenspeed_scheduler ext was built with
    TOKENSPEED_FLAT_KVCACHE. A missing package or an older / radix-built
    ext reports False — the radix-safe default (never delivers flat tables).
    """
    try:
        # Local import: module must stay importable without the compiled ext.
        import tokenspeed_scheduler
    except ImportError:
        return False
    return bool(getattr(tokenspeed_scheduler, "FLAT_KVCACHE", False))


# layer_type label -> retention. GPT-OSS uses the first two, Qwen3.5 GDN
# layers use "linear_attention"; unknown labels raise.
_LAYER_TYPE_RETENTION: Dict[str, Retention] = {
    "full_attention": "full_history",
    "sliding_attention": "sliding_window",
    # State groups ride full_history retention: the C++ side keys the
    # mamba-state kind on family == State && retention != SlidingWindow.
    "linear_attention": "full_history",
}

# Labels whose group is state-family (recurrent state rows, not KV history).
_STATE_LAYER_TYPES = {"linear_attention"}


def hybrid_slab_group_size(
    layer_types: Optional[Sequence[str]],
    *,
    speculative_enabled: bool,
    sliding_window_tokens: int | Sequence[int | None] | None = None,
) -> Optional[int]:
    """Group size for the hybrid slab KV layout (one layer of EACH group
    shares a K/V slab), or None to keep the legacy per-layer layout.

    Single source (canonical) for both the sizing divisor (registry KV
    profile) and the buffer layout (_create_buffers) -- the two must never
    disagree. Safe only with the flat ext (its single BlockPool owns each
    page id by at most one group, so paired layers' live rows never
    overlap) and equal group sizes. Unknown labels degrade to None -- the
    predicate gates an optimization, so it must not raise.

    Multi-window models (a per-layer window sequence with >1 distinct
    window) degrade to None: the slab pairing is per raw label, not per
    (retention, window) group.
    """
    if speculative_enabled or not scheduler_ext_flat_kvcache():
        return None
    if not layer_types:
        return None
    counts: dict[str, int] = {}
    for label in layer_types:
        # State rows are not byte-equal with KV rows, so no slab pairing.
        if label not in _LAYER_TYPE_RETENTION or label in _STATE_LAYER_TYPES:
            return None
        counts[label] = counts.get(label, 0) + 1
    if len(counts) < 2:
        return None
    if sliding_window_tokens is not None and not isinstance(sliding_window_tokens, int):
        if not isinstance(sliding_window_tokens, Sequence) or len(
            sliding_window_tokens
        ) != len(layer_types):
            return None
        distinct = {
            w
            for label, w in zip(layer_types, sliding_window_tokens)
            if _LAYER_TYPE_RETENTION[label] == "sliding_window"
            and isinstance(w, int)
            and not isinstance(w, bool)
            and w > 0
        }
        if len(distinct) > 1:
            return None
    sizes = set(counts.values())
    if len(sizes) != 1:
        return None
    return sizes.pop()


def validate_flat_scheduler_config(
    *,
    flat_kvcache_ext: bool,
    paged_cache_groups: Sequence[object],
    attn_backend: object,
    kv_pool: object,
    speculative_enabled: bool,
) -> None:
    """Fail fast, before the C++ ``Scheduler`` ctor, when a flat-built ext
    cannot drive this setup: a paged-groups backend that is not flat-group
    capable, or zero published groups. No-op on a radix build.
    """
    if not flat_kvcache_ext:
        return
    backend_name = type(attn_backend).__name__
    pool_name = type(kv_pool).__name__
    uses_paged = bool(getattr(attn_backend, "uses_paged_cache_groups", False))
    uses_flat = bool(getattr(attn_backend, "uses_flat_cache_groups", False))
    if uses_paged and not uses_flat:
        raise RuntimeError(
            "flat scheduler build (TOKENSPEED_FLAT_KVCACHE) does not support "
            f"this model's cache layout yet: attention backend {backend_name} "
            f"(KV pool {pool_name}) consumes paged-cache groups through the "
            "radix scheduler's populate path, which the flat build compiles "
            "out — CUDA graphs would silently replay against stale capture "
            "placeholders. Use a radix-built tokenspeed_scheduler extension "
            "for this model."
        )
    if not paged_cache_groups:
        if speculative_enabled:
            cause = (
                "speculative decoding is enabled, which gates paged-cache "
                "group publication off"
            )
            action = (
                "Disable speculative decoding or use a radix-built "
                "tokenspeed_scheduler extension."
            )
        else:
            cause = (
                f"KV pool {pool_name} publishes no paged-cache groups (e.g. "
                "mamba/state-only pools)"
            )
            action = (
                "Use a radix-built tokenspeed_scheduler extension for this "
                "model."
            )
        raise RuntimeError(
            "flat scheduler build (TOKENSPEED_FLAT_KVCACHE) requires at least "
            f"one paged-cache group, but {cause}. {action}"
        )


def compute_paged_cache_group_page_counts(
    specs: Sequence[PagedCacheGroupSpec],
    *,
    max_live_requests: int,
    max_scheduled_tokens: int,
    max_total_tokens: int,
    max_context_len: int,
    safety_margin: int = 0,
) -> Dict[str, int]:
    # Local import: keeps this module torch-free at import time.
    from tokenspeed.runtime.utils.common import ceil_div

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
    if safety_margin < 0:
        raise ValueError(f"safety_margin must be >= 0, got {safety_margin}")

    counts: Dict[str, int] = {}
    for spec in specs:
        raw_per_page = spec.rows_per_page * spec.entry_stride_tokens
        if raw_per_page <= 0:
            raise ValueError(
                f"PagedCacheGroupSpec {spec.group_id}: rows_per_page * "
                "entry_stride_tokens must be > 0"
            )
        # Mamba-state kind = family "state" AND retention != sliding_window
        # (the C++ side keys it the same way); V4's sliding-window state tail
        # buffers keep the sliding-window formula below.
        if spec.family == "state" and spec.retention == "full_history":
            # State group: 2 live pages/request (the W=2 write window) +
            # floor(T/P) snapshot pages (snapshots are bounded by the shared
            # page-id space), capped at the full-history count.
            full_history_total = (
                ceil_div(max_total_tokens, raw_per_page)
                + max_live_requests
                + _PAGED_CACHE_GROUP_DUMMY_PAGES
                + safety_margin
            )
            state_total = (
                max_live_requests * 2
                + max_total_tokens // raw_per_page
                + _PAGED_CACHE_GROUP_DUMMY_PAGES
                + safety_margin
            )
            total = min(state_total, full_history_total)
        elif spec.retention == "full_history":
            full_pages = ceil_div(max_total_tokens, raw_per_page)
            total = (
                full_pages
                + max_live_requests
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


def _layer_specs(
    layer_types: Sequence[str],
    sliding_window_tokens: int | Sequence[int | None] | None,
) -> list[tuple[str, Retention, int | None]]:
    """Per-layer (group_id, retention, window). group_id is the bare label
    unless sliding layers carry more than one distinct window (then
    label_<window>), so single-window models keep byte-identical ids.
    A scalar window broadcasts to sliding layers; a sequence lines up 1:1."""
    if isinstance(sliding_window_tokens, str):
        raise ValueError(
            "_layer_specs: sliding_window_tokens must be None, an int, or a "
            f"sequence of int/None, got {sliding_window_tokens!r}"
        )
    if sliding_window_tokens is None or isinstance(sliding_window_tokens, int):
        if isinstance(sliding_window_tokens, bool):
            raise ValueError(
                "_layer_specs: sliding_window_tokens must be None, an int, or "
                f"a sequence of int/None, got {sliding_window_tokens!r}"
            )
        windows: list[int | None] = [sliding_window_tokens] * len(layer_types)
        scalar = True
    elif not isinstance(sliding_window_tokens, Sequence):
        raise ValueError(
            "_layer_specs: sliding_window_tokens must be None, an int, or a "
            f"sequence of int/None, got {sliding_window_tokens!r}"
        )
    else:
        windows = list(sliding_window_tokens)
        scalar = False
        if len(windows) != len(layer_types):
            raise ValueError(
                f"_layer_specs: sliding_window_tokens has {len(windows)} "
                f"entries but layer_types has {len(layer_types)}"
            )
    rows: list[tuple[str, Retention, int | None]] = []
    for i, (label, raw) in enumerate(zip(layer_types, windows)):
        retention = _LAYER_TYPE_RETENTION.get(label)
        if retention is None:
            raise ValueError(
                f"_layer_specs: unknown layer_type {label!r} at layer {i}; "
                f"expected one of {sorted(_LAYER_TYPE_RETENTION)}"
            )
        if raw is not None and (isinstance(raw, bool) or not isinstance(raw, int)):
            raise ValueError(
                f"_layer_specs: layer {i} ({label!r}) window must be None or "
                f"an int, got {raw!r}"
            )
        window = raw
        if retention == "sliding_window":
            if window is None or window <= 0:
                raise ValueError(
                    f"_layer_specs: layer {i} ({label!r}) is sliding but its "
                    f"window is not a positive int (got {raw!r})"
                )
        else:
            if not scalar and window is not None and window > 0:
                raise ValueError(
                    f"_layer_specs: layer {i} ({label!r}) is full-history but "
                    f"carries sliding window {window}; mislabeled layer_type?"
                )
            window = None
        rows.append((label, retention, window))
    distinct = {w for _, r, w in rows if r == "sliding_window"}
    multi_window = len(distinct) > 1
    return [
        (
            f"{label}_{window}"
            if multi_window and retention == "sliding_window"
            else label,
            retention,
            window,
        )
        for label, retention, window in rows
    ]


def layer_group_ids(
    *,
    layer_types: Sequence[str],
    sliding_window_tokens: int | Sequence[int | None] | None,
) -> list[str]:
    """Per-layer paged-cache group id — the single source multi-window models
    will assign ``PagedAttention(group_id=...)`` from (today gpt_oss.py
    assigns group_id=layer_type, identical in the single-window case), so
    ``flat_block_tables`` keys line up with the published group specs."""
    return [gid for gid, _, _ in _layer_specs(layer_types, sliding_window_tokens)]


def group_specs_from_layer_types(
    *,
    layer_types: Sequence[str],
    sliding_window_tokens: int | Sequence[int | None] | None,
    page_size: int,
) -> list[PagedCacheGroupSpec]:
    """Derive paged-cache group specs from per-layer attention types.

    vLLM-style spec-value grouping: layers collapse into one group per
    distinct (retention, window). Group order = first-appearance order.

    Args:
        layer_types: Per-layer labels: "full_attention" / "sliding_attention"
            / "linear_attention" (state-family, e.g. Qwen3.5 GDN).
        sliding_window_tokens: One window for all sliding layers (today's HF
            scalar), or a per-layer sequence (multi-window models; full-layer
            positions must be None).
        page_size: Tokens per page (uniform across groups).

    Raises:
        ValueError: unknown label; window sequence length mismatch; sliding
            layer without a positive window; full layer carrying a window.
    """
    specs: list[PagedCacheGroupSpec] = []
    seen: set[str] = set()
    for gid, retention, window in _layer_specs(layer_types, sliding_window_tokens):
        if gid in seen:
            continue
        seen.add(gid)
        specs.append(
            PagedCacheGroupSpec(
                group_id=gid,
                retention=retention,
                rows_per_page=page_size,
                entry_stride_tokens=1,
                sliding_window_tokens=window,
                family="state" if gid in _STATE_LAYER_TYPES else "history",
            )
        )
    return specs


__all__ = [
    "PagedCacheGroupSpec",
    "Retention",
    "compute_paged_cache_group_page_counts",
    "group_specs_from_layer_types",
    "hybrid_slab_group_size",
    "layer_group_ids",
    "scheduler_ext_flat_kvcache",
    "validate_flat_scheduler_config",
]
