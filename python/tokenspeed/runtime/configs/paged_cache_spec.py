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


# Paged-cache label vocabulary (NOT the HF checkpoint's serialized enum:
# Qwen3.5 checkpoints spell full attention "attention").
FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"

# layer_type label -> retention. GPT-OSS uses the first two, Qwen3.5 GDN
# layers use "linear_attention"; unknown labels raise.
_LAYER_TYPE_RETENTION: dict[str, Retention] = {
    FULL_ATTENTION: "full_history",
    "sliding_attention": "sliding_window",
    # State groups ride full_history retention: the C++ side keys the
    # mamba-state kind on family == State && retention != SlidingWindow.
    LINEAR_ATTENTION: "full_history",
}

# Labels whose group is state-family (recurrent state rows, not KV history).
STATE_LAYER_TYPES = frozenset({LINEAR_ATTENTION})


def hybrid_slab_group_size(
    layer_types: Sequence[str] | None,
    *,
    speculative_enabled: bool,
    sliding_window_tokens: int | Sequence[int | None] | None = None,
) -> int | None:
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
        if label not in _LAYER_TYPE_RETENTION or label in STATE_LAYER_TYPES:
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


def _flat_kv_backend(attn_backend: object) -> object:
    """The backend whose KV-table consumption matters for flat safety: the
    backend itself, or a composite's full-attention sub-backend (hybrid's
    per-layer KV routing lives there and is user-selectable). The linear
    side consumes only the state group's table through its own explicit
    flat path and is out of scope here.
    """
    sub = getattr(attn_backend, "full_attn_backend", None)
    if sub is not None:
        return _flat_kv_backend(sub)
    return attn_backend


def validate_flat_scheduler_config(
    *,
    flat_kvcache_ext: bool,
    paged_cache_groups: Sequence[object],
    attn_backend: object,
    kv_pool: object,
    speculative_enabled: bool,
) -> None:
    """Fail fast, before the C++ ``Scheduler`` ctor, when a flat-built ext
    cannot drive this setup: a backend (or hybrid sub-backend) that would not
    consume the per-group flat tables, or zero published groups. No-op on a
    radix build.
    """
    if not flat_kvcache_ext:
        return
    pool_name = type(kv_pool).__name__
    backend = _flat_kv_backend(attn_backend)
    backend_name = type(backend).__name__
    uses_paged = bool(getattr(backend, "uses_paged_cache_groups", False))
    uses_flat = bool(getattr(backend, "uses_flat_cache_groups", False))
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
    if len(paged_cache_groups) > 1 and not uses_flat:
        # A table-blind backend on a multi-group pool would index every
        # layer through the C++ single-table fallback (a first-group
        # sample) — with slab-aliased layouts that silently corrupts KV
        # past the sliding window. Refuse at startup instead.
        raise RuntimeError(
            "flat scheduler build (TOKENSPEED_FLAT_KVCACHE): KV pool "
            f"{pool_name} publishes {len(paged_cache_groups)} cache groups "
            f"but attention backend {backend_name} does not consume flat "
            "per-group tables (uses_flat_cache_groups=False); the single-"
            "table fallback would serve one group's pages to every layer, "
            "silently corrupting KV. Pick a flat-capable attention backend "
            "or use a radix-built tokenspeed_scheduler extension."
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
                "Use a radix-built tokenspeed_scheduler extension for this " "model."
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
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
    safety_margin: int = 0,
) -> dict[str, int]:
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
                + protected_pages
                + _PAGED_CACHE_GROUP_DUMMY_PAGES
                + safety_margin
            )
            state_total = (
                max_live_requests * 2
                + max_total_tokens // raw_per_page
                + protected_pages
                + _PAGED_CACHE_GROUP_DUMMY_PAGES
                + safety_margin
            )
            total = min(state_total, full_history_total)
        elif spec.retention == "full_history":
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
            # Capacity tracks resident history before the next token.
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
            (
                f"{label}_{window}"
                if multi_window and retention == "sliding_window"
                else label
            ),
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
                family="state" if gid in STATE_LAYER_TYPES else "history",
            )
        )
    return specs


def publish_paged_cache_groups(
    *,
    layer_types: Sequence[str],
    sliding_window_tokens: int | Sequence[int | None] | None,
    page_size: int,
    speculative_enabled: bool,
    max_live_requests: int,
    max_scheduled_tokens: int,
    max_total_tokens: int,
    max_context_len: int,
) -> tuple[list[PagedCacheGroupSpec], dict[str, int]] | None:
    """Publication rule (canonical) for a KV pool's paged-cache groups.

    Publish groups iff the scheduler ext is flat-built (a radix ext never
    delivers flat tables — capture would bind dead buffers) and spec decode
    is off (flat tables do not support spec-expanded metadata; non-empty
    groups would also disable the overlap scheduler under spec). Publication
    is THE upstream signal every flat consumer keys off.
    TODO(flat+spec): publish under spec.

    Args:
        layer_types: Per-layer paged-cache labels (empty -> single
            full-history group).
        sliding_window_tokens / page_size: Forwarded to
            group_specs_from_layer_types.
        speculative_enabled: Gates publication off.
        max_live_requests / max_scheduled_tokens / max_total_tokens /
            max_context_len: Sizing inputs for
            compute_paged_cache_group_page_counts.

    Returns:
        (specs, page_counts) when publishing, None when publication is
        gated off (radix ext or spec decode).
    """
    if speculative_enabled or not scheduler_ext_flat_kvcache():
        return None
    specs = group_specs_from_layer_types(
        layer_types=tuple(layer_types) or (FULL_ATTENTION,),
        sliding_window_tokens=sliding_window_tokens,
        page_size=page_size,
    )
    counts = compute_paged_cache_group_page_counts(
        specs,
        max_live_requests=max_live_requests,
        max_scheduled_tokens=max(0, int(max_scheduled_tokens)),
        max_total_tokens=max_total_tokens,
        max_context_len=max_context_len,
    )
    return specs, counts


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
    # Local import: keeps this module torch-free at import time.
    from tokenspeed.runtime.utils.common import ceil_div

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
        # Capture uses a conservative metadata bound; it does not change the
        # per-token attention history counted as window - 1 above.
        retention_bound = min(window, max_context_len)
        live_tokens = retention_bound + reservation_horizon
        return ceil_div(live_tokens, raw_per_page) + 1
    if spec.retention == "full_history":
        live_tokens = max_context_len + reservation_horizon
        return ceil_div(live_tokens, raw_per_page)
    raise ValueError(
        f"PagedCacheGroupSpec {spec.group_id}: unsupported retention "
        f"{spec.retention!r}"
    )


__all__ = [
    "FULL_ATTENTION",
    "LINEAR_ATTENTION",
    "PagedCacheGroupSpec",
    "Retention",
    "STATE_LAYER_TYPES",
    "compute_max_logical_pages_for_capture",
    "compute_paged_cache_group_page_counts",
    "group_specs_from_layer_types",
    "hybrid_slab_group_size",
    "layer_group_ids",
    "publish_paged_cache_groups",
    "scheduler_ext_flat_kvcache",
    "validate_flat_scheduler_config",
]
