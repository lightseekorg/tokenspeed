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

from collections.abc import Mapping, Sequence
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
    # Per-group page tokens; None -> scheduler global block_size, else a multiple of it.
    block_size: int | None = None
    # Physical child CacheBlocks packed into one shared LCM parent.
    cache_blocks_per_lcm_block: int = 1
    # None preserves standalone/non-PD behavior; PD plans set this explicitly.
    transfer_policy: TransferPolicy | None = None


_PAGED_CACHE_GROUP_DUMMY_PAGES = 1


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

# Sliding sub-groups make each slab bound by one layer of every group — no dead slab rows.
_SLIDING_SUBGROUP_PREFIX = "sliding_attention_"


def _retention_for_label(label: str) -> Retention | None:
    """Retention for a paged-cache layer_type label, or None if unknown.

    Args:
        label: A layer_type label — one of the exact vocabulary in
            ``_LAYER_TYPE_RETENTION``, or a sliding sub-group label
            ``sliding_attention_<k>`` (k a decimal index).

    Returns:
        The label's retention, or None for labels outside the vocabulary.
    """
    retention = _LAYER_TYPE_RETENTION.get(label)
    if retention is not None:
        return retention
    if (
        label.startswith(_SLIDING_SUBGROUP_PREFIX)
        and label[len(_SLIDING_SUBGROUP_PREFIX) :].isdigit()
    ):
        return "sliding_window"
    return None


def hybrid_slab_group_size(
    layer_types: Sequence[str] | None,
    *,
    sliding_window_tokens: int | Sequence[int | None] | None = None,
) -> int | None:
    """Slab count for the hybrid slab KV layout (the i-th layer of EACH
    group shares slab i), or None when the model cannot share slabs.

    Single source (canonical) for both the sizing divisor (registry KV
    profile) and the buffer layout (_create_buffers) -- the two must never
    disagree. The scheduler's single BlockPool owns each page id by at most
    one group, so paired layers' live rows never overlap. Unknown labels
    degrade to None -- the predicate gates an optimization, so it must not
    raise.

    Groups may be unequal in size (e.g. Inkling's 55 sliding + 11 full):
    the slab count is the LARGEST group's layer count; slabs beyond a
    smaller group's count are single-layer. Equal groups keep the original
    gpt-oss pairing (count == group size). Sliding sub-group labels
    ("sliding_attention_<k>") count as separate groups — equal-count
    sub-groups make every slab fully bound (Inkling 5x11 + 11 -> 11 slabs).

    Multi-window models (a per-layer window sequence with >1 distinct
    window) degrade to None: the slab pairing is per raw label, not per
    (retention, window) group.
    """
    if not layer_types:
        return None
    counts: dict[str, int] = {}
    for label in layer_types:
        # State rows are not byte-equal with KV rows, so no slab pairing.
        if _retention_for_label(label) is None or label in STATE_LAYER_TYPES:
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
            if _retention_for_label(label) == "sliding_window"
            and isinstance(w, int)
            and not isinstance(w, bool)
            and w > 0
        }
        if len(distinct) > 1:
            return None
    return max(counts.values())


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
        retention = _retention_for_label(label)
        if retention is None:
            raise ValueError(
                f"_layer_specs: unknown layer_type {label!r} at layer {i}; "
                f"expected one of {sorted(_LAYER_TYPE_RETENTION)} or a "
                f"sliding sub-group label '{_SLIDING_SUBGROUP_PREFIX}<k>'"
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
    ``block_tables`` keys line up with the published group specs."""
    return [gid for gid, _, _ in _layer_specs(layer_types, sliding_window_tokens)]


def split_recurrent_state_groups(layer_types: Sequence[str]) -> list[str]:
    """Split each recurrent run by its position between history layers.

    Qwen3.5 repeats ``state-0, state-1, state-2, full``. The three state
    positions have independent physical layouts while repeated occurrences of
    one position remain one cache group.
    """
    group_ids = []
    state_position = 0
    for label in layer_types:
        if label in STATE_LAYER_TYPES:
            group_ids.append(f"{label}_{state_position}")
            state_position += 1
        else:
            group_ids.append(label)
            state_position = 0
    return group_ids


def group_specs_from_layer_types(
    *,
    layer_types: Sequence[str],
    group_ids: Sequence[str] | None = None,
    sliding_window_tokens: int | Sequence[int | None] | None,
    page_size: int,
    page_sizes: Mapping[str, int] | None = None,
    cache_blocks_per_lcm_block: Mapping[str, int] | None = None,
) -> list[PagedCacheGroupSpec]:
    """Derive paged-cache group specs from per-layer attention types.

    vLLM-style spec-value grouping: layers collapse into one group per
    distinct (retention, window). Group order = first-appearance order.

    Args:
        layer_types: Per-layer labels: "full_attention" / "sliding_attention"
            (or sliding sub-group labels "sliding_attention_<k>") /
            "linear_attention" (state-family, e.g. Qwen3.5 GDN).
        group_ids: Optional physical group id per layer. Retention and family
            still come from ``layer_types``.
        sliding_window_tokens: One window for all sliding layers (today's HF
            scalar), or a per-layer sequence (multi-window models; full-layer
            positions must be None).
        page_size: Tokens per page (the scheduler's base block size).
        page_sizes: Per-group page sizes keyed by group id (heterogeneous
            block sizes); values must be positive multiples of page_size.
            Groups not listed use page_size.
        cache_blocks_per_lcm_block: Per-group physical packing. Omitted groups
            use one CacheBlock per physical parent.

    Raises:
        ValueError: unknown label; window sequence length mismatch; sliding
            layer without a positive window; full layer carrying a window.
    """
    sizes = dict(page_sizes or {})
    for gid, ps in sizes.items():
        if ps <= 0 or ps % page_size:
            raise ValueError(
                f"page_sizes[{gid!r}] = {ps} must be a positive "
                f"multiple of page_size {page_size}"
            )
    packing = dict(cache_blocks_per_lcm_block or {})
    for gid, count in packing.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(
                f"cache_blocks_per_lcm_block[{gid!r}] = {count!r} must be "
                "a positive int"
            )
    layer_specs = _layer_specs(layer_types, sliding_window_tokens)
    resolved_group_ids = (
        list(group_ids) if group_ids is not None else [gid for gid, _, _ in layer_specs]
    )
    if len(resolved_group_ids) != len(layer_types):
        raise ValueError(
            f"group_ids has {len(resolved_group_ids)} entries but layer_types "
            f"has {len(layer_types)}"
        )

    specs: list[PagedCacheGroupSpec] = []
    seen: dict[str, tuple[Retention, int | None, Family]] = {}
    for layer_id, ((_, retention, window), gid) in enumerate(
        zip(layer_specs, resolved_group_ids)
    ):
        if not gid:
            raise ValueError(f"group_ids[{layer_id}] must be non-empty")
        family: Family = (
            "state" if layer_types[layer_id] in STATE_LAYER_TYPES else "history"
        )
        if gid in seen:
            if seen[gid] != (retention, window, family):
                raise ValueError(f"group_id {gid!r} mixes incompatible layer policies")
            continue
        seen[gid] = (retention, window, family)
        ps = sizes.pop(gid, None) or page_size
        group_packing = packing.pop(gid, 1)
        specs.append(
            PagedCacheGroupSpec(
                group_id=gid,
                retention=retention,
                rows_per_page=ps,
                entry_stride_tokens=1,
                sliding_window_tokens=window,
                family=family,
                # Always explicit: unset groups inherit the C++ gcd base, which a finer extra group silently lowers.
                block_size=ps,
                cache_blocks_per_lcm_block=group_packing,
            )
        )
    if sizes:
        raise ValueError(f"page_sizes for unknown groups: {sorted(sizes)}")
    if packing:
        raise ValueError(
            "cache_blocks_per_lcm_block for unknown groups: " f"{sorted(packing)}"
        )
    return specs


def publish_paged_cache_groups(
    *,
    layer_types: Sequence[str],
    group_ids: Sequence[str] | None = None,
    sliding_window_tokens: int | Sequence[int | None] | None,
    page_size: int,
    page_sizes: Mapping[str, int] | None = None,
    cache_blocks_per_lcm_block: Mapping[str, int] | None = None,
    extra_groups: Sequence[PagedCacheGroupSpec] = (),
    max_live_requests: int,
    max_scheduled_tokens: int,
    max_total_tokens: int,
    max_context_len: int,
) -> tuple[list[PagedCacheGroupSpec], dict[str, int]]:
    """Publication rule (canonical) for a KV pool's paged-cache groups.

    Every cache pool publishes its scheduler groups. Speculative decoding is
    supported: verify writes per-group [bs*N] locations and the drafter
    consumes the full-attention group's table (mirrored into req_to_page each
    step).

    Args:
        layer_types: Per-layer paged-cache labels (empty -> single
            full-history group).
        sliding_window_tokens / page_size: Forwarded to
            group_specs_from_layer_types.
        max_live_requests / max_scheduled_tokens / max_total_tokens /
            max_context_len: Sizing inputs for
            compute_paged_cache_group_page_counts.

    Returns:
        The group specs and their page counts.
    """
    packing = dict(cache_blocks_per_lcm_block or {})
    for spec in extra_groups:
        planned = packing.pop(spec.group_id, None)
        if planned is not None and planned != spec.cache_blocks_per_lcm_block:
            raise ValueError(
                f"extra_groups[{spec.group_id!r}] packing "
                f"{spec.cache_blocks_per_lcm_block} does not match LCM plan "
                f"{planned}"
            )
    specs = group_specs_from_layer_types(
        layer_types=tuple(layer_types) or (FULL_ATTENTION,),
        group_ids=group_ids,
        sliding_window_tokens=sliding_window_tokens,
        page_size=page_size,
        page_sizes=page_sizes,
        cache_blocks_per_lcm_block=packing,
    )
    # Model-declared groups outside the layer-type vocabulary (e.g. paged sconv columns).
    for spec in extra_groups:
        if any(sp.group_id == spec.group_id for sp in specs):
            raise ValueError(f"extra_groups: duplicate group id {spec.group_id!r}")
        # Smaller extra-group blocks lower the gcd base by design; MakeCoordinator asserts divisibility.
        if spec.block_size is not None and spec.block_size <= 0:
            raise ValueError(f"extra_groups[{spec.group_id!r}]: block_size must be > 0")
        if spec.cache_blocks_per_lcm_block <= 0:
            raise ValueError(
                f"extra_groups[{spec.group_id!r}]: "
                "cache_blocks_per_lcm_block must be > 0"
            )
        specs.append(spec)
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
    "split_recurrent_state_groups",
    "validate_scheduler_config",
]
