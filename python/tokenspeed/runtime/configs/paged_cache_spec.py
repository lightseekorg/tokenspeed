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

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence

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
    """Whether the installed tokenspeed_scheduler ext is a flat-KV-cache build.

    True iff the compiled extension was built with TOKENSPEED_FLAT_KVCACHE
    (flat KvCacheCoordinator scheduler). Failure-tolerant probe: a missing
    package, or an older / radix-built extension without the ``FLAT_KVCACHE``
    attribute, reports False. False must keep paged-cache group publication
    off — the radix scheduler never populates ``flat_block_tables``, so the
    flat CUDA-graph capture path must stay inactive against it.
    """
    try:
        # Local import: keeps this module importable without the compiled
        # scheduler extension (pure-spec unit tests).
        import tokenspeed_scheduler
    except ImportError:
        return False
    return bool(getattr(tokenspeed_scheduler, "FLAT_KVCACHE", False))


def hybrid_slab_group_size(
    layer_types: Optional[Sequence[str]],
    *,
    speculative_enabled: bool,
) -> Optional[int]:
    """Group size for the hybrid slab KV layout, or None to keep the legacy
    per-layer-buffer layout.

    The slab layout (M12) shares one K/V slab between one layer from EACH
    group (vLLM-style aliasing) and divides the memory budget by
    layers-per-group instead of total layers -- the byte-level capacity win
    for hybrid models. Safe only when (a) the flat scheduler ext is active
    (single BlockPool guarantees a page id is owned by at most one group at
    a time, so paired layers' live rows never overlap) and (b) every group
    has the SAME layer count (equal slab fan-out). Per-layer page bytes are
    uniform for MHA pools by construction (single head_num/head_dim/dtype).
    This predicate is the SINGLE source for both the sizing divisor
    (registry profile) and the buffer layout (_create_buffers) -- the two
    must never disagree.

    Unlike ``group_specs_from_layer_types`` (which raises on an unknown
    label, since spec publication must fail loudly), an unknown label here
    returns None: the predicate gates an optimization, so unrecognized
    input degrades to the safe legacy layout.
    """
    if speculative_enabled or not scheduler_ext_flat_kvcache():
        return None
    if not layer_types:
        return None
    counts: dict[str, int] = {}
    for label in layer_types:
        if label not in ("sliding_attention", "full_attention"):
            return None
        counts[label] = counts.get(label, 0) + 1
    if len(counts) < 2:
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
    """Fail fast when a flat-built scheduler ext cannot drive this runtime setup.

    Called at scheduler-config assembly, before the C++ ``Scheduler`` ctor. Two
    misconfigurations are rejected (both no-ops on a radix build, where groups
    are transport-only):

    1. A backend that consumes paged-cache groups through the radix scheduler's
       populate path but is not flat-group capable (``uses_paged_cache_groups``
       without ``uses_flat_cache_groups``, e.g. DeepSeek V4/MLA). The flat build
       compiles that path out and silently drops the specs' granularity fields,
       so CUDA graphs would replay against stale capture placeholders — garbage
       output with no error. The backend class flags are the same signals the
       CUDA-graph wrapper keys its capture/replay paths off, so gating on them
       rejects exactly the configs that would reach the broken path.
    2. Zero published groups (spec decode gates MHA publication off; state-only
       pools like mamba publish none). The flat Scheduler ctor would otherwise
       die inside MakeCoordinator with no hint at the actual knob.
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
    # Local import: keeps this module torch-free at import time so the pure
    # spec dataclasses + group_specs_from_layer_types load without torch.
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
        if spec.retention == "full_history":
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


# layer_type label -> retention. GPT-OSS uses these two; unknown labels raise.
_LAYER_TYPE_RETENTION: Dict[str, Retention] = {
    "full_attention": "full_history",
    "sliding_attention": "sliding_window",
}


def group_specs_from_layer_types(
    *,
    layer_types: Sequence[str],
    sliding_window_tokens: Optional[int],
    page_size: int,
) -> list[PagedCacheGroupSpec]:
    """Derive paged-cache group specs from a model's per-layer attention types.

    Mirrors vLLM's spec-value grouping: layers sharing an attention type
    collapse into one group. Group order = first-appearance order of the layer
    type. group_id is the layer-type label itself, so downstream
    ``flat_block_tables`` keys line up with it.

    Args:
        layer_types: Per-layer attention-type labels (e.g. from
            ``hf_config.layer_types``): ``"full_attention"`` /
            ``"sliding_attention"``.
        sliding_window_tokens: Window size for sliding layers; required (>0) when
            any ``"sliding_attention"`` layer is present, else may be None.
        page_size: Tokens per page; used as ``rows_per_page`` for every group
            (uniform page size across groups).

    Returns:
        One ``PagedCacheGroupSpec`` per distinct attention type, in
        first-appearance order.

    Raises:
        ValueError: on an unknown layer-type label, or a sliding layer without a
            positive ``sliding_window_tokens``.
    """
    specs: list[PagedCacheGroupSpec] = []
    seen: set[str] = set()
    for label in layer_types:
        if label in seen:
            continue
        retention = _LAYER_TYPE_RETENTION.get(label)
        if retention is None:
            raise ValueError(
                f"group_specs_from_layer_types: unknown layer_type {label!r}; "
                f"expected one of {sorted(_LAYER_TYPE_RETENTION)}"
            )
        window: Optional[int] = None
        if retention == "sliding_window":
            window = (
                None
                if sliding_window_tokens is None
                else int(sliding_window_tokens)
            )
            if window is None or window <= 0:
                raise ValueError(
                    f"group_specs_from_layer_types: layer_type {label!r} is "
                    "sliding but sliding_window_tokens is not a positive int "
                    f"(got {sliding_window_tokens!r})"
                )
        seen.add(label)
        specs.append(
            PagedCacheGroupSpec(
                group_id=label,
                retention=retention,
                rows_per_page=page_size,
                entry_stride_tokens=1,
                sliding_window_tokens=window,
                family="history",
            )
        )
    return specs


__all__ = [
    "PagedCacheGroupSpec",
    "Retention",
    "compute_paged_cache_group_page_counts",
    "group_specs_from_layer_types",
    "hybrid_slab_group_size",
    "scheduler_ext_flat_kvcache",
    "validate_flat_scheduler_config",
]
