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


# layer_type label -> retention. GPT-OSS uses the first two, Qwen3.5 GDN
# layers use "linear_attention"; unknown labels raise.
_LAYER_TYPE_RETENTION: dict[str, Retention] = {
    FULL_ATTENTION: "full_history",
    "sliding_attention": "sliding_window",
    # State groups ride full_history retention: the C++ side keys the
    # mamba-state kind on family == State && retention != SlidingWindow.
    LINEAR_ATTENTION: "full_history",
    # DSA attends sparsely but retains the full KV history (the indexer
    # selects from it), so its layers publish as full-history groups.
    "deepseek_sparse_attention": "full_history",
}

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


def _layer_retention_windows(
    layer_types: Sequence[str],
    sliding_window_tokens: int | Sequence[int | None] | None,
) -> list[tuple[Retention, int | None]]:
    """Validate the per-layer labels/windows and return (retention, window)
    per layer. A scalar window broadcasts to sliding layers; a sequence
    lines up 1:1 (full-history positions must be None)."""
    if isinstance(sliding_window_tokens, str):
        raise ValueError(  # noqa: TRY004 - preserve the existing API contract
            "sliding_window_tokens must be None, an int, or a "
            f"sequence of int/None, got {sliding_window_tokens!r}"
        )
    if sliding_window_tokens is None or isinstance(sliding_window_tokens, int):
        if isinstance(sliding_window_tokens, bool):
            raise ValueError(
                "sliding_window_tokens must be None, an int, or "
                f"a sequence of int/None, got {sliding_window_tokens!r}"
            )
        windows: list[int | None] = [sliding_window_tokens] * len(layer_types)
        scalar = True
    elif not isinstance(sliding_window_tokens, Sequence):
        raise ValueError(
            "sliding_window_tokens must be None, an int, or a "
            f"sequence of int/None, got {sliding_window_tokens!r}"
        )
    else:
        windows = list(sliding_window_tokens)
        scalar = False
        if len(windows) != len(layer_types):
            raise ValueError(
                f"sliding_window_tokens has {len(windows)} "
                f"entries but layer_types has {len(layer_types)}"
            )
    rows: list[tuple[Retention, int | None]] = []
    for i, (label, raw) in enumerate(zip(layer_types, windows)):
        retention = _retention_for_label(label)
        if retention is None:
            raise ValueError(
                f"unknown layer_type {label!r} at layer {i}; "
                f"expected one of {sorted(_LAYER_TYPE_RETENTION)} or a "
                f"sliding sub-group label '{_SLIDING_SUBGROUP_PREFIX}<k>'"
            )
        if raw is not None and (isinstance(raw, bool) or not isinstance(raw, int)):
            raise ValueError(
                f"layer {i} ({label!r}) window must be None or " f"an int, got {raw!r}"
            )
        window = raw
        if retention == "sliding_window":
            if window is None or window <= 0:
                raise ValueError(
                    f"layer {i} ({label!r}) is sliding but its "
                    f"window is not a positive int (got {raw!r})"
                )
        else:
            if not scalar and window is not None and window > 0:
                raise ValueError(
                    f"layer {i} ({label!r}) is full-history but "
                    f"carries sliding window {window}; mislabeled layer_type?"
                )
            window = None
        rows.append((retention, window))
    return rows


def layer_group_ids(
    *,
    layer_types: Sequence[str],
    sliding_window_tokens: int | Sequence[int | None] | None,
) -> list[str]:
    """Per-layer paged-cache group id — the single derivation the recipes
    and multi-window models assign ``PagedAttention(group_id=...)`` from
    (today gpt_oss.py assigns group_id=layer_type, identical in the
    single-window case), so ``block_tables`` keys line up with the
    published group specs.

    The id is the bare label unless sliding layers carry more than one
    distinct window (then ``label_<window>``), so single-window models keep
    byte-identical ids."""
    rows = _layer_retention_windows(layer_types, sliding_window_tokens)
    distinct = {w for r, w in rows if r == "sliding_window"}
    multi_window = len(distinct) > 1
    return [
        (
            f"{label}_{window}"
            if multi_window and retention == "sliding_window"
            else label
        )
        for label, (retention, window) in zip(layer_types, rows)
    ]


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
    group_ids: Sequence[str],
    sliding_window_tokens: int | Sequence[int | None] | None,
    page_size: int,
    page_sizes: Mapping[str, int] | None = None,
    cache_blocks_per_lcm_block: Mapping[str, int] | None = None,
) -> list[PagedCacheGroupSpec]:
    """Derive paged-cache group specs from per-layer attention types.

    vLLM-style spec-value grouping: layers collapse into one group per
    distinct group id. Group order = first-appearance order.

    Args:
        layer_types: Per-layer labels: "full_attention" / "sliding_attention"
            (or sliding sub-group labels "sliding_attention_<k>") /
            "linear_attention" (state-family, e.g. Qwen3.5 GDN). Retention
            and family always come from these labels.
        group_ids: Physical group id per layer. The cache recipe is the
            single source of these ids: derive them with ``layer_group_ids``
            for label-equivalent grouping, or supply a finer split (hybrid
            models, e.g. ``split_recurrent_state_groups``). Layers sharing a
            group id must agree on (retention, window, family).
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
            layer without a positive window; full layer carrying a window;
            group id shared across incompatible layer policies.
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
    layer_specs = _layer_retention_windows(layer_types, sliding_window_tokens)
    resolved_group_ids = list(group_ids)
    if len(resolved_group_ids) != len(layer_types):
        raise ValueError(
            f"group_ids has {len(resolved_group_ids)} entries but layer_types "
            f"has {len(layer_types)}"
        )

    specs: list[PagedCacheGroupSpec] = []
    seen: dict[str, tuple[Retention, int | None, Family]] = {}
    for layer_id, ((retention, window), gid) in enumerate(
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
                cache_blocks_per_lcm_block=group_packing,
            )
        )
    if sizes:
        raise ValueError(f"page_sizes for unknown groups: {sorted(sizes)}")
    if packing:
        raise ValueError(
            f"cache_blocks_per_lcm_block for unknown groups: {sorted(packing)}"
        )
    return specs


def apply_pd_transfer_policies(
    specs: Sequence[PagedCacheGroupSpec],
) -> list[PagedCacheGroupSpec]:
    """Stamp PD-disaggregation transfer policies onto group specs.

    State groups transfer only their trailing snapshot; history groups
    transfer the full suffix.
    """
    from dataclasses import replace

    return [
        replace(
            spec,
            transfer_policy=(
                "latest_snapshot" if spec.family == "state" else "full_suffix"
            ),
        )
        for spec in specs
    ]


def build_paged_cache_group_specs(
    *,
    layer_types: Sequence[str],
    group_ids: Sequence[str],
    sliding_window_tokens: int | Sequence[int | None] | None,
    page_size: int,
    cache_blocks_per_lcm_block: Mapping[str, int] | None = None,
    pd_disaggregation_enabled: bool = False,
) -> tuple[PagedCacheGroupSpec, ...]:
    """Build the scheduler group specs one cache recipe publishes.

    The recipe computes these once and carries them via
    ``CachePoolSpec.paged_cache_group_specs``; page counts always come from
    the memory plan (the pool binds both into the runtime contract).
    Groups outside the layer-type vocabulary (e.g. Inkling's paged sconv
    columns) are plain tuple concatenation on the result — the pool
    validates uniqueness and planning for every group the same way.

    Args:
        layer_types: Per-layer paged-cache labels (empty -> every layer in
            ``group_ids`` is full-history).
        group_ids: Physical group id per layer, from the cache recipe
            (``CachePoolSpec.layer_group_ids``). Must line up 1:1 with
            layer_types when both are non-empty.
        sliding_window_tokens / page_size / cache_blocks_per_lcm_block:
            Forwarded to group_specs_from_layer_types.
        pd_disaggregation_enabled: Stamp PD transfer policies on the specs.
    """
    resolved_group_ids = tuple(group_ids)
    if not resolved_group_ids:
        raise ValueError(
            "build_paged_cache_group_specs requires per-layer group_ids; the "
            "cache recipe is their single source: derive them with "
            "layer_group_ids(...) and carry them via "
            "CachePoolSpec.layer_group_ids"
        )
    resolved_layer_types = tuple(layer_types) or (FULL_ATTENTION,) * len(
        resolved_group_ids
    )
    specs = group_specs_from_layer_types(
        layer_types=resolved_layer_types,
        group_ids=resolved_group_ids,
        sliding_window_tokens=sliding_window_tokens,
        page_size=page_size,
        cache_blocks_per_lcm_block=cache_blocks_per_lcm_block,
    )
    if pd_disaggregation_enabled:
        specs = apply_pd_transfer_policies(specs)
    return tuple(specs)


__all__ = [
    "FULL_ATTENTION",
    "LINEAR_ATTENTION",
    "PagedCacheGroupSpec",
    "Retention",
    "STATE_LAYER_TYPES",
    "apply_pd_transfer_policies",
    "build_paged_cache_group_specs",
    "compute_max_logical_pages_for_capture",
    "compute_paged_cache_group_page_counts",
    "group_specs_from_layer_types",
    "hybrid_slab_group_size",
    "layer_group_ids",
    "split_recurrent_state_groups",
    "validate_scheduler_config",
]
