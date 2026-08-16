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
class CacheGroupSpec:
    """One cache group's scheduler-facing layout.

    A spec declares exactly one geometry shape:

    * **Row geometry** (``rows_per_page`` + ``entry_stride_tokens``) — for
      paged KV-cache consumers, whose CacheBlocks physically hold rows of
      entries (per-token KV, sliding windows, compressed entries).
    * **Checkpoint** (``checkpoint_granularity``) — for snapshot-style state
      groups (recurrent/conv state), whose CacheBlocks each hold one state
      snapshot taken every ``checkpoint_granularity`` tokens. Such a group
      has no rows and no pages; only state-family groups may use this shape.

    Family-agnostic consumers address either shape through
    ``block_granularity``; ``page_size`` exists only for row geometry.
    """

    group_id: str
    retention: Retention
    rows_per_page: int | None = None
    entry_stride_tokens: int | None = None
    sliding_window_tokens: int | None = None
    # History stores token history; State stores recurrent state. Retention
    # determines whether either family is full-history or sliding.
    family: Family = "history"
    # None preserves standalone/non-PD behavior; PD plans set this explicitly.
    transfer_policy: TransferPolicy | None = None
    # Snapshot-state shape: raw-token span between two state checkpoints.
    checkpoint_granularity: int | None = None

    def __post_init__(self) -> None:
        has_rows = (
            self.rows_per_page is not None or self.entry_stride_tokens is not None
        )
        if self.checkpoint_granularity is not None:
            if has_rows:
                raise ValueError(
                    f"group {self.group_id!r}: checkpoint_granularity is "
                    "mutually exclusive with rows_per_page/entry_stride_tokens"
                )
            if self.family != "state":
                raise ValueError(
                    f"group {self.group_id!r}: checkpoint_granularity is only "
                    "valid for state-family groups"
                )
            if self.checkpoint_granularity <= 0:
                raise ValueError(
                    f"group {self.group_id!r}: checkpoint_granularity must be "
                    f"> 0, got {self.checkpoint_granularity}"
                )
            return
        if self.rows_per_page is None or self.entry_stride_tokens is None:
            raise ValueError(
                f"group {self.group_id!r}: declare either row geometry "
                "(rows_per_page + entry_stride_tokens) or "
                "checkpoint_granularity"
            )
        if self.rows_per_page <= 0 or self.entry_stride_tokens <= 0:
            raise ValueError(
                f"group {self.group_id!r}: rows_per_page and "
                "entry_stride_tokens must be > 0, got "
                f"{self.rows_per_page} and {self.entry_stride_tokens}"
            )

    @property
    def page_size(self) -> int:
        """Raw-token span of one CacheBlock; row-geometry (paged KV) only."""
        if self.rows_per_page is None or self.entry_stride_tokens is None:
            raise TypeError(
                f"group {self.group_id!r}: page_size is undefined for "
                "checkpoint (snapshot-state) groups; use block_granularity "
                "or checkpoint_granularity"
            )
        return self.rows_per_page * self.entry_stride_tokens

    @property
    def block_granularity(self) -> int:
        """Raw-token span of one block-table slot, valid for every shape."""
        if self.checkpoint_granularity is not None:
            return self.checkpoint_granularity
        return self.page_size


_CACHE_GROUP_DUMMY_PAGES = 1

# Token span of one interleaved mxfp8 KV-scale tile. The fused FP8 attention
# kernels store k_scale/v_scale in tiles of this many tokens, which changes
# both the scale field shape and its head-partition axis.
MXFP8_KV_SCALE_TILE_TOKENS = 128


def _ceil_div(dividend: int, divisor: int) -> int:
    return (dividend + divisor - 1) // divisor


# Cache-group label vocabulary (NOT the HF checkpoint's serialized enum:
# Qwen3.5 checkpoints spell full attention "attention").
FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"

# Labels whose group is state-family (recurrent state rows, not KV history).
STATE_LAYER_TYPES = frozenset({LINEAR_ATTENTION})


def validate_scheduler_config(
    *,
    attn_backend: object,
    kv_pool: object,
) -> None:
    """Validate that the attention backend consumes every cache-group family
    the KV pool's runtime contract publishes.

    Args:
        attn_backend: The (possibly composite) attention backend; composite
            backends aggregate their sub-backends' consumer families in
            their own ``cache_consumer_families``.
        kv_pool: The KV pool whose ``runtime_contract`` group specs name the
            required families.

    Raises:
        RuntimeError: When the pool publishes no runtime contract, or when a
            contract family (e.g. a hybrid pool's ``state`` group) has no
            consumer in the backend — that group's tables would go unread,
            dying on a capture-path assert at best and silently reading the
            wrong pages at worst.
    """
    arena = getattr(kv_pool, "arena", None)
    contract = getattr(arena, "runtime_contract", None)
    if contract is None:
        raise RuntimeError(
            f"KV pool {type(kv_pool).__name__} publishes no "
            "CacheRuntimeContract. Every pool must be built from a "
            "cache recipe (kv_cache.recipes.setup.prepare_cache_setup)."
        )
    required_families = frozenset(spec.family for spec in contract.group_specs)
    supported_families = frozenset(getattr(attn_backend, "cache_consumer_families", ()))
    missing_families = required_families - supported_families
    if missing_families:
        raise RuntimeError(
            "cache pool requires consumer families "
            f"{sorted(required_families)}, but backend "
            f"{type(attn_backend).__name__} is missing "
            f"{sorted(missing_families)}"
        )


def compute_cache_group_page_counts(
    specs: Sequence[CacheGroupSpec],
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
        raise ValueError("overlapped cache sizing requires decode_input_tokens > 0")
    if safety_margin < 0:
        raise ValueError(f"safety_margin must be >= 0, got {safety_margin}")

    counts: dict[str, int] = {}
    for spec in specs:
        block_granularity = spec.block_granularity
        protected_pages = max_live_requests * _ceil_div(
            overlap_schedule_depth * decode_input_tokens, block_granularity
        )
        # Mamba-state kind = family "state" AND retention != sliding_window
        # (the C++ side keys it the same way); V4's sliding-window state tail
        # buffers keep the sliding-window formula below.
        if spec.family == "state" and spec.retention == "full_history":
            # State group: 2 live pages/request (the W=2 write window) +
            # floor(T/P) snapshot pages (snapshots are bounded by the shared
            # page-id space), capped at the full-history count.
            full_history_total = (
                _ceil_div(max_total_tokens, block_granularity)
                + max_live_requests
                + protected_pages
                + _CACHE_GROUP_DUMMY_PAGES
                + safety_margin
            )
            state_total = (
                max_live_requests * 2
                + max_total_tokens // block_granularity
                + protected_pages
                + _CACHE_GROUP_DUMMY_PAGES
                + safety_margin
            )
            total = min(state_total, full_history_total)
        elif spec.retention == "full_history":
            full_pages = _ceil_div(max_total_tokens, block_granularity)
            total = (
                full_pages
                + max_live_requests
                + protected_pages
                + _CACHE_GROUP_DUMMY_PAGES
                + safety_margin
            )
        elif spec.retention == "sliding_window":
            window = spec.sliding_window_tokens
            if window is None or window <= 0:
                raise ValueError(
                    f"CacheGroupSpec {spec.group_id}: sliding group missing "
                    "positive sliding_window_tokens"
                )
            # Capacity tracks resident history before the next token.
            resident_tokens_per_req = min(max(window - 1, 0), max_context_len)
            resident_pages = max_live_requests * _ceil_div(
                resident_tokens_per_req, block_granularity
            )
            scheduled_tokens = min(max_scheduled_tokens, max_total_tokens)
            scheduled_pages = _ceil_div(scheduled_tokens, block_granularity)
            total = (
                resident_pages
                + scheduled_pages
                + max_live_requests
                + protected_pages
                + _CACHE_GROUP_DUMMY_PAGES
                + safety_margin
            )
        else:
            raise ValueError(
                f"CacheGroupSpec {spec.group_id}: unsupported retention "
                f"{spec.retention!r}"
            )
        counts[spec.group_id] = int(total)
    return counts


def compute_max_logical_pages_for_capture(
    spec: CacheGroupSpec,
    *,
    max_context_len: int,
    max_tokens_per_req: int = 1,
    overlap_schedule_depth: int = 0,
) -> int:
    """Return CUDA Graph block-table width for one cache group.

    Decode admission reserves the current verify span plus one span for each
    overlapped schedule.  Include that complete reservation horizon here: a
    request close to the model context limit can still expose the reserved
    pages in its scheduler block-table row before the accepted tokens are
    truncated by the request-length limit.

    Args:
        spec: Cache group layout and retention policy.
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
    block_granularity = spec.block_granularity
    reservation_horizon = (overlap_schedule_depth + 1) * max_tokens_per_req
    if spec.retention == "sliding_window":
        window = spec.sliding_window_tokens
        if window is None or window <= 0:
            raise ValueError(
                f"CacheGroupSpec {spec.group_id}: sliding group missing "
                "positive sliding_window_tokens"
            )
        # Capture uses a conservative metadata bound; it does not change the
        # per-token attention history counted as window - 1 above.
        retention_bound = min(window, max_context_len)
        live_tokens = retention_bound + reservation_horizon
        return _ceil_div(live_tokens, block_granularity) + 1
    if spec.retention == "full_history":
        live_tokens = max_context_len + reservation_horizon
        return _ceil_div(live_tokens, block_granularity)
    raise ValueError(
        f"CacheGroupSpec {spec.group_id}: unsupported retention {spec.retention!r}"
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
    profile) and the planned field layout -- the two must never disagree. The scheduler's single BlockPool owns each page id by at most
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
    """Per-layer cache group id — the single derivation the recipes
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


def group(
    *,
    layer_types: Sequence[str],
    group_ids: Sequence[str],
    sliding_window_tokens: int | Sequence[int | None] | None,
    prefix_granularity: int,
    fields_for_layer,
    page_sizes: Mapping[str, int] | None = None,
    pd_disaggregation_enabled: bool = False,
) -> tuple[tuple[CacheGroupSpec, tuple], ...]:
    """Walk the layers once, building each group whole.

    A cache group has two halves -- the scheduler-facing spec and the bytes
    its fields occupy -- and one walk produces both: every layer contributes
    its retention/family policy to its group's spec and its fields to that
    same group's field list, keyed by the one ``group_id`` variable. There is
    no second enumeration to disagree with, so nothing needs cross-checking.

    Physical packing is deliberately absent from the specs: how many of a
    group's CacheBlocks share one LCM parent is solved by the memory plan and
    published by the arena.

    vLLM-style spec-value grouping: layers collapse into one group per
    distinct group id. Group order = first-appearance order.

    Args:
        layer_types: Per-layer labels: "full_attention" / "sliding_attention"
            (or sliding sub-group labels "sliding_attention_<k>") /
            "linear_attention" (state-family, e.g. Qwen3.5 GDN). Retention
            and family always come from these labels. Empty means every layer
            is full-history.
        group_ids: Physical group id per layer. The cache recipe is the
            single source of these ids: derive them with ``layer_group_ids``
            for label-equivalent grouping, or supply a finer split (hybrid
            models, e.g. ``split_recurrent_state_groups``). Layers sharing a
            group id must agree on (retention, window, family).
        sliding_window_tokens: One window for all sliding layers (today's HF
            scalar), or a per-layer sequence (multi-window models; full-layer
            positions must be None).
        prefix_granularity: Scheduler-wide prefix granularity in tokens.
        fields_for_layer: ``(layer_id, group_id, occurrence) -> fields``. The
            occurrence is how many layers of this group already declared
            fields, i.e. this layer's slot in the group's plane numbering; a
            layer that declares nothing does not consume a slot.
        page_sizes: Per-group page sizes keyed by group id (heterogeneous
            block sizes); values must be positive multiples of
            prefix_granularity. Groups not listed use prefix_granularity.
        pd_disaggregation_enabled: Stamp PD transfer policies on the specs.

    Returns:
        ``(spec, fields)`` pairs in first-appearance order, ready for
        ``pack``.

    Raises:
        ValueError: unknown label; window sequence length mismatch; sliding
            layer without a positive window; full layer carrying a window;
            group id shared across incompatible layer policies; a group whose
            layers declare no fields.
    """
    resolved_group_ids = tuple(group_ids)
    if not resolved_group_ids:
        raise ValueError(
            "group requires per-layer group_ids; the "
            "cache recipe is their single source: derive them with "
            "layer_group_ids(...) and carry them via "
            "CachePoolSpec.layer_group_ids"
        )
    resolved_layer_types = tuple(layer_types) or (FULL_ATTENTION,) * len(
        resolved_group_ids
    )
    if len(resolved_group_ids) != len(resolved_layer_types):
        raise ValueError(
            f"group_ids has {len(resolved_group_ids)} entries but layer_types "
            f"has {len(resolved_layer_types)}"
        )
    sizes = dict(page_sizes or {})
    for gid, ps in sizes.items():
        if ps <= 0 or ps % prefix_granularity:
            raise ValueError(
                f"page_sizes[{gid!r}] = {ps} must be a positive "
                f"multiple of prefix_granularity {prefix_granularity}"
            )
    layer_policies = _layer_retention_windows(
        resolved_layer_types, sliding_window_tokens
    )

    specs: dict[str, CacheGroupSpec] = {}
    fields: dict[str, tuple] = {}
    occurrences: dict[str, int] = {}
    for layer_id, ((retention, window), gid) in enumerate(
        zip(layer_policies, resolved_group_ids)
    ):
        if not gid:
            raise ValueError(f"group_ids[{layer_id}] must be non-empty")
        family: Family = (
            "state"
            if resolved_layer_types[layer_id] in STATE_LAYER_TYPES
            else "history"
        )
        if gid in specs:
            existing = specs[gid]
            if (
                existing.retention,
                existing.sliding_window_tokens,
                existing.family,
            ) != (
                retention,
                window,
                family,
            ):
                raise ValueError(f"group_id {gid!r} mixes incompatible layer policies")
        else:
            specs[gid] = _layer_group_spec(
                group_id=gid,
                retention=retention,
                window=window,
                family=family,
                block_tokens=sizes.pop(gid, None) or prefix_granularity,
            )
        occurrence = occurrences.get(gid, 0)
        declared = tuple(fields_for_layer(layer_id, gid, occurrence))
        if declared:
            occurrences[gid] = occurrence + 1
            fields[gid] = fields.get(gid, ()) + declared
    if sizes:
        raise ValueError(f"page_sizes for unknown groups: {sorted(sizes)}")
    barren = sorted(set(specs) - set(fields))
    if barren:
        raise ValueError(
            f"cache groups {barren} declare no fields; a group with no bytes "
            "cannot be addressed"
        )
    published = specs.values()
    if pd_disaggregation_enabled:
        published = apply_pd_transfer_policies(tuple(published))
    return tuple((spec, fields[spec.group_id]) for spec in published)


def _layer_group_spec(
    *,
    group_id: str,
    retention: Retention,
    window: int | None,
    family: Family,
    block_tokens: int,
) -> CacheGroupSpec:
    """One group's spec in the shape its family declares."""
    if family == "state":
        # Snapshot-state groups have no rows: one CacheBlock holds one
        # recurrent-state checkpoint taken every `block_tokens` tokens.
        return CacheGroupSpec(
            group_id=group_id,
            retention=retention,
            sliding_window_tokens=window,
            family=family,
            checkpoint_granularity=block_tokens,
        )
    return CacheGroupSpec(
        group_id=group_id,
        retention=retention,
        rows_per_page=block_tokens,
        entry_stride_tokens=1,
        sliding_window_tokens=window,
        family=family,
    )


def apply_pd_transfer_policies(
    specs: Sequence[CacheGroupSpec],
) -> list[CacheGroupSpec]:
    """Stamp PD-disaggregation transfer policies onto group specs.

    Full-history state groups transfer only their trailing snapshot. Sliding
    state is rolling token history and therefore transfers its complete
    retained suffix, like an attention-history group.
    """
    from dataclasses import replace

    return [
        replace(
            spec,
            transfer_policy=(
                "latest_snapshot"
                if spec.family == "state" and spec.retention == "full_history"
                else "full_suffix"
            ),
        )
        for spec in specs
    ]


__all__ = [
    "FULL_ATTENTION",
    "LINEAR_ATTENTION",
    "CacheGroupSpec",
    "Retention",
    "STATE_LAYER_TYPES",
    "apply_pd_transfer_policies",
    "group",
    "compute_max_logical_pages_for_capture",
    "compute_cache_group_page_counts",
    "hybrid_slab_group_size",
    "layer_group_ids",
    "split_recurrent_state_groups",
    "validate_scheduler_config",
]
