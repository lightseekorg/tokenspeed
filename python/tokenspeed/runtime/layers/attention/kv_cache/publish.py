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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Publication rules mapping cache recipes to paged-cache group specs.

The contract types (``PagedCacheGroupSpec``, the layer_type label
vocabulary) and the scheduler-side sizing/validation live in
``tokenspeed.runtime.configs.paged_cache_spec``; this module owns the
KV-pool side: deriving per-layer group ids and publishing the group specs
a pool hands to the scheduler.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tokenspeed.runtime.configs.paged_cache_spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    STATE_LAYER_TYPES,
    Family,
    PagedCacheGroupSpec,
    Retention,
    compute_paged_cache_group_page_counts,
)

# layer_type label -> retention. GPT-OSS uses the first two, Qwen3.5 GDN
# layers use "linear_attention"; unknown labels raise.
_LAYER_TYPE_RETENTION: dict[str, Retention] = {
    FULL_ATTENTION: "full_history",
    "sliding_attention": "sliding_window",
    # State groups ride full_history retention: the C++ side keys the
    # mamba-state kind on family == State && retention != SlidingWindow.
    LINEAR_ATTENTION: "full_history",
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


def publish_paged_cache_groups(
    *,
    layer_types: Sequence[str],
    group_ids: Sequence[str],
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
    consumes the full-attention group's table (published into the batch-ordered
    draft page table each step).

    Args:
        layer_types: Per-layer paged-cache labels (empty -> every layer in
            ``group_ids`` is full-history).
        group_ids: Physical group id per layer, from the cache recipe
            (``CachePoolSpec.layer_group_ids``). Must line up 1:1 with
            layer_types when both are non-empty.
        sliding_window_tokens / page_size: Forwarded to
            group_specs_from_layer_types.
        max_live_requests / max_scheduled_tokens / max_total_tokens /
            max_context_len: Sizing inputs for
            compute_paged_cache_group_page_counts.

    Returns:
        The group specs and their page counts.
    """
    resolved_group_ids = tuple(group_ids)
    if not resolved_group_ids:
        raise ValueError(
            "publish_paged_cache_groups requires per-layer group_ids; the "
            "cache recipe is their single source: derive them with "
            "layer_group_ids(...) and carry them via "
            "CachePoolSpec.layer_group_ids"
        )
    resolved_layer_types = tuple(layer_types) or (FULL_ATTENTION,) * len(
        resolved_group_ids
    )
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
        layer_types=resolved_layer_types,
        group_ids=resolved_group_ids,
        sliding_window_tokens=sliding_window_tokens,
        page_size=page_size,
        page_sizes=page_sizes,
        cache_blocks_per_lcm_block=packing,
    )
    # Model-declared groups outside the layer-type vocabulary (e.g. paged sconv columns).
    for spec in extra_groups:
        if any(sp.group_id == spec.group_id for sp in specs):
            raise ValueError(f"extra_groups: duplicate group id {spec.group_id!r}")
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


__all__ = [
    "group_specs_from_layer_types",
    "hybrid_slab_group_size",
    "layer_group_ids",
    "publish_paged_cache_groups",
    "split_recurrent_state_groups",
]
