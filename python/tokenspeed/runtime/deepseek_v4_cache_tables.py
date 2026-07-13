"""Pure cache-table source contract for the DeepSeek V4 runtime backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

CacheTableSourceKind = Literal["none", "radix", "flat"]


@dataclass(frozen=True)
class DeepseekV4CacheTableSource:
    """Exactly one active table ABI normalized to group-keyed mappings."""

    kind: CacheTableSourceKind
    tables: Mapping[str, Any]
    base_offsets: Mapping[str, Any]


@dataclass(frozen=True)
class DeepseekV4FlatLocPolicy:
    """Owner-local group contract for V4 flat KV writes.

    A V4 flat page id is local to the pool selected by ``group_id``.  It can
    therefore never be converted through the radix-era scalar
    ``req_to_page``/``page_size`` ABI.  The policy records the exact target and
    draft group subsets that must instead be delivered to their respective
    backends.
    """

    target_group_ids: tuple[str, ...]
    draft_group_ids: tuple[str, ...]


def _canonical_group_ids(specs: Sequence[Any], *, owner: str) -> tuple[str, ...]:
    group_ids = tuple(str(getattr(spec, "group_id", "")) for spec in specs)
    if not group_ids or any(not group_id for group_id in group_ids):
        raise RuntimeError(
            f"DeepSeek V4 flat {owner} cache loc policy requires non-empty "
            "owner-local group ids"
        )
    if len(group_ids) != len(set(group_ids)):
        raise RuntimeError(
            f"DeepSeek V4 flat {owner} cache loc policy has duplicate groups: "
            f"{group_ids}"
        )
    return group_ids


def resolve_deepseek_v4_flat_loc_policy(
    *,
    target_backend: Any,
    target_pool: Any,
    draft_backend: Any | None,
    draft_pool: Any | None,
    speculative_algorithm: str | None,
) -> DeepseekV4FlatLocPolicy | None:
    """Resolve the group-keyed write-loc policy for an active V4 flat arena.

    ``None`` means the target is on radix or another legacy cache layout.  An
    active V4 flat arena must opt in explicitly via the backend capability and
    must expose owner-local views that are exact subsets of the shared
    scheduler union.  Speculative execution additionally requires a matching
    draft-side arena/backend; silently borrowing the target view is forbidden.
    """

    target_plan = getattr(target_pool, "flat_memory_plan", None)
    requires_group_keyed = bool(
        getattr(target_backend, "requires_group_keyed_cache_locs", False)
    )
    if target_plan is None:
        return None
    if not requires_group_keyed:
        raise RuntimeError(
            "DeepSeek V4 flat arena requires a backend that declares "
            "group-keyed cache locations; scalar req_to_page fallback is "
            "forbidden"
        )

    scheduler_group_ids = {
        str(getattr(spec, "group_id", ""))
        for spec in getattr(target_pool, "scheduler_group_specs", ())
    }
    if not scheduler_group_ids or "" in scheduler_group_ids:
        raise RuntimeError(
            "DeepSeek V4 flat cache loc policy requires a non-empty canonical "
            "scheduler group union"
        )
    target_group_ids = _canonical_group_ids(
        tuple(getattr(target_pool, "paged_cache_group_specs", ()) or ()),
        owner="target",
    )
    if not set(target_group_ids).issubset(scheduler_group_ids):
        raise RuntimeError(
            "DeepSeek V4 flat target cache groups are not a subset of the "
            f"scheduler union: target={sorted(target_group_ids)}, "
            f"scheduler={sorted(scheduler_group_ids)}"
        )

    if speculative_algorithm is None:
        return DeepseekV4FlatLocPolicy(target_group_ids, ())

    if draft_backend is None or draft_pool is None:
        raise RuntimeError(
            "DeepSeek V4 flat speculative decoding requires a draft backend "
            "and an owner-local draft cache view"
        )
    draft_plan = getattr(draft_pool, "flat_memory_plan", None)
    if draft_plan is None or not getattr(
        draft_backend, "requires_group_keyed_cache_locs", False
    ):
        raise RuntimeError(
            "DeepSeek V4 flat speculative decoding requires group-keyed draft "
            "cache locations; a target req_to_page mirror is not a valid "
            "draft page domain"
        )
    target_fingerprint = getattr(target_plan, "plan_fingerprint", None)
    draft_fingerprint = getattr(draft_plan, "plan_fingerprint", None)
    if (
        not isinstance(target_fingerprint, str)
        or not target_fingerprint
        or target_fingerprint != draft_fingerprint
    ):
        raise RuntimeError(
            "DeepSeek V4 flat target/draft cache loc policies disagree on the "
            f"shared plan: target={target_fingerprint!r}, "
            f"draft={draft_fingerprint!r}"
        )
    draft_group_ids = _canonical_group_ids(
        tuple(getattr(draft_pool, "paged_cache_group_specs", ()) or ()),
        owner="draft",
    )
    if not set(draft_group_ids).issubset(scheduler_group_ids):
        raise RuntimeError(
            "DeepSeek V4 flat draft cache groups are not a subset of the "
            f"scheduler union: draft={sorted(draft_group_ids)}, "
            f"scheduler={sorted(scheduler_group_ids)}"
        )
    return DeepseekV4FlatLocPolicy(target_group_ids, draft_group_ids)


def legacy_flat_loc_group_id(
    group_specs: Sequence[Any],
    *,
    legacy_page_size: int,
) -> str | None:
    """Return the only flat group safe for the radix-era scalar loc ABI.

    The compatibility mirror is valid only for one absolute, stride-1,
    full-history group whose raw page span exactly equals the scalar page
    size.  In particular, a V4 compressed group with stride 4 must never be
    selected merely because it happens to be the first full-history group.
    """

    specs = tuple(group_specs)
    if len(specs) != 1:
        return None
    spec = specs[0]
    if (
        getattr(spec, "family", "history") != "history"
        or getattr(spec, "retention", None) != "full_history"
        or getattr(spec, "table_layout", "absolute") != "absolute"
        or getattr(spec, "entry_stride_tokens", None) != 1
        or getattr(spec, "block_size_tokens", None) != legacy_page_size
    ):
        return None
    group_id = str(getattr(spec, "group_id", ""))
    return group_id or None


def resolve_deepseek_v4_cache_table_source(
    *,
    paged_tables: Mapping[str, Any] | None,
    paged_base_offsets: Mapping[str, Any] | None,
    flat_tables: Mapping[str, Any] | None,
    flat_base_offsets: Mapping[str, Any] | None,
    require_source: bool = False,
) -> DeepseekV4CacheTableSource:
    """Select radix or flat tables and enforce the flat atomic ABI.

    Radix compact-table bases remain backward compatible: full-history keys
    may be omitted and therefore default to zero in the existing adapter.
    Flat export is stricter: every table group, including full-history groups,
    must have one explicit base vector and no base-only group may be present.
    """

    paged_active = paged_tables is not None or paged_base_offsets is not None
    flat_active = flat_tables is not None or flat_base_offsets is not None
    if paged_active and flat_active:
        raise RuntimeError(
            "DeepSeek V4 cache metadata received both radix paged tables and "
            "flat block tables; exactly one table source is allowed"
        )
    if flat_active:
        if flat_tables is None or flat_base_offsets is None:
            raise RuntimeError(
                "DeepSeek V4 flat cache metadata requires both block tables "
                "and explicit base offsets"
            )
        table_keys = {str(key) for key in flat_tables}
        base_keys = {str(key) for key in flat_base_offsets}
        if not table_keys:
            raise RuntimeError(
                "DeepSeek V4 flat cache metadata requires at least one owner-local "
                "cache group"
            )
        if table_keys != base_keys:
            raise RuntimeError(
                "DeepSeek V4 flat cache table/base group mismatch: "
                f"tables={sorted(table_keys)}, bases={sorted(base_keys)}"
            )
        return DeepseekV4CacheTableSource(
            kind="flat",
            tables={str(key): value for key, value in flat_tables.items()},
            base_offsets={str(key): value for key, value in flat_base_offsets.items()},
        )
    if paged_active:
        return DeepseekV4CacheTableSource(
            kind="radix",
            tables={str(key): value for key, value in (paged_tables or {}).items()},
            base_offsets={
                str(key): value for key, value in (paged_base_offsets or {}).items()
            },
        )
    if require_source:
        raise RuntimeError(
            "DeepSeek V4 cache metadata is missing its required radix/flat "
            "table source"
        )
    return DeepseekV4CacheTableSource(kind="none", tables={}, base_offsets={})
