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

"""Source-neutral contracts for grouped cache tables and write locations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

CacheTableSourceKind = Literal["none", "radix", "flat"]
FLAT_PACKED_META_MAX_OFFSET = (1 << 31) - 1


@dataclass(frozen=True)
class CacheTableSource:
    """Exactly one active table ABI normalized to group-keyed mappings."""

    kind: CacheTableSourceKind
    tables: Mapping[str, Any]
    base_offsets: Mapping[str, Any]
    # True only for persistent ring-slot sources whose tensor schema and
    # capacity were validated when the flat memory plan was constructed.
    planned: bool = False
    # Pool-set generation observed when this source's contents were published.
    # Immutable wrappers keep pre-wake sources stale even when a ring slot is
    # later reused; tables/base mappings remain persistent and allocation-free.
    generation: int | None = None
    # Planned sources carry one packed device upload plus owner-local unpack
    # metadata. Unplanned/radix sources leave this unset.
    packed: "FlatPackedTableUnion | FlatPackedTableOwnerSource | None" = None


@dataclass(frozen=True)
class FlatPackedTableOwnerSource:
    """One owner's view of a planned packed table/base upload."""

    buffer: Any
    unpack_meta: Any
    group_ids: tuple[str, ...]


@dataclass(frozen=True)
class FlatPackedTableUnion:
    """Scheduler-union packed upload with owner-local unpack metadata."""

    buffer: Any
    unpack_meta_by_owner: Mapping[str, Any]
    group_ids_by_owner: Mapping[str, tuple[str, ...]]

    def bind(
        self, owner: str, expected_group_ids: Sequence[str]
    ) -> FlatPackedTableOwnerSource:
        try:
            unpack_meta = self.unpack_meta_by_owner[owner]
            group_ids = self.group_ids_by_owner[owner]
        except KeyError as exc:
            raise RuntimeError(
                f"{owner} flat cache input has no packed owner layout"
            ) from exc
        expected_group_ids = tuple(expected_group_ids)
        expected = frozenset(expected_group_ids)
        if len(expected) != len(expected_group_ids):
            raise RuntimeError(
                f"{owner} flat cache binding has duplicate expected groups"
            )
        actual = frozenset(group_ids)
        if len(actual) != len(group_ids) or actual != expected:
            raise RuntimeError(
                f"{owner} flat packed groups disagree with owner binding: "
                f"packed={sorted(actual)}, expected={sorted(expected)}"
            )
        return FlatPackedTableOwnerSource(
            buffer=self.buffer,
            unpack_meta=unpack_meta,
            group_ids=group_ids,
        )


@dataclass(frozen=True, slots=True)
class FlatPackedTableDestination:
    """Stable offsets for one group in an exact-width packed graph buffer."""

    group_id: str
    table_offset: int
    table_cols: int
    base_offset: int


def make_flat_packed_table_destinations(
    capture_cols_by_group: Mapping[str, int],
    *,
    batch_rows: int,
) -> tuple[tuple[FlatPackedTableDestination, ...], int]:
    """Return deterministic exact-width table/base spans and total int32s."""

    if (
        isinstance(batch_rows, bool)
        or not isinstance(batch_rows, int)
        or batch_rows < 0
    ):
        raise ValueError("flat packed batch_rows must be an integer >= 0")
    offset = 0
    destinations = []
    for group_id, raw_cols in sorted(capture_cols_by_group.items()):
        if not group_id:
            raise ValueError("flat packed destination group_id must be non-empty")
        if isinstance(raw_cols, bool) or not isinstance(raw_cols, int) or raw_cols <= 0:
            raise ValueError(
                f"flat packed destination {group_id!r} columns must be positive"
            )
        table_offset = offset
        offset += batch_rows * raw_cols
        base_offset = offset
        offset += batch_rows
        if offset > FLAT_PACKED_META_MAX_OFFSET:
            raise OverflowError(
                "flat packed destination exceeds int32 metadata offsets: "
                f"group={group_id!r}, elements={offset}"
            )
        destinations.append(
            FlatPackedTableDestination(
                group_id=str(group_id),
                table_offset=table_offset,
                table_cols=raw_cols,
                base_offset=base_offset,
            )
        )
    return tuple(destinations), offset


@dataclass(frozen=True)
class CacheTableBinding:
    """Initialization-time authority for one backend's cache-table ABI."""

    kind: Literal["radix", "flat"]
    group_ids: tuple[str, ...] = ()
    group_keyed_cache_locs: bool = False


class _CacheTableProjection(Mapping[str, Any]):
    """Fixed-key, zero-copy view over one stable scheduler-union mapping."""

    __slots__ = ("_group_id_set", "_group_ids", "_mapping")

    def __init__(self, mapping: Mapping[str, Any], group_ids: tuple[str, ...]) -> None:
        self._mapping = mapping
        self._group_ids = group_ids
        self._group_id_set = frozenset(group_ids)

    def __getitem__(self, group_id: str) -> Any:
        if group_id not in self._group_id_set:
            raise KeyError(group_id)
        return self._mapping[group_id]

    def __iter__(self):
        return iter(self._group_ids)

    def __len__(self) -> int:
        return len(self._group_ids)


class FlatCacheTableOwnerView:
    """Owner-local zero-copy projection of a validated flat table union.

    Planned ring-slot sources are persistent, so their projections are built
    once per slot and reused. The per-forward hot path performs one identity
    lookup instead of rebinding every group into fresh dictionaries.
    """

    __slots__ = ("group_ids", "_group_id_set", "_planned_sources")

    def __init__(self, group_ids: Sequence[str]) -> None:
        self.group_ids = tuple(group_ids)
        self._group_id_set = frozenset(self.group_ids)
        self._planned_sources: dict[
            int, tuple[Mapping[str, Any], int | None, CacheTableSource]
        ] = {}

    def bind(
        self,
        source: CacheTableSource | None,
        *,
        owner: str,
    ) -> CacheTableSource | None:
        """Project a scheduler union without copying or rebinding entries."""

        if not self.group_ids:
            return None
        if source is None:
            raise RuntimeError(
                f"{owner} flat cache input is missing its required table source"
            )
        # The backend/pool ABI is bound once at startup, and only the Flat
        # extractor can reach this owner view.  Re-checking the source-kind
        # discriminator here would put the initialization decision back on the
        # per-forward path.
        if source.planned:
            cached = self._planned_sources.get(id(source.tables))
            if (
                cached is not None
                and cached[0] is source.tables
                and cached[1] == source.generation
            ):
                return cached[2]
        missing_tables = self._group_id_set.difference(source.tables)
        missing_bases = self._group_id_set.difference(source.base_offsets)
        if missing_tables or missing_bases:
            raise RuntimeError(
                f"{owner} flat cache input is incomplete: "
                f"missing_tables={sorted(missing_tables)}, "
                f"missing_bases={sorted(missing_bases)}"
            )
        projected = CacheTableSource(
            kind="flat",
            tables=_CacheTableProjection(source.tables, self.group_ids),
            base_offsets=_CacheTableProjection(source.base_offsets, self.group_ids),
            planned=source.planned,
            generation=source.generation,
            packed=(
                source.packed.bind(owner, self.group_ids)
                if isinstance(source.packed, FlatPackedTableUnion)
                else source.packed
            ),
        )
        if source.planned:
            # The mapping object identifies one stable ring slot. Replacing the
            # value on generation change keeps this cache bounded by ring depth.
            self._planned_sources[id(source.tables)] = (
                source.tables,
                source.generation,
                projected,
            )
        return projected


def require_flat_cache_generation(value: Any, *, where: str) -> int:
    """Return one validated pool/source generation at its owner boundary."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(
            f"{where}: flat cache generation must be an integer >= 0, got {value!r}"
        )
    return value


def validate_flat_cache_generation(
    actual_generation: Any,
    *,
    expected_generation: int,
    where: str,
) -> int:
    """Reject cache metadata published before the current pool generation."""

    if (
        isinstance(expected_generation, bool)
        or not isinstance(expected_generation, int)
        or expected_generation < 0
    ):
        raise ValueError(
            "expected cache table generation must be an integer >= 0, "
            f"got {expected_generation!r}"
        )
    actual_generation = require_flat_cache_generation(
        actual_generation,
        where=where,
    )
    if actual_generation != expected_generation:
        raise RuntimeError(
            f"{where}: stale flat cache table generation: "
            f"source={actual_generation!r}, arena={expected_generation}"
        )
    return actual_generation


def validate_flat_table_base_offsets(
    tables: Mapping[str, Any],
    base_offsets: Mapping[str, Any],
    *,
    where: str,
    min_rows: int | None = None,
) -> frozenset[str]:
    """Validate the atomic flat table/base ABI from tensor metadata."""

    # Keep torch out of this source-neutral module's import path. Legacy
    # conversion is the only caller that needs tensor metadata inspection.
    import torch

    table_keys = frozenset(tables or {})
    base_keys = frozenset(base_offsets or {})
    if table_keys != base_keys:
        raise RuntimeError(
            f"{where}: flat block table/base group mismatch: "
            f"tables={sorted(table_keys)}, bases={sorted(base_keys)}"
        )
    for group_id in table_keys:
        table = tables[group_id]
        bases = base_offsets[group_id]
        if not isinstance(table, torch.Tensor) or not isinstance(bases, torch.Tensor):
            raise TypeError(
                f"{where}: flat group {group_id!r} table/base must be tensors"
            )
        if table.dim() != 2 or bases.dim() != 1:
            raise RuntimeError(
                f"{where}: flat group {group_id!r} requires rank-2/rank-1 "
                f"table/base, got {tuple(table.shape)}/{tuple(bases.shape)}"
            )
        if table.dtype != torch.int32 or bases.dtype != torch.int32:
            raise RuntimeError(
                f"{where}: flat group {group_id!r} table/base must be int32, "
                f"got {table.dtype}/{bases.dtype}"
            )
        if table.device != bases.device or table.shape[0] != bases.shape[0]:
            raise RuntimeError(
                f"{where}: flat group {group_id!r} table/base disagree: "
                f"devices={table.device}/{bases.device}, "
                f"rows={table.shape[0]}/{bases.shape[0]}"
            )
        if min_rows is not None and table.shape[0] < min_rows:
            raise RuntimeError(
                f"{where}: flat group {group_id!r} has {table.shape[0]} rows "
                f"but needs at least {min_rows}"
            )
    return table_keys


def validate_flat_cache_table_source(
    source: CacheTableSource,
    *,
    where: str,
    min_rows: int | None = None,
) -> None:
    """Validate an unplanned flat table/base pair at its conversion boundary."""

    if source.kind != "flat":
        raise RuntimeError(f"{where}: expected flat source, got {source.kind!r}")
    if not source.tables:
        raise RuntimeError(f"{where}: flat cache metadata requires at least one group")
    validate_flat_table_base_offsets(
        source.tables,
        source.base_offsets,
        where=where,
        min_rows=min_rows,
    )


def _canonical_group_ids(specs: Sequence[Any], *, owner: str) -> tuple[str, ...]:
    group_ids = tuple(str(getattr(spec, "group_id", "")) for spec in specs)
    if not group_ids or any(not group_id for group_id in group_ids):
        raise RuntimeError(
            f"flat {owner} cache locations require non-empty owner-local group ids"
        )
    if len(group_ids) != len(set(group_ids)):
        raise RuntimeError(
            f"flat {owner} cache locations contain duplicate groups: {group_ids}"
        )
    return group_ids


def validate_speculative_flat_bindings(
    *,
    target_pool: Any,
    target_binding: CacheTableBinding,
    draft_pool: Any | None,
    draft_binding: CacheTableBinding | None,
    speculative_algorithm: str | None,
) -> None:
    """Validate speculative owners share one plan and the scheduler's groups."""

    if (
        speculative_algorithm is None
        or target_binding.kind != "flat"
        or not target_binding.group_keyed_cache_locs
    ):
        return
    if draft_pool is None or draft_binding is None:
        raise RuntimeError(
            "flat speculative decoding requires a draft backend and owner-local "
            "draft cache view"
        )
    if draft_binding.kind != "flat" or not draft_binding.group_keyed_cache_locs:
        raise RuntimeError(
            "flat speculative decoding requires group-keyed draft cache locations; "
            "the target scalar page domain cannot be borrowed"
        )
    target_plan = getattr(target_pool, "flat_memory_plan", None)
    draft_plan = getattr(draft_pool, "flat_memory_plan", None)
    if draft_plan is not target_plan:
        raise RuntimeError(
            "flat target and draft cache views must share one canonical plan object"
        )
    scheduler_specs = tuple(
        getattr(target_pool, "scheduler_group_specs", None)
        or getattr(target_pool, "paged_cache_group_specs", ())
    )
    scheduler_group_ids = set(_canonical_group_ids(scheduler_specs, owner="scheduler"))
    missing_target_groups = set(target_binding.group_ids) - scheduler_group_ids
    if missing_target_groups:
        raise RuntimeError(
            "flat target cache groups are absent from the scheduler union: "
            f"missing={sorted(missing_target_groups)}, "
            f"scheduler={sorted(scheduler_group_ids)}"
        )
    missing_draft_groups = set(draft_binding.group_ids) - scheduler_group_ids
    if missing_draft_groups:
        raise RuntimeError(
            "flat draft cache groups are absent from the scheduler union: "
            f"missing={sorted(missing_draft_groups)}, "
            f"scheduler={sorted(scheduler_group_ids)}"
        )


def resolve_cache_table_binding(
    *,
    backend: Any,
    pool: Any,
    flat_scheduler_active: bool,
) -> CacheTableBinding:
    """Bind one backend/pool pair to radix or flat exactly once.

    A dual-capability backend selects flat only when its pool carries the
    canonical flat memory plan.  A flat-only backend always selects flat.
    """

    if backend is None or pool is None:
        raise ValueError("cache table binding requires both a backend and a pool")
    uses_flat = bool(getattr(backend, "uses_flat_cache_groups", False))
    uses_paged = bool(getattr(backend, "uses_paged_cache_groups", False))
    has_flat_plan = getattr(pool, "flat_memory_plan", None) is not None
    flat_active = (
        flat_scheduler_active and uses_flat and (not uses_paged or has_flat_plan)
    )
    if flat_scheduler_active and has_flat_plan and not uses_flat:
        raise RuntimeError(
            "flat cache plan requires a backend that consumes flat cache groups"
        )
    if not flat_active:
        return CacheTableBinding(kind="radix")
    group_ids = _canonical_group_ids(
        tuple(getattr(pool, "paged_cache_group_specs", ()) or ()),
        owner="backend",
    )
    group_keyed_cache_locs = bool(
        getattr(backend, "requires_group_keyed_cache_locs", False)
    )
    if has_flat_plan:
        if not group_keyed_cache_locs:
            raise RuntimeError(
                "flat cache plan requires a backend with group-keyed cache "
                "locations; scalar req_to_page fallback is forbidden"
            )
        scheduler_group_ids = _canonical_group_ids(
            tuple(getattr(pool, "scheduler_group_specs", ()) or ()),
            owner="scheduler",
        )
        if not set(group_ids).issubset(scheduler_group_ids):
            raise RuntimeError(
                "flat backend cache groups are not a subset of the scheduler "
                f"union: backend={sorted(group_ids)}, "
                f"scheduler={sorted(scheduler_group_ids)}"
            )
    return CacheTableBinding(
        kind="flat",
        group_ids=group_ids,
        group_keyed_cache_locs=group_keyed_cache_locs,
    )


def legacy_flat_loc_group_id(
    group_specs: Sequence[Any], *, legacy_page_size: int
) -> str | None:
    """Return the sole group safe to mirror into the legacy scalar scratch ABI."""

    if (
        isinstance(legacy_page_size, bool)
        or not isinstance(legacy_page_size, int)
        or legacy_page_size <= 0
    ):
        return None

    candidates: list[str] = []
    for spec in group_specs:
        block_size = getattr(spec, "block_size", None)
        if (
            getattr(spec, "family", "history") != "history"
            or getattr(spec, "retention", None) != "full_history"
            or getattr(spec, "table_layout", "absolute") != "absolute"
            or getattr(spec, "entry_stride_tokens", None) != 1
            or isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size != legacy_page_size
        ):
            continue
        group_id = str(getattr(spec, "group_id", ""))
        if group_id:
            candidates.append(group_id)
    return candidates[0] if len(candidates) == 1 else None


def resolve_cache_table_source(
    *,
    paged_tables: Mapping[str, Any] | None,
    paged_base_offsets: Mapping[str, Any] | None,
    flat_tables: Mapping[str, Any] | None,
    flat_base_offsets: Mapping[str, Any] | None,
    flat_generation: int | None = None,
    require_source: bool = False,
) -> CacheTableSource:
    """Select exactly one radix or flat table source.

    Flat tables always carry an explicit logical-base vector for every group.
    Radix tables retain their legacy optional-base behavior.
    """

    paged_active = paged_tables is not None or paged_base_offsets is not None
    flat_active = flat_tables is not None or flat_base_offsets is not None
    if paged_active and flat_active:
        raise RuntimeError(
            "cache metadata received both radix and flat tables; exactly one "
            "table source is allowed"
        )
    if flat_active:
        if flat_tables is None or flat_base_offsets is None:
            raise RuntimeError(
                "flat cache metadata requires both block tables and base offsets"
            )
        table_keys = set(flat_tables)
        base_keys = set(flat_base_offsets)
        if any(not isinstance(key, str) or not key for key in table_keys | base_keys):
            raise RuntimeError("flat cache table group ids must be non-empty strings")
        if not table_keys:
            raise RuntimeError("flat cache metadata requires at least one group")
        if table_keys != base_keys:
            raise RuntimeError(
                "flat cache table/base group mismatch: "
                f"tables={sorted(table_keys)}, bases={sorted(base_keys)}"
            )
        if flat_generation is not None and (
            isinstance(flat_generation, bool)
            or not isinstance(flat_generation, int)
            or flat_generation < 0
        ):
            raise ValueError(
                f"flat_generation must be an integer >= 0, got {flat_generation!r}"
            )
        return CacheTableSource(
            kind="flat",
            tables=flat_tables,
            base_offsets=flat_base_offsets,
            generation=flat_generation,
        )
    if paged_active:
        paged_tables = paged_tables or {}
        paged_base_offsets = paged_base_offsets or {}
        if any(
            not isinstance(key, str) or not key
            for key in set(paged_tables) | set(paged_base_offsets)
        ):
            raise RuntimeError("radix cache table group ids must be non-empty strings")
        return CacheTableSource(
            kind="radix",
            tables=paged_tables,
            base_offsets=paged_base_offsets,
        )
    if require_source:
        raise RuntimeError("cache metadata is missing its required table source")
    return CacheTableSource(kind="none", tables={}, base_offsets={})


__all__ = [
    "CacheTableBinding",
    "CacheTableSource",
    "CacheTableSourceKind",
    "FLAT_PACKED_META_MAX_OFFSET",
    "FlatCacheTableOwnerView",
    "FlatPackedTableDestination",
    "FlatPackedTableOwnerSource",
    "FlatPackedTableUnion",
    "legacy_flat_loc_group_id",
    "make_flat_packed_table_destinations",
    "require_flat_cache_generation",
    "resolve_cache_table_binding",
    "resolve_cache_table_source",
    "validate_speculative_flat_bindings",
    "validate_flat_cache_table_source",
    "validate_flat_table_base_offsets",
]
