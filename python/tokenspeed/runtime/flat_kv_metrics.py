"""Pool-aware observability helpers for heterogeneous flat KV caches."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _field(snapshot: Any, name: str) -> int:
    value = snapshot[name] if isinstance(snapshot, Mapping) else getattr(snapshot, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"flat pool snapshot {name} must be an int")
    return value


def _pool_id(snapshot: Any) -> str:
    value = (
        snapshot["pool_id"]
        if isinstance(snapshot, Mapping)
        else getattr(snapshot, "pool_id")
    )
    if not isinstance(value, str) or not value:
        raise TypeError("flat pool snapshot pool_id must be a non-empty string")
    return value


def _summary_from_totals(
    *,
    active_bytes: int,
    capacity_bytes: int,
    pressure_numerator: int,
    pressure_denominator: int,
    equivalent_total_pages: int,
) -> dict[str, int | float]:
    active_equivalent_pages, pressure_equivalent_pages = _equivalent_pages_from_totals(
        active_bytes=active_bytes,
        capacity_bytes=capacity_bytes,
        pressure_numerator=pressure_numerator,
        pressure_denominator=pressure_denominator,
        equivalent_total_pages=equivalent_total_pages,
    )
    return {
        "active_bytes": active_bytes,
        "capacity_bytes": capacity_bytes,
        "byte_utilization": active_bytes / capacity_bytes if capacity_bytes else 0.0,
        "pressure": pressure_numerator / pressure_denominator,
        "active_equivalent_pages": active_equivalent_pages,
        "pressure_equivalent_pages": pressure_equivalent_pages,
    }


def _equivalent_pages_from_totals(
    *,
    active_bytes: int,
    capacity_bytes: int,
    pressure_numerator: int,
    pressure_denominator: int,
    equivalent_total_pages: int,
) -> tuple[int, int]:
    if (
        isinstance(equivalent_total_pages, bool)
        or not isinstance(equivalent_total_pages, int)
        or equivalent_total_pages < 0
    ):
        raise ValueError("equivalent_total_pages must be a non-negative int")
    if min(active_bytes, capacity_bytes, pressure_numerator) < 0:
        raise ValueError("flat pool aggregate counters must be non-negative")
    if active_bytes > capacity_bytes:
        raise ValueError("flat pool active bytes exceed capacity bytes")
    if pressure_denominator <= 0 or pressure_numerator > pressure_denominator:
        raise ValueError("flat pool aggregate pressure ratio is invalid")
    active_equivalent_pages = (
        (active_bytes * equivalent_total_pages + capacity_bytes - 1) // capacity_bytes
        if capacity_bytes
        else 0
    )
    pressure_equivalent_pages = (
        pressure_numerator * equivalent_total_pages + pressure_denominator - 1
    ) // pressure_denominator
    return (
        min(equivalent_total_pages, active_equivalent_pages),
        min(equivalent_total_pages, pressure_equivalent_pages),
    )


def flat_pool_equivalent_pages(
    aggregate: Any,
    *,
    equivalent_total_pages: int,
) -> tuple[int, int]:
    """Return fixed-size active/pressure page scalars without metric rows."""

    active_bytes = _field(aggregate, "active_bytes")
    capacity_bytes = _field(aggregate, "capacity_bytes")
    pressure_numerator = _field(aggregate, "pressure_numerator")
    pressure_denominator = _field(aggregate, "pressure_denominator")
    return _equivalent_pages_from_totals(
        active_bytes=active_bytes,
        capacity_bytes=capacity_bytes,
        pressure_numerator=pressure_numerator,
        pressure_denominator=pressure_denominator,
        equivalent_total_pages=equivalent_total_pages,
    )


def summarize_flat_pool_aggregate(
    aggregate: Any,
    *,
    equivalent_total_pages: int,
) -> dict[str, int | float]:
    """Convert the scheduler's allocation-free scalar aggregate to metrics."""

    return _summary_from_totals(
        active_bytes=_field(aggregate, "active_bytes"),
        capacity_bytes=_field(aggregate, "capacity_bytes"),
        pressure_numerator=_field(aggregate, "pressure_numerator"),
        pressure_denominator=_field(aggregate, "pressure_denominator"),
        equivalent_total_pages=equivalent_total_pages,
    )


def collect_flat_pool_metrics(
    snapshots: Iterable[Any],
    *,
    equivalent_total_pages: int,
) -> tuple[dict[str, int | float], tuple[dict[str, int | float | str], ...]]:
    """Validate snapshots and return aggregate plus per-pool metric rows.

    Existing scheduler/load APIs expose scalar page counters.  For a
    heterogeneous pool-set, the returned equivalent page counts preserve their
    range while encoding byte-weighted active utilization and max-pool pressure;
    they are never formed by summing unrelated local page domains.
    """
    rows = tuple(snapshots)
    if not rows:
        raise ValueError("flat pool snapshots must be non-empty")

    active_bytes = 0
    capacity_bytes = 0
    pressure_numerator = 0
    pressure_denominator = 1
    seen_pool_ids: set[str] = set()
    metric_rows: list[dict[str, int | float | str]] = []
    for snapshot in rows:
        pool_id = _pool_id(snapshot)
        if pool_id in seen_pool_ids:
            raise ValueError(f"duplicate flat pool snapshot {pool_id!r}")
        seen_pool_ids.add(pool_id)
        usable = _field(snapshot, "usable_blocks")
        free = _field(snapshot, "free_blocks")
        active = _field(snapshot, "active_blocks")
        reserved = _field(snapshot, "reserved_blocks")
        bytes_per_block = _field(snapshot, "bytes_per_block")
        if usable <= 0:
            raise ValueError("flat pool usable_blocks must be positive")
        if (
            min(
                free,
                active,
                reserved,
                bytes_per_block,
            )
            < 0
        ):
            raise ValueError("flat pool snapshot counters must be non-negative")
        if free > usable or active > usable or active != usable - free:
            raise ValueError("flat pool active/free counters do not conserve capacity")
        pool_active_bytes = active * bytes_per_block
        pool_capacity_bytes = usable * bytes_per_block
        active_bytes += pool_active_bytes
        capacity_bytes += pool_capacity_bytes
        available_unreserved = max(0, min(usable, free - reserved))
        used_or_reserved = usable - available_unreserved
        pool_pressure = used_or_reserved / usable
        if used_or_reserved * pressure_denominator > pressure_numerator * usable:
            pressure_numerator = used_or_reserved
            pressure_denominator = usable
        cached_evictable = _field(snapshot, "cached_evictable_blocks")
        pinned_cached = _field(snapshot, "pinned_cached_blocks")
        if min(cached_evictable, pinned_cached) < 0:
            raise ValueError("flat pool snapshot counters must be non-negative")
        if cached_evictable > free or pinned_cached > active:
            raise ValueError("flat pool cached counters exceed their ownership domains")
        metric_rows.append(
            {
                "pool_id": pool_id,
                "usable_blocks": usable,
                "free_blocks": free,
                "active_blocks": active,
                "cached_evictable_blocks": cached_evictable,
                "pinned_cached_blocks": pinned_cached,
                "reserved_blocks": reserved,
                "bytes_per_block": bytes_per_block,
                "active_bytes": pool_active_bytes,
                "capacity_bytes": pool_capacity_bytes,
                "pressure": pool_pressure,
            }
        )

    summary = _summary_from_totals(
        active_bytes=active_bytes,
        capacity_bytes=capacity_bytes,
        pressure_numerator=pressure_numerator,
        pressure_denominator=pressure_denominator,
        equivalent_total_pages=equivalent_total_pages,
    )
    return summary, tuple(metric_rows)


__all__ = [
    "collect_flat_pool_metrics",
    "flat_pool_equivalent_pages",
    "summarize_flat_pool_aggregate",
]
