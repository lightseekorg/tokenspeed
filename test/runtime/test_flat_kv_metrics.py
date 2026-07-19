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

"""Focused contracts for the allocation-free Flat KV aggregate metrics ABI."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tokenspeed.runtime.flat_kv_metrics import (
    collect_flat_pool_metrics,
    flat_pool_equivalent_pages,
    summarize_flat_pool_aggregate,
)


@pytest.mark.parametrize(
    ("aggregate", "equivalent_pages", "expected"),
    (
        (
            SimpleNamespace(
                active_bytes=30,
                capacity_bytes=100,
                pressure_numerator=3,
                pressure_denominator=4,
            ),
            10,
            (3, 8, 0.3, 0.75),
        ),
        (
            SimpleNamespace(
                active_bytes=0,
                capacity_bytes=0,
                pressure_numerator=0,
                pressure_denominator=1,
            ),
            10,
            (0, 0, 0.0, 0.0),
        ),
        (
            SimpleNamespace(
                active_bytes=0,
                capacity_bytes=3,
                pressure_numerator=2,
                pressure_denominator=3,
            ),
            9,
            (0, 6, 0.0, 2 / 3),
        ),
    ),
)
def test_scalar_aggregate_maps_to_equivalent_pages(
    aggregate, equivalent_pages, expected
):
    summary = summarize_flat_pool_aggregate(
        aggregate,
        equivalent_total_pages=equivalent_pages,
    )

    assert (
        flat_pool_equivalent_pages(
            aggregate,
            equivalent_total_pages=equivalent_pages,
        )
        == expected[:2]
    )
    assert (
        summary["active_equivalent_pages"],
        summary["pressure_equivalent_pages"],
        summary["byte_utilization"],
        summary["pressure"],
    ) == expected


def test_scalar_aggregate_matches_detailed_snapshot_math():
    snapshots = (
        {
            "pool_id": "history",
            "usable_blocks": 4,
            "free_blocks": 2,
            "active_blocks": 2,
            "cached_evictable_blocks": 0,
            "pinned_cached_blocks": 0,
            "reserved_blocks": 1,
            "bytes_per_block": 100,
        },
        {
            "pool_id": "state",
            "usable_blocks": 8,
            "free_blocks": 6,
            "active_blocks": 2,
            "cached_evictable_blocks": 0,
            "pinned_cached_blocks": 0,
            "reserved_blocks": 0,
            "bytes_per_block": 25,
        },
    )
    detailed, _ = collect_flat_pool_metrics(
        snapshots,
        equivalent_total_pages=20,
    )
    aggregate = SimpleNamespace(
        active_bytes=250,
        capacity_bytes=600,
        pressure_numerator=3,
        pressure_denominator=4,
    )

    assert (
        summarize_flat_pool_aggregate(
            aggregate,
            equivalent_total_pages=20,
        )
        == detailed
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("active_bytes", -1),
        ("capacity_bytes", -1),
        ("pressure_numerator", -1),
        ("pressure_denominator", 0),
    ),
)
def test_scalar_aggregate_rejects_invalid_counters(field, value):
    values = {
        "active_bytes": 0,
        "capacity_bytes": 1,
        "pressure_numerator": 0,
        "pressure_denominator": 1,
    }
    values[field] = value

    with pytest.raises(ValueError):
        summarize_flat_pool_aggregate(
            SimpleNamespace(**values),
            equivalent_total_pages=1,
        )


def _event_loop_with_metrics(*, enabled: bool):
    event_loop_module = pytest.importorskip("tokenspeed.runtime.engine.event_loop")

    class Scheduler:
        snapshot_calls = 0

        @staticmethod
        def flat_pool_aggregate():
            return SimpleNamespace(
                active_bytes=30,
                capacity_bytes=100,
                pressure_numerator=3,
                pressure_denominator=4,
            )

        def flat_pool_snapshots(self):
            self.snapshot_calls += 1
            return (
                {
                    "pool_id": "history",
                    "usable_blocks": 4,
                    "free_blocks": 2,
                    "active_blocks": 2,
                    "cached_evictable_blocks": 0,
                    "pinned_cached_blocks": 0,
                    "reserved_blocks": 1,
                    "bytes_per_block": 25,
                },
            )

        @staticmethod
        def waiting_size():
            return 0

    loop = event_loop_module.EventLoop.__new__(event_loop_module.EventLoop)
    loop.scheduler = Scheduler()
    loop.metrics = SimpleNamespace(enabled=enabled)
    loop.max_total_num_tokens = 10
    loop.server_args = SimpleNamespace(block_size=1)
    loop._next_flat_pool_detail_at = 0.0
    return event_loop_module, loop


def test_metrics_disabled_iteration_never_materializes_pool_rows():
    _, loop = _event_loop_with_metrics(enabled=False)

    stats = loop._get_scheduler_stats()

    assert loop.scheduler.snapshot_calls == 0
    assert stats["num_active_pages"] == 3
    assert stats["num_cached_pages"] == 8
    assert stats["flat_summary"] is None
    assert stats["flat_pool_metrics"] == ()


def test_enabled_pool_rows_are_rate_limited(monkeypatch):
    event_loop_module, loop = _event_loop_with_metrics(enabled=True)
    monkeypatch.setattr(event_loop_module.time, "monotonic", lambda: 10.0)

    first = loop._get_scheduler_stats()
    second = loop._get_scheduler_stats()

    assert loop.scheduler.snapshot_calls == 1
    assert len(first["flat_pool_metrics"]) == 1
    assert second["flat_pool_metrics"] == ()
