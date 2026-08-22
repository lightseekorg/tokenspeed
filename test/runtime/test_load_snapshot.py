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

"""Tests for the scheduler-published load snapshot cache contract."""

from __future__ import annotations

import pytest

from tokenspeed.runtime.engine.io_struct import LoadSnapshot
from tokenspeed.runtime.engine.load_snapshot import LoadSnapshotStore

VALID_FOR_SECONDS = 1.0


class FakeClock:
    """A controllable monotonic clock for cache expiry behavior."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_snapshot(**overrides) -> LoadSnapshot:
    fields = dict(
        epoch="boot-a",
        sequence=1,
        dp_rank=0,
        num_running_reqs=2,
        num_waiting_reqs=3,
        num_active_pages=4,
        num_used_pages=5,
        max_total_pages=10,
        valid_for_ms=int(VALID_FOR_SECONDS * 1_000),
    )
    fields.update(overrides)
    return LoadSnapshot(**fields)


def test_load_snapshot_is_immutable_and_positional():
    """Snapshots cannot be mutated after their wire values are assigned."""
    snapshot = LoadSnapshot("boot-a", 1, 0, 2, 3, 4, 5, 10, 1_000)

    with pytest.raises(AttributeError):
        snapshot.sequence = 2


def test_store_rejects_duplicate_and_retired_epoch_without_refreshing_age():
    """Duplicate and retired messages cannot revive stale rank state."""
    clock = FakeClock()
    store = LoadSnapshotStore(dp_size=1, clock=clock)

    assert store.accept(make_snapshot(epoch="a", sequence=1))
    clock.advance(0.1)
    assert not store.accept(make_snapshot(epoch="a", sequence=1))
    assert store.accept(make_snapshot(epoch="b", sequence=1))
    assert not store.accept(make_snapshot(epoch="a", sequence=2))


def test_store_rejects_non_increasing_sequence_in_current_epoch():
    """Only strictly newer versions can replace a rank's current epoch."""
    store = LoadSnapshotStore(dp_size=1, clock=FakeClock())

    assert store.accept(make_snapshot(sequence=2))
    assert not store.accept(make_snapshot(sequence=2))
    assert not store.accept(make_snapshot(sequence=1))
    assert store.accept(make_snapshot(sequence=3))


@pytest.mark.parametrize(
    "overrides",
    [
        {"dp_rank": -1},
        {"dp_rank": 2},
        {"sequence": -1},
        {"num_running_reqs": -1},
        {"num_waiting_reqs": -1},
        {"num_active_pages": -1},
        {"num_used_pages": -1},
        {"max_total_pages": 0},
        {"num_active_pages": 11},
        {"num_used_pages": 11},
        {"valid_for_ms": -1},
    ],
)
def test_store_rejects_invalid_snapshot_values(overrides):
    """Malformed values never enter the replica cache."""
    store = LoadSnapshotStore(dp_size=2, clock=FakeClock())

    assert not store.accept(make_snapshot(**overrides))
    assert store.fresh_snapshots() == []


def test_store_projects_only_a_complete_fresh_rank_set():
    """Public loads stay unavailable until every rank has a live snapshot."""
    clock = FakeClock()
    store = LoadSnapshotStore(dp_size=2, clock=clock)

    assert store.project_loads() == []
    assert store.accept(make_snapshot(dp_rank=0))
    assert store.project_loads() == []
    assert store.accept(make_snapshot(dp_rank=1))
    loads = store.project_loads()
    assert [load.dp_rank for load in loads] == [0, 1]
    assert [(load.num_reqs, load.num_waiting_reqs, load.num_pages) for load in loads] == [
        (5, 3, 5),
        (5, 3, 5),
    ]

    clock.advance(VALID_FOR_SECONDS + 0.001)

    assert not store.is_complete_fresh()
    assert store.fresh_snapshots() == []
    assert store.project_loads() == []


def test_store_projection_checks_completeness_at_one_clock_read():
    """A rank expiring during projection cannot leak a partial public result."""
    reads = iter((0.0, 0.5, 0.999, 1.001))
    store = LoadSnapshotStore(dp_size=2, clock=lambda: next(reads))

    assert store.accept(make_snapshot(dp_rank=0))
    assert store.accept(make_snapshot(dp_rank=1))

    assert [load.dp_rank for load in store.project_loads()] == [0, 1]


def test_store_uses_local_receipt_time_and_does_not_refresh_rejection_age():
    """Only accepted messages extend a snapshot's locally measured lifetime."""
    clock = FakeClock()
    store = LoadSnapshotStore(dp_size=1, clock=clock)

    assert store.accept(make_snapshot())
    clock.advance(0.9)
    assert not store.accept(make_snapshot())
    clock.advance(0.101)

    assert store.fresh_snapshots() == []


def test_store_rejects_non_positive_dp_size():
    """A store must have at least one rank to form a complete replica."""
    with pytest.raises(ValueError, match="dp_size"):
        LoadSnapshotStore(dp_size=0)
