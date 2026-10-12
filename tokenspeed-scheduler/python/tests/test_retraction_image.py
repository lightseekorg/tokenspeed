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

"""The retraction image through the Python bindings, with a Host L2 cache.

A retracted request's hash-complete prefix travels as ordinary L2 entries
(a stream-ordered ``WriteBackOp`` whose Host rows stay pinned for the
request) and only the tail -- here empty, so the slot-state blob alone --
goes to the snapshot pool. The way back is one ``RestoreOp`` whose rows name
their Host tier; prefix blocks still cached on the Device are claimed rather
than copied. ``test_kv_cache.py`` covers the same cycle without L2.
"""

from __future__ import annotations

import pytest
from conftest import _advance, _find_forward_op, _finish, _spec

ts = pytest.importorskip("tokenspeed_scheduler")

PAGE = 2
DEVICE_PAGES = 17


def _make_config() -> ts.SchedulerConfig:
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = PAGE
    cfg.num_device_pages = DEVICE_PAGES
    cfg.num_host_pages = 33
    cfg.num_snapshot_pages = 9
    cfg.max_retracted_requests = 2
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.disable_l2_cache = False
    cfg.disable_prefix_cache = False
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="full",
            block_granularity=PAGE,
            total_pages=DEVICE_PAGES,
            retention=ts.CacheRetention.FullHistory,
            family=ts.CacheGroupFamily.History,
        )
    ]
    return cfg


def _ack(scheduler, event) -> None:
    execution_event = ts.ExecutionEvent()
    execution_event.add_event(event)
    scheduler.advance(execution_event)


def _ops(plan, kind) -> list:
    return [op for op in plan.cache if isinstance(op, kind)]


def _host_rows(op) -> dict[str, int]:
    """key -> Host page for every row of a WriteBackOp."""
    return {
        key: page
        for keys, pages in zip(op.content_hashes, op.dst_pages)
        for key, page in zip(keys, pages)
    }


def _ack_write_backs(scheduler, plan, host_rows: dict[str, int]) -> None:
    for op in _ops(plan, ts.Cache.WriteBackOp):
        host_rows.update(_host_rows(op))
        for op_id in op.op_ids:
            event = ts.Cache.WriteBackDoneEvent()
            event.op_id = int(op_id)
            _ack(scheduler, event)


def test_image_splits_into_l2_prefix_and_blob_and_waits_for_both_acks() -> None:
    scheduler = ts.Scheduler(_make_config())
    request_ids = ("a", "b", "c", "d")
    # a: two full pages plus a one-token tail; the others one page each.
    scheduler.submit_requests(
        [_spec("a", [1, 2, 3, 4, 5])]
        + [
            _spec(r, [100 * i + 1, 100 * i + 2])
            for i, r in enumerate(request_ids[1:], 1)
        ]
    )
    prefill_plan = scheduler.next_execution_plan()
    assert tuple(_find_forward_op(prefill_plan).request_ids) == request_ids
    host_rows: dict[str, int] = {}
    _ack_write_backs(scheduler, prefill_plan, host_rows)
    for index, request_id in enumerate(request_ids):
        _advance(scheduler, request_id, [1000 + index])

    # Decode until growth blocks every grower; a, with the most to release,
    # gives way in the round that grants the freed pages to the others.
    a_row: list[int] = []
    next_token = 2000
    for _ in range(16):
        plan = scheduler.next_execution_plan()
        op = _find_forward_op(plan)
        scheduled = () if op is None else tuple(op.request_ids)
        snapshots = _ops(plan, ts.Cache.SnapshotOp)
        if snapshots:
            assert scheduled == ("b", "c", "d")
            break
        if "a" in scheduled:
            a_row = list(dict(op.block_tables)["full"][scheduled.index("a")])
        _ack_write_backs(scheduler, plan, host_rows)
        for request_id in scheduled:
            _advance(scheduler, request_id, [next_token])
            next_token += 1
    else:
        pytest.fail("decode growth never retracted a request")

    [snapshot] = snapshots
    [write_back] = _ops(plan, ts.Cache.WriteBackOp)
    assert scheduler.retracted_size() == 1
    assert scheduler.request_token_size("a") == 11
    # 10 computed tokens = five hash-complete pages: the two prefill pages are
    # Host-cached already (pinned now), the three decode pages are stored by
    # a stream-ordered write-back that releases its Device sources at once.
    assert len(a_row) == 5
    assert list(write_back.source_pinned) == [False]
    [stored_pages] = write_back.src_pages
    assert sorted(stored_pages) == sorted(a_row[2:])
    [stored_keys] = write_back.content_hashes
    assert all(stored_keys)
    assert not set(stored_keys) & set(
        host_rows
    ), "only pages not yet on Host are stored"
    assert scheduler.host_pool_pinned_blocks() == 2
    # No tail: the snapshot leg carries only the slot-state blob.
    assert list(snapshot.request_ids) == ["a"]
    assert [list(row) for row in snapshot.src_pages] == [[]]
    assert [list(row) for row in snapshot.dst_pages] == [[]]
    [snapshot_slot] = snapshot.snapshot_slots
    assert snapshot_slot >= 1
    assert scheduler.snapshot_pool_free_blocks() == 8

    for request_id in scheduled:
        _advance(scheduler, request_id, [next_token])
        next_token += 1
    # Both legs must land before a can be restored: the snapshot ACK alone,
    # with the Device otherwise idle, restores nothing.
    done = ts.Cache.SnapshotDoneEvent()
    done.op_id = int(snapshot.op_ids[0])
    _ack(scheduler, done)
    for request_id in ("b", "c", "d"):
        _finish(scheduler, request_id)
    idle = scheduler.next_execution_plan()
    assert _find_forward_op(idle) is None
    assert not _ops(idle, ts.Cache.RestoreOp)
    assert scheduler.retracted_size() == 1
    _ack_write_backs(scheduler, idle, host_rows)  # the finished requests' own stores

    landed = ts.Cache.WriteBackDoneEvent()
    landed.op_id = int(write_back.op_ids[0])
    _ack(scheduler, landed)
    host_rows.update(_host_rows(write_back))
    assert scheduler.host_pool_pinned_blocks() == 5

    restore_plan = scheduler.next_execution_plan()
    assert _find_forward_op(restore_plan) is None
    [restore] = _ops(restore_plan, ts.Cache.RestoreOp)
    assert list(restore.request_ids) == ["a"]
    assert list(restore.snapshot_slots) == [snapshot_slot]
    [tiers] = restore.source_tiers
    [sources] = restore.src_pages
    [destinations] = restore.dst_pages
    [keys] = restore.content_hashes
    # Every copied row is an L2 entry (its key travels with it); the prefix
    # pages still cached on the Device are claimed and do not appear.
    assert set(tiers) == {int(ts.Cache.HostTier.L2)}
    assert 0 < len(sources) <= 5
    assert [host_rows[key] for key in keys] == list(sources)
    # A copy fills its destination whole: the plan zeroes none of them.
    zeroed = set(dict(restore_plan.pages_to_zero)["full"])
    assert not set(destinations) & zeroed
    assert scheduler.retracted_size() == 1
    assert scheduler.decoding_size() == 0

    restored = ts.Cache.RestoreDoneEvent()
    restored.op_id = int(restore.op_ids[0])
    _ack(scheduler, restored)
    assert scheduler.retracted_size() == 0
    assert scheduler.decoding_size() == 1
    # The pins are released; the entries stay published.
    assert scheduler.host_pool_pinned_blocks() == 0

    resume = _find_forward_op(scheduler.next_execution_plan())
    assert tuple(resume.request_ids) == ("a",)
    assert list(resume.input_lengths) == [1]
    assert list(resume.extend_prefix_lens) == []
    row = list(dict(resume.block_tables)["full"])[0]
    row = list(row)
    assert len(row) == len(a_row) + 1
    copied_slots = {row.index(page) for page in destinations}
    assert copied_slots <= set(range(5))
    for slot in set(range(5)) - copied_slots:
        assert row[slot] == a_row[slot], "a claimed slot keeps its Device block"
        assert row[slot] not in zeroed
    assert row[5] in zeroed
    assert len(set(row)) == len(row)


def test_finish_while_retracted_releases_the_pins_at_the_acks() -> None:
    scheduler = ts.Scheduler(_make_config())
    request_ids = ("a", "b", "c", "d")
    scheduler.submit_requests(
        [_spec("a", [1, 2, 3, 4, 5])]
        + [
            _spec(r, [100 * i + 1, 100 * i + 2])
            for i, r in enumerate(request_ids[1:], 1)
        ]
    )
    plan = scheduler.next_execution_plan()
    host_rows: dict[str, int] = {}
    _ack_write_backs(scheduler, plan, host_rows)
    for index, request_id in enumerate(request_ids):
        _advance(scheduler, request_id, [1000 + index])
    next_token = 2000
    for _ in range(16):
        plan = scheduler.next_execution_plan()
        snapshots = _ops(plan, ts.Cache.SnapshotOp)
        if snapshots:
            break
        _ack_write_backs(scheduler, plan, host_rows)
        for request_id in _find_forward_op(plan).request_ids:
            _advance(scheduler, request_id, [next_token])
            next_token += 1
    else:
        pytest.fail("decode growth never retracted a request")
    [snapshot] = snapshots
    [write_back] = _ops(plan, ts.Cache.WriteBackOp)
    assert scheduler.host_pool_pinned_blocks() == 2

    _finish(scheduler, "a")
    assert scheduler.retracted_size() == 0
    assert scheduler.waiting_size() == 0
    # The already-cached entries are unpinned at once; the in-flight ones
    # become plain evictable entries when their store lands.
    assert scheduler.host_pool_pinned_blocks() == 0
    landed = ts.Cache.WriteBackDoneEvent()
    landed.op_id = int(write_back.op_ids[0])
    _ack(scheduler, landed)
    done = ts.Cache.SnapshotDoneEvent()
    done.op_id = int(snapshot.op_ids[0])
    _ack(scheduler, done)
    assert scheduler.host_pool_pinned_blocks() == 0
    assert scheduler.snapshot_pool_free_blocks() == 8
    assert scheduler.decoding_size() == 3
