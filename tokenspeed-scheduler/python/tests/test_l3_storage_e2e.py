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

"""End-to-end scheduler tests for Mooncake Store L3 under compact Host KV.

CI does not run a Mooncake master. These tests drive the scheduler control
plane the runtime uses with ``--kvstore-storage-backend mooncake`` (and the
in-process ``memory`` stand-in): cross-instance ``register_storage_keys``
after ``batch_exists``, write-back object keys, and L3-only load-back with
``prefetch_from_storage``.
"""

from __future__ import annotations

import pytest
from conftest import _advance, _finish, _spec

ts = pytest.importorskip("tokenspeed_scheduler")


def _l3_config(
    *,
    num_device_pages: int = 32,
    num_host_pages: int = 32,
    with_swa: bool = False,
) -> ts.SchedulerConfig:
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 2
    cfg.num_device_pages = num_device_pages
    cfg.num_host_pages = num_host_pages
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.enable_l3_storage = True
    cfg.disable_l2_cache = False
    cfg.disable_prefix_cache = False
    groups = [
        ts.CacheGroupConfig(
            group_id="full",
            rows_per_page=cfg.prefix_granularity,
            entry_stride_tokens=1,
            total_pages=cfg.num_device_pages,
            retention=ts.CacheRetention.FullHistory,
            family=ts.CacheGroupFamily.History,
        )
    ]
    if with_swa:
        groups.append(
            ts.CacheGroupConfig(
                group_id="swa",
                rows_per_page=cfg.prefix_granularity,
                entry_stride_tokens=1,
                total_pages=cfg.num_device_pages,
                retention=ts.CacheRetention.SlidingWindow,
                sliding_window_tokens=4,
                family=ts.CacheGroupFamily.State,
            )
        )
    cfg.cache_groups = groups
    return cfg


def _find_write_back(plan):
    for op in plan.cache:
        if isinstance(op, ts.Cache.WriteBackOp):
            return op
    return None


def _find_load_back(plan):
    for op in plan.cache:
        if isinstance(op, ts.Cache.LoadBackOp):
            return op
    return None


def _ack_write_back(scheduler, op_id: int) -> None:
    event = ts.Cache.WriteBackDoneEvent()
    event.op_id = int(op_id)
    execution_event = ts.ExecutionEvent()
    execution_event.add_event(event)
    scheduler.advance(execution_event)


def _ack_load_back(scheduler, op_id: int) -> None:
    event = ts.Cache.LoadBackDoneEvent()
    event.op_id = int(op_id)
    execution_event = ts.ExecutionEvent()
    execution_event.add_event(event)
    scheduler.advance(execution_event)


def _run_to_finalize(scheduler, spec) -> object:
    scheduler.submit_requests([spec])
    scheduler.next_execution_plan()  # prefill
    _advance(scheduler, spec.request_id, [9001])
    return scheduler.next_execution_plan()  # PrefillDone -> Decoding: drain


def _finish_and_reap(scheduler, request_id: str) -> None:
    _advance(scheduler, request_id, [9002])
    _finish(scheduler, request_id)
    scheduler.next_execution_plan()


def _prefetch_flags(load_op) -> list[int]:
    return [int(flag) for row in load_op.prefetch_from_storage for flag in row]


def test_l3_cold_miss_does_not_prefetch() -> None:
    scheduler = ts.Scheduler(_l3_config())
    scheduler.submit_requests([_spec("r1", list(range(1, 9)))])
    plan = scheduler.next_execution_plan()
    assert _find_load_back(plan) is None
    assert any(dict(op.block_tables) for op in plan.forward)


def test_l3_register_storage_keys_emits_prefetch_loadback() -> None:
    """Cross-instance Mooncake path: batch_exists → register → prefetch H2D."""

    scheduler = ts.Scheduler(_l3_config())
    tokens = list(range(1, 9))
    hashes = scheduler.prefix_hashes_for_tokens(tokens)
    assert hashes
    group_ids, expanded, offsets = scheduler.expand_prefix_keys(hashes)
    assert group_ids
    scheduler.register_storage_keys(group_ids, expanded, offsets)

    scheduler.submit_requests([_spec("r1", tokens)])
    plan = scheduler.next_execution_plan()
    load = _find_load_back(plan)
    assert load is not None, "L3-only prefix must emit LoadBackOp"
    assert list(load.op_ids)
    flags = _prefetch_flags(load)
    assert flags
    assert all(flag != 0 for flag in flags)
    assert any(row for row in load.content_hashes)
    _ack_load_back(scheduler, load.op_ids[0])


def test_l3_unregister_storage_keys_removes_stale_remote_hit() -> None:
    scheduler = ts.Scheduler(_l3_config())
    tokens = list(range(1, 9))
    hashes = scheduler.prefix_hashes_for_tokens(tokens)
    group_ids, expanded, offsets = scheduler.expand_prefix_keys(hashes)
    scheduler.register_storage_keys(group_ids, expanded, offsets)
    scheduler.unregister_storage_keys(group_ids, expanded, offsets)

    scheduler.submit_requests([_spec("r1", tokens)])
    plan = scheduler.next_execution_plan()
    assert _find_load_back(plan) is None


def test_l3_writeback_carries_object_keys() -> None:
    scheduler = ts.Scheduler(_l3_config(with_swa=True))
    spec = _spec("r1", list(range(1, 9)))
    finalize = _run_to_finalize(scheduler, spec)
    write_back = _find_write_back(finalize)
    assert write_back is not None, "finalize must drain a streaming Host write-back"
    assert list(write_back.op_ids)
    hashes = [content_hash for row in write_back.content_hashes for content_hash in row]
    assert hashes
    assert all(content_hash for content_hash in hashes)
    offsets = [int(offset) for row in write_back.page_offsets for offset in row]
    assert len(offsets) == len(hashes)

    _finish_and_reap(scheduler, spec.request_id)
    _ack_write_back(scheduler, write_back.op_ids[0])
    scheduler.next_execution_plan()


def test_l3_host_eviction_still_prefetches_registered_prefix() -> None:
    """Same-instance reuse after Host is full: L3 keys outlive L2 eviction."""

    cfg = _l3_config(num_device_pages=13, num_host_pages=7, with_swa=True)
    scheduler = ts.Scheduler(cfg)

    r1 = _spec("r1", list(range(1, 9)))
    wb1 = _find_write_back(_run_to_finalize(scheduler, r1))
    assert wb1 is not None
    _finish_and_reap(scheduler, "r1")
    _ack_write_back(scheduler, wb1.op_ids[0])
    scheduler.next_execution_plan()

    churn = _spec("churn", list(range(501, 511)))
    wb2 = _find_write_back(_run_to_finalize(scheduler, churn))
    assert wb2 is not None, "a full Host pool must replace r1's committed entries"
    _finish_and_reap(scheduler, "churn")
    _ack_write_back(scheduler, wb2.op_ids[0])
    scheduler.next_execution_plan()

    scheduler.submit_requests([_spec("r3", list(range(1, 11)))])
    plan = scheduler.next_execution_plan()
    load = _find_load_back(plan)
    assert load is not None, "Host-evicted L3 prefix must still emit LoadBackOp"
    flags = _prefetch_flags(load)
    assert flags
    assert all(flag != 0 for flag in flags), "replaced Host pages must prefetch from L3"
    _ack_load_back(scheduler, load.op_ids[0])
