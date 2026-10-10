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
after ``batch_exists``, write-back object keys, and the pre-admission
``PrefetchOp`` that fills Host before an L3 hit is admitted as a Host hit.
"""

from __future__ import annotations

import inspect

import pytest
from conftest import _advance, _finish, _spec

ts = pytest.importorskip("tokenspeed_scheduler")


def _l3_config(
    *,
    num_device_pages: int,
    num_host_pages: int,
    with_swa: bool,
    min_prefetch_pages: int,
) -> ts.SchedulerConfig:
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 2
    cfg.num_device_pages = num_device_pages
    cfg.num_host_pages = num_host_pages
    cfg.num_snapshot_pages = num_device_pages  # ample: nothing here is retracted
    cfg.max_retracted_requests = 8
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.enable_l3_storage = True
    cfg.l3_prefetch_min_pages = min_prefetch_pages
    cfg.disable_l2_cache = False
    cfg.disable_prefix_cache = False
    groups = [
        ts.CacheGroupConfig(
            group_id="full",
            block_granularity=cfg.prefix_granularity,
            total_pages=cfg.num_device_pages,
            retention=ts.CacheRetention.FullHistory,
            family=ts.CacheGroupFamily.History,
        )
    ]
    if with_swa:
        groups.append(
            ts.CacheGroupConfig(
                group_id="swa",
                block_granularity=cfg.prefix_granularity,
                total_pages=cfg.num_device_pages,
                retention=ts.CacheRetention.SlidingWindow,
                sliding_window_tokens=4,
                family=ts.CacheGroupFamily.History,
            )
        )
    cfg.cache_groups = groups
    return cfg


def _find_op(plan, kind):
    for op in plan.cache:
        if isinstance(op, kind):
            return op
    return None


def _ack_write_back(scheduler, op_id: int) -> None:
    event = ts.Cache.WriteBackDoneEvent()
    event.op_id = int(op_id)
    execution_event = ts.ExecutionEvent()
    execution_event.add_event(event)
    scheduler.advance(execution_event)


def _ack_load_back(scheduler, op_id: int) -> None:
    execution_event = ts.ExecutionEvent()
    execution_event.add_event(ts.Cache.LoadBackDoneEvent(int(op_id)))
    scheduler.advance(execution_event)


def _ack_prefetch(scheduler, op_id: int, landed_pages: int) -> None:
    execution_event = ts.ExecutionEvent()
    execution_event.add_event(ts.Cache.PrefetchDoneEvent(int(op_id), int(landed_pages)))
    scheduler.advance(execution_event)


def _register_prompt(scheduler, tokens):
    hashes = scheduler.prefix_hashes_for_tokens(tokens)
    assert hashes
    group_ids, expanded, offsets = scheduler.expand_prefix_keys(hashes)
    assert group_ids
    scheduler.register_storage_keys(group_ids, expanded, offsets)
    return hashes, (group_ids, expanded, offsets)


def _run_to_finalize(scheduler, spec) -> object:
    scheduler.submit_requests([spec])
    scheduler.next_execution_plan()  # prefill
    _advance(scheduler, spec.request_id, [9001])
    return scheduler.next_execution_plan()  # PrefillDone -> Decoding: drain


def _finish_and_reap(scheduler, request_id: str) -> None:
    _advance(scheduler, request_id, [9002])
    _finish(scheduler, request_id)
    scheduler.next_execution_plan()


def test_l3_config_requires_every_knob() -> None:
    parameters = inspect.signature(_l3_config).parameters
    for name in (
        "num_device_pages",
        "num_host_pages",
        "with_swa",
        "min_prefetch_pages",
    ):
        assert parameters[name].default is inspect.Parameter.empty


def test_l3_requires_an_explicit_prefetch_threshold() -> None:
    cfg = _l3_config(
        num_device_pages=32, num_host_pages=32, with_swa=False, min_prefetch_pages=1
    )
    cfg.l3_prefetch_min_pages = 0
    with pytest.raises(ValueError, match="l3_prefetch_min_pages"):
        ts.Scheduler(cfg)
    cfg.enable_l3_storage = False
    cfg.l3_prefetch_min_pages = 2
    with pytest.raises(ValueError, match="l3_prefetch_min_pages"):
        ts.Scheduler(cfg)


def test_l3_cold_miss_does_not_prefetch() -> None:
    scheduler = ts.Scheduler(
        _l3_config(
            num_device_pages=32, num_host_pages=32, with_swa=False, min_prefetch_pages=1
        )
    )
    scheduler.submit_requests([_spec("r1", list(range(1, 9)))])
    plan = scheduler.next_execution_plan()
    assert _find_op(plan, ts.Cache.PrefetchOp) is None
    assert _find_op(plan, ts.Cache.LoadBackOp) is None
    assert any(dict(op.block_tables) for op in plan.forward)


def test_l3_register_storage_keys_emits_a_prefetch_before_admission() -> None:
    """Cross-instance Mooncake path: batch_exists -> register -> PrefetchOp ->
    PrefetchDone -> admission as a Host hit with plain L2 rows."""

    scheduler = ts.Scheduler(
        _l3_config(
            num_device_pages=32, num_host_pages=32, with_swa=False, min_prefetch_pages=1
        )
    )
    tokens = list(range(1, 9))
    hashes, _ = _register_prompt(scheduler, tokens)

    scheduler.submit_requests([_spec("r1", tokens)])
    plan = scheduler.next_execution_plan()
    assert (
        _find_op(plan, ts.Cache.LoadBackOp) is None
    ), "nothing is loaded before the objects are on Host"
    assert not list(plan.forward[0].request_ids) if plan.forward else True
    prefetch = _find_op(plan, ts.Cache.PrefetchOp)
    assert prefetch is not None, "L3-only prefix must emit a PrefetchOp"
    assert list(prefetch.request_ids) == ["r1"]
    assert list(prefetch.first_pages) == [0]
    assert list(prefetch.num_pages) == [len(hashes)]
    [rows] = prefetch.content_hashes
    assert list(rows) == hashes, "one full-group row per prefix page, in prefix order"
    [page_indices] = prefetch.page_indices
    assert list(page_indices) == list(range(len(hashes)))
    assert all(page > 0 for page in prefetch.host_pages[0])
    assert scheduler.waiting_size() == 1, "Prefetching counts as waiting"
    assert scheduler.active_lcm_blocks() == 0, "no Device page is held while fetching"

    _ack_prefetch(scheduler, prefetch.op_ids[0], len(hashes))
    assert scheduler.host_pool_pinned_blocks() == len(
        hashes
    ), "the request pins what landed until admission"
    admit = scheduler.next_execution_plan()
    load = _find_op(admit, ts.Cache.LoadBackOp)
    assert (
        load is not None
    ), "the landed pages come back from Host under the first chunk"
    assert not hasattr(load, "prefetch_from_storage")
    op = admit.forward[0]
    assert list(op.request_ids) == ["r1"]
    assert list(op.extend_prefix_lens) == [2 * len(hashes)]
    _ack_load_back(scheduler, load.op_ids[0])
    assert scheduler.host_pool_pinned_blocks() == 0


def test_l3_partial_landing_admits_on_the_landed_prefix() -> None:
    scheduler = ts.Scheduler(
        _l3_config(
            num_device_pages=32, num_host_pages=32, with_swa=False, min_prefetch_pages=1
        )
    )
    tokens = list(range(1, 9))
    hashes, _ = _register_prompt(scheduler, tokens)
    scheduler.submit_requests([_spec("r1", tokens)])
    prefetch = _find_op(scheduler.next_execution_plan(), ts.Cache.PrefetchOp)
    assert prefetch is not None
    assert list(prefetch.num_pages) == [3]

    _ack_prefetch(scheduler, prefetch.op_ids[0], 1)
    assert (
        scheduler.host_pool_pinned_blocks() == 1
    ), "only the landed page is an entry, pinned for r1"
    admit = scheduler.next_execution_plan()
    assert (
        _find_op(admit, ts.Cache.PrefetchOp) is None
    ), "the unlanded keys are forgotten"
    op = admit.forward[0]
    assert list(op.request_ids) == ["r1"]
    assert list(op.extend_prefix_lens) == [2]
    assert list(op.input_lengths) == [6]


def test_l3_short_host_pool_truncates_the_prefetch() -> None:
    """A Host-starved L3 hit fetches what fits and computes the rest."""

    scheduler = ts.Scheduler(
        _l3_config(
            num_device_pages=32, num_host_pages=3, with_swa=False, min_prefetch_pages=1
        )
    )
    tokens = list(range(1, 9))
    _register_prompt(scheduler, tokens)

    scheduler.submit_requests([_spec("r1", tokens)])
    prefetch = _find_op(scheduler.next_execution_plan(), ts.Cache.PrefetchOp)
    assert prefetch is not None
    assert list(prefetch.num_pages) == [2], "two usable Host pages"
    _ack_prefetch(scheduler, prefetch.op_ids[0], 2)
    plan = scheduler.next_execution_plan()
    assert plan.forward
    op = plan.forward[0]
    assert list(op.extend_prefix_lens) == [4]
    assert list(op.input_lengths) == [4]
    assert op.extend_prefix_lens[0] + op.input_lengths[0] == op.prefill_lengths[0]


def test_l3_prefetch_below_the_threshold_is_computed() -> None:
    scheduler = ts.Scheduler(
        _l3_config(
            num_device_pages=32, num_host_pages=32, with_swa=False, min_prefetch_pages=4
        )
    )
    tokens = list(range(1, 9))
    _register_prompt(scheduler, tokens)  # three prefix pages
    scheduler.submit_requests([_spec("r1", tokens)])
    plan = scheduler.next_execution_plan()
    assert _find_op(plan, ts.Cache.PrefetchOp) is None
    op = plan.forward[0]
    assert list(op.request_ids) == ["r1"]
    assert list(op.extend_prefix_lens) == [0]


def test_l3_host_shortage_skips_a_page_it_cannot_fetch_whole() -> None:
    """A fine-group Host shortage must not fetch half a prefix page."""

    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 4
    cfg.num_device_pages = 32
    cfg.num_host_pages = 2
    cfg.num_snapshot_pages = 32
    cfg.max_retracted_requests = 8
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.enable_l3_storage = True
    cfg.l3_prefetch_min_pages = 1
    cfg.disable_l2_cache = False
    cfg.disable_prefix_cache = False
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="full_fine",
            block_granularity=2,
            total_pages=cfg.num_device_pages,
            retention=ts.CacheRetention.FullHistory,
            family=ts.CacheGroupFamily.History,
        ),
        ts.CacheGroupConfig(
            group_id="full_coarse",
            block_granularity=4,
            total_pages=cfg.num_device_pages,
            retention=ts.CacheRetention.FullHistory,
            family=ts.CacheGroupFamily.History,
        ),
    ]
    scheduler = ts.Scheduler(cfg)
    tokens = list(range(1, 9))
    _register_prompt(scheduler, tokens)

    scheduler.submit_requests([_spec("r1", tokens)])
    plan = scheduler.next_execution_plan()
    assert (
        _find_op(plan, ts.Cache.PrefetchOp) is None
    ), "one usable Host page against three rows"
    assert plan.forward
    op = plan.forward[0]
    assert list(op.extend_prefix_lens) == [0]
    assert list(op.input_lengths) == [8]
    assert op.extend_prefix_lens[0] + op.input_lengths[0] == op.prefill_lengths[0]


def test_l3_prefetching_request_holds_no_head_of_line() -> None:
    scheduler = ts.Scheduler(
        _l3_config(
            num_device_pages=32, num_host_pages=32, with_swa=False, min_prefetch_pages=1
        )
    )
    tokens = list(range(1, 9))
    _register_prompt(scheduler, tokens)
    scheduler.submit_requests(
        [_spec("waiter", tokens), _spec("later", list(range(101, 105)))]
    )
    plan = scheduler.next_execution_plan()
    prefetch = _find_op(plan, ts.Cache.PrefetchOp)
    assert prefetch is not None
    assert list(prefetch.request_ids) == ["waiter"]
    assert list(plan.forward[0].request_ids) == [
        "later"
    ], "a later prompt is admitted past the prefetch"


def test_l3_unregister_storage_keys_removes_stale_remote_hit() -> None:
    scheduler = ts.Scheduler(
        _l3_config(
            num_device_pages=32, num_host_pages=32, with_swa=False, min_prefetch_pages=1
        )
    )
    tokens = list(range(1, 9))
    _, (group_ids, expanded, offsets) = _register_prompt(scheduler, tokens)
    scheduler.unregister_storage_keys(group_ids, expanded, offsets)

    scheduler.submit_requests([_spec("r1", tokens)])
    plan = scheduler.next_execution_plan()
    assert _find_op(plan, ts.Cache.PrefetchOp) is None
    assert _find_op(plan, ts.Cache.LoadBackOp) is None


def test_l3_writeback_carries_object_keys() -> None:
    scheduler = ts.Scheduler(
        _l3_config(
            num_device_pages=32, num_host_pages=32, with_swa=True, min_prefetch_pages=1
        )
    )
    spec = _spec("r1", list(range(1, 9)))
    finalize = _run_to_finalize(scheduler, spec)
    write_back = _find_op(finalize, ts.Cache.WriteBackOp)
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
    """Host eviction keeps Mooncake objects; submit-time register restores the shadow."""

    cfg = _l3_config(
        num_device_pages=13, num_host_pages=7, with_swa=True, min_prefetch_pages=1
    )
    scheduler = ts.Scheduler(cfg)

    r1 = _spec("r1", list(range(1, 9)))
    wb1 = _find_op(_run_to_finalize(scheduler, r1), ts.Cache.WriteBackOp)
    assert wb1 is not None
    _finish_and_reap(scheduler, "r1")
    _ack_write_back(scheduler, wb1.op_ids[0])
    scheduler.next_execution_plan()

    churn = _spec("churn", list(range(501, 511)))
    wb2 = _find_op(_run_to_finalize(scheduler, churn), ts.Cache.WriteBackOp)
    assert wb2 is not None, "a full Host pool must replace r1's committed entries"
    _finish_and_reap(scheduler, "churn")
    _ack_write_back(scheduler, wb2.op_ids[0])
    scheduler.next_execution_plan()

    r3_tokens = list(range(1, 11))
    _register_prompt(scheduler, r3_tokens)
    scheduler.submit_requests([_spec("r3", r3_tokens)])
    plan = scheduler.next_execution_plan()
    prefetch = _find_op(plan, ts.Cache.PrefetchOp)
    assert (
        prefetch is not None
    ), "a Host-evicted L3 prefix must be fetched again before admission"
    assert list(prefetch.num_pages) == [4]
    assert len(prefetch.host_pages[0]) == 6, "4 full + 2 swa rows"
    _ack_prefetch(scheduler, prefetch.op_ids[0], 4)
    admit = scheduler.next_execution_plan()
    load = _find_op(admit, ts.Cache.LoadBackOp)
    assert load is not None
    assert list(admit.forward[0].extend_prefix_lens) == [8]
    _ack_load_back(scheduler, load.op_ids[0])


def test_l3_storage_prefix_hash_and_register_bindings() -> None:
    cfg = _l3_config(
        num_device_pages=32, num_host_pages=32, with_swa=True, min_prefetch_pages=1
    )
    scheduler = ts.Scheduler(cfg)
    assert hasattr(scheduler, "prefix_hashes_for_tokens")
    assert not hasattr(
        scheduler, "waiting_prefix_hashes"
    ), "the prefetch op is the probe now"
    hashes = scheduler.prefix_hashes_for_tokens([1, 2, 3, 4, 5])
    assert hashes
    group_ids, expanded, offsets = scheduler.expand_prefix_keys(hashes)
    assert group_ids
    assert len(group_ids) == len(expanded) == len(offsets)
    scheduler.register_storage_keys(group_ids, expanded, offsets)
