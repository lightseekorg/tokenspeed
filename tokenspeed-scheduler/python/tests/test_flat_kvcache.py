"""Binding-smoke tests for the flat KV-cache path, driven through the
nanobind binding (no GPU).

The scheduler scenarios themselves are covered by the C++ suites
(``tests/cpp/test_flat_kvcache_lifecycle.cpp`` and
``test_flat_kvcache_scenarios.cpp``); this module keeps the marshalling
surface honest: per-group flat block tables (incl. the sliding-window null
hole), the four-group Kimi-K3 namespace with finish/abort page restoration,
atomic OOM deferral, and the readmit op's ``prefill_lengths`` /
``extend_prefix_lens`` fields.

The nanobind extension is build-identical between the radix and flat builds --
``FlatForwardOp.flat_block_tables`` is always exposed -- but it is only
*populated* when the extension is built with ``TOKENSPEED_FLAT_KVCACHE=ON``.
On the default (radix) build it stays empty, so the whole module is
``skipif``-guarded on a behavioral probe and is a no-op there.
"""

from __future__ import annotations

import pytest
from conftest import (
    K3_GROUP_IDS,
)
from conftest import _advance as _advance_tokens
from conftest import (
    _find_flat_op,
    _finish,
    _make_k3_config,
)
from conftest import _positive as _positive_pages
from conftest import (
    _spec,
    requires_flat_build,
)

# conftest guards the ext import, so skip resolution can happen here.
ts = pytest.importorskip("tokenspeed_scheduler")

pytestmark = requires_flat_build


def _make_config() -> ts.SchedulerConfig:
    cfg = ts.SchedulerConfig()
    cfg.block_size = 2
    cfg.num_device_pages = 32
    cfg.num_host_pages = 32
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.enable_l3_storage = False
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = True

    full = ts.PagedCacheGroupConfig(
        group_id="full",
        rows_per_page=cfg.block_size,
        entry_stride_tokens=1,
        total_pages=cfg.num_device_pages,
        retention=ts.PagedCacheRetention.FullHistory,
        family=ts.PagedCacheGroupFamily.History,
    )
    swa = ts.PagedCacheGroupConfig(
        group_id="swa",
        rows_per_page=cfg.block_size,
        entry_stride_tokens=1,
        total_pages=cfg.num_device_pages,
        retention=ts.PagedCacheRetention.SlidingWindow,
        sliding_window_tokens=4,
        family=ts.PagedCacheGroupFamily.State,
    )
    cfg.paged_cache_groups = [full, swa]
    return cfg


def _make_spec(
    request_id: str, num_pages: int, page_size: int = 2, start: int = 1
) -> ts.RequestSpec:
    # page_size must stay in sync with cfg.block_size: the token count below
    # (num_pages * page_size) is what determines how many pages get allocated.
    return _spec(request_id, list(range(start, start + num_pages * page_size)))


def _abort(scheduler, request_id: str) -> None:
    event = ts.ForwardEvent.Abort()
    event.request_id = request_id
    execution_event = ts.ExecutionEvent()
    execution_event.add_event(event)
    scheduler.advance(execution_event)


def test_decode_slides_swa_window_to_null_hole():
    scheduler = ts.Scheduler(_make_config())
    scheduler.submit_requests([_make_spec("r1", num_pages=2)])

    scheduler.next_execution_plan()  # prefill
    _advance_tokens(scheduler, "r1", [42])

    last_plan = None
    token = 43
    # sliding_window_tokens=4, page_size=2 => window spans 2 pages; ~4 decode
    # steps push total pages past 2, so the oldest page slides out and leaves a
    # null hole in the swa block table.
    for _ in range(4):
        last_plan = scheduler.next_execution_plan()
        assert _find_flat_op(last_plan) is not None
        _advance_tokens(scheduler, "r1", [token])
        token += 1

    op = _find_flat_op(last_plan)
    assert op is not None
    tables = dict(op.flat_block_tables)

    full_row = list(tables["full"][0])
    # page id 0 is the reserved null-block sentinel: >0 means a real page, 0
    # means a hole. The full-history group should never develop a hole.
    assert all(
        page_id > 0 for page_id in full_row
    ), "full row should keep history with no null/padding hole"

    swa_row = list(tables["swa"][0])
    assert (
        0 in swa_row
    ), "swa row should contain a null hole after the sliding window slides"


def test_k3_four_groups_share_one_global_id_namespace() -> None:
    scheduler = ts.Scheduler(_make_k3_config())
    before = scheduler.available_kv_pages()
    assert before == 32
    scheduler.submit_requests([_make_spec("r1", num_pages=2)])
    op = _find_flat_op(scheduler.next_execution_plan())
    assert op is not None
    tables = dict(op.flat_block_tables)
    assert tuple(tables) == K3_GROUP_IDS
    positive_by_group = {
        group_id: _positive_pages(tables[group_id][0]) for group_id in K3_GROUP_IDS
    }
    real_by_group = {
        group_id: set(pages) for group_id, pages in positive_by_group.items()
    }
    fresh_count = sum(len(pages) for pages in positive_by_group.values())
    all_real = set().union(*real_by_group.values())
    assert len(all_real) == fresh_count
    for index, left in enumerate(real_by_group):
        for right in tuple(real_by_group)[index + 1 :]:
            assert real_by_group[left].isdisjoint(real_by_group[right])

    _abort(scheduler, "r1")
    scheduler.next_execution_plan()
    assert scheduler.available_kv_pages() == before


def test_k3_finish_restores_all_usable_pages() -> None:
    scheduler = ts.Scheduler(_make_k3_config())
    before = scheduler.available_kv_pages()
    scheduler.submit_requests([_make_spec("r1", num_pages=2)])
    assert _find_flat_op(scheduler.next_execution_plan()) is not None
    _advance_tokens(scheduler, "r1", [42])
    _finish(scheduler, "r1")
    scheduler.next_execution_plan()
    assert scheduler.available_kv_pages() == before


def test_k3_oom_is_atomic_across_all_four_tables() -> None:
    scheduler = ts.Scheduler(_make_k3_config())
    before = scheduler.available_kv_pages()
    scheduler.submit_requests([_make_spec("oom", num_pages=8)])

    for _ in range(3):
        deferred = scheduler.next_execution_plan()
        assert _find_flat_op(deferred) is None
        assert tuple(deferred.flat_oom_request_ids) == ()
        assert all(not dict(op.flat_block_tables) for op in deferred.forward)
        assert scheduler.available_kv_pages() == before


def _make_k3_128k_config(num_device_pages: int) -> ts.SchedulerConfig:
    cfg = _make_k3_config()
    cfg.block_size = 1_536
    cfg.num_device_pages = num_device_pages
    cfg.max_scheduled_tokens = 8_192
    cfg.max_batch_size = 1
    for group in cfg.paged_cache_groups:
        group.rows_per_page = cfg.block_size
        group.total_pages = num_device_pages
    return cfg


def test_k3_128k_requires_group_aware_shared_pool_geometry() -> None:
    prompt = _spec("128k", list(range(131_072)))

    undersized = ts.Scheduler(_make_k3_128k_config(86))
    undersized.submit_requests([prompt])
    completed = 0
    for _ in range(16):
        if _find_flat_op(undersized.next_execution_plan()) is None:
            break
        completed += 1
    assert completed < 16

    corrected = ts.Scheduler(_make_k3_128k_config(108))
    before = corrected.available_kv_pages()
    assert before == 107
    corrected.submit_requests([prompt])
    for chunk in range(16):
        assert _find_flat_op(corrected.next_execution_plan()) is not None, chunk
    _advance_tokens(corrected, "128k", [131_072])
    assert _find_flat_op(corrected.next_execution_plan()) is not None
    _finish(corrected, "128k")
    corrected.next_execution_plan()
    assert corrected.available_kv_pages() == before


def _drive_k3_to_retract(scheduler) -> dict[str, dict[int, int]]:
    request_ids = ("a", "b", "c", "d")
    scheduler.submit_requests(
        [
            _make_spec(request_id, num_pages=1, start=1 + index * 100)
            for index, request_id in enumerate(request_ids)
        ]
    )
    prefill = _find_flat_op(scheduler.next_execution_plan())
    assert prefill is not None
    assert tuple(prefill.request_ids) == request_ids
    pre_retract_pages = {group_id: {} for group_id in K3_GROUP_IDS}

    def record_positive_slots(op, row_index: int) -> None:
        tables = dict(op.flat_block_tables)
        for group_id in K3_GROUP_IDS:
            for logical_slot, page in enumerate(tables[group_id][row_index]):
                if page <= 0:
                    continue
                previous = pre_retract_pages[group_id].setdefault(logical_slot, page)
                assert previous == page

    record_positive_slots(prefill, 0)
    for index, request_id in enumerate(request_ids):
        _advance_tokens(scheduler, request_id, [1000 + index])

    consecutive_empty_rounds = 0
    next_token = 2000
    for _ in range(32):
        op = _find_flat_op(scheduler.next_execution_plan())
        scheduled = () if op is None else tuple(op.request_ids)
        if not scheduled:
            consecutive_empty_rounds += 1
            if consecutive_empty_rounds == 2:
                break
            continue
        consecutive_empty_rounds = 0
        if "a" in scheduled:
            a_row = scheduled.index("a")
            record_positive_slots(op, a_row)
        for request_id in scheduled:
            _advance_tokens(scheduler, request_id, [next_token])
            next_token += 1

    assert consecutive_empty_rounds == 2
    assert scheduler.available_kv_pages() == 11
    assert scheduler.waiting_size() == 1
    assert scheduler.decoding_size() == 3
    assert scheduler.get_request_token_size("a") == 11
    return pre_retract_pages


def test_k3_readmit_rebuilds_all_four_tables_and_restores_pages() -> None:
    """Binding-marshalling smoke for readmit: the only python test that reads
    ``op.prefill_lengths`` through the real nanobind property (the C++ suite
    covers the retract/readmit scheduler scenarios themselves)."""
    scheduler = ts.Scheduler(_make_k3_config())
    before = scheduler.available_kv_pages()
    pre_retract_pages = _drive_k3_to_retract(scheduler)

    for request_id in ("b", "c", "d"):
        _finish(scheduler, request_id)

    readmit = _find_flat_op(scheduler.next_execution_plan())
    assert readmit is not None
    assert tuple(readmit.request_ids) == ("a",)
    assert tuple(readmit.prefill_lengths) == (11,)
    assert readmit.extend_prefix_lens[0] + readmit.input_lengths[0] == 11
    tables = dict(readmit.flat_block_tables)
    assert tuple(tables) == K3_GROUP_IDS
    page_size = _make_k3_config().block_size
    assert readmit.extend_prefix_lens[0] % page_size == 0
    prefix_slots = readmit.extend_prefix_lens[0] // page_size
    assert prefix_slots == 4
    expected_slots = (readmit.prefill_lengths[0] + page_size - 1) // page_size
    assert expected_slots == 6

    all_positive_entries = []
    restored_pages = set()
    fresh_tail_entries = []
    for group_id in K3_GROUP_IDS:
        row = tuple(tables[group_id][0])
        assert len(row) == expected_slots
        group_positive = _positive_pages(row)
        assert group_positive
        all_positive_entries.extend(group_positive)

        restored_in_group = []
        for index, page in enumerate(row[:prefix_slots]):
            if page > 0:
                assert page == pre_retract_pages[group_id].get(index)
                restored_in_group.append(page)
        assert restored_in_group
        restored_pages.update(restored_in_group)

        tail = row[prefix_slots:]
        assert len(tail) == 2
        assert all(page > 0 for page in tail)
        group_tail = _positive_pages(tail)
        assert len(group_tail) == 2
        fresh_tail_entries.extend(group_tail)

    assert len(set(all_positive_entries)) == len(all_positive_entries)
    assert len(set(fresh_tail_entries)) == len(fresh_tail_entries)
    assert set(fresh_tail_entries).isdisjoint(restored_pages)

    _advance_tokens(scheduler, "a", [3000])
    scheduler.next_execution_plan()
    _advance_tokens(scheduler, "a", [3001])
    _finish(scheduler, "a")
    scheduler.next_execution_plan()
    assert scheduler.available_kv_pages() == before
