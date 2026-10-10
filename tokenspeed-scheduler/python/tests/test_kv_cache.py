"""Binding-smoke tests for the cache-group scheduler (no GPU).

The scheduler scenarios themselves are covered by the C++ suites
(``tests/cpp/test_kvcache_lifecycle.cpp`` and
``test_kvcache_scenarios.cpp``); this module keeps the marshalling
surface honest: per-group block tables (including the sliding-window null
hole), the four-group Kimi-K3 namespace with finish/abort page restoration,
atomic OOM deferral, and the readmit op's ``prefill_lengths`` /
``extend_prefix_lens`` fields.

"""

from __future__ import annotations

import pytest
from conftest import (
    K3_GROUP_IDS,
)
from conftest import _advance as _advance_tokens
from conftest import (
    _find_forward_op,
    _finish,
    _make_k3_config,
)
from conftest import _positive as _positive_pages
from conftest import (
    _spec,
)

# conftest guards the ext import, so skip resolution can happen here.
ts = pytest.importorskip("tokenspeed_scheduler")


def _make_config() -> ts.SchedulerConfig:
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 2
    cfg.num_device_pages = 32
    cfg.num_host_pages = 32
    cfg.num_snapshot_pages = 1  # never retracts
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.enable_l3_storage = False
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = True

    full = ts.CacheGroupConfig(
        group_id="full",
        block_granularity=cfg.prefix_granularity,
        total_pages=cfg.num_device_pages,
        retention=ts.CacheRetention.FullHistory,
        family=ts.CacheGroupFamily.History,
    )
    swa = ts.CacheGroupConfig(
        group_id="swa",
        block_granularity=cfg.prefix_granularity,
        total_pages=cfg.num_device_pages,
        retention=ts.CacheRetention.SlidingWindow,
        sliding_window_tokens=4,
        family=ts.CacheGroupFamily.History,
    )
    cfg.cache_groups = [full, swa]
    return cfg


def test_prefix_replay_tokens_binding_defaults_to_zero_and_round_trips() -> None:
    cfg = _make_config()
    assert cfg.prefix_replay_tokens == 0
    cfg.prefix_replay_tokens = 4
    assert cfg.prefix_replay_tokens == 4


def _make_spec(
    request_id: str, num_pages: int, prefix_granularity: int = 2, start: int = 1
) -> ts.RequestSpec:
    # prefix_granularity must stay in sync with cfg.prefix_granularity: the token count below
    # (num_pages * prefix_granularity) is what determines how many pages get allocated.
    return _spec(request_id, list(range(start, start + num_pages * prefix_granularity)))


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
    # sliding_window_tokens=4, prefix_granularity=2 => window spans 2 pages; ~4 decode
    # steps push total pages past 2, so the oldest page slides out and leaves a
    # null hole in the swa block table.
    for _ in range(4):
        last_plan = scheduler.next_execution_plan()
        assert _find_forward_op(last_plan) is not None
        _advance_tokens(scheduler, "r1", [token])
        token += 1

    op = _find_forward_op(last_plan)
    assert op is not None
    tables = dict(op.block_tables)

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


def test_forward_batch_uses_per_group_block_tables_as_the_only_page_table():
    scheduler = ts.Scheduler(_make_config())
    scheduler.submit_requests([_make_spec("r1", num_pages=2)])

    op = _find_forward_op(scheduler.next_execution_plan())
    assert op is not None
    arrays = op.block_tables_arrays()
    assert set(arrays) == {"full", "swa"}
    assert all(array.shape[0] == 1 for array in arrays.values())
    assert not hasattr(op, "occupied_pages")
    assert not hasattr(op, "begins")
    assert not hasattr(op, "sizes")


def test_pages_to_zero_arrays_view_the_plan_without_python_ints():
    import numpy as np

    scheduler = ts.Scheduler(_make_config())
    scheduler.submit_requests([_make_spec("r1", num_pages=2)])

    plan = scheduler.next_execution_plan()
    arrays = plan.pages_to_zero_arrays()
    assert set(arrays) == set(plan.pages_to_zero) == {"full", "swa"}
    for group_id, pages in plan.pages_to_zero.items():
        array = arrays[group_id]
        assert array.dtype == np.int32 and array.ndim == 1
        assert array.tolist() == list(pages)
    assert any(array.size for array in arrays.values())

    # The views borrow the plan's storage; the plan must outlive them even
    # when the caller drops its own reference.
    full = arrays["full"]
    expected = full.tolist()
    del plan, arrays
    assert full.tolist() == expected


def test_k3_four_groups_share_one_global_id_namespace() -> None:
    scheduler = ts.Scheduler(_make_k3_config())
    before = scheduler.available_lcm_blocks()
    assert before == 32
    scheduler.submit_requests([_make_spec("r1", num_pages=2)])
    plan = scheduler.next_execution_plan()
    op = _find_forward_op(plan)
    assert op is not None
    tables = dict(op.block_tables)
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
    pages_to_zero = {
        group_id: list(page_ids)
        for group_id, page_ids in dict(plan.pages_to_zero).items()
    }
    assert set(pages_to_zero) == set(K3_GROUP_IDS)
    for group_id in K3_GROUP_IDS:
        assert set(pages_to_zero[group_id]) == real_by_group[group_id]

    _abort(scheduler, "r1")
    scheduler.next_execution_plan()
    assert scheduler.available_lcm_blocks() == before


def test_k3_finish_restores_all_usable_pages() -> None:
    scheduler = ts.Scheduler(_make_k3_config())
    before = scheduler.available_lcm_blocks()
    assert scheduler.empty_lcm_blocks() == before
    assert scheduler.active_lcm_blocks() == 0
    scheduler.submit_requests([_make_spec("r1", num_pages=2)])
    assert _find_forward_op(scheduler.next_execution_plan()) is not None
    assert scheduler.active_lcm_blocks() > 0
    assert scheduler.empty_lcm_blocks() + scheduler.active_lcm_blocks() == before
    _advance_tokens(scheduler, "r1", [42])
    _finish(scheduler, "r1")
    scheduler.next_execution_plan()
    assert scheduler.available_lcm_blocks() == before
    # The finished request's pages stay resident as cache-only parents: they
    # are evictable (available) but neither empty nor active.
    assert scheduler.active_lcm_blocks() == 0
    assert scheduler.empty_lcm_blocks() < before


def _make_k3_128k_config(num_device_pages: int) -> ts.SchedulerConfig:
    cfg = _make_k3_config()
    cfg.prefix_granularity = 128
    cfg.num_device_pages = num_device_pages
    cfg.max_scheduled_tokens = 8_192
    cfg.max_batch_size = 1
    for group in cfg.cache_groups:
        group.block_granularity = cfg.prefix_granularity
        group.cache_blocks_per_lcm_block = (
            12 if group.group_id == K3_GROUP_IDS[0] else 1
        )
        group.total_pages = (
            1 + (num_device_pages - 1) * group.cache_blocks_per_lcm_block
        )
    return cfg


def test_k3_reports_group_aware_single_request_capacity() -> None:
    # Each sparse State group needs input, aligned checkpoint, final state,
    # and banked growth. The three groups therefore leave 272
    # of the 284 usable parents for Full KV. K_full=12 and P=128 expose
    # 272 * 12 * 128 tokens.
    scheduler = ts.Scheduler(_make_k3_128k_config(285))
    assert scheduler.max_single_request_tokens() == 417_792


def test_k3_128k_requires_group_aware_shared_pool_geometry() -> None:
    prompt = _spec("128k", list(range(131_072)))

    # Twelve State parents plus 86 Full parents admit 128K; one fewer Full parent
    # is 512 tokens short because each Full parent carries 12 * 128 tokens.
    undersized = ts.Scheduler(_make_k3_128k_config(98))
    assert undersized.max_single_request_tokens() < 131_072

    corrected = ts.Scheduler(_make_k3_128k_config(99))
    before = corrected.available_lcm_blocks()
    assert before == 98
    corrected.submit_requests([prompt])
    completed_tokens = 0
    for chunk in range(32):
        op = _find_forward_op(corrected.next_execution_plan())
        assert op is not None, chunk
        completed_tokens += op.input_lengths[0]
        if completed_tokens == 131_072:
            break
    assert completed_tokens == 131_072
    _advance_tokens(corrected, "128k", [131_072])
    assert _find_forward_op(corrected.next_execution_plan()) is not None
    _finish(corrected, "128k")
    corrected.next_execution_plan()
    assert corrected.available_lcm_blocks() == before


@pytest.mark.parametrize("block_granularity", [1, 2, 4])
@pytest.mark.parametrize("chunk_tokens", [4, 8, 9])
@pytest.mark.parametrize("decode_width", [1, 3])
@pytest.mark.parametrize("overlap_depth", [0, 1])
@pytest.mark.parametrize("prefix_cache_enabled", [False, True])
def test_accepted_state_prompts_can_prefill_and_start_decode(
    block_granularity: int,
    chunk_tokens: int,
    decode_width: int,
    overlap_depth: int,
    prefix_cache_enabled: bool,
) -> None:
    """An empty pool must serve every prompt below its advertised startup bound."""
    for usable_blocks in range(2, 9):
        cfg = ts.SchedulerConfig()
        cfg.prefix_granularity = 4
        cfg.num_device_pages = usable_blocks + 1
        cfg.num_snapshot_pages = 1  # never retracts
        cfg.max_scheduled_tokens = chunk_tokens
        cfg.max_batch_size = 1
        cfg.disable_l2_cache = True
        cfg.disable_prefix_cache = not prefix_cache_enabled
        cfg.decode_input_tokens = decode_width
        cfg.overlap_schedule_depth = overlap_depth
        cfg.cache_groups = [
            ts.CacheGroupConfig(
                group_id="state",
                block_granularity=block_granularity,
                total_pages=usable_blocks + 1,
                retention=ts.CacheRetention.FullHistory,
                family=ts.CacheGroupFamily.State,
            )
        ]
        capacity = ts.Scheduler(cfg).max_single_request_tokens()
        for prompt_tokens in range(1, min(16, capacity - decode_width) + 1):
            scheduler = ts.Scheduler(cfg)
            spec = _spec("r", list(range(prompt_tokens)))
            spec.max_new_tokens = min(decode_width + 1, capacity - prompt_tokens)
            scheduler.submit_requests([spec])
            computed = 0
            while computed < prompt_tokens:
                batch = _find_forward_op(scheduler.next_execution_plan())
                assert batch is not None, (
                    usable_blocks,
                    prompt_tokens,
                    computed,
                    capacity,
                )
                assert list(batch.request_ids) == ["r"]
                assert batch.input_lengths[0] > 0
                computed += batch.input_lengths[0]
                _advance_tokens(
                    scheduler, "r", [101] if computed == prompt_tokens else []
                )
            # The completing admission must also secure the first decode step.
            if spec.max_new_tokens > 1:
                assert _find_forward_op(scheduler.next_execution_plan()) is not None


@pytest.mark.parametrize("finish_after_first_decode", [False, True])
@pytest.mark.parametrize("decode_width", [1, 3])
@pytest.mark.parametrize("state_granularity", [1, 2, 4])
@pytest.mark.parametrize("prompt_tokens", [3, 7, 8])
@pytest.mark.parametrize("truncate_output", [False, True])
def test_decode_reuses_only_prefill_state_boundary(
    finish_after_first_decode: bool,
    decode_width: int,
    state_granularity: int,
    prompt_tokens: int,
    truncate_output: bool,
) -> None:
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 4
    cfg.num_device_pages = 33
    cfg.num_host_pages = 0
    cfg.num_snapshot_pages = 1  # never retracts
    cfg.max_scheduled_tokens = 32
    cfg.max_batch_size = 2
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = False
    cfg.decode_input_tokens = decode_width
    cfg.overlap_schedule_depth = 0
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="state",
            block_granularity=state_granularity,
            total_pages=33,
            retention=ts.CacheRetention.FullHistory,
            family=ts.CacheGroupFamily.State,
        )
    ]
    scheduler = ts.Scheduler(cfg)
    request = _spec("r", list(range(1, prompt_tokens + 1)))
    request.max_new_tokens = 30
    scheduler.submit_requests([request])
    assert _find_forward_op(scheduler.next_execution_plan()) is not None
    _advance_tokens(scheduler, "r", [prompt_tokens + 1])
    assert _find_forward_op(scheduler.next_execution_plan()) is not None
    next_token = prompt_tokens + 2
    visible_tokens = 1 if truncate_output else decode_width
    _advance_tokens(
        scheduler, "r", list(range(next_token, next_token + visible_tokens))
    )
    if not finish_after_first_decode:
        assert _find_forward_op(scheduler.next_execution_plan()) is not None
        _advance_tokens(
            scheduler,
            "r",
            list(range(next_token + visible_tokens, next_token + 2 * visible_tokens)),
        )
    _finish(scheduler, "r")
    scheduler.next_execution_plan()

    # Decode cannot create a reusable checkpoint, even at an aligned accepted
    # endpoint or when host-side stopping truncates the accepted output.
    reuse = _spec("reuse", list(range(1, next_token + visible_tokens)) + [90, 91])
    reuse.max_new_tokens = 4
    scheduler.submit_requests([reuse])
    batch = _find_forward_op(scheduler.next_execution_plan())
    assert batch is not None
    assert list(batch.extend_prefix_lens) == [prompt_tokens // 4 * 4]


def _ack_cache_event(scheduler, event) -> None:
    execution_event = ts.ExecutionEvent()
    execution_event.add_event(event)
    scheduler.advance(execution_event)


def _find_cache_op(plan, kind):
    found = [op for op in plan.cache if isinstance(op, kind)]
    assert len(found) <= 1
    return found[0] if found else None


def _rows(op) -> dict[str, list[int]]:
    return {group_id: list(rows[0]) for group_id, rows in dict(op.block_tables).items()}


def _drive_k3_to_retract(scheduler) -> tuple[object, dict[str, list[int]]]:
    """Decode a/b/c/d until growth evicts the smallest grower, a.

    Returns the SnapshotOp that carries a's image and a's last block tables."""
    request_ids = ("a", "b", "c", "d")
    scheduler.submit_requests(
        [
            _make_spec(request_id, num_pages=1, start=1 + index * 100)
            for index, request_id in enumerate(request_ids)
        ]
    )
    prefill = _find_forward_op(scheduler.next_execution_plan())
    assert prefill is not None
    assert tuple(prefill.request_ids) == request_ids
    for index, request_id in enumerate(request_ids):
        _advance_tokens(scheduler, request_id, [1000 + index])

    a_rows = _rows(prefill)
    next_token = 2000
    for _ in range(32):
        before_plan = (
            scheduler.active_lcm_blocks(),
            scheduler.empty_lcm_blocks(),
            scheduler.available_lcm_blocks(),
        )
        plan = scheduler.next_execution_plan()
        op = _find_forward_op(plan)
        scheduled = () if op is None else tuple(op.request_ids)
        snapshot = _find_cache_op(plan, ts.Cache.SnapshotOp)
        if snapshot is not None:
            # Retract-and-grant happen in one round: the blocked grower runs
            # on the pages a just released.
            assert before_plan == (29, 3, 3)
            assert scheduled == ("b",)
            break
        if "a" in scheduled:
            a_rows = {
                group_id: list(rows[scheduled.index("a")])
                for group_id, rows in dict(op.block_tables).items()
            }
        for request_id in scheduled:
            _advance_tokens(scheduler, request_id, [next_token])
            next_token += 1
    else:
        pytest.fail("decode growth never retracted a request")

    assert tuple(scheduler.request_token_size(r) for r in request_ids) == (
        11,
        9,
        9,
        9,
    )
    assert scheduler.retracted_size() == 1
    assert scheduler.waiting_size() == 1
    assert scheduler.decoding_size() == 3
    _advance_tokens(scheduler, "b", [next_token])
    return snapshot, a_rows


def test_k3_retraction_images_every_group_and_restores_it_in_place() -> None:
    """Without L2 the whole image lives in the snapshot pool; the restore
    copies it back into fresh pages and decode resumes where it stopped."""
    cfg = _make_k3_config()
    assert cfg.num_host_pages == 0
    assert cfg.disable_l2_cache
    cfg.num_snapshot_pages = 33
    cfg.max_retracted_requests = 4
    scheduler = ts.Scheduler(cfg)
    before = scheduler.available_lcm_blocks()
    assert scheduler.snapshot_pool_free_blocks() == 32
    snapshot, a_rows = _drive_k3_to_retract(scheduler)

    # The image: every positive slot of every group, 11 computed tokens = five
    # History blocks plus one working block per State group, each copied to
    # its own snapshot-pool page along with the slot-state blob of a's row.
    assert list(snapshot.request_ids) == ["a"]
    assert len(snapshot.op_ids) == 1
    [snapshot_slot] = snapshot.snapshot_slots
    assert snapshot_slot >= 1
    [request_pool_index] = snapshot.request_pool_indices
    assert request_pool_index >= 1
    [group_ids] = snapshot.group_ids
    [src_pages] = snapshot.src_pages
    [dst_pages] = snapshot.dst_pages
    expected_sources = [
        (group_index, page)
        for group_index, group_id in enumerate(K3_GROUP_IDS)
        for page in _positive_pages(a_rows[group_id])
    ]
    assert list(zip(group_ids, src_pages)) == expected_sources
    assert len(expected_sources) == 5 + 3
    assert len(set(dst_pages)) == len(dst_pages) == 8
    assert scheduler.snapshot_pool_free_blocks() == 32 - 8
    assert scheduler.host_pool_pinned_blocks() == 0

    # Finishing the others frees the Device, but a cannot be restored before
    # its image has landed.
    for request_id in ("b", "c", "d"):
        _finish(scheduler, request_id)
    idle = scheduler.next_execution_plan()
    assert _find_forward_op(idle) is None
    assert _find_cache_op(idle, ts.Cache.RestoreOp) is None
    assert scheduler.retracted_size() == 1
    assert scheduler.active_lcm_blocks() == 0

    done = ts.Cache.SnapshotDoneEvent()
    done.op_id = int(snapshot.op_ids[0])
    _ack_cache_event(scheduler, done)
    restore_plan = scheduler.next_execution_plan()
    assert _find_forward_op(restore_plan) is None
    restore = _find_cache_op(restore_plan, ts.Cache.RestoreOp)
    assert restore is not None
    assert list(restore.request_ids) == ["a"]
    assert list(restore.snapshot_slots) == [snapshot_slot]
    [restored_pool_index] = restore.request_pool_indices
    assert restored_pool_index >= 1
    # Every row comes back from the snapshot pool (no L2 leg to claim from),
    # in image order, into pages the same plan zeroes first.
    [restore_groups] = restore.group_ids
    [restore_sources] = restore.src_pages
    [restore_destinations] = restore.dst_pages
    [source_tiers] = restore.source_tiers
    assert list(restore_groups) == list(group_ids)
    assert list(restore_sources) == list(dst_pages)
    assert set(source_tiers) == {int(ts.Cache.HostTier.SnapshotPool)}
    assert all(not key for key in restore.content_hashes[0])
    zeroed = dict(restore_plan.pages_to_zero)
    for group_index, page in zip(restore_groups, restore_destinations):
        assert page in zeroed[K3_GROUP_IDS[group_index]]
    # The restore holds the rebuilt tables plus decode growth, yet nothing is
    # schedulable until the copy lands and the pool keeps a's blocks.
    assert scheduler.retracted_size() == 1
    assert scheduler.decoding_size() == 0
    assert scheduler.active_lcm_blocks() == 8 + 4
    assert scheduler.snapshot_pool_free_blocks() == 32 - 8
    while_restoring = scheduler.next_execution_plan()
    assert _find_forward_op(while_restoring) is None
    assert not list(while_restoring.cache)

    restored = ts.Cache.RestoreDoneEvent()
    restored.op_id = int(restore.op_ids[0])
    _ack_cache_event(scheduler, restored)
    assert scheduler.retracted_size() == 0
    assert scheduler.waiting_size() == 0
    assert scheduler.decoding_size() == 1
    assert scheduler.snapshot_pool_free_blocks() == 32
    assert scheduler.request_token_size("a") == 11

    # Decode continues from token 11 with no prefill: the block tables are
    # the restore destinations in their image slots plus one growth block.
    resume = _find_forward_op(scheduler.next_execution_plan())
    assert resume is not None
    assert tuple(resume.request_ids) == ("a",)
    assert list(resume.input_lengths) == [1]
    assert list(resume.extend_prefix_lens) == []
    rows = _rows(resume)
    restored_rows = {group_id: [] for group_id in K3_GROUP_IDS}
    for group_index, page in zip(restore_groups, restore_destinations):
        restored_rows[K3_GROUP_IDS[group_index]].append(page)
    for group_id in K3_GROUP_IDS:
        row = rows[group_id]
        assert len(row) == len(a_rows[group_id]) + 1
        assert [slot for slot, page in enumerate(row) if page == 0] == [
            slot for slot, page in enumerate(a_rows[group_id]) if page == 0
        ]
        assert _positive_pages(row)[:-1] == restored_rows[group_id]

    _advance_tokens(scheduler, "a", [3000])
    assert _find_forward_op(scheduler.next_execution_plan()) is not None
    _advance_tokens(scheduler, "a", [3001])
    _finish(scheduler, "a")
    scheduler.next_execution_plan()
    assert scheduler.available_lcm_blocks() == before
    assert scheduler.active_lcm_blocks() == 0
    assert scheduler.snapshot_pool_free_blocks() == 32


def test_k3_finish_while_retracted_returns_the_image() -> None:
    cfg = _make_k3_config()
    cfg.num_snapshot_pages = 33
    cfg.max_retracted_requests = 4
    scheduler = ts.Scheduler(cfg)
    snapshot, _ = _drive_k3_to_retract(scheduler)
    assert scheduler.snapshot_pool_free_blocks() == 32 - 8
    _finish(scheduler, "a")
    assert scheduler.retracted_size() == 0
    assert scheduler.waiting_size() == 0
    # The copy into the pool pages is still in flight: they return at its ACK.
    assert scheduler.snapshot_pool_free_blocks() == 32 - 8
    done = ts.Cache.SnapshotDoneEvent()
    done.op_id = int(snapshot.op_ids[0])
    _ack_cache_event(scheduler, done)
    assert scheduler.snapshot_pool_free_blocks() == 32
    assert scheduler.decoding_size() == 3


def _make_replay_config() -> ts.SchedulerConfig:
    """full (closed) + swa regenerated by bounded replay; P=8, budget 64."""
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 8
    cfg.num_device_pages = 256
    cfg.num_host_pages = 256
    cfg.num_snapshot_pages = 1  # never retracts
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.enable_l3_storage = False
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = False
    full = ts.CacheGroupConfig(
        group_id="full",
        block_granularity=cfg.prefix_granularity,
        total_pages=cfg.num_device_pages,
        retention=ts.CacheRetention.FullHistory,
        family=ts.CacheGroupFamily.History,
    )
    swa = ts.CacheGroupConfig(
        group_id="swa",
        block_granularity=4,
        total_pages=cfg.num_device_pages,
        retention=ts.CacheRetention.SlidingWindow,
        sliding_window_tokens=16,
        family=ts.CacheGroupFamily.History,
        replayable=True,
    )
    cfg.cache_groups = [full, swa]
    return cfg


def test_cache_group_config_replayable_round_trips() -> None:
    swa = _make_replay_config().cache_groups[1]
    assert swa.replayable is True
    swa.replayable = False
    assert swa.replayable is False
    swa.validate()
    swa.replayable = True
    swa.validate()
    full = _make_replay_config().cache_groups[0]
    assert full.replayable is False
    full.replayable = True
    with pytest.raises(ValueError, match="sliding History group"):
        full.validate()


def test_prefix_hit_replays_window_and_exposes_extend_replay_lens() -> None:
    scheduler = ts.Scheduler(_make_replay_config())
    tokens = list(range(1, 33))
    scheduler.submit_requests([_spec("r1", tokens)])
    op = _find_forward_op(scheduler.next_execution_plan())
    assert op is not None
    assert list(op.extend_prefix_lens) == [0]
    assert list(op.extend_replay_lens) == [0]
    _advance_tokens(scheduler, "r1", [9001])
    scheduler.next_execution_plan()  # first decode publishes the prefix pages
    _advance_tokens(scheduler, "r1", [9002])
    _finish(scheduler, "r1")
    scheduler.next_execution_plan()

    # r2 shares r1's 32 tokens and adds 8: the closed group hits P=32 alone
    # and the swa window [16, 32) is re-fed ahead of the new tokens.
    scheduler.submit_requests([_spec("r2", tokens + list(range(901, 909)))])
    op = _find_forward_op(scheduler.next_execution_plan())
    assert op is not None
    assert list(op.extend_prefix_lens) == [16]
    assert list(op.extend_replay_lens) == [16]
    assert list(op.input_lengths) == [24]
    assert list(op.input_ids) == tokens[16:] + list(range(901, 909))
    swa_row = list(dict(op.block_tables)["swa"][0])
    assert swa_row[:4] == [0, 0, 0, 0], "no swa page below the replay window"
    assert all(page > 0 for page in swa_row[4:11]), "private swa pages from s=16"
    assert "extend_replay_lens=[16]" in repr(op)


def test_final_chunk_is_never_shorter_than_the_replay_window() -> None:
    scheduler = ts.Scheduler(_make_replay_config())
    tokens = list(range(1, 71))
    scheduler.submit_requests([_spec("r1", tokens)])
    # 70 tokens under a 64-token budget would leave a 6-token final chunk; the
    # first chunk is shortened so the final chunk holds the 16-token window.
    chunk1 = _find_forward_op(scheduler.next_execution_plan())
    assert chunk1 is not None
    assert list(chunk1.input_lengths) == [54]
    assert list(chunk1.extend_replay_lens) == [0]
    chunk2 = _find_forward_op(scheduler.next_execution_plan())
    assert chunk2 is not None
    assert list(chunk2.extend_prefix_lens) == [54]
    assert list(chunk2.extend_replay_lens) == [0]
    assert list(chunk2.input_lengths) == [16]
    assert list(chunk2.input_ids) == tokens[54:70]
