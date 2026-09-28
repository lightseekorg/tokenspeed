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

"""CPU Plan A recipe -> real C++ scheduler/LCM lifecycle tests.

Only model/serving configuration and forward completion tokens are supplied by
Python. All admission, tables, prefix publication, reclaim and release are real;
no arena payloads, attention kernels or model forward are executed.
"""

from collections import Counter
from itertools import combinations
from test.runtime.cache.test_dots3_note_recipe import PACKING, _layout
from test.runtime.cache.test_dots3_note_recipe import inputs as inputs
from test.runtime.cache.test_dots3_note_recipe import mtp_inputs as mtp_inputs

import pytest
import tokenspeed_scheduler as ts

from tokenspeed.runtime.layers.attention.kv_cache.recipes.dots3_note import (
    Dots3NoteRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (
    cache_group_config,
)

GROUPS = tuple(PACKING)
SWA_GROUPS = GROUPS[1:]


def _scheduler(inputs, *, parents, chunk_tokens, prefix_cache):
    recipe = Dots3NoteRecipe(**inputs)
    layout = _layout(recipe)
    recipe.check_layout(layout)
    # Bind the production layout to an explicit small pool for lifecycle pressure;
    # recipe capacity sizing has its own tests. No arena allocation is necessary.
    plan = layout.bind(parents)
    config = ts.SchedulerConfig()
    config.prefix_hash_lookahead_tokens = int(inputs["decode_input_tokens"] > 1)
    config.role = ts.SchedulerConfig.Role.Fused
    config.prefix_granularity = plan.prefix_granularity
    config.num_device_pages = parents + 1
    config.num_host_pages = 0
    config.max_batch_size = 2
    config.max_scheduled_tokens = chunk_tokens
    config.decode_input_tokens = inputs["decode_input_tokens"]
    config.overlap_schedule_depth = inputs["overlap_schedule_depth"]
    config.disable_prefix_cache = not prefix_cache
    config.disable_l2_cache = True
    config.enable_l3_storage = False
    config.enable_mixed_prefill_decode = False
    config.prefix_replay_tokens = 0
    config.cache_groups = [
        cache_group_config(
            spec,
            total_pages=plan.group(spec.group_id).page_count,
            cache_blocks_per_lcm_block=plan.group(
                spec.group_id
            ).cache_blocks_per_lcm_block,
        )
        for spec, _ in recipe.groups()
    ]
    assert config.prefix_granularity == 64
    packing = PACKING | ({"draft.swa": 15} if recipe.num_draft_layers else {})
    assert [g.group_id for g in config.cache_groups] == list(packing)
    assert [g.block_granularity for g in config.cache_groups] == [64] + [32] * (
        len(packing) - 1
    )
    assert [g.cache_blocks_per_lcm_block for g in config.cache_groups] == list(
        packing.values()
    )
    assert all(not g.replayable for g in config.cache_groups)
    assert [g.sliding_window_tokens for g in config.cache_groups[1:]] == [513] * (
        len(packing) - 1
    )
    scheduler = ts.Scheduler(config)
    assert scheduler.empty_lcm_blocks() == parents
    for gid, count in packing.items():
        assert scheduler.cache_group_total_pages(gid) == 1 + parents * count
        assert scheduler.cache_group_available_pages(gid) == parents * count
    return scheduler


def _submit(scheduler, request_id, tokens, *, max_new_tokens):
    spec = ts.RequestSpec()
    spec.request_id = request_id
    spec.tokens = tokens
    spec.max_new_tokens = max_new_tokens
    scheduler.submit_requests([spec])


def _advance(scheduler, request_id, tokens):
    event = ts.ForwardEvent.ExtendResult()
    event.request_id = request_id
    event.tokens = tokens
    scheduler.advance(ts.ExecutionEvent().add_event(event))


def _release(scheduler, request_id, *, abort):
    event = ts.ForwardEvent.Abort() if abort else ts.ForwardEvent.Finish()
    event.request_id = request_id
    scheduler.advance(ts.ExecutionEvent().add_event(event))


def _assert_idle(scheduler):
    # The unified path may emit an empty ForwardBatch even with no requests.
    plan = scheduler.next_execution_plan()
    assert all(not batch.request_ids for batch in plan.forward)
    assert scheduler.waiting_size() == 0
    assert scheduler.prefilling_size() == 0
    assert scheduler.decoding_size() == 0
    assert scheduler.active_lcm_blocks() == 0


def _batch(plan):
    assert len(plan.forward) == 1, f"expected a forward, got {plan}"
    assert not plan.cache
    assert plan.remote_prefill is None and plan.remote_decode is None
    batch = plan.forward[0]
    assert set(batch.block_tables) in (set(GROUPS), set(GROUPS) | {"draft.swa"})
    assert all(n == 0 for n in batch.extend_replay_lens)
    return batch


def _tables(batch, request_id):
    row = batch.request_ids.index(request_id)
    return {gid: list(rows[row]) for gid, rows in batch.block_tables.items()}


def _parents(gid, pages):
    # Page zero is null; positive group-local IDs encode parent and child.
    packing = 15 if gid == "draft.swa" else PACKING[gid]
    return {(page - 1) // packing + 1 for page in pages if page > 0}


def _assert_parent_exclusivity(scheduler, *batches):
    by_group = {
        gid: _parents(
            gid,
            [
                page
                for batch in batches
                for row in batch.block_tables[gid]
                for page in row
            ],
        )
        for gid in batches[0].block_tables
    }
    for left, right in combinations(by_group, 2):
        assert by_group[left].isdisjoint(by_group[right]), (left, right, by_group)
    assert scheduler.active_lcm_blocks() == sum(map(len, by_group.values()))


@pytest.mark.parametrize("prompt", [31, 32, 63, 64, 512, 513])
def test_mtp_reserve_verify_acceptance_and_retention(mtp_inputs, prompt):
    width = mtp_inputs["decode_input_tokens"]
    scheduler = _scheduler(
        mtp_inputs, parents=160, chunk_tokens=1024, prefix_cache=False
    )
    _submit(scheduler, "mtp", list(range(prompt)), max_new_tokens=32)
    prefill = _batch(scheduler.next_execution_plan())
    assert prefill.input_lengths == [prompt]
    original = _tables(prefill, "mtp")
    _assert_parent_exclusivity(scheduler, prefill)
    for gid, pages in original.items():
        size = 64 if gid == "full" else 32
        assert all(pages[p // size] > 0 for p in range(prompt - 1, prompt + width))
    if prompt >= 512:
        assert len(_parents("draft.swa", original["draft.swa"])) >= 2
    _advance(scheduler, "mtp", [10000])
    computed = prompt
    # All-rejected, all-accepted and a partial verify use the same scheduler path.
    for accepted in (1, width, max(1, width - 1)):
        batch = _batch(scheduler.next_execution_plan())
        assert batch.num_extends() == 0 and batch.input_lengths == [width]
        tables = _tables(batch, "mtp")
        _assert_parent_exclusivity(scheduler, batch)
        for gid in (*SWA_GROUPS, "draft.swa"):
            expired = max(0, computed - 512) // 32
            assert tables[gid][:expired] == [0] * expired
            assert all(
                tables[gid][p // 32] > 0
                for p in range(max(0, computed - 512), computed + width)
            )
        _advance(
            scheduler, "mtp", list(range(10001 + computed, 10001 + computed + accepted))
        )
        computed += accepted
    _release(scheduler, "mtp", abort=False)
    _assert_idle(scheduler)
    assert scheduler.available_lcm_blocks() == 160


def test_mtp_ordinary_prefix_reuses_draft_lookback(mtp_inputs):
    scheduler = _scheduler(
        mtp_inputs, parents=160, chunk_tokens=1024, prefix_cache=True
    )
    tokens = list(range(512))
    _submit(scheduler, "writer", tokens, max_new_tokens=1)
    original = _tables(_batch(scheduler.next_execution_plan()), "writer")
    _advance(scheduler, "writer", [10000])
    _release(scheduler, "writer", abort=False)
    _assert_idle(scheduler)
    _submit(scheduler, "reader", tokens + [10000], max_new_tokens=1)
    plan = scheduler.next_execution_plan()
    batch = _batch(plan)
    assert batch.extend_prefix_lens == [512]
    assert batch.input_lengths == [1] and batch.extend_replay_lens == [0]
    tables = _tables(batch, "reader")
    for gid, pages in tables.items():
        end = 8 if gid == "full" else 16
        assert pages[:end] == original[gid][:end]
        assert set(pages[:end]).isdisjoint(plan.pages_to_zero.get(gid, []))
    _assert_parent_exclusivity(scheduler, batch)
    _advance(scheduler, "reader", [20001])
    _release(scheduler, "reader", abort=False)
    _assert_idle(scheduler)


def test_allocation_uses_exclusive_parents_and_packing_five(inputs):
    scheduler = _scheduler(inputs, parents=64, chunk_tokens=256, prefix_cache=False)
    for i in range(2):
        _submit(
            scheduler, f"r{i}", list(range(i * 1000, i * 1000 + 64)), max_new_tokens=1
        )
    plan = scheduler.next_execution_plan()
    batch = _batch(plan)
    assert batch.input_lengths == [64, 64]
    _assert_parent_exclusivity(scheduler, batch)
    assert scheduler.active_lcm_blocks() == 18  # 4 Full + 6 + 6 + ceil(6/5).
    for gid, rows in batch.block_tables.items():
        pages = [page for row in rows for page in row]
        assert all(page > 0 for page in pages)
        assert len(pages) == len(set(pages))  # Unrelated requests never share a child.
        assert set(plan.pages_to_zero[gid]) == set(pages)
        assert [len(row) for row in rows] == ([2, 2] if gid == "full" else [3, 3])
    small = batch.block_tables["swa.2"]
    occupancy = Counter((page - 1) // 5 + 1 for row in small for page in row)
    assert sorted(occupancy.values()) == [1, 5]
    assert _parents("swa.2", small[0]) & _parents("swa.2", small[1])


@pytest.mark.parametrize("prefix_cache", [False, True])
@pytest.mark.parametrize("computed", [512, 513, 543, 544, 545, 671, 672])
def test_swa513_reclaims_only_whole_expired_pages(inputs, prefix_cache, computed):
    scheduler = _scheduler(
        inputs, parents=160, chunk_tokens=1024, prefix_cache=prefix_cache
    )
    _submit(scheduler, "r", list(range(computed)), max_new_tokens=2)
    initial = _tables(_batch(scheduler.next_execution_plan()), "r")
    empty_before = scheduler.empty_lcm_blocks()
    _advance(scheduler, "r", [10_000])
    batch = _batch(scheduler.next_execution_plan())
    assert batch.num_extends() == 0
    tables = _tables(batch, "r")
    assert tables["full"] == initial["full"]
    assert all(page > 0 for page in tables["full"])
    # The next query is at `computed`: it needs tokens [computed-512, computed].
    expired = max(0, computed - 512) // 32
    for gid in SWA_GROUPS:
        assert tables[gid][:expired] == [0] * expired
        assert tables[gid][expired:] == initial[gid][expired:]
        assert all(page > 0 for page in tables[gid][expired:])
        for token in range(max(0, computed - 512), computed + 1):
            assert tables[gid][token // 32] > 0, (gid, computed, token)
    # Disabling prefix reuse does not disable publication. The newest hashed
    # boundary publishes its lookback before retention releases request refs.
    cached_end = computed // 64 * 2
    cached_start = max(0, cached_end - 16)
    released_parents = sum(
        len(
            _parents(gid, initial[gid])
            - _parents(gid, tables[gid] + initial[gid][cached_start:cached_end])
        )
        for gid in SWA_GROUPS
    )
    assert scheduler.empty_lcm_blocks() - empty_before == released_parents
    _assert_parent_exclusivity(scheduler, batch)


def test_chunked_prefill_preserves_logical_slots_and_full_history(inputs):
    scheduler = _scheduler(inputs, parents=160, chunk_tokens=256, prefix_cache=True)
    tokens = list(range(1537))
    _submit(scheduler, "r", tokens, max_new_tokens=2)
    full = None
    previous = None
    for start in range(0, len(tokens), 256):
        end = min(start + 256, len(tokens))
        batch = _batch(scheduler.next_execution_plan())
        assert batch.request_ids == ["r"]
        assert batch.extend_prefix_lens == [start]
        assert batch.input_lengths == [end - start]
        assert batch.input_ids == tokens[start:end]
        tables = _tables(batch, "r")
        if full is None:
            full = tables["full"]
            assert len(full) == 25  # Fused first admission prepays prompt + generation.
        assert tables["full"] == full
        expired = max(0, start - 512) // 32
        for gid in SWA_GROUPS:
            assert len(tables[gid]) == (end + int(end == len(tokens)) + 31) // 32
            assert tables[gid][:expired] == [0] * expired
            assert all(page > 0 for page in tables[gid][expired:])
            if previous is not None:
                assert (
                    tables[gid][expired : len(previous[gid])] == previous[gid][expired:]
                )
        _assert_parent_exclusivity(scheduler, batch)
        previous = tables
        _advance(scheduler, "r", [10_000] if end == len(tokens) else [])
    decode = _batch(scheduler.next_execution_plan())
    assert decode.num_extends() == 0
    assert scheduler.decoding_size() == 1
    assert _tables(decode, "r")["full"] == full
    _advance(scheduler, "r", [10_001])
    _release(scheduler, "r", abort=False)
    _assert_idle(scheduler)
    assert scheduler.active_lcm_blocks() == 0
    assert scheduler.available_lcm_blocks() == 160


@pytest.mark.parametrize(
    "prompt_tokens, changed_token, hit_tokens, chunk_tokens",
    [
        (64, None, 64, 2048),
        (512, None, 512, 2048),
        (576, None, 576, 2048),
        (1024, None, 1024, 2048),
        (1024, 0, 0, 2048),
        # A single 1024-token chunk publishes only the SWA lookback [512,1024).
        # Earlier Full hits lack their complete SWA lookbacks, so cannot resume.
        (1024, 543, 0, 2048),
        (1024, 1023, 0, 2048),
        # Publishing the intermediate endpoint supplies the missing old pages.
        (1024, 543, 512, 512),
        (1024, 1023, 960, 512),
    ],
)
def test_prefix_hit_and_miss_keep_every_swa_lookback(
    inputs, prompt_tokens, changed_token, hit_tokens, chunk_tokens
):
    scheduler = _scheduler(
        inputs, parents=192, chunk_tokens=chunk_tokens, prefix_cache=True
    )
    tokens = list(range(prompt_tokens))
    _submit(scheduler, "writer", tokens, max_new_tokens=1)
    for start in range(0, prompt_tokens, chunk_tokens):
        batch = _batch(scheduler.next_execution_plan())
        assert batch.extend_prefix_lens == [start]
        assert batch.input_ids == tokens[start : start + chunk_tokens]
        original = _tables(batch, "writer")
        _advance(
            scheduler,
            "writer",
            [10_000] if start + chunk_tokens >= prompt_tokens else [],
        )
    _release(scheduler, "writer", abort=False)
    _assert_idle(scheduler)
    assert scheduler.active_lcm_blocks() == 0
    assert scheduler.empty_lcm_blocks() < 192
    assert scheduler.available_lcm_blocks() == 192

    reader_tokens = tokens + [20_000]
    if changed_token is not None:
        reader_tokens[changed_token] = 30_000
    _submit(scheduler, "reader", reader_tokens, max_new_tokens=1)
    plan = scheduler.next_execution_plan()
    batch = _batch(plan)
    assert batch.extend_prefix_lens == [hit_tokens]
    first_chunk = min(len(reader_tokens) - hit_tokens, chunk_tokens)
    if hit_tokens == 0 and changed_token not in (None, 0):
        # The ordinary promotion boundary splits recomputation at Full's hit;
        # it does not skip tokens whose SWA lookback is absent.
        first_chunk = min(first_chunk, changed_token // 64 * 64)
    assert batch.input_ids == reader_tokens[hit_tokens : hit_tokens + first_chunk]
    assert batch.input_lengths == [first_chunk]
    tables = _tables(batch, "reader")
    assert tables["full"][: hit_tokens // 64] == original["full"][: hit_tokens // 64]
    for gid in SWA_GROUPS:
        first, end = max(0, hit_tokens - 512) // 32, hit_tokens // 32
        assert tables[gid][:first] == [0] * first
        assert tables[gid][first:end] == original[gid][first:end]
        assert all(page > 0 for page in tables[gid][first:])
        assert set(tables[gid][first:end]).isdisjoint(plan.pages_to_zero.get(gid, []))
    assert set(tables["full"][: hit_tokens // 64]).isdisjoint(
        plan.pages_to_zero.get("full", [])
    )
    if hit_tokens == 0:
        for gid in GROUPS:
            assert set(tables[gid]) == set(plan.pages_to_zero[gid])
    _assert_parent_exclusivity(scheduler, batch)
    first_end = hit_tokens + first_chunk
    _advance(scheduler, "reader", [20_001] if first_end == len(reader_tokens) else [])
    for start in range(first_end, len(reader_tokens), chunk_tokens):
        end = min(start + chunk_tokens, len(reader_tokens))
        batch = _batch(scheduler.next_execution_plan())
        assert batch.extend_prefix_lens == [start]
        assert batch.input_ids == reader_tokens[start:end]
        for gid in SWA_GROUPS:
            row = _tables(batch, "reader")[gid]
            assert all(page > 0 for page in row[max(0, start - 512) // 32 :])
        _advance(scheduler, "reader", [20_001] if end == len(reader_tokens) else [])
    _release(scheduler, "reader", abort=False)
    _assert_idle(scheduler)
    assert scheduler.available_lcm_blocks() == 192


@pytest.mark.parametrize("parents, expected_hit", [(192, 0), (512, 1024)])
def test_swa_eviction_blocks_prefix_hit_even_while_full_history_is_pinned(
    inputs, parents, expected_hit
):
    scheduler = _scheduler(inputs, parents=parents, chunk_tokens=256, prefix_cache=True)
    tokens = list(range(4096))
    _submit(scheduler, "source", tokens, max_new_tokens=2)
    full = None
    for start in range(0, len(tokens), 256):
        batch = _batch(scheduler.next_execution_plan())
        assert batch.extend_prefix_lens == [start]
        assert batch.input_ids == tokens[start : start + 256]
        tables = _tables(batch, "source")
        if full is None:
            full = tables["full"]
        assert tables["full"] == full
        _advance(scheduler, "source", [10_000] if start == 3840 else [])
    source = _batch(scheduler.next_execution_plan())
    assert source.num_extends() == 0
    assert _tables(source, "source")["full"] == full
    assert all(page > 0 for page in full)
    _advance(scheduler, "source", [10_001])
    if expected_hit == 0:
        assert scheduler.empty_lcm_blocks() == 0

    # Full KV stays pinned by the live source in both cases. With 192 parents,
    # growth evicts old SWA prefix entries; Full alone must not authorize a hit.
    branch_tokens = tokens[:1024] + [20_000]
    _submit(scheduler, "branch", branch_tokens, max_new_tokens=1)
    plan = scheduler.next_execution_plan()
    branch = _batch(plan)
    assert branch.request_ids == ["branch"]
    assert branch.extend_prefix_lens == [expected_hit]
    assert branch.input_ids == branch_tokens[expected_hit : expected_hit + 256]
    _assert_parent_exclusivity(scheduler, source, branch)
    tables = _tables(branch, "branch")
    if expected_hit == 0:
        for gid in GROUPS:
            assert all(page > 0 for page in tables[gid])
            assert set(tables[gid]) == set(plan.pages_to_zero[gid])
        # These are new Full pages, not holes masquerading as reused history.
        assert set(tables["full"]).isdisjoint(full)
    else:
        assert tables["full"][:16] == full[:16]
        for gid in SWA_GROUPS:
            assert tables[gid][:16] == [0] * 16
            assert all(page > 0 for page in tables[gid][16:33])

    _advance(scheduler, "branch", [20_001] if expected_hit else [])
    _release(scheduler, "branch", abort=True)
    _release(scheduler, "source", abort=False)
    _assert_idle(scheduler)
    assert scheduler.available_lcm_blocks() == parents
    assert scheduler.clear_cache()
    assert scheduler.empty_lcm_blocks() == parents


@pytest.mark.parametrize("prefix_cache", [False, True])
@pytest.mark.parametrize("abort", [False, True], ids=["finish", "abort"])
def test_release_and_reallocation_preserve_shared_parent_ownership(
    inputs, prefix_cache, abort
):
    scheduler = _scheduler(
        inputs, parents=20, chunk_tokens=128, prefix_cache=prefix_cache
    )
    previous_parents = set()
    for cycle in range(3):
        for i in range(2):
            start = cycle * 1000 + i * 100
            _submit(
                scheduler, f"r{i}", list(range(start, start + 64)), max_new_tokens=4
            )
        batch = _batch(scheduler.next_execution_plan())
        assert batch.request_ids == ["r0", "r1"]
        assert batch.extend_prefix_lens == [0, 0]
        _assert_parent_exclusivity(scheduler, batch)
        current_parents = set().union(
            *(
                _parents(gid, [page for row in rows for page in row])
                for gid, rows in batch.block_tables.items()
            )
        )
        assert len(current_parents) == 18
        if previous_parents:
            assert len(previous_parents & current_parents) >= 16
        previous_parents = current_parents
        survivor = _tables(batch, "r1")
        assert _parents("swa.2", _tables(batch, "r0")["swa.2"]) & _parents(
            "swa.2", survivor["swa.2"]
        )
        for i in range(2):
            _advance(scheduler, f"r{i}", [10_000])
        decode = _batch(scheduler.next_execution_plan())
        assert decode.num_extends() == 0
        for i in range(2):
            _advance(scheduler, f"r{i}", [10_001])
        _release(scheduler, "r0", abort=abort)
        assert scheduler.active_lcm_blocks() == sum(
            len(_parents(gid, pages)) for gid, pages in survivor.items()
        )
        # Releasing one owner cannot free the other request's packed children.
        remaining = _batch(scheduler.next_execution_plan())
        assert remaining.request_ids == ["r1"]
        assert _tables(remaining, "r1") == survivor
        assert scheduler.request_token_size("r0") == -1
        _assert_parent_exclusivity(scheduler, remaining)
        _advance(scheduler, "r1", [10_002])
        _release(scheduler, "r1", abort=abort)
        _assert_idle(scheduler)
        assert scheduler.request_token_size("r1") == -1
        assert scheduler.available_lcm_blocks() == 20
        # Both settings publish completed blocks; only matching is disabled.
        assert scheduler.empty_lcm_blocks() < 20
    assert scheduler.clear_cache()
    assert scheduler.empty_lcm_blocks() == 20
    # Unlike available_lcm_blocks, this gauge excludes cache-only eviction.
    for gid in GROUPS:
        assert scheduler.cache_group_available_pages(gid) == 20 * PACKING[gid]
