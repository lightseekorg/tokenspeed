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

"""A lone request within max_single_request_tokens completes on a fragmented
pool (docs/design/scheduler.md §1.4), through the public bindings.

Fused role, prefix cache on, no Host tier, no snapshot pool. A history group
of PACK 8-token blocks per LCM parent beside a 64-token sliding-window group
of one block per parent. Short one-page prompts admitted in one plan each
take a history parent (the page plus the block reserved for their first
decode) and a sliding parent, and finish on their first token: every history
parent is left [cached page | hole] and every sliding page stays cached. One
long prompt then arrives alone. Its first chunk prepays all its history
blocks, which fill the holes first, one per parent, so its blocks span more
parents than a dense layout needs; a later chunk needs sliding parents that
evicting the cached pages beside its own blocks cannot free. Before the
compaction it waited for good; now its reserve blocks are re-placed into its
own parents and the parents they leave fund the retry in the same plan. The
copy contract of the compaction's Device copies is the owner copy-in's: no
self-copy, each destination written once and never zeroed in the same plan,
both ends in one shard's bucket.
"""

from __future__ import annotations

import pytest

ts = pytest.importorskip("tokenspeed_scheduler")

from conftest import _event, _spec  # noqa: E402

P = 8
HISTORY = "full_attention"


def _config(parents: int, chunk: int, packing: int, shards: int) -> ts.SchedulerConfig:
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = P
    cfg.num_device_pages = parents + 1  # page 0 is the null sentinel
    cfg.num_host_pages = 0
    cfg.disable_l2_cache = True
    cfg.enable_l3_storage = False
    cfg.num_snapshot_pages = 1
    cfg.max_retracted_requests = 0
    cfg.max_scheduled_tokens = chunk
    cfg.max_batch_size = 64
    cfg.decode_input_tokens = 1
    cfg.overlap_schedule_depth = 0
    cfg.disable_prefix_cache = False
    cfg.role = ts.SchedulerConfig.Role.Fused
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id=HISTORY,
            block_granularity=P,
            total_pages=(parents + 1) * packing,
            attn_kind=ts.AttnKind.Full,
            cache_blocks_per_lcm_block=packing,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
            shard_count=shards,
        ),
        ts.CacheGroupConfig(
            group_id="sliding",
            block_granularity=P,
            total_pages=parents + 1,
            attn_kind=ts.AttnKind.SlidingWindow,
            sliding_window_tokens=64,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        ),
    ]
    return cfg


def _submit(scheduler, request_id: str, tokens: list[int]) -> None:
    spec = _spec(request_id, tokens)
    spec.max_new_tokens = 1
    scheduler.submit_requests([spec])


def _step(scheduler, shards: int):
    """One plan, every row landed: an intermediate chunk produces no token, a
    last chunk or a decode one, on which the request finishes. Checks the copy
    contract and returns the plan."""
    plan = scheduler.next_execution_plan()
    for group, copies in plan.pages_to_copy.items():
        destinations = [destination for _, destination in copies]
        assert len(set(destinations)) == len(destinations), group
        assert not set(destinations) & set(plan.pages_to_zero[group]), group
        for source, destination in copies:
            assert source != destination, group
            if group == HISTORY:
                assert (source - 1) % shards == (destination - 1) % shards
    events = ts.ExecutionEvent()
    for op in plan.forward:
        for i, request_id in enumerate(op.request_ids):
            mid_chunk = (
                i < op.num_extends()
                and op.extend_prefix_lens[i] + op.input_lengths[i]
                < op.prefill_lengths[i]
            )
            events.add_event(
                _event("ExtendResult", request_id, tokens=[] if mid_chunk else [7])
            )
            if not mid_chunk:
                events.add_event(_event("Finish", request_id))
    scheduler.advance(events)
    return plan


def _fragment(scheduler, prompts: int, shards: int) -> None:
    """The one-page prompts, admitted together and finished at once."""
    for k in range(prompts):
        _submit(scheduler, f"a{k}", [100 + 50 * k + j for j in range(P)])
    for _ in range(4):
        if scheduler.request_token_size(f"a{prompts - 1}") == -1:
            break
        _step(scheduler, shards)
    assert scheduler.request_token_size(f"a{prompts - 1}") == -1


# 28 parents, two history blocks per parent: eight prompts leave eight holes;
# a 126-token prompt (the bound is 128) holds 16 history blocks and the
# copy-in block of its prompt end, which fill the holes first, so after its
# first 64-token chunk they span 13 history parents instead of 9; its second
# chunk needs eight sliding parents and the sliding copy-in block's, which
# seven cached sliding pages cannot fund. The compaction re-places the block
# alone in a fresh parent (slot 14, page 5) and the copy-in block (page 8)
# into the slots of the cached pages beside the request's own blocks (pages 13
# and 19, cleared; page 13 zeroed, page 19 left to the copy-in that fills it
# whole), and the two parents they leave fund the retry.
def test_lone_request_within_the_bound_completes_on_a_fragmented_pool() -> None:
    scheduler = ts.Scheduler(_config(parents=28, chunk=64, packing=2, shards=1))
    assert scheduler.max_single_request_tokens() == 128
    _fragment(scheduler, prompts=8, shards=1)
    assert scheduler.empty_lcm_blocks() == 12
    assert scheduler.available_lcm_blocks() == 28

    _submit(scheduler, "R", list(range(3000, 3126)))
    first = _step(scheduler, shards=1)
    (op,) = first.forward
    assert list(op.request_ids) == ["R"]
    assert list(op.input_lengths) == [64]
    row = [page for page in op.block_tables[HISTORY][0] if page != -1]
    assert row == [14, 20, 26, 32, 38, 44, 49, 50, 51, 52, 53, 54, 55, 56, 5, 2]
    assert scheduler.empty_lcm_blocks() == 0
    assert scheduler.host_cache_stats().lone_compactions == 0

    second = _step(scheduler, shards=1)
    (op,) = second.forward
    assert list(op.request_ids) == ["R"]
    assert list(op.extend_prefix_lens) == [64]
    row = [page for page in op.block_tables[HISTORY][0] if page != -1]
    assert row == [14, 20, 26, 32, 38, 44, 49, 50, 51, 52, 53, 54, 55, 56, 13, 2]
    assert second.pages_to_zero[HISTORY] == [13]
    assert all(not copies for copies in second.pages_to_copy.values())
    stats = scheduler.host_cache_stats()
    assert stats.lone_compactions == 1
    assert stats.lone_compaction_moves == 2
    assert stats.device_evictions.compaction.blocks == 2

    _step(scheduler, shards=1)
    assert scheduler.request_token_size("R") == -1
    assert scheduler.host_cache_stats().lone_compactions == 1


# The same prompt on an empty pool takes the same two plans and never
# compacts: the compaction runs only when an admission fails.
def test_lone_request_on_an_empty_pool_never_compacts() -> None:
    scheduler = ts.Scheduler(_config(parents=28, chunk=64, packing=2, shards=1))
    _submit(scheduler, "R", list(range(3000, 3126)))
    for _ in range(2):
        plan = _step(scheduler, shards=1)
        assert any(op.request_ids for op in plan.forward)
    _step(scheduler, shards=1)
    assert scheduler.request_token_size("R") == -1
    assert scheduler.host_cache_stats().lone_compactions == 0
    assert scheduler.host_cache_stats().device_evictions.compaction.blocks == 0


# Two KV-parallel shards, one block of each per parent: the twelve prompts'
# holes all sit in bucket 1, where the prompt end's block (slot 31, page 20)
# and its copy-in block land first and the balanced allocator fills seven
# more before opening fresh parents. Those two keep their bucket: each
# re-places within bucket 1 into a slot one of the request's free unwritten
# blocks gives up (pages 82 and 6), and those blocks move to the bucket-0
# slots of cached pages beside the request's own blocks (pages 31 and 37,
# cleared): only an unwritten block without a copy-in block beside it may
# change bucket, and nothing is copied.
def test_unwritten_blocks_re_place_across_buckets_under_kv_parallelism() -> None:
    scheduler = ts.Scheduler(_config(parents=44, chunk=128, packing=2, shards=2))
    assert scheduler.max_single_request_tokens() == 256
    _fragment(scheduler, prompts=12, shards=2)
    assert scheduler.empty_lcm_blocks() == 20

    _submit(scheduler, "R", list(range(3000, 3254)))
    first = _step(scheduler, shards=2)
    (op,) = first.forward
    before = [page for page in op.block_tables[HISTORY][0] if page != -1]
    holes, fresh, late = (
        [32, 38, 44, 50, 56, 62, 68],
        list(range(73, 89)),
        [5, 6, 11, 12, 17, 18, 23, 24, 20],
    )
    assert before == holes + fresh + late
    second = _step(scheduler, shards=2)
    (op,) = second.forward
    assert list(op.extend_prefix_lens) == [128]
    after = [page for page in op.block_tables[HISTORY][0] if page != -1]
    expected = list(before)
    expected[16], expected[24], expected[31] = 31, 37, 6
    assert after == expected
    # A page's bucket is (page - 1) % shards: the prompt end's block stays in
    # bucket 1 (20 -> 6), the displaced free blocks leave it (82 -> 31, 6 -> 37).
    assert [(page - 1) % 2 for page in (20, 6, 82, 31, 37)] == [1, 1, 1, 0, 0]
    # Page 82, the copy-in block's new place, is not zeroed.
    assert second.pages_to_zero[HISTORY] == [31, 37, 6]
    assert all(not copies for copies in second.pages_to_copy.values())
    stats = scheduler.host_cache_stats()
    assert stats.lone_compactions == 1
    assert stats.lone_compaction_moves == 4
    assert stats.device_evictions.compaction.blocks == 2
    _step(scheduler, shards=2)
    assert scheduler.request_token_size("R") == -1


# Under KV parallelism a lone first admission whose probe hits a Device tail
# can fail every plan with nothing to move (docs/design/scheduler.md §1.4):
# three parents of one block of each of two buckets, the history group alone.
# X (17 tokens) leaves its two pages in bucket 1 and its one-token tail in
# bucket 0. R, X plus 14 tokens, needs the tail's private copy beside the
# tail and its prompt end's block with that block's copy-in block in one
# bucket, which neither bucket has room for. On the fused role R is aborted
# with AbortReason.NoProgress at the 4096th consecutive empty plan
# (Scheduler::kNoProgressAbortPlans), and the runtime fails it toward its
# client.
def test_lone_first_admission_that_can_never_fit_is_aborted() -> None:
    cfg = _config(parents=3, chunk=16, packing=2, shards=2)
    cfg.cache_groups = cfg.cache_groups[:1]
    scheduler = ts.Scheduler(cfg)
    assert scheduler.max_single_request_tokens() == 32
    prompt = list(range(20000, 20017))
    _submit(scheduler, "X", prompt)
    for _ in range(4):
        _step(scheduler, shards=2)
    assert scheduler.request_token_size("X") == -1

    _submit(scheduler, "R", prompt + list(range(50000, 50014)))
    aborts = []
    for plan_number in range(1, 4097):
        plan = _step(scheduler, shards=2)
        assert not any(op.request_ids for op in plan.forward)
        aborts += [(plan_number, abort) for abort in plan.aborts]
    assert [(n, abort.request_id, abort.reason) for n, abort in aborts] == [
        (4096, "R", ts.AbortReason.NoProgress)
    ]
    assert "no progress for 4096 consecutive plans" in aborts[0][1].detail
    assert scheduler.host_cache_stats().lone_compactions == 0
    _step(scheduler, shards=2)
    assert scheduler.request_token_size("R") == -1
