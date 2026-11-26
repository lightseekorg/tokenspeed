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

"""Prefill schedulers sharing one L2 pool as their Host L2. The test plays
the runtime: it answers each scheduler's lookups and commands from a real
SharedL2Pool and completes load-backs and writes, when it chooses to. No
bytes move."""

from __future__ import annotations

import pytest
import tokenspeed_scheduler as ts
from conftest import _advance, _event, _finish, _spec

GRAIN = 2
ROLES = ts.SchedulerConfig.Role


def groups(pages):
    full = ts.CacheGroupConfig(
        group_id="kv",
        block_granularity=GRAIN,
        total_pages=pages,
        transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        attn_kind=ts.AttnKind.Full,
    )
    state = ts.CacheGroupConfig(
        group_id="state",
        block_granularity=0,
        total_pages=pages,
        transfer_policy=ts.CacheTransferPolicy.LatestSnapshot,
        attn_kind=ts.AttnKind.State,
    )
    return [full, state]


def config(
    *,
    kv_events=False,
    role=ROLES.Fused,
    max_writes_in_flight=2,
    max_keys_per_write=64,
    device_pages=64,
    chunk=64,
):
    cfg = ts.SchedulerConfig()
    cfg.role = role
    cfg.enable_kv_cache_events = kv_events
    cfg.prefix_granularity = GRAIN
    cfg.num_snapshot_pages = 1
    cfg.num_device_pages = device_pages
    # The proxy Host tier holds no bytes: twice the pool.
    cfg.num_host_pages = 129
    cfg.disable_l2_cache = False
    cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteThrough
    cfg.host_group_packing = [1, 1]
    cfg.max_scheduled_tokens = chunk
    cfg.max_batch_size = 8
    cfg.max_resident_requests = cfg.max_batch_size
    cfg.cache_groups = groups(device_pages)
    cfg.shared_l2_config = ts.SharedL2Config(
        max_writes_in_flight=max_writes_in_flight, max_keys_per_write=max_keys_per_write
    )
    return cfg


def scheduler(**kwargs):
    return ts.Scheduler(config(**kwargs))


def owner(blocks=64):
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = GRAIN
    cfg.cache_groups = groups(64)
    cfg.host_group_packing = [1, 1]
    cfg.checkpoint_retention_requests = 64
    return ts.SharedL2Pool(cfg, blocks, 0)


class Runtime:
    """One scheduler's runtime: it answers the scheduler's lookups and
    carries out its shared L2 ops against the pool, as the event loop, the
    pool client and the load executor do."""

    def __init__(self, pool, client, *, complete_loads=True, complete_writes=True):
        self.pool, self.client = pool, client
        self.complete_loads = complete_loads
        self.complete_writes = complete_writes
        self.lookups, self.loads, self.writes, self.touched = [], [], [], []
        self.released = []
        self.forwarded = []
        # (group, proxy Host block) -> the pool page a load from it reads.
        self.pages = {}
        self.open_loads = []
        self.open_writes = []

    def answer(self, lookup):
        request_id, hashes, tokens, max_tokens = lookup
        return (
            request_id,
            *self.pool.lookup(self.client, hashes, tokens, max_tokens),
        )

    def apply(self, s, answers):
        for group, host_block, _, pool_page in s.apply_shared_l2_answers(answers):
            self.pages[(group, host_block)] = pool_page

    def lookup(self, s):
        lookups = s.start_shared_l2_lookups()
        self.lookups.extend(lookups)
        self.apply(s, [self.answer(lookup) for lookup in lookups])

    def write(self, op):
        ticket, pages = self.pool.reserve(self.client, op.keys, op.buckets)
        # Ticket 0 reserves nothing: there is nothing to commit.
        if ticket != 0:
            self.pool.commit(self.client, ticket, op.checkpoints)
        return ticket, pages

    def plan(self, s, *, lookup=True):
        if lookup:
            self.lookup(s)
        plan = s.next_execution_plan()
        for batch in plan.forward:
            self.forwarded.extend(batch.request_ids)
        e = ts.ExecutionEvent()
        for op in plan.cache:
            if isinstance(op, ts.Cache.LoadBackOp):
                for op_id, groups_, sources in zip(
                    op.op_ids, op.group_ids, op.src_pages
                ):
                    # Every source stands for a page the pool pins.
                    self.loads.append(
                        [
                            (group, self.pages[(group, source)])
                            for group, source in zip(groups_, sources)
                        ]
                    )
                    self.open_loads.append(op_id)
            elif isinstance(op, ts.Cache.SharedL2ReleaseOp):
                self.released.append(op.ticket)
                self.pool.release(self.client, op.ticket)
            elif isinstance(op, ts.Cache.SharedL2WriteOp):
                self.writes.append(op)
                if self.complete_writes:
                    self.write(op)
                    e.add_event(ts.Cache.SharedL2WriteDone(op.op_id))
                else:
                    self.open_writes.append(op)
            elif isinstance(op, ts.Cache.SharedL2TouchOp):
                self.touched.extend(op.keys)
                self.pool.touch(op.keys)
            else:
                raise AssertionError(f"unexpected cache operation {op}")
        if self.complete_loads:
            self.finish_loads(e)
        s.advance(e)
        return plan

    def finish_loads(self, e=None):
        event = e if e is not None else ts.ExecutionEvent()
        for op_id in self.open_loads:
            event.add_event(ts.Cache.LoadBackDoneEvent(op_id))
        self.open_loads = []
        return event

    def finish_writes(self):
        """Carries out the writes left open and reports them."""
        event = ts.ExecutionEvent()
        for op in self.open_writes:
            self.write(op)
            event.add_event(ts.Cache.SharedL2WriteDone(op.op_id))
        self.open_writes = []
        return event


def prefill(s, runtime, rid, tokens):
    """Plan until the request's first forward; return that forward batch."""
    s.submit_requests([_spec(rid, tokens)])
    for _ in range(6):
        plan = runtime.plan(s)
        for batch in plan.forward:
            if rid in batch.request_ids:
                _advance(s, rid, [1000])
                return batch
    raise AssertionError(f"{rid} was never scheduled")


def publish(writer, runtime, rid, tokens):
    batch = prefill(writer, runtime, rid, tokens)
    for _ in range(3):
        runtime.plan(writer)
    return batch


def settled(s, pool):
    """Nothing of the rank is left at the pool or in its proxy."""
    return (
        s.host_pool_cached_blocks() == 0
        and pool.stats()["pinned_blocks"] == 0
        and pool.stats()["reserved_blocks"] == 0
        and pool.stats()["open_lookups"] == 0
    )


def test_a_rank_resumes_from_the_pool_where_another_rank_published():
    pool = owner()
    writer, reader = scheduler(), scheduler()
    writer_rt, reader_rt = Runtime(pool, 1), Runtime(pool, 2)
    prompt = list(range(9))

    first = publish(writer, writer_rt, "a", prompt)
    assert list(first.extend_prefix_lens) == [0]
    # Four pages, the tail's block and the State block, with the
    # checkpoint: its tokens, page hashes and tail.
    [write] = writer_rt.writes
    assert [key[1] for key in write.keys] == [0] * 5 + [1]
    assert write.buckets == [0] * 6
    [(tokens, hashes, tail, _)] = write.checkpoints
    assert len(tokens) == 4 * len(prompt) and len(hashes) == 4 and tail[0] == 1
    assert tail[1] == write.keys[-1][2]
    assert pool.stats()["cached_blocks"] == 6
    assert pool.stats()["checkpoints"] == 1
    assert settled(writer, pool)

    second = prefill(reader, reader_rt, "b", prompt + [50, 51])
    # The reader resumes at the writer's prompt end: its pages, its tail and
    # the State checkpoint there all load from the pool, one load-back.
    assert list(second.extend_prefix_lens) == [len(prompt)]
    [load] = reader_rt.loads
    assert sorted(group for group, _ in load) == [0] * 5 + [1]
    assert len(set(load)) == len(load)
    assert reader.host_cache_stats().prefix_hits.host_hit_tokens == len(prompt)
    for _ in range(3):
        reader_rt.plan(reader)
    assert settled(reader, pool)
    assert reader.host_cache_stats().host_evictions.released.blocks == 6


def test_a_waiting_prompt_is_admitted_only_after_its_lookup():
    pool = owner()
    writer, reader = scheduler(), scheduler()
    writer_rt, reader_rt = Runtime(pool, 1), Runtime(pool, 2)
    prompt = list(range(9))
    publish(writer, writer_rt, "a", prompt)

    reader.submit_requests([_spec("b", prompt + [50, 51])])
    # Not asked yet, then asked and not answered: no plan admits it.
    reader_rt.plan(reader, lookup=False)
    [lookup] = reader.start_shared_l2_lookups()
    assert lookup[0] == "b" and lookup[3] == len(prompt) + 1
    for _ in range(3):
        reader_rt.plan(reader, lookup=False)
    assert "b" not in reader_rt.forwarded
    reader_rt.apply(reader, [reader_rt.answer(lookup)])
    [batch] = [b for b in reader_rt.plan(reader).forward if "b" in b.request_ids]
    assert list(batch.extend_prefix_lens) == [len(prompt)]
    # One lookup per prompt.
    for _ in range(3):
        reader_rt.plan(reader)
    assert not reader_rt.lookups


def test_the_pool_pins_a_hit_until_its_load_completes():
    pool = owner()
    writer, reader = scheduler(), scheduler()
    writer_rt = Runtime(pool, 1)
    reader_rt = Runtime(pool, 2, complete_loads=False)
    prompt = list(range(9))
    publish(writer, writer_rt, "a", prompt)

    prefill(reader, reader_rt, "b", prompt + [50, 51])
    assert reader_rt.open_loads
    for _ in range(3):
        reader_rt.plan(reader)
    # The lookup's hold went with the admission; the load-back holds on.
    assert pool.stats()["pinned_blocks"] == 6 and not reader_rt.released
    assert reader.host_pool_pinned_blocks() == 6
    reader.advance(reader_rt.finish_loads())
    reader_rt.plan(reader)
    assert len(reader_rt.released) == 1
    assert settled(reader, pool)


def test_a_request_gone_before_admission_lets_its_hit_go():
    pool = owner()
    writer, reader = scheduler(), scheduler()
    writer_rt, reader_rt = Runtime(pool, 1), Runtime(pool, 2)
    prompt = list(range(9))
    publish(writer, writer_rt, "a", prompt)

    def abort(rid):
        e = ts.ExecutionEvent()
        e.add_event(_event("Abort", rid))
        reader.advance(e)

    # Answered: the proxy entries go, then the pool's pins.
    reader.submit_requests([_spec("b", prompt + [50, 51])])
    reader_rt.lookup(reader)
    assert pool.stats()["pinned_blocks"] == 6
    abort("b")
    for _ in range(2):
        reader_rt.plan(reader)
    assert len(reader_rt.released) == 1
    assert settled(reader, pool)

    # Gone between its lookup and the answer, which a runtime that answers a
    # lookup in the call that sent it never sees: the answer names a request
    # the scheduler no longer knows, so no lookup that waits, and is refused.
    reader.submit_requests([_spec("c", prompt + [60, 61])])
    [lookup] = reader.start_shared_l2_lookups()
    late = reader_rt.answer(lookup)
    abort("c")
    reader_rt.plan(reader)
    with pytest.raises(ValueError, match="no lookup that waits"):
        reader.apply_shared_l2_answers([late])
    assert "b" not in reader_rt.forwarded and "c" not in reader_rt.forwarded
    assert not reader_rt.loads and reader.host_pool_cached_blocks() == 0


def test_prompts_sharing_a_pool_prefix_both_resume_from_it():
    pool = owner()
    writer, reader = scheduler(), scheduler()
    writer_rt, reader_rt = Runtime(pool, 1), Runtime(pool, 2)
    prompt = list(range(9))
    publish(writer, writer_rt, "a", prompt)

    reader.submit_requests(
        [_spec("x", prompt + [50, 51]), _spec("y", prompt + [60, 61])]
    )
    prefixes = {}
    for _ in range(6):
        for batch in reader_rt.plan(reader).forward:
            for rid, prefix in zip(batch.request_ids, batch.extend_prefix_lens):
                prefixes.setdefault(rid, prefix)
                _advance(reader, rid, [1000])
    assert prefixes == {"x": len(prompt), "y": len(prompt)}
    # Admitted together, as from Host L2: the first loads the whole hit, and
    # the second takes the full pages the first one's admission published on
    # the Device and loads the tail's block and the State checkpoint again,
    # into blocks of its own.
    first, second = reader_rt.loads
    assert len(first) == 6 and set(second) < set(first)
    assert sorted(group for group, _ in second) == [0, 1]
    # The second lookup's pages were proxied already: its ticket went back
    # at once, and the first one's once both loads had landed.
    assert len(reader_rt.released) == 2
    assert reader.host_cache_stats().host_evictions.released.blocks == 6
    assert settled(reader, pool)


def step(s, runtime, rid, prompt_len):
    """One plan; the request's chunk, if it ran, completes (only the last
    chunk of the prompt samples a token)."""
    plan = runtime.plan(s)
    batch = next((b for b in plan.forward if rid in b.request_ids), None)
    if batch is None:
        return None
    row = list(batch.request_ids).index(rid)
    if row >= len(batch.extend_prefix_lens):
        _advance(s, rid, [1000])  # a decode step
        return None
    end = batch.extend_prefix_lens[row] + batch.input_lengths[row]
    _advance(s, rid, [1000] if end == prompt_len else [])
    return batch


def test_a_device_hit_refreshes_the_pool_copies():
    pool = owner()
    s, runtime = scheduler(), Runtime(pool, 1)
    prompt = list(range(9))
    publish(s, runtime, "a", prompt)
    assert not runtime.touched
    again = prompt + [70]
    s.submit_requests([_spec("b", again)])
    batches = [step(s, runtime, "b", len(again)) for _ in range(3)]
    first = next(b for b in batches if b is not None)
    assert list(first.extend_prefix_lens) == [len(prompt)]
    # The Device served the hit: nothing loads, and the pool is told which of
    # its copies were used.
    assert not runtime.loads
    written = {key for op in runtime.writes[:1] for key in op.keys}
    assert set(runtime.touched) == written
    assert settled(s, pool)


def test_a_prompt_the_device_covers_asks_the_pool_nothing():
    pool = owner()
    s, runtime = scheduler(), Runtime(pool, 1)
    prompt = list(range(9))
    publish(s, runtime, "a", prompt)
    asked = len(runtime.lookups)
    # All but the last token is cached on the Device, and the last one is
    # always recomputed: the pool has nothing to add.
    again = prompt + [70]
    s.submit_requests([_spec("b", again[:10])])
    batches = [step(s, runtime, "b", 10) for _ in range(3)]
    assert next(b for b in batches if b is not None)
    assert len(runtime.lookups) == asked


def test_blocks_loaded_from_the_pool_are_not_written_back():
    pool = owner()
    writer, reader = scheduler(), scheduler()
    writer_rt, reader_rt = Runtime(pool, 1), Runtime(pool, 2)
    prompt = list(range(9))
    publish(writer, writer_rt, "a", prompt)
    written = {key for op in writer_rt.writes for key in op.keys}

    publish(reader, reader_rt, "b", prompt + [50, 51, 52, 53])
    assert reader_rt.loads
    rewritten = {key for op in reader_rt.writes for key in op.keys}
    assert rewritten and not rewritten & written
    # The pool was not asked for a page it holds either.
    assert pool.stats()["cached_blocks"] == len(written) + len(rewritten)


def test_kv_the_pool_holds_past_its_checkpoints_is_a_junction():
    pool = owner()
    writer, branch, sibling = scheduler(), scheduler(), scheduler()
    writer_rt, branch_rt, sibling_rt = (
        Runtime(pool, 1),
        Runtime(pool, 2),
        Runtime(pool, 3),
    )
    shared = list(range(8))
    publish(writer, writer_rt, "a", shared + [10, 11, 12, 13])

    # The pool holds the shared KV but no checkpoint inside it: the branch
    # computes from the start and cuts its first chunk where that KV ends.
    branch_prompt = shared + [20, 21, 22, 23]
    branch.submit_requests([_spec("b", branch_prompt)])
    batches = [step(branch, branch_rt, "b", len(branch_prompt)) for _ in range(6)]
    first = next(b for b in batches if b is not None)
    assert list(first.extend_prefix_lens) == [0]
    assert list(first.input_lengths) == [len(shared)]
    assert not branch_rt.loads

    # The checkpoint cut there reached the pool: a third rank resumes at it.
    third = prefill(sibling, sibling_rt, "c", shared + [30, 31, 32])
    assert list(third.extend_prefix_lens) == [len(shared)]


@pytest.mark.parametrize("role", [ROLES.P, ROLES.Fused])
def test_a_junction_cut_publishes_its_checkpoint_to_the_pool(role):
    pool = owner()
    writer, writer_rt = scheduler(), Runtime(pool, 1)
    shared = list(range(8))
    publish(writer, writer_rt, "a", shared + [10, 11, 12, 13])
    checkpoints = pool.stats()["checkpoints"]

    s, runtime = scheduler(role=role), Runtime(pool, 2)
    s.submit_requests(
        [
            _spec("branch", shared + [20, 21, 22, 23]),
            _spec("other", list(range(100, 106))),
        ]
    )
    if role == ROLES.P:
        e = ts.ExecutionEvent()
        for rid in ("branch", "other"):
            e.add_event(ts.PD.BootstrappedEvent(rid))
        s.advance(e)
    plan = runtime.plan(s)
    rows = {
        rid: tokens
        for batch in plan.forward
        for rid, tokens in zip(batch.request_ids, batch.input_lengths)
    }
    # The branch stops where the pool's KV ends, on either role.
    assert rows["branch"] == len(shared)
    _advance(s, "branch", [])
    if "other" in rows:
        _advance(s, "other", [1000])
    # Its next admission publishes the checkpoint there, and the write that
    # carries it brings the pool the State block and its checkpoint alone: the
    # pool holds the KV pages already.
    for _ in range(3):
        runtime.plan(s)
    carried = [
        op for op in runtime.writes if any(len(p[1]) == 4 for p in op.checkpoints)
    ]
    assert carried
    assert pool.stats()["checkpoints"] > checkpoints
    sibling, sibling_rt = scheduler(), Runtime(pool, 3)
    third = prefill(sibling, sibling_rt, "c", shared + [30, 31, 32])
    assert list(third.extend_prefix_lens) == [len(shared)]


def stored_blocks(s):
    return {
        tuple(event.token_ids)
        for event in s.drain_kv_events()
        if event.kind == "BlockStored"
    }


def test_kv_events_report_the_pages_a_rank_holds_on_its_device():
    pool = owner()
    writer, reader = scheduler(kv_events=True), scheduler(kv_events=True)
    writer_rt, reader_rt = Runtime(pool, 1), Runtime(pool, 2)
    prompt = list(range(9))
    publish(writer, writer_rt, "a", prompt)
    assert stored_blocks(writer)

    second = prefill(reader, reader_rt, "b", prompt + [50, 51])
    assert list(second.extend_prefix_lens) == [len(prompt)]
    for _ in range(3):
        reader_rt.plan(reader)
    # The reader's events name what its Device holds: the pages it loaded
    # from the pool and the ones it computed.
    assert stored_blocks(reader) == {(0, 1), (2, 3), (4, 5), (6, 7), (8, 50), (51,)}


def test_an_answer_outside_its_prompt_is_refused():
    pool = owner()
    writer, reader = scheduler(), scheduler()
    writer_rt, reader_rt = Runtime(pool, 1), Runtime(pool, 2)
    prompt = list(range(9))
    publish(writer, writer_rt, "a", prompt)

    reader.submit_requests([_spec("b", prompt + [50, 51])])
    [lookup] = reader.start_shared_l2_lookups()
    request_id, ticket, num_tokens, kv_tokens, tail, pages = reader_rt.answer(lookup)
    # A hit past the prompt, and a KV reach past what the lookup asked for.
    for hit_tokens, reach in ((40, kv_tokens), (num_tokens, 40)):
        with pytest.raises(ValueError, match="outside the prompt"):
            reader.apply_shared_l2_answers(
                [(request_id, ticket, hit_tokens, reach, tail, pages)]
            )
    # Nothing of the answer was taken, and no plan runs the prompt.
    assert reader.host_pool_cached_blocks() == 0
    reader_rt.plan(reader, lookup=False)
    assert "b" not in reader_rt.forwarded


def test_a_cold_pool_leaves_the_request_to_ordinary_admission():
    pool = owner()
    s, runtime = scheduler(), Runtime(pool, 1)
    batch = prefill(s, runtime, "a", list(range(9)))
    assert list(batch.extend_prefix_lens) == [0]
    assert not runtime.loads and not runtime.released
    assert pool.stats()["pinned_blocks"] == 0


def _with_l3(cfg):
    cfg.enable_l3_storage = True
    cfg.storage_config = ts.StorageConfig(
        8,
        32,
        64,
        4,
        32,
        2,
        1_000_000,
        100,
        ts.StoragePressure(8, 1048576, 1000),
        ts.StoragePressure(4, 524288, 500),
    )


def _without_host_tier(cfg):
    cfg.disable_l2_cache = True


def _without_prefix_caching(cfg):
    cfg.disable_prefix_cache = True


def _on_the_decode_role(cfg):
    cfg.role = ROLES.D


def _with_a_snapshot_pool(cfg):
    cfg.num_snapshot_pages = 8
    cfg.max_retracted_requests = 4


def _with_sharded_groups(cfg):
    sharded = ts.CacheGroupConfig(
        group_id="kv",
        block_granularity=GRAIN,
        total_pages=64,
        transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        attn_kind=ts.AttnKind.Full,
        shard_count=2,
    )
    sharded.cache_blocks_per_lcm_block = 2
    cfg.cache_groups = [sharded, cfg.cache_groups[1]]
    cfg.host_group_packing = [2, 1]


def _writing_evictions_back(cfg):
    cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteBack


def _without_write_room(cfg):
    cfg.shared_l2_config = ts.SharedL2Config(
        max_writes_in_flight=0, max_keys_per_write=64
    )


def _with_peer_sharing(cfg):
    cfg.peer_config = ts.PeerConfig(
        name="rank",
        incarnation=1,
        max_leases=8,
        soft_pin_us=1000,
        hard_pin_us=1000,
        lookup_timeout_us=1000,
        min_fetch_tokens=GRAIN,
        landing_tier=ts.PeerLandingTier.Device,
        max_host_landing_blocks=0,
        host_ready_timeout_us=0,
        versioned_shards=False,
    )


@pytest.mark.parametrize(
    "breaks, message",
    [
        (_with_l3, "excludes L3 storage"),
        (_without_host_tier, "requires a Host L2 cache"),
        (_without_prefix_caching, "requires prefix caching"),
        (_on_the_decode_role, "decode role"),
        (_with_a_snapshot_pool, "never retract and take no snapshot pool"),
        (_with_sharded_groups, "requires replicated groups"),
        (_writing_evictions_back, "WriteThrough"),
        (_without_write_room, "positive write limits"),
        (_with_peer_sharing, "peer prefix sharing"),
    ],
)
def test_a_shared_l2_config_refuses_what_its_proxy_cannot_serve(breaks, message):
    cfg = config()
    breaks(cfg)
    with pytest.raises(ValueError, match=message):
        ts.Scheduler(cfg)


def test_a_flush_refused_for_a_write_in_flight_keeps_the_queued_writes():
    written = []
    for flush_while_writing in (False, True):
        pool, s = owner(), scheduler(max_writes_in_flight=1, max_keys_per_write=1)
        rt = Runtime(pool, 1, complete_writes=False)
        publish(s, rt, "a", list(range(9)))
        assert len(rt.writes) == 1
        if flush_while_writing:
            assert not s.can_clear_cache()
            assert not s.clear_cache()
            assert not s.clear_l1_cache()
        for _ in range(40):
            s.advance(rt.finish_writes())
            rt.plan(s)
        written.append(sorted(key for op in rt.writes for key in op.keys))
    assert len(written[0]) > 1 and written[1] == written[0]


def test_a_flush_drops_the_queued_writes_once_none_is_in_flight():
    pool, s = owner(), scheduler(max_writes_in_flight=1, max_keys_per_write=1)
    rt = Runtime(pool, 1, complete_writes=False)
    prefill(s, rt, "a", list(range(9)))
    _finish(s, "a")
    for _ in range(3):
        rt.plan(s)
    assert len(rt.writes) == 1
    queued = s.host_cache_stats().store_candidates_dropped_by_flush
    s.advance(rt.finish_writes())
    assert s.clear_cache()
    assert s.host_cache_stats().store_candidates_dropped_by_flush > queued
    for _ in range(5):
        rt.plan(s)
    assert len(rt.writes) == 1
    assert settled(s, pool)


def test_writes_left_open_pin_no_more_than_their_bound_and_admissions_go_on():
    pool = owner()
    s = scheduler(max_writes_in_flight=2, max_keys_per_write=3, device_pages=24)
    rt = Runtime(pool, 1, complete_writes=False)
    # Six prompts of nine tokens publish six blocks each; the runtime never
    # completes a write.
    for index in range(6):
        prompt = [100 * index + token for token in range(9)]
        batch = prefill(s, rt, f"r{index}", prompt)
        assert list(batch.extend_prefix_lens) == [0]
        _finish(s, f"r{index}")
        for _ in range(3):
            rt.plan(s)
    # Two writes of three blocks are out; every other publication waits in
    # the store queue without a pin, so the Device kept evicting them for the
    # later prompts.
    assert len(rt.writes) == 2 and len(rt.open_writes) == 2
    assert all(len(op.keys) == 3 for op in rt.writes)
    # No request is left: the Device blocks out of reach are the six the two
    # writes read.
    assert s.active_lcm_blocks() == 0
    assert s.available_lcm_blocks() == 23 - 6
    assert s.host_cache_stats().publication_stores.evicted_before_drain == 0
    # Once the runtime catches up, what the Device still holds is written
    # and what it evicted meanwhile is counted.
    for _ in range(40):
        s.advance(rt.finish_writes())
        rt.plan(s)
    stores = s.host_cache_stats().publication_stores
    assert stores.evicted_before_drain > 0
    assert sum(stores.blocks) + stores.evicted_before_drain == 36
    assert settled(s, pool)


def test_a_finish_offers_its_entries_again_and_the_pool_takes_the_new_ones():
    pool = owner()
    s, rt = scheduler(), Runtime(pool, 1, complete_writes=False)
    prompt = list(range(9))
    publish(s, rt, "a", prompt)
    # The prompt end is written while the request decodes.
    [first] = rt.open_writes
    s.advance(rt.finish_writes())
    assert pool.stats()["cached_blocks"] == 6
    for _ in range(3):
        _advance(s, "a", [1000])
    _finish(s, "a")
    rt.plan(s)
    # The finish queues every entry of the request, as for a Host tier of the
    # rank's own, which would skip what it holds. The pool does that: it
    # grants pages for the entries it lacks alone.
    [last] = rt.open_writes
    assert set(first.keys[:4]) < set(last.keys)
    _, pages = rt.write(last)
    granted = [key for key, page in zip(last.keys, pages) if page >= 1]
    assert granted and not set(granted) & set(first.keys)
    assert all(page == -1 for key, page in zip(last.keys, pages) if key in first.keys)
    rt.open_writes = []
    s.advance(_event_write_done(last.op_id))
    rt.plan(s)
    assert settled(s, pool)
    # Only a write in flight has a completion.
    with pytest.raises(ValueError, match="unknown shared L2 write"):
        s.advance(_event_write_done(last.op_id))


def _event_write_done(op_id):
    event = ts.ExecutionEvent()
    event.add_event(ts.Cache.SharedL2WriteDone(op_id))
    return event
