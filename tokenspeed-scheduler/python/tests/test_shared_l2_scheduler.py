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

"""Prefill schedulers sharing one L2 pool as their Host L2: the test plays
the runtime, answering lookups and commands from a real SharedL2Pool and
completing load-backs from the proxy Host tier (no bytes move)."""

from __future__ import annotations

import pytest
import tokenspeed_scheduler as ts
from conftest import _advance, _finish, _spec

GRAIN = 2


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
    role=ts.SchedulerConfig.Role.Fused,
    max_writes=2,
    max_write_keys=64,
):
    cfg = ts.SchedulerConfig()
    cfg.role = role
    if role == ts.SchedulerConfig.Role.P:
        cfg.decode_allocation_policy = (
            ts.SchedulerConfig.DecodeAllocationPolicy.Independent
        )
    cfg.enable_kv_cache_events = kv_events
    cfg.prefix_granularity = GRAIN
    cfg.num_snapshot_pages = 1
    cfg.num_device_pages = 64
    # The proxy Host tier: twice the pool.
    cfg.num_host_pages = 129
    cfg.disable_l2_cache = False
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.cache_groups = groups(64)
    cfg.shared_l2_config = ts.SharedL2Config(
        max_writes=max_writes, max_write_keys=max_write_keys
    )
    return cfg


def scheduler(**kwargs):
    return ts.Scheduler(config(**kwargs))


def owner():
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = GRAIN
    cfg.cache_groups = groups(64)
    return ts.SharedL2Pool(cfg, 64, 0)


class Runtime:
    """Answers one scheduler's shared L2 lookups and commands from the pool,
    as the event loop and the load executor do."""

    def __init__(self, pool, client, *, complete_loads=True, complete_writes=True):
        self.pool, self.client = pool, client
        self.complete_loads = complete_loads
        self.complete_writes = complete_writes
        self.lookups, self.loads, self.writes, self.touched = [], [], [], []
        self.forwarded = []
        self.pages = {}
        self.open_loads = []
        self.open_writes = []

    def lookup(self, s):
        lookups = s.shared_l2_lookups()
        self.lookups.extend(lookups)
        results = []
        for rid, hashes, tokens, max_tokens in lookups:
            ticket, num_tokens, kv_tokens, tail, slots = self.pool.probe(
                self.client, list(hashes), tokens, max_tokens
            )
            results.append((rid, ticket, num_tokens, kv_tokens, tail, list(slots)))
        for group, block, page in s.apply_shared_l2_lookups(results):
            self.pages[(group, block)] = page

    def plan(self, s, *, lookup=True):
        if lookup:
            self.lookup(s)
        plan = s.next_execution_plan()
        for batch in plan.forward:
            self.forwarded.extend(batch.request_ids)
        e = ts.ExecutionEvent()
        for op in plan.cache:
            if isinstance(op, ts.Cache.LoadBackOp):
                for op_id, groups_, sources, _ in zip(
                    op.op_ids, op.group_ids, op.src_pages, op.dst_pages
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
                self.pool.release(self.client, op.ticket)
            elif isinstance(op, ts.Cache.SharedL2WriteOp):
                self.writes.append(op)
                ticket, _ = self.pool.reserve(self.client, list(op.keys))
                self.pool.commit(
                    self.client,
                    ticket,
                    list(op.publications),
                )
                if self.complete_writes:
                    e.add_event(ts.Cache.SharedL2WriteDone(op.op_id))
                else:
                    self.open_writes.append(op.op_id)
            elif isinstance(op, ts.Cache.SharedL2TouchOp):
                self.touched.extend(op.keys)
                self.pool.touch(list(op.keys))
        if self.complete_loads:
            self.finish_loads(e)
        s.advance(e)
        return plan

    def finish_writes(self):
        event = ts.ExecutionEvent()
        for op_id in self.open_writes:
            event.add_event(ts.Cache.SharedL2WriteDone(op_id))
        self.open_writes = []
        return event

    def finish_loads(self, e=None):
        event = e if e is not None else ts.ExecutionEvent()
        for op_id in self.open_loads:
            event.add_event(ts.Cache.LoadBackDoneEvent(op_id))
        self.open_loads = []
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
    prefill(writer, runtime, rid, tokens)
    for _ in range(3):
        runtime.plan(writer)


def test_a_rank_resumes_from_the_pool_where_another_rank_published():
    pool = owner()
    writer, reader = scheduler(), scheduler()
    writer_rt, reader_rt = Runtime(pool, 1), Runtime(pool, 2)
    prompt = list(range(9))

    first = prefill(writer, writer_rt, "a", prompt)
    assert list(first.extend_prefix_lens) == [0]
    for _ in range(3):
        writer_rt.plan(writer)
    assert writer_rt.writes and any(op.publications for op in writer_rt.writes)
    assert pool.stats()["reserved_blocks"] == 0

    second = prefill(reader, reader_rt, "b", prompt + [50, 51])
    # The reader resumes at the writer's prompt end: its pages, its tail and
    # the State checkpoint there all load from the pool, one load-back.
    assert list(second.extend_prefix_lens) == [len(prompt)]
    [load] = reader_rt.loads
    assert {group for group, _ in load} == {0, 1}
    assert len({page for _, page in load}) == len(load)
    reader_rt.plan(reader)
    assert pool.stats()["pinned_blocks"] == 0


def test_a_waiting_prompt_is_admitted_only_after_its_lookup():
    pool = owner()
    writer, reader = scheduler(), scheduler()
    writer_rt, reader_rt = Runtime(pool, 1), Runtime(pool, 2)
    prompt = list(range(9))
    publish(writer, writer_rt, "a", prompt)

    reader.submit_requests([_spec("b", prompt + [50, 51])])
    reader_rt.plan(reader, lookup=False)
    assert "b" not in reader_rt.forwarded
    [batch] = [b for b in reader_rt.plan(reader).forward if "b" in b.request_ids]
    assert list(batch.extend_prefix_lens) == [len(prompt)]
    # One lookup per prompt.
    reader_rt.plan(reader)
    assert [lookup[0] for lookup in reader_rt.lookups] == ["b"]


def test_the_pool_pins_a_hit_until_its_load_completes():
    pool = owner()
    writer, reader = scheduler(), scheduler()
    writer_rt = Runtime(pool, 1)
    reader_rt = Runtime(pool, 2, complete_loads=False)
    prompt = list(range(9))
    publish(writer, writer_rt, "a", prompt)

    prefill(reader, reader_rt, "b", prompt + [50, 51])
    assert reader_rt.open_loads
    reader_rt.plan(reader)
    assert pool.stats()["pinned_blocks"] > 0
    reader.advance(reader_rt.finish_loads())
    reader_rt.plan(reader)
    assert pool.stats()["pinned_blocks"] == 0


def test_a_request_gone_before_admission_releases_its_lookup():
    pool = owner()
    writer, reader = scheduler(), scheduler()
    writer_rt, reader_rt = Runtime(pool, 1), Runtime(pool, 2)
    prompt = list(range(9))
    publish(writer, writer_rt, "a", prompt)

    reader.submit_requests([_spec("b", prompt + [50, 51])])
    reader_rt.lookup(reader)
    assert pool.stats()["pinned_blocks"] > 0
    abort = ts.ForwardEvent.Abort()
    abort.request_id = "b"
    e = ts.ExecutionEvent()
    e.add_event(abort)
    reader.advance(e)
    for _ in range(2):
        reader_rt.plan(reader)
    assert "b" not in reader_rt.forwarded and not reader_rt.loads
    assert pool.stats()["pinned_blocks"] == 0


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
    # Admitted together, each loads the shared pages into its own blocks
    # (the Device has no checkpoint there yet), as from Host L2.
    first, second = reader_rt.loads
    assert sorted(first) == sorted(second)
    assert pool.stats()["pinned_blocks"] == 0


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
    assert not runtime.loads and runtime.touched


def test_blocks_loaded_from_the_pool_are_not_written_back():
    pool = owner()
    writer, reader = scheduler(), scheduler()
    writer_rt, reader_rt = Runtime(pool, 1), Runtime(pool, 2)
    prompt = list(range(9))
    publish(writer, writer_rt, "a", prompt)
    written = {tuple(key) for op in writer_rt.writes for key in op.keys}

    publish(reader, reader_rt, "b", prompt + [50, 51, 52, 53])
    assert reader_rt.loads
    rewritten = {tuple(key) for op in reader_rt.writes for key in op.keys}
    assert rewritten and not rewritten & written


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

    # The checkpoint cut there reached the pool: a third rank resumes at it.
    third = prefill(sibling, sibling_rt, "c", shared + [30, 31, 32])
    assert list(third.extend_prefix_lens) == [len(shared)]


def test_a_junction_cut_leaves_the_prefill_roles_round_to_the_prompts_behind_it():
    pool = owner()
    writer, writer_rt = scheduler(), Runtime(pool, 1)
    shared = list(range(8))
    publish(writer, writer_rt, "a", shared + [10, 11, 12, 13])

    roles = ts.SchedulerConfig.Role
    for client, (role, joins) in enumerate(
        ((roles.P, True), (roles.Fused, False)), start=2
    ):
        s, runtime = scheduler(role=role), Runtime(pool, client)
        s.submit_requests(
            [
                _spec("branch", shared + [20, 21, 22, 23]),
                _spec("other", list(range(100, 106))),
            ]
        )
        if role == roles.P:
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
        # The branch stops where the pool's KV ends, so its checkpoint lands
        # there; on the prefill role the prompt behind it fills the round.
        assert rows["branch"] == len(shared)
        assert ("other" in rows) == joins


def stored_blocks(s):
    return {
        tuple(event.token_ids)
        for event in s.drain_kv_events()
        if event.kind == "BlockStored"
    }


def test_kv_events_report_only_the_ranks_own_publications():
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
    # The reader reports its own publication; the loaded tail stays private.
    assert stored_blocks(reader) == {(51,)}


def test_a_cold_pool_leaves_the_request_to_ordinary_admission():
    pool = owner()
    s, runtime = scheduler(), Runtime(pool, 1)
    batch = prefill(s, runtime, "a", list(range(9)))
    assert list(batch.extend_prefix_lens) == [0]
    assert not runtime.loads
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


def _on_the_decode_role(cfg):
    cfg.role = ts.SchedulerConfig.Role.D


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


@pytest.mark.parametrize(
    "breaks, message",
    [
        (_with_l3, "exclusive"),
        (_without_host_tier, "proxy Host tier"),
        (_on_the_decode_role, "decode role"),
        (_with_a_snapshot_pool, "do not retract"),
        (_with_sharded_groups, "replicated groups"),
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
        pool, s = owner(), scheduler(max_writes=1, max_write_keys=1)
        rt = Runtime(pool, 1, complete_writes=False)
        publish(s, rt, "a", list(range(9)))
        assert len(rt.writes) == 1
        if flush_while_writing:
            assert not s.clear_cache()
        for _ in range(40):
            s.advance(rt.finish_writes())
            rt.plan(s)
        written.append(sorted(tuple(key) for op in rt.writes for key in op.keys))
    assert len(written[0]) > 1 and written[1] == written[0]


def test_a_flush_drops_the_queued_writes_once_none_is_in_flight():
    pool, s = owner(), scheduler(max_writes=1, max_write_keys=1)
    rt = Runtime(pool, 1, complete_writes=False)
    prefill(s, rt, "a", list(range(9)))
    _finish(s, "a")
    for _ in range(3):
        rt.plan(s)
    assert len(rt.writes) == 1
    s.advance(rt.finish_writes())
    assert s.clear_cache()
    for _ in range(5):
        rt.plan(s)
    assert len(rt.writes) == 1
