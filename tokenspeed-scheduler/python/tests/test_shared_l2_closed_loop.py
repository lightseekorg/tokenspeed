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

"""Randomized closed loops over several ranks sharing one tiny L2 pool.

Each rank is the closed-loop driver of test_closed_loop_random with a shared
L2 pool as its Host L2, and the test plays its runtime: every lookup is
answered before the plan that follows it; pool writes complete late;
releases reach the owner late; load-backs are acknowledged late, out of
order and now and then twice. The ranks draw their prompts from the same
system prefixes and continue each other's finished sequences, so a prefix
one rank published is resumed on another, on a pool small enough to evict
all the time.

No scheduler may raise; no prompt computes what the pool's answer pinned for
it; no pool page is granted to a write while a load-back still reads it; and
once everything is acknowledged, no request is left waiting and neither the
ranks nor the pool hold anything.
"""

from __future__ import annotations

import random

import pytest

ts = pytest.importorskip("tokenspeed_scheduler")

import test_closed_loop_random as closed_loop  # noqa: E402

P = closed_loop.P
ROUNDS = 200


class _Pool:
    """The pool the ranks share, with what must never happen at its pages."""

    def __init__(self, seed: int, blocks: int, max_writes: int, max_write_keys: int):
        self.rng = random.Random(seed)
        self.blocks = blocks
        self.limits = (max_writes, max_write_keys)
        self.chunk = self.rng.choice([P, 2 * P, 3 * P])
        self.pool = None
        self.groups = None
        # (group, pool page) -> load-backs reading it.
        self.reading: dict[tuple[int, int], int] = {}
        self.granted = 0

    def configure(self, cfg) -> None:
        """A rank's settings: the first rank's drawn groups are the pool's,
        and every later rank takes the same ones over its own Device."""
        if self.groups is None:
            self.groups = list(cfg.cache_groups)
            owner = ts.SchedulerConfig()
            owner.prefix_granularity = cfg.prefix_granularity
            owner.cache_groups = self.groups
            owner.host_group_packing = [1] * len(self.groups)
            owner.checkpoint_retention_requests = 8
            self.pool = ts.SharedL2Pool(owner, self.blocks, self.rng.choice([0, 2]))
        else:
            groups = []
            for group in self.groups:
                state = group.attn_kind == ts.AttnKind.State
                kwargs = dict(
                    group_id=group.group_id,
                    block_granularity=group.block_granularity,
                    total_pages=cfg.num_device_pages
                    * (1 if state else P // group.block_granularity),
                    attn_kind=group.attn_kind,
                    transfer_policy=group.transfer_policy,
                )
                if group.attn_kind == ts.AttnKind.SlidingWindow:
                    kwargs["sliding_window_tokens"] = group.sliding_window_tokens
                groups.append(ts.CacheGroupConfig(**kwargs))
            cfg.cache_groups = groups
        cfg.host_group_packing = [1] * len(self.groups)
        cfg.max_scheduled_tokens = self.chunk
        cfg.shared_l2_config = ts.SharedL2Config(
            max_writes=self.limits[0], max_write_keys=self.limits[1]
        )


class _Planned:
    """A rank's scheduler whose every plan passes through the rank's runtime."""

    def __init__(self, rank):
        self._rank = rank

    def __getattr__(self, name):
        return getattr(self._rank.real, name)

    def next_execution_plan(self):
        self._rank.before_plan()
        plan = self._rank.real.next_execution_plan()
        self._rank.after_plan(plan)
        return plan


class _Rank(closed_loop._Loop):
    def __init__(self, seed: int, role, shared: _Pool, client: int):
        super().__init__(
            seed,
            role,
            # The proxy Host tier, past the pool.
            host_pages=shared.blocks + 9,
            policy=ts.SchedulerConfig.HostWritePolicy.WriteThrough,
            kv_shards=1,
            num_snapshot_pages=1,
            max_retracted_requests=0,
            force_retraction_interval=0,
            mixed=False,
            inject_aborts=True,
            shuffle_groups=False,
            configure=shared.configure,
        )
        self.shared, self.pool, self.client = shared, shared.pool, client
        self.real = self.scheduler
        self.scheduler = _Planned(self)
        # The pool client draws from its own stream.
        self.io = random.Random(seed + 3)
        self.commands: list[tuple[int, int]] = []
        self.writes: list[tuple[int, object]] = []
        # For each prompt the pool answered, the tokens it pinned for it.
        self.pinned_for: dict[str, int] = {}
        self.started: set[str] = set()
        # (group, proxy Host block) -> pool page; load op -> the pages it reads.
        self.page_of: dict[tuple[int, int], int] = {}
        self.reads: dict[int, list[tuple[int, int]]] = {}
        self.pool_resumes = 0

    def before_plan(self) -> None:
        # Every lookup is answered before the plan, as the event loop does.
        for lookup in self.real.take_shared_l2_lookups():
            request_id, hashes, tokens, max_tokens = lookup
            result = self.pool.probe(self.client, hashes, tokens, max_tokens)
            # The prompt must resume at least where the pool pinned.
            self.pinned_for[request_id] = result[1]
            for group, block, _, page in self.real.apply_shared_l2_lookups(
                [(request_id, *result)]
            ):
                self.page_of[(group, block)] = page

    def after_plan(self, plan) -> None:
        for op in plan.forward:
            for i in range(op.num_extends()):
                request_id = op.request_ids[i]
                if request_id in self.started:
                    continue
                self.started.add(request_id)
                pinned = self.pinned_for.get(request_id, 0)
                assert op.extend_prefix_lens[i] >= pinned, (
                    f"{request_id} computes from {op.extend_prefix_lens[i]}: "
                    f"the pool held {pinned}"
                )
        for op in plan.cache:
            if isinstance(op, ts.Cache.LoadBackOp):
                for op_id, groups, sources in zip(
                    op.op_ids, op.group_ids, op.src_pages
                ):
                    # Every source is a proxy block the runtime knows.
                    pages = [
                        (g, self.page_of[(g, src)]) for g, src in zip(groups, sources)
                    ]
                    self.reads[op_id] = pages
                    for page in pages:
                        self.shared.reading[page] = self.shared.reading.get(page, 0) + 1
                    self.pool_resumes += 1
            elif isinstance(op, ts.Cache.SharedL2ReleaseOp):
                self.commands.append((self.round + self.io.randrange(3), op.ticket))
            elif isinstance(op, ts.Cache.SharedL2TouchOp):
                self.pool.touch(op.keys)
            elif isinstance(op, ts.Cache.SharedL2WriteOp):
                assert len(op.keys) <= self.shared.limits[1]
                self.writes.append((self.round + self.io.randrange(4), op))
        # Every write not reported yet is one the scheduler has in flight.
        assert len(self.writes) <= self.shared.limits[0]
        self.carry_out(everything=False)

    def carry_out(self, everything: bool) -> None:
        """The releases and write completions that are due."""
        due = [t for when, t in self.commands if everything or when <= self.round]
        self.commands = [
            (when, t)
            for when, t in self.commands
            if not (everything or when <= self.round)
        ]
        for ticket in due:
            self.pool.release(self.client, ticket)
        writes = [op for when, op in self.writes if everything or when <= self.round]
        self.writes = [
            (when, op)
            for when, op in self.writes
            if not (everything or when <= self.round)
        ]
        events = ts.ExecutionEvent()
        for op in writes:
            ticket, pages = self.pool.reserve(self.client, op.keys, op.buckets)
            for key, page in zip(op.keys, pages):
                if page >= 1:
                    assert (
                        key[1],
                        page,
                    ) not in self.shared.reading, (
                        f"pool page {(key[1], page)} is rewritten under a load-back"
                    )
                    self.shared.granted += 1
            assert self.pool.commit(self.client, ticket, op.publications) == (
                ticket != 0
            )
            events.add_event(ts.Cache.SharedL2WriteDone(op.write_id))
        if writes:
            self.real.advance(events)

    def acknowledge(self, everything: bool) -> None:
        landed = [
            ack.op_id
            for when, ack in self.acks
            if (everything or when <= self.round)
            and isinstance(ack, ts.Cache.LoadBackDoneEvent)
        ]
        super().acknowledge(everything)
        for op_id in landed:
            for page in self.reads.pop(op_id):
                self.shared.reading[page] -= 1
                if self.shared.reading[page] == 0:
                    del self.shared.reading[page]


def _run(seed: int, roles) -> dict:
    shared = _Pool(
        seed,
        blocks=random.Random(seed).choice([12, 20, 32]),
        max_writes=2,
        max_write_keys=4,
    )
    ranks = [
        _Rank(seed * 10 + index, role, shared, client=index + 1)
        for index, role in enumerate(roles)
    ]
    # One workload over every rank: the same system prefixes, and each rank
    # continues what any rank finished.
    for rank in ranks[1:]:
        rank.systems, rank.done = ranks[0].systems, ranks[0].done
    for rank in ranks:
        rank.submit(rank.rng.randrange(2, 6))
    for round_ in range(ROUNDS):
        for rank in ranks:
            rank.round = round_
            rank.step(ROUNDS)
    # No new prompt: every command is carried out and every operation
    # acknowledged, round after round, until every request has left its
    # scheduler and a plan has nothing left to run.
    for round_ in range(ROUNDS, 10 * ROUNDS):
        idle = True
        for rank in ranks:
            rank.round = round_
            while rank.pending:
                rank.land(rank.pending.pop(0))
            rank.acknowledge(everything=True)
            if rank.prefill_only:
                rank.peer(None)
            plan = rank.scheduler.next_execution_plan()
            rank.carry_out(everything=True)
            rank.queue_acks(plan)
            if rank.prefill_only:
                rank.peer(plan)
            for request_id in [
                r for r in rank.live if rank.real.request_token_size(r) == -1
            ]:
                del rank.live[request_id]
            if (
                rank.live
                or rank.stopping
                or any(op.request_ids for op in plan.forward)
                or plan.cache
                or plan.remote_decode is not None
                or rank.acks
                or rank.writes
                or rank.commands
            ):
                idle = False
                rank.pending.append(plan)
        if idle:
            break
    else:
        pytest.fail("the ranks did not drain")
    for rank in ranks:
        assert rank.real.host_pool_cached_blocks() == 0, "a proxy entry is left"
        assert rank.real.waiting_size() == 0
        assert rank.real.can_clear_cache()
    assert not shared.reading
    pool = shared.pool.stats()
    assert (pool["pinned_blocks"], pool["reserved_blocks"], pool["lookups"]) == (
        0,
        0,
        0,
    )
    return {
        "pool resumes": sum(rank.pool_resumes for rank in ranks),
        "pool evictions": pool["evicted_blocks"],
        "pages written": shared.granted,
        "aborts": sum(rank.aborts for rank in ranks),
        "pool hits": sum(
            bool(pinned) for rank in ranks for pinned in rank.pinned_for.values()
        ),
    }


@pytest.mark.parametrize(
    "roles",
    [("Fused", "Fused"), ("P", "P", "P"), ("Fused", "P", "Fused")],
    ids="-".join,
)
def test_ranks_sharing_a_tiny_pool_neither_stall_nor_leak(roles) -> None:
    activity: dict[str, int] = {}
    for seed in range(8):
        for name, count in _run(
            seed, [getattr(ts.SchedulerConfig.Role, role) for role in roles]
        ).items():
            activity[name] = activity.get(name, 0) + count
    for name, count in activity.items():
        assert count > 0, f"no {name} across the seeds"
