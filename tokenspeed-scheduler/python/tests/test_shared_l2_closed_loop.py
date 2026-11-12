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
L2 pool as its Host L2, and the test plays its runtime: lookups are answered
zero to a few rounds late, some are given up and some of those answered
after all; pool writes complete late, and some fail before or after their
reservation; releases and abandons reach the owner late; load-backs are
acknowledged late, out of order and now and then twice. The ranks draw their
prompts from the same system prefixes and continue each other's finished
sequences, so a prefix one rank published is resumed on another, on a pool
small enough to evict all the time.

No scheduler may raise; no prompt is admitted before its lookup is settled,
or computes what the pool's answer pinned for it; no pool page is granted to
a write while a load-back still reads it; and once everything is
acknowledged, no request is left waiting and neither the ranks nor the pool
hold anything.
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
        self.answers: list[tuple[int, tuple]] = []
        self.give_ups: list[tuple[int, int]] = []
        self.commands: list[tuple[int, object]] = []
        self.writes: list[tuple[int, object]] = []
        # The requests whose lookup is out, by lookup id, and for each one
        # answered the tokens the pool pinned for it.
        self.asked: dict[int, str] = {}
        self.pinned_for: dict[str, int] = {}
        self.started: set[str] = set()
        # (group, proxy Host block) -> pool page; load op -> the pages it reads.
        self.page_of: dict[tuple[int, int], int] = {}
        self.reads: dict[int, list[tuple[int, int]]] = {}
        self.pool_resumes = 0
        self.failed_writes = 0
        # The runtime's lease on the owner's pins, as it last reported it.
        self.lease_lost = False
        self.lease_losses = 0

    def before_plan(self) -> None:
        for lookup in self.real.shared_l2_lookups():
            self.asked[lookup[0]] = lookup[1]
            if self.io.random() < 0.15:
                # The runtime will give this one up; its probe may still
                # reach the owner, before or after.
                self.give_ups.append((self.round + self.io.randrange(1, 3), lookup[0]))
                if self.io.random() < 0.6:
                    self.answers.append((self.round + self.io.randrange(5), lookup))
            else:
                self.answers.append((self.round + self.io.randrange(3), lookup))
        self.settle(everything=False)
        # The lease goes and comes back, with loads in flight or not; the
        # runtime reports it after the round's answers, as the event loop
        # does.
        if self.io.random() < (0.3 if self.lease_lost else 0.04):
            self.lease_lost = not self.lease_lost
            self.lease_losses += self.lease_lost
        self.real.set_shared_l2_lease_lost(self.lease_lost)
        if self.lease_lost:
            # The waiting prompts gave their hits up: they may compute.
            self.pinned_for = {
                request_id: pinned
                for request_id, pinned in self.pinned_for.items()
                if request_id in self.started
            }

    def settle(self, everything: bool) -> None:
        """The give-ups and answers that are due."""
        due = [i for when, i in self.give_ups if everything or when <= self.round]
        self.give_ups = [
            (when, i)
            for when, i in self.give_ups
            if not (everything or when <= self.round)
        ]
        self.real.abandon_shared_l2_lookups(due)
        for lookup_id in due:
            self.asked.pop(lookup_id, None)
        answers = [
            answer for when, answer in self.answers if everything or when <= self.round
        ]
        self.answers = [
            (when, answer)
            for when, answer in self.answers
            if not (everything or when <= self.round)
        ]
        for lookup_id, request_id, hashes, tokens, max_tokens in answers:
            result = self.pool.probe(self.client, lookup_id, hashes, tokens, max_tokens)
            waited = self.real.shared_l2_stats().pending_lookups
            created = self.real.apply_shared_l2_lookups([(lookup_id, *result)])
            if self.real.shared_l2_stats().pending_lookups < waited:
                # The scheduler was waiting for it: the prompt must resume
                # at least where the pool pinned.
                self.pinned_for[request_id] = result[1]
                del self.asked[lookup_id]
            else:
                assert not created
            for group, block, _, page in created:
                self.page_of[(group, block)] = page

    def after_plan(self, plan) -> None:
        for op in plan.forward:
            for i in range(op.num_extends()):
                request_id = op.request_ids[i]
                if request_id in self.started:
                    continue
                self.started.add(request_id)
                assert (
                    request_id not in self.asked.values()
                ), f"{request_id} is admitted before its lookup is settled"
                pinned = self.pinned_for.get(request_id, 0)
                assert op.extend_prefix_lens[i] >= pinned, (
                    f"{request_id} computes from {op.extend_prefix_lens[i]}: "
                    f"the pool held {pinned}"
                )
        for op in plan.cache:
            if isinstance(op, ts.Cache.LoadBackOp):
                assert (
                    not self.lease_lost
                ), "a load starts from the pool without the lease"
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
                self.commands.append(
                    (self.round + self.io.randrange(3), ("release", op.ticket))
                )
            elif isinstance(op, ts.Cache.SharedL2AbandonOp):
                self.commands.append(
                    (
                        self.round + self.io.randrange(3),
                        ("abandon_lookup", op.lookup_id),
                    )
                )
            elif isinstance(op, ts.Cache.SharedL2TouchOp):
                self.pool.touch(op.keys)
            elif isinstance(op, ts.Cache.SharedL2WriteOp):
                assert len(op.keys) <= self.shared.limits[1]
                self.writes.append((self.round + self.io.randrange(4), op))
        assert self.real.shared_l2_stats().writes_in_flight <= self.shared.limits[0]
        self.carry_out(everything=False)

    def carry_out(self, everything: bool) -> None:
        """The owner calls and write completions that are due."""
        due = [c for when, c in self.commands if everything or when <= self.round]
        self.commands = [
            (when, c)
            for when, c in self.commands
            if not (everything or when <= self.round)
        ]
        for name, argument in due:
            getattr(self.pool, name)(self.client, argument)
        writes = [op for when, op in self.writes if everything or when <= self.round]
        self.writes = [
            (when, op)
            for when, op in self.writes
            if not (everything or when <= self.round)
        ]
        events = ts.ExecutionEvent()
        for op in writes:
            draw = self.io.random()
            if draw < 0.08:
                # Given up before the reservation arrived; it may arrive late.
                self.pool.abandon_write(self.client, op.write_id)
                if self.io.random() < 0.5:
                    assert (
                        self.pool.reserve(
                            self.client, op.write_id, op.keys, op.buckets
                        )[0]
                        == 0
                    )
            elif draw < 0.16:
                # Given up after it, and its commit arrives late.
                ticket, _ = self.pool.reserve(
                    self.client, op.write_id, op.keys, op.buckets
                )
                self.pool.abandon_write(self.client, op.write_id)
                assert not self.pool.commit(self.client, ticket, op.publications)
            else:
                ticket, pages = self.pool.reserve(
                    self.client, op.write_id, op.keys, op.buckets
                )
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
            self.failed_writes += draw < 0.16
            events.add_event(ts.Cache.SharedL2WriteDone(op.write_id, draw >= 0.16))
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
    # No new prompt: every lookup is settled, every command carried out and
    # every operation acknowledged, round after round, until every request
    # has left its scheduler and a plan has nothing left to run.
    for round_ in range(ROUNDS, 10 * ROUNDS):
        idle = True
        for rank in ranks:
            rank.round = round_
            while rank.pending:
                rank.land(rank.pending.pop(0))
            rank.acknowledge(everything=True)
            rank.settle(everything=True)
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
                or rank.answers
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
        stats = rank.real.shared_l2_stats()
        assert (stats.pending_lookups, stats.tickets, stats.writes_in_flight) == (
            0,
            0,
            0,
        )
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
        "abandoned lookups": sum(
            rank.real.shared_l2_stats().abandoned_lookups for rank in ranks
        ),
        "failed writes": sum(
            rank.real.shared_l2_stats().failed_writes for rank in ranks
        ),
        "aborts": sum(rank.aborts for rank in ranks),
        "junction or pool hits": sum(
            rank.real.shared_l2_stats().hits for rank in ranks
        ),
        "lease losses": sum(rank.lease_losses for rank in ranks),
        "hits given up for the lease": sum(
            rank.real.shared_l2_stats().lease_expired_hits for rank in ranks
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
