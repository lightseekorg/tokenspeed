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

"""Randomized closed loops over the scheduler, driven as the event loop does.

Plan k+1 is built before plan k's results land (overlap depth 1); a finish is
decided on the landed token while the next decode is already dispatched;
every planned forward row lands exactly one result (an intermediate chunk an
empty one). Prompts share system prefixes, repeat each other exactly and
continue finished sequences as next turns, so hits land at pages, inside
pages and at finish checkpoints, on a pool small enough to evict. The
scheduler must never raise. The D-role tail hit inside a prompt's last block
of a sliding-window group was found this way.

The fused and P roles also run with a Host L2: under the write-through
policy, which stores what is published, and on the P role under the
write-back policy, which copies what an admission evicts to the Host in the
same plan (without the sliding-window group, which write-back refuses).
Every Host op is acknowledged zero to three plans late, out of order and now
and then twice. Under write-back every write-back must be stream-ordered (no
pinned source); under either policy, once everything is acknowledged no Host
entry is pinned and nothing is in flight.
"""

from __future__ import annotations

import random

import pytest

ts = pytest.importorskip("tokenspeed_scheduler")

P = 8
HISTORY = "full_attention"
STATES = ("linear_attention_0", "linear_attention_1")
ROUNDS = 300


def _config(
    role, pages: int, history_block: int, sliding: bool, host_pages: int, policy
) -> "ts.SchedulerConfig":
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = P
    cfg.num_device_pages = pages
    cfg.num_host_pages = host_pages
    cfg.num_snapshot_pages = 1
    cfg.max_scheduled_tokens = 48
    cfg.max_batch_size = 6
    cfg.enable_l3_storage = False
    # A Host tier states what feeds it; Validate refuses what it cannot run.
    cfg.disable_l2_cache = host_pages == 0
    cfg.host_write_policy = policy
    cfg.disable_prefix_cache = False
    cfg.decode_input_tokens = 1
    cfg.overlap_schedule_depth = 1
    cfg.role = role
    groups = [
        ts.CacheGroupConfig(
            group_id=HISTORY,
            block_granularity=history_block,
            total_pages=pages * (P // history_block),
            attn_kind=ts.AttnKind.Full,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        )
    ]
    if sliding:
        groups.append(
            ts.CacheGroupConfig(
                group_id="sliding",
                block_granularity=history_block,
                total_pages=pages * (P // history_block),
                attn_kind=ts.AttnKind.SlidingWindow,
                sliding_window_tokens=12,
                transfer_policy=ts.CacheTransferPolicy.FullSuffix,
            )
        )
    groups += [
        ts.CacheGroupConfig(
            group_id=group_id,
            block_granularity=0,
            total_pages=pages,
            attn_kind=ts.AttnKind.State,
            transfer_policy=ts.CacheTransferPolicy.LatestSnapshot,
        )
        for group_id in STATES
    ]
    cfg.cache_groups = groups
    return cfg


def _event(kind: str, request_id: str, **fields):
    event = getattr(ts.ForwardEvent, kind)()
    event.request_id = request_id
    for name, value in fields.items():
        setattr(event, name, value)
    return event


class _Loop:
    def __init__(self, seed: int, role, host_pages: int, policy):
        self.rng = random.Random(seed)
        self.role = role
        self.uses_write_back = policy == ts.SchedulerConfig.HostWritePolicy.WriteBack
        self.remote = role == ts.SchedulerConfig.Role.D
        self.prefill_only = role == ts.SchedulerConfig.Role.P
        self.scheduler = ts.Scheduler(
            _config(
                role,
                pages=self.rng.choice([24, 40, 64]),
                history_block=self.rng.choice([P, P // 2]),
                # Drawn either way, so a seed keeps its other draws.
                sliding=self.rng.random() < 0.4 and not self.uses_write_back,
                host_pages=host_pages,
                policy=policy,
            )
        )
        # Unacknowledged cache ops: (the round their ACK is due, the ACK).
        self.acks: list[tuple[int, object]] = []
        self.round = 0
        self.write_back_rows = 0
        self.systems = [
            [self.rng.randrange(2, 500) for _ in range(self.rng.randrange(3, 40))]
            for _ in range(3)
        ]
        self.live: dict[str, dict] = {}
        self.done: list[list[int]] = []
        self.pending: list = []
        self.next_id = 0
        self.next_token = 5000

    def _token(self) -> int:
        self.next_token += 1
        return self.next_token

    def submit(self, count: int) -> None:
        specs = []
        for _ in range(count):
            user = [
                self.rng.randrange(2, 500) for _ in range(self.rng.randrange(1, 30))
            ]
            draw = self.rng.random()
            if draw < 0.15 and self.live:
                tokens = list(self.rng.choice(list(self.live.values()))["prompt"])
            elif draw < 0.45 and self.done:
                tokens = list(self.rng.choice(self.done))
                if self.rng.random() < 0.5:
                    tokens += user
            else:
                tokens = self.rng.choice(self.systems) + user
            max_new = self.rng.randrange(1, 10)
            tokens = tokens[
                : max(1, self.scheduler.max_single_request_tokens() - max_new - 1)
            ]
            spec = ts.RequestSpec()
            spec.request_id = request_id = f"r{self.next_id}"
            self.next_id += 1
            spec.tokens = tokens
            spec.max_new_tokens = max_new
            specs.append(spec)
            self.live[request_id] = {
                "prompt": tokens,
                "max_new": max_new,
                "generated": 0,
                "finished": False,
                "bootstrapped": self.role == ts.SchedulerConfig.Role.Fused,
                "transferring": False,
                "out": [],
            }
        self.scheduler.submit_requests(specs)

    def land(self, plan) -> None:
        """One result per planned row; finishes decided on the landed token."""
        events = ts.ExecutionEvent()
        finishes = []
        for op in plan.forward:
            request_ids = list(op.request_ids)
            num_extends = op.num_extends()
            for i, request_id in enumerate(request_ids):
                state = self.live.get(request_id)
                is_decode = i >= num_extends
                mid_chunk = not is_decode and (
                    op.extend_prefix_lens[i] + op.input_lengths[i]
                    < op.prefill_lengths[i]
                )
                if mid_chunk:
                    events.add_event(_event("ExtendResult", request_id, tokens=[]))
                    continue
                token = self._token()
                events.add_event(_event("ExtendResult", request_id, tokens=[token]))
                if state is None:
                    continue
                state["out"].append(token)
                if state["finished"] or self.prefill_only:
                    continue  # a finished request's delayed row; the P never finishes itself
                state["generated"] += 1
                if state["generated"] >= state["max_new"] or self.rng.random() < 0.03:
                    state["finished"] = True
                    finishes.append(request_id)
                elif is_decode:
                    events.add_event(
                        _event(
                            "UpdateReserveNumTokens",
                            request_id,
                            reserve_num_tokens_in_next_schedule_event=1,
                        )
                    )
        for request_id in finishes:
            kind = "Abort" if self.rng.random() < 0.1 else "Finish"
            events.add_event(_event(kind, request_id))
        self.scheduler.advance(events)

    def peer(self, plan) -> None:
        """The transfer peer: bootstraps waiting prompts; on the D lands the
        plan's remote prefills, on the P acknowledges (or fails) the decodes
        it was handed."""
        events = ts.ExecutionEvent()
        count = 0
        if plan is not None and plan.remote_decode is not None:
            for request_id in plan.remote_decode.request_ids:
                self.live[request_id]["transferring"] = True
        for request_id, state in self.live.items():
            if not state["bootstrapped"] and self.rng.random() < 0.7:
                events.add_event(ts.PD.BootstrappedEvent(request_id))
                state["bootstrapped"] = True
                count += 1
            if state["transferring"] and self.rng.random() < 0.5:
                kind = (
                    ts.PD.FailedEvent
                    if self.rng.random() < 0.1
                    else ts.PD.SucceededEvent
                )
                events.add_event(kind(request_id))
                state["transferring"] = False
                state["finished"] = True
                count += 1
        if plan is not None and plan.remote_prefill is not None:
            for request_id in plan.remote_prefill.request_ids:
                token = self._token()
                events.add_event(ts.PD.RemotePrefillDoneEvent(request_id, token))
                state = self.live[request_id]
                state["out"].append(token)
                state["generated"] += 1
                count += 1
        if count:
            self.scheduler.advance(events)

    def queue_acks(self, plan) -> None:
        """The plan's cache copies, each acknowledged zero to three rounds on."""
        for op in plan.cache:
            if isinstance(op, ts.Cache.WriteBackOp):
                if self.uses_write_back:
                    assert not any(
                        op.source_pinned
                    ), "a write-back pins no Device source"
                self.write_back_rows += sum(len(rows) for rows in op.src_pages)
                done = ts.Cache.WriteBackDoneEvent
            elif isinstance(op, ts.Cache.LoadBackOp):
                done = ts.Cache.LoadBackDoneEvent
            elif isinstance(op, ts.Cache.SnapshotOp):
                done = ts.Cache.SnapshotDoneEvent
            elif isinstance(op, ts.Cache.RestoreOp):
                done = ts.Cache.RestoreDoneEvent
            else:
                continue
            for op_id in op.op_ids:
                ack = done()
                ack.op_id = op_id
                self.acks.append((self.round + self.rng.randrange(4), ack))

    def acknowledge(self, everything: bool) -> None:
        """The due ACKs in a random order, one in twenty of them twice."""
        due = [ack for when, ack in self.acks if everything or when <= self.round]
        self.acks = [
            (when, ack)
            for when, ack in self.acks
            if not (everything or when <= self.round)
        ]
        self.rng.shuffle(due)
        for ack in due:
            for _ in range(2 if self.rng.random() < 0.05 else 1):
                events = ts.ExecutionEvent()
                events.add_event(ack)
                self.scheduler.advance(events)

    def run(self, rounds: int) -> None:
        self.submit(self.rng.randrange(2, 6))
        for self.round in range(rounds):
            if self.rng.random() < 0.3:
                self.submit(self.rng.randrange(1, 4))
            if self.remote or self.prefill_only:
                self.peer(None)
            plan = self.scheduler.next_execution_plan()
            self.queue_acks(plan)
            self.pending.append(plan)
            if self.remote or self.prefill_only:
                self.peer(plan)
            if len(self.pending) > 1:
                self.land(self.pending.pop(0))
            if self.rng.random() < 0.1:
                while self.pending:
                    self.land(self.pending.pop(0))
            for request_id in [r for r, s in self.live.items() if s["finished"]]:
                if self.scheduler.request_token_size(request_id) == -1:
                    state = self.live.pop(request_id)
                    # Every landed token but the last was consumed.
                    sequence = state["prompt"] + state["out"][:-1]
                    if len(self.done) < 20:
                        self.done.append(sequence)
                    else:
                        self.done[self.rng.randrange(20)] = sequence
            self.acknowledge(everything=False)


# Seed 42 on the D role reaches the sliding-window tail hit inside a prompt's
# last block.
@pytest.mark.parametrize("role", ["Fused", "D", "P"])
@pytest.mark.parametrize("seed", [*range(12), 42])
def test_random_closed_loop_never_raises(role: str, seed: int) -> None:
    _Loop(
        seed,
        getattr(ts.SchedulerConfig.Role, role),
        host_pages=0,
        policy=ts.SchedulerConfig.HostWritePolicy.Unspecified,
    ).run(ROUNDS)


# Write-back runs on the P role alone; write-through on both roles that
# compute prompts.
@pytest.mark.parametrize(
    "policy, role",
    [("WriteBack", "P"), ("WriteThrough", "P"), ("WriteThrough", "Fused")],
)
@pytest.mark.parametrize("host_pages", [9, 17, 33])
@pytest.mark.parametrize("seed", range(12))
def test_random_host_acks_leave_nothing_in_flight(
    policy: str, role: str, host_pages: int, seed: int
) -> None:
    loop = _Loop(
        seed,
        getattr(ts.SchedulerConfig.Role, role),
        host_pages=host_pages,
        policy=getattr(ts.SchedulerConfig.HostWritePolicy, policy),
    )
    loop.run(ROUNDS)
    # No new prompt: land what is out and acknowledge everything, round after
    # round, until every request has left the scheduler and a plan has
    # nothing left to run -- the plans that follow finish the live requests
    # and copy and load more (a finish stores at the plan after it),
    # acknowledged in turn.
    for loop.round in range(ROUNDS, 10 * ROUNDS):
        while loop.pending:
            loop.land(loop.pending.pop(0))
        loop.acknowledge(everything=True)
        if loop.remote or loop.prefill_only:
            loop.peer(None)
        plan = loop.scheduler.next_execution_plan()
        loop.queue_acks(plan)
        if loop.remote or loop.prefill_only:
            loop.peer(plan)
        # Finished, or aborted by the scheduler for capacity.
        for request_id in [
            r for r in loop.live if loop.scheduler.request_token_size(r) == -1
        ]:
            del loop.live[request_id]
        if (
            not loop.live
            and not any(op.request_ids for op in plan.forward)
            and not plan.cache
            and plan.remote_decode is None
            and not loop.acks
        ):
            break
        loop.pending.append(plan)
    else:
        pytest.fail("the loop did not drain")
    assert loop.write_back_rows > 0, "the Host was written"
    assert loop.scheduler.host_pool_pinned_blocks() == 0
    # No copy is in flight: a cache flush could go ahead.
    assert loop.scheduler.can_clear_cache()
