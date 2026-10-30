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

"""Exercise the compute budget through real scheduler plans and completions."""

import math

import pytest
from tokenspeed_scheduler import (
    PD,
    AttnKind,
    CacheGroupConfig,
    CacheTransferPolicy,
    ExecutionEvent,
    ForwardEvent,
    PrefillComputeConfig,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)


def config(compute, *, mixed, state, replay_window):
    cfg = SchedulerConfig()
    cfg.prefix_granularity = 4
    cfg.max_scheduled_tokens = 32
    cfg.max_batch_size = 8
    cfg.num_device_pages = 512
    cfg.num_snapshot_pages = 1
    cfg.disable_l2_cache = True
    cfg.enable_mixed_prefill_decode = mixed
    cfg.prefill_compute = compute
    groups = [CacheGroupConfig("mla", 4, 512, AttnKind.Full)]
    if state:
        groups.append(CacheGroupConfig("kda", 0, 512, AttnKind.State))
    if replay_window:
        groups.append(
            CacheGroupConfig(
                "swa",
                4,
                512,
                AttnKind.SlidingWindow,
                sliding_window_tokens=replay_window,
                replayable=True,
            )
        )
    cfg.cache_groups = groups
    return cfg


def submit(scheduler, rid, tokens):
    request = RequestSpec()
    request.request_id = rid
    request.tokens = tokens
    scheduler.submit_requests([request])


def complete(scheduler, rid, *, finish):
    done = ForwardEvent.ExtendResult()
    done.request_id = rid
    done.tokens = [123456] if finish else []
    scheduler.advance(ExecutionEvent().add_event(done))
    if finish:
        end = ForwardEvent.Finish()
        end.request_id = rid
        scheduler.advance(ExecutionEvent().add_event(end))


def forward(scheduler):
    plan = scheduler.next_execution_plan()
    assert not plan.aborts
    assert len(plan.forward) == 1
    return plan.forward[0]


@pytest.mark.parametrize("state", [False, True])
def test_context_growth_reduces_chunk_and_preserves_all_tokens(state):
    compute = PrefillComputeConfig(36, 0, 1, 0)
    scheduler = Scheduler(config(compute, mixed=False, state=state, replay_window=0))
    tokens = list(range(32))
    submit(scheduler, "long", tokens)
    cursor = 0
    lengths = []
    while cursor < len(tokens):
        batch = forward(scheduler)
        (q,) = batch.input_lengths
        assert batch.extend_prefix_lens == [cursor]
        assert batch.input_ids == tokens[cursor : cursor + q]
        assert compute.cost(q, cursor) <= compute.budget_ns
        # The next token would exceed the cost or complete the prompt.
        assert (
            cursor + q == len(tokens) or compute.cost(q + 1, cursor) > compute.budget_ns
        )
        cursor += q
        lengths.append(q)
        complete(scheduler, "long", finish=cursor == len(tokens))
    assert lengths[:3] == [8, 3, 2]


def test_budget_is_shared_across_requests_and_resets_each_plan():
    compute = PrefillComputeConfig(20, 1, 0, 5)
    scheduler = Scheduler(config(compute, mixed=False, state=False, replay_window=0))
    submit(scheduler, "a", [1] * 5)
    submit(scheduler, "b", [2] * 5)
    submit(scheduler, "c", [3] * 5)
    batch = forward(scheduler)
    assert batch.request_ids == ["a", "b"]
    assert batch.input_lengths == [5, 5]
    for rid in batch.request_ids:
        complete(scheduler, rid, finish=True)
    assert forward(scheduler).request_ids == ["c"]


@pytest.mark.parametrize("state", [False, True])
def test_first_admission_uses_actual_prefix_hit(state):
    compute = PrefillComputeConfig(136, 0, 1, 0)
    scheduler = Scheduler(config(compute, mixed=False, state=state, replay_window=0))
    prefix = list(range(16))
    submit(scheduler, "seed", prefix)
    assert forward(scheduler).input_lengths == [16]
    complete(scheduler, "seed", finish=True)
    submit(scheduler, "hit", prefix + list(range(100, 116)))
    batch = forward(scheduler)
    assert batch.extend_prefix_lens == [16]
    assert batch.input_lengths == [6]
    assert batch.input_ids == list(range(100, 106))


def test_tiny_budget_makes_progress_but_only_once_per_round():
    compute = PrefillComputeConfig(0.01, 10, 1, 100)
    scheduler = Scheduler(config(compute, mixed=False, state=True, replay_window=0))
    submit(scheduler, "a", [1, 2])
    submit(scheduler, "b", [3, 4])
    seen = []
    for _ in range(4):
        batch = forward(scheduler)
        assert batch.input_lengths == [1]
        (rid,) = batch.request_ids
        seen.append(rid)
        complete(scheduler, rid, finish=seen.count(rid) == 2)
    assert seen == ["a", "a", "b", "b"]


def test_replay_cost_and_final_window_do_not_stall():
    compute = PrefillComputeConfig(1, 1, 0, 0)
    scheduler = Scheduler(config(compute, mixed=False, state=True, replay_window=4))
    prefix = list(range(8))
    submit(scheduler, "seed", prefix)
    for i in range(2):
        assert forward(scheduler).input_lengths == [4]
        complete(scheduler, "seed", finish=i == 1)
    submit(scheduler, "hit", prefix + [8, 9, 10, 11])
    batch = forward(scheduler)
    assert batch.extend_prefix_lens == [4]
    assert batch.extend_replay_lens == [4]
    assert batch.input_lengths == [8]
    assert batch.input_ids == list(range(4, 12))


def test_replay_tokens_are_charged_to_shared_budget():
    compute = PrefillComputeConfig(12, 1, 0, 0)
    scheduler = Scheduler(config(compute, mixed=False, state=True, replay_window=4))
    submit(scheduler, "seed", list(range(8)))
    forward(scheduler)
    complete(scheduler, "seed", finish=True)
    submit(scheduler, "hit", list(range(12)))
    submit(scheduler, "new", list(range(100, 112)))
    batch = forward(scheduler)
    assert batch.request_ids == ["hit", "new"]
    assert batch.input_lengths == [8, 4]
    assert batch.extend_replay_lens == [4, 0]


def test_disabled_budget_retains_token_limited_plan():
    scheduler = Scheduler(config(None, mixed=False, state=False, replay_window=0))
    submit(scheduler, "long", list(range(64)))
    assert forward(scheduler).input_lengths == [32]
    complete(scheduler, "long", finish=False)
    batch = forward(scheduler)
    assert batch.extend_prefix_lens == [32]
    assert batch.input_lengths == [32]


def test_large_time_budget_never_exceeds_hard_token_limit():
    scheduler = Scheduler(
        config(
            PrefillComputeConfig(1e20, 1, 1, 0),
            mixed=False,
            state=True,
            replay_window=0,
        )
    )
    submit(scheduler, "long", list(range(64)))
    assert forward(scheduler).input_lengths == [32]


@pytest.mark.parametrize(
    "values",
    [
        (0, 1, 1, 0),
        (-1, 1, 1, 0),
        (math.nan, 1, 1, 0),
        (math.inf, 1, 1, 0),
        (1, -1, 1, 0),
        (1, 1, -1, 0),
        (1, 1, 1, -1),
        (1, 0, 0, 1),
        (1, math.inf, 1, 0),
        (1, 1, math.nan, 0),
        (1, 1, 1, math.inf),
    ],
)
def test_invalid_calibration_rejected_by_scheduler(values):
    with pytest.raises(ValueError, match="prefill compute budget"):
        Scheduler(
            config(
                PrefillComputeConfig(*values), mixed=False, state=False, replay_window=0
            )
        )


def test_decode_remote_admission_is_not_chunked_by_local_compute_budget():
    cfg = config(
        PrefillComputeConfig(1, 100, 100, 0), mixed=False, state=False, replay_window=0
    )
    cfg.role = SchedulerConfig.Role.D
    # nanobind returns a copy of the vector; assign the configured group back.
    groups = cfg.cache_groups
    groups[0].transfer_policy = CacheTransferPolicy.FullSuffix
    cfg.cache_groups = groups
    scheduler = Scheduler(cfg)
    submit(scheduler, "remote", list(range(64)))
    scheduler.advance(ExecutionEvent().add_event(PD.BootstrappedEvent("remote")))
    plan = scheduler.next_execution_plan()
    assert plan.remote_prefill is not None
    assert plan.remote_prefill.input_lengths == [64]


@pytest.mark.parametrize("mixed", [False, True])
def test_decode_keeps_its_existing_scheduling_priority(mixed):
    compute = PrefillComputeConfig(4, 1, 0, 0)
    scheduler = Scheduler(config(compute, mixed=mixed, state=True, replay_window=0))
    submit(scheduler, "decoding", [1, 2, 3, 4])
    forward(scheduler)
    done = ForwardEvent.ExtendResult()
    done.request_id = "decoding"
    done.tokens = [5]
    scheduler.advance(ExecutionEvent().add_event(done))
    submit(scheduler, "prefill", list(range(100, 116)))
    batch = forward(scheduler)
    assert batch.num_extends() == 1
    assert batch.input_lengths[0] == 4
    assert batch.request_ids[0] == "prefill"
    assert ("decoding" in batch.request_ids) == mixed


def test_checkpoint_in_forward_remains_valid_with_smaller_compute_chunks():
    compute = PrefillComputeConfig(36, 0, 1, 0)
    cfg = config(compute, mixed=False, state=True, replay_window=0)
    cfg.max_scheduled_tokens = 8
    cfg.state_checkpoint_in_forward = True
    scheduler = Scheduler(cfg)
    tokens = list(range(17))
    submit(scheduler, "long", tokens)
    cursor = 0
    while cursor < len(tokens):
        batch = forward(scheduler)
        (q,) = batch.input_lengths
        assert batch.input_ids == tokens[cursor : cursor + q]
        assert compute.cost(q, cursor) <= compute.budget_ns
        (checkpoint,) = batch.extend_state_checkpoint_lens
        assert checkpoint == 0 or (
            0 < checkpoint < q and checkpoint % cfg.prefix_granularity == 0
        )
        cursor += q
        complete(scheduler, "long", finish=cursor == len(tokens))


def test_prefill_role_applies_budget_before_remote_decode_handoff():
    cfg = config(
        PrefillComputeConfig(36, 0, 1, 0), mixed=False, state=False, replay_window=0
    )
    cfg.role = SchedulerConfig.Role.P
    groups = cfg.cache_groups
    groups[0].transfer_policy = CacheTransferPolicy.FullSuffix
    cfg.cache_groups = groups
    scheduler = Scheduler(cfg)
    submit(scheduler, "long", list(range(32)))
    scheduler.advance(ExecutionEvent().add_event(PD.BootstrappedEvent("long")))
    batch = forward(scheduler)
    assert batch.input_lengths == [8]
    complete(scheduler, "long", finish=False)
    batch = forward(scheduler)
    assert batch.extend_prefix_lens == [8]
    assert batch.input_lengths == [3]
