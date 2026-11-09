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

"""Exercise asynchronous confidence feedback through real CUDA state."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.execution.verify_layout import ConfidenceFeedback

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA feedback")


def _owners(width, *, budget_policy):
    states = RuntimeStates(8, 16, width, device="cuda")
    states.init_draft_confidence(width - 1)
    feedback = ConfidenceFeedback(width, 4, 4, budget_policy=budget_policy)
    for queries in range(1, width + 1):
        feedback.planner.set_baseline(1, queries, 0.8 + queries * 0.2)
    return states, feedback


def _round(states, feedback, key, slot, rates, *, prefill, accepted, queries=None):
    layout = (
        ForwardOutputLayout(1, 1, 0, states.future_input_map.shape[1])
        if prefill
        else ForwardOutputLayout(0, 0, 1, states.future_input_map.shape[1])
    )
    slots = torch.tensor([slot], device="cuda")
    if prefill:
        states.reset_states(slots, torch.zeros(1, dtype=torch.int32, device="cuda"))
    feedback.prepare([key], [slot], layout)
    states.publish_draft_confidence(
        slots, torch.tensor([rates], device="cuda"), group=None
    )
    queries = 0 if prefill else len(rates) + 1 if queries is None else queries
    feedback.record(
        states.draft_confidence[slots],
        states.draft_confidence_generation[slots],
        states.previous_draft_confidence[slots],
        torch.tensor([queries], dtype=torch.int32, device="cuda"),
        torch.tensor([accepted], dtype=torch.int32, device="cuda"),
        None,
        None,
        None,
    )
    torch.cuda.synchronize()
    feedback.poll()


def test_missing_scores_do_not_assume_perfect_acceptance():
    _, feedback = _owners(8, budget_policy="confidence")
    assert feedback.select(1, ["new"]) == 2
    assert feedback.summary()["fallback_fraction"] == 1


def test_request_scores_survive_unrelated_prefill_batches():
    states, feedback = _owners(3, budget_policy="confidence")
    _round(states, feedback, "decode", 1, [0.1, 0.1], prefill=True, accepted=0)
    _round(states, feedback, "decode", 1, [0.1, 0.1], prefill=False, accepted=1)
    _round(states, feedback, "prefill", 2, [0.9, 0.9], prefill=True, accepted=0)
    feedback.prepare(["decode"], [1], ForwardOutputLayout(0, 0, 1, 3))
    assert feedback.select(1, ["decode"]) == 1
    assert feedback.summary()["fallback_fraction"] == 0


def test_stale_scores_cannot_substitute_for_generation_n_minus_two():
    states, feedback = _owners(3, budget_policy="confidence")
    _round(states, feedback, "request", 1, [0.1, 0.1], prefill=True, accepted=0)
    _round(states, feedback, "request", 1, [0.1, 0.1], prefill=False, accepted=1)
    feedback.prepare(["request"], [1], ForwardOutputLayout(0, 0, 1, 3))
    assert feedback.select(1, ["request"]) == 1
    # Launch another draft without a feedback sample, as a full feedback
    # queue does. The next budget must not reuse the previous generation.
    states.publish_draft_confidence(
        torch.tensor([1], device="cuda"),
        torch.tensor([[0.9, 0.9]], device="cuda"),
        group=None,
    )
    feedback.prepare(["request"], [1], ForwardOutputLayout(0, 0, 1, 3))
    assert feedback.select(1, ["request"]) == 1
    states.publish_draft_confidence(
        torch.tensor([1], device="cuda"),
        torch.tensor([[0.9, 0.9]], device="cuda"),
        group=None,
    )
    feedback.prepare(["request"], [1], ForwardOutputLayout(0, 0, 1, 3))
    assert feedback.select(1, ["request"]) == 2
    assert feedback.summary()["fallback_fraction"] == pytest.approx(1 / 3)


def test_budget_uses_n_minus_two_without_waiting_for_n_minus_one_feedback():
    states, feedback = _owners(3, budget_policy="confidence")
    _round(states, feedback, "request", 1, [0.1, 0.1], prefill=True, accepted=0)
    feedback.prepare(["request"], [1], ForwardOutputLayout(0, 0, 1, 3))
    states.publish_draft_confidence(
        torch.tensor([1], device="cuda"),
        torch.tensor([[0.9, 0.9]], device="cuda"),
        group=None,
    )
    # Generation 2 has not returned to the CPU. Its predecessor was copied
    # when produced, so generation 3 can still select its exact N-2 scores.
    feedback.prepare(["request"], [1], ForwardOutputLayout(0, 0, 1, 3))
    assert feedback.select(1, ["request"]) == 1
    assert feedback.summary()["fallback_fraction"] == 0


def test_device_generation_mismatch_is_rejected():
    states, feedback = _owners(3, budget_policy="confidence")
    _round(states, feedback, "request", 1, [0.1, 0.1], prefill=True, accepted=0)
    feedback.prepare(["request"], [1], ForwardOutputLayout(0, 0, 1, 3))
    slots = torch.tensor([1], device="cuda")
    states.publish_draft_confidence(
        slots, torch.tensor([[0.1, 0.1]], device="cuda"), group=None
    )
    # Supply scores with a device generation ahead of the scheduled draft.
    feedback.record(
        states.draft_confidence[slots],
        states.draft_confidence_generation[slots] + 1,
        states.previous_draft_confidence[slots],
        torch.tensor([3], dtype=torch.int32, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        None,
        None,
        None,
    )
    torch.cuda.synchronize()
    feedback.poll()
    feedback.prepare(["request"], [1], ForwardOutputLayout(0, 0, 1, 3))
    states.publish_draft_confidence(
        slots, torch.tensor([[0.1, 0.1]], device="cuda"), group=None
    )
    feedback.prepare(["request"], [1], ForwardOutputLayout(0, 0, 1, 3))
    assert feedback.select(1, ["request"]) == 2
    assert feedback.summary()["rejected_generations"] == 1
    assert feedback.summary()["fallback_fraction"] == 1


def test_recompute_and_slot_reuse_discard_old_request_scores():
    states, feedback = _owners(3, budget_policy="confidence")
    _round(states, feedback, "old", 1, [0.1, 0.1], prefill=True, accepted=0)
    _round(states, feedback, "old", 1, [0.1, 0.1], prefill=False, accepted=1)
    _round(states, feedback, "new", 1, [0.9, 0.9], prefill=True, accepted=0)
    _round(states, feedback, "new", 1, [0.9, 0.9], prefill=False, accepted=3)
    _round(states, feedback, "new", 1, [0.2, 0.2], prefill=True, accepted=0)
    feedback.prepare(["new"], [1], ForwardOutputLayout(0, 0, 1, 3))
    feedback.select(1, ["new"])
    assert feedback.summary()["fallback_fraction"] == 1
    assert feedback.summary()["rejected_generations"] == 0


def test_restored_device_generation_rebases_a_new_host_owner():
    states, feedback = _owners(3, budget_policy="confidence")
    _round(states, feedback, "restored", 1, [0.1, 0.1], prefill=True, accepted=0)
    _round(states, feedback, "restored", 1, [0.1, 0.1], prefill=False, accepted=1)
    image = torch.empty(states.slot_state_bytes(), dtype=torch.uint8, device="cuda")
    states.export_slot_state(1, image, None, request_id="restored")
    _round(states, feedback, "replacement", 1, [0.9, 0.9], prefill=True, accepted=0)
    states.import_slot_state(2, image, None, request_id="restored")
    for _ in range(2):
        _round(states, feedback, "restored", 2, [0.1, 0.1], prefill=False, accepted=1)
    feedback.prepare(["restored"], [2], ForwardOutputLayout(0, 0, 1, 3))
    assert feedback.select(1, ["restored"]) == 1
    assert feedback.summary()["rejected_generations"] == 0
    assert feedback.summary()["fallback_fraction"] == 0


def test_confidence_error_censors_positions_after_first_rejection():
    states, feedback = _owners(4, budget_policy="confidence")
    _round(states, feedback, "request", 1, [0.5] * 3, prefill=True, accepted=0)
    _round(states, feedback, "request", 1, [0.9] * 3, prefill=False, accepted=2)
    summary = feedback.summary()
    assert summary["predicted_length"] == 1.875
    assert summary["accepted_length"] == 2
    assert summary["confidence_mae"] == 0.5
    assert feedback.confidence_observations == 2


@pytest.mark.parametrize("budget_policy,expected", [("confidence", 4), ("history", 1)])
def test_actual_rejection_controls_history_budget(budget_policy, expected):
    states, feedback = _owners(4, budget_policy=budget_policy)
    _round(states, feedback, "request", 1, [0.99] * 3, prefill=True, accepted=0)
    _round(states, feedback, "request", 1, [0.99] * 3, prefill=False, accepted=1)
    feedback.prepare(["request"], [1], ForwardOutputLayout(0, 0, 1, 4))
    assert feedback.select(1, ["request"]) == expected
    if budget_policy == "history":
        assert feedback.summary()["history_fraction"] == 1
        assert feedback.summary()["history_generation_lag"] == 2


@pytest.mark.parametrize("large_graph_ms,expected", [(1.6, 4), (10.0, 2)])
def test_full_budget_grows_only_when_predicted_gain_pays_for_cost(
    large_graph_ms, expected
):
    states, feedback = _owners(4, budget_policy="history")
    feedback.planner.set_baseline(1, 3, large_graph_ms)
    feedback.planner.set_baseline(1, 4, large_graph_ms)
    _round(states, feedback, "request", 1, [0.99] * 3, prefill=True, accepted=0)
    _round(
        states, feedback, "request", 1, [0.99] * 3, prefill=False, accepted=2, queries=2
    )
    feedback.prepare(["request"], [1], ForwardOutputLayout(0, 0, 1, 4))
    assert feedback.select(1, ["request"]) == expected
    assert feedback.summary()["history_censored_fraction"] == 1
    assert feedback.summary()["history_growth_rounds"] == int(expected > 2)


def test_history_does_not_use_acceptance_newer_than_n_minus_two():
    states, feedback = _owners(4, budget_policy="history")
    _round(states, feedback, "request", 1, [0.99] * 3, prefill=True, accepted=0)
    _round(states, feedback, "request", 1, [0.99] * 3, prefill=False, accepted=1)
    _round(states, feedback, "request", 1, [0.99] * 3, prefill=False, accepted=4)
    # Generation 2's acceptance is already copied, but this budget may use
    # only generation 1. Its actual rejection must still win over high scores.
    assert feedback.select(1, ["request"]) == 1


@pytest.mark.parametrize("recompute", [False, True])
def test_history_is_reset_when_request_lifetime_changes(recompute):
    states, feedback = _owners(4, budget_policy="history")
    _round(states, feedback, "old", 1, [0.99] * 3, prefill=True, accepted=0)
    _round(states, feedback, "old", 1, [0.99] * 3, prefill=False, accepted=1)
    feedback.prepare(["old"], [1], ForwardOutputLayout(0, 0, 1, 4))
    assert feedback.select(1, ["old"]) == 1
    key = "old" if recompute else "new"
    _round(states, feedback, key, 1, [0.99] * 3, prefill=True, accepted=0)
    _round(states, feedback, key, 1, [0.99] * 3, prefill=False, accepted=4)
    feedback.prepare([key], [1], ForwardOutputLayout(0, 0, 1, 4))
    assert feedback.select(1, [key]) == 4
    assert feedback.summary()["rejected_generations"] == 0
