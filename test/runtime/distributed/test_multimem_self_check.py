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

"""The startup self-check refuses an in-switch reduction that does not hold."""

import pytest
import torch
from tokenspeed_kernel.ops.communication.triton import multimem_probe_payload

from tokenspeed.runtime.distributed.comm_backend import self_check
from tokenspeed.runtime.distributed.comm_backend.auto import Collective, Route
from tokenspeed.runtime.distributed.comm_backend.self_check import (
    MultimemSelfCheckError,
    verify_multimem_all_reduce,
)

WORLD = (0, 1, 2, 3)
ATTN = (0, 1)
HIDDEN = 64


class FakeBackend:
    """A backend whose all-reduce is a rank-ordered fp32 fold of the probe
    slices, optionally perturbed on chosen calls."""

    def __init__(self, *, routes: dict, perturb_calls=()):
        self.routes = routes
        self.perturb_calls = set(perturb_calls)
        self.calls = 0
        self.pinned = []

    def route(self, collective, tensor, group):
        assert collective is Collective.ALL_REDUCE
        if group in self.pinned:
            return Route.ORDERED_FOLD
        return self.routes[group]

    def pin_ordered_fold(self, group):
        self.pinned.append(group)

    def all_reduce(self, tensor, group):
        rows = tensor.shape[0]
        total = torch.zeros(rows, HIDDEN, dtype=torch.float32)
        for member in range(len(group)):
            total += multimem_probe_payload(
                member,
                len(group),
                self_check._PROBE_ROWS,
                HIDDEN,
                "cpu",
                self_check._PROBE_SEED,
            )[:rows].float()
        out = total.to(torch.bfloat16)
        self.calls += 1
        if self.calls in self.perturb_calls:
            out[0, 0] = out[0, 0] + 1
        return out


@pytest.fixture
def single_process(monkeypatch):
    """Stand in for the world collectives: one rank, groups all agree."""
    monkeypatch.setattr(
        self_check, "_world_agrees", lambda verdict, world, device: verdict
    )
    monkeypatch.setattr(
        self_check,
        "_gather_over_world",
        lambda tensor, world: [tensor.clone()] * len(world),
    )


def _verify(backend, **overrides):
    kwargs = dict(
        groups=(("attention TP", ATTN), ("MoE TP-EP", WORLD)),
        world_group=WORLD,
        rank=0,
        hidden_size=HIDDEN,
        device=torch.device("cpu"),
        repetitions=8,
    )
    kwargs.update(overrides)
    return verify_multimem_all_reduce(backend, **kwargs)


def test_a_stable_switch_passes_and_names_the_verified_kinds(single_process):
    backend = FakeBackend(routes={ATTN: Route.MULTIMEM, WORLD: Route.ORDERED_FOLD})
    assert _verify(backend) == [("attention TP", Route.MULTIMEM)]
    # 8 repetitions plus the half-payload check, on the switched kind only.
    assert backend.calls == 9
    assert backend.pinned == []


def test_a_repetition_that_differs_refuses_startup(single_process):
    backend = FakeBackend(
        routes={ATTN: Route.MULTIMEM, WORLD: Route.MULTIMEM}, perturb_calls={3}
    )
    with pytest.raises(
        MultimemSelfCheckError, match="not run-stable: repetition 2 of 8"
    ):
        _verify(backend)


def test_a_row_count_that_changes_the_bits_refuses_startup(single_process):
    backend = FakeBackend(
        routes={ATTN: Route.MULTIMEM, WORLD: Route.MULTIMEM}, perturb_calls={9}
    )
    with pytest.raises(MultimemSelfCheckError, match="not batch-invariant"):
        _verify(backend)


def test_two_groups_of_a_kind_that_disagree_are_pinned_to_the_fold(monkeypatch):
    # The switch's order is a property of the GPU set: two groups of a kind
    # reducing the same payload differently is a topology, not a fault, and
    # the kind stays bitwise on the fold.
    monkeypatch.setattr(
        self_check, "_world_agrees", lambda verdict, world, device: verdict
    )

    def other_group_differs(tensor, world):
        other = tensor.clone()
        other[1, 1] = other[1, 1] + 1
        return [tensor.clone(), tensor.clone(), other, other]

    monkeypatch.setattr(self_check, "_gather_over_world", other_group_differs)
    backend = FakeBackend(routes={ATTN: Route.MULTIMEM, WORLD: Route.MULTIMEM})
    outcome = _verify(backend)
    assert outcome == [
        ("attention TP", Route.ORDERED_FOLD),
        ("MoE TP-EP", Route.ORDERED_FOLD),
    ]
    assert backend.pinned == [ATTN, WORLD]
    assert backend.route(Collective.ALL_REDUCE, None, ATTN) is Route.ORDERED_FOLD


def test_ranks_that_disagree_on_the_route_are_pinned_to_the_fold(monkeypatch):
    monkeypatch.setattr(
        self_check, "_world_agrees", lambda verdict, world, device: False
    )
    monkeypatch.setattr(
        self_check, "_gather_over_world", lambda tensor, world: [tensor] * len(world)
    )
    backend = FakeBackend(routes={ATTN: Route.MULTIMEM, WORLD: Route.MULTIMEM})
    assert _verify(backend) == [
        ("attention TP", Route.ORDERED_FOLD),
        ("MoE TP-EP", Route.ORDERED_FOLD),
    ]
    assert backend.pinned == [ATTN, WORLD]
    assert backend.calls == 0
    # A rank whose own group is off the switch simply skips the kind.
    backend = FakeBackend(routes={ATTN: Route.ORDERED_FOLD, WORLD: Route.ORDERED_FOLD})
    assert _verify(backend) == []
    assert backend.pinned == []


def test_groups_of_one_and_folded_kinds_are_skipped(single_process):
    backend = FakeBackend(routes={(0,): Route.MULTIMEM, WORLD: Route.ORDERED_FOLD})
    assert _verify(backend, groups=(("attention TP", (0,)), ("dense TP", WORLD))) == []
    assert backend.calls == 0


def test_the_same_group_under_two_kinds_is_decided_once(single_process):
    backend = FakeBackend(routes={ATTN: Route.MULTIMEM, WORLD: Route.ORDERED_FOLD})
    outcome = _verify(
        backend,
        groups=(("attention TP", ATTN), ("dense TP", ATTN), ("MoE TP-EP", WORLD)),
    )
    assert outcome == [("attention TP", Route.MULTIMEM), ("dense TP", Route.MULTIMEM)]
    assert backend.calls == 9


def test_the_probe_payload_separates_association_orders():
    # The payload the self-check reduces must expose an order change in
    # software, or it could not expose one in the switch.
    world = 8
    slices = [
        multimem_probe_payload(rank, world, 16, HIDDEN, "cpu", seed=11).float()
        for rank in range(world)
    ]
    forward = sum(slices[1:], slices[0])
    backward = sum(reversed(slices[:-1]), slices[-1])
    differing = (forward.to(torch.bfloat16) != backward.to(torch.bfloat16)).float()
    assert differing.mean() > 0.25
    # Every rank derives the same assignment: exactly one +B and one -B per
    # element, so the exact sum is small.
    stacked = torch.stack(slices)
    assert torch.equal((stacked.abs() >= 16).sum(0), torch.full((16, HIDDEN), 2))
