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

"""Verification budget decisions use native cost and actual batch occupancy."""

import itertools
import math

import pytest

from tokenspeed.runtime.execution.drafter.verify_budget import VerifyBudgetPolicy


def policy(*, enabled=True, min_bs=2, width=6):
    return VerifyBudgetPolicy(enabled, min_bs, width, 0.02)


@pytest.mark.parametrize(
    "enabled,actual,graph,expected",
    [
        (False, 32, 32, False),
        (True, 1, 2, False),
        (True, 2, 4, False),
        (True, 2, None, False),
        (True, 0, 0, False),
        (True, 2, 2, True),
        (True, 3, 3, True),
    ],
)
def test_admission(enabled, actual, graph, expected):
    assert (
        policy(enabled=enabled).admits(batch_size=actual, graph_batch_size=graph)
        is expected
    )


def choose(p, scores, *, native=10.0, overhead=0.0, costs=None):
    return p.select(
        scores,
        graph_batch_size=len(scores),
        native_step_ms=native,
        adaptive_step_ms={} if costs is None else costs,
        policy_ms=overhead,
    )


def test_native_cost_is_not_adaptive_full_graph_cost():
    scores = [[0.0] * 5] * 2
    assert choose(policy(), scores, costs={4: 4.0, 12: 100.0}) == 4
    assert choose(policy(), scores, native=1.0, costs={4: 4.0, 12: 100.0}) == 12
    assert choose(policy(), scores, overhead=10.0, costs={4: 4.0}) == 12


def test_missing_costs_and_unknown_feedback_use_native():
    assert choose(policy(), [[0.0] * 5] * 2) == 12
    assert choose(policy(), [[math.inf] * 5] * 2, costs={4: 1.0}) == 12
    assert choose(policy(), [[0.0] * 5] * 2, native=math.nan, costs={4: 1.0}) == 12


def test_disabled_policy_does_not_inspect_scores_or_costs():
    assert choose(policy(enabled=False), [[], []], costs={-1: -1.0}) == 12
    assert choose(policy(min_bs=32), [[], []], costs={-1: -1.0}) == 12


def test_generic_width_and_stable_extreme_scores():
    p = policy(width=3)
    assert choose(p, [[-1000.0, -1000.0]] * 2, costs={2: 1.0}) == 2
    assert choose(p, [[1000.0, 1000.0]] * 2, costs={2: 9.0}) == 6


@pytest.mark.parametrize("costs", [{1: 1.0}, {13: 1.0}, {4: 0.0}, {4: math.nan}])
def test_invalid_cost_measurements(costs):
    with pytest.raises(ValueError):
        choose(policy(), [[0.0] * 5] * 2, costs=costs)


def test_distributed_budget_uses_all_rank_rewards():
    policy = VerifyBudgetPolicy(
        enabled=True, min_batch_size=1, verify_width=3, min_gain=0.0
    )
    costs = dict(
        graph_batch_size=1,
        native_step_ms=3.0,
        adaptive_step_ms={1: 1.1, 2: 2.1},
        policy_ms=0.0,
    )
    high = [[20.0, 20.0]]
    low = [[-20.0, -20.0]]
    assert policy.select_distributed((high, [[math.nan, 0.0]]), **costs) == 3
    assert policy.select(high, **costs) == 3
    assert policy.select(low, **costs) == 1
    assert policy.select_distributed((high, low), **costs) == 1
    assert policy.select_distributed((low, high), **costs) == 1


def test_distributed_budget_preserves_single_rank_selection():
    policy = VerifyBudgetPolicy(
        enabled=True, min_batch_size=1, verify_width=3, min_gain=0.02
    )
    scores = [[1.0, -0.5], [2.0, 0.0]]
    costs = dict(
        graph_batch_size=2,
        native_step_ms=5.0,
        adaptive_step_ms={2: 2.0, 4: 3.0},
        policy_ms=0.1,
    )
    expected = policy.select(scores, **costs)
    assert policy.select_distributed((scores,), **costs) == expected
    assert policy.select_distributed((scores, scores), **costs) == expected


@pytest.mark.parametrize("sizes", [(8, 7), (7, 7), (8, 6, 4, 2, 0, 0, 0, 0)])
def test_distributed_budget_accepts_padding_and_idle_ranks(sizes):
    policy = VerifyBudgetPolicy(
        enabled=True, min_batch_size=4, verify_width=6, min_gain=0.02
    )
    scores = [[[-20.0] * 5 for _ in range(size)] for size in sizes]
    assert (
        policy.select_distributed(
            scores,
            graph_batch_size=8,
            native_step_ms=8.0,
            adaptive_step_ms={8: 2.0, 16: 3.0},
            policy_ms=0.0,
        )
        == 8
    )


def test_distributed_budget_rejects_undersized_graph():
    with pytest.raises(ValueError, match="cover every rank"):
        policy().select_distributed(
            ([[0.0] * 5] * 3, []),
            graph_batch_size=2,
            native_step_ms=3.0,
            adaptive_step_ms={},
            policy_ms=0.0,
        )


@pytest.mark.parametrize("sizes", [(2, 1), (1, 0), (2, 2), (0, 0)])
def test_distributed_budget_matches_exhaustive_prefix_rewards(sizes):
    policy = VerifyBudgetPolicy(
        enabled=True, min_batch_size=1, verify_width=3, min_gain=0.02
    )
    ranks = [
        [[float(rank - row), float(row - rank - 1)] for row in range(size)]
        for rank, size in enumerate(sizes)
    ]
    physical_bs, full = 2, 6
    costs = {2: 0.8, 3: 1.0, 4: 1.4, 5: 2.0}
    native_ms, overhead = 2.5, 0.1

    def reward(capacity):
        total = 0.0
        for rows in ranks:
            prefixes = []
            for row in rows:
                p0, p1 = (1 / (1 + math.exp(-score)) for score in row)
                prefixes.append((0.0, p0, p0 + p0 * p1))
            # Padding keeps an anchor; only real rows can earn draft rewards.
            best = max(
                sum(prefixes[i][n] for i, n in enumerate(counts))
                for counts in itertools.product(range(3), repeat=len(rows))
                if sum(counts) <= capacity - physical_bs
            )
            total += len(rows) + best
        return total

    expected = full
    best_rate = reward(full) / native_ms * (1 + policy.min_gain)
    for capacity, cost in sorted(costs.items(), reverse=True):
        rate = reward(capacity) / (cost + overhead)
        if rate > best_rate:
            expected, best_rate = capacity, rate
    assert (
        policy.select_distributed(
            ranks,
            graph_batch_size=physical_bs,
            native_step_ms=native_ms,
            adaptive_step_ms=costs,
            policy_ms=overhead,
        )
        == expected
    )
