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

"""Acceptance-history budgets through real forecasts and the native planner."""

import pytest
from tokenspeed_scheduler import ConfidenceBudgetPlanner

from tokenspeed.runtime.execution.confidence_history import AcceptanceHistory
from tokenspeed.runtime.utils.server_args import ServerArgs


def expected_length(rates):
    total, prefix = 1.0, 1.0
    for rate in rates:
        prefix *= rate
        total += prefix
    return total


def test_rejections_smooth_the_actual_accepted_length():
    history = AcceptanceHistory(6, observation_weight=0.25)
    history.observe(1, queries=6, accepted=5)
    forecast = history.forecast([0.99] * 5, maximum_generation=1)
    assert expected_length(forecast.rates) == 5
    history.observe(2, queries=6, accepted=2)
    forecast = history.forecast([0.99] * 5, maximum_generation=2)
    assert expected_length(forecast.rates) == pytest.approx(0.75 * 5 + 0.25 * 2)
    assert not forecast.censored


@pytest.mark.parametrize("censored", [False, True])
def test_same_length_distinguishes_rejection_from_budget_limit(censored):
    history = AcceptanceHistory(6, observation_weight=0.25)
    history.observe(1, queries=3 if censored else 6, accepted=3)
    forecast = history.forecast([0.9] * 5, maximum_generation=1)
    planner = ConfidenceBudgetPlanner(6)
    for queries, cost in [(3, 1.0), (4, 1.1), (6, 3.0)]:
        planner.set_baseline(1, queries, cost)
    assert planner.select(1, [forecast.observed_rates]) == 3
    assert planner.select(1, [forecast.rates]) == (4 if censored else 3)


def test_anchor_only_budget_recovers_after_a_previous_rejection():
    history = AcceptanceHistory(6, observation_weight=0.25)
    history.observe(1, queries=6, accepted=1)
    history.observe(2, queries=1, accepted=1)
    forecast = history.forecast([0.9] * 5, maximum_generation=2)
    assert forecast.censored
    assert forecast.rates == pytest.approx([0.9] * 5)
    planner = ConfidenceBudgetPlanner(6)
    planner.set_baseline(1, 1, 1.0)
    planner.set_baseline(1, 6, 1.5)
    assert planner.select(1, [forecast.rates]) == 6


def test_older_generation_forecast_is_not_rewritten_by_later_observations():
    history = AcceptanceHistory(6, observation_weight=0.25)
    history.observe(1, queries=6, accepted=1)
    history.observe(3, queries=6, accepted=6)
    old = history.forecast([0.9] * 5, maximum_generation=2)
    new = history.forecast([0.9] * 5, maximum_generation=3)
    assert old.generation == 1 and expected_length(old.rates) == 1
    assert new.generation == 3 and expected_length(new.rates) > 1


def test_full_window_success_has_no_unobserved_tail():
    history = AcceptanceHistory(6, observation_weight=0.25)
    history.observe(1, queries=6, accepted=6)
    forecast = history.forecast([0.1] * 5, maximum_generation=1)
    assert not forecast.censored
    assert expected_length(forecast.rates) == 6


@pytest.mark.parametrize("policy", ["confidence", "history"])
def test_adaptive_budget_policies_require_supported_drafts(policy):
    args = ServerArgs(
        model="x", speculative_algorithm="DSPARK", speculative_verify_policy=policy
    )
    assert args.speculative_verify_policy == policy
    with pytest.raises(ValueError, match="requires DSPARK"):
        ServerArgs(
            model="x", speculative_algorithm="EAGLE3", speculative_verify_policy=policy
        )
