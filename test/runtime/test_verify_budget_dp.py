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


import math

import pytest

from tokenspeed.runtime.execution.drafter.verify_budget import VerifyBudgetPolicy


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


def test_distributed_budget_rejects_unbalanced_inputs():
    policy = VerifyBudgetPolicy(
        enabled=True, min_batch_size=1, verify_width=3, min_gain=0.02
    )
    with pytest.raises(ValueError, match="equal local batches"):
        policy.select_distributed(
            ([[0.0, 0.0]], []),
            graph_batch_size=1,
            native_step_ms=3.0,
            adaptive_step_ms={},
            policy_ms=0.0,
        )
