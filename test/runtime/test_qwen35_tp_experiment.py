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

"""Check benchmark aggregation independently of the inference implementation."""

import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(scripts))
from qwen35_tp_experiment import summarize_wave


def test_concurrent_throughput_uses_wave_duration():
    requests = [
        dict(output_ids=list(range(32)), ttft_ms=100, tpot_ms=20),
        dict(output_ids=list(range(64)), ttft_ms=200, tpot_ms=30),
    ]
    result = summarize_wave(requests, 4.0)
    assert result["output_tokens"] == 96
    assert result["output_tokens_per_second"] == 24
    assert result["ttft_ms"] == 150
    assert result["tpot_ms"] == 25


@pytest.mark.parametrize("requests,elapsed", [([], 1), ([{}], 0), ([{}], -1)])
def test_incomplete_measurements_are_rejected(requests, elapsed):
    with pytest.raises(ValueError):
        summarize_wave(requests, elapsed)


def test_logit_comparison_checks_values_and_greedy_choice():
    import torch
    from qwen35_tp_compare import compare_logits

    baseline = torch.tensor([[1.0, 2.0, 3.0]])
    result = compare_logits(baseline + 0.001, baseline)
    assert result["passed"]
    assert result["greedy_equal"]
    assert not compare_logits(-baseline, baseline)["passed"]


def test_logit_comparison_rejects_nan_and_shape_mismatch():
    import torch
    from qwen35_tp_compare import compare_logits

    for actual in (torch.tensor([[float("nan")]]), torch.ones(2)):
        with pytest.raises(ValueError):
            compare_logits(actual, torch.ones(1, 1))


def test_profile_intervals_do_not_double_count_overlap():
    from qwen35_tp_profile_summary import duration, merge_intervals

    assert merge_intervals([(3, 5), (0, 2), (1, 4), (7, 8)]) == [(0, 5), (7, 8)]
    compute = [(0, 3), (5, 7)]
    communication = [(2, 6)]
    overlap = (
        duration(compute) + duration(communication) - duration(compute + communication)
    )
    assert duration(compute + communication) == 7
    assert overlap == 2
    assert duration(compute + communication) - duration(compute) == 2


def test_probability_diagnostic_is_invariant_to_uniform_logit_shift():
    import torch
    from qwen35_tp_compare import compare_logits

    expected = torch.tensor([[1.0, 2.0, 3.0]])
    result = compare_logits(expected + 10, expected)
    assert not result["passed"]  # The original absolute-logit gate stays enforced.
    assert result["greedy_equal"]
    assert abs(result["kl_reference_to_parallel"]) < 1e-6
