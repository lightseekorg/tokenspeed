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

"""Engine-free behavior tests for fixed-sample probability calibration."""

import json
import math
from dataclasses import replace
from test.runtime.sampling import run_sampling_calibration as calibration

import pytest


@pytest.mark.parametrize(
    "samples,low,high,power_one,power_two",
    [
        (1000, 64, 140, 0.00193242627066, 0.0308788531435),
        (10000, 884, 1120, 0.265735939961718, 0.993753589816),
        (40000, 3767, 4236, 0.995869607852, 1.0),
    ],
)
def test_binomial_critical_counts_and_reference_power(
    samples, low, high, power_one, power_two
):
    distribution = calibration.BinomialDistribution(samples, 0.1)
    tail = 0.001 / 20
    assert distribution.critical_counts(tail) == (low, high)
    assert distribution.at_most(low) <= tail < distribution.at_most(low + 1)
    assert distribution.at_least(high) <= tail < distribution.at_least(high - 1)
    for probability, expected in ((0.11, power_one), (0.12, power_two)):
        alternative = calibration.BinomialDistribution(samples, probability)
        actual = alternative.at_most(low) + alternative.at_least(high)
        assert actual == pytest.approx(expected, abs=2e-12)


def test_binomial_small_distribution_matches_exact_probabilities():
    distribution = calibration.BinomialDistribution(4, 0.5)
    assert distribution.mass == [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]
    assert distribution.at_most(1) == 5 / 16
    assert distribution.at_least(3) == 5 / 16
    assert distribution.at_most(-1) == distribution.at_least(5) == 0
    assert distribution.at_most(4) == distribution.at_least(0) == 1


def test_gsm_plan_assigns_all_ten_thousand_events_before_generation():
    prompts = [f"Question {i}" for i in range(100)]
    plans = calibration.plan_requests(
        prompts, mode="gsm8k", samples=10000, tokens_per_request=128
    )
    assert len(plans) == 79
    assert [plan.prompt for plan in plans] == prompts[:79]
    assert [plan.index for plan in plans] == list(range(79))
    assert [plan.tokens for plan in plans[:-1]] == [128] * 78
    assert plans[-1].tokens == 16
    assert sum(plan.tokens for plan in plans) == 10000


def test_prefill_plan_repeats_one_prompt_with_no_decode_outputs():
    plans = calibration.plan_requests(
        ["A fixed prompt"], mode="prefill", samples=1000, tokens_per_request=1
    )
    assert len(plans) == 1000
    assert all(plan.prompt == "A fixed prompt" and plan.tokens == 1 for plan in plans)
    params = calibration.sampling_parameters(plans[0])
    assert params["temperature"] == params["top_p"] == 1
    assert params["top_k"] == -1 and params["min_p"] == 0
    assert params["max_new_tokens"] == 1
    assert params["ignore_eos"] and params["no_stop_trim"]
    assert params["skip_special_tokens"] is False
    assert "seed" not in params and "sampling_seed" not in params


def response(*, tokens, probabilities, prefixes):
    return calibration.SamplingResponse(
        output_ids=tokens,
        output_token_logprobs=[math.log(value) for value in probabilities],
        output_token_cumulative_probs=prefixes,
        finish_reason="length",
        finish_details={"type": "length"},
    )


def collector(samples):
    return calibration.QuantileCalibration(
        samples=samples, bins=10, alpha=0.001, client_rng_seed=1234
    )


def test_large_probability_atom_spans_several_mass_bins():
    count = 1000
    result = collector(count)
    added = result.add_response(
        calibration.RequestPlan(0, "x", count),
        response(
            tokens=[4] * count, probabilities=[0.75] * count, prefixes=[0.0] * count
        ),
    )
    assert {observation.bin_index for observation in added} == set(range(8))
    assert all(0 <= observation.quantile < 0.75 for observation in added)
    assert sum(result.phase_counts["prefill"]) == 1
    assert sum(result.phase_counts["decode"]) == count - 1


def test_equal_probability_tokens_use_their_reported_prefix_mass():
    result = collector(2)
    added = result.add_response(
        calibration.RequestPlan(0, "x", 2),
        response(tokens=[10, 11], probabilities=[0.1, 0.1], prefixes=[0.3, 0.4]),
    )
    assert [value.bin_index for value in added] == [3, 4]
    assert added[0].token_id == 10 and added[1].token_id == 11


def test_mass_at_the_upper_endpoint_stays_in_last_bin():
    result = collector(1)
    added = result.add_response(
        calibration.RequestPlan(0, "x", 1),
        response(tokens=[8], probabilities=[1e-20], prefixes=[1.0]),
    )
    assert added[0].bin_index == 9
    assert added[0].quantile < 1


@pytest.mark.parametrize(
    "change",
    [
        {"output_token_cumulative_probs": []},
        {"output_token_logprobs": []},
        {"output_ids": []},
        {"output_ids": [True]},
        {"output_token_logprobs": [float("nan")]},
        {"output_token_logprobs": [0.01]},
        {"output_token_logprobs": [-1000.0]},
        {"output_token_cumulative_probs": [float("inf")]},
        {"output_token_cumulative_probs": [-0.01]},
        {"output_token_cumulative_probs": [1.01]},
        {"output_token_cumulative_probs": [0.99]},
        {"finish_reason": "abort", "finish_details": {"error": "failed"}},
    ],
)
def test_invalid_metadata_never_contributes_a_partial_sample(change):
    result = collector(1)
    valid = response(tokens=[3], probabilities=[0.1], prefixes=[0.0])
    with pytest.raises(calibration.CalibrationError):
        result.add_response(
            calibration.RequestPlan(0, "x", 1), replace(valid, **change)
        )
    assert result.observations == []
    assert result.counts == [0] * 10


def test_out_of_order_response_cannot_change_the_fixed_sample():
    result = collector(1)
    valid = response(tokens=[3], probabilities=[0.1], prefixes=[0.0])
    with pytest.raises(calibration.CalibrationError, match="predeclared request order"):
        result.add_response(calibration.RequestPlan(1, "x", 1), valid)
    with pytest.raises(calibration.CalibrationError, match="exactly 1 samples"):
        result.report()


def observations_for_counts(counts):
    tokens, probabilities, prefixes = [], [], []
    for bucket, count in enumerate(counts):
        tokens.extend([bucket] * count)
        probabilities.extend([0.001] * count)
        prefixes.extend([(bucket + 0.25) / 10] * count)
    return response(tokens=tokens, probabilities=probabilities, prefixes=prefixes)


def test_report_applies_inclusive_familywise_threshold_and_keeps_phase_descriptive():
    counts = [1120] + [987] * 8 + [984]
    assert sum(counts) == 10000
    result = collector(10000)
    result.add_response(
        calibration.RequestPlan(0, "x", 10000), observations_for_counts(counts)
    )
    report = result.report()
    assert report["status"] == "FAIL"
    assert report["familywise_pvalue"] <= 0.001
    assert report["bin_results"][0]["count"] == 1120
    assert report["bin_results"][0]["rejected"]
    assert report["phase_counts_are_descriptive_only"] is True
    assert report["reference_power"][1]["power"] == pytest.approx(0.993753589816)


def test_report_accepts_counts_just_inside_the_predeclared_boundary():
    counts = [1119] + [987] * 8 + [985]
    result = collector(10000)
    result.add_response(
        calibration.RequestPlan(0, "x", 10000), observations_for_counts(counts)
    )
    assert result.report()["status"] == "PASS"


def test_dataset_template_preserves_line_order_and_literal_braces(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"question":"First {x}"}\n{"question":"Second"}\n')
    prompts = calibration.read_dataset(
        path, question_field="question", prompt_template="Question: {question}\nAnswer:"
    )
    assert prompts == ["Question: First {x}\nAnswer:", "Question: Second\nAnswer:"]
    with pytest.raises(ValueError, match="contain"):
        calibration.read_dataset(
            path, question_field="question", prompt_template="ignored"
        )


def test_insufficient_dataset_is_an_error_not_repeated_questions():
    with pytest.raises(ValueError, match="Need 2 dataset prompts"):
        calibration.plan_requests(
            ["one"], mode="gsm8k", samples=3, tokens_per_request=2
        )


def test_invalid_experiment_config_preserves_an_invalid_report(tmp_path):
    config = tmp_path / "engine.json"
    config.write_text('["--model", "/path/to/model", "--enable-output-logprobs"]')
    directory = tmp_path / "results"
    directory.mkdir()
    args = calibration.parse_args(
        [
            "--mode",
            "prefill",
            "--run-name",
            "invalid-config",
            "--engine-config",
            str(config),
            "--model",
            "/path/to/model",
            "--num-engines",
            "1",
            "--routing",
            "round_robin",
            "--startup-timeout",
            "60",
            "--no-tokenizer-trust-remote-code",
            "--no-add-special-tokens",
            "--samples",
            "1000",
            "--bins",
            "10",
            "--alpha",
            "0.001",
            "--client-rng-seed",
            "42",
            "--batch-size",
            "16",
            "--prompt",
            "A fixed prompt",
            "--output-dir",
            str(directory),
        ]
    )
    report = calibration.run(args)
    assert report["status"] == "INVALID"
    assert report["validated_samples"] == 0
    assert "--enable-output-cumulative-probs" in report["error"]
    assert json.loads((directory / "report.json").read_text()) == report


def test_raw_snapshot_preserves_invalid_values_without_mutating_the_response():
    response = {"output_token_logprobs": [float("nan"), float("inf"), -1.0]}
    snapshot = calibration.json_snapshot(response)
    assert snapshot["output_token_logprobs"] == ["nan", "inf", -1.0]
    assert math.isnan(response["output_token_logprobs"][0])
