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

"""Engine-free behavior tests for the batch state-isolation experiment."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from test.runtime import run_batch_state_isolation as isolation  # noqa: E402


def completion(*, ids, logprobs, reason="length", details=None):
    return {
        "output_ids": ids,
        "output_token_logprobs": logprobs,
        "finish_reason": reason,
        "finish_details": details,
        "prompt_tokens": 3,
        "completion_tokens": len(ids),
        "cached_tokens": 0,
        "input_token_logprobs": None,
    }


def test_default_matrix_covers_each_required_batch_shape():
    cases = {case.name: case for case in isolation.default_cases()}
    assert tuple(cases) == (
        "single-1",
        "single-2",
        "equal-length",
        "long-short",
        "reversed",
        "shared-prefix-single",
        "shared-prefix",
        "shared-prefix-reversed",
        "four-request",
    )
    assert cases["reversed"].target_index == 1
    assert cases["shared-prefix-reversed"].target_index == 1
    assert len(cases["four-request"].request_names) == 4
    isolation.validate_cases(list(cases.values()), isolation.default_prompt_set())


def test_validate_cases_requires_an_earlier_reference_case():
    with pytest.raises(ValueError, match="earlier case"):
        isolation.validate_cases(
            [isolation.BatchCase("candidate", ("target",), 0, "reference")],
            isolation.default_prompt_set(),
        )


def test_comparison_reports_token_and_prior_logprob_divergences_separately():
    reference = completion(ids=[4, 7, 9], logprobs=[-0.1, -0.2, -0.3])
    candidate = completion(ids=[4, 8, 9], logprobs=[-0.1, -0.25, -0.3])
    compared = isolation.comparison_record(reference, candidate)
    assert compared["status"] == "FAIL"
    assert compared["first_token_difference"] == {
        "position": 1,
        "expected": 7,
        "actual": 8,
    }
    assert compared["first_logprob_difference"] == {
        "position": 1,
        "expected": -0.2,
        "actual": -0.25,
    }
    assert compared["first_logprob_difference_before_token_divergence"] is None


def test_comparison_keeps_finish_metadata_in_the_oracle():
    compared = isolation.comparison_record(
        completion(ids=[4], logprobs=[-0.1], reason="length"),
        completion(ids=[4], logprobs=[-0.1], reason="abort", details={"type": "abort"}),
    )
    assert compared["status"] == "FAIL"
    assert compared["tokens_equal"] and compared["logprobs_equal"]
    assert not compared["finish_equal"]


def test_shared_prefix_batch_cases_compare_with_the_single_request_baseline():
    cases = {case.name: case for case in isolation.default_cases()}
    assert cases["shared-prefix"].reference_case == "shared-prefix-single"
    assert cases["shared-prefix-reversed"].reference_case == "shared-prefix-single"


def test_load_observation_does_not_equate_an_empty_sample_with_no_batch():
    observed = isolation.batch_observation([], expected_requests=4)
    assert observed == {
        "samples": 0,
        "max_running_requests": 0,
        "observed_concurrent_requests": False,
        "expected_requests": 4,
        "conclusion": "inconclusive: no sampled load snapshot showed two running requests",
    }


def test_load_observation_requires_current_in_flight_requests():
    observed = isolation.batch_observation(
        [{"elapsed_s": 0.1, "engines": [[2, 0, 3, 10]], "in_flight": [0]}],
        expected_requests=2,
    )
    assert observed["max_running_requests"] == 2
    assert not observed["observed_concurrent_requests"]


def test_first_pass_engine_arguments_disable_interfering_features(tmp_path):
    args = isolation.parse_args(["--output-dir", str(tmp_path)])
    argv = isolation.engine_args(args, prefix_caching=False)
    assert argv == [
        "--model",
        isolation.DEFAULT_MODEL,
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--seed",
        "42",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "4",
        "--chunked-prefill-size",
        "-1",
        "--disable-prefix-caching",
        "--disable-prefill-graph",
        "--enforce-eager",
        "--disable-kvstore",
        "--enable-output-logprobs",
        "--gpu-memory-utilization",
        "0.55",
    ]
