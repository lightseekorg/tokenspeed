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

"""The native frontend keeps diagnostics aligned while joining output chunks."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.entrypoints.wire_driver import Completion


def _chunk(ids, logprobs, cumulative_probs, reason, reasoning_tokens=0):
    return {
        "output_ids": ids,
        "output_token_logprobs": logprobs,
        "output_token_cumulative_probs": cumulative_probs,
        "finish_reason": reason,
        "finish_details": None if reason is None else {"type": reason},
        "prompt_tokens": 3,
        "completion_tokens": 3,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": 0,
        "input_token_logprobs": None,
        "spec_verify_ct": None,
        "accept_draft_tokens": None,
    }


def test_completion_joins_diagnostics_with_their_tokens_and_logprobs():
    completion = Completion()
    completion.add(_chunk([4, 9], [-0.2, -2.0], [0.8, 0.99], None, reasoning_tokens=2))
    completion.add(_chunk([2], [-0.4], [0.9], "stop", reasoning_tokens=5))
    assert completion.output_ids == [4, 9, 2]
    assert completion.output_token_logprobs == [-0.2, -2.0, -0.4]
    assert completion.output_token_cumulative_probs == [0.8, 0.99, 0.9]
    assert completion.reasoning_tokens == 5
    assert completion.finish_reason == "stop"


def test_empty_finish_chunk_does_not_repeat_diagnostics():
    completion = Completion()
    completion.add(_chunk([4], [-0.2], [0.8], None))
    completion.add(_chunk([], [], [], "length"))
    assert completion.output_ids == [4]
    assert completion.output_token_logprobs == [-0.2]
    assert completion.output_token_cumulative_probs == [0.8]
    assert completion.finish_reason == "length"


def test_completion_without_diagnostics_keeps_the_column_empty():
    completion = Completion()
    completion.add(_chunk([4], [-0.2], [], "length"))
    assert completion.output_token_cumulative_probs == []


@pytest.mark.parametrize("logprobs,probs", [([-0.2], [0.8, 0.9]), ([], [0.8])])
def test_completion_rejects_misaligned_diagnostics_before_adding_tokens(
    logprobs, probs
):
    completion = Completion()
    with pytest.raises(ValueError, match="must match"):
        completion.add(_chunk([4], logprobs, probs, "length"))
    assert completion.output_ids == []
    assert completion.output_token_logprobs == []
    assert completion.output_token_cumulative_probs == []


def _execution_snapshot(rounds=1, *, admission=7):
    return dict(
        version=1,
        instance_id="fixture",
        role="fused",
        dp_rank=0,
        physical_rank=0,
        request_id="r",
        admission_id=admission,
        origin="serving",
        assigned_rounds=rounds,
        completed_rounds=rounds,
        through_round=rounds,
        missing_rounds=0,
        committed_tokens=rounds,
        evidence_valid=True,
        pre_stop_bins=[
            dict(
                path="decode_graph",
                verify_width=8,
                accepted_before_stop=4,
                force_single=False,
                count=rounds,
            )
        ],
        bins=[
            dict(
                phase="decode",
                path="decode_graph",
                verify_width=8,
                kept_prefix=1,
                count=rounds,
            )
        ],
    )


def test_execution_snapshots_replace_cumulative_counts_and_survive_empty_finish():
    completion = Completion()
    for rounds in (1, 2):
        chunk = _chunk([4], [], [], None)
        chunk["execution_observations"] = _execution_snapshot(rounds)
        completion.add(chunk)
    completion.add(_chunk([], [], [], "length"))
    assert completion.execution_observations["completed_rounds"] == 2
    assert completion.execution_observations["bins"][0]["verify_width"] == 8
    assert completion.execution_observations["bins"][0]["count"] == 2


@pytest.mark.parametrize("change", ["identity", "count", "width", "version"])
def test_bad_execution_snapshot_does_not_mutate_completion(change):
    completion = Completion()
    chunk = _chunk([4], [], [], None)
    chunk["execution_observations"] = _execution_snapshot(2)
    completion.add(chunk)
    bad = _execution_snapshot(2)
    if change == "identity":
        bad["admission_id"] = 8
    elif change == "count":
        bad = _execution_snapshot(1)
    elif change == "width":
        bad["bins"][0]["verify_width"] = 0
    else:
        bad["version"] = 99
    chunk["execution_observations"] = bad
    with pytest.raises(ValueError):
        completion.add(chunk)
    assert completion.output_ids == [4]
    assert completion.execution_observations == _execution_snapshot(2)


def test_explicit_budget_omission_keeps_ordinary_tokens_and_invalidates_evidence():
    completion = Completion()
    chunk = _chunk([4], [], [], None)
    chunk["execution_observations"] = _execution_snapshot(1)
    completion.add(chunk)
    omitted = _execution_snapshot(2)
    omitted.update(
        evidence_valid=False, histograms_omitted=True, bins=[], pre_stop_bins=[]
    )
    chunk = _chunk([5], [], [], "length")
    chunk["execution_observations"] = omitted
    completion.add(chunk)
    assert completion.output_ids == [4, 5]
    assert completion.finish_reason == "length"
    assert completion.execution_observations == omitted
    assert not completion.execution_observations["evidence_valid"]
