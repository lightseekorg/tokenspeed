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

"""Negative tests for the draft-graph oracle; no checkpoint or GPU needed."""

from __future__ import annotations

import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from test.runtime.draft_graph_validation import (  # noqa: E402
    compare_payloads,
    compare_proposals,
    validate_trace_counts,
)


def _call(path="graph", *, token=7, gap=1.0, key="same-input"):
    return {
        "bs": 1,
        "num_extends": 1,
        "accept": [1],
        "first_step": {"path": path, "input_key": key},
        "draft": [[token, token]],
        "margin": [[gap, gap]],
    }


def test_eager_controls_cannot_hide_all_broken_replays():
    graph = [_call("eager") for _ in range(95)] + [_call(token=8) for _ in range(5)]
    eager = [_call("eager") for _ in range(100)]
    errors, summary = compare_proposals(graph, eager, 1.0, 0.05)
    assert sum("replay proposal differs" in error for error in errors) == 5
    assert "5 replay rows" in summary


def test_eager_only_trace_proves_no_graph_correctness():
    errors, _ = compare_proposals([_call("eager")], [_call("eager")], 1.0, 0.05)
    assert any("no actual replay rows" in error for error in errors)


@pytest.mark.parametrize(
    "mutation", ["input", "path", "rows", "columns", "nan", "outlier", "accept"]
)
def test_bad_trace_fails_closed(mutation):
    graph, eager = _call(), _call("eager")
    if mutation == "input":
        graph["first_step"]["input_key"] = "different-input"
    elif mutation == "path":
        del graph["first_step"]["path"]
    elif mutation == "rows":
        graph["draft"] = []
    elif mutation == "columns":
        graph["margin"][0] = []
    elif mutation == "nan":
        graph["margin"][0][0] = float("nan")
    elif mutation == "outlier":
        graph["margin"][0][0] = 10.0
    else:
        graph["accept"] = [2]
    errors, _ = compare_proposals([graph], [eager], 1.0, 0.05)
    assert errors


def test_valid_replay_passes_and_percentage_override_is_rejected():
    assert not compare_proposals([_call()], [_call("eager")], 1.0, 0.05)[0]
    with pytest.raises(ValueError, match="min_match must be 1.0"):
        compare_proposals([_call()], [_call("eager")], 0.95, 0.05)


def _payload():
    return {
        "inputs": {
            name: torch.ones(2, 3)
            for name in ("input_ids", "positions", "hidden", "gather_ids", "frontier")
        },
        "outputs": {
            name: torch.ones(2, 3) for name in ("logits", "latent_kv", "index_k")
        },
    }


@pytest.mark.parametrize("name", ["logits", "latent_kv", "index_k"])
def test_tensor_corruption_fails_even_when_proposals_match(name):
    graph, eager = _payload(), _payload()
    graph["outputs"][name][1, 2] = 10.0
    errors = compare_payloads(
        graph, eager, logit_atol=0.05, logit_rtol=0.01, kv_atol=0.004, kv_rtol=0.02
    )
    assert any(name in error for error in errors)


def test_missing_and_nonfinite_tensor_evidence_fails():
    graph, eager = _payload(), _payload()
    graph["outputs"]["logits"][0, 0] = float("nan")
    del graph["inputs"]["frontier"]
    assert compare_payloads(
        graph, eager, logit_atol=0.05, logit_rtol=0.01, kv_atol=0.004, kv_rtol=0.02
    )


def test_payload_accepts_identical_inputs_and_small_output_rounding():
    eager = _payload()
    graph = copy.deepcopy(eager)
    graph["outputs"]["logits"] += 0.001
    assert not compare_payloads(
        graph, eager, logit_atol=0.05, logit_rtol=0.01, kv_atol=0.004, kv_rtol=0.02
    )
    graph["inputs"]["hidden"][0, 0] += 0.001
    assert compare_payloads(
        graph, eager, logit_atol=0.05, logit_rtol=0.01, kv_atol=0.004, kv_rtol=0.02
    )


def test_trace_cannot_drop_calls_from_both_runs():
    run = {"trace": [_call()], "counts": {"replays": 2, "eager_extends": 0}}
    assert validate_trace_counts(run)
    run["counts"]["replays"] = 1
    assert not validate_trace_counts(run)
    run["trace"].append(_call("eager"))
    assert validate_trace_counts(run)
    run["counts"]["eager_extends"] = 1
    assert not validate_trace_counts(run)
