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

"""CPU-only tests of medium-tail numerical and measurement boundaries."""

import argparse
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from medium_fused_tail_contract import (
    BT_TUNING,
    REFERENCE_ATOL,
    REFERENCE_L2,
    REFERENCE_RTOL,
    measurement_config,
    reference_passes,
)


@pytest.mark.parametrize("tokens", (256, 257, 384, 512, 768, 832, 896, 960, 1023, 1024))
def test_matched_boundary_and_partial_tiles(tokens):
    config = measurement_config(tokens, 5, 8, 20, 4, 3)
    assert config["shared_staging_in_both"]
    assert "identical_bt" in config["baseline"]
    assert "ar2_clone" in config["baseline"]
    assert "identical_bt" in config["candidate"]
    assert not config["producer_included"]
    assert not config["serving_measurement"]
    assert not config["formal_sampling"]
    assert not config["two_batch_acceptance"]


@pytest.mark.parametrize("tokens", (True, 0, 255, 1025, 4096, 256.0))
def test_out_of_scope_rows_rejected(tokens):
    with pytest.raises(ValueError):
        measurement_config(tokens, 5, 8, 20, 4, 3)


@pytest.mark.parametrize(
    "args",
    (
        (256, 2, 8, 20, 4, 3),
        (256, 5, 0, 20, 4, 3),
        (256, 5, 8, 0, 4, 3),
        (256, 5, 8, 20, 3, 3),
        (256, 5, 8, 20, 5, 3),
        (256, 5, 8, 20, 4, 2),
    ),
)
def test_sampling_validation(args):
    with pytest.raises(ValueError):
        measurement_config(*args)


def test_formal_sampling_not_acceptance():
    config = measurement_config(1024, 31, 8, 20, 4, 128)
    assert config["formal_sampling"]
    assert not config["two_batch_acceptance"]
    assert BT_TUNING == {
        "max_tokens": 1024,
        "elements_per_thread": 2,
        "threads": 256,
        "prefetch_group": 1,
        "reduction_threads": 224,
        "rms_threads": 448,
        "enable_pdl": True,
    }


def test_original_numerical_bounds():
    assert (REFERENCE_ATOL, REFERENCE_RTOL, REFERENCE_L2) == (0.046875, 0.01, 0.006)
    assert reference_passes(0.0, 0.0, 0.0)
    assert reference_passes(0.05, 0.0059, 1.0)
    assert not reference_passes(0.0, 0.00601, 1.0)
    assert not reference_passes(0.056876, 0.005, 1.0)


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf, -1.0))
def test_nonfinite_and_negative_metrics_rejected(value):
    assert not reference_passes(value, 0.001, 1.0)
    assert not reference_passes(0.01, value, 1.0)
    assert not reference_passes(0.01, 0.001, value)


def test_harness_keeps_baseline_and_candidate_boundaries():
    source = Path(__file__).with_name("bench_medium_fused_tail.py").read_text()
    module = ast.parse(source)
    pair = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "TailPair"
    )
    methods = {
        node.name: ast.get_source_segment(source, node)
        for node in pair.body
        if isinstance(node, ast.FunctionDef)
    }
    baseline, candidate = methods["baseline"], methods["candidate"]
    assert (
        baseline.index("multimem_stage")
        < baseline.index("self.bt(")
        < baseline.index("owner.add_(")
    )
    assert (
        baseline.index("owner.add_(")
        < baseline.index("owner.addmm_(")
        < baseline.index("multimem_all_reduce_staged")
    )
    assert ".clone()" in baseline
    assert (
        candidate.index("plan.stage(")
        < candidate.index("self.bt(")
        < candidate.index("outputs.append(plan())")
    )
    for timed in (baseline, candidate):
        assert "all_gather_object" not in timed
        assert "prepare_fused" not in timed
        assert "SharedRsWorkspace.allocate" not in timed
    assert "stable_gain_accepted=False" in source


def test_optional_ab_stage_cli_preserves_original_command(monkeypatch):
    source = Path(__file__).with_name("bench_medium_fused_tail.py").read_text()
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    )
    namespace = {"argparse": argparse, "Path": Path, "__doc__": "CPU CLI check"}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "medium-cli", "exec"),
        namespace,
    )
    argv = [
        "bench",
        "--tokens",
        "256",
        "--implementation",
        "inherited",
        "--tile-m",
        "256",
        "--tile-n",
        "128",
        "--two-cta",
        "1",
        "--cluster-m",
        "2",
        "--cluster-n",
        "1",
        "--cluster-cap",
        "0",
        "--scheduler-type",
        "static_persistent",
        "--c-stages",
        "0",
        "--addend-stages",
        "1",
        "--reduce-vectors",
        "4",
        "--rounds",
        "5",
        "--warmup",
        "8",
        "--replays",
        "20",
        "--generations",
        "3",
        "--rank-skew-cycles",
        "0",
        "--output",
        "result.json",
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert namespace["parse_args"]().ab_stages == 0
    monkeypatch.setattr("sys.argv", argv + ["--ab-stages", "6"])
    assert namespace["parse_args"]().ab_stages == 6


def test_inherited_rejects_nonzero_ab_override():
    source = Path(__file__).with_name("bench_medium_fused_tail.py").read_text()
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_tuning"
    )
    namespace = {
        "FusedRsUpProjectionTuning": lambda **kwargs: kwargs,
        "fused_rs_up_projection_config": lambda tokens: {
            "tuning": {"token_config": tokens}
        },
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "medium-tuning", "exec"),
        namespace,
    )
    args = SimpleNamespace(
        implementation="inherited",
        tile_m=256,
        tile_n=128,
        two_cta=1,
        cluster_m=2,
        cluster_n=1,
        cluster_cap=0,
        scheduler_type="static_persistent",
        c_stages=0,
        addend_stages=1,
        reduce_vectors=4,
        rank_skew_cycles=0,
        ab_stages=0,
    )
    assert namespace["make_tuning"](args) == {"token_config": 8192}
    args.ab_stages = 6
    with pytest.raises(ValueError, match="inherited control"):
        namespace["make_tuning"](args)
    tree = ast.parse(ast.get_source_segment(source, function))
    values = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "values"
            for target in node.targets
        )
    )
    assert "ab_stages" in [key.value for key in values.keys]
