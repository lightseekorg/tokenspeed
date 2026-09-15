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

"""CPU checks for real producer timing, one-copy policy, and alias-safe checks."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from medium_fused_tail_producer_contract import (
    ARMS,
    COMPARISONS,
    producer_measurement_config,
)


@pytest.mark.parametrize("tokens", (256, 384, 512, 768, 832, 896, 960, 1024))
def test_producer_boundary_is_distinct(tokens):
    config = producer_measurement_config(tokens, 5, 8, 20, 3)
    assert config["producer_included"]
    assert config["producer_activation_shape"] == [tokens, 768]
    assert config["producer_weight_shape"] == [7168, 768]
    assert config["producer_solution"] == "torch"
    assert "baseline" not in config and "candidate" not in config
    assert "shared_staging_in_both" not in config
    assert config["correctness_snapshots_before_next_arm"]
    assert config["same_bt_for_all_arms"]
    assert config["same_fused_plan_for_copied_and_direct"]
    assert not config["formal_sampling"]
    assert not config["trace_verified_no_staging"]
    assert not config["serving_measurement"]


def test_formal_sampling_does_not_claim_acceptance():
    config = producer_measurement_config(1024, 31, 8, 20, 128)
    assert config["formal_sampling"]
    assert not config["two_batch_acceptance"]
    assert COMPARISONS == (
        ("baseline", "copied"),
        ("copied", "direct"),
        ("baseline", "direct"),
    )


def extract_function(name):
    source = Path(__file__).with_name("bench_medium_fused_tail_producer.py").read_text()
    module = ast.parse(source)
    return next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


class FakeTensor:
    def __init__(self, label, trace):
        self.label, self.trace = label, trace

    def data_ptr(self):
        return self.label

    def __getitem__(self, index):
        return self

    def t(self):
        return self

    def add_(self, value):
        self.trace.append(("add", self.label))

    def addmm_(self, latent, weight):
        self.trace.append(("addmm", self.label))

    def clone(self):
        self.trace.append(("clone", self.label))
        return self


class FakePlan:
    def __init__(self, slot, trace):
        self.input_view = FakeTensor(("symmetric", slot), trace)
        self.inputs = [FakeTensor("latent", trace)]
        self.trace, self.slot = trace, slot

    def stage(self, partial):
        self.trace.append(("stage", self.slot))

    def __call__(self):
        self.trace.append(("fused", self.slot))
        return FakeTensor(("output", self.slot), self.trace)


@pytest.mark.parametrize("mode", ARMS)
def test_execution_has_real_producer_per_layer_and_exact_copy_policy(mode):
    trace = []

    def producer(activation, weight, *, out, solution):
        assert solution == "torch"
        expected = "symmetric" if mode == "direct" else "ordinary"
        assert out.label == (expected, activation)
        trace.append(("producer", activation))
        return out

    def stage(partial, group_name, capacity):
        assert capacity == 1024
        trace.append(("stage", partial.label[1]))
        return partial

    def bt(gemm2, weights, indices, gamma):
        trace.append(("BT", gemm2))
        return FakeTensor("latent", trace)

    def allreduce(partial, group_name):
        trace.append(("AR2", partial.label))
        return partial

    namespace = {
        "ARMS": ARMS,
        "kimi3_shared_down_projection": producer,
        "multimem_stage": stage,
        "multimem_all_reduce_staged": allreduce,
    }
    exec(
        compile(
            ast.Module(body=[extract_function("run_mode")], type_ignores=[]),
            "producer-execution",
            "exec",
        ),
        namespace,
    )
    pair = SimpleNamespace(
        _check_stream=lambda: None,
        producers=[(slot, slot) for slot in range(4)],
        plans=[FakePlan(slot, trace) for slot in range(4)],
        rank=0,
        group_name="test",
        bt=bt,
        layers=[
            SimpleNamespace(
                shared_partial=FakeTensor(("ordinary", slot), trace),
                gemm2_output=slot,
                expert_weights=None,
                expanded_idx=None,
                gamma=None,
                prefix=FakeTensor("prefix", trace),
                up_weight=FakeTensor("weight", trace),
            )
            for slot in range(4)
        ],
    )
    assert len(namespace["run_mode"](pair, mode)) == 4
    sequence = {
        "baseline": ["producer", "stage", "BT", "add", "addmm", "AR2", "clone"],
        "copied": ["producer", "stage", "BT", "fused"],
        "direct": ["producer", "BT", "fused"],
    }[mode]
    assert [event[0] for event in trace] == sequence * 4
    assert [event[1] for event in trace if event[0] == "producer"] == list(range(4))


def test_replay_snapshots_before_another_arm_runs():
    trace = []
    shared = {"value": None}

    class Graph:
        def __init__(self, name):
            self.name = name

        def replay(self):
            shared["value"] = self.name
            trace.append("replay-" + self.name)

    def snapshot(values):
        trace.append("snapshot-" + shared["value"])
        return shared["value"]

    namespace = {"snapshot": snapshot}
    exec(
        compile(
            ast.Module(body=[extract_function("replay_observations")], type_ignores=[]),
            "producer-replay",
            "exec",
        ),
        namespace,
    )
    result = namespace["replay_observations"](
        SimpleNamespace(rank=0),
        {name: Graph(name) for name in ARMS},
        {name: shared for name in ARMS},
        0,
        0,
    )
    assert result == {name: name for name in ARMS}
    assert trace == [
        event for name in ARMS for event in ("replay-" + name, "snapshot-" + name)
    ]
