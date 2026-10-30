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

from __future__ import annotations

import json

import pytest
import torch
from tokenspeed_kernel.op_profile import (
    CostRegistry,
    OpProfile,
    ProfileSession,
    profiled,
)
from tokenspeed_kernel.profile_costs import DenseMatmulCost
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, register_kernel
from tokenspeed_kernel.selection import SelectedKernel


@pytest.fixture
def mm(fresh_registry):
    def matmul(a, b, *, out=None):
        return torch.mm(a, b, out=out)

    registry = KernelRegistry.get()
    spec = KernelSpec(
        name="profile_test_mm", family="gemm", mode="mm", solution="reference"
    )
    result = registry.register(spec, matmul, profile=OpProfile(cost_rule="dense_mm"))
    assert result is None
    return registry.get_impl(spec.name)


class Layer:
    def __init__(self, op):
        self.op = op

    @profiled(level="layer")
    def forward(self, a, b, *, out=None):
        return self.op(a, b, out=out)


def session_for(*layers):
    session = ProfileSession(run_id="test", model_id="target", rank=2, backend="none")
    for i, layer in enumerate(layers):
        session.bind_layer(layer, layer_id=i)
    return session


def costs():
    registry = CostRegistry()
    registry.register("dense_mm", DenseMatmulCost(lhs="a", rhs="b", output="return"))
    return registry


def test_repeated_layers_iterations_and_selected_call(mm, tmp_path):
    layer = Layer(SelectedKernel("profile_test_mm", mm))
    session = session_for(layer)
    a, b = torch.ones(2, 3), torch.ones(3, 4)
    out = torch.empty(2, 4)
    assert layer.forward(a, b, out=out) is out
    assert session.ops == []
    with session.iteration(iter_id=10):
        assert layer.forward(a, b, out=out) is out
        layer.forward(a=a, b=b)
    with session.iteration(iter_id=11):
        layer.forward(a, b)
    torch.testing.assert_close(out, torch.full((2, 4), 3.0))
    assert [(op.iter_id, op.inner_id, op.occurrence) for op in session.ops] == [
        (10, 0, 0),
        (10, 1, 1),
        (11, 0, 0),
    ]
    assert session.ops[0].inputs["a"].shape == (2, 3)
    assert session.ops[0].outputs["return"].logical_bytes == 32
    path = tmp_path / "result.json"
    session.export(path, costs=costs())
    report = json.loads(path.read_text())
    summary = report["analysis"]["layers"][0]
    assert summary["complete_cost"] == dict(
        useful_flops=96,
        executed_flops=96,
        logical_read_bytes=144,
        logical_write_bytes=64,
    )
    assert summary["covered_op_count"] == summary["op_count"] == 2
    assert summary["mfu"] is None and summary["estimated_mbu"] is None


def test_exception_restores_layer_and_session(mm):
    layer = Layer(mm)
    session = session_for(layer)
    a, b = torch.ones(2, 3), torch.ones(3, 4)
    with session.iteration(iter_id=0):
        with pytest.raises(RuntimeError):
            layer.forward(a, torch.ones(5, 4))
        mm(a, b)
        layer.forward(a, b)
    assert [op.layer_id for op in session.ops] == [0, None, 0]
    assert [op.inner_id for op in session.ops] == [0, 0, 1]
    summary = costs().analyze(session)["layers"][0]
    assert summary["complete_cost"] is None
    assert summary["covered_op_count"] == 1
    assert session.layers[0].error == "RuntimeError"
    with pytest.raises(ValueError, match="already recorded"):
        with session.iteration(iter_id=0):
            pass
    mm(a, b)
    assert len(session.ops) == 3


def test_nested_layers_restore_parent(mm):
    child = Layer(mm)

    class Parent:
        @profiled(level="layer")
        def forward(self, a, b):
            mm(a, b)
            child.forward(a, b)
            return mm(a, b)

    parent = Parent()
    session = session_for(parent, child)
    with session.iteration(iter_id=0):
        parent.forward(torch.ones(2, 3), torch.ones(3, 4))
    assert [(op.layer_id, op.inner_id) for op in session.ops] == [
        (0, 0),
        (1, 0),
        (0, 1),
    ]
    assert [row["op_count"] for row in costs().analyze(session)["layers"]] == [2, 1]


def test_missing_cost_and_unbound_identity(mm):
    layer = Layer(mm)
    session = ProfileSession(run_id="test", model_id="target", rank=0, backend="none")
    with session.iteration(iter_id=0):
        with pytest.raises(ValueError, match="Bind the layer"):
            layer.forward(torch.ones(2, 3), torch.ones(3, 4))
        mm(torch.ones(2, 3), torch.ones(3, 4))
    result = CostRegistry().analyze(session)
    assert result["ops"][0]["cost"] is None
    assert result["layers"][0]["complete_cost"] is None


def test_noncontiguous_mixed_storage_and_output_roles(fresh_registry):
    @register_kernel(
        "test",
        "cast_mm",
        name="cast_mm",
        solution="reference",
        signatures=frozenset(),
        profile=OpProfile(cost_rule="dense_mm"),
    )
    def cast_mm(a, b):
        return (a.float() @ b.float()).half()

    a = torch.ones(3, 2, dtype=torch.float16).t()
    b = torch.ones(3, 4, dtype=torch.float32)
    session = session_for()
    with session.iteration(iter_id=0):
        cast_mm(a, b)
    assert session.ops[0].inputs["a"].stride == (1, 2)
    cost = costs().analyze(session)["ops"][0]["cost"]
    assert cost["logical_read_bytes"] == 12 + 48
    assert cost["logical_write_bytes"] == 16
    assert cost["estimated_hbm_bytes"] is None


def test_registration_without_profile_preserves_behavior(fresh_registry):
    def function(x, *, scale):
        return x * scale

    registered = register_kernel(
        "test", "plain", name="plain", solution="reference", signatures=frozenset()
    )(function)
    assert registered(3, scale=4) == 12
    session = session_for()
    with session.iteration(iter_id=0):
        assert registered(5, scale=2) == 10
    assert session.ops == []


def test_stateful_registration_explicitly_rejects_profile(fresh_registry):
    class Stateful:
        def __call__(self, x):
            return x

    with pytest.raises(TypeError, match="synchronous registered functions"):
        register_kernel(
            "test",
            "stateful",
            name="stateful",
            solution="reference",
            signatures=frozenset(),
            profile=OpProfile(cost_rule=None),
        )(Stateful)
    assert KernelRegistry.get().get_by_name("stateful") is None

    registry = KernelRegistry.get()
    spec = KernelSpec(name="stateful", family="test", mode="stateful")
    registry.register(spec, lambda x: x + 1)
    with pytest.raises(TypeError, match="synchronous registered functions"):
        registry.register(spec, Stateful, profile=OpProfile(cost_rule=None))
    assert registry.get_impl("stateful")(2) == 3


def test_nested_session_rejected_and_recovery(mm):
    first, second = session_for(), session_for()
    with first.iteration(iter_id=0):
        with pytest.raises(RuntimeError, match="Nested"):
            with second.iteration(iter_id=0):
                pass
        mm(torch.ones(2, 3), torch.ones(3, 4))
    with second.iteration(iter_id=0):
        mm(torch.ones(2, 3), torch.ones(3, 4))
    assert len(first.ops) == len(second.ops) == 1


def test_composite_registration_records_outer_call_once(mm):
    @register_kernel(
        "gemm",
        "mm",
        name="composite",
        solution="reference",
        signatures=frozenset(),
        profile=OpProfile(cost_rule="dense_mm"),
    )
    def composite(a, b):
        return mm(a, b)

    session = session_for()
    with session.iteration(iter_id=0):
        composite(torch.ones(2, 3), torch.ones(3, 4))
        mm(torch.ones(2, 3), torch.ones(3, 4))
    assert [op.implementation for op in session.ops] == ["composite", "profile_test_mm"]
    assert costs().analyze(session)["layers"][0]["complete_cost"]["useful_flops"] == 96


def test_supplied_arguments_are_not_discarded(mm):
    layer = Layer(mm)
    session = session_for(layer)
    with session.iteration(iter_id=0):
        with pytest.raises(TypeError):
            mm(torch.ones(2, 3), torch.ones(3, 4), unsupported=True)
        layer.forward(torch.ones(2, 3), torch.ones(3, 4))
    assert len(session.ops) == 1


def test_profiler_labels_show_operator_and_layer_before_serialized_identity(
    mm, tmp_path
):
    layer = Layer(mm)
    session = ProfileSession(
        run_id='run/"quoted"', model_id="target", rank=2, backend="torch"
    )
    session.bind_layer(layer, layer_id=7)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        with session.iteration(iter_id=3):
            result = layer.forward(torch.ones(2, 3), torch.ones(3, 4))
    torch.testing.assert_close(result, torch.full((2, 4), 3.0))
    path = tmp_path / "readable.json"
    profiler.export_chrome_trace(str(path))
    names = {event["name"] for event in json.loads(path.read_text())["traceEvents"]}
    assert session.ops[0].trace_label in names
    assert session.layers[0].trace_label in names
    assert session.ops[0].trace_label.startswith(
        "layer=7 | op=gemm.mm | rank=2 iter=3 call=0 | "
    )
    assert session.layers[0].trace_label.startswith(
        "layer=7 | Layer.forward | rank=2 iter=3 call=0 | "
    )
