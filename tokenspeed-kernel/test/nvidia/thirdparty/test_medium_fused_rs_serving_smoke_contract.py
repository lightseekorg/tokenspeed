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

"""CPU checks of serving-adapter smoke scope, ordering and rejection coverage."""

import ast
from pathlib import Path

import pytest


def read_harness():
    source = Path(__file__).with_name("smoke_medium_fused_rs_serving.py").read_text()
    return source, ast.parse(source)


def load_function(name, namespace):
    _, module = read_harness()
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "serving-smoke-contract",
            "exec",
        ),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize(
    "tokens,generations,skew",
    (
        ([256, 1024], 3, 0),
        ([256, 832, 1024], 128, 10000),
        ([256, 384, 512, 768, 832, 896, 960, 1024], 128, 10000),
    ),
)
def test_supported_multi_bucket_controls(tokens, generations, skew):
    validate = load_function(
        "validate_options",
        {
            "MEDIUM_FUSED_RS_SERVING_CANDIDATE_TOKENS": frozenset(
                (256, 384, 512, 768, 832, 896, 960, 1024)
            )
        },
    )
    validate(tokens, generations, skew)


@pytest.mark.parametrize(
    "tokens,generations,skew",
    (
        ([256], 3, 0),
        ([256, 256], 3, 0),
        ([255, 1024], 3, 0),
        ([256, 4096], 3, 0),
        ([256, 1024], 2, 0),
        ([256, 1024], 3, -1),
    ),
)
def test_invalid_scope_rejected(tokens, generations, skew):
    validate = load_function(
        "validate_options",
        {
            "MEDIUM_FUSED_RS_SERVING_CANDIDATE_TOKENS": frozenset(
                (256, 384, 512, 768, 832, 896, 960, 1024)
            )
        },
    )
    with pytest.raises(ValueError):
        validate(tokens, generations, skew)


def test_real_producer_before_each_chained_layer():
    trace = []

    class Tensor:
        shape = (256, 7168)

    class Adapter:
        def __init__(self, slot):
            self.slot = slot

        def input_view(self, m):
            assert m == 256
            return ("symmetric", self.slot)

        def __call__(self, latent, weight, prefix, shared):
            expected = original if self.slot == 0 else ("output", 0)
            assert prefix is expected or prefix == expected
            assert shared == ("symmetric", self.slot)
            trace.append(("adapter", self.slot))
            return ("output", self.slot)

    def producer(activation, weight, *, out, solution):
        assert solution == "torch"
        trace.append(("producer", activation))
        return out

    original = Tensor()
    run = load_function("run_layers", {"kimi3_shared_down_projection": producer})
    outputs = run(
        {"residual": original, "latent": [0, 1], "producers": [(0, 0), (1, 1)]},
        [Adapter(0), Adapter(1)],
        [0, 1],
    )
    assert outputs == [("output", 0), ("output", 1)]
    assert trace == [("producer", 0), ("adapter", 0), ("producer", 1), ("adapter", 1)]


def test_dynamic_pointer_rejection_and_persistent_alias_checks_are_explicit():
    source, module = read_harness()
    functions = {
        node.name: ast.get_source_segment(source, node)
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "weight changed" in functions["rejection_checks"]
    assert "compilation must finish during warmup" in functions["rejection_checks"]
    assert "not fresh._plans" in functions["rejection_checks"]
    assert (
        "warm_latent" in functions["make_case"]
        and "warm_residual" in functions["make_case"]
    )
    assert "[x.clone() for x in run_layers" in functions["make_case"]
    assert "flags.append(torch.all(actual == expected))" in functions["run"]
    assert '"full_model_or_ttft_qualified": False' in functions["main"]
    assert '"performance_measured": False' in functions["main"]
