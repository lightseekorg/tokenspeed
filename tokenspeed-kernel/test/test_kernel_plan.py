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

"""Real registry planning and numerical execution, without GPU dependencies."""

import pytest
import torch
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.selection import (
    KernelPlan,
    NoKernelFoundError,
    kernel_override,
)
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

pytestmark = pytest.mark.usefixtures("fresh_registry")
SIGNATURE = format_signature(a=dense_tensor_format(torch.float32))


def _register(name, *, priority, invariant, traits):
    # A row-local reduction whose order is independent of the number of rows.
    def reduce_rows(x):
        output = torch.zeros(x.shape[0], dtype=x.dtype)
        for column in range(x.shape[1]):
            output = output + x[:, column]
        return output

    spec = KernelSpec(
        name=name,
        family="test_plan",
        mode="reduce",
        solution=name,
        format_signatures=frozenset({SIGNATURE}),
        priority=priority,
        features=frozenset({"batch_invariant"}) if invariant else frozenset(),
        traits=traits,
    )
    KernelRegistry.get().register(spec, reduce_rows)


def _plan(platform):
    return KernelPlan.select(
        "test_plan",
        "reduce",
        SIGNATURE,
        features=frozenset({"batch_invariant"}),
        static_traits={"k": 4},
        variable_traits=frozenset({"m"}),
        solution=None,
        declared_traits=frozenset(),
        ignore_layout=False,
        platform=platform,
    )


def test_plan_survives_registry_changes_and_batch_changes(h100_platform):
    _register("fixed", priority=1, invariant=True, traits={})
    plan = _plan(h100_platform)
    _register("later_faster", priority=19, invariant=True, traits={})
    x = torch.tensor([[1e10, -1e10, 3.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    for count in (1, 2, 17):
        inputs = x.repeat(count, 1)
        plan.validate(
            "test_plan",
            "reduce",
            SIGNATURE,
            traits={"m": inputs.shape[0], "k": 4},
            features=None,
            solution=None,
            override=None,
        )
        selected = plan.kernel
        assert selected.name == "fixed"
        assert torch.equal(selected(inputs), torch.tensor([4.0, 10.0]).repeat(count))


@pytest.mark.parametrize(
    "traits", [{name: frozenset({16})} for name in ("m", "m_min", "m_max", "m_align")]
)
def test_plan_rejects_batch_limited_provider(h100_platform, traits):
    _register("limited", priority=19, invariant=True, traits=traits)
    with pytest.raises(NoKernelFoundError):
        _plan(h100_platform)


def test_plan_requires_declared_invariance(h100_platform):
    _register("ordinary", priority=19, invariant=False, traits={})
    with pytest.raises(NoKernelFoundError):
        _plan(h100_platform)


def test_plan_checks_geometry_and_does_not_follow_new_override(h100_platform):
    _register("fixed", priority=1, invariant=True, traits={})
    plan = _plan(h100_platform)
    with pytest.raises(ValueError, match="static trait k changed"):
        plan.validate(
            "test_plan",
            "reduce",
            SIGNATURE,
            traits={"m": 1, "k": 8},
            features=None,
            solution=None,
            override=None,
        )
    with kernel_override("test_plan", "reduce", "another"):
        with pytest.raises(ValueError, match="change override"):
            plan.validate(
                "test_plan",
                "reduce",
                SIGNATURE,
                traits={"m": 1, "k": 4},
                features=None,
                solution=None,
                override=None,
            )


@pytest.mark.parametrize("names", [("z", "a"), ("a", "z")])
def test_ties_do_not_depend_on_registration_order(h100_platform, names):
    for name in names:
        _register(name, priority=1, invariant=True, traits={})
    assert _plan(h100_platform).kernel.name == "a"


def test_initial_override_cannot_bypass_capability(h100_platform):
    _register("ordinary", priority=19, invariant=False, traits={})
    _register("fixed", priority=1, invariant=True, traits={})
    with kernel_override("test_plan", "reduce", "ordinary"):
        with pytest.raises(NoKernelFoundError):
            _plan(h100_platform)


def test_plan_rejects_changed_format(h100_platform):
    _register("fixed", priority=1, invariant=True, traits={})
    plan = _plan(h100_platform)
    signature = format_signature(a=dense_tensor_format(torch.bfloat16))
    with pytest.raises(ValueError, match="Input contract changed"):
        plan.validate(
            "test_plan",
            "reduce",
            signature,
            traits={"k": 4, "m": 1},
            features=None,
            solution=None,
            override=None,
        )


def test_plan_checks_every_value_in_a_static_domain(h100_platform):
    _register("both", priority=1, invariant=True, traits={"q_len": frozenset({1, 2})})
    plan = KernelPlan.select(
        "test_plan",
        "reduce",
        SIGNATURE,
        features=frozenset({"batch_invariant"}),
        static_traits={"k": 4, "q_len": frozenset({1, 2})},
        variable_traits=frozenset({"m"}),
        solution=None,
        declared_traits=frozenset(),
        ignore_layout=False,
        platform=h100_platform,
    )
    for q_len in (1, 2):
        plan.validate(
            "test_plan",
            "reduce",
            SIGNATURE,
            traits={"k": 4, "m": 3, "q_len": q_len},
            features=None,
            solution=None,
            override=None,
        )
    with pytest.raises(ValueError, match="static trait q_len changed"):
        plan.validate(
            "test_plan",
            "reduce",
            SIGNATURE,
            traits={"k": 4, "m": 3, "q_len": 3},
            features=None,
            solution=None,
            override=None,
        )


def test_plan_does_not_infer_a_semantic_promise_from_an_absent_trait(h100_platform):
    _register("silent", priority=19, invariant=True, traits={})
    _register(
        "ordered",
        priority=1,
        invariant=True,
        traits={"slot_order": frozenset({"sorted"})},
    )
    plan = KernelPlan.select(
        "test_plan",
        "reduce",
        SIGNATURE,
        features=frozenset({"batch_invariant"}),
        static_traits={"k": 4, "slot_order": "sorted"},
        variable_traits=frozenset({"m"}),
        declared_traits=frozenset({"slot_order"}),
        solution=None,
        ignore_layout=False,
        platform=h100_platform,
    )
    assert plan.kernel.name == "ordered"


@pytest.mark.parametrize(
    "dimension,semantic", [("topk", "topk_layout"), ("m", "m_layout")]
)
def test_variable_dimension_does_not_reject_static_layout(
    h100_platform, dimension, semantic
):
    _register(
        "ordered",
        priority=1,
        invariant=True,
        traits={semantic: frozenset({"global_slots"})},
    )
    plan = KernelPlan.select(
        "test_plan",
        "reduce",
        SIGNATURE,
        features=frozenset({"batch_invariant"}),
        static_traits={"k": 4, semantic: "global_slots"},
        variable_traits=frozenset({"m", dimension}),
        declared_traits=frozenset({semantic}),
        solution=None,
        ignore_layout=False,
        platform=h100_platform,
    )
    for rows in (1, 7):
        inputs = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
        plan.validate(
            "test_plan",
            "reduce",
            SIGNATURE,
            traits={"k": 4, "m": rows, dimension: rows, semantic: "global_slots"},
            features=None,
            solution=None,
            override=None,
        )
        assert torch.equal(plan.kernel(inputs), inputs.sum(dim=1))
    with pytest.raises(ValueError, match="static trait"):
        plan.validate(
            "test_plan",
            "reduce",
            SIGNATURE,
            traits={"k": 4, "m": 1, dimension: 1, semantic: "different"},
            features=None,
            solution=None,
            override=None,
        )
