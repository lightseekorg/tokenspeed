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

"""Selection of monokernel layers through the kernel registry."""

import pytest
from tokenspeed_kernel.ops.monokernel import (
    MONOKERNEL_SIGNATURE,
    PlanKey,
    prepare_monokernel_layer,
    prepare_monokernel_state,
)
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import register_kernel

pytestmark = pytest.mark.usefixtures("fresh_registry")

STATE_ARGS = dict(
    group="group",
    rank_in_group=3,
    max_requests=2,
    failure_record_path="/run/record-3",
    device="cpu",
)


@pytest.fixture(autouse=True)
def _amd_platform(monkeypatch, mi350_platform):
    # The test host has no GPU; selection reads the platform singleton.
    monkeypatch.setattr(Platform, "_instance", mi350_platform)


def _register_leaf(recipe):
    @register_kernel(
        "monokernel",
        "prepare",
        name=f"test_monokernel_{recipe}",
        solution="test",
        signatures={MONOKERNEL_SIGNATURE},
        traits={"recipe": frozenset({recipe}), "verify_width": frozenset({8})},
    )
    class Leaf:
        def __init__(self, *, plan_keys, weights, state):
            self.plan_keys = plan_keys
            self.weights = weights
            self.state = state

        @staticmethod
        def prepare_state(**kwargs):
            return ("state", kwargs)

    return Leaf


def test_selected_leaf_receives_every_argument():
    leaf = _register_leaf("muse_spark")
    state = prepare_monokernel_state(recipe="muse_spark", **STATE_ARGS)
    assert state == ("state", STATE_ARGS)

    keys = [PlanKey(1, "sliding", "full"), PlanKey(2, "sliding", "full")]
    layer = prepare_monokernel_layer(
        recipe="muse_spark",
        plan_keys=keys,
        traits={"verify_width": 8},
        weights={"w": 1},
        state=state,
    )
    assert isinstance(layer, leaf)
    assert (layer.plan_keys, layer.weights, layer.state) == (keys, {"w": 1}, state)


def test_unserved_recipe_or_traits_prepare_nothing():
    _register_leaf("muse_spark")
    assert prepare_monokernel_state(recipe="other", **STATE_ARGS) is None
    state = prepare_monokernel_state(recipe="muse_spark", **STATE_ARGS)
    common = dict(plan_keys=[PlanKey(1, "full", "regular")], weights={}, state=state)
    assert prepare_monokernel_layer(recipe="other", traits={}, **common) is None
    assert (
        prepare_monokernel_layer(
            recipe="muse_spark", traits={"verify_width": 4}, **common
        )
        is None
    )


def test_no_state_prepares_no_layer():
    _register_leaf("muse_spark")
    assert (
        prepare_monokernel_layer(
            recipe="muse_spark",
            plan_keys=[PlanKey(1, "full", "regular")],
            traits={"verify_width": 8},
            weights={},
            state=None,
        )
        is None
    )
