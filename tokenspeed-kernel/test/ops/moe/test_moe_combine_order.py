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

"""``moe_plan(combine_order=...)``: the plan names how routed slots meet.

``"rank"`` is what every apply kernel serves (a per-rank partial the caller
reduces); ``"slot"`` is an in-kernel EP fold only kernels declaring the
``combine_order`` trait provide. The plan must carry the choice and the EP
group to the kernel, and refuse a kernel that cannot honour it.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe import COMBINE_ORDERS, moe_plan
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, Priority
from tokenspeed_kernel.selection import NoKernelFoundError
from tokenspeed_kernel.signature import format_signatures

_SIGNATURES = frozenset(format_signatures("x", "dense", {torch.bfloat16}))
_COMMON_TRAITS = {
    "weight_dtype": frozenset({"unquant"}),
    "routing_mode": frozenset({"precomputed_topk"}),
    "supports_ep": frozenset({True}),
}


def _register(
    name: str,
    *,
    combine_orders: tuple[str, ...] | None,
    priority: int = Priority.PORTABLE,
) -> None:
    traits = dict(_COMMON_TRAITS)
    if combine_orders is not None:
        traits["combine_order"] = frozenset(combine_orders)
    spec = KernelSpec(
        name=name,
        family="moe",
        mode="apply",
        solution=name,
        format_signatures=_SIGNATURES,
        traits=traits,
        priority=priority,
    )
    KernelRegistry.get().register(spec, lambda **kwargs: None)


def _plan(combine_order: str, **overrides) -> dict:
    kwargs = dict(
        input_dtype=torch.bfloat16,
        routing_mode="precomputed_topk",
        ep_size=1,
        hidden=None,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=True,
        fast_math=False,
        combine_order=combine_order,
    )
    kwargs.update(overrides)
    return moe_plan("unquant", **kwargs)


@pytest.fixture
def two_leaves(fresh_registry, h100_platform):
    _ = fresh_registry
    real_platform = Platform.get()
    Platform.override(h100_platform)
    # The tuned, silent kernel outranks the folding one, as a vendor's
    # batch-invariant leaf (reference priority) sits below the tuned kernels.
    _register("silent_moe_apply", combine_orders=None, priority=Priority.PERFORMANT)
    _register(
        "folding_moe_apply",
        combine_orders=("rank", "slot"),
        priority=Priority.REFERENCE,
    )
    yield
    Platform.override(real_platform)


def test_orders_are_the_two_documented_ones():
    assert COMBINE_ORDERS == ("rank", "slot")


def test_rank_plans_any_kernel_and_records_the_order(two_leaves):
    plan = _plan("rank", solution="silent_moe_apply")
    assert plan["apply_kernel_name"] == "silent_moe_apply"
    assert plan["combine_order"] == "rank"
    assert _plan("rank", solution="folding_moe_apply")["combine_order"] == "rank"


def test_slot_needs_a_kernel_declaring_the_trait(two_leaves):
    plan = _plan("slot", solution="folding_moe_apply")
    assert plan["apply_kernel_name"] == "folding_moe_apply"
    assert plan["combine_order"] == "slot"
    # A silent kernel is not excluded by the trait (it declares nothing), so
    # an unpinned plan that lands on it is refused rather than reduced twice.
    with pytest.raises(ValueError, match="does not declare the combine_order"):
        _plan("slot")


def test_declared_orders_are_enforced(fresh_registry, h100_platform):
    _ = fresh_registry
    real_platform = Platform.get()
    Platform.override(h100_platform)
    try:
        _register("rank_only_moe_apply", combine_orders=("rank",))
        assert _plan("rank")["apply_kernel_name"] == "rank_only_moe_apply"
        # Declaring the opposite order excludes the kernel at selection.
        with pytest.raises(NoKernelFoundError):
            _plan("slot", solution="rank_only_moe_apply")
    finally:
        Platform.override(real_platform)


def test_slot_under_ep_carries_the_process_group(two_leaves):
    group = object()
    plan = _plan("slot", ep_size=2, process_group=group, solution="folding_moe_apply")
    assert plan["process_group"] is group
    with pytest.raises(ValueError, match="EP process group"):
        _plan("slot", ep_size=2, solution="folding_moe_apply")
    # A single EP rank folds locally and needs no group.
    plan = _plan("slot", ep_size=1, solution="folding_moe_apply")
    assert plan["process_group"] is None


def test_unknown_order_is_refused(two_leaves):
    with pytest.raises(ValueError, match="combine_order must be one of"):
        _plan("tree")
