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

import pytest
import torch
from tokenspeed_kernel.ops.moe import moe_plan
from tokenspeed_kernel.ops.moe.gluon import (
    GLUON_FP8_CHANNEL_APPLY,
    GLUON_FP8_CHANNEL_POLICY,
    gluon_fp8_channel_bf16_precomputed_moe_apply,
)
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import NoKernelFoundError


def _plan(platform, **overrides):
    kwargs = {
        "weight_dtype": "fp8_channel",
        "input_dtype": torch.bfloat16,
        "activation": "silu",
        "routing_mode": "precomputed_topk",
        "ep_size": 1,
        "ispp": 640,
        "hidden": 5120,
        "swiglu_form": "standard",
        "activation_clamped": False,
        "expert_id_repeats": False,
        "fast_math": False,
        "combine_order": "rank",
    }
    kwargs.update(overrides)
    original = Platform.get()
    try:
        Platform.override(platform)
        KernelRegistry.get().clear_cache()
        return moe_plan(**kwargs)
    finally:
        Platform.override(original)
        KernelRegistry.get().clear_cache()


@pytest.mark.parametrize("intermediate", [320, 640])
def test_default_plan_keeps_native_fp8(mi350_platform, intermediate):
    assert (
        _plan(mi350_platform, ispp=intermediate)["apply_kernel_name"]
        == "triton_fp8_channel_fused_precomputed_moe_apply"
    )


@pytest.mark.parametrize("intermediate", [320, 640])
def test_explicit_gluon_plan_uses_channel_candidate(mi350_platform, intermediate):
    plan = _plan(mi350_platform, solution="gluon", ispp=intermediate)
    assert plan["apply_kernel_name"] == GLUON_FP8_CHANNEL_APPLY
    assert plan["weight_preprocessor"] is None
    assert not plan["supports_deferred_finalize"]


@pytest.mark.parametrize(
    "dtype,width,expected",
    [
        ("unquant", 640, "gluon_bf16_precomputed_moe_apply"),
        ("fp8", 1024, "gluon_fp8_block_precomputed_moe_apply"),
    ],
)
def test_legacy_gluon_plans_keep_existing_implementation(
    mi350_platform, dtype, width, expected
):
    assert (
        _plan(mi350_platform, weight_dtype=dtype, ispp=width, solution="gluon")[
            "apply_kernel_name"
        ]
        == expected
    )


@pytest.mark.parametrize(
    "override",
    [
        {"ep_size": 2},
        {"ispp": 352},
        {"hidden": 2048},
        {"activation_clamped": True},
        {"expert_id_repeats": True},
        {"requires_deferred_finalize": True},
        {"with_bias": True},
    ],
)
def test_gluon_plan_rejects_unsupported_geometry(mi350_platform, override):
    with pytest.raises(NoKernelFoundError):
        _plan(mi350_platform, solution="gluon", **override)


@pytest.mark.parametrize(
    "name,policy",
    [
        (None, None),
        (GLUON_FP8_CHANNEL_APPLY, None),
        ("triton_fp8_channel_fused_precomputed_moe_apply", GLUON_FP8_CHANNEL_POLICY),
        (GLUON_FP8_CHANNEL_APPLY, "fp8ch_route_accum_fp32"),
    ],
)
def test_apply_requires_both_explicit_name_and_policy(name, policy):
    with pytest.raises(
        ValueError, match="explicit implementation and precision_policy"
    ):
        gluon_fp8_channel_bf16_precomputed_moe_apply(
            {"apply_kernel_name": name, "precision_policy": policy},
            None,
            None,
            None,
        )


@pytest.mark.parametrize(
    "options,error",
    [
        ({"enable_pdl": True}, "enable_pdl"),
        ({"plan_extra": {"output_scale": 2.0}}, "unit output scale"),
        ({"plan_extra": {"combine_order": "slot"}}, "rank-local"),
    ],
)
def test_apply_rejects_unsupported_launch_before_device_use(options, error):
    options = dict(options)
    plan = {
        "apply_kernel_name": GLUON_FP8_CHANNEL_APPLY,
        "precision_policy": GLUON_FP8_CHANNEL_POLICY,
    }
    plan.update(options.pop("plan_extra", {}))
    with pytest.raises(ValueError, match=error):
        gluon_fp8_channel_bf16_precomputed_moe_apply(
            plan, None, torch.nn.Module(), None, **options
        )


@pytest.mark.parametrize("intermediate", [320, 640])
def test_correct_policy_still_rejects_non_gpu_tensors(intermediate):
    weights = torch.nn.Module()
    weights.w13_weight = torch.empty(
        256, 2 * intermediate, 5120, device="meta", dtype=torch.float8_e4m3fn
    )
    weights.w2_weight = torch.empty(
        256, 5120, intermediate, device="meta", dtype=torch.float8_e4m3fn
    )
    weights.w13_weight_scale = torch.empty(256, 2 * intermediate, device="meta")
    weights.w2_weight_scale = torch.empty(256, 5120, device="meta")
    with pytest.raises(ValueError, match="contiguous GPU tensors"):
        gluon_fp8_channel_bf16_precomputed_moe_apply(
            {
                "apply_kernel_name": GLUON_FP8_CHANNEL_APPLY,
                "precision_policy": GLUON_FP8_CHANNEL_POLICY,
            },
            torch.empty(1, 5120, device="meta", dtype=torch.bfloat16),
            weights,
            None,
            topk_ids=torch.zeros(1, 8, dtype=torch.int32),
            topk_weights=torch.ones(1, 8),
        )
