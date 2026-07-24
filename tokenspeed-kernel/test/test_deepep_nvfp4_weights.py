"""Regression tests for DeepEP NVFP4 weight preprocessing."""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

if not current_platform().is_nvidia:
    pytest.skip(
        "flashinfer cutedsl DeepEP NVFP4 kernels are NVIDIA-only",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.moe import moe_plan
from tokenspeed_kernel.ops.moe.flashinfer.cutedsl_deepep_nvfp4 import (
    flashinfer_cutedsl_deepep_nvfp4_moe_weights,
)


def test_deepep_nvfp4_quant_scales_are_contiguous_per_expert() -> None:
    weights = torch.nn.Module()
    weights.num_local_experts = 2
    weights.w13_weight = torch.empty((2, 8, 8))
    weights.w13_weight_scale = torch.nn.Parameter(torch.ones((2, 3, 5)))
    weights.w13_weight_scale_2 = torch.nn.Parameter(torch.ones((2, 1)))
    weights.w13_input_scale = torch.nn.Parameter(torch.tensor([2.0, 4.0]))
    weights.w2_weight_scale = torch.nn.Parameter(torch.ones((2, 3, 5)))
    weights.w2_weight_scale_2 = torch.nn.Parameter(torch.ones(2))
    weights.w2_input_scale = torch.nn.Parameter(torch.tensor([4.0, 8.0]))

    flashinfer_cutedsl_deepep_nvfp4_moe_weights({}, weights)

    assert weights.w13_input_scale_quant.shape == (2,)
    assert weights.w13_input_scale_quant.is_contiguous()
    torch.testing.assert_close(
        weights.w13_input_scale_quant,
        torch.full((2,), 0.25),
    )
    assert weights.w2_input_scale_quant.shape == (2,)
    assert weights.w2_input_scale_quant.is_contiguous()
    torch.testing.assert_close(
        weights.w2_input_scale_quant,
        torch.full((2,), 0.125),
    )


def test_deepep_plan_preserves_static_low_latency_token_capacity(require) -> None:
    require("moe", "apply", "flashinfer_cutedsl", torch.bfloat16, "x")
    plan = moe_plan(
        "nvfp4",
        input_dtype=torch.bfloat16,
        activation="silu",
        a2a_backend="deepep",
        ep_size=2,
        ispp=128,
        deepep_group=object(),
        low_latency_max_num_tokens_per_gpu=256,
        solution="flashinfer_cutedsl",
    )

    assert plan["low_latency_max_num_tokens_per_gpu"] == 256
