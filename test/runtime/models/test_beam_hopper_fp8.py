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

"""Beam's FP8 checkpoint on NVIDIA SM90: admission, MoE plan and loading.

No GPU kernels are launched; the platform is overridden to SM90.
"""

import dataclasses
import os
import sys

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, Platform, current_platform

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.models import beam

# The CI runner executes registered test files as standalone scripts.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from test.runtime.models.beam_fixture import tiny_config  # noqa: E402
from test.runtime.models.test_beam_moe_quant import (  # noqa: E402
    NVFP4,
    _post_load,
    ct,
    fp8_checkpoint,
)
from test.runtime.models.test_beam_mxfp4 import mxfp4_config  # noqa: E402

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

FP8_CHANNEL_APPLY = "triton_fp8_channel_precomputed_moe_apply"
PREFIX = "model.layers.1.mlp"


@pytest.fixture
def hopper_platform():
    original = current_platform()
    Platform.override(
        dataclasses.replace(original, vendor="nvidia", arch_version=ArchVersion(9, 0))
    )
    yield
    Platform.override(original)


def _beam_geometry():
    config = tiny_config()
    config.hidden_size = 5120
    config.moe_intermediate_size = 2560
    config.num_experts = 256
    config.num_experts_per_tok = 8
    config.n_group = 8
    config.topk_group = 4
    return config


@pytest.mark.parametrize("backend", ["auto", "triton"])
@pytest.mark.parametrize("tp_size", [1, 8])
def test_hopper_fp8_experts_plan_the_triton_fp8_channel_moe(
    server, hopper_platform, backend, tp_size
):
    server(backend)
    cls = beam.beam_moe_class(quantized_experts=True)
    assert cls is beam.BeamVendorMoE
    with torch.device("meta"):
        layer = cls(
            _beam_geometry(), Mapping(rank=0, world_size=tp_size), 1, PREFIX, ct()
        )
    experts = layer.experts
    assert experts.plan["apply_kernel_name"] == FP8_CHANNEL_APPLY
    assert experts.plan.get("precision_policy", "default") == "default"
    assert experts.plan.get("output_scale", 1.0) == 1.0
    assert experts.supports_precomputed_topk
    assert not experts.support_routing and not experts.supports_deferred_finalize
    assert not layer.kernel_routing
    assert experts.w13_weight.dtype == torch.float8_e4m3fn
    assert experts.w2_weight.dtype == torch.float8_e4m3fn
    assert experts.w13_weight_scale.dtype == torch.float32
    assert experts.w13_weight_scale.shape == (256, 2 * 2560 // tp_size)
    assert experts.w2_weight_scale.shape == (256, 5120)


@pytest.mark.parametrize("backend", ["auto", "triton"])
def test_hopper_fp8_route_applies_the_scale_once(server, hopper_platform, backend):
    """Beam's FP32 routes carry the routed scale; the experts add none."""
    server(backend)
    layer = beam.beam_moe_class(quantized_experts=True)(
        tiny_config(), Mapping(rank=0, world_size=1), 1, PREFIX, ct()
    )
    layer.gate.e_score_correction_bias.data.zero_()
    logits = torch.arange(16, dtype=torch.float32).view(2, 8) / 10
    actual = layer.topk(torch.zeros(2, 128), logits)
    expected, ids = beam.beam_route_weights(
        logits, layer.gate.e_score_correction_bias, 2, True, 2, 1, None
    )
    assert actual.topk_weights.dtype == torch.float32
    assert torch.equal(actual.topk_ids, ids)
    assert torch.equal(actual.topk_weights, expected * 2.5)
    assert layer.topk.topk_config.routed_scaling_factor is None
    assert layer.experts.plan.get("output_scale", 1.0) == 1.0


def test_hopper_bf16_experts_keep_beam_moe(server, hopper_platform):
    server("auto")
    cls = beam.beam_moe_class(quantized_experts=False)
    assert cls is beam.BeamMoE
    layer = cls(tiny_config(), Mapping(rank=0, world_size=1), 1, PREFIX)
    assert layer.experts.plan["precision_policy"] == "bf16_route_accum_fp32"
    assert layer.experts.plan["apply_kernel_name"] == (
        "triton_bf16_precomputed_moe_apply"
    )


@pytest.mark.parametrize(
    "backend,quant_config,match",
    [
        ("auto", NVFP4, "no vendor MoE for nvfp4 experts on NVIDIA SM90"),
        ("triton", NVFP4, "no vendor MoE for nvfp4 experts on NVIDIA SM90"),
        ("flashinfer_trtllm", None, "no vendor MoE for BF16 experts on NVIDIA SM90"),
        ("auto", mxfp4_config(), "MXFP4 experts require AMD gfx950"),
    ],
    ids=["nvfp4-auto", "nvfp4-triton", "bf16-vendor", "mxfp4"],
)
def test_hopper_refuses_experts_without_a_kernel(
    server, hopper_platform, backend, quant_config, match
):
    server(backend)
    cls = beam.beam_moe_class(quantized_experts=quant_config is not None)
    assert cls is beam.BeamVendorMoE
    with pytest.raises(ValueError, match=match):
        cls(tiny_config(), Mapping(rank=0, world_size=1), 1, PREFIX, quant_config)


def test_hopper_fp8_checkpoint_keeps_expert_codes_and_scales(server, hopper_platform):
    """The experts keep the checkpoint's FP8 codes and FP32 channel scales;
    the dense FP8 linears, with no SM90 FP8 GEMM, are dequantized after
    loading, so attention keeps no fused FP8 projection."""
    server("auto")
    model = beam.BeamForCausalLM(tiny_config(), Mapping(rank=0, world_size=1), ct())
    checkpoint, _ = fp8_checkpoint()
    assert model.load_weights(checkpoint.items()) == checkpoint.keys()
    experts = model.model.layers[1].mlp.experts
    for expert in range(8):
        prefix = f"{PREFIX}.experts.{expert}"
        for half, projection in enumerate(("gate_proj", "up_proj")):
            rows = slice(half * 64, (half + 1) * 64)
            assert torch.equal(
                experts.w13_weight[expert, rows].view(torch.uint8),
                checkpoint[f"{prefix}.{projection}.weight"].view(torch.uint8),
            )
            assert torch.equal(
                experts.w13_weight_scale[expert, rows],
                checkpoint[f"{prefix}.{projection}.weight_scale"].float().flatten(),
            )
        assert torch.equal(
            experts.w2_weight[expert].view(torch.uint8),
            checkpoint[f"{prefix}.down_proj.weight"].view(torch.uint8),
        )
    _post_load(model)
    attn = model.model.layers[0].self_attn
    assert attn.qkv_gate_weight is None
    assert attn.qkv_proj.weight.dtype == torch.bfloat16
