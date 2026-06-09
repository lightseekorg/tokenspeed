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

import math

import pytest
import torch


def _has_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _scale_inv_for(tensor: torch.Tensor) -> torch.Tensor:
    scale = (tensor.detach().abs().amax().to(torch.float32) / (448.0 * 6.0)).clamp(
        min=1e-8
    )
    return (1.0 / scale).view(1)


def test_interleave_linear_and_gate_layout() -> None:
    from tokenspeed.runtime.layers.dense.nvfp4 import interleave_linear_and_gate

    linear = torch.arange(128 * 4, dtype=torch.uint8).reshape(128, 4)
    gate = torch.arange(128 * 4, 256 * 4, dtype=torch.uint8).reshape(128, 4)
    actual = interleave_linear_and_gate(
        torch.cat([linear, gate], dim=0),
        group_size=64,
        dim=0,
    )

    expected = torch.cat(
        [
            linear[:64],
            gate[:64],
            linear[64:128],
            gate[64:128],
        ],
        dim=0,
    )
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not _has_sm100(), reason="Blackwell SM100 CUDA GPU required")
def test_nvfp4_process_weights_releases_normal_fc1_tensors() -> None:
    from torch import nn
    from torch.nn.parameter import Parameter

    from tokenspeed.runtime.layers.dense.nvfp4 import Nvfp4LinearMethod

    class QuantConfig:
        group_size = 16

    layer = nn.Module()
    layer.prefix = "test.gate_up_proj"
    layer.interleave_linear_and_gate = True
    layer.weight = Parameter(
        torch.randint(0, 256, (256, 8), device="cuda", dtype=torch.uint8),
        requires_grad=False,
    )
    layer.weight_scale = Parameter(
        torch.empty((256, 4), device="cuda", dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.input_scale = Parameter(
        torch.tensor([2.0], device="cuda", dtype=torch.float32),
        requires_grad=False,
    )
    layer.weight_scale_2 = Parameter(
        torch.tensor([3.0], device="cuda", dtype=torch.float32),
        requires_grad=False,
    )

    Nvfp4LinearMethod(QuantConfig()).process_weights_after_loading(layer)

    assert not hasattr(layer, "weight")
    assert not hasattr(layer, "weight_scale")
    assert not hasattr(layer, "weight_scale_interleaved")
    assert hasattr(layer, "weight_swiglu_interleaved")
    assert hasattr(layer, "weight_scale_swiglu_interleaved")


@pytest.mark.skipif(not _has_sm100(), reason="Blackwell SM100 CUDA GPU required")
def test_nvfp4_process_weights_releases_normal_weight_scale() -> None:
    from torch import nn
    from torch.nn.parameter import Parameter

    from tokenspeed.runtime.layers.dense.nvfp4 import Nvfp4LinearMethod

    class QuantConfig:
        group_size = 16

    layer = nn.Module()
    layer.prefix = "test.down_proj"
    layer.interleave_linear_and_gate = False
    layer.weight = Parameter(
        torch.randint(0, 256, (128, 8), device="cuda", dtype=torch.uint8),
        requires_grad=False,
    )
    layer.weight_scale = Parameter(
        torch.empty((128, 4), device="cuda", dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.input_scale = Parameter(
        torch.tensor([2.0], device="cuda", dtype=torch.float32),
        requires_grad=False,
    )
    layer.weight_scale_2 = Parameter(
        torch.tensor([3.0], device="cuda", dtype=torch.float32),
        requires_grad=False,
    )

    Nvfp4LinearMethod(QuantConfig()).process_weights_after_loading(layer)

    assert hasattr(layer, "weight")
    assert not hasattr(layer, "weight_scale")
    assert hasattr(layer, "weight_scale_interleaved")
    assert not hasattr(layer, "weight_swiglu_interleaved")
    assert not hasattr(layer, "weight_scale_swiglu_interleaved")


@pytest.mark.skipif(not _has_sm100(), reason="Blackwell SM100 CUDA GPU required")
@pytest.mark.parametrize(
    ("m", "k", "i"),
    [
        pytest.param(1, 7168, 512, id="deepseek_v3_kimi_k25_tp4_shared_decode"),
        pytest.param(128, 7168, 512, id="deepseek_v3_kimi_k25_tp4_shared_prefill"),
        pytest.param(1, 7168, 4608, id="deepseek_v3_kimi_k25_tp4_dense_decode"),
        pytest.param(128, 7168, 4608, id="deepseek_v3_kimi_k25_tp4_dense_prefill"),
    ],
)
def test_nvfp4_gemm_swiglu_nvfp4_quant_matches_unfused_model_shapes(
    m: int,
    k: int,
    i: int,
) -> None:
    import tokenspeed_kernel
    from tokenspeed_kernel.ops.gemm.cute_dsl import (
        nvfp4_gemm_swiglu_nvfp4_quant,
    )
    from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize
    from tokenspeed_kernel.registry import load_builtin_kernels
    from tokenspeed_kernel.thirdparty.cuda import silu_and_mul_fuse_nvfp4_quant

    from tokenspeed.runtime.layers.dense.nvfp4 import (
        interleave_linear_and_gate,
        swizzle_blockscale_2d,
    )

    load_builtin_kernels()

    torch.manual_seed(1000 + m + i)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w1 = (
        torch.randn(2 * i, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    ).contiguous()
    w2 = (
        torch.randn(k, i, device="cuda", dtype=torch.bfloat16) / math.sqrt(i)
    ).contiguous()

    x_scale_inv = _scale_inv_for(x)
    w1_scale_inv = _scale_inv_for(w1)
    w2_scale_inv = _scale_inv_for(w2)

    x_fp4, x_scale = fp4_quantize(x, x_scale_inv, enable_pdl=True)
    w1_fp4, w1_scale = fp4_quantize(
        w1,
        w1_scale_inv,
        is_sf_swizzled_layout=False,
        enable_pdl=True,
    )
    w2_fp4, w2_scale = fp4_quantize(
        w2,
        w2_scale_inv,
        is_sf_swizzled_layout=False,
        enable_pdl=True,
    )
    w1_scale_swizzled = swizzle_blockscale_2d(w1_scale)
    w2_scale_swizzled = swizzle_blockscale_2d(w2_scale)

    fc1_alpha = (1.0 / x_scale_inv) * (1.0 / w1_scale_inv)
    gate_up = tokenspeed_kernel.mm(
        x_fp4,
        w1_fp4.T,
        A_scales=x_scale,
        B_scales=w1_scale_swizzled.T,
        out_dtype=torch.bfloat16,
        alpha=fc1_alpha,
        quant="nvfp4",
        enable_pdl=True,
        expected_kernel_name="flashinfer_mm_nvfp4",
    ).view(m, 2 * i)

    silu_out = (
        torch.nn.functional.silu(gate_up[:, :i].float()) * gate_up[:, i:].float()
    ).to(torch.bfloat16)
    down_input_scale_inv = _scale_inv_for(silu_out)
    ref_fp4, ref_scale = silu_and_mul_fuse_nvfp4_quant(
        gate_up,
        down_input_scale_inv,
        enable_pdl=True,
    )

    gate_fp4, linear_fp4 = w1_fp4.chunk(2, dim=0)
    linear_gate_fp4 = torch.cat((linear_fp4, gate_fp4), dim=0)
    gate_scale, linear_scale = w1_scale.chunk(2, dim=0)
    linear_gate_scale = torch.cat((linear_scale, gate_scale), dim=0)

    fused_fp4, fused_scale = nvfp4_gemm_swiglu_nvfp4_quant(
        x_fp4,
        x_scale,
        interleave_linear_and_gate(linear_gate_fp4, group_size=64, dim=0),
        swizzle_blockscale_2d(
            interleave_linear_and_gate(linear_gate_scale, group_size=64, dim=0)
        ),
        fc1_alpha,
        down_input_scale_inv,
        enable_pdl=True,
    )

    fc2_alpha = (1.0 / down_input_scale_inv) * (1.0 / w2_scale_inv)
    ref = tokenspeed_kernel.mm(
        ref_fp4,
        w2_fp4.T,
        A_scales=ref_scale,
        B_scales=w2_scale_swizzled.T,
        out_dtype=torch.bfloat16,
        alpha=fc2_alpha,
        quant="nvfp4",
        enable_pdl=True,
        expected_kernel_name="flashinfer_mm_nvfp4",
    ).view(m, k)
    actual = tokenspeed_kernel.mm(
        fused_fp4,
        w2_fp4.T,
        A_scales=fused_scale,
        B_scales=w2_scale_swizzled.T,
        out_dtype=torch.bfloat16,
        alpha=fc2_alpha,
        quant="nvfp4",
        enable_pdl=True,
        expected_kernel_name="flashinfer_mm_nvfp4",
    ).view(m, k)
    torch.cuda.synchronize()

    diff = (ref.float() - actual.float()).abs().flatten()
    assert diff.mean().item() < 0.03
    assert torch.quantile(diff, 0.99).item() < 0.12
    assert diff.max().item() < 0.25
