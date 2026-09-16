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
    ).view(m, k)
    actual = tokenspeed_kernel.mm(
        fused_fp4,
        w2_fp4.T,
        A_scales=fused_scale,
        B_scales=w2_scale_swizzled.T,
        out_dtype=torch.bfloat16,
        alpha=fc2_alpha,
        quant="nvfp4",
    ).view(m, k)
    torch.cuda.synchronize()

    diff = (ref.float() - actual.float()).abs().flatten()
    assert diff.mean().item() < 0.03
    assert torch.quantile(diff, 0.99).item() < 0.12
    assert diff.max().item() < 0.25


def _autotune_operands(m: int, k: int, i: int):
    from tokenspeed_kernel.ops.gemm.cute_dsl import _round_up

    n = 2 * i
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(515)

    def packed(rows: int, cols: int):
        return torch.randint(
            0, 256, (rows, cols), dtype=torch.uint8, device=dev, generator=g
        )

    def scales(rows: int, cols: int):
        # 0x7F / 0xFF are e4m3 NaN; keep them out so results stay comparable.
        return torch.randint(
            1, 127, (rows, cols), dtype=torch.uint8, device=dev, generator=g
        ).view(torch.float8_e4m3fn)

    return [
        packed(m, k // 2),
        scales(_round_up(m, 128), _round_up(k // 16, 4)),
        packed(n, k // 2),
        scales(_round_up(n, 128), _round_up(k // 16, 4)),
        torch.ones(1, 1, dtype=torch.float32, device=dev),
        torch.ones(1, dtype=torch.float32, device=dev),
        torch.empty((m, i // 2), dtype=torch.uint8, device=dev),
        torch.empty(
            (_round_up(m, 128), _round_up(i // 16, 4)),
            dtype=torch.float8_e4m3fn,
            device=dev,
        ),
    ]


@pytest.mark.skipif(not _has_sm100(), reason="Blackwell SM100 CUDA GPU required")
@pytest.mark.parametrize(
    ("m", "k", "i"),
    [
        pytest.param(1, 7168, 512, id="shared_decode"),
        pytest.param(1024, 7168, 4608, id="dense_prefill"),
        pytest.param(32, 2048, 256, id="qwen3_5_a3b_tp2_dense"),
    ],
)
def test_nvfp4_gemm_swiglu_tactics_agree_with_heuristic(m: int, k: int, i: int) -> None:
    """Every tunable tactic must compute what the untuned fallback computes.

    Guards against tile shapes that ``can_implement()`` accepts but the kernel
    silently mis-computes -- the autotuner would otherwise pick one for being
    fast at doing less work.
    """
    from tokenspeed_kernel.ops.gemm.cute_dsl import (
        _Nvfp4GemmSwigluNvfp4QuantRunner,
    )

    runner = _Nvfp4GemmSwigluNvfp4QuantRunner.get(
        "float4_e2m1fn", "float8_e4m3fn", "float4_e2m1fn", 16, False, 3, True, False
    )
    inputs = _autotune_operands(m, k, i)
    out, out_scale = inputs[6], inputs[7]

    tactics = runner.get_valid_tactics(inputs, None)
    assert tactics, "no valid tactic for a production shape"
    n = 2 * i
    for tactic in tactics:
        # The persistent scheduler rounds the cluster grid up and never
        # bounds-checks per-CTA tile coords; a non-dividing cluster_n makes
        # trailing CTAs overwrite real output (garbage Qwen3.5 generations).
        (_, tile_n), (_, cluster_n) = tactic
        assert (
            n // tile_n
        ) % cluster_n == 0, f"tactic {tactic} must be filtered for n={n}"

    out.zero_()
    out_scale.zero_()
    runner.forward(inputs, tactic=-1)
    torch.cuda.synchronize()
    ref, ref_scale = out.clone(), out_scale.view(torch.uint8).clone()

    for tactic in tactics:
        out.zero_()
        out_scale.zero_()
        runner.forward(inputs, tactic=tactic)
        torch.cuda.synchronize()
        assert torch.equal(out, ref) and torch.equal(
            out_scale.view(torch.uint8), ref_scale
        ), f"tactic {tactic} disagrees with the heuristic fallback"


@pytest.mark.skipif(not _has_sm100(), reason="Blackwell SM100 CUDA GPU required")
def test_nvfp4_gemm_swiglu_autotune_populates_cache() -> None:
    """One call inside a tuning window fills every smaller shape bucket."""
    from flashinfer.autotuner import AutoTuner, autotune
    from tokenspeed_kernel.ops.gemm.cute_dsl import nvfp4_gemm_swiglu_nvfp4_quant

    m, k, i = 256, 7168, 512
    a, a_scale, b, b_scale, alpha, global_scale, _, _ = _autotune_operands(m, k, i)

    tuner = AutoTuner.get()
    before = sum(
        1 for key in tuner.profiling_cache if "nvfp4_gemm_swiglu" in key.custom_op
    )
    with autotune():
        tuned, tuned_scale = nvfp4_gemm_swiglu_nvfp4_quant(
            a, a_scale, b, b_scale, alpha, global_scale
        )
    torch.cuda.synchronize()
    after = sum(
        1 for key in tuner.profiling_cache if "nvfp4_gemm_swiglu" in key.custom_op
    )
    assert after > before + 1, "a tuning window must fill more than the observed bucket"

    untuned, untuned_scale = nvfp4_gemm_swiglu_nvfp4_quant(
        a, a_scale, b, b_scale, alpha, global_scale
    )
    torch.cuda.synchronize()
    assert torch.equal(tuned, untuned) and torch.equal(
        tuned_scale.view(torch.uint8), untuned_scale.view(torch.uint8)
    )
