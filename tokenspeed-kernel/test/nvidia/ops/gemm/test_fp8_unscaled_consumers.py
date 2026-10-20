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

"""An FP8 GEMM run with unit scales, and the kernels that apply its scales.

An FP8 W8A8 linear with per-token activation and per-channel weight scales
can run on the registry's per-tensor FP8 GEMM (cuBLASLt) with both scales 1:
its BF16 output is the codes' product rounded once. The next kernel that reads
that output applies ``token_scale * channel_scale`` (``fp8_dequantize_unscaled``)
before its own work: the SwiGLU with the down projection's per-token
quantization (``silu_and_mul_quantize_fp8``), and Beam's Q/K norms with the
V and output gate columns (``fused_qk_rmsnorm_rope`` / ``qk_rmsnorm_ssmax`` with
``dequant=``).

Range: ``|code| <= 448``, so ``|sum| <= 448^2 * K`` (8.2e8 at K = 4,096), far
inside BF16's range; the unscaled output's rounding is relative (2^-9), as the
scaled GEMM's own output rounding is.
"""

import pytest
import torch
from tokenspeed_kernel.ops.activation.triton import silu_and_mul
from tokenspeed_kernel.ops.gemm import mm as kernel_mm
from tokenspeed_kernel.ops.gemm import triton_fp8_channel
from tokenspeed_kernel.ops.layernorm.qk_rmsnorm_ssmax import (
    TORCH_QK_RMSNORM_SSMAX,
    TRITON_FUSED_QK_RMSNORM_SSMAX,
    qk_rmsnorm_ssmax,
)
from tokenspeed_kernel.ops.layernorm.triton import (
    UnscaledFP8QKVGate,
    fused_qk_rmsnorm_rope,
)
from tokenspeed_kernel.ops.quantization import silu_and_mul_quantize_fp8
from tokenspeed_kernel.ops.quantization import triton as quant_triton
from tokenspeed_kernel.ops.quantization.triton import (
    dequantize_unscaled_fp8,
    fp8_quantize_per_token,
)
from tokenspeed_kernel.platform import pdl_enabled
from utils import assert_no_triton_compile, kernel_supported

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not kernel_supported("triton_mm_fp8_channel_small_m"),
    reason="the FP8 dense decode path is registered for SM100 to SM103 (GPU only)",
)

FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
# Beam's TP8 attention shard: 6 query heads, 1 KV head, head_dim 128.
Q_HEADS, KV_HEADS, HEAD_DIM = 6, 1, 128
QW, KVW = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM


def _weight(n, k, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=generator) / k**0.5
    scales = w.abs().amax(1, keepdim=True) / 448.0
    return (w / scales).clamp(-448, 448).to(FP8), scales


def _rows(m, k, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=generator)
    x[:, :: max(1, k // 7)] *= 20
    return x.to(BF16)


def _unscaled(a8, w8):
    unit = torch.ones(1, device="cuda")
    return kernel_mm(
        a8, w8.t(), A_scales=unit, B_scales=unit, out_dtype=BF16, quant="fp8"
    )


def _rel(out, ref):
    return ((out.double() - ref).norm() / ref.norm()).item()


# (N, K): Llama-3.1-8B shards at 8-way tensor parallelism (fused Q/K/V,
# output, gate/up, down), and Beam's TP8 shards (fused Q/K/V + output gate,
# dense gate/up, down, output).
@pytest.mark.parametrize(
    "n,k",
    [
        (768, 4096),
        (4096, 512),
        (3584, 4096),
        (4096, 1792),
        (1792, 5120),
        (5120, 5120),
        (5120, 2560),
        (5120, 768),
    ],
)
@pytest.mark.parametrize("m", [1, 4, 17, 64])
def test_the_unit_scale_gemm_is_the_rounded_code_product(m, n, k):
    w8, _ = _weight(n, k, seed=n + k)
    a8, _ = fp8_quantize_per_token(_rows(m, k, seed=m))
    out = _unscaled(a8, w8)
    exact = a8.double() @ w8.double().t()
    magnitude = a8.double().abs() @ w8.double().abs().t()
    ulp = torch.where(
        exact == 0,
        torch.zeros_like(exact),
        torch.exp2(torch.floor(torch.log2(exact.abs().clamp_min(1e-300))) - 7),
    )
    excess = (out.double() - exact).abs() - ((k + 2) * 2.0**-24 * magnitude + ulp)
    assert excess.max().item() <= 0, f"worst excess {excess.max().item():.3e}"


def test_the_largest_code_product_fits_bf16():
    k = 4096
    a8 = torch.full((2, k), 448.0, device="cuda").to(FP8)
    w8 = torch.full((16, k), -448.0, device="cuda").to(FP8)
    out = _unscaled(a8, w8)
    # 448^2 * 4096 = 2^24 * 49: exact in FP32 sums and in BF16.
    assert bool((out.float() == -(448.0**2) * k).all())


# 2,560 and 320: Beam's dense and expert intermediate size per TP8 rank.
@pytest.mark.parametrize("width", [2048, 192, 2560, 320])
@pytest.mark.parametrize("m", [1, 7, 64])
def test_silu_and_mul_dequantizes_then_quantizes_per_token(m, width):
    k = 4096
    w8, w_scales = _weight(2 * width, k, seed=width)
    a8, a_scales = fp8_quantize_per_token(_rows(m, k, seed=20 + m))
    y = _unscaled(a8, w8)
    codes, scales = silu_and_mul_quantize_fp8(
        y, a_scales, w_scales, enable_pdl=pdl_enabled()
    )
    # The statements it replaces: the scaled GEMM's BF16 output, SwiGLU in
    # FP32 rounded to BF16, per-token quantization of that.
    scaled = dequantize_unscaled_fp8(y, a_scales, w_scales)
    gate, up = scaled.float().split(width, -1)
    act = (gate * torch.sigmoid(gate) * up).to(BF16)
    expected_codes, expected_scales = fp8_quantize_per_token(act)
    torch.testing.assert_close(scales, expected_scales, rtol=2**-7, atol=0)
    steps = (
        codes.view(torch.uint8).int() - expected_codes.view(torch.uint8).int()
    ).abs()
    assert steps.max().item() <= 1 and (steps > 0).float().mean().item() <= 1e-3
    # FP64: the same error as the scaled path's (scaled small-M GEMM, then
    # silu_and_mul, then the quantizer) against the exact SwiGLU.
    exact = (a8.double() * a_scales.double()) @ (w8.double() * w_scales.double()).t()
    g64, u64 = exact.split(width, -1)
    reference = torch.nn.functional.silu(g64) * u64
    ours = _rel(codes.double() * scales.double(), reference)
    out = triton_fp8_channel.triton_mm_fp8_channel_small_m(
        a8, w8.t(), a_scales, w_scales, BF16
    )
    base_codes, base_scales = fp8_quantize_per_token(silu_and_mul(out))
    base = _rel(base_codes.double() * base_scales.double(), reference)
    print(
        f"M={m} I={width}: rel vs FP64 unscaled path {ours:.5f} scaled path {base:.5f}"
    )
    assert ours <= 1.1 * base + 1e-3


def test_row_counts_do_not_recompile():
    """One compiled binary serves every row count: the width and the strides
    are runtime arguments and only the grid follows M."""
    width, k = 2048, 4096
    w8, w_scales = _weight(2 * width, k, seed=40)

    def run(m):
        a8, a_scales = fp8_quantize_per_token(_rows(m, k, seed=40 + m))
        silu_and_mul_quantize_fp8(
            _unscaled(a8, w8), a_scales, w_scales, enable_pdl=pdl_enabled()
        )

    run(1)
    with assert_no_triton_compile(quant_triton._silu_and_mul_quantize_fp8_kernel):
        for m in (7, 16, 64):
            run(m)


def _qkvg(m, seed):
    k = 5120
    w8, w_scales = _weight(2 * QW + 2 * KVW, k, seed=seed)
    a8, a_scales = fp8_quantize_per_token(_rows(m, k, seed=seed + m))
    raw = _unscaled(a8, w8)
    q, k_, v, gate = raw.split([QW, KVW, KVW, QW], -1)
    sq, sk, sv, sg = w_scales.split([QW, KVW, KVW, QW])
    dequant = UnscaledFP8QKVGate(
        token_scales=a_scales,
        q_scales=sq,
        k_scales=sk,
        v=v,
        v_scales=sv,
        gate=gate,
        gate_scales=sg,
        v_out=torch.empty(m, KVW, device="cuda", dtype=BF16),
        gate_out=torch.empty(m, QW, device="cuda", dtype=BF16),
    )
    scaled = dequantize_unscaled_fp8(raw, a_scales, w_scales)
    return q, k_, dequant, scaled.split([QW, KVW, KVW, QW], -1)


def _norm_weights():
    generator = torch.Generator(device="cuda").manual_seed(5)
    return [
        (1 + 0.1 * torch.randn(HEAD_DIM, device="cuda", generator=generator)).to(BF16)
        for _ in range(2)
    ]


@pytest.mark.parametrize("m", [1, 5, 64])
@pytest.mark.parametrize(
    "kernel",
    ["fused_qk_rmsnorm_rope", TRITON_FUSED_QK_RMSNORM_SSMAX, TORCH_QK_RMSNORM_SSMAX],
)
def test_the_qk_norms_dequantize_first(m, kernel):
    """Bit for bit the kernel on the scaled GEMM's BF16 output (the PyTorch
    dequantization), and V and the output gate are that output's columns."""
    q, k, dequant, (sq, sk, sv, sg) = _qkvg(m, seed=30)
    q_weight, k_weight = _norm_weights()
    positions = torch.arange(100, 100 + m, device="cuda", dtype=torch.int64)
    call = dict(solution=None, override=kernel)
    if kernel == "fused_qk_rmsnorm_rope":
        inv = 1.0 / (
            10000.0 ** (torch.arange(0, HEAD_DIM, 2, device="cuda") / HEAD_DIM)
        )
        freqs = torch.outer(torch.arange(4096, device="cuda").float(), inv)
        cache = torch.cat([freqs.cos(), freqs.sin()], -1)
        args = (q_weight, k_weight, cache, positions, 1e-6, Q_HEADS, KV_HEADS, HEAD_DIM)
        got = fused_qk_rmsnorm_rope(q, k, *args, dequant=dequant)
        expected = fused_qk_rmsnorm_rope(sq, sk, *args)
    else:
        got = qk_rmsnorm_ssmax(
            q, k, q_weight, k_weight, 1e-6, positions, dequant=dequant, **call
        )
        expected = qk_rmsnorm_ssmax(sq, sk, q_weight, k_weight, 1e-6, positions, **call)
    assert torch.equal(got[0], expected[0]) and torch.equal(got[1], expected[1])
    assert torch.equal(dequant.v_out, sv) and torch.equal(dequant.gate_out, sg)
