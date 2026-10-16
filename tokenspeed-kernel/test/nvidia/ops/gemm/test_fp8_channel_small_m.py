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

"""The small-M FP8 GEMM and its per-token FP8 quantizer.

Every GEMM result is held to the FP64 product of the same FP8 codes and FP32
scales: ``|out - exact| <= (K + 2) * 2^-24 * sum|a||w| * |sa| * sb`` (FP32
accumulation of exact FP8 products, two scale multiplies) plus one ulp of the
16-bit output. Quantized activations are compared bit for bit.
"""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import mm as kernel_mm
from tokenspeed_kernel.ops.gemm import triton_fp8_channel
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.ops.quantization import triton as quant_triton
from tokenspeed_kernel.ops.quantization.triton import fp8_quantize_per_token
from utils import assert_no_triton_compile, kernel_supported

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not kernel_supported("triton_mm_fp8_channel_small_m"),
    reason="the small-M FP8 GEMM is registered for SM100 to SM103 (GPU only)",
)

FP8 = torch.float8_e4m3fn
# (N, K) 8-way tensor-parallel shards of public configs' dense projections:
# Llama-3.1-8B fused Q/K/V, output, gate/up and down; Qwen3-30B-A3B fused
# Q/K/V and output; Llama-3.3-70B down. They cover both weight-row tiles
# (N up to 2,048 and above, N = 2,048 itself) and every K tile (512 for K a
# multiple of 512 from 2,048 on, K = 2,048 itself; 256 for K = 512 and 1,792).
# (200, 384) adds a partial last weight block and K tiles of 128. Beam's TP8
# rank shards (Q/K/V + output gate, output projection, dense gate/up, dense
# down) take their measured ``_SMALL_M_TILES`` entries.
SHAPES = [
    (768, 4096),
    (4096, 512),
    (3584, 4096),
    (4096, 1792),
    (768, 2048),
    (2048, 512),
    (8192, 3584),
    (200, 384),
    (1792, 5120),
    (5120, 768),
    (5120, 5120),
    (5120, 2560),
]


def _weight(n, k, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=generator) / k**0.5
    scales = w.abs().amax(1, keepdim=True) / 448.0
    return (w / scales).clamp(-448, 448).to(FP8), scales


def _rows(m, k, seed, dtype=torch.bfloat16):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=generator)
    # A few large outliers per row, as hidden states carry.
    x[:, :: max(1, k // 7)] *= 20
    return x.to(dtype)


def _assert_within_bound(out, a8, a_scales, w8, w_scales):
    sa = a_scales.double().reshape(-1, 1).expand(a8.shape[0], 1)
    sb = w_scales.double().reshape(1, -1).expand(1, w8.shape[0])
    exact = (a8.double() @ w8.double().t()) * sa * sb
    magnitude = (a8.double().abs() @ w8.double().abs().t()) * sa.abs() * sb
    mantissa = 7 if out.dtype == torch.bfloat16 else 10
    ulp = torch.where(
        exact == 0,
        torch.zeros_like(exact),
        torch.exp2(torch.floor(torch.log2(exact.abs().clamp_min(1e-300))) - mantissa),
    )
    limit = (a8.shape[1] + 2) * 2.0**-24 * magnitude + ulp
    excess = (out.double() - exact).abs() - limit
    assert excess.max().item() <= 0, f"worst excess {excess.max().item():.3e}"


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 2, 8, 9, 16, 17, 33, 64])
def test_fp8_rows_match_the_fp64_product(m, n, k):
    w8, w_scales = _weight(n, k, seed=n + k)
    a8, a_scales = fp8_quantize_per_token(_rows(m, k, seed=m))
    out = triton_fp8_channel.triton_mm_fp8_channel_small_m(
        a8, w8.t(), a_scales, w_scales, torch.bfloat16
    )
    _assert_within_bound(out, a8, a_scales, w8, w_scales)


@pytest.mark.parametrize("a_layout", ["M1", "M", "tensor"])
@pytest.mark.parametrize("b_layout", ["N1", "tensor"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_scale_layouts(a_layout, b_layout, out_dtype):
    m, n, k = 12, 768, 4096
    w8, w_scales = _weight(n, k, seed=1)
    if b_layout == "tensor":
        w_scales = w_scales.max().reshape(1)
    x = _rows(m, k, seed=2)
    if a_layout == "tensor":
        a_scales = (x.float().abs().amax() / 448.0).reshape(1, 1)
        a8 = (x.float() / a_scales).clamp(-448, 448).to(FP8)
    else:
        a8, a_scales = fp8_quantize_per_token(x)
        if a_layout == "M":
            a_scales = a_scales.reshape(-1)
    out = triton_fp8_channel.triton_mm_fp8_channel_small_m(
        a8, w8.t(), a_scales, w_scales, out_dtype
    )
    _assert_within_bound(out, a8, a_scales, w8, w_scales)


def test_the_quantizer_follows_trtllms_per_token_rule():
    """quantize_fp8's token quantizer up to 64 rows (fp8_quantize_per_token):
    scale max(amax / 448, 1 / (448 * 512)), codes RN(x * min(448 / amax,
    448 * 512)) saturated, in FP32 with IEEE divisions; a zero row is zeros."""
    x = _rows(9, 4096, seed=5)
    x[3] = 0
    codes, scales = quantize_fp8(x, granularity="token")
    amax = x.float().abs().amax(1, keepdim=True)
    # IEEE divisions by tensors: on CUDA a division by a Python scalar
    # (amax / 448.0) multiplies by its rounded reciprocal, and 448.0 / amax
    # likewise; either can differ from the quantizer's div_rn by one ulp.
    expected_scales = (amax / torch.full_like(amax, 448.0)).clamp_min(
        1.0 / (448.0 * 512.0)
    )
    multiplier = (torch.full_like(amax, 448.0) / amax).clamp_max(448.0 * 512.0)
    expected = (x.float() * multiplier).clamp(-448, 448).to(FP8)
    assert torch.equal(scales, expected_scales)
    assert torch.equal(codes.view(torch.uint8), expected.view(torch.uint8))
    assert not codes[3].float().any()


def test_the_quantizer_agrees_with_trtllm():
    """TRT-LLM's quantize_e4m3_activation (quantize_fp8's token quantizer, the
    path above 64 rows) on the same rows. Its scales come back in the input dtype (BF16
    here): within one BF16 rounding (2^-8 relative) of the FP32 scales, which
    a GB200 run found 60 of 64 rows off by up to 0.33%. Both
    dequantize the rows within half an E4M3 step (2^-4 relative to amax),
    TRT-LLM's plus its scale's BF16 rounding."""
    x = _rows(64, 4096, seed=6)
    codes, scales = fp8_quantize_per_token(x)
    trt_codes, trt_scales = quantize_fp8(x, granularity="token", solution="trtllm")
    print(
        f"TRT-LLM scale dtype before the FP32 conversion: "
        f"{torch.ops.tensorrt_llm.quantize_e4m3_activation(x)[1].dtype}"
    )
    torch.testing.assert_close(trt_scales, scales, rtol=2**-8, atol=0)
    amax = x.float().abs().amax(1, keepdim=True)
    for name, (c, s) in {
        "triton": (codes, scales),
        "trtllm": (trt_codes, trt_scales),
    }.items():
        error = ((c.float() * s - x.float()).abs() / amax).max().item()
        print(f"{name}: max |dequant - x| / amax = {error:.3e}")
        assert error <= 2.0**-4 + 2.0**-8, (name, error)


def test_out_buffer_and_strided_rows():
    m, n, k = 5, 768, 4096
    w8, w_scales = _weight(n, k, seed=7)
    wide = torch.zeros(m, k + 256, device="cuda", dtype=FP8)
    a8, a_scales = fp8_quantize_per_token(_rows(m, k, seed=8))
    wide[:, 128 : 128 + k] = a8
    out_wide = torch.zeros(m, n + 64, device="cuda", dtype=torch.bfloat16)
    result = triton_fp8_channel.triton_mm_fp8_channel_small_m(
        wide[:, 128 : 128 + k],
        w8.t(),
        a_scales,
        w_scales,
        torch.bfloat16,
        out=out_wide[:, :n],
    )
    assert result.data_ptr() == out_wide.data_ptr()
    _assert_within_bound(out_wide[:, :n], a8, a_scales, w8, w_scales)
    assert bool((out_wide[:, n:] == 0).all())


@pytest.mark.parametrize("m", [1, 9, 64, 65])
def test_mm_quantizes_online(m):
    """tokenspeed_kernel.ops.gemm.mm with 16-bit rows and no A scales: the Triton
    quantizer and the small-M GEMM up to 64 rows (exactly the GEMM of
    fp8_quantize_per_token's codes), another kernel after the TRT-LLM quantizer
    from 65; both within FP8_DYNAMIC's activation error of the FP64 product."""
    n, k = 768, 4096
    w8, w_scales = _weight(n, k, seed=9)
    x = _rows(m, k, seed=10)
    out = kernel_mm(x, w8.t(), B_scales=w_scales, out_dtype=torch.bfloat16, quant="fp8")
    if m <= 64:
        a8, a_scales = fp8_quantize_per_token(x)
        _assert_within_bound(out, a8, a_scales, w8, w_scales)
    exact = x.double() @ (w8.double() * w_scales.double()).t()
    rel = ((out.double() - exact).norm() / exact.norm()).item()
    assert rel < 0.06, rel


def test_row_counts_do_not_recompile():
    n, k = 768, 4096
    w8, w_scales = _weight(n, k, seed=11)

    def run(m):
        a8, a_scales = quantize_fp8(_rows(m, k, seed=m), granularity="token")
        triton_fp8_channel.triton_mm_fp8_channel_small_m(
            a8, w8.t(), a_scales, w_scales, torch.bfloat16
        )

    for m in (1, 16, 17, 32, 33, 64):
        run(m)
    with assert_no_triton_compile(
        triton_fp8_channel._fp8_channel_small_m_kernel,
        quant_triton._fp8_token_quantize_kernel,
    ):
        for m in (2, 3, 5, 7, 8, 10, 15, 18, 31, 40, 63):
            run(m)
