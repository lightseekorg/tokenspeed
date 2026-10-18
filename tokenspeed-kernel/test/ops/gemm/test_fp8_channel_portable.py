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

"""Native W8A8 tests for Beam's dense shards; synthetic data, no model weights."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.ops.quantization import triton as quant_triton
from tokenspeed_kernel.selection import kernel_override
from utils import assert_no_triton_compile, kernel_supported

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not kernel_supported("triton_mm_fp8_scaled")
    or not kernel_supported("triton_quantize_fp8_per_token_portable"),
    reason="portable E4M3 tests require the scaled GEMM and token quantizer",
)


@pytest.fixture(autouse=True)
def portable_kernels():
    # Exercise these implementations even when a vendor specialization wins.
    with (
        kernel_override("gemm", "mm", "triton_mm_fp8_scaled"),
        kernel_override(
            "quantization", "fp8_with_scale", "triton_quantize_fp8_per_token_portable"
        ),
    ):
        yield


def _reference_quantize(x):
    # Use double precision to independently round the two IEEE FP32 divisions.
    amax = x.float().abs().amax(-1, keepdim=True)
    scales = (amax.double() / 448.0).float().clamp_min(1.0 / (448.0 * 512.0))
    multiplier = (448.0 / amax.double()).float().clamp_max(448.0 * 512.0)
    codes = (x.float() * multiplier).clamp(-448, 448).to(torch.float8_e4m3fn)
    return codes, scales


@pytest.mark.parametrize("m", [0, 1, 2, 4, 8, 16, 65, 129])
@pytest.mark.parametrize("k", [320, 640, 2560, 5120])
def test_per_token_codes_match_independently_rounded_reference(m, k):
    torch.manual_seed(193)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    if m:
        x[0].zero_()
    if m > 1:
        x[1].mul_(2**-20)
    if m > 2:
        x[2].mul_(128)
    codes, scales = quantize_fp8(x, granularity="token", enable_pdl=False)
    expected_codes, expected_scales = _reference_quantize(x)
    torch.testing.assert_close(
        codes.view(torch.uint8), expected_codes.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)


@pytest.mark.parametrize("m", [0, 1, 2, 4, 8, 16, 65, 129])
@pytest.mark.parametrize(
    "n,k", [(1792, 5120), (5120, 768), (5120, 5120), (5120, 2560), (37, 333)]
)
def test_dense_beam_shards_match_fp32_accumulator_then_scale(m, n, k):
    torch.manual_seed(194)
    x = torch.randint(-8, 9, (m, k), device="cuda").to(torch.bfloat16)
    if m:
        x[:, 0] = 448
        x *= 2.0 ** (torch.arange(m, device="cuda").remainder(9) - 4).reshape(-1, 1)
    weight = torch.randint(-8, 9, (n, k), device="cuda").to(torch.float8_e4m3fn)
    # Integer FP8 products accumulate exactly in FP32 at these K sizes. This
    # isolates the two scale multiplications and final BF16 rounding.
    scales = torch.linspace(0.001, 0.01, n, device="cuda").reshape(n, 1)
    out = torch.empty(m, n, device="cuda", dtype=x.dtype)
    result = mm(x, weight.t(), B_scales=scales, out_dtype=x.dtype, quant="fp8", out=out)
    codes, token_scales = _reference_quantize(x)
    acc = (codes.double() @ weight.double().t()).float()
    expected = ((acc * token_scales) * scales.t()).to(torch.bfloat16)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result is out


def test_token_quantizer_reuses_compilation_for_changing_row_count():
    x = torch.randn(17, 5120, device="cuda", dtype=torch.bfloat16)
    quantize_fp8(x[:1], granularity="token", enable_pdl=False)
    with assert_no_triton_compile(quant_triton._fp8_token_quantize_kernel):
        for rows in (2, 4, 8, 16, 17):
            quantize_fp8(x[:rows], granularity="token", enable_pdl=False)


def test_all_finite_e4m3_products_have_exact_native_decode():
    bits = torch.arange(256, device="cuda", dtype=torch.uint8)
    bits = bits[torch.isfinite(bits.view(torch.float8_e4m3fn).float())]
    q = torch.zeros(bits.numel(), 256, device="cuda", dtype=torch.float8_e4m3fn)
    q.view(torch.uint8)[:, 0].copy_(bits)
    ones = torch.ones(bits.numel(), 1, device="cuda")
    actual = mm(
        q, q.t(), A_scales=ones, B_scales=ones, out_dtype=torch.float32, quant="fp8"
    )
    expected = (q.double() @ q.double().t()).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_random_dense_output_matches_epilogue_of_native_fp32_accumulator():
    torch.manual_seed(194)
    m, n, k = 2, 1792, 5120
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-8, 9, (n, k), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.linspace(0.001, 0.01, n, device="cuda").reshape(n, 1)
    codes, token_scales = _reference_quantize(x)
    raw = mm(
        codes,
        weight.t(),
        A_scales=torch.ones(m, 1, device="cuda"),
        B_scales=torch.ones(n, 1, device="cuda"),
        out_dtype=torch.float32,
        quant="fp8",
    )
    actual = mm(x, weight.t(), B_scales=scales, out_dtype=x.dtype, quant="fp8")
    expected = ((raw * token_scales) * scales.t()).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", [1, 16, 65])
def test_dense_graph_replay_matches_eager_after_input_changes(m):
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1792, 5120, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.full((1792, 1), 0.01, device="cuda")
    out = torch.empty(m, 1792, device="cuda", dtype=torch.bfloat16)

    def run():
        return mm(
            x, weight.t(), B_scales=scales, out_dtype=x.dtype, quant="fp8", out=out
        )

    # Warm the same path before capture; the output allocation belongs to the
    # caller and the captured quantization buffers belong to the graph pool.
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for magnitude in (0.0, 0.001, 1.0, 16.0):
        x.normal_().mul_(magnitude)
        expected = run().clone()
        out.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
