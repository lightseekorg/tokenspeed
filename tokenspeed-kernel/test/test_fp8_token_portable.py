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

"""CPU behavior of portable FP8 dispatch; only GPU execution is emulated."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.gemm import triton as gemm_triton
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.ops.quantization import triton as quant_triton
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry


@pytest.fixture
def amd(mi350_platform):
    original = Platform.get()
    Platform.override(mi350_platform)
    KernelRegistry.get().clear_cache()
    yield
    Platform.override(original)
    KernelRegistry.get().clear_cache()


def _quantize_cpu(x, enable_pdl):
    assert not enable_pdl
    amax = x.float().abs().amax(-1, keepdim=True)
    scales = (amax / 448.0).clamp_min(1.0 / (448.0 * 512.0))
    multiplier = (448.0 / amax).clamp_max(448.0 * 512.0)
    return (x.float() * multiplier).clamp(-448, 448).to(torch.float8_e4m3fn), scales


@pytest.mark.parametrize("m", [1, 16, 64, 65, 257])
def test_online_dense_quantization_preserves_token_and_channel_scales(
    amd, monkeypatch, m
):
    # CPU execution replaces unavailable GPU bodies, while public operation
    # selection and argument handling are exercised unchanged.
    monkeypatch.setattr(quant_triton, "fp8_quantize_per_token", _quantize_cpu)

    def scaled_cpu(
        a, b, scale_a, scale_b, out_dtype, bias, out, fp8_accumulation_streams
    ):
        value = a.float() @ b.float()
        value = value * scale_a.reshape(-1, 1)
        value = value * scale_b.reshape(1, -1)
        value = value.to(out_dtype)
        if out is not None:
            out.copy_(value)
            return out
        return value

    monkeypatch.setattr(gemm_triton, "triton_scaled_mm", scaled_cpu)
    x = torch.linspace(-2, 2, m * 32).reshape(m, 32).to(torch.bfloat16)
    x[0].zero_()
    weight = torch.arange(7 * 32).reshape(7, 32).remainder(9).to(torch.float8_e4m3fn)
    scales = torch.linspace(0.001, 0.1, 7).reshape(7, 1)
    out = torch.empty(m, 7, dtype=torch.bfloat16)
    result = mm(x, weight.t(), B_scales=scales, out_dtype=x.dtype, quant="fp8", out=out)
    codes, token_scales = _quantize_cpu(x, enable_pdl=False)
    expected = (codes.float() @ weight.float().t()) * token_scales
    expected = (expected * scales.t()).to(x.dtype)
    assert result is out
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_empty_quantization_and_dense_output_need_no_gpu(amd):
    x = torch.empty(0, 5120, dtype=torch.bfloat16)
    codes, scales = quantize_fp8(x, granularity="token")
    assert codes.shape == x.shape and codes.dtype == torch.float8_e4m3fn
    assert scales.shape == (0, 1) and scales.dtype == torch.float32
    output = mm(
        x,
        torch.empty(7, 5120, dtype=torch.float8_e4m3fn).t(),
        B_scales=torch.ones(7, 1),
        out_dtype=x.dtype,
        quant="fp8",
    )
    assert output.shape == (0, 7) and output.dtype == x.dtype


def test_portable_quantizer_rejects_dependent_launch(amd):
    with pytest.raises(ValueError, match="does not support enable_pdl"):
        quantize_fp8(
            torch.ones(1, 32, dtype=torch.bfloat16),
            granularity="token",
            enable_pdl=True,
        )


@pytest.mark.parametrize("options", [{"alpha": torch.ones(1)}, {"block_size": [1, 32]}])
def test_scaled_gemm_rejects_unsupported_scaling(amd, options):
    with pytest.raises(ValueError, match="takes no alpha or block_size"):
        gemm_triton.triton_mm_fp8_scaled(
            torch.ones(1, 32).to(torch.float8_e4m3fn),
            torch.ones(7, 32).to(torch.float8_e4m3fn).t(),
            torch.ones(1, 1),
            torch.ones(7, 1),
            torch.bfloat16,
            **options,
        )
