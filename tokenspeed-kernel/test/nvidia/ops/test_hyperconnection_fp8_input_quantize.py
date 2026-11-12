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

"""Compare the actual input quantizer with canonical FP32 rounding."""

import pytest
import torch
from test_hyperconnection_fp8 import fp8_device as fp8_device
from tokenspeed_kernel.ops.residual.cute_dsl_fp8 import _quantize_input
from tokenspeed_kernel.platform import pdl_enabled
from utils import assert_no_cute_compile


def _reference(x):
    values = x.float()
    maximum = values.abs().amax(-1, keepdim=True)
    limit = torch.full_like(maximum, 448.0)
    scales = (maximum / limit).clamp_min(1.0 / (448.0 * 512.0))
    multiplier = (limit / maximum).clamp_max(448.0 * 512.0)
    return (values * multiplier).clamp(-448, 448).to(
        torch.float8_e4m3fn
    ), scales.reshape(-1)


def _check(codes, scales, x):
    expected_codes, expected_scales = _reference(x)
    nan = torch.isnan(expected_codes)
    assert torch.equal(torch.isnan(codes), nan)
    assert torch.equal(
        codes.view(torch.uint8)[~nan], expected_codes.view(torch.uint8)[~nan]
    )
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("columns", [10240, 32768])
@pytest.mark.parametrize("pdl", [False, True])
@torch.inference_mode()
def test_quantize_bits_scales_and_replay(dtype, columns, pdl):
    pdl_enabled(pdl)
    torch.manual_seed(248)
    x = torch.randn((17, columns), device="cuda", dtype=dtype)
    x[0].zero_()
    x[1].fill_(2.0**-20)
    x[2].mul_(2.0**-16)
    x[3].mul_(256.0)
    x[4, 0] = float("nan")
    x[5, 0] = float("inf")
    x[6, 0] = -float("inf")
    _check(*_quantize_input(x), x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with assert_no_cute_compile():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            codes, scales = _quantize_input(x)
        graph.replay()
        stream.synchronize()
        _check(codes, scales, x)
        with torch.cuda.stream(stream):
            x.mul_(0.5)
            graph.replay()
        stream.synchronize()
        _check(codes, scales, x)
