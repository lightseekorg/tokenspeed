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


"""A main FP8 linear at a skinny-GEMM shape takes the gfx950 skinny FP8 GEMM
in its 17-128 row window, at the precision of the FP8 GEMM it replaces: within
one BF16 rounding plus FP32 summation error of the FP64 product."""

import os
import sys

import pytest
import torch
from tokenspeed_kernel.ops.gemm import _mm_selected_kernel_name
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.quantization.w8a8_fp8 import W8A8Fp8Config

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_cdna4,
    reason="the skinny FP8 GEMM is gfx950-only",
)


@pytest.mark.parametrize("rows", [16, 64, 129])
def test_w8a8_fp8_linear_takes_the_skinny_gemm_in_its_window(rows):
    torch.manual_seed(rows)
    # An FP8 checkpoint: per-channel E4M3 codes and [N, 1] FP32 scales.
    with torch.device("cuda"):
        layer = ReplicatedLinear(
            8192,
            4608,
            bias=False,
            quant_config=W8A8Fp8Config(is_checkpoint_fp8_serialized=True),
            prefix="proj",
        )
    codes, scales = quantize_fp8(
        (torch.randn(4608, 8192, device="cuda") * 0.02).bfloat16(),
        granularity="token",
    )
    layer.weight.data.copy_(codes)
    layer.weight_scale.data.copy_(scales.reshape(layer.weight_scale.shape))
    layer.quant_method.process_weights_after_loading(layer)
    x = torch.randn(rows, 8192, device="cuda").bfloat16()

    out = layer.quant_method.apply(layer, x)

    # The operands the method passes to gemm.mm, and the kernel they select.
    codes, scales = quantize_fp8(x, granularity="token")
    operands = dict(A_scales=scales, B_scales=layer.weight_scale)
    selected = _mm_selected_kernel_name(
        codes,
        layer.weight,
        **operands,
        out_dtype=torch.bfloat16,
        quant="fp8",
        features=None,
    )
    in_window = 17 <= rows <= 128
    assert (selected == "gluon_mm_fp8_skinny_gfx950") == in_window
    a = codes.double() * scales.double().reshape(-1, 1)
    b = layer.weight.double() * layer.weight_scale.double().reshape(1, -1)
    exact = a @ b
    ulp = torch.exp2(torch.floor(torch.log2(exact.abs().clamp_min(2.0**-126))) - 7)
    bound = ulp / 2 + a.shape[1] * 2.0**-24 * (a.abs() @ b.abs())
    assert bool(((out.double() - exact).abs() <= bound).all())
