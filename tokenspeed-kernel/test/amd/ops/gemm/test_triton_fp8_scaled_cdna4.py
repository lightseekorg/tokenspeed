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

import pytest
import torch
from utils import (
    assert_no_triton_compile,
    int_specialization_class,
    is_cdna4,
    warm_specialization_classes,
)

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

import tokenspeed_kernel  # noqa: E402
from tokenspeed_kernel.ops.gemm import triton as triton_gemm  # noqa: E402
from tokenspeed_kernel.ops.gemm.fp8_utils import (  # noqa: E402
    per_token_group_quant_fp8,
)


def _quantized_operands(m: int, n: int, k: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=generator)
    w = 0.02 * torch.randn(
        n, k, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    w_scale = w.abs().amax(dim=1, keepdim=True).float() / 448.0
    w_fp8 = (w.float() / w_scale).to(torch.float8_e4m3fn)
    x_fp8, x_scale = per_token_group_quant_fp8(x, k)
    return x_fp8, x_scale, w_fp8, w_scale


@pytest.mark.parametrize("m,n,k", [(1, 6272, 7168), (64, 7168, 1536), (300, 512, 256)])
def test_fp8_per_channel_mm_matches_dequantized_reference(m, n, k):
    x_fp8, x_scale, w_fp8, w_scale = _quantized_operands(m, n, k, seed=m)
    out = tokenspeed_kernel.mm(
        x_fp8,
        w_fp8.t(),
        A_scales=x_scale,
        B_scales=w_scale,
        out_dtype=torch.bfloat16,
        quant="fp8",
    )
    expected = (x_fp8.float() * x_scale) @ (w_fp8.float() * w_scale).t()
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)


def test_fp8_per_channel_mm_does_not_recompile_across_batches():
    n, k = 512, 256
    _, _, w_fp8, w_scale = _quantized_operands(1, n, k, seed=0)

    def run(m: int) -> None:
        generator = torch.Generator(device="cuda").manual_seed(m)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=generator)
        x_fp8, x_scale = per_token_group_quant_fp8(x, k)
        tokenspeed_kernel.mm(
            x_fp8,
            w_fp8.t(),
            A_scales=x_scale,
            B_scales=w_scale,
            out_dtype=torch.bfloat16,
            quant="fp8",
        )

    def key(m: int):
        # Tiles follow the power-of-two M bucket; M specializes on 1 and %16.
        bucket = min(max(32, 1 << (m - 1).bit_length()), 256)
        return (bucket, int_specialization_class(m))

    sweep = (3, 17, 45, 100, 250, 1000, 4097)
    warm_specialization_classes(run, key, sweep, range(1, 5000))
    with assert_no_triton_compile(triton_gemm.scaled_mm_kernel):
        for m in sweep:
            run(m)
