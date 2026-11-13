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

"""Decode-size BF16 GEMM with an in-kernel ordered split-K fold."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm.triton_skinny import (
    SKINNY_MM_MAX_ROWS,
    _skinny_mm_kernel,
    skinny_mm,
    skinny_mm_supported,
)
from utils import assert_no_triton_compile, is_nvidia

pytestmark = pytest.mark.skipif(
    not is_nvidia(), reason="skinny_mm requires an NVIDIA GPU"
)


@pytest.mark.parametrize("shape", [(3792, 4608), (4608, 1536), (384, 4608)])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
def test_skinny_mm_is_one_rounding_of_the_fp32_product(shape, out_dtype, with_bias):
    n, k = shape
    generator = torch.Generator(device="cuda").manual_seed(n + k)
    weight = (torch.randn((n, k), generator=generator, device="cuda") * 0.02).bfloat16()
    x = torch.randn((SKINNY_MM_MAX_ROWS, k), generator=generator, device="cuda")
    x = x.bfloat16()
    bias = (
        torch.randn(n, generator=generator, device="cuda").bfloat16()
        if with_bias
        else None
    )
    out = skinny_mm(x, weight, bias, out_dtype=out_dtype)
    expected = x.double() @ weight.double().T
    if bias is not None:
        expected += bias.double()
    assert out.dtype == out_dtype
    # FP32 accumulation over K plus one rounding to the output dtype.
    tolerance = 1e-5 if out_dtype == torch.float32 else 8e-3
    torch.testing.assert_close(
        out.double(), expected, rtol=tolerance, atol=tolerance * expected.abs().max()
    )
    # A row's bits do not depend on the rows beside it.
    for rows in (1, 3, 8, 17):
        assert torch.equal(
            skinny_mm(x[:rows], weight, bias, out_dtype=out_dtype), out[:rows]
        )


def test_skinny_mm_short_weight_rows_match_a_taller_weight():
    # 384 rows take narrow tiles and 640 rows wide ones; both split K in six.
    generator = torch.Generator(device="cuda").manual_seed(11)
    weight = torch.randn((640, 4608), generator=generator, device="cuda").bfloat16()
    x = torch.randn((SKINNY_MM_MAX_ROWS, 4608), generator=generator, device="cuda")
    x = x.bfloat16()
    bias = torch.randn(640, generator=generator, device="cuda").bfloat16()
    for out_dtype in (torch.bfloat16, torch.float32):
        for rows in (1, 8, SKINNY_MM_MAX_ROWS):
            tall = skinny_mm(x[:rows], weight, bias, out_dtype=out_dtype)
            short = skinny_mm(x[:rows], weight[:384], bias[:384], out_dtype=out_dtype)
            assert torch.equal(short, tall[:, :384])


def test_skinny_mm_reads_a_column_slice_like_its_copy():
    generator = torch.Generator(device="cuda").manual_seed(5)
    wide = torch.randn((768, 9728), generator=generator, device="cuda").bfloat16()
    weight = wide[:, 512:]
    x = torch.randn((7, 9216), generator=generator, device="cuda").bfloat16()
    bias = torch.randn(768, generator=generator, device="cuda").bfloat16()
    assert skinny_mm_supported(x, weight)
    assert torch.equal(
        skinny_mm(x, weight, bias, out_dtype=torch.float32),
        skinny_mm(x, weight.contiguous(), bias, out_dtype=torch.float32),
    )


def test_skinny_mm_takes_decode_rows_only():
    weight = torch.zeros((128, 512), dtype=torch.bfloat16, device="cuda")
    assert skinny_mm_supported(
        torch.zeros((8, 512), dtype=torch.bfloat16, device="cuda"), weight
    )
    assert not skinny_mm_supported(
        torch.zeros((SKINNY_MM_MAX_ROWS + 1, 512), dtype=torch.bfloat16, device="cuda"),
        weight,
    )
    assert not skinny_mm_supported(
        torch.zeros((8, 500), dtype=torch.bfloat16, device="cuda"),
        torch.zeros((128, 500), dtype=torch.bfloat16, device="cuda"),
    )


def test_skinny_mm_row_counts_do_not_recompile():
    weight = torch.randn((3792, 4608), device="cuda").bfloat16()
    x = torch.randn((SKINNY_MM_MAX_ROWS, 4608), device="cuda").bfloat16()
    for rows in (1, 16):
        skinny_mm(x[:rows], weight, None, out_dtype=torch.bfloat16)
    with assert_no_triton_compile(_skinny_mm_kernel):
        for rows in (2, 7, 8, 24, 32):
            skinny_mm(x[:rows], weight, None, out_dtype=torch.bfloat16)
