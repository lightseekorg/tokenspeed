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
    pytest.skip(
        "AMD CDNA4 is required for the skinny MXFP8 GEMM tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.gemm import mm as kernel_mm  # noqa: E402
from tokenspeed_kernel.profiling import ShapeCapture  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8 import skinny  # noqa: E402


def _inputs(m: int, n: int, k: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((m, k), device="cuda", generator=g).to(torch.float8_e4m3fn)
    b = torch.randn((n, k), device="cuda", generator=g).to(torch.float8_e4m3fn)
    # UE8M0 scales 2^-3..2^3, as in the numerics suite.
    a_scales = torch.randint(
        124, 131, (m, k // 32), device="cuda", dtype=torch.uint8, generator=g
    )
    b_scales = torch.randint(
        124, 131, (n, k // 32), device="cuda", dtype=torch.uint8, generator=g
    )
    return a, b, a_scales, b_scales


def _reference(a, b, a_scales, b_scales) -> torch.Tensor:
    def dequantize(values, scales):
        return values.double() * torch.exp2(scales.double() - 127).repeat_interleave(
            32, dim=1
        )

    return dequantize(a, a_scales) @ dequantize(b, b_scales).T


def _check(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # One output rounding plus fp32 accumulation. FP8 MFMAs miss this: they sum
    # products at reduced precision, off by ~0.05 on these inputs.
    torch.testing.assert_close(
        actual.double(), expected, rtol=2.0**-8, atol=5e-3, check_dtype=False
    )


def _launch(a, b, a_scales, b_scales, out_dtype=torch.bfloat16, out=None):
    return skinny.launch_triton_mm_mxfp8_skinny_gfx950(
        a, b, a_scales, b_scales, out_dtype, None, [1, 32], out
    )


@pytest.mark.parametrize("m", [1, 6, 16, 17, 32, 33, 64])
@pytest.mark.parametrize("n,k", [(1792, 5120), (300, 1056)])
def test_skinny_mxfp8_gemm_matches_fp64_reference(m: int, n: int, k: int) -> None:
    a, b, a_scales, b_scales = _inputs(m, n, k)
    _check(_launch(a, b, a_scales, b_scales), _reference(a, b, a_scales, b_scales))


def test_skinny_mxfp8_gemm_fp16_output() -> None:
    a, b, a_scales, b_scales = _inputs(6, 512, 2048)
    actual = _launch(a, b, a_scales, b_scales, out_dtype=torch.float16)
    assert actual.dtype == torch.float16
    _check(actual, _reference(a, b, a_scales, b_scales))


def test_skinny_mxfp8_gemm_accepts_padded_operands_and_strided_scales() -> None:
    m, n, k = 6, 300, 1056
    a_full, b_full, _, _ = _inputs(m, n, k + 64)
    a, b = a_full[:, :k], b_full[:, :k]
    g = torch.Generator(device="cuda").manual_seed(1)
    a_scales = torch.randint(
        124, 131, (m, k // 16), device="cuda", dtype=torch.uint8, generator=g
    )[:, ::2]
    b_scales = torch.randint(
        124, 131, (n, k // 16), device="cuda", dtype=torch.uint8, generator=g
    )[:, ::2]
    out = torch.empty((m, n + 17), device="cuda", dtype=torch.bfloat16)[:, :n]

    actual = _launch(a, b, a_scales, b_scales, out=out)

    assert actual is out
    _check(actual, _reference(a, b, a_scales, b_scales))


@pytest.mark.parametrize(
    "m,expected",
    [
        (1, "triton_mm_mxfp8_skinny_gfx950"),
        (64, "triton_mm_mxfp8_skinny_gfx950"),
        (65, "triton_mm_fp8_blockscale"),
    ],
)
def test_mxfp8_public_api_selects_skinny_for_decode_rows(m: int, expected: str) -> None:
    a, b, a_scales, b_scales = _inputs(m, 1792, 5120)
    ShapeCapture.reset()
    capture = ShapeCapture.get()
    capture.enabled = True
    try:
        kernel_mm(
            a,
            b,
            A_scales=a_scales,
            B_scales=b_scales,
            out_dtype=torch.bfloat16,
            block_size=[1, 32],
            quant="mxfp8",
        )
    finally:
        capture.enabled = False

    assert capture._records[-1].kernel_name == expected


def test_skinny_mxfp8_gemm_does_not_recompile_across_batch_rows() -> None:
    n, k = 512, 1024
    a, b, a_scales, b_scales = _inputs(64, n, k)
    full = _launch(a, b, a_scales, b_scales)

    def run(rows):
        # Rows are independent: a shorter batch is a prefix.
        out = _launch(a[:rows], b, a_scales[:rows], b_scales)
        torch.testing.assert_close(out, full[:rows], rtol=0, atol=0)

    # One binary per power-of-two row bucket (16, 32, 64) and integer class.
    sweep = (2, 5, 12, 16, 20, 31, 32, 40, 48, 63, 64)
    warm_specialization_classes(
        run,
        lambda rows: (
            max(16, 1 << (rows - 1).bit_length()),
            int_specialization_class(rows),
        ),
        sweep,
        pool=(1, 3, 15, 17, 18, 33, 34, 50),
    )
    with assert_no_triton_compile(skinny.triton_mm_mxfp8_skinny_gfx950):
        for rows in sweep:
            run(rows)
