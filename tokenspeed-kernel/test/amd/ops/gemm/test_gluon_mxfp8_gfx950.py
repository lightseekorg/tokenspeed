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

from types import SimpleNamespace

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for MXFP8 Gluon GEMM tests",
        allow_module_level=True,
    )

import tokenspeed_kernel  # noqa: E402
from tokenspeed_kernel.profiling import ShapeCapture  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.mm import (  # noqa: E402
    _mxfp8_launch_metadata,
    launch_gluon_mm_mxfp8_gfx950,
    supports_mxfp8_gemm_shape,
)


def _inputs(
    m: int, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    a = (torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.05).to(
        torch.float8_e4m3fn
    )
    b = (torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.05).to(
        torch.float8_e4m3fn
    )
    a_scales = torch.randint(124, 129, (m, k // 32), device="cuda", dtype=torch.uint8)
    b_scales = torch.randint(124, 129, (n, k // 32), device="cuda", dtype=torch.uint8)
    return a, b, a_scales, b_scales


def _dequantize(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rows, k = values.shape
    return (
        values.float().reshape(rows, k // 32, 32)
        * torch.exp2(scales.float() - 127).unsqueeze(-1)
    ).reshape(rows, k)


@pytest.mark.parametrize("k", [512, 1280])
def test_mxfp8_gemm_matches_dequantized_reference(k: int) -> None:
    m, n = 256, 256
    a, b, a_scales, b_scales = _inputs(m, n, k)
    backing = torch.empty((m, n + 17), device="cuda", dtype=torch.bfloat16)
    out = backing[:, :n]

    actual = launch_gluon_mm_mxfp8_gfx950(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        alpha=None,
        block_size=[1, 32],
        out=out,
    )
    expected = (_dequantize(a, a_scales) @ _dequantize(b, b_scales).T).to(actual.dtype)

    assert actual is out
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_mxfp8_gemm_accepts_row_padded_operands_and_strided_scales() -> None:
    m, n, k = 256, 256, 512
    a_backing = (
        torch.randn((m, k + 16), device="cuda", dtype=torch.bfloat16) * 0.05
    ).to(torch.float8_e4m3fn)
    b_backing = (
        torch.randn((n, k + 16), device="cuda", dtype=torch.bfloat16) * 0.05
    ).to(torch.float8_e4m3fn)
    a = a_backing[:, :k]
    b = b_backing[:, :k]

    groups = k // 32
    a_scale_backing = torch.randint(
        124, 129, (m, groups * 2), device="cuda", dtype=torch.uint8
    )
    b_scale_backing = torch.randint(
        124, 129, (n, groups * 2), device="cuda", dtype=torch.uint8
    )
    a_scales = a_scale_backing[:, ::2]
    b_scales = b_scale_backing[:, ::2]

    actual = launch_gluon_mm_mxfp8_gfx950(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        alpha=None,
        block_size=[1, 32],
        out=None,
    )
    expected = (_dequantize(a, a_scales) @ _dequantize(b, b_scales).T).to(actual.dtype)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_mxfp8_gemm_async_scales_accept_row_padding() -> None:
    m, n, k = 256, 256, 512
    a, b, _, _ = _inputs(m, n, k)
    groups = k // 32
    a_scale_backing = torch.randint(
        124, 129, (m, groups + 4), device="cuda", dtype=torch.uint8
    )
    b_scale_backing = torch.randint(
        124, 129, (n, groups + 4), device="cuda", dtype=torch.uint8
    )
    a_scales = a_scale_backing[:, :groups]
    b_scales = b_scale_backing[:, :groups]

    actual = launch_gluon_mm_mxfp8_gfx950(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        alpha=None,
        block_size=[1, 32],
        out=None,
    )
    expected = (_dequantize(a, a_scales) @ _dequantize(b, b_scales).T).to(actual.dtype)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "m,n,k,expected",
    [
        (1024, 1792, 5120, "gluon_mm_mxfp8_gfx950"),
        (256, 4096, 1280, "triton_mm_fp8_blockscale"),
        (1024, 256, 512, "triton_mm_fp8_blockscale"),
    ],
)
def test_mxfp8_gemm_public_api_selects_measured_prefill_shapes(
    m: int, n: int, k: int, expected: str
) -> None:
    a, b, a_scales, b_scales = _inputs(m, n, k)
    ShapeCapture.reset()
    capture = ShapeCapture.get()
    capture.enabled = True
    try:
        tokenspeed_kernel.mm(
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


def test_mxfp8_gemm_rejects_non_mxfp8_scale_contract() -> None:
    a, b, a_scales, b_scales = _inputs(256, 256, 512)

    with pytest.raises(ValueError, match=r"block_size=\[1, 32\]"):
        launch_gluon_mm_mxfp8_gfx950(
            a,
            b,
            a_scales,
            b_scales,
            torch.bfloat16,
            alpha=None,
            block_size=[128, 128],
            out=None,
        )


def test_mxfp8_shape_contract() -> None:
    assert supports_mxfp8_gemm_shape(256, 1536, 4096)
    assert supports_mxfp8_gemm_shape(256, 16384, 1024)
    assert supports_mxfp8_gemm_shape(256, 4096, 1280)
    assert supports_mxfp8_gemm_shape(1024, 1792, 5120)
    assert not supports_mxfp8_gemm_shape(128, 4096, 1280)
    assert not supports_mxfp8_gemm_shape(256, 3968, 1280)
    assert not supports_mxfp8_gemm_shape(256, 4096, 384)


def test_mxfp8_launch_metadata_reports_flops_and_tensor_bytes() -> None:
    m, n, k = 1024, 4096, 1280
    output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    metadata = _mxfp8_launch_metadata(
        None,
        SimpleNamespace(name="mxfp8"),
        {"M": m, "N": n, "K": k, "c_ptr": output},
    )

    expected_bytes = m * k + n * k + (m + n) * (k // 32) + m * n * 2
    assert metadata == {
        "name": "mxfp8",
        "flops8": 2 * m * n * k,
        "bytes": expected_bytes,
    }
