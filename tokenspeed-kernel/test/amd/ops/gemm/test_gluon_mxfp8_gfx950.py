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
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for MXFP8 Gluon GEMM tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.gemm import mm as kernel_mm
from tokenspeed_kernel.profiling import ShapeCapture  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.mm import (  # noqa: E402
    _mxfp8_launch_metadata,
    _mxfp8_num_splits,
    gluon_mm_mxfp8_gfx950,
    gluon_mm_mxfp8_reduce_gfx950,
    launch_gluon_mm_mxfp8_gfx950,
    supports_mxfp8_gemm_shape,
)

# noqa: E402


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


@pytest.mark.parametrize(
    "m,n,k",
    [
        # Ragged M tile, N not a multiple of 256, split-K partials.
        (641, 1152, 5120),
        # Ragged M and N tiles, unsplit.
        (1531, 1296, 1280),
        # K tail (18 scale groups): shifted K walk, preloaded scale rows.
        (8175, 5120, 576),
        # Fewer rows and columns than one tile, K tail of 800.
        (37, 208, 800),
    ],
)
def test_mxfp8_gemm_ragged_shapes_match_dequantized_reference(
    m: int, n: int, k: int
) -> None:
    a, b, a_scales, b_scales = _inputs(m, n, k)

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


@pytest.mark.parametrize("k", [5120, 576])
def test_mxfp8_gemm_row_counts_reuse_compiled_kernels(k: int) -> None:
    n = 1152
    a, b, a_scales, b_scales = _inputs(8192, n, k)

    def run(rows: int) -> None:
        launch_gluon_mm_mxfp8_gfx950(
            a[:rows],
            b,
            a_scales[:rows],
            b_scales,
            torch.bfloat16,
            alpha=None,
            block_size=[1, 32],
            out=None,
        )

    # Warm the split-K and the direct-output launches.
    run(641)
    run(8192)
    with assert_no_triton_compile(gluon_mm_mxfp8_gfx950, gluon_mm_mxfp8_reduce_gfx950):
        for rows in (300, 1531, 2049, 6622, 8175):
            run(rows)


@pytest.mark.parametrize("k", [512, 1056])
def test_mxfp8_gemm_accepts_row_padded_operands_and_strided_scales(k: int) -> None:
    # K=1056 also takes the K-tail path with direct scale loads.
    m, n = 300, 256
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
    "m,n,k,selected",
    [
        (1024, 1792, 5120, True),
        (8175, 1152, 5120, True),
        (641, 5120, 576, True),
        (257, 256, 512, True),
        (256, 4096, 1280, False),
    ],
)
def test_mxfp8_gemm_public_api_selects_prefill_rows(
    m: int, n: int, k: int, selected: bool
) -> None:
    a, b, a_scales, b_scales = _inputs(m, n, k)
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

    kernel_name = capture._records[-1].kernel_name
    assert (kernel_name == "gluon_mm_mxfp8_gfx950") == selected


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
    assert supports_mxfp8_gemm_shape(8175, 1152, 5120)
    assert supports_mxfp8_gemm_shape(641, 5120, 576)
    assert supports_mxfp8_gemm_shape(1, 16, 288)
    assert not supports_mxfp8_gemm_shape(256, 4100, 1280)
    assert not supports_mxfp8_gemm_shape(256, 4096, 256)
    assert not supports_mxfp8_gemm_shape(256, 4096, 400)


def test_mxfp8_split_k_only_for_underfilled_long_k_launches() -> None:
    assert _mxfp8_num_splits(641, 1152, 5120) > 1
    assert _mxfp8_num_splits(8192, 1152, 5120) == 1
    assert _mxfp8_num_splits(641, 8192, 1280) == 1


@pytest.mark.parametrize(
    ("splits", "out_dtype"), [(1, torch.bfloat16), (4, torch.float32)]
)
def test_mxfp8_launch_metadata_reports_flops_and_tensor_bytes(
    splits: int, out_dtype: torch.dtype
) -> None:
    m, n, k = 1024, 4096, 1280
    # Split-K writes into an FP32 [splits, M, N] partial buffer.
    output = torch.empty((splits, m, n), device="cuda", dtype=out_dtype)

    metadata = _mxfp8_launch_metadata(
        None,
        SimpleNamespace(name="mxfp8"),
        {"M": m, "N": n, "K": k, "SPLITS": splits, "c_ptr": output},
    )

    expected_bytes = (
        m * k + n * k + (m + n) * (k // 32) + m * n * splits * output.element_size()
    )
    assert metadata == {
        "name": "mxfp8",
        "flops8": 2 * m * n * k,
        "bytes": expected_bytes,
    }
