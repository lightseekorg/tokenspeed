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
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for the gfx950 MXFP8 decode GEMM tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.gemm import mm  # noqa: E402
from tokenspeed_kernel.ops.quantization import quantize_fp8  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8 import decode  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.decode import (  # noqa: E402
    gluon_mm_mxfp8_decode_gfx950,
    launch_gluon_mm_mxfp8_decode_gfx950,
)

# Inputs dequantize exactly and both sides accumulate in FP32 before one BF16
# rounding, so outputs differ by at most one BF16 ULP (2**-8 relative) plus
# FP32 accumulation-order noise on near-zero outputs.
_ATOL = 1e-3
_RTOL = 2**-8


def _operands(m: int, n: int, k: int):
    a = (torch.randn((m, k), device="cuda") * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device="cuda") * 0.5).to(torch.float8_e4m3fn)
    a_scales = torch.randint(120, 130, (m, k // 32), device="cuda", dtype=torch.uint8)
    b_scales = torch.randint(118, 128, (n, k // 32), device="cuda", dtype=torch.uint8)
    return a, b, a_scales, b_scales


def _dequantize(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    decoded = scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32, dim=1)
    return values.float() * decoded


def _reference(a, b, a_scales, b_scales) -> torch.Tensor:
    return _dequantize(a, a_scales) @ _dequantize(b, b_scales).T


def _run(a, b, a_scales, b_scales, out=None) -> torch.Tensor:
    return launch_gluon_mm_mxfp8_decode_gfx950(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        alpha=None,
        block_size=[1, 32],
        out=out,
    )


@pytest.mark.parametrize(
    "m,n,k",
    [
        # DeepSeek V4.1 TP4 decode projections across the M buckets.
        (1, 8192, 1280),
        (6, 5120, 2048),
        (24, 1152, 5120),
        (48, 1792, 5120),
        (96, 5120, 576),
        (192, 8192, 1280),
        (256, 1536, 5120),
        # Ragged N and a K that is not a multiple of the K tile.
        (37, 1000, 800),
    ],
)
def test_matches_dequantized_reference(m: int, n: int, k: int) -> None:
    torch.manual_seed(0)
    a, b, a_scales, b_scales = _operands(m, n, k)
    actual = _run(a, b, a_scales, b_scales)
    torch.testing.assert_close(
        actual.float(), _reference(a, b, a_scales, b_scales), atol=_ATOL, rtol=_RTOL
    )


def test_writes_strided_out_deterministically() -> None:
    torch.manual_seed(0)
    m, n, k = 6, 1152, 5120  # split-K route
    a, b, a_scales, b_scales = _operands(m, n, k)
    backing = torch.empty((m, n + 24), device="cuda", dtype=torch.bfloat16)
    out = backing[:, :n]
    actual = _run(a, b, a_scales, b_scales, out=out)
    assert actual is out
    torch.testing.assert_close(
        out.float(), _reference(a, b, a_scales, b_scales), atol=_ATOL, rtol=_RTOL
    )
    # The split-K reduction sums partials in split order.
    for _ in range(3):
        assert torch.equal(_run(a, b, a_scales, b_scales), actual)


def test_row_count_does_not_recompile() -> None:
    torch.manual_seed(0)
    n, k = 1792, 5120
    a, b, a_scales, b_scales = _operands(256, n, k)
    expected = _reference(a, b, a_scales, b_scales)

    def run(rows: int) -> torch.Tensor:
        return _run(a[:rows], b, a_scales[:rows], b_scales)

    # One row count per M bucket compiles each tile configuration.
    for rows in (1, 32, 33, 128, 129):
        run(rows)
    with assert_no_triton_compile(gluon_mm_mxfp8_decode_gfx950):
        for rows in (2, 5, 6, 10, 12, 20, 24, 40, 48, 80, 96, 120, 144, 186, 192):
            # Rows are independent: a shorter batch is a prefix.
            torch.testing.assert_close(
                run(rows).float(), expected[:rows], atol=_ATOL, rtol=_RTOL
            )


def test_hip_graph_replay_uses_private_counters() -> None:
    torch.manual_seed(0)
    m, n, k = 12, 1152, 5120  # split-K route
    a, b, a_scales, b_scales = _operands(m, n, k)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        # Eager warmup on the capture stream fills the per-stream caches.
        _run(a, b, a_scales, b_scales, out=out)
        eager_counters = dict(decode._counter_cache)
        with torch.cuda.graph(graph, stream=stream):
            _run(a, b, a_scales, b_scales, out=out)
    stream.synchronize()
    assert decode._counter_cache == eager_counters

    for _ in range(2):
        new_a, _, new_scales, _ = _operands(m, n, k)
        a.copy_(new_a)
        a_scales.copy_(new_scales)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out.float(),
            _reference(a, b, a_scales, b_scales),
            atol=_ATOL,
            rtol=_RTOL,
        )


@pytest.mark.parametrize("m", [6, 192])
def test_mm_selects_decode_kernel(monkeypatch, m: int) -> None:
    torch.manual_seed(0)
    n, k = 5120, 2048
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    a, a_scales = quantize_fp8(
        x,
        granularity="token_group",
        group_size=32,
        scale_encoding="ue8m0",
        enable_pdl=False,
        override="triton_quantize_fp8_group32_ue8m0",
        solution=None,
    )
    _, b, _, b_scales = _operands(1, n, k)
    launches = []
    launch = decode._launch
    monkeypatch.setattr(
        decode, "_launch", lambda *args: launches.append(args) or launch(*args)
    )
    actual = mm(
        a,
        b,
        A_scales=a_scales,
        B_scales=b_scales,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[1, 32],
    )
    assert len(launches) == 1
    torch.testing.assert_close(
        actual.float(), _reference(a, b, a_scales, b_scales), atol=_ATOL, rtol=_RTOL
    )
