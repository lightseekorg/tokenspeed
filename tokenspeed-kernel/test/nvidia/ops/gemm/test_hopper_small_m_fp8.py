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

"""Hopper SM90 small-M block-FP8 GEMM and TMA-aligned MN-major scale tests."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm.hopper_small_m_fp8 import (
    _hopper_small_m_fp8_kernel,
    hopper_small_m_fp8_mm,
    hopper_small_m_fp8_supported,
)
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.platform import Platform
from utils import assert_no_triton_compile, is_nvidia

pytestmark = pytest.mark.skipif(
    not is_nvidia(), reason="Hopper FP8 tests require an NVIDIA GPU"
)


@pytest.mark.parametrize("rows", [1, 6, 16, 24])
def test_trtllm_quantize_fp8_mn_major_tma_matches_dense(rows: int) -> None:
    """TMA-aligned MN-major FP8 quantizer matches dense quantizer with 4-float row alignment."""
    torch.manual_seed(67 + rows)
    device = torch.device("cuda")
    x = torch.randn(rows, 3072, dtype=torch.bfloat16, device=device)
    q_tma, s_tma = quantize_fp8(
        x,
        granularity="token_group",
        group_size=128,
        scale_encoding="float32",
        override="trtllm_quantize_fp8_mn_major_tma",
    )
    q_ref, s_ref = quantize_fp8(
        x,
        granularity="token_group",
        group_size=128,
        scale_encoding="float32",
        override="trtllm_quantize_fp8_mn_major",
    )
    aligned = ((rows + 3) // 4) * 4
    assert s_tma.stride() == (1, aligned)
    assert torch.equal(q_tma.view(torch.uint8), q_ref.view(torch.uint8))
    torch.testing.assert_close(s_tma, s_ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("m", "n", "k", "mn_major"),
    [
        (1, 512, 1024, True),
        (6, 3072, 3072, True),
        (16, 2048, 3072, False),
        (24, 3072, 3072, True),
    ],
)
def test_hopper_small_m_fp8_mm_and_cuda_graph(
    m: int, n: int, k: int, mn_major: bool
) -> None:
    """Verify Hopper small-M W8A8 block-FP8 GEMM accuracy and multi-replay CUDA graph safety."""
    if not Platform.get().is_hopper:
        pytest.skip("Hopper SM90 only")

    torch.manual_seed(101 + m + n)
    device = torch.device("cuda")
    a = (torch.randn(m, k, device=device) * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, device=device) * 0.5).to(torch.float8_e4m3fn)
    k_blks = k // 128
    if mn_major:
        aligned_m = max(((m + 3) // 4) * 4, 4)
        sa = (torch.rand(k_blks, aligned_m, device=device) * 0.03 + 0.01).t()[:m]
    else:
        sa = torch.rand(m, k_blks, device=device) * 0.03 + 0.01
    sb = torch.rand((n + 127) // 128, k_blks, device=device) * 0.03 + 0.01
    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    assert hopper_small_m_fp8_supported(a, b, sa, sb, torch.bfloat16)

    def _ref() -> torch.Tensor:
        return (a.float() * sa.repeat_interleave(128, dim=1)) @ (
            b.float()
            * sb.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:n]
        ).T

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        hopper_small_m_fp8_mm(a, b, sa, sb, out=out, enable_pdl=False)
        torch.testing.assert_close(out.float(), _ref(), atol=2e-4, rtol=5e-3)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hopper_small_m_fp8_mm(a, b, sa, sb, out=out, enable_pdl=True)
            hopper_small_m_fp8_mm(a, b, sa, sb, out=out, enable_pdl=True)
        for factor in (0.0, 0.5, 1.5, -0.75):
            a.copy_((torch.randn(m, k, device=device) * factor).to(torch.float8_e4m3fn))
            graph.replay()
            torch.testing.assert_close(out.float(), _ref(), atol=2e-4, rtol=5e-3)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def test_hopper_small_m_fp8_row_counts_do_not_recompile() -> None:
    """Varying M within the same BM bucket must not trigger Triton recompilation."""
    if not Platform.get().is_hopper:
        pytest.skip("Hopper SM90 only")

    device = torch.device("cuda")
    n, k = 3072, 3072
    b = (torch.randn(n, k, device=device) * 0.5).to(torch.float8_e4m3fn)
    sb = torch.rand(n // 128, k // 128, device=device) * 0.03 + 0.01
    for m in (1, 17):
        a = (torch.randn(m, k, device=device) * 0.5).to(torch.float8_e4m3fn)
        sa = torch.rand(k // 128, m, device=device).T
        hopper_small_m_fp8_mm(a, b, sa, sb, enable_pdl=False)
    with assert_no_triton_compile(_hopper_small_m_fp8_kernel):
        for m in (2, 6, 16, 20, 32):
            a = (torch.randn(m, k, device=device) * 0.5).to(torch.float8_e4m3fn)
            sa = torch.rand(k // 128, m, device=device).T
            hopper_small_m_fp8_mm(a, b, sa, sb, enable_pdl=False)
