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

"""The M == 1 per-tensor FP8 skinny GEMV: selection, results and capture."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import _gemm_format_signature
from tokenspeed_kernel.ops.gemm import cute_dsl as cute_dsl_ops
from tokenspeed_kernel.ops.gemm import flashinfer as flashinfer_ops
from tokenspeed_kernel.ops.gemm import mm as kernel_mm
from tokenspeed_kernel.selection import select_kernel

pytestmark = pytest.mark.skipif(
    not hasattr(cute_dsl_ops, "cute_dsl_mm_fp8_tensor_scaled_m1"),
    reason="the FP8 skinny GEMV needs a Blackwell NVIDIA GPU",
)


def _operands(m: int, n: int, k: int):
    g = torch.Generator(device="cuda").manual_seed(m + n + k)
    a = (torch.randn(m, k, generator=g, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    weight = (torch.randn(n, k, generator=g, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    return (
        a,
        weight.t(),
        torch.tensor([0.03], device="cuda"),
        torch.tensor([0.02], device="cuda"),
    )


def _selected(m: int, n: int, k: int) -> str:
    a, b, a_scale, b_scale = _operands(m, n, k)
    return select_kernel(
        "gemm",
        "mm",
        _gemm_format_signature(a, b, a_scale, b_scale, torch.bfloat16, "fp8", None),
        traits={
            "m": m,
            "n": n,
            "k": k,
            "a_inner_stride_one": True,
            "b_inner_stride_one": False,
            "out_dtype": torch.bfloat16,
        },
    ).name


def test_only_single_rows_with_aligned_k_select_the_skinny_gemv():
    assert _selected(1, 4096, 8192) == "cute_dsl_mm_fp8_tensor_scaled_m1"
    assert _selected(2, 4096, 8192) == "flashinfer_mm_fp8_tensor_scaled"
    assert _selected(1, 4096, 5376) == "flashinfer_mm_fp8_tensor_scaled"


@pytest.mark.parametrize(
    "n,k", [(18560, 4096), (4096, 8192), (5376, 4096), (4608, 4096), (4096, 4096)]
)
def test_matches_the_cublaslt_fp8_gemm(n, k):
    a, b, a_scale, b_scale = _operands(1, n, k)
    expected = flashinfer_ops.flashinfer_mm_fp8_tensor_scaled(
        a, b, a_scale, b_scale, torch.bfloat16
    )
    out = kernel_mm(
        a, b, A_scales=a_scale, B_scales=b_scale, out_dtype=torch.bfloat16, quant="fp8"
    )
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    strided = torch.empty(1, 2 * n, dtype=torch.bfloat16, device="cuda")[:, ::2]
    assert not strided.is_contiguous()
    cute_dsl_ops.cute_dsl_mm_fp8_tensor_scaled_m1(
        a, b, a_scale, b_scale, torch.bfloat16, out=strided
    )
    torch.testing.assert_close(strided, expected, rtol=1e-2, atol=1e-2)
    padded = (torch.randn(n, k + 16, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    padded[:, :k].copy_(b.t())
    torch.testing.assert_close(
        cute_dsl_ops.cute_dsl_mm_fp8_tensor_scaled_m1(
            a, padded[:, :k].t(), a_scale, b_scale, torch.bfloat16
        ),
        expected,
        rtol=1e-2,
        atol=1e-2,
    )


def test_capture_before_warmup_runs_cublaslt_and_replays_after():
    a, b, a_scale, b_scale = _operands(1, 4096, 6144)
    expected = flashinfer_ops.flashinfer_mm_fp8_tensor_scaled(
        a, b, a_scale, b_scale, torch.bfloat16
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cold = cute_dsl_ops.cute_dsl_mm_fp8_tensor_scaled_m1(
            a, b, a_scale, b_scale, torch.bfloat16
        )
    graph.replay()
    torch.testing.assert_close(cold, expected, rtol=0, atol=0)
    cute_dsl_ops.cute_dsl_mm_fp8_tensor_scaled_m1(
        a, b, a_scale, b_scale, torch.bfloat16
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        warm = cute_dsl_ops.cute_dsl_mm_fp8_tensor_scaled_m1(
            a, b, a_scale, b_scale, torch.bfloat16
        )
    a.copy_((torch.randn(1, 6144, device="cuda") * 0.5).to(torch.float8_e4m3fn))
    changed = flashinfer_ops.flashinfer_mm_fp8_tensor_scaled(
        a, b, a_scale, b_scale, torch.bfloat16
    )
    assert not torch.equal(changed, expected)
    graph.replay()
    torch.testing.assert_close(warm, changed, rtol=1e-2, atol=1e-2)
