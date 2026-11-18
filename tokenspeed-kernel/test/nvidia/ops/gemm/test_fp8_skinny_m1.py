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
from tokenspeed_kernel.profiling import ShapeCapture
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


@pytest.mark.parametrize("offset", ["activation", "weight", "both", "output"])
def test_alignment_contract_through_mm_and_warmed_graph(offset):
    a, b, a_scale, b_scale = _operands(1, 128, 2048)

    def offset_copy(tensor):
        storage = torch.empty(
            tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device
        )
        return storage[1:].view(tensor.shape).copy_(tensor)

    if offset in ("activation", "both"):
        a = offset_copy(a)
        assert a.is_contiguous() and a.data_ptr() % 16 != 0
    if offset in ("weight", "both"):
        b = offset_copy(b.t()).t()
        assert b.t().is_contiguous() and b.data_ptr() % 16 != 0
    storage = torch.full((130,), 42, dtype=torch.bfloat16, device=a.device)
    out = storage[1:-1].view(1, 128)
    assert out.is_contiguous() and out.data_ptr() % 16 != 0

    def run():
        return kernel_mm(
            a,
            b,
            A_scales=a_scale,
            B_scales=b_scale,
            out_dtype=torch.bfloat16,
            quant="fp8",
            out=out,
        )

    if offset != "output":
        argument = "B" if offset == "weight" else "A"
        with pytest.raises(
            ValueError,
            match=rf"{argument} to be 16-byte aligned; got address % 16 = 1 bytes, storage_offset = 1 elements",
        ):
            run()
        assert torch.all(storage == 42)
        return

    capture = ShapeCapture.get()
    capture.clear()
    previous = capture.enabled
    capture.enabled = True
    try:
        assert run() is out
    finally:
        capture.enabled = previous
    assert capture._records[-1].kernel_name == "cute_dsl_mm_fp8_tensor_scaled_m1"
    expected = (a.float() * a_scale) @ (b.float() * b_scale)
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=3e-5)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    a.copy_((a.float() * 0.5).to(a.dtype))
    b.t().copy_((b.t().float() * 0.5).to(b.dtype))
    expected = (a.float() * a_scale) @ (b.float() * b_scale)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=3e-5)
    assert storage[0] == 42 and storage[-1] == 42


def test_cold_capture_rejects_unaligned_output_then_warmed_capture_accepts_it():
    # A distinct shape keeps this regression independent of earlier warmups.
    a, b, a_scale, b_scale = _operands(1, 132, 2048)
    key = (a.device.index or 0, b.shape[1], a.shape[1])
    cute_dsl_ops._fp8_skinny_warmed.discard(key)
    storage = torch.full((134,), 42, device="cuda", dtype=torch.bfloat16)
    out = storage[1:-1].view(1, 132)
    assert out.is_contiguous() and out.data_ptr() % 16 != 0

    def run():
        return kernel_mm(
            a,
            b,
            A_scales=a_scale,
            B_scales=b_scale,
            out_dtype=torch.bfloat16,
            quant="fp8",
            out=out,
        )

    graph = torch.cuda.CUDAGraph()
    with pytest.raises(ValueError, match="out to be 16-byte aligned"):
        with torch.cuda.graph(graph):
            run()
    assert torch.all(storage == 42)
    assert run() is out
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    a.copy_((a.float() * 0.5).to(a.dtype))
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    expected = (a.float() * a_scale) @ (b.float() * b_scale)
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=3e-5)
    assert storage[0] == 42 and storage[-1] == 42


def test_unaligned_padded_weights_are_normalized_before_skinny_gemv():
    a, b, a_scale, b_scale = _operands(1, 128, 2048)
    storage = torch.empty(128 * 2064 + 1, device="cuda", dtype=b.dtype)
    weight = storage[1:].view(128, 2064)[:, :2048]
    weight.copy_(b.t())
    assert not weight.is_contiguous() and weight.data_ptr() % 16 != 0
    result = kernel_mm(
        a,
        weight.t(),
        A_scales=a_scale,
        B_scales=b_scale,
        out_dtype=torch.bfloat16,
        quant="fp8",
    )
    expected = (a.float() * a_scale) @ (b.float() * b_scale)
    torch.testing.assert_close(result.float(), expected, rtol=1e-2, atol=3e-5)
