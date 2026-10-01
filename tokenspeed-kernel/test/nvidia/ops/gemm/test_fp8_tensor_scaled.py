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

"""Per-tensor scaled FP8 GEMM selection and its FlashInfer cuBLASLt kernel."""

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.gemm import _gemm_format_signature
from tokenspeed_kernel.ops.gemm import flashinfer as flashinfer_ops
from tokenspeed_kernel.selection import select_kernel

pytestmark = pytest.mark.skipif(
    flashinfer_ops.bmm_fp8 is flashinfer_ops.error_fn,
    reason="FlashInfer bmm_fp8 needs a Blackwell NVIDIA GPU",
)


def _operands(m: int, k: int, n: int):
    g = torch.Generator(device="cuda").manual_seed(m + n)
    a = torch.randn(m, k, generator=g, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.randn(n, k, generator=g, device="cuda").to(torch.float8_e4m3fn)
    a_scale = torch.tensor([0.03], device="cuda")
    b_scale = torch.tensor([0.02], device="cuda")
    expected = (a.float() * a_scale) @ (weight.float() * b_scale).t()
    return a, weight.t(), a_scale, b_scale, expected


@pytest.mark.parametrize("m", [1, 7, 256])
def test_column_major_weights_select_cublaslt_and_match_the_reference(m):
    a, b, a_scale, b_scale, expected = _operands(m, 512, 384)
    out = tokenspeed_kernel.mm(
        a, b, A_scales=a_scale, B_scales=b_scale, out_dtype=torch.bfloat16, quant="fp8"
    )
    error = (out.float() - expected).norm() / expected.norm()
    assert error.item() < 1e-2
    selected = select_kernel(
        "gemm",
        "mm",
        _gemm_format_signature(a, b, a_scale, b_scale, torch.bfloat16, "fp8", None),
        traits={"a_inner_stride_one": True, "b_inner_stride_one": False},
    )
    assert selected.name == "flashinfer_mm_fp8_tensor_scaled"


def test_row_major_weights_fall_to_the_general_kernel():
    a, b, a_scale, b_scale, expected = _operands(4, 512, 384)
    b = b.contiguous()
    out = tokenspeed_kernel.mm(
        a, b, A_scales=a_scale, B_scales=b_scale, out_dtype=torch.bfloat16, quant="fp8"
    )
    assert ((out.float() - expected).norm() / expected.norm()).item() < 1e-2


def test_strided_output_rows_are_written_in_place():
    a, b, a_scale, b_scale, expected = _operands(8, 512, 384)
    buffer = torch.zeros(8, 512, device="cuda", dtype=torch.bfloat16)
    view = buffer[:, :384]
    result = flashinfer_ops.flashinfer_mm_fp8_tensor_scaled(
        a, b, a_scale, b_scale, torch.bfloat16, out=view
    )
    assert result.data_ptr() == view.data_ptr()
    assert ((view.float() - expected).norm() / expected.norm()).item() < 1e-2
    assert torch.count_nonzero(buffer[:, 384:]) == 0


def test_padded_operand_rows_are_read_with_their_strides():
    a, b, a_scale, b_scale, expected = _operands(8, 512, 384)
    a_wide = torch.zeros(8, 640, device="cuda").to(torch.float8_e4m3fn)
    a_wide[:, :512] = a
    b_wide = torch.zeros(384, 640, device="cuda").to(torch.float8_e4m3fn)
    b_wide[:, :512] = b.t()
    result = flashinfer_ops.flashinfer_mm_fp8_tensor_scaled(
        a_wide[:, :512], b_wide[:, :512].t(), a_scale, b_scale, torch.bfloat16
    )
    assert ((result.float() - expected).norm() / expected.norm()).item() < 1e-2
