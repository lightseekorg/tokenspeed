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

"""Per-token/per-channel scaled FP8 GEMM selection and its PyTorch CUTLASS kernel."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import _gemm_format_signature
from tokenspeed_kernel.ops.gemm import mm as kernel_mm
from tokenspeed_kernel.ops.gemm.torch import TORCH_FP8_SCALED_MIN_M
from tokenspeed_kernel.selection import select_kernel
from utils import kernel_supported, make_fp8_per_channel_gemm_operands

pytestmark = pytest.mark.skipif(
    not kernel_supported("torch_mm_fp8_scaled"),
    reason="PyTorch's row-wise scaled FP8 GEMM is registered for SM100",
)


def _selected(m: int, n: int, k: int) -> str:
    a = torch.empty(m, k, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.empty(n, k, device="cuda", dtype=torch.float8_e4m3fn).t()
    a_scale = torch.ones(1, device="cuda")
    b_scales = torch.ones(n, 1, device="cuda")
    signature = _gemm_format_signature(
        a, b, a_scale, b_scales, torch.bfloat16, "fp8", None
    )
    traits = {
        "m": m,
        "n": n,
        "k": k,
        "a_inner_stride_one": True,
        "b_inner_stride_one": False,
        "b_layout": "KN",
    }
    return select_kernel("gemm", "mm", signature, traits=traits).name


def test_channel_scales_select_cutlass_from_the_min_rows_on_aligned_shapes():
    assert _selected(TORCH_FP8_SCALED_MIN_M, 1536, 1024) == "torch_mm_fp8_scaled"
    assert _selected(TORCH_FP8_SCALED_MIN_M - 1, 1536, 1024) == "triton_mm_fp8_scaled"
    assert _selected(TORCH_FP8_SCALED_MIN_M, 1000, 1024) == "triton_mm_fp8_scaled"
    assert _selected(TORCH_FP8_SCALED_MIN_M, 1536, 1000) == "triton_mm_fp8_scaled"


@pytest.mark.parametrize("m", [TORCH_FP8_SCALED_MIN_M, 1024])
@pytest.mark.parametrize(
    ("a_per_token", "b_per_channel"), [(False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_scales_apply_to_the_fp32_accumulator(m, a_per_token, b_per_channel, out_dtype):
    a, a_scales, b, b_scales = make_fp8_per_channel_gemm_operands(m, 1536, 1024, seed=m)
    if not a_per_token:
        a_scales = a_scales.max().reshape(1)
    if not b_per_channel:
        b_scales = b_scales.max().reshape(1)
    out = kernel_mm(
        a,
        b.t(),
        A_scales=a_scales,
        B_scales=b_scales,
        out_dtype=out_dtype,
        quant="fp8",
        override="torch_mm_fp8_scaled",
    )
    expected = (a.float() * a_scales) @ (b.float() * b_scales.reshape(-1, 1)).t()
    # FP32 accumulation, then 16-bit output rounding.
    torch.testing.assert_close(out.float(), expected, rtol=2**-8, atol=5e-4)


def test_output_buffer_is_written_in_place():
    a, a_scales, b, b_scales = make_fp8_per_channel_gemm_operands(
        128, 1536, 1024, seed=0
    )
    out = torch.empty(128, 1536, device="cuda", dtype=torch.bfloat16)
    result = kernel_mm(
        a,
        b.t(),
        A_scales=a_scales,
        B_scales=b_scales,
        out=out,
        out_dtype=torch.bfloat16,
        quant="fp8",
        override="torch_mm_fp8_scaled",
    )
    assert result.data_ptr() == out.data_ptr()
    expected = (a.float() * a_scales) @ (b.float() * b_scales).t()
    torch.testing.assert_close(out.float(), expected, rtol=2**-8, atol=5e-4)
