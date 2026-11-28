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
from utils import assert_no_triton_compile, is_rdna4

if not is_rdna4():
    pytest.skip("AMD RDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.numerics.torch.quantization import (  # noqa: E402
    dequantize_mxfp4,
)
from tokenspeed_kernel.ops.gemm import (  # noqa: E402
    mxfp4_linear,
    mxfp4_swiglu_linear,
    prepare_mxfp4_linear,
)
from tokenspeed_kernel.ops.gemm._triton.mxfp4 import triton_mm_mxfp4  # noqa: E402
from tokenspeed_kernel.ops.gemm.triton import (  # noqa: E402
    triton_mxfp4_linear,
    triton_mxfp4_swiglu_linear,
)
from tokenspeed_kernel.ops.quantization.triton import mxfp4_quantize  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx120x.gemm.mxfp4_decode import (  # noqa: E402
    gluon_mm_mxfp4_decode_gfx120x,
    gluon_mm_mxfp4_decode_quantize_gfx120x,
)
from tokenspeed_kernel_amd.ops.gfx120x.gemm.mxfp4_decode import (  # noqa: E402
    launch_gluon_mm_mxfp4_decode_gfx120x as mxfp4_decode,
)


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize("n,k", [(5120, 6144), (16480, 5120), (5120, 17408)])
def test_mxfp4_decode_matches_tiled_gemm(m: int, n: int, k: int) -> None:
    torch.manual_seed(m + n + k)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 3
    weight = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    scale = torch.randint(118, 124, (n, k // 32), device="cuda", dtype=torch.uint8)
    out = mxfp4_decode(x, weight, scale)
    # The tiled kernel quantizes with mxfp4_quantize; four rows take it.
    a, a_scale = mxfp4_quantize(x)
    rows = torch.arange(4, device="cuda") % m
    expected = triton_mm_mxfp4(a[rows], weight, a_scale[rows], scale, torch.bfloat16)
    torch.testing.assert_close(out.float(), expected[:m].float(), atol=1e-3, rtol=2**-8)


@pytest.mark.parametrize("m", [4, 77, 512])
@pytest.mark.parametrize("n,k", [(1000, 2080), (5120, 17408)])
def test_tiled_mxfp4_gemm_matches_reference(m: int, n: int, k: int) -> None:
    torch.manual_seed(m + n + k)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 3
    weight = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    scale = torch.randint(118, 124, (n, k // 32), device="cuda", dtype=torch.uint8)
    a, a_scale = mxfp4_quantize(x)
    out = triton_mm_mxfp4(a, weight, a_scale, scale, torch.bfloat16)
    expected = dequantize_mxfp4(a, a_scale) @ dequantize_mxfp4(weight, scale).T
    torch.testing.assert_close(out.float(), expected, atol=1e-3, rtol=2**-8)


@pytest.mark.parametrize("m", [1, 2, 3, 77])
@pytest.mark.parametrize("swiglu", [False, True])
def test_mxfp4_linear_matches_portable_kernel(m: int, swiglu: bool) -> None:
    torch.manual_seed(m)
    x = torch.randn((m, 5120), device="cuda", dtype=torch.bfloat16) * 3
    weight = torch.randint(0, 256, (2048, 2560), device="cuda", dtype=torch.uint8)
    scale = torch.randint(118, 124, (2048, 160), device="cuda", dtype=torch.uint8)
    prepared = prepare_mxfp4_linear(weight, scale, swiglu=swiglu)
    op, portable = (
        (mxfp4_swiglu_linear, triton_mxfp4_swiglu_linear)
        if swiglu
        else (mxfp4_linear, triton_mxfp4_linear)
    )
    out = op(x, prepared)
    expected = portable(x, weight, scale, out=None)
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-3, rtol=2**-8)


def test_mxfp4_decode_row_count_does_not_recompile() -> None:
    weight = torch.randint(0, 256, (5120, 3072), device="cuda", dtype=torch.uint8)
    scale = torch.randint(118, 124, (5120, 192), device="cuda", dtype=torch.uint8)
    x = torch.randn((3, 2 * 6144), device="cuda", dtype=torch.bfloat16)

    mxfp4_decode(x[:2, :6144], weight, scale)
    kernels = (gluon_mm_mxfp4_decode_gfx120x, gluon_mm_mxfp4_decode_quantize_gfx120x)
    with assert_no_triton_compile(*kernels):
        # Row views of a wider tensor change the row stride, not the binary.
        for m in (1, 3, 2):
            mxfp4_decode(x[:m, 6144:], weight, scale)
