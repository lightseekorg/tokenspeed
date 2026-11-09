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

from tokenspeed_kernel.ops.gemm import mm  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx120x.gemm.bf16_decode import (  # noqa: E402
    gluon_mm_bf16_decode_gfx120x,
)


@pytest.mark.parametrize("m", [1, 2, 3, 4])
@pytest.mark.parametrize("n,k", [(248320, 5120), (1000, 256)])
def test_bf16_decode_mm_matches_fp32(m: int, n: int, k: int) -> None:
    torch.manual_seed(m + n)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    out = mm(a, w)
    expected = (a.float() @ w.float().T).to(torch.bfloat16)
    # FP32 sums in another order: one BF16 rounding apart.
    torch.testing.assert_close(out, expected, rtol=8e-3, atol=1e-2)


def test_bf16_decode_mm_rows_share_one_compile() -> None:
    w = torch.randn((512, 256), device="cuda", dtype=torch.bfloat16)
    a = torch.randn((5, 256), device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        mm(a, w), (a.float() @ w.float().T).to(torch.bfloat16), rtol=8e-3, atol=1e-2
    )
    mm(a[:2], w)
    mm(a[:3], w)
    with assert_no_triton_compile(gluon_mm_bf16_decode_gfx120x):
        mm(a[:4], w)
