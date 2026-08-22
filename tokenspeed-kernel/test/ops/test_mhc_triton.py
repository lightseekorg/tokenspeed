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
from tokenspeed_kernel.ops.mhc.triton import _mhc_prenorm_gemm_triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def test_mhc_prenorm_gemm_covers_all_columns_for_hc5() -> None:
    torch.manual_seed(17)
    tokens = 3
    hc_mult = 5
    hidden_size = 64
    n = 2 * hc_mult + hc_mult**2
    x = torch.randn(
        (tokens, hc_mult * hidden_size), device="cuda", dtype=torch.bfloat16
    )
    fn = torch.randn((n, x.shape[1]), device="cuda", dtype=torch.float32)
    out_mul = torch.empty((1, tokens, n), device="cuda", dtype=torch.float32)
    out_sqrsum = torch.empty((1, tokens), device="cuda", dtype=torch.float32)

    _mhc_prenorm_gemm_triton(x, fn, out_mul, out_sqrsum, n_splits=1)

    torch.testing.assert_close(
        out_mul.sum(dim=0),
        x.float() @ fn.T,
        rtol=2e-4,
        atol=2e-3,
    )
    torch.testing.assert_close(
        out_sqrsum.sum(dim=0),
        x.float().square().sum(dim=1),
        rtol=2e-5,
        atol=2e-3,
    )
