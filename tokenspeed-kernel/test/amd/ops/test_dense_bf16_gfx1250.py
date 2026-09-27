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

"""Dense decode split-K and TDM tail regression coverage."""

import pytest
import torch
from utils import is_cdna5

if not is_cdna5():
    pytest.skip("AMD CDNA5 is required", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
    gluon_wmma_tdm_dense_gfx1250,
)


@pytest.mark.parametrize("split_k", [1, 2, 4, 8, None])
@pytest.mark.parametrize("m,n", [(1, 80), (17, 128)])
def test_dense_split_k_strided_output_and_replay(split_k, m, n):
    torch.manual_seed(1250)
    k = 8192
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    storage = torch.full((m, n + 16), float("nan"), device="cuda", dtype=a.dtype)
    out = storage[:, :n]
    result = gluon_wmma_tdm_dense_gfx1250(a, b, out=out, split_k=split_k)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, a @ b.T, atol=1e-2, rtol=1e-2)
    assert torch.isnan(storage[:, n:]).all()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gluon_wmma_tdm_dense_gfx1250(a, b, out=out, split_k=split_k)
    a.copy_(torch.randn_like(a) / k**0.5)
    graph.replay()
    torch.testing.assert_close(out, a @ b.T, atol=1e-2, rtol=1e-2)
    assert torch.isnan(storage[:, n:]).all()


@pytest.mark.parametrize("k", [128, 256, 384])
def test_dense_short_k_drains_tdm(k):
    torch.manual_seed(k)
    a = torch.randn(2, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    b = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
    expected = a @ b.T
    for _ in range(3):
        actual = gluon_wmma_tdm_dense_gfx1250(a, b, split_k=1)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("k,split_k", [(1024, 0), (1024, 3), (1024, 8), (8192, 16)])
def test_dense_rejects_invalid_split(k, split_k):
    a = torch.empty(2, k, device="cuda", dtype=torch.bfloat16)
    b = torch.empty(64, k, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="one of|divide|full TDM pipeline"):
        gluon_wmma_tdm_dense_gfx1250(a, b, split_k=split_k)
