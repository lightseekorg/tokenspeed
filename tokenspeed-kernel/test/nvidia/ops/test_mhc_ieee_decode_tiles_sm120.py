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

import pytest
import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.residual.triton import (
    _mhc_prenorm_gemm_triton,
    _mhc_prenorm_gemm_triton_kernel,
    mhc_pre_mix_hc4,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120",
)


@pytest.mark.parametrize("rows", [1, 6, 12, 24, 32, 48, 64, 96])
def test_mhc_ieee_decode_tiles(rows):
    torch.manual_seed(55)
    k, n, ns = 20480, 24, 80
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda") / k**0.5
    scale = torch.ones(3, device="cuda")
    base = torch.randn(24, device="cuda")
    reference = None
    for candidate in (False, True):
        mul = torch.empty(ns, rows, n, device="cuda")
        sq = torch.empty(ns, rows, device="cuda")
        pre = torch.empty(rows, 4, device="cuda")
        post = torch.empty_like(pre)
        comb = torch.empty(rows, 16, device="cuda")

        def run():
            if candidate:
                _mhc_prenorm_gemm_triton(x, weight, mul, sq, ns)
            else:
                _mhc_prenorm_gemm_triton_kernel[(ns, triton.cdiv(rows, 16), 1)](
                    x,
                    weight,
                    mul,
                    sq,
                    rows,
                    K=k,
                    N=n,
                    SPLIT_K=256,
                    BLOCK_M=16,
                    BLOCK_N=32,
                    BLOCK_K=64,
                    DOT_PRECISION="ieee",
                    num_warps=4,
                    num_stages=1,
                )
            mhc_pre_mix_hc4(
                mul,
                sq,
                scale,
                base,
                pre,
                post,
                comb,
                hidden_size=5120,
                rms_eps=1e-6,
                hc_eps=1e-6,
                sinkhorn_iters=20,
                n_splits=ns,
                num_tokens=rows,
            )

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
        torch.cuda.synchronize()
        actual = [t.clone() for t in (mul, sq, pre, post, comb)]
        if reference is None:
            reference = actual
        else:
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
