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
from tokenspeed_kernel.ops.gemm import mm, prepare_mxfp8_weight_scales
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.platform import ArchVersion, Platform

pytestmark = pytest.mark.skipif(
    not Platform.get().is_nvidia or Platform.get().arch_version != ArchVersion(12, 0),
    reason="SM120 only",
)


def check(actual, expected):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.01, atol=0.02)
    relative_rmse = (
        (actual.float() - expected).square().mean() / expected.square().mean()
    ).sqrt()
    assert relative_rmse < 0.003


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 1792, 5120),
        (6, 1792, 5120),
        (24, 5120, 8192),
        (192, 576, 5120),
        (192, 5120, 288),
        (8192, 1792, 5120),
    ],
)
def test_prepared_gemm_graph(m, n, k):
    pytest.importorskip("deep_gemm_sm120")
    torch.manual_seed(37)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    b = (torch.randn(n, k, device="cuda") * 16).to(torch.float8_e4m3fn)
    sb = torch.full((n, k // 32), 120, device="cuda", dtype=torch.uint8)
    packed = prepare_mxfp8_weight_scales(b, sb, solution="deepgemm")
    assert (packed is None) == (k % 128 != 0)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(
        B_scales=packed if packed is not None else sb,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[1, 32],
        prepacked_scales=packed is not None,
        out=out,
    )

    def forward():
        return mm(x, b, **kwargs)

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = forward()
    assert result.data_ptr() == out.data_ptr()
    for _ in range(2):
        x.normal_(0, 0.1)
        graph.replay()
        a, sa = quantize_fp8(
            x, granularity="token_group", group_size=32, scale_encoding="ue8m0"
        )
        reference = (
            a.float() * torch.exp2(sa.float() - 127).repeat_interleave(32, 1)
        ) @ (b.float() * 0.0078125).T
        check(out, reference)
        result = mm(a, b, A_scales=sa, **kwargs)
        check(result, reference)
    if packed is not None:
        with pytest.raises(ValueError):
            mm(x, b, **{**kwargs, "override": "flashinfer_mm_mxfp8"})


def test_prepared_rejects_bad_output():
    pytest.importorskip("deep_gemm_sm120")
    x = torch.ones(6, 5120, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(1792, 5120, device="cuda").to(torch.float8_e4m3fn)
    sb = torch.full((1792, 160), 120, device="cuda", dtype=torch.uint8)
    packed = prepare_mxfp8_weight_scales(b, sb, solution="deepgemm")
    with pytest.raises(ValueError):
        mm(
            x,
            b,
            B_scales=packed,
            quant="mxfp8",
            block_size=[1, 32],
            prepacked_scales=True,
            out=torch.empty(6, 1792, device="cuda", dtype=torch.float32),
        )
