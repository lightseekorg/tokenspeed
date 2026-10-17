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
from tokenspeed_kernel.ops.gemm import _gemm_format_signature, mm
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.selection import select_kernel

pytestmark = pytest.mark.skipif(not Platform.get().is_hopper, reason="Hopper only")


def operands(m, n, k):
    torch.manual_seed(7)
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.rand(m, (k + 127) // 128, device="cuda") * 0.03 + 0.01
    sb = torch.rand((n + 127) // 128, (k + 127) // 128, device="cuda") * 0.03 + 0.01
    return a, b, sa, sb


def choose(a, b, sa, sb, solution):
    m, k = a.shape
    n = b.shape[0]
    sig = _gemm_format_signature(a, b, sa, sb, torch.bfloat16, "mxfp8", [128, 128])
    return select_kernel(
        "gemm",
        "mm",
        sig,
        solution=solution,
        traits={
            "m": m,
            "n": n,
            "k": k,
            "a_inner_stride_one": True,
            "b_inner_stride_one": True,
            "block_scale_layout": "canonical",
            "out_dtype": torch.bfloat16,
            "out_inner_stride_one": True,
        },
    ).name


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 256, 4096),
        (2, 2112, 4096),
        (17, 512, 512),
        (64, 1024, 512),
        (129, 256, 256),
        (1, 4096, 128),
    ],
)
def test_selection_values_out_and_graph(m, n, k):
    a, b, sa, sb = operands(m, n, k)
    if m == 17:
        sa = sa.t().contiguous().t()

    assert choose(a, b, sa, sb, None) == "triton_mm_fp8_blockscale"
    assert choose(a, b, sa, sb, "deepgemm") == "deepgemm_mm_fp8_blockscale"
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

    def run():
        return mm(
            a,
            b,
            A_scales=sa,
            B_scales=sb,
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[128, 128],
            solution="deepgemm",
            out=out,
        )

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False

    def check():
        ref = (a.float() * sa.repeat_interleave(128, 1)[:, :k]) @ (
            b.float() * sb.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
        ).T
        torch.testing.assert_close(out.float(), ref, atol=2e-4, rtol=5e-3)

    try:
        assert run().data_ptr() == out.data_ptr()
        check()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        for factor in [0.0, 0.5, 2.0]:
            a.copy_((torch.randn(m, k, device="cuda") * factor).to(a.dtype))
            g.replay()
            check()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("bad", ["shape", "dtype", "scale", "alpha", "block", "layout"])
def test_forced_override_rejects_unsupported(bad):
    a, b, sa, sb = operands(1, 256, 256)
    kwargs = dict(
        A_scales=sa,
        B_scales=sb,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[128, 128],
        override="deepgemm_mm_fp8_blockscale",
    )
    if bad == "shape":
        b = b[:255]
        assert choose(a, b, sa, sb, None) == "triton_mm_fp8_blockscale"
    elif bad == "dtype":
        kwargs["out_dtype"] = torch.float16
    elif bad == "scale":
        kwargs["A_scales"] = sa.to(torch.uint8)
    elif bad == "alpha":
        kwargs["alpha"] = torch.ones((), device="cuda")
    elif bad == "block":
        kwargs["block_size"] = [64, 128]
    elif bad == "layout":
        b = b.t()
    with pytest.raises((ValueError, AssertionError)):
        mm(a, b, **kwargs)
