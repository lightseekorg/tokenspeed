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


"""gfx950 FP32 projection dispatch, accumulation and graph replay."""

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("gfx950 required", allow_module_level=True)

from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.fp32_output import (
    gluon_mm_bf16_fp32_narrow_gfx950,
    launch_gluon_mm_bf16_fp32_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import gluon_mm_a16w16_medium_gfx950


def project(a, b, out):
    return mm(
        a,
        b,
        A_scales=None,
        B_scales=None,
        bias=None,
        out=out,
        out_dtype=torch.float32,
        alpha=None,
        block_size=None,
        quant=None,
        override=None,
        solution=None,
        prepacked_scales=False,
    )


@pytest.mark.parametrize("n,k", [(8, 4608), (384, 4608), (3792, 4608), (4608, 1536)])
def test_gfx950_projection_batch_invariance_and_accuracy(n, k):
    torch.manual_seed(8)
    source = torch.randn((1, k), device="cuda", dtype=torch.bfloat16) * 0.3
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.03
    expected = source.double() @ weight.double().T
    first = project(source, weight, None)
    torch.testing.assert_close(first.double(), expected, rtol=2e-6, atol=2e-6)
    # The shared MFMA kernel specializes runtime scalars by alignment class.
    for count in (16, 17):
        project(source.expand(count, -1).contiguous(), weight, None)
    with assert_no_triton_compile(
        gluon_mm_bf16_fp32_narrow_gfx950, gluon_mm_a16w16_medium_gfx950
    ):
        for count in (2, 4, 8, 24, 32, 64, 71, 79, 128, 256, 1287, 1962, 2048):
            inputs = torch.randn((count, k), device="cuda", dtype=torch.bfloat16)
            row = count // 2
            inputs[row].copy_(source[0])
            actual = project(inputs, weight, None)
            assert torch.equal(actual[row : row + 1], first)
    assert not torch.equal(first, first.bfloat16().float())


@pytest.mark.parametrize("n,k", [(8, 4608), (384, 4608), (3792, 4608), (4608, 1536)])
@pytest.mark.parametrize("padding,offset", [(3, 0), (4, 1)])
def test_gfx950_projection_strided_output_graph(n, k, padding, offset):
    torch.manual_seed(11)
    a = torch.empty((17, k + padding), device="cuda", dtype=torch.bfloat16)[
        :, offset : offset + k
    ]
    b = torch.empty((n, k + padding + 2), device="cuda", dtype=torch.bfloat16)[
        :, offset : offset + k
    ]
    a.normal_(mean=0, std=0.3)
    b.normal_(mean=0, std=0.03)
    backing = torch.full((17, n + 3), 12345.0, device="cuda")
    out = backing[:, :n]
    assert project(a, b, out) is out
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        project(a, b, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        project(a, b, out)
    for _ in range(3):
        a.normal_(mean=0, std=0.3)
        b.normal_(mean=0, std=0.03)
        graph.replay()
        captured = out.clone()
        torch.testing.assert_close(
            captured.double(), a.double() @ b.double().T, rtol=2e-6, atol=2e-6
        )
        project(a, b, out)
        assert torch.equal(out, captured)
        assert (backing[:, n:] == 12345.0).all()


@pytest.mark.parametrize("operand", ["a", "b", "out"])
def test_gfx950_projection_rejects_buffer_offset_wrap(operand):
    a = torch.empty((2, 4608), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((384, 4608), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((2, 384), device="cuda", dtype=torch.float32)
    if operand == "a":
        a = torch.empty_strided(a.shape, (2**31, 1), device=a.device, dtype=a.dtype)
    elif operand == "b":
        b = torch.empty_strided(b.shape, (2**23, 1), device=b.device, dtype=b.dtype)
    else:
        out = torch.empty_strided(
            out.shape, (2**30, 1), device=out.device, dtype=out.dtype
        )
    with pytest.raises(ValueError, match="32-bit buffer offsets"):
        project(a, b, out)


def test_gfx950_projection_rejects_split_k_offset_overflow():
    a = torch.empty((1, 4608), device="cuda", dtype=torch.bfloat16).expand(65536, -1)
    b = torch.empty((3792, 4608), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((1, 3792), device="cuda", dtype=torch.float32).expand(65536, -1)
    counters = torch.empty(0, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="Unsupported FP32 projection shape"):
        launch_gluon_mm_bf16_fp32_gfx950(a, b, out, counters=counters)
