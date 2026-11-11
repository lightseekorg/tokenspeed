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

"""Blackwell packed activation quantization and dense DeepGEMM execution."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import mm, prepare_deepgemm_fp8_weight
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.platform import pdl_enabled

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires SM100 or SM103",
)


def unpack(scales, groups):
    shifts = torch.arange(4, device=scales.device) * 8
    exponents = ((scales.to(torch.int64)[..., None] >> shifts) & 255).flatten(1)[
        :, :groups
    ]
    return torch.exp2(exponents.float() - 127)


def reference_quant(x):
    m, k = x.shape
    groups = x.float().view(m, k // 128, 128)
    raw = groups.abs().amax(-1).clamp_min(1e-10) / 448
    bits = raw.view(torch.int32)
    exponent = (((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0)).clamp(1, 254)
    scales = (exponent << 23).view(torch.float32)
    codes = (
        (groups / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(m, k)
    )
    return codes, scales, exponent


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 192, 640),
        (4, 256, 2560),
        (32, 256, 1536),
        (1, 2560, 6144),
        (128, 256, 2560),
        (16384, 256, 1536),
    ],
)
@pytest.mark.parametrize("pdl", [False, True])
def test_registered_deepgemm_values_out_graph(m, n, k, pdl):
    pytest.importorskip("deep_gemm")
    previous = pdl_enabled()
    pdl_enabled(pdl)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.03
        qw, sw = quantize_fp8(w, granularity="block", block_size=[128, 128])
        original_weight = qw.float() * sw.repeat_interleave(128, 0)[
            :n
        ].repeat_interleave(128, 1)
        qw, packed_w = prepare_deepgemm_fp8_weight(qw, sw)
        prepared_weight = qw.float() * unpack(packed_w, k // 128).repeat_interleave(
            128, 1
        )
        assert (
            prepared_weight - original_weight
        ).norm() / original_weight.norm() < 0.04
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

        def run():
            return mm(
                x,
                qw,
                B_scales=packed_w,
                quant="mxfp8",
                block_size=[128, 128],
                out=out,
                solution="deepgemm",
            )

        def check():
            qa, sa, _ = reference_quant(x)
            dw = qw.float() * unpack(packed_w, k // 128).repeat_interleave(128, 1)
            ref = (qa.float() * sa.repeat_interleave(128, 1)) @ dw.t()
            torch.testing.assert_close(out.float(), ref, rtol=0.01, atol=0.001)

        assert run().data_ptr() == out.data_ptr()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        x.normal_().mul_(0.2)
        graph.replay()
        check()
    finally:
        pdl_enabled(previous)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.mark.parametrize("num_splits", [2, 3, 4, 6])
@pytest.mark.parametrize("pdl", [False, True])
def test_split_k_graph_and_variable_rows(num_splits, pdl):
    pytest.importorskip("deep_gemm")
    from tokenspeed_kernel.ops.gemm.deepgemm import (
        _reduce_split_k_kernel,
        block_fp8_mm_split_k,
    )
    from tokenspeed_kernel.thirdparty.deepgemm import block_fp8_mm
    from utils import assert_no_triton_compile

    torch.manual_seed(43)
    k, n = 6144, 512
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.03
    b, b_scale = quantize_fp8(w, granularity="block", block_size=[128, 128])
    b, b_scale = prepare_deepgemm_fp8_weight(b, b_scale)
    x = torch.randn(32, k, device="cuda", dtype=torch.bfloat16) * 0.1

    def operands(rows):
        return quantize_fp8(
            x[:rows],
            granularity="token_group",
            group_size=128,
            scale_encoding="packed_ue8m0",
            enable_pdl=pdl,
        )

    def run(rows):
        a, a_scale = operands(rows)
        out = x.new_empty((rows, n))
        return block_fp8_mm_split_k(
            a, b, a_scale, b_scale, out, num_splits=num_splits, enable_pdl=pdl
        )

    def check(rows, actual):
        a, a_scale = operands(rows)
        expected = block_fp8_mm(
            a, b, a_scale, b_scale, x.new_empty((rows, n)), enable_pdl=pdl
        )
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.002)

    old_pdl = pdl_enabled()
    pdl_enabled(pdl)
    try:
        run(4)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run(4)
        x.normal_().mul_(0.2)
        graph.replay()
        check(4, actual)
        with assert_no_triton_compile(_reduce_split_k_kernel):
            for rows in (1, 7, 17, 32):
                check(rows, run(rows))
    finally:
        pdl_enabled(old_pdl)
