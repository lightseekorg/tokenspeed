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


def quant(x, pdl):
    return quantize_fp8(
        x,
        granularity="token_group",
        group_size=128,
        scale_encoding="packed_ue8m0",
        enable_pdl=pdl,
    )


@pytest.mark.parametrize(
    "m,k",
    [
        (1, 128),
        (3, 384),
        (5, 640),
        (4, 2560),
        (32, 1536),
        (128, 2560),
        (129, 640),
        (16384, 1536),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_packed_codes_scales_and_padding(m, k, dtype):
    torch.manual_seed(42)
    x = torch.randn(m, k, device="cuda", dtype=dtype) * 0.1
    x[:, :128] = 0
    if k >= 384:
        x[:, 128:256] = 0.4375  # Exact power-of-two scale, no log2 bump.
        x[:, 256:384] *= 1e-20
    actual, packed = quant(x, True)
    expected, scales, exponent = reference_quant(x)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    torch.testing.assert_close(unpack(packed, k // 128), scales, rtol=0, atol=0)
    assert packed.dtype == torch.int32
    assert packed.shape == (m, (k // 128 + 3) // 4)
    padded = (m + 3) // 4 * 4
    assert packed.stride() == (1, padded)
    all_rows = torch.as_strided(packed, (padded, packed.shape[1]), (1, padded)).to(
        torch.int64
    )
    shifts = torch.arange(4, device=x.device) * 8
    bytes_ = ((all_rows[..., None] >> shifts) & 255).flatten(1)
    assert torch.equal(bytes_[:m, : k // 128], exponent.to(torch.int64))
    assert torch.count_nonzero(bytes_[m:]) == 0
    assert torch.count_nonzero(bytes_[:, k // 128 :]) == 0


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("m", [1, 5, 128])
def test_packed_graph_refresh(m, pdl):
    x = torch.randn(m, 640, device="cuda", dtype=torch.bfloat16)
    quant(x, pdl)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, packed = quant(x, pdl)
    for factor in [0.0, 0.5, 8.0]:
        x.normal_().mul_(factor)
        graph.replay()
        expected, scales, _ = reference_quant(x)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        torch.testing.assert_close(unpack(packed, 5), scales, rtol=0, atol=0)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 192, 640),
        (4, 256, 2560),
        (32, 256, 1536),
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


def test_packed_rows_reuse_compilation():
    from tokenspeed_kernel.ops.quantization.triton import (
        _fp8_group128_quantize_packed_kernel,
    )
    from utils import assert_no_triton_compile

    for m in [1, 8, 129]:
        quant(torch.zeros(m, 640, device="cuda", dtype=torch.bfloat16), False)
    with assert_no_triton_compile(_fp8_group128_quantize_packed_kernel):
        for m in [2, 3, 4, 9, 16, 31, 32, 64, 128, 130, 256]:
            quant(torch.zeros(m, 640, device="cuda", dtype=torch.bfloat16), False)
