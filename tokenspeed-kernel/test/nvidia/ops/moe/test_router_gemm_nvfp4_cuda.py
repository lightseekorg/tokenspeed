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

"""Router producer arithmetic, graph reuse and dispatch boundaries."""

import pytest
import torch
from tokenspeed_kernel.ops.moe import try_router_gemm_nvfp4
from tokenspeed_kernel.platform import ArchVersion, current_platform, pdl_enabled

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or current_platform().arch_version != ArchVersion(10, 7),
    reason="requires SM107",
)


@pytest.fixture(autouse=True)
def pdl():
    previous = pdl_enabled()
    pdl_enabled(True)
    yield
    pdl_enabled(previous)


def inputs(rows, hidden=2560, experts=512):
    generator = torch.Generator(device="cuda").manual_seed(29)
    x = torch.randn(rows, hidden, generator=generator, device="cuda").bfloat16()
    weight = (
        torch.randn(experts, hidden, generator=generator, device="cuda") * 0.05
    ).bfloat16()
    scale = torch.tensor([128.0], device="cuda")
    return x, weight, scale


def check(actual, x, weight, scale):
    from flashinfer import fp4_quantize

    logits, (packed, scales) = actual
    expected_logits = (x.double() @ weight.double().T).to(torch.bfloat16)
    torch.testing.assert_close(logits, expected_logits, atol=1e-3, rtol=1e-2)
    expected_packed, expected_scales = fp4_quantize(
        x, scale, is_sf_swizzled_layout=False, enable_pdl=True
    )

    def fp4_codes(data):
        # Positive and negative FP4 zero are the same value; other codes are
        # unique. Match the existing quantizer's values, not zero sign bits.
        codes = torch.stack((data & 15, data >> 4), dim=-1)
        return torch.where(codes == 8, 0, codes)

    torch.testing.assert_close(
        fp4_codes(packed), fp4_codes(expected_packed), atol=0, rtol=0
    )
    torch.testing.assert_close(
        scales.view(torch.uint8),
        expected_scales.view(torch.uint8).reshape_as(scales),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("rows", [1, 4])
def test_router_producer_eager_and_graph(rows):
    x, weight, scale = inputs(rows)
    check(try_router_gemm_nvfp4(x, weight, scale), x, weight, scale)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = try_router_gemm_nvfp4(x, weight, scale)
    torch.cuda.current_stream().wait_stream(stream)
    for factor in (0.0, -0.75, 1.25):
        x.copy_(torch.randn_like(x) * factor)
        weight.mul_(-1)
        scale.mul_(0.5)
        graph.replay()
        check(actual, x, weight, scale)


def test_dispatch_fallbacks_and_override():
    x, weight, scale = inputs(8)
    assert try_router_gemm_nvfp4(x, weight, scale) is None
    unaligned = torch.empty(x[:4].numel() + 1, device=x.device, dtype=x.dtype)[
        1:
    ].view_as(x[:4])
    unaligned.copy_(x[:4])
    assert try_router_gemm_nvfp4(unaligned, weight, scale) is None
    pdl_enabled(False)
    assert try_router_gemm_nvfp4(x[:4], weight, scale) is None
    with pytest.raises(ValueError, match="PDL"):
        try_router_gemm_nvfp4(x[:4], weight, scale, override="cuda_router_gemm_nvfp4")
    pdl_enabled(True)
    # Different static dimensions exercise the tile's contract, not just its
    # measured admission. Warm M1; capturing other row counts must not compile.
    x, weight, scale = inputs(8, hidden=512, experts=64)
    override = "cuda_router_gemm_nvfp4"
    assert try_router_gemm_nvfp4(x, weight, scale) is None
    check(
        try_router_gemm_nvfp4(x[:1], weight, scale, override=override),
        x[:1],
        weight,
        scale,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        results = [
            try_router_gemm_nvfp4(x[:rows], weight, scale, override=override)
            for rows in (2, 4, 8)
        ]
    torch.cuda.current_stream().wait_stream(stream)
    x.neg_()
    graph.replay()
    for rows, result in zip((2, 4, 8), results, strict=True):
        check(result, x[:rows], weight, scale)
