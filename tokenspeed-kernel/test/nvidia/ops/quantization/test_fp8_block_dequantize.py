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

"""Weight-only decode must retain the FP32 product before BF16 rounding."""

import pytest
import torch
from tokenspeed_kernel.ops.quantization import dequantize_fp8_block

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "shape", [(128, 256), (129, 131), (3, 256, 384), (0, 128, 128)]
)
def test_fp8_block_dequantize(shape):
    q = (torch.randn(shape, device="cuda") * 20).to(torch.float8_e4m3fn)
    scales = (
        torch.rand(
            (*shape[:-2], (shape[-2] + 127) // 128, (shape[-1] + 127) // 128),
            device="cuda",
        )
        + 0.01
    )
    expanded = scales.repeat_interleave(128, -2).repeat_interleave(128, -1)[
        ..., : shape[-2], : shape[-1]
    ]
    expected = (q.float() * expanded).bfloat16()
    before = q.view(torch.uint8).clone()
    actual = dequantize_fp8_block(q, scales)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(q.view(torch.uint8), before)


def test_graph_uses_updated_codes_and_scales():
    q = torch.ones((256, 256), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.ones((2, 2), device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dequantize_fp8_block(q, scales)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = dequantize_fp8_block(q, scales)
    q.copy_(
        torch.full_like(scales, 2).repeat_interleave(128, 0).repeat_interleave(128, 1)
    )
    scales.fill_(0.37)
    graph.replay()
    assert torch.equal(output, torch.full_like(output, 0.74))


@pytest.mark.parametrize(
    "scales", [torch.ones(1, 1), torch.ones(2, 2, dtype=torch.bfloat16)]
)
def test_rejects_malformed_scale_storage(scales):
    q = torch.ones((256, 256), device="cuda").to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="128x128 FP32"):
        dequantize_fp8_block(q, scales.cuda())
