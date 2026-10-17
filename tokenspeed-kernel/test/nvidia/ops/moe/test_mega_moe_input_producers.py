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

"""Direct NVFP4 and routing stores preserve the existing producer results."""

import pytest
import torch
from utils import assert_no_triton_compile, is_nvidia

if not is_nvidia() or not torch.cuda.is_available():
    pytest.skip("NVIDIA GPU required", allow_module_level=True)
if torch.cuda.get_device_capability()[0] < 10:
    pytest.skip("Blackwell GPU required", allow_module_level=True)

from tokenspeed_kernel.ops.moe import moe_topk  # noqa: E402
from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (  # noqa: E402
    _kimi3_sigmoid_bias_topk_kernel,
)
from tokenspeed_kernel.ops.moe._triton.minimax_topk import (  # noqa: E402
    _minimax_biased_grouped_topk_kernel,
    _scale_sigmoid_weights_kernel,
)
from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize  # noqa: E402


@pytest.mark.parametrize("tokens", [0, 1, 3, 16, 17, 257])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_nvfp4_output_buffers(tokens, enable_pdl):
    torch.manual_seed(713)
    hidden = 3584
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([1.25], device="cuda")
    padded = (tokens + 15) // 16 * 16
    value_count = tokens * hidden // 2
    scale_count = padded * hidden // 16
    value_storage = torch.full(
        (value_count + 256,), 123, device="cuda", dtype=torch.uint8
    )
    scale_storage = torch.full(
        (scale_count + 256,), 123, device="cuda", dtype=torch.uint8
    )
    outputs = (
        value_storage[:value_count].view(tokens, hidden // 2),
        scale_storage[:scale_count].view(padded, hidden // 16),
    )
    kwargs = dict(is_sf_swizzled_layout=False, enable_pdl=enable_pdl)
    actual = fp4_quantize(x, scale, out=outputs, **kwargs)
    if tokens:
        expected = fp4_quantize(x, scale, **kwargs)
        for got, ref in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, ref, atol=0, rtol=0)
        assert actual[0].data_ptr() == outputs[0].data_ptr()
        assert actual[1].data_ptr() == outputs[1].data_ptr()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fp4_quantize(x, scale, out=outputs, **kwargs)
        x.normal_()
        expected = fp4_quantize(x, scale, **kwargs)
        graph.replay()
        for got, ref in zip(captured, expected, strict=True):
            torch.testing.assert_close(got, ref, atol=0, rtol=0)
    assert torch.all(value_storage[value_count:] == 123)
    assert torch.all(scale_storage[scale_count:] == 123)


def _route(logits, bias, normalize, out):
    return moe_topk(
        logits,
        16,
        score_function="sigmoid",
        selection_method="topk",
        renormalize=normalize,
        routed_scaling_factor=2.5,
        correction_bias=bias,
        topk_indices_dtype=torch.int32 if out is None else torch.int64,
        topk_weights_dtype=torch.bfloat16,
        out=out,
    )


@pytest.mark.parametrize("tokens", [0, 1, 3, 64, 256, 257, 1024])
@pytest.mark.parametrize("normalize", [False, True])
def test_routing_output_buffers(tokens, normalize):
    torch.manual_seed(713)
    logits = torch.randn(tokens, 896, device="cuda")
    bias = torch.randn(896, device="cuda") * 0.01
    outputs = (
        torch.empty(tokens, 16, device="cuda", dtype=torch.float32),
        torch.empty(tokens, 16, device="cuda", dtype=torch.int64),
    )
    expected = _route(logits, bias, normalize, None)
    actual = _route(logits, bias, normalize, outputs)
    assert actual[0] is outputs[0]
    assert actual[1] is outputs[1]
    torch.testing.assert_close(actual[0], expected[0].float(), atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1].long(), atol=0, rtol=0)
    if tokens:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _route(logits, bias, normalize, outputs)
        logits.normal_()
        expected = _route(logits, bias, normalize, None)
        graph.replay()
        torch.testing.assert_close(captured[0], expected[0].float(), atol=0, rtol=0)
        torch.testing.assert_close(captured[1], expected[1].long(), atol=0, rtol=0)


def test_routing_runtime_row_counts_do_not_compile():
    bias = torch.randn(896, device="cuda")
    for tokens in [3, 257]:
        logits = torch.randn(tokens, 896, device="cuda")
        out = (
            torch.empty(tokens, 16, device="cuda"),
            torch.empty(tokens, 16, device="cuda", dtype=torch.int64),
        )
        _route(logits, bias, False, out)
    with assert_no_triton_compile(
        _kimi3_sigmoid_bias_topk_kernel,
        _minimax_biased_grouped_topk_kernel,
        _scale_sigmoid_weights_kernel,
    ):
        for tokens in [5, 17, 264, 513]:
            logits = torch.randn(tokens, 896, device="cuda")
            out = (
                torch.empty(tokens, 16, device="cuda"),
                torch.empty(tokens, 16, device="cuda", dtype=torch.int64),
            )
            _route(logits, bias, False, out)
