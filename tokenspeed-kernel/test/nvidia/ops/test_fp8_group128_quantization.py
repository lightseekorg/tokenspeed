# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Direct canonical-scale FP8 quantization on Blackwell."""

import pytest
import torch
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.ops.quantization.trtllm import _per_token_group_quant_8bit

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires an SM10x GPU",
)


def _quantize(x, *, enable_pdl=False):
    return quantize_fp8(
        x,
        granularity="token_group",
        group_size=128,
        scale_encoding="float32",
        enable_pdl=enable_pdl,
        override="triton_quantize_fp8_group128",
    )


@pytest.mark.parametrize(
    "m,k", [(1, 128), (3, 384), (8, 8192), (32, 2048), (129, 8192), (4096, 2048)]
)
def test_direct_scales_and_roundtrip(m, k):
    torch.manual_seed(42)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    # Every group owns its scale; include zero groups and varied magnitudes.
    x[:, :128] = 0
    if k > 128:
        x[:, 128:256] *= 100
    if k >= 384:
        x[:, 256:384] *= 1e-20
    q, scales = _quantize(x)
    assert q.shape == x.shape
    assert q.dtype == torch.float8_e4m3fn
    assert scales.shape == (m, k // 128)
    assert scales.dtype == torch.float32
    assert scales.is_contiguous()
    assert torch.all(scales[:, 0] > 0)
    dequantized = q.float() * scales.repeat_interleave(128, dim=-1)
    torch.testing.assert_close(
        dequantized, x.float(), atol=scales.max().item() * 2**-10, rtol=2**-4
    )
    native_q, native_s = _per_token_group_quant_8bit(x, 128, False)
    assert torch.equal(q.view(torch.uint8), native_q.view(torch.uint8))
    torch.testing.assert_close(scales, native_s.t(), atol=0, rtol=0)


@pytest.mark.parametrize("m", [1, 8, 129])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_graph_replay_refreshes_values_and_scales(m, enable_pdl):
    x = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16)
    _quantize(x, enable_pdl=enable_pdl)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_q, actual_s = _quantize(x, enable_pdl=enable_pdl)
    x.normal_().mul_(3)
    graph.replay()
    expected_q, expected_s = _quantize(x, enable_pdl=enable_pdl)
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    torch.testing.assert_close(actual_s, expected_s, atol=0, rtol=0)


def test_batch_changes_reuse_compilation():
    from tokenspeed_kernel.ops.quantization.triton import _fp8_group128_quantize_kernel
    from utils import assert_no_triton_compile

    def run(m):
        x = torch.zeros(m, 2048, device="cuda", dtype=torch.bfloat16)
        _quantize(x)

    # Warm each group-count bucket and scalar divisibility specialization.
    for m in [1, 8, 129, 144]:
        run(m)
    with assert_no_triton_compile(_fp8_group128_quantize_kernel):
        for m in [2, 3, 9, 16, 31, 64, 128, 130, 256, 4096]:
            run(m)
