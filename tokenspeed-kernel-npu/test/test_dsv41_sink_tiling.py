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

"""Sink row tiling preserves pointwise BF16 output and native replay."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu._triton import triton
from tokenspeed_kernel_npu.ops.dsv41_attention_fusion import _sink, apply_sink


def original(output, maximum, total, sink, counts):
    tokens, heads, dim = output.shape
    result = torch.empty_like(output)
    if tokens:
        _sink[(min(triton.cdiv(tokens * heads, 4), 256),)](
            output,
            maximum,
            total,
            sink,
            counts,
            result,
            tokens * heads,
            heads,
            dim,
            4,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return result


@pytest.mark.parametrize(
    "tokens,heads", [(0, 16), (1, 4), (8, 16), (17, 9), (48, 16), (192, 4), (257, 16)]
)
def test_exact_sink_tiling_and_native_replay(tokens, heads):
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(117)
    x = torch.randn(tokens, heads, 512, device="npu", dtype=torch.bfloat16)
    maximum = torch.randn(tokens, heads, device="npu")
    total = torch.rand(tokens, heads, device="npu") + 0.1
    sink = torch.randn(heads, device="npu")
    counts = torch.ones(tokens, device="npu", dtype=torch.int32)
    apply_sink(x, maximum, total, sink, counts)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        actual = apply_sink(x, maximum, total, sink, counts)
    for step in range(4):
        x.copy_(torch.randn_like(x))
        maximum.mul_(3)
        sink.add_(step)
        counts.fill_(step % 2)
        if tokens:
            counts[::3] = 0
            if step == 3:
                x[0, 0, 0] = float("nan")
                maximum[0, 0] = -float("inf")
                total[0, 0] = 0
        graph.replay()
        torch.npu.synchronize()
        expected = original(x, maximum, total, sink, counts)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
        # Independent expression on CPU; the BF16 SFA output is rounded
        # before the factor, matching the existing attention contract.
        factor = torch.sigmoid(maximum.cpu() + total.cpu().log() - sink.cpu())
        ref = torch.where(
            counts.cpu()[:, None, None] > 0, x.cpu().float() * factor[:, :, None], 0
        ).bfloat16()
        torch.testing.assert_close(
            actual.cpu(), ref, atol=0.0001, rtol=0.0079, equal_nan=True
        )
