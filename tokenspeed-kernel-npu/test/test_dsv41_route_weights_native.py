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

"""Unique route mapping: exact values, bounded writes and mutable native graphs."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu.ops.dsv41_route_weights import route_weights
from tokenspeed_kernel_npu.ops.dsv41_route_weights_native import (
    _library,
    route_weights_native,
)


def reference(mapping, weights):
    n = mapping.numel()
    output = torch.zeros(n + 1, device=weights.device, dtype=torch.float32)
    output.scatter_add_(0, torch.where(mapping >= 0, mapping, n).long(), weights)
    return output[:n]


def inputs(n, active):
    mapping = torch.full((n,), -1, dtype=torch.int32)
    sources = torch.randperm(n)[:active]
    mapping[sources] = torch.randperm(active).int()
    return mapping.npu(), torch.randn(n, device="npu")


def assert_bits(a, b):
    a, b = a.cpu(), b.cpu()
    nan = torch.isnan(b)
    assert torch.equal(torch.isnan(a), nan)
    assert torch.equal(a.view(torch.int32)[~nan], b.view(torch.int32)[~nan])


@pytest.mark.parametrize("n", [513, 960, 1152, 4095, 4096])
@pytest.mark.parametrize("fraction", [0, 1 / 16, 1])
def test_native_exact_and_output_guards(n, fraction):
    torch.npu.set_device(0)
    assert _library() is not None
    torch.manual_seed(371)
    mapping, weights = inputs(n, int(n * fraction))
    weights[:5].copy_(
        torch.tensor(
            [-0.0, 0.0, float("inf"), float("-inf"), float("nan")], device="npu"
        )
    )
    expected = reference(mapping, weights)
    assert_bits(route_weights(mapping, weights), expected)
    backing = torch.full((n + 32,), 71.0, device="npu")
    out = backing[16 : 16 + n]
    _library().launch_route_weights(
        mapping.data_ptr(),
        weights.data_ptr(),
        out.data_ptr(),
        n,
        16,
        torch.npu.current_stream().npu_stream,
    )
    torch.npu.synchronize()
    assert_bits(out, expected)
    assert torch.equal(backing[:16].cpu(), torch.full((16,), 71.0))
    assert torch.equal(backing[16 + n :].cpu(), torch.full((16,), 71.0))


@pytest.mark.parametrize("n", [513, 1152, 4096])
def test_native_mutable_graph(n):
    torch.npu.set_device(0)
    mapping, weights = inputs(n, n // 16)
    route_weights(mapping, weights)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = route_weights(mapping, weights)
    for active in [0, n, n // 16]:
        new_map, new_weights = inputs(n, active)
        mapping.copy_(new_map)
        weights.copy_(new_weights)
        graph.replay()
        torch.npu.synchronize()
        assert_bits(output, reference(mapping, weights))


def test_geometry_fallback():
    torch.npu.set_device(0)
    mapping, weights = inputs(1152, 72)
    assert route_weights_native(mapping.long(), weights) is None
    backing = torch.empty(2304, device="npu")
    backing[::2].copy_(weights)
    assert route_weights_native(mapping, backing[::2]) is None
    assert_bits(route_weights(mapping.long(), weights), reference(mapping, weights))
    assert_bits(route_weights(mapping, backing[::2]), reference(mapping, weights))
    assert route_weights_native(mapping[:0], weights[:0]) is None
