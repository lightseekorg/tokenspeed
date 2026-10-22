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

"""Runtime SwiGLU accuracy on CPU and Ascend, including clamped inputs."""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.activation import SiluAndMul


@pytest.fixture(params=["cpu", "npu"])
def device(request):
    if request.param == "npu" and not current_platform().is_npu:
        pytest.skip("requires an Ascend NPU")
    return request.param


def _reference(x, limit):
    gate, up = x.double().chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp_max(limit)
        up = up.clamp(-limit, limit)
    return gate * up / (1 + torch.exp(-gate))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("limit", [None, 2.0])
@pytest.mark.parametrize(
    "shape,row_stride",
    [((0, 2048), 1), ((1, 80), 1), ((17, 80), 2), ((3, 7, 80), 1), ((2048, 2048), 1)],
)
def test_silu_and_mul_matches_reference(device, dtype, limit, shape, row_stride):
    generator = torch.Generator().manual_seed(20261022)
    storage_shape = (shape[0] * row_stride, *shape[1:])
    storage = (torch.randn(storage_shape, generator=generator) * 2).to(dtype)
    x = storage.to(device)[::row_stride]
    actual = SiluAndMul(swiglu_limit=limit)(x)
    expected = _reference(storage[::row_stride], limit)

    assert actual.shape == (*shape[:-1], shape[-1] // 2)
    assert actual.dtype == dtype
    assert actual.device == x.device
    atol, rtol = {
        torch.float32: (2**-16, 2**-10),
        torch.float16: (2**-9, 2**-9),
        torch.bfloat16: (2**-6, 2**-6),
    }[dtype]
    torch.testing.assert_close(actual.cpu().double(), expected, atol=atol, rtol=rtol)


def test_silu_and_mul_rejects_odd_width(device):
    with pytest.raises(ValueError, match="even"):
        SiluAndMul()(torch.empty(2, 3, device=device))


def test_silu_and_mul_rejects_fp8_output(device):
    with pytest.raises(NotImplementedError, match="fp8_out"):
        SiluAndMul()(torch.ones(2, 8, device=device), fp8_out=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_npu_silu_and_mul_graph_replay(dtype):
    if not current_platform().is_npu:
        pytest.skip("requires an Ascend NPU")
    x = torch.linspace(-8, 8, 17 * 80, dtype=dtype).reshape(17, 80).to("npu")
    activation = SiluAndMul()
    expected = activation(x).clone()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = activation(x)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    x.neg_()
    expected = activation(x).clone()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
