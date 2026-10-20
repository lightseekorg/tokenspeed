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

from __future__ import annotations

import pytest
import tokenspeed_kernel
import torch
import torch.nn.functional as F
import torch_npu
from tokenspeed_kernel.platform import current_platform


@pytest.fixture(autouse=True, scope="module")
def _requires_npu():
    if not current_platform().is_npu:
        pytest.skip("requires an Ascend NPU")
    torch.npu.set_device(0)


@pytest.mark.parametrize(
    ("shape", "transpose"),
    (
        ((3072, 512), False),
        ((3072, 384), False),
        ((1536, 3072), True),
        ((576, 3072), True),
        ((6144, 1536), False),
        ((3072, 4096), False),
    ),
)
@pytest.mark.parametrize("batch_size", (1, 2))
def test_weight_nz_exact_shapes_eager_and_graph(shape, transpose, batch_size):
    torch.manual_seed(11100 + batch_size + shape[0])
    source = torch.randn(shape, device="npu:0", dtype=torch.bfloat16) * 0.01
    prepared = tokenspeed_kernel.prepare_weight_nz(source, transpose=transpose)
    assert torch_npu.get_npu_format(prepared) == 29
    assert prepared.shape == (shape[::-1] if transpose else shape)

    def apply(hidden):
        return hidden @ prepared if transpose else F.linear(hidden, prepared)

    hidden = torch.randn(batch_size, shape[1], device="npu:0", dtype=torch.bfloat16)
    for _ in range(4):
        output = apply(hidden)
    expected = F.linear(hidden, source)
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph,
        stream=torch.npu.Stream(),
        auto_dispatch_capture=True,
    ):
        output = apply(hidden)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    first = output.clone()

    hidden.copy_(torch.randn_like(hidden))
    expected = F.linear(hidden, source)
    graph.replay()
    torch.npu.synchronize()
    assert not torch.equal(first, output)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)


def test_weight_nz_restores_internal_format_setting():
    config = torch.npu.config
    had_setting = hasattr(config, "allow_internal_format")
    previous = getattr(config, "allow_internal_format", None)
    try:
        if had_setting:
            del config.allow_internal_format
        weight = torch.randn(32, 32, device="npu:0", dtype=torch.bfloat16)
        prepared = tokenspeed_kernel.prepare_weight_nz(weight)
        assert torch_npu.get_npu_format(prepared) == 29
        assert not hasattr(config, "allow_internal_format")
    finally:
        if had_setting:
            config.allow_internal_format = previous
