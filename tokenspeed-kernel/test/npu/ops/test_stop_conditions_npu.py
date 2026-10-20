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

"""NPU graph stop conditions preserve CPU results for padding and empty outputs."""

import pytest
import torch

pytest.importorskip("torch_npu")
from tokenspeed_kernel.ops.metadata import apply_stop_conditions


def batch(shape, seed):
    size, extends, outputs, width, padding = shape
    gen = torch.Generator().manual_seed(seed)
    pool = size + 4
    slots = torch.arange(size, dtype=torch.int64)
    if size > 1:
        slots[-1] = padding
    valid = torch.randint(0, 100, (pool,), generator=gen, dtype=torch.int32)
    inputs = torch.full((size,), 8, dtype=torch.int32)
    prompts = valid[:extends] + inputs[:extends]
    limits = valid + 1 + torch.randint(1, 17, (pool,), generator=gen, dtype=torch.int32)
    stops = torch.randint(-1, 20, (pool, 16), generator=gen, dtype=torch.int32)
    stopped = torch.rand(pool, generator=gen) < 0.2
    tokens = torch.randint(
        0, 20, (outputs + (size - extends) * width,), generator=gen, dtype=torch.int32
    )
    accepts = torch.randint(0, width + 1, (size,), generator=gen, dtype=torch.int32)
    accepts[:extends] = 1
    return [slots, tokens, accepts, inputs, prompts, valid, stops, limits, stopped]


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 8, -1),
        (1, 1, 0, 8, -1),
        (2, 2, 1, 8, -1),
        (8, 8, 8, 8, 1 << 40),
        (8, 0, 0, 8, -1),
        (32, 0, 0, 8, -1),
        (65, 5, 2, 3, -1),
        (200, 70, 70, 5, -1),
    ],
)
def test_graph_stop_conditions_match_cpu(shape):
    if not torch.npu.is_available():
        pytest.skip("requires NPU")
    _, extends, outputs, width, padding = shape
    kwargs = dict(
        num_extends=extends,
        num_prefill_outputs=outputs,
        decode_width=width,
        padding_index=padding,
    )
    host = batch(shape, 0)
    device = [tensor.npu() for tensor in host]
    for _ in range(2):
        apply_stop_conditions(*device, **kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        apply_stop_conditions(*device, **kwargs)
    for seed in range(12):
        host = batch(shape, seed)
        for dst, src in zip(device, host):
            dst.copy_(src)
        apply_stop_conditions(*host, **kwargs)
        graph.replay()
        torch.npu.synchronize()
        for index in (2, 8):
            torch.testing.assert_close(device[index].cpu(), host[index], rtol=0, atol=0)
