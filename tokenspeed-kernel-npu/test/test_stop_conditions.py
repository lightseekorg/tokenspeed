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

"""Ascend stop conditions preserve the CPU oracle in eager and Graph execution."""

import pytest
import torch
from tokenspeed_kernel.ops.metadata import apply_stop_conditions
from tokenspeed_kernel.ops.metadata.stop_conditions import _apply_stop_conditions_torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend NPU required"
)


@pytest.mark.parametrize(
    "batch,extends,outputs,width",
    [
        (9, 0, 0, 8),
        (12, 5, 3, 8),
        (6, 6, 0, 1),
        (6, 6, 6, 1),
        (1, 0, 0, 1),
        (200, 70, 70, 5),
    ],
)
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("stop_count", [0, 16])
def test_stop_conditions_match_cpu(batch, extends, outputs, width, capture, stop_count):
    gen = torch.Generator().manual_seed(batch + width)
    pool = batch + 4
    slots = torch.randperm(pool - 1, generator=gen)[:batch].to(torch.int64)
    if batch > 1:
        slots[-2:] = pool - 1  # repeated graph padding
    valid = torch.randint(0, 50, (pool,), generator=gen, dtype=torch.int32)
    limits = valid + torch.randint(
        0, width + 3, (pool,), generator=gen, dtype=torch.int32
    )
    stop_ids = torch.randint(
        -2, 12, (pool, stop_count), generator=gen, dtype=torch.int32
    )
    stopped = torch.rand(pool, generator=gen) < 0.2
    tokens = torch.randint(
        0, 12, (outputs + (batch - extends) * width,), generator=gen, dtype=torch.int32
    )
    accepts = torch.randint(0, width + 1, (batch,), generator=gen, dtype=torch.int32)
    accepts[:extends] = 1
    inputs = torch.full((batch,), width, dtype=torch.int32)
    prompts = (valid[slots[:extends]] + inputs[:extends]).clone()
    prompts[::2] += 3
    initial = [
        slots,
        tokens,
        accepts,
        inputs,
        prompts,
        valid,
        stop_ids,
        limits,
        stopped,
    ]
    device = [tensor.to("npu") for tensor in initial]
    kwargs = dict(
        num_extends=extends,
        num_prefill_outputs=outputs,
        decode_width=width,
        padding_index=pool - 1,
    )
    apply_stop_conditions(*device, **kwargs)  # compile before capture
    if capture:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            apply_stop_conditions(*device, **kwargs)
        # Replay must read updated buffers, including stop IDs and accepted widths.
        initial[1] = (tokens + 3) % 12
        initial[6] = torch.where(stop_ids >= 0, (stop_ids + 2) % 12, stop_ids)
        for dst, src in zip(device, initial, strict=True):
            dst.copy_(src)
        graph.replay()
    expected = [tensor.clone() for tensor in initial]
    _apply_stop_conditions_torch(*expected, **kwargs)
    torch.npu.synchronize()
    torch.testing.assert_close(device[2].cpu(), expected[2], rtol=0, atol=0)
    torch.testing.assert_close(device[8].cpu(), expected[8], rtol=0, atol=0)
