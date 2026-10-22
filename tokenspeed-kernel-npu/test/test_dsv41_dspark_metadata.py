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

"""Exact DSpark integer metadata, including strided inputs and mutable replay."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel.ops.attention import dsv41


@pytest.mark.parametrize(
    "n,window,block,hc",
    [
        (0, 128, 5, 4),
        (1, 128, 5, 4),
        (8, 128, 5, 4),
        (32, 128, 5, 4),
        (7, 5, 3, 2),
        (33, 128, 5, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_block_reference_and_replay(n, window, block, hc, dtype):
    torch.npu.set_device(0)
    torch.manual_seed(37)
    bonus = torch.randint(0, 100, (n, 3), device="npu", dtype=dtype)[:, 0]
    start = torch.arange(n * 2, device="npu", dtype=dtype)[::2]
    history = torch.randint(-10, 900, (n, window + 3), device="npu", dtype=dtype)[
        :, :window
    ]
    if dtype == torch.int64 and n:
        history[0, 0] = 2**40 + 65
        history[-1, -1] = -(2**40)
        start[0] = 2**40

    def run():
        return dsv41.dspark_block(bonus, start, history, 97, 64, hc, block)

    eager = run()
    expected = dsv41.dspark_block(
        bonus.cpu(), start.cpu(), history.cpu(), 97, 64, hc, block
    )
    for a, b in zip(eager, expected):
        torch.testing.assert_close(a.cpu(), b, atol=0, rtol=0)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        captured = run()
    for _ in range(3):
        bonus.add_(1)
        start.add_(11)
        history.sub_(5)
        graph.replay()
        torch.npu.synchronize()
        expected = dsv41.dspark_block(
            bonus.cpu(), start.cpu(), history.cpu(), 97, 64, hc, block
        )
        for a, b in zip(captured, expected):
            torch.testing.assert_close(a.cpu(), b, atol=0, rtol=0)


@pytest.mark.parametrize(
    "bs,extends,outputs",
    [(0, 0, 0), (1, 0, 0), (8, 0, 0), (8, 3, 1), (8, 8, 3), (32, 2, 2), (33, 0, 0)],
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_anchors_preserve_incomplete_prefill_and_graph(bs, extends, outputs, dtype):
    torch.npu.set_device(0)
    width, spec = 6, 6
    tokens = (
        torch.arange(outputs + (bs - extends) * width, device="npu", dtype=dtype) + 100
    )
    accept = torch.arange(bs, device="npu", dtype=dtype) - 3
    if bs:
        accept[-1] = 2**40 if dtype == torch.int64 else 2**30
    positions = (
        torch.arange((bs - extends) * width, device="npu", dtype=torch.int64) + 2**40
    )
    storage = torch.full((bs, spec + 2), -999, device="npu", dtype=torch.int32)
    nxt = storage[:, :spec]
    starts = torch.empty(bs - extends, device="npu", dtype=torch.int64)

    def run():
        dsv41.dspark_anchors(
            tokens,
            accept,
            positions,
            extends,
            width,
            nxt,
            starts,
            num_prefill_outputs=outputs,
        )

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        run()
    for _ in range(3):
        storage.fill_(-999)
        tokens.add_(7)
        positions.add_(3)
        accept.sub_(1)
        graph.replay()
        torch.npu.synchronize()
        reference = torch.full((bs, spec + 2), -999, dtype=torch.int32)
        refstarts = torch.empty(bs - extends, dtype=torch.int64)
        dsv41.dspark_anchors(
            tokens.cpu(),
            accept.cpu(),
            positions.cpu(),
            extends,
            width,
            reference[:, :spec],
            refstarts,
            num_prefill_outputs=outputs,
        )
        torch.testing.assert_close(storage.cpu(), reference, atol=0, rtol=0)
        torch.testing.assert_close(starts.cpu(), refstarts, atol=0, rtol=0)
