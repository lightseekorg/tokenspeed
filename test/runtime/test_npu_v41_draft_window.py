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

"""Draft windows use strided arena row copies without touching adjacent fields."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed.runtime.models.deepseek_v41_dspark import (
    _quantized_kv,
    _WindowAttention,
    _WindowSelection,
    _write_window_rows,
)


@pytest.mark.parametrize("pages", [7, 1294])
@pytest.mark.parametrize("batch", [1, 8])
def test_strided_window_mutable_graph(pages, batch):
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(987)
    # Extra fields before/after every page reproduce the arena's noncontiguity.
    backing = torch.randn(pages + 2, 192, 512, device="npu", dtype=torch.bfloat16)
    window = backing[1 : pages + 1, 64:128]
    expected = backing.cpu().clone()
    expected_window = expected[1 : pages + 1, 64:128]
    slots = torch.arange(batch * 6, device="npu", dtype=torch.int64) + 64
    slots[-1] = -1
    values = torch.randn(batch * 6, 512, device="npu", dtype=torch.bfloat16)
    history = torch.arange(batch * 128, device="npu", dtype=torch.int64).reshape(
        batch, 128
    ) % (pages * 64)
    history[:, :3] = -1
    safe = history.clamp_min(0)
    selection = _WindowSelection(
        safe // 64, safe % 64, torch.empty(0, device="npu", dtype=torch.int32)
    )
    positions = torch.arange(batch * 5, device="npu", dtype=torch.int64)
    backend = _WindowAttention(
        positions, window, history, 5, selection, torch.zeros_like(positions)
    )
    swa = torch.randn(batch * 5, 512, device="npu", dtype=torch.bfloat16)

    def run():
        _write_window_rows(window, slots, values)
        return backend._workspace(swa, positions, None, None, None)

    run()
    backing.copy_(expected.npu())
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph, stream=torch.npu.Stream(device=window.device), auto_dispatch_capture=True
    ):
        actual = run()
    for step in range(4):
        slots.add_(1)
        slots[-1] = -1
        values.add_(0.25)
        swa.mul_(0.5)
        history.copy_((history + 13).remainder(pages * 64))
        history[:, :3] = -1
        graph.replay()
        torch.npu.synchronize()
        host_slots = slots.cpu()
        valid = host_slots >= 0
        vslots = host_slots[valid]
        expected_window[vslots // 64, vslots % 64] = values.cpu()[valid]
        safe = history.cpu().clamp_min(0)
        reference_history = expected_window[safe // 64, safe % 64]
        reference_rows = _quantized_kv(swa).cpu().reshape(batch, 5, 512)
        torch.testing.assert_close(
            actual.cpu(),
            torch.cat((reference_history, reference_rows), 1),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(backing.cpu(), expected, atol=0, rtol=0)
