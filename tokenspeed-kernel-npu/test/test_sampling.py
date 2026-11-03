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

"""Operator-owned scratch and min-p filtering survive changed-input NPU Graph replay."""

import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires Ascend NPU"
)


@pytest.mark.parametrize("rows,vocab", [(1, 4), (3, 1025), (32, 163840)])
def test_sampling_dispatch_matches_cpu_and_replays(rows, vocab):
    from tokenspeed_kernel_npu.ops.sampling import (
        SamplingWorkspace,
        min_p_renorm_prob,
        sample_from_probs,
    )

    workspace = SamplingWorkspace.create(rows, vocab, "npu")
    probs = torch.empty((rows, vocab), dtype=torch.float32, device="npu")
    coins = torch.empty(rows, dtype=torch.float32, device="npu")
    min_ps = torch.empty_like(coins)
    out = torch.empty(rows, dtype=torch.int32, device="npu")

    def run():
        return sample_from_probs(
            min_p_renorm_prob(probs, min_ps), coins, workspace, out=out
        )

    def update(variant):
        values = (torch.arange(vocab).remainder(31) + 1).double()
        if variant:
            values = values.flip(0)
        host = values.expand(rows, -1).clone()
        host_min = torch.linspace(0.0, 0.8, rows)
        host_coins = torch.linspace(0.123456, 0.891234, rows)
        kept = host * (host >= host.amax(1, keepdim=True) * host_min[:, None])
        expected = (
            torch.searchsorted(
                kept.cumsum(1),
                host_coins.double()[:, None] * kept.sum(1, keepdim=True),
                right=True,
            )
            .squeeze(1)
            .int()
        )
        probs.copy_(host.float())
        min_ps.copy_(host_min)
        coins.copy_(host_coins)
        return expected

    expected = update(0)
    result = run()
    assert result.data_ptr() == out.data_ptr()
    assert torch.equal(result.cpu(), expected)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    for variant in (1, 0):
        expected = update(variant)
        graph.replay()
        assert captured.data_ptr() == out.data_ptr()
        assert torch.equal(captured.cpu(), expected)


def test_sampling_workspace_rejects_invalid_geometry():
    from tokenspeed_kernel_npu.ops.sampling import SamplingWorkspace

    for rows, vocab in ((0, 4), (1, 0)):
        with pytest.raises(ValueError, match="must be positive"):
            SamplingWorkspace.create(rows, vocab, "meta")
