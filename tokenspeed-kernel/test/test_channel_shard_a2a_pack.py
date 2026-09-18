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

"""Exact packing and graph-replay checks shared by GPU vendors."""

import pytest
import torch
from tokenspeed_kernel.ops.communication.triton import (
    triton_pack_channel_shards_for_a2a,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("peers,width", [(2, 34), (4, 12288)])
@pytest.mark.parametrize(
    "rows,padded,strided",
    [
        (1, 1, False),
        (0, 1, False),
        (1, 1, True),
        (3, 8, False),
        (8, 8, False),
        (3, 8, True),
        (0, 8, True),
    ],
)
def test_channel_shard_a2a_pack_and_replay(peers, width, rows, padded, strided):
    storage = torch.randn(
        rows, width * (2 if strided else 1), device="cuda", dtype=torch.bfloat16
    )
    inputs = storage[:, ::2] if strided else storage
    scratch = torch.empty(
        peers, padded, width // peers, device="cuda", dtype=inputs.dtype
    )

    def reference():
        expected = torch.zeros_like(scratch)
        expected[:, :rows].copy_(
            inputs.reshape(rows, peers, width // peers).permute(1, 0, 2)
        )
        return expected.flatten(0, 1)

    result = triton_pack_channel_shards_for_a2a(inputs, scratch)
    torch.testing.assert_close(result, reference(), rtol=0, atol=0)
    assert result.data_ptr() == (
        inputs.data_ptr() if rows == padded == 1 and not strided else scratch.data_ptr()
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = triton_pack_channel_shards_for_a2a(inputs, scratch)
    for scale in (1.0, -2.0, 0.0):
        inputs.mul_(scale)
        scratch.fill_(17)
        graph.replay()
        torch.testing.assert_close(captured, reference(), rtol=0, atol=0)
