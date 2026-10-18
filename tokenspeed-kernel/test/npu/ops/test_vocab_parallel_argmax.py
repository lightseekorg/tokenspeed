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

"""Portable vocabulary-parallel key packing on Ascend."""

import pytest
import torch

pytest.importorskip("torch_npu")
from tokenspeed_kernel.ops.sampling.triton.vocab_parallel_argmax import (
    argmax_key_index,
    vocab_parallel_argmax_keys,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("capture", [False, True])
def test_sharded_argmax_matches_cpu_with_ties_and_signed_zero(dtype, capture):
    host = torch.randn(8, 8192, generator=torch.Generator().manual_seed(51)).to(dtype)
    host[0, 4097] = host[0, 9] = 100
    host[1] = -1
    host[1, 8], host[1, 7000] = -0.0, 0.0
    host[2] = float("-inf")
    host[3, 8191] = float("inf")
    data = host.npu()
    keys = [torch.empty(8, device="npu", dtype=torch.int64) for _ in range(2)]

    def run():
        for shard in range(2):
            vocab_parallel_argmax_keys(
                data[:, shard * 4096 : (shard + 1) * 4096],
                vocab_offset=shard * 4096,
                out=keys[shard],
            )
        reduced = torch.stack(keys).amax(0)
        return argmax_key_index(reduced, out=torch.empty_like(reduced))

    actual = run()
    torch.testing.assert_close(actual.cpu(), host.argmax(-1), rtol=0, atol=0)
    if capture:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = run()
        for turn in range(3):
            replacement = host.roll(turn + 1, dims=1)
            data.copy_(replacement)
            graph.replay()
            torch.testing.assert_close(
                actual.cpu(), replacement.argmax(-1), rtol=0, atol=0
            )
        graph.reset()
