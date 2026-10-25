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

from test.utils import assert_no_triton_compile

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsv41._triton.kvp import (
    _pack_decode_history,
    pack_decode_history,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def reference(cache, table, rows, virtual_count, degree, rank):
    result = torch.zeros((table.numel() + 1, 64, cache.shape[2]), dtype=torch.uint8)
    host = cache.cpu()
    for i, page in enumerate(table.cpu().tolist()):
        if 0 < page < virtual_count and (page - 1) % degree == rank and i * 64 < rows:
            result[i + 1] = host[(page - 1) // degree + 1]
    return result.to(cache.device)


@pytest.mark.parametrize("width", [68, 132])
@pytest.mark.parametrize("degree", [1, 2, 8])
@pytest.mark.parametrize("rows", [0, 1, 63, 64, 65, 1024])
def test_pack_history_bytes(width, degree, rows):
    cache = torch.randint(
        0, 256, (18, 64, width + 16), dtype=torch.uint8, device="cuda"
    )[:, :, :width]
    table = torch.tensor(
        [9, 0, 2, -1, 7, 17, 18, 1, 4, 13], dtype=torch.int32, device="cuda"
    ).repeat_interleave(2)[::2]
    visible = torch.tensor(rows, dtype=torch.int32, device="cuda")
    for rank in range(degree):
        actual = pack_decode_history(
            cache, table, visible, virtual_block_count=17, degree=degree, rank=rank
        )
        torch.testing.assert_close(
            actual, reference(cache, table, rows, 17, degree, rank), rtol=0, atol=0
        )


def test_pack_history_graph_updates():
    cache = torch.randint(0, 256, (4, 64, 148), dtype=torch.uint8, device="cuda")[
        :, :, :132
    ]
    table = torch.tensor([1, 9, 0, -1], dtype=torch.int64, device="cuda")
    visible = torch.tensor(1, dtype=torch.int64, device="cuda")

    def run():
        return pack_decode_history(
            cache, table, visible, virtual_block_count=17, degree=8, rank=0
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for rows in (0, 1, 65, 256):
        visible.fill_(rows)
        cache.random_(0, 256)
        table[:2].copy_(table[:2].flip(0))
        graph.replay()
        torch.testing.assert_close(
            actual, reference(cache, table, rows, 17, 8, 0), rtol=0, atol=0
        )


def test_pack_history_capacity_does_not_recompile():
    cache = torch.zeros((18, 64, 132), dtype=torch.uint8, device="cuda")
    visible = torch.tensor(65, dtype=torch.int32, device="cuda")

    def run(capacity):
        table = torch.ones(capacity, dtype=torch.int32, device="cuda")
        return pack_decode_history(
            cache, table, visible, virtual_block_count=129, degree=8, rank=0
        )

    run(16)
    with assert_no_triton_compile(_pack_decode_history):
        for capacity in (0, 1, 17, 256):
            actual = run(capacity)
            assert actual.shape == (capacity + 1, 64, 132)
            assert torch.count_nonzero(actual) == 0
