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

"""Exact stable KVP compaction and mutable NPU graph replay."""

import pytest
import torch

pytest.importorskip("torch_npu")
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import compact_kvp_pages

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an Ascend NPU"
)


def _inputs(columns, subpages, phase, *, rows):
    gen = torch.Generator().manual_seed(902 + phase)
    table = torch.randint(
        -3 * subpages,
        257 * subpages,
        (rows, columns),
        generator=gen,
        dtype=torch.int32,
    )
    if columns > 3:
        table[:, :3] = torch.tensor([0, subpages, 256 * subpages])
    if phase == 1:
        table.zero_()
    elif phase == 2:
        table.fill_(subpages)
    endpoints = torch.tensor(
        [0, 1, 127, 128, 129, columns * 128 - 1, columns * 128, 262144],
        dtype=torch.int32,
    )
    lengths = endpoints[torch.arange(rows) % 8].clamp(0, columns * 128)
    return table, lengths


def _reference(table, lengths, *, subpages, rank, reserve):
    result = torch.zeros_like(table)
    tokens = torch.zeros_like(lengths)
    prefix = torch.zeros((table.shape[0], table.shape[1] + 1), dtype=torch.int32)
    for row, pages in enumerate(table.tolist()):
        count = local_tokens = 0
        length = int(lengths[row])
        for col, page in enumerate(pages):
            block = page // subpages
            owned = (
                0 < block < 256
                and (block - 1) % 8 == rank
                and (reserve or col * 128 < length)
            )
            if owned:
                result[row, count] = ((block - 1) // 8 + 1) * subpages + page % subpages
                count += 1
                local_tokens += min(128, max(length - col * 128, 0))
            prefix[row, col + 1] = count
        tokens[row] = local_tokens
    return result, tokens, prefix if reserve else None


@pytest.mark.parametrize("columns", [1, 31, 32, 257, 2057])
@pytest.mark.parametrize("subpages", [1, 8])
@pytest.mark.parametrize("reserve", [False, True])
@pytest.mark.parametrize("rank", range(8))
def test_compact_pages_exact(columns, subpages, reserve, rank):
    table, lengths = _inputs(columns, subpages, 0, rows=8)
    expected = _reference(table, lengths, subpages=subpages, rank=rank, reserve=reserve)
    table_npu, lengths_npu = table.npu(), lengths.npu()
    output = torch.full_like(table_npu, -99)
    local_lengths = torch.full_like(lengths_npu, -99)
    prefix = (
        torch.empty((8, columns + 1), dtype=torch.int32, device="npu")
        if reserve
        else None
    )
    compact_kvp_pages(
        table_npu,
        lengths_npu,
        page_size=128,
        block_granularity=subpages * 128,
        virtual_block_count=256,
        degree=8,
        rank=rank,
        out=output,
        local_lengths=local_lengths,
        page_prefix=prefix,
    )
    for actual, wanted in zip((output, local_lengths, prefix), expected):
        if wanted is not None:
            torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)


@pytest.mark.parametrize("reserve", [False, True])
def test_compact_pages_graph_replay(reserve):
    table, lengths = _inputs(2057, 8, 0, rows=64)
    table_npu, lengths_npu = table.npu(), lengths.npu()
    output = torch.empty_like(table_npu)
    local_lengths = torch.empty_like(lengths_npu)
    prefix = (
        torch.empty((64, 2058), dtype=torch.int32, device="npu") if reserve else None
    )

    def run():
        compact_kvp_pages(
            table_npu,
            lengths_npu,
            page_size=128,
            block_granularity=1024,
            virtual_block_count=256,
            degree=8,
            rank=0,
            out=output,
            local_lengths=local_lengths,
            page_prefix=prefix,
        )

    pointers = output.data_ptr(), local_lengths.data_ptr()
    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for phase in [1, 2, 3, 0]:
        table, lengths = _inputs(2057, 8, phase, rows=64)
        table_npu.copy_(table)
        lengths_npu.copy_(lengths)
        graph.replay()
        torch.npu.synchronize()
        expected = _reference(table, lengths, subpages=8, rank=0, reserve=reserve)
        for actual, wanted in zip((output, local_lengths, prefix), expected):
            if wanted is not None:
                torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
        assert pointers == (output.data_ptr(), local_lengths.data_ptr())
