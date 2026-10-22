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

"""Token-local MLA preserves request boundaries and bounded page reads."""

from dataclasses import dataclass

import pytest
import torch

from tokenspeed.runtime.layers.attention.backends.paged.ascend_dsa import (
    _AscendDSAPrefillPlan,
)
from tokenspeed.runtime.layers.attention.backends.paged.ascend_mla import (
    AscendMLABackend,
    mla_prefill_spans,
)
from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement


@pytest.mark.parametrize("degree", [1, 2, 4, 8, 16])
@pytest.mark.parametrize(
    "extends,prefixes",
    [
        ([16384], [0]),
        ([16384], [114688]),
        ([3, 0, 9], [127, 55, 1025]),
        ([0, 1, 0], [0, 13, 900]),
        ([0, 0], [12, 0]),
    ],
)
def test_local_query_spans_preserve_causal_positions(degree, extends, prefixes):
    expected = [
        (req, prefix + pos)
        for req, (extent, prefix) in enumerate(zip(extends, prefixes, strict=True))
        for pos in range(extent)
    ]
    total = sum(extends)
    counts = [total // degree + (rank < total % degree) for rank in range(degree)]
    first = 0
    for rows in counts:
        spans = mla_prefill_spans(extends, prefixes, first=first, rows=rows)
        actual = []
        active_requests = []
        offset = 0
        for req, extent in enumerate(extends):
            if max(first, offset) < min(first + rows, offset + extent):
                active_requests.append(req)
            offset += extent
        for req, (begin, end, past) in zip(active_requests, spans, strict=True):
            actual.extend((req, past + pos) for pos in range(end - begin))
        assert actual == expected[first : first + rows]
        assert sum(end - begin for begin, end, _ in spans) == rows
        first += rows


@pytest.mark.parametrize("first,rows", [(-1, 1), (0, -1), (3, 2)])
def test_invalid_query_partition_fails(first, rows):
    with pytest.raises(ValueError):
        mla_prefill_spans([4], [0], first=first, rows=rows)


@dataclass
class ReceivePlane:
    kv: torch.Tensor


@pytest.mark.parametrize("begin,end", [(0, 1), (5, 129), (127, 257), (0, 384)])
def test_bounded_reads_follow_the_mapped_table_without_changing_storage(begin, end):
    raw = torch.arange(7 * 128 * 576, dtype=torch.int32).reshape(-1, 576)
    table = torch.tensor([[4, 1, 6], [6, 4, 1]])
    backend = object.__new__(AscendMLABackend)
    backend.kernel_page_size = 128
    backend._prefill_pages = table
    backend._prefill_runtime = ReceivePlane(raw)
    before = raw.clone()
    for sequence in range(2):
        expected = torch.cat([raw[p * 128 : (p + 1) * 128] for p in table[sequence]])
        actual = backend.read_prefill_rows(sequence, begin, end)
        torch.testing.assert_close(actual, expected[begin:end], rtol=0, atol=0)
    torch.testing.assert_close(raw, before, rtol=0, atol=0)


@dataclass
class RequestMetadata:
    extend_seq_lens_cpu: torch.Tensor
    extend_prefix_lens_cpu: torch.Tensor
    page_table: torch.Tensor

    @property
    def extend_prefix_lens(self):
        return self.extend_prefix_lens_cpu

    @property
    def cum_extend_seq_lens(self):
        return torch.cat(
            (torch.zeros(1, dtype=torch.int64), self.extend_seq_lens_cpu.cumsum(0))
        )


@pytest.mark.parametrize("degree", [1, 2, 8, 16])
@pytest.mark.parametrize("page_size", [64, 128])
def test_head_shards_share_full_query_metadata_and_receive_page_addresses(
    degree, page_size
):
    metadata = RequestMetadata(
        torch.tensor([129, 0, 17]),
        torch.tensor([127, 500, 31]),
        torch.tensor([[2, 3, 4, 5], [0, 0, 0, 0], [6, 0, 0, 0]], dtype=torch.int32),
    )
    placement = CachePlacement(128, 9, tuple(range(degree)), 0)
    plans = [
        _AscendDSAPrefillPlan.build(
            metadata,
            placement,
            degree=degree,
            rank=rank,
            page_size=page_size,
            topk=1,
            local_cache_pages=32,
            query_partition=(0, 146),
        )
        for rank in range(degree)
    ]
    for plan in plans:
        assert (plan.row_start, plan.row_end) == (0, 146)
        assert sum(plan.row_counts) == 146
        torch.testing.assert_close(plan.query_cu, torch.tensor([0, 129, 146]))
        torch.testing.assert_close(plan.key_cu, torch.tensor([0, 256, 304]))
        torch.testing.assert_close(plan.page_table, plans[0].page_table)
        assert plan.query_seq.tolist() == [0] * 129 + [1] * 17


@dataclass
class CPUOwnerCommit:
    """Stand in only for the native NPU byte-copy kernel."""

    current: torch.Tensor
    cache: torch.Tensor

    def patch(self, layer_id, slots, direct, *, width, block_rows, virtual_blocks):
        live = (slots >= block_rows) & (slots < virtual_blocks * block_rows)
        self.cache.index_copy_(0, slots[live].long(), self.current[live])


@pytest.mark.parametrize("prefixes,extends", [([0], [129]), ([127, 255], [3, 2])])
def test_prefill_commits_router_slots_across_request_and_page_boundaries(
    prefixes, extends
):
    from tokenspeed.runtime.layers.attention.backends.paged.write_locations import (
        extend_write_locations,
    )

    page = 128
    table = torch.tensor([[2, 5, 3], [6, 4, 1]], dtype=torch.int32)[: len(extends)]
    total = sum(extends)
    slots = torch.empty((1, total), dtype=torch.int32)
    extend_write_locations(
        table[None],
        torch.tensor([page]),
        torch.tensor(prefixes),
        torch.tensor(extends),
        total,
        slots,
        page_sizes_cpu=(page,),
    )
    current = torch.arange(total * 576, dtype=torch.int32).reshape(total, 576)
    cache = torch.full((7 * page, 576), -1, dtype=torch.int32)
    backend = object.__new__(AscendMLABackend)
    backend._prefill_runtime = CPUOwnerCommit(current, cache)
    backend._placement = CachePlacement(page, 7, (0,), 0)
    backend._prefill_plan = _AscendDSAPrefillPlan.build(
        RequestMetadata(torch.tensor(extends), torch.tensor(prefixes), table),
        backend._placement,
        degree=1,
        rank=0,
        page_size=page,
        topk=1,
        local_cache_pages=7,
        query_partition=(0, total),
    )
    backend.consume_prefill_kv(7, slots[0])
    expected = torch.full_like(cache, -1)
    token = 0
    for req, (prefix, count) in enumerate(zip(prefixes, extends, strict=True)):
        for pos in range(prefix, prefix + count):
            expected[int(table[req, pos // page]) * page + pos % page] = current[token]
            token += 1
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
