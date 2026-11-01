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

"""Prefill page staging preserves virtual-slot identity and causal endpoints."""

from dataclasses import dataclass

import pytest
import torch

from tokenspeed.runtime.layers.attention.dcp.ascend import AscendPrefillPlan
from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement


@dataclass
class PrefillMetadata:
    extend_seq_lens_cpu: torch.Tensor
    extend_prefix_lens_cpu: torch.Tensor
    page_table: torch.Tensor

    @property
    def cum_extend_seq_lens(self):
        return torch.cat(
            (
                torch.zeros(1, dtype=torch.int32),
                self.extend_seq_lens_cpu.int().cumsum(0).int(),
            )
        )

    @property
    def extend_prefix_lens(self):
        return self.extend_prefix_lens_cpu


@pytest.mark.parametrize("degree", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("granularity", [128, 512])
@pytest.mark.parametrize(
    "extends,prefixes",
    [
        ([3, 2], [127, 255]),
        ([129, 0, 257], [0, 260, 127]),
        ([1], [0]),
        ([17, 23], [0, 0]),
        ([16384], [0]),
        ([129, 0, 257], [16384, 260, 127]),
    ],
)
def test_request_pages_and_query_causal_spans(degree, granularity, extends, prefixes):
    page = 128
    per_block = granularity // page
    lengths = [p + n for p, n in zip(prefixes, extends)]
    columns = max((n + page - 1) // page for n in lengths) + 2
    table = torch.full((len(extends), columns), 10000, dtype=torch.int32)
    # Requests share/reorder physical pages; invalid padding must be ignored.
    for request, length in enumerate(lengths):
        count = (length + page - 1) // page
        table[request, :count] = torch.arange(per_block, per_block + count).flip(0)
    metadata = PrefillMetadata(torch.tensor(extends), torch.tensor(prefixes), table)
    last_page = max(
        int(table[i, : (length + page - 1) // page].max())
        for i, length in enumerate(lengths)
        if length
    )
    virtual_blocks = last_page // per_block + 1
    local_capacity = ((virtual_blocks - 2) // degree + 2) * per_block
    plans = [
        AscendPrefillPlan.build(
            metadata,
            CachePlacement(granularity, virtual_blocks, tuple(range(degree)), rank),
            degree=degree,
            rank=rank,
            page_size=page,
            topk=2048,
            local_cache_pages=local_capacity,
        )
        for rank in range(degree)
    ]
    global_pages = {0}
    for request, length in enumerate(lengths):
        global_pages.update(table[request, : (length + page - 1) // page].tolist())
    caches = [{} for _ in range(degree)]
    for virtual in global_pages:
        block, subpage = divmod(virtual, per_block)
        owner = (block - 1) % degree if block else 0
        physical = ((block - 1) // degree + 1) * per_block + subpage if block else 0
        caches[owner][physical] = torch.arange(page, dtype=torch.int64) + virtual * page
    send_rows = -(-sum(extends) // (degree * page)) * page
    gathered = torch.zeros(
        (degree * local_capacity + degree * send_rows // page, page), dtype=torch.int64
    )
    for rank, cache in enumerate(caches):
        for physical, values in cache.items():
            gathered[rank * local_capacity + physical] = values
    current_values = []
    for request, (prefix, count) in enumerate(zip(prefixes, extends)):
        positions = torch.arange(prefix, prefix + count)
        current_values.extend(
            (
                table[request, positions // page].long() * page + positions % page
            ).tolist()
        )
    for owner, count in enumerate(plans[0].row_counts):
        start = sum(plans[0].row_counts[:owner])
        target = degree * local_capacity * page + owner * send_rows
        gathered.view(-1)[target : target + count] = torch.tensor(
            current_values[start : start + count]
        )
    assert sum(plans[0].row_counts) == sum(extends)
    assert max(plans[0].row_counts) - min(plans[0].row_counts) <= 1
    global_sequence = [
        request for request, count in enumerate(extends) for _ in range(count)
    ]
    global_ends = [
        prefix + i + 1 for prefix, count in zip(prefixes, extends) for i in range(count)
    ]
    for plan in plans:
        active_requests = [
            request
            for request, count in enumerate(extends)
            if sum(extends[:request]) < plan.row_end
            and sum(extends[: request + 1]) > plan.row_start
            and count
        ]
        expected_sequence = [
            active_requests.index(request)
            for request in global_sequence[plan.row_start : plan.row_end]
        ]
        assert plan.query_seq.tolist() == expected_sequence
        assert plan.scan_count.tolist() == [
            min(end, 2048) for end in global_ends[plan.row_start : plan.row_end]
        ]
        assert plan.query_cu[-1] == plan.row_end - plan.row_start
        for compact, request in enumerate(active_requests):
            length = lengths[request]
            positions = torch.arange(length)
            actual = gathered[
                plan.page_table[compact, positions // page].long(), positions % page
            ]
            expected = (
                table[request, positions // page].long() * page + positions % page
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        q_lens = torch.diff(plan.query_cu).tolist()
        k_lens = torch.diff(plan.key_cu).tolist()
        active_requests = [
            request
            for request, count in enumerate(extends)
            if sum(extends[:request]) < plan.row_end
            and sum(extends[: request + 1]) > plan.row_start
            and count
        ]
        assert plan.index_table.data_ptr() == plan.page_table.data_ptr()
        for request, qlen, klen in zip(active_requests, q_lens, k_lens):
            offset = sum(extends[:request])
            start = min(max(plan.row_start - offset, 0), extends[request])
            end = min(max(plan.row_end - offset, 0), extends[request])
            assert qlen == end - start
            assert klen == prefixes[request] + end


def test_prefill_rejects_mismatched_request_lengths():
    metadata = PrefillMetadata(
        torch.tensor([2, 3]), torch.tensor([0]), torch.ones(2, 1, dtype=torch.int32)
    )
    with pytest.raises(ValueError, match="request page-table"):
        AscendPrefillPlan.build(
            metadata,
            None,
            degree=1,
            rank=0,
            page_size=128,
            topk=2048,
            local_cache_pages=8,
        )
