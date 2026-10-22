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

"""The KPool prefill plan maps every token, pool and tail to the right rows.

Expected values come from a per-token walk of the KPool layout: a request
whose cached prefix is ``start`` writes completed pools ``start // kpool``
onward (the first one completed from the tail buffer) and leaves the
remainder in its tail.
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.attention.kpool import build_kpool_prefill_plan

KPOOL = 4
ROWS_PER_PAGE = 8
TAIL_ROWS = 6


def _expected(starts, lengths, table, slots, capacity):
    tokens = {
        "positions": [],
        "req_ids": [],
        "causal": [],
        "row_start": [],
        "row_end": [],
    }
    workspace = []
    pools = {
        k: [] for k in ("slot", "from_tail", "tail_slot", "tail_rows", "chunk_rows")
    }
    tails = {k: [] for k in ("src", "slot", "dst", "count")}
    offset = 0
    for req, (start, length) in enumerate(zip(starts, lengths)):
        row_start = len(workspace)
        for pool in range((start + length) // KPOOL):
            workspace.append(
                table[req][pool // ROWS_PER_PAGE] * ROWS_PER_PAGE + pool % ROWS_PER_PAGE
            )
        for i in range(length):
            causal = start + i + 1
            tokens["positions"].append(causal - 1)
            tokens["req_ids"].append(req)
            tokens["causal"].append(causal)
            tokens["row_start"].append(row_start)
            tokens["row_end"].append(row_start + causal // KPOOL)
        position = start
        while position // KPOOL < (start + length) // KPOOL:
            pool = position // KPOOL
            n_from_tail = position % KPOOL
            pools["slot"].append(
                table[req][pool // ROWS_PER_PAGE] * ROWS_PER_PAGE + pool % ROWS_PER_PAGE
            )
            pools["from_tail"].append([j < n_from_tail for j in range(KPOOL)])
            pools["tail_slot"].append([slots[req]])
            pools["tail_rows"].append(
                [(pool * KPOOL + j) % TAIL_ROWS for j in range(KPOOL)]
            )
            pools["chunk_rows"].extend(
                offset + (position - start) + max(j - n_from_tail, 0)
                for j in range(KPOOL)
            )
            position = (pool + 1) * KPOOL
        if position < start + length:
            tails["src"].append(offset + position - start)
            tails["slot"].append(slots[req])
            tails["dst"].append(position)
            tails["count"].append(start + length - position)
        offset += length
    padding = capacity - offset
    for values in tokens.values():
        values.extend([0] * padding)
    return tokens, workspace, pools, tails


@pytest.mark.parametrize(
    "starts,lengths,capacity",
    [
        ([0], [13], 16),  # fresh prompt with a partial final pool
        ([8192], [12], 16),  # chunk 2 of a long prompt: no completed pool
        ([3, 0, 10], [9, 0, 6], 24),  # unaligned prefixes, an empty extend
        ([5, 1], [3, 4], 7),  # pools completed from tails, no padding
    ],
)
def test_plan_maps_tokens_pools_and_tails(starts, lengths, capacity) -> None:
    max_pages = (
        max(s + n for s, n in zip(starts, lengths)) // KPOOL
    ) // ROWS_PER_PAGE + 1
    table = [
        [100 * (req + 1) + page for page in range(max_pages)]
        for req in range(len(starts))
    ]
    slots = [7, 3, 11][: len(starts)]
    plan = build_kpool_prefill_plan(
        prefix_lens_cpu=torch.tensor(starts),
        extend_lens_cpu=torch.tensor(lengths),
        index_block_table=torch.tensor(table, dtype=torch.int32),
        request_slots=torch.tensor(slots, dtype=torch.int32),
        kpool=KPOOL,
        index_rows_per_page=ROWS_PER_PAGE,
        tail_rows_per_request=TAIL_ROWS,
        token_capacity=capacity,
    )
    tokens, workspace, pools, tails = _expected(starts, lengths, table, slots, capacity)

    assert plan.num_prefill_tokens == sum(lengths)
    assert plan.positions.tolist() == tokens["positions"]
    assert plan.req_ids.tolist() == tokens["req_ids"]
    assert plan.causal_lens.tolist() == tokens["causal"]
    assert plan.row_starts.tolist() == tokens["row_start"]
    assert plan.row_ends.tolist() == tokens["row_end"]
    assert plan.pool_workspace_slots.tolist() == workspace
    assert plan.query_start_loc.tolist() == [
        0,
        *torch.tensor(lengths).cumsum(0).tolist(),
    ]
    assert plan.seq_lens.tolist() == [s + n for s, n in zip(starts, lengths)]
    assert plan.max_seq_len == max(s + n for s, n in zip(starts, lengths))
    assert plan.max_num_pools == max((s + n) // KPOOL for s, n in zip(starts, lengths))

    write = plan.write
    assert write.pool_write_slots.tolist() == pools["slot"]
    assert write.pool_from_tail.tolist() == pools["from_tail"]
    assert write.pool_tail_slots.tolist() == pools["tail_slot"]
    assert write.pool_tail_rows.tolist() == pools["tail_rows"]
    assert write.pool_chunk_rows.tolist() == pools["chunk_rows"]
    assert write.tail_chunk_src.tolist() == tails["src"]
    assert write.tail_slots.tolist() == tails["slot"]
    assert write.tail_dst_positions.tolist() == tails["dst"]
    assert write.tail_write_counts.tolist() == tails["count"]


def test_plan_rejects_capacity_below_the_tokens() -> None:
    with pytest.raises(ValueError, match="capacity is smaller"):
        build_kpool_prefill_plan(
            prefix_lens_cpu=torch.tensor([0]),
            extend_lens_cpu=torch.tensor([9]),
            index_block_table=torch.zeros((1, 2), dtype=torch.int32),
            request_slots=torch.zeros(1, dtype=torch.int32),
            kpool=KPOOL,
            index_rows_per_page=ROWS_PER_PAGE,
            tail_rows_per_request=TAIL_ROWS,
            token_capacity=8,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_plan_builds_on_device_without_synchronizing() -> None:
    """The plan is built at the first sparse layer behind queued GPU work; a
    synchronizing upload there would stall the forward thread."""
    table = torch.arange(4, dtype=torch.int32, device="cuda").view(1, 4)
    slots = torch.tensor([2], dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        plan = build_kpool_prefill_plan(
            prefix_lens_cpu=torch.tensor([6]),
            extend_lens_cpu=torch.tensor([21]),
            index_block_table=table,
            request_slots=slots,
            kpool=KPOOL,
            index_rows_per_page=ROWS_PER_PAGE,
            tail_rows_per_request=TAIL_ROWS,
            token_capacity=32,
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")
    tokens, workspace, pools, tails = _expected([6], [21], table.tolist(), [2], 32)
    assert plan.causal_lens.tolist() == tokens["causal"]
    assert plan.pool_workspace_slots.tolist() == workspace
    assert plan.write.pool_chunk_rows.tolist() == pools["chunk_rows"]
    assert plan.write.tail_dst_positions.tolist() == tails["dst"]
