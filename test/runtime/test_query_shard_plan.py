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

"""Row layout behavior, independent of distributed execution."""

import sys

import pytest
import torch
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.execution.query_shard import QueryShardPlan

register_cuda_ci(est_time=5, suite="runtime-1gpu")


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("lengths", [[], [1], [2, 1, 3], [17, 9, 31], [32, 48]])
def test_row_payloads_survive_sharding_and_selected_row_gathers(size, lengths):
    total = sum(lengths)
    plans = [
        QueryShardPlan.from_forward(
            total_tokens=total,
            input_lengths=lengths,
            size=size,
            rank=rank,
            device="cpu",
        )
        for rank in range(size)
    ]
    # Unique payloads expose a wrong row even when request lengths coincide.
    span = torch.arange(total * 3, dtype=torch.int64).reshape(total, 3)
    local = [span[p.local_row_ids] for p in plans]
    gathered = torch.cat(local)
    assert sorted(torch.cat([p.local_row_ids for p in plans]).tolist()) == list(
        range(total)
    )
    # Select a noncontiguous, reversed subset, as consumers need not request
    # rows in rank order or span order.
    selected = torch.arange(total, dtype=torch.int64)[::2].flip(0)
    splits = [p.split_rows(selected) for p in plans]
    selected_gather = torch.cat([x[s.local_ids] for x, s in zip(local, splits)])
    sampled = torch.cumsum(torch.tensor(lengths, dtype=torch.int64), 0) - 1
    sampled_gather = torch.cat([x[p.local_sampled_index] for x, p in zip(local, plans)])
    for p, split in zip(plans, splits):
        assert torch.equal(p.to_span_order(gathered), span)
        assert torch.equal(p.to_gather_order(span), gathered)
        assert torch.equal(selected_gather[split.to_rows_order], span[selected])
        assert torch.equal(sampled_gather[p.sampled_to_request], span[sampled])
        assert split.rows_per_rank == splits[0].rows_per_rank


@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_successive_row_selections_keep_their_gather_order(size, device):
    lengths = [5, 1, 9]
    plans = [
        QueryShardPlan.from_forward(
            total_tokens=sum(lengths),
            input_lengths=lengths,
            size=size,
            rank=rank,
            device=device,
        )
        for rank in range(size)
    ]
    span = torch.arange(sum(lengths) * 3, device=device).reshape(-1, 3)
    local = [span[p.local_row_ids] for p in plans]
    selections = [
        torch.tensor(rows, dtype=torch.int64)
        for rows in ([14, 0, 6, 2], [], [1, 8, 13])
    ]
    # Keep the earlier results alive while the same plans split other rows.
    splits = [[p.split_rows(rows) for p in plans] for rows in selections]
    for rows, selection_splits in zip(selections, splits):
        gathered = torch.cat(
            [shard[s.local_ids.to(device)] for shard, s in zip(local, selection_splits)]
        )
        for split in selection_splits:
            assert torch.equal(
                gathered[split.to_rows_order.to(device)], span[rows.to(device)]
            )


@pytest.mark.parametrize("size", [2, 3, 4, 8])
def test_equal_segments_balance_causal_visible_key_work_per_request(size):
    # A row at position i sees prefix + i + 1 keys. Each rank's early/late
    # pair has equal total work when the request divides into equal segments.
    lengths = [4 * size, 6 * size]
    prefix_lengths = [0, 101]
    costs = torch.cat(
        [torch.arange(p + 1, p + n + 1) for p, n in zip(prefix_lengths, lengths)]
    )
    per_rank = []
    for rank in range(size):
        plan = QueryShardPlan.from_forward(
            total_tokens=sum(lengths),
            input_lengths=lengths,
            size=size,
            rank=rank,
            device="cpu",
        )
        per_rank.append(costs[plan.local_row_ids].sum().item())
    assert len(set(per_rank)) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
