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

"""The CUDA DCP candidate merge selects what the portable merge selects."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_merge_candidates
from tokenspeed_kernel.ops.attention.dsa._triton.candidate_exchange import (
    _select_through_threshold,
)
from tokenspeed_kernel.ops.attention.dsa.cuda import has_ragged_decode_topk
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not has_ragged_decode_topk(),
    reason="requires the CUDA persistent top-k kernel",
)


def _records(degree: int, rows: int, topk: int, seed: int) -> torch.Tensor:
    """Every rank's local candidates as the all-gather lays them out, with
    the invalid, mandatory and short rows a DCP step produces."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randperm(1 << 24, device="cuda", generator=generator)
    ids = ids[: degree * rows * topk].int().view(degree, rows, topk)
    scores = torch.rand((degree, rows, topk), device="cuda", generator=generator)
    # Row 0: invalid offsets and scores, mandatory windows, a signed zero.
    ids[:, 0, :3] = -1
    scores[:, 0, 3:8] = torch.tensor(
        [torch.nan, -torch.inf, torch.inf, torch.inf, -0.0], device="cuda"
    )
    if rows > 2:
        # A short history: fewer valid candidates in all than topk.
        scores[:, 1, topk // 4 :] = -torch.inf
        # A request whose history no rank owns.
        ids[:, 2] = -1
    return torch.stack((ids, scores.view(torch.int32)), dim=-1)


@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 7, 64])
@pytest.mark.parametrize("topk", [512, 2048])
def test_cuda_merge_selects_the_portable_merges_candidates(degree, rows, topk):
    records = _records(degree, rows, topk, seed=degree * 100 + rows)

    positions, counts = dsa_merge_candidates(records, topk=topk, solution="cuda")
    expected, expected_counts = dsa_merge_candidates(
        records, topk=topk, solution="triton"
    )

    torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)
    for row in range(rows):
        count = int(counts[row])
        assert sorted(positions[row, :count].tolist()) == sorted(
            expected[row, :count].tolist()
        )
        assert (positions[row, count:] == -1).all()
    if rows > 2:
        assert int(counts[2]) == 0


def _selected_scores(records: torch.Tensor, positions, counts, row: int) -> list:
    ids, bits = records[:, row].reshape(-1, 2).unbind(-1)
    score = dict(zip(ids.tolist(), bits.view(torch.float32).tolist()))
    return sorted(score[p] for p in positions[row, : int(counts[row])].tolist())


@pytest.mark.parametrize("degree", [2, 8])
@pytest.mark.parametrize("topk", [512, 2048])
def test_cuda_merge_selects_the_same_tied_candidates_every_time(degree, topk):
    # Ties at the cutoff resolve by atomic order; every KVP rank must keep the same set.
    records = _records(degree, 16, topk, seed=degree + topk)
    generator = torch.Generator(device="cuda").manual_seed(topk)
    tied = torch.randint(
        0, 8, records.shape[:3], device="cuda", generator=generator
    ).float()
    records[..., 1] = tied.view(torch.int32)

    positions, counts = dsa_merge_candidates(records, topk=topk, solution="cuda")

    for _ in range(10):
        again, again_counts = dsa_merge_candidates(records, topk=topk, solution="cuda")
        torch.testing.assert_close(again, positions, atol=0, rtol=0)
        torch.testing.assert_close(again_counts, counts, atol=0, rtol=0)
    one, one_count = dsa_merge_candidates(
        records[:, 3:4].contiguous(), topk=topk, solution="cuda"
    )
    torch.testing.assert_close(one[0], positions[3], atol=0, rtol=0)
    torch.testing.assert_close(one_count[0], counts[3], atol=0, rtol=0)
    expected, expected_counts = dsa_merge_candidates(
        records, topk=topk, solution="triton"
    )
    torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)
    for row in range(16):
        assert _selected_scores(records, positions, counts, row) == _selected_scores(
            records, expected, expected_counts, row
        )


def test_cuda_merge_launches_one_specialization_across_batch_rows():
    for rows in (1, 16):
        dsa_merge_candidates(
            _records(2, rows, 2048, seed=rows), topk=2048, solution="cuda"
        )
    with assert_no_triton_compile(_select_through_threshold):
        for rows in (3, 8, 37, 64):
            dsa_merge_candidates(
                _records(2, rows, 2048, seed=rows), topk=2048, solution="cuda"
            )
