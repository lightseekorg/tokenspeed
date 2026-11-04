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

"""GFX950 DCP selection: the row-bounded logical top-k over owned rows of a
table-wide buffer, and the Gluon candidate merge against the portable one."""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD GFX950 is required for the Gluon DCP top-k", allow_module_level=True
    )

from tokenspeed_kernel.ops.attention.dsa import dsa_merge_candidates  # isort: skip
from tokenspeed_kernel.ops.attention.dsa._triton.candidate_exchange import (  # isort: skip
    _extract_rank_major_scores,
    _select_source_through_threshold,
    _select_through_threshold,
    rank_major_scores,
    rank_major_scores_with_bounds,
    select_through_threshold,
    select_through_threshold_by_source,
)
from tokenspeed_kernel.selection import select_kernel  # isort: skip
from tokenspeed_kernel.signature import (  # isort: skip
    dense_tensor_format,
    format_signature,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (  # isort: skip
    gluon_dsa_logical_topk_gfx950,
    gluon_dsa_row_bounded_logical_topk_gfx950,
)


@pytest.mark.parametrize("rows", [1, 48, 64])
@pytest.mark.parametrize("topk", [512, 2048])
def test_row_bounded_topk_selects_each_rows_largest_owned_scores(rows, topk):
    # Owned rows a fraction of a max-model-len-wide buffer whose unwritten
    # tail holds larger values than any owned score.
    width = 1 << 20
    generator = torch.Generator(device="cuda").manual_seed(rows + topk)
    logits = torch.rand((rows, width), device="cuda", generator=generator)
    lengths = torch.randint(
        topk // 2, 4 * topk, (rows,), device="cuda", generator=generator
    ).int()
    lengths[0] = 0
    if rows > 2:
        lengths[1] = topk - 3
        lengths[2] = 70000
    starts = torch.zeros_like(lengths)
    for row in range(rows):
        logits[row, int(lengths[row]) :] += 2.0

    offsets, counts = gluon_dsa_row_bounded_logical_topk_gfx950(
        logits, starts, lengths, topk=topk
    )
    _, expected_counts = gluon_dsa_logical_topk_gfx950(
        logits, starts, lengths, topk=topk
    )

    torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)
    for row in range(rows):
        count = int(counts[row])
        length = int(lengths[row])
        assert count == min(topk, length)
        assert (offsets[row, count:] == -1).all()
        mine = offsets[row, :count].long()
        assert bool(((mine >= 0) & (mine < length)).all())
        assert len(set(mine.tolist())) == count
        if count:
            assert torch.equal(
                logits[row, mine].sort().values,
                logits[row, :length].topk(count).values.sort().values,
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


def _selected_scores(records: torch.Tensor, positions, counts, row: int) -> list:
    ids, bits = records[:, row].reshape(-1, 2).unbind(-1)
    score = dict(zip(ids.tolist(), bits.view(torch.float32).tolist()))
    return sorted(score[p] for p in positions[row, : int(counts[row])].tolist())


def test_gluon_merge_is_the_default_gfx950_merge():
    signature = format_signature(records=dense_tensor_format(torch.int32))
    for topk in (512, 1024, 2048):
        kernel = select_kernel(
            "attention", "dsa_merge_candidates", signature, traits={"topk": topk}
        )
        assert kernel.name == "gluon_dsa_merge_candidates_gfx950"


@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 7, 64])
@pytest.mark.parametrize("topk", [512, 2048])
def test_gluon_merge_selects_the_portable_merges_candidates(degree, rows, topk):
    records = _records(degree, rows, topk, seed=degree * 100 + rows)

    positions, counts = dsa_merge_candidates(records, topk=topk, solution="gluon")
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


@pytest.mark.parametrize("degree", [2, 8])
@pytest.mark.parametrize("topk", [512, 2048])
def test_gluon_merge_selects_the_same_tied_candidates_every_time(degree, topk):
    # Eight distinct scores: the top-k cutoff always falls inside a tied
    # group. Every KVP rank merges the same records and must keep the same
    # candidates.
    records = _records(degree, 16, topk, seed=degree + topk)
    generator = torch.Generator(device="cuda").manual_seed(topk)
    tied = torch.randint(
        0, 8, records.shape[:3], device="cuda", generator=generator
    ).float()
    records[..., 1] = tied.view(torch.int32)

    positions, counts = dsa_merge_candidates(records, topk=topk, solution="gluon")

    for _ in range(10):
        again, again_counts = dsa_merge_candidates(records, topk=topk, solution="gluon")
        torch.testing.assert_close(again, positions, atol=0, rtol=0)
        torch.testing.assert_close(again_counts, counts, atol=0, rtol=0)
    one, one_count = dsa_merge_candidates(
        records[:, 3:4].contiguous(), topk=topk, solution="gluon"
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


@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 7, 64])
@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("tied", [False, True])
def test_per_source_selection_matches_per_row_selection(degree, rows, topk, tied):
    records = _records(degree, rows, topk, seed=degree * 1000 + rows + topk)
    if tied:
        generator = torch.Generator(device="cuda").manual_seed(rows)
        ties = torch.randint(
            0, 8, records.shape[:3], device="cuda", generator=generator
        )
        records[..., 1] = ties.float().view(torch.int32)
    scores = rank_major_scores(records)
    order, _ = gluon_dsa_logical_topk_gfx950(
        scores,
        torch.zeros((rows,), device="cuda", dtype=torch.int32),
        torch.full((rows,), scores.shape[1], device="cuda", dtype=torch.int32),
        topk=topk,
    )

    positions, counts = select_through_threshold_by_source(
        records, scores, order, topk=topk
    )
    expected, expected_counts = select_through_threshold(
        records, scores, order, topk=topk
    )

    torch.testing.assert_close(positions, expected, atol=0, rtol=0)
    torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)


@pytest.mark.parametrize("degree", [2, 8])
@pytest.mark.parametrize("rows", [1, 64])
def test_rank_major_scores_with_bounds_cover_every_candidate(degree, rows):
    records = _records(degree, rows, 1024, seed=degree + rows)

    scores, starts, ends = rank_major_scores_with_bounds(records)

    torch.testing.assert_close(scores, rank_major_scores(records), atol=0, rtol=0)
    assert (starts == 0).all()
    assert (ends == degree * 1024).all()


def test_gluon_merge_launches_one_specialization_across_batch_rows():
    # Small batches select per source, large ones per row.
    for rows in (1, 16, 600):
        dsa_merge_candidates(
            _records(2, rows, 2048, seed=rows), topk=2048, solution="gluon"
        )
    with assert_no_triton_compile(
        _extract_rank_major_scores,
        _select_through_threshold,
        _select_source_through_threshold,
    ):
        for rows in (3, 8, 37, 64, 513, 700):
            dsa_merge_candidates(
                _records(2, rows, 2048, seed=rows), topk=2048, solution="gluon"
            )
