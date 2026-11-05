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

from tokenspeed_kernel.ops.attention.dsa import (  # isort: skip
    dsa_merge_candidates,
    dsa_select_owned_candidates,
)
from tokenspeed_kernel.ops.attention.dsa._triton.candidate_exchange import (  # isort: skip
    _extract_rank_major_scores,
    _rank_major_planar_scores,
    _select_owned_through_threshold,
    _select_source_through_threshold,
    _select_through_threshold,
    rank_major_scores,
    rank_major_scores_with_bounds,
    select_through_threshold,
    select_through_threshold_by_source,
)
from tokenspeed_kernel.ops.attention.dsa.triton import (  # isort: skip
    local_topk_to_global_slots,
)
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import (  # isort: skip
    compact_owned_slots,
    virtual_slots_to_local,
)
from tokenspeed_kernel.selection import select_kernel  # isort: skip
from tokenspeed_kernel.signature import (  # isort: skip
    dense_tensor_format,
    format_signature,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (  # isort: skip
    gluon_dsa_live_logical_topk_gfx950,
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


@pytest.mark.parametrize("width", [256, 1 << 16])
def test_row_bounded_topk_without_starts_begins_every_row_at_column_zero(width):
    # Narrower and wider than the top-k: the sort and the radix paths.
    rows, topk = 16, 512
    generator = torch.Generator(device="cuda").manual_seed(width)
    logits = torch.rand((rows, width), device="cuda", generator=generator)
    lengths = torch.randint(
        0, width + 1, (rows,), device="cuda", generator=generator
    ).int()
    offsets, counts = gluon_dsa_row_bounded_logical_topk_gfx950(
        logits, None, lengths, topk=topk
    )
    expected, expected_counts = gluon_dsa_row_bounded_logical_topk_gfx950(
        logits, torch.zeros_like(lengths), lengths, topk=topk
    )

    torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)
    # Equal-score order is unspecified; the selected sets must agree.
    torch.testing.assert_close(
        offsets.sort(dim=1).values, expected.sort(dim=1).values, atol=0, rtol=0
    )


@pytest.mark.parametrize("rows", [1, 64])
@pytest.mark.parametrize("topk", [512, 2048])
def test_live_topk_keeps_every_live_score_of_rows_holding_at_most_topk(rows, topk):
    # Merged candidate rows: padding rows with no live score, short
    # histories with fewer live scores than topk, live scores scattered
    # among -inf ones, and mandatory +inf ones.
    width = 8 * topk
    generator = torch.Generator(device="cuda").manual_seed(rows + topk)
    logits = torch.rand((rows, width), device="cuda", generator=generator)
    live_counts = torch.randint(
        0, width, (rows,), device="cuda", generator=generator
    ).tolist()
    live_counts[: min(rows, 4)] = [0, topk - 5, topk, topk + 1][:rows]
    starts = torch.zeros((rows,), device="cuda", dtype=torch.int32)
    ends = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    live = torch.zeros((rows, width), device="cuda", dtype=torch.bool)
    for row, count in enumerate(live_counts):
        columns = torch.randperm(width, device="cuda", generator=generator)
        live[row, columns[:count]] = True
    logits[~live] = -torch.inf
    if rows > 4:
        logits[3, live[3].nonzero()[:3, 0]] = torch.inf
        # A row narrower than topk.
        starts[4], ends[4] = 100, 100 + topk // 2
        live[4, :100] = False
        live[4, 100 + topk // 2 :] = False

    offsets, counts = gluon_dsa_live_logical_topk_gfx950(
        logits, starts, ends, topk=topk
    )

    for row in range(rows):
        live_columns = live[row].nonzero()[:, 0]
        count = int(counts[row])
        assert count == min(topk, live_columns.numel())
        assert (offsets[row, count:] == -1).all()
        mine = offsets[row, :count].long()
        assert len(set(mine.tolist())) == count
        assert bool(live[row, mine].all())
        if live_columns.numel() <= topk:
            assert sorted(mine.tolist()) == live_columns.tolist()
        else:
            assert torch.equal(
                logits[row, mine].sort().values,
                logits[row, live_columns].topk(topk).values.sort().values,
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


# K3.1's cache geometry: 64-token index pages inside 128-token cache blocks,
# block ``b`` on rank ``(b - 1) % degree``.
_PAGE, _BLOCK = 64, 128


def _owned_candidates(degree: int, rows: int, q_len: int, topk: int, seed: int):
    """A virtual index-page table and every rank's candidates, all on pages
    the rank owns: positions ``[degree, rows, topk]`` (``-1`` invalid) and
    their scores (``-inf`` invalid), with mandatory, signed-zero, short and
    empty rows."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    requests = rows // q_len
    blocks_per_request = 24 * degree
    virtual_blocks = requests * blocks_per_request + 1
    blocks = 1 + torch.randperm(
        virtual_blocks - 1, device="cuda", generator=generator
    ).view(requests, blocks_per_request)
    pages_per_block = _BLOCK // _PAGE
    table = blocks.unsqueeze(-1) * pages_per_block + torch.arange(
        pages_per_block, device="cuda"
    )
    table = table.view(requests, -1).int()
    owner = (table // pages_per_block - 1) % degree
    positions = torch.full((degree, rows, topk), -1, device="cuda", dtype=torch.int32)
    for source in range(degree):
        for row in range(rows):
            pages = (owner[row // q_len] == source).nonzero().flatten()
            tokens = pages.unsqueeze(1) * _PAGE + torch.arange(_PAGE, device="cuda")
            tokens = tokens.flatten()
            pick = torch.randperm(tokens.numel(), device="cuda", generator=generator)
            pick = tokens[pick[:topk]]
            positions[source, row, : pick.numel()] = pick.int()
    scores = torch.rand((degree, rows, topk), device="cuda", generator=generator)
    scores[:, 0, 3:6] = torch.tensor([torch.inf, torch.inf, -0.0], device="cuda")
    scores[:, 0, 6] = 0.0
    if rows > 2:
        # A short history: fewer valid candidates in all than topk.
        positions[:, 1, topk // (2 * degree) :] = -1
        # A request step whose history no rank owns.
        positions[:, 2] = -1
    scores[positions < 0] = -torch.inf
    return table, virtual_blocks, positions, scores


def _local_pages(table, virtual_blocks: int, degree: int, rank: int):
    slots, owned = virtual_slots_to_local(
        table.long() * _PAGE,
        rows_per_page=_BLOCK,
        virtual_block_count=virtual_blocks,
        degree=degree,
        rank=rank,
    )
    return torch.where(owned, slots // _PAGE, -1).int()


def _merged_then_compacted(table, virtual_blocks, positions, scores, q_len, topk):
    """Every rank's local slots through the full exchange: the merge's
    selection by source, its scheduler slots, then each rank's owned share."""
    degree = positions.shape[0]
    records = torch.stack((positions, scores.view(torch.int32)), dim=-1)
    merged = rank_major_scores(records)
    order = torch.topk(merged, topk, dim=-1).indices
    selected, counts = select_through_threshold_by_source(
        records, merged, order, topk=topk
    )
    slots, lens = local_topk_to_global_slots(
        local_topk_offsets=selected,
        block_table=table,
        block_size=_PAGE,
        q_len_per_req=q_len,
    )
    torch.testing.assert_close(lens, counts, atol=0, rtol=0)
    return [
        compact_owned_slots(
            slots,
            lens,
            rows_per_page=_BLOCK,
            virtual_block_count=virtual_blocks,
            degree=degree,
            rank=rank,
        )
        for rank in range(degree)
    ]


@pytest.mark.parametrize("solution", ["gluon", "triton"])
@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("rows,q_len", [(1, 1), (7, 1), (64, 4)])
@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("tied", [False, True])
def test_owned_selection_is_each_ranks_share_of_the_merge(
    solution, degree, rows, q_len, topk, tied
):
    table, virtual_blocks, positions, scores = _owned_candidates(
        degree, rows, q_len, topk, seed=degree * 1000 + rows + topk
    )
    if tied:
        # The cutoff falls inside a tied group that spans the ranks.
        generator = torch.Generator(device="cuda").manual_seed(rows + topk)
        ties = torch.randint(
            0, 8, scores.shape, device="cuda", generator=generator
        ).float()
        scores = torch.where(positions < 0, scores, ties)
    expected = _merged_then_compacted(
        table, virtual_blocks, positions, scores, q_len, topk
    )
    # Each rank's shard of a send buffer wider than the scores.
    gathered = torch.full(
        (degree, rows * topk + 32), torch.nan, device="cuda", dtype=torch.float32
    )
    gathered[:, : rows * topk] = scores.view(degree, -1)
    gathered = gathered[:, : rows * topk].view(degree, rows, topk)
    requests = (torch.arange(rows, device="cuda") // q_len).int()

    for rank, (slots, counts) in enumerate(expected):
        local_slots, local_counts = dsa_select_owned_candidates(
            gathered,
            positions[rank].contiguous(),
            _local_pages(table, virtual_blocks, degree, rank),
            requests,
            rank=rank,
            topk=topk,
            page_size=_PAGE,
            solution=solution,
        )

        torch.testing.assert_close(local_slots, slots, atol=0, rtol=0)
        torch.testing.assert_close(local_counts, counts, atol=0, rtol=0)
    total = sum(counts for _, counts in expected)
    assert (total <= topk).all()
    if rows > 2:
        assert (total[2] == 0).all()


def test_owned_selection_writes_into_the_callers_rows():
    degree, rows, topk = 4, 7, 512
    table, virtual_blocks, positions, scores = _owned_candidates(
        degree, rows, 1, topk, seed=5
    )
    (slots, counts), *_ = _merged_then_compacted(
        table, virtual_blocks, positions, scores, 1, topk
    )
    out = torch.full((rows + 2, topk), 7, device="cuda", dtype=torch.int32)
    lens_out = torch.full((rows + 2,), 7, device="cuda", dtype=torch.int32)

    local_slots, local_counts = dsa_select_owned_candidates(
        scores,
        positions[0].contiguous(),
        _local_pages(table, virtual_blocks, degree, 0),
        torch.arange(rows, device="cuda", dtype=torch.int32),
        rank=0,
        topk=topk,
        page_size=_PAGE,
        solution="gluon",
        out=out[1 : rows + 1],
        lens_out=lens_out[1 : rows + 1],
    )

    assert local_slots.data_ptr() == out[1].data_ptr()
    torch.testing.assert_close(out[1 : rows + 1], slots, atol=0, rtol=0)
    torch.testing.assert_close(lens_out[1 : rows + 1], counts, atol=0, rtol=0)
    assert (out[0] == 7).all() and (out[rows + 1] == 7).all()
    assert int(lens_out[0]) == 7 and int(lens_out[rows + 1]) == 7


def test_owned_selection_launches_one_specialization_across_batch_rows():
    def select(rows):
        table, virtual_blocks, positions, scores = _owned_candidates(
            2, rows, 1, 2048, seed=rows
        )
        dsa_select_owned_candidates(
            scores,
            positions[1].contiguous(),
            _local_pages(table, virtual_blocks, 2, 1),
            torch.arange(rows, device="cuda", dtype=torch.int32),
            rank=1,
            topk=2048,
            page_size=_PAGE,
            solution="gluon",
        )

    select(1)
    with assert_no_triton_compile(
        _rank_major_planar_scores, _select_owned_through_threshold
    ):
        for rows in (3, 8, 37, 64):
            select(rows)
