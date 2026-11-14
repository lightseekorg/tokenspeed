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
table-wide buffer, and each rank's owned share of the merged selection."""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD GFX950 is required for the Gluon DCP top-k", allow_module_level=True
    )

from tokenspeed_kernel.ops.attention.dsa import (  # isort: skip
    dsa_select_owned_candidates,
    dsa_select_owned_candidates_supported,
)
from tokenspeed_kernel.ops.attention.dsa._triton.candidate_exchange import (  # isort: skip
    rank_major_scores,
    select_through_threshold,
)
from tokenspeed_kernel.ops.attention.dsa.triton import (  # isort: skip
    local_topk_to_global_slots,
)
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import (  # isort: skip
    compact_owned_slots,
    virtual_slots_to_local,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.owned_selection import (  # isort: skip
    _select_owned_from_gathered_kernel,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (  # isort: skip
    gluon_dsa_row_bounded_logical_topk_gfx950,
)


def _assert_row_topk(logits, offsets, counts, lengths, row, topk):
    """Row ``row`` selects ``min(topk, length)`` distinct columns of its
    prefix holding its largest scores, then ``-1``."""
    count = int(counts[row])
    length = int(lengths[row])
    assert count == min(topk, length)
    assert (offsets[row, count:] == -1).all()
    mine = offsets[row, :count].long()
    assert bool(((mine >= 0) & (mine < length)).all())
    assert len(set(mine.tolist())) == count
    assert torch.equal(
        logits[row, mine].sort().values,
        logits[row, :length].topk(count).values.sort().values,
    )


@pytest.mark.parametrize(
    "topk,width",
    [
        (512, 1 << 20),
        (2048, 1 << 20),
        # Narrower than the top-k: the sort path.
        (512, 256),
    ],
)
def test_row_bounded_topk_selects_each_rows_largest_owned_scores(topk, width):
    # Owned rows a fraction of a max-model-len-wide buffer whose unwritten
    # tail holds larger values than any owned score; without starts every
    # row begins at column zero.
    rows = 64
    generator = torch.Generator(device="cuda").manual_seed(rows + topk)
    logits = torch.rand((rows, width), device="cuda", generator=generator)
    lengths = torch.randint(
        topk // 2, 4 * topk, (rows,), device="cuda", generator=generator
    ).int()
    lengths[0] = 0
    lengths[1] = topk - 3
    lengths[2] = 70000
    lengths %= width + 1
    for row in range(rows):
        logits[row, int(lengths[row]) :] += 2.0

    offsets, counts = gluon_dsa_row_bounded_logical_topk_gfx950(
        logits, None, lengths, topk=topk
    )

    for row in range(rows):
        _assert_row_topk(logits, offsets, counts, lengths, row, topk)


def test_row_bounded_topk_reads_rows_past_int32_offsets():
    # A max-model-len-wide buffer whose last rows start past 2**31 logits;
    # the rows before them are empty and never read.
    rows, width, topk, tail = 2112, 1 << 20, 2048, 4
    generator = torch.Generator(device="cuda").manual_seed(31)
    logits = torch.empty((rows, width), device="cuda")
    logits[-tail:] = torch.rand((tail, width), device="cuda", generator=generator)
    lengths = torch.zeros((rows,), device="cuda", dtype=torch.int32)
    lengths[-tail:] = torch.tensor([topk - 3, 70000, 300000, width])
    starts = torch.zeros_like(lengths)
    assert (rows - tail) * logits.stride(0) > 2**31

    offsets, counts = gluon_dsa_row_bounded_logical_topk_gfx950(
        logits, starts, lengths, topk=topk
    )

    assert (counts[:-tail] == 0).all()
    for row in range(rows - tail, rows):
        _assert_row_topk(logits, offsets, counts, lengths, row, topk)


# K3.1's cache geometry: 64-token index pages inside 128-token cache blocks,
# block ``b`` on rank ``(b - 1) % degree``.
_PAGE, _BLOCK = 64, 128


def _owned_candidates(degree: int, rows: int, q_len: int, topk: int, seed: int):
    """A virtual index-page table and every rank's candidates, all on pages
    the rank owns: positions ``[degree, rows, topk]`` (``-1`` invalid) and
    their scores (``-inf`` invalid), with mandatory, signed-zero, NaN, short
    and empty rows."""
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
    scores[:, 0, 7] = torch.nan
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
    selection (``cuda_dsa_merge_candidates``' threshold selection over an
    exact top-k), its scheduler slots, then each rank's owned share."""
    degree = positions.shape[0]
    records = torch.stack((positions, scores.view(torch.int32)), dim=-1)
    merged = rank_major_scores(records)
    order = torch.topk(merged, topk, dim=-1).indices
    selected, counts = select_through_threshold(records, merged, order, topk=topk)
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


# Eight ranks, 16 requests of 4 verify rows, top-2048: K3.1's KVP8 decode.
# The score bands that stress the radix passes and the tie quota run there.
@pytest.mark.parametrize(
    "degree,rows,q_len,topk,band",
    [(2, 64, 4, 512, "uniform"), (4, 64, 4, 2048, "uniform")]
    + [(8, 64, 4, topk, "uniform") for topk in (512, 2048)]
    + [(8, 7, 1, 2048, "uniform")]
    + [(8, 64, 4, 2048, band) for band in ("tied", "narrow", "single")],
)
def test_owned_selection_is_each_ranks_share_of_the_merge(
    degree, rows, q_len, topk, band
):
    table, virtual_blocks, positions, scores = _owned_candidates(
        degree, rows, q_len, topk, seed=degree * 1000 + rows + topk
    )
    if band == "tied":
        # The cutoff falls inside a tied group that spans the ranks.
        generator = torch.Generator(device="cuda").manual_seed(rows + topk)
        ties = torch.randint(
            0, 8, scores.shape, device="cuda", generator=generator
        ).float()
        scores = torch.where(positions < 0, scores, ties)
    elif band == "narrow":
        # Scores sharing their sign, exponent and leading mantissa bits,
        # beside the mandatory, zero and NaN ones.
        scores = torch.where(
            scores.isfinite() & (scores > 0), 5 + scores * 1e-3, scores
        )
    elif band == "single":
        scores = torch.where(positions < 0, scores, 2.5)
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
            solution="gluon",
        )

        torch.testing.assert_close(local_slots, slots, atol=0, rtol=0)
        torch.testing.assert_close(local_counts, counts, atol=0, rtol=0)
    total = sum(counts for _, counts in expected)
    assert (total <= topk).all()
    assert (total[2] == 0).all()


def test_owned_selection_reads_positions_past_int32_offsets():
    # This rank's candidate positions in rows an int32 stride apart, the last
    # row (one with candidates) past 2**31 elements, selected into the
    # caller's rows of wider buffers.
    degree, rows, topk = 2, 4, 2048
    table, virtual_blocks, positions, scores = _owned_candidates(
        degree, rows, 1, topk, seed=23
    )
    (slots, counts), *_ = _merged_then_compacted(
        table, virtual_blocks, positions, scores, 1, topk
    )
    assert int(counts[-1]) > 0
    stride = (2**31 // (rows - 1) + 64) // 64 * 64
    storage = torch.empty((rows - 1) * stride + topk, device="cuda", dtype=torch.int32)
    far = storage.as_strided((rows, topk), (stride, 1))
    far.copy_(positions[0])
    assert (rows - 1) * far.stride(0) > 2**31
    out = torch.full((rows + 2, topk), 7, device="cuda", dtype=torch.int32)
    lens_out = torch.full((rows + 2,), 7, device="cuda", dtype=torch.int32)

    local_slots, _ = dsa_select_owned_candidates(
        scores,
        far,
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
    with assert_no_triton_compile(_select_owned_from_gathered_kernel):
        for rows in (3, 8, 37, 64):
            select(rows)


def test_owned_selection_refuses_gathered_scores_it_cannot_take():
    assert dsa_select_owned_candidates_supported(
        ranks=2, rows=1, topk=2048, rank_stride=2048
    )
    # Such KVP decodes keep the merge.
    for ranks, rank_stride in (
        # Not a power of two.
        (3, 2048),
        # Every rank's share of a row past the registers.
        (16, 2048),
        # Rank chunks read past 32-bit offsets.
        (8, 2**31 // 7),
    ):
        assert not dsa_select_owned_candidates_supported(
            ranks=ranks, rows=1, topk=2048, rank_stride=rank_stride
        )
