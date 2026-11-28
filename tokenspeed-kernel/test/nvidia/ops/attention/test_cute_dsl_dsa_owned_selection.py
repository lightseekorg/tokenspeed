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

"""Check the CuTe DSL owned selection against the compacted merge.

The reference is the merge's selection compacted to each rank's owned slots.
Every rank's rows must match bit for bit, with mandatory, signed zero, NaN,
short and empty rows and score bands that stress the radix digits and the tie
quota; then positions past int32 offsets, one compile across batch rows, and
the sizes it refuses.
"""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_cute_compile, is_nvidia

if not is_nvidia():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)
pytest.importorskip("cutlass")
from tokenspeed_kernel import platform  # noqa: E402

if (
    not platform.ArchVersion(10, 0)
    <= platform.current_platform().arch_version
    <= platform.ArchVersion(10, 3)
):
    pytest.skip("requires SM100 or SM103", allow_module_level=True)

from tokenspeed_kernel.ops.attention.dsa import (  # noqa: E402
    dsa_select_owned_candidates,
    dsa_select_owned_candidates_supported,
)
from tokenspeed_kernel.ops.attention.dsa._triton.candidate_exchange import (  # noqa: E402
    rank_major_scores,
    select_through_threshold,
)
from tokenspeed_kernel.ops.attention.dsa.triton import (  # noqa: E402
    local_topk_to_global_slots,
)
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import (  # noqa: E402
    compact_owned_slots,
    virtual_slots_to_local,
)

_PAGE, _BLOCK = 64, 128


def _owned_candidates(degree: int, rows: int, q_len: int, topk: int, seed: int):
    """Build a virtual index-page table and every rank's owned candidates.

    Returns positions ``[degree, rows, topk]`` (``-1`` invalid) on pages the
    rank owns and their scores (``-inf`` invalid), with mandatory,
    signed-zero, NaN, short, and empty rows.
    """
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
    """Return every rank's local slots through the full exchange.

    That is the merge's selection (``cuda_dsa_merge_candidates``' threshold
    selection over an exact top-k), its scheduler slots, then each rank's
    owned share.
    """
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
    + [(8, 64, 4, 2048, band) for band in ("tied", "narrow", "single", "short")],
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
        # Scores sharing their sign, exponent, and leading mantissa bits,
        # beside the mandatory, zero, and NaN ones.
        scores = torch.where(
            scores.isfinite() & (scores > 0), 5 + scores * 1e-3, scores
        )
    elif band == "single":
        scores = torch.where(positions < 0, scores, 2.5)
    elif band == "short":
        # Every row's history fits the top-k: each rank keeps all it holds.
        positions[:, :, topk // (2 * degree) :] = -1
        scores[positions < 0] = -torch.inf
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
        output_page_stride = _PAGE + 16
        local_slots, local_counts = dsa_select_owned_candidates(
            gathered,
            positions[rank].contiguous(),
            _local_pages(table, virtual_blocks, degree, rank),
            requests,
            rank=rank,
            topk=topk,
            page_size=_PAGE,
            output_page_stride=output_page_stride,
            solution="cute_dsl",
        )

        slots = torch.where(
            slots >= 0, slots // _PAGE * output_page_stride + slots % _PAGE, -1
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
        solution="cute_dsl",
        out=out[1 : rows + 1],
        lens_out=lens_out[1 : rows + 1],
    )

    assert local_slots.data_ptr() == out[1].data_ptr()
    torch.testing.assert_close(out[1 : rows + 1], slots, atol=0, rtol=0)
    torch.testing.assert_close(lens_out[1 : rows + 1], counts, atol=0, rtol=0)
    assert (out[0] == 7).all() and (out[rows + 1] == 7).all()
    assert int(lens_out[0]) == 7 and int(lens_out[rows + 1]) == 7


def test_owned_selection_compiles_once_across_batch_rows():
    def select(rows):
        table, virtual_blocks, positions, scores = _owned_candidates(
            8, rows, 1, 2048, seed=rows
        )
        dsa_select_owned_candidates(
            scores,
            positions[1].contiguous(),
            _local_pages(table, virtual_blocks, 8, 1),
            torch.arange(rows, device="cuda", dtype=torch.int32),
            rank=1,
            topk=2048,
            page_size=_PAGE,
            solution="cute_dsl",
        )

    select(1)
    with assert_no_cute_compile():
        for rows in (3, 8, 37, 64):
            select(rows)


def test_owned_selection_refuses_gathered_scores_it_cannot_take():
    assert dsa_select_owned_candidates_supported(
        ranks=8, rows=64, topk=2048, rank_stride=64 * 2048 + 1152
    )
    # Such KVP decodes keep the merge.
    for ranks, topk, rank_stride in (
        # Not a power of two.
        (3, 2048, 2048),
        # Every rank's share of a row past the registers.
        (32, 2048, 2048),
        # A top-k under the CTA's threads' minimum.
        (2, 256, 256),
        # Rank chunks that overlap.
        (8, 2048, 2047),
    ):
        assert not dsa_select_owned_candidates_supported(
            ranks=ranks, rows=1, topk=topk, rank_stride=rank_stride
        )
