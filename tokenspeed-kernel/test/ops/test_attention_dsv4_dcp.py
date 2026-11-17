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

"""DeepSeek V4 decode context parallelism kernels.

Virtual block translation, the LSE merge of per-shard attention partials, and
the sink applied once after merging.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.dsv4 import (
    torch_dsv4_merge_partials,
)
from tokenspeed_kernel.numerics.torch.kvcache import (
    torch_virtual_slots_to_local,
)
from tokenspeed_kernel.ops.attention.dsa._triton import candidate_exchange as exchange
from tokenspeed_kernel.ops.attention.dsa._triton import index_candidates as candidates
from tokenspeed_kernel.ops.attention.dsv4._triton import dcp
from tokenspeed_kernel.ops.attention.dsv4._triton.dcp import (
    dcp_apply_sink,
    dcp_weight_for_reduce_scatter,
    normalize_dcp_partials,
)
from tokenspeed_kernel.ops.kvcache import triton_cache_placement
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import (
    compact_owned_slots,
    virtual_slots_to_local,
)
from utils import assert_no_triton_compile

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

# ---------------------------------------------------------------------------
# Virtual block translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 64, 256])
@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("parents,packing", [(3, 1), (2, 4), (5, 2)])
def test_owned_pages_partition_the_virtual_space(rows, degree, parents, packing):
    virtual_count = 1 + parents * degree * packing
    local_pages = 1 + parents * packing
    slots = torch.arange(-3, virtual_count * rows + 5, dtype=torch.int64)
    covered = torch.zeros(virtual_count, dtype=torch.int64)
    for rank in range(degree):
        local, owned = virtual_slots_to_local(
            slots,
            rows_per_page=rows,
            virtual_block_count=virtual_count,
            degree=degree,
            rank=rank,
        )
        expected_local, expected_owned = torch_virtual_slots_to_local(
            slots,
            rows_per_page=rows,
            virtual_block_count=virtual_count,
            degree=degree,
            rank=rank,
        )
        assert torch.equal(local, expected_local)
        assert torch.equal(owned, expected_owned)
        # Unowned slots park on slot 0; nothing negative, null or out of range is owned.
        assert not owned[slots < rows].any()
        assert not owned[slots >= virtual_count * rows].any()
        assert (local[~owned] == 0).all()
        pages = local[owned] // rows
        assert pages.min() >= 1 and pages.max() == local_pages - 1
        # Each owned local page receives exactly ``rows`` slots.
        assert torch.equal(
            torch.bincount(pages, minlength=local_pages)[1:],
            torch.full((local_pages - 1,), rows),
        )
        covered += torch.bincount(slots[owned] // rows, minlength=virtual_count)
    # Every non-null virtual block has exactly one owner across the ranks.
    assert covered[0] == 0
    assert (covered[1:] == rows).all()


@requires_cuda
@pytest.mark.parametrize("rows", [1, 64])
@pytest.mark.parametrize("degree", [1, 4])
def test_virtual_slot_translation_matches_between_cpu_and_cuda(rows, degree):
    virtual_count = 1 + 7 * degree * 4
    slots = torch.randint(-5, virtual_count * rows + 9, (1531,), dtype=torch.int64)
    for rank in range(degree):
        cpu_local, cpu_owned = virtual_slots_to_local(
            slots,
            rows_per_page=rows,
            virtual_block_count=virtual_count,
            degree=degree,
            rank=rank,
        )
        cuda_local, cuda_owned = virtual_slots_to_local(
            slots.cuda(),
            rows_per_page=rows,
            virtual_block_count=virtual_count,
            degree=degree,
            rank=rank,
        )
        assert torch.equal(cuda_local.cpu(), cpu_local)
        assert torch.equal(cuda_owned.cpu(), cpu_owned)


@requires_cuda
def test_virtual_slot_translation_reuses_caller_outputs():
    slots = torch.arange(0, 400, dtype=torch.int32, device="cuda").view(20, 20)
    out = torch.empty_like(slots)
    mask = torch.empty(slots.shape, dtype=torch.bool, device="cuda")
    local, owned = virtual_slots_to_local(
        slots,
        rows_per_page=1,
        virtual_block_count=401,
        degree=2,
        rank=1,
        out=out,
        owner_mask=mask,
    )
    assert local.data_ptr() == out.data_ptr()
    assert owned.data_ptr() == mask.data_ptr()
    expected_local, expected_owned = torch_virtual_slots_to_local(
        slots.long().cpu(), rows_per_page=1, virtual_block_count=401, degree=2, rank=1
    )
    assert torch.equal(local.cpu().long(), expected_local)
    assert torch.equal(owned.cpu(), expected_owned)


def test_virtual_slot_translation_rejects_invalid_geometry():
    slots = torch.arange(8, dtype=torch.int64)
    with pytest.raises(ValueError, match="owner rank"):
        virtual_slots_to_local(
            slots, rows_per_page=1, virtual_block_count=9, degree=2, rank=2
        )
    with pytest.raises(ValueError, match="usable pages"):
        virtual_slots_to_local(
            slots, rows_per_page=1, virtual_block_count=1, degree=1, rank=0
        )
    with pytest.raises(TypeError):
        virtual_slots_to_local(
            slots.float(), rows_per_page=1, virtual_block_count=9, degree=1, rank=0
        )


def _owned_selection(rows, width, *, rows_per_page, virtual_count, seed):
    # Virtual slots with -1 holes and null-block, negative and past-capacity entries.
    generator = torch.Generator().manual_seed(seed)
    slots = torch.randint(
        -5,
        virtual_count * rows_per_page + 3 * rows_per_page,
        (rows, width),
        generator=generator,
        dtype=torch.int64,
    )
    holes = torch.rand((rows, width), generator=generator) < 0.1
    return slots.masked_fill(holes, -1).to(torch.int32)


def _compacted_reference(slots, lengths, **geometry):
    local, owned = torch_virtual_slots_to_local(slots.long(), **geometry)
    width = slots.shape[1]
    if lengths is not None:
        owned &= torch.arange(width) < lengths.long().clamp(0, width).unsqueeze(1)
    expected = torch.full(slots.shape, -1, dtype=torch.int32)
    for row in range(slots.shape[0]):
        kept = local[row][owned[row]]
        expected[row, : kept.numel()] = kept.to(torch.int32)
    return expected, owned.sum(1).to(torch.int32)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("rows_per_page", [64, 128])
@pytest.mark.parametrize("lengths_kind", ["none", "partial", "zero"])
@pytest.mark.parametrize("width", [1, 600, 2051])
def test_compact_owned_slots_matches_reference(
    device, degree, rows_per_page, lengths_kind, width
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    rows = 7
    virtual_count = 1 + 5 * degree
    slots = _owned_selection(
        rows,
        width,
        rows_per_page=rows_per_page,
        virtual_count=virtual_count,
        seed=degree * 131 + width,
    )
    if lengths_kind == "none":
        lengths = None
    elif lengths_kind == "zero":
        lengths = torch.zeros(rows, dtype=torch.int32)
    else:
        # Clamped below zero and past the width.
        lengths = torch.tensor(
            [0, 1, width // 2, width - 1, width, width + 9, -3], dtype=torch.int32
        )
    geometry = dict(
        rows_per_page=rows_per_page, virtual_block_count=virtual_count, degree=degree
    )
    for rank in range(degree):
        expected, expected_counts = _compacted_reference(
            slots, lengths, rank=rank, **geometry
        )
        local, counts = compact_owned_slots(
            slots.to(device),
            None if lengths is None else lengths.to(device),
            rank=rank,
            **geometry,
        )
        assert local.dtype == torch.int32 and counts.dtype == torch.int32
        assert local.shape == slots.shape and local.is_contiguous()
        assert torch.equal(local.cpu(), expected)
        assert torch.equal(counts.cpu(), expected_counts)
    # int64 slots and lengths, and a column-strided view, compact alike.
    view = torch.cat((slots, slots), dim=1).long().to(device)[:, ::2]
    local, counts = compact_owned_slots(
        view,
        None if lengths is None else lengths.to(device).long(),
        rank=degree - 1,
        **geometry,
    )
    expected, expected_counts = _compacted_reference(
        view.cpu(), lengths, rank=degree - 1, **geometry
    )
    assert torch.equal(local.cpu(), expected)
    assert torch.equal(counts.cpu(), expected_counts)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_compact_owned_slots_partitions_selection(device, degree):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    rows, width, rows_per_page = 9, 2048, 128
    virtual_count = 1 + 24 * degree
    generator = torch.Generator().manual_seed(degree)
    # Distinct owned slots, as top-k selects them, with stale entries past each length.
    capacity = (virtual_count - 1) * rows_per_page
    slots = torch.stack(
        [
            torch.randperm(capacity, generator=generator)[:width] + rows_per_page
            for _ in range(rows)
        ]
    ).to(torch.int32)
    lengths = torch.randint(0, width + 1, (rows,), generator=generator)
    lengths[:3] = torch.tensor([0, 1, width])
    lengths = lengths.to(torch.int32)
    # One row selects only rank 0's blocks: other ranks receive nothing.
    rank0 = torch.arange(rows_per_page, virtual_count * rows_per_page)
    rank0 = rank0[(rank0 // rows_per_page - 1) % degree == 0][: width // 2]
    slots[3, : rank0.numel()] = rank0.to(torch.int32)
    lengths[3] = rank0.numel()
    total = torch.zeros(rows, dtype=torch.int32)
    restored = [[] for _ in range(rows)]
    for rank in range(degree):
        local, counts = compact_owned_slots(
            slots.to(device),
            lengths.to(device),
            rows_per_page=rows_per_page,
            virtual_block_count=virtual_count,
            degree=degree,
            rank=rank,
        )
        local, counts = local.cpu().long(), counts.cpu()
        total += counts
        for row in range(rows):
            count = int(counts[row])
            assert (local[row, count:] == -1).all()
            mine = local[row, :count]
            # Back to virtual slots through the cyclic rule.
            block = (mine // rows_per_page - 1) * degree + rank + 1
            virtual = block * rows_per_page + mine % rows_per_page
            live = slots[row, : int(lengths[row])].long()
            # Selection order is kept within the rank.
            assert torch.equal(
                virtual, live[(live // rows_per_page - 1) % degree == rank]
            )
            restored[row].append(virtual)
    assert torch.equal(total, lengths)
    for row in range(rows):
        union = torch.cat(restored[row]).sort().values
        assert torch.equal(union, slots[row, : int(lengths[row])].long().sort().values)


@requires_cuda
def test_compact_owned_slots_graph_replay():
    geometry = dict(rows_per_page=128, virtual_block_count=97, degree=8, rank=3)

    def selection(rows, width, seed):
        return _owned_selection(
            rows, width, rows_per_page=128, virtual_count=97, seed=seed
        )

    rows, width = 64 * 8, 2048
    slots = selection(rows, width, 0).cuda()
    lengths = torch.randint(0, width + 1, (rows,), dtype=torch.int32).cuda()
    compact_owned_slots(slots, lengths, **geometry)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        local, counts = compact_owned_slots(slots, lengths, **geometry)
    for seed in range(1, 4):
        slots.copy_(selection(rows, width, seed))
        lengths.copy_(torch.randint(0, width + 1, (rows,), dtype=torch.int32))
        graph.replay()
        expected, expected_counts = _compacted_reference(
            slots.cpu(), lengths.cpu(), **geometry
        )
        assert torch.equal(local.cpu(), expected)
        assert torch.equal(counts.cpu(), expected_counts)

    # Rows, widths and strides are runtime: one binary per placement and lengths form.
    compact_owned_slots(selection(4, 512, 7).cuda(), None, **geometry)
    with assert_no_triton_compile(triton_cache_placement._compact_owned_slots):
        for rows, width in ((1, 2048), (3, 2051), (16, 512), (33, 2048)):
            slots = selection(rows, width, rows).cuda()
            lengths = torch.full((rows,), width // 2, dtype=torch.int32).cuda()
            for row_lengths in (None, lengths):
                local, counts = compact_owned_slots(slots, row_lengths, **geometry)
                expected, expected_counts = _compacted_reference(
                    slots.cpu(),
                    None if row_lengths is None else row_lengths.cpu(),
                    **geometry,
                )
                assert torch.equal(local.cpu(), expected)
                assert torch.equal(counts.cpu(), expected_counts)


def test_compact_owned_slots_rejects_invalid_inputs():
    slots = torch.arange(8, dtype=torch.int64).view(2, 4)
    geometry = dict(rows_per_page=1, virtual_block_count=9)
    with pytest.raises(ValueError, match="owner rank"):
        compact_owned_slots(slots, None, degree=2, rank=2, **geometry)
    with pytest.raises(ValueError, match="usable pages"):
        compact_owned_slots(
            slots, None, rows_per_page=1, virtual_block_count=1, degree=1, rank=0
        )
    with pytest.raises(TypeError):
        compact_owned_slots(slots.float(), None, degree=1, rank=0, **geometry)
    with pytest.raises(ValueError, match="rows, width"):
        compact_owned_slots(slots.flatten(), None, degree=1, rank=0, **geometry)
    with pytest.raises(ValueError, match="lengths"):
        compact_owned_slots(
            slots, torch.ones(3, dtype=torch.int32), degree=1, rank=0, **geometry
        )
    local, counts = compact_owned_slots(
        slots[:0], torch.ones(0, dtype=torch.int32), degree=1, rank=0, **geometry
    )
    assert local.shape == (0, 4) and counts.shape == (0,)


# ---------------------------------------------------------------------------
# Partial normalization and the LSE merge
# ---------------------------------------------------------------------------


@requires_cuda
def test_normalize_marks_empty_and_non_finite_rows_as_empty():
    tokens, heads, dim = 5, 4, 8
    output = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    original = output.clone()
    lse = torch.randn(tokens, heads, device="cuda")
    lse[3, 1] = torch.inf  # FlashMLA's value for a row that attended to nothing.
    lse[4, 2] = torch.nan
    swa_lens = torch.tensor([3, 0, 0, 2, 5], device="cuda", dtype=torch.int32)
    extra_lens = torch.tensor([0, 0, 4, 1, 1], device="cuda", dtype=torch.int32)

    normalized_output, normalized_lse = normalize_dcp_partials(
        output, lse, swa_lens, extra_lens
    )

    assert normalized_output.data_ptr() == output.data_ptr()
    empty_rows = torch.tensor([False, True, False, False, False], device="cuda")
    assert torch.equal(
        normalized_output[empty_rows], torch.zeros_like(original[empty_rows])
    )
    assert torch.equal(
        normalized_output[~empty_rows][:, [0, 3]], original[~empty_rows][:, [0, 3]]
    )
    assert torch.isneginf(normalized_lse[1]).all()
    assert torch.isneginf(normalized_lse[3, 1]) and torch.isneginf(normalized_lse[4, 2])
    assert torch.equal(
        normalized_output[3, 1], torch.zeros(dim, device="cuda", dtype=torch.bfloat16)
    )
    kept = torch.isfinite(lse) & ~empty_rows[:, None]
    assert torch.equal(normalized_lse[kept], lse[kept])


@requires_cuda
@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_weight_and_sink_match_the_reference_merge_with_empty_shards(degree, dtype):
    torch.manual_seed(degree)
    tokens, local_heads, dim = 6, 4, 16
    heads = local_heads * degree
    partials = [
        torch.randn(tokens, heads, dim, device="cuda", dtype=dtype)
        for _ in range(degree)
    ]
    lses = [torch.randn(tokens, heads, device="cuda") * 3 for _ in range(degree)]
    # Token 0 is empty on shard 1, token 1 is empty everywhere, token 2 leaks
    # FlashMLA's +inf from shard 0 without a matching zero partial.
    lses[1][0] = -torch.inf
    partials[1][0] = 0
    for rank in range(degree):
        lses[rank][1] = -torch.inf
        partials[rank][1] = 0
    lses[0][2] = torch.inf
    sink = torch.randn(heads, device="cuda")

    expected, expected_lse = torch_dsv4_merge_partials(partials, lses, sink)
    all_lse = torch.stack(lses, dim=0)
    summed = None
    global_lses = []
    for rank in range(degree):
        weighted, global_lse = dcp_weight_for_reduce_scatter(
            partials[rank], all_lse, rank
        )
        assert (
            weighted.shape == (heads, tokens, dim) and weighted.dtype == torch.float32
        )
        assert global_lse.shape == (tokens, local_heads)
        summed = weighted if summed is None else summed + weighted
        global_lses.append(global_lse)
    actual = torch.cat(
        [
            dcp_apply_sink(
                summed[rank * local_heads : (rank + 1) * local_heads].movedim(0, 1),
                global_lses[rank],
                sink[rank * local_heads : (rank + 1) * local_heads].contiguous(),
                dtype=dtype,
            )
            for rank in range(degree)
        ],
        dim=1,
    )
    global_lse = torch.cat(global_lses, dim=1)

    assert not actual.isnan().any() and not global_lse.isnan().any()
    assert torch.equal(torch.isneginf(global_lse), torch.isneginf(expected_lse))
    finite = torch.isfinite(expected_lse)
    torch.testing.assert_close(
        global_lse[finite], expected_lse[finite], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)
    assert torch.equal(actual[1], torch.zeros_like(actual[1]))


@requires_cuda
def test_weight_propagates_an_unexpected_nan_lse():
    partials = torch.ones(2, 4, 8, device="cuda", dtype=torch.bfloat16)
    all_lse = torch.zeros(2, 2, 4, device="cuda")
    all_lse[1, 0, 1] = torch.nan  # shard 1, token 0, head 1 (rank 0's head slice)
    weighted, global_lse = dcp_weight_for_reduce_scatter(partials, all_lse, 0)
    assert weighted[1, 0].isnan().all()
    assert not weighted[0].isnan().any() and not weighted[2:].isnan().any()
    assert not weighted[1, 1].isnan().any()
    assert global_lse[0, 1].isnan()
    assert not global_lse[0, 0].isnan() and not global_lse[1].isnan().any()


@requires_cuda
def test_weight_and_sink_reject_inconsistent_shapes():
    partials = torch.ones(2, 4, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="topology"):
        dcp_weight_for_reduce_scatter(partials, torch.zeros(3, 2, 4, device="cuda"), 0)
    with pytest.raises(ValueError, match="shapes disagree"):
        dcp_weight_for_reduce_scatter(partials, torch.zeros(2, 2, 3, device="cuda"), 0)
    with pytest.raises(ValueError, match="shapes disagree"):
        dcp_apply_sink(
            partials.float(),
            torch.zeros(2, 4, device="cuda"),
            torch.zeros(2, device="cuda"),
            dtype=torch.bfloat16,
        )


# Candidate transport and head-owner merging share the DCP graph contract.


@pytest.mark.parametrize("rows", [0, 1, 7])
@pytest.mark.parametrize("degree", [1, 3, 8])
@pytest.mark.parametrize("width,topk", [(16, 8), (2053, 2049)])
@requires_cuda
def test_rank_major_merge_preserves_order_and_bits(rows, degree, width, topk):
    ids = torch.arange(degree * rows * width, device="cuda", dtype=torch.int32)
    ids = ids.reshape(degree, rows, width) + 2**24 + 1
    scores = torch.rand((degree, rows, width), device="cuda")
    if rows:
        ids[:, 0, -1] = -1
        scores[:, 0, :4] = torch.tensor(
            [torch.inf, -torch.inf, torch.nan, -0.0], device="cuda"
        )
        scores[:, -1, 4:] = -torch.inf
    records = torch.stack((ids, scores.view(torch.int32)), dim=-1)
    actual, counts = exchange.merge_packed_index_candidates(records, topk=topk)
    row_ids = ids.permute(1, 0, 2).reshape(rows, degree * width)
    row_scores = scores.permute(1, 0, 2).reshape(rows, degree * width)
    valid = (row_ids >= 0) & ~torch.isnan(row_scores) & (row_scores > -torch.inf)
    values, order = torch.topk(torch.where(valid, row_scores, -torch.inf), topk, dim=-1)
    expected = torch.where(values > -torch.inf, row_ids.gather(1, order), -1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        counts, (values > -torch.inf).sum(1).int(), atol=0, rtol=0
    )


@pytest.mark.parametrize("degree", [2, 3, 8])
@pytest.mark.parametrize("rows_per_owner", [1, 5])
@pytest.mark.parametrize("width,topk", [(16, 8), (2048, 2048)])
@pytest.mark.parametrize("tied", [False, True])
@requires_cuda
def test_rank_major_merge_row_slices_match_full_merge(
    degree, rows_per_owner, width, topk, tied
):
    """An owner merging its equal slice of the rows over every rank's
    candidates selects what merging every row selects for that slice: the
    merge is row-independent, which the KVP prefill's row-owner exchange
    relies on to merge each row on one rank only."""
    generator = torch.Generator(device="cuda").manual_seed(4117 + degree + width)
    rows = degree * rows_per_owner
    shape = (degree, rows, width)
    base = 2**24 + 1
    ids = torch.arange(degree * rows * width, device="cuda", dtype=torch.int32)
    ids = ids.reshape(shape) + base
    if tied:
        # Four score values: selections cut through runs of equal scores.
        scores = torch.randint(0, 4, shape, generator=generator, device="cuda").float()
    else:
        # Distinct scores order every candidate: the selection is unique.
        scores = torch.randperm(
            degree * rows * width, generator=generator, device="cuda"
        )
        scores = (scores + 1).float().reshape(shape)
    # Row 0: an invalid candidate of each kind, and one mandatory candidate.
    ids[:, 0, -1] = -1
    scores[:, 0, 1] = -torch.inf
    scores[:, 0, 2] = torch.nan
    scores[0, 0, 0] = torch.inf
    # The last row is cold: one valid candidate per rank.
    scores[:, -1, 1:] = -torch.inf
    if rows > 2:
        # Row 1 has none, as an exchange's padding rows.
        ids[:, 1] = -1
    records = torch.stack((ids, scores.view(torch.int32)), dim=-1)
    full_positions, full_counts = exchange.merge_packed_index_candidates(
        records, topk=topk
    )
    valid = (ids >= 0) & ~torch.isnan(scores) & (scores > -torch.inf)
    candidates = valid.sum(dim=(0, 2))
    torch.testing.assert_close(
        full_counts, candidates.clamp_max(topk).int(), atol=0, rtol=0
    )
    # Every candidate's score by position, -inf where invalid.
    by_position = torch.where(valid, scores, -torch.inf).flatten()

    def selected_scores(positions):
        picked = by_position[(positions - base).clamp_min(0).long()]
        picked = torch.where(positions >= 0, picked, -torch.inf)
        return picked.sort(dim=-1, descending=True).values

    owned = rows // degree
    for owner in range(degree):
        part = slice(owner * owned, (owner + 1) * owned)
        positions, counts = exchange.merge_packed_index_candidates(
            records[:, part].contiguous(), topk=topk
        )
        torch.testing.assert_close(counts, full_counts[part], atol=0, rtol=0)
        if not tied:
            torch.testing.assert_close(positions, full_positions[part], atol=0, rtol=0)
            continue
        # Ties may order or cut differently; scores and fully-selected rows still agree.
        torch.testing.assert_close(
            selected_scores(positions),
            selected_scores(full_positions[part]),
            atol=0,
            rtol=0,
        )
        whole = candidates[part] <= topk
        torch.testing.assert_close(
            positions[whole].sort(dim=-1).values,
            full_positions[part][whole].sort(dim=-1).values,
            atol=0,
            rtol=0,
        )


@requires_cuda
def test_threshold_selection_ignores_which_tied_candidates_a_topk_chose():
    # Three 2.0s tie at the top-4 cutoff; -inf, NaN and negative offsets never count.
    ids = torch.tensor(
        [[[10, 11, 12, 13]], [[20, 21, -1, 23]]], device="cuda", dtype=torch.int32
    )
    scores = torch.tensor(
        [[[2.0, -torch.inf, 5.0, torch.nan]], [[2.0, 9.0, 7.0, 2.0]]],
        device="cuda",
    )
    records = torch.stack((ids, scores.view(torch.int32)), dim=-1)
    rank_major = exchange.rank_major_scores(records)
    # Two exact top-4s keeping different tied candidates, in different orders.
    orders = [
        torch.tensor([[5, 2, 7, 4]], device="cuda", dtype=torch.int32),
        torch.tensor([[0, 5, 2, 4]], device="cuda", dtype=torch.int64),
    ]

    results = [
        exchange.select_through_threshold(records, rank_major, order, topk=4)
        for order in orders
    ]

    for positions, counts in results:
        # 9.0, 5.0 and the first two 2.0 ties (10, 20; not 23), in candidate order.
        torch.testing.assert_close(
            positions,
            torch.tensor([[10, 12, 20, 21]], device="cuda", dtype=torch.int32),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            counts, torch.tensor([4], device="cuda", dtype=torch.int32), atol=0, rtol=0
        )


@requires_cuda
def test_candidate_gather_bounds_reads_with_uninitialized_tail():
    logits = torch.full((2, 128), torch.nan, device="cuda")
    values = torch.tensor(
        [[3.0, 4.0, torch.inf, torch.nan, -torch.inf]] * 2, device="cuda"
    )
    offsets = torch.tensor(
        [[0, 63, 64, 1 << 29, -1]] * 2, device="cuda", dtype=torch.int32
    )
    positions = torch.tensor([[9, 7], [5, 3]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([[64], [0]], device="cuda", dtype=torch.int32)
    ids, scores = candidates.gather_index_candidates(
        offsets,
        logits,
        positions,
        64,
        lengths=lengths,
        selected_scores=values,
    )
    torch.testing.assert_close(
        ids,
        torch.tensor(
            [[576, 639, -1, -1, -1], [-1] * 5], device="cuda", dtype=torch.int32
        ),
    )
    torch.testing.assert_close(
        scores,
        torch.tensor(
            [[3.0, 4.0, -torch.inf, -torch.inf, -torch.inf], [-torch.inf] * 5],
            device="cuda",
        ),
    )


@requires_cuda
def test_length_aware_helpers_do_not_specialize_on_batch_geometry():
    def run(rows, width):
        logits = torch.randn(rows, width, device="cuda")
        positions = torch.arange(
            (width + 63) // 64, device="cuda", dtype=torch.int32
        ).repeat(rows, 1)
        lengths = torch.full((rows, 1), width, dtype=torch.int32, device="cuda")
        candidates.mask_index_scores(
            logits, positions, lengths, lengths.flatten(), 64, 0, 0, mask_tail=False
        )
        offsets = torch.arange(8, device="cuda", dtype=torch.int32).repeat(rows, 1)
        return candidates.gather_index_candidates(
            offsets,
            logits,
            positions,
            64,
            lengths=lengths,
            selected_scores=logits[:, :8],
        )

    run(4, 256)
    with assert_no_triton_compile(
        candidates._mask_scores, candidates._gather_candidates
    ):
        for rows, width in ((1, 64), (3, 128), (16, 516)):
            run(rows, width)


def _partials(degree, tokens, dim, dtype, strided):
    heads = degree * 2
    outputs = torch.randn(
        degree, tokens, heads, dim * (2 if strided else 1), device="cuda", dtype=dtype
    )
    if strided:
        outputs = outputs[..., ::2]
    lses = torch.randn(degree, tokens, heads * 2, device="cuda")[..., ::2] * 3
    if tokens >= 5:
        lses[0, 0] = torch.inf
        outputs[0, 0] = torch.nan
        lses[:, 1] = -torch.inf
        outputs[:, 1] = torch.nan
        lses[0, 2, 0] = torch.nan
        if degree > 1:
            lses[0, 3] = -1000
            lses[1:, 3] = 1000
            outputs[0, 3] = torch.nan
    return outputs, lses


def _transport_reference(outputs, lses, sink):
    lses = torch.where(torch.isposinf(lses), -torch.inf, lses)
    maximum = lses.max(0).values
    safe = torch.where(torch.isneginf(maximum), 0, maximum)
    mass = (lses - safe).exp()
    denominator = mass.sum(0)
    weights = mass / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    values = torch.where(torch.isneginf(lses)[..., None], 0, outputs.float())
    result = (values * weights[..., None]).sum(0)
    if sink is not None:
        normalizer = torch.where(
            denominator == 0, -torch.inf, maximum + denominator.log()
        )
        factor = torch.where(
            torch.isneginf(normalizer), 0, 1 / (1 + (sink - normalizer).exp())
        )
        result *= factor[..., None]
    return result


@pytest.mark.parametrize("degree", [1, 3, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "dim,strided,use_sink", [(15, True, True), (512, False, False)]
)
@requires_cuda
def test_partial_transport_and_merge(degree, dtype, dim, strided, use_sink):
    torch.manual_seed(819)
    outputs, lses = _partials(degree, 5, dim, dtype, strided)
    sink = torch.randn(degree * 2, device="cuda") if use_sink else None
    sent = [
        dcp.dispatch_dcp_attention_partials(outputs[r], lses[r], degree=degree)
        for r in range(degree)
    ]
    expected = _transport_reference(outputs, lses, sink)
    for owner in range(degree):
        received = torch.stack([payload[owner] for payload in sent])
        local_sink = None if sink is None else sink[owner * 2 : (owner + 1) * 2]
        kwargs = dict(head_dim=dim, partial_dtype=dtype, sink=local_sink)
        actual = dcp.combine_dcp_attention_partials(
            received, output_dtype=torch.float32, **kwargs
        )
        torch.testing.assert_close(
            actual,
            expected[:, owner * 2 : (owner + 1) * 2],
            rtol=3e-6,
            atol=2e-6,
            equal_nan=True,
        )
        rounded = dcp.combine_dcp_attention_partials(
            received, output_dtype=dtype, **kwargs
        )
        torch.testing.assert_close(
            rounded, actual.to(dtype), rtol=0, atol=0, equal_nan=True
        )
        assert rounded.is_contiguous()


@requires_cuda
def test_partial_dispatch_writes_lent_storage():
    torch.manual_seed(820)
    degree, dim = 4, 33
    outputs, lses = _partials(degree, 5, dim, torch.bfloat16, True)
    expected = dcp.dispatch_dcp_attention_partials(outputs[0], lses[0], degree=degree)
    # Lent storage: a view at the start of a larger buffer, stale bytes in it.
    buffer = torch.full((expected.numel() + 64,), -5, device="cuda", dtype=torch.int32)
    lent = buffer[: expected.numel()].view(expected.shape)
    got = dcp.dispatch_dcp_attention_partials(
        outputs[0], lses[0], degree=degree, out=lent
    )
    assert got.data_ptr() == buffer.data_ptr()
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert (buffer[expected.numel() :] == -5).all()
    with pytest.raises(ValueError, match="contiguous int32"):
        dcp.dispatch_dcp_attention_partials(
            outputs[0], lses[0], degree=degree, out=lent[:, :, :1]
        )


@requires_cuda
def test_candidate_and_partial_fusions_do_not_recompile():
    degree, width, topk, dim = 2, 16, 8, 16

    def run(rows):
        records = torch.zeros(
            (degree, rows, width, 2), device="cuda", dtype=torch.int32
        )
        records[..., 0] = torch.arange(width, device="cuda", dtype=torch.int32)
        records[..., 1] = torch.arange(width, device="cuda").float().view(torch.int32)
        selected, counts = exchange.merge_packed_index_candidates(records, topk=topk)
        outputs, lses = _partials(degree, rows, dim, torch.bfloat16, False)
        packed = dcp.dispatch_dcp_attention_partials(outputs[0], lses[0], degree=degree)
        merged = dcp.combine_dcp_attention_partials(
            packed,
            head_dim=dim,
            partial_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            sink=None,
        )
        return records, packed, selected, counts, merged

    run(4)
    with assert_no_triton_compile(
        exchange._extract_rank_major_scores,
        exchange._gather_selected_candidates,
        dcp._dispatch_dcp_attention_partials,
        dcp._combine_dcp_attention_partials,
    ):
        for rows in (1, 3, 16, 33):
            run(rows)
