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

import os
import sys
from dataclasses import replace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.dcp.metadata import (
    CompactDCPLayout,
    CompactDCPMetadata,
    DCPPageTableMetadata,
    PositionPreservingDCPLayout,
    PositionPreservingDCPMetadata,
    refresh_dcp_page_table_metadata,
)
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    max_owned_index_pages,
    owned_history_rows,
    resolve_cache_slots,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("block_size", [64, 128])
def test_layouts_share_placement_and_preserve_partial_tail(device, degree, block_size):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Nonconsecutive ownership IDs; second row includes null/out-of-capacity IDs.
    blocks = torch.tensor([[2, 5, 8], [0, 10, 3]], dtype=torch.int32, device=device)
    lengths = torch.tensor([2 * block_size + 17, 0], dtype=torch.int32, device=device)
    subpages = block_size // 64
    pages = (
        (blocks[..., None] * subpages + torch.arange(subpages, device=device))
        .reshape(2, -1)
        .int()
    )
    token_totals = torch.zeros_like(lengths)
    for rank in range(degree):
        common = dict(virtual_block_count=10, degree=degree, rank=rank, previous=None)
        preserved = refresh_dcp_page_table_metadata(
            page_table=blocks, layout=PositionPreservingDCPLayout(), **common
        )
        compact = refresh_dcp_page_table_metadata(
            page_table=pages,
            layout=CompactDCPLayout(lengths, 64, block_size),
            **common,
        )
        assert isinstance(preserved, PositionPreservingDCPMetadata)
        assert isinstance(compact, CompactDCPMetadata)
        assert isinstance(compact, DCPPageTableMetadata)
        expected_pages = []
        expected_tokens = 0
        for col, block in enumerate([2, 5, 8]):
            owned = (block - 1) % degree == rank
            local_block = (block - 1) // degree + 1
            assert preserved.local_page_table[0, col].item() == (
                local_block if owned else -1
            )
            if owned:
                count = min(block_size, int(lengths[0]) - col * block_size)
                expected_tokens += count
                expected_pages.extend(
                    local_block * subpages + i for i in range((count + 63) // 64)
                )
        assert (
            compact.local_page_table[0, : len(expected_pages)].tolist()
            == expected_pages
        )
        assert not compact.local_page_table[0, len(expected_pages) :].any()
        assert compact.local_seq_lens.tolist() == [expected_tokens, 0]
        assert not compact.local_page_table[1].any()
        assert preserved.local_page_table[1, :2].tolist() == [-1, -1]
        token_totals += compact.local_seq_lens
    torch.testing.assert_close(token_totals, lengths)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_refresh_row_views_and_graph_replay(compact, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    table = torch.tensor(
        [[2, 3, 4], [1, 2, 3], [2, 3, 4]], dtype=torch.int32, device=device
    )
    lengths = torch.tensor([145, 129, 128], dtype=torch.int32, device=device)

    def layout(n):
        return (
            CompactDCPLayout(lengths[:n], 64, 64)
            if compact
            else PositionPreservingDCPLayout()
        )

    common = dict(virtual_block_count=9, degree=2, rank=1)
    metadata = refresh_dcp_page_table_metadata(
        page_table=table, layout=layout(3), previous=None, **common
    )
    pointers = (
        metadata.local_page_table.data_ptr(),
        (metadata.local_seq_lens if compact else metadata.owner_mask).data_ptr(),
    )

    def refresh(n):
        updated = refresh_dcp_page_table_metadata(
            page_table=table[:n],
            layout=layout(n),
            previous=metadata.slice_requests(0, n),
            **common,
        )
        assert updated.local_page_table.data_ptr() == pointers[0]
        assert (
            updated.local_seq_lens if compact else updated.owner_mask
        ).data_ptr() == pointers[1]

    refresh(1)
    refresh(3)
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            refresh(1)
    table[0].copy_(torch.tensor([3, 2, 4], device=device))
    lengths[0] = 65
    if device == "cuda":
        graph.replay()
    else:
        refresh(1)
    if compact:
        assert metadata.local_page_table[0].tolist() == [1, 0, 0]
        assert metadata.local_seq_lens[0].item() == 1
    else:
        assert metadata.local_page_table[0].tolist() == [-1, 1, 2]
        assert metadata.owner_mask[0].tolist() == [False, True, True]


def test_refresh_rejects_layout_topology_and_page_geometry_changes():
    table = torch.tensor([[2, 3]], dtype=torch.int32)
    lengths = torch.tensor([65], dtype=torch.int32)
    layout = CompactDCPLayout(lengths, 64, 128)
    kwargs = dict(
        page_table=table, virtual_block_count=10, degree=2, rank=0, layout=layout
    )
    metadata = refresh_dcp_page_table_metadata(previous=None, **kwargs)
    for changes, message in [
        ({"layout": PositionPreservingDCPLayout()}, "layout changed"),
        ({"rank": 1}, "topology changed"),
        ({"virtual_block_count": 12}, "capacity changed"),
        ({"layout": replace(layout, block_granularity=64)}, "geometry changed"),
        ({"layout": replace(layout, page_size=0)}, "whole kernel pages"),
        ({"layout": replace(layout, seq_lens=torch.ones(2))}, "lengths must match"),
        (
            # A captured graph would keep replaying against the old buffers.
            {"page_table": torch.tensor([[2, 3, 4]], dtype=torch.int32)},
            "buffers do not match",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            refresh_dcp_page_table_metadata(previous=metadata, **(kwargs | changes))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("block_size", [64, 128])
def test_query_visibility_reserve_pages_advance_and_reanchor(
    device, degree, block_size
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Ownership follows physical virtual IDs, not logical page positions.
    blocks = [[5, 2, 8, 1], [2, 5, 1, 8], [0, 0, 0, 0]]
    subpages = block_size // 64
    table = torch.tensor(
        [[b * subpages + s for b in row for s in range(subpages)] for row in blocks],
        dtype=torch.int32,
        device=device,
    )
    visible = torch.tensor([[62, 63, 64, 65]] * 2 + [[0] * 4], device=device).int()
    lengths = visible[:, -1].contiguous()
    for rank in range(degree):
        common = dict(virtual_block_count=10, degree=degree, rank=rank)
        layout = CompactDCPLayout(lengths, 64, block_size, visible)
        metadata = refresh_dcp_page_table_metadata(
            page_table=table, layout=layout, previous=None, **common
        )
        buffers = (
            metadata.local_page_table,
            metadata.page_prefix,
            metadata.local_visible_lens,
            metadata.local_seq_lens,
        )
        pointers = [t.data_ptr() for t in buffers]
        compact_before = metadata.local_page_table.clone()

        def check(endpoints):
            expected = []
            for row, row_blocks in enumerate(blocks):
                # Enumerate visible global tokens independently of prefix arithmetic.
                owned = [
                    (b > 0 and (b - 1) % degree == rank)
                    for b in row_blocks
                    for _ in range(block_size)
                ]
                expected.append([sum(owned[:end]) for end in endpoints[row]])
                pages = [
                    ((b - 1) // degree + 1) * subpages + s
                    for b in row_blocks
                    if b > 0 and (b - 1) % degree == rank
                    for s in range(subpages)
                ]
                assert metadata.local_page_table[row].tolist() == pages + [0] * (
                    table.shape[1] - len(pages)
                )
            assert metadata.local_visible_lens.tolist() == expected
            assert metadata.local_seq_lens.tolist() == [row[-1] for row in expected]
            assert [t.data_ptr() for t in buffers] == pointers

        metadata.refresh_visible_lengths(visible)
        if device == "cuda":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                metadata.refresh_visible_lengths(visible)
        # Advance across kernel and ownership pages; rollback; noncausal draft;
        # exact full-table endpoint; idle; then restore a live sequence.
        for endpoints in (
            [62, 63, 64, 65],
            [126, 127, 128, 129],
            [190, 191, 192, 193],
            [61, 62, 63, 64],
            [129] * 4,
            list(range(4 * block_size - 3, 4 * block_size + 1)),
            [0] * 4,
            [62, 63, 64, 65],
        ):
            expected_endpoints = [endpoints, endpoints, [0] * 4]
            visible.copy_(torch.tensor(expected_endpoints, device=device))
            metadata.local_visible_lens.fill_(-1)
            metadata.local_seq_lens.fill_(-1)
            if device == "cuda":
                graph.replay()
            else:
                metadata.refresh_visible_lengths(visible)
            check(expected_endpoints)
            torch.testing.assert_close(metadata.local_page_table, compact_before)

        # A row view must update the parent's buffers, including the query bounds.
        subset = metadata.slice_requests(1, 2)
        visible[1].zero_()
        subset.refresh_visible_lengths(visible[1:2])
        assert metadata.local_visible_lens[1].tolist() == [0] * 4
        assert metadata.local_seq_lens[1].item() == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("queries", [1, 3])
def test_query_visibility_full_refresh_reuses_storage(device, queries):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    table = torch.tensor([[5, 2, 8, 1]], dtype=torch.int32, device=device)
    visible = torch.tensor(
        [list(range(66 - queries, 66))], dtype=torch.int32, device=device
    )
    lengths = torch.tensor([65], dtype=torch.int32, device=device)
    layout = CompactDCPLayout(lengths, 64, 64, visible)
    common = dict(
        page_table=table, layout=layout, virtual_block_count=10, degree=2, rank=0
    )
    metadata = refresh_dcp_page_table_metadata(previous=None, **common)

    def refresh():
        updated = refresh_dcp_page_table_metadata(previous=metadata, **common)
        assert updated.page_prefix.data_ptr() == metadata.page_prefix.data_ptr()
        assert (
            updated.local_visible_lens.data_ptr()
            == metadata.local_visible_lens.data_ptr()
        )

    refresh()
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            refresh()
    table.copy_(torch.tensor([[2, 5, 1, 8]], device=device))
    if device == "cuda":
        graph.replay()
    else:
        refresh()
    assert metadata.local_page_table.tolist() == [[3, 1, 0, 0]]
    assert metadata.page_prefix.tolist() == [[0, 0, 1, 2, 2]]
    assert metadata.local_visible_lens.tolist() == [[0] * (queries - 1) + [1]]
    assert metadata.local_seq_lens.tolist() == [1]
    with pytest.raises(ValueError, match="visibility buffers changed"):
        refresh_dcp_page_table_metadata(
            previous=metadata,
            **(common | {"layout": replace(layout, visible_lens=None)}),
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("block_size", [64, 128])
def test_verify_query_lengths_cross_ownership_boundaries(device, block_size):
    """A verify window's per-query endpoints, fed through the layout's
    ``visible_lens``, map to the owned-token counts a brute-force scan of the
    cyclic placement gives, across kernel-page and ownership-block edges."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    blocks = torch.tensor([[2, 5, 8], [0, 0, 0]], dtype=torch.int32, device=device)
    subpages = block_size // 64
    pages = (
        blocks[..., None] * subpages + torch.arange(subpages, device=device)
    ).reshape(2, -1)
    offsets = torch.arange(-5, 1, dtype=torch.int32, device=device)
    for length in (1, block_size - 1, block_size + 2, 2 * block_size + 1):
        lens = torch.tensor([length, 0], dtype=torch.int32, device=device)
        causal = (lens[:, None] + offsets).clamp_min(0)
        noncausal = lens[:, None].expand(-1, 6).contiguous()
        for rank in range(4):
            expected = [
                sum(
                    (int(blocks[0, t // block_size]) - 1) % 4 == rank
                    for t in range(max(0, length - 5 + i))
                )
                for i in range(6)
            ]
            for visible, rows in ((causal, expected), (noncausal, [expected[-1]] * 6)):
                metadata = refresh_dcp_page_table_metadata(
                    page_table=pages,
                    virtual_block_count=10,
                    degree=4,
                    rank=rank,
                    layout=CompactDCPLayout(lens, 64, block_size, visible),
                    previous=None,
                )
                assert metadata.local_visible_lens.tolist() == [rows, [0] * 6]
                assert metadata.local_seq_lens.tolist() == [expected[-1], 0]


def _skewed_index_table(seq_lens, *, degree, subpages, virtual_block_count, seed):
    """A host kernel-page table dealing each request distinct blocks, about
    70% of the first request's from rank 0, with stale pages past every
    length and a wider mirror than the lengths need."""
    generator = torch.Generator().manual_seed(seed)
    page = 64
    counts = [-(-length // (page * subpages)) for length in seq_lens]
    # Block ``v`` lives on rank ``(v - 1) % degree``.
    per_rank = (virtual_block_count - 1) // degree
    pools = [
        (torch.randperm(per_rank, generator=generator) * degree + rank + 1).tolist()
        for rank in range(degree)
    ]
    columns = max(counts) * subpages + 5
    table = torch.randint(
        1,
        virtual_block_count * subpages,
        (len(seq_lens), columns),
        generator=generator,
    )
    for request, count in enumerate(counts):
        blocks = []
        for index in range(count):
            if request == 0 and index < count * 7 // 10:
                owner = 0
            else:
                live = [rank for rank in range(degree) if pools[rank]]
                owner = live[int(torch.randint(len(live), (1,), generator=generator))]
            blocks.append(pools[owner].pop())
        pages = [block * subpages + sub for block in blocks for sub in range(subpages)]
        table[request, : len(pages)] = torch.tensor(pages)
    return table.to(torch.int32)


def _device_owned_pages(table, causal, request, placement, page):
    """The pages a query row keeps on its rank: the DCP indexer's local table
    (``resolve_cache_slots`` over the table's slots, as ``_local_index_table``
    builds it) filtered as the candidate compaction filters it."""
    slots, owned = resolve_cache_slots(table[request].long() * page, placement)
    local = torch.where(owned, slots // page, -1)
    starts = torch.arange(table.shape[1]) * page
    return int(((local > 0) & (starts < causal)).sum())


@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("holes", [False, True])
def test_max_owned_index_pages_bounds_every_rank(degree, block_size, holes):
    """The host bound covers every query row on every rank, is reached by
    some request's last row, and ignores pages without an owner (the null
    block, a block past the capacity) instead of refusing them."""
    page, virtual_block_count = 64, 1 + 128 * degree
    subpages = block_size // page
    seq_lens = [5000, 1, 64 * 7 + 3, 130]
    table = _skewed_index_table(
        seq_lens,
        degree=degree,
        subpages=subpages,
        virtual_block_count=virtual_block_count,
        seed=degree * block_size + holes,
    )
    if holes:
        # A null page and a page past the capacity inside request 0's length.
        table[0, 3] = 0
        table[0, 9] = virtual_block_count * subpages
        with pytest.raises(ValueError, match="holes"):
            owned_history_rows(
                table,
                torch.tensor(seq_lens),
                page_size=page,
                placement=CachePlacement(block_size, virtual_block_count, (0,), 0),
            )
    group = tuple(range(degree))
    bounds, reached = set(), 0
    for rank in group:
        placement = CachePlacement(block_size, virtual_block_count, group, rank)
        bounds.add(
            max_owned_index_pages(
                table,
                torch.tensor(seq_lens, dtype=torch.int32),
                page_size=page,
                placement=placement,
            )
        )
        for request, length in enumerate(seq_lens):
            # The rows of a chunk ending at the request's length.
            for causal in sorted({1, max(1, length // 2), max(1, length - 1), length}):
                owned = _device_owned_pages(table, causal, request, placement, page)
                assert owned <= min(bounds)
            reached = max(
                reached, _device_owned_pages(table, length, request, placement, page)
            )
    # Every rank computes the same bound, and it is tight.
    assert len(bounds) == 1
    assert bounds == {reached}
    # One owner ([B, 0, ..., 0], a capture dummy's): all pages of the longest request.
    one_owner = torch.full_like(table, subpages * (1 + degree))
    assert max_owned_index_pages(
        one_owner,
        torch.tensor(seq_lens),
        page_size=page,
        placement=CachePlacement(block_size, virtual_block_count, group, 0),
    ) == -(-max(seq_lens) // page)


def test_max_owned_index_pages_of_no_pages_is_zero():
    placement = CachePlacement(128, 9, (0, 1), 1)
    for table, lengths in (
        (torch.zeros((0, 4), dtype=torch.int32), torch.zeros(0, dtype=torch.int32)),
        (torch.tensor([[2, 3]], dtype=torch.int32), torch.tensor([0])),
        (torch.tensor([[0, 0]], dtype=torch.int32), torch.tensor([100])),
    ):
        assert (
            max_owned_index_pages(table, lengths, page_size=64, placement=placement)
            == 0
        )
    with pytest.raises(ValueError, match="rows for"):
        max_owned_index_pages(
            torch.zeros((2, 4), dtype=torch.int32),
            torch.tensor([1]),
            page_size=64,
            placement=placement,
        )


def test_write_slots_are_reused_per_forward_and_slots(monkeypatch):
    """Every layer's write at the forward's slots reuses one resolution; other
    slots, rewritten slots and another placement resolve again, each with
    ``resolve_cache_slots``'s result. A graph capture resolves the forward's
    slots itself rather than reusing an eager resolution, and an eager write
    never reuses the capture's."""
    from types import SimpleNamespace

    import tokenspeed.runtime.layers.attention.dcp.placement as placement_mod

    capturing = [False]
    monkeypatch.setattr(
        placement_mod, "current_platform", lambda: SimpleNamespace(is_amd=True)
    )
    monkeypatch.setattr(
        placement_mod.torch.cuda, "is_current_stream_capturing", lambda: capturing[0]
    )
    # Inside the capacity, so ranks and rewritten slots resolve differently.
    placement = CachePlacement(128, 33, (0, 1, 2, 3), 2)
    # The forward's SparseTopKShare field.
    share = SimpleNamespace(write_slots=[])
    loc = torch.arange(200, 264, dtype=torch.int64) * 7

    first = placement_mod.resolve_write_slots(loc, placement, share)
    for view in (loc, loc[:64], loc.view(64)):
        again = placement_mod.resolve_write_slots(view, placement, share)
        assert again[0] is first[0] and again[1] is first[1]
    expected = resolve_cache_slots(loc, placement)
    assert torch.equal(first[0], expected[0]) and torch.equal(first[1], expected[1])

    other = replace(placement, rank=1)
    for view, where in ((loc[:32], placement), (loc, other)):
        result = placement_mod.resolve_write_slots(view, where, share)
        expected = resolve_cache_slots(view, where)
        assert torch.equal(result[0], expected[0])
        assert torch.equal(result[1], expected[1])
    loc.add_(128)
    result = placement_mod.resolve_write_slots(loc, placement, share)
    expected = resolve_cache_slots(loc, placement)
    assert torch.equal(result[0], expected[0]) and torch.equal(result[1], expected[1])

    # SparseTopKShare.clear at the next forward's metadata build.
    share.write_slots = []
    eager = placement_mod.resolve_write_slots(loc, placement, share)
    capturing[0] = True
    captured = placement_mod.resolve_write_slots(loc, placement, share)
    assert captured[0] is not eager[0]
    again = placement_mod.resolve_write_slots(loc, placement, share)
    assert again[0] is captured[0]
    capturing[0] = False
    after = placement_mod.resolve_write_slots(loc, placement, share)
    assert after[0] is eager[0]

    # Slots made under inference mode carry no version, so they are not kept.
    share.write_slots = []
    with torch.inference_mode():
        inference_loc = loc.clone()
    for _ in range(2):
        result = placement_mod.resolve_write_slots(inference_loc, placement, share)
        expected = resolve_cache_slots(inference_loc, placement)
        assert torch.equal(result[0], expected[0])
        assert torch.equal(result[1], expected[1])
    assert share.write_slots == []
