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

from dataclasses import replace

import pytest
import torch

from tokenspeed.runtime.layers.attention.dcp.metadata import (
    CompactDCPLayout,
    CompactDCPMetadata,
    DCPPageTableMetadata,
    PositionPreservingDCPLayout,
    PositionPreservingDCPMetadata,
    refresh_dcp_page_table_metadata,
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
