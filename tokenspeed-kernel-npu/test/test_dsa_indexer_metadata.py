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

"""Replay the grouped Indexer inputs across a DCP ownership boundary."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed.runtime.layers.attention.backends.paged.ascend_dsa import (
    _AscendDSAContextParallel,
)


@pytest.mark.parametrize("rank", range(8))
def test_grouped_metadata_graph_reads_changed_and_restored_causal_ends(rank):
    # Request pages never alias. Their ninth pages belong to different ranks.
    pages = [list(range(1, 9)) + [17], list(range(9, 17)) + [18]]
    table = torch.tensor(pages, dtype=torch.int32, device="npu").repeat_interleave(8, 0)
    lengths = torch.arange(1017, 1025, dtype=torch.int32, device="npu").repeat(2)
    q_ends = torch.arange(1, 17, dtype=torch.int32, device="npu")
    dcp = _AscendDSAContextParallel(
        degree=8,
        rank=rank,
        ranks=tuple(range(8)),
        virtual_block_count=19,
    )
    dcp.allocate_decode_buffers(16, 9, "npu")

    def prepare():
        dcp.prepare_metadata(
            seq_lens=lengths,
            page_table=table,
            page_size=128,
            block_granularity=128,
            init_tokens=2,
            local_tokens=8,
            query_width=1,
            query_group_size=8,
            slots=None,
        )
        return dcp.indexer_metadata(
            q_ends, dcp.metadata(start=0, end=16), query_group_size=8
        )

    for _ in range(3):
        dcp.clear_forward_metadata()
        prepare()
    torch.npu.synchronize()
    dcp.clear_forward_metadata()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        grouped = prepare()
    pointers = [
        t.data_ptr()
        for t in (grouped.page_table, grouped.seq_lens, grouped.visible_lengths)
    ]
    for delta in (0, 8, 0):
        visible = list(range(1017 + delta, 1025 + delta)) * 2
        lengths.copy_(torch.tensor(visible, dtype=torch.int32, device="npu"))
        graph.replay()
        torch.npu.synchronize()

        def owned(request, start, end):
            return sum(
                (pages[request][p // 128] - 1) % 8 == rank for p in range(start, end)
            )

        local = [owned(i // 8, 0, n) for i, n in enumerate(visible)]
        windows = [owned(i // 8, max(n - 8, 0), n) for i, n in enumerate(visible)]
        assert grouped.q_ends.cpu().tolist() == [8, 16]
        assert grouped.visible_lengths.cpu().tolist() == local
        assert grouped.seq_lens.cpu().tolist() == local[7::8]
        assert grouped.local_counts.cpu().tolist() == windows
        assert grouped.init_counts.cpu().tolist() == [owned(i, 0, 2) for i in range(2)]
        assert [
            t.data_ptr()
            for t in (grouped.page_table, grouped.seq_lens, grouped.visible_lengths)
        ] == pointers
    graph.reset()


def _cpu_metadata(
    pages, lengths, slots, *, width, subpages, degree, rank, initial, window, capacity
):
    """Independent integer oracle using logical page/token interval intersections."""
    columns = len(pages[0])
    tables, local, starts, windows = [], [], [], []
    for source, final in zip(pages, lengths):
        owned = [
            (col, page)
            for col, page in enumerate(source)
            if 0 < page // subpages < capacity
            and (page // subpages - 1) % degree == rank
        ]
        packed = [
            ((page // subpages - 1) // degree + 1) * subpages + page % subpages
            for _, page in owned
        ]
        for offset in range(width):
            end = max(1, final + offset + 1 - width) if width > 1 else final

            def count(start, stop):
                return sum(
                    max(0, min(stop, (col + 1) * 128) - max(start, col * 128))
                    for col, _ in owned
                )

            tables.append(packed + [0] * (columns - len(packed)))
            local.append(count(0, end))
            starts.append(count(0, min(end, initial)))
            windows.append(count(max(0, end - window), end))
    mapped = []
    block_rows = subpages * 128
    for slot in slots:
        block = slot // block_rows - 1
        mapped.append(
            (block // degree + 1) * block_rows + slot % block_rows
            if block_rows <= slot < capacity * block_rows and block % degree == rank
            else -1
        )
    last = slice(width - 1, None, width)
    return [
        tables,
        local,
        starts,
        windows,
        list(range(1, len(local) + 1)),
        tables[last],
        local[last],
        starts[last],
        list(range(width, len(local) + 1, width)),
        mapped,
    ]


@pytest.mark.parametrize(
    "batch,width,columns",
    [(1, 8, 31), (2, 8, 513), (4, 8, 1024), (8, 8, 1024), (64, 8, 1024), (8, 1, 2057)],
)
@pytest.mark.parametrize("subpages", [1, 8])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "degree,rank", [(1, 0), (2, 1), (4, 3), *((8, rank) for rank in range(8)), (16, 15)]
)
def test_fused_dsa_cp_metadata_exact_replay(
    batch, width, columns, subpages, slot_dtype, degree, rank
):
    capacity = 4097
    gen = torch.Generator().manual_seed(173 + batch + columns)
    host_pages = torch.randint(
        -1,
        (capacity + 1) * subpages,
        (batch, columns),
        generator=gen,
        dtype=torch.int32,
    )
    # Check source row strides, null/foreign pages, reserve pages and suffix clearing.
    carrier = torch.zeros((batch, columns + 3), dtype=torch.int32, device="npu")
    table = carrier[:, :columns]
    table.copy_(host_pages)
    host_lengths = torch.tensor(
        [
            min(columns * 128, x)
            for x in ([0, 1, 127, 128, 129, 511, 1025, 131072] * ((batch + 7) // 8))[
                :batch
            ]
        ],
        dtype=torch.int32,
    )
    lengths = host_lengths.npu()
    raw = torch.arange(257, dtype=slot_dtype) * subpages * 128
    raw[:5] = torch.tensor(
        [-1, 0, subpages * 128 - 1, capacity * subpages * 128, 2**30], dtype=slot_dtype
    )
    if slot_dtype == torch.int64:
        raw[-1] = 2**60
    slots = raw.npu()
    rows = batch * width
    dcp = _AscendDSAContextParallel(
        degree=degree,
        rank=rank,
        ranks=tuple(range(degree)),
        virtual_block_count=capacity,
    )
    dcp.allocate_decode_buffers(max(rows, slots.numel()), columns, "npu")

    def prepare():
        dcp.clear_forward_metadata()
        for _ in range(2):
            dcp.prepare_metadata(
                seq_lens=lengths,
                page_table=table,
                page_size=128,
                block_granularity=subpages * 128,
                init_tokens=131,
                local_tokens=129,
                query_width=width,
                query_group_size=width,
                slots=slots,
            )
        query = dcp.metadata(start=0, end=rows)
        index = dcp.indexer_metadata(dcp.query_ends, query, query_group_size=width)
        return [
            query.page_table,
            query.seq_lens,
            query.init_counts,
            query.local_counts,
            dcp.query_ends,
            index.page_table,
            index.seq_lens,
            index.init_counts,
            index.q_ends,
            dcp.write_slots,
        ]

    for _ in range(2):
        prepare()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = prepare()
    pointers = [t.data_ptr() for t in outputs]
    for changed in (False, True, False):
        pages = host_pages.flip(1).contiguous() if changed else host_pages
        ends = (host_lengths + 13).clamp_max(columns * 128) if changed else host_lengths
        source_slots = raw.flip(0).contiguous() if changed else raw
        table.copy_(pages)
        lengths.copy_(ends)
        slots.copy_(source_slots)
        graph.replay()
        torch.npu.synchronize()
        expected = _cpu_metadata(
            pages.tolist(),
            ends.tolist(),
            source_slots.tolist(),
            width=width,
            subpages=subpages,
            degree=degree,
            rank=rank,
            initial=131,
            window=129,
            capacity=capacity,
        )
        for actual, wanted in zip(outputs, expected):
            torch.testing.assert_close(
                actual.cpu(), torch.tensor(wanted, dtype=actual.dtype), rtol=0, atol=0
            )
        assert [t.data_ptr() for t in outputs] == pointers
    graph.reset()


def test_metadata_temporaries_survive_foreign_stream_allocator_reuse():
    pages = (torch.arange(8 * 513).reshape(8, 513) % 17).int()
    lengths = torch.full((8,), 65000, dtype=torch.int32)
    slots = torch.arange(128, 136, dtype=torch.int64)
    expected = _cpu_metadata(
        pages.tolist(),
        lengths.tolist(),
        slots.tolist(),
        width=1,
        subpages=1,
        degree=8,
        rank=0,
        initial=131,
        window=129,
        capacity=17,
    )
    dcp = _AscendDSAContextParallel(
        degree=8, rank=0, ranks=tuple(range(8)), virtual_block_count=17
    )
    dcp.allocate_decode_buffers(8, 513, "npu")
    producer, consumer = torch.npu.current_stream(), torch.npu.Stream()
    for _ in range(16):
        table, ends, locations = pages.npu(), lengths.npu(), slots.npu()
        with torch.npu.stream(consumer):
            consumer.wait_stream(producer)
            dcp.prepare_metadata(
                seq_lens=ends,
                page_table=table,
                page_size=128,
                block_granularity=128,
                init_tokens=131,
                local_tokens=129,
                query_width=1,
                query_group_size=1,
                slots=locations,
            )
            local = dcp.metadata(start=0, end=8)
            index = dcp.indexer_metadata(dcp.query_ends, local, query_group_size=1)
            outputs = [
                local.page_table,
                local.seq_lens,
                local.init_counts,
                local.local_counts,
                dcp.query_ends,
                index.page_table,
                index.seq_lens,
                index.init_counts,
                index.q_ends,
                dcp.write_slots,
            ]
        dcp.clear_forward_metadata()
        del table, ends, locations
        # Reusing the producer's size classes must not overwrite live consumer inputs.
        trash = [
            torch.full((8, 513), 7, dtype=torch.int32, device="npu") for _ in range(8)
        ]
        trash += [
            torch.full((8,), 7, dtype=torch.int64, device="npu") for _ in range(8)
        ]
        consumer.synchronize()
        for actual, wanted in zip(outputs, expected):
            torch.testing.assert_close(
                actual.cpu(), torch.tensor(wanted, dtype=actual.dtype), rtol=0, atol=0
            )
        del trash
