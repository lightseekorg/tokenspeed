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

"""The MLA prefix-chunk gather plan: the host's per-block segments, the
per-row layout they expand to, and the all-gather output rows every rank
reads back."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    prefix_chunk_segments,
)

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    ),
]


def _random_layout(
    generator: torch.Generator,
    *,
    prefix_lens: list[int],
    granularity: int,
    virtual_block_count: int,
) -> tuple[torch.Tensor, list[list[int]]]:
    """A host block table dealing each request distinct random blocks (``-1``
    ragged padding past them, as the bridge pads), and every request's
    virtual slot per prefix position."""
    blocks_needed = [-(-length // granularity) for length in prefix_lens]
    blocks = (
        torch.randperm(virtual_block_count - 1, generator=generator)[
            : sum(blocks_needed)
        ]
        + 1
    ).tolist()
    columns = max(blocks_needed) + 3  # the table may be wider
    table = torch.full((len(prefix_lens), columns), -1, dtype=torch.int32)
    slots = []
    for request, (length, count) in enumerate(zip(prefix_lens, blocks_needed)):
        mine, blocks = blocks[:count], blocks[count:]
        table[request, :count] = torch.tensor(mine, dtype=torch.int32)
        slots.append(
            [
                mine[p // granularity] * granularity + p % granularity
                for p in range(length)
            ]
        )
    return table, slots


def _chunk_slots(slots: list[list[int]], chunk_len: int) -> list[list[int]]:
    """``chunk.get_chunks_paged``'s loop order: per loop, request-major."""
    num_chunks = -(-max(len(s) for s in slots) // chunk_len)
    return [
        [slot for s in slots for slot in s[j * chunk_len : (j + 1) * chunk_len]]
        for j in range(num_chunks)
    ]


# The virtual block ids every layout draws from and every placement owns.
VIRTUAL_BLOCK_COUNT = 4096


def _placement(granularity: int, degree: int, rank: int) -> CachePlacement:
    return CachePlacement(
        block_granularity=granularity,
        virtual_block_count=VIRTUAL_BLOCK_COUNT,
        group=tuple(range(degree)),
        rank=rank,
    )


CASES = [
    # prefix_lens, granularity, chunk_len, degree
    ([300], 128, 128, 4),
    ([1000, 37, 513], 128, 100, 4),
    ([5000, 4097], 128, 2048, 16),
    ([129, 0, 64], 64, 64, 2),
    ([7], 4, 3, 2),
    # More loops than one window covers.
    ([5000, 4097], 128, 512, 4),
    # Loops shorter than a block (several loops cut one block), an odd
    # degree, prefixes ending mid-block, five windows.
    ([1000, 333, 0, 77], 128, 50, 3),
]


@pytest.mark.parametrize("prefix_lens,granularity,chunk_len,degree", CASES)
def test_prefix_chunk_segments_match_per_row_owners_and_slots(
    prefix_lens, granularity, chunk_len, degree
):
    """Expanded row by row, the segments name every loop row's owner and
    the owner's physical slot, in the loops' row order, and never span two
    blocks."""
    table, slots = _random_layout(
        torch.Generator().manual_seed(7),
        prefix_lens=prefix_lens,
        granularity=granularity,
        virtual_block_count=VIRTUAL_BLOCK_COUNT,
    )
    chunks = _chunk_slots(slots, chunk_len)
    segments = prefix_chunk_segments(
        table,
        torch.tensor(prefix_lens),
        chunk_len=chunk_len,
        num_chunks=len(chunks),
        placement=_placement(granularity, degree, 0),
    )
    assert segments.loop_rows.tolist() == [len(chunk) for chunk in chunks]
    assert (segments.rows > 0).all() and (segments.rows <= granularity).all()
    owners, local_slots = [], []
    for loop, row, rows, owner, local in zip(
        segments.loop.tolist(),
        segments.row.tolist(),
        segments.rows.tolist(),
        segments.owner.tolist(),
        segments.local_slot.tolist(),
    ):
        # Segments tile the loops' rows in order.
        assert row == len(owners)
        owners.extend((loop, owner) for _ in range(rows))
        local_slots.extend(range(local, local + rows))
    expected_owners, expected_slots = [], []
    for loop, chunk in enumerate(chunks):
        for slot in chunk:
            block = slot // granularity
            expected_owners.append((loop, (block - 1) % degree))
            expected_slots.append(
                ((block - 1) // degree + 1) * granularity + slot % granularity
            )
    assert owners == expected_owners
    assert local_slots == expected_slots


def test_prefix_chunk_segments_refuse_a_hole_or_a_short_table():
    placement = CachePlacement(
        block_granularity=4, virtual_block_count=16, group=(0, 1), rank=0
    )
    for table in ([[1, 0, 3]], [[1, -1, 3]]):
        # A hole (0) or ragged padding (-1) below the prefix has no owner.
        with pytest.raises(ValueError, match="holes"):
            prefix_chunk_segments(
                torch.tensor(table, dtype=torch.int32),
                torch.tensor([6]),
                chunk_len=6,
                num_chunks=1,
                placement=placement,
            )
    with pytest.raises(ValueError, match="holes"):
        # A block past the virtual capacity has no owner either.
        prefix_chunk_segments(
            torch.tensor([[1, 16]], dtype=torch.int32),
            torch.tensor([6]),
            chunk_len=6,
            num_chunks=1,
            placement=placement,
        )
    with pytest.raises(ValueError, match="does not cover"):
        prefix_chunk_segments(
            torch.tensor([[1, 2]], dtype=torch.int32),
            torch.tensor([9]),
            chunk_len=9,
            num_chunks=1,
            placement=placement,
        )


def _per_row_windows(
    chunks: list[list[int]], *, granularity: int, degree: int, rank: int
):
    """The plan of every window, row by row: each owner's rows go out in
    order, rank-major, every owner padded to the largest count; rank
    ``rank`` sends the physical slots of its rows, then the null slot 0."""
    from tokenspeed.runtime.layers.attention.dcp.cache import (
        PREFIX_HISTORY_WINDOW_LOOPS,
    )

    windows = []
    for first in range(0, len(chunks), PREFIX_HISTORY_WINDOW_LOOPS):
        loops = chunks[first : first + PREFIX_HISTORY_WINDOW_LOOPS]
        slots = [slot for loop in loops for slot in loop]
        owners = [(slot // granularity - 1) % degree for slot in slots]
        counts = tuple(owners.count(k) for k in range(degree))
        send_rows = max(counts)
        seen = [0] * degree
        gathered_index, fetch = [], [0] * send_rows
        for slot, owner in zip(slots, owners):
            gathered_index.append(owner * send_rows + seen[owner])
            if owner == rank:
                fetch[seen[owner]] = (
                    (slot // granularity - 1) // degree + 1
                ) * granularity + slot % granularity
            seen[owner] += 1
        bounds = [0]
        for loop in loops:
            bounds.append(bounds[-1] + len(loop))
        windows.append((counts, send_rows, fetch, gathered_index, tuple(bounds)))
    return windows


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prefix_lens,granularity,chunk_len,degree", CASES)
def test_the_plan_matches_the_per_row_layout(
    device, prefix_lens, granularity, chunk_len, degree
):
    """Every rank's per-block plan expands to exactly the per-row layout:
    the same send slots, tail included, and the same all-gather row for
    every loop row, window by window."""
    from tokenspeed.runtime.layers.attention.dcp.cache import (
        plan_prefix_history_gather,
    )

    table, slots = _random_layout(
        torch.Generator().manual_seed(11),
        prefix_lens=prefix_lens,
        granularity=granularity,
        virtual_block_count=VIRTUAL_BLOCK_COUNT,
    )
    chunks = _chunk_slots(slots, chunk_len)
    loops = [torch.tensor(chunk, dtype=torch.int32, device=device) for chunk in chunks]
    for rank in sorted({0, degree // 2, degree - 1}):
        placement = _placement(granularity, degree, rank)
        windows = plan_prefix_history_gather(
            loops,
            block_table_cpu=table,
            prefix_lens_cpu=torch.tensor(prefix_lens),
            chunk_len=chunk_len,
            placement=placement,
        )
        expected = _per_row_windows(
            chunks, granularity=granularity, degree=degree, rank=rank
        )
        assert len(windows) == len(expected)
        for window, (counts, send_rows, fetch, gathered_index, bounds) in zip(
            windows, expected
        ):
            assert window.group == placement.group and window.rank == rank
            assert window.owned_rows_per_rank == counts
            assert window.send_rows == send_rows
            assert window.loop_bounds == bounds
            assert window.fetch_slots.dtype == torch.int32
            assert window.fetch_slots.tolist() == fetch
            assert window.gathered_index.dtype == torch.int32
            assert window.gathered_index.tolist() == gathered_index


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ratio", [2, 4])
def test_the_plan_matches_the_loops_cut_from_the_expanded_table(monkeypatch, ratio):
    """The plan takes owners and send slots from the host block table and
    only counts the loops; the loops are cut on the device from the kernel
    page table the router expands from that same table. With kernel pages
    smaller than a block, ragged ``-1`` padding and a ``0`` hole past a
    prefix, every rank's plan equals the per-row layout derived from the
    loops' own slots."""
    from tokenspeed.runtime.layers.attention import chunk
    from tokenspeed.runtime.layers.attention.backends.paged.group_tables import (
        GroupTableSpec,
        GroupTableStacks,
    )
    from tokenspeed.runtime.layers.attention.dcp.cache import (
        plan_prefix_history_gather,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    granularity, degree = 8, 4
    # An empty prefix, one ending on a block bound, two ending mid-block.
    prefix_lens = [53, 0, 40, 17]
    bs = len(prefix_lens)
    # Five rows of every request a loop: loops cut blocks; eleven loops,
    # three windows.
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 5 * bs)
    monkeypatch.setitem(global_server_args_dict, "mla_chunk_multiplier", 1)
    table, _ = _random_layout(
        torch.Generator().manual_seed(3),
        prefix_lens=prefix_lens,
        granularity=granularity,
        virtual_block_count=VIRTUAL_BLOCK_COUNT,
    )
    # A retracted tail: a hole right past two prefixes' blocks.
    for request in (0, 3):
        table[request, -(-prefix_lens[request] // granularity)] = 0
    stacks = GroupTableStacks(
        [
            GroupTableSpec(
                group_id="full",
                block_granularity=granularity,
                kernel_page_size=granularity // ratio,
                max_num_pages=table.shape[1] * ratio,
            )
        ],
        max_bs=bs,
        max_tokens_per_req=1,
        max_extend_tokens=0,
        device="cuda",
    )
    stacks.fill(bs, bs, {"full": table.cuda()})
    prefix = torch.tensor(prefix_lens, dtype=torch.int32)
    _, loops, _, _, _ = chunk.build_chunked_prefill_metadata_arrays(
        prefix.cuda(), prefix, stacks.table("full", bs), granularity // ratio
    )
    chunks = [loop.tolist() for loop in loops]
    assert len(chunks) == 11
    for rank in range(degree):
        windows = plan_prefix_history_gather(
            loops,
            block_table_cpu=table,
            prefix_lens_cpu=prefix,
            chunk_len=chunk.prefix_chunk_len(bs),
            placement=_placement(granularity, degree, rank),
        )
        expected = _per_row_windows(
            chunks, granularity=granularity, degree=degree, rank=rank
        )
        assert len(windows) == len(expected) == 3
        for window, (counts, send_rows, fetch, gathered_index, bounds) in zip(
            windows, expected
        ):
            assert window.owned_rows_per_rank == counts
            assert window.send_rows == send_rows
            assert window.loop_bounds == bounds
            assert window.fetch_slots.tolist() == fetch
            assert window.gathered_index.tolist() == gathered_index


def test_the_plan_refuses_loops_the_block_table_does_not_cut():
    from tokenspeed.runtime.layers.attention.dcp.cache import (
        plan_prefix_history_gather,
    )

    placement = _placement(4, 2, 0)
    with pytest.raises(ValueError, match="rows are not"):
        plan_prefix_history_gather(
            [torch.arange(4, 9, dtype=torch.int32)],
            block_table_cpu=torch.tensor([[1, 2]], dtype=torch.int32),
            prefix_lens_cpu=torch.tensor([6]),
            chunk_len=6,
            placement=placement,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prefix_lens,granularity,chunk_len,degree", CASES)
def test_every_rank_reads_the_position_order_rows(
    device, prefix_lens, granularity, chunk_len, degree
):
    """Emulate each window's all-gather over ``degree`` owners: every rank's
    send rows come from its own physical cache, and every rank's plan must
    read the rank-major, padded output back as each loop's rows, in order."""
    from tokenspeed.runtime.layers.attention.dcp.cache import (
        PREFIX_HISTORY_WINDOW_LOOPS,
        plan_prefix_history_gather,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import (
        resolve_cache_slots,
    )

    generator = torch.Generator().manual_seed(11)
    rows = granularity
    table, slots = _random_layout(
        generator,
        prefix_lens=prefix_lens,
        granularity=granularity,
        virtual_block_count=VIRTUAL_BLOCK_COUNT,
    )
    chunks = [
        torch.tensor(chunk, dtype=torch.int32, device=device)
        for chunk in _chunk_slots(slots, chunk_len)
    ]
    placements = [_placement(rows, degree, rank) for rank in range(degree)]
    # Every virtual slot's payload is distinct; rank r stores the rows it
    # owns at its local slots and garbage everywhere else.
    width = 8
    local_rows = ((VIRTUAL_BLOCK_COUNT - 2) // degree + 2) * rows
    caches = [
        torch.full((local_rows, width), -1, dtype=torch.int32, device=device)
        for _ in range(degree)
    ]
    every_slot = torch.cat(chunks).to(torch.int64)
    payload = every_slot.unsqueeze(1) * width + torch.arange(width, device=device)
    for placement, cache in zip(placements, caches):
        local, owned = resolve_cache_slots(every_slot, placement)
        cache[local[owned]] = payload[owned].to(torch.int32)

    plans = [
        plan_prefix_history_gather(
            chunks,
            block_table_cpu=table,
            prefix_lens_cpu=torch.tensor(prefix_lens),
            chunk_len=chunk_len,
            placement=placement,
        )
        for placement in placements
    ]
    expected_windows = -(-len(chunks) // PREFIX_HISTORY_WINDOW_LOOPS)
    assert all(len(plan) == expected_windows for plan in plans)
    loop = 0
    for windows in zip(*plans):
        send_rows = windows[0].send_rows
        assert all(w.send_rows == send_rows for w in windows)
        assert all(w.loop_bounds == windows[0].loop_bounds for w in windows)
        gathered = torch.full(
            (degree * send_rows, width), -2, dtype=torch.int32, device=device
        )
        for rank, (window, cache) in enumerate(zip(windows, caches)):
            sent = cache.index_select(0, window.fetch_slots)
            gathered[rank * send_rows : rank * send_rows + sent.shape[0]] = sent
        bounds = windows[0].loop_bounds
        for start, end in zip(bounds[:-1], bounds[1:]):
            expected = chunks[loop].to(torch.int64).unsqueeze(1) * width + torch.arange(
                width, device=device
            )
            for window in windows:
                assert window.gathered_index.dtype == torch.int32
                index = window.gathered_index[start:end]
                torch.testing.assert_close(
                    gathered.index_select(0, index).to(torch.int64), expected
                )
            loop += 1
    assert loop == len(chunks)


class _StubLatentPool:
    """Per-layer ``[slots, 1, width]`` int32 latent planes, ``None`` for the
    layers without one (another history group's); reads are plain gathers
    split in two halves."""

    def __init__(self, planes: list[torch.Tensor | None]):
        self.kv_buffer = planes

    def get_key_buffer(self, layer_id):
        return self.kv_buffer[layer_id]

    def history_group_by_layer(self):
        return {
            i: "latent" if plane is not None else "other"
            for i, plane in enumerate(self.kv_buffer)
        }

    def read_latent_rows(self, rows, loc, dst_dtype):
        picked = rows.index_select(0, loc.to(torch.int64))
        half = picked.shape[-1] // 2
        return picked[..., :half], picked[..., half:]


class _RecordingBackend:
    """Two emulated owners: rank 1 records what it sends; rank 0 fills each
    gather from its own rows and rank 1's recorded ones, in call order."""

    def __init__(self, rank: int, peer_sends: list[torch.Tensor] | None):
        self.rank = rank
        self.peer_sends = peer_sends
        self.sends: list[torch.Tensor] = []

    def all_gather_single_async(self, gathered, send, group):
        from types import SimpleNamespace

        assert group == (0, 1)
        rows = send.shape[0]
        self.sends.append(send.clone())
        if self.rank == 1:
            gathered.fill_(0xFF)
        else:
            gathered[:rows] = send
            gathered[rows:] = self.peer_sends[len(self.sends) - 1]
        return SimpleNamespace(wait=lambda: None)


@pytest.mark.parametrize("device", DEVICES)
def test_read_prefix_history_prefetches_one_window_ahead(device):
    """Every layer's loops read their rows back through windows that the
    previous window -- of this layer, else of the previous latent layer --
    already launched; the gathers run in the same order on every rank."""
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.dcp.cache import PrefixHistoryGather
    from tokenspeed.runtime.layers.attention.dcp.placement import (
        resolve_cache_slots,
    )

    degree, rows, chunk_len = 2, 128, 512
    prefix_lens = [5000, 300]
    virtual_block_count = 512
    table, slots = _random_layout(
        torch.Generator().manual_seed(5),
        prefix_lens=prefix_lens,
        granularity=rows,
        virtual_block_count=virtual_block_count,
    )
    chunks = [
        torch.tensor(chunk, dtype=torch.int32, device=device)
        for chunk in _chunk_slots(slots, chunk_len)
    ]
    every_slot = torch.cat(chunks).to(torch.int64)
    # A hybrid stack: latent planes on layers 1, 3 and 4 only.
    latent_layers = [1, 3, 4]
    width = 8
    local_rows = ((virtual_block_count - 2) // degree + 2) * rows

    def payload(layer_id, slot):
        return (
            layer_id * 1_000_000
            + slot.to(torch.int64).unsqueeze(1) * width
            + torch.arange(width, device=device)
        ).to(torch.int32)

    runs = {}
    peer_sends = None
    for rank in (1, 0):  # rank 1 first: rank 0 replays its sends
        placement = CachePlacement(
            block_granularity=rows,
            virtual_block_count=virtual_block_count,
            group=tuple(range(degree)),
            rank=rank,
        )
        local, owned = resolve_cache_slots(every_slot, placement)
        planes = [None] * 6
        for layer_id in latent_layers:
            plane = torch.full(
                (local_rows, 1, width), -1, dtype=torch.int32, device=device
            )
            plane[local[owned], 0] = payload(layer_id, every_slot[owned])
            planes[layer_id] = plane
        pool = _StubLatentPool(planes)
        backend = _RecordingBackend(rank, peer_sends)
        history = PrefixHistoryGather(placement, comm=backend)
        history.prepare(
            chunks,
            block_table_cpu=table,
            prefix_lens_cpu=torch.tensor(prefix_lens),
            chunk_len=chunk_len,
        )
        num_windows = len(history.windows)
        assert num_windows > 1
        launched_after_layer = []
        outputs = {}
        for layer_id in latent_layers:
            outputs[layer_id] = [
                torch.cat(pair, dim=-1)
                for pair in history.read(
                    pool,
                    SimpleNamespace(layer_id=layer_id),
                    chunks,
                    dst_dtype=torch.int32,
                )
            ]
            launched_after_layer.append(len(backend.sends))
        if device == "cuda":
            torch.cuda.synchronize()
        runs[rank] = (outputs, launched_after_layer)
        peer_sends = backend.sends

    outputs, launched_after_layer = runs[0]
    # One window ahead: draining a layer leaves the next latent layer's first
    # window launched; the last layer has none after it.
    assert launched_after_layer == [
        num_windows + 1,
        2 * num_windows + 1,
        3 * num_windows,
    ]
    for layer_id in latent_layers:
        assert len(outputs[layer_id]) == len(chunks)
        for chunk, got in zip(chunks, outputs[layer_id]):
            torch.testing.assert_close(got, payload(layer_id, chunk).unsqueeze(1))


def test_history_gather_plans_at_first_read():
    """Preparing a forward's gathers plans nothing: the windows are planned
    when the first MLA layer reads them, off the forward's critical start."""
    from tokenspeed.runtime.layers.attention.dcp.cache import PrefixHistoryGather

    placement = CachePlacement(
        block_granularity=4, virtual_block_count=16, group=(0, 1), rank=0
    )
    history = PrefixHistoryGather(placement, comm=None)
    loops = [torch.tensor([4, 5, 6, 7, 0, 0], dtype=torch.int32)]
    with pytest.raises(RuntimeError, match="host block table"):
        history.prepare(
            loops, block_table_cpu=None, prefix_lens_cpu=torch.tensor([6]), chunk_len=6
        )
    history.prepare(
        loops,
        # Block 0 below the prefix: an unplannable table, refused only on read.
        block_table_cpu=torch.tensor([[1, 0, 3]], dtype=torch.int32),
        prefix_lens_cpu=torch.tensor([6]),
        chunk_len=6,
    )
    with pytest.raises(ValueError, match="holes"):
        history.windows

    # A forward without prefix loops (a prefill with no cached prefix) has
    # no host table to plan from and reads nothing.
    no_loops = []
    history.prepare(
        no_loops, block_table_cpu=None, prefix_lens_cpu=torch.tensor([0]), chunk_len=3
    )
    assert list(history.read(None, None, no_loops, dst_dtype=torch.int32)) == []
