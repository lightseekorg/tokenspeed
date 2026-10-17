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

"""The MLA prefix-chunk gather plan: host owner counts and the all-gather
output rows every rank reads back."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    owned_prefix_chunk_rows,
)


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


CASES = [
    # prefix_lens, granularity, chunk_len, degree
    ([300], 128, 128, 4),
    ([1000, 37, 513], 128, 100, 4),
    ([5000, 4097], 128, 2048, 16),
    ([129, 0, 64], 64, 64, 2),
    ([7], 4, 3, 2),
    # More loops than one window covers.
    ([5000, 4097], 128, 512, 4),
]


@pytest.mark.parametrize("prefix_lens,granularity,chunk_len,degree", CASES)
def test_owned_prefix_chunk_rows_matches_per_row_owners(
    prefix_lens, granularity, chunk_len, degree
):
    generator = torch.Generator().manual_seed(7)
    virtual_block_count = 4096
    placement = CachePlacement(
        block_granularity=granularity,
        virtual_block_count=virtual_block_count,
        group=tuple(range(degree)),
        rank=0,
    )
    table, slots = _random_layout(
        generator,
        prefix_lens=prefix_lens,
        granularity=granularity,
        virtual_block_count=virtual_block_count,
    )
    chunks = _chunk_slots(slots, chunk_len)
    counts = owned_prefix_chunk_rows(
        table,
        torch.tensor(prefix_lens),
        chunk_len=chunk_len,
        num_chunks=len(chunks),
        placement=placement,
    )
    rows = placement.block_granularity
    expected = torch.zeros((len(chunks), degree), dtype=torch.int64)
    for j, chunk in enumerate(chunks):
        for slot in chunk:
            expected[j, (slot // rows - 1) % degree] += 1
    torch.testing.assert_close(counts, expected)


def test_owned_prefix_chunk_rows_refuses_a_hole_or_a_short_table():
    placement = CachePlacement(
        block_granularity=4, virtual_block_count=16, group=(0, 1), rank=0
    )
    with pytest.raises(ValueError, match="holes"):
        owned_prefix_chunk_rows(
            torch.tensor([[1, 0, 3]], dtype=torch.int32),
            torch.tensor([6]),
            chunk_len=6,
            num_chunks=1,
            placement=placement,
        )
    with pytest.raises(ValueError, match="does not cover"):
        owned_prefix_chunk_rows(
            torch.tensor([[1, 2]], dtype=torch.int32),
            torch.tensor([9]),
            chunk_len=9,
            num_chunks=1,
            placement=placement,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("prefix_lens,granularity,chunk_len,degree", CASES)
def test_every_rank_reads_the_position_order_rows(
    prefix_lens, granularity, chunk_len, degree
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
    virtual_block_count = 4096
    rows = granularity
    table, slots = _random_layout(
        generator,
        prefix_lens=prefix_lens,
        granularity=granularity,
        virtual_block_count=virtual_block_count,
    )
    chunks = [
        torch.tensor(chunk, dtype=torch.int32, device="cuda")
        for chunk in _chunk_slots(slots, chunk_len)
    ]
    placements = [
        CachePlacement(
            block_granularity=rows,
            virtual_block_count=virtual_block_count,
            group=tuple(range(degree)),
            rank=rank,
        )
        for rank in range(degree)
    ]
    # Every virtual slot's payload is distinct; rank r stores the rows it
    # owns at its local slots and garbage everywhere else.
    width = 8
    local_rows = ((virtual_block_count - 2) // degree + 2) * rows
    caches = [
        torch.full((local_rows, width), -1, dtype=torch.int32, device="cuda")
        for _ in range(degree)
    ]
    every_slot = torch.cat(chunks).to(torch.int64)
    payload = every_slot.unsqueeze(1) * width + torch.arange(width, device="cuda")
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
            (degree * send_rows, width), -2, dtype=torch.int32, device="cuda"
        )
        for rank, (window, cache) in enumerate(zip(windows, caches)):
            sent = cache.index_select(0, window.fetch_slots)
            gathered[rank * send_rows : rank * send_rows + sent.shape[0]] = sent
        bounds = windows[0].loop_bounds
        for start, end in zip(bounds[:-1], bounds[1:]):
            expected = chunks[loop].to(torch.int64).unsqueeze(1) * width + torch.arange(
                width, device="cuda"
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
    split in two halves. Like the MLA pool, a key-buffer read first waits for
    its layer's Host load when a load tracker is set."""

    def __init__(self, planes: list[torch.Tensor | None]):
        self.kv_buffer = planes
        self.layerwise_load_tracker = None

    def get_key_buffer(self, layer_id):
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_read_prefix_history_prefetches_one_window_ahead():
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
        torch.tensor(chunk, dtype=torch.int32, device="cuda")
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
            + torch.arange(width, device="cuda")
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
                (local_rows, 1, width), -1, dtype=torch.int32, device="cuda"
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_a_prefetched_window_waits_for_its_layers_host_load():
    """A Host load brings a prefix's later pages in layer by layer on a side
    stream, held back before the second latent layer's copy, behind the
    pools' real ``LayerwiseLoadTracker``. Reading the first latent layer
    prefetches the second one's first window, and that gather reads the
    loaded rows: it waits for its layer's load. The loaded pages are all rank
    1's, so rank 0 arms no load and sends only rows it held already, yet
    gathers every loaded row from rank 1."""
    from types import SimpleNamespace

    from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker
    from tokenspeed.runtime.layers.attention.dcp.cache import PrefixHistoryGather
    from tokenspeed.runtime.layers.attention.dcp.placement import (
        resolve_cache_slots,
    )

    degree, page_size, subpages, chunk_len = 2, 32, 4, 512
    virtual_block_count = 512
    rows = page_size * subpages
    # Blocks 1 to 3 (owners 0, 1, 0) stay on the Device; the Host load brings
    # even blocks, rank 1's, for the rest of the prefix.
    resident_blocks = [1, 2, 3]
    blocks = resident_blocks + list(range(4, 44, 2))
    prefix_len = len(blocks) * rows - 50
    pages = [block * subpages + sub for block in blocks for sub in range(subpages)]
    # The host block table the loops' kernel pages were expanded from.
    table = torch.zeros((1, len(blocks) + 3), dtype=torch.int32)
    table[0, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    chunks = [
        torch.tensor(chunk, dtype=torch.int32, device="cuda")
        for chunk in _chunk_slots(
            [
                [
                    pages[p // page_size] * page_size + p % page_size
                    for p in range(prefix_len)
                ]
            ],
            chunk_len,
        )
    ]
    every_slot = torch.cat(chunks).to(torch.int64)
    loaded = every_slot // rows > max(resident_blocks)
    latent_layers = [1, 3, 4]
    width = 8
    local_rows = ((virtual_block_count - 2) // degree + 2) * rows
    # Every layer's distinct payload of every slot, in loop order.
    payloads = {
        layer_id: (
            layer_id * 1_000_000
            + every_slot.unsqueeze(1) * width
            + torch.arange(width, device="cuda")
        ).to(torch.int32)
        for layer_id in latent_layers
    }

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
        resident = owned & ~loaded
        landing = owned & loaded
        planes = [None] * 6
        for layer_id in latent_layers:
            plane = torch.full(
                (local_rows, 1, width), -1, dtype=torch.int32, device="cuda"
            )
            plane[local[resident], 0] = payloads[layer_id][resident]
            planes[layer_id] = plane
        backend = _RecordingBackend(rank, peer_sends)
        history = PrefixHistoryGather(placement, comm=backend)
        history.prepare(
            chunks,
            block_table_cpu=table,
            prefix_lens_cpu=torch.tensor([prefix_len]),
            chunk_len=chunk_len,
        )
        # Planned before the load starts, so the reads below enqueue at once.
        num_windows = len(history.windows)
        assert num_windows > 1
        pool = _StubLatentPool(planes)
        pool.layerwise_load_tracker = LayerwiseLoadTracker(len(planes))
        load_stream = None
        if bool(landing.any()):
            # Every layer's copy in order on the load's own stream, each
            # recording its layer's event; the second latent layer's is late.
            # The rows are selected first: a masked copy would wait for the
            # load stream on the host.
            landing_rows = local[landing]
            landing_payloads = {
                layer_id: payloads[layer_id][landing] for layer_id in latent_layers
            }
            index = pool.layerwise_load_tracker.begin_load()
            events = pool.layerwise_load_tracker.event_sets[index].layer_done_events
            load_stream = torch.cuda.Stream()
            load_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(load_stream):
                for layer_id, plane in enumerate(planes):
                    if layer_id == latent_layers[1]:
                        torch.cuda._sleep(200_000_000)
                    if plane is not None:
                        plane[landing_rows, 0] = landing_payloads[layer_id]
                    events[layer_id].record(load_stream)
            pool.layerwise_load_tracker.set_consumers(index)
        else:
            pool.layerwise_load_tracker.set_consumers(-1)
        launched_after_layer = []
        outputs = {}
        for layer_id in latent_layers:
            outputs[layer_id] = torch.cat(
                [
                    torch.cat(pair, dim=-1)
                    for pair in history.read(
                        pool,
                        SimpleNamespace(layer_id=layer_id),
                        chunks,
                        dst_dtype=torch.int32,
                    )
                ]
            )
            launched_after_layer.append(len(backend.sends))
        # The load has not landed yet: every read above was enqueued ahead of
        # the delayed layers' copies.
        assert load_stream is None or not load_stream.query()
        torch.cuda.synchronize()
        runs[rank] = (outputs, launched_after_layer)
        peer_sends = backend.sends

    outputs, launched_after_layer = runs[0]
    # The second latent layer's first window was launched while the first
    # layer was read, ahead of its own load.
    assert launched_after_layer == [
        num_windows + 1,
        2 * num_windows + 1,
        3 * num_windows,
    ]
    for layer_id in latent_layers:
        torch.testing.assert_close(outputs[layer_id], payloads[layer_id].unsqueeze(1))


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
