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
    page_size: int,
    subpages: int,
    virtual_block_count: int,
) -> tuple[torch.Tensor, list[list[int]]]:
    """A host kernel-page table dealing each request distinct random blocks,
    and every request's virtual slot per prefix position."""
    pages_needed = [-(-length // page_size) for length in prefix_lens]
    blocks_needed = [-(-pages // subpages) for pages in pages_needed]
    blocks = (
        torch.randperm(virtual_block_count - 1, generator=generator)[
            : sum(blocks_needed)
        ]
        + 1
    ).tolist()
    columns = max(blocks_needed) * subpages + 3  # the mirror may be wider
    table = torch.zeros((len(prefix_lens), columns), dtype=torch.int32)
    slots = []
    for request, (length, count) in enumerate(zip(prefix_lens, blocks_needed)):
        mine, blocks = blocks[:count], blocks[count:]
        pages = [block * subpages + sub for block in mine for sub in range(subpages)]
        table[request, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
        slots.append(
            [pages[p // page_size] * page_size + p % page_size for p in range(length)]
        )
    return table, slots


def _chunk_plan(prefix_lens, chunk_len, reuse):
    from tokenspeed.runtime.layers.attention.chunk import chunking, plan_prefix_chunks

    lengths = torch.tensor(prefix_lens)
    if reuse:
        return plan_prefix_chunks(lengths, chunk_len * len(prefix_lens))
    return chunking(
        lengths, -(-max(prefix_lens) // chunk_len), len(prefix_lens), chunk_len
    )


def _chunk_slots(slots: list[list[int]], boundaries: torch.Tensor) -> list[list[int]]:
    return [
        [
            slot
            for request, (start, end) in enumerate(zip(starts, ends))
            for slot in slots[request][start:end]
        ]
        for starts, ends in zip(boundaries[:-1].tolist(), boundaries[1:].tolist())
    ]


CASES = [
    # prefix_lens, page_size, subpages, chunk_len, degree
    ([300], 64, 2, 128, 4),
    ([1000, 37, 513], 64, 2, 100, 4),
    ([5000, 4097], 32, 4, 2048, 16),
    ([129, 0, 64], 64, 1, 64, 2),
    ([7], 1, 4, 3, 2),
    ([30000, 0, 1000, 45000], 64, 2, 8192, 16),
    # More loops than one window covers.
    ([5000, 4097], 32, 4, 512, 4),
]


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("prefix_lens,page_size,subpages,chunk_len,degree", CASES)
def test_owned_prefix_chunk_rows_matches_per_row_owners(
    prefix_lens, page_size, subpages, chunk_len, degree, reuse
):
    generator = torch.Generator().manual_seed(7)
    virtual_block_count = 4096
    placement = CachePlacement(
        block_granularity=page_size * subpages,
        virtual_block_count=virtual_block_count,
        group=tuple(range(degree)),
        rank=0,
    )
    table, slots = _random_layout(
        generator,
        prefix_lens=prefix_lens,
        page_size=page_size,
        subpages=subpages,
        virtual_block_count=virtual_block_count,
    )
    bounds = _chunk_plan(prefix_lens, chunk_len, reuse).boundaries
    chunks = _chunk_slots(slots, bounds)
    counts = owned_prefix_chunk_rows(
        table,
        bounds,
        page_size=page_size,
        placement=placement,
    )
    rows = placement.block_granularity
    expected = torch.zeros((len(chunks), degree), dtype=torch.int64)
    for j, chunk in enumerate(chunks):
        for slot in chunk:
            expected[j, (slot // rows - 1) % degree] += 1
    torch.testing.assert_close(counts, expected)


def test_owned_prefix_chunk_rows_refuses_a_hole():
    placement = CachePlacement(
        block_granularity=4, virtual_block_count=16, group=(0, 1), rank=0
    )
    with pytest.raises(ValueError, match="holes"):
        owned_prefix_chunk_rows(
            torch.tensor([[4, 0, 12]], dtype=torch.int32),
            torch.tensor([[0], [3]]),
            page_size=1,
            placement=placement,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("plan", ["nccl", "fused", "narrow_slot", "both_bounds"])
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("prefix_lens,page_size,subpages,chunk_len,degree", CASES)
def test_every_rank_reads_the_position_order_rows(
    prefix_lens, page_size, subpages, chunk_len, degree, reuse, plan
):
    """Emulate each window's all-gather over ``degree`` owners: every rank's
    send rows come from its own physical cache, and every rank's plan must
    read the rank-major, padded output back as each loop's rows, in order.

    Every rank plans the same windows. A window keeps its bounds unless one
    loop alone exceeds the rows bound, and takes every loop it can: the NCCL
    windows of ``PREFIX_HISTORY_WINDOW_LOOPS`` loops, the fused ones packed
    into a slot of the loops' capacity (which then holds every window), a
    slot narrower than some loops, and both bounds at once."""
    from tokenspeed.runtime.layers.attention.dcp.cache import (
        PREFIX_HISTORY_WINDOW_LOOPS,
        plan_prefix_history_gather,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import (
        resolve_cache_slots,
    )

    # Either chunk plan expands at most this many rows a loop.
    capacity = chunk_len * len(prefix_lens)
    max_loops, max_send_rows = {
        "nccl": (PREFIX_HISTORY_WINDOW_LOOPS, None),
        "fused": (None, capacity),
        "narrow_slot": (None, max(1, capacity // (2 * degree))),
        "both_bounds": (2, capacity),
    }[plan]
    generator = torch.Generator().manual_seed(11)
    virtual_block_count = 4096
    rows = page_size * subpages
    table, slots = _random_layout(
        generator,
        prefix_lens=prefix_lens,
        page_size=page_size,
        subpages=subpages,
        virtual_block_count=virtual_block_count,
    )
    bounds = _chunk_plan(prefix_lens, chunk_len, reuse).boundaries
    chunks = [
        torch.tensor(chunk, dtype=torch.int32, device="cuda")
        for chunk in _chunk_slots(slots, bounds)
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
            page_table_cpu=table,
            chunk_bounds_cpu=bounds,
            page_size=page_size,
            placement=placement,
            max_loops=max_loops,
            max_send_rows=max_send_rows,
        )
        for placement in placements
    ]
    assert all(len(windows) == len(plans[0]) for windows in plans)
    if plan == "nccl":
        assert len(plans[0]) == -(-len(chunks) // PREFIX_HISTORY_WINDOW_LOOPS)
    counts = owned_prefix_chunk_rows(
        table, bounds, page_size=page_size, placement=placements[0]
    ).tolist()
    loop = 0
    for windows in zip(*plans):
        send_rows = windows[0].send_rows
        owned = windows[0].owned_rows_per_rank
        taken = len(windows[0].loop_bounds) - 1
        assert all(w.send_rows == send_rows for w in windows)
        assert all(w.owned_rows_per_rank == owned for w in windows)
        assert all(w.loop_bounds == windows[0].loop_bounds for w in windows)
        assert max_loops is None or taken <= max_loops
        assert max_send_rows is None or taken == 1 or send_rows <= max_send_rows
        if plan == "fused":
            assert send_rows <= capacity
        if loop + taken < len(chunks):
            # The next loop would break a bound.
            joined = max(mine + more for mine, more in zip(owned, counts[loop + taken]))
            overflows = max_send_rows is not None and joined > max_send_rows
            assert taken == max_loops or overflows
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_windows_pack_loops_by_their_busiest_owner():
    """Loops whose rows all sit on one owner fill a slot of one loop's rows
    alone, and loops dealt over the owners share one while their busiest
    owner fits: the owners' counts decide, not an even split of the slot. Four
    loops a window would put 640 rows on one owner."""
    from tokenspeed.runtime.layers.attention.dcp.cache import (
        PREFIX_HISTORY_WINDOW_LOOPS,
        plan_prefix_history_gather,
    )

    degree, page_size, subpages = 4, 64, 2
    rows = page_size * subpages
    # Loops of two blocks: two loops on owner 0's blocks only, then four
    # going round the owners.
    blocks = [1, 5, 9, 13] + list(range(17, 25))
    pages = [block * subpages + sub for block in blocks for sub in range(subpages)]
    prefix_len = len(blocks) * rows
    bounds = _chunk_plan([prefix_len], 2 * rows, reuse=True).boundaries
    chunks = [
        torch.tensor(chunk, dtype=torch.int32, device="cuda")
        for chunk in _chunk_slots(
            [
                [
                    pages[p // page_size] * page_size + p % page_size
                    for p in range(prefix_len)
                ]
            ],
            bounds,
        )
    ]
    layout = dict(
        page_table_cpu=torch.tensor([pages], dtype=torch.int32),
        chunk_bounds_cpu=bounds,
        page_size=page_size,
        placement=CachePlacement(
            block_granularity=rows,
            virtual_block_count=64,
            group=tuple(range(degree)),
            rank=0,
        ),
    )
    fused = plan_prefix_history_gather(
        chunks, **layout, max_loops=None, max_send_rows=2 * rows
    )
    assert [w.loop_bounds for w in fused] == [
        (0, 256),
        (0, 256),
        (0, 256, 512, 768, 1024),
    ]
    assert [w.owned_rows_per_rank for w in fused] == [
        (256, 0, 0, 0),
        (256, 0, 0, 0),
        (256, 256, 256, 256),
    ]
    assert [w.send_rows for w in fused] == [256] * 3
    nccl = plan_prefix_history_gather(
        chunks, **layout, max_loops=PREFIX_HISTORY_WINDOW_LOOPS, max_send_rows=None
    )
    assert [w.send_rows for w in nccl] == [640, 128]
    with pytest.raises(ValueError, match="positive"):
        plan_prefix_history_gather(chunks, **layout, max_loops=None, max_send_rows=0)


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
@pytest.mark.parametrize("reuse", [False, True])
def test_read_prefix_history_prefetches_one_window_ahead(reuse):
    """Every layer's loops read their rows back through windows that the
    previous window -- of this layer, else of the previous latent layer --
    already launched; the gathers run in the same order on every rank."""
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.dcp.cache import PrefixHistoryGather
    from tokenspeed.runtime.layers.attention.dcp.placement import (
        resolve_cache_slots,
    )

    degree, page_size, subpages, chunk_len = 2, 32, 4, 1024
    prefix_lens = [20000, 300]
    virtual_block_count = 512
    rows = page_size * subpages
    table, slots = _random_layout(
        torch.Generator().manual_seed(5),
        prefix_lens=prefix_lens,
        page_size=page_size,
        subpages=subpages,
        virtual_block_count=virtual_block_count,
    )
    bounds = _chunk_plan(prefix_lens, chunk_len, reuse).boundaries
    chunks = [
        torch.tensor(chunk, dtype=torch.int32, device="cuda")
        for chunk in _chunk_slots(slots, bounds)
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
        history = PrefixHistoryGather(
            placement, page_size=page_size, comm=backend, transport=None
        )
        history.prepare(
            chunks,
            page_table_cpu=table,
            chunk_bounds_cpu=bounds,
        )
        num_windows = len(history.windows(pool))
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
@pytest.mark.parametrize("reuse", [False, True])
def test_a_prefetched_window_waits_for_its_layers_host_load(reuse):
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
    table = torch.zeros((1, len(pages) + 3), dtype=torch.int32)
    table[0, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
    bounds = _chunk_plan([prefix_len], chunk_len, reuse).boundaries
    chunks = [
        torch.tensor(chunk, dtype=torch.int32, device="cuda")
        for chunk in _chunk_slots(
            [
                [
                    pages[p // page_size] * page_size + p % page_size
                    for p in range(prefix_len)
                ]
            ],
            bounds,
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
        history = PrefixHistoryGather(
            placement, page_size=page_size, comm=backend, transport=None
        )
        history.prepare(
            chunks,
            page_table_cpu=table,
            chunk_bounds_cpu=bounds,
        )
        pool = _StubLatentPool(planes)
        # Planned before the load starts, so the reads below enqueue at once.
        num_windows = len(history.windows(pool))
        assert num_windows > 1
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
    history = PrefixHistoryGather(placement, page_size=1, comm=None, transport=None)
    loops = [torch.tensor([4, 5, 6], dtype=torch.int32)]
    with pytest.raises(RuntimeError, match="host page table"):
        history.prepare(
            loops, page_table_cpu=None, chunk_bounds_cpu=torch.tensor([[0], [3]])
        )
    history.prepare(
        loops,
        # Page 0 below the prefix: an unplannable table, refused only on read.
        page_table_cpu=torch.tensor([[4, 0, 12]], dtype=torch.int32),
        chunk_bounds_cpu=torch.tensor([[0], [3]]),
    )
    with pytest.raises(ValueError, match="holes"):
        history.windows(_StubLatentPool([None]))

    # A forward without prefix loops (a prefill with no cached prefix) has
    # no host table to plan from and reads nothing.
    no_loops = []
    history.prepare(no_loops, page_table_cpu=None, chunk_bounds_cpu=torch.tensor([[0]]))
    assert list(history.read(None, None, no_loops, dst_dtype=torch.int32)) == []


class _Work:
    """A collective's work that counts its waits on its backend and fails
    like an errored collective's while the backend says so."""

    def __init__(self, backend):
        self._backend = backend

    def wait(self):
        self._backend.waits += 1
        if self._backend.failing:
            raise RuntimeError("collective failed")


class _CountingBackend(_RecordingBackend):
    """Rank 1 of ``_RecordingBackend``, whose works count their waits."""

    def __init__(self):
        super().__init__(1, None)
        self.waits = 0
        self.failing = False

    def all_gather_single_async(self, gathered, send, group):
        super().all_gather_single_async(gathered, send, group)
        return _Work(self)


def _two_layer_reader():
    """A rank-1 reader of two latent layers (1 and 3) over two owners, whose
    forward plans three windows, and its pool; its gathers count their
    waits."""
    from tokenspeed.runtime.layers.attention.dcp.cache import PrefixHistoryGather

    page_size, subpages, chunk_len = 32, 4, 512
    prefix_lens, virtual_block_count = [5000, 300], 512
    table, slots = _random_layout(
        torch.Generator().manual_seed(5),
        prefix_lens=prefix_lens,
        page_size=page_size,
        subpages=subpages,
        virtual_block_count=virtual_block_count,
    )
    bounds = _chunk_plan(prefix_lens, chunk_len, reuse=True).boundaries
    loops = [
        torch.tensor(chunk, dtype=torch.int32, device="cuda")
        for chunk in _chunk_slots(slots, bounds)
    ]
    rows = page_size * subpages
    placement = CachePlacement(
        block_granularity=rows,
        virtual_block_count=virtual_block_count,
        group=(0, 1),
        rank=1,
    )
    backend = _CountingBackend()
    history = PrefixHistoryGather(
        placement, page_size=page_size, comm=backend, transport=None
    )
    planes = [None] * 4
    for layer_id in (1, 3):
        planes[layer_id] = torch.zeros(
            (((virtual_block_count - 2) // 2 + 2) * rows, 1, 8),
            dtype=torch.int32,
            device="cuda",
        )
    pool = _StubLatentPool(planes)

    def prepare():
        history.prepare(loops, page_table_cpu=table, chunk_bounds_cpu=bounds)

    def read(layer_id):
        from types import SimpleNamespace

        return history.read(
            pool, SimpleNamespace(layer_id=layer_id), loops, dst_dtype=torch.int32
        )

    return history, pool, backend, prepare, read


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_a_read_stopped_early_waits_for_its_windows():
    """A read closed after any yield waits for every gather it left in flight
    (its next window, or the next layer's first) and drops it; reads after it
    gather every window afresh."""
    history, pool, backend, prepare, read = _two_layer_reader()
    prepare()
    windows = len(history.windows(pool))
    loops = sum(len(window.loop_bounds) - 1 for window in history.windows(pool))
    assert windows == 3
    for taken in range(1, loops + 1):
        prepare()
        reading = read(1)
        for _ in range(taken):
            next(reading)
        reading.close()
        assert backend.waits == len(backend.sends)
        sends = len(backend.sends)
        assert len(list(read(1))) == loops and len(list(read(3))) == loops
        assert len(backend.sends) == sends + 2 * windows
        assert backend.waits == len(backend.sends)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_an_older_read_never_touches_a_newer_forward():
    """A read resumed after the next prepare raises, and neither it nor one
    collected or closed on another thread then waits for or drops the newer
    forward's window in flight; the next prepare waits for what such a read
    left."""
    import gc
    import threading

    _, _, backend, prepare, read = _two_layer_reader()
    prepare()
    older = read(1)
    next(older)
    prepare()
    newer = read(1)
    next(newer)
    counts = (len(backend.sends), backend.waits)
    assert counts[0] - counts[1] == 1
    with pytest.raises(RuntimeError, match="resumed after"):
        next(older)
    assert (len(backend.sends), backend.waits) == counts
    collected = read(1)
    next(collected)
    prepare()
    newest = read(1)
    next(newest)
    counts = (len(backend.sends), backend.waits)
    del collected
    gc.collect()
    closer = threading.Thread(target=newest.close)
    closer.start()
    closer.join()
    assert (len(backend.sends), backend.waits) == counts
    prepare()
    assert backend.waits == len(backend.sends)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_a_failed_cleanup_wait_tries_every_window():
    """Every wait of a stopped read's cleanup is tried and the windows are
    dropped even when one fails; the error that stopped the read stays the
    one raised, with the cleanup's in a note, unless a close stopped it."""
    _, _, backend, prepare, read = _two_layer_reader()
    prepare()
    first = read(1)
    next(first)
    second = read(3)
    next(second)
    # In flight: the first layer's next window and the second layer's.
    assert len(backend.sends) - backend.waits == 2
    waits = backend.waits
    backend.failing = True
    with pytest.raises(ValueError, match="consumer failed") as stopped:
        second.throw(ValueError("consumer failed"))
    assert backend.waits == waits + 2
    assert any("collective failed" in note for note in stopped.value.__notes__)
    # Dropped: the next prepare waits for neither again.
    backend.failing = False
    prepare()
    assert backend.waits == waits + 2

    reading = read(1)
    next(reading)
    backend.failing = True
    with pytest.raises(RuntimeError, match="collective failed"):
        reading.close()
    waits = backend.waits
    backend.failing = False
    prepare()
    assert backend.waits == waits


def _startup_args(mapping, *, timeout_s, **overrides):
    """A prefill launch's resolved server arguments with FP8 MLA rows, on
    ``mapping`` (the test's KVP group), with ``overrides`` of the fields the
    startup verdict reads set the way resolution leaves them: some of those
    values need a model or hardware to pass the arguments' own checks."""
    from tokenspeed.runtime.utils.server_args import ServerArgs

    args = ServerArgs(
        model="x",
        world_size=2,
        disaggregation_mode="prefill",
        kv_parallel_size=2,
        kv_cache_dtype="fp8_e4m3",
        distributed_timeout_seconds=timeout_s,
    )
    args.mapping = mapping
    args.moe_backend = overrides.pop("moe_backend", args.moe_backend)
    args.draft_moe_backend = overrides.pop("draft_moe_backend", None)
    args.speculative_algorithm = overrides.pop("speculative_algorithm", None)
    args.all2all_backend = overrides.pop("all2all_backend", args.all2all_backend)
    args.disaggregation_mode = overrides.pop(
        "disaggregation_mode", args.disaggregation_mode
    )
    args.kv_cache_quant_method = overrides.pop(
        "kv_cache_quant_method", args.kv_cache_quant_method
    )
    args.attention_backend = overrides.pop("attention_backend", None)
    assert not overrides, f"unknown startup fields {sorted(overrides)}"
    return args


def _mla_model_config(attention_arch):
    """The MLA geometry the startup reads (576-byte FP8 rows); a launch's
    ModelConfig needs the model's HF config."""
    from types import SimpleNamespace

    return SimpleNamespace(
        attention_arch=attention_arch, kv_lora_rank=512, qk_rope_head_dim=64
    )


@pytest.fixture
def startup_verdicts(monkeypatch):
    """The verdicts this rank brings to the startup vote. The vote needs the
    group's processes: the factory is stood in for by one that records its
    arguments and refuses. The MLA history capacity is the defaults' (8192
    x 4 rows), as the server's arguments resolve it."""
    from tokenspeed_kernel.ops.communication import cute_dsl

    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 8192)
    monkeypatch.setitem(global_server_args_dict, "mla_chunk_multiplier", 4)
    verdicts = []
    monkeypatch.setattr(
        cute_dsl,
        "create_paged_row_all_gather_state",
        lambda group, device, **kwargs: verdicts.append(kwargs),
    )
    monkeypatch.setattr(
        pg_manager, "get_process_group", lambda backend, group, name=None: None
    )
    return verdicts


@pytest.mark.parametrize(
    "arch,overrides,plans,eligible",
    [
        # Model attention, server arguments, built MoE layers' (solution,
        # all-to-all) plans, the verdict.
        ("MLA", {}, [], True),
        ("MLA", {}, [("triton", "none"), ("flashinfer_trtllm", "none")], True),
        ("MLA", {"moe_backend": "mega_moe"}, [], False),
        ("MLA", {"moe_backend": "deep_gemm_mega_moe"}, [], False),
        (
            "MLA",
            {"draft_moe_backend": "a_plugin_moe", "speculative_algorithm": "EAGLE3"},
            [],
            False,
        ),
        # Without speculation no draft runs its MoE backend.
        ("MLA", {"draft_moe_backend": "a_plugin_moe"}, [], True),
        ("MLA", {"all2all_backend": "deepep"}, [], False),
        ("MLA", {"disaggregation_mode": "decode"}, [], False),
        ("MLA", {"kv_cache_quant_method": "per_token_head"}, [], False),
        ("MLA", {"attention_backend": "deepseek_v41"}, [], False),
        ("MHA", {}, [], False),
        ("MLA", {}, [("triton", "none"), ("mega_moe", "none")], False),
        ("MLA", {}, [("flashinfer_trtllm", "deepep")], False),
    ],
)
def test_startup_keeps_nccl_where_a_fused_window_could_deadlock(
    monkeypatch, startup_verdicts, arch, overrides, plans, eligible
):
    """The verdict a rank brings to the startup vote: the fused all-gather
    only for MLA prefix history read through ``PrefixHistoryGather`` in
    unquantized rows, outside a decode role, and when no configured kernel
    needs the whole GPU while it waits across ranks -- judged on the server
    arguments (target and draft MoE backends, the all-to-all) and on the
    built models' MoE plans, the draft's included."""
    from types import SimpleNamespace

    from tokenspeed.runtime.configs.model_config import AttentionArch
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.execution.distributed_initializer import (
        DistributedConfig,
    )
    from tokenspeed.runtime.layers.attention.dcp.cache import (
        _FUSED_HISTORY_TRANSPORTS,
        prepare_prefix_history_transport,
    )
    from tokenspeed.runtime.layers.moe import expert as expert_mod
    from tokenspeed.runtime.layers.moe.expert import MoELayer
    from tokenspeed.runtime.utils.env import global_server_args_dict

    # MoE layers whose plan names a solution and an all-to-all, without
    # weights (as test_moe_ispp_padding builds them).
    monkeypatch.setattr(
        expert_mod, "get_moe_backend", lambda: SimpleNamespace(value="triton")
    )
    monkeypatch.setattr(expert_mod, "create_layer_weights", lambda *a, **k: None)
    monkeypatch.setitem(global_server_args_dict, "moe_mxfp4_fp8_activation", False)
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
    models = []
    for solution, a2a_backend in plans:
        monkeypatch.setattr(
            expert_mod,
            "kernel_moe_plan",
            lambda weight_dtype, **kwargs: {
                "solution": solution,
                "a2a_backend": a2a_backend,
                "apply_kernel_name": "fake",
            },
        )
        models.append(
            torch.nn.ModuleList(
                [
                    MoELayer(
                        top_k=2,
                        num_experts=8,
                        hidden_size=256,
                        intermediate_size=512,
                        quant_config=None,
                        layer_index=0,
                        prefix="model.layers.0.mlp.experts",
                        tp_rank=0,
                        tp_size=1,
                    )
                ]
            )
        )
    mapping = Mapping(
        rank=0, world_size=2, attn_tp_size=2, attn_dp_size=1, attn_kvp_size=2
    )
    prepare_prefix_history_transport(
        _startup_args(mapping, timeout_s=None, **overrides),
        _mla_model_config(AttentionArch[arch]),
        models,
        torch.device("cpu"),
    )
    (verdict,) = startup_verdicts
    assert verdict["eligible"] is eligible
    if eligible:
        assert verdict["row_bytes"] == 576
        # One loop's history rows per rank.
        assert verdict["cap_bytes"] == 8192 * 4 * 576
        # The process groups' default timeout less 30 s.
        default_s = DistributedConfig.distributed_timeout_seconds
        assert verdict["timeout_ns"] == (default_s - 30) * 10**9
    assert _FUSED_HISTORY_TRANSPORTS == {}


@pytest.mark.parametrize(
    "chunked_prefill_size,multiplier,cap_rows",
    # The last: a slot of 2 x 2**21 rows of 576 bytes, over 2 GiB.
    [(2048, 2, 4096), (16384, 4, 65536), (-1, 4, None), (524288, 4, None)],
)
def test_startup_sizes_the_slot_from_the_history_capacity(
    monkeypatch, startup_verdicts, chunked_prefill_size, multiplier, cap_rows
):
    """A rank's part of a receive slot holds the MLA history capacity in
    rows, what the chunk planner expands at most a loop; without chunked
    prefill no loop reads history, and a slot of 2 GiB or more is past the
    factory's bound on the receive workspace: the group keeps NCCL."""
    from tokenspeed.runtime.configs.model_config import AttentionArch
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.attention.dcp.cache import (
        prepare_prefix_history_transport,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(
        global_server_args_dict, "chunked_prefill_size", chunked_prefill_size
    )
    monkeypatch.setitem(global_server_args_dict, "mla_chunk_multiplier", multiplier)
    mapping = Mapping(
        rank=0, world_size=2, attn_tp_size=2, attn_dp_size=1, attn_kvp_size=2
    )
    prepare_prefix_history_transport(
        _startup_args(mapping, timeout_s=None),
        _mla_model_config(AttentionArch.MLA),
        [],
        torch.device("cpu"),
    )
    (verdict,) = startup_verdicts
    assert verdict["eligible"] is (cap_rows is not None)
    if cap_rows is not None:
        assert verdict["cap_bytes"] == cap_rows * verdict["row_bytes"]


def test_startup_keeps_nccl_beside_fused_dp_projections(startup_verdicts):
    """A built DP projection whose workspace holds a fused projection state
    keeps the NCCL all-gather. Only GB300 builds that state: a stand-in takes
    its place."""
    from tokenspeed.runtime.configs.model_config import AttentionArch
    from tokenspeed.runtime.distributed.comm_backend.projection import (
        ProjectionSpec,
        ProjectionWorkspace,
    )
    from tokenspeed.runtime.distributed.mapping import DenseLayerMapping, Mapping
    from tokenspeed.runtime.layers.attention.dcp.cache import (
        prepare_prefix_history_transport,
    )
    from tokenspeed.runtime.layers.linear import DPRowParallelLinear

    row = DPRowParallelLinear(
        256,
        128,
        parallel=DenseLayerMapping(rank=0, world_size=2, tp_size=2, dp_size=1),
        params_dtype=torch.bfloat16,
        quant_config=None,
        prefix="o_proj",
    )
    row.projection_workspace = ProjectionWorkspace(
        ProjectionSpec(
            group=(0, 1),
            kind="row",
            input_size=256,
            output_size=128,
            max_tokens=16,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        ),
        backend=None,
    )
    row.projection_workspace.fused_row = object()
    mapping = Mapping(
        rank=0, world_size=2, attn_tp_size=2, attn_dp_size=1, attn_kvp_size=2
    )
    prepare_prefix_history_transport(
        _startup_args(mapping, timeout_s=None),
        _mla_model_config(AttentionArch.MLA),
        [torch.nn.ModuleList([row])],
        torch.device("cpu"),
    )
    (verdict,) = startup_verdicts
    assert verdict["eligible"] is False
