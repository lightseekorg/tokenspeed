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

"""The data path of a shared L2 pool: where a rank's blocks and the pool's
pages are, the copies of a write, and the load executor that reads pool pages
back layer by layer.

The pool's memory is plain host bytes and a memory copy stands in for RDMA.
The CPU tests keep the ready words in plain memory too; the tests that need a
GPU run the real words and CUDA streams."""

import ctypes
import faulthandler
import json
import os
import sys
import threading
import time
import types
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from test.runtime.shared_l2_test_utils import (  # noqa: E402
    ServedOwner,
    lookup_fields,
    wait_until,
    write_fields,
)

import tokenspeed_engine.scheduler as ts  # noqa: E402

from tokenspeed.runtime.cache.host_storage import (  # noqa: E402
    HostCacheStorage,
    compute_host_group_packing,
    compute_host_lcm_block_bytes,
)
from tokenspeed.runtime.cache.l2.layerwise_load import (  # noqa: E402
    LayerwiseLoadTracker,
)
from tokenspeed.runtime.cache.shared_l2 import executor as executor_module  # noqa: E402
from tokenspeed.runtime.cache.shared_l2.client import (  # noqa: E402
    SharedL2Client,
    SharedL2ClientConfig,
    SharedL2PoolView,
    SharedL2Provider,
)
from tokenspeed.runtime.cache.shared_l2.executor import (  # noqa: E402
    HostSignals,
    SharedL2LoadExecutor,
)
from tokenspeed.runtime.cache.shared_l2.layout import (  # noqa: E402
    SharedL2Layout,
    build_shared_l2_layout,
)
from tokenspeed.runtime.cache.shared_l2.transfer import (  # noqa: E402
    RdmaEngine,
    SharedL2Transfer,
)
from tokenspeed.runtime.cache.shared_l2.wire import OwnerConnection  # noqa: E402
from tokenspeed.runtime.cache.transfer.layout import (  # noqa: E402
    CacheField,
    CacheGroupLayout,
    CacheTransferLayout,
)
from tokenspeed.runtime.cache.transfer.ops import RestoreOp  # noqa: E402
from tokenspeed.runtime.engine.scheduler_utils import make_config  # noqa: E402
from tokenspeed.runtime.execution.breakable_cuda_graph import (  # noqa: E402
    BreakableCapture,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (  # noqa: E402
    cache_group_config,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (  # noqa: E402
    CacheGroupSpec,
)
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="runs CUDA streams and their memory waits"
)

KV, STATE = 0, 1
# Device block ids 1..8 and the null block.
BLOCKS = 9
SPECS = (
    CacheGroupSpec(
        "kv", "full_history", rows_per_page=1, entry_stride_tokens=2, replayable=False
    ),
    CacheGroupSpec("state", "state", replayable=False),
)
# Two layers: the first reads k, the second v and the state.
CONSUMERS = (("layer.0.k",), ("layer.1.v", "layer.1.s"))
# A pool of 8 LCM blocks on two providers.
PROVIDER_BLOCKS = (3, 5)


class Arena:
    """Bytes standing in for a rank's Device cache: a KV group whose two
    fields, in one buffer, two layers read, and a State group. A KV block
    fills a pool LCM block; two State blocks fit in one, with bytes to spare.
    The Device packs four and eight blocks to its own LCM block."""

    def __init__(self, device="cpu"):
        # A KV block's row: k in bytes 0..12, v in bytes 16..32.
        self.kv = torch.zeros(BLOCKS, 32, dtype=torch.uint8, device=device)
        self.state = torch.zeros(BLOCKS, 16, dtype=torch.uint8, device=device)
        self.layout = SharedL2Layout.build(
            CacheTransferLayout(
                num_lcm_blocks=4,
                groups=(
                    CacheGroupLayout(
                        "kv",
                        4,
                        (
                            CacheField("layer.0.k", 0, 0, 32, 12),
                            CacheField("layer.1.v", 0, 16, 32, 16),
                        ),
                    ),
                    CacheGroupLayout(
                        "state", 8, (CacheField("layer.1.s", 1, 0, 16, 12),)
                    ),
                ),
                buffers=(self.kv, self.state),
                consumers=CONSUMERS,
            ),
            view_layers=(len(CONSUMERS),),
            group_specs=SPECS,
            cache_groups=[
                cache_group_config(spec, total_pages=0, cache_blocks_per_lcm_block=n)
                for spec, n in zip(SPECS, (4, 8))
            ],
            prefix_granularity=2,
        )

    def fill(self, group, block, seed):
        if group == KV:
            self.kv[block, :12] = torch.arange(seed, seed + 12, dtype=torch.uint8)
            self.kv[block, 16:] = torch.arange(seed + 50, seed + 66, dtype=torch.uint8)
        else:
            self.state[block, :12] = torch.arange(seed, seed + 12, dtype=torch.uint8)

    def fields(self, group, block):
        """The payload of every field of a block, in pool page order."""
        if group == KV:
            return [
                bytes(self.kv[block, :12].tolist()),
                bytes(self.kv[block, 16:].tolist()),
            ]
        return [bytes(self.state[block, :12].tolist())]

    def zero(self):
        self.kv.zero_()
        self.state.zero_()

    def copy_from(self, other):
        self.kv.copy_(other.kv)
        self.state.copy_(other.state)


class Pool:
    """A pool's memory on two providers, as one hello describes it."""

    def __init__(self, lcm_bytes):
        self.memory = [
            ctypes.create_string_buffer(blocks * lcm_bytes)
            for blocks in PROVIDER_BLOCKS
        ]
        self.view = SharedL2PoolView(
            client=1,
            providers=(
                SharedL2Provider("p0", ctypes.addressof(self.memory[0]), 3, 0),
                SharedL2Provider("p1", ctypes.addressof(self.memory[1]), 5, 3),
            ),
            lcm_bytes=lcm_bytes,
        )


class MemoryEngine:
    """Copies between host addresses, as RDMA does between a rank and a
    provider."""

    def __init__(self):
        # (direction, session) of every copy, in order.
        self.copies = []
        self.fail_reads = False
        # Reads of ranges of these sizes wait for ``resume``.
        self.held_sizes = frozenset()
        self.resume = threading.Event()
        self.read_delay_s = 0.0

    def read(self, session, local, remote, lengths):
        self.copies.append(("read", session))
        if self.held_sizes & set(lengths):
            assert self.resume.wait(10)
        time.sleep(self.read_delay_s)
        if self.fail_reads:
            raise RuntimeError(f"shared L2 read from {session} failed")
        self.copy(local, remote, lengths)

    def write(self, session, local, remote, lengths):
        self.copies.append(("write", session))
        self.copy(remote, local, lengths)

    def copy(self, destinations, sources, lengths):
        for destination, source, size in zip(
            destinations, sources, lengths, strict=True
        ):
            ctypes.memmove(destination, source, size)

    def close(self):
        pass


class CudaEngine(MemoryEngine):
    """Copies between CUDA memory and the pool's host memory from any thread."""

    def __init__(self):
        super().__init__()
        self._cuda = ctypes.CDLL("libcuda.so.1")

    def copy(self, destinations, sources, lengths):
        torch.cuda.set_device(0)
        for destination, source, size in zip(
            destinations, sources, lengths, strict=True
        ):
            result = self._cuda.cuMemcpy(
                ctypes.c_uint64(destination),
                ctypes.c_uint64(source),
                ctypes.c_size_t(size),
            )
            assert result == 0


class PlainWords:
    """The start and ready words in plain host memory. A stream write lands
    when the test lets it; a stream wait is kept, and ``drain`` returns once
    every wait kept so far on a stream would have passed, as synchronizing
    that stream does."""

    def __init__(self, num_words):
        self.words = np.zeros(num_words, dtype=np.uint32)
        # (stream, word, value) of every wait.
        self.waits = []
        self.hold_writes = False
        self._held = []
        self.closed = False

    def write(self, stream, word, value):
        if self.hold_writes:
            self._held.append((word, value))
        else:
            self.words[word] = value

    def release_writes(self):
        self.hold_writes = False
        for word, value in self._held:
            self.words[word] = value

    def wait(self, stream, word, value):
        self.waits.append((stream, word, value))

    def passed(self, word, value):
        return (int(self.words[word]) - value) & 0xFFFFFFFF < 0x80000000

    def drain(self, stream):
        wait_until(
            lambda: all(
                self.passed(word, value)
                for waiting, word, value in self.waits
                if waiting == stream
            )
        )

    def close(self):
        self.closed = True


@dataclass
class LoadBackOp:
    """A ``Cache.LoadBackOp`` batch of one operation; the binding's type has
    no constructor."""

    op_id: int
    rows: list  # (group, proxy Host block, Device block)

    @property
    def op_ids(self):
        return [self.op_id]

    @property
    def group_ids(self):
        return [[group for group, _, _ in self.rows]]

    @property
    def src_pages(self):
        return [[source for _, source, _ in self.rows]]

    @property
    def dst_pages(self):
        return [[destination for _, _, destination in self.rows]]


class Rank:
    """A rank's load path over a pool: its arena, its tracker and executor,
    and the proxy pages its scheduler would have mapped."""

    def __init__(
        self, pool, *, device="cpu", engine=None, words=PlainWords, load_pool=None
    ):
        self.pool = pool
        self.arena = Arena(device)
        self.engine = engine if engine is not None else MemoryEngine()
        self.tracker = LayerwiseLoadTracker(len(CONSUMERS))
        self.proxy_pages = {}
        # The tickets a load may still start from, unless a pool client's
        # ``load_pool`` decides.
        self.allowed_tickets = {1}
        # What the executor had stopped when it closed the pool client.
        self.closed_after = []
        self.executor = SharedL2LoadExecutor(
            self.arena.layout,
            self.engine,
            signals=words,
            trackers=[(self.tracker, len(CONSUMERS))],
            proxy_page=lambda group, block: self.proxy_pages[group, block],
            load_pool=load_pool
            or (
                lambda ticket: (
                    self.pool.view if ticket in self.allowed_tickets else None
                )
            ),
            load_workers=2,
            num_lcm_blocks=self.pool.view.num_lcm_blocks,
            close=lambda: self.closed_after.append(
                [thread.name for thread in threading.enumerate()]
            ),
        )
        self.words = self.executor._signals
        # The stream the forward's layer waits go to.
        self.forward = executor_module.device_module.current_stream()

    def load(self, op_id, rows, *, ticket=1, stream=None):
        """Submit the load of ``(group, pool page, Device block)`` rows, each
        page behind a proxy Host block of its own."""
        proxied = []
        for group, page, block in rows:
            proxy = 100 + len(self.proxy_pages)
            self.proxy_pages[group, proxy] = (ticket, 0, page)
            proxied.append((group, proxy, block))
        if stream is None:
            stream = self.forward
        self.executor.submit_load_backs(
            [LoadBackOp(op_id, proxied)], prerequisite_stream=stream
        )

    def wait_for_layers(self):
        """The forward's waits, on the current stream."""
        for layer in range(len(CONSUMERS)):
            self.tracker.wait_for_layer(layer)

    def run_forward(self):
        """The forward's waits on plain words: returns once all have passed."""
        self.wait_for_layers()
        self.words.drain(self.forward)


@pytest.fixture
def pool():
    return Pool(Arena().layout.lcm_bytes)


@pytest.fixture
def ranks(pool, monkeypatch):
    """Builds ranks over the pool; their executors are shut down afterwards."""
    monkeypatch.setattr(executor_module.Cache, "LoadBackOp", LoadBackOp)
    built = []

    def build(**settings) -> Rank:
        built.append(Rank(pool, **settings))
        return built[-1]

    yield build
    for rank in built:
        rank.engine.resume.set()
        if not rank.closed_after:
            rank.executor.shutdown()


# Pool pages of both groups on both providers, and the Device blocks a writer
# holds them in: (group, page, block). Two of the State pages share an LCM
# block.
STORED = [
    (KV, 1, 3),
    (KV, 3, 5),
    (KV, 8, 1),
    (STATE, 3, 4),
    (STATE, 4, 8),
    (STATE, 9, 2),
]


def store(pool, engine=None) -> Arena:
    """A writer rank fills its blocks and copies them to the pool's pages."""
    writer = Arena()
    for group, page, block in STORED:
        writer.fill(group, block, seed=10 * page + group)
    SharedL2Transfer(writer.layout, engine or MemoryEngine()).write(
        pool.view,
        [((0, group, "hash", 0), 0, page, block) for group, page, block in STORED],
    )
    return writer


# A reader loads the same pages into other Device blocks.
LOADED = [(group, page, BLOCKS - block) for group, page, block in STORED]


def assert_loaded(reader: Arena, writer: Arena) -> None:
    for (group, _, written), (_, _, loaded) in zip(STORED, LOADED, strict=True):
        assert reader.fields(group, loaded) == writer.fields(group, written)


def test_a_pool_is_laid_out_as_a_ranks_host_tier(pool):
    layout = Arena().layout
    transfer = layout.transfer
    assert layout.lcm_bytes == compute_host_lcm_block_bytes(transfer, group_specs=SPECS)
    assert layout.host_packing == compute_host_group_packing(
        transfer, group_specs=SPECS
    )
    assert (layout.lcm_bytes, layout.host_packing) == (28, (1, 2))
    assert (layout.block_bytes, layout.field_offsets) == ((28, 12), ((0, 12), (0,)))

    # A page lies where the Host tier puts the block of that id, on the
    # provider that holds its LCM block.
    host_tier = SimpleNamespace(
        group_ids=("kv", "state"),
        group_packing=layout.host_packing,
        host_cache_block_bytes=layout.block_bytes,
        host_lcm_block_bytes=layout.lcm_bytes,
        num_host_lcm_blocks=sum(PROVIDER_BLOCKS),
    )
    bases = {provider.session: provider for provider in pool.view.providers}
    for group, pages in ((KV, 8), (STATE, 16)):
        for page in range(1, pages + 1):
            session, address = layout.page_address(pool.view, group, 0, page)
            provider = bases[session]
            in_provider = address - provider.base
            assert (
                0 <= in_provider <= provider.lcm_blocks * 28 - layout.block_bytes[group]
            )
            assert provider.start * 28 + in_provider == (
                HostCacheStorage.host_block_offset(host_tier, group, page)
            )
    assert layout.page_address(pool.view, STATE, 0, 6)[0] == "p0"
    assert layout.page_address(pool.view, STATE, 0, 7)[0] == "p1"
    # Two State pages of an LCM block lie back to back.
    first, second = (
        layout.page_address(pool.view, STATE, 0, page)[1] for page in (7, 8)
    )
    assert second - first == 12

    for page in (0, 17):
        with pytest.raises(IndexError, match="out of range"):
            layout.page_address(pool.view, STATE, 0, page)
    with pytest.raises(ValueError, match="bucket 0 only"):
        layout.page_address(pool.view, KV, 1, 1)
    other = SharedL2PoolView(client=1, providers=pool.view.providers, lcm_bytes=56)
    with pytest.raises(ValueError, match="LCM block holds 56 bytes"):
        layout.page_address(other, KV, 0, 1)


def test_the_geometry_is_the_schedulers_view_of_the_groups():
    layout = Arena().layout
    assert layout.geometry == {
        "prefix_granularity": 2,
        "lcm_bytes": 28,
        "groups": [
            {
                "group_id": "kv",
                "attn_kind": "Full",
                "block_granularity": 2,
                "cache_blocks_per_lcm_block": 4,
                "host_packing": 1,
                "shard_count": 1,
                "sliding_window_tokens": None,
                "sink_tokens": 0,
                "replayable": False,
                "checkpoint_granularity": 1,
            },
            {
                "group_id": "state",
                "attn_kind": "State",
                "block_granularity": 0,
                "cache_blocks_per_lcm_block": 8,
                "host_packing": 2,
                "shard_count": 1,
                "sliding_window_tokens": None,
                "sink_tokens": 0,
                "replayable": False,
                "checkpoint_granularity": 1,
            },
        ],
    }
    # What crosses the owner's wire unchanged.
    assert json.loads(json.dumps(layout.geometry)) == layout.geometry
    assert layout.registration_regions() == tuple(
        (buffer.data_ptr(), buffer.nbytes) for buffer in layout.transfer.buffers
    )


def test_a_sharded_group_cannot_join_a_pool():
    layout = Arena().layout
    sharded = (
        CacheGroupSpec(
            "kv",
            "full_history",
            rows_per_page=1,
            entry_stride_tokens=2,
            replayable=False,
            shard_count=2,
        ),
        SPECS[1],
    )
    with pytest.raises(ValueError, match="replicated cache groups only"):
        SharedL2Layout.build(
            layout.transfer,
            view_layers=layout.view_layers,
            group_specs=sharded,
            cache_groups=[],
            prefix_granularity=2,
        )


class CachePoolView:
    """A cache pool as the layout reads it: its transfer layout and its
    arena's group specs. A real pool needs a GPU arena."""

    def __init__(self, transfer):
        self._transfer = transfer
        self.arena = SimpleNamespace(
            runtime_contract=SimpleNamespace(group_specs=SPECS)
        )

    def cache_transfer_layout(self):
        return self._transfer


def test_the_layout_is_read_from_the_cache_pools():
    by_hand = Arena().layout
    target = CachePoolView(by_hand.transfer)
    # The scheduler's groups, as the device build hands them over.
    cache_groups = [
        cache_group_config(spec, total_pages=0, cache_blocks_per_lcm_block=packing)
        for spec, packing in zip(SPECS, (4, 8))
    ]

    layout = build_shared_l2_layout(
        target, None, cache_groups=cache_groups, prefix_granularity=2
    )
    assert layout == by_hand

    # A draft view of the same arena, with a field of its own in the KV group:
    # its layer follows the target's, and its payload joins the pool page.
    arena = torch.zeros(BLOCKS, 64, dtype=torch.uint8)
    target = CachePoolView(
        CacheTransferLayout(
            num_lcm_blocks=4,
            groups=(
                CacheGroupLayout(
                    "kv",
                    4,
                    (
                        CacheField("layer.0.k", 0, 0, 64, 12),
                        CacheField("layer.1.v", 0, 16, 64, 16),
                    ),
                ),
                CacheGroupLayout("state", 8, (CacheField("layer.1.s", 0, 32, 64, 12),)),
            ),
            buffers=(arena,),
            consumers=CONSUMERS,
        )
    )
    draft = CachePoolView(
        CacheTransferLayout(
            num_lcm_blocks=4,
            groups=(
                CacheGroupLayout("kv", 4, (CacheField("layer.2.k", 0, 48, 64, 6),)),
            ),
            buffers=(arena,),
            consumers=(("layer.2.k",),),
        )
    )
    layout = build_shared_l2_layout(
        target, draft, cache_groups=cache_groups, prefix_granularity=2
    )
    assert layout.view_layers == (2, 1)
    assert layout.transfer.consumers == CONSUMERS + (("layer.2.k",),)
    assert (layout.block_bytes, layout.lcm_bytes, layout.host_packing) == (
        (34, 12),
        34,
        (1, 2),
    )
    assert layout.field_offsets[KV] == (0, 12, 28)
    assert layout.registration_regions() == ((arena.data_ptr(), arena.nbytes),)


def test_the_rdma_engine_registers_the_cache_and_moves_ranges(monkeypatch):
    class TransferEngine:
        """Mooncake's engine, which needs an RDMA device, as the rank calls
        it."""

        registered = []
        calls = []
        result = 0

        def initialize(self, hostname, metadata, protocol, device_name):
            assert (metadata, protocol, device_name) == (
                "P2PHANDSHAKE",
                "rdma",
                "mlx5_0",
            )
            return 0

        def register_memory(self, address, size):
            self.registered.append((address, size))
            return 0

        def unregister_memory(self, address):
            self.registered.remove(
                next(region for region in self.registered if region[0] == address)
            )
            return 0

        def batch_transfer_sync_read(self, session, local, remote, lengths):
            self.calls.append(("read", session, local, remote, lengths))
            return self.result

        def batch_transfer_sync_write(self, session, local, remote, lengths):
            self.calls.append(("write", session, local, remote, lengths))
            return self.result

    module = types.ModuleType("mooncake.engine")
    module.TransferEngine = TransferEngine
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.engine", module)
    regions = Arena().layout.registration_regions()

    engine = RdmaEngine(device_name="mlx5_0", regions=regions)
    assert TransferEngine.registered == list(regions)
    engine.read("p0", [1], [2], [3])
    engine.write("p1", [4], [5], [6])
    assert TransferEngine.calls == [
        ("read", "p0", [1], [2], [3]),
        ("write", "p1", [4], [5], [6]),
    ]
    # A copy Mooncake reports as failed raises, for a load or a write to fail.
    TransferEngine.result = -1
    with pytest.raises(RuntimeError, match="read from p0 failed"):
        engine.read("p0", [1], [2], [3])
    with pytest.raises(RuntimeError, match="write to p1 failed"):
        engine.write("p1", [4], [5], [6])

    # The rank's writes and loads share the engine: the second close is void.
    engine.close()
    engine.close()
    assert TransferEngine.registered == []


def test_a_write_then_a_load_restores_every_field_of_every_group(pool, ranks):
    engine = MemoryEngine()
    writer = store(pool, engine)
    # Each page went to the provider that holds it.
    assert sorted(engine.copies) == [("write", "p0"), ("write", "p1")]

    reader = ranks()
    reader.load(7, LOADED)
    reader.run_forward()

    assert_loaded(reader.arena, writer)
    wait_until(lambda: [event.op_id for event in reader.executor.poll_results()] == [7])
    assert reader.executor.poll_results() == []
    # Only the loaded blocks changed, and of those only the payloads.
    expected = Arena()
    for (group, _, written), (_, _, loaded) in zip(STORED, LOADED, strict=True):
        payloads = writer.fields(group, written)
        if group == KV:
            expected.kv[loaded, :12] = torch.tensor(
                list(payloads[0]), dtype=torch.uint8
            )
            expected.kv[loaded, 16:] = torch.tensor(
                list(payloads[1]), dtype=torch.uint8
            )
        else:
            expected.state[loaded, :12] = torch.tensor(
                list(payloads[0]), dtype=torch.uint8
            )
    assert torch.equal(reader.arena.kv, expected.kv)
    assert torch.equal(reader.arena.state, expected.state)


def test_a_layer_is_ready_once_its_fields_are_there(pool, ranks):
    writer = store(pool)
    reader = ranks()
    # The second layer's fields (v and the state) stay on their providers.
    reader.engine.held_sizes = frozenset({16})

    reader.load(7, LOADED)
    reader.tracker.wait_for_layer(0)
    reader.words.drain(reader.forward)

    # The first layer's wait passed with its field loaded and no other.
    ((_, first_word, seq),) = [
        wait for wait in reader.words.waits if wait[0] == reader.forward
    ]
    assert reader.words.passed(first_word, seq)
    assert not reader.words.passed(first_word + 1, seq)
    block = LOADED[0][2]
    assert reader.arena.fields(KV, block)[0] == writer.fields(KV, STORED[0][2])[0]
    assert not reader.arena.kv[:, 16:].any() and not reader.arena.state.any()
    assert reader.executor.poll_results() == []

    reader.engine.resume.set()
    reader.run_forward()
    assert_loaded(reader.arena, writer)


def test_a_load_starts_behind_the_plans_page_zeroing(pool, ranks):
    writer = store(pool)
    reader = ranks()
    reader.words.hold_writes = True

    reader.load(7, LOADED)
    # The start word has not landed: the zeroing may still be running.
    time.sleep(0.05)
    assert reader.engine.copies == []
    assert not reader.arena.kv.any()

    reader.words.release_writes()
    reader.run_forward()
    assert_loaded(reader.arena, writer)


def test_the_whole_load_is_one_event_behind_the_last_layers_word(pool, ranks):
    store(pool)
    reader = ranks()
    before = list(reader.tracker.event_sets[0].layer_done_events)

    reader.load(7, LOADED)

    # Before any layer wait of the forward, a wait for the last layer's word
    # sits on a stream of the executor's own, and the event recorded behind it
    # stands for every layer of the load.
    ((stream, word, seq),) = reader.words.waits
    assert stream != reader.forward
    assert word == len(CONSUMERS) and reader.words.words[0] == seq
    index = reader.tracker.current_load_index
    events = reader.tracker.event_sets[index].layer_done_events
    assert len(set(map(id, events))) == 1 and events[0] not in before
    assert reader.tracker.consumer_indices == (index,)
    reader.words.drain(stream)


def test_consecutive_loads_take_turns_on_the_words(pool, ranks):
    writer = store(pool)
    reader = ranks()
    for op_id in range(1, 6):
        reader.arena.zero()
        reader.load(op_id, LOADED)
        reader.run_forward()
        assert_loaded(reader.arena, writer)
    done = []
    wait_until(lambda: done.extend(reader.executor.poll_results()) or len(done) == 5)
    assert [event.op_id for event in done] == [1, 2, 3, 4, 5]


def test_a_plan_without_a_load_waits_for_none(pool, ranks):
    store(pool)
    reader = ranks()
    reader.load(7, LOADED)
    reader.run_forward()
    waits = len(reader.words.waits)

    reader.executor.submit_load_backs([], prerequisite_stream=None)

    assert reader.tracker.consumer_indices == ()
    reader.run_forward()
    assert len(reader.words.waits) == waits


def test_a_load_does_not_start_once_the_pins_may_be_gone(pool, ranks):
    store(pool)
    reader = ranks()
    reader.allowed_tickets = {2}

    reader.load(7, LOADED, ticket=1)
    # The forward is let go, with nothing read into its blocks.
    reader.run_forward()

    assert reader.engine.copies == []
    assert not reader.arena.kv.any() and not reader.arena.state.any()
    with pytest.raises(RuntimeError, match="shared L2 load failed") as failure:
        reader.executor.poll_results()
    assert "lease on the pool pages ran out" in str(failure.value.__cause__)


@pytest.mark.parametrize(
    "case", ["silent owner", "restarted owner", "past the start bound"]
)
def test_a_planned_load_starts_until_its_start_bound_from_a_pool_the_rank_left(
    pool, ranks, case
):
    """The pool's client decides: its lease on a hit, and the pool a load
    planned from the hit reads, with the owner silent or restarted."""
    owner = ServedOwner(num_providers=2, joined_providers=0, lcm_bytes=None)
    for provider in pool.view.providers:
        owner.admin.request(
            "provider",
            session=provider.session,
            base=provider.base,
            bytes=provider.lcm_blocks * pool.view.lcm_bytes,
        )
    late = case == "past the start bound"
    config = SharedL2ClientConfig.from_json(
        json.dumps(
            {
                "owner": owner.address,
                "device_names": [""],
                "request_timeout_ms": 100,
                "connect_timeout_s": 5,
                "lookup_deadline_ms": 3000,
                # The rank leaves a silent pool long before the start bound of
                # its tickets; or it stays and the start bound passes.
                "no_pool_after_s": 30 if late else 1,
                "lease_s": 0.2 if late else 30,
                "load_start_margin_s": 0.2 if late else 5,
                "load_workers": 1,
                "write_workers": 1,
                "scheduler": {"max_writes": 2, "max_write_keys": 64},
            }
        )
    )
    layout = Arena().layout
    client = SharedL2Client(
        config,
        geometry=layout.geometry,
        deployment_id="deployment",
        transfer=SharedL2Transfer(layout, MemoryEngine()),
        connection=OwnerConnection(owner.address, timeout_ms=100),
    )
    try:
        # A hit of the pool, and the load of a plan made from it whose start
        # word has not landed.
        prompt = [1, 2, 3, 4, 5]
        client.submit_write(1, *write_fields(prompt))
        wait_until(lambda: client.poll_writes() == [(1, True)])
        writer = store(pool)
        client.submit_lookups([(10, "request", *lookup_fields(prompt))])
        answered = []
        wait_until(lambda: answered.extend(client.poll_lookups()) or answered)
        ticket = answered[0][1]
        assert ticket != 0
        reader = ranks(load_pool=client.load_pool)
        reader.words.hold_writes = True
        reader.load(7, LOADED, ticket=ticket)

        if case == "restarted owner":
            # The new owner has no provider: the rank finds no pool to join.
            owner.close()
            owner = ServedOwner(address=owner.address, joined_providers=0)
        else:
            owner.stall()
        if late:
            wait_until(lambda: client.load_pool(ticket) is None)
            assert not client.no_pool
        else:
            wait_until(lambda: client.no_pool)
        reader.words.release_writes()
        reader.run_forward()

        if late:
            with pytest.raises(RuntimeError, match="shared L2 load failed") as failure:
                reader.executor.poll_results()
            assert "lease on the pool pages ran out" in str(failure.value.__cause__)
            assert reader.engine.copies == []
        else:
            assert_loaded(reader.arena, writer)
            wait_until(
                lambda: [event.op_id for event in reader.executor.poll_results()] == [7]
            )
    finally:
        client.close()
        owner.close()


def test_a_failed_read_lets_the_forward_go_and_fails_the_next_poll(pool, ranks):
    store(pool)
    reader = ranks()
    reader.engine.fail_reads = True

    reader.load(7, LOADED)
    reader.run_forward()

    with pytest.raises(RuntimeError, match="shared L2 load failed") as failure:
        reader.executor.poll_results()
    assert "read from p" in str(failure.value.__cause__)
    # It stays failed: the engine is to stop.
    with pytest.raises(RuntimeError, match="shared L2 load failed"):
        reader.executor.poll_results()


def test_a_load_is_refused_during_graph_capture(pool, ranks, monkeypatch):
    store(pool)
    reader = ranks()
    monkeypatch.setattr(executor_module, "get_is_capture_mode", lambda: True)
    with pytest.raises(RuntimeError, match="outside CUDA Graph capture"):
        reader.load(7, LOADED)
    # A plan without a load is fine there, as with a Host L2.
    reader.executor.submit_load_backs([], prerequisite_stream=None)
    assert reader.engine.copies == []


def test_a_pool_rank_runs_no_other_host_cache_op(ranks):
    reader = ranks()
    # A scheduler with a local Host L2 plans the write-back of a prompt's
    # pages in the plan after the one that computes them.
    scheduler = ts.Scheduler(
        make_config(
            num_device_pages=32,
            max_scheduled_tokens=64,
            max_batch_size=8,
            max_resident_requests=8,
            prefix_granularity=2,
            num_host_pages=65,
            disable_l2_cache=False,
            enable_l3_storage=False,
            role="fused",
            num_snapshot_pages=1,
            host_group_packing=(1,),
            max_retracted_requests=0,
            host_write_policy="write_through",
            prefill_decode_interval=0,
            prefill_decode_min_interval=0,
            storage_config=None,
            cache_groups=[ts.CacheGroupConfig("full", 2, 32, ts.AttnKind.Full)],
        )
    )
    request = ts.RequestSpec()
    request.request_id = "r"
    request.tokens = list(range(1, 8))
    scheduler.submit_requests([request])
    scheduler.next_execution_plan()
    (write_back,) = scheduler.next_execution_plan().cache
    with pytest.raises(TypeError, match="writes the pool, not a Host L2"):
        reader.executor.submit_write_backs(
            [write_back], prerequisite_stream=None, fence_stream=None
        )
    reader.executor.submit_write_backs(
        [LoadBackOp(1, [])], prerequisite_stream=None, fence_stream=None
    )
    # Nor a restore from a snapshot pool, which a pool rank does not have.
    restore = RestoreOp(
        op_id=2,
        request_id="r",
        admission_id=1,
        request_pool_index=0,
        snapshot_slot=0,
        transfers=(),
        source_tier=(),
    )
    with pytest.raises(TypeError, match="does not run RestoreOp"):
        reader.executor.submit_load_backs([restore], prerequisite_stream=None)


def test_the_executor_describes_the_proxy_tier_to_the_device_build(ranks):
    executor = ranks().executor
    # No local Host bytes, no snapshot pool; the proxy tier has room for the
    # pool twice over, packed as the pool is.
    assert executor.host_storage is None and executor.block_owners is None
    assert executor.num_host_pages == 2 * sum(PROVIDER_BLOCKS) + 1
    assert executor.host_group_packing == (1, 2)
    assert (executor.num_snapshot_pages, executor.max_retracted_requests) == (1, 0)
    assert len(executor.layout.groups) == 2


def test_a_shutdown_stops_the_loads_and_then_closes_the_pool_client(pool, ranks):
    store(pool)
    reader = ranks()
    reader.engine.read_delay_s = 0.1
    reader.load(7, LOADED)

    reader.executor.shutdown()

    # The load in flight ended first: the client's writes share its engine.
    (threads,) = reader.closed_after
    assert not [name for name in threads if name.startswith("shared-l2-")]
    assert reader.words.closed
    assert [event.op_id for event in reader.executor.poll_results()] == [7]


def pool_config() -> str:
    return json.dumps(
        {
            "owner": "tcp://127.0.0.1:1",
            "device_names": ["mlx5_0"],
            "request_timeout_ms": 1000,
            "connect_timeout_s": 60,
            "lookup_deadline_ms": 5,
            "no_pool_after_s": 10,
            "lease_s": 20,
            "load_workers": 2,
            "write_workers": 2,
            "scheduler": {"max_writes": 4, "max_write_keys": 256},
        }
    )


def test_server_args_take_a_pool_in_place_of_the_host_l2():
    args = ServerArgs(
        model="x", max_num_seqs=8, disable_kvstore=True, shared_l2_config=pool_config()
    )
    assert not args.enable_kvstore
    # The pool is written when a page is published, and its ranks never retract.
    assert args.kvstore_write_policy == "write_through"
    assert args.retraction_snapshot_pool_disabled
    assert args.retraction_snapshot_max_requests == 0

    prefill = ServerArgs(
        model="x",
        max_num_seqs=8,
        disable_kvstore=True,
        disaggregation_mode="prefill",
        shared_l2_config=pool_config(),
    )
    assert prefill.shared_l2_config == pool_config()


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"disable_kvstore": False}, "does not support the local Host L2"),
        (
            {"disable_kvstore": False, "kvstore_storage_backend": "memory"},
            "--kvstore-storage-backend",
        ),
        (
            {"disaggregation_mode": "decode"},
            "does not support --disaggregation-mode decode or encode",
        ),
        (
            {"disaggregation_mode": "encode"},
            "does not support --disaggregation-mode decode or encode$",
        ),
        (
            {"world_size": 2, "nprocs_per_node": 2},
            "does not support attention TP > 1$",
        ),
        (
            {"world_size": 2, "nprocs_per_node": 2, "pipeline_parallel_size": 2},
            "does not support --pipeline-parallel-size > 1$",
        ),
        (
            {
                "kv_offload_config": {
                    "layers": [0],
                    "hot_tokens": 1024,
                    "host_gb": 1,
                    "overlap": False,
                }
            },
            "does not support --kv-offload-config$",
        ),
        (
            {"enable_memory_saver": True, "weight_memory_policy": "cpu_backup"},
            "does not support --enable-memory-saver$",
        ),
        (
            {"kvstore_write_policy": "write_back"},
            "write_back does not support --shared-l2-config",
        ),
        ({"enable_prefix_caching": False}, "disabled prefix caching"),
        (
            {
                "kv_parallel_size": 2,
                "world_size": 2,
                "nprocs_per_node": 2,
                "disaggregation_mode": "prefill",
            },
            r"--kv-parallel-size > 1 \(the first version of the pool holds "
            "replicated KV only\\)",
        ),
        ({"retraction_snapshot_ratio": 0.5}, "does not retract"),
        ({"retraction_snapshot_max_requests": 4}, "does not retract"),
        ({"debug_force_retraction_interval": 2}, "does not retract"),
        (
            {"enable_startup_requests": True},
            "--enable-startup-requests is not supported with --shared-l2-config$",
        ),
        ({"shared_l2_config": '{"owner": "tcp://127.0.0.1:1"}'}, "needs exactly"),
    ],
)
def test_server_args_refuse_what_a_pool_does_not_support(settings, message):
    fields = dict(
        model="x", max_num_seqs=8, disable_kvstore=True, shared_l2_config=pool_config()
    )
    fields.update(settings)
    with pytest.raises(ValueError, match=message):
        ServerArgs(**fields)


def test_server_args_join_no_pool_by_default():
    args = ServerArgs(model="x", max_num_seqs=8)
    assert args.shared_l2_config is None and args.enable_kvstore


# The tests below run the real ready words and CUDA streams.


def slow_reader(pool, ranks, *, device):
    """A reader whose reads take long enough to still run when the forward
    reaches its waits."""
    engine = CudaEngine() if device == "cuda" else MemoryEngine()
    engine.read_delay_s = 0.3
    return ranks(
        device=device,
        engine=engine,
        words=lambda count: HostSignals(torch.cuda.current_device(), count),
    )


@needs_cuda
def test_gpu_bytes_loaded_layer_by_layer_equal_the_bytes_written(pool, ranks):
    writer = store(pool)
    reader = slow_reader(pool, ranks, device="cuda")

    reader.load(7, LOADED, stream=torch.cuda.current_stream())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        reader.tracker.wait_for_layer(0)
        # Behind the first layer's wait, its field is there.
        first = reader.arena.kv.clone()
        reader.tracker.wait_for_layer(1)
    stream.synchronize()

    block = LOADED[0][2]
    assert bytes(first[block, :12].tolist()) == writer.fields(KV, STORED[0][2])[0]
    assert_loaded(reader.arena, writer)
    wait_until(lambda: [event.op_id for event in reader.executor.poll_results()] == [7])


@needs_cuda
def test_gpu_a_load_completes_while_the_forward_holds_the_driver(pool, ranks, capfd):
    """Loading a kernel on first use holds the CUDA driver until the GPU
    drains: a forward that does so while its stream waits on a load needs the
    loader to release that wait without the driver. A deadlock ends the
    process."""
    writer = store(pool)
    # Host cache memory: the reader's copies make no CUDA call either.
    reader = slow_reader(pool, ranks, device="cpu")
    # Kernels this process has not run, so the product below loads them.
    a = torch.empty(333, 517, dtype=torch.float64, device="cuda")
    b = torch.empty(517, 211, dtype=torch.float64, device="cuda")

    reader.load(7, LOADED, stream=torch.cuda.current_stream())
    stream = torch.cuda.Stream()
    # Uncaptured, so the watchdog's report survives the exit.
    with capfd.disabled():
        faulthandler.dump_traceback_later(60, exit=True)
        try:
            with torch.cuda.stream(stream):
                reader.wait_for_layers()
                torch.mm(a, b)
            stream.synchronize()
        finally:
            faulthandler.cancel_dump_traceback_later()

    assert_loaded(reader.arena, writer)


@needs_cuda
def test_gpu_the_completion_event_waits_for_the_whole_load(pool, ranks):
    writer = store(pool)
    reader = slow_reader(pool, ranks, device="cuda")

    reader.load(7, LOADED, stream=torch.cuda.current_stream())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # What a graph replay does: no wait of its own on any layer.
        reader.tracker.wait_for_all_layers()
        seen = Arena("cuda")
        seen.copy_from(reader.arena)
    stream.synchronize()

    assert_loaded(seen, writer)
    # The next load on these words finds the event done.
    for op_id in (8, 9, 10):
        reader.load(op_id, LOADED, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert_loaded(reader.arena, writer)


@needs_cuda
def test_gpu_a_prefill_graph_replay_waits_for_its_load(pool, ranks):
    writer = store(pool)
    reader = slow_reader(pool, ranks, device="cuda")
    seen = Arena("cuda")
    torch.cuda.synchronize()

    # A captured segment reads loaded pages; the capture notes the layer it
    # reaches instead of waiting.
    capture = BreakableCapture()
    with capture:
        reader.tracker.wait_for_layer(len(CONSUMERS) - 1)
        seen.copy_from(reader.arena)
    assert [bool(waits) for waits in capture.segment_waits] == [True]

    # Every replay waits for its own plan's load before the segment.
    for op_id in (7, 8):
        reader.arena.zero()
        seen.zero()
        reader.load(op_id, LOADED, stream=torch.cuda.current_stream())
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            capture.replay()
        stream.synchronize()
        assert_loaded(seen, writer)
