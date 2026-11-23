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

"""The data path of a shared L2 pool on a rank's Python side: where the rank's
blocks are and how a pool page holds one, and the forward thread's part of a
load, which hands the load to the engine host.

The host's loader reads the pool's pages (``shared_l2/load.rs`` of
``ts-control``, tested there over test memory). Here a stand-in takes its
place: it copies a writer's blocks layer by layer and sets the ready words.
The CPU tests keep the signal words in plain memory; the tests that need a GPU
run the real words and CUDA streams."""

import ctypes
import faulthandler
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from test.runtime.shared_l2_test_utils import wait_until  # noqa: E402

import tokenspeed_engine.scheduler as ts  # noqa: E402
from tokenspeed_kernel.platform import current_platform  # noqa: E402

from tokenspeed.runtime.cache.host_storage import (  # noqa: E402
    compute_host_group_packing,
    compute_host_lcm_block_bytes,
)
from tokenspeed.runtime.cache.l2.layerwise_load import (  # noqa: E402
    LayerwiseLoadTracker,
)
from tokenspeed.runtime.cache.shared_l2 import executor as executor_module  # noqa: E402
from tokenspeed.runtime.cache.shared_l2.client import (  # noqa: E402
    SharedL2PoolView,
    SharedL2Provider,
)
from tokenspeed.runtime.cache.shared_l2.executor import (  # noqa: E402
    HostSignals,
    SharedL2LoadExecutor,
)
from tokenspeed.runtime.cache.shared_l2.layout import (  # noqa: E402
    SharedL2Layout,
    SharedL2PageField,
    build_shared_l2_layout,
)
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

# The signal words are CUDA stream memory operations, called in the NVIDIA
# driver library (``HostSignals``): torch.cuda alone is there on AMD too.
needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_nvidia,
    reason="runs CUDA streams and their memory waits in the NVIDIA driver library",
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
# A pool of 8 LCM blocks on two providers, as a hello describes it.
PROVIDER_BLOCKS = (3, 5)
POOL = SharedL2PoolView(
    client=1,
    providers=(
        SharedL2Provider("p0", 1 << 40, 3, 0),
        SharedL2Provider("p1", 1 << 41, 5, 3),
    ),
    lcm_bytes=28,
)


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


def memory_copy(destination: int, source: int, size: int) -> None:
    """A copy between host addresses."""
    ctypes.memmove(destination, source, size)


def cuda_copy(destination: int, source: int, size: int) -> None:
    """A copy between CUDA memory and host memory, from any thread."""
    torch.cuda.set_device(0)
    result = ctypes.CDLL("libcuda.so.1").cuMemcpy(
        ctypes.c_uint64(destination), ctypes.c_uint64(source), ctypes.c_size_t(size)
    )
    assert result == 0


class StandInHandle:
    """Stands in for the host's shared L2 handle (``SharedL2Handle``), which
    runs a rank's loads in a server: one thread takes the loads in the order
    they were handed over, waits for a load's start word, copies each layer's
    fields and sets the layer's ready word, as the host's loader does from
    the pool. A proxy Host block here stands for the writer's Device block of
    the same id."""

    def __init__(self, writer, reader, *, copy):
        self._writer, self._reader, self._copy = writer, reader, copy
        # The rank's words, once its executor made them.
        self.words = None
        # Every load handed over: (op_ids, rows, start_word, seq).
        self.loads = []
        # How long every layer's copies take.
        self.layer_delay_s = 0.0
        # What a hand-over raises, once set: the loader refuses the load.
        self.refusal = None
        # Whether the close reports that every thread ended.
        self.stops = True
        self._closing = False
        self.closed = False
        # Called when the close comes, before anything is stopped.
        self.before_close = lambda: None
        self._jobs = ThreadPoolExecutor(1, thread_name_prefix="stand-in-load")

    def submit_load(self, *, op_ids, rows, start_word, seq):
        if self.refusal is not None:
            raise RuntimeError(self.refusal)
        self.loads.append((list(op_ids), list(rows), start_word, seq))
        self._jobs.submit(self._load, list(rows), start_word, seq)

    def _load(self, rows, start_word, seq):
        words = self.words.words
        # The start word lands once the plan's page zeroing is done; a load
        # still waiting for it at the close is given up.
        while (int(words[start_word]) - seq) & 0xFFFFFFFF >= 0x80000000:
            if self._closing:
                return
            time.sleep(20e-6)
        for layer in range(len(CONSUMERS)):
            time.sleep(self.layer_delay_s)
            for group, source, destination in rows:
                for written, loaded in zip(
                    self._writer.layout.page_fields(group),
                    self._reader.layout.page_fields(group),
                    strict=True,
                ):
                    if loaded.layer == layer:
                        self._copy(
                            loaded.device_address + destination * loaded.device_stride,
                            written.device_address + source * written.device_stride,
                            loaded.bytes,
                        )
            words[start_word + 1 + layer] = seq

    def close(self):
        self.before_close()
        self._closing = True
        self._jobs.shutdown(wait=True)
        self.closed = True
        return self.stops


class PlainSignals:
    """The start and ready words in plain host memory. A stream write lands
    when the test lets it; a stream wait is kept, and ``drain`` returns once
    every wait kept so far on a stream would have passed, as synchronizing
    that stream does."""

    def __init__(self, num_words):
        self.words = np.zeros(num_words, dtype=np.uint32)
        self.address = self.words.ctypes.data
        # (stream, word, value) of every wait, and of every write.
        self.waits = []
        self.writes = []
        self.hold_writes = False
        self._held = []
        self.closed = False

    def write(self, stream, word, value):
        self.writes.append((stream, word, value))
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
    """A rank's load path: its arena, its tracker and executor, and the
    stand-in for the host's shared L2 handle."""

    def __init__(self, writer, *, device="cpu", words=PlainSignals, copy=memory_copy):
        self.arena = Arena(device)
        self.tracker = LayerwiseLoadTracker(len(CONSUMERS))
        self.shared_l2_handle = StandInHandle(writer, self.arena, copy=copy)
        self.executor = SharedL2LoadExecutor(
            self.arena.layout,
            self.shared_l2_handle,
            signals=words,
            trackers=[(self.tracker, len(CONSUMERS))],
            pool=POOL,
        )
        self.words = self.executor._signals
        self.shared_l2_handle.words = self.words
        # The stream the forward's layer waits go to.
        self.forward = executor_module.device_module.current_stream()

    def load(self, op_id, rows, *, stream=None):
        """Submit the load of ``(group, proxy Host block, Device block)``
        rows."""
        if stream is None:
            stream = self.forward
        self.executor.submit_load_backs(
            [LoadBackOp(op_id, rows)], prerequisite_stream=stream
        )

    def wait_for_layers(self):
        """The forward's waits, on the current stream."""
        for layer in range(len(CONSUMERS)):
            self.tracker.wait_for_layer(layer)

    def run_forward(self):
        """The forward's waits on plain words: returns once all have passed."""
        self.wait_for_layers()
        self.words.drain(self.forward)


# The Device blocks a writer holds, of both groups: (group, block).
STORED = [(KV, 3), (KV, 5), (KV, 1), (STATE, 4), (STATE, 8), (STATE, 2)]
# A reader loads each into another Device block, through the proxy Host block
# of the writer block's id: (group, proxy Host block, Device block).
LOADED = [(group, block, BLOCKS - block) for group, block in STORED]


@pytest.fixture
def writer():
    """A rank whose blocks hold what a reader loads."""
    arena = Arena()
    for group, block in STORED:
        arena.fill(group, block, seed=10 * block + group)
    return arena


@pytest.fixture
def ranks(writer, monkeypatch):
    """Builds ranks that load the writer's blocks; their executors are shut
    down afterwards."""
    monkeypatch.setattr(executor_module.Cache, "LoadBackOp", LoadBackOp)
    built = []

    def build(**settings) -> Rank:
        built.append(Rank(writer, **settings))
        return built[-1]

    yield build
    for rank in built:
        if not rank.shared_l2_handle.closed:
            rank.executor.shutdown()


def assert_loaded(reader: Arena, writer: Arena) -> None:
    for group, written, loaded in LOADED:
        assert reader.fields(group, loaded) == writer.fields(group, written)


def test_a_pool_is_laid_out_as_a_ranks_host_tier():
    layout = Arena().layout
    transfer = layout.transfer
    lcm_bytes = layout.geometry["lcm_bytes"]
    assert lcm_bytes == compute_host_lcm_block_bytes(transfer, group_specs=SPECS)
    assert layout.host_packing == compute_host_group_packing(
        transfer, group_specs=SPECS
    )
    assert (lcm_bytes, layout.host_packing) == (28, (1, 2))
    assert (layout.block_bytes, layout.field_offsets) == ((28, 12), ((0, 12), (0,)))


def test_the_host_gets_every_field_of_a_group_in_page_order():
    arena = Arena()
    kv, state = arena.kv.data_ptr(), arena.state.data_ptr()
    assert arena.layout.page_fields(KV) == (
        SharedL2PageField(
            device_address=kv, device_stride=32, bytes=12, page_offset=0, layer=0
        ),
        SharedL2PageField(
            device_address=kv + 16, device_stride=32, bytes=16, page_offset=12, layer=1
        ),
    )
    assert arena.layout.page_fields(STATE) == (
        SharedL2PageField(
            device_address=state, device_stride=16, bytes=12, page_offset=0, layer=1
        ),
    )


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
    assert (layout.block_bytes, layout.geometry["lcm_bytes"], layout.host_packing) == (
        (34, 12),
        34,
        (1, 2),
    )
    assert layout.field_offsets[KV] == (0, 12, 28)
    assert layout.registration_regions() == ((arena.data_ptr(), arena.nbytes),)
    # The draft's field joins the group's page, and its layer follows the
    # target's.
    assert layout.page_fields(KV)[2] == SharedL2PageField(
        device_address=arena.data_ptr() + 48,
        device_stride=64,
        bytes=6,
        page_offset=28,
        layer=2,
    )


def test_a_plans_load_is_handed_to_the_host_behind_its_start_word(writer, ranks):
    reader = ranks()
    reader.words.hold_writes = True

    reader.load(7, LOADED)

    # The host got the load's operation, its rows, the first of its words and
    # the value that marks it there.
    ((op_ids, rows, start_word, seq),) = reader.shared_l2_handle.loads
    assert (op_ids, rows, start_word, seq) == ([7], LOADED, 0, 1)
    # The start word is written on the stream the plan's page zeroing ran on:
    # it has not landed, the zeroing may still be running, and nothing is
    # loaded.
    assert reader.words.writes == [(reader.forward, start_word, seq)]
    time.sleep(0.05)
    assert not reader.arena.kv.any() and not reader.arena.state.any()

    reader.words.release_writes()
    reader.run_forward()
    assert_loaded(reader.arena, writer)


def test_a_load_the_host_refuses_puts_no_wait_on_a_stream(ranks):
    reader = ranks()
    reader.shared_l2_handle.refusal = (
        "a shared L2 load reads proxy Host block 99 of group 0, which stands "
        "for no pool page"
    )

    with pytest.raises(RuntimeError, match="stands for no pool page"):
        reader.load(7, [(KV, 99, 1)])

    # Neither a start word nor a wait: no stream is left waiting on a word
    # that nobody sets.
    assert reader.words.writes == [] and reader.words.waits == []


def test_the_forwards_layers_wait_for_their_own_words(writer, ranks):
    reader = ranks()
    reader.shared_l2_handle.layer_delay_s = 0.05

    reader.load(7, LOADED)
    reader.tracker.wait_for_layer(0)
    reader.words.drain(reader.forward)

    # The first layer's wait passed with its field loaded and no other.
    ((_, first_ready_word, seq),) = [
        wait for wait in reader.words.waits if wait[0] == reader.forward
    ]
    assert reader.words.passed(first_ready_word, seq)
    assert not reader.words.passed(first_ready_word + 1, seq)
    block = LOADED[0][2]
    assert reader.arena.fields(KV, block)[0] == writer.fields(KV, STORED[0][1])[0]
    assert not reader.arena.kv[:, 16:].any() and not reader.arena.state.any()

    reader.run_forward()
    assert_loaded(reader.arena, writer)


def test_the_whole_load_is_one_event_behind_the_last_layers_word(ranks):
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


def test_consecutive_loads_take_turns_on_the_words(writer, ranks):
    reader = ranks()
    for op_id in range(1, 6):
        reader.arena.zero()
        reader.load(op_id, LOADED)
        reader.run_forward()
        assert_loaded(reader.arena, writer)
    # Each load has a value of its own, on the word set its tracker gave it:
    # a start word and a ready word per layer.
    width = len(CONSUMERS) + 1
    sets = len(reader.tracker.event_sets)
    assert [seq for _, _, _, seq in reader.shared_l2_handle.loads] == [1, 2, 3, 4, 5]
    start_words = {start_word for _, _, start_word, _ in reader.shared_l2_handle.loads}
    assert len(start_words) > 1
    assert start_words <= {index * width for index in range(sets)}
    # The host's loader reads and sets them where the executor says.
    assert reader.executor.signal_words == (reader.words.address, sets * width)


def test_a_plan_without_a_load_waits_for_none(ranks):
    reader = ranks()
    reader.load(7, LOADED)
    reader.run_forward()
    waits = len(reader.words.waits)

    reader.executor.submit_load_backs([], prerequisite_stream=None)

    assert reader.tracker.consumer_indices == ()
    reader.run_forward()
    assert len(reader.words.waits) == waits


def test_a_load_is_refused_during_graph_capture(ranks, monkeypatch):
    reader = ranks()
    monkeypatch.setattr(executor_module, "get_is_capture_mode", lambda: True)
    with pytest.raises(RuntimeError, match="outside CUDA Graph capture"):
        reader.load(7, LOADED)
    # A plan without a load is fine there, as with a Host L2.
    reader.executor.submit_load_backs([], prerequisite_stream=None)
    assert reader.shared_l2_handle.loads == []


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


@pytest.fixture
def synchronized(monkeypatch):
    """Every synchronization of the device's streams by the executor."""
    calls = []
    monkeypatch.setattr(
        executor_module.device_module, "synchronize", lambda: calls.append(True)
    )
    return calls


def test_a_shutdown_closes_the_hosts_client_and_then_frees_the_words(
    ranks, synchronized
):
    reader = ranks()
    handle = reader.shared_l2_handle
    handle.layer_delay_s = 0.1
    # Whether the words were freed when the close came.
    words_closed_at_close = []
    handle.before_close = lambda: words_closed_at_close.append(reader.words.closed)
    reader.load(7, LOADED)

    reader.executor.shutdown()

    # The load in flight ended first: it sets the words, which the streams
    # then pass before the words are freed.
    assert handle.closed and words_closed_at_close == [False]
    assert synchronized == [True]
    assert reader.words.closed


def test_a_shutdown_leaves_the_words_a_thread_that_still_runs_uses(ranks, synchronized):
    reader = ranks()
    # A thread of the host's client did not end in time: it may set a word
    # later, and a stream may wait on a word nobody sets any more.
    reader.shared_l2_handle.stops = False

    reader.executor.shutdown()

    # Neither the streams nor the words are touched.
    assert reader.shared_l2_handle.closed
    assert synchronized == [] and not reader.words.closed


def pool_config() -> str:
    return json.dumps(
        {
            "owner": "tcp://127.0.0.1:1",
            "device_names": ["mlx5_0"],
            "request_timeout_ms": 1000,
            "connect_timeout_s": 60,
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


# The tests below run the real signal words and CUDA streams.


def slow_reader(ranks, *, device):
    """A reader whose load takes long enough to still run when the forward
    reaches its waits."""
    reader = ranks(
        device=device,
        copy=cuda_copy if device == "cuda" else memory_copy,
        words=lambda count: HostSignals(torch.cuda.current_device(), count),
    )
    reader.shared_l2_handle.layer_delay_s = 0.3
    return reader


@needs_cuda
def test_gpu_bytes_loaded_layer_by_layer_equal_the_bytes_written(writer, ranks):
    reader = slow_reader(ranks, device="cuda")

    reader.load(7, LOADED, stream=torch.cuda.current_stream())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        reader.tracker.wait_for_layer(0)
        # Behind the first layer's wait, its field is there.
        first = reader.arena.kv.clone()
        reader.tracker.wait_for_layer(1)
    stream.synchronize()

    block = LOADED[0][2]
    assert bytes(first[block, :12].tolist()) == writer.fields(KV, STORED[0][1])[0]
    assert_loaded(reader.arena, writer)


@needs_cuda
def test_gpu_a_load_completes_while_the_forward_holds_the_driver(writer, ranks, capfd):
    """Loading a kernel on first use holds the CUDA driver until the GPU
    drains: a forward that does so while its stream waits on a load needs the
    loader to release that wait without the driver. A deadlock ends the
    process."""
    # Host cache memory: the reader's copies make no CUDA call either.
    reader = slow_reader(ranks, device="cpu")
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
def test_gpu_the_completion_event_waits_for_the_whole_load(writer, ranks):
    reader = slow_reader(ranks, device="cuda")

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
def test_gpu_a_prefill_graph_replay_waits_for_its_load(writer, ranks):
    reader = slow_reader(ranks, device="cuda")
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
