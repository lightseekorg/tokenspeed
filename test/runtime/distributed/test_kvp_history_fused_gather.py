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

"""KVP prefix-history reads through the fused all-gather on GB300: the real
``PrefixHistoryGather``, the real NCCL all-gather on the named KVP history
groups, the real fused transport built by the startup function, and the MLA
pool's own latent read.

Every case reads a forward through an NCCL reader and a fused reader of the
same rank-sharded fp8 MLA planes (every byte value, NaN and denormal patterns
included) and requires both outputs bitwise equal to the rows the planes
hold. Every verdict that one rank could reach alone is agreed over the world
first, so that a failing case fails on every rank and the next case's
collectives still line up.

``_worker`` runs every case on an initialized world; the test spawns it on two
and four local GPUs. Run as a script under ``torchrun`` (eight ranks across
two nodes), the same worker adds the case with real all-reduces in flight and
skips the single-node ones.
"""

import gc
import os
import random
import sys
import threading
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=600, suite="runtime-2gpu")

from test.runtime.test_kvp_prefix_chunk_gather import (  # noqa: E402
    _chunk_slots,
    _CountingBackend,
    _mla_model_config,
    _random_layout,
    _startup_args,
    _StubLatentPool,
    _Work,
)

# A MLA latent row: 512 latent and 64 RoPE elements (fp8: 576 bytes).
ROW = 576
FP8 = torch.float8_e4m3fn
PAGE_SIZE = 64
SUBPAGES = 2
GRANULARITY = PAGE_SIZE * SUBPAGES
# The startup's process-group timeout: in-kernel waits end after 10 s.
TIMEOUT_S = 20
# The process groups' own timeout (NCCL watchdog).
PG_TIMEOUT_S = 300
LATENT_LAYERS = (1, 3, 4)
NUM_LAYERS = 6
# ~0.5 s of device sleep: a load, a kernel or a stream held back while reads
# enqueue.
HOLD_CYCLES = 1_000_000_000


class _LatentPool(_StubLatentPool):
    """The stub pool reading rows through the MLA pool's own latent read
    (``get_mla_kv_buffer_triton``)."""

    def __init__(self, planes: list[torch.Tensor | None]):
        super().__init__(planes)
        self.kv_lora_rank = 512
        self.qk_rope_head_dim = 64

    def read_latent_rows(self, rows, loc, dst_dtype):
        from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool

        return MLATokenToKVPool.read_latent_rows(self, rows, loc, dst_dtype)


@dataclass
class _World:
    rank: int
    device: torch.device
    group: tuple[int, ...]
    mapping: object
    comm: object
    gloo: object
    one_node: bool


@dataclass
class _Layout:
    """One forward's prefix loops over a host kernel-page table."""

    table: torch.Tensor
    prefix_lens: list[int]
    virtual_block_count: int
    chunk_len: int
    chunks: list[torch.Tensor]
    every_slot: torch.Tensor


def _agree(world: _World, failure: str | None, what: str) -> None:
    """Fail on every rank when any rank failed (``failure``: this rank's
    reason, None when fine)."""
    flag = torch.tensor([0 if failure else 1], dtype=torch.int32, device=world.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    assert int(flag) == 1, f"{what} failed on some rank; this rank: {failure or 'ok'}"


def _layout(
    device,
    *,
    prefix_lens,
    chunk_len,
    virtual_block_count=4096,
    blocks=None,
    seed=0,
):
    """Random distinct blocks per request (``_random_layout``), or the given
    ``blocks`` of one request in order."""
    if blocks is None:
        table, slots = _random_layout(
            torch.Generator().manual_seed(seed),
            prefix_lens=prefix_lens,
            page_size=PAGE_SIZE,
            subpages=SUBPAGES,
            virtual_block_count=virtual_block_count,
        )
    else:
        (length,) = prefix_lens
        pages = [block * SUBPAGES + sub for block in blocks for sub in range(SUBPAGES)]
        table = torch.zeros((1, len(pages) + 3), dtype=torch.int32)
        table[0, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
        slots = [
            [pages[p // PAGE_SIZE] * PAGE_SIZE + p % PAGE_SIZE for p in range(length)]
        ]
    chunks = [
        torch.tensor(chunk, dtype=torch.int32, device=device)
        for chunk in _chunk_slots(slots, chunk_len)
    ]
    return _Layout(
        table=table,
        prefix_lens=list(prefix_lens),
        virtual_block_count=virtual_block_count,
        chunk_len=chunk_len,
        chunks=chunks,
        every_slot=torch.cat(chunks).to(torch.int64),
    )


def _layouts(device):
    return {
        # One long request on consecutive blocks.
        "sequential": _layout(
            device, prefix_lens=[6000], chunk_len=512, blocks=list(range(1, 48))
        ),
        "fragmented": _layout(device, prefix_lens=[5000, 3000, 700], chunk_len=512),
        # bs > 1, short segments, a request without history.
        "small_segments": _layout(
            device, prefix_lens=[1000, 37, 513, 129, 0, 77], chunk_len=50, seed=3
        ),
        # One window a layer, eight, and a last window of two loops.
        "one_window": _layout(device, prefix_lens=[2000], chunk_len=2048, seed=4),
        "eight_windows": _layout(device, prefix_lens=[4096], chunk_len=128, seed=5),
        "partial_window": _layout(device, prefix_lens=[4096], chunk_len=300, seed=6),
    }


def _payload(layer_id: int, slots: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``[n, 1, ROW]`` rows of ``dtype`` for the virtual ``slots`` of a layer:
    every byte value (NaN and denormal patterns included), distinct per
    layer, slot and byte."""
    width = ROW * torch.empty((), dtype=dtype).element_size()
    byte = torch.arange(width, device=slots.device)
    mixed = (
        slots.to(torch.int64).unsqueeze(1) * 1_000_003
        + layer_id * 7919
        + byte * 2_654_435_761
    )
    rows = ((mixed ^ (mixed >> 13)) & 0xFF).to(torch.uint8)
    return rows.view(dtype).unsqueeze(1)


def _placement(world: _World, layout: _Layout):
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

    return CachePlacement(
        block_granularity=GRANULARITY,
        virtual_block_count=layout.virtual_block_count,
        group=world.group,
        rank=world.rank,
    )


def _planes(world: _World, layout: _Layout, *, dtype, payload_layer_offset):
    """This rank's ``dtype`` latent planes: the payload of every history row
    it owns (of layer ``id + payload_layer_offset``: another layer's bytes
    stand for rows a Host load has not brought yet), zero elsewhere."""
    from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots

    local, owned = resolve_cache_slots(layout.every_slot, _placement(world, layout))
    local_rows = (
        (layout.virtual_block_count - 2) // len(world.group) + 2
    ) * GRANULARITY
    width = ROW * torch.empty((), dtype=dtype).element_size()
    planes: list[torch.Tensor | None] = [None] * NUM_LAYERS
    for layer_id in LATENT_LAYERS:
        plane = torch.zeros(
            (local_rows, 1, width), dtype=torch.uint8, device=world.device
        )
        plane[local[owned]] = _payload(
            layer_id + payload_layer_offset, layout.every_slot[owned], dtype
        ).view(torch.uint8)
        planes[layer_id] = plane.view(dtype)
    return planes, local, owned


def _reader(world: _World, layout: _Layout, transport):
    from tokenspeed.runtime.layers.attention.dcp.cache import PrefixHistoryGather

    return PrefixHistoryGather(
        _placement(world, layout),
        page_size=PAGE_SIZE,
        comm=world.comm,
        transport=transport,
    )


def _prepare(history, layout: _Layout) -> None:
    history.prepare(
        layout.chunks,
        page_table_cpu=layout.table,
        prefix_lens_cpu=torch.tensor(layout.prefix_lens),
        chunk_len=layout.chunk_len,
    )


def _read(history, pool, layout: _Layout, layer_id: int):
    return history.read(
        pool,
        SimpleNamespace(layer_id=layer_id),
        layout.chunks,
        dst_dtype=torch.bfloat16,
    )


def _forward(history, pool, layout: _Layout, *, between):
    """One forward reading every latent layer's loops; ``between`` (or None)
    runs after every loop the reader yields."""
    _prepare(history, layout)
    out = {}
    for layer_id in LATENT_LAYERS:
        out[layer_id] = []
        for pair in _read(history, pool, layout, layer_id):
            out[layer_id].append(pair)
            if between is not None:
                between(layer_id)
    torch.cuda.synchronize()
    return out


def _expected(pool, layout: _Layout, *, dtype):
    return {
        layer_id: [
            pool.read_latent_rows(
                _payload(layer_id, chunk, dtype),
                torch.arange(chunk.numel(), dtype=torch.int32, device=chunk.device),
                torch.bfloat16,
            )
            for chunk in layout.chunks
        ]
        for layer_id in LATENT_LAYERS
    }


def _assert_rows(world: _World, got, want, what: str) -> None:
    failure = None
    if got.keys() != want.keys():
        failure = f"layers {sorted(got)} != {sorted(want)}"
    for layer_id in want:
        if failure is not None:
            break
        if len(got[layer_id]) != len(want[layer_id]):
            failure = f"layer {layer_id}: {len(got[layer_id])} loops"
            break
        for loop, (pair, expected) in enumerate(zip(got[layer_id], want[layer_id])):
            if not all(
                torch.equal(part.view(torch.int16), other.view(torch.int16))
                for part, other in zip(pair, expected)
            ):
                failure = f"layer {layer_id} loop {loop}: bytes differ"
                break
    _agree(world, failure, what)


def _assert_clean(world: _World, transport) -> None:
    """No error record, and the same epoch on every rank."""
    epochs = [
        torch.zeros(1, dtype=torch.int64, device=world.device) for _ in world.group
    ]
    dist.all_gather(epochs, torch.tensor([transport.state.epoch], device=world.device))
    error = transport.state.read_error()
    seen = sorted({int(e) for e in epochs})
    failure = None
    if error is not None:
        failure = f"error record {error}"
    elif len(seen) != 1:
        failure = f"epochs {seen}"
    _agree(world, failure, "the fused state's error record and epochs")


def _own_transport(world: _World, *, cap_rows: int, modes):
    """A transport on a state of its own (fresh zero slots, its own epochs):
    collective, every rank alike."""
    from tokenspeed_kernel.ops.communication.cute_dsl import (
        create_paged_row_all_gather_state,
    )

    from tokenspeed.runtime.layers.attention.dcp.cache import FusedHistoryTransport

    state = create_paged_row_all_gather_state(
        world.gloo,
        world.device,
        row_bytes=ROW,
        cap_bytes=cap_rows * ROW,
        timeout_ns=(TIMEOUT_S // 2) * 10**9,
        modes=modes,
        eligible=True,
    )
    _agree(world, None if state is not None else "no state", "an own state")
    return FusedHistoryTransport(state)


def _nccl_then_fused(world: _World, layout: _Layout, transport):
    """Read a forward through an NCCL reader and a fused one; both must be the
    planes' rows, and every fused window takes one epoch."""
    planes, _, _ = _planes(world, layout, dtype=FP8, payload_layer_offset=0)
    pool = _LatentPool(planes)
    want = _expected(pool, layout, dtype=FP8)
    got = _forward(_reader(world, layout, None), pool, layout, between=None)
    _assert_rows(world, got, want, "the NCCL reader")
    history = _reader(world, layout, transport)
    epoch = transport.state.epoch
    got = _forward(history, pool, layout, between=None)
    _assert_rows(world, got, want, "the fused reader")
    assert transport.state.epoch - epoch == len(history.windows) * len(LATENT_LAYERS)
    _assert_clean(world, transport)
    return history, pool


def _case_startup(world: _World):
    """The startup function refuses on every rank alike -- a configured
    kernel that could deadlock with a fused window, a decode role, an
    allocation failure on one rank -- and the group's readers then gather on
    NCCL; otherwise it builds the group's transport once, and the readers
    gather through it."""
    from tokenspeed.runtime.configs.model_config import AttentionArch
    from tokenspeed.runtime.layers.attention.dcp.cache import (
        _FUSED_HISTORY_TRANSPORTS,
        PrefixHistoryGather,
        prepare_prefix_history_transport,
    )

    layout = _layouts(world.device)["fragmented"]
    planes, _, _ = _planes(world, layout, dtype=FP8, payload_layer_offset=0)
    pool = _LatentPool(planes)
    want = _expected(pool, layout, dtype=FP8)

    def startup(**overrides):
        prepare_prefix_history_transport(
            _startup_args(world.mapping, timeout_s=TIMEOUT_S, **overrides),
            _mla_model_config(AttentionArch.MLA),
            [],
            world.device,
        )

    for overrides in (
        {"moe_backend": "mega_moe"},
        {"draft_moe_backend": "a_plugin_moe", "speculative_algorithm": "EAGLE3"},
        {"all2all_backend": "deepep"},
        {"disaggregation_mode": "decode"},
    ):
        startup(**overrides)
        _agree(
            world,
            None if world.group not in _FUSED_HISTORY_TRANSPORTS else "built",
            f"the refusal of {overrides}",
        )
    if world.rank == 1:
        with patch.object(symm_mem, "empty", side_effect=RuntimeError("no memory")):
            startup()
    else:
        startup()
    _agree(
        world,
        None if world.group not in _FUSED_HISTORY_TRANSPORTS else "built",
        "an allocation failure on one rank",
    )
    refused = PrefixHistoryGather.on_history_groups(
        _placement(world, layout), page_size=PAGE_SIZE
    )
    _assert_rows(
        world,
        _forward(refused, pool, layout, between=None),
        want,
        "a refused group's reader",
    )

    startup()
    transport = _FUSED_HISTORY_TRANSPORTS.get(world.group)
    _agree(world, None if transport is not None else "not built", "the startup")
    epoch = transport.state.epoch
    startup()
    _agree(
        world,
        (
            None
            if _FUSED_HISTORY_TRANSPORTS[world.group] is transport
            and transport.state.epoch == epoch
            else "rebuilt"
        ),
        "a repeated startup",
    )
    reader = PrefixHistoryGather.on_history_groups(
        _placement(world, layout), page_size=PAGE_SIZE
    )
    got = _forward(reader, pool, layout, between=None)
    _assert_rows(world, got, want, "the group's reader")
    assert transport.state.epoch - epoch == len(reader.windows) * len(LATENT_LAYERS)
    _assert_clean(world, transport)
    return transport


def _case_layouts(world: _World, transport):
    expected_windows = {"one_window": 1, "eight_windows": 8, "partial_window": 4}
    for name, layout in _layouts(world.device).items():
        history, _ = _nccl_then_fused(world, layout, transport)
        if name in expected_windows:
            assert len(history.windows) == expected_windows[name]
        if name == "partial_window":
            assert len(history.windows[-1].loop_bounds) - 1 == 2


def _case_late_ranks(world: _World, transport):
    """Late consumers on half the ranks: a device sleep after every loop they
    read, which also delays the ready event of their next launch."""

    def late(_layer_id):
        if world.rank % 2 == 0:
            torch.cuda._sleep(2_000_000)

    for name in ("fragmented", "eight_windows"):
        layout = _layouts(world.device)[name]
        planes, _, _ = _planes(world, layout, dtype=FP8, payload_layer_offset=0)
        pool = _LatentPool(planes)
        want = _expected(pool, layout, dtype=FP8)
        got = _forward(_reader(world, layout, transport), pool, layout, between=late)
        _assert_rows(world, got, want, f"late ranks, {name}")
    _assert_clean(world, transport)


def _case_closed_after_every_yield(world: _World, transport):
    """A target read closed after any yield drops its windows: the draft reads
    next with no target prepare in between, and the target's next forward
    reads in full. Older reads never touch a newer forward: one resumed after
    the next prepare raises, one collected after it does nothing, and one
    closed on another thread cleans up nothing (the next prepare drains it)."""
    layout = _layout(world.device, prefix_lens=[1200, 300], chunk_len=128, seed=7)
    planes, _, _ = _planes(world, layout, dtype=FP8, payload_layer_offset=0)
    pool = _LatentPool(planes)
    want = _expected(pool, layout, dtype=FP8)
    target = _reader(world, layout, transport)
    draft = _reader(world, layout, transport)
    loops = len(layout.chunks)
    for layer_index, layer_id in enumerate(LATENT_LAYERS):
        for taken in range(1, loops + 1):
            _prepare(target, layout)
            for earlier in LATENT_LAYERS[:layer_index]:
                assert len(list(_read(target, pool, layout, earlier))) == loops
            reading = _read(target, pool, layout, layer_id)
            for _ in range(taken):
                next(reading)
            reading.close()
            _prepare(draft, layout)
            got = {
                every: list(_read(draft, pool, layout, every))
                for every in LATENT_LAYERS
            }
            torch.cuda.synchronize()
            _assert_rows(world, got, want, f"the draft after a close ({taken} yields)")
    got = _forward(target, pool, layout, between=None)
    _assert_rows(world, got, want, "the target after the closes")

    # Resumed after the next prepare: refused.
    _prepare(target, layout)
    resumed = _read(target, pool, layout, LATENT_LAYERS[0])
    next(resumed)
    _prepare(target, layout)
    newer = _read(target, pool, layout, LATENT_LAYERS[0])
    next(newer)
    with pytest.raises(RuntimeError, match="resumed after"):
        next(resumed)
    # Collected after the next prepare, without a close: no cleanup.
    _prepare(target, layout)
    collected = _read(target, pool, layout, LATENT_LAYERS[0])
    next(collected)
    _prepare(target, layout)
    got = {}
    for layer_id in LATENT_LAYERS:
        reading = _read(target, pool, layout, layer_id)
        got[layer_id] = [next(reading)]
        if layer_id == LATENT_LAYERS[0]:
            del collected
            gc.collect()
        got[layer_id] += list(reading)
    torch.cuda.synchronize()
    _assert_rows(world, got, want, "a read after a collected older one")
    # Closed on another thread: nothing dropped until the next prepare.
    _prepare(target, layout)
    reading = _read(target, pool, layout, LATENT_LAYERS[0])
    next(reading)
    closer = threading.Thread(target=reading.close)
    closer.start()
    closer.join()
    _prepare(draft, layout)
    with pytest.raises(RuntimeError, match="two readers"):
        list(_read(draft, pool, layout, LATENT_LAYERS[0]))
    _prepare(target, layout)
    _prepare(draft, layout)
    got = {every: list(_read(draft, pool, layout, every)) for every in LATENT_LAYERS}
    torch.cuda.synchronize()
    _assert_rows(world, got, want, "the draft after a close on another thread")
    _assert_clean(world, transport)


def _case_target_then_draft(world: _World, transport):
    """The draft reads its forward right after the target's, with no other
    target prepare between, on the same transport, forward after forward:
    the target's last layer prefetches nothing and its last window is
    released once its reads are enqueued, so the draft finds the transport
    free."""
    layout = _layouts(world.device)["fragmented"]
    planes, _, _ = _planes(world, layout, dtype=FP8, payload_layer_offset=0)
    pool = _LatentPool(planes)
    want = _expected(pool, layout, dtype=FP8)
    target = _reader(world, layout, transport)
    draft = _reader(world, layout, transport)
    epoch = transport.state.epoch
    for forward in range(3):
        got = _forward(target, pool, layout, between=None)
        _assert_rows(world, got, want, f"the target of forward {forward}")
        got = _forward(draft, pool, layout, between=None)
        _assert_rows(world, got, want, f"the draft of forward {forward}")
    windows = len(target.windows) * len(LATENT_LAYERS)
    assert transport.state.epoch - epoch == 6 * windows
    _assert_clean(world, transport)


def _case_release_before_the_last_yield(world: _World, transport):
    """A window's slot is released once its last loop's read is enqueued,
    before that loop is yielded: at that yield the transport holds only the
    window prefetched after it."""
    layout = _layouts(world.device)["eight_windows"]
    planes, _, _ = _planes(world, layout, dtype=FP8, payload_layer_offset=0)
    pool = _LatentPool(planes)
    want = _expected(pool, layout, dtype=FP8)
    history = _reader(world, layout, transport)
    _prepare(history, layout)
    failure = None
    got = {}
    for layer_id in LATENT_LAYERS:
        got[layer_id] = []
        reading = _read(history, pool, layout, layer_id)
        for window_index, window in enumerate(history.windows):
            for loop in range(len(window.loop_bounds) - 1):
                got[layer_id].append(next(reading))
                # The windows held: holds() says only whether there are any.
                held = len(transport._open)
                last = loop == len(window.loop_bounds) - 2
                prefetched = not (
                    layer_id == LATENT_LAYERS[-1]
                    and window_index == len(history.windows) - 1
                )
                if last and held != int(prefetched) and failure is None:
                    failure = (
                        f"layer {layer_id} window {window_index}: {held} windows "
                        "held at its last yield"
                    )
        assert list(reading) == []
    torch.cuda.synchronize()
    _agree(world, failure, "the release before the last yield")
    _assert_rows(world, got, want, "the reads around the releases")
    _assert_clean(world, transport)


def _case_prepare_drains(world: _World, transport):
    """A forward that stopped after its first take, neither read on nor
    closed: the next prepare waits for both windows in flight and drains the
    transport, so the draft reads next. A prepare called on another stream
    still makes the transport's consumer stream wait for the windows."""
    layout = _layouts(world.device)["fragmented"]
    planes, _, _ = _planes(world, layout, dtype=FP8, payload_layer_offset=0)
    pool = _LatentPool(planes)
    want = _expected(pool, layout, dtype=FP8)
    target = _reader(world, layout, transport)
    draft = _reader(world, layout, transport)
    _prepare(target, layout)
    abandoned = _read(target, pool, layout, LATENT_LAYERS[0])
    next(abandoned)
    _prepare(target, layout)
    _prepare(draft, layout)
    got = {every: list(_read(draft, pool, layout, every)) for every in LATENT_LAYERS}
    torch.cuda.synchronize()
    _assert_rows(world, got, want, "the draft after an abandoned target")
    del abandoned

    # Two windows launched (no take; only the reader's own launch step does
    # that), the second ~2 s late on every rank (rank 0's kernel starts late;
    # the others wait for its credit), then a prepare on a side stream: the
    # side stream does not wait, the consumer does.
    _prepare(target, layout)
    target._launch(pool, LATENT_LAYERS[0], 0)
    if world.rank == 0:
        with torch.cuda.stream(transport.state.stream):
            torch.cuda._sleep(4 * HOLD_CYCLES)
    target._launch(pool, LATENT_LAYERS[0], 1)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _prepare(target, layout)
        side_done = torch.cuda.Event()
        side_done.record()
    consumer_done = torch.cuda.Event()
    consumer_done.record()
    side_done.synchronize()
    waited = not consumer_done.query()
    torch.cuda.synchronize()
    _agree(
        world,
        None if waited else "the consumer did not wait",
        "a prepare on a side stream",
    )
    _assert_clean(world, transport)


def _case_over_cap_alternates(world: _World):
    """Windows alternately spread over every owner (128 rows each) and held
    by one (512 rows), on a 256-row slot: inside every layer, fused, NCCL,
    fused, ... Epochs count the fused windows only."""
    degree = len(world.group)
    blocks = []
    for window in range(6):
        base = 1 + window * 4 * degree
        step = 1 if window % 2 == 0 else degree
        blocks += [base + i * step for i in range(4)]
    layout = _layout(
        world.device,
        prefix_lens=[len(blocks) * GRANULARITY - 10],
        chunk_len=128,
        blocks=blocks,
    )
    transport = _own_transport(world, cap_rows=256, modes=("multicast", "peer"))
    planes, _, _ = _planes(world, layout, dtype=FP8, payload_layer_offset=0)
    pool = _LatentPool(planes)
    want = _expected(pool, layout, dtype=FP8)
    history = _reader(world, layout, transport)
    epoch = transport.state.epoch
    got = _forward(history, pool, layout, between=None)
    _assert_rows(world, got, want, "alternating windows")
    assert [window.send_rows > 256 for window in history.windows] == [False, True] * 3
    assert transport.state.epoch - epoch == 3 * len(LATENT_LAYERS)
    _assert_clean(world, transport)
    transport.state.close()


def _case_other_row_width(world: _World, transport):
    """A plane of 2-byte elements (1152-byte rows, not the compiled 576), and
    int64 virtual slots (which plan int64 fetch slots, not the kernel's
    int32), take the NCCL all-gather in every window, on every rank,
    bitwise."""
    layout = _layouts(world.device)["fragmented"]
    planes, _, _ = _planes(world, layout, dtype=torch.bfloat16, payload_layer_offset=0)
    pool = _LatentPool(planes)
    want = _expected(pool, layout, dtype=torch.bfloat16)
    epoch = transport.state.epoch
    got = _forward(_reader(world, layout, transport), pool, layout, between=None)
    _assert_rows(world, got, want, "2-byte rows")
    wide = replace(layout, chunks=[chunk.to(torch.int64) for chunk in layout.chunks])
    planes, _, _ = _planes(world, wide, dtype=FP8, payload_layer_offset=0)
    pool = _LatentPool(planes)
    want = _expected(pool, wide, dtype=FP8)
    got = _forward(_reader(world, wide, transport), pool, wide, between=None)
    _assert_rows(world, got, want, "int64 slots")
    assert transport.state.epoch == epoch
    _assert_clean(world, transport)


def _case_owner_without_rows(world: _World, transport):
    """Only ranks 0 and 1 own history blocks: the others send no row, yet
    enter and leave every window."""
    degree = len(world.group)
    blocks = [first + k * degree for k in range(4) for first in (1, 2)]
    layout = _layout(
        world.device,
        prefix_lens=[len(blocks) * GRANULARITY - 10],
        chunk_len=256,
        blocks=blocks,
    )
    history, _ = _nccl_then_fused(world, layout, transport)
    for window in history.windows:
        assert window.owned_rows_per_rank[2:] == (0,) * (degree - 2)


def _case_neighbour_rows_written(world: _World, transport):
    """Requests ending mid-block leave history rows whose cache lines (576 B
    rows: every other row boundary is inside a 128 B line) they share with
    the next rows; another thread keeps rewriting every non-history row of
    their blocks on a third stream while the fused forward is enqueued. The
    history rows read back intact."""
    lengths = [128 * k + 2 * k + 1 for k in range(1, 13)]
    layout = _layout(world.device, prefix_lens=lengths, chunk_len=128, seed=11)
    planes, local, owned = _planes(world, layout, dtype=FP8, payload_layer_offset=0)
    pool = _LatentPool(planes)
    want = _expected(pool, layout, dtype=FP8)
    blocks = torch.unique(local[owned] // GRANULARITY)
    rows = (
        blocks.unsqueeze(1) * GRANULARITY
        + torch.arange(GRANULARITY, device=world.device)
    ).flatten()
    neighbours = rows[~torch.isin(rows, local[owned])]
    _agree(world, None if neighbours.numel() else "no neighbour row", "the layout")
    noise = torch.randint(
        0, 256, (neighbours.numel(), 1, ROW), dtype=torch.uint8, device=world.device
    )
    writer = torch.cuda.Stream()
    writer.wait_stream(torch.cuda.current_stream())
    started, stop = threading.Event(), threading.Event()

    def write():
        # Bounded; enqueued while the forward is, so the writes run among
        # its gathers on the device.
        with torch.cuda.device(world.device), torch.cuda.stream(writer):
            for _ in range(4000):
                for layer_id in LATENT_LAYERS:
                    planes[layer_id].view(torch.uint8)[neighbours] = noise
                started.set()
                if stop.is_set():
                    break

    writing = threading.Thread(target=write)
    writing.start()
    started.wait()
    history = _reader(world, layout, transport)
    _prepare(history, layout)
    got = {
        layer_id: list(_read(history, pool, layout, layer_id))
        for layer_id in LATENT_LAYERS
    }
    stop.set()
    writing.join()
    torch.cuda.synchronize()
    _assert_rows(world, got, want, "fused reads next to written rows")
    _assert_clean(world, transport)


def _host_load_planes(world: _World, layout: _Layout, loaded_from_block: int):
    """Planes whose rows in blocks from ``loaded_from_block`` on are a Host
    load's: they hold another layer's bytes until the load lands. Returns the
    planes' pool, the loaded rows' mask and local rows, each latent layer's
    loaded bytes, and the rows every read must see."""
    planes, local, owned = _planes(world, layout, dtype=FP8, payload_layer_offset=100)
    loaded = layout.every_slot // GRANULARITY >= loaded_from_block
    resident, landing = owned & ~loaded, owned & loaded
    _agree(
        world,
        None if bool(landing.any()) and bool(resident.any()) else "no landing row",
        "the layout",
    )
    for layer_id in LATENT_LAYERS:
        planes[layer_id].view(torch.uint8)[local[resident]] = _payload(
            layer_id, layout.every_slot[resident], FP8
        ).view(torch.uint8)
    pool = _LatentPool(planes)
    payloads = {
        layer_id: _payload(layer_id, layout.every_slot[landing], FP8).view(torch.uint8)
        for layer_id in LATENT_LAYERS
    }
    return pool, landing, local[landing], payloads, _expected(pool, layout, dtype=FP8)


def _host_load_case(world, layout, readers, *, loaded_from_block, late_cycles, skew):
    """One forward per reader of ``readers`` on one pool, each after its own
    Host load (``begin_load`` per forward, so the tracker's event sets
    rotate) of every row in blocks from ``loaded_from_block`` on, layer by
    layer on a load stream, the second latent layer's copy held back
    ``late_cycles``. The planes hold another layer's bytes in those rows
    until the copy lands: every read must see the loaded bytes. With a long
    hold, the reads must all be enqueued ahead of it (else the case tested
    nothing). ``skew``: random device sleeps of up to that many cycles after
    every loop."""
    from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker

    pool, landing, landing_rows, payloads, want = _host_load_planes(
        world, layout, loaded_from_block
    )
    planes = pool.kv_buffer
    pool.layerwise_load_tracker = LayerwiseLoadTracker(NUM_LAYERS)
    rng = random.Random(world.rank)
    for index, history in enumerate(readers):
        for layer_id in LATENT_LAYERS:
            planes[layer_id].view(torch.uint8)[landing_rows] = _payload(
                layer_id + 100, layout.every_slot[landing], FP8
            ).view(torch.uint8)
        _prepare(history, layout)
        history.windows  # planned before the load starts: the reads enqueue at once
        tracker = pool.layerwise_load_tracker
        load_index = tracker.begin_load()
        events = tracker.event_sets[load_index].layer_done_events
        load = torch.cuda.Stream()
        load.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(load):
            for layer_id, plane in enumerate(planes):
                if layer_id == LATENT_LAYERS[1]:
                    torch.cuda._sleep(late_cycles)
                if plane is not None:
                    plane.view(torch.uint8)[landing_rows] = payloads[layer_id]
                events[layer_id].record(load)
        tracker.set_consumers(load_index)
        out = {}
        for layer_id in LATENT_LAYERS:
            out[layer_id] = []
            for pair in _read(history, pool, layout, layer_id):
                out[layer_id].append(pair)
                if skew:
                    torch.cuda._sleep(rng.randrange(skew))
        pending = not load.query()
        torch.cuda.synchronize()
        if late_cycles >= HOLD_CYCLES:
            _agree(
                world,
                None if pending else "the load landed before the reads were enqueued",
                "the case's premise",
            )
        _assert_rows(world, out, want, f"forward {index} after a Host load")


def _case_host_load_event(world: _World, transport):
    """Event mode: a window prefetched while the previous latent layer reads
    waits for its own layer's load (the wait stays on the consumer stream, in
    ``get_key_buffer``, and the window's ready event follows it)."""
    layout = _layout(world.device, prefix_lens=[6000, 2500], chunk_len=512, seed=12)
    readers = (_reader(world, layout, None), _reader(world, layout, transport))
    _host_load_case(
        world,
        layout,
        readers,
        loaded_from_block=1500,
        late_cycles=HOLD_CYCLES,
        skew=0,
    )
    _assert_clean(world, transport)


def _case_late_release_flags(world: _World, transport):
    """Flags mode, the release submitted late: the reads enqueue every
    layer's spin wait (the real ``wait_layer_ready``) and windows first; only
    then another thread submits, on a load stream, each layer's copy and its
    flag's gpu-scope release (the shape of the transfer kernel's
    ``_signal_layer_ready``).

    A first launch (JIT and module load) while a spin waits can hang, so
    every kernel runs before any spin waits: the copy and both kernels for
    every layer index, then an NCCL and a fused forward behind released
    flags. The releasing thread starts before the reads and waits for them
    at most a minute: a read whose enqueue still blocks gets its release and
    fails the case's premise on every rank instead of hanging. The NCCL
    reader runs first, so a hang under ``CUDA_DEVICE_MAX_CONNECTIONS=1``
    there is the NCCL path's own."""
    from tokenspeed_kernel._triton import tl, triton
    from tokenspeed_kernel.ops.kvcache.triton import wait_layer_ready

    from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker

    @triton.jit
    def release_flag(flag_ptr, index):
        tl.atomic_xchg(flag_ptr + index, 1, sem="release", scope="gpu")

    layout = _layout(world.device, prefix_lens=[6000, 2500], chunk_len=512, seed=12)
    pool, landing, landing_rows, payloads, want = _host_load_planes(world, layout, 1500)
    planes = pool.kv_buffer
    readers = {
        "NCCL": _reader(world, layout, None),
        "fused": _reader(world, layout, transport),
    }
    load = torch.cuda.Stream()
    flags = torch.ones(NUM_LAYERS, dtype=torch.int32, device=world.device)
    # Warm the copy and both kernels for every layer index (Triton keys an
    # integer on 1 and on divisibility by 16): released flags, so the spins
    # return at once. The copies leave every row loaded.
    load.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(load):
        for layer_id in range(NUM_LAYERS):
            if planes[layer_id] is not None:
                planes[layer_id].view(torch.uint8)[landing_rows] = payloads[layer_id]
            release_flag[(1,)](flags, layer_id)
    for layer_id in range(NUM_LAYERS):
        wait_layer_ready(flags, layer_id)
    torch.cuda.synchronize()
    tracker = LayerwiseLoadTracker(NUM_LAYERS)
    pool.layerwise_load_tracker = tracker
    # Released flags first (their spins return at once; the release rewrites
    # the same bytes and flags), then each layer's rows and flag withheld.
    for name, late in (
        ("NCCL", False),
        ("fused", False),
        ("NCCL", True),
        ("fused", True),
    ):
        when = "late" if late else "released"
        if late:
            for layer_id in LATENT_LAYERS:
                planes[layer_id].view(torch.uint8)[landing_rows] = _payload(
                    layer_id + 100, layout.every_slot[landing], FP8
                ).view(torch.uint8)
            flags.zero_()
        load_index = tracker.begin_load()
        events = tracker.event_sets[load_index]
        events.layer_ready_init_event.record()
        events.layer_ready_flags = flags
        events.wait_layer_ready = wait_layer_ready
        tracker.set_consumers(load_index)
        load.wait_stream(torch.cuda.current_stream())
        enqueued, gave_up = threading.Event(), threading.Event()

        def release():
            if not enqueued.wait(60):
                gave_up.set()
            with torch.cuda.device(world.device), torch.cuda.stream(load):
                for layer_id, plane in enumerate(planes):
                    if plane is not None:
                        plane.view(torch.uint8)[landing_rows] = payloads[layer_id]
                    release_flag[(1,)](flags, layer_id)

        releaser = threading.Thread(target=release)
        releaser.start()
        history = readers[name]
        _prepare(history, layout)
        out = {
            layer_id: list(_read(history, pool, layout, layer_id))
            for layer_id in LATENT_LAYERS
        }
        enqueued.set()
        releaser.join()
        torch.cuda.synchronize()
        events.layer_ready_flags = None
        events.wait_layer_ready = None
        # Behind released flags a slow enqueue (a first compile) is no failure.
        _agree(
            world,
            (
                "the reads took over 60 s to enqueue"
                if late and gave_up.is_set()
                else None
            ),
            f"the premise of {name} ({when})",
        )
        _assert_rows(world, out, want, f"{name} after a {when} release")
    _assert_clean(world, transport)


def _case_event_sets_rotate(world: _World, transport):
    """One pool, one Host load per forward over 44 forwards (the tracker's
    three event sets reused many times), with random device skew on every
    rank: the fused ready / done event pairs run over 1000 epochs."""
    layout = _layout(world.device, prefix_lens=[4096, 900], chunk_len=128, seed=14)
    fused = _reader(world, layout, transport)
    start = transport.state.epoch
    _host_load_case(
        world,
        layout,
        (_reader(world, layout, None),) + (fused,) * 43,
        loaded_from_block=1500,
        late_cycles=20_000_000,
        skew=300_000,
    )
    assert transport.state.epoch - start >= 1000
    _assert_clean(world, transport)


def _case_nccl_window_then_fused_after_host_load(world: _World):
    """Each layer's first window past the slot takes the NCCL all-gather; its
    later windows are fused and still read the layer's loaded rows."""
    layout = _layout(
        world.device, prefix_lens=[5000, 1500, 1500], chunk_len=256, seed=15
    )
    planner = _reader(world, layout, None)
    _prepare(planner, layout)
    sizes = [window.send_rows for window in planner.windows]
    cap_rows = max(sizes[1:])
    assert sizes[0] > cap_rows
    transport = _own_transport(world, cap_rows=cap_rows, modes=("multicast", "peer"))
    epoch = transport.state.epoch
    _host_load_case(
        world,
        layout,
        (_reader(world, layout, None), _reader(world, layout, transport)),
        loaded_from_block=1000,
        late_cycles=HOLD_CYCLES,
        skew=0,
    )
    assert transport.state.epoch - epoch == (len(sizes) - 1) * len(LATENT_LAYERS)
    _assert_clean(world, transport)
    transport.state.close()


def _case_allocator_churn(world: _World, transport):
    """Garbage written into fresh allocations on the current stream between
    every two loops and after every forward (activation and planner-buffer
    sizes): the reads do not change."""
    layout = _layouts(world.device)["small_segments"]
    planes, _, _ = _planes(world, layout, dtype=FP8, payload_layer_offset=0)
    pool = _LatentPool(planes)
    want = _expected(pool, layout, dtype=FP8)
    readers = (_reader(world, layout, None), _reader(world, layout, transport))
    sizes = (8 << 20, 4 * layout.every_slot.numel(), 64 << 10)

    def churn(_layer_id):
        for size in sizes:
            torch.empty(size, dtype=torch.uint8, device=world.device).fill_(0xA5)

    for _ in range(3):
        for history in readers:
            got = _forward(history, pool, layout, between=churn)
            _assert_rows(world, got, want, "reads with churn")
            churn(None)
    _assert_clean(world, transport)


def _case_capture(world: _World):
    """A fused window launched under graph capture raises before any epoch is
    used (no captured forward reads prefix history). A fused window launched
    before a capture is not taken under it, and a read resumed inside a
    capture begun between two of its yields -- of its own stream or of
    another -- raises before it enqueues anything and leaves its windows to
    its next prepare, outside the capture, so the transport's epochs go on.
    Runs on a stream of its own, the transport's consumer."""
    layout = _layouts(world.device)["fragmented"]
    planes, _, _ = _planes(world, layout, dtype=FP8, payload_layer_offset=0)
    pool = _LatentPool(planes)
    want = _expected(pool, layout, dtype=FP8)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        transport = _own_transport(world, cap_rows=4096, modes=("multicast", "peer"))
        history = _reader(world, layout, transport)
        got = _forward(history, pool, layout, between=None)
        _assert_rows(world, got, want, "the capture case's first forward")
        _prepare(history, layout)
        history.windows  # planned before the capture
        epoch = transport.state.epoch
        # thread_local: the NCCL watchdog thread queries events meanwhile.
        with torch.cuda.graph(
            torch.cuda.CUDAGraph(), stream=stream, capture_error_mode="thread_local"
        ):
            with pytest.raises(RuntimeError, match="launched under graph capture"):
                list(_read(history, pool, layout, LATENT_LAYERS[-1]))
        assert transport.state.epoch == epoch and not transport.holds(history)

        _prepare(history, layout)
        reading = _read(history, pool, layout, LATENT_LAYERS[0])
        next(reading)  # window 0 taken, window 1 launched: the last epoch
        with torch.cuda.graph(
            torch.cuda.CUDAGraph(), capture_error_mode="thread_local"
        ):
            with pytest.raises(RuntimeError, match="taken under graph capture"):
                transport.taken(transport.state.epoch, history)
        # Resumed in a capture of its own stream, and in one on a stream of
        # the capture's own: either raises and leaves its windows in flight.
        for capture_stream in (stream, None):
            _prepare(history, layout)
            reading = _read(history, pool, layout, LATENT_LAYERS[0])
            next(reading)
            with torch.cuda.graph(
                torch.cuda.CUDAGraph(),
                stream=capture_stream,
                capture_error_mode="thread_local",
            ):
                with pytest.raises(RuntimeError, match="under a graph capture"):
                    next(reading)
            assert transport.holds(history)
        _prepare(history, layout)
        assert not transport.holds(history)
        got = _forward(history, pool, layout, between=None)
        _assert_rows(world, got, want, "a forward after the captures")
    torch.cuda.synchronize()
    _assert_clean(world, transport)
    transport.state.close()


def _case_transport_invariants(world: _World, transport):
    """Refused before any epoch or event changes: a second reader
    interleaving, a third window in flight, and any stream but the consumer
    stream -- for a take, a release (kept for a retry) and a launch after
    every window was released. A read resumed on another stream inside a
    window raises before it enqueues anything and drains its windows."""
    layout = _layouts(world.device)["fragmented"]
    planes, _, _ = _planes(world, layout, dtype=FP8, payload_layer_offset=0)
    pool = _LatentPool(planes)
    want = _expected(pool, layout, dtype=FP8)
    one = _reader(world, layout, transport)
    other = _reader(world, layout, transport)
    _prepare(one, layout)
    windows = one.windows
    assert len(windows) >= 3
    cache = pool.get_key_buffer(LATENT_LAYERS[0])

    def launch(window, reader):
        return transport.launch(cache, window, 7, reader=reader)

    first = launch(windows[0], one)
    epoch = transport.state.epoch
    with pytest.raises(RuntimeError, match="two readers"):
        launch(windows[1], other)
    assert transport.state.epoch == epoch
    second = launch(windows[1], one)
    epoch = transport.state.epoch
    with pytest.raises(RuntimeError, match="third fused history window"):
        launch(windows[2], one)
    assert transport.state.epoch == epoch
    side = torch.cuda.Stream()
    with torch.cuda.stream(side), pytest.raises(RuntimeError, match="history take"):
        transport.taken(first[2], one)
    transport.taken(first[2], one)
    first[1].wait()
    with torch.cuda.stream(side), pytest.raises(RuntimeError, match="history release"):
        transport.release(first[2], one)
    transport.release(first[2], one)
    transport.taken(second[2], one)
    second[1].wait()
    transport.release(second[2], one)
    with torch.cuda.stream(side), pytest.raises(RuntimeError, match="history launch"):
        launch(windows[2], one)
    assert transport.state.epoch == epoch

    _prepare(one, layout)
    reading = _read(one, pool, layout, LATENT_LAYERS[0])
    next(reading)
    with torch.cuda.stream(side), pytest.raises(
        RuntimeError, match="another thread or stream"
    ):
        next(reading)
    assert not transport.holds(one)
    got = _forward(one, pool, layout, between=None)
    _assert_rows(world, got, want, "a forward after the refusals")
    _assert_clean(world, transport)


def _case_failed_cleanup_poisons(world: _World):
    """A read stopped with a window in flight whose cleanup fails a wait
    leaves its transport poisoned: every later launch raises, on every rank,
    even of a window that would take the NCCL all-gather."""
    from tokenspeed.runtime.layers.attention.dcp.cache import _LaunchedGather

    layout = _layouts(world.device)["fragmented"]
    planes, _, _ = _planes(world, layout, dtype=FP8, payload_layer_offset=0)
    pool = _LatentPool(planes)
    transport = _own_transport(world, cap_rows=4096, modes=("multicast", "peer"))
    history = _reader(world, layout, transport)
    _prepare(history, layout)
    reading = _read(history, pool, layout, LATENT_LAYERS[0])
    next(reading)  # window 0 taken, window 1 launched
    # Another gather in flight, whose wait fails as an errored collective's:
    # injected, since a real collective does not fail on demand.
    backend = _CountingBackend()
    backend.failing = True
    history._launched[(NUM_LAYERS, 0)] = _LaunchedGather(
        rows=torch.empty(0), work=_Work(backend), epoch=None
    )
    failure = "the close raised nothing"
    try:
        reading.close()
    except RuntimeError as error:
        failure = None if "collective failed" in str(error) else repr(error)
    if failure is None and not transport.poisoned:
        failure = "the transport is not poisoned"
    if failure is None:
        wide, _, _ = _planes(
            world, layout, dtype=torch.bfloat16, payload_layer_offset=0
        )
        failure = "a launch went on"
        try:
            transport.launch(
                wide[LATENT_LAYERS[0]], history.windows[0], 0, reader=history
            )
        except RuntimeError as error:
            failure = None if "refuses launches" in str(error) else repr(error)
    torch.cuda.synchronize()
    _agree(world, failure, "the poisoned transport")
    _assert_clean(world, transport)
    transport.state.close()


def _case_plane_beyond_2gib(world: _World, transport):
    """A plane of ~2.2 GB per layer with the history in its last blocks:
    source byte offsets past 2**31."""
    from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots

    degree = len(world.group)
    rows_needed = (2**31 + (1 << 26)) // ROW
    virtual_block_count = degree * (rows_needed // GRANULARITY) + 2
    top = virtual_block_count - 1
    blocks = list(range(top - 4 * degree, top))
    layout = _layout(
        world.device,
        prefix_lens=[len(blocks) * GRANULARITY - 5],
        chunk_len=256,
        blocks=blocks,
        virtual_block_count=virtual_block_count,
    )
    local, owned = resolve_cache_slots(layout.every_slot, _placement(world, layout))
    beyond = int(local[owned].max()) * ROW > 2**31
    _agree(world, None if beyond else "no source byte past 2**31", "the layout")
    _nccl_then_fused(world, layout, transport)


def _case_large_first_epoch(world: _World):
    """A state whose epochs go on from 2**40 + 1: the flag words hold
    int64 epochs."""
    transport = _own_transport(world, cap_rows=4096, modes=("multicast", "peer"))
    # The state's epoch counter, moved on with no window in flight.
    transport.state.epoch = 2**40
    _nccl_then_fused(world, _layouts(world.device)["eight_windows"], transport)
    assert transport.state.epoch > 2**40
    transport.state.close()


def _case_peer(world: _World):
    """The peer-store transport (no multicast), forced."""
    transport = _own_transport(world, cap_rows=4096, modes=("peer",))
    assert transport.state.mode == "peer"
    _nccl_then_fused(world, _layouts(world.device)["sequential"], transport)
    transport.state.close()


def _case_real_collectives_in_flight(world: _World, transport):
    """After every latent layer, with the next layer's first window in
    flight -- rank 1's fused stream sleeps before that window's kernel, so
    every rank's kernel waits for it, and rank 1 delays its loops -- the
    attention group's real all-reduces run on the current stream: trtllm
    mnnvl one-shot and two-shot, and NCCL past their range. Nothing hangs,
    the reads and sums are exact, and each kind of all-reduce completed on
    the device before such a window did."""
    from tokenspeed.runtime.distributed.comm_backend import initialize_comm_backend

    backend = initialize_comm_backend(use_pynccl=False)
    group = tuple(world.mapping.attn.tp_group)
    armed = backend.trtllm_ar.configure_group(
        rank=group.index(dist.get_rank()),
        group=group,
        max_token_num=2048,
        hidden_dim=7168,
    )
    _agree(world, None if armed else "trtllm all-reduce unavailable", "arming")
    layout = _layouts(world.device)["fragmented"]
    planes, _, _ = _planes(world, layout, dtype=FP8, payload_layer_offset=0)
    pool = _LatentPool(planes)
    want = _expected(pool, layout, dtype=FP8)
    readers = (_reader(world, layout, None), _reader(world, layout, transport))
    total = sum(range(1, len(group) + 1))
    loops = len(layout.chunks)
    seen = {layer_id: 0 for layer_id in LATENT_LAYERS}
    sizes = (16, 1024, 8192)
    overlaps, wrong = dict.fromkeys(sizes, 0), []
    current = {"fused": False}
    rng = random.Random(world.rank)

    def all_reduces(layer_id):
        seen[layer_id] += 1
        held = current["fused"] and layer_id != LATENT_LAYERS[-1]
        windows = readers[1].windows if current["fused"] else readers[0].windows
        last_window_loops = len(windows[-1].loop_bounds) - 1
        if world.rank == 1:
            torch.cuda._sleep(rng.randrange(4_000_000))
        if held and world.rank == 1 and seen[layer_id] == loops - last_window_loops:
            # The last loop before the layer's last take, which launches the
            # next layer's first window: its kernel starts ~125 ms late.
            with torch.cuda.stream(transport.state.stream):
                torch.cuda._sleep(HOLD_CYCLES // 4)
        if seen[layer_id] != loops:
            return
        for tokens in sizes:
            x = torch.full(
                (tokens, 7168),
                dist.get_rank() + 1,
                dtype=torch.bfloat16,
                device=world.device,
            )
            out = backend.trtllm_ar.all_reduce(x, group)
            reduced = torch.cuda.Event()
            reduced.record()
            reduced.synchronize()
            # Finished on the device before the next layer's first window,
            # the only one in flight after the layer's last read (and the
            # only work of the state's stream besides rank 1's sleep), did.
            if held and transport.holds(readers[1]):
                overlaps[tokens] += not transport.state.stream.query()
            if not torch.equal(out, torch.full_like(x, total)):
                wrong.append(tokens)

    for _ in range(3):
        for fused, history in zip((False, True), readers):
            current["fused"] = fused
            seen.update({layer_id: 0 for layer_id in LATENT_LAYERS})
            got = _forward(history, pool, layout, between=all_reduces)
            _assert_rows(world, got, want, "reads with all-reduces in flight")
    _agree(world, f"wrong sums at {wrong}" if wrong else None, "the all-reduces")
    most = torch.tensor([overlaps[tokens] for tokens in sizes], device=world.device)
    dist.all_reduce(most, op=dist.ReduceOp.MAX)
    assert bool((most > 0).all()), (
        f"all-reduces of {sizes} tokens done while a fused window ran: "
        f"{most.tolist()}"
    )
    _assert_clean(world, transport)


def _case_compiles_once(world: _World, transport):
    """A fused forward compiles neither the all-gather (the state compiled
    it once, at startup) nor any Triton read kernel after an NCCL forward of
    the same layout (the slot views take the gathered rows'
    specializations)."""
    import importlib.util

    from tokenspeed_kernel.ops.communication._cute_dsl import paged_row_all_gather
    from tokenspeed_kernel.ops.kvcache.triton import (
        _get_mla_kv_buffer_kernel,
        _get_mla_kv_buffer_per_loc_kernel,
    )

    test_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "tokenspeed-kernel",
        "test",
    )
    spec = importlib.util.spec_from_file_location(
        "kernel_test_utils", os.path.join(test_dir, "utils.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    layout = _layouts(world.device)["small_segments"]
    planes, _, _ = _planes(world, layout, dtype=FP8, payload_layer_offset=0)
    pool = _LatentPool(planes)
    want = _expected(pool, layout, dtype=FP8)
    got = _forward(_reader(world, layout, None), pool, layout, between=None)
    _assert_rows(world, got, want, "the NCCL reader")
    compiles = patch.object(
        paged_row_all_gather,
        "compile_paged_row_all_gather",
        wraps=paged_row_all_gather.compile_paged_row_all_gather,
    )
    with compiles as compile_calls, helper.assert_no_triton_compile(
        _get_mla_kv_buffer_kernel, _get_mla_kv_buffer_per_loc_kernel
    ):
        got = _forward(_reader(world, layout, transport), pool, layout, between=None)
    _assert_rows(world, got, want, "the fused reader")
    assert compile_calls.call_count == 0
    _assert_clean(world, transport)


def _worker(rank: int, world_size: int, init_method: str) -> None:
    """Every case on a fresh world of ``world_size`` ranks; the device is
    ``LOCAL_RANK`` under torchrun, else ``rank``."""
    from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map

    from tokenspeed.runtime.distributed.comm_backend import set_global_backend
    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.mapping import KVP_HISTORY_GROUP_NAME, Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.layers.attention.dcp.cache import (
        _FUSED_HISTORY_TRANSPORTS,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    mapping = Mapping(
        rank=rank,
        world_size=world_size,
        attn_tp_size=world_size,
        attn_dp_size=1,
        attn_kvp_size=world_size,
    )
    pg_manager.init_distributed(
        mapping,
        distributed_init_method=init_method,
        backend="nccl",
        timeout=PG_TIMEOUT_S,
        device_id=device,
    )
    pg_manager.register_process_group("nccl", mapping.world_group, dist.group.WORLD)
    group = tuple(mapping.attn.kvp_group)
    pg_manager.init_process_group(group)
    pg_manager.init_process_group(group, backend="nccl", name=KVP_HISTORY_GROUP_NAME)
    gather_fabric_map()
    set_global_backend(NcclBackend())
    world = _World(
        rank=group.index(dist.get_rank()),
        device=device,
        group=group,
        mapping=mapping,
        comm=NcclBackend().for_named_groups(KVP_HISTORY_GROUP_NAME),
        gloo=pg_manager.get_process_group("gloo", group),
        one_node=int(os.environ.get("LOCAL_WORLD_SIZE", world_size)) == world_size,
    )
    transport = _case_startup(world)
    _case_layouts(world, transport)
    _case_late_ranks(world, transport)
    _case_closed_after_every_yield(world, transport)
    _case_target_then_draft(world, transport)
    _case_release_before_the_last_yield(world, transport)
    _case_prepare_drains(world, transport)
    _case_over_cap_alternates(world)
    _case_other_row_width(world, transport)
    _case_owner_without_rows(world, transport)
    _case_host_load_event(world, transport)
    _case_late_release_flags(world, transport)
    _case_allocator_churn(world, transport)
    _case_large_first_epoch(world)
    if world.one_node:
        # Single-node premises (neighbour rows, event-set rotation, one
        # layer's first window past the slot) or single-node budgets.
        _case_transport_invariants(world, transport)
        _case_neighbour_rows_written(world, transport)
        _case_event_sets_rotate(world, transport)
        _case_nccl_window_then_fused_after_host_load(world)
        _case_capture(world)
        _case_failed_cleanup_poisons(world)
        _case_plane_beyond_2gib(world, transport)
        _case_peer(world)
    else:
        _case_real_collectives_in_flight(world, transport)
    _case_compiles_once(world, transport)
    torch.cuda.synchronize()
    dist.barrier()
    _FUSED_HISTORY_TRANSPORTS.pop(group).state.close()
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_kvp_history_fused_gather(tmp_path, world_size):
    if torch.cuda.device_count() < world_size or torch.cuda.get_device_capability() != (
        10,
        3,
    ):
        pytest.skip(f"requires {world_size} node-local GB300 GPUs")
    mp.spawn(
        _worker,
        args=(world_size, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    # Under torchrun, e.g. eight ranks across two nodes.
    _worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), "env://")
