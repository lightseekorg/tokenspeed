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

"""Ranks share one L2 pool end to end: real schedulers, clients and owner
service. CPU bytes stand in for pool memory and a memory copy for RDMA; the
load tests run the real load executor against CUDA streams."""

import ctypes
import faulthandler
import json
import socket
import threading
import time

import pytest
import tokenspeed_scheduler as ts
import torch

from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker
from tokenspeed.runtime.cache.shared_l2.client import (
    SharedL2Client,
    SharedL2ClientConfig,
)
from tokenspeed.runtime.cache.shared_l2.layout import (
    FieldLayout,
    GroupLayout,
    SharedL2Layout,
)
from tokenspeed.runtime.cache.shared_l2.owner import SharedL2Owner
from tokenspeed.runtime.cache.shared_l2.wire import OwnerConnection
from tokenspeed.runtime.engine.scheduler_utils import cache_ops_from_plan
from tokenspeed.runtime.engine.shared_l2_hooks import SharedL2Hooks

GRAIN = 2
PAGES = 65  # Scheduler blocks 1..64 and the null page.
# Two layers: the first reads k, the second v and the state.
CONSUMERS = (("k",), ("v", "s"))


class Arena:
    """Bytes standing in for a rank's Device cache: Full KV and a State."""

    def __init__(self, device="cpu"):
        self.k = torch.zeros(PAGES, 8, dtype=torch.uint8, device=device)
        self.v = torch.zeros(PAGES, 8, dtype=torch.uint8, device=device)
        self.state = torch.zeros(PAGES, 32, dtype=torch.uint8, device=device)
        self.layout = SharedL2Layout(
            groups=(
                GroupLayout(
                    group_id="kv",
                    packing=2,
                    fields=(
                        FieldLayout("k", self.k.data_ptr(), 8, 6),
                        FieldLayout("v", self.v.data_ptr(), 8, 8),
                    ),
                    attn_kind="Full",
                    block_granularity=GRAIN,
                    sliding_window_tokens=None,
                    replayable=False,
                ),
                GroupLayout(
                    group_id="state",
                    packing=1,
                    fields=(FieldLayout("s", self.state.data_ptr(), 32, 28),),
                    attn_kind="State",
                    block_granularity=0,
                    sliding_window_tokens=None,
                    replayable=False,
                ),
            ),
            registration_regions=(),
            prefix_granularity=GRAIN,
        )

    def block(self, group, block_id):
        if group == 0:
            return bytes(self.k[block_id, :6].tolist() + self.v[block_id, :8].tolist())
        return bytes(self.state[block_id, :28].tolist())

    def fill(self, group, block_id, seed):
        if group == 0:
            self.k[block_id, :6] = torch.arange(seed, seed + 6, dtype=torch.uint8)
            self.v[block_id, :8] = torch.arange(seed + 50, seed + 58, dtype=torch.uint8)
        else:
            self.state[block_id, :28] = torch.arange(seed, seed + 28, dtype=torch.uint8)


class MemoryTransfer:
    """Copies between CPU addresses; one in-process provider session."""

    def read(self, session, local, remote, lengths):
        for dst, src, size in zip(local, remote, lengths):
            ctypes.memmove(dst, src, size)
        return True

    def write(self, session, local, remote, lengths):
        for src, dst, size in zip(local, remote, lengths):
            ctypes.memmove(dst, src, size)
        return True

    def close(self):
        pass


class SlowMemoryTransfer(MemoryTransfer):
    """A provider whose reads take long enough to still run once the forward
    waits on them."""

    def read(self, session, local, remote, lengths):
        time.sleep(0.5)
        return super().read(session, local, remote, lengths)


class CudaTransfer:
    """Copies between CUDA memory and CPU pool memory from any thread."""

    def __init__(self):
        self._cuda = ctypes.CDLL("libcuda.so.1")

    def _copy(self, destinations, sources, lengths):
        torch.cuda.set_device(0)
        for dst, src, size in zip(destinations, sources, lengths):
            result = self._cuda.cuMemcpy(
                ctypes.c_uint64(dst), ctypes.c_uint64(src), ctypes.c_size_t(size)
            )
            assert result == 0
        return True

    def read(self, session, local, remote, lengths):
        return self._copy(local, remote, lengths)

    def write(self, session, local, remote, lengths):
        return self._copy(remote, local, lengths)

    def close(self):
        pass


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def owner_address():
    layout = Arena().layout
    address = f"tcp://127.0.0.1:{free_port()}"
    owner = SharedL2Owner(
        address=address,
        lcm_bytes=layout.lcm_bytes,
        num_providers=1,
        headroom_lcm_blocks=0,
        client_timeout_s=60,
    )
    stop = threading.Event()
    thread = threading.Thread(target=owner.serve, args=(stop,), daemon=True)
    thread.start()
    pool_memory = ctypes.create_string_buffer(64 * layout.lcm_bytes)
    provider = OwnerConnection(address, timeout_ms=5000)
    provider.request(
        "provider",
        session="memory",
        base=ctypes.addressof(pool_memory),
        bytes=len(pool_memory),
    )
    yield address
    provider.close()
    stop.set()
    thread.join(timeout=5)
    owner.close()


def config_fields(address):
    return {
        "owner": address,
        "namespace": "test",
        "device_names": [""],
        "request_timeout_ms": 5000,
        "connect_timeout_s": 5,
        "load_workers": 2,
        "write_workers": 1,
        "scheduler": {"max_writes": 2, "max_write_keys": 64},
    }


def client_config(address):
    return SharedL2ClientConfig.from_json(json.dumps(config_fields(address)))


def scheduler(config, num_host_pages):
    cfg = ts.SchedulerConfig()
    cfg.role = ts.SchedulerConfig.Role.Fused
    cfg.prefix_granularity = GRAIN
    cfg.num_snapshot_pages = 1
    cfg.num_device_pages = 64
    cfg.num_host_pages = num_host_pages
    cfg.disable_l2_cache = False
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.shared_l2_config = config.scheduler
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="kv",
            block_granularity=GRAIN,
            total_pages=64,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
            attn_kind=ts.AttnKind.Full,
        ),
        ts.CacheGroupConfig(
            group_id="state",
            block_granularity=0,
            total_pages=64,
            transfer_policy=ts.CacheTransferPolicy.LatestSnapshot,
            attn_kind=ts.AttnKind.State,
        ),
    ]
    return ts.Scheduler(cfg)


class Rank:
    """One rank's event loop: lookups before each plan, commands to the
    client, load-backs to the executor (or acknowledged at once without one)."""

    def __init__(self, address, *, device="cpu", transfer=None):
        self.address = address
        self.arena = Arena(device)
        config = client_config(address)
        self.client = SharedL2Client(
            config,
            layout=self.arena.layout,
            transfer=transfer if transfer is not None else MemoryTransfer(),
            connection=OwnerConnection(address, timeout_ms=5000),
        )
        self.scheduler = scheduler(config, 2 * self.client.num_lcm_blocks + 1)
        self.executor = None
        self.tracker = None
        self.lookups, self.loads, self.writes = [], [], []
        self.pending = []

    def use_executor(self):
        from tokenspeed.runtime.cache.shared_l2.executor import SharedL2LoadExecutor

        self.tracker = LayerwiseLoadTracker(len(CONSUMERS))
        self.executor = SharedL2LoadExecutor(
            self.client, trackers=[(self.tracker, len(CONSUMERS))], consumers=CONSUMERS
        )

    def round(self, before_submit=None):
        e = ts.ExecutionEvent()
        for event in self.client.poll_events():
            e.add_event(event)
        if self.executor is not None:
            for event in self.executor.poll_results():
                e.add_event(event)
        else:
            for op_id in self.pending:
                e.add_event(ts.Cache.LoadBackDoneEvent(op_id))
            self.pending = []
        self.scheduler.advance(e)
        lookups = self.scheduler.shared_l2_lookups()
        self.lookups += lookups
        self.client.map_proxy_pages(
            self.scheduler.apply_shared_l2_lookups(self.client.lookup(lookups))
        )
        plan = self.scheduler.next_execution_plan()
        ops = cache_ops_from_plan(plan)
        writes = [
            op for op in ops.shared_l2 if isinstance(op, ts.Cache.SharedL2WriteOp)
        ]
        if before_submit is not None:
            before_submit(writes)
        self.writes += writes
        self.client.submit(ops.shared_l2)
        for op in ops.device:
            if isinstance(op, ts.Cache.LoadBackOp):
                for op_id, groups, sources, destinations in zip(
                    op.op_ids, op.group_ids, op.src_pages, op.dst_pages
                ):
                    self.loads.append(list(zip(groups, sources, destinations)))
                    self.pending.append(op_id)
        if self.executor is not None:
            self.executor.submit_load_backs(
                ops.device, prerequisite_stream=torch.cuda.current_stream()
            )
            self.pending = []
        return plan

    def settle(self, before_submit=None, rounds=20):
        for _ in range(rounds):
            self.round(before_submit)
            time.sleep(0.02)

    def submit(self, rid, tokens):
        spec = ts.RequestSpec()
        spec.request_id, spec.tokens = rid, list(tokens)
        self.scheduler.submit_requests([spec])

    def first_chunks(self, rids, before_submit=None):
        """Plan until every request ran its first chunk; their batches' rows."""
        rows = {}
        deadline = time.monotonic() + 5
        while len(rows) < len(rids) and time.monotonic() < deadline:
            plan = self.round(before_submit)
            for batch in plan.forward:
                for rid, prefix in zip(batch.request_ids, batch.extend_prefix_lens):
                    if rid in rids and rid not in rows:
                        rows[rid] = prefix
                        result = ts.ForwardEvent.ExtendResult()
                        result.request_id, result.tokens = rid, [1000]
                        e = ts.ExecutionEvent()
                        e.add_event(result)
                        self.scheduler.advance(e)
            time.sleep(0.02)
        assert len(rows) == len(rids), f"never scheduled: {set(rids) - set(rows)}"
        return rows

    def pinned(self):
        owner = OwnerConnection(self.address, timeout_ms=5000)
        try:
            return owner.request("stats")["stats"]["pinned_blocks"]
        finally:
            owner.close()

    def close(self):
        if self.executor is not None:
            self.executor.shutdown()
        self.client.close()


def publish(writer, rid, tokens, before_submit=None):
    writer.submit(rid, tokens)
    writer.first_chunks([rid], before_submit)
    writer.settle(before_submit)


def publish_filled(writer, rid, tokens):
    """Publish a prompt whose blocks hold distinct bytes; returns those bytes."""
    written = set()

    def fill(writes):
        for op in writes:
            for key, block in zip(op.keys, op.blocks):
                group = key[0]
                writer.arena.fill(group, block, 7 * block + group)
                written.add(writer.arena.block(group, block))

    publish(writer, rid, tokens, fill)
    assert written
    return written


def test_one_owner_request_looks_up_every_waiting_prompt(owner_address):
    writer, reader = Rank(owner_address), Rank(owner_address)
    try:
        prompt = list(range(9))
        publish(writer, "a", prompt)

        reader.submit("hit", prompt + [50, 51])
        reader.submit("cold", list(range(100, 111)))
        rows = reader.first_chunks(["hit", "cold"])
        assert rows == {"hit": len(prompt), "cold": 0}
        assert sorted(lookup[0] for lookup in reader.lookups) == ["cold", "hit"]
        [load] = reader.loads
        # Every source stands for the pool page its lookup pinned.
        assert all(
            reader.client.pool_page(group, source) > 0 for group, source, _ in load
        )
        reader.settle()
        assert reader.pinned() == 0
    finally:
        writer.close()
        reader.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="loads into CUDA memory")
def test_a_rank_loads_the_bytes_another_rank_wrote_layer_by_layer(owner_address):
    writer = Rank(owner_address)
    reader = Rank(owner_address, device="cuda", transfer=CudaTransfer())
    reader.use_executor()
    try:
        prompt = list(range(9))
        written = publish_filled(writer, "a", prompt)

        reader.submit("b", prompt + [50, 51])
        assert reader.first_chunks(["b"]) == {"b": len(prompt)}
        [load] = reader.loads
        # The forward's layer-wise waits return once each layer landed.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for layer in range(len(CONSUMERS)):
                reader.tracker.wait_for_layer(layer)
        stream.synchronize()
        loaded = {
            reader.arena.block(group, destination) for group, _, destination in load
        }
        assert loaded == written
        reader.settle()
        assert reader.pinned() == 0
    finally:
        writer.close()
        reader.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="waits on a CUDA stream")
def test_a_load_completes_while_the_forward_holds_the_cuda_driver(owner_address, capfd):
    """Loading a kernel on first use holds the CUDA driver until the GPU drains:
    a forward that does so while its stream waits on a load needs the loader to
    release that wait without the driver. A deadlock ends the process."""
    writer = Rank(owner_address)
    # CPU cache memory: the reader's copies make no CUDA call either.
    reader = Rank(owner_address, transfer=SlowMemoryTransfer())
    reader.use_executor()
    # Kernels this process has not run, so the GEMM below loads them.
    a = torch.empty(333, 517, dtype=torch.float64, device="cuda")
    b = torch.empty(517, 211, dtype=torch.float64, device="cuda")
    try:
        prompt = list(range(9))
        written = publish_filled(writer, "a", prompt)

        reader.submit("b", prompt + [50, 51])
        assert reader.first_chunks(["b"]) == {"b": len(prompt)}
        [load] = reader.loads
        stream = torch.cuda.Stream()
        # Uncaptured, so the watchdog's report survives the exit.
        with capfd.disabled():
            faulthandler.dump_traceback_later(60, exit=True)
            try:
                with torch.cuda.stream(stream):
                    for layer in range(len(CONSUMERS)):
                        reader.tracker.wait_for_layer(layer)
                    torch.mm(a, b)
                stream.synchronize()
            finally:
                faulthandler.cancel_dump_traceback_later()
        loaded = {
            reader.arena.block(group, destination) for group, _, destination in load
        }
        assert loaded == written
        reader.settle()
        assert reader.pinned() == 0
    finally:
        writer.close()
        reader.close()


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda fields: fields.pop("load_workers"), "needs exactly"),
        (lambda fields: fields.update(extra=1), "needs exactly"),
        (
            lambda fields: fields["scheduler"].pop("max_writes"),
            "scheduler needs exactly",
        ),
        (
            lambda fields: fields["scheduler"].update(max_writes=0),
            "positive write limits",
        ),
        (lambda fields: fields.update(write_workers=0), "at least one"),
    ],
)
def test_a_client_config_names_every_field_and_usable_limits(change, message):
    fields = config_fields("tcp://127.0.0.1:1")
    change(fields)
    with pytest.raises(ValueError, match=message):
        SharedL2ClientConfig.from_json(json.dumps(fields))


class _ArrivingPrompts:
    """The event loop's intake: one prompt arrives per call while any remain."""

    def __init__(self, rank, prompts):
        self.rank = rank
        self.scheduler = rank.scheduler
        self.pending = list(prompts)

    def _process_new_requests(self) -> bool:
        if not self.pending:
            return False
        self.rank.submit(*self.pending.pop(0))
        return True


class _ReadyAfter:
    """A forward's result that lands after a few polls."""

    def __init__(self, polls):
        self.polls = polls

    def ready(self) -> bool:
        self.polls -= 1
        return self.polls < 0


def test_a_forward_wait_looks_up_the_prompts_arriving_meanwhile(owner_address):
    writer, reader = Rank(owner_address), Rank(owner_address)
    try:
        prompt = list(range(9))
        publish(writer, "a", prompt)

        loop = _ArrivingPrompts(reader, [("hit", prompt + [50, 51])])
        SharedL2Hooks(loop, reader.client).wait_for_forward(_ReadyAfter(polls=3))
        assert not loop.pending and reader.scheduler.shared_l2_lookups() == []
        # The plan admits it from the pool without a lookup of its own.
        assert reader.first_chunks(["hit"]) == {"hit": len(prompt)}
        assert not reader.lookups
        reader.settle()
        assert reader.pinned() == 0
    finally:
        writer.close()
        reader.close()
