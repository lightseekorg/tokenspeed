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

"""Real scheduler and asynchronous Host-byte restoration at a nonaligned tail.

D2H arrival is supplied as CPU bytes. Scheduling, keys, queues, storage IO and
publication use the production classes: the C++ scheduler and the engine
host's L3 executor (``tokenspeed_engine.L3Executor``). The Mooncake variant
starts a local master (``MOONCAKE_REQUIRE_MASTER=1`` requires it) with an
independent segment owner.
"""

from __future__ import annotations

import hashlib
import uuid
from contextlib import ExitStack
from test.l3_test_utils import (
    mooncake_client,
    mooncake_master,
    start_l3_executor,
    storage_ops,
)

import pytest
import tokenspeed_scheduler as ts
import torch

# Every group's 8-byte blocks, one per LCM block: block ``b`` of any group is
# at ``(b - 1) * 8``, and the scheduler never gives one LCM block to two
# groups at once.
_BLOCK_BYTES = 8
# ``num_host_pages`` with the null block excluded.
_HOST_BLOCKS = 31


def _host():
    return torch.zeros(_HOST_BLOCKS * _BLOCK_BYTES, dtype=torch.uint8)


def _block(host, block):
    offset = (block - 1) * _BLOCK_BYTES
    return host[offset : offset + _BLOCK_BYTES]


def _send(scheduler, event):
    events = ts.ExecutionEvent()
    events.add_event(event)
    scheduler.advance(events)


def _scheduler(kinds):
    config = ts.SchedulerConfig()
    config.prefix_granularity = 2
    config.num_snapshot_pages = 1
    config.num_device_pages = 64
    config.num_host_pages = _HOST_BLOCKS + 1
    config.host_group_packing = [1 for _ in kinds]
    config.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteThrough
    config.max_scheduled_tokens = 64
    config.max_batch_size = 8
    config.enable_l3_storage = True
    config.storage_config = ts.StorageConfig(
        8,
        64,
        31,
        16,
        64,
        2,
        100,
        100,
        ts.StoragePressure(8, 1048576, 1000),
        ts.StoragePressure(4, 524288, 500),
    )
    config.cache_groups = [
        ts.CacheGroupConfig(
            group_id=kind,
            block_granularity=0 if kind == "state" else 2,
            total_pages=64,
            attn_kind={
                "full": ts.AttnKind.Full,
                "swa": ts.AttnKind.SlidingWindow,
                "state": ts.AttnKind.State,
            }[kind],
            sliding_window_tokens=4 if kind == "swa" else None,
        )
        for kind in kinds
    ]
    scheduler = ts.Scheduler(config)
    _send(scheduler, ts.Cache.StorageTick(1, True, True, ts.StoragePressure(0, 0, 0)))
    return scheduler


def _submit(scheduler, tokens, *, reuse_bound):
    request = ts.RequestSpec()
    request.request_id = "r"
    request.tokens = tokens
    request.max_cached_prefix_tokens = reuse_bound
    scheduler.submit_requests([request])
    return storage_ops(scheduler.next_execution_plan())[0]


def _complete(scheduler, executor, operation):
    """Run ``operation`` to completion and feed it back: its per-key success."""
    import time

    executor.submit([operation])
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        _, completions, _ = executor.poll()
        if completions:
            ((op_id, success, retired, object_failures),) = completions
            assert retired
            assert op_id == operation.op_id
            _send(
                scheduler,
                ts.Cache.StorageDone(op_id, success, retired, object_failures),
            )
            return success
        time.sleep(0.001)
    pytest.fail(f"storage operation {operation.op_id} did not retire")


@pytest.fixture(scope="module")
def mooncake_tail_client():
    """A store client of a local master whose segment another client owns."""
    with mooncake_master() as master:
        owner_host = _host()
        owner = start_l3_executor(
            owner_host,
            groups=((1, _BLOCK_BYTES, 1),),
            lcm_block_bytes=_BLOCK_BYTES,
            owner_rank=0,
            tp_rank=0,
            deployment_id=f"tail-owner-{uuid.uuid4().hex}",
            mooncake=mooncake_client(master, global_segment_size=64 << 20),
        )
        try:
            yield mooncake_client(master, global_segment_size=0)
        finally:
            owner.close(10.0)


@pytest.mark.parametrize("backend_kind", ["memory", "mooncake"])
@pytest.mark.parametrize("kinds", [("full",), ("state",), ("full", "swa", "state")])
@pytest.mark.parametrize("suffix_tokens", [1, 4])
def test_cold_tail_restores_actual_put_bytes(
    backend_kind, kinds, suffix_tokens, request
):
    mooncake = (
        request.getfixturevalue("mooncake_tail_client")
        if backend_kind == "mooncake"
        else None
    )
    namespace = f"tail-roundtrip-{uuid.uuid4().hex}"

    def executor(host):
        return start_l3_executor(
            host,
            groups=((1, _BLOCK_BYTES, 1),) * len(kinds),
            lcm_block_bytes=_BLOCK_BYTES,
            owner_rank=0,
            tp_rank=0,
            deployment_id=namespace,
            mooncake=mooncake,
        )

    expected = {}
    with ExitStack() as cleanup:
        writer, writer_host = _scheduler(kinds), _host()
        writer_io = executor(writer_host)
        cleanup.callback(writer_io.close, 5.0)
        query = _submit(writer, list(range(5)), reuse_bound=4)
        assert not any(_complete(writer, writer_io, query))
        plan = writer.next_execution_plan()
        assert list(plan.forward[0].extend_prefix_lens) == [0]
        advanced = ts.ForwardEvent.ExtendResult()
        advanced.request_id, advanced.tokens = "r", [77]
        _send(writer, advanced)
        finished = ts.ForwardEvent.Finish()
        finished.request_id = "r"
        _send(writer, finished)
        for operation in writer.next_execution_plan().cache:
            if isinstance(operation, ts.Cache.WriteBackOp):
                for op_id in operation.op_ids:
                    done = ts.Cache.WriteBackDoneEvent()
                    done.op_id = op_id
                    _send(writer, done)
        puts = [
            op
            for op in storage_ops(writer.next_execution_plan())
            if op.kind == ts.StorageKind.Put
        ]
        assert puts
        for operation in puts:
            for key, block in zip(operation.keys, operation.host_blocks, strict=True):
                key = tuple(key)
                payload = hashlib.sha256(repr(key).encode()).digest()[:_BLOCK_BYTES]
                expected[key] = payload
                _block(writer_host, block).copy_(
                    torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                )
            assert all(_complete(writer, writer_io, operation))
        # The store keeps the objects after their writer closes.
        writer_io.close(5.0)
        writer_host.zero_()

        reader, reader_host = _scheduler(kinds), _host()
        reader_io = executor(reader_host)
        cleanup.callback(reader_io.close, 5.0)
        query = _submit(reader, list(range(5)) + [88] * suffix_tokens, reuse_bound=5)
        _complete(reader, reader_io, query)
        get = next(
            op
            for op in storage_ops(reader.next_execution_plan())
            if op.kind == ts.StorageKind.Get
        )
        assert all(tuple(key) in expected for key in get.keys)
        assert [
            sum(key[0] == gid for key in get.keys) for gid in range(len(kinds))
        ] == [{"full": 3, "swa": 2, "state": 1}[kind] for kind in kinds]
        assert reader.host_pool_cached_blocks() == 0
        assert all(_complete(reader, reader_io, get))
        for key, block in zip(get.keys, get.host_blocks, strict=True):
            assert bytes(_block(reader_host, block).tolist()) == expected[tuple(key)]
        plan = reader.next_execution_plan()
        assert list(plan.forward[0].extend_prefix_lens) == [5]
        assert list(plan.forward[0].input_lengths) == [suffix_tokens]
        assert any(isinstance(op, ts.Cache.LoadBackOp) for op in plan.cache)
        assert reader.storage_quarantined() == 0
        assert reader_io.delete_namespace()
