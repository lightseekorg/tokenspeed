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
publication use the production classes. MOONCAKE_REQUIRE_MASTER starts a local
master and independent memory owner; MOONCAKE_GPU_TEST_CONFIG reuses an owner.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import ExitStack
from test.l3_test_utils import storage_ops

import pytest
import tokenspeed_scheduler as ts
import torch

from tokenspeed.runtime.cache.l3.backend import (
    MemoryKvStore,
    copy_host_bytes,
    write_host_bytes,
)
from tokenspeed.runtime.cache.l3.config import StorageRuntimeConfig
from tokenspeed.runtime.cache.l3.executor import L3Executor


class _Host:
    def __init__(self):
        self.host_buffer = torch.zeros(3 * 64 * 8, dtype=torch.uint8)

    def host_block_range(self, group_id, block_id):
        assert 0 <= group_id < 3 and 1 <= block_id < 64
        return (group_id * 64 + block_id - 1) * 8, 8


def _send(scheduler, event):
    events = ts.ExecutionEvent()
    events.add_event(event)
    scheduler.advance(events)


def _scheduler(kinds):
    config = ts.SchedulerConfig()
    config.prefix_granularity = 2
    config.num_snapshot_pages = 1
    config.num_device_pages = 64
    config.num_host_pages = 32
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
    executor.submit_operations([operation])
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        completions = executor.poll().completions
        if completions:
            assert len(completions) == 1
            result = completions[0]
            assert result.retired
            assert result.op_id == operation.op_id
            _send(
                scheduler,
                ts.Cache.StorageDone(
                    result.op_id,
                    list(result.success),
                    result.retired,
                    list(result.object_failures),
                ),
            )
            return result
        time.sleep(0.001)
    pytest.fail(f"storage operation {operation.op_id} did not retire")


@pytest.fixture(scope="module")
def mooncake_tail_config():
    config_json = os.environ.get("MOONCAKE_GPU_TEST_CONFIG")
    if config_json:
        yield json.loads(config_json)
        return
    if os.environ.get("MOONCAKE_REQUIRE_MASTER", "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        pytest.skip("set MOONCAKE_REQUIRE_MASTER or MOONCAKE_GPU_TEST_CONFIG")

    from test.test_l3_mooncake_master import TestMooncakeMasterLiveTest, _Host

    from tokenspeed.runtime.cache.l3.mooncake import MooncakeKvStore

    TestMooncakeMasterLiveTest.setup_class()
    try:
        config = TestMooncakeMasterLiveTest()._extra_config()
        owner_host = _Host(nbytes=4096, page=64)
        owner = MooncakeKvStore(
            config, host_buffer=owner_host.host_buffer, tp_size=1, pp_size=1
        )
        try:
            yield dict(config, global_segment_size=0)
        finally:
            owner.close()
    finally:
        TestMooncakeMasterLiveTest.teardown_class()


@pytest.mark.parametrize("backend_kind", ["memory", "mooncake"])
@pytest.mark.parametrize("kinds", [("full",), ("state",), ("full", "swa", "state")])
@pytest.mark.parametrize("suffix_tokens", [1, 4])
def test_cold_tail_restores_actual_put_bytes(
    backend_kind, kinds, suffix_tokens, request
):
    native_config = (
        request.getfixturevalue("mooncake_tail_config")
        if backend_kind == "mooncake"
        else None
    )
    shared = MemoryKvStore()
    namespace = f"tail-roundtrip-{uuid.uuid4().hex}"
    async_config = StorageRuntimeConfig.from_extra_config(
        None, host_blocks=31, prefix_granularity=2
    )

    def executor(host):
        if backend_kind == "memory":
            backend = shared
        else:
            from tokenspeed.runtime.cache.l3.mooncake import MooncakeKvStore

            config = native_config
            assert config["global_segment_size"] == 0
            backend = MooncakeKvStore(
                config, host_buffer=host.host_buffer, tp_size=1, pp_size=1
            )
        return L3Executor(
            backend,
            host,
            key_prefix=namespace,
            rank=0,
            ownership=None,
            config=async_config,
            prefix_for_weight_version=None,
        )

    expected = {}
    with ExitStack() as cleanup:
        writer, writer_host = _scheduler(kinds), _Host()
        writer_io = executor(writer_host)
        cleanup.callback(writer_io.close, timeout_s=5)
        query = _submit(writer, list(range(5)), reuse_bound=4)
        assert not any(_complete(writer, writer_io, query).success)
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
                payload = hashlib.sha256(repr(key).encode()).digest()[:8]
                expected[key] = payload
                offset, _ = writer_host.host_block_range(key[0], block)
                write_host_bytes(writer_host.host_buffer, offset, payload)
            assert all(_complete(writer, writer_io, operation).success)
        if backend_kind == "mooncake":
            # The independent owner retains native objects after client close.
            writer_io.close(timeout_s=5)
        else:
            # MemoryKvStore itself is the shared owner in the CPU variant.
            writer_io._shutdown_workers(timeout_s=5)
        writer_host.host_buffer.zero_()

        reader, reader_host = _scheduler(kinds), _Host()
        reader_io = executor(reader_host)
        cleanup.callback(reader_io.close, timeout_s=5)
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
        assert all(_complete(reader, reader_io, get).success)
        for key, block in zip(get.keys, get.host_blocks, strict=True):
            offset, size = reader_host.host_block_range(key[0], block)
            assert (
                copy_host_bytes(reader_host.host_buffer, offset, size)
                == expected[tuple(key)]
            )
        plan = reader.next_execution_plan()
        assert list(plan.forward[0].extend_prefix_lens) == [5]
        assert list(plan.forward[0].input_lengths) == [suffix_tokens]
        assert any(isinstance(op, ts.Cache.LoadBackOp) for op in plan.cache)
        assert reader.storage_quarantined() == 0
        assert reader_io.backend.remove_by_prefix(namespace)
