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

"""L3 object identity survives a change of cyclic Host owner."""

from __future__ import annotations

import time

from tokenspeed.runtime.cache.l3.backend import MemoryKvStore
from tokenspeed.runtime.cache.l3.config import StorageRuntimeConfig
from tokenspeed.runtime.cache.l3.executor import L3Executor, StorageJob
from tokenspeed.runtime.cache.transfer.ownership import BlockOwnerTranslation


class Host:
    def __init__(self):
        self.host_buffer = bytearray([0xA5] * 64)
        self.host_cache_block_bytes = (4, 4)

    def host_block_range(self, group, block):
        return (block - 1) * 8 + group * 4, 4


class Backend(MemoryKvStore):
    def __init__(self):
        super().__init__()
        self.writes = []
        self.reads = []
        self.fail_query = False
        self.fail_get = False

    def batch_put_from(self, keys, buffer, offsets, sizes):
        self.writes.append((tuple(keys), tuple(offsets), tuple(sizes)))
        return super().batch_put_from(keys, buffer, offsets, sizes)

    def batch_get_into(self, keys, buffer, offsets, sizes):
        self.reads.append((tuple(keys), tuple(offsets), tuple(sizes)))
        if self.fail_get:
            raise RuntimeError("read backend unavailable")
        return super().batch_get_into(keys, buffer, offsets, sizes)

    def batch_exists(self, keys):
        if self.fail_query:
            raise RuntimeError("metadata unavailable")
        return super().batch_exists(keys)


def make_executor(backend, host, rank, *, extra_config):
    owners = BlockOwnerTranslation(
        shard_counts=(2, 1),
        device_virtual_counts=(9, 5),
        host_virtual_counts=(9, 5),
        rank=rank % 2,
    )
    return L3Executor(
        backend,
        host,
        key_prefix="kvp",
        rank=rank,
        ownership=owners,
        config=StorageRuntimeConfig.from_extra_config(
            extra_config, host_blocks=32, prefix_granularity=2
        ),
        prefix_for_weight_version=None,
    )


def completion(executor):
    until = time.monotonic() + 3
    while time.monotonic() < until:
        facts = executor.poll().completions
        if facts:
            assert len(facts) == 1
            return facts[0]
        time.sleep(0.001)
    raise AssertionError("completion did not retire")


def test_page_keys_follow_replica_while_state_keys_follow_rank():
    backend = Backend()
    a, b, other = (
        make_executor(backend, Host(), rank, extra_config=None) for rank in [0, 1, 2]
    )
    try:
        assert a._object_key("h", 0, 0) == b._object_key("h", 0, 0)
        assert a._object_key("h", 0, 0) != other._object_key("h", 0, 0)
        assert a._object_key("h", 1, 0) != b._object_key("h", 1, 0)
    finally:
        for client in (a, b, other):
            client._shutdown_workers(timeout_s=2)
        backend.close()


def test_source_owner_one_restores_on_destination_owner_zero():
    backend = Backend()
    source = [Host(), Host()]
    source[1].host_buffer[:4] = b"KV01"
    writers = [
        make_executor(backend, host, rank, extra_config=None)
        for rank, host in enumerate(source)
    ]
    target = [Host(), Host()]
    readers = [
        make_executor(backend, host, rank, extra_config=None)
        for rank, host in enumerate(target)
    ]
    for kind, clients, block in (("put", writers, 2), ("get", readers, 3)):
        for executor in clients:
            try:
                keys, offsets, sizes = executor._page_ranges([(0, block, "h", 0)])
                executor.submit(
                    StorageJob(1, kind, tuple(keys), tuple(offsets), tuple(sizes))
                )
                fact = completion(executor)
                assert fact.success == (True,) and fact.retired
                assert fact.object_failures == (False,)
            finally:
                # These clients deliberately share one in-memory backend.
                executor._shutdown_workers(timeout_s=2)
    assert len(backend.writes) == 1
    assert len(backend.reads) == 1
    assert target[0].host_buffer[8:12] == b"KV01"
    assert target[1].host_buffer == bytearray([0xA5] * 64)
    backend.close()


def test_async_nonowner_is_neutral_and_counts_no_io_bytes():
    backend = Backend()
    host = Host()
    owned = make_executor(backend, host, 1, extra_config=None)
    executor = owned
    try:
        keys, offsets, sizes = owned._page_ranges([(0, 3, "h", 0)])
        assert offsets == [None]
        executor.submit(StorageJob(1, "get", tuple(keys), tuple(offsets), tuple(sizes)))
        fact = completion(executor)
        assert (
            fact.success == (True,)
            and fact.retired
            and fact.object_failures == (False,)
        )
        assert not backend.reads
        row = next(row for row in executor.snapshot().operations if row.kind == "get")
        assert row.success_keys == row.success_bytes == 0
    finally:
        executor.close(timeout_s=2)


def test_empty_owner_job_cannot_recover_an_unhealthy_backend():
    backend = Backend()
    backend.fail_query = True
    owned = make_executor(backend, Host(), 1, extra_config=None)
    executor = owned
    try:
        executor.submit(StorageJob(1, "query", (owned._object_key("h", 0, 0),), (), ()))
        assert completion(executor).success == (False,)
        before = executor.snapshot().health
        assert before.state != "healthy"
        keys, offsets, sizes = owned._page_ranges([(0, 3, "h", 0)])
        executor.submit(StorageJob(2, "get", tuple(keys), tuple(offsets), tuple(sizes)))
        assert completion(executor).success == (True,)
        after = executor.snapshot().health
        assert after == before
        assert not backend.reads
    finally:
        executor.close(timeout_s=2)


def test_mixed_owner_results_keep_object_failures_in_logical_order():
    backend = Backend()
    owned = make_executor(backend, Host(), 0, extra_config=None)
    keys, offsets, sizes = owned._page_ranges(
        [(0, 2, "absent-peer", 0), (0, 3, "absent-local", 0), (1, 1, "state", 0)]
    )
    backend.batch_put_from([keys[2]], bytearray(b"DATA"), [0], [4])
    executor = owned
    try:
        executor.submit(StorageJob(1, "get", tuple(keys), tuple(offsets), tuple(sizes)))
        fact = completion(executor)
        assert fact.success == (True, False, True)
        assert fact.object_failures == (False, True, False)
        assert fact.retired
        row = next(row for row in executor.snapshot().operations if row.kind == "get")
        assert row.success_keys == 1 and row.failed_keys == 1 and row.success_bytes == 4
        assert backend.reads[0][0] == (keys[1], keys[2])
    finally:
        executor.close(timeout_s=2)


def test_backend_failure_and_cooldown_preserve_nonowner_neutrality():
    backend = Backend()
    backend.fail_get = True
    owned = make_executor(
        backend, Host(), 0, extra_config='{"async": {"recovery_interval_ms": 60000}}'
    )
    keys, offsets, sizes = owned._page_ranges([(0, 2, "peer", 0), (0, 3, "local", 0)])
    executor = owned
    try:
        for op_id in (1, 2):
            executor.submit(
                StorageJob(op_id, "get", tuple(keys), tuple(offsets), tuple(sizes))
            )
            fact = completion(executor)
            assert fact.success == (True, False)
            assert fact.retired and fact.object_failures == (False, False)
        assert len(backend.reads) == 1
        row = next(row for row in executor.snapshot().operations if row.kind == "get")
        assert row.failed_keys == 2 and row.success_keys == row.success_bytes == 0
    finally:
        executor.close(timeout_s=2)
