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

from __future__ import annotations

import os
import unittest

from tokenspeed.runtime.cache.l3.backend import (
    MemoryKvStore,
    storage_key_prefix,
    storage_object_key,
)
from tokenspeed.runtime.cache.l3.executor import L3HostStore
from tokenspeed.runtime.cache.l3.factory import create_kvstore_storage_backend
from tokenspeed.runtime.cache.l3.mooncake import MooncakeStoreConfig, parse_extra_config


class _FakeHost:
    def __init__(self, payload: bytes, *, size: int = 64):
        self.host_buffer = bytearray(size)
        self.host_buffer[: len(payload)] = payload
        self._payload_size = len(payload)

    def host_block_range(self, group_index: int, block_id: int) -> tuple[int, int]:
        del group_index
        offset = (int(block_id) - 1) * self._payload_size
        return offset, self._payload_size


class StorageKeyTest(unittest.TestCase):
    def test_object_key_includes_group_offset_and_rank(self):
        self.assertEqual(
            storage_object_key("abc", 1, 2, prefix="model", rank=3),
            "model_abc|g1|o2|r3",
        )
        self.assertEqual(storage_object_key("abc", 0, 0), "abc|g0|o0|r0")
        self.assertEqual(storage_key_prefix("org/model"), "org_model")


class MemoryKvStoreTest(unittest.TestCase):
    def test_round_trips_host_bytes(self):
        store = MemoryKvStore()
        host = bytearray(b"\x00" * 16)
        host[4:8] = b"kvkv"
        self.assertEqual(store.batch_put_from(["k"], host, [4], [4]), [True])
        self.assertEqual(store.batch_exists(["k", "missing"]), [True, False])
        dest = bytearray(b"\xff" * 16)
        self.assertEqual(store.batch_get_into(["k"], dest, [8], [4]), [True])
        self.assertEqual(dest[8:12], b"kvkv")
        host[4:8] = b"xxxx"
        self.assertEqual(store.batch_put_from(["k"], host, [4], [4]), [True])
        dest2 = bytearray(16)
        store.batch_get_into(["k"], dest2, [0], [4])
        self.assertEqual(dest2[:4], b"kvkv")
        store.close()
        self.assertEqual(store.batch_exists(["k"]), [False])


class L3HostStoreTest(unittest.TestCase):
    def test_backups_and_prefetches_packed_pages(self):
        backend = MemoryKvStore()
        host = _FakeHost(b"abcdefgh")
        l3 = L3HostStore(backend, host, key_prefix="m", rank=1)
        pages = [(0, 1, "h0", 0)]
        self.assertEqual(l3.backup(pages), [True])
        self.assertEqual(l3.exists(pages), [True])
        host.host_buffer[:8] = b"\x00" * 8
        self.assertEqual(l3.prefetch(pages), [True])
        self.assertEqual(host.host_buffer[:8], b"abcdefgh")
        groups, hashes, offsets = l3.present_keys([0, 0], ["h0", "miss"], [0, 0])
        self.assertEqual(groups, [0])
        self.assertEqual(hashes, ["h0"])
        self.assertEqual(offsets, [0])
        self.assertEqual(
            l3.present_keys([0], ["h0"], [0], exists=[False]), ([], [], [])
        )
        l3.close()


class FactoryTest(unittest.TestCase):
    def test_memory_and_unknown_backend(self):
        backend = create_kvstore_storage_backend("memory", None, host_buffer=object())
        self.assertIsInstance(backend, MemoryKvStore)
        self.assertIsNone(
            create_kvstore_storage_backend(None, None, host_buffer=object())
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            create_kvstore_storage_backend("nfs", None, host_buffer=object())


class MooncakeConfigTest(unittest.TestCase):
    def test_from_mapping_and_size_suffixes(self):
        saved = {
            key: os.environ.pop(key)
            for key in ("MOONCAKE_MASTER", "MOONCAKE_CLIENT")
            if key in os.environ
        }
        try:
            with self.assertRaisesRegex(ValueError, "master_server_address"):
                MooncakeStoreConfig.from_mapping({})
            config = MooncakeStoreConfig.from_mapping(
                {
                    "client_server_address": "127.0.0.1:50051",
                    "global_segment_size": "2gb",
                    "protocol": "rdma",
                    "tenant_id": "ts",
                }
            )
            self.assertEqual(config.master_server_address, "127.0.0.1:50051")
            self.assertEqual(config.global_segment_size, 2 * 1024 * 1024 * 1024)
            self.assertEqual(config.protocol, "rdma")
            self.assertEqual(config.tenant_id, "ts")
            parsed = parse_extra_config('{"master_server_address": "host:1"}')
            self.assertEqual(parsed["master_server_address"], "host:1")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                parse_extra_config("[]")
        finally:
            os.environ.update(saved)


if __name__ == "__main__":
    unittest.main()
