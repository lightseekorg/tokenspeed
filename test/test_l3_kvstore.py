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
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from tokenspeed.runtime.cache.l3.backend import (
    MemoryKvStore,
    cache_layout_signature,
    storage_key_prefix,
    storage_object_key,
)
from tokenspeed.runtime.cache.l3.executor import L3HostStore
from tokenspeed.runtime.cache.l3.factory import create_kvstore_storage_backend
from tokenspeed.runtime.cache.l3.mooncake import (
    MooncakeKvStore,
    MooncakeStoreConfig,
    parse_extra_config,
)


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

    def test_prefix_is_stable_and_separates_incompatible_cache_objects(self):
        def prefix(**overrides):
            values = {
                "model_name": "org/model",
                "revision": "abc",
                "weight_version": "v1",
                "cache_signature": "layout",
                "pipeline_rank": 0,
                "draft_model": "",
                "draft_revision": "",
                "draft_weight_version": "",
            }
            values.update(overrides)
            return storage_key_prefix(**values)

        base = prefix()
        self.assertEqual(base, prefix())
        self.assertTrue(base.startswith("tsl3v1-"))
        self.assertNotEqual(base, prefix(model_name="org_model"))
        self.assertNotEqual(base, prefix(revision="def"))
        self.assertNotEqual(base, prefix(weight_version="v2"))
        self.assertNotEqual(base, prefix(cache_signature="other"))
        self.assertNotEqual(base, prefix(pipeline_rank=1))
        self.assertNotEqual(base, prefix(draft_model="org/draft"))
        with self.assertRaises(TypeError):
            storage_key_prefix("org/model")

    def test_layout_signature_includes_dtype_and_byte_geometry(self):
        field = SimpleNamespace(
            field_id="k0",
            device_buffer_index=0,
            device_block_zero_offset_bytes=16,
            block_stride_bytes=32,
            payload_bytes=24,
        )
        layout = SimpleNamespace(
            groups=(
                SimpleNamespace(
                    group_id="full", cache_blocks_per_lcm_block=1, fields=(field,)
                ),
            )
        )
        fp16 = cache_layout_signature(layout, cache_dtype="float16")
        self.assertEqual(fp16, cache_layout_signature(layout, cache_dtype="float16"))
        self.assertNotEqual(
            fp16, cache_layout_signature(layout, cache_dtype="bfloat16")
        )
        changed = SimpleNamespace(
            groups=(
                SimpleNamespace(
                    group_id="full",
                    cache_blocks_per_lcm_block=1,
                    fields=(SimpleNamespace(**{**vars(field), "payload_bytes": 20}),),
                ),
            )
        )
        self.assertNotEqual(
            fp16, cache_layout_signature(changed, cache_dtype="float16")
        )


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

    def test_namespace_clear_deletes_objects_without_changing_the_prefix(self):
        backend = MemoryKvStore()
        host = _FakeHost(b"abcdefgh")
        l3 = L3HostStore(backend, host, key_prefix="m", rank=1)
        pages = [(0, 1, "h0", 0)]
        self.assertEqual(l3.backup(pages), [True])
        old_key = l3.object_key("h0", 0, 0)
        l3.rotate_namespace()
        self.assertEqual(old_key, l3.object_key("h0", 0, 0))
        self.assertEqual(l3.exists(pages), [False])
        restarted = L3HostStore(backend, host, key_prefix="m", rank=1)
        self.assertEqual(restarted.object_key("h0", 0, 0), old_key)
        self.assertEqual(restarted.exists(pages), [False])

    def test_clear_raises_when_remote_delete_fails_and_keeps_the_prefix(self):
        backend = mock.Mock()
        backend.remove_by_prefix.side_effect = RuntimeError("delete failed")
        l3 = L3HostStore(backend, _FakeHost(b"abcdefgh"), key_prefix="m", rank=1)
        old_key = l3.object_key("h0", 0, 0)

        with self.assertRaisesRegex(RuntimeError, "delete failed"):
            l3.rotate_namespace()

        self.assertEqual(old_key, l3.object_key("h0", 0, 0))


class FactoryTest(unittest.TestCase):
    def test_memory_and_unknown_backend(self):
        backend = create_kvstore_storage_backend(
            "memory", None, host_buffer=object(), tp_size=1, pp_size=1
        )
        self.assertIsInstance(backend, MemoryKvStore)
        self.assertIsNone(
            create_kvstore_storage_backend(
                None, None, host_buffer=object(), tp_size=1, pp_size=1
            )
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            create_kvstore_storage_backend(
                "nfs", None, host_buffer=object(), tp_size=1, pp_size=1
            )


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

    def test_default_global_segment_matches_runtime_flag_default(self):
        saved = os.environ.pop("MOONCAKE_GLOBAL_SEGMENT_SIZE", None)
        try:
            config = MooncakeStoreConfig.from_mapping(
                {"master_server_address": "127.0.0.1:50051"}
            )
            self.assertEqual(config.global_segment_size, 4 * 1024**3)
        finally:
            if saved is not None:
                os.environ["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = saved


class MooncakeKvStoreTest(unittest.TestCase):
    def test_non_default_tenant_is_never_silently_dropped(self):
        class _Store:
            def setup(self, *args, **kwargs):
                del args
                if "tenant_id" in kwargs:
                    raise TypeError("old Mooncake")
                raise AssertionError("must not retry without the requested tenant")

        store_module = types.ModuleType("mooncake.store")
        store_module.MooncakeDistributedStore = _Store
        package = types.ModuleType("mooncake")
        package.store = store_module
        host = SimpleNamespace(
            data_ptr=lambda: 1, numel=lambda: 8, element_size=lambda: 1
        )
        with mock.patch.dict(
            sys.modules, {"mooncake": package, "mooncake.store": store_module}
        ):
            with self.assertRaisesRegex(RuntimeError, "does not support tenant_id"):
                MooncakeKvStore(
                    {
                        "master_server_address": "127.0.0.1:50051",
                        "tenant_id": "tenant-a",
                    },
                    host_buffer=host,
                    tp_size=1,
                    pp_size=1,
                )

    def test_concurrent_create_is_treated_as_idempotent_success(self):
        class _Store:
            def __init__(self):
                self.exists_calls = 0

            def batch_is_exist(self, keys):
                self.exists_calls += 1
                return [0] * len(keys) if self.exists_calls == 1 else [1] * len(keys)

            def batch_put_from(self, keys, ptrs, sizes):
                del ptrs, sizes
                return [-1] * len(keys)

        adapter = object.__new__(MooncakeKvStore)
        adapter.store = _Store()
        host = SimpleNamespace(data_ptr=lambda: 100)
        self.assertEqual(adapter.batch_put_from(["k"], host, [0], [8]), [True])

    def test_namespace_clear_uses_anchored_escaped_regex(self):
        adapter = object.__new__(MooncakeKvStore)
        store = mock.Mock()
        adapter.store = store
        store.remove_by_regex.return_value = 0

        adapter.remove_by_prefix("model.v1_")

        store.remove_by_regex.assert_called_once_with(r"^model\.v1_.*", True)

    def test_namespace_clear_failure_is_not_reported_as_success(self):
        adapter = object.__new__(MooncakeKvStore)
        adapter.store = mock.Mock()
        adapter.store.remove_by_regex.return_value = -1
        with self.assertRaisesRegex(RuntimeError, "Failed to clear"):
            adapter.remove_by_prefix("model_")

    def test_segment_is_divided_across_tp_and_pp_ranks(self):
        captured = {}

        class _Store:
            def setup(self, *_args, **_kwargs):
                captured["segment"] = _args[2]
                return 0

            def register_buffer(self, ptr, size):
                del ptr, size
                return 0

        store_module = types.ModuleType("mooncake.store")
        store_module.MooncakeDistributedStore = _Store
        package = types.ModuleType("mooncake")
        package.store = store_module
        host = SimpleNamespace(
            data_ptr=lambda: 1, numel=lambda: 8, element_size=lambda: 1
        )
        with mock.patch.dict(
            sys.modules, {"mooncake": package, "mooncake.store": store_module}
        ):
            MooncakeKvStore(
                {
                    "master_server_address": "127.0.0.1:50051",
                    "global_segment_size": 8 * 1024**3,
                },
                host_buffer=host,
                tp_size=2,
                pp_size=2,
            )
        self.assertEqual(captured["segment"], 2 * 1024**3)


if __name__ == "__main__":
    unittest.main()
