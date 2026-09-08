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

import inspect
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from tokenspeed.runtime.cache.l3.backend import (
    L3_FLUSH_REQUIRES_WEIGHT_VERSION,
    L3UnreadKeySet,
    MemoryKvStore,
    cache_layout_signature,
    l3_cache_quantization_id,
    l3_checkpoint_id,
    l3_pages_newly_published,
    l3_unread_key_capacity,
    resolve_l3_weight_version,
    share_l3_checkpoint_ids,
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
    def test_object_key_includes_group_offset_and_ranks(self):
        self.assertEqual(
            storage_object_key("abc", 1, 2, prefix="model", rank=3, cp_rank=4),
            "model_abc|g1|o2|r3|c4",
        )
        self.assertEqual(
            storage_object_key("abc", 0, 0, prefix="", rank=0, cp_rank=0),
            "abc|g0|o0|r0|c0",
        )
        self.assertNotEqual(
            storage_object_key("abc", 0, 0, prefix="", rank=0, cp_rank=0),
            storage_object_key("abc", 0, 0, prefix="", rank=0, cp_rank=1),
        )

    def test_prefix_is_stable_and_separates_incompatible_cache_objects(self):
        def prefix(**overrides):
            values = {
                "model_name": "org/model",
                "revision": "abc",
                "weight_version": "v1",
                "model_overrides": {},
                "cache_signature": "layout",
                "pipeline_rank": 0,
                "attn_tp_size": 1,
                "cp_size": 1,
                "draft_model": "",
                "draft_revision": "",
                "draft_weight_version": "",
                "cache_quantization": "",
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
        self.assertNotEqual(base, prefix(attn_tp_size=8))
        self.assertNotEqual(prefix(attn_tp_size=8), prefix(attn_tp_size=16))
        self.assertNotEqual(base, prefix(cp_size=2))
        self.assertNotEqual(prefix(cp_size=2), prefix(cp_size=4))
        self.assertNotEqual(base, prefix(draft_model="org/draft"))
        self.assertNotEqual(base, prefix(cache_quantization='{"quantization":"fp8"}'))
        self.assertNotEqual(base, prefix(model_overrides={"rope_theta": 10000.0}))
        self.assertNotEqual(
            prefix(model_overrides={"rope_theta": 10000.0}),
            prefix(model_overrides={"rope_scaling": {"type": "linear"}}),
        )
        self.assertEqual(
            prefix(model_overrides={"b": 2, "a": 1}),
            prefix(model_overrides={"a": 1, "b": 2}),
        )
        with self.assertRaises(TypeError):
            storage_key_prefix("org/model")
        with self.assertRaises(TypeError):
            prefix(model_overrides=["rope_theta"])
        signature = inspect.signature(storage_key_prefix)
        self.assertIs(
            signature.parameters["cache_quantization"].default, inspect.Parameter.empty
        )
        self.assertIs(signature.parameters["revision"].default, inspect.Parameter.empty)
        self.assertIs(
            signature.parameters["model_overrides"].default, inspect.Parameter.empty
        )
        self.assertIs(
            signature.parameters["attn_tp_size"].default, inspect.Parameter.empty
        )

    def _checkpoint_id(self, model_path, *, load_format, hf_config, revision):
        return l3_checkpoint_id(
            model_path,
            hf_config=hf_config,
            revision=revision,
            load_format=load_format,
            model_loader_extra_config={},
        )

    def test_checkpoint_id_prefers_loaded_commit_over_moving_branch(self):
        commit = "a" * 40
        other = "b" * 40
        self.assertEqual(
            self._checkpoint_id(
                "org/model",
                hf_config=SimpleNamespace(_commit_hash=commit),
                revision="main",
                load_format="auto",
            ),
            f"{commit}:auto",
        )
        self.assertNotEqual(
            self._checkpoint_id(
                "org/model",
                hf_config=SimpleNamespace(_commit_hash=commit),
                revision="main",
                load_format="auto",
            ),
            self._checkpoint_id(
                "org/model",
                hf_config=SimpleNamespace(_commit_hash=other),
                revision="main",
                load_format="auto",
            ),
        )
        self.assertNotEqual(
            self._checkpoint_id(
                "org/model",
                hf_config=SimpleNamespace(_commit_hash=commit),
                revision="main",
                load_format="auto",
            ),
            self._checkpoint_id(
                "org/model",
                hf_config=SimpleNamespace(_commit_hash=commit),
                revision="main",
                load_format="pt",
            ),
        )

    def test_checkpoint_id_uses_snapshot_directory_commit(self):
        commit = "c" * 40
        with tempfile.TemporaryDirectory() as root:
            repo = os.path.join(root, "hub", "models--org--model")
            snapshot = os.path.join(repo, "snapshots", commit)
            os.makedirs(os.path.join(repo, "refs"))
            os.makedirs(snapshot)
            with open(os.path.join(snapshot, "config.json"), "w") as handle:
                handle.write("{}")
            self.assertEqual(
                self._checkpoint_id(
                    snapshot,
                    hf_config=SimpleNamespace(),
                    revision="main",
                    load_format="safetensors",
                ),
                f"{commit}:safetensors",
            )

    def test_checkpoint_id_does_not_trust_hub_snapshots_without_refs(self):
        commit = "f" * 40
        with tempfile.TemporaryDirectory() as root:
            repo = os.path.join(root, "hub", "models--org--model")
            snapshot = os.path.join(repo, "snapshots", commit)
            os.makedirs(snapshot)
            with open(os.path.join(snapshot, "config.json"), "w") as handle:
                handle.write('{"model_type":"x"}')
            with open(os.path.join(snapshot, "model.safetensors"), "wb") as handle:
                handle.write(b"aaa")
            checkpoint_id = self._checkpoint_id(
                snapshot,
                load_format="auto",
                hf_config=SimpleNamespace(),
                revision="",
            )
            self.assertTrue(checkpoint_id.startswith("local-"))
            self.assertNotEqual(checkpoint_id, f"{commit}:auto")

    def test_checkpoint_id_does_not_trust_snapshots_dir_outside_hf_hub_cache(self):
        commit = "e" * 40
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            first = os.path.join(first_root, "models", "snapshots", commit)
            second = os.path.join(second_root, "models", "snapshots", commit)
            os.makedirs(first)
            os.makedirs(second)
            for directory, payload in ((first, b"aaa"), (second, b"bbb")):
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write('{"model_type":"x"}')
                with open(os.path.join(directory, "model.safetensors"), "wb") as handle:
                    handle.write(payload)
            first_id = self._checkpoint_id(
                first,
                load_format="auto",
                hf_config=SimpleNamespace(),
                revision="",
            )
            second_id = self._checkpoint_id(
                second,
                load_format="auto",
                hf_config=SimpleNamespace(),
                revision="",
            )
            self.assertTrue(first_id.startswith("local-"))
            self.assertTrue(second_id.startswith("local-"))
            self.assertNotEqual(first_id, second_id)
            self.assertNotEqual(first_id, f"{commit}:auto")

    def test_checkpoint_id_fingerprints_local_weight_bytes(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory, payload in ((first, b"aaa"), (second, b"bbb")):
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write('{"model_type":"x"}')
                with open(os.path.join(directory, "model.safetensors"), "wb") as handle:
                    handle.write(payload)
            self.assertEqual(
                os.path.getsize(os.path.join(first, "model.safetensors")), 3
            )
            self.assertEqual(
                os.path.getsize(os.path.join(second, "model.safetensors")), 3
            )
            self.assertNotEqual(
                self._checkpoint_id(
                    first,
                    load_format="auto",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
                self._checkpoint_id(
                    second,
                    load_format="auto",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
            )
            self.assertTrue(
                self._checkpoint_id(
                    first,
                    load_format="auto",
                    hf_config=SimpleNamespace(),
                    revision="",
                ).startswith("local-")
            )

    def test_checkpoint_id_does_not_trust_hex_basename_outside_hf_snapshots(self):
        commit = "d" * 40
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            first = os.path.join(first_root, commit)
            second = os.path.join(second_root, commit)
            os.makedirs(first)
            os.makedirs(second)
            for directory, payload in ((first, b"aaa"), (second, b"bbb")):
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write('{"model_type":"x"}')
                with open(os.path.join(directory, "model.safetensors"), "wb") as handle:
                    handle.write(payload)
            first_id = self._checkpoint_id(
                first,
                load_format="auto",
                hf_config=SimpleNamespace(),
                revision="",
            )
            second_id = self._checkpoint_id(
                second,
                load_format="auto",
                hf_config=SimpleNamespace(),
                revision="",
            )
            self.assertTrue(first_id.startswith("local-"))
            self.assertTrue(second_id.startswith("local-"))
            self.assertNotEqual(first_id, second_id)
            self.assertNotEqual(first_id, f"{commit}:auto")

    def test_checkpoint_id_ignores_inherited_commit_on_local_dir(self):
        inherited = "a" * 40
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory, payload in ((first, b"aaa"), (second, b"bbb")):
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write('{"model_type":"x"}')
                with open(os.path.join(directory, "model.safetensors"), "wb") as handle:
                    handle.write(payload)
            first_id = self._checkpoint_id(
                first,
                hf_config=SimpleNamespace(_commit_hash=inherited),
                revision="",
                load_format="auto",
            )
            second_id = self._checkpoint_id(
                second,
                hf_config=SimpleNamespace(_commit_hash=inherited),
                revision="",
                load_format="auto",
            )
            self.assertNotEqual(first_id, second_id)
            self.assertFalse(first_id.startswith(inherited))
            self.assertTrue(first_id.startswith("local-"))

    def test_checkpoint_id_fingerprints_local_hf_quant_config(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory, kv_algo in ((first, "FP8"), (second, "INT8")):
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write('{"model_type":"x"}')
                with open(os.path.join(directory, "model.safetensors"), "wb") as handle:
                    handle.write(b"weights")
                with open(
                    os.path.join(directory, "hf_quant_config.json"), "w"
                ) as handle:
                    handle.write(
                        '{"quantization":{"quant_algo":"NVFP4",'
                        f'"kv_cache_quant_algo":"{kv_algo}"}}'
                    )
            self.assertNotEqual(
                self._checkpoint_id(
                    first,
                    load_format="auto",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
                self._checkpoint_id(
                    second,
                    load_format="auto",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
            )

    def test_checkpoint_id_fingerprints_local_config_code(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory, source in (
                (first, "class Config: rope_theta = 10000\n"),
                (second, "class Config: rope_theta = 500000\n"),
            ):
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write(
                        '{"model_type":"x","auto_map":{"AutoConfig":"configuration.Config"}}'
                    )
                with open(os.path.join(directory, "model.safetensors"), "wb") as handle:
                    handle.write(b"weights")
                with open(os.path.join(directory, "configuration.py"), "w") as handle:
                    handle.write(source)
            self.assertNotEqual(
                self._checkpoint_id(
                    first,
                    load_format="auto",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
                self._checkpoint_id(
                    second,
                    load_format="auto",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
            )

    def test_checkpoint_id_fingerprints_imported_custom_code_subdirectories(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory, helper in (
                (first, "def rotary_dim():\n    return 64\n"),
                (second, "def rotary_dim():\n    return 128\n"),
            ):
                helpers = os.path.join(directory, "model_helpers")
                os.makedirs(helpers)
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write(
                        '{"model_type":"x","auto_map":{"AutoConfig":"configuration.Config"}}'
                    )
                with open(os.path.join(directory, "model.safetensors"), "wb") as handle:
                    handle.write(b"weights")
                with open(os.path.join(directory, "configuration.py"), "w") as handle:
                    handle.write("from model_helpers.attention import rotary_dim\n")
                with open(os.path.join(helpers, "__init__.py"), "w") as handle:
                    handle.write("")
                with open(os.path.join(helpers, "attention.py"), "w") as handle:
                    handle.write(helper)
            first_id = self._checkpoint_id(
                first,
                load_format="auto",
                hf_config=SimpleNamespace(),
                revision="",
            )
            second_id = self._checkpoint_id(
                second,
                load_format="auto",
                hf_config=SimpleNamespace(),
                revision="",
            )
            self.assertTrue(first_id.startswith("local-"))
            self.assertTrue(second_id.startswith("local-"))
            self.assertNotEqual(first_id, second_id)

    def test_checkpoint_id_uses_selected_load_format_weight_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "config.json"), "w") as handle:
                handle.write("{}")
            with open(os.path.join(directory, "model.safetensors"), "wb") as handle:
                handle.write(b"safe-weights")
            with open(os.path.join(directory, "pytorch_model.bin"), "wb") as handle:
                handle.write(b"bin-weights")
            with open(os.path.join(directory, "model.pt"), "wb") as handle:
                handle.write(b"pt-weights")
            safetensors_id = self._checkpoint_id(
                directory,
                load_format="safetensors",
                hf_config=SimpleNamespace(),
                revision="",
            )
            bin_id = self._checkpoint_id(
                directory,
                load_format="npcache",
                hf_config=SimpleNamespace(),
                revision="",
            )
            pt_id = self._checkpoint_id(
                directory,
                load_format="pt",
                hf_config=SimpleNamespace(),
                revision="",
            )
            auto_id = self._checkpoint_id(
                directory,
                load_format="auto",
                hf_config=SimpleNamespace(),
                revision="",
            )
            self.assertNotEqual(safetensors_id, bin_id)
            self.assertNotEqual(safetensors_id, pt_id)
            self.assertNotEqual(bin_id, pt_id)
            self.assertNotEqual(auto_id, safetensors_id)
            self.assertTrue(safetensors_id.endswith(":safetensors"))
            self.assertTrue(auto_id.endswith(":auto"))

    def test_checkpoint_id_ignores_unselected_weight_files(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory, leftover in ((first, b"aaa"), (second, b"bbb")):
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write("{}")
                with open(os.path.join(directory, "model.safetensors"), "wb") as handle:
                    handle.write(b"same-safe")
                with open(os.path.join(directory, "pytorch_model.bin"), "wb") as handle:
                    handle.write(leftover)
            self.assertEqual(
                self._checkpoint_id(
                    first,
                    load_format="safetensors",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
                self._checkpoint_id(
                    second,
                    load_format="safetensors",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
            )
            self.assertNotEqual(
                self._checkpoint_id(
                    first,
                    load_format="npcache",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
                self._checkpoint_id(
                    second,
                    load_format="npcache",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
            )

    def test_checkpoint_id_fingerprints_sharded_state_weight_files(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory, payload in ((first, b"rank0-a"), (second, b"rank0-b")):
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write("{}")
                with open(
                    os.path.join(directory, "model-rank-0-part-0.safetensors"),
                    "wb",
                ) as handle:
                    handle.write(payload)
            self.assertNotEqual(
                self._checkpoint_id(
                    first,
                    load_format="sharded_state",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
                self._checkpoint_id(
                    second,
                    load_format="sharded_state",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
            )

    def test_checkpoint_id_fingerprints_sharded_state_pattern_files(self):
        """Custom shard patterns must hash the files the loader would open."""

        from tokenspeed.runtime.cache.l3 import backend as l3_backend

        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "config.json"), "w") as handle:
                handle.write("{}")
            default_shard = os.path.join(directory, "model-rank-0-part-0.safetensors")
            alt_shard = os.path.join(directory, "alt-rank-0-part-0.bin")
            with open(default_shard, "wb") as handle:
                handle.write(b"default-shard")
            with open(alt_shard, "wb") as handle:
                handle.write(b"alt-shard-a")
            default_id = l3_checkpoint_id(
                directory,
                hf_config=SimpleNamespace(),
                revision="",
                load_format="sharded_state",
                model_loader_extra_config={},
            )
            extra = {"pattern": "alt-rank-{rank}-part-{part}.bin"}
            alt_id = l3_checkpoint_id(
                directory,
                hf_config=SimpleNamespace(),
                revision="",
                load_format="sharded_state",
                model_loader_extra_config=extra,
            )
            self.assertNotEqual(default_id, alt_id)
            with open(default_shard, "wb") as handle:
                handle.write(b"default-shard-changed")
            l3_backend._local_checkpoint_fingerprint.cache_clear()
            self.assertEqual(
                alt_id,
                l3_checkpoint_id(
                    directory,
                    hf_config=SimpleNamespace(),
                    revision="",
                    load_format="sharded_state",
                    model_loader_extra_config=extra,
                ),
            )
            with open(alt_shard, "wb") as handle:
                handle.write(b"alt-shard-b")
            l3_backend._local_checkpoint_fingerprint.cache_clear()
            self.assertNotEqual(
                alt_id,
                l3_checkpoint_id(
                    directory,
                    hf_config=SimpleNamespace(),
                    revision="",
                    load_format="sharded_state",
                    model_loader_extra_config=extra,
                ),
            )

    def test_checkpoint_id_fingerprints_npcache_numpy_files(self):
        """When np/ exists, npcache loads it and must not hash only *.bin."""

        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory, np_payload in ((first, b"np-a"), (second, b"np-b")):
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write("{}")
                with open(os.path.join(directory, "pytorch_model.bin"), "wb") as handle:
                    handle.write(b"same-bin")
                os.makedirs(os.path.join(directory, "np"))
                with open(
                    os.path.join(directory, "np", "weight_names.json"),
                    "w",
                ) as handle:
                    handle.write('["w"]')
                with open(os.path.join(directory, "np", "w"), "wb") as handle:
                    handle.write(np_payload)
            self.assertNotEqual(
                self._checkpoint_id(
                    first,
                    load_format="npcache",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
                self._checkpoint_id(
                    second,
                    load_format="npcache",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
            )

    def test_checkpoint_id_fingerprints_mistral_shard_index(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory, mapped in (
                (first, "consolidated.00.safetensors"),
                (second, "consolidated.01.safetensors"),
            ):
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write("{}")
                with open(
                    os.path.join(directory, "consolidated.00.safetensors"),
                    "wb",
                ) as handle:
                    handle.write(b"shard-a")
                with open(
                    os.path.join(directory, "consolidated.01.safetensors"),
                    "wb",
                ) as handle:
                    handle.write(b"shard-b")
                with open(
                    os.path.join(directory, "consolidated.safetensors.index.json"),
                    "w",
                ) as handle:
                    handle.write(f'{{"weight_map":{{"w":"{mapped}"}}}}')
            self.assertNotEqual(
                self._checkpoint_id(
                    first,
                    load_format="mistral",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
                self._checkpoint_id(
                    second,
                    load_format="mistral",
                    hf_config=SimpleNamespace(),
                    revision="",
                ),
            )

    def test_checkpoint_id_rejects_unsupported_load_format(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "config.json"), "w") as handle:
                handle.write("{}")
            with self.assertRaises(ValueError):
                self._checkpoint_id(
                    directory,
                    load_format="extensible",
                    hf_config=SimpleNamespace(),
                    revision="",
                )

    def test_checkpoint_id_requires_load_format(self):
        signature = inspect.signature(l3_checkpoint_id)
        self.assertIs(
            signature.parameters["load_format"].default, inspect.Parameter.empty
        )
        self.assertIs(
            signature.parameters["model_loader_extra_config"].default,
            inspect.Parameter.empty,
        )
        with self.assertRaises(TypeError):
            l3_checkpoint_id(
                "org/model",
                hf_config=SimpleNamespace(_commit_hash="a" * 40),
                revision="main",
            )
        with self.assertRaises(ValueError):
            self._checkpoint_id(
                "org/model",
                hf_config=SimpleNamespace(_commit_hash="a" * 40),
                revision="main",
                load_format="  ",
            )

    def test_local_fingerprint_is_cached_per_directory(self):
        from tokenspeed.runtime.cache.l3 import backend as l3_backend

        l3_backend._local_checkpoint_fingerprint.cache_clear()
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "config.json"), "w") as handle:
                handle.write("{}")
            with open(os.path.join(directory, "model.safetensors"), "wb") as handle:
                handle.write(b"weights")
            with mock.patch.object(
                l3_backend,
                "_update_file_digest",
                wraps=l3_backend._update_file_digest,
            ) as digest:
                first = self._checkpoint_id(
                    directory,
                    load_format="auto",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
                second = self._checkpoint_id(
                    directory,
                    load_format="auto",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            self.assertEqual(first, second)
            self.assertEqual(digest.call_count, 2)

    def test_share_l3_checkpoint_ids_combines_rank_local_digests(self):
        seen = []

        def gather_from(rows):
            def gather(payload):
                seen.append(list(payload))
                return rows

            return gather

        shared = ["local-aaa:sharded_state", ""]
        self.assertEqual(
            share_l3_checkpoint_ids(
                shared,
                rank=0,
                world_size=1,
                gather=gather_from([shared]),
            ),
            shared,
        )
        self.assertEqual(seen, [])
        self.assertEqual(
            share_l3_checkpoint_ids(
                shared,
                rank=1,
                world_size=2,
                gather=gather_from([shared, shared]),
            ),
            shared,
        )
        signature = inspect.signature(share_l3_checkpoint_ids)
        self.assertIs(signature.parameters["gather"].default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            share_l3_checkpoint_ids(
                shared,
                rank=0,
                world_size=2,
            )
        rank0 = ["local-rank0:sharded_state", ""]
        rank1_a = ["local-rank1a:sharded_state", ""]
        rank1_b = ["local-rank1b:sharded_state", ""]
        combined_a = share_l3_checkpoint_ids(
            rank0,
            rank=0,
            world_size=2,
            gather=gather_from([rank0, rank1_a]),
        )
        combined_b = share_l3_checkpoint_ids(
            rank0,
            rank=0,
            world_size=2,
            gather=gather_from([rank0, rank1_b]),
        )
        self.assertTrue(combined_a[0].startswith("local-"))
        self.assertNotEqual(combined_a[0], rank0[0])
        self.assertNotEqual(combined_a, combined_b)
        self.assertEqual(combined_a[1], "")
        self.assertEqual(combined_b[1], "")
        with self.assertRaises(ValueError):
            share_l3_checkpoint_ids(
                rank0,
                rank=0,
                world_size=2,
                gather=gather_from([rank0]),
            )

    def test_checkpoint_id_rejects_unpinned_remote_without_commit(self) -> None:
        with self.assertRaises(ValueError):
            l3_checkpoint_id(
                "org/unpinned-model",
                hf_config=SimpleNamespace(),
                revision="main",
                load_format="auto",
                model_loader_extra_config={},
            )

    def test_cache_quantization_id_hashes_scale_file_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "a.json")
            second = os.path.join(directory, "b.json")
            with open(first, "w") as handle:
                handle.write('{"scale":1}')
            with open(second, "w") as handle:
                handle.write('{"scale":2}')
            same_copy = os.path.join(directory, "a-copy.json")
            with open(same_copy, "w") as handle:
                handle.write('{"scale":1}')
            first_id = l3_cache_quantization_id(
                quantization="fp8",
                quantization_param_path=first,
                draft_quantization="",
            )
            self.assertNotEqual(
                first_id,
                l3_cache_quantization_id(
                    quantization="fp8",
                    quantization_param_path=second,
                    draft_quantization="",
                ),
            )
            self.assertEqual(
                first_id,
                l3_cache_quantization_id(
                    quantization="fp8",
                    quantization_param_path=same_copy,
                    draft_quantization="",
                ),
            )
            self.assertNotEqual(
                first_id,
                l3_cache_quantization_id(
                    quantization="",
                    quantization_param_path="",
                    draft_quantization="",
                ),
            )
            self.assertNotEqual(
                first_id,
                l3_cache_quantization_id(
                    quantization="fp8",
                    quantization_param_path=first,
                    draft_quantization="fp8",
                ),
            )
            signature = inspect.signature(l3_cache_quantization_id)
            self.assertIs(
                signature.parameters["quantization_param_path"].default,
                inspect.Parameter.empty,
            )
            self.assertIs(
                signature.parameters["draft_quantization"].default,
                inspect.Parameter.empty,
            )
            with self.assertRaises(TypeError):
                l3_cache_quantization_id(
                    quantization="fp8", quantization_param_path=first
                )

    def test_resolve_l3_weight_version_does_not_mint_a_successor(self):
        self.assertIsNone(
            resolve_l3_weight_version(
                "v1",
                None,
                flush_cache=True,
                storage_backend="memory",
            )
        )
        self.assertIsNone(
            resolve_l3_weight_version(
                "v1",
                None,
                flush_cache=True,
                storage_backend=None,
            )
        )
        self.assertIsNone(
            resolve_l3_weight_version(
                "v1",
                None,
                flush_cache=False,
                storage_backend="memory",
            )
        )
        self.assertEqual(
            resolve_l3_weight_version(
                "v1",
                "explicit",
                flush_cache=True,
                storage_backend="memory",
            ),
            "explicit",
        )
        self.assertIn("require weight_version", L3_FLUSH_REQUIRES_WEIGHT_VERSION)

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
        strided = SimpleNamespace(
            groups=(
                SimpleNamespace(
                    group_id="full",
                    cache_blocks_per_lcm_block=1,
                    fields=(
                        SimpleNamespace(**{**vars(field), "block_stride_bytes": 64}),
                    ),
                ),
            )
        )
        self.assertNotEqual(
            fp16, cache_layout_signature(strided, cache_dtype="float16")
        )
        shifted = SimpleNamespace(
            groups=(
                SimpleNamespace(
                    group_id="full",
                    cache_blocks_per_lcm_block=1,
                    fields=(
                        SimpleNamespace(
                            **{
                                **vars(field),
                                "device_buffer_index": 1,
                                "device_block_zero_offset_bytes": 4096,
                            }
                        ),
                    ),
                ),
            )
        )
        self.assertEqual(fp16, cache_layout_signature(shifted, cache_dtype="float16"))


class L3UnreadKeySetTest(unittest.TestCase):
    def test_forget_after_mark_allows_reuse(self):
        unread = L3UnreadKeySet(capacity=4)
        unread.mark([0], ["h4"], [0])
        self.assertTrue(unread.contains(0, "h4", 0))
        unread.forget([0], ["h4"], [0])
        self.assertFalse(unread.contains(0, "h4", 0))

    def test_forget_pages_matches_backup_tuples(self):
        unread = L3UnreadKeySet(capacity=4)
        unread.mark([1], ["h5"], [2])
        unread.forget_pages([(1, 99, "h5", 2)])
        self.assertFalse(unread.contains(1, "h5", 2))

    def test_newly_published_pages_exclude_objects_that_already_existed(self):
        missing = (0, 1, "h-new", 0)
        present = (0, 2, "h-old", 0)
        self.assertEqual(
            l3_pages_newly_published([missing, present], [False, True]),
            [missing],
        )
        self.assertEqual(l3_pages_newly_published([present], [True]), [])
        self.assertEqual(l3_pages_newly_published([missing], [False, True]), [])

    def test_capacity_counts_packed_cache_blocks_not_lcm_parents(self):
        self.assertEqual(
            l3_unread_key_capacity(num_host_pages=2, cache_blocks_per_lcm_block=(4, 1)),
            10,
        )
        unread = L3UnreadKeySet(
            capacity=l3_unread_key_capacity(
                num_host_pages=1, cache_blocks_per_lcm_block=(4, 1)
            )
        )
        unread.mark(
            [0, 0, 0, 0, 1],
            ["h0", "h1", "h2", "h3", "h4"],
            [0, 1, 2, 3, 0],
        )
        self.assertTrue(unread.contains(0, "h0", 0))
        self.assertTrue(unread.contains(1, "h4", 0))

    def test_capacity_evicts_oldest_failure(self):
        unread = L3UnreadKeySet(capacity=2)
        unread.mark([0, 0, 0], ["h1", "h2", "h3"], [0, 0, 0])
        self.assertFalse(unread.contains(0, "h1", 0))
        self.assertTrue(unread.contains(0, "h2", 0))
        self.assertTrue(unread.contains(0, "h3", 0))

    def test_clear_forgets_every_key(self):
        unread = L3UnreadKeySet(capacity=2)
        unread.mark([0], ["h4"], [0])
        unread.clear()
        self.assertFalse(unread.contains(0, "h4", 0))

    def test_capacity_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "capacity must be positive"):
            L3UnreadKeySet(capacity=0)

    def test_unread_capacity_rejects_non_positive_packing(self):
        with self.assertRaisesRegex(ValueError, "cache_blocks_per_lcm_block"):
            l3_unread_key_capacity(num_host_pages=2, cache_blocks_per_lcm_block=(4, 0))


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
        l3 = L3HostStore(backend, host, key_prefix="m", rank=1, cp_rank=0)
        pages = [(0, 1, "h0", 0)]
        self.assertEqual(l3.backup(pages), [True])
        self.assertEqual(l3.exists(pages), [True])
        host.host_buffer[:8] = b"\x00" * 8
        self.assertEqual(l3.prefetch(pages), [True])
        self.assertEqual(host.host_buffer[:8], b"abcdefgh")
        groups, hashes, offsets = l3.present_keys(
            [0, 0], ["h0", "miss"], [0, 0], exists=None
        )
        self.assertEqual(groups, [0])
        self.assertEqual(hashes, ["h0"])
        self.assertEqual(offsets, [0])
        self.assertEqual(
            l3.present_keys([0], ["h0"], [0], exists=[False]), ([], [], [])
        )
        l3.close()

    def test_present_keys_requires_exists_mask(self):
        backend = MemoryKvStore()
        host = _FakeHost(b"abcdefgh")
        l3 = L3HostStore(backend, host, key_prefix="m", rank=1, cp_rank=0)
        signature = inspect.signature(l3.present_keys)
        self.assertIs(signature.parameters["exists"].default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            l3.present_keys([0], ["h0"], [0])
        l3.close()

    def test_namespace_clear_deletes_objects_without_changing_the_prefix(self):
        backend = MemoryKvStore()
        host = _FakeHost(b"abcdefgh")
        l3 = L3HostStore(backend, host, key_prefix="m", rank=1, cp_rank=0)
        pages = [(0, 1, "h0", 0)]
        self.assertEqual(l3.backup(pages), [True])
        old_key = l3.object_key("h0", 0, 0)
        l3.rotate_namespace()
        self.assertEqual(old_key, l3.object_key("h0", 0, 0))
        self.assertEqual(l3.exists(pages), [False])
        restarted = L3HostStore(backend, host, key_prefix="m", rank=1, cp_rank=0)
        self.assertEqual(restarted.object_key("h0", 0, 0), old_key)
        self.assertEqual(restarted.exists(pages), [False])

    def test_clear_returns_false_when_remote_delete_fails_and_keeps_the_prefix(self):
        backend = mock.Mock()
        backend.remove_by_prefix.return_value = False
        l3 = L3HostStore(
            backend, _FakeHost(b"abcdefgh"), key_prefix="m", rank=1, cp_rank=0
        )
        old_key = l3.object_key("h0", 0, 0)

        self.assertFalse(l3.rotate_namespace())

        self.assertEqual(old_key, l3.object_key("h0", 0, 0))

    def test_clear_returns_false_when_remote_delete_raises(self):
        backend = mock.Mock()
        backend.remove_by_prefix.side_effect = RuntimeError("delete failed")
        l3 = L3HostStore(
            backend, _FakeHost(b"abcdefgh"), key_prefix="m", rank=1, cp_rank=0
        )
        old_key = l3.object_key("h0", 0, 0)

        self.assertFalse(l3.rotate_namespace())

        self.assertEqual(old_key, l3.object_key("h0", 0, 0))

    def test_set_key_prefix_republishes_under_the_new_namespace(self):
        backend = MemoryKvStore()
        host = _FakeHost(b"abcdefgh")
        l3 = L3HostStore(backend, host, key_prefix="v1", rank=0, cp_rank=0)
        pages = [(0, 1, "h0", 0)]
        self.assertEqual(l3.backup(pages), [True])
        old_key = l3.object_key("h0", 0, 0)
        l3.rotate_namespace()
        l3.set_key_prefix("v2")
        self.assertNotEqual(old_key, l3.object_key("h0", 0, 0))
        self.assertEqual(l3.exists(pages), [False])
        self.assertEqual(l3.backup(pages), [True])
        self.assertTrue(l3.object_key("h0", 0, 0).startswith("v2_"))
        self.assertEqual(backend.batch_exists([old_key]), [False])


class FactoryTest(unittest.TestCase):
    def test_memory_and_unknown_backend(self):
        backend = create_kvstore_storage_backend(
            "memory", None, host_buffer=object(), tp_size=1, cp_size=1, pp_size=1
        )
        self.assertIsInstance(backend, MemoryKvStore)
        self.assertIsNone(
            create_kvstore_storage_backend(
                None, None, host_buffer=object(), tp_size=1, cp_size=1, pp_size=1
            )
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            create_kvstore_storage_backend(
                "nfs", None, host_buffer=object(), tp_size=1, cp_size=1, pp_size=1
            )

    def test_cp_size_has_no_default(self):
        param = inspect.signature(create_kvstore_storage_backend).parameters["cp_size"]
        self.assertIs(param.default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            create_kvstore_storage_backend(
                "memory", None, host_buffer=object(), tp_size=1, pp_size=1
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

    def test_from_mapping_none_uses_mooncake_master_env(self):
        saved = {
            key: os.environ.pop(key)
            for key in ("MOONCAKE_MASTER", "MOONCAKE_CLIENT")
            if key in os.environ
        }
        os.environ["MOONCAKE_MASTER"] = "env-master:50051"
        try:
            config = MooncakeStoreConfig.from_mapping(None)
            self.assertEqual(config.master_server_address, "env-master:50051")
        finally:
            os.environ.pop("MOONCAKE_MASTER", None)
            os.environ.update(saved)

    def test_default_global_segment_matches_runtime_flag_default(self):
        saved = os.environ.pop("MOONCAKE_GLOBAL_SEGMENT_SIZE", None)
        try:
            config = MooncakeStoreConfig.from_mapping(
                {"master_server_address": "127.0.0.1:50051"}
            )
            self.assertEqual(config.global_segment_size, 4 * 1024**3)
            self.assertEqual(config.tenant_id, "default")
        finally:
            if saved is not None:
                os.environ["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = saved

    def test_tenant_id_has_no_constructor_default(self):
        signature = inspect.signature(MooncakeStoreConfig)
        self.assertIs(
            signature.parameters["tenant_id"].default, inspect.Parameter.empty
        )
        with self.assertRaises(TypeError):
            MooncakeStoreConfig(
                "localhost",
                "P2PHANDSHAKE",
                4 * 1024**3,
                "tcp",
                "",
                "127.0.0.1:50051",
                "",
            )


class MooncakeKvStoreTest(unittest.TestCase):
    def test_extra_config_has_no_default(self):
        param = inspect.signature(MooncakeKvStore.__init__).parameters["extra_config"]
        self.assertIs(param.default, inspect.Parameter.empty)
        cp_param = inspect.signature(MooncakeKvStore.__init__).parameters["cp_size"]
        self.assertIs(cp_param.default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            MooncakeKvStore(host_buffer=object(), tp_size=1, cp_size=1, pp_size=1)
        with self.assertRaises(TypeError):
            MooncakeKvStore(None, host_buffer=object(), tp_size=1, pp_size=1)

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
                    cp_size=1,
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

    def test_truncated_exists_does_not_ack_unuploaded_keys(self):
        class _Store:
            def batch_is_exist(self, keys):
                del keys
                return [0]

            def batch_put_from(self, keys, ptrs, sizes):
                del keys, ptrs, sizes
                raise AssertionError("truncated exists must not issue puts")

        adapter = object.__new__(MooncakeKvStore)
        adapter.store = _Store()
        host = SimpleNamespace(data_ptr=lambda: 100)
        self.assertEqual(
            adapter.batch_put_from(["k0", "k1"], host, [0, 8], [8, 8]),
            [False, False],
        )

    def test_truncated_exists_probe_is_rejected(self):
        adapter = object.__new__(MooncakeKvStore)
        adapter.store = SimpleNamespace(batch_is_exist=lambda keys: [1])
        with self.assertRaisesRegex(ValueError, "batch_is_exist returned 1"):
            adapter.batch_exists(["k0", "k1"])

    def test_truncated_put_results_do_not_ack_unuploaded_keys(self):
        class _Store:
            def batch_is_exist(self, keys):
                return [0] * len(keys)

            def batch_put_from(self, keys, ptrs, sizes):
                del ptrs, sizes
                return [0]

        adapter = object.__new__(MooncakeKvStore)
        adapter.store = _Store()
        host = SimpleNamespace(data_ptr=lambda: 100)
        self.assertEqual(
            adapter.batch_put_from(["k0", "k1"], host, [0, 8], [8, 8]),
            [False, False],
        )

    def test_truncated_get_into_is_rejected(self):
        adapter = object.__new__(MooncakeKvStore)
        adapter.store = SimpleNamespace(batch_get_into=lambda keys, ptrs, sizes: [1])
        host = SimpleNamespace(data_ptr=lambda: 100)
        with self.assertRaisesRegex(ValueError, "batch_get_into returned 1"):
            adapter.batch_get_into(["k0", "k1"], host, [0, 8], [8, 8])

    def test_namespace_clear_uses_anchored_escaped_regex(self):
        adapter = object.__new__(MooncakeKvStore)
        store = mock.Mock()
        adapter.store = store
        store.remove_by_regex.return_value = 0

        self.assertTrue(adapter.remove_by_prefix("model.v1_"))

        store.remove_by_regex.assert_called_once_with(r"^model\.v1_.*", True)

    def test_namespace_clear_without_remove_by_regex_is_failure(self):
        adapter = object.__new__(MooncakeKvStore)
        adapter.store = object()
        self.assertFalse(adapter.remove_by_prefix("model_"))

    def test_namespace_clear_failure_is_not_reported_as_success(self):
        adapter = object.__new__(MooncakeKvStore)
        adapter.store = mock.Mock()
        adapter.store.remove_by_regex.return_value = -1
        self.assertFalse(adapter.remove_by_prefix("model_"))

    def test_segment_is_divided_across_tp_cp_and_pp_ranks(self):
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
        extra = {
            "master_server_address": "127.0.0.1:50051",
            "global_segment_size": 8 * 1024**3,
        }
        with mock.patch.dict(
            sys.modules, {"mooncake": package, "mooncake.store": store_module}
        ):
            MooncakeKvStore(
                extra,
                host_buffer=host,
                tp_size=2,
                cp_size=2,
                pp_size=2,
            )
        self.assertEqual(captured["segment"], 1 * 1024**3)
        with mock.patch.dict(
            sys.modules, {"mooncake": package, "mooncake.store": store_module}
        ):
            MooncakeKvStore(
                extra,
                host_buffer=host,
                tp_size=1,
                cp_size=4,
                pp_size=1,
            )
        self.assertEqual(captured["segment"], 2 * 1024**3)


if __name__ == "__main__":
    unittest.main()
