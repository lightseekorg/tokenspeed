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

import json
import os
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest

from tokenspeed.runtime.cache.l3.config import (
    MooncakeStoreConfig,
    MooncakeStoreSettings,
    StorageRuntimeConfig,
    parse_extra_config,
)
from tokenspeed.runtime.cache.l3.identity import (
    L3_RUNTIME_COMPAT,
    _ext_def_file_from_yaml_text,
    cache_layout_signature,
    l3_cache_quantization_id,
    l3_checkpoint_id,
    resolve_l3_weight_version,
    share_l3_checkpoint_ids,
    storage_deployment_id,
)


@contextmanager
def _chdir(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _deployment_facts():
    return {
        "model_name": "org/model",
        "revision": "abc",
        "model_overrides": {},
        "cache_signature": "layout",
        "pipeline_rank": 0,
        "attn_tp_size": 1,
        "group_shard_counts": (1,),
        "draft_model": "",
        "draft_revision": "",
        "cache_quantization": "",
        "runtime_compat": L3_RUNTIME_COMPAT,
        "attention_backend": "mha",
        "draft_attention_backend": "",
        "skip_softmax_threshold": 0.0,
        "eagle3_layers_to_capture": [],
        "kv_producer": {"numerics": "resolved", "attn_kvp_size": 1},
    }


class TestStorageKeyTest:
    def test_cp_group_layouts_do_not_share_replica_keys(self):
        facts = _deployment_facts()
        facts.update(
            attn_tp_size=4,
            kv_producer={
                "attn_kvp_size": 4,
                "attn_qcp_size": 2,
                "attn_head_tp_size": 1,
            },
        )
        identities = set()
        for shards in ((1, 1), (2, 1), (4, 1), (1, 4), (4, 4)):
            facts["group_shard_counts"] = shards
            identities.add(storage_deployment_id(**facts))
        assert len(identities) == 5
        assert storage_deployment_id(**facts) == storage_deployment_id(
            **{**facts, "group_shard_counts": list(facts["group_shard_counts"])}
        )

    @pytest.mark.parametrize("shards", ["41", b"41"])
    def test_cp_group_layout_rejects_text(self, shards):
        facts = {**_deployment_facts(), "group_shard_counts": shards}
        with pytest.raises(TypeError, match="group_shard_counts"):
            storage_deployment_id(**facts)

    @pytest.mark.parametrize("shards", [(), (0,), (2, -1)])
    def test_cp_group_layout_requires_positive_counts(self, shards):
        facts = {**_deployment_facts(), "group_shard_counts": shards}
        with pytest.raises(ValueError, match="group_shard_counts"):
            storage_deployment_id(**facts)

    def test_identity_is_stable_and_separates_incompatible_cache_objects(self):
        # The weight versions join the identity in the engine host's
        # namespace (``ts-l3``'s ``key_prefix``).
        def prefix(**overrides):
            return storage_deployment_id(**{**_deployment_facts(), **overrides})

        base = prefix()
        assert base == prefix()
        assert base != prefix(kv_producer={"numerics": "different"})
        assert base != prefix(model_name="org_model")
        assert base != prefix(revision="def")
        assert base != prefix(cache_signature="other")
        assert base != prefix(pipeline_rank=1)
        assert base != prefix(attn_tp_size=8)
        assert prefix(attn_tp_size=8) != prefix(attn_tp_size=16)
        assert base != prefix(draft_model="org/draft")
        assert base != prefix(cache_quantization='{"quantization":"fp8"}')
        assert base != prefix(model_overrides={"rope_theta": 10000.0})
        assert prefix(model_overrides={"rope_theta": 10000.0}) != prefix(
            model_overrides={"rope_scaling": {"type": "linear"}}
        )
        assert prefix(model_overrides={"b": 2, "a": 1}) == prefix(
            model_overrides={"a": 1, "b": 2}
        )
        assert (prefix(model_overrides={"b": 2, "a": 1})) == (
            prefix(model_overrides={"a": 1, "b": 2})
        )
        assert (base) != (prefix(attention_backend="fa3"))
        assert (prefix(attention_backend="fa3")) != (prefix(attention_backend="fa4"))
        assert (base) != (prefix(draft_attention_backend="mha"))
        assert (prefix(attention_backend="msa:trtllm")) != (
            prefix(attention_backend="msa:fa3")
        )
        assert (prefix(draft_attention_backend="msa:trtllm")) != (
            prefix(draft_attention_backend="msa:fa3")
        )
        assert (prefix(draft_attention_backend="fa3")) != (
            prefix(draft_attention_backend="fa4")
        )
        assert base != prefix(runtime_compat="previous-epoch")
        assert base == prefix(runtime_compat="2")
        assert base != prefix(runtime_compat="1")
        assert base != prefix(runtime_compat="3")
        assert base != prefix(runtime_compat="4")
        assert prefix(runtime_compat="1") == prefix(runtime_compat="1")
        assert base != prefix(skip_softmax_threshold=0.01)
        assert prefix(skip_softmax_threshold=0.0) == prefix(skip_softmax_threshold=0.0)
        assert (base) != (prefix(eagle3_layers_to_capture=[2, 16, 29]))
        assert (prefix(eagle3_layers_to_capture=[2, 16, 29])) != (
            prefix(eagle3_layers_to_capture=[1, 16, 29])
        )
        assert (prefix(eagle3_layers_to_capture=[2, 16, 29])) == (
            prefix(eagle3_layers_to_capture=[2, 16, 29])
        )
        with pytest.raises((TypeError)):
            prefix(eagle3_layers_to_capture="2,16,29")
        with pytest.raises((TypeError)):
            prefix(model_overrides=["rope_theta"])
        with pytest.raises(ValueError):
            prefix(kv_producer={})

    def test_identity_follows_its_mutable_inputs(self):
        facts = _deployment_facts()
        deployment = storage_deployment_id(**facts)
        facts["model_overrides"]["rope_theta"] = 10000.0
        facts["kv_producer"]["numerics"] = "different"
        facts["eagle3_layers_to_capture"].append(4)
        assert storage_deployment_id(**facts) != deployment

    def test_structured_producer_facts_have_canonical_order(self):
        facts = _deployment_facts()
        forward = storage_deployment_id(**facts)
        facts["kv_producer"] = {"attn_kvp_size": 1, "numerics": "resolved"}
        assert storage_deployment_id(**facts) == forward
        facts["kv_producer"]["attn_kvp_size"] = 2
        assert storage_deployment_id(**facts) != forward

    def _checkpoint_id(self, model_path, *, load_format, hf_config, revision):
        return l3_checkpoint_id(
            model_path,
            hf_config=hf_config,
            revision=revision,
            load_format=load_format,
            model_loader_extra_config={},
            ext_yaml="",
        )

    def test_checkpoint_id_prefers_loaded_commit_over_moving_branch(self):
        commit = "a" * 40
        other = "b" * 40
        assert (
            self._checkpoint_id(
                "org/model",
                hf_config=SimpleNamespace(_commit_hash=commit),
                revision="main",
                load_format="auto",
            )
        ) == (f"{commit}:auto")
        assert (
            self._checkpoint_id(
                "org/model",
                hf_config=SimpleNamespace(_commit_hash=commit),
                revision="main",
                load_format="auto",
            )
        ) != (
            self._checkpoint_id(
                "org/model",
                hf_config=SimpleNamespace(_commit_hash=other),
                revision="main",
                load_format="auto",
            )
        )
        assert (
            self._checkpoint_id(
                "org/model",
                hf_config=SimpleNamespace(_commit_hash=commit),
                revision="main",
                load_format="auto",
            )
        ) != (
            self._checkpoint_id(
                "org/model",
                hf_config=SimpleNamespace(_commit_hash=commit),
                revision="main",
                load_format="pt",
            )
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
            assert (
                self._checkpoint_id(
                    snapshot,
                    hf_config=SimpleNamespace(),
                    revision="main",
                    load_format="safetensors",
                )
            ) == (f"{commit}:safetensors")

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
            assert checkpoint_id.startswith("local-")
            assert (checkpoint_id) != (f"{commit}:auto")

    def test_checkpoint_id_does_not_trust_snapshots_dir_outside_hf_hub_cache(self):
        commit = "e" * 40
        with (
            tempfile.TemporaryDirectory() as first_root,
            tempfile.TemporaryDirectory() as second_root,
        ):
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
            assert first_id.startswith("local-")
            assert second_id.startswith("local-")
            assert (first_id) != (second_id)
            assert (first_id) != (f"{commit}:auto")

    def test_checkpoint_id_fingerprints_local_weight_bytes(self):
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            for directory, payload in ((first, b"aaa"), (second, b"bbb")):
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write('{"model_type":"x"}')
                with open(os.path.join(directory, "model.safetensors"), "wb") as handle:
                    handle.write(payload)
            assert (os.path.getsize(os.path.join(first, "model.safetensors"))) == (3)
            assert (os.path.getsize(os.path.join(second, "model.safetensors"))) == (3)
            assert (
                self._checkpoint_id(
                    first,
                    load_format="auto",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            ) != (
                self._checkpoint_id(
                    second,
                    load_format="auto",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            )
            assert self._checkpoint_id(
                first,
                load_format="auto",
                hf_config=SimpleNamespace(),
                revision="",
            ).startswith("local-")

    def test_checkpoint_id_does_not_trust_hex_basename_outside_hf_snapshots(self):
        commit = "d" * 40
        with (
            tempfile.TemporaryDirectory() as first_root,
            tempfile.TemporaryDirectory() as second_root,
        ):
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
            assert first_id.startswith("local-")
            assert second_id.startswith("local-")
            assert (first_id) != (second_id)
            assert (first_id) != (f"{commit}:auto")

    def test_checkpoint_id_ignores_inherited_commit_on_local_dir(self):
        inherited = "a" * 40
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
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
            assert (first_id) != (second_id)
            assert not (first_id.startswith(inherited))
            assert first_id.startswith("local-")

    def test_checkpoint_id_fingerprints_local_hf_quant_config(self):
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
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
            assert (
                self._checkpoint_id(
                    first,
                    load_format="auto",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            ) != (
                self._checkpoint_id(
                    second,
                    load_format="auto",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            )

    def test_checkpoint_id_fingerprints_local_config_code(self):
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
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
            assert (
                self._checkpoint_id(
                    first,
                    load_format="auto",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            ) != (
                self._checkpoint_id(
                    second,
                    load_format="auto",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            )

    def test_checkpoint_id_fingerprints_imported_custom_code_subdirectories(self):
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
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
            assert first_id.startswith("local-")
            assert second_id.startswith("local-")
            assert (first_id) != (second_id)

    def test_checkpoint_id_fingerprints_symlinked_custom_code_packages(self):
        with tempfile.TemporaryDirectory() as root:
            first_pkg = os.path.join(root, "pkg_a")
            second_pkg = os.path.join(root, "pkg_b")
            first_ckpt = os.path.join(root, "ckpt_a")
            second_ckpt = os.path.join(root, "ckpt_b")
            os.makedirs(first_pkg)
            os.makedirs(second_pkg)
            os.makedirs(first_ckpt)
            os.makedirs(second_ckpt)
            with open(os.path.join(first_pkg, "__init__.py"), "w") as handle:
                handle.write("")
            with open(os.path.join(first_pkg, "attention.py"), "w") as handle:
                handle.write("def rotary_dim():\n    return 64\n")
            with open(os.path.join(second_pkg, "__init__.py"), "w") as handle:
                handle.write("")
            with open(os.path.join(second_pkg, "attention.py"), "w") as handle:
                handle.write("def rotary_dim():\n    return 128\n")
            for directory, package in (
                (first_ckpt, first_pkg),
                (second_ckpt, second_pkg),
            ):
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write(
                        '{"model_type":"x","auto_map":{"AutoConfig":"configuration.Config"}}'
                    )
                with open(os.path.join(directory, "model.safetensors"), "wb") as handle:
                    handle.write(b"weights")
                with open(os.path.join(directory, "configuration.py"), "w") as handle:
                    handle.write("from model_helpers.attention import rotary_dim\n")
                os.symlink(package, os.path.join(directory, "model_helpers"))
            first_id = self._checkpoint_id(
                first_ckpt,
                load_format="auto",
                hf_config=SimpleNamespace(),
                revision="",
            )
            second_id = self._checkpoint_id(
                second_ckpt,
                load_format="auto",
                hf_config=SimpleNamespace(),
                revision="",
            )
            assert first_id.startswith("local-")
            assert second_id.startswith("local-")
            assert (first_id) != (second_id)
            cycle = os.path.join(root, "cycle")
            os.makedirs(cycle)
            with open(os.path.join(cycle, "config.json"), "w") as handle:
                handle.write("{}")
            os.symlink(cycle, os.path.join(cycle, "loop"))
            assert self._checkpoint_id(
                cycle,
                load_format="auto",
                hf_config=SimpleNamespace(),
                revision="",
            ).startswith("local-")

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
            assert (safetensors_id) != (bin_id)
            assert (safetensors_id) != (pt_id)
            assert (bin_id) != (pt_id)
            assert (auto_id) != (safetensors_id)
            assert safetensors_id.endswith(":safetensors")
            assert auto_id.endswith(":auto")

    def test_checkpoint_id_ignores_unselected_weight_files(self):
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            for directory, leftover in ((first, b"aaa"), (second, b"bbb")):
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write("{}")
                with open(os.path.join(directory, "model.safetensors"), "wb") as handle:
                    handle.write(b"same-safe")
                with open(os.path.join(directory, "pytorch_model.bin"), "wb") as handle:
                    handle.write(leftover)
            assert (
                self._checkpoint_id(
                    first,
                    load_format="safetensors",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            ) == (
                self._checkpoint_id(
                    second,
                    load_format="safetensors",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            )
            assert (
                self._checkpoint_id(
                    first,
                    load_format="npcache",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            ) != (
                self._checkpoint_id(
                    second,
                    load_format="npcache",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            )

    def test_checkpoint_id_fingerprints_sharded_state_weight_files(self):
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            for directory, payload in ((first, b"rank0-a"), (second, b"rank0-b")):
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write("{}")
                with open(
                    os.path.join(directory, "model-rank-0-part-0.safetensors"),
                    "wb",
                ) as handle:
                    handle.write(payload)
            assert (
                self._checkpoint_id(
                    first,
                    load_format="sharded_state",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            ) != (
                self._checkpoint_id(
                    second,
                    load_format="sharded_state",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            )

    def test_checkpoint_id_fingerprints_sharded_state_pattern_files(self):
        """Custom shard patterns must hash the files the loader would open."""

        from tokenspeed.runtime.cache.l3 import identity as l3_identity

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
                ext_yaml="",
            )
            extra = {"pattern": "alt-rank-{rank}-part-{part}.bin"}
            alt_id = l3_checkpoint_id(
                directory,
                hf_config=SimpleNamespace(),
                revision="",
                load_format="sharded_state",
                model_loader_extra_config=extra,
                ext_yaml="",
            )
            assert (default_id) != (alt_id)
            with open(default_shard, "wb") as handle:
                handle.write(b"default-shard-changed")
            l3_identity._local_checkpoint_fingerprint.cache_clear()
            assert (alt_id) == (
                l3_checkpoint_id(
                    directory,
                    hf_config=SimpleNamespace(),
                    revision="",
                    load_format="sharded_state",
                    model_loader_extra_config=extra,
                    ext_yaml="",
                )
            )
            with open(alt_shard, "wb") as handle:
                handle.write(b"alt-shard-b")
            l3_identity._local_checkpoint_fingerprint.cache_clear()
            assert (alt_id) != (
                l3_checkpoint_id(
                    directory,
                    hf_config=SimpleNamespace(),
                    revision="",
                    load_format="sharded_state",
                    model_loader_extra_config=extra,
                    ext_yaml="",
                )
            )

    def test_checkpoint_id_fingerprints_npcache_numpy_files(self):
        """When np/ exists, npcache loads it and must not hash only *.bin."""

        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
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
            assert (
                self._checkpoint_id(
                    first,
                    load_format="npcache",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            ) != (
                self._checkpoint_id(
                    second,
                    load_format="npcache",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            )

    def test_checkpoint_id_fingerprints_mistral_shard_index(self):
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
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
            assert (
                self._checkpoint_id(
                    first,
                    load_format="mistral",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            ) != (
                self._checkpoint_id(
                    second,
                    load_format="mistral",
                    hf_config=SimpleNamespace(),
                    revision="",
                )
            )

    def test_checkpoint_id_rejects_unsupported_load_format(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "config.json"), "w") as handle:
                handle.write("{}")
            with pytest.raises((ValueError)):
                self._checkpoint_id(
                    directory,
                    load_format="unknown-loader",
                    hf_config=SimpleNamespace(),
                    revision="",
                )

    def test_ext_def_file_parses_loader_yaml_forms(self):
        assert (_ext_def_file_from_yaml_text("ext_def_file: ext.py\n")) == ("ext.py")
        assert (_ext_def_file_from_yaml_text("ext_def_file : ext.py\n")) == ("ext.py")
        assert (_ext_def_file_from_yaml_text('"ext_def_file": ext.py\n')) == ("ext.py")
        assert (_ext_def_file_from_yaml_text("'ext_def_file': 'ext.py'\n")) == (
            "ext.py"
        )
        assert (
            _ext_def_file_from_yaml_text("{ext_def_file: ext.py, context: {}}\n")
        ) == ("ext.py")
        assert (_ext_def_file_from_yaml_text('{"ext_def_file": "ext.py"}\n')) == (
            "ext.py"
        )
        assert (_ext_def_file_from_yaml_text("---\n{ext_def_file: ext.py}\n")) == (
            "ext.py"
        )
        assert (
            _ext_def_file_from_yaml_text("context:\n  ext_def_file: nested.py\n")
        ) is None
        assert (
            _ext_def_file_from_yaml_text("{context: {ext_def_file: nested.py}}\n")
        ) is None

    def test_checkpoint_id_hashes_extensible_yaml_and_extension_code(self):
        commit = "d" * 40
        with tempfile.TemporaryDirectory() as root:
            repo = os.path.join(root, "hub", "models--org--model")
            snapshot = os.path.join(repo, "snapshots", commit)
            os.makedirs(os.path.join(repo, "refs"))
            os.makedirs(snapshot)
            with open(os.path.join(snapshot, "config.json"), "w") as handle:
                handle.write("{}")
            with open(os.path.join(snapshot, "model.safetensors"), "wb") as handle:
                handle.write(b"same-weights")

            def write_yaml(name: str, ext_def_name: str) -> str:
                yaml_path = os.path.join(root, name)
                ext_path = os.path.join(root, ext_def_name)
                with open(yaml_path, "w", encoding="utf-8") as handle:
                    handle.write(f"ext_def_file: {ext_path}\n")
                return yaml_path

            def write_ext_def(name: str, body: str) -> None:
                with open(os.path.join(root, name), "w", encoding="utf-8") as handle:
                    handle.write(body)

            write_ext_def("a.py", "PROCESSOR = 'a'\n")
            write_ext_def("b.py", "PROCESSOR = 'b'\n")
            yaml_a = write_yaml("a.yaml", "a.py")
            yaml_b = write_yaml("b.yaml", "b.py")
            yaml_a_copy = write_yaml("a-copy.yaml", "a.py")

            def extensible_id(ext_yaml: str) -> str:
                return l3_checkpoint_id(
                    snapshot,
                    hf_config=SimpleNamespace(),
                    revision="main",
                    load_format="extensible",
                    model_loader_extra_config={},
                    ext_yaml=ext_yaml,
                )

            id_a = extensible_id(yaml_a)
            id_b = extensible_id(yaml_b)
            assert id_a.startswith(f"{commit}:extensible:ext-")
            assert (id_a) != (f"{commit}:extensible")
            assert (id_a) != (id_b)
            assert (id_a) == (extensible_id(yaml_a_copy))
            quoted_yaml = os.path.join(root, "quoted.yaml")
            with open(quoted_yaml, "w", encoding="utf-8") as handle:
                handle.write(f'ext_def_file: "{os.path.join(root, "a.py")}"\n')
            quoted_id = extensible_id(quoted_yaml)
            assert (quoted_id) != (id_a)
            write_ext_def("a.py", "PROCESSOR = 'a-changed'\n")
            assert (id_a) != (extensible_id(yaml_a))
            assert (quoted_id) != (extensible_id(quoted_yaml))
            with pytest.raises((ValueError)):
                extensible_id("")

    def test_checkpoint_id_hashes_extension_code_for_quoted_key_and_flow_yaml(self):
        commit = "c" * 40
        with tempfile.TemporaryDirectory() as root:
            repo = os.path.join(root, "hub", "models--org--model")
            snapshot = os.path.join(repo, "snapshots", commit)
            os.makedirs(os.path.join(repo, "refs"))
            os.makedirs(snapshot)
            with open(os.path.join(snapshot, "config.json"), "w") as handle:
                handle.write("{}")
            with open(os.path.join(snapshot, "model.safetensors"), "wb") as handle:
                handle.write(b"same-weights")
            ext_path = os.path.join(root, "ext.py")
            with open(ext_path, "w", encoding="utf-8") as handle:
                handle.write("PROCESSOR = 'v1'\n")
            quoted_key = os.path.join(root, "quoted-key.yaml")
            spaced = os.path.join(root, "spaced.yaml")
            flow = os.path.join(root, "flow.yaml")
            with open(quoted_key, "w", encoding="utf-8") as handle:
                handle.write(f'"ext_def_file": {ext_path}\n')
            with open(spaced, "w", encoding="utf-8") as handle:
                handle.write(f"ext_def_file : {ext_path}\n")
            with open(flow, "w", encoding="utf-8") as handle:
                handle.write(f"{{ext_def_file: {ext_path}, context: {{}}}}\n")

            def extensible_id(ext_yaml: str) -> str:
                return l3_checkpoint_id(
                    snapshot,
                    hf_config=SimpleNamespace(),
                    revision="main",
                    load_format="extensible",
                    model_loader_extra_config={},
                    ext_yaml=ext_yaml,
                )

            quoted_id = extensible_id(quoted_key)
            spaced_id = extensible_id(spaced)
            flow_id = extensible_id(flow)
            assert quoted_id.startswith(f"{commit}:extensible:ext-")
            assert spaced_id.startswith(f"{commit}:extensible:ext-")
            assert flow_id.startswith(f"{commit}:extensible:ext-")
            with open(ext_path, "w", encoding="utf-8") as handle:
                handle.write("PROCESSOR = 'v2'\n")
            assert (quoted_id) != (extensible_id(quoted_key))
            assert (spaced_id) != (extensible_id(spaced))
            assert (flow_id) != (extensible_id(flow))

    def test_checkpoint_id_fingerprints_extensible_local_weights_and_yaml(self):
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            yaml_path = os.path.join(first, "ext.yaml")
            ext_def = os.path.join(first, "ext.py")
            with open(ext_def, "w", encoding="utf-8") as handle:
                handle.write("PROCESSOR = 'shared'\n")
            with open(yaml_path, "w", encoding="utf-8") as handle:
                handle.write(f"ext_def_file: {ext_def}\n")
            for directory, payload in ((first, b"weights-a"), (second, b"weights-b")):
                with open(os.path.join(directory, "config.json"), "w") as handle:
                    handle.write("{}")
                with open(os.path.join(directory, "model.safetensors"), "wb") as handle:
                    handle.write(payload)
            first_id = l3_checkpoint_id(
                first,
                hf_config=SimpleNamespace(),
                revision="",
                load_format="extensible",
                model_loader_extra_config={},
                ext_yaml=yaml_path,
            )
            second_id = l3_checkpoint_id(
                second,
                hf_config=SimpleNamespace(),
                revision="",
                load_format="extensible",
                model_loader_extra_config={},
                ext_yaml=yaml_path,
            )
            assert first_id.startswith("local-")
            assert (first_id) != (second_id)
            other_yaml = os.path.join(second, "other.yaml")
            with open(other_yaml, "w", encoding="utf-8") as handle:
                handle.write(f"ext_def_file: {ext_def}\ncontext: {{}}\n")
            other_id = l3_checkpoint_id(
                first,
                hf_config=SimpleNamespace(),
                revision="",
                load_format="extensible",
                model_loader_extra_config={},
                ext_yaml=other_yaml,
            )
            assert (first_id) != (other_id)

    def test_checkpoint_id_resolves_ext_def_file_from_cwd_like_loader(self):
        commit = "e" * 40
        with (
            tempfile.TemporaryDirectory() as yaml_dir,
            tempfile.TemporaryDirectory() as cwd_dir,
        ):
            repo = os.path.join(yaml_dir, "hub", "models--org--model")
            snapshot = os.path.join(repo, "snapshots", commit)
            os.makedirs(os.path.join(repo, "refs"))
            os.makedirs(snapshot)
            with open(os.path.join(snapshot, "config.json"), "w") as handle:
                handle.write("{}")
            with open(os.path.join(snapshot, "model.safetensors"), "wb") as handle:
                handle.write(b"same-weights")
            yaml_path = os.path.join(yaml_dir, "ext.yaml")
            yaml_ext = os.path.join(yaml_dir, "ext.py")
            cwd_ext = os.path.join(cwd_dir, "ext.py")
            with open(yaml_path, "w", encoding="utf-8") as handle:
                handle.write("ext_def_file: ext.py\n")
            with open(yaml_ext, "w", encoding="utf-8") as handle:
                handle.write("PROCESSOR = 'yaml-dir'\n")
            with open(cwd_ext, "w", encoding="utf-8") as handle:
                handle.write("PROCESSOR = 'cwd'\n")

            def extensible_id() -> str:
                return l3_checkpoint_id(
                    snapshot,
                    hf_config=SimpleNamespace(),
                    revision="main",
                    load_format="extensible",
                    model_loader_extra_config={},
                    ext_yaml=yaml_path,
                )

            with _chdir(cwd_dir):
                id_cwd = extensible_id()
                with open(yaml_ext, "w", encoding="utf-8") as handle:
                    handle.write("PROCESSOR = 'yaml-dir-changed'\n")
                assert (id_cwd) == (extensible_id())
                with open(cwd_ext, "w", encoding="utf-8") as handle:
                    handle.write("PROCESSOR = 'cwd-changed'\n")
                assert (id_cwd) != (extensible_id())

    def test_checkpoint_id_hashes_imported_extension_helpers(self):
        commit = "f" * 40
        with tempfile.TemporaryDirectory() as cwd_dir:
            repo = os.path.join(cwd_dir, "hub", "models--org--model")
            snapshot = os.path.join(repo, "snapshots", commit)
            os.makedirs(os.path.join(repo, "refs"))
            os.makedirs(snapshot)
            with open(os.path.join(snapshot, "config.json"), "w") as handle:
                handle.write("{}")
            with open(os.path.join(snapshot, "model.safetensors"), "wb") as handle:
                handle.write(b"same-weights")
            pkg = os.path.join(cwd_dir, "pkg")
            os.makedirs(pkg)
            with open(os.path.join(cwd_dir, "ext.py"), "w", encoding="utf-8") as handle:
                handle.write("import helper\nfrom pkg import inner\n")
            with open(
                os.path.join(cwd_dir, "helper.py"), "w", encoding="utf-8"
            ) as handle:
                handle.write("VALUE = 1\n")
            with open(
                os.path.join(pkg, "__init__.py"), "w", encoding="utf-8"
            ) as handle:
                handle.write("")
            with open(os.path.join(pkg, "inner.py"), "w", encoding="utf-8") as handle:
                handle.write("VALUE = 1\n")
            with open(
                os.path.join(cwd_dir, "unused.py"), "w", encoding="utf-8"
            ) as handle:
                handle.write("VALUE = 1\n")
            yaml_path = os.path.join(cwd_dir, "ext.yaml")
            with open(yaml_path, "w", encoding="utf-8") as handle:
                handle.write("ext_def_file: ext.py\n")

            def extensible_id() -> str:
                return l3_checkpoint_id(
                    snapshot,
                    hf_config=SimpleNamespace(),
                    revision="main",
                    load_format="extensible",
                    model_loader_extra_config={},
                    ext_yaml=yaml_path,
                )

            with _chdir(cwd_dir):
                id_base = extensible_id()
                with open(
                    os.path.join(cwd_dir, "unused.py"), "w", encoding="utf-8"
                ) as handle:
                    handle.write("VALUE = 2\n")
                assert (id_base) == (extensible_id())
                with open(
                    os.path.join(cwd_dir, "helper.py"), "w", encoding="utf-8"
                ) as handle:
                    handle.write("VALUE = 2\n")
                id_helper = extensible_id()
                assert (id_base) != (id_helper)
                with open(
                    os.path.join(pkg, "inner.py"), "w", encoding="utf-8"
                ) as handle:
                    handle.write("VALUE = 2\n")
                assert (id_helper) != (extensible_id())

    def test_checkpoint_id_hashes_package_relative_extension_imports(self):
        commit = "1" * 40
        with tempfile.TemporaryDirectory() as cwd_dir:
            repo = os.path.join(cwd_dir, "hub", "models--org--model")
            snapshot = os.path.join(repo, "snapshots", commit)
            os.makedirs(os.path.join(repo, "refs"))
            os.makedirs(snapshot)
            with open(os.path.join(snapshot, "config.json"), "w") as handle:
                handle.write("{}")
            with open(os.path.join(snapshot, "model.safetensors"), "wb") as handle:
                handle.write(b"same-weights")
            pkg = os.path.join(cwd_dir, "pkg")
            os.makedirs(pkg)
            with open(os.path.join(cwd_dir, "ext.py"), "w", encoding="utf-8") as handle:
                handle.write("import pkg\n")
            with open(
                os.path.join(pkg, "__init__.py"), "w", encoding="utf-8"
            ) as handle:
                handle.write("from . import helper\n")
            with open(os.path.join(pkg, "helper.py"), "w", encoding="utf-8") as handle:
                handle.write("VALUE = 1\n")
            with open(
                os.path.join(cwd_dir, "helper.py"), "w", encoding="utf-8"
            ) as handle:
                handle.write("VALUE = 'parent'\n")
            yaml_path = os.path.join(cwd_dir, "ext.yaml")
            with open(yaml_path, "w", encoding="utf-8") as handle:
                handle.write("ext_def_file: ext.py\n")

            def extensible_id() -> str:
                return l3_checkpoint_id(
                    snapshot,
                    hf_config=SimpleNamespace(),
                    revision="main",
                    load_format="extensible",
                    model_loader_extra_config={},
                    ext_yaml=yaml_path,
                )

            with _chdir(cwd_dir):
                id_base = extensible_id()
                with open(
                    os.path.join(cwd_dir, "helper.py"), "w", encoding="utf-8"
                ) as handle:
                    handle.write("VALUE = 'parent-changed'\n")
                assert (id_base) == (extensible_id())
                with open(
                    os.path.join(pkg, "helper.py"), "w", encoding="utf-8"
                ) as handle:
                    handle.write("VALUE = 2\n")
                assert (id_base) != (extensible_id())

    def test_checkpoint_id_ignores_ext_yaml_unless_extensible(self):
        commit = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            yaml_path = os.path.join(directory, "ext.yaml")
            with open(yaml_path, "w", encoding="utf-8") as handle:
                handle.write("ext_def_file: missing.py\n")
            without_yaml = l3_checkpoint_id(
                "org/model",
                hf_config=SimpleNamespace(_commit_hash=commit),
                revision="main",
                load_format="auto",
                model_loader_extra_config={},
                ext_yaml="",
            )
            with_yaml = l3_checkpoint_id(
                "org/model",
                hf_config=SimpleNamespace(_commit_hash=commit),
                revision="main",
                load_format="auto",
                model_loader_extra_config={},
                ext_yaml=yaml_path,
            )
            assert (without_yaml) == (with_yaml)
            assert (without_yaml) == (f"{commit}:auto")

    def test_checkpoint_id_rejects_blank_load_format(self):
        with pytest.raises((ValueError)):
            self._checkpoint_id(
                "org/model",
                hf_config=SimpleNamespace(_commit_hash="a" * 40),
                revision="main",
                load_format="  ",
            )

    def test_local_fingerprint_is_cached_per_directory(self):
        from tokenspeed.runtime.cache.l3 import identity as l3_identity

        l3_identity._local_checkpoint_fingerprint.cache_clear()
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "config.json"), "w") as handle:
                handle.write("{}")
            with open(os.path.join(directory, "model.safetensors"), "wb") as handle:
                handle.write(b"weights")
            with mock.patch.object(
                l3_identity,
                "_update_file_digest",
                wraps=l3_identity._update_file_digest,
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
            assert (first) == (second)
            assert (digest.call_count) == (2)

    def test_share_l3_checkpoint_ids_combines_rank_local_digests(self):
        seen = []

        def gather_from(rows):
            def gather(payload):
                seen.append(list(payload))
                return rows

            return gather

        shared = ["local-aaa:sharded_state", ""]
        assert (
            share_l3_checkpoint_ids(
                shared,
                rank=0,
                world_size=1,
                gather=gather_from([shared]),
            )
        ) == (shared)
        assert (seen) == ([])
        assert (
            share_l3_checkpoint_ids(
                shared,
                rank=1,
                world_size=2,
                gather=gather_from([shared, shared]),
            )
        ) == (shared)
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
        assert combined_a[0].startswith("local-")
        assert (combined_a[0]) != (rank0[0])
        assert (combined_a) != (combined_b)
        assert (combined_a[1]) == ("")
        assert (combined_b[1]) == ("")
        with pytest.raises((ValueError)):
            share_l3_checkpoint_ids(
                rank0,
                rank=0,
                world_size=2,
                gather=gather_from([rank0]),
            )

    def test_checkpoint_id_rejects_unpinned_remote_without_commit(self) -> None:
        with pytest.raises((ValueError)):
            l3_checkpoint_id(
                "org/unpinned-model",
                hf_config=SimpleNamespace(),
                revision="main",
                load_format="auto",
                model_loader_extra_config={},
                ext_yaml="",
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
            assert (first_id) != (
                l3_cache_quantization_id(
                    quantization="fp8",
                    quantization_param_path=second,
                    draft_quantization="",
                )
            )
            assert (first_id) == (
                l3_cache_quantization_id(
                    quantization="fp8",
                    quantization_param_path=same_copy,
                    draft_quantization="",
                )
            )
            assert (first_id) != (
                l3_cache_quantization_id(
                    quantization="",
                    quantization_param_path="",
                    draft_quantization="",
                )
            )
            assert (first_id) != (
                l3_cache_quantization_id(
                    quantization="fp8",
                    quantization_param_path=first,
                    draft_quantization="fp8",
                )
            )

    def test_resolve_l3_weight_version_preserves_requested_identity(self):
        assert resolve_l3_weight_version(None) is None
        assert resolve_l3_weight_version("explicit") == "explicit"
        assert resolve_l3_weight_version(12) == "12"

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
        assert (fp16) == (cache_layout_signature(layout, cache_dtype="float16"))
        assert (fp16) != (cache_layout_signature(layout, cache_dtype="bfloat16"))
        changed = SimpleNamespace(
            groups=(
                SimpleNamespace(
                    group_id="full",
                    cache_blocks_per_lcm_block=1,
                    fields=(SimpleNamespace(**{**vars(field), "payload_bytes": 20}),),
                ),
            )
        )
        assert (fp16) != (cache_layout_signature(changed, cache_dtype="float16"))
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
        assert (fp16) != (cache_layout_signature(strided, cache_dtype="float16"))
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
        assert (fp16) == (cache_layout_signature(shifted, cache_dtype="float16"))


class TestMooncakeConfigTest:
    def test_from_mapping_and_size_suffixes(self):
        saved = {
            key: os.environ.pop(key)
            for key in ("MOONCAKE_MASTER", "MOONCAKE_CLIENT")
            if key in os.environ
        }
        try:
            with pytest.raises((ValueError), match=("master_server_address")):
                MooncakeStoreConfig.from_mapping({})
            config = MooncakeStoreConfig.from_mapping(
                {
                    "client_server_address": "127.0.0.1:50051",
                    "global_segment_size": "2gb",
                    "protocol": "rdma",
                    "tenant_id": "ts",
                }
            )
            assert (config.master_server_address) == ("127.0.0.1:50051")
            assert (config.global_segment_size) == (2 * 1024 * 1024 * 1024)
            assert (config.protocol) == ("rdma")
            assert (config.tenant_id) == ("ts")
            parsed = parse_extra_config('{"master_server_address": "host:1"}')
            assert (parsed["master_server_address"]) == ("host:1")
            with pytest.raises((ValueError), match=("JSON object")):
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
            assert (config.master_server_address) == ("env-master:50051")
        finally:
            os.environ.pop("MOONCAKE_MASTER", None)
            os.environ.update(saved)

    def test_default_global_segment_matches_runtime_flag_default(self):
        saved = os.environ.pop("MOONCAKE_GLOBAL_SEGMENT_SIZE", None)
        try:
            config = MooncakeStoreConfig.from_mapping(
                {"master_server_address": "127.0.0.1:50051"}
            )
            assert (config.global_segment_size) == (4 * 1024**3)
            assert (config.tenant_id) == ("default")
        finally:
            if saved is not None:
                os.environ["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = saved

    def test_segment_is_divided_across_tp_and_pp_ranks(self):
        def segment(size, *, tp_size, pp_size):
            return MooncakeStoreSettings.resolve(
                {
                    "master_server_address": "127.0.0.1:50051",
                    "global_segment_size": size,
                },
                library="libmooncake_store.so",
                tp_size=tp_size,
                pp_size=pp_size,
            ).global_segment_size

        assert segment(8 * 1024**3, tp_size=4, pp_size=2) == 1024**3
        assert segment(8 * 1024**3, tp_size=4, pp_size=1) == 2 * 1024**3
        # Zero mounts nothing: the client uses independently owned segments.
        assert segment(0, tp_size=8, pp_size=1) == 0
        for size in (-1, 1):
            with pytest.raises(ValueError, match="too small"):
                segment(size, tp_size=8, pp_size=1)


class TestStorageRuntimeConfigTest:
    @pytest.mark.parametrize(
        "name",
        [
            "max_operations",
            "max_keys_per_operation",
            "max_host_blocks",
            "metadata_entries",
            "max_prefetch_tokens",
            "min_prefetch_tokens",
        ],
    )
    def test_limits_reject_values_outside_the_scheduler_integer_range(self, name):
        with pytest.raises(ValueError, match="int32 limit"):
            StorageRuntimeConfig.from_extra_config(
                json.dumps({"async": {"metadata_entries": 128, name: 2**31}}),
                host_blocks=2**32,
                prefix_granularity=2,
            )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"queue_low_operations": 8},
            {"queue_high_bytes": 128 * 1024 * 1024},
            {"operation_low_age_ms": 500},
            {"queue_low_operations": -1},
        ],
    )
    def test_pressure_rejects_invalid_hysteresis(self, overrides):
        with pytest.raises(ValueError):
            StorageRuntimeConfig.from_extra_config(
                json.dumps({"async": overrides}), host_blocks=32, prefix_granularity=2
            )

    def test_one_credit_pressure_can_recover_on_an_empty_queue(self):
        config = StorageRuntimeConfig.from_extra_config(
            '{"async":{"max_operations":1}}', host_blocks=32, prefix_granularity=2
        )
        assert config.scheduler.pressure_high.queued_operations == 1
        assert config.scheduler.pressure_low.queued_operations == 0
