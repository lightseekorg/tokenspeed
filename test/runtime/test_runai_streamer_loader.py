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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
import hashlib
import json
import os
import stat
import struct
import sys
import tempfile
import types
from importlib.util import find_spec
from pathlib import Path

import pytest
import safetensors.torch
import torch
from safetensors import safe_open

from tokenspeed.runtime.cache.l3.identity import l3_checkpoint_id
from tokenspeed.runtime.configs.load_config import LoadConfig, LoadFormat
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.model_loader.loader import (
    DefaultModelLoader,
    RunaiModelStreamerLoader,
    get_model_loader,
)
from tokenspeed.runtime.model_loader.object_store import (
    RemoteSafetensorsFile,
    is_object_store_uri,
    join_object_store_uri,
    parse_object_store_uri,
    resolve_metadata_cache_dir,
    sync_object_store_metadata,
)
from tokenspeed.runtime.model_loader.weight_utils import (
    filter_safetensors_files_by_weight_names,
    open_safetensors_slice,
    runai_streamer_weights_iterator,
)
from tokenspeed.runtime.utils.server_args import ServerArgs

RUNAI_AVAILABLE = find_spec("runai_model_streamer") is not None


class TestRunaiStreamerConfig:
    """CLI, ServerArgs, LoadConfig, and extra-config validation."""

    def test_cli_flags_wire_load_format_and_extra_config(self) -> None:
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        extra = json.dumps(
            {"concurrency": 8, "memory_limit": 4 * 1024**3, "distributed": True}
        )
        args = parser.parse_args(
            [
                "--model",
                "gs://example-bucket/model",
                "--load-format",
                "runai_streamer",
                "--model-loader-extra-config",
                extra,
            ]
        )
        assert args.load_format == "runai_streamer"
        load_config = LoadConfig(
            load_format=args.load_format,
            model_loader_extra_config=args.model_loader_extra_config,
        )
        assert load_config.load_format == LoadFormat.RUNAI_STREAMER
        loader = get_model_loader(load_config)
        assert isinstance(loader, RunaiModelStreamerLoader)
        assert loader.concurrency == 8
        assert loader.memory_limit == 4 * 1024**3
        assert loader.distributed is True

    def test_server_args_parses_extra_config_once(self) -> None:
        args = object.__new__(ServerArgs)
        args.model_loader_extra_config = '{"concurrency": 4, "distributed": true}'
        args._parse_model_loader_extra_config()
        assert args.model_loader_extra_config == {"concurrency": 4, "distributed": True}

        args.model_loader_extra_config = None
        args._parse_model_loader_extra_config()
        assert args.model_loader_extra_config == {}

        for bad in ("{not-json", "[1, 2]", "42", 123):
            args.model_loader_extra_config = bad
            with pytest.raises(ValueError):
                args._parse_model_loader_extra_config()

    @pytest.mark.parametrize(
        "bad_config",
        [
            {"unknown_option": 1},
            {"concurrency": 0},
            {"concurrency": True},
            {"memory_limit": -1},
            {"memory_limit": False},
            {"distributed": "yes"},
        ],
    )
    def test_invalid_extra_config_is_rejected(self, bad_config: dict) -> None:
        load_config = LoadConfig(
            load_format="runai_streamer", model_loader_extra_config=bad_config
        )
        with pytest.raises(ValueError):
            RunaiModelStreamerLoader(load_config)

    def test_distributed_with_checkpoint_load_group_raises(
        self, tmp_path: Path
    ) -> None:
        safetensors.torch.save_file(
            {"w": torch.zeros((2, 2), dtype=torch.bfloat16)},
            str(tmp_path / "model.safetensors"),
        )
        loader = RunaiModelStreamerLoader(
            LoadConfig(
                load_format="runai_streamer",
                model_loader_extra_config={"distributed": True},
            )
        )
        source = DefaultModelLoader.Source(
            model_or_path=str(tmp_path),
            revision=None,
            weight_files=None,
            prefix="",
            fall_back_to_pt=False,
        )
        with pytest.raises(ValueError, match="checkpoint_load_group"):
            loader._get_weights_iterator(
                source, None, (0, 1), Mapping(world_size=2, rank=0)
            )


class TestL3Identity:
    """L3 checkpoint identity treats runai_streamer tuning knobs as weight-invariant."""

    def test_runai_tuning_knobs_and_format_match_safetensors_identity(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "config.json").write_text(
            '{"model_type": "test"}', encoding="utf-8"
        )
        (tmp_path / "model.safetensors").write_bytes(b"identical-weight-bytes")

        id_safetensors = l3_checkpoint_id(
            str(tmp_path),
            hf_config=None,
            revision="",
            load_format="safetensors",
            model_loader_extra_config={},
            ext_yaml="",
        )
        id_runai_prefill = l3_checkpoint_id(
            str(tmp_path),
            hf_config=None,
            revision="",
            load_format="runai_streamer",
            model_loader_extra_config={"concurrency": 8, "distributed": True},
            ext_yaml="",
        )
        id_runai_decode = l3_checkpoint_id(
            str(tmp_path),
            hf_config=None,
            revision="",
            load_format="runai_streamer",
            model_loader_extra_config={
                "concurrency": 16,
                "memory_limit": 4 * 1024**3,
                "distributed": False,
            },
            ext_yaml="",
        )
        assert id_runai_prefill == id_safetensors
        assert id_runai_decode == id_safetensors


class TestObjectStoreUriHelpers:
    def test_parse_and_join_uris(self) -> None:
        assert is_object_store_uri("gs://my-bucket/org/model")
        assert is_object_store_uri("s3://my-bucket/org/model/")
        assert not is_object_store_uri("/local/path/model")
        assert not is_object_store_uri("org/model")

        assert parse_object_store_uri("gs://bucket/a/b/") == ("gs", "bucket", "a/b")
        assert parse_object_store_uri("s3://bucket") == ("s3", "bucket", "")
        assert (
            join_object_store_uri("gs://bucket/prefix/", "/shard.safetensors")
            == "gs://bucket/prefix/shard.safetensors"
        )

    def test_invalid_uris_raise(self) -> None:
        with pytest.raises(ValueError):
            parse_object_store_uri("http://example.com/model")
        with pytest.raises(ValueError):
            parse_object_store_uri("gs:///missing-bucket")

    def test_resolve_metadata_cache_dir_prefers_user_owned_locations(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        explicit = str(tmp_path / "dl")
        assert resolve_metadata_cache_dir(explicit) == os.path.join(
            explicit, "tokenspeed-object-store-meta"
        )

        hf_home = str(tmp_path / "hf")
        xdg_cache = str(tmp_path / "xdg")
        monkeypatch.setenv("HF_HOME", hf_home)
        monkeypatch.setenv("XDG_CACHE_HOME", xdg_cache)
        assert resolve_metadata_cache_dir(None) == os.path.join(
            hf_home, "tokenspeed-object-store-meta"
        )

        monkeypatch.delenv("HF_HOME", raising=False)
        assert resolve_metadata_cache_dir(None) == os.path.join(
            xdg_cache, "tokenspeed", "object-store-meta"
        )

        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        resolved_default = resolve_metadata_cache_dir(None)
        assert resolved_default.startswith(os.path.expanduser("~"))
        assert not resolved_default.startswith(tempfile.gettempdir())


class TestSyncObjectStoreMetadata:
    """Metadata sync: weight skipping, manifest reuse, stale refresh, and security."""

    def test_sync_skips_weights_reuses_marker_and_refreshes_on_version_change(
        self, tmp_path: Path
    ) -> None:
        uri = "gs://example-bucket/org/model"
        remote_state: dict[str, tuple[str, bytes]] = {
            "config.json": ("gen:1", b'{"v": 1}'),
            "encoding/encoding.py": ("gen:1", b"ENC = 1\n"),
            "model-00001-of-00001.safetensors": ("gen:1", b"HUGE_WEIGHTS"),
            "legacy.bin": ("gen:1", b"BIN_WEIGHTS"),
        }
        downloaded: list[str] = []

        def fake_list(_uri: str) -> list[tuple[str, str]]:
            return sorted((k, v[0]) for k, v in remote_state.items())

        def fake_download(obj_uri: str, dst: str) -> None:
            rel = obj_uri.removeprefix(uri + "/")
            downloaded.append(rel)
            Path(dst).write_bytes(remote_state[rel][1])

        cache_root = str(tmp_path / "cache")
        local_dir = sync_object_store_metadata(
            uri,
            cache_dir=cache_root,
            list_objects=fake_list,
            download_object=fake_download,
        )
        assert sorted(downloaded) == ["config.json", "encoding/encoding.py"]
        assert (Path(local_dir) / "config.json").read_bytes() == b'{"v": 1}'
        assert (
            Path(local_dir) / "encoding" / "encoding.py"
        ).read_bytes() == b"ENC = 1\n"
        assert not (Path(local_dir) / "model-00001-of-00001.safetensors").exists()
        assert not (Path(local_dir) / "legacy.bin").exists()
        assert stat.S_IMODE(os.lstat(local_dir).st_mode) == 0o700

        # Repeat sync with unchanged remote versions must reuse cache without downloads.
        downloaded.clear()
        local_dir_2 = sync_object_store_metadata(
            uri,
            cache_dir=cache_root,
            list_objects=fake_list,
            download_object=fake_download,
        )
        assert local_dir_2 == local_dir
        assert downloaded == []

        # Mutate config.json version, add tokenizer.json, and remove encoding/encoding.py.
        remote_state["config.json"] = ("gen:2", b'{"v": 2}')
        remote_state["tokenizer.json"] = ("gen:1", b"{}")
        del remote_state["encoding/encoding.py"]

        sync_object_store_metadata(
            uri,
            cache_dir=cache_root,
            list_objects=fake_list,
            download_object=fake_download,
        )
        assert sorted(downloaded) == ["config.json", "tokenizer.json"]
        assert (Path(local_dir) / "config.json").read_bytes() == b'{"v": 2}'
        assert (Path(local_dir) / "tokenizer.json").read_bytes() == b"{}"
        assert not (Path(local_dir) / "encoding" / "encoding.py").exists()

    def test_sync_rejects_path_traversal(self, tmp_path: Path) -> None:
        uri = "gs://example-bucket/org/evil"
        downloads: list[str] = []

        with pytest.raises(ValueError, match="Unsafe object key"):
            sync_object_store_metadata(
                uri,
                cache_dir=str(tmp_path / "cache"),
                list_objects=lambda _u: [("../escape.json", "gen:1")],
                download_object=lambda u, d: downloads.append(d),
            )
        assert downloads == []
        assert not (tmp_path / "escape.json").exists()

    def test_sync_rejects_symlinked_cache_dir(self, tmp_path: Path) -> None:
        uri = "gs://example-bucket/org/symlinked"
        cache_root = tmp_path / "cache"
        base_cache = Path(resolve_metadata_cache_dir(str(cache_root)))
        base_cache.mkdir(parents=True, mode=0o700)
        digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
        decoy = tmp_path / "decoy"
        decoy.mkdir()
        os.symlink(decoy, base_cache / digest)

        with pytest.raises(PermissionError, match="symlinked"):
            sync_object_store_metadata(
                uri,
                cache_dir=str(cache_root),
                list_objects=lambda _u: [("config.json", "gen:1")],
                download_object=lambda _u, d: Path(d).write_bytes(b"{}"),
            )


class TestRemoteSafetensorsFileAndSlice:
    """Byte-range safetensors decoding and row slicing against real safetensors files."""

    def test_get_tensor_and_row_slices_match_safetensors(self) -> None:
        tensors: dict[str, torch.Tensor] = {
            "w_bf16": torch.randn((10, 16), dtype=torch.bfloat16),
            "w_fp32": torch.randn((4, 8), dtype=torch.float32),
            "w_fp8": torch.randn((10, 32), dtype=torch.float32).to(torch.float8_e4m3fn),
            "w_u8": torch.randint(0, 255, (10, 12), dtype=torch.uint8),
            "w_i64": torch.arange(6, dtype=torch.int64),
        }
        if hasattr(torch, "float8_e8m0fnu"):
            tensors["w_e8m0"] = torch.randint(0, 255, (10, 8), dtype=torch.uint8).view(
                torch.float8_e8m0fnu
            )
        if hasattr(torch, "float4_e2m1fn_x2"):
            tensors["w_f4"] = torch.randint(0, 255, (6, 8), dtype=torch.uint8).view(
                torch.float4_e2m1fn_x2
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "shard.safetensors")
            safetensors.torch.save_file(tensors, path)
            raw_bytes = Path(path).read_bytes()
            ref_loaded = safetensors.torch.load_file(path)

            ranges_read: list[tuple[int, int]] = []

            def range_reader(start: int, end: int) -> bytes:
                ranges_read.append((start, end))
                return raw_bytes[start:end]

            remote = RemoteSafetensorsFile(
                "gs://bucket/shard.safetensors", range_reader=range_reader
            )
            assert sorted(remote.keys()) == sorted(tensors.keys())

            for name, expected in ref_loaded.items():
                got = remote.get_tensor(name)
                assert got.dtype == expected.dtype
                assert got.shape == expected.shape
                assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))

            # Verify row slicing parity against safe_open for both FP8 and E8M0 scales.
            with safe_open(path, framework="pt", device="cpu") as ref_handle:
                for slice_name in (
                    "w_fp8",
                    *(["w_e8m0"] if "w_e8m0" in tensors else []),
                ):
                    ranges_read.clear()
                    table_slice = remote.get_slice(slice_name)
                    ref_slice = ref_handle.get_slice(slice_name)
                    assert table_slice.get_shape() == ref_slice.get_shape()
                    chunks = [
                        table_slice[0:4, :],
                        table_slice[4:7, :],
                        table_slice[7:10, :],
                    ]
                    reconstructed = torch.cat(chunks, dim=0)
                    assert reconstructed.dtype == ref_loaded[slice_name].dtype
                    assert torch.equal(
                        reconstructed.view(torch.uint8),
                        ref_loaded[slice_name].view(torch.uint8),
                    )
                    assert (
                        sum(end - start for start, end in ranges_read)
                        == ref_loaded[slice_name].nbytes
                    )

                    single_row = table_slice[3]
                    assert single_row.dtype == ref_loaded[slice_name].dtype
                    assert torch.equal(
                        single_row.view(torch.uint8),
                        ref_loaded[slice_name][3].view(torch.uint8),
                    )
                    empty = table_slice[5:5, :]
                    assert empty.shape == (0, ref_loaded[slice_name].shape[1])

    def test_1d_subbyte_f4_slice_raises(self) -> None:
        header = json.dumps(
            {"f4_1d": {"dtype": "F4", "shape": [16], "data_offsets": [0, 8]}}
        ).encode("utf-8")
        raw = struct.pack("<Q", len(header)) + header + bytes(8)
        remote = RemoteSafetensorsFile(
            "gs://bucket/f4_1d.safetensors",
            range_reader=lambda s, e: raw[s:e],
        )
        got = remote.get_tensor("f4_1d")
        assert got.shape == (8,)
        with pytest.raises(ValueError, match="rank < 2"):
            remote.get_slice("f4_1d")

    def test_corrupt_header_raises(self) -> None:
        bad_len = struct.pack("<Q", 2**60)
        with pytest.raises(ValueError):
            RemoteSafetensorsFile(
                "gs://bucket/bad.safetensors",
                range_reader=lambda s, e: bad_len[s:e],
            )


class TestRunaiStreamerMixedShardIterator:
    """Mixed-shard byte-range splitting and scoped env var restoration."""

    def test_mixed_shard_splits_range_reads_and_restores_env_before_yield(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        shard1_tensors = {
            "layer0.weight": torch.randn((4, 8), dtype=torch.bfloat16),
            "layer0.bias": torch.randn((4,), dtype=torch.bfloat16),
        }
        shard2_tensors = {
            "layer1.proj.weight": torch.randn((4, 8), dtype=torch.bfloat16),
            "layer1.engram.embed.weight": torch.randint(
                0, 255, (64, 64), dtype=torch.uint8
            ),
        }
        shard3_tensors = {
            "layer2.engram.embed.weight": torch.randint(
                0, 255, (32, 32), dtype=torch.uint8
            ),
        }

        shard_bytes: dict[str, bytes] = {}
        shard_tensors_by_uri: dict[str, dict[str, torch.Tensor]] = {}
        for idx, data in enumerate(
            (shard1_tensors, shard2_tensors, shard3_tensors), start=1
        ):
            name = f"model-0000{idx}-of-00003.safetensors"
            local_path = tmp_path / name
            safetensors.torch.save_file(data, str(local_path))
            uri = f"gs://example-bucket/model/{name}"
            shard_bytes[uri] = local_path.read_bytes()
            shard_tensors_by_uri[uri] = data

        index_weight_map = {
            "layer0.weight": "model-00001-of-00003.safetensors",
            "layer0.bias": "model-00001-of-00003.safetensors",
            "layer1.proj.weight": "model-00002-of-00003.safetensors",
            "layer1.engram.embed.weight": "model-00002-of-00003.safetensors",
            "layer2.engram.embed.weight": "model-00003-of-00003.safetensors",
        }

        streamed_files: list[str] = []
        env_during_stream: list[tuple[str | None, str | None]] = []

        class _FakeStreamer:
            def __init__(self) -> None:
                self._files: list[str] = []

            def __enter__(self):
                env_during_stream.append(
                    (
                        os.environ.get("RUNAI_STREAMER_CONCURRENCY"),
                        os.environ.get("RUNAI_STREAMER_MEMORY_LIMIT"),
                    )
                )
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def stream_files(
                self, files: list[str], is_distributed: bool = False
            ) -> None:
                assert is_distributed is False
                self._files = list(files)
                streamed_files.extend(files)

            def get_tensors(self):
                for uri in self._files:
                    for k, v in shard_tensors_by_uri[uri].items():
                        yield k, v

        fake_mod = types.ModuleType("runai_model_streamer")
        fake_mod.SafetensorsStreamer = _FakeStreamer  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "runai_model_streamer", fake_mod)
        monkeypatch.delenv("RUNAI_STREAMER_CONCURRENCY", raising=False)
        monkeypatch.delenv("RUNAI_STREAMER_MEMORY_LIMIT", raising=False)

        ranges_by_uri: dict[str, list[tuple[int, int]]] = {}

        def fake_remote_file_factory(uri: str) -> RemoteSafetensorsFile:
            raw = shard_bytes[uri]
            recorded: list[tuple[int, int]] = []
            ranges_by_uri[uri] = recorded

            def reader(start: int, end: int) -> bytes:
                recorded.append((start, end))
                return raw[start:end]

            return RemoteSafetensorsFile(uri, range_reader=reader)

        uris = list(shard_bytes.keys())
        it = runai_streamer_weights_iterator(
            uris,
            distributed=False,
            weight_name_filter=lambda name: "engram.embed" not in name,
            index_weight_map=index_weight_map,
            concurrency=6,
            memory_limit=1024,
            remote_file_factory=fake_remote_file_factory,
        )

        yielded: list[tuple[str, torch.Tensor]] = []
        for name, tensor in it:
            # Env vars must already be restored before the very first yield.
            assert "RUNAI_STREAMER_CONCURRENCY" not in os.environ
            assert "RUNAI_STREAMER_MEMORY_LIMIT" not in os.environ
            yielded.append((name, tensor))

        assert env_during_stream == [("6", "1024")]
        assert streamed_files == [uris[0]]
        assert list(ranges_by_uri.keys()) == [uris[1]]

        # Rejected tensor's byte range in the mixed shard must never be read.
        shard2_raw = shard_bytes[uris[1]]
        hdr_len = struct.unpack("<Q", shard2_raw[:8])[0]
        hdr = json.loads(shard2_raw[8 : 8 + hdr_len].decode("utf-8"))
        data_base = 8 + hdr_len
        rej_start = data_base + hdr["layer1.engram.embed.weight"]["data_offsets"][0]
        rej_end = data_base + hdr["layer1.engram.embed.weight"]["data_offsets"][1]
        for start, end in ranges_by_uri[uris[1]]:
            assert end <= rej_start or start >= rej_end

        # Every accepted tensor is yielded exactly once and is byte-equal.
        names = [n for n, _ in yielded]
        assert len(names) == len(set(names))
        got_map = dict(yielded)
        assert set(got_map.keys()) == {
            "layer0.weight",
            "layer0.bias",
            "layer1.proj.weight",
        }
        expected_map = {
            **shard1_tensors,
            "layer1.proj.weight": shard2_tensors["layer1.proj.weight"],
        }
        for name, expected in expected_map.items():
            assert torch.equal(got_map[name], expected)


class TestFilterAndSliceHelpers:
    def test_filter_safetensors_files_supports_object_store_uris(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index = {
                "weight_map": {
                    "model.layer0.weight": "model-00001-of-00002.safetensors",
                    "model.engram.embed.weight": "model-00002-of-00002.safetensors",
                }
            }
            with open(
                os.path.join(tmpdir, "model.safetensors.index.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(index, f)

            uris = [
                "gs://bucket/model/model-00001-of-00002.safetensors",
                "gs://bucket/model/model-00002-of-00002.safetensors",
            ]
            kept = filter_safetensors_files_by_weight_names(
                uris,
                tmpdir,
                "model.safetensors.index.json",
                lambda name: "engram.embed" not in name,
            )
            assert kept == ["gs://bucket/model/model-00001-of-00002.safetensors"]

    def test_open_safetensors_slice_reads_local_shard(self) -> None:
        table = torch.randint(0, 255, (8, 16), dtype=torch.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            safetensors.torch.save_file(
                {"layers.1.engram.embed.weight": table},
                os.path.join(tmpdir, "model-00001-of-00001.safetensors"),
            )
            with open_safetensors_slice(
                tmpdir,
                "model-00001-of-00001.safetensors",
                "layers.1.engram.embed.weight",
            ) as tensor_slice:
                assert tuple(tensor_slice.get_shape()) == (8, 16)
                assert torch.equal(tensor_slice[2:6, :], table[2:6, :])

            with pytest.raises(FileNotFoundError):
                with open_safetensors_slice(
                    tmpdir, "missing.safetensors", "layers.1.engram.embed.weight"
                ):
                    pass


@pytest.mark.skipif(not RUNAI_AVAILABLE, reason="runai_model_streamer is not installed")
class TestRunaiStreamerLocalParity:
    def test_local_shards_match_default_safetensors_loader(self) -> None:
        shard1 = {
            "layer0.weight": torch.randn((8, 16), dtype=torch.bfloat16),
            "layer0.bias": torch.randn((8,), dtype=torch.bfloat16),
        }
        shard2 = {
            "layer1.weight": torch.randn((8, 16), dtype=torch.bfloat16),
            "skip.table": torch.randint(0, 255, (16, 16), dtype=torch.uint8),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            safetensors.torch.save_file(
                shard1, os.path.join(tmpdir, "model-00001-of-00002.safetensors")
            )
            safetensors.torch.save_file(
                shard2, os.path.join(tmpdir, "model-00002-of-00002.safetensors")
            )
            index = {
                "weight_map": {
                    "layer0.weight": "model-00001-of-00002.safetensors",
                    "layer0.bias": "model-00001-of-00002.safetensors",
                    "layer1.weight": "model-00002-of-00002.safetensors",
                    "skip.table": "model-00002-of-00002.safetensors",
                }
            }
            with open(
                os.path.join(tmpdir, "model.safetensors.index.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(index, f)

            source = DefaultModelLoader.Source(
                model_or_path=tmpdir,
                revision=None,
                weight_files=None,
                prefix="",
                fall_back_to_pt=False,
            )
            mapping = Mapping(world_size=1, rank=0)
            predicate = lambda name: name != "skip.table"

            ref_loader = DefaultModelLoader(
                LoadConfig(
                    load_format="safetensors",
                    weight_loader_prefetch_checkpoints=False,
                )
            )
            stream_loader = RunaiModelStreamerLoader(
                LoadConfig(
                    load_format="runai_streamer",
                    model_loader_extra_config={"concurrency": 2},
                )
            )

            ref = dict(
                ref_loader._get_weights_iterator(source, predicate, None, mapping)
            )
            got = dict(
                stream_loader._get_weights_iterator(source, predicate, None, mapping)
            )
            assert set(got.keys()) == set(ref.keys())
            assert "skip.table" not in got
            for name, ref_tensor in ref.items():
                assert torch.equal(got[name], ref_tensor)


class TestObjectStoreMetadataResolution:
    def test_resolve_checkpoint_metadata_dir_handles_local_cache_and_hub(
        self, monkeypatch
    ) -> None:
        from tokenspeed.runtime.model_loader import object_store

        with tempfile.TemporaryDirectory() as tmpdir:
            local_model = os.path.join(tmpdir, "local-model")
            os.makedirs(local_model)
            assert (
                object_store.resolve_checkpoint_metadata_dir(local_model, sync=True)
                == local_model
            )
            assert (
                object_store.resolve_checkpoint_metadata_dir("org/hub-model", sync=True)
                is None
            )

            uri = "gs://example-bucket/org/cached-model"
            cache_root = os.path.join(tmpdir, "cache")

            def forbid_sync(*args, **kwargs):
                raise AssertionError("sync_object_store_metadata must not be called")

            monkeypatch.setattr(object_store, "sync_object_store_metadata", forbid_sync)
            # Uncached URI with sync=False returns None with no network call.
            assert (
                object_store.resolve_checkpoint_metadata_dir(
                    uri, download_dir=cache_root, sync=False
                )
                is None
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            uri = "gs://example-bucket/org/cached-model"
            cache_root = os.path.join(tmpdir, "cache")
            synced_dir = sync_object_store_metadata(
                uri,
                cache_dir=cache_root,
                list_objects=lambda _u: [("config.json", "gen:1")],
                download_object=lambda _src, dst: Path(dst).write_text(
                    '{"architectures": ["LlamaForCausalLM"]}', encoding="utf-8"
                ),
            )
            monkeypatch.setattr(object_store, "sync_object_store_metadata", forbid_sync)
            # Cached URI with sync=False returns the local synced dir without network.
            assert (
                object_store.resolve_checkpoint_metadata_dir(
                    uri, download_dir=cache_root, sync=False
                )
                == synced_dir
            )

    def test_count_dspark_stages_resolves_object_store_uri(self, monkeypatch) -> None:
        from tokenspeed.runtime.model_loader import object_store
        from tokenspeed.runtime.utils import spec_block_geometry

        uri = "gs://example-bucket/org/dspark-model"
        with tempfile.TemporaryDirectory() as tmpdir:
            synced_dir = os.path.join(tmpdir, "synced")
            os.makedirs(synced_dir)
            index_payload = {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001.safetensors",
                    "mtp.0.proj.weight": "model-00002.safetensors",
                    "mtp.1.proj.weight": "model-00002.safetensors",
                    "mtp.2.proj.weight": "model-00002.safetensors",
                }
            }
            Path(os.path.join(synced_dir, "model.safetensors.index.json")).write_text(
                json.dumps(index_payload), encoding="utf-8"
            )

            seen: list[tuple[str, str | None]] = []

            def fake_sync(target_uri: str, cache_dir: str | None = None) -> str:
                seen.append((target_uri, cache_dir))
                return synced_dir

            def forbid_hf(*args, **kwargs):
                raise AssertionError(
                    "hf_hub_download must not be called for object-store URIs"
                )

            monkeypatch.setattr(object_store, "sync_object_store_metadata", fake_sync)
            monkeypatch.setattr(spec_block_geometry, "hf_hub_download", forbid_hf)

            stages = spec_block_geometry.count_dspark_stages(
                uri, download_dir="/custom/cache"
            )
            assert stages == 3
            assert seen == [(uri, "/custom/cache")]

    def test_checkpoint_architectures_and_defaults_for_object_store_uri(
        self, monkeypatch
    ) -> None:
        import huggingface_hub

        from tokenspeed.runtime.model_loader import object_store
        from tokenspeed.runtime.utils import server_args as server_args_mod
        from tokenspeed.runtime.utils.server_args import (
            ServerArgs,
            _checkpoint_architectures,
        )

        uri = "gs://example-bucket/org/beam-model"
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_cache = os.path.join(tmpdir, "empty-cache")
            populated_cache = os.path.join(tmpdir, "populated-cache")
            synced_dir = sync_object_store_metadata(
                uri,
                cache_dir=populated_cache,
                list_objects=lambda _u: [("config.json", "gen:1")],
                download_object=lambda _src, dst: Path(dst).write_text(
                    '{"architectures": ["BeamForCausalLM"]}', encoding="utf-8"
                ),
            )

            def forbid_hf(*args, **kwargs):
                raise AssertionError("HF hub must not be queried for object-store URIs")

            monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbid_hf)
            monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", forbid_hf)

            # Uncached + download=False: returns None without touching HF hub or syncing.
            assert (
                _checkpoint_architectures(
                    uri, None, download=False, download_dir=empty_cache
                )
                is None
            )
            # Cached + download=False: reads synced config.json without network.
            assert _checkpoint_architectures(
                uri, None, download=False, download_dir=populated_cache
            ) == ["BeamForCausalLM"]

            # download=True: syncs metadata and applies Beam defaults on ServerArgs.
            seen: list[tuple[str, str | None]] = []

            def fake_sync(target_uri: str, cache_dir: str | None = None) -> str:
                seen.append((target_uri, cache_dir))
                return synced_dir

            monkeypatch.setattr(object_store, "sync_object_store_metadata", fake_sync)
            monkeypatch.setattr(
                server_args_mod, "get_amdgpu_memory_capacity", lambda: 80000
            )
            monkeypatch.setattr(
                server_args_mod, "get_nvgpu_memory_capacity", lambda: 80000
            )
            args = ServerArgs(
                model=uri,
                load_format="runai_streamer",
                download_dir=empty_cache,
            )
            assert not args.disable_tf32
            args.resolve_model_defaults(download=True)
            assert args.disable_tf32 is True
            assert seen == [(uri, empty_cache)]

    def test_model_config_reads_hf_quant_config_from_object_store_uri(
        self, monkeypatch
    ) -> None:
        import huggingface_hub
        from transformers import PretrainedConfig

        from tokenspeed.runtime.configs.model_config import ModelConfig
        from tokenspeed.runtime.model_loader import object_store

        uri = "gs://example-bucket/org/nvfp4-model"
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(os.path.join(tmpdir, "hf_quant_config.json")).write_text(
                json.dumps({"quantization": {"quant_algo": "NVFP4"}}),
                encoding="utf-8",
            )

            def fake_sync(target_uri: str, cache_dir: str | None = None) -> str:
                assert target_uri == uri
                assert cache_dir == "/custom/cache"
                return tmpdir

            def forbid_snapshot(*args, **kwargs):
                raise AssertionError(
                    "snapshot_download must not be called for object-store URIs"
                )

            monkeypatch.setattr(object_store, "sync_object_store_metadata", fake_sync)
            monkeypatch.setattr(huggingface_hub, "snapshot_download", forbid_snapshot)

            cfg = ModelConfig.__new__(ModelConfig)
            cfg.model_path = uri
            cfg.revision = None
            cfg.download_dir = "/custom/cache"
            cfg.hf_config = PretrainedConfig()

            parsed = cfg._parse_quant_hf_config()
            assert parsed == {"quant_method": "modelopt", "quant_algo": "NVFP4"}

    def test_l3_checkpoint_id_and_resolved_model_dir_reject_object_store_uri(
        self, monkeypatch
    ) -> None:
        import huggingface_hub

        from tokenspeed.runtime.cache.l3.identity import _resolved_model_dir

        def forbid_snapshot(*args, **kwargs):
            raise AssertionError(
                "snapshot_download must not be called for object-store URIs"
            )

        monkeypatch.setattr(huggingface_hub, "snapshot_download", forbid_snapshot)
        uri = "gs://example-bucket/org/model"
        assert _resolved_model_dir(uri, revision="") is None
        with pytest.raises(ValueError, match="immutable checkpoint id"):
            l3_checkpoint_id(
                model_path=uri,
                revision="",
                hf_config=object(),
                load_format="runai_streamer",
                model_loader_extra_config={"concurrency": 16},
                ext_yaml=None,
            )

    def test_non_runai_loaders_reject_object_store_uri(self) -> None:
        from tokenspeed.runtime.model_loader.loader import ShardedStateLoader

        uri = "gs://example-bucket/org/model"
        default_loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
        with pytest.raises(ValueError, match="runai_streamer"):
            default_loader._prepare_weights(uri, None, False, None)

        sharded_loader = ShardedStateLoader(LoadConfig(load_format="sharded_state"))
        with pytest.raises(ValueError, match="runai_streamer"):
            sharded_loader._prepare_weights(uri, None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
