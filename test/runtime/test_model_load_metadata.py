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

"""Startup checkpoint metadata remains usable after the source is removed."""

import json
import os
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.engine.io_struct import UpdateWeightFromDiskReqInput
from tokenspeed.runtime.execution import weight_loader
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.execution.weight_loader import WeightLoader
from tokenspeed.runtime.model_loader import loader as loader_module
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


class _CheckpointLinear(torch.nn.Linear):
    """Canonical checkpoint weights are transposed during loading."""

    def __init__(self):
        super().__init__(3, 4)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, tensor in weights:
            params[name].data.copy_(
                tensor.reshape(3, 4).T if name == "weight" else tensor
            )


def _write_checkpoint(directory, scale):
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4) * scale,
        "bias": torch.arange(4, dtype=torch.float32) * scale,
    }
    save_file(tensors, str(directory / "model.safetensors"))
    return tensors


def _startup_runner(source, startup_format, model_type, monkeypatch):
    # Only accelerator setup is substituted; model/checkpoint IO and reloads are real.
    monkeypatch.setattr(weight_loader, "get_available_gpu_memory", lambda *args: 0.0)
    monkeypatch.setattr(
        weight_loader, "DeviceConfig", lambda device: SimpleNamespace(device=device)
    )
    monkeypatch.setattr(loader_module, "_initialize_model", lambda *args: model_type())
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = "cpu"
    runner.gpu_id = 0
    runner.checkpoint_load_group = None
    runner.mapping = Mapping(rank=0)
    runner.model_config = SimpleNamespace(
        model_path=str(source),
        mapping=runner.mapping,
        revision=None,
        dtype=torch.float32,
        vocab_size=4,
    )
    runner.server_args = SimpleNamespace(
        enable_memory_saver=False,
        weight_memory_policy=None,
        load_format=startup_format,
        download_dir=None,
        model_loader_extra_config={},
        ext_yaml=None,
        weight_loader_prefetch_checkpoints=False,
        weight_loader_prefetch_num_threads=1,
        kv_cache_dtype="bfloat16",
        quantization_param_path=None,
        model_update_draft_weights=None,
    )
    runner.is_draft_worker = False
    runner.memory_saver_adapter = TorchMemorySaverAdapter.create(False, device="cpu")
    runner._disk_reload_shapes = {}
    return runner


@pytest.mark.parametrize(
    "startup_format,source_kind",
    [
        ("auto", "local"),
        ("dummy", "local"),
        ("runai_streamer", "local"),
        ("auto", "hub"),
    ],
)
def test_disk_reload_keeps_shape_validation_after_initial_source_is_removed(
    tmp_path, monkeypatch, startup_format, source_kind
):
    if startup_format == "runai_streamer":
        pytest.importorskip("runai_model_streamer")
    initial = tmp_path / "initial"
    _write_checkpoint(initial, scale=1)
    model_type = _CheckpointLinear
    if source_kind == "hub":
        monkeypatch.delenv("TOKENSPEED_USE_MODELSCOPE", raising=False)
        monkeypatch.setattr(
            loader_module,
            "download_weights_from_hf",
            lambda *args, **kwargs: str(initial),
        )
        monkeypatch.setattr(
            loader_module, "download_safetensors_index_file_from_hf", lambda *args: None
        )

        def offline(*args, **kwargs):
            raise ConnectionError("the hub became unavailable after loading")

        class OfflineAfterLoad(_CheckpointLinear):
            def load_weights(self, weights):
                super().load_weights(weights)
                monkeypatch.setattr(loader_module, "download_weights_from_hf", offline)

        model_type = OfflineAfterLoad
    runner = _startup_runner(initial, startup_format, model_type, monkeypatch)
    if source_kind == "hub":
        runner.model_config.model_path = "fixture/model"
    runner.load_model()
    before = [parameter.detach().clone() for parameter in runner.model.parameters()]
    shutil.rmtree(initial)

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    save_file(
        {"bias": torch.full((4,), 13.0), "weight": torch.ones(8, 3)},
        str(invalid / "model.safetensors"),
    )
    ok, message = runner.update_weights_from_disk(
        UpdateWeightFromDiskReqInput(model_path=str(invalid), load_format="auto")
    )
    assert not ok and "has shape" in message
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(runner.model.parameters(), before)
    )

    replacement = tmp_path / "replacement"
    expected = _write_checkpoint(replacement, scale=7)
    ok, message = runner.update_weights_from_disk(
        UpdateWeightFromDiskReqInput(model_path=str(replacement), load_format="auto")
    )
    assert ok, message
    assert torch.equal(runner.model.weight, expected["weight"].T)
    assert torch.equal(runner.model.bias, expected["bias"])


@pytest.mark.parametrize(
    "source", ["s3://test/model", "gs://test/model", "fixture/model"]
)
def test_optional_metadata_does_not_download_weights(tmp_path, monkeypatch, source):
    def offline(*args, **kwargs):
        raise ConnectionError("metadata must not resolve a remote checkpoint")

    monkeypatch.setattr(loader_module, "download_weights_from_hf", offline)
    assert (
        WeightLoader.checkpoint_shapes(
            SimpleNamespace(model_path=source), _CheckpointLinear()
        )
        == {}
    )


@pytest.mark.parametrize("with_metadata", [False, True], ids=["model", "metadata"])
def test_empty_snapshot_does_not_reprobe_the_source(
    tmp_path, monkeypatch, with_metadata
):
    expected = _write_checkpoint(tmp_path, scale=3)
    runner = _startup_runner(tmp_path, "pt", _CheckpointLinear, monkeypatch)
    torch.save(expected, tmp_path / "model.pt")
    # PT loading must not probe an unrelated safetensors file afterwards.
    (tmp_path / "model.safetensors").write_bytes(b"not a readable shard")
    load = (
        WeightLoader.load_model_with_metadata
        if with_metadata
        else WeightLoader.load_model
    )
    loaded = load(
        model_config=runner.model_config,
        server_args=runner.server_args,
        device="cpu",
        gpu_id=0,
        memory_saver_adapter=runner.memory_saver_adapter,
        checkpoint_load_group=None,
        is_draft_worker=False,
    )
    model = loaded.model if with_metadata else loaded
    assert torch.equal(model.weight, expected["weight"].T)
    assert torch.equal(model.bias, expected["bias"])
    if with_metadata:
        assert loaded.checkpoint_shapes == {}


@pytest.mark.parametrize(
    "startup_format",
    ["sharded_state", loader_module.ShardedStateLoader],
    ids=["sharded-state", "custom-loader"],
)
def test_rank_local_state_does_not_define_canonical_checkpoint_shapes(
    tmp_path, monkeypatch, startup_format
):
    initial = tmp_path / "initial"
    initial.mkdir()
    reference = _CheckpointLinear()
    save_file(reference.state_dict(), str(initial / "model-rank-0-part-0.safetensors"))
    runner = _startup_runner(initial, startup_format, _CheckpointLinear, monkeypatch)
    runner.load_model()
    assert torch.equal(runner.model.weight, reference.weight)
    assert torch.equal(runner.model.bias, reference.bias)
    shutil.rmtree(initial)

    replacement = tmp_path / "replacement"
    expected = _write_checkpoint(replacement, scale=7)
    ok, message = runner.update_weights_from_disk(
        UpdateWeightFromDiskReqInput(model_path=str(replacement), load_format="auto")
    )
    assert ok, message
    assert torch.equal(runner.model.weight, expected["weight"].T)
    assert torch.equal(runner.model.bias, expected["bias"])


@pytest.mark.parametrize("startup_format", ["dummy", "auto"])
@pytest.mark.parametrize(
    "checkpoint_state,error",
    [
        ("truncated", ValueError),
        ("index-ahead", RuntimeError),
        ("missing-weight-map", KeyError),
        ("invalid-weight-map", TypeError),
    ],
)
def test_incomplete_checkpoint_only_prevents_real_weight_loading(
    tmp_path, monkeypatch, caplog, startup_format, checkpoint_state, error
):
    _write_checkpoint(tmp_path, scale=1)
    shard = tmp_path / "model.safetensors"
    if checkpoint_state == "truncated":
        shard.write_bytes(shard.read_bytes()[:-1])
    else:
        index = {
            "index-ahead": {"weight_map": {"weight": "pending.safetensors"}},
            "missing-weight-map": {},
            "invalid-weight-map": {"weight_map": None},
        }[checkpoint_state]
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    runner = _startup_runner(tmp_path, startup_format, _CheckpointLinear, monkeypatch)
    if startup_format == "auto":
        with pytest.raises(error):
            runner.load_model()
    else:
        runner.load_model()
        assert runner._disk_reload_shapes == {}
        assert torch.isfinite(runner.model.weight).all()
        assert "optional dummy checkpoint metadata" in caplog.text


def test_pt_reload_discards_the_previous_checkpoint_shapes(tmp_path, monkeypatch):
    initial = tmp_path / "initial"
    tensors = _write_checkpoint(initial, scale=1)
    runner = _startup_runner(initial, "auto", _CheckpointLinear, monkeypatch)
    runner.load_model()
    shutil.rmtree(initial)

    for file_format, scale in (("pt", 3), ("safetensors", 7)):
        path = tmp_path / file_format
        path.mkdir()
        expected = {name: tensor * scale for name, tensor in tensors.items()}
        expected["weight"] = expected["weight"].flatten()
        save = torch.save if file_format == "pt" else save_file
        save(expected, str(path / f"model.{file_format}"))
        ok, message = runner.update_weights_from_disk(
            UpdateWeightFromDiskReqInput(model_path=str(path), load_format=file_format)
        )
        assert ok, message
        assert torch.equal(runner.model.weight, expected["weight"].reshape(3, 4).T)
        assert torch.equal(runner.model.bias, expected["bias"])
