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

from test.runtime.models.checkpoint_fixture import (
    _runner,
    _TinyModel,
    _write_checkpoint,
)

from tokenspeed.runtime.engine.io_struct import UpdateWeightFromDiskReqInput
from tokenspeed.runtime.execution import weight_loader
from tokenspeed.runtime.execution.weight_loader import WeightLoader
from tokenspeed.runtime.model_loader import loader as loader_module
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


def _startup_runner(source, startup_format, model_type, monkeypatch):
    runner = _runner(model_type(), source)
    runner.server_args.load_format = startup_format
    runner.server_args.kv_cache_dtype = "bfloat16"
    runner.server_args.quantization_param_path = None
    runner.server_args.model_update_draft_weights = None
    runner.model_config.dtype = torch.float32
    runner.model_config.vocab_size = 4
    runner.is_draft_worker = False
    runner.memory_saver_adapter = TorchMemorySaverAdapter.create(False, device="cpu")
    # Substitute accelerator reporting/placement; checkpoint IO and reloads are real.
    monkeypatch.setattr(weight_loader, "get_available_gpu_memory", lambda *args: 0.0)
    monkeypatch.setattr(
        weight_loader, "DeviceConfig", lambda device: SimpleNamespace(device=device)
    )
    monkeypatch.setattr(loader_module, "_initialize_model", lambda *args: model_type())
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
    model_type = _TinyModel
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

        class OfflineAfterLoad(_TinyModel):
            def load_weights(self, weights):
                super().load_weights(weights)
                monkeypatch.setattr(loader_module, "download_weights_from_hf", offline)

        model_type = OfflineAfterLoad
    runner = _startup_runner(initial, startup_format, model_type, monkeypatch)
    if source_kind == "hub":
        runner.model_config.model_path = "fixture/model"
    runner.load_model()
    runner.model.seen.clear()
    before = [parameter.detach().clone() for parameter in runner.model.parameters()]
    shutil.rmtree(initial)

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    save_file(
        {"b": torch.full((4,), 13.0), "w": torch.ones(8, 3)},
        str(invalid / "model.safetensors"),
    )
    ok, message = runner.update_weights_from_disk(
        UpdateWeightFromDiskReqInput(model_path=str(invalid), load_format="auto")
    )
    assert not ok and "has shape" in message
    assert runner.model.seen == []
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
    assert torch.equal(runner.model.w, expected["w"])
    assert torch.equal(runner.model.b, expected["b"])


@pytest.mark.parametrize(
    "source", ["s3://test/model", "gs://test/model", "fixture/model"]
)
def test_optional_metadata_does_not_download_weights(tmp_path, monkeypatch, source):
    def offline(*args, **kwargs):
        raise ConnectionError("metadata must not resolve a remote checkpoint")

    monkeypatch.setattr(loader_module, "download_weights_from_hf", offline)
    runner = _runner(_TinyModel(), tmp_path)
    runner.model_config.model_path = source
    assert WeightLoader.checkpoint_shapes(runner.model_config, runner.model) == {}


@pytest.mark.parametrize("with_metadata", [False, True], ids=["model", "metadata"])
def test_empty_snapshot_does_not_reprobe_the_source(
    tmp_path, monkeypatch, with_metadata
):
    expected = _write_checkpoint(tmp_path, scale=3)
    runner = _startup_runner(tmp_path, "pt", _TinyModel, monkeypatch)
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
    assert torch.equal(model.w, expected["w"])
    assert torch.equal(model.b, expected["b"])
    if with_metadata:
        assert loaded.checkpoint_shapes == {}
