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

"""KV caches run at unit scale, so a checkpoint or scale file may only carry ones."""

import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution import weight_loader  # noqa: E402
from tokenspeed.runtime.model_loader.weight_utils import (  # noqa: E402
    require_unit_kv_scale_file,
    require_unit_kv_scales,
)

KV_SCALE_NAMES = [
    "layers.3.self_attn.k_proj.k_scale",
    "layers.3.self_attn.v_proj.v_scale",
    "layers.3.self_attn.attn.kv_scale",
    "layers.3.self_attn.attn_k_scale",
    "layers.3.self_attn.attn_v_scale",
    "layers.3.self_attn.k_proj.output_scale",
    "layers.3.self_attn.v_proj.output_scale",
]


def test_weights_without_kv_scales_pass_through_unchanged():
    weights = [("layers.0.self_attn.k_proj.weight", torch.ones(4, 4))]
    assert list(require_unit_kv_scales(weights)) == weights


@pytest.mark.parametrize("name", KV_SCALE_NAMES)
def test_unit_kv_scales_pass_through(name):
    weights = [(name, torch.ones(4))]
    assert list(require_unit_kv_scales(weights)) == weights


@pytest.mark.parametrize("name", KV_SCALE_NAMES)
@pytest.mark.parametrize("value", [0.5, 2.0, 1 + 2**-23, float("nan")])
def test_a_non_unit_kv_scale_fails_the_load(name, value):
    """One non-unit entry of a per-head scale is enough."""
    scale = torch.ones(4)
    scale[2] = value
    with pytest.raises(ValueError, match=name):
        list(require_unit_kv_scales([(name, scale)]))


def test_weight_scales_are_not_mistaken_for_kv_scales():
    weights = [
        ("layers.0.self_attn.k_proj.weight_scale", torch.tensor(0.5)),
        ("layers.0.self_attn.k_scale.weight", torch.tensor(0.5)),
    ]
    assert list(require_unit_kv_scales(weights)) == weights


def test_the_checkpoint_error_names_every_non_unit_scale():
    """The stream is drained before raising, so the message lists them all."""
    weights = [
        ("layers.0.self_attn.k_proj.k_scale", torch.tensor(0.5)),
        ("layers.0.mlp.weight", torch.ones(2)),
        ("layers.5.self_attn.v_proj.v_scale", torch.tensor(2.0)),
    ]
    seen = []
    with pytest.raises(ValueError) as info:
        for name, _ in require_unit_kv_scales(weights):
            seen.append(name)
    assert seen == [name for name, _ in weights]
    assert "layers.0.self_attn.k_proj.k_scale" in str(info.value)
    assert "layers.5.self_attn.v_proj.v_scale" in str(info.value)


def _scale_file(tmp_path, scales: dict[int, dict[int, float]]) -> str:
    path = tmp_path / "kv_scales.json"
    path.write_text(
        json.dumps(
            {
                "model_type": "llama",
                "kv_cache": {"dtype": "float8_e4m3fn", "scaling_factor": scales},
            }
        )
    )
    return str(path)


def test_a_unit_scale_file_is_accepted(tmp_path):
    require_unit_kv_scale_file(_scale_file(tmp_path, {0: {0: 1.0, 1: 1.0}}))


@pytest.mark.parametrize("value", [0.75, 1.5, 1 + 2**-52, float("nan")])
def test_a_scale_file_with_a_non_unit_scale_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="TP rank 0 layer 1"):
        require_unit_kv_scale_file(
            _scale_file(tmp_path, {0: {0: 1.0, 1: value, 2: 1.0}})
        )


def test_every_tp_rank_of_a_scale_file_is_checked(tmp_path):
    with pytest.raises(ValueError, match="TP rank 1 layer 0"):
        require_unit_kv_scale_file(_scale_file(tmp_path, {0: {0: 1.0}, 1: {0: 0.5}}))


@pytest.mark.parametrize(
    "contents",
    [
        None,
        "{}",
        "not json",
        '{"kv_cache": {"scaling_factor": [1.0]}}',
        '{"kv_cache": {"scaling_factor": {"0": [1.0]}}}',
        '{"kv_cache": {"scaling_factor": {"0": {"0": "abc"}}}}',
        '{"kv_cache": {"scaling_factor": {"0": {"0": null}}}}',
        '{"kv_cache": {"scaling_factor": {"0": {"0": [1.0]}}}}',
        '{"kv_cache": null}',
        "[1.0]",
    ],
)
def test_an_unreadable_scale_file_names_the_expected_layout(tmp_path, contents):
    path = tmp_path / "kv_scales.json"
    if contents is not None:
        path.write_text(contents)
    with pytest.raises(ValueError, match="kv_cache.scaling_factor"):
        require_unit_kv_scale_file(str(path))


def _the_load_begins(*args):
    raise RuntimeError("the weights would load now")


@pytest.mark.parametrize(
    "kv_cache_dtype,path,checked",
    [
        ("fp8_e4m3", "kv_scales.json", True),
        ("auto", "kv_scales.json", False),
        ("fp8_e4m3", None, False),
    ],
)
def test_the_loader_checks_the_scale_file_of_an_fp8_cache(
    monkeypatch, kv_cache_dtype, path, checked
):
    """The check runs before any weight loads, and only for an FP8 cache."""
    checks = []
    monkeypatch.setattr(weight_loader, "require_unit_kv_scale_file", checks.append)
    monkeypatch.setattr(weight_loader, "get_available_gpu_memory", _the_load_begins)
    server_args = SimpleNamespace(
        kv_cache_dtype=kv_cache_dtype, quantization_param_path=path
    )
    with pytest.raises(RuntimeError, match="would load now"):
        weight_loader.WeightLoader.load_model(None, server_args, "cuda", 0, None)
    assert checks == ([path] if checked else [])


def test_a_checkpoint_load_rejects_a_non_unit_kv_scale(tmp_path):
    from safetensors.torch import save_file

    from tokenspeed.runtime.configs.load_config import LoadConfig
    from tokenspeed.runtime.model_loader.loader import DefaultModelLoader

    save_file(
        {
            "model.layers.0.self_attn.k_proj.weight": torch.zeros(2, 2),
            "model.layers.0.self_attn.k_proj.k_scale": torch.tensor(0.5),
        },
        str(tmp_path / "model.safetensors"),
    )
    loader = DefaultModelLoader(LoadConfig())
    model = SimpleNamespace(fall_back_to_pt_during_load=False, secondary_weights=())
    model_config = SimpleNamespace(model_path=str(tmp_path), revision=None)
    with pytest.raises(ValueError, match="k_proj.k_scale"):
        list(loader._get_all_weights(model_config, model))


def test_a_distributed_update_finishes_its_load_before_rejecting_a_scale(monkeypatch):
    """The trainer broadcasts every weight, and the model completes its load,
    post-load derivations included, before the update reports the scale."""
    import torch.distributed as dist

    from tokenspeed.runtime.execution.model_runner import ModelRunner

    received = []
    monkeypatch.setattr(
        dist, "broadcast", lambda buf, src, group: received.append(buf.fill_(0.5))
    )
    loaded = []

    def load_weights(weights):
        loaded.extend(name for name, _ in weights)
        loaded.append("post_load")

    runner = object.__new__(ModelRunner)
    runner._weight_update_pg = object()
    runner._weight_update_device = torch.device("cuda")
    runner.model = SimpleNamespace(load_weights=load_weights)
    names = ["layers.0.self_attn.k_proj.k_scale", "layers.0.mlp.weight"]
    ok, message = runner.update_weights_from_distributed(
        SimpleNamespace(names=names, dtype_names=["float32"] * 2, shapes=[[], [2]])
    )
    assert not ok and "k_proj.k_scale" in message
    assert len(received) == len(names)
    assert loaded == [*names, "post_load"]
