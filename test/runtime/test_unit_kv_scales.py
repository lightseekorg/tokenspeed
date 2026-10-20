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

"""The startup load hands a checkpoint's KV-cache scales to the model; a scale
file and a weight update may only carry ones."""

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
    non_unit_kv_scale_message,
    record_non_unit_kv_scales,
    require_unit_kv_scale_file,
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


def _screen(weights):
    """The weight-update screen: weights passed through and the scales rejected."""
    rejected: list[str] = []
    return list(record_non_unit_kv_scales(weights, rejected)), rejected


def test_weights_without_kv_scales_pass_through_unchanged():
    weights = [("layers.0.self_attn.k_proj.weight", torch.ones(4, 4))]
    assert _screen(weights) == (weights, [])


@pytest.mark.parametrize("name", KV_SCALE_NAMES)
def test_a_weight_update_drops_unit_kv_scales(name):
    assert _screen([(name, torch.ones(4))]) == ([], [])


@pytest.mark.parametrize("name", KV_SCALE_NAMES)
@pytest.mark.parametrize("value", [0.5, 2.0, 1 + 2**-23, float("nan")])
def test_a_weight_update_rejects_a_non_unit_kv_scale(name, value):
    """One non-unit entry of a per-head scale is enough."""
    scale = torch.ones(4)
    scale[2] = value
    _, rejected = _screen([(name, scale)])
    assert len(rejected) == 1 and rejected[0].startswith(name)


def test_weight_scales_are_not_mistaken_for_kv_scales():
    weights = [
        ("layers.0.self_attn.k_proj.weight_scale", torch.tensor(0.5)),
        ("layers.0.self_attn.k_scale.weight", torch.tensor(0.5)),
    ]
    assert _screen(weights) == (weights, [])


def test_the_rejection_names_every_non_unit_scale():
    weights = [
        ("layers.0.self_attn.k_proj.k_scale", torch.tensor(0.5)),
        ("layers.0.mlp.weight", torch.ones(2)),
        ("layers.5.self_attn.v_proj.v_scale", torch.tensor(2.0)),
    ]
    passed, rejected = _screen(weights)
    assert [name for name, _ in passed] == ["layers.0.mlp.weight"]
    message = non_unit_kv_scale_message(rejected)
    assert "layers.0.self_attn.k_proj.k_scale" in message
    assert "layers.5.self_attn.v_proj.v_scale" in message


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
        weight_loader.WeightLoader.load_model(
            None, server_args, "cuda", 0, None, checkpoint_load_group=None
        )
    assert checks == ([path] if checked else [])


class _Linear(torch.nn.Linear):
    """A model whose ``load_weights`` knows only its own parameters."""

    def load_weights(self, weights):
        for name, tensor in weights:
            self.get_parameter(name).data.copy_(tensor)


@pytest.mark.parametrize("k_scale", [1.0, 0.5])
def test_the_startup_load_refuses_scales_the_model_does_not_load(
    tmp_path, monkeypatch, k_scale
):
    """The loader keeps every KV-cache scale from ``load_weights``; one other
    than one needs the model's ``load_kv_cache_scales``."""
    from safetensors.torch import save_file

    from tokenspeed.runtime.configs.load_config import LoadConfig
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.model_loader import loader

    save_file(
        {
            "weight": torch.ones(2, 4),
            "bias": torch.zeros(2),
            "attn.k_scale": torch.tensor([k_scale]),
            "attn.v_scale": torch.tensor([1.0]),
        },
        str(tmp_path / "model.safetensors"),
    )
    monkeypatch.setattr(loader, "_initialize_model", lambda *args: _Linear(4, 2))

    def load():
        return loader.DefaultModelLoader(LoadConfig()).load_model(
            model_config=SimpleNamespace(
                model_path=str(tmp_path),
                revision=None,
                dtype=torch.float32,
                mapping=Mapping(rank=0, world_size=1),
            ),
            device_config=SimpleNamespace(device="cpu"),
        )

    if k_scale == 1.0:
        assert torch.equal(load().weight, torch.ones(2, 4))
    else:
        with pytest.raises(ValueError, match=r"_Linear: .*attn\.k_scale=\[0\.5\]"):
            load()


def _attention():
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

    return PagedAttention(4, 16, 0.25, 2, 0, rotary_emb=None, qk_norm=None)


def test_a_layer_takes_per_tensor_kv_scales():
    from tokenspeed_kernel.ops.attention.prologue import FP8TensorScales

    attn = _attention()
    attn.load_kv_cache_scale("layers.0.mixer.k_proj.k_scale", torch.tensor([0.5]))
    assert attn.fp8_kv_scales == FP8TensorScales(0.5, 1.0)
    attn.load_kv_cache_scale("layers.0.mixer.v_proj.v_scale", torch.tensor(0.25))
    assert attn.fp8_kv_scales == FP8TensorScales(0.5, 0.25)
    with pytest.raises(NotImplementedError, match="per-tensor"):
        attn.load_kv_cache_scale("layers.0.mixer.k_proj.k_scale", torch.ones(2))


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("leaf", ["mha", "qsa", "trtllm"])
@pytest.mark.parametrize("cache_dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_an_fp8_cache_is_written_at_the_scales_only_for_a_leaf_reading_them(
    hybrid, leaf, cache_dtype
):
    from tokenspeed_kernel.ops.attention.prologue import FP8TensorScales, HeadKVCache

    from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
        HybridLinearAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.mha import (
        MHAAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.qsa import (
        QSAAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.router import (
        CacheGroupRouter,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.trtllm import (
        TRTLLMMHAAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.state.mamba import (
        MambaAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        FULL_ATTENTION,
    )

    leaf_cls = {
        "mha": MHAAttnBackend,
        "qsa": QSAAttnBackend,
        "trtllm": TRTLLMMHAAttnBackend,
    }[leaf]
    router = CacheGroupRouter(
        lambda group, page_size: None,
        is_draft=False,
        spec_num_tokens=1,
        device="cpu",
        consumed_group_ids=(FULL_ATTENTION,),
    )
    router.leaves = {FULL_ATTENTION: leaf_cls.__new__(leaf_cls)}
    backend = (
        HybridLinearAttnBackend(router, MambaAttnBackend.__new__(MambaAttnBackend), [0])
        if hybrid
        else router
    )
    attn = _attention()
    attn.load_kv_cache_scale("a.k_scale", torch.tensor(0.5))
    attn.bind_cache_group(FULL_ATTENTION)
    rows = torch.zeros(8, 2, 16, dtype=cache_dtype)
    # The forward context and the pool's write target, as a forward hands them.
    ctx = SimpleNamespace(
        attn_backend=backend,
        token_to_kv_pool=SimpleNamespace(
            kv_write_target=lambda layer_id, slots, owned: HeadKVCache(
                rows, rows, None, slots
            )
        ),
    )
    if cache_dtype is not torch.float8_e4m3fn:
        # Another dtype holds the rows unscaled, so any leaf reads them.
        assert attn._local_target(torch.arange(3), ctx).scales is None
    elif leaf == "mha":
        target = attn._local_target(torch.arange(3), ctx)
        assert target.scales == FP8TensorScales(0.5, 1.0)
    else:
        with pytest.raises(NotImplementedError, match="does not read"):
            attn._local_target(torch.arange(3), ctx)


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
    # The scale never reaches the model.
    assert loaded == ["layers.0.mlp.weight", "post_load"]
