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
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.configs.deepseek_v4_config import DeepseekV4Config
from tokenspeed.runtime.configs.model_config import (
    is_audio_model,
    is_multimodal_model,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v4 import (
    DeepseekV4AttentionBackend,
)
from tokenspeed.runtime.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
    DeepseekV4MoEGate,
)

CHECKPOINT_FIXTURE = (
    Path(__file__).parents[1] / "fixtures" / "deepseek_v4_vision" / "checkpoint.json"
)
MODEL_INDEX = (
    Path(os.environ["DSV4_VISION_MODEL_PATH"]) / "model.safetensors.index.json"
    if "DSV4_VISION_MODEL_PATH" in os.environ
    else None
)


def test_config_detection_activates_only_vision_enabled_v4():
    text = DeepseekV4Config(architectures=["DeepseekV4ForCausalLM"], vision_n_layers=0)
    vision = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"], vision_n_layers=32
    )
    assert not is_multimodal_model(text.architectures, text)
    assert is_multimodal_model(vision.architectures, vision)
    assert not is_audio_model(vision.architectures, vision)


def test_existing_multimodal_and_audio_detection_remains_config_optional():
    assert is_multimodal_model(["KimiK3ForConditionalGeneration"])
    assert is_audio_model(["Qwen3ASRForConditionalGeneration"])
    assert not is_multimodal_model(["DeepseekV4ForCausalLM"])


def test_prefill_graph_capability_defaults_safe_and_disables_all_request_v4():
    ctx = SimpleNamespace()
    multimodal_context = SimpleNamespace()

    safe = PrefillGraph.__new__(PrefillGraph)
    safe._multimodal_graph_safe = True
    safe._multimodal_input_embeds = lambda *_: None
    safe._replay_bucket = lambda _: 64
    assert safe.can_run(ctx, None)
    assert safe.can_run(ctx, multimodal_context)

    unsafe = PrefillGraph.__new__(PrefillGraph)
    unsafe._multimodal_graph_safe = False
    unsafe._multimodal_input_embeds = lambda *_: None
    unsafe._replay_bucket = lambda _: 64
    assert unsafe.can_run(ctx, None)
    assert not unsafe.can_run(ctx, multimodal_context)


def test_forward_context_and_attention_backends_publish_real_vision_seams():
    assert "dsv4_vision" in ForwardContext.__dataclass_fields__
    assert ForwardContext.__dataclass_fields__["dsv4_vision"].default is None
    assert (
        "vision"
        in inspect.signature(
            DeepseekV4AttentionBackend.forward_deepseek_v4_prefill
        ).parameters
    )
    assert (
        "vision"
        in inspect.signature(
            DeepseekV4AttentionBackend.forward_deepseek_v4_mixed
        ).parameters
    )
    assert "forward" not in DeepseekV4ForCausalLM.__dict__
    assert "prepare_model_kwargs" in DeepseekV4ForCausalLM.__dict__


def _fake_multimodal_context(start=2, end=4, prefix=0, length=6):
    item = SimpleNamespace(
        modality=SimpleNamespace(name="IMAGE"), offsets=[(start, end)]
    )
    return SimpleNamespace(
        mm_inputs=[SimpleNamespace(mm_items=[item])],
        extend_prefix_lens=[prefix],
        extend_seq_lens=[length],
    )


def _fake_inner_model():
    return SimpleNamespace(
        config=SimpleNamespace(vision_max_n_token=384),
        hc_mult=2,
        pp_start_layer=0,
        pp_end_layer=0,
        layers=[],
        dspark_layers_to_capture=(),
        mapping=SimpleNamespace(is_last_pp_rank=False),
    )


def _fake_forward_context():
    return SimpleNamespace(
        dsv4_vision=object(),
        capture_hidden_mode=None,
    )


@pytest.mark.parametrize("vision_first", [True, False])
def test_inner_forward_builds_and_clears_payload_in_both_orders(vision_first):
    model = _fake_inner_model()
    ctx = _fake_forward_context()
    input_ids = torch.tensor([900000, 900001, 900002, 7, 8, 9])
    input_embeds = torch.zeros(6, 4)
    vision_context = _fake_multimodal_context()

    def run(multimodal_context):
        DeepseekV4Model.forward(
            model,
            input_ids,
            torch.arange(6),
            ctx,
            input_embeds=input_embeds,
            dsv4_multimodal_context=multimodal_context,
        )
        return ctx.dsv4_vision

    ordered = [vision_context, None] if vision_first else [None, vision_context]
    results = [run(value) for value in ordered]
    if vision_first:
        assert results[0] is not None and results[0].intersects_span
        assert results[1] is None
    else:
        assert results[0] is None
        assert results[1] is not None and results[1].intersects_span


def test_prepare_model_kwargs_is_unchanged_when_vision_is_off_or_absent():
    model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
    nn.Module.__init__(model)
    model.is_multimodal_active = False
    input_embeds = torch.randn(2, 4)
    kwargs = {
        "input_embeds": input_embeds,
        "multimodal_context": _fake_multimodal_context(),
    }
    result = model.prepare_model_kwargs(SimpleNamespace(), torch.tensor([1, 2]), kwargs)
    assert set(result) == {"input_embeds"}
    assert result["input_embeds"] is input_embeds

    model.is_multimodal_active = True
    result = model.prepare_model_kwargs(
        SimpleNamespace(), torch.tensor([1, 2]), {"input_embeds": input_embeds}
    )
    assert set(result) == {"input_embeds"}
    assert result["input_embeds"] is input_embeds


def test_bias_vl_is_the_only_vision_gate_parameter_addition():
    common = {
        "n_routed_experts": 4,
        "hidden_size": 8,
        "num_hash_layers": 0,
        "num_hidden_layers": 43,
        "topk_method": "noaux_tc",
    }
    text_gate = DeepseekV4MoEGate(
        SimpleNamespace(**common, vision_n_layers=0), layer_index=2
    )
    vision_gate = DeepseekV4MoEGate(
        SimpleNamespace(**common, vision_n_layers=1), layer_index=2
    )
    assert set(text_gate.state_dict()) == {
        "weight",
        "e_score_correction_bias",
    }
    assert set(vision_gate.state_dict()) == {
        "weight",
        "e_score_correction_bias",
        "bias_vl",
    }
    assert vision_gate.bias_vl.dtype == torch.float32

    hash_config = dict(common)
    hash_config.update(
        vision_n_layers=1,
        num_hash_layers=1,
        vocab_size=16,
        num_experts_per_tok=2,
    )
    hash_gate = DeepseekV4MoEGate(
        SimpleNamespace(**hash_config),
        layer_index=0,
    )
    assert set(hash_gate.state_dict()) == {"weight", "tid2eid", "bias_vl"}


def test_exact_bias_mapping_never_creates_correction_bias_vl():
    model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
    assert (
        model._map_weight_name("layers.3.ffn.gate.bias")
        == "model.layers.3.ffn.gate.e_score_correction_bias"
    )
    assert (
        model._map_weight_name("layers.3.ffn.gate.bias_vl")
        == "model.layers.3.ffn.gate.bias_vl"
    )
    assert "e_score_correction_bias_vl" not in model._map_weight_name(
        "layers.3.ffn.gate.bias_vl"
    )


def test_manifest_diff_and_w1_collision_cutout_are_exact():
    if MODEL_INDEX is None or not MODEL_INDEX.exists():
        pytest.skip(f"checkpoint index not available: {MODEL_INDEX}")
    names = set(json.loads(MODEL_INDEX.read_text())["weight_map"])
    model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)

    bias_candidates = {name for name in names if ".ffn.gate.bias" in name}
    changed = set()
    for name in bias_candidates:
        old = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
        new = model._map_weight_name(name)
        if name.startswith("layers."):
            old = "model." + old
        if old != new:
            changed.add(name)
    assert len(changed) == 46
    assert all(name.endswith(".ffn.gate.bias_vl") for name in changed)

    checkpoint_fixture = json.loads(CHECKPOINT_FIXTURE.read_text())
    collisions = set(checkpoint_fixture["vision_w1_collision_names"])
    assert len(collisions) == 34
    assert all(model._is_vision_weight_name(name) for name in collisions)
    shared_w1 = {
        name
        for name in names
        if name.startswith("layers.")
        and ".shared_experts.w1" in name
        and ".experts." not in name
    }
    assert len(shared_w1) == 86
    assert not any(model._is_vision_weight_name(name) for name in shared_w1)


def test_exact_vision_loader_rejects_shape_dtype_and_renamed_tensors():
    parameter = nn.Parameter(torch.empty(2, 3, dtype=torch.bfloat16))
    assert (
        DeepseekV4ForCausalLM._load_exact_parameter(
            "vision.test", parameter, torch.ones(2, 3, dtype=torch.bfloat16)
        )
        == 12
    )
    with pytest.raises(ValueError, match="shape"):
        DeepseekV4ForCausalLM._load_exact_parameter(
            "vision.test", parameter, torch.ones(3, 2, dtype=torch.bfloat16)
        )
    with pytest.raises(ValueError, match="dtype"):
        DeepseekV4ForCausalLM._load_exact_parameter(
            "vision.test", parameter, torch.ones(2, 3, dtype=torch.float32)
        )

    renamed = "vision.blocks.0.mlp.gate_proj.weight"
    assert DeepseekV4ForCausalLM._is_vision_weight_name(renamed)
    assert renamed != "vision.blocks.0.mlp.w1.weight"
