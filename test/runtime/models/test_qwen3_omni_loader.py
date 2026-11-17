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
"""Qwen3-Omni renames checkpoint names for the shared ``BaseCausalLM`` loader."""

import os
import sys

import torch
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.layers.quantization.base_config import (  # noqa: E402
    QuantizationConfig,
)
from tokenspeed.runtime.models.qwen3_omni import (  # noqa: E402
    Qwen3OmniMoeForConditionalGeneration,
)

HIDDEN = 4
NUM_EXPERTS = 2


def _shard_loader(param, loaded_weight, shard_id):
    rows = {"q": 0, "k": 1, "v": 2}[shard_id]
    param.data[rows * HIDDEN : (rows + 1) * HIDDEN] = loaded_weight


def _expert_loader(param, loaded_weight, *, shard_id, local_expert_id):
    rows = {
        "w1": slice(0, HIDDEN),
        "w3": slice(HIDDEN, 2 * HIDDEN),
        "w2": slice(0, HIDDEN),
    }[shard_id]
    param.data[local_expert_id, rows] = loaded_weight


def _linear(rows: int, stacked: bool) -> nn.Module:
    module = nn.Module()
    module.weight = nn.Parameter(
        torch.zeros(rows * HIDDEN, HIDDEN), requires_grad=False
    )
    if stacked:
        module.weight.weight_loader = _shard_loader
    return module


def _attention(qkv_name: str, stacked: bool) -> nn.Module:
    attention = nn.Module()
    attention.add_module(qkv_name, _linear(3, stacked=stacked))
    return attention


class _TextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = nn.Module()
        layer.self_attn = _attention("qkv_proj", stacked=True)
        layer.mlp = nn.Module()
        layer.mlp.experts = nn.Module()
        for name, rows in (("w13_weight", 2 * HIDDEN), ("w2_weight", HIDDEN)):
            param = nn.Parameter(
                torch.zeros(NUM_EXPERTS, rows, HIDDEN), requires_grad=False
            )
            param.weight_loader = _expert_loader
            layer.mlp.experts.register_parameter(name, param)
        self.layers = nn.ModuleList([layer])


class _TinyQwen3Omni(Qwen3OmniMoeForConditionalGeneration):
    """The real loader over CPU stand-ins for the text, vision and audio models."""

    def __init__(self) -> None:
        super().__init__(
            config=Qwen3OmniMoeConfig(
                thinker_config={
                    "text_config": {
                        "hidden_size": HIDDEN,
                        "vocab_size": 8,
                        "num_experts": NUM_EXPERTS,
                    }
                }
            ),
            mapping=Mapping(rank=0, world_size=1),
            is_multimodal_active=False,
        )
        self.is_multimodal_active = True
        block = nn.Module()
        block.attn = _attention("qkv_proj", stacked=False)
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList([block])
        audio_layer = nn.Module()
        audio_layer.self_attn = _attention("qkv_proj", stacked=True)
        audio_layer.self_attn.proj = _linear(1, stacked=False)
        self.audio_tower = nn.Module()
        self.audio_tower.layers = nn.ModuleList([audio_layer])

    def resolve_model(
        self,
        config: Qwen3OmniMoeConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> nn.Module:
        return _TextModel()


def _value(seed: int) -> torch.Tensor:
    return torch.full((HIDDEN, HIDDEN), float(seed))


def test_thinker_weights_reach_their_parameters():
    model = _TinyQwen3Omni()
    text = "thinker.model.layers.0"
    audio = "thinker.audio_tower.layers.0.self_attn"
    stream = [
        (f"{text}.self_attn.q_proj.weight", _value(1)),
        (f"{text}.self_attn.k_proj.weight", _value(2)),
        (f"{text}.self_attn.v_proj.weight", _value(3)),
        (
            "thinker.visual.blocks.0.attn.qkv.weight",
            torch.full((3 * HIDDEN, HIDDEN), 4.0),
        ),
        (f"{audio}.q_proj.weight", _value(5)),
        (f"{audio}.k_proj.weight", _value(6)),
        (f"{audio}.v_proj.weight", _value(7)),
        (f"{audio}.out_proj.weight", _value(8)),
        ("talker.model.layers.0.self_attn.q_proj.weight", _value(9)),
    ]
    for expert in range(NUM_EXPERTS):
        for proj, offset in (("gate_proj", 10), ("up_proj", 20), ("down_proj", 30)):
            stream.append(
                (f"{text}.mlp.experts.{expert}.{proj}.weight", _value(offset + expert))
            )

    loaded = model.load_weights(stream)

    layer = model.model.layers[0]
    audio_attn = model.audio_tower.layers[0].self_attn
    for qkv, seeds in (
        (layer.self_attn.qkv_proj.weight, (1, 2, 3)),
        (audio_attn.qkv_proj.weight, (5, 6, 7)),
    ):
        for rows, seed in enumerate(seeds):
            assert torch.equal(qkv[rows * HIDDEN : (rows + 1) * HIDDEN], _value(seed))
    # The vision tower's fused qkv is one checkpoint tensor, not a v_proj shard.
    vision_qkv = model.visual.blocks[0].attn.qkv_proj.weight
    assert torch.equal(vision_qkv, torch.full((3 * HIDDEN, HIDDEN), 4.0))
    assert torch.equal(audio_attn.proj.weight, _value(8))
    w13, w2 = layer.mlp.experts.w13_weight, layer.mlp.experts.w2_weight
    for expert in range(NUM_EXPERTS):
        assert torch.equal(w13[expert, :HIDDEN], _value(10 + expert))
        assert torch.equal(w13[expert, HIDDEN:], _value(20 + expert))
        assert torch.equal(w2[expert], _value(30 + expert))
    assert loaded == {
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w2_weight",
        "visual.blocks.0.attn.qkv_proj.weight",
        "audio_tower.layers.0.self_attn.qkv_proj.weight",
        "audio_tower.layers.0.self_attn.proj.weight",
    }
