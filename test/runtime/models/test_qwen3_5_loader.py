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

"""Qwen3.5-MoE renames checkpoint names for the shared ``BaseCausalLM`` loader."""

import os
import sys

import torch
from torch import nn

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.qwen3_5_config import (  # noqa: E402
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.layers.quantization.base_config import (  # noqa: E402
    QuantizationConfig,
)
from tokenspeed.runtime.models.qwen3_5 import (  # noqa: E402
    Qwen3_5MoeForConditionalGeneration,
)

HIDDEN = 4
NUM_EXPERTS = 2


def _shard_loader(param, loaded_weight, shard_id):
    if isinstance(shard_id, tuple):
        start, count = shard_id[0], len(shard_id)
    else:
        start = (
            {"q": 0, "k": 1, "v": 2}[shard_id]
            if isinstance(shard_id, str)
            else shard_id
        )
        count = 1
    param.data[start * HIDDEN : (start + count) * HIDDEN] = loaded_weight


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


class _TextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = nn.Module()
        layer.qkv_proj = _linear(3, stacked=True)
        layer.mlp = nn.Module()
        layer.mlp.shared_expert = nn.Module()
        layer.mlp.shared_expert.gate_up_proj = _linear(2, stacked=True)
        layer.mlp.experts = nn.Module()
        for name, rows in (("w13_weight", 2 * HIDDEN), ("w2_weight", HIDDEN)):
            param = nn.Parameter(
                torch.zeros(NUM_EXPERTS, rows, HIDDEN), requires_grad=False
            )
            param.weight_loader = _expert_loader
            layer.mlp.experts.register_parameter(name, param)
        self.layers = nn.ModuleList([layer])


class _TinyQwen3_5Moe(Qwen3_5MoeForConditionalGeneration):
    """The real loader over CPU stand-ins for the text and vision models."""

    def __init__(self) -> None:
        super().__init__(
            config=Qwen3_5MoeConfig(
                text_config={
                    "hidden_size": HIDDEN,
                    "vocab_size": 8,
                    "num_experts": NUM_EXPERTS,
                }
            ),
            mapping=Mapping(rank=0, world_size=1),
            is_multimodal_active=False,
        )
        self.is_multimodal_active = True
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.qkv_proj = _linear(3, stacked=False)
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList([block])

    def resolve_model(
        self,
        config: Qwen3_5MoeTextConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> nn.Module:
        return _TextModel()


def _value(seed: int) -> torch.Tensor:
    return torch.full((HIDDEN, HIDDEN), float(seed))


def test_text_and_vision_weights_reach_their_parameters():
    model = _TinyQwen3_5Moe()
    text = "model.language_model.layers.0"
    stream = [
        (f"{text}.self_attn.q_proj.weight", _value(1)),
        (f"{text}.self_attn.k_proj.weight", _value(2)),
        (f"{text}.self_attn.v_proj.weight", _value(3)),
        (f"{text}.mlp.shared_expert.gate_proj.weight", _value(4)),
        (f"{text}.mlp.shared_expert.up_proj.weight", _value(5)),
        (
            "model.visual.blocks.0.attn.qkv.weight",
            torch.full((3 * HIDDEN, HIDDEN), 6.0),
        ),
    ]
    for expert in range(NUM_EXPERTS):
        for proj, offset in (("gate_proj", 10), ("up_proj", 20), ("down_proj", 30)):
            stream.append(
                (f"{text}.mlp.experts.{expert}.{proj}.weight", _value(offset + expert))
            )

    loaded = model.load_weights(stream)

    layer = model.model.layers[0]
    qkv = layer.qkv_proj.weight
    for rows, seed in enumerate((1, 2, 3)):
        assert torch.equal(qkv[rows * HIDDEN : (rows + 1) * HIDDEN], _value(seed))
    gate_up = layer.mlp.shared_expert.gate_up_proj.weight
    assert torch.equal(gate_up[:HIDDEN], _value(4))
    assert torch.equal(gate_up[HIDDEN:], _value(5))
    # The vision tower's fused qkv is one checkpoint tensor, not a v_proj shard.
    vision_qkv = model.visual.blocks[0].attn.qkv_proj.weight
    assert torch.equal(vision_qkv, torch.full((3 * HIDDEN, HIDDEN), 6.0))
    w13, w2 = layer.mlp.experts.w13_weight, layer.mlp.experts.w2_weight
    for expert in range(NUM_EXPERTS):
        assert torch.equal(w13[expert, :HIDDEN], _value(10 + expert))
        assert torch.equal(w13[expert, HIDDEN:], _value(20 + expert))
        assert torch.equal(w2[expert], _value(30 + expert))
    assert loaded == {
        "model.layers.0.qkv_proj.weight",
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight",
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w2_weight",
        "visual.blocks.0.attn.qkv_proj.weight",
    }


def test_split_and_merged_gdn_layers_load_their_checkpoint_shards():
    model = _TinyQwen3_5Moe()
    split = nn.Module()
    split.in_proj_qkvz = _linear(4, stacked=True)
    split.in_proj_ba = _linear(2, stacked=True)
    model.model.layers[0].linear_attn = split
    merged = nn.Module()
    merged.in_proj_qkvzba = _linear(6, stacked=True)
    second_layer = nn.Module()
    second_layer.linear_attn = merged
    model.model.layers.append(second_layer)

    rows = [[_value(seed) for seed in range(start, start + 6)] for start in (1, 7)]
    stream = []
    for layer, (q, k, v, z, b, a) in enumerate(rows):
        prefix = f"model.language_model.layers.{layer}.linear_attn"
        stream.extend(
            (f"{prefix}.{name}.weight", weight)
            for name, weight in (
                ("in_proj_qkv", torch.cat((q, k, v))),
                ("in_proj_z", z),
                ("in_proj_b", b),
                ("in_proj_a", a),
            )
        )

    loaded = model.load_weights(stream)

    assert loaded == {
        "model.layers.0.linear_attn.in_proj_qkvz.weight",
        "model.layers.0.linear_attn.in_proj_ba.weight",
        "model.layers.1.linear_attn.in_proj_qkvzba.weight",
    }
    torch.testing.assert_close(split.in_proj_qkvz.weight, torch.cat(rows[0][:4]))
    torch.testing.assert_close(split.in_proj_ba.weight, torch.cat(rows[0][4:]))
    torch.testing.assert_close(merged.in_proj_qkvzba.weight, torch.cat(rows[1]))
