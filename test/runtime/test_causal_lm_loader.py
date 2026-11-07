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

"""``BaseCausalLM.load_weights`` stacks projections and loads local experts.

It skips other ranks' experts and returns the parameters that received data.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn
from transformers import PretrainedConfig

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.layers.moe import (  # noqa: E402
    ExpertCheckpointSchema,
    MoECheckpointLoader,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.models.base import BaseCausalLM  # noqa: E402

HIDDEN = 4
NUM_EXPERTS, EP_SIZE, EP_RANK = 4, 2, 1


def _shard_loader(param, loaded_weight, shard_id):
    rows = {"q": 0, "k": 1, "v": 2, 0: 0, 1: 1}[shard_id]
    param.data[rows * HIDDEN : (rows + 1) * HIDDEN] = loaded_weight


def _expert_loader(param, loaded_weight, *, shard_id, local_expert_id):
    rows = {
        "w1": slice(0, HIDDEN),
        "w3": slice(HIDDEN, 2 * HIDDEN),
        "w2": slice(0, HIDDEN),
    }[shard_id]
    param.data[local_expert_id, rows] = loaded_weight


def _stacked(rows: int) -> nn.Module:
    module = nn.Module()
    module.weight = nn.Parameter(
        torch.zeros(rows * HIDDEN, HIDDEN), requires_grad=False
    )
    module.weight.weight_loader = _shard_loader
    return module


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = _stacked(3)
        self.input_layernorm = nn.Module()
        self.input_layernorm.weight = nn.Parameter(
            torch.zeros(HIDDEN), requires_grad=False
        )
        self.mlp = nn.Module()
        self.mlp.experts = nn.Module()
        num_local = NUM_EXPERTS // EP_SIZE
        self.mlp.experts.w13_weight = nn.Parameter(
            torch.zeros(num_local, 2 * HIDDEN, HIDDEN), requires_grad=False
        )
        self.mlp.experts.w2_weight = nn.Parameter(
            torch.zeros(num_local, HIDDEN, HIDDEN), requires_grad=False
        )
        self.mlp.experts.w13_weight.weight_loader = _expert_loader
        self.mlp.experts.w2_weight.weight_loader = _expert_loader


class _TinyMoeLM(BaseCausalLM):
    def __init__(self) -> None:
        super().__init__(
            config=PretrainedConfig(),
            mapping=Mapping(rank=0, world_size=1),
            encoder_only=True,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer()])

    def get_stacked_params_mapping(self) -> list[tuple[str, str, int | str]]:
        return [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

    def get_expert_checkpoint_loader(self) -> MoECheckpointLoader:
        return build_moe_checkpoint_loader(
            params_dict=dict(self.named_parameters()),
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
            ),
            num_experts=NUM_EXPERTS,
            ep_rank=EP_RANK,
            ep_size=EP_SIZE,
        )


def _value(seed: int) -> torch.Tensor:
    return torch.full((HIDDEN, HIDDEN), float(seed))


def test_projections_stack_and_only_this_ranks_experts_load():
    model = _TinyMoeLM()
    prefix = "model.layers.0"
    stream = [
        (f"{prefix}.self_attn.q_proj.weight", _value(1)),
        (f"{prefix}.self_attn.k_proj.weight", _value(2)),
        (f"{prefix}.self_attn.v_proj.weight", _value(3)),
        (f"{prefix}.self_attn.rotary_emb.inv_freq", torch.ones(HIDDEN)),
        (f"{prefix}.input_layernorm.weight", torch.full((HIDDEN,), 4.0)),
    ]
    for expert in range(NUM_EXPERTS):
        for proj, offset in (("gate_proj", 10), ("up_proj", 20), ("down_proj", 30)):
            stream.append(
                (
                    f"{prefix}.mlp.experts.{expert}.{proj}.weight",
                    _value(offset + expert),
                )
            )

    loaded = model.load_weights(stream)

    layer = model.model.layers[0]
    qkv = layer.self_attn.qkv_proj.weight
    for rows, seed in enumerate((1, 2, 3)):
        assert torch.equal(qkv[rows * HIDDEN : (rows + 1) * HIDDEN], _value(seed))
    assert torch.equal(layer.input_layernorm.weight, torch.full((HIDDEN,), 4.0))
    # Rank 1 of 2 owns experts 2 and 3, in its local slots 0 and 1.
    w13, w2 = layer.mlp.experts.w13_weight, layer.mlp.experts.w2_weight
    for local, expert in enumerate((2, 3)):
        assert torch.equal(w13[local, :HIDDEN], _value(10 + expert))
        assert torch.equal(w13[local, HIDDEN:], _value(20 + expert))
        assert torch.equal(w2[local], _value(30 + expert))
    assert loaded == {
        f"{prefix}.self_attn.qkv_proj.weight",
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.mlp.experts.w13_weight",
        f"{prefix}.mlp.experts.w2_weight",
    }
