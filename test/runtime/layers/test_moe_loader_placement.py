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

"""MoE checkpoint loading under an expert placement: every slot gets its expert."""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.moe.loader import (
    _build_placed_expert_plan,
    _select_local_experts,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema
from tokenspeed.runtime.moe.expert_location import ExpertLocationMetadata

_SCHEMA = ExpertCheckpointSchema(
    gate_proj_name="gate_proj", up_proj_name="up_proj", down_proj_name="down_proj"
)


def _placement(physical_to_logical, num_logical, ep_size, ep_rank):
    return ExpertLocationMetadata.from_physical_to_logical_map(
        torch.tensor(physical_to_logical),
        num_logical,
        ep_size=ep_size,
        ep_rank=ep_rank,
        ep_rank_nodes=(0,) * ep_size,
    )


class _Slots(torch.nn.Module):
    """Per-slot parameters whose loader records (shard, slot) <- tensor."""

    def __init__(self, num_local: int, hidden: int = 4):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(
            torch.zeros(num_local, 2 * hidden, hidden), requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(
            torch.zeros(num_local, hidden, hidden), requires_grad=False
        )
        self.writes: list[tuple[str, int]] = []

        def loader(param, loaded_weight, *, shard_id, local_expert_id):
            self.writes.append((shard_id, local_expert_id))
            if shard_id == "w1":
                param.data[local_expert_id, :hidden] = loaded_weight
            elif shard_id == "w3":
                param.data[local_expert_id, hidden:] = loaded_weight
            else:
                param.data[local_expert_id] = loaded_weight

        self.w13_weight.weight_loader = loader
        self.w2_weight.weight_loader = loader


def _params(layers: dict[int, _Slots]) -> dict:
    return {
        f"model.layers.{layer_id}.mlp.experts.{name}": param
        for layer_id, module in layers.items()
        for name, param in module.named_parameters()
    }


def test_a_replicated_expert_fills_every_local_slot_and_remote_ones_are_skipped():
    # Layer 0: rank 1 holds experts 3, 0, 2; layer 1: rank 1 holds 0, 1, 1.
    placement = _placement([[0, 1, 2, 3, 0, 2], [3, 2, 1, 0, 1, 1]], 4, 2, ep_rank=1)
    layers = {0: _Slots(3), 1: _Slots(3)}
    loader = build_moe_checkpoint_loader(
        params_dict=_params(layers),
        expert_schema=_SCHEMA,
        num_experts=6,
        ep_rank=1,
        ep_size=2,
        expert_placement=placement,
    )

    # Layer 1, expert 1 sits in two local slots: both receive the tensor.
    down = torch.full((4, 4), 7.0)
    assert loader.matches("model.layers.1.mlp.experts.1.down_proj.weight")
    assert loader.load("model.layers.1.mlp.experts.1.down_proj.weight", down) == (
        "model.layers.1.mlp.experts.w2_weight"
    )
    assert layers[1].writes == [("w2", 1), ("w2", 2)]
    torch.testing.assert_close(layers[1].w2_weight[1], down)
    torch.testing.assert_close(layers[1].w2_weight[2], down)
    assert not layers[1].w2_weight[0].any()

    # The same logical expert maps to a different slot in another layer.
    loader.load("model.layers.0.mlp.experts.0.gate_proj.weight", torch.ones(4, 4))
    assert layers[0].writes == [("w1", 1)]

    # Experts this rank does not hold in that layer are not this loader's.
    assert not loader.matches("model.layers.1.mlp.experts.2.down_proj.weight")
    assert not loader.matches("model.layers.0.mlp.experts.1.up_proj.weight")
    assert loader.is_expert_checkpoint_weight(
        "model.layers.0.mlp.experts.1.up_proj.weight"
    )
    # The placement is per layer: a layer it does not know is nobody's.
    assert not loader.matches("model.layers.2.mlp.experts.0.gate_proj.weight")


def test_fused_checkpoint_tensors_gather_each_slots_expert():
    placement = _placement([[0, 1, 2, 3, 0, 2], [3, 2, 1, 0, 1, 1]], 4, 2, ep_rank=1)
    layers = {1: _Slots(3)}
    loader = build_moe_checkpoint_loader(
        params_dict=_params(layers),
        fused_schema=ExpertCheckpointSchema(
            gate_up_fused_name="gate_up_proj", down_proj_name="down_proj"
        ),
        num_experts=6,
        ep_rank=1,
        ep_size=2,
        expert_placement=placement,
    )
    # [logical experts, ...]: expert e's down projection is all e's.
    down = torch.arange(4, dtype=torch.float32).view(4, 1, 1).expand(4, 4, 4)
    loader.load("model.layers.1.mlp.experts.down_proj", down)
    assert layers[1].writes == [("w2", 0), ("w2", 1), ("w2", 2)]
    # Slots of rank 1 in layer 1 hold experts 0, 1, 1.
    assert layers[1].w2_weight[:, 0, 0].tolist() == [0.0, 1.0, 1.0]


def test_placement_plan_for_896_slots_over_128_ranks_has_7_slots_per_rank():
    physical_to_logical = (torch.arange(896) % 768).view(1, 896)
    nodes = tuple(r // 8 for r in range(128))
    placement = ExpertLocationMetadata.from_physical_to_logical_map(
        physical_to_logical, 768, ep_size=128, ep_rank=5, ep_rank_nodes=nodes
    )
    assert placement.num_local_physical_experts == 7
    plan = _build_placed_expert_plan(_SCHEMA, expert_placement=placement, ep_rank=5)
    assert len(plan) == 7 * 3
    assert [entry.local_expert_id for entry in plan[::3]] == list(range(7))
    assert plan[0].checkpoint_weight_name == "experts.35.gate_proj."
    assert plan[-1].checkpoint_weight_name == "experts.41.down_proj."
    # The last rank's slots wrap onto replicas of experts 121..127.
    last = _build_placed_expert_plan(
        _SCHEMA,
        expert_placement=ExpertLocationMetadata.from_physical_to_logical_map(
            physical_to_logical, 768, ep_size=128, ep_rank=127, ep_rank_nodes=nodes
        ),
        ep_rank=127,
    )
    assert last[0].checkpoint_weight_name == "experts.121.gate_proj."


def test_loader_geometry_must_match_the_placement():
    placement = _placement([[0, 1, 2, 3, 0, 2]], 4, 2, ep_rank=0)
    with pytest.raises(ValueError, match="physical experts"):
        build_moe_checkpoint_loader(
            params_dict={},
            expert_schema=_SCHEMA,
            num_experts=4,
            ep_rank=0,
            ep_size=2,
            expert_placement=placement,
        )
    with pytest.raises(ValueError, match="ep_size"):
        build_moe_checkpoint_loader(
            params_dict={},
            expert_schema=_SCHEMA,
            num_experts=6,
            ep_rank=0,
            ep_size=3,
            expert_placement=placement,
        )


def test_without_a_placement_the_plan_is_the_contiguous_one():
    layers = {0: _Slots(2)}
    loader = build_moe_checkpoint_loader(
        params_dict=_params(layers),
        expert_schema=_SCHEMA,
        num_experts=4,
        ep_rank=1,
        ep_size=2,
    )
    loader.load("model.layers.0.mlp.experts.3.down_proj.weight", torch.ones(4, 4))
    assert layers[0].writes == [("w2", 1)]
    assert not loader.matches("model.layers.0.mlp.experts.0.down_proj.weight")


def test_fused_local_experts_slice_a_contiguous_range_and_gather_otherwise():
    stacked = torch.arange(6).view(6, 1)
    contiguous = _select_local_experts(stacked, [2, 3, 4])
    assert contiguous.tolist() == [[2], [3], [4]]
    assert contiguous.data_ptr() == stacked[2].data_ptr()  # a view, no copy
    gathered = _select_local_experts(stacked, [0, 1, 1])
    assert gathered.tolist() == [[0], [1], [1]]
    assert gathered.data_ptr() != stacked.data_ptr()
    assert _select_local_experts(stacked, [4, 3]).tolist() == [[4], [3]]
    assert _select_local_experts(stacked, []).shape[0] == 0
