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

import torch

from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights.loaders import make_weight_loader
from tokenspeed.runtime.layers.moe.weights.nvfp4 import create_nvfp4_weight_pair

_TP_SIZE = 4
_BLOCK_SIZE = 128
_INTERMEDIATE_SIZE = 640
_PADDED_INTERMEDIATE_SIZE = 1024
_LOCAL_INTERMEDIATE_SIZE = _PADDED_INTERMEDIATE_SIZE // _TP_SIZE
_SCALE_BLOCKS = (_INTERMEDIATE_SIZE + _BLOCK_SIZE - 1) // _BLOCK_SIZE
_LOCAL_SCALE_BLOCKS = _LOCAL_INTERMEDIATE_SIZE // _BLOCK_SIZE


def _spec(tp_rank: int) -> MoELayerSpec:
    return MoELayerSpec(
        top_k=2,
        num_experts=1,
        num_local_experts=1,
        hidden_size=1,
        intermediate_size=_PADDED_INTERMEDIATE_SIZE,
        activation="silu",
        tp_rank=tp_rank,
        tp_size=_TP_SIZE,
        ep_rank=0,
        ep_size=1,
    )


def _expected_shard(source: torch.Tensor, rank: int, dim: int) -> torch.Tensor:
    shape = list(source.shape)
    shape[dim] = _LOCAL_INTERMEDIATE_SIZE
    expected = torch.zeros(shape, dtype=source.dtype)
    start = rank * _LOCAL_INTERMEDIATE_SIZE
    length = min(_LOCAL_INTERMEDIATE_SIZE, max(0, source.shape[dim] - start))
    if length:
        expected.narrow(dim, 0, length).copy_(source.narrow(dim, start, length))
    return expected


def _expected_scale_shard(source: torch.Tensor, rank: int, dim: int) -> torch.Tensor:
    shape = list(source.shape)
    shape[dim] = _LOCAL_SCALE_BLOCKS
    expected = torch.ones(shape, dtype=source.dtype)
    start = rank * _LOCAL_SCALE_BLOCKS
    length = min(_LOCAL_SCALE_BLOCKS, max(0, source.shape[dim] - start))
    if length:
        expected.narrow(dim, 0, length).copy_(source.narrow(dim, start, length))
    return expected


def test_global_padding_precedes_w13_weight_and_scale_tp_sharding() -> None:
    gate_weight = torch.arange(1, _INTERMEDIATE_SIZE + 1, dtype=torch.float32).reshape(
        -1, 1
    )
    up_weight = gate_weight + 1000
    gate_scale = (
        torch.arange(1, _SCALE_BLOCKS + 1, dtype=torch.float32).reshape(-1, 1) / 1000
    )
    up_scale = gate_scale + 0.1

    for rank in range(_TP_SIZE):
        loader = make_weight_loader(_spec(rank))
        weight = torch.nn.Parameter(
            torch.zeros(1, 2 * _LOCAL_INTERMEDIATE_SIZE, 1),
            requires_grad=False,
        )
        scale = torch.nn.Parameter(
            torch.ones(1, 2 * _LOCAL_SCALE_BLOCKS, 1),
            requires_grad=False,
        )

        loader(weight, gate_weight, "w1", local_expert_id=0)
        loader(weight, up_weight, "w3", local_expert_id=0)
        loader(scale, gate_scale, "w1", local_expert_id=0)
        loader(scale, up_scale, "w3", local_expert_id=0)

        torch.testing.assert_close(
            weight[0, :_LOCAL_INTERMEDIATE_SIZE],
            _expected_shard(gate_weight, rank, dim=0),
        )
        torch.testing.assert_close(
            weight[0, _LOCAL_INTERMEDIATE_SIZE:],
            _expected_shard(up_weight, rank, dim=0),
        )
        torch.testing.assert_close(
            scale[0, :_LOCAL_SCALE_BLOCKS],
            _expected_scale_shard(gate_scale, rank, dim=0),
        )
        torch.testing.assert_close(
            scale[0, _LOCAL_SCALE_BLOCKS:],
            _expected_scale_shard(up_scale, rank, dim=0),
        )


def test_global_padding_precedes_w2_weight_and_scale_tp_sharding() -> None:
    down_weight = torch.arange(1, _INTERMEDIATE_SIZE + 1, dtype=torch.float32).reshape(
        1, -1
    )
    down_scale = (
        torch.arange(1, _SCALE_BLOCKS + 1, dtype=torch.float32).reshape(1, -1) / 1000
    )

    for rank in range(_TP_SIZE):
        loader = make_weight_loader(_spec(rank))
        weight = torch.nn.Parameter(
            torch.zeros(1, 1, _LOCAL_INTERMEDIATE_SIZE),
            requires_grad=False,
        )
        scale = torch.nn.Parameter(
            torch.ones(1, 1, _LOCAL_SCALE_BLOCKS),
            requires_grad=False,
        )

        loader(weight, down_weight, "w2", local_expert_id=0)
        loader(scale, down_scale, "w2", local_expert_id=0)

        torch.testing.assert_close(weight[0], _expected_shard(down_weight, rank, dim=1))
        torch.testing.assert_close(
            scale[0], _expected_scale_shard(down_scale, rank, dim=1)
        )


def test_nvfp4_packed_weight_and_scale_use_the_same_padded_tp_shard() -> None:
    group_size = 16
    hidden_size = 16
    padded_intermediate_size = 768
    local_intermediate_size = padded_intermediate_size // _TP_SIZE
    spec = MoELayerSpec(
        top_k=2,
        num_experts=1,
        num_local_experts=1,
        hidden_size=hidden_size,
        intermediate_size=padded_intermediate_size,
        activation="silu",
        tp_rank=3,
        tp_size=_TP_SIZE,
        ep_rank=0,
        ep_size=1,
    )
    layer = torch.nn.Module()
    create_nvfp4_weight_pair(spec, layer, group_size=group_size)
    layer.w2_weight.weight_loader(
        layer.w2_weight,
        torch.ones(hidden_size, _INTERMEDIATE_SIZE // 2, dtype=torch.uint8),
        "w2",
        local_expert_id=0,
    )
    layer.w2_weight_scale.weight_loader(
        layer.w2_weight_scale,
        torch.full(
            (hidden_size, _INTERMEDIATE_SIZE // group_size),
            2,
            dtype=torch.float8_e4m3fn,
        ),
        shard_id="w2",
        local_expert_id=0,
    )

    real_values = _INTERMEDIATE_SIZE - 3 * local_intermediate_size
    assert torch.all(layer.w2_weight[..., : real_values // 2] == 1)
    assert torch.count_nonzero(layer.w2_weight[..., real_values // 2 :]) == 0
    assert torch.all(layer.w2_weight_scale[..., : real_values // group_size] == 2)
    assert (
        torch.count_nonzero(
            layer.w2_weight_scale[..., real_values // group_size :].float()
        )
        == 0
    )
