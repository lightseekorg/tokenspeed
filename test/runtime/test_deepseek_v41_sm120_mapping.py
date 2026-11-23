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

import pytest

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.models.deepseek_v41 import _validate_sm120_deepep_mapping


def _mapping(world: int, rank: int, qcp: bool) -> Mapping:
    return Mapping(
        rank=rank,
        world_size=world,
        attn_tp_size=world if qcp else 1,
        attn_dp_size=1 if qcp else world,
        attn_qcp_size=world if qcp else 1,
        attn_head_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=world,
        nprocs_per_node=min(world, 8),
        nnodes=max(world // 8, 1),
    )


@pytest.mark.parametrize("world", [4, 8, 16, 32])
def test_wire1_accepts_data_parallel_expert_groups(world):
    for rank in range(world):
        _validate_sm120_deepep_mapping(_mapping(world, rank, False), "nvfp4_fp8")


@pytest.mark.parametrize("qcp", [False, True])
def test_elastic_preserves_existing_prefill_topologies(qcp):
    _validate_sm120_deepep_mapping(_mapping(8, 0, qcp), "elastic_fp8_bf16")


@pytest.mark.parametrize("world", [4, 8, 16, 32])
def test_elastic_accepts_data_parallel_expert_groups(world):
    for rank in range(world):
        _validate_sm120_deepep_mapping(_mapping(world, rank, False), "elastic_fp8_bf16")


def test_unknown_wire_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        _validate_sm120_deepep_mapping(_mapping(8, 0, False), "invalid")


@pytest.mark.parametrize("size", [2, 64])
@pytest.mark.parametrize("wire", ["nvfp4_fp8", "elastic_fp8_bf16"])
def test_unvalidated_ep_size_is_rejected(size, wire):
    with pytest.raises(ValueError, match="EP in"):
        _validate_sm120_deepep_mapping(_mapping(size, 0, False), wire)


def test_head_tensor_parallel_is_rejected():
    mapping = Mapping(
        rank=0,
        world_size=32,
        attn_tp_size=2,
        attn_dp_size=16,
        attn_head_tp_size=2,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=32,
        nprocs_per_node=8,
        nnodes=4,
    )
    with pytest.raises(ValueError, match="TP1"):
        _validate_sm120_deepep_mapping(mapping, "nvfp4_fp8")


@pytest.mark.parametrize(
    "decode,max_rows,expected", [(False, 18, 32), (False, 2051, 2048), (True, 7, 7)]
)
def test_active_and_idle_ranks_use_identical_dispatch_capacities(
    monkeypatch, decode, max_rows, expected
):
    from types import SimpleNamespace

    import torch
    from tokenspeed_kernel.thirdparty.deepep_sm120_experiment import (
        buffer,
        deepep_flashinfer_fp8_opt,
    )

    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.models import deepseek_v41 as v41
    from tokenspeed.runtime.models.deepseek_v41_moe import DeepseekV4MoE
    from tokenspeed.runtime.utils.cuda_stream import StreamFork

    monkeypatch.setattr(buffer, "CAPACITY", 2048)
    capacities = []

    def apply(_buffer, x, _state, _ids, _weights, **kwargs):
        capacities.append(kwargs["num_max_tokens_per_rank"])
        return torch.zeros_like(x)

    def routes(self, x, ids):
        return SimpleNamespace(
            topk_ids=torch.zeros((x.shape[0], 1), dtype=torch.int32),
            topk_weights=torch.ones((x.shape[0], 1)),
            output_scale=1.0,
        )

    monkeypatch.setattr(deepep_flashinfer_fp8_opt, "apply_deepep_flashinfer", apply)
    monkeypatch.setattr(v41.DeepseekV41MoE, "_compute_topk_output", routes)
    monkeypatch.setattr(DeepseekV4MoE, "_compute_topk_output", routes)
    monkeypatch.setattr(DeepseekV4MoE, "_forward_shared_experts", lambda self, x: None)
    moe = v41.DeepseekV41MoE.__new__(v41.DeepseekV41MoE)
    torch.nn.Module.__init__(moe)
    moe.use_sm120_deepep = True
    moe.sm120_deepep_buffer = object()
    moe.sm120_shared_overlap = False
    moe.stream_fork = StreamFork(None)
    moe.experts = SimpleNamespace(_moe_backend_state=None, num_experts=384)
    traces = []
    for rows, mode in [
        (max_rows, ForwardMode.DECODE if decode else ForwardMode.EXTEND),
        (0, ForwardMode.IDLE),
    ]:
        capacities.clear()
        x = torch.empty((rows, 8), dtype=torch.bfloat16)
        moe(
            x,
            None,
            max_rows,
            max_rows,
            ctx=SimpleNamespace(forward_mode=mode, all_decode_or_idle=decode),
        )
        traces.append(list(capacities))
    assert traces[0] == traces[1]
    assert traces[0][0] == expected
    if max_rows > 2048:
        assert traces[0] == [2048, 4]
