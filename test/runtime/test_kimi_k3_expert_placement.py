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

"""Kimi-K3 expert placement: 896 routed experts on 900 slots at EP20."""

import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig  # noqa: E402
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.layers.moe.topk import (  # noqa: E402
    ExpertLocationDispatchInfo,
    TopK,
    TopKOutputFormat,
)
from tokenspeed.runtime.layers.moe.utils import (  # noqa: E402
    All2AllBackend,
    MoeBackend,
)
from tokenspeed.runtime.models import kimi_k3  # noqa: E402
from tokenspeed.runtime.models.kimi_k3 import KimiLinearMoE  # noqa: E402
from tokenspeed.runtime.moe.expert_location import (  # noqa: E402
    ExpertLocationMetadata,
    set_global_expert_location_metadata,
)

_EXPERTS, _SLOTS, _EP, _LAYERS = 896, 900, 20, 4


def _placement(ep_rank: int, device: str) -> ExpertLocationMetadata:
    # Trivial placement, four GPUs per node: experts 0..3 sit on rank 0 (node
    # 0) and again in slots 896..899 on rank 19 (node 4).
    physical_to_logical = (torch.arange(_SLOTS) % _EXPERTS).repeat(_LAYERS, 1)
    return ExpertLocationMetadata.from_physical_to_logical_map(
        physical_to_logical.to(device),
        _EXPERTS,
        ep_size=_EP,
        ep_rank=ep_rank,
        ep_rank_nodes=tuple(rank // 4 for rank in range(_EP)),
    )


@pytest.mark.parametrize(
    "moe_backend,all2all_backend",
    [
        (MoeBackend.MEGA_MOE, All2AllBackend.NONE),
        (MoeBackend.FLASHINFER_TRTLLM, All2AllBackend.FLASHINFER),
        (MoeBackend.FLASHINFER_TRTLLM, All2AllBackend.AGRS),
    ],
)
def test_k3_moe_builds_physical_slots_and_a_static_router_map(
    monkeypatch, moe_backend, all2all_backend
):
    class Experts(nn.Module):
        """Stands in for the routed experts, whose setup needs the EP group."""

        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.supports_precomputed_topk = True
            self.topk_output_format = TopKOutputFormat.STANDARD
            self.plan = {"a2a_backend": "none"}
            self.topk_weights_dtype = torch.float32
            self.sorted_route_rows: frozenset[int] = frozenset()

    for key, value in [
        ("ep_dispatch_algorithm", "static"),
        ("max_prefill_tokens", 8192),
        ("max_num_seqs", 128),
    ]:
        monkeypatch.setitem(kimi_k3.global_server_args_dict, key, value)
    monkeypatch.setattr(kimi_k3, "MoELayer", Experts)
    monkeypatch.setattr(kimi_k3, "get_moe_backend", lambda: moe_backend)
    monkeypatch.setattr(kimi_k3, "get_all2all_backend", lambda: all2all_backend)
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.process_group_manager."
        "process_group_manager.get_device_process_group",
        lambda ranks: object(),
    )
    transports = []
    monkeypatch.setattr(
        kimi_k3,
        "get_flashinfer_moe_alltoall",
        lambda **kwargs: transports.append(kwargs) or object(),
    )
    set_global_expert_location_metadata(_placement(19, "cpu"))
    try:
        layer = KimiLinearMoE(
            config=KimiLinearConfig(
                hidden_size=64,
                routed_expert_hidden_size=32,
                moe_intermediate_size=32,
                num_experts=_EXPERTS,
                num_experts_per_token=16,
                num_shared_experts=1,
            ),
            mapping=Mapping(
                rank=19,
                world_size=_EP,
                attn_dp_size=_EP,
                moe_ep_size=_EP,
                nprocs_per_node=4,
            ),
            layer_index=3,
            model_scope="test",
            moe_block_count=1,
            quant_config=None,
            prefix="moe",
            alt_stream=None,
        )
    finally:
        set_global_expert_location_metadata(None)
    assert layer.experts.kwargs["num_experts"] == _SLOTS
    assert layer.gate.weight.shape[0] == _EXPERTS
    # The all-to-all sends each route to the rank owning its slot.
    assert [t["num_experts"] for t in transports] == (
        [_SLOTS] if all2all_backend is All2AllBackend.FLASHINFER else []
    )
    # Rank 19 keeps experts 0..3 on its own replica slots.
    dispatch = layer.expert_dispatch_info.partial_logical_to_rank_dispatch_physical_map
    assert dispatch[:5].tolist() == [896, 897, 898, 899, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="K3 routing runs on CUDA")
@pytest.mark.parametrize("tokens", [3, 300])  # fused mapped kernel; gather after
@pytest.mark.parametrize("ep_rank", [0, 19])
def test_k3_router_selects_physical_slot_ids(tokens: int, ep_rank: int):
    placement = _placement(ep_rank, "cuda")
    info = ExpertLocationDispatchInfo.init_new(
        layer_id=3,
        ep_dispatch_algorithm="static",
        expert_location_metadata=placement,
        all_to_all_ep=True,
    )
    # Experts 0..3, the replicated ones, win every token.
    bias = torch.zeros(_EXPERTS, device="cuda")
    bias[:4] = 10.0
    topk = TopK(
        top_k=16,
        layer_id=3,
        renormalize=True,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        correction_bias=bias,
        routed_scaling_factor=2.5,
        output_format=TopKOutputFormat.STANDARD,
        topk_weights_dtype=torch.bfloat16,
    )
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn(tokens, _EXPERTS, device="cuda", generator=generator)
    hidden = torch.empty(tokens, 8, device="cuda")

    logical = topk(hidden, logits, expert_location_dispatch_info=None)
    physical = topk(hidden, logits, expert_location_dispatch_info=info)
    logical_ids = logical.topk_ids.long()
    physical_ids = physical.topk_ids.long()

    assert torch.equal(physical.topk_weights, logical.topk_weights)
    # Every slot holds the expert the router chose.
    assert torch.equal(placement.physical_to_logical_map[3][physical_ids], logical_ids)
    # Rank 0 keeps its own slots 0..3; rank 19 its replicas 896..899.
    replicas = physical_ids[logical_ids < 4] - logical_ids[logical_ids < 4]
    assert (replicas == (0 if ep_rank == 0 else _EXPERTS)).all()
