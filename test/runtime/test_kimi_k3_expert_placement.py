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
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.configs.kimi_k3_config import (  # noqa: E402
    KimiK3Config,
    KimiLinearConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.layers.moe.latent import Kimi3MoEExecutionPlan  # noqa: E402
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
from tokenspeed.runtime.models.deepseek_v3 import (  # noqa: E402
    Eagle3DeepseekV2ForCausalLM,
)
from tokenspeed.runtime.models.kimi_k3 import (  # noqa: E402
    KimiLinearMoE,
    _check_k3_expert_placement,
)
from tokenspeed.runtime.moe.expert_location import (  # noqa: E402
    ExpertLocationMetadata,
    build_expert_placement,
    compute_logical_to_rank_dispatch_physical_map,
    set_global_expert_location_metadata,
)

_EXPERTS, _SLOTS, _EP, _LAYERS = 896, 900, 20, 4


def _mapping(rank: int) -> Mapping:
    return Mapping(
        rank=rank,
        world_size=_EP,
        attn_dp_size=_EP,
        moe_ep_size=_EP,
        nprocs_per_node=4,
    )


def _placement(ep_rank: int, device: str, ep_size: int = _EP):
    # Trivial placement, four GPUs per node: experts 0..3 sit on rank 0 (node
    # 0) and again in slots 896..899 on rank 19 (node 4).
    physical_to_logical = (torch.arange(_SLOTS) % _EXPERTS).repeat(_LAYERS, 1)
    return ExpertLocationMetadata.from_physical_to_logical_map(
        physical_to_logical.to(device),
        _EXPERTS,
        ep_size=ep_size,
        ep_rank=ep_rank,
        ep_rank_nodes=tuple(rank // 4 for rank in range(ep_size)),
    )


@pytest.fixture
def static_dispatch(monkeypatch):
    monkeypatch.setitem(
        kimi_k3.global_server_args_dict, "ep_dispatch_algorithm", "static"
    )


def test_k3_entry_opts_in_with_its_text_geometry():
    config = KimiK3Config(architectures=["KimiK3ForConditionalGeneration"])
    model_config = SimpleNamespace(
        hf_config=config, quantization=None, model_profile_architecture=None
    )
    args = SimpleNamespace(
        ep_num_redundant_experts=4,
        init_expert_location="trivial",
        expert_distribution_recorder_mode=None,
        ep_dispatch_algorithm="static",
        enable_eplb=False,
        device="cpu",
        mapping=_mapping(19),
    )
    placement = build_expert_placement(args, model_config)
    assert placement.num_layers == config.text_config.num_hidden_layers
    assert placement.num_logical_experts == _EXPERTS
    assert placement.num_physical_experts == _SLOTS
    assert placement.num_local_physical_experts == 45
    assert placement.local_slot_logical_experts(5, 19) == [*range(855, 896), 0, 1, 2, 3]
    assert placement.rank_dispatch_map()[5, :4].tolist() == [896, 897, 898, 899]
    # Without the redundant slots 896 experts do not divide over EP20.
    args.ep_num_redundant_experts = 0
    args.expert_distribution_recorder_mode = "stat"
    with pytest.raises(ValueError, match="do not divide"):
        build_expert_placement(args, model_config)


def test_k3_accepts_a_placement_only_on_the_attention_dp_megamoe_path(
    monkeypatch, static_dispatch
):
    config = KimiLinearConfig(num_experts=_EXPERTS)
    mapping = _mapping(0)
    mega_moe = Kimi3MoEExecutionPlan.build(mapping, MoeBackend.MEGA_MOE, None)
    _check_k3_expert_placement(_placement(0, "cpu"), config, mapping, 3, mega_moe, None)

    trtllm = Kimi3MoEExecutionPlan(
        use_native=False,
        use_trtllm=True,
        use_mega_moe=False,
        owns_ep_communication=False,
        overlap_shared_experts=False,
        joint_moe_reduce=False,
    )
    attn_tp = Mapping(rank=0, world_size=_EP, attn_tp_size=_EP, moe_ep_size=_EP)
    refusals = [
        ((_placement(0, "cpu"), config, mapping, 3, trtllm), "mega_moe"),
        ((_placement(0, "cpu"), config, attn_tp, 3, mega_moe), "attention DP"),
        (
            (
                _placement(0, "cpu"),
                KimiLinearConfig(num_experts=768),
                mapping,
                3,
                mega_moe,
            ),
            "logical experts",
        ),
        ((_placement(0, "cpu"), config, mapping, _LAYERS, mega_moe), "outside"),
        ((_placement(0, "cpu", ep_size=10), config, mapping, 3, mega_moe), "ep_size"),
    ]
    for args, match in refusals:
        with pytest.raises(ValueError, match=match):
            _check_k3_expert_placement(*args, None)

    recording = _placement(0, "cpu")
    recording.enable_load_recording(mapping)
    with pytest.raises(ValueError, match="--enable-eplb"):
        _check_k3_expert_placement(recording, config, mapping, 3, mega_moe, None)
    monkeypatch.setitem(
        kimi_k3.global_server_args_dict, "ep_dispatch_algorithm", "dynamic"
    )
    with pytest.raises(ValueError, match="--ep-dispatch-algorithm static"):
        _check_k3_expert_placement(
            _placement(0, "cpu"), config, mapping, 3, mega_moe, None
        )


def test_k3_moe_builds_physical_slots_and_a_static_router_map(
    monkeypatch, static_dispatch
):
    class Experts(nn.Module):
        """Stands in for MegaMoE, whose setup needs the 20-rank EP group."""

        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.supports_precomputed_topk = True
            self.topk_output_format = TopKOutputFormat.STANDARD
            self.plan = {"a2a_backend": "none"}

    monkeypatch.setattr(kimi_k3, "MoELayer", Experts)
    monkeypatch.setattr(kimi_k3, "get_moe_backend", lambda: MoeBackend.MEGA_MOE)
    monkeypatch.setattr(kimi_k3, "get_all2all_backend", lambda: All2AllBackend.NONE)
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
            mapping=_mapping(19),
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
    assert layer.topk.topk_config.layer_id == 3
    info = layer.expert_dispatch_info
    assert info.layer_id == 3 and info.replica_dispatch is None
    assert info.partial_logical_to_rank_dispatch_physical_map[:5].tolist() == [
        896,
        897,
        898,
        899,
        4,
    ]


def test_k3_moe_refuses_a_placement_under_deepep(monkeypatch):
    monkeypatch.setattr(kimi_k3, "get_all2all_backend", lambda: All2AllBackend.DEEPEP)
    set_global_expert_location_metadata(_placement(0, "cpu"))
    try:
        with pytest.raises(ValueError, match="deepep"):
            kimi_k3.create_kimi_linear_moe(
                config=KimiLinearConfig(num_experts=_EXPERTS),
                mapping=_mapping(0),
                layer_index=3,
                model_scope="test",
                moe_block_count=1,
                quant_config=None,
                prefix="moe",
                alt_stream=None,
            )
    finally:
        set_global_expert_location_metadata(None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="K3 routing runs on CUDA")
@pytest.mark.parametrize("tokens", [3, 300])  # fused mapped kernel; gather after
@pytest.mark.parametrize("ep_rank", [0, 5, 16, 19])
def test_k3_router_writes_physical_slot_ids_into_the_megamoe_buffers(
    tokens: int, ep_rank: int
):
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

    def megamoe_buffers():
        # MegaMoE's workspace views: FP32 weights, INT32 ids.
        return (
            torch.empty(tokens, 16, dtype=torch.float32, device="cuda"),
            torch.full((tokens, 16), -1, dtype=torch.int32, device="cuda"),
        )

    logical = megamoe_buffers()
    topk(hidden, logits, expert_location_dispatch_info=None, out=logical)
    physical = megamoe_buffers()
    routed = topk(hidden, logits, expert_location_dispatch_info=info, out=physical)

    assert routed.topk_ids.data_ptr() == physical[1].data_ptr()
    assert routed.topk_weights.data_ptr() == physical[0].data_ptr()
    assert torch.equal(physical[0], logical[0])
    assert torch.equal(physical[1], placement.rank_dispatch_map()[3][logical[1].long()])
    # Every slot holds the expert the router chose, on the rank that owns it.
    assert torch.equal(
        placement.physical_to_logical_map[3][physical[1].long()].to(torch.int32),
        logical[1],
    )
    replicas = physical[1][logical[1] < 4] - logical[1][logical[1] < 4]
    if ep_rank == 0:
        assert (replicas == 0).all()  # its own slots 0..3
    elif ep_rank in (16, 19):
        assert (replicas == _EXPERTS).all()  # rank 19's slots 896..899
    else:
        assert ((replicas == 0) | (replicas == _EXPERTS)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_static_rank_dispatch_map_builds_under_a_cuda_default_device():
    # Model construction runs with CUDA as the default device; the host-side
    # nearest-replica table must not pick it up.
    replicas = torch.tensor([[[0, -1], [1, 4], [2, -1], [3, 5]]], dtype=torch.int64)
    with torch.device("cuda"):
        dispatch = compute_logical_to_rank_dispatch_physical_map(
            logical_to_all_physical_map=replicas,
            num_physical_experts=6,
            ep_rank_nodes=[0, 1],
            ep_rank=1,
        )
    assert dispatch.device.type == "cpu"
    # Rank 1 owns slots 3..5: it keeps its own replicas of experts 1 and 3 and
    # sends experts 0 and 2 to their only slots.
    assert dispatch.tolist() == [[0, 4, 2, 3]]


class _DenseDrafterWeights(Eagle3DeepseekV2ForCausalLM):
    """The EAGLE3 MLA drafter's real weight path over one dense projection;
    attention post-processing needs MLA modules this test does not build."""

    def __init__(self, config, mapping):
        nn.Module.__init__(self)
        self.config = config
        self.mapping = mapping
        self.quant_config = None
        self.model = nn.Module()
        self.model.midlayer = nn.Module()
        self.model.midlayer.mlp = nn.Module()
        self.model.midlayer.mlp.down_proj = nn.Linear(4, 2, bias=False)

    def post_load_weights(self):
        pass


def test_eagle3_mla_drafter_loads_at_an_ep_size_that_does_not_divide_the_experts():
    # The drafter's config carries the target's 896 routed experts, but its one
    # decoder layer is dense: EP20 must not need an expert partition for it.
    drafter = _DenseDrafterWeights(
        PretrainedConfig(n_routed_experts=_EXPERTS, num_hidden_layers=1),
        _mapping(rank=3),
    )
    weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    loaded = drafter.load_weights([("layers.0.mlp.down_proj.weight", weight)])
    assert loaded == {"model.midlayer.mlp.down_proj.weight"}
    assert torch.equal(drafter.model.midlayer.mlp.down_proj.weight.data, weight)
