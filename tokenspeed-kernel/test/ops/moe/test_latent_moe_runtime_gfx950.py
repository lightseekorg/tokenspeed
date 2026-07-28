from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from kimi3_reference import (
    mxfp4_situ_latent_moe_reference,
)
from torch import nn
from utils import make_mxfp4_moe_weights


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip("latent MXFP4 runtime test requires gfx950", allow_module_level=True)

from tokenspeed.runtime.layers.activation import SituAndMul  # noqa: E402
from tokenspeed.runtime.layers.layernorm import RMSNorm  # noqa: E402
from tokenspeed.runtime.layers.moe.expert import MoELayer  # noqa: E402
from tokenspeed.runtime.layers.moe.latent import LatentMoELayer  # noqa: E402
from tokenspeed.runtime.layers.moe.topk import TopK  # noqa: E402
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config  # noqa: E402


class _TestRouter(nn.Linear):
    def __init__(self, hidden_size: int, num_experts: int) -> None:
        super().__init__(
            hidden_size,
            num_experts,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32, device="cuda")
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.float(), self.weight.float())


class _TestSharedExperts(nn.Module):
    def __init__(
        self,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.gate_weight = nn.Parameter(gate_weight, requires_grad=False)
        self.up_weight = nn.Parameter(up_weight, requires_grad=False)
        self.down_weight = nn.Parameter(down_weight, requires_grad=False)
        self.activation = SituAndMul(beta=4.0, linear_beta=25.0)
        self.forward_streams: list[torch.cuda.Stream] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.forward_streams.append(torch.cuda.current_stream())
        gate = F.linear(hidden_states, self.gate_weight)
        up = F.linear(hidden_states, self.up_weight)
        return F.linear(
            self.activation(torch.cat((gate, up), dim=-1)),
            self.down_weight,
        )


@pytest.mark.parametrize(
    ("overlap_shared_experts", "graph_phase", "expect_overlap"),
    [(False, False, False), (True, False, True), (True, True, False)],
)
def test_latent_moe_runtime_composes_real_mxfp4_moe_layer(
    overlap_shared_experts: bool,
    graph_phase: bool,
    expect_overlap: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260718)
    tokens, hidden_size = 5, 64
    latent_size = intermediate_size = 32
    num_experts, top_k = 4, 2

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        experts = MoELayer(
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=latent_size,
            intermediate_size=intermediate_size,
            quant_config=Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
            layer_index=1,
            activation="situ",
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
            routing_mode="precomputed_topk",
        ).cuda()
    finally:
        torch.set_default_dtype(previous_dtype)

    raw = make_mxfp4_moe_weights(
        num_experts,
        latent_size,
        intermediate_size,
        generator,
        scale_range=(118, 126),
    )
    w13_packed, w13_scales = raw["w13_weight"], raw["w13_scale"]
    w2_packed, w2_scales = raw["w2_weight"], raw["w2_scale"]
    experts.w13_weight.data.copy_(w13_packed)
    experts.w13_weight_scale.data.copy_(w13_scales)
    experts.w2_weight.data.copy_(w2_packed)
    experts.w2_weight_scale.data.copy_(w2_scales)
    experts.process_weights_after_loading(experts)

    router = _TestRouter(hidden_size, num_experts)
    router.weight.data.normal_(generator=generator).mul_(hidden_size**-0.5)
    router.e_score_correction_bias.data.normal_(generator=generator).mul_(0.01)

    routed_down = torch.nn.Linear(
        hidden_size,
        latent_size,
        bias=False,
        dtype=torch.bfloat16,
        device="cuda",
    )
    routed_up = torch.nn.Linear(
        latent_size,
        hidden_size,
        bias=False,
        dtype=torch.bfloat16,
        device="cuda",
    )
    routed_down.weight.data.normal_(generator=generator).mul_(hidden_size**-0.5)
    routed_up.weight.data.normal_(generator=generator).mul_(latent_size**-0.5)
    routed_norm = RMSNorm(latent_size, eps=1e-5).cuda().to(torch.bfloat16)
    routed_norm.weight.data.normal_(mean=1.0, std=0.1, generator=generator)

    shared_gate = (
        torch.randn(
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * hidden_size**-0.5
    )
    shared_up = (
        torch.randn(
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * hidden_size**-0.5
    )
    shared_down = (
        torch.randn(
            hidden_size,
            2 * intermediate_size,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * (2 * intermediate_size) ** -0.5
    )
    shared_experts = _TestSharedExperts(
        shared_gate,
        shared_up,
        shared_down,
    )
    topk = TopK(
        top_k=top_k,
        use_grouped_topk=True,
        topk_group=1,
        num_expert_group=1,
        renormalize=True,
        correction_bias=router.e_score_correction_bias,
        routed_scaling_factor=1.0,
        output_format=experts.topk_output_format,
    )
    shared_expert_stream = torch.cuda.Stream() if overlap_shared_experts else None
    monkeypatch.setattr(
        "tokenspeed.runtime.layers.moe.latent.get_is_cuda_graph_phase",
        lambda: graph_phase,
    )
    primary_stream = torch.cuda.current_stream()
    shared_reduce_streams: list[torch.cuda.Stream] = []

    def shared_reduce(value: torch.Tensor) -> torch.Tensor:
        shared_reduce_streams.append(torch.cuda.current_stream())
        return value

    layer = LatentMoELayer(
        router=router,
        topk=topk,
        routed_down_proj=routed_down,
        experts=experts,
        routed_norm=routed_norm,
        routed_up_proj=routed_up,
        shared_experts=shared_experts,
        shared_reduce=shared_reduce,
        shared_expert_stream=shared_expert_stream,
    )
    hidden_states = torch.randn(
        tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    actual = layer(hidden_states)
    actual_shared_streams = list(shared_experts.forward_streams)
    expected = mxfp4_situ_latent_moe_reference(
        hidden_states,
        router.weight,
        router.e_score_correction_bias,
        routed_down.weight,
        routed_up.weight,
        w13_packed,
        w13_scales,
        w2_packed,
        w2_scales,
        top_k=top_k,
        num_expert_group=1,
        topk_group=1,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        rms_norm_weight=routed_norm.weight,
        rms_norm_eps=1e-5,
        shared_expert=shared_experts,
    )
    torch.cuda.synchronize()

    assert actual.shape == (tokens, hidden_size)
    assert actual.dtype == torch.bfloat16
    expected_shared_stream = shared_expert_stream if expect_overlap else primary_stream
    assert actual_shared_streams == [expected_shared_stream]
    assert shared_reduce_streams == [primary_stream]
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
