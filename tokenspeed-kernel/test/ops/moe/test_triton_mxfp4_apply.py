from __future__ import annotations

import pytest
import tokenspeed_kernel
import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import current_platform


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("activation", ["silu", "situ"])
def test_triton_mxfp4_moe_matches_torch(activation: str) -> None:
    if not current_platform().is_amd:
        pytest.skip("Triton MXFP4 MoE is registered for AMD GPUs")

    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    w13 = torch.randint(
        0, 256, (4, 256, 64), device="cuda", dtype=torch.uint8, generator=generator
    )
    w13_scale = torch.full((4, 256, 4), 120, device="cuda", dtype=torch.uint8)
    w2 = torch.randint(
        0, 256, (4, 128, 64), device="cuda", dtype=torch.uint8, generator=generator
    )
    w2_scale = torch.full((4, 128, 4), 120, device="cuda", dtype=torch.uint8)
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0]], device="cuda", dtype=torch.int32
    )
    topk_weights = torch.tensor(
        [[0.6, 0.4], [0.7, 0.3], [0.2, 0.8], [0.5, 0.5]],
        device="cuda",
        dtype=torch.float32,
    )

    weights = torch.nn.Module()
    weights.w13_weight = w13
    weights.w13_weight_scale = w13_scale
    weights.w2_weight = w2
    weights.w2_weight_scale = w2_scale
    weights.top_k = 2
    weights.w13_input_layout = "concatenated"
    weights.activation_situ_beta = 4.0
    weights.activation_situ_linear_beta = 25.0
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation=activation,
        routing_mode="precomputed_topk",
        ispp=128,
        internal_activation_dtype="mxfp4",
        solution="triton",
    )
    tokenspeed_kernel.moe_process_weights(plan, weights)
    actual = tokenspeed_kernel.moe_apply(
        plan,
        x,
        weights,
        torch.empty((4, 4), device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
        dtype=torch.float32,
    )

    x_packed, x_scale = tokenspeed_kernel.quantize_mxfp4(
        x, scale_layout="linear", solution="triton"
    )
    x_codes = torch.stack((x_packed & 0xF, x_packed >> 4), dim=-1).flatten(-2)
    x_dequant = (
        table[x_codes.long()]
        * torch.exp2(x_scale.float() - 127).repeat_interleave(32, dim=-1)
    ).to(torch.bfloat16)

    w13_codes = torch.stack((w13 & 0xF, w13 >> 4), dim=-1).flatten(-2)
    w13_dequant = (
        table[w13_codes.long()]
        * torch.exp2(w13_scale.float() - 127).repeat_interleave(32, dim=-1)
    ).to(torch.bfloat16)
    w2_codes = torch.stack((w2 & 0xF, w2 >> 4), dim=-1).flatten(-2)
    w2_dequant = (
        table[w2_codes.long()]
        * torch.exp2(w2_scale.float() - 127).repeat_interleave(32, dim=-1)
    ).to(torch.bfloat16)

    expected = torch.zeros((4, 128), device="cuda", dtype=torch.float32)
    for expert_id in range(4):
        token_ids, slots = torch.where(topk_ids == expert_id)
        gate_up = F.linear(x_dequant[token_ids], w13_dequant[expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        if activation == "situ":
            gate = gate.float()
            up = up.float()
            gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
            up = 25.0 * torch.tanh(up / 25.0)
            intermediate = (gate * up).to(torch.bfloat16)
        else:
            intermediate = (F.silu(gate) * up).to(torch.bfloat16)
        intermediate_packed, intermediate_scale = tokenspeed_kernel.quantize_mxfp4(
            intermediate, scale_layout="linear", solution="triton"
        )
        intermediate_codes = torch.stack(
            (intermediate_packed & 0xF, intermediate_packed >> 4), dim=-1
        ).flatten(-2)
        intermediate_dequant = (
            table[intermediate_codes.long()]
            * torch.exp2(intermediate_scale.float() - 127).repeat_interleave(32, dim=-1)
        ).to(torch.bfloat16)
        expert_output = F.linear(intermediate_dequant, w2_dequant[expert_id]).to(
            torch.bfloat16
        )
        expected.index_add_(
            0,
            token_ids,
            expert_output.float() * topk_weights[token_ids, slots, None],
        )

    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)
