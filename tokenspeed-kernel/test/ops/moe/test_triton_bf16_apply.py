from __future__ import annotations

import pytest
import tokenspeed_kernel
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_triton_bf16_moe_matches_torch() -> None:
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    w13 = (
        torch.randn(
            4, 64, 128, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            4, 128, 32, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.02
    )
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
    weights.w2_weight = w2
    weights.top_k = 2
    plan = tokenspeed_kernel.moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ispp=32,
        solution="triton",
    )
    actual = tokenspeed_kernel.moe_apply(
        plan,
        x,
        weights,
        torch.empty((4, 4), device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    expected = torch.zeros((4, 128), device="cuda", dtype=torch.float32)
    for expert_id in range(4):
        token_ids, slots = torch.where(topk_ids == expert_id)
        gate_up = F.linear(x[token_ids].float(), w13[expert_id].float())
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = (F.silu(gate) * up).to(torch.bfloat16)
        expert_output = F.linear(intermediate.float(), w2[expert_id].float()).to(
            torch.bfloat16
        )
        expected.index_add_(
            0,
            token_ids,
            expert_output.float() * topk_weights[token_ids, slots, None],
        )

    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.01)
