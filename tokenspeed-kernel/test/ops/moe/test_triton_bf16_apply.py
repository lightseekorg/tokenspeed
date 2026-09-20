from __future__ import annotations

import pytest
import tokenspeed_kernel
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("activation", ["silu", "situ"])
def test_triton_bf16_moe_matches_reference(activation: str) -> None:
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
    weights.activation_situ_beta = 4.0
    weights.activation_situ_linear_beta = 25.0

    def run(solution: str) -> torch.Tensor:
        plan = tokenspeed_kernel.moe_plan(
            "unquant",
            input_dtype=torch.bfloat16,
            activation=activation,
            routing_mode="precomputed_topk",
            ispp=32,
            solution=solution,
        )
        return tokenspeed_kernel.moe_apply(
            plan,
            x,
            weights,
            torch.empty((4, 4), device="cuda"),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

    actual = run("triton")
    expected = run("reference")
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.01)
