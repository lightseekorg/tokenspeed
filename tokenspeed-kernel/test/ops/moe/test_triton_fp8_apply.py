from __future__ import annotations

from types import SimpleNamespace

import pytest
import tokenspeed_kernel
import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import current_platform


def _block_fp8(
    shape: tuple[int, int, int], generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_experts, rows, cols = shape
    weight = torch.randn(shape, device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    scale = (
        torch.rand(
            (num_experts, rows // 128, cols // 128),
            device="cuda",
            generator=generator,
        )
        + 0.5
    ) / cols**0.5
    dequantized = weight.float() * scale.repeat_interleave(128, 1).repeat_interleave(
        128, 2
    )
    return weight, scale, dequantized


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("num_tokens", [5, 33])
@pytest.mark.parametrize(
    ("activation", "swiglu_limit"), [("silu", None), ("swiglu", 1.5)]
)
def test_triton_fp8_moe_matches_torch(
    num_tokens: int, activation: str, swiglu_limit: float | None
) -> None:
    if not current_platform().is_amd:
        pytest.skip("Triton FP8 MoE is registered for AMD GPUs")

    num_experts, top_k, hidden_size, intermediate_size = 4, 2, 384, 256
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(
        num_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    w13, w13_scale, w13_dequant = _block_fp8(
        (num_experts, 2 * intermediate_size, hidden_size), generator
    )
    w2, w2_scale, w2_dequant = _block_fp8(
        (num_experts, hidden_size, intermediate_size), generator
    )
    topk_ids = (
        torch.rand(num_tokens, num_experts, device="cuda", generator=generator)
        .argsort(dim=1)[:, :top_k]
        .to(torch.int32)
    )
    topk_ids[0, 1] = -1
    topk_weights = torch.rand(
        num_tokens, top_k, device="cuda", dtype=torch.float32, generator=generator
    )

    weights = torch.nn.Module()
    weights.w13_weight = w13
    weights.w13_weight_scale_inv = w13_scale
    weights.w2_weight = w2
    weights.w2_weight_scale_inv = w2_scale
    weights.top_k = top_k
    if swiglu_limit is not None:
        weights.swiglu_arg = SimpleNamespace(alpha=None, limit=swiglu_limit)
    plan = tokenspeed_kernel.moe_plan(
        "fp8",
        input_dtype=torch.bfloat16,
        activation=activation,
        routing_mode="precomputed_topk",
        ispp=intermediate_size,
        fp8_scale_block_shape=(128, 128),
        internal_activation_dtype="input",
        solution="triton",
        hidden=hidden_size,
        swiglu_form=("standard" if activation == "swiglu" else None),
        activation_clamped=swiglu_limit is not None,
        expert_id_repeats=False,
        fast_math=True,
    )
    assert plan["apply_kernel_name"] == "triton_fp8_block_precomputed_moe_apply"
    tokenspeed_kernel.moe_process_weights(plan, weights)
    actual = tokenspeed_kernel.moe_apply(
        plan,
        x,
        weights,
        torch.empty((num_tokens, num_experts), device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    expected = torch.zeros(
        (num_tokens, hidden_size), device="cuda", dtype=torch.float32
    )
    for expert_id in range(num_experts):
        token_ids, slots = torch.where(topk_ids == expert_id)
        gate_up = F.linear(x[token_ids].float(), w13_dequant[expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        if swiglu_limit is not None:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(-swiglu_limit, swiglu_limit)
        intermediate = (F.silu(gate) * up).to(torch.bfloat16)
        expert_output = F.linear(intermediate.float(), w2_dequant[expert_id]).to(
            torch.bfloat16
        )
        expected.index_add_(
            0,
            token_ids,
            expert_output.float() * topk_weights[token_ids, slots, None],
        )

    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.01)
