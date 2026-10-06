from __future__ import annotations

from types import SimpleNamespace

import pytest
import tokenspeed_kernel
import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import ArchVersion, current_platform
from utils import (
    assert_no_triton_compile,
    int_specialization_class,
    warm_specialization_classes,
)


def _skip_unless_supported() -> None:
    platform = current_platform()
    floor = {"amd": ArchVersion(9, 5), "nvidia": ArchVersion(8, 9)}.get(platform.vendor)
    if floor is None or not platform.arch_version >= floor:
        pytest.skip("Triton FP8 MoE requires gfx950+ or SM89+")


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


def _experts(
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    generator: torch.Generator,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    w13, w13_scale, w13_dequant = _block_fp8(
        (num_experts, 2 * intermediate_size, hidden_size), generator
    )
    w2, w2_scale, w2_dequant = _block_fp8(
        (num_experts, hidden_size, intermediate_size), generator
    )
    weights = torch.nn.Module()
    weights.w13_weight = w13
    weights.w13_weight_scale_inv = w13_scale
    weights.w2_weight = w2
    weights.w2_weight_scale_inv = w2_scale
    weights.top_k = top_k
    return weights, w13_dequant, w2_dequant


def _topk_ids(
    num_tokens: int, num_experts: int, top_k: int, generator: torch.Generator
) -> torch.Tensor:
    return (
        torch.rand(num_tokens, num_experts, device="cuda", generator=generator)
        .argsort(dim=1)[:, :top_k]
        .to(torch.int32)
    )


def _plan(
    activation: str,
    swiglu_limit: float | None,
    hidden_size: int,
    intermediate_size: int,
) -> dict:
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
        combine_order="rank",
    )
    assert plan["apply_kernel_name"] == "triton_fp8_block_precomputed_moe_apply"
    return plan


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("num_tokens", [5, 33])
@pytest.mark.parametrize(
    ("activation", "swiglu_limit"), [("silu", None), ("swiglu", 1.5)]
)
def test_triton_fp8_moe_matches_torch(
    num_tokens: int, activation: str, swiglu_limit: float | None
) -> None:
    _skip_unless_supported()

    num_experts, top_k, hidden_size, intermediate_size = 4, 2, 384, 256
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(
        num_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weights, w13_dequant, w2_dequant = _experts(
        num_experts, top_k, hidden_size, intermediate_size, generator
    )
    topk_ids = _topk_ids(num_tokens, num_experts, top_k, generator)
    topk_ids[0, 1] = -1
    topk_weights = torch.rand(
        num_tokens, top_k, device="cuda", dtype=torch.float32, generator=generator
    )

    if swiglu_limit is not None:
        weights.swiglu_arg = SimpleNamespace(alpha=None, limit=swiglu_limit)
    plan = _plan(activation, swiglu_limit, hidden_size, intermediate_size)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_triton_fp8_moe_token_count():
    _skip_unless_supported()
    from tokenspeed_kernel.ops.moe.triton import _common, fp8

    num_experts, top_k, hidden_size, intermediate_size = 4, 2, 256, 128
    max_tokens = 1500
    generator = torch.Generator(device="cuda").manual_seed(0)
    weights, _, _ = _experts(
        num_experts, top_k, hidden_size, intermediate_size, generator
    )
    plan = _plan("silu", None, hidden_size, intermediate_size)
    tokenspeed_kernel.moe_process_weights(plan, weights)
    x = torch.randn(
        max_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_ids = _topk_ids(max_tokens, num_experts, top_k, generator)
    topk_weights = torch.rand(
        max_tokens, top_k, device="cuda", dtype=torch.float32, generator=generator
    )

    def run(tokens):
        return tokenspeed_kernel.moe_apply(
            plan,
            x[:tokens],
            weights,
            torch.empty((tokens, num_experts), device="cuda"),
            topk_weights=topk_weights[:tokens],
            topk_ids=topk_ids[:tokens],
        )

    def key(tokens):
        routes = tokens * top_k
        counts = (
            tokens,
            routes,
            _common._num_programs(x.device, routes, intermediate_size, 32),
            _common._num_programs(x.device, routes, hidden_size, 32),
        )
        # Up to 16 tokens take the small row tile, up to 128 routes the small
        # routing block.
        return (*map(int_specialization_class, counts), tokens <= 16, routes <= 128)

    # Tokens are independent: a shorter batch is a prefix of one that uses
    # the same row tile.
    small, full = run(16), run(max_tokens)
    sweep = (3, 7, 12, 24, 100, 300, 1000, 1483)
    warm_specialization_classes(run, key, sweep, range(1, max_tokens))
    with assert_no_triton_compile(
        fp8._stage1_kernel,
        fp8._stage2_kernel,
        _common._routing_kernel,
        _common._combine_kernel,
    ):
        for tokens in sweep:
            expected = small if tokens <= 16 else full
            torch.testing.assert_close(run(tokens), expected[:tokens], rtol=0, atol=0)
