from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.moe import moe_apply as kernel_moe_apply
from tokenspeed_kernel.ops.moe import moe_plan as kernel_moe_plan
from utils import (
    assert_no_triton_compile,
    int_specialization_class,
    warm_specialization_classes,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("graph_replay", [False, True])
def test_bf16_route_combine_rounds_before_accumulate_and_scale(graph_replay):
    from tokenspeed_kernel.ops.moe._triton.bf16 import _combine

    generator = torch.Generator(device="cpu").manual_seed(47)
    routes_cpu = torch.randn(8, 128, generator=generator).to(torch.bfloat16)
    weights_cpu = torch.rand(2, 4, generator=generator, dtype=torch.float32)
    weights_cpu /= weights_cpu.sum(dim=1, keepdim=True)
    routes = routes_cpu.cuda()
    weights = weights_cpu.cuda()
    output = torch.empty((2, 128), device="cuda", dtype=torch.bfloat16)

    def expected_values():
        total = torch.zeros((2, 128), dtype=torch.float32)
        for slot in range(4):
            weighted = routes_cpu.view(2, 4, 128)[:, slot].float()
            weighted = (weighted * weights_cpu[:, slot, None]).to(torch.bfloat16)
            total += weighted.float()
        return (total * 2.5).to(torch.bfloat16), total

    expected, total = expected_values()
    wrong_post_cast = (total.to(torch.bfloat16).float() * 2.5).to(torch.bfloat16)
    wrong_unrounded = (
        (routes_cpu.view(2, 4, 128).float() * weights_cpu[:, :, None]).sum(dim=1) * 2.5
    ).to(torch.bfloat16)
    assert not torch.equal(expected, wrong_post_cast)
    assert not torch.equal(expected, wrong_unrounded)
    _combine(
        routes,
        weights,
        output,
        None,
        0,
        round_weighted_routes=True,
        output_scale=2.5,
    )
    graph = None
    if graph_replay:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _combine(
                routes,
                weights,
                output,
                None,
                0,
                round_weighted_routes=True,
                output_scale=2.5,
            )
        graph.replay()
    assert torch.equal(output.cpu(), expected)
    # Replay must read live route values/weights, not captured snapshots.
    routes_cpu.neg_()
    weights_cpu = weights_cpu.roll(1, dims=1)
    routes.copy_(routes_cpu)
    weights.copy_(weights_cpu)
    if graph is None:
        _combine(
            routes,
            weights,
            output,
            None,
            0,
            round_weighted_routes=True,
            output_scale=2.5,
        )
    else:
        graph.replay()
    expected, _ = expected_values()
    assert torch.equal(output.cpu(), expected)
    if graph is not None:
        graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("activation", ["silu", "situ"])
@pytest.mark.parametrize("num_tokens", [4, 48, 80])
@pytest.mark.parametrize("intermediate", [32, 128])
@pytest.mark.parametrize("precision_policy", ["default", "bf16_route_accum_fp32"])
def test_triton_bf16_moe_matches_torch(
    activation: str, num_tokens: int, intermediate: int, precision_policy: str
) -> None:
    """Four tokens run 16-row tiles; with the deep pipeline 48 tokens spread one
    expert's 24 routes over two 16-row tiles and 80 (40 per expert) take 64-row
    tiles, while a 32-wide intermediate keeps the shallow token-count rule.
    On SM90 the three counts take the 16-, 32- and 64-row tiles."""
    if activation == "situ" and precision_policy != "default":
        pytest.skip("The opt-in precision policy supports SiLU only")
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(
        num_tokens, 128, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    w13 = (
        torch.randn(
            4,
            2 * intermediate,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            4,
            128,
            intermediate,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.02
    )
    # Descending expert pairs exercise the ascending-expert sort that the
    # rounding policy applies before the FP32 route accumulation.
    first = torch.arange(num_tokens, device="cuda") % 4
    topk_ids = torch.stack(((first + 1) % 4, first), dim=1).to(torch.int32)
    topk_weights = torch.rand(
        (num_tokens, 2), device="cuda", dtype=torch.float32, generator=generator
    )

    weights = torch.nn.Module()
    weights.w13_weight = w13
    weights.w2_weight = w2
    weights.top_k = 2
    weights.activation_situ_beta = 4.0
    weights.activation_situ_linear_beta = 25.0
    plan = kernel_moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation=activation,
        routing_mode="precomputed_topk",
        ispp=32,
        solution="triton",
        hidden=None,
        swiglu_form=("standard" if activation == "swiglu" else None),
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
    )
    if precision_policy != "default":
        plan["precision_policy"] = precision_policy
        plan["output_scale"] = 2.5
    actual = kernel_moe_apply(
        plan,
        x,
        weights,
        torch.empty((num_tokens, 4), device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    expected = torch.zeros((num_tokens, 128), device="cuda", dtype=torch.float32)
    for expert_id in range(4):
        token_ids, slots = torch.where(topk_ids == expert_id)
        gate_up = F.linear(x[token_ids].float(), w13[expert_id].float())
        gate, up = gate_up.chunk(2, dim=-1)
        if precision_policy != "default":
            gate = gate.to(torch.bfloat16).float()
            up = up.to(torch.bfloat16).float()
        if activation == "situ":
            gate = gate.to(torch.bfloat16).float()
            up = up.to(torch.bfloat16).float()
            gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
            up = 25.0 * torch.tanh(up / 25.0)
            intermediate = (gate * up).to(torch.bfloat16)
        else:
            intermediate = (F.silu(gate) * up).to(torch.bfloat16)
        expert_output = F.linear(intermediate.float(), w2[expert_id].float()).to(
            torch.bfloat16
        )
        weighted = expert_output.float() * topk_weights[token_ids, slots, None]
        if precision_policy != "default":
            weighted = weighted.to(torch.bfloat16).float()
        expected.index_add_(0, token_ids, weighted)

    if precision_policy != "default":
        expected = (expected * 2.5).to(torch.bfloat16)

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=0.02,
        atol=0.0002 if precision_policy != "default" else 0.01,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_triton_bf16_moe_token_count():
    """Token counts reuse the stage binaries of their row tile.

    Small batches launch fewer persistent programs than the grid's cap, so the
    program count must stay a runtime value.
    """
    from tokenspeed_kernel.ops.moe._triton import bf16
    from tokenspeed_kernel.ops.moe._triton import common as _common

    num_experts, top_k, hidden_size, intermediate_size = 4, 2, 128, 64
    max_tokens = 300
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(
        max_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weights = torch.nn.Module()
    weights.w13_weight = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.02
    )
    weights.w2_weight = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.02
    )
    weights.top_k = top_k
    first = torch.arange(max_tokens, device="cuda") % num_experts
    topk_ids = torch.stack(((first + 1) % num_experts, first), dim=1).to(torch.int32)
    topk_weights = torch.rand(
        (max_tokens, top_k), device="cuda", dtype=torch.float32, generator=generator
    )
    plan = kernel_moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ispp=intermediate_size,
        solution="triton",
        hidden=None,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
    )
    plan["precision_policy"] = "bf16_route_accum_fp32"
    plan["output_scale"] = 1.0

    def run(tokens):
        return kernel_moe_apply(
            plan,
            x[:tokens],
            weights,
            torch.empty((tokens, num_experts), device="cuda"),
            topk_weights=topk_weights[:tokens],
            topk_ids=topk_ids[:tokens],
        )

    def tiles(tokens):
        # The row tile on SM90 (by routes per expert) and elsewhere (by tokens).
        routes_per_expert = tokens * top_k / num_experts
        return routes_per_expert <= 8, routes_per_expert < 32, tokens <= 16

    def key(tokens):
        routes = tokens * top_k
        counts = (tokens, routes)
        return (*map(int_specialization_class, counts), *tiles(tokens), routes <= 128)

    # Tokens are independent: a shorter batch is a prefix of the longest one
    # that uses the same row tile.
    expected = {tiles(tokens): run(tokens) for tokens in (16, 63, max_tokens)}
    sweep = (3, 7, 12, 17, 24, 40, 62, 65, 100, 255, 299)
    warm_specialization_classes(run, key, sweep, range(1, max_tokens))
    with assert_no_triton_compile(
        bf16._stage1_kernel,
        bf16._stage2_kernel,
        _common._routing_kernel,
        _common._combine_kernel,
    ):
        for tokens in sweep:
            torch.testing.assert_close(
                run(tokens), expected[tiles(tokens)][:tokens], rtol=0, atol=0
            )
