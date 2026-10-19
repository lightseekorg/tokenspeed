from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.numerics.torch.quantization import dequantize_mxfp4
from tokenspeed_kernel.ops.moe import moe_apply as kernel_moe_apply
from tokenspeed_kernel.ops.moe import moe_plan as kernel_moe_plan
from tokenspeed_kernel.ops.moe import moe_process_weights as kernel_moe_process_weights
from tokenspeed_kernel.ops.quantization import quantize_mxfp4 as kernel_quantize_mxfp4
from tokenspeed_kernel.platform import current_platform

# E2M1 code points, in the packed nibble order.


def _dequantize(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return dequantize_mxfp4(packed, scale).to(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("activation", ["silu", "situ"])
# 192 is a multiple of 64 but not 128, like a 640-wide expert sharded by TP2.
@pytest.mark.parametrize("intermediate_size", [128, 192])
def test_triton_mxfp4_moe_matches_torch(
    activation: str, intermediate_size: int
) -> None:
    if not current_platform().is_amd:
        pytest.skip("Triton MXFP4 MoE is registered for AMD GPUs")

    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    w13 = torch.randint(
        0,
        256,
        (4, 2 * intermediate_size, 64),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    w13_scale = torch.full(
        (4, 2 * intermediate_size, 4), 120, device="cuda", dtype=torch.uint8
    )
    w2 = torch.randint(
        0,
        256,
        (4, 128, intermediate_size // 2),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    w2_scale = torch.full(
        (4, 128, intermediate_size // 32), 120, device="cuda", dtype=torch.uint8
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
    weights.w13_weight_scale = w13_scale
    weights.w2_weight = w2
    weights.w2_weight_scale = w2_scale
    weights.top_k = 2
    weights.w13_input_layout = "concatenated"
    weights.activation_situ_beta = 4.0
    weights.activation_situ_linear_beta = 25.0
    plan = kernel_moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation=activation,
        routing_mode="precomputed_topk",
        ispp=intermediate_size,
        internal_activation_dtype="mxfp4",
        solution="triton",
        hidden=None,
        swiglu_form=("standard" if activation == "swiglu" else None),
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
    )
    plan["precision_policy"] = "default"
    plan["output_scale"] = 1.0
    kernel_moe_process_weights(plan, weights)
    actual = kernel_moe_apply(
        plan,
        x,
        weights,
        torch.empty((4, 4), device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    x_packed, x_scale = kernel_quantize_mxfp4(
        x, scale_layout="linear", solution="triton"
    )
    x_dequant = _dequantize(x_packed, x_scale)
    w13_dequant = _dequantize(w13, w13_scale)
    w2_dequant = _dequantize(w2, w2_scale)

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
        intermediate_packed, intermediate_scale = kernel_quantize_mxfp4(
            intermediate, scale_layout="linear", solution="triton"
        )
        intermediate_dequant = _dequantize(intermediate_packed, intermediate_scale)
        expert_output = F.linear(intermediate_dequant, w2_dequant[expert_id]).to(
            torch.bfloat16
        )
        weighted = expert_output.float() * topk_weights[token_ids, slots, None]
        expected.index_add_(0, token_ids, weighted)

    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_triton_mxfp4_moe_defers_local_ep_routes() -> None:
    """Deferred EP keeps local routes canonical and marks remote routes absent."""
    if not current_platform().is_amd:
        pytest.skip("Triton MXFP4 MoE is registered for AMD GPUs")

    generator = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    w13 = torch.randint(
        0, 256, (2, 256, 64), device="cuda", dtype=torch.uint8, generator=generator
    )
    w13_scale = torch.full((2, 256, 4), 120, device="cuda", dtype=torch.uint8)
    w2 = torch.randint(
        0, 256, (2, 128, 64), device="cuda", dtype=torch.uint8, generator=generator
    )
    w2_scale = torch.full((2, 128, 4), 120, device="cuda", dtype=torch.uint8)
    topk_ids = torch.tensor([[0, 2], [3, 1], [2, 3]], device="cuda", dtype=torch.int32)
    topk_weights = torch.tensor(
        [[0.6, 0.4], [0.7, 0.3], [0.2, 0.8]],
        device="cuda",
        dtype=torch.float32,
    )

    weights = torch.nn.Module()
    weights.w13_weight = w13
    weights.w13_weight_scale = w13_scale
    weights.w2_weight = w2
    weights.w2_weight_scale = w2_scale
    weights.top_k = 2
    weights.num_experts = 4
    weights.num_local_experts = 2
    weights.ep_rank = 1
    weights.ep_size = 2
    weights.w13_input_layout = "concatenated"
    plan = kernel_moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="silu",
        requires_deferred_finalize=True,
        routing_mode="precomputed_topk",
        a2a_backend="none",
        ep_size=2,
        ispp=128,
        hidden=128,
        swiglu_form="standard",
        activation_clamped=False,
        expert_id_repeats=False,
        fp8_scale_block_shape=None,
        internal_activation_dtype="mxfp4",
        with_bias=False,
        process_group=None,
        deepep_mode=None,
        deepep_low_latency_max_num_tokens_per_gpu=None,
        persistent_max_num_tokens_per_gpu=None,
        fast_math=False,
        combine_order="rank",
        solution="triton",
    )
    assert plan["supports_deferred_finalize"] is True
    assert plan["supports_prequantized_input"] is False
    kernel_moe_process_weights(plan, weights)
    routes, aux, expanded_idx = kernel_moe_apply(
        plan,
        x,
        weights,
        torch.empty((3, 4), device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_tokens_global=3,
        max_num_tokens_per_gpu=3,
        do_finalize=False,
        low_latency=None,
        overlap_fn=None,
        shared_input=None,
        shared_weight=None,
        shared_out=None,
        x_quantized=None,
    )

    assert aux is None
    expected_idx = torch.tensor([-1, 1, 2, -1, 4, 5], device="cuda", dtype=torch.int32)
    torch.testing.assert_close(expanded_idx, expected_idx, atol=0, rtol=0)
    remote = torch.tensor([0, 3], device="cuda")
    assert torch.count_nonzero(routes.index_select(0, remote)) == 0

    x_packed, x_scale = kernel_quantize_mxfp4(
        x,
        global_scale=None,
        scale_size=32,
        scale_layout="linear",
        enable_pdl=False,
        override=None,
        solution="triton",
    )
    x_dequant = _dequantize(x_packed, x_scale)
    w13_dequant = _dequantize(w13, w13_scale)
    w2_dequant = _dequantize(w2, w2_scale)
    expected = torch.zeros_like(routes)
    for route, global_expert in enumerate(topk_ids.flatten().tolist()):
        if global_expert < 2:
            continue
        token = route // topk_ids.shape[1]
        local_expert = global_expert - 2
        gate_up = F.linear(x_dequant[token], w13_dequant[local_expert])
        gate, up = gate_up.chunk(2, dim=-1)
        activated = (F.silu(gate) * up).to(torch.bfloat16)
        activated_packed, activated_scale = kernel_quantize_mxfp4(
            activated.unsqueeze(0),
            global_scale=None,
            scale_size=32,
            scale_layout="linear",
            enable_pdl=False,
            override=None,
            solution="triton",
        )
        expected[route] = F.linear(
            _dequantize(activated_packed, activated_scale).squeeze(0),
            w2_dequant[local_expert],
        ).to(torch.bfloat16)

    torch.testing.assert_close(routes, expected, rtol=0.02, atol=0.02)
