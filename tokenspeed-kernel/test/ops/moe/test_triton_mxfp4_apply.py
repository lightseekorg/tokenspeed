from __future__ import annotations

from types import SimpleNamespace

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
        features=frozenset(),
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
        features=frozenset(),
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


# GLM-5.3 Flash's experts: text_config.swiglu_limit, hidden_size and
# moe_intermediate_size (2048 over TP8) of /models/GLM-5.3-Flash/config.json.
GLM53_FLASH_SWIGLU_LIMIT = 10.0
GLM53_FLASH_HIDDEN = 4096
GLM53_FLASH_TP8_INTERMEDIATE = 256
GLM53_FLASH_TOP_K = 8


def _glm53_flash_case(num_tokens: int, num_experts: int = 16):
    """GLM-sized experts whose gate/up projections reach well past the limit."""
    generator = torch.Generator(device="cuda").manual_seed(53)
    hidden, inter = GLM53_FLASH_HIDDEN, GLM53_FLASH_TP8_INTERMEDIATE
    x = torch.randn(
        num_tokens, hidden, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    w13 = torch.randint(
        0,
        256,
        (num_experts, 2 * inter, hidden // 2),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    # Scale 2^-3: gate/up have a standard deviation of about 24, so most of
    # them lie outside [-10, 10] and the clamp decides the result.
    w13_scale = torch.full(
        (num_experts, 2 * inter, hidden // 32), 124, device="cuda", dtype=torch.uint8
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden, inter // 2),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    w2_scale = torch.full(
        (num_experts, hidden, inter // 32), 120, device="cuda", dtype=torch.uint8
    )
    scores = torch.rand(num_tokens, num_experts, device="cuda", generator=generator)
    topk_ids = scores.argsort(dim=-1)[:, :GLM53_FLASH_TOP_K].to(torch.int32)
    topk_weights = torch.rand(
        num_tokens, GLM53_FLASH_TOP_K, device="cuda", generator=generator
    )
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return x, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights


def _swiglu_mxfp4_reference(
    x, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights, limit
) -> torch.Tensor:
    """MXFP4 activations and weights, GLM's clamp (glm53_flash.py), then the
    MXFP4 requantization of the intermediate. Both GEMMs sum the exact MXFP4
    products in FP32 and round once to BF16, as the kernel does (a BF16 GEMM
    may reduce in lower precision over the 4096-wide hidden size)."""
    x_packed, x_scale = kernel_quantize_mxfp4(
        x, scale_layout="linear", solution="triton"
    )
    x_dequant = dequantize_mxfp4(x_packed, x_scale).float()
    w13_dequant = dequantize_mxfp4(w13, w13_scale).float()
    w2_dequant = dequantize_mxfp4(w2, w2_scale).float()
    expected = torch.zeros(x.shape, device="cuda", dtype=torch.float32)
    for expert_id in range(w13.shape[0]):
        token_ids, slots = torch.where(topk_ids == expert_id)
        if token_ids.numel() == 0:
            continue
        gate_up = F.linear(x_dequant[token_ids], w13_dequant[expert_id])
        gate, up = gate_up.to(torch.bfloat16).chunk(2, dim=-1)
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        # SiLU as the kernel's epilogue computes it, rounded to BF16.
        silu = (gate.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
        intermediate = (silu * up).to(torch.bfloat16)
        intermediate_packed, intermediate_scale = kernel_quantize_mxfp4(
            intermediate, scale_layout="linear", solution="triton"
        )
        expert_output = F.linear(
            dequantize_mxfp4(intermediate_packed, intermediate_scale).float(),
            w2_dequant[expert_id],
        ).to(torch.bfloat16)
        weighted = expert_output.float() * topk_weights[token_ids, slots, None]
        expected.index_add_(0, token_ids, weighted)
    return expected


def _apply_swiglu_mxfp4(
    x, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights, limit, activation
) -> torch.Tensor:
    weights = torch.nn.Module()
    weights.w13_weight = w13
    weights.w13_weight_scale = w13_scale
    weights.w2_weight = w2
    weights.w2_weight_scale = w2_scale
    weights.top_k = topk_ids.shape[1]
    weights.w13_input_layout = "concatenated"
    weights.activation = activation
    weights.swiglu_arg = (
        None if limit is None else SimpleNamespace(alpha=1.0, limit=limit)
    )
    plan = kernel_moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation=activation,
        routing_mode="precomputed_topk",
        ispp=w2.shape[2] * 2,
        internal_activation_dtype="mxfp4",
        solution="triton",
        hidden=x.shape[1],
        swiglu_form="standard",
        activation_clamped=limit is not None,
        expert_id_repeats=False,
        fast_math=False,
        combine_order="rank",
        features=frozenset(),
    )
    assert plan["apply_kernel_name"] == "triton_mxfp4_precomputed_moe_apply"
    plan["precision_policy"] = "default"
    plan["output_scale"] = 1.0
    kernel_moe_process_weights(plan, weights)
    return kernel_moe_apply(
        plan,
        x,
        weights,
        torch.empty((x.shape[0], w13.shape[0]), device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("activation", ["swiglu", "silu"])
# 8 tokens take the 16-row tiles, 40 tokens the 64-row tiles.
@pytest.mark.parametrize("num_tokens", [8, 40])
def test_triton_mxfp4_moe_applies_glm53_flash_swiglu_clamp(
    activation: str, num_tokens: int
) -> None:
    """MXFP4 activations with GLM-5.3 Flash's clamped SwiGLU (TP8 experts)."""
    if not current_platform().is_amd:
        pytest.skip("Triton MXFP4 MoE is registered for AMD GPUs")
    case = _glm53_flash_case(num_tokens)
    actual = _apply_swiglu_mxfp4(
        *case, limit=GLM53_FLASH_SWIGLU_LIMIT, activation=activation
    ).float()
    expected = _swiglu_mxfp4_reference(*case, limit=GLM53_FLASH_SWIGLU_LIMIT)
    unclamped = _swiglu_mxfp4_reference(*case, limit=None)

    scale = expected.abs().max().item()
    error = (actual - expected).abs()
    print(
        f"glm53 clamped mxfp4 activation={activation} tokens={num_tokens}: "
        f"max|out|={scale:.4g} max_abs_err={error.max().item():.4g} "
        f"max_err/max|out|={error.max().item() / scale:.3g} "
        f"mean_abs_err={error.mean().item():.4g} "
        f"max|unclamped-clamped|={(unclamped - expected).abs().max().item():.4g}"
    )
    # The clamp decides this case: without it the outputs move by more than
    # their own range. The MXFP4 requantization of the intermediate turns a
    # one-ulp BF16 difference of gate/up (the kernel's FP32 sums against the
    # reference's) into a whole E2M1 step, so the bound is statistical:
    # about 3% of max|out| at worst and 0.4% on average here.
    assert (unclamped - expected).abs().max().item() > scale
    assert error.max().item() <= 0.05 * scale
    assert error.mean().item() <= 0.01 * scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_triton_mxfp4_moe_without_limit_stays_unclamped() -> None:
    """The same GLM-sized experts with no limit compute plain SwiGLU."""
    if not current_platform().is_amd:
        pytest.skip("Triton MXFP4 MoE is registered for AMD GPUs")
    case = _glm53_flash_case(40)
    actual = _apply_swiglu_mxfp4(*case, limit=None, activation="swiglu").float()
    expected = _swiglu_mxfp4_reference(*case, limit=None)
    clamped = _swiglu_mxfp4_reference(*case, limit=GLM53_FLASH_SWIGLU_LIMIT)
    scale = expected.abs().max().item()
    error = (actual - expected).abs()
    print(
        f"glm53 unclamped mxfp4: max|out|={scale:.4g} "
        f"max_abs_err={error.max().item():.4g} "
        f"mean_abs_err={error.mean().item():.4g} "
        f"max_err/max|out|={error.max().item() / scale:.3g}"
    )
    assert (clamped - expected).abs().max().item() > 0.5 * scale
    # Unclamped gate/up reach about 100, so the E2M1 steps of the
    # intermediate (and the same requantization noise) are coarser.
    assert error.max().item() <= 0.1 * scale
    assert error.mean().item() <= 0.02 * scale
    # A limit that never binds leaves the result bit-identical.
    inactive = _apply_swiglu_mxfp4(*case, limit=1.0e6, activation="swiglu").float()
    assert torch.equal(inactive, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_triton_mxfp4_moe_refuses_limit_with_situ() -> None:
    if not current_platform().is_amd:
        pytest.skip("Triton MXFP4 MoE is registered for AMD GPUs")
    case = _glm53_flash_case(4, num_experts=8)
    with pytest.raises(ValueError, match="SiTU MoE does not take a SwiGLU limit"):
        _apply_swiglu_mxfp4(*case, limit=GLM53_FLASH_SWIGLU_LIMIT, activation="situ")
