from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.moe import torch_moe_experts
from tokenspeed_kernel.numerics.torch.quantization import (
    quantize_fp8_rows,
    round_to_bf16_exact,
)
from tokenspeed_kernel.ops.moe import moe_apply as kernel_moe_apply
from tokenspeed_kernel.ops.moe import moe_plan as kernel_moe_plan
from tokenspeed_kernel.platform import ArchVersion, Platform, current_platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import NoKernelFoundError
from utils import (
    assert_no_triton_compile,
    int_specialization_class,
    warm_specialization_classes,
)

APPLY = "triton_fp8_channel_precomputed_moe_apply"


def _skip_unless_supported() -> None:
    platform = current_platform()
    if not (
        torch.cuda.is_available()
        and platform.is_nvidia
        and ArchVersion(9, 0) <= platform.arch_version <= ArchVersion(10, 3)
    ):
        pytest.skip("Triton FP8 channel MoE requires NVIDIA SM90 to SM103")


def _plan(**overrides) -> dict:
    kwargs = dict(
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=192,
        hidden=2048,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
        # On SM100 and SM103 FlashInfer's TRT-LLM kernel ranks first.
        solution="triton",
    )
    kwargs.update(overrides)
    return kernel_moe_plan("fp8_channel", **kwargs)


def _plan_on(platform, **overrides) -> dict:
    registry = KernelRegistry.get()
    real_platform = Platform.get()
    try:
        Platform.override(platform)
        registry.clear_cache()
        return _plan(**overrides)
    finally:
        Platform.override(real_platform)
        registry.clear_cache()


@pytest.mark.parametrize(
    "platform_fixture", ["h100_platform", "b200_platform", "b300_platform"]
)
@pytest.mark.parametrize("ispp,ep_size", [(192, 1), (384, 1), (1536, 8)])
def test_fp8_channel_plan_selects_the_triton_kernel(
    platform_fixture: str, ispp: int, ep_size: int, request: pytest.FixtureRequest
) -> None:
    plan = _plan_on(
        request.getfixturevalue(platform_fixture), ispp=ispp, ep_size=ep_size
    )

    assert plan["apply_kernel_name"] == APPLY
    assert plan["weight_preprocessor"] is None
    assert plan["supports_precomputed_topk"]
    assert not plan["support_routing"]
    assert not plan["supports_deferred_finalize"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"routing_mode": "kernel_routing"},
        {"requires_deferred_finalize": True},
        {"ispp": 48},
        {"hidden": 2000},
        {"activation_clamped": True},
        {"activation": "swiglu", "swiglu_form": "generalized"},
        {"with_bias": True},
        {"input_dtype": torch.float16},
    ],
)
def test_fp8_channel_plan_refuses_layers_the_kernel_cannot_run(
    overrides: dict, b200_platform
) -> None:
    with pytest.raises(NoKernelFoundError):
        _plan_on(b200_platform, **overrides)


def test_fp8_channel_plan_needs_sm90_or_newer(a100_platform) -> None:
    with pytest.raises(NoKernelFoundError):
        _plan_on(a100_platform)


# ---------------------------------------------------------------------------
# GPU numerics against an FP64 emulation of the scheme's rounding points.
# ---------------------------------------------------------------------------


def _moe_reference(x, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, offset):
    def dequantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return weight.double() * scale.double()[..., None]

    def activation(fc1: torch.Tensor) -> torch.Tensor:
        gate, up = round_to_bf16_exact(fc1).chunk(2, dim=-1)
        return round_to_bf16_exact(gate * torch.sigmoid(gate) * up)

    def quantize(values: torch.Tensor) -> torch.Tensor:
        codes, scale = quantize_fp8_rows(values)
        return codes.double() * scale.double()[:, None]

    output = torch_moe_experts(
        x,
        dequantize(w13, w13_scale),
        dequantize(w2, w2_scale),
        topk_ids.long(),
        topk_weights,
        activation=activation,
        w13_bias=None,
        w2_bias=None,
        quantize_input=quantize,
        quantize_intermediate=quantize,
        round_expert_output=round_to_bf16_exact,
        first_expert=offset,
    )
    return round_to_bf16_exact(output)


def _quantized(shape: tuple[int, ...], generator: torch.Generator):
    weight = torch.randn(shape, device="cuda", generator=generator) * 0.02
    scale = (weight.abs().amax(dim=-1) / 448.0).clamp(min=2.0**-126)
    codes = (weight / scale[..., None]).to(torch.float8_e4m3fn)
    return codes.contiguous(), scale.contiguous()


def _layer(num_experts, hidden, ispp, ep_rank, ep_size, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    layer = torch.nn.Module()
    layer.w13_weight, layer.w13_weight_scale = _quantized(
        (num_experts, 2 * ispp, hidden), generator
    )
    layer.w2_weight, layer.w2_weight_scale = _quantized(
        (num_experts, hidden, ispp), generator
    )
    layer.ep_rank, layer.ep_size = ep_rank, ep_size
    return layer


def test_fp8_channel_row_quantization_is_exact() -> None:
    _skip_unless_supported()
    from tokenspeed_kernel.ops.moe._triton.fp8_channel import _quantize_rows

    generator = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(6, 192, device="cuda", generator=generator) * 3
    x[0] = 0
    x[1, :8] = torch.tensor([448.0, 6.5, 7.5, 13.0, 15.0, 26.0, 30.0, 0.0078125])
    x = x.to(torch.bfloat16)
    expert_ids = torch.tensor([0, 1, -1, 3, 1, 9], device="cuda", dtype=torch.int32)

    codes, scales = _quantize_rows(x)
    expected_codes, expected_scales = quantize_fp8_rows(x)
    assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(scales, expected_scales)

    routed_codes, routed_scales = _quantize_rows(x, expert_ids, 4)
    valid = (expert_ids >= 0) & (expert_ids < 4)
    assert torch.equal(
        routed_codes[valid].view(torch.uint8), expected_codes[valid].view(torch.uint8)
    )
    assert torch.equal(routed_scales[valid], expected_scales[valid])


@pytest.mark.parametrize("num_tokens", [1, 5, 20])
@pytest.mark.parametrize("ispp,ep_rank,ep_size", [(64, 0, 1), (96, 0, 1), (64, 1, 2)])
@pytest.mark.parametrize("use_tma_gather", [None, False])
def test_fp8_channel_moe_matches_fp64(
    num_tokens: int, ispp: int, ep_rank: int, ep_size: int, use_tma_gather
) -> None:
    _skip_unless_supported()
    from tokenspeed_kernel.ops.moe._triton import fp8_channel

    hidden, num_experts, top_k = 256, 4, 3
    layer = _layer(num_experts, hidden, ispp, ep_rank, ep_size)
    generator = torch.Generator(device="cuda").manual_seed(num_tokens)
    x = (torch.randn(num_tokens, hidden, device="cuda", generator=generator) * 2).to(
        torch.bfloat16
    )
    logits = torch.randn(
        num_tokens, num_experts * ep_size, device="cuda", generator=generator
    )
    topk_weights, topk_ids = torch.topk(logits.softmax(dim=-1), top_k, dim=-1)
    topk_weights = (topk_weights * 2.5).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    offset = ep_rank * num_experts

    if use_tma_gather is None:
        plan = _plan(ispp=ispp, hidden=hidden, ep_size=ep_size)
        actual = kernel_moe_apply(
            plan, x, layer, None, topk_weights=topk_weights, topk_ids=topk_ids
        )
    else:
        actual = fp8_channel._triton_fp8_channel_moe_impl(
            x,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            offset,
            use_tma_gather=use_tma_gather,
        )
    expected = _moe_reference(
        x,
        layer.w13_weight,
        layer.w13_weight_scale,
        layer.w2_weight,
        layer.w2_weight_scale,
        topk_weights,
        topk_ids,
        offset,
    )

    local = ((topk_ids - offset >= 0) & (topk_ids - offset < num_experts)).any(dim=1)
    assert torch.all(actual[~local] == 0)
    error = (actual.double() - expected).norm() / expected.norm().clamp(min=1e-30)
    # FP32 accumulation order may flip a rare BF16 or E4M3 rounding.
    assert error < 2e-3


def test_fp8_channel_moe_graph_replay_matches_eager() -> None:
    _skip_unless_supported()
    hidden, ispp, num_experts, top_k, num_tokens = 256, 64, 4, 2, 7
    layer = _layer(num_experts, hidden, ispp, 0, 1, seed=5)
    generator = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn(num_tokens, hidden, device="cuda", generator=generator).to(
        torch.bfloat16
    )
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), device="cuda", generator=generator
    ).to(torch.int32)
    topk_weights = torch.rand(num_tokens, top_k, device="cuda", generator=generator)
    plan = _plan(ispp=ispp, hidden=hidden)

    def run():
        return kernel_moe_apply(
            plan, x, layer, None, topk_weights=topk_weights, topk_ids=topk_ids
        )

    eager = run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)
    x.copy_(-x)
    graph.replay()
    assert torch.equal(captured, run())


def test_fp8_channel_moe_token_count_does_not_recompile() -> None:
    _skip_unless_supported()
    from tokenspeed_kernel._triton import triton
    from tokenspeed_kernel.ops.moe._triton import common as _common
    from tokenspeed_kernel.ops.moe._triton import fp8_channel

    hidden, ispp, num_experts, top_k, max_tokens = 256, 64, 4, 2, 1500
    layer = _layer(num_experts, hidden, ispp, 0, 1, seed=7)
    generator = torch.Generator(device="cuda").manual_seed(13)
    x = torch.randn(max_tokens, hidden, device="cuda", generator=generator).to(
        torch.bfloat16
    )
    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, device="cuda", generator=generator)[:top_k]
            for _ in range(max_tokens)
        ]
    ).to(torch.int32)
    topk_weights = torch.rand(
        max_tokens, top_k, device="cuda", dtype=torch.float32, generator=generator
    )
    plan = _plan(ispp=ispp, hidden=hidden)
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count

    def run(tokens):
        return kernel_moe_apply(
            plan,
            x[:tokens],
            layer,
            None,
            topk_weights=topk_weights[:tokens],
            topk_ids=topk_ids[:tokens],
        )

    def key(tokens):
        routes = tokens * top_k
        config = fp8_channel._launch_config(tokens, hidden, ispp)
        programs = (
            min(
                config[f"{stage}_ctas_per_sm"] * num_sms,
                routes * triton.cdiv(size, config[f"{stage}_block_n"]),
            )
            for stage, size in (("fc1", ispp), ("fc2", hidden))
        )
        # The launch config is bucketed by token count; up to 128 routes take
        # the small routing block.
        return (
            *map(int_specialization_class, (tokens, routes, *programs)),
            tuple(config.values()),
            routes <= 128,
        )

    # Each token's output depends only on its own routes, so a shorter batch
    # is compared with the prefix of the FP64 emulation of the full one.
    expected = _moe_reference(
        x,
        layer.w13_weight,
        layer.w13_weight_scale,
        layer.w2_weight,
        layer.w2_weight_scale,
        topk_weights,
        topk_ids,
        0,
    )
    sweep = (1, 3, 7, 12, 24, 100, 300, 700, 1000, 1483)
    warm_specialization_classes(run, key, sweep, range(1, max_tokens))
    with assert_no_triton_compile(
        fp8_channel._fc1_kernel,
        fp8_channel._fc2_kernel,
        fp8_channel._quantize_rows_kernel,
        _common._routing_kernel,
        _common._combine_kernel,
    ):
        for tokens in sweep:
            actual = run(tokens).double()
            reference = expected[:tokens]
            error = (actual - reference).norm() / reference.norm().clamp(min=1e-30)
            assert error < 2e-3
