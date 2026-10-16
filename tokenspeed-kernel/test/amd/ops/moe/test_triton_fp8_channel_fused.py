from __future__ import annotations

import pytest
import torch
from ops.moe.test_triton_fp8_channel_apply import _layer, _plan, _plan_on
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.moe import moe_apply
from tokenspeed_kernel.ops.moe._triton import fp8_channel_fused
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.selection import NoKernelFoundError
from utils import (
    assert_no_triton_compile,
    int_specialization_class,
    warm_specialization_classes,
)

APPLY = "triton_fp8_channel_fused_precomputed_moe_apply"


@triton.jit
def _dot_probe_kernel(a_ptr, b_ptr, out_ptr, seed):
    rows = tl.arange(0, 16)
    k = tl.arange(0, 64)
    a = tl.load(a_ptr + rows[:, None] * 64 + k[None, :])
    b = tl.load(b_ptr + rows[:, None] * 64 + k[None, :])
    acc = tl.full((16, 16), seed, tl.float32)
    result = fp8_channel_fused._fp8_dot(a, b.T, acc)
    tl.store(out_ptr + rows[:, None] * 16 + rows[None, :], result)


@triton.jit
def _raw_projection_kernel(a_ptr, b_ptr, out_ptr, K: tl.constexpr, N: tl.constexpr):
    rows = tl.arange(0, 16)
    cols = tl.program_id(0) * 64 + tl.arange(0, 64)
    ks = tl.arange(0, 64)
    acc = tl.zeros((16, 64), tl.float32)
    for start in range(0, K, 64):
        a = tl.load(
            a_ptr + rows[:, None] * K + (start + ks)[None, :],
            rows[:, None] < 2,
            other=0.0,
        )
        b = tl.load(b_ptr + cols[:, None] * K + (start + ks)[None, :])
        acc = fp8_channel_fused._fp8_dot(a, b.T, acc)
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], acc, rows[:, None] < 2)


def test_native_fp8_dot_retains_exact_mixed_magnitude_products():
    _require_gpu()
    out = torch.empty(16, 16, device="cuda")
    # Every K position must preserve a small product beside a large product.
    # The sum is exactly representable in FP32 for every order of addition.
    for first in range(64):
        for second in range(64):
            if first == second:
                continue
            a = torch.zeros(16, 64, device="cuda")
            a[:, first], a[:, second] = 256.0, 0.25
            for sign in (-1.0, 1.0):
                b = a.clone()
                b[:, second] *= sign
                aq, bq = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
                for seed in (0.0, 1.0):
                    _dot_probe_kernel[(1,)](aq, bq, out, seed, num_warps=4)
                    assert torch.all(out == 65536.0 + sign * 0.0625 + seed)


def test_native_fp8_dot_obeys_fp32_rounding_bound():
    _require_gpu()
    generator = torch.Generator(device="cuda").manual_seed(29)
    values = [
        (torch.randn(16, 64, device="cuda", generator=generator) * 100)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        for _ in range(2)
    ]
    a, b = values
    out = torch.empty(16, 16, device="cuda")
    _dot_probe_kernel[(1,)](a, b, out, 0.0, num_warps=4)
    reference = a.double() @ b.double().T
    sum_abs = a.double().abs() @ b.double().abs().T
    unit_roundoff = 2.0**-24
    gamma = 63 * unit_roundoff / (1 - 63 * unit_roundoff)
    assert torch.all((out.double() - reference).abs() <= gamma * sum_abs)


@pytest.mark.parametrize("ispp,ep_size", [(320, 1), (640, 1), (2560, 8)])
def test_fused_channel_plan_admits_beam_partitions(mi350_platform, ispp, ep_size):
    plan = _plan_on(mi350_platform, hidden=5120, ispp=ispp, ep_size=ep_size)
    assert plan["apply_kernel_name"] == APPLY
    assert plan["weight_preprocessor"] is None
    assert plan["supports_precomputed_topk"]
    assert not plan["supports_deferred_finalize"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"ispp": 48},
        {"hidden": 2000},
        {"input_dtype": torch.float16},
        {"with_bias": True},
        {"requires_deferred_finalize": True},
        {"activation_clamped": True},
        {"routing_mode": "kernel_routing"},
        {"activation": "swiglu", "swiglu_form": "generalized"},
    ],
)
def test_fused_channel_plan_refuses_unsupported_contract(mi350_platform, overrides):
    with pytest.raises(NoKernelFoundError):
        _plan_on(mi350_platform, **overrides)


def _require_gpu():
    platform = current_platform()
    if not (
        torch.cuda.is_available()
        and platform.is_amd
        and platform.arch_version == ArchVersion(9, 5)
    ):
        pytest.skip("Fused FP8 channel experts require gfx950")


def test_fused_channel_rejects_unsupported_launch_request():
    from tokenspeed_kernel.ops.moe.triton import (
        triton_fp8_channel_fused_precomputed_moe_apply,
    )

    with pytest.raises(ValueError, match="enable_pdl=True"):
        triton_fp8_channel_fused_precomputed_moe_apply(
            {},
            torch.empty(0, 128, dtype=torch.bfloat16),
            torch.nn.Module(),
            None,
            enable_pdl=True,
        )


def _quantize_reference(x, *, route):
    values = x.float()
    amax = values.abs().amax(dim=1, keepdim=True)
    if route:
        amax = amax.clamp(min=1e-6)
    scales = (amax.double() / 448).float().clamp(min=1 / (448 * 512))
    multiplier = (448 / amax.double()).float().clamp(max=448 * 512)
    scaled = values * multiplier
    if route:
        scaled = scaled.bfloat16().float()
    return scaled.clamp(-448, 448).to(torch.float8_e4m3fn), scales


def _reference(x, layer, weights, ids):
    codes, scales = _quantize_reference(x, route=False)
    intermediate = layer.w2_weight.shape[2]
    route_outputs = torch.zeros(
        (*ids.shape, x.shape[1]), device=x.device, dtype=torch.bfloat16
    )
    local_ids = ids.long() - layer.ep_rank * layer.w13_weight.shape[0]
    for expert in range(layer.w13_weight.shape[0]):
        tokens, slots = torch.where(local_ids == expert)
        if not tokens.numel():
            continue
        fc1 = (codes[tokens].double() @ layer.w13_weight[expert].double().T).float()
        fc1 = (fc1 * scales[tokens]) * layer.w13_weight_scale[expert]
        gate, up = fc1[:, :intermediate], fc1[:, intermediate:]
        activated = (gate * torch.sigmoid(gate) * up).bfloat16()
        act_codes, act_scales = _quantize_reference(activated, route=True)
        fc2 = (act_codes.double() @ layer.w2_weight[expert].double().T).float()
        route_outputs[tokens, slots] = (
            (fc2 * act_scales) * layer.w2_weight_scale[expert]
        ).bfloat16()
    combined = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        combined += route_outputs[:, slot].float() * weights[:, slot, None]
    return combined.bfloat16()


def test_route_quantizer_matches_bf16_before_fp8_boundary():
    _require_gpu()
    # A dense BF16 sweep includes scaled values straddling FP8 midpoints.
    values = (
        torch.arange(1, 4097, device="cuda", dtype=torch.float32)
        .reshape(16, 256)
        .bfloat16()
    )
    values[0] = 0
    values[1] *= 1e-10
    values[2] *= -1e-10
    values[3] *= -1
    actual, scales = fp8_channel_fused._quantize_routes(values)
    expected, expected_scales = _quantize_reference(values, route=True)
    direct, _ = _quantize_reference(values, route=False)
    assert not torch.equal(expected.view(torch.uint8), direct.view(torch.uint8))
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scales, expected_scales.flatten())


def test_fc1_keeps_gate_and_up_in_fp32_until_activation():
    _require_gpu()
    hidden, intermediate = 128, 32
    codes = torch.ones(1, hidden, device="cuda").to(torch.float8_e4m3fn)
    input_scale = torch.ones(1, device="cuda")
    weight = torch.ones(1, 2 * intermediate, hidden, device="cuda").to(
        torch.float8_e4m3fn
    )
    gate = torch.linspace(-3.3, 3.7, intermediate, device="cuda")
    up = torch.linspace(0.71, 2.13, intermediate, device="cuda")
    weight_scale = torch.cat((gate, up)).reshape(1, -1) / hidden
    output = torch.empty(1, intermediate, device="cuda", dtype=torch.bfloat16)
    route_ids = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    counts = torch.ones(1, device="cuda", dtype=torch.int32)
    fp8_channel_fused._expert_matmul_kernel[(1,)](
        codes,
        input_scale,
        weight,
        weight_scale,
        output,
        route_ids,
        counts,
        1,
        1,
        hidden,
        intermediate,
        1,
        1,
        1,
        True,
        16,
        64,
        64,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    expected = (gate * torch.sigmoid(gate) * up).bfloat16()
    rounded_gate, rounded_up = gate.bfloat16().float(), up.bfloat16().float()
    separately_rounded = (
        rounded_gate * torch.sigmoid(rounded_gate) * rounded_up
    ).bfloat16()
    assert not torch.equal(expected, separately_rounded)
    assert torch.equal(output[0], expected)


@pytest.mark.parametrize(
    "intermediate,seed", [(320, 19), (640, 19)] + [(2560, seed) for seed in range(16)]
)
def test_fused_experts_beam_hidden_width_and_top8(intermediate, seed):
    _require_gpu()
    layer = _layer(8, 5120, intermediate, 0, 1)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(2, 5120, device="cuda", generator=generator).bfloat16()
    ids = torch.arange(8, device="cuda", dtype=torch.int32).expand(2, -1).contiguous()
    weights = torch.rand(2, 8, device="cuda", generator=generator) * 0.37
    plan = _plan(hidden=5120, ispp=intermediate)
    actual = moe_apply(plan, x, layer, None, topk_weights=weights, topk_ids=ids)
    expected = _reference(x, layer, weights, ids)
    # Independent FP32 reduction orders can cross a BF16 midpoint, even when
    # every stage obeys its rounding contract. Final elementwise thresholds
    # below one BF16 step would incorrectly require identical reduction orders.
    relative = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative < 2e-3


def test_seed1_fc2_midpoint_crossings_obey_stage_contract():
    _require_gpu()
    from tokenspeed_kernel.ops.moe._triton.common import (
        _combine,
        _prepare_routed_output,
    )
    from tokenspeed_kernel.ops.quantization import quantize_fp8

    hidden, intermediate, experts, topk = 5120, 2560, 8, 8
    layer = _layer(experts, hidden, intermediate, 0, 1)
    generator = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(2, hidden, device="cuda", generator=generator).bfloat16()
    weights = torch.rand(2, topk, device="cuda", generator=generator) * 0.37
    ids = (
        torch.arange(topk, device="cuda", dtype=torch.int32).expand(2, -1).contiguous()
    )
    routes, counts, routed, output = _prepare_routed_output(x, ids, experts)
    codes, scales = quantize_fp8(
        x, granularity="token", scale_encoding="float32", enable_pdl=False
    )
    activated = torch.zeros(2 * topk, intermediate, device="cuda", dtype=torch.bfloat16)

    def stage(values, token_scales, weight, weight_scales, destination, first):
        fp8_channel_fused._expert_matmul_kernel[(256,)](
            values,
            token_scales,
            weight,
            weight_scales,
            destination,
            routes,
            counts,
            2,
            256,
            hidden,
            intermediate,
            experts,
            8,
            topk,
            first,
            16,
            64,
            64,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )

    stage(codes, scales, layer.w13_weight, layer.w13_weight_scale, activated, True)
    act_codes, act_scales = fp8_channel_fused._quantize_routes(activated)
    expected_codes, expected_scales = _quantize_reference(activated, route=True)
    assert torch.equal(act_codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(act_scales, expected_scales.flatten())
    stage(act_codes, act_scales, layer.w2_weight, layer.w2_weight_scale, routed, False)
    gamma = (intermediate - 1) * 2.0**-24 / (1 - (intermediate - 1) * 2.0**-24)
    for expert in range(experts):
        indices = torch.arange(2, device="cuda") * topk + expert
        values = act_codes[indices].contiguous()
        raw = torch.empty(2, hidden, device="cuda")
        _raw_projection_kernel[(hidden // 64,)](
            values,
            layer.w2_weight[expert],
            raw,
            intermediate,
            hidden,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
        exact = values.double() @ layer.w2_weight[expert].double().T
        sum_abs = values.double().abs() @ layer.w2_weight[expert].double().abs().T
        assert torch.all((raw.double() - exact).abs() <= gamma * sum_abs)
        scaled = (raw * act_scales[indices, None]) * layer.w2_weight_scale[expert]
        expected = (exact.float() * act_scales[indices, None]) * layer.w2_weight_scale[
            expert
        ]
        assert torch.equal(routed[indices], scaled.bfloat16())
        different = scaled.bfloat16() != expected.bfloat16()
        if different.any():
            actual_bf16, expected_bf16 = (
                scaled.bfloat16().float(),
                expected.bfloat16().float(),
            )
            midpoint = (actual_bf16 + expected_bf16) * 0.5
            assert torch.all(
                ((scaled - midpoint) * (expected - midpoint))[different] <= 0
            )
    _combine(routed, weights, output)
    combined = torch.zeros_like(output, dtype=torch.float32)
    for slot in range(topk):
        values = routed.reshape(2, topk, hidden)[:, slot].double()
        combined = (
            combined.double() + values * weights[:, slot, None].double()
        ).float()
    assert torch.equal(output, combined.bfloat16())


@pytest.mark.parametrize("tokens", [0, 1, 2, 4, 8, 16, 65])
@pytest.mark.parametrize(
    "intermediate,ep_rank,ep_size",
    [(320, 0, 1), (352, 0, 1), (640, 1, 2), (2560, 0, 1)],
)
def test_fused_experts_match_rounding_reference(tokens, intermediate, ep_rank, ep_size):
    _require_gpu()
    hidden, experts, topk = 256, 4, 3
    layer = _layer(experts, hidden, intermediate, ep_rank, ep_size)
    generator = torch.Generator(device="cuda").manual_seed(19)
    x = torch.randn(tokens, hidden, device="cuda", generator=generator).bfloat16()
    ids = torch.randint(
        -1,
        experts * ep_size + 1,
        (tokens, topk),
        device="cuda",
        generator=generator,
        dtype=torch.int32,
    )
    weights = torch.rand(tokens, topk, device="cuda", generator=generator) * 2.5
    plan = _plan(hidden=hidden, ispp=intermediate, ep_size=ep_size)
    actual = moe_apply(plan, x, layer, None, topk_weights=weights, topk_ids=ids)
    if not tokens:
        assert actual.shape == x.shape
        return
    expected = _reference(x, layer, weights, ids)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    relative = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp(min=1e-30)
    assert relative < 2e-3
    outside = ((ids < ep_rank * experts) | (ids >= (ep_rank + 1) * experts)).all(dim=1)
    assert torch.count_nonzero(actual[outside]) == 0


def test_fused_expert_graph_refreshes_routes_and_inputs():
    _require_gpu()
    layer = _layer(4, 256, 320, 0, 1)
    x = torch.randn(7, 256, device="cuda").bfloat16()
    ids = torch.zeros(7, 2, device="cuda", dtype=torch.int32)
    weights = torch.rand(7, 2, device="cuda")
    plan = _plan(hidden=256, ispp=320)

    def run():
        return moe_apply(plan, x, layer, None, topk_weights=weights, topk_ids=ids)

    eager = run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    assert torch.equal(captured, eager)
    ids.fill_(3)
    x.neg_()
    weights.mul_(0.37)
    graph.replay()
    assert torch.equal(captured, run())


@pytest.mark.parametrize("experts,ep_size", [(32, 8), (64, 4)])
def test_fused_expert_last_ep_rank_owns_only_local_ids(experts, ep_size):
    _require_gpu()
    rank = ep_size - 1
    layer = _layer(experts, 256, 320, rank, ep_size)
    start = rank * experts
    x = torch.randn(4, 256, device="cuda").bfloat16()
    ids = torch.tensor(
        [
            [start, start + experts - 1, start - 1],
            [start - 1, -1, start + experts],
            [start + experts - 1, start, 0],
            [start, start, start + experts - 1],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    weights = torch.tensor([[0.1234567, 0.8765432, 0.2345678]] * 4, device="cuda")
    plan = _plan(hidden=256, ispp=320, ep_size=ep_size)
    actual = moe_apply(plan, x, layer, None, topk_weights=weights, topk_ids=ids)
    expected = _reference(x, layer, weights, ids)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    assert torch.count_nonzero(actual[1]) == 0


def test_fused_expert_token_count_does_not_recompile():
    _require_gpu()
    from tokenspeed_kernel.ops.moe._triton import common
    from tokenspeed_kernel.ops.quantization import triton as quantization

    layer = _layer(4, 256, 320, 0, 1)
    x = torch.randn(130, 256, device="cuda").bfloat16()
    ids = torch.zeros(130, 2, device="cuda", dtype=torch.int32)
    weights = torch.rand(130, 2, device="cuda")
    plan = _plan(hidden=256, ispp=320)

    def run(rows):
        return moe_apply(
            plan,
            x[:rows],
            layer,
            None,
            topk_weights=weights[:rows],
            topk_ids=ids[:rows],
        )

    def key(rows):
        return (
            int_specialization_class(rows),
            int_specialization_class(rows * 2),
            rows * 2 <= 128,
        )

    sweep = (1, 3, 7, 16, 31, 63, 65, 97, 129)
    warm_specialization_classes(run, key, sweep, range(1, 130))
    with assert_no_triton_compile(
        fp8_channel_fused._expert_matmul_kernel,
        fp8_channel_fused._route_quantize_kernel,
        common._routing_kernel,
        common._combine_kernel,
        quantization._fp8_token_quantize_kernel,
    ):
        for rows in sweep:
            actual = run(rows)
            torch.testing.assert_close(
                actual,
                _reference(x[:rows], layer, weights[:rows], ids[:rows]),
                atol=2e-3,
                rtol=2e-3,
            )
