# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The gfx1250 warp-pipelined FP8 x MXFP4 MoE kernels against a PyTorch
reference over dequantized weights and against the A8W4 fused MoE."""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import pytest
import torch
from tokenspeed_kernel.numerics.torch.activation import swiglu, torch_situ_and_mul
from tokenspeed_kernel.numerics.torch.moe import DequantizedExperts, torch_moe_experts
from tokenspeed_kernel.numerics.torch.quantization import dequantize_mxfp4
from utils import assert_no_triton_compile, is_cdna5, make_mxfp4_moe_weights

if not is_cdna5():
    pytest.skip(
        "AMD CDNA5 is required for the gfx1250 warp-pipelined MoE tests",
        allow_module_level=True,
    )

from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4 import (  # noqa: E402
    decode_warp_pipelined,
    down_warp_pipelined,
    gate_up_warp_pipelined,
    warp_pipelined,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.fused import (  # noqa: E402
    gluon_mxfp_precomputed_mxfp4_fused_moe,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.weight_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx1250_moe_weights,
)

_SWIGLU = dict(swiglu_alpha=1.702, swiglu_limit=7.0, swiglu_beta=1.0)
_SITU = dict(situ_beta=4.0, situ_linear_beta=25.0)


class Case(NamedTuple):
    tokens: int
    experts: int
    hidden: int
    intermediate: int
    top_k: int
    activation: str
    bias: bool
    static_scales: bool


class Layer(NamedTuple):
    module: torch.nn.Module
    raw: dict[str, torch.Tensor]
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor


def _make_layer(case: Case, seed: int) -> Layer:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    raw = make_mxfp4_moe_weights(
        case.experts, case.hidden, case.intermediate, generator
    )
    module = torch.nn.Module()
    for name, key in (
        ("w13_weight", "w13_weight"),
        ("w13_weight_scale", "w13_scale"),
        ("w2_weight", "w2_weight"),
        ("w2_weight_scale", "w2_scale"),
    ):
        setattr(module, name, torch.nn.Parameter(raw[key].clone(), requires_grad=False))
    if case.bias:
        raw["w13_bias"] = 0.05 * torch.randn(
            case.experts, 2 * case.intermediate, device="cuda", generator=generator
        )
        raw["w2_bias"] = 0.05 * torch.randn(
            case.experts, case.hidden, device="cuda", generator=generator
        )
        module.w13_weight_bias = torch.nn.Parameter(
            raw["w13_bias"].clone(), requires_grad=False
        )
        module.w2_weight_bias = torch.nn.Parameter(
            raw["w2_bias"].clone(), requires_grad=False
        )
    if case.static_scales:
        module.w13_input_scale = torch.nn.Parameter(
            torch.tensor([0.5 / 448.0], device="cuda"), requires_grad=False
        )
        module.w2_input_scale = torch.nn.Parameter(
            torch.tensor([0.25 / 448.0], device="cuda"), requires_grad=False
        )
    preprocess_gluon_mxfp4_gfx1250_moe_weights({}, module)

    hidden_states = 0.1 * torch.randn(
        case.tokens,
        case.hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    # Distinct random experts per token, so routes cover every expert unevenly.
    scores = torch.rand(case.tokens, case.experts, device="cuda", generator=generator)
    topk_ids = scores.topk(case.top_k, dim=-1).indices.to(torch.int32)
    topk_weights = torch.softmax(
        torch.rand(case.tokens, case.top_k, device="cuda", generator=generator),
        dim=-1,
    )
    return Layer(module, raw, hidden_states, topk_weights, topk_ids)


def _activation(case: Case):
    return warp_pipelined.make_activation(case.activation, **_SWIGLU, **_SITU)


def _run_warp_pipelined(
    layer: Layer,
    case: Case,
    *,
    decode: bool,
    gate_up_tiles=None,
    down_tiles=None,
) -> torch.Tensor:
    module = layer.module
    return warp_pipelined.gluon_mxfp_precomputed_mxfp4_warp_pipelined_moe(
        layer.hidden_states,
        layer.topk_weights,
        layer.topk_ids,
        module.w13_weight_triton_tensor,
        module.w2_weight_triton_tensor,
        w13_mx_scale=module.w13_precision_config.b_mx_scale,
        w2_mx_scale=module.w2_precision_config.b_mx_scale,
        w13_bias=getattr(module, "w13_weight_bias", None),
        w2_bias=getattr(module, "w2_weight_bias", None),
        activation=_activation(case),
        out_dtype=torch.bfloat16,
        decode=decode,
        gate_up_tiles=gate_up_tiles,
        down_tiles=down_tiles,
    )


def _run_fused_a8w4(layer: Layer, case: Case) -> torch.Tensor:
    module = layer.module
    activation_args = _SITU if case.activation == "situ" else _SWIGLU
    return gluon_mxfp_precomputed_mxfp4_fused_moe(
        layer.hidden_states,
        layer.topk_weights,
        layer.topk_ids,
        module.w13_weight_triton_tensor,
        module.w2_weight_triton_tensor,
        w13_mx_scale=module.w13_precision_config.b_mx_scale,
        w2_mx_scale=module.w2_precision_config.b_mx_scale,
        w13_bias=getattr(module, "w13_weight_bias", None),
        w2_bias=getattr(module, "w2_weight_bias", None),
        activation_dtype="fp8",
        activation=case.activation,
        **activation_args,
    )


def _reference(layer: Layer, case: Case) -> torch.Tensor:
    """Each GEMM input is fake-quantized to E4M3 by its static scale (1 when
    absent); each route's output rounds to BF16 before the weighted sum."""
    raw = layer.raw
    x_scale = layer.module.w13_weight_triton_tensor.act_scale
    h_scale = layer.module.w2_weight_triton_tensor.act_scale

    def fake_fp8(scale: torch.Tensor | None):
        def quantize(values: torch.Tensor) -> torch.Tensor:
            if scale is None:
                return values.float().to(torch.float8_e4m3fn).float()
            q = (values.float() / scale).to(torch.float8_e4m3fn)
            return q.float() * scale

        return quantize

    if case.activation == "situ":

        def activation(gate_up: torch.Tensor) -> torch.Tensor:
            # The kernel rounds the gate/up product to BF16 before SiTU.
            return torch_situ_and_mul(
                gate_up.bfloat16(), _SITU["situ_beta"], _SITU["situ_linear_beta"]
            ).float()

    else:

        def activation(gate_up: torch.Tensor) -> torch.Tensor:
            return swiglu(
                gate_up,
                alpha=_SWIGLU["swiglu_alpha"],
                limit=_SWIGLU["swiglu_limit"],
                beta=_SWIGLU["swiglu_beta"],
            )

    return torch_moe_experts(
        layer.hidden_states,
        DequantizedExperts(dequantize_mxfp4, raw["w13_weight"], raw["w13_scale"]),
        DequantizedExperts(dequantize_mxfp4, raw["w2_weight"], raw["w2_scale"]),
        layer.topk_ids,
        layer.topk_weights,
        activation=activation,
        w13_bias=raw.get("w13_bias"),
        w2_bias=raw.get("w2_bias"),
        quantize_input=fake_fp8(x_scale),
        quantize_intermediate=fake_fp8(h_scale),
        round_expert_output=torch.Tensor.bfloat16,
        first_expert=0,
    ).bfloat16()


def _assert_matches(actual: torch.Tensor, layer: Layer, case: Case) -> None:
    torch.cuda.synchronize()
    expected = _reference(layer, case)
    assert torch.count_nonzero(expected).item() > 0
    # FP8 intermediates flip by one unit where the accumulation order differs.
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=5e-2)
    torch.testing.assert_close(
        actual, _run_fused_a8w4(layer, case), atol=2e-2, rtol=5e-2
    )


PREFILL_CASES = [
    # Whole 128 x 512 x 256 tiles in both GEMMs.
    Case(512, 8, 1024, 1024, 4, "swiglu", False, False),
    # Partial M tiles, a partial N tile in both GEMMs (gate/up N = 1792,
    # down N = 1152) and a partial K tile in the down GEMM (K = 896), with bias.
    Case(300, 8, 1152, 896, 4, "swiglu", True, False),
    # SiTU and static FP8 activation scales; experts left without rows.
    Case(96, 16, 1024, 768, 2, "situ", False, True),
]


@pytest.mark.parametrize("case", PREFILL_CASES, ids=str)
@pytest.mark.parametrize("down_persistent", [False, True])
@pytest.mark.parametrize("partitioned", [False, True])
def test_prefill_matches_reference_gfx1250(
    case: Case, down_persistent: bool, partitioned: bool
) -> None:
    layer = _make_layer(case, seed=20261116 + case.tokens)
    actual = _run_warp_pipelined(
        layer,
        case,
        decode=False,
        gate_up_tiles=replace(
            warp_pipelined.PREFILL_GATE_UP_TILES,
            resolve_partition_conflicts=partitioned,
        ),
        down_tiles=replace(
            warp_pipelined.PREFILL_DOWN_TILES,
            persistent=down_persistent,
            resolve_partition_conflicts=partitioned,
        ),
    )
    _assert_matches(actual, layer, case)


# Partitioned LDS changes where each value lands, not which values the WMMAs
# reduce, so the results match the unpartitioned kernels bit for bit; this
# also covers 64-row tiles, wide gather indices and two-workgroup clusters.
@pytest.mark.parametrize(
    ("block_m", "cta_m", "tokens"),
    [(128, 1, 512), (64, 1, 512), (64, 2, 1024), (128, 1, 66_000)],
)
def test_partitioned_gate_up_matches_unpartitioned_gfx1250(
    block_m: int, cta_m: int, tokens: int
) -> None:
    case = Case(
        tokens, 4, 1024, 768, 1 if tokens > 65_535 else 4, "swiglu", True, False
    )
    layer = _make_layer(case, seed=17)
    gate_up = replace(
        warp_pipelined.PREFILL_GATE_UP_TILES, block_m=block_m, cta_m=cta_m
    )
    plain, partitioned = (
        _run_warp_pipelined(
            layer,
            case,
            decode=False,
            gate_up_tiles=replace(gate_up, resolve_partition_conflicts=flag),
            down_tiles=warp_pipelined.PREFILL_DOWN_TILES,
        )
        for flag in (False, True)
    )
    torch.cuda.synchronize()
    assert torch.count_nonzero(plain).item() > 0
    torch.testing.assert_close(partitioned, plain, atol=0.0, rtol=0.0)


# Down K of 384 (Kimi K3 TP8, a partial second K tile) and 512 (GLM-5.3-Flash)
# fit only a two-slot ring.
@pytest.mark.parametrize("intermediate", [384, 512])
@pytest.mark.parametrize("down_persistent", [False, True])
def test_prefill_short_down_k_matches_reference_gfx1250(
    intermediate: int, down_persistent: bool
) -> None:
    case = Case(256, 16, 1024, intermediate, 4, "situ", True, False)
    layer = _make_layer(case, seed=intermediate)
    actual = _run_warp_pipelined(
        layer,
        case,
        decode=False,
        gate_up_tiles=warp_pipelined.PREFILL_GATE_UP_TILES,
        down_tiles=replace(
            warp_pipelined.PREFILL_DOWN_TILES,
            num_buffers=2,
            persistent=down_persistent,
        ),
    )
    _assert_matches(actual, layer, case)


@pytest.mark.parametrize("cta_m", [2, 4])
def test_prefill_cluster_multicast_matches_single_workgroup_gfx1250(cta_m: int) -> None:
    case = Case(1024, 4, 1024, 1024, 4, "swiglu", True, False)
    layer = _make_layer(case, seed=7)
    tiles = dict(
        gate_up_tiles=warp_pipelined.PREFILL_GATE_UP_TILES,
        down_tiles=warp_pipelined.PREFILL_DOWN_TILES,
    )
    single = _run_warp_pipelined(layer, case, decode=False, **tiles)
    clustered = _run_warp_pipelined(
        layer,
        case,
        decode=False,
        gate_up_tiles=replace(
            tiles["gate_up_tiles"], block_m=256 // cta_m, cta_m=cta_m
        ),
        down_tiles=replace(tiles["down_tiles"], block_m=256 // cta_m, cta_m=cta_m),
    )
    torch.cuda.synchronize()
    # A cluster tile computes each row with the same K order as one workgroup.
    torch.testing.assert_close(clustered, single, atol=0.0, rtol=0.0)


DECODE_CASES = [
    Case(1, 32, 2048, 1536, 4, "swiglu", False, False),
    Case(4, 32, 2048, 1536, 4, "swiglu", True, False),
    # A partial last K tile in the gate/up GEMM (K = 2304) and SiTU.
    Case(16, 32, 2304, 1536, 8, "situ", False, True),
]


@pytest.mark.parametrize("case", DECODE_CASES, ids=str)
@pytest.mark.parametrize("async_store", [False, True])
def test_decode_matches_reference_gfx1250(case: Case, async_store: bool) -> None:
    layer = _make_layer(case, seed=31 + case.tokens)
    actual = _run_warp_pipelined(
        layer,
        case,
        decode=True,
        gate_up_tiles=warp_pipelined.DECODE_TILES,
        down_tiles=replace(warp_pipelined.DECODE_TILES, async_store=async_store),
    )
    _assert_matches(actual, layer, case)


def test_decode_short_down_k_matches_reference_gfx1250() -> None:
    # Kimi K3 TP8 shards the intermediate to 384: three 128-wide K tiles.
    case = Case(4, 16, 3584, 384, 8, "situ", False, False)
    layer = _make_layer(case, seed=5)
    short_k = warp_pipelined.select_decode_tiles(case.intermediate)
    assert short_k.block_k == 128
    actual = _run_warp_pipelined(
        layer,
        case,
        decode=True,
        gate_up_tiles=warp_pipelined.DECODE_TILES,
        down_tiles=short_k,
    )
    _assert_matches(actual, layer, case)


def test_prefill_wide_gather_indices_match_reference_gfx1250() -> None:
    # More tokens than 16-bit gather indices address: tiles index relative to
    # their lowest row.
    case = Case(66_000, 4, 1024, 768, 1, "swiglu", False, False)
    layer = _make_layer(case, seed=11)
    actual = _run_warp_pipelined(
        layer,
        case,
        decode=False,
        gate_up_tiles=warp_pipelined.PREFILL_GATE_UP_TILES,
        down_tiles=warp_pipelined.PREFILL_DOWN_TILES,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, _reference(layer, case), atol=2e-2, rtol=5e-2)


# A real 64-bit offset needs a multi-gigabyte weight slab, so force the
# predicate instead.
@pytest.mark.parametrize("decode", [False, True])
def test_upcast_indices_match_narrow_indices_gfx1250(
    decode: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = (DECODE_CASES if decode else PREFILL_CASES)[1]
    layer = _make_layer(case, seed=3)
    tiles = (
        dict(
            gate_up_tiles=warp_pipelined.DECODE_TILES,
            down_tiles=warp_pipelined.DECODE_TILES,
        )
        if decode
        else dict(
            gate_up_tiles=warp_pipelined.PREFILL_GATE_UP_TILES,
            down_tiles=warp_pipelined.PREFILL_DOWN_TILES,
        )
    )
    narrow = _run_warp_pipelined(layer, case, decode=decode, **tiles)
    monkeypatch.setattr(warp_pipelined, "should_upcast_indices", lambda *args: True)
    wide = _run_warp_pipelined(layer, case, decode=decode, **tiles)
    torch.cuda.synchronize()
    assert torch.count_nonzero(narrow).item() > 0
    torch.testing.assert_close(wide, narrow, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("decode", [False, True])
def test_token_count_does_not_recompile_gfx1250(decode: bool) -> None:
    case = Case(0, 16, 2048, 1536, 4, "swiglu", False, False)
    tiles = (
        dict(
            gate_up_tiles=warp_pipelined.DECODE_TILES,
            down_tiles=warp_pipelined.DECODE_TILES,
        )
        if decode
        else dict(
            gate_up_tiles=warp_pipelined.PREFILL_GATE_UP_TILES,
            down_tiles=warp_pipelined.PREFILL_DOWN_TILES,
        )
    )
    warm, sweep = ((16, 1), (2, 3, 7, 12)) if decode else ((512, 777), (130, 999, 2047))

    def run(tokens: int) -> None:
        layer = _make_layer(case._replace(tokens=tokens), seed=tokens)
        _run_warp_pipelined(layer, case, decode=decode, **tiles)

    for tokens in warm:
        run(tokens)
    kernels = (
        (decode_warp_pipelined.gluon_moe_decode_warp_pipelined_gfx1250,)
        if decode
        else (
            gate_up_warp_pipelined.gluon_moe_gate_up_warp_pipelined_gfx1250,
            down_warp_pipelined.gluon_moe_down_warp_pipelined_gfx1250,
        )
    )
    with assert_no_triton_compile(*kernels):
        for tokens in sweep:
            run(tokens)


# The default tiles: a 64-row prefill (about 32 rows per expert) with a
# two-slot down ring, and short-K decode tiles.
@pytest.mark.parametrize("decode", [False, True])
def test_selected_tiles_match_reference_gfx1250(decode: bool) -> None:
    case = Case(8 if decode else 256, 32, 1024, 512, 4, "swiglu", True, True)
    layer = _make_layer(case, seed=13)
    _assert_matches(_run_warp_pipelined(layer, case, decode=decode), layer, case)


def test_selection_follows_measured_gains_gfx1250() -> None:
    select = warp_pipelined.select_warp_pipelined_gemms
    # Kimi K3 TP8 gate/up cleared 10% from 8 to 32 tokens; its down never did.
    k3 = (896, 3584, 384, 16, True)
    assert [select(*k3, tokens) for tokens in (4, 8, 32, 64)] == [
        (False, False),
        (True, False),
        (True, False),
        (False, False),
    ]
    # The same shape with SwiGLU, and any unmeasured shape, keep A8W4.
    assert select(896, 3584, 384, 16, False, 16) == (False, False)
    assert select(16, 3584, 384, 16, True, 16) == (False, False)
    # Past the largest measured batch the A8W4 kernels stay in place too.
    assert select(128, 2880, 2880, 4, False, 8192) == (True, True)
    assert select(128, 2880, 2880, 4, False, 32768) == (False, False)


def _launched_kernels(fn) -> set[str]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        result = fn()
        torch.cuda.synchronize()
    return result, {event.key for event in prof.key_averages()}


# 16 tokens run the warp-pipelined gate/up and the A8W4 down; 4 tokens run
# A8W4 for both.
@pytest.mark.parametrize("tokens", [4, 16])
def test_kimi_k3_tp8_apply_uses_measured_kernels_gfx1250(tokens: int) -> None:
    from tokenspeed_kernel.numerics.torch.moe import mxfp4_situ_moe_experts
    from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights

    experts, hidden, intermediate, top_k = 896, 3584, 384, 16
    generator = torch.Generator(device="cuda").manual_seed(tokens)
    raw = make_mxfp4_moe_weights(experts, hidden, intermediate, generator)
    module = torch.nn.Module()
    for name, key in (
        ("w13_weight", "w13_weight"),
        ("w13_weight_scale", "w13_scale"),
        ("w2_weight", "w2_weight"),
        ("w2_weight_scale", "w2_scale"),
    ):
        setattr(module, name, torch.nn.Parameter(raw[key].clone(), requires_grad=False))
    module.top_k, module.num_experts, module.ep_size = top_k, experts, 1
    module.activation_situ_beta = _SITU["situ_beta"]
    module.activation_situ_linear_beta = _SITU["situ_linear_beta"]
    plan = moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=intermediate,
        internal_activation_dtype="input",
        solution="gluon",
        hidden=None,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
    )
    assert plan["apply_kernel_name"] == (
        "gluon_mxfp4_a8w4_situ_gfx1250_precomputed_moe_apply"
    )
    moe_process_weights(plan, module)
    hidden_states = 0.1 * torch.randn(
        tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    scores = torch.rand(tokens, experts, device="cuda", generator=generator)
    topk_ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
    topk_weights = torch.softmax(
        torch.rand(tokens, top_k, device="cuda", generator=generator), dim=-1
    )
    router_logits = torch.zeros(tokens, experts, device="cuda")

    actual, kernels = _launched_kernels(
        lambda: moe_apply(
            plan,
            hidden_states,
            module,
            router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
    )
    warp_pipelined_ran = any(
        "gluon_moe_decode_warp_pipelined_gfx1250" in name for name in kernels
    )
    assert warp_pipelined_ran == (tokens == 16)

    expected = mxfp4_situ_moe_experts(
        hidden_states,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        topk_ids,
        topk_weights,
        activation_dtype=torch.float8_e4m3fn,
        situ_beta=_SITU["situ_beta"],
        situ_linear_beta=_SITU["situ_linear_beta"],
    )
    assert torch.count_nonzero(expected).item() > 0
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=5e-2)
