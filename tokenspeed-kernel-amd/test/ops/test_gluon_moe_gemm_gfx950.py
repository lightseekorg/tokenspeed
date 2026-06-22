from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip(
        "AMD GFX950 is required for Gluon MoE GEMM correctness tests",
        allow_module_level=True,
    )


from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import (  # noqa: E402
    default_route,
    fp8_quantize,
    gluon_mxfp_fused_moe,
)
from tokenspeed_kernel_amd.ops.moe.mxfp4_gfx950_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)

HIDDEN_SIZE = 288
INTERMEDIATE_SIZE = 256
E = 8
TOPK = 2
MXFP4_BLOCK = 32
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
W13_ACT_SCALE = 0.125
W2_ACT_SCALE = 0.125
# E2M1 codes for 0, +0.5, +1, -0.5, -1.
WEIGHT_NIBBLES = (0, 1, 2, 9, 10)
# E8M0 block scales centered around a useful non-unit exponent.
WEIGHT_SCALE_EXPONENTS = (123, 124, 125)
GEMM_ATOL = 0.05
GEMM_RTOL = 0.01

# The split-GEMM combine helper is no longer a safe public correctness surface
# after upstream's preshuffled-W combine changes. Exercise the fused MoE entry
# point on decode shapes that share the same routing as the torch reference.
KEY_NUM_TOKEN_VALUES = (2, 16)
KEY_NUM_TOKENS = [
    pytest.param(2, id="tokens2_routedM4"),
    pytest.param(16, id="tokens16_routedM32"),
]

_E2M1_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


@dataclass
class RawMxfp4Weights:
    w13_weight: torch.Tensor
    w13_scale: torch.Tensor
    w2_weight: torch.Tensor
    w2_scale: torch.Tensor


@dataclass
class Mxfp4Weights:
    w13_weight: Any
    w2_weight: Any
    w13_bias: torch.Tensor | None
    w2_bias: torch.Tensor | None
    w13_precision_config: Any
    w2_precision_config: Any
    w13_act_scale: torch.Tensor
    w2_act_scale: torch.Tensor


@dataclass
class Mxfp4WeightVariants:
    raw: RawMxfp4Weights
    nonpreshuffled: Mxfp4Weights
    preshuffled: Mxfp4Weights


@dataclass
class RouteAndInputs:
    ragged_metadata: Any
    gather_indx: torch.Tensor
    scatter_indx: torch.Tensor
    gate_scal: torch.Tensor
    gemm1_input: torch.Tensor


@dataclass
class TorchReference:
    gemm1_output: torch.Tensor
    gemm2_output: torch.Tensor


def _make_mxfp4_weight_bytes(
    shape: tuple[int, ...],
    *,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    nibbles = torch.tensor(WEIGHT_NIBBLES, device=device, dtype=torch.uint8)
    lo = nibbles[
        torch.randint(0, len(WEIGHT_NIBBLES), shape, device=device, generator=generator)
    ]
    hi = nibbles[
        torch.randint(0, len(WEIGHT_NIBBLES), shape, device=device, generator=generator)
    ]
    return lo | (hi << 4)


def _make_e8m0_scales(
    shape: tuple[int, ...],
    *,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    exponents = torch.tensor(WEIGHT_SCALE_EXPONENTS, device=device, dtype=torch.uint8)
    return exponents[
        torch.randint(
            0, len(WEIGHT_SCALE_EXPONENTS), shape, device=device, generator=generator
        )
    ]


def _make_raw_mxfp4_weights() -> RawMxfp4Weights:
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260610)
    return RawMxfp4Weights(
        w13_weight=_make_mxfp4_weight_bytes(
            (E, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 2),
            device=device,
            generator=generator,
        ),
        w13_scale=_make_e8m0_scales(
            (E, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // MXFP4_BLOCK),
            device=device,
            generator=generator,
        ),
        w2_weight=_make_mxfp4_weight_bytes(
            (E, HIDDEN_SIZE, INTERMEDIATE_SIZE // 2),
            device=device,
            generator=generator,
        ),
        w2_scale=_make_e8m0_scales(
            (E, HIDDEN_SIZE, INTERMEDIATE_SIZE // MXFP4_BLOCK),
            device=device,
            generator=generator,
        ),
    )


def _make_module(raw: RawMxfp4Weights) -> torch.nn.Module:
    module = torch.nn.Module()
    module.w13_weight = torch.nn.Parameter(raw.w13_weight.clone(), requires_grad=False)
    module.w13_weight_scale = torch.nn.Parameter(
        raw.w13_scale.clone(), requires_grad=False
    )
    module.w2_weight = torch.nn.Parameter(raw.w2_weight.clone(), requires_grad=False)
    module.w2_weight_scale = torch.nn.Parameter(
        raw.w2_scale.clone(), requires_grad=False
    )
    module.w13_weight_bias = torch.nn.Parameter(
        torch.zeros((E, 2 * INTERMEDIATE_SIZE), device="cuda", dtype=torch.float32),
        requires_grad=False,
    )
    module.w2_weight_bias = torch.nn.Parameter(
        torch.zeros((E, HIDDEN_SIZE), device="cuda", dtype=torch.float32),
        requires_grad=False,
    )
    module.w13_input_scale = torch.nn.Parameter(
        torch.full((E,), W13_ACT_SCALE, device="cuda", dtype=torch.float32),
        requires_grad=False,
    )
    module.w2_input_scale = torch.nn.Parameter(
        torch.full((E,), W2_ACT_SCALE, device="cuda", dtype=torch.float32),
        requires_grad=False,
    )
    return module


def _make_preprocessed_weights(
    raw: RawMxfp4Weights,
    *,
    preshuffle: bool,
) -> Mxfp4Weights:
    module = _make_module(raw)
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, module, preshuffle=preshuffle)

    return Mxfp4Weights(
        w13_weight=module.w13_weight_triton_tensor,
        w2_weight=module.w2_weight_triton_tensor,
        w13_bias=module.w13_weight_bias,
        # Keep combine bias out of this test so padded-preshuffle and
        # unpadded LDS paths share the same reference.
        w2_bias=None,
        w13_precision_config=module.w13_precision_config,
        w2_precision_config=module.w2_precision_config,
        w13_act_scale=module.w13_act_scale,
        w2_act_scale=module.w2_act_scale,
    )


@pytest.fixture(scope="module")
def mxfp4_weights() -> Mxfp4WeightVariants:
    raw = _make_raw_mxfp4_weights()
    return Mxfp4WeightVariants(
        raw=raw,
        nonpreshuffled=_make_preprocessed_weights(raw, preshuffle=False),
        preshuffled=_make_preprocessed_weights(raw, preshuffle=True),
    )


def _make_hidden_and_router(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(9000 + num_tokens)
    hidden_states = (
        torch.randint(
            -4,
            5,
            (num_tokens, HIDDEN_SIZE),
            device="cuda",
            generator=generator,
        ).to(torch.float32)
        / 16.0
    ).to(torch.bfloat16)
    router_logits = torch.randn(
        (num_tokens, E),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).to(torch.bfloat16)
    return hidden_states, router_logits


def _route_and_inputs(num_tokens: int, weights: Mxfp4Weights) -> RouteAndInputs:
    hidden_states, router_logits = _make_hidden_and_router(num_tokens)
    ragged_metadata, gather_indx, scatter_indx, gate_scal = default_route(
        router_logits,
        TOPK,
        dtype=router_logits.dtype,
    )
    assert int(ragged_metadata.slice_sizes.sum()) == num_tokens * TOPK
    return RouteAndInputs(
        ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        scatter_indx=scatter_indx,
        gate_scal=gate_scal,
        gemm1_input=fp8_quantize(hidden_states, weights.w13_act_scale),
    )


@pytest.fixture(scope="module")
def route_inputs(
    mxfp4_weights: Mxfp4WeightVariants,
) -> dict[int, RouteAndInputs]:
    return {
        num_tokens: _route_and_inputs(num_tokens, mxfp4_weights.nonpreshuffled)
        for num_tokens in KEY_NUM_TOKEN_VALUES
    }


def _mxfp4_dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    lo = lut[(packed & 0x0F).long()]
    hi = lut[(packed >> 4).long()]
    unpacked = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], -1)
    scale = torch.pow(2.0, scale.to(torch.float32) - 127.0)
    scale = scale.repeat_interleave(MXFP4_BLOCK, dim=-1)
    return unpacked * scale[..., : unpacked.shape[-1]]


def _swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    gate_up = gate_up.reshape(*gate_up.shape[:-1], INTERMEDIATE_SIZE, 2)
    gate = torch.minimum(
        gate_up[..., 0],
        torch.tensor(SWIGLU_LIMIT, device=gate_up.device),
    )
    linear = torch.clamp(gate_up[..., 1], -SWIGLU_LIMIT, SWIGLU_LIMIT)
    return (gate / (1.0 + torch.exp(-SWIGLU_ALPHA * gate))) * (linear + 1.0)


def _torch_reference(
    num_tokens: int,
    raw: RawMxfp4Weights,
    route: RouteAndInputs,
    weights: Mxfp4Weights,
) -> TorchReference:
    hidden = route.gemm1_input.to(torch.float32) * weights.w13_act_scale
    slice_offs = route.ragged_metadata.slice_offs.detach().cpu().tolist()
    gather = route.gather_indx.detach().cpu().tolist()
    scatter = route.scatter_indx.detach().cpu().tolist()
    gate_scal = route.gate_scal.to(torch.float32)

    gemm1 = torch.empty(
        (num_tokens * TOPK, INTERMEDIATE_SIZE),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    output = torch.zeros((num_tokens, HIDDEN_SIZE), device="cuda", dtype=torch.float32)

    deq_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _expert_weights(expert: int) -> tuple[torch.Tensor, torch.Tensor]:
        if expert not in deq_cache:
            deq_cache[expert] = (
                _mxfp4_dequant(raw.w13_weight[expert], raw.w13_scale[expert]),
                _mxfp4_dequant(raw.w2_weight[expert], raw.w2_scale[expert]),
            )
        return deq_cache[expert]

    for expert in range(E):
        start, end = slice_offs[expert], slice_offs[expert + 1]
        if start == end:
            continue
        w13, w2 = _expert_weights(expert)
        for row in range(start, end):
            token = gather[row]
            gate_up = hidden[token : token + 1] @ w13.T
            if weights.w13_bias is not None:
                gate_up = gate_up + weights.w13_bias[expert][None, :]
            inter = _swiglu(gate_up)
            inter_fp8 = (inter / weights.w2_act_scale).to(torch.float8_e4m3fn)
            gemm1[row] = inter_fp8.squeeze(0)

            inter_dequant = inter_fp8.to(torch.float32) * weights.w2_act_scale
            second = inter_dequant @ w2.T
            if weights.w2_bias is not None:
                second = second + weights.w2_bias[expert][None, :]
            out_token = scatter[row] // TOPK
            output[out_token] += gate_scal[row] * second.squeeze(0)

    return TorchReference(
        gemm1_output=gemm1,
        gemm2_output=output.to(torch.bfloat16),
    )


def _run_gluon_moe(
    num_tokens: int,
    weights: Mxfp4Weights,
) -> torch.Tensor:
    hidden_states, router_logits = _make_hidden_and_router(num_tokens)
    return gluon_mxfp_fused_moe(
        hidden_states,
        router_logits,
        weights.w13_weight,
        weights.w2_weight,
        w13_bias=weights.w13_bias,
        w2_bias=weights.w2_bias,
        w13_precision_config=weights.w13_precision_config,
        w2_precision_config=weights.w2_precision_config,
        w13_act_scale=weights.w13_act_scale,
        w2_act_scale=weights.w2_act_scale,
        top_k=TOPK,
        swiglu_alpha=SWIGLU_ALPHA,
        swiglu_limit=SWIGLU_LIMIT,
    )


def _assert_gluon_matches_torch(
    num_tokens: int,
    *,
    raw: RawMxfp4Weights,
    weights: Mxfp4Weights,
    route_inputs: dict[int, RouteAndInputs],
) -> None:
    route = route_inputs[num_tokens]
    reference = _torch_reference(num_tokens, raw, route, weights)
    output = _run_gluon_moe(num_tokens, weights)

    torch.testing.assert_close(
        output.float(),
        reference.gemm2_output.float(),
        atol=GEMM_ATOL,
        rtol=GEMM_RTOL,
    )


@pytest.mark.parametrize("num_tokens", KEY_NUM_TOKENS)
def test_gluon_moe_without_preshuffle_matches_torch_gfx950(
    num_tokens: int,
    mxfp4_weights: Mxfp4WeightVariants,
    route_inputs: dict[int, RouteAndInputs],
) -> None:
    _assert_gluon_matches_torch(
        num_tokens,
        raw=mxfp4_weights.raw,
        weights=mxfp4_weights.nonpreshuffled,
        route_inputs=route_inputs,
    )


@pytest.mark.parametrize("num_tokens", KEY_NUM_TOKENS)
def test_gluon_moe_with_preshuffle_matches_torch_gfx950(
    num_tokens: int,
    mxfp4_weights: Mxfp4WeightVariants,
    route_inputs: dict[int, RouteAndInputs],
) -> None:
    _assert_gluon_matches_torch(
        num_tokens,
        raw=mxfp4_weights.raw,
        weights=mxfp4_weights.preshuffled,
        route_inputs=route_inputs,
    )
