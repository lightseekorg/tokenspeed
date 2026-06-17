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
    FnSpecs,
    FusedActivation,
    default_route,
    fp8_quantize,
    gluon_mxfp_ragged_matmul,
    shuffle_weight_for_gluon_dot_layout,
    swiglu_fn,
)
from tokenspeed_kernel_amd.ops.moe.mxfp4_gfx950_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)

HIDDEN_SIZE = 288
INTERMEDIATE_SIZE = 256
E = 8
TOPK = 2
MXFP4_BLOCK = 32
GLUON_COMBINE_BLOCK_N = 128
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

KEY_NUM_TOKEN_VALUES = (1, 2, 16, 17, 64)
KEY_NUM_TOKENS = [
    pytest.param(1, id="tokens1_routedM2"),
    pytest.param(2, id="tokens2_routedM4"),
    pytest.param(16, id="tokens16_routedM32"),
    pytest.param(17, id="tokens17_routedM34_blockm_regression"),
    pytest.param(64, id="tokens64_routedM128"),
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
    module.w2_weight_scale = torch.nn.Parameter(raw.w2_scale.clone(), requires_grad=False)
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


def _pad_w2_to_block_n(module: torch.nn.Module, block_n: int) -> None:
    original_n = int(module.w2_weight.shape[-2])
    module._w2_logical_n = original_n
    if original_n % block_n == 0:
        return

    n_padded = (original_n + block_n - 1) // block_n * block_n
    extra_n = n_padded - original_n
    w2_weight = module.w2_weight.data
    w2_scale = module.w2_weight_scale.data
    module.w2_weight = torch.nn.Parameter(
        torch.cat(
            [
                w2_weight,
                torch.zeros(
                    *w2_weight.shape[:-2],
                    extra_n,
                    w2_weight.shape[-1],
                    dtype=w2_weight.dtype,
                    device=w2_weight.device,
                ),
            ],
            dim=-2,
        ),
        requires_grad=False,
    )
    module.w2_weight_scale = torch.nn.Parameter(
        torch.cat(
            [
                w2_scale,
                torch.full(
                    (*w2_scale.shape[:-2], extra_n, w2_scale.shape[-1]),
                    127,
                    dtype=w2_scale.dtype,
                    device=w2_scale.device,
                ),
            ],
            dim=-2,
        ),
        requires_grad=False,
    )


def _raw_storage(obj: Any) -> torch.Tensor:
    storage = getattr(obj, "storage", None)
    data = getattr(storage, "data", None)
    if isinstance(data, torch.Tensor):
        return data
    assert isinstance(obj, torch.Tensor)
    return obj


def _attach_gluon_bpreshuffle(module: torch.nn.Module) -> None:
    targets = (
        ("w13_weight_triton_tensor", None),
        ("w2_weight_triton_tensor", getattr(module, "_w2_logical_n", None)),
    )
    for attr, logical_n in targets:
        raw = _raw_storage(getattr(module, attr))
        shuffled = shuffle_weight_for_gluon_dot_layout(raw)
        if logical_n is not None and logical_n != shuffled.shape[-1]:
            shuffled.original_n = int(logical_n)
            raw.original_n = int(logical_n)
        raw._gluon_shuffled = shuffled


def _make_preprocessed_weights(
    raw: RawMxfp4Weights,
    *,
    preshuffle: bool,
) -> Mxfp4Weights:
    module = _make_module(raw)
    if preshuffle:
        _pad_w2_to_block_n(module, GLUON_COMBINE_BLOCK_N)

    preprocess_gluon_mxfp4_gfx950_moe_weights({}, module)

    if preshuffle:
        _attach_gluon_bpreshuffle(module)

    return Mxfp4Weights(
        w13_weight=module.w13_weight_triton_tensor,
        w2_weight=module.w2_weight_triton_tensor,
        w13_bias=module.w13_weight_bias,
        # Keep combine bias out of this test so padded-preshuffle and unpadded
        # LDS paths share the same reference.
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


def _run_gluon_gemms(
    route: RouteAndInputs,
    weights: Mxfp4Weights,
) -> tuple[torch.Tensor, torch.Tensor]:
    activation = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (SWIGLU_ALPHA, SWIGLU_LIMIT),
    )

    gemm1_output = gluon_mxfp_ragged_matmul(
        route.gemm1_input,
        weights.w13_weight,
        weights.w13_bias,
        a_ragged_metadata=route.ragged_metadata,
        gather_indx=route.gather_indx,
        precision_config=weights.w13_precision_config,
        fused_activation=activation,
        out_quant_scale=weights.w2_act_scale,
    )

    gemm2_output = gluon_mxfp_ragged_matmul(
        gemm1_output,
        weights.w2_weight,
        weights.w2_bias,
        a_ragged_metadata=route.ragged_metadata,
        scatter_indx=route.scatter_indx,
        precision_config=weights.w2_precision_config,
        gammas=route.gate_scal,
        n_tokens=route.gate_scal.shape[0] // TOPK,
        n_expts_act=TOPK,
    )

    torch.cuda.synchronize()
    return gemm1_output, gemm2_output


def _assert_gluon_matches_torch(
    num_tokens: int,
    *,
    raw: RawMxfp4Weights,
    weights: Mxfp4Weights,
    route_inputs: dict[int, RouteAndInputs],
) -> None:
    route = route_inputs[num_tokens]
    reference = _torch_reference(num_tokens, raw, route, weights)
    gluon_gemm1, gluon_gemm2 = _run_gluon_gemms(route, weights)

    torch.testing.assert_close(
        gluon_gemm1.float(),
        reference.gemm1_output.float(),
        atol=GEMM_ATOL,
        rtol=GEMM_RTOL,
    )
    torch.testing.assert_close(
        gluon_gemm2.float(),
        reference.gemm2_output.float(),
        atol=GEMM_ATOL,
        rtol=GEMM_RTOL,
    )


@pytest.mark.parametrize("num_tokens", KEY_NUM_TOKENS)
def test_gluon_moe_gemms_without_preshuffle_match_torch_gfx950(
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
def test_gluon_moe_gemms_with_preshuffle_match_torch_gfx950(
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
