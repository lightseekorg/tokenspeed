from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.moe.triton_kernels import FnSpecs, FusedActivation, swiglu_fn
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import kernel_override

HIDDEN_SIZE = 2880
INTERMEDIATE_SIZE = 2880
E = 128
TOPK = 2
MXFP4_BLOCK = 32
GLUON_COMBINE_BLOCK_N = 128
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
W13_ACT_SCALE = 0.125
W2_ACT_SCALE = 0.125
# E2M1 codes for 0, +0.5, +1, -0.5, -1.
WEIGHT_NIBBLES = (0, 1, 2, 9, 10)
# e8m0 block scales centered around the previous uniform exponent 124.
WEIGHT_SCALE_EXPONENTS = (123, 124, 125)
GEMM_ATOL = 0.05
RTOL = 0.01

KEY_NUM_TOKEN_VALUES = (1, 2, 16, 17, 64, 4096, 8192)
KEY_NUM_TOKENS = [
    pytest.param(1, id="tokens1_routedM2"),
    pytest.param(2, id="tokens2_routedM4"),
    pytest.param(16, id="tokens16_routedM32"),
    pytest.param(17, id="tokens17_routedM34_blockm_regression"),
    pytest.param(64, id="tokens64_routedM128"),
    pytest.param(4096, id="tokens4096_routedM8192"),
    pytest.param(8192, id="tokens8192_routedM16384"),
]


requires_gfx950 = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_cdna4),
    reason="Gluon GPT-OSS MoE GEMM kernels are gfx950 (CDNA4) only",
)


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
    nonpreshuffled: Mxfp4Weights
    preshuffled: Mxfp4Weights


@dataclass
class TritonReference:
    ragged_metadata: Any
    gather_indx: Any
    scatter_indx: Any
    gate_scal: torch.Tensor
    hidden_dtype: torch.dtype
    gemm1_input: torch.Tensor
    gemm2_input: torch.Tensor
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
            (E, HIDDEN_SIZE, INTERMEDIATE_SIZE // 2), device=device, generator=generator
        ),
        w2_scale=_make_e8m0_scales(
            (E, HIDDEN_SIZE, INTERMEDIATE_SIZE // MXFP4_BLOCK),
            device=device,
            generator=generator,
        ),
    )


def _make_backend():
    from tokenspeed.runtime.layers.moe.backends.mxfp4 import gluon_kernel
    from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
    from tokenspeed.runtime.layers.quantization import Mxfp4Config

    spec = MoELayerSpec(
        top_k=TOPK,
        num_experts=E,
        num_local_experts=E,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        activation="swiglu",
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
    )
    return gluon_kernel.Mxfp4GluonKernelBackend(
        BackendKey("gfx950", "mxfp4", "gluon"),
        spec,
        Mxfp4Config(is_checkpoint_mxfp4_serialized=True, is_w4a8_fp8=True),
    )


def _copy_raw_weights(layer: torch.nn.Module, raw: RawMxfp4Weights) -> None:
    layer.w13_weight.data.copy_(raw.w13_weight)
    layer.w13_weight_scale.data.copy_(raw.w13_scale)
    layer.w2_weight.data.copy_(raw.w2_weight)
    layer.w2_weight_scale.data.copy_(raw.w2_scale)
    layer.w13_weight_bias.data.zero_()
    layer.w2_weight_bias.data.zero_()
    layer.w13_input_scale.data.fill_(W13_ACT_SCALE)
    layer.w2_input_scale.data.fill_(W2_ACT_SCALE)


def _make_preprocessed_weights(
    raw: RawMxfp4Weights,
    *,
    preshuffle: bool,
) -> Mxfp4Weights:
    backend = _make_backend()
    layer = torch.nn.Module()
    layer.activation = "swiglu"
    layer.swiglu_arg = None
    backend.create_layer_weights(layer, with_bias=True)
    layer.to(raw.w13_weight.device)
    _copy_raw_weights(layer, raw)

    if preshuffle:
        from tokenspeed.runtime.layers.moe.backends.mxfp4 import gluon_kernel

        gluon_kernel._pad_w2_to_block_n(layer, GLUON_COMBINE_BLOCK_N)

    backend.process_weights_after_loading(layer)

    if preshuffle:
        gluon_kernel._attach_gluon_bpreshuffle(layer)

    return Mxfp4Weights(
        w13_weight=layer.w13_weight_triton_tensor,
        w2_weight=layer.w2_weight_triton_tensor,
        w13_bias=layer.w13_weight_bias,
        # W2 preshuffle pads output N before the Gluon-specific shuffle.
        # Keep combine bias out of this test so the same Triton reference is
        # valid for both padded-preshuffle and unpadded LDS paths.
        w2_bias=None,
        w13_precision_config=layer.w13_precision_config,
        w2_precision_config=layer.w2_precision_config,
        w13_act_scale=layer.w13_act_scale,
        w2_act_scale=layer.w2_act_scale,
    )


@pytest.fixture(scope="module")
def mxfp4_weights() -> Mxfp4WeightVariants:
    raw_weights = _make_raw_mxfp4_weights()
    return Mxfp4WeightVariants(
        nonpreshuffled=_make_preprocessed_weights(raw_weights, preshuffle=False),
        preshuffled=_make_preprocessed_weights(raw_weights, preshuffle=True),
    )


def _make_hidden_and_router(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(9000 + num_tokens)
    hidden_states = (
        torch.randint(
            -4, 5, (num_tokens, HIDDEN_SIZE), device="cuda", generator=generator
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


def _make_gemm2_input(num_tokens: int, scale: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(19000 + num_tokens)
    exact_values = (
        torch.randint(
            -4,
            5,
            (num_tokens * TOPK, INTERMEDIATE_SIZE),
            device="cuda",
            generator=generator,
        ).to(torch.float32)
        / 16.0
    ).to(torch.bfloat16)
    return tokenspeed_kernel.quantize_fp8(
        exact_values,
        scale=scale,
        solution="triton",
    )


def _swiglu_activation() -> FusedActivation:
    return FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (SWIGLU_ALPHA, SWIGLU_LIMIT),
    )


def _compute_triton_reference(
    num_tokens: int,
    weights: Mxfp4Weights,
) -> TritonReference:
    hidden_states, router_logits = _make_hidden_and_router(num_tokens)

    with kernel_override("moe", "route", "triton_kernels_routing"):
        ragged_metadata, gather_indx, scatter_indx, gate_scal = (
            tokenspeed_kernel.moe_route(
                router_logits,
                TOPK,
                sm_first=False,
                dtype=router_logits.dtype,
                traits={"output_type": "ragged_metadata"},
                expected_kernel_name="triton_kernels_routing",
            )
        )

    assert int(ragged_metadata.slice_sizes.sum()) == num_tokens * TOPK

    gemm1_input = tokenspeed_kernel.quantize_fp8(
        hidden_states,
        scale=weights.w13_act_scale,
        solution="triton",
    )
    gemm2_input = _make_gemm2_input(num_tokens, weights.w2_act_scale)

    with torch.no_grad():
        with kernel_override("moe", "experts", "triton_kernels_dispatch_gemm"):
            gemm1_output = tokenspeed_kernel.moe_experts(
                gemm1_input,
                weights.w13_weight,
                weights.w13_bias,
                a_ragged_metadata=ragged_metadata,
                gather_indx=gather_indx,
                precision_config=weights.w13_precision_config,
                fused_activation=_swiglu_activation(),
                dtype=hidden_states.dtype,
                features={"ragged_metadata", "dispatch_gemm"},
                expected_kernel_name="triton_kernels_dispatch_gemm",
            )

        with kernel_override("moe", "experts", "triton_kernels_gemm_combine"):
            gemm2_output = tokenspeed_kernel.moe_experts(
                gemm2_input,
                weights.w2_weight,
                weights.w2_bias,
                a_ragged_metadata=ragged_metadata,
                scatter_indx=scatter_indx,
                precision_config=weights.w2_precision_config,
                gammas=gate_scal,
                n_tokens=num_tokens,
                n_expts_act=TOPK,
                dtype=hidden_states.dtype,
                features={"ragged_metadata", "gemm_combine"},
                expected_kernel_name="triton_kernels_gemm_combine",
            )

    torch.cuda.synchronize()
    return TritonReference(
        ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        scatter_indx=scatter_indx,
        gate_scal=gate_scal,
        hidden_dtype=hidden_states.dtype,
        gemm1_input=gemm1_input,
        gemm2_input=gemm2_input,
        gemm1_output=gemm1_output,
        gemm2_output=gemm2_output,
    )


@pytest.fixture(scope="module")
def triton_references(
    mxfp4_weights: Mxfp4WeightVariants,
) -> dict[int, TritonReference]:
    return {
        num_tokens: _compute_triton_reference(num_tokens, mxfp4_weights.nonpreshuffled)
        for num_tokens in KEY_NUM_TOKEN_VALUES
    }


def _run_gluon_gemms(
    reference: TritonReference,
    weights: Mxfp4Weights,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        with kernel_override("moe", "experts", "gluon_dispatch_gemm"):
            gemm1_output = tokenspeed_kernel.moe_experts(
                reference.gemm1_input,
                weights.w13_weight,
                weights.w13_bias,
                a_ragged_metadata=reference.ragged_metadata,
                gather_indx=reference.gather_indx,
                precision_config=weights.w13_precision_config,
                fused_activation=_swiglu_activation(),
                dtype=reference.gemm1_input.dtype,
                weight_format="mxfp4",
                fp8_scale_granularity="tensor",
                features={"ragged_metadata", "dispatch_gemm"},
                traits={"weight_dtype": "mxfp4"},
                expected_kernel_name="gluon_dispatch_gemm",
            )

        with kernel_override("moe", "experts", "gluon_gemm_combine"):
            gemm2_output = tokenspeed_kernel.moe_experts(
                reference.gemm2_input,
                weights.w2_weight,
                weights.w2_bias,
                a_ragged_metadata=reference.ragged_metadata,
                scatter_indx=reference.scatter_indx,
                precision_config=weights.w2_precision_config,
                gammas=reference.gate_scal,
                n_tokens=reference.gate_scal.shape[0] // TOPK,
                n_expts_act=TOPK,
                dtype=reference.gemm2_input.dtype,
                weight_format="mxfp4",
                fp8_scale_granularity="tensor",
                features={"ragged_metadata", "gemm_combine"},
                traits={"weight_dtype": "mxfp4"},
                expected_kernel_name="gluon_gemm_combine",
            )

    torch.cuda.synchronize()
    return gemm1_output, gemm2_output


def _assert_gluon_matches_triton(
    num_tokens: int,
    *,
    weights: Mxfp4Weights,
    triton_references: dict[int, TritonReference],
) -> None:
    reference = triton_references[num_tokens]
    gluon_gemm1, gluon_gemm2 = _run_gluon_gemms(reference, weights)

    torch.testing.assert_close(
        gluon_gemm1.float(),
        reference.gemm1_output.float(),
        atol=GEMM_ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        gluon_gemm2.float(),
        reference.gemm2_output.float(),
        atol=GEMM_ATOL,
        rtol=RTOL,
    )


@requires_gfx950
@pytest.mark.parametrize("num_tokens", KEY_NUM_TOKENS)
def test_gluon_moe_gemms_without_preshuffle_match_triton_gfx950(
    num_tokens: int,
    mxfp4_weights: Mxfp4WeightVariants,
    triton_references: dict[int, TritonReference],
) -> None:
    _assert_gluon_matches_triton(
        num_tokens,
        weights=mxfp4_weights.nonpreshuffled,
        triton_references=triton_references,
    )


@requires_gfx950
@pytest.mark.parametrize("num_tokens", KEY_NUM_TOKENS)
def test_gluon_moe_gemms_with_preshuffle_match_triton_gfx950(
    num_tokens: int,
    mxfp4_weights: Mxfp4WeightVariants,
    triton_references: dict[int, TritonReference],
) -> None:
    _assert_gluon_matches_triton(
        num_tokens,
        weights=mxfp4_weights.preshuffled,
        triton_references=triton_references,
    )
