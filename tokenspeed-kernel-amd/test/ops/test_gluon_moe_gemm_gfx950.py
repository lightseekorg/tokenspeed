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


_IS_GFX950 = _is_gfx950()
if not _IS_GFX950:
    pytest.skip(
        "Gluon GPT-OSS MoE GEMM kernels are gfx950 (CDNA4) only",
        allow_module_level=True,
    )

from tokenspeed_kernel_amd.ops.moe import fused_mxfp_gfx950 as gluon_moe
from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import (
    default_route,
    fp8_quantize,
)
from tokenspeed_kernel_amd.ops.moe.mxfp4_gfx950_preprocess import (
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)
from tokenspeed_kernel_amd.ops.moe.utils import FnSpecs, FusedActivation, swiglu_fn

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
E2M1_POSITIVE_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
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


def test_gluon_dot_preshuffle_records_layout_block_n() -> None:
    w = torch.arange(128 * 128, dtype=torch.uint8).reshape(128, 128)
    shuffled = gluon_moe.shuffle_weight_for_gluon_dot_layout(w, block_n=128)

    assert shuffled.is_shuffled_for_gluon_dot is True
    assert shuffled.original_k_pk == 128
    assert shuffled.gluon_dot_block_k_pk == 128
    assert shuffled.gluon_dot_block_n == 128


def test_preshuffled_layout_selection_clamps_block_n_256_without_slicen() -> None:
    from tokenspeed_kernel_amd.ops.moe import fused_mxfp_gfx950 as gluon_moe

    w = torch.empty((128, 128), dtype=torch.uint8)
    w.gluon_dot_block_n = 128

    block_n, use_slice_mn, use_slice_n = gluon_moe._align_block_n_to_preshuffled_layout(
        w,
        block_m=128,
        block_n=256,
        block_k=256,
        scale_load_mode="swizzle",
        x_format="e4m3",
        has_x_block_scale=False,
        has_w_block_scale=True,
        use_slice_mn=None,
        use_slice_n=None,
    )

    assert block_n == 128
    assert use_slice_mn is False
    assert use_slice_n is False


def test_preshuffled_layout_selection_keeps_block_n_256_for_slicen() -> None:
    from tokenspeed_kernel_amd.ops.moe import fused_mxfp_gfx950 as gluon_moe

    w = torch.empty((128, 128), dtype=torch.uint8)
    w.gluon_dot_block_n = 128

    block_n, use_slice_mn, use_slice_n = gluon_moe._align_block_n_to_preshuffled_layout(
        w,
        block_m=64,
        block_n=256,
        block_k=256,
        scale_load_mode="swizzle",
        x_format="e4m3",
        has_x_block_scale=False,
        has_w_block_scale=True,
        use_slice_mn=None,
        use_slice_n=None,
    )

    assert block_n == 256
    assert use_slice_mn is False
    assert use_slice_n is None


def test_preshuffled_layout_selection_clamps_when_slicen_is_incompatible() -> None:
    from tokenspeed_kernel_amd.ops.moe import fused_mxfp_gfx950 as gluon_moe

    w = torch.empty((128, 128), dtype=torch.uint8)
    w.gluon_dot_block_n = 128

    block_n, use_slice_mn, use_slice_n = gluon_moe._align_block_n_to_preshuffled_layout(
        w,
        block_m=128,
        block_n=256,
        block_k=256,
        scale_load_mode="transpose",
        x_format="e4m3",
        has_x_block_scale=False,
        has_w_block_scale=True,
        use_slice_mn=None,
        use_slice_n=None,
    )

    assert block_n == 128
    assert use_slice_mn is False
    assert use_slice_n is False


def test_autotune_block_promotes_small_dispatch_shape() -> None:
    block_m, block_n, _block_k, _num_warps, use_slice_n, small = (
        gluon_moe._autotune_block(
            1024,
            5760,
            2880,
            do_swiglu=True,
            slice_size=8,
            use_slice_n=None,
            large_slice_size=128,
            large_m=16384,
        )
    )

    assert (block_m, block_n, use_slice_n) == (16, 256, None)
    assert small is True


def test_autotune_block_forces_large_dispatch_slicen() -> None:
    block_m, block_n, _block_k, _num_warps, use_slice_n, small = (
        gluon_moe._autotune_block(
            16384,
            5760,
            2880,
            do_swiglu=True,
            slice_size=128,
            use_slice_n=None,
            large_slice_size=128,
            large_m=16384,
        )
    )

    assert (block_m, block_n, use_slice_n) == (128, 256, True)
    assert small is False


@pytest.mark.parametrize(
    ("op", "m", "expected"),
    [
        ("dispatch", 1024, (1, gluon_moe._CDNA4_NUM_XCDS, None, False)),
        ("dispatch", 2048, (1, 4, None, False)),
        ("dispatch", 4096, (1, gluon_moe._CDNA4_NUM_XCDS, True, False)),
        ("dispatch", 8192, (1, None, None, False)),
        ("combine", 1024, (1, gluon_moe._CDNA4_NUM_XCDS, None, False)),
        ("combine", 2048, (1, 4, None, False)),
        ("combine", 4096, (1, gluon_moe._CDNA4_NUM_XCDS, True, False)),
        ("combine", 8192, (1, 4, None, False)),
        ("combine", 16384, (1, 4, None, True)),
    ],
)
def test_prefill_launch_tuning_routes(
    op: str,
    m: int,
    expected: tuple[int | None, int | None, bool | None, bool],
) -> None:
    actual = gluon_moe._prefill_launch_tuning(
        op,
        m=m,
        use_slice_mn=False,
    )

    assert actual == expected


def test_prefill_launch_tuning_ignores_slice_mn() -> None:
    assert gluon_moe._prefill_launch_tuning(
        "combine",
        m=4096,
        use_slice_mn=True,
    ) == (1, None, None, False)


def test_prefill_slice_resolver_prefers_slicen_by_default() -> None:
    use_slice_mn, use_slice_n = gluon_moe._resolve_prefill_slice_modes(
        use_slice_mn=None,
        use_slice_n=True,
        block_m=128,
        block_n=256,
        block_k=256,
        num_buffers=2,
        scale_load_mode="swizzle",
        x_format="e2m1",
        has_x_block_scale=True,
        has_w_block_scale=True,
    )

    assert use_slice_mn is False
    assert use_slice_n is True


def test_prefill_slice_resolver_honors_explicit_slicemn() -> None:
    use_slice_mn, use_slice_n = gluon_moe._resolve_prefill_slice_modes(
        use_slice_mn=True,
        use_slice_n=True,
        block_m=128,
        block_n=256,
        block_k=256,
        num_buffers=2,
        scale_load_mode="swizzle",
        x_format="e2m1",
        has_x_block_scale=True,
        has_w_block_scale=True,
    )

    assert use_slice_mn is True
    assert use_slice_n is False


requires_gfx950 = pytest.mark.skipif(
    not _IS_GFX950,
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
    raw: RawMxfp4Weights
    nonpreshuffled: Mxfp4Weights
    preshuffled: Mxfp4Weights


@dataclass
class TorchReference:
    ragged_metadata: Any
    gather_indx: Any
    scatter_indx: Any
    gate_scal: torch.Tensor
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


def _make_weight_module(raw: RawMxfp4Weights) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.activation = "swiglu"
    layer.swiglu_arg = None
    layer.w13_weight = torch.nn.Parameter(raw.w13_weight.clone(), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        raw.w13_scale.clone(), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(raw.w2_weight.clone(), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(
        raw.w2_scale.clone(), requires_grad=False
    )
    layer.w13_weight_bias = torch.nn.Parameter(
        torch.zeros(E, 2 * INTERMEDIATE_SIZE, device=raw.w13_weight.device),
        requires_grad=False,
    )
    layer.w2_weight_bias = torch.nn.Parameter(
        torch.zeros(E, HIDDEN_SIZE, device=raw.w13_weight.device),
        requires_grad=False,
    )
    layer.w13_input_scale = torch.nn.Parameter(
        torch.full((E,), W13_ACT_SCALE, device=raw.w13_weight.device),
        requires_grad=False,
    )
    layer.w2_input_scale = torch.nn.Parameter(
        torch.full((E,), W2_ACT_SCALE, device=raw.w13_weight.device),
        requires_grad=False,
    )
    return layer


def _make_preprocessed_weights(
    raw: RawMxfp4Weights,
    *,
    preshuffle: bool,
) -> Mxfp4Weights:
    layer = _make_weight_module(raw)
    plan = {"internal_activation_dtype": "fp8"}
    preprocess_gluon_mxfp4_gfx950_moe_weights(plan, layer, preshuffle=preshuffle)

    return Mxfp4Weights(
        w13_weight=layer.w13_weight_triton_tensor,
        w2_weight=layer.w2_weight_triton_tensor,
        w13_bias=layer.w13_weight_bias,
        w2_bias=layer.w2_weight_bias,
        w13_precision_config=layer.w13_precision_config,
        w2_precision_config=layer.w2_precision_config,
        w13_act_scale=layer.w13_act_scale,
        w2_act_scale=layer.w2_act_scale,
    )


@pytest.fixture(scope="module")
def mxfp4_weights() -> Mxfp4WeightVariants:
    raw_weights = _make_raw_mxfp4_weights()
    return Mxfp4WeightVariants(
        raw=raw_weights,
        nonpreshuffled=_make_preprocessed_weights(raw_weights, preshuffle=False),
        preshuffled=_make_preprocessed_weights(raw_weights, preshuffle=True),
    )


def test_gluon_preshuffle_keeps_w2_bias_logical_n(
    mxfp4_weights: Mxfp4WeightVariants,
) -> None:
    w2_raw = gluon_moe._extract_gluon_raw_w(mxfp4_weights.preshuffled.w2_weight)
    assert mxfp4_weights.preshuffled.w2_bias is not None
    assert getattr(w2_raw, "original_n") == HIDDEN_SIZE
    assert w2_raw.shape[-1] > HIDDEN_SIZE
    assert mxfp4_weights.preshuffled.w2_bias.shape[-1] == HIDDEN_SIZE


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


def _quantize_mxfp4_for_test(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] % MXFP4_BLOCK != 0:
        raise ValueError("MXFP4 test quantization requires 32-element blocks")

    x_blocks = x.to(torch.float32).reshape(
        *x.shape[:-1],
        x.shape[-1] // MXFP4_BLOCK,
        MXFP4_BLOCK,
    )
    max_abs = x_blocks.abs().amax(dim=-1)
    min_exp = torch.full_like(max_abs, -127.0)
    scale_exp = torch.where(
        max_abs > 0,
        torch.floor(torch.log2(max_abs)) - 2,
        min_exp,
    ).clamp(-127, 127)

    scaled = x_blocks * torch.exp2(-scale_exp).unsqueeze(-1)
    abs_scaled = scaled.abs()
    codes = torch.zeros_like(abs_scaled, dtype=torch.uint8)
    for threshold in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        codes += (abs_scaled >= threshold).to(torch.uint8)
    codes |= (scaled < 0).to(torch.uint8) * 8

    packed_blocks = codes[..., 0::2] | (codes[..., 1::2] << 4)
    packed = packed_blocks.reshape(*x.shape[:-1], x.shape[-1] // 2).contiguous()
    scales = (scale_exp.to(torch.int16) + 127).to(torch.uint8).contiguous()
    return packed, scales


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
    return fp8_quantize(
        exact_values,
        scale=scale,
    )


def _swiglu_activation() -> FusedActivation:
    return FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (SWIGLU_ALPHA, SWIGLU_LIMIT),
    )


def _mxfp4_dequant(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    positive = torch.tensor(
        E2M1_POSITIVE_VALUES, device=packed.device, dtype=torch.float32
    )
    lut = torch.cat((positive, -positive))
    lo = lut[(packed & 0x0F).long()]
    hi = lut[(packed >> 4).long()]
    values = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], -1)
    block_scales = torch.exp2(scales.to(torch.float32) - 127.0)
    scaled = values.reshape(
        *values.shape[:-1], values.shape[-1] // MXFP4_BLOCK, MXFP4_BLOCK
    )
    return (scaled * block_scales.unsqueeze(-1)).reshape_as(values)


def _fp8_dequant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32) * scale.to(torch.float32).reshape(())


def _swiglu_reference(gate_up: torch.Tensor) -> torch.Tensor:
    gate, linear = gate_up.reshape(gate_up.shape[0], -1, 2).unbind(dim=-1)
    gate = torch.clamp(gate, max=SWIGLU_LIMIT)
    linear = torch.clamp(linear, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    sigmoid = 1.0 / (1.0 + torch.exp(-SWIGLU_ALPHA * gate))
    return (gate * sigmoid) * (linear + 1.0)


def _expert_ranges(ragged_metadata: Any) -> list[tuple[int, int]]:
    slice_sizes = ragged_metadata.slice_sizes.to(torch.int64).tolist()
    ranges = []
    offset = 0
    for size in slice_sizes:
        end = offset + int(size)
        ranges.append((offset, end))
        offset = end
    return ranges


def _compute_torch_gemm1_reference(
    raw: RawMxfp4Weights,
    gemm1_input: torch.Tensor,
    weights: Mxfp4Weights,
    ragged_metadata: Any,
    gather_indx: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(
        (gather_indx.numel(), INTERMEDIATE_SIZE),
        device=gemm1_input.device,
        dtype=torch.bfloat16,
    )
    x = _fp8_dequant(gemm1_input, weights.w13_act_scale)
    for expert, (start, end) in enumerate(_expert_ranges(ragged_metadata)):
        if start == end:
            continue
        row_idx = gather_indx[start:end].long()
        w13 = _mxfp4_dequant(raw.w13_weight[expert], raw.w13_scale[expert])
        gate_up = x[row_idx] @ w13.T
        if weights.w13_bias is not None:
            gate_up = gate_up + weights.w13_bias[expert][None, :]
        output[start:end] = _swiglu_reference(gate_up).to(torch.bfloat16)
    return output


def _compute_torch_gemm1_mxfp4_reference(
    raw: RawMxfp4Weights,
    gemm1_input: torch.Tensor,
    gemm1_scale: torch.Tensor,
    weights: Mxfp4Weights,
    ragged_metadata: Any,
    gather_indx: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(
        (gather_indx.numel(), INTERMEDIATE_SIZE),
        device=gemm1_input.device,
        dtype=torch.bfloat16,
    )
    x = _mxfp4_dequant(gemm1_input, gemm1_scale)
    for expert, (start, end) in enumerate(_expert_ranges(ragged_metadata)):
        if start == end:
            continue
        row_idx = gather_indx[start:end].long()
        w13 = _mxfp4_dequant(raw.w13_weight[expert], raw.w13_scale[expert])
        gate_up = x[row_idx] @ w13.T
        if weights.w13_bias is not None:
            gate_up = gate_up + weights.w13_bias[expert][None, :]
        output[start:end] = _swiglu_reference(gate_up).to(torch.bfloat16)
    return output


def _compute_torch_gemm2_reference(
    raw: RawMxfp4Weights,
    gemm2_input: torch.Tensor,
    weights: Mxfp4Weights,
    ragged_metadata: Any,
    scatter_indx: torch.Tensor,
    gate_scal: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    routed = torch.empty(
        (num_tokens * TOPK, HIDDEN_SIZE),
        device=gemm2_input.device,
        dtype=torch.bfloat16,
    )
    x = _fp8_dequant(gemm2_input, weights.w2_act_scale)
    for expert, (start, end) in enumerate(_expert_ranges(ragged_metadata)):
        if start == end:
            continue
        w2 = _mxfp4_dequant(raw.w2_weight[expert], raw.w2_scale[expert])
        expert_out = x[start:end] @ w2.T
        if weights.w2_bias is not None:
            expert_out = expert_out + weights.w2_bias[expert][None, :]
        expert_out = expert_out * gate_scal[start:end].to(torch.float32)[:, None]
        routed[scatter_indx[start:end].long()] = expert_out.to(torch.bfloat16)
    return routed.view(num_tokens, TOPK, HIDDEN_SIZE).sum(dim=1)


def _compute_torch_reference(
    num_tokens: int,
    raw: RawMxfp4Weights,
    weights: Mxfp4Weights,
) -> TorchReference:
    hidden_states, router_logits = _make_hidden_and_router(num_tokens)

    ragged_metadata, gather_indx, scatter_indx, gate_scal = default_route(
        router_logits,
        TOPK,
        dtype=router_logits.dtype,
    )

    assert int(ragged_metadata.slice_sizes.sum()) == num_tokens * TOPK

    gemm1_input = fp8_quantize(
        hidden_states,
        scale=weights.w13_act_scale,
    )
    gemm2_input = _make_gemm2_input(num_tokens, weights.w2_act_scale)

    with torch.no_grad():
        gemm1_output = _compute_torch_gemm1_reference(
            raw, gemm1_input, weights, ragged_metadata, gather_indx
        )
        gemm2_output = _compute_torch_gemm2_reference(
            raw,
            gemm2_input,
            weights,
            ragged_metadata,
            scatter_indx,
            gate_scal,
            num_tokens,
        )

    torch.cuda.synchronize()
    return TorchReference(
        ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        scatter_indx=scatter_indx,
        gate_scal=gate_scal,
        gemm1_input=gemm1_input,
        gemm2_input=gemm2_input,
        gemm1_output=gemm1_output,
        gemm2_output=gemm2_output,
    )


@pytest.fixture(scope="module")
def torch_references(
    mxfp4_weights: Mxfp4WeightVariants,
) -> dict[int, TorchReference]:
    return {
        num_tokens: _compute_torch_reference(
            num_tokens, mxfp4_weights.raw, mxfp4_weights.nonpreshuffled
        )
        for num_tokens in KEY_NUM_TOKEN_VALUES
    }


def _run_gluon_gemms(
    reference: TorchReference,
    weights: Mxfp4Weights,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        gemm1_output = gluon_moe.gluon_mxfp_ragged_matmul(
            reference.gemm1_input,
            weights.w13_weight,
            weights.w13_bias,
            w_mx_scale=weights.w13_precision_config.b_mx_scale,
            x_global_scale=weights.w13_act_scale,
            out_dtype=weights.w13_precision_config.out_dtype,
            a_ragged_metadata=reference.ragged_metadata,
            gather_indx=reference.gather_indx,
            fused_activation=_swiglu_activation(),
        )

        gemm2_output = gluon_moe.gluon_mxfp_ragged_matmul(
            reference.gemm2_input,
            weights.w2_weight,
            weights.w2_bias,
            w_mx_scale=weights.w2_precision_config.b_mx_scale,
            x_global_scale=weights.w2_act_scale,
            out_dtype=weights.w2_precision_config.out_dtype,
            a_ragged_metadata=reference.ragged_metadata,
            scatter_indx=reference.scatter_indx,
            gammas=reference.gate_scal,
            n_tokens=reference.gate_scal.shape[0] // TOPK,
            n_expts_act=TOPK,
        )

    torch.cuda.synchronize()
    return gemm1_output, gemm2_output


def _assert_gluon_matches_torch(
    num_tokens: int,
    *,
    weights: Mxfp4Weights,
    torch_references: dict[int, TorchReference],
) -> None:
    reference = torch_references[num_tokens]
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
def test_gluon_moe_gemms_without_preshuffle_match_torch_gfx950(
    num_tokens: int,
    mxfp4_weights: Mxfp4WeightVariants,
    torch_references: dict[int, TorchReference],
) -> None:
    _assert_gluon_matches_torch(
        num_tokens,
        weights=mxfp4_weights.nonpreshuffled,
        torch_references=torch_references,
    )


@requires_gfx950
@pytest.mark.parametrize("num_tokens", KEY_NUM_TOKENS)
def test_gluon_moe_gemms_with_preshuffle_match_torch_gfx950(
    num_tokens: int,
    mxfp4_weights: Mxfp4WeightVariants,
    torch_references: dict[int, TorchReference],
) -> None:
    _assert_gluon_matches_torch(
        num_tokens,
        weights=mxfp4_weights.preshuffled,
        torch_references=torch_references,
    )


@requires_gfx950
@pytest.mark.parametrize("num_tokens", KEY_NUM_TOKENS)
@pytest.mark.parametrize("variant", ("nonpreshuffled", "preshuffled"))
def test_gluon_moe_gemm1_dynamic_mxfp4_gather_scales_match_torch_gfx950(
    num_tokens: int,
    mxfp4_weights: Mxfp4WeightVariants,
    variant: str,
) -> None:
    weights = getattr(mxfp4_weights, variant)
    hidden_states, router_logits = _make_hidden_and_router(num_tokens)
    ragged_metadata, gather_indx, _scatter_indx, _gate_scal = default_route(
        router_logits,
        TOPK,
        dtype=router_logits.dtype,
    )
    gemm1_input, gemm1_scale = _quantize_mxfp4_for_test(hidden_states)

    with torch.no_grad():
        actual = gluon_moe.gluon_mxfp_ragged_matmul(
            gemm1_input,
            weights.w13_weight,
            weights.w13_bias,
            w_mx_scale=weights.w13_precision_config.b_mx_scale,
            x_mx_scale=gemm1_scale,
            x_format="e2m1",
            out_dtype=weights.w13_precision_config.out_dtype,
            a_ragged_metadata=ragged_metadata,
            gather_indx=gather_indx,
            fused_activation=_swiglu_activation(),
        )
        expected = _compute_torch_gemm1_mxfp4_reference(
            mxfp4_weights.raw,
            gemm1_input,
            gemm1_scale,
            weights,
            ragged_metadata,
            gather_indx,
        )

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=GEMM_ATOL,
        rtol=RTOL,
    )
