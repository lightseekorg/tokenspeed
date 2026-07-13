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

from tokenspeed_kernel_amd.ops.moe import fused_mxfp_gfx950 as gluon_moe  # noqa: E402
from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import (  # noqa: E402
    _dynamic_mxfp4_route,
    default_biased_route,
    default_grouped_route,
    default_route,
    fp8_quantize,
    gluon_biased_grouped_fused_route,
    gluon_mxfp_dynamic_mxfp4_fused_moe,
    gluon_mxfp_ragged_matmul,
)
from tokenspeed_kernel_amd.ops.moe.mxfp4_gfx950_preprocess import (  # noqa: E402
    _interleave_gate_up_rows,
    _make_k_packed_mxfp4_weight,
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)
from tokenspeed_kernel_amd.ops.moe.utils import (  # noqa: E402
    FnSpecs,
    FusedActivation,
    swiglu_fn,
)

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


def test_single_k_tile_gate_matches_k_iters_one_gfx950() -> None:
    from tokenspeed_kernel_amd.ops.moe import fused_mxfp_gfx950 as gluon_moe

    # A single BLOCK_K tile (K_ITERS == 1) must be flagged so the host disables
    # SliceN and routes the shape to the full-N decode schedule.
    assert gluon_moe._is_single_k_tile(256, 256) is True
    assert gluon_moe._is_single_k_tile(128, 256) is True
    # Two or more K tiles keep the pipelined/SliceN schedules.
    assert gluon_moe._is_single_k_tile(257, 256) is False
    assert gluon_moe._is_single_k_tile(512, 256) is False


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


def _make_random_mxfp4_quantized_tensor(
    logical_shape: tuple[int, ...],
    *,
    device: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logical_shape[-1] % MXFP4_BLOCK != 0:
        raise ValueError(
            f"MXFP4 test tensor K must be divisible by {MXFP4_BLOCK}, "
            f"got {logical_shape[-1]}"
        )
    return (
        _make_mxfp4_weight_bytes(
            (*logical_shape[:-1], logical_shape[-1] // 2),
            device=device,
            generator=generator,
        ),
        _make_e8m0_scales(
            (*logical_shape[:-1], logical_shape[-1] // MXFP4_BLOCK),
            device=device,
            generator=generator,
        ),
    )


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
    layer.w13_input_layout = "interleaved"
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


def test_preprocess_releases_raw_mxfp4_parameters() -> None:
    device = "cuda"
    e, h, i = 2, 64, 64
    layer = torch.nn.Module()
    layer.w13_input_layout = "concatenated"
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros(e, 2 * i, h // 2, dtype=torch.uint8, device=device),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.full(
            (e, 2 * i, h // MXFP4_BLOCK),
            124,
            dtype=torch.uint8,
            device=device,
        ),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros(e, h, i // 2, dtype=torch.uint8, device=device),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.full(
            (e, h, i // MXFP4_BLOCK),
            124,
            dtype=torch.uint8,
            device=device,
        ),
        requires_grad=False,
    )
    layer.w13_weight_bias = torch.nn.Parameter(
        torch.zeros(e, 2 * i, device=device), requires_grad=False
    )
    layer.w2_weight_bias = torch.nn.Parameter(
        torch.zeros(e, h, device=device), requires_grad=False
    )
    layer.w13_input_scale = torch.nn.Parameter(
        torch.ones(e, device=device), requires_grad=False
    )
    layer.w2_input_scale = torch.nn.Parameter(
        torch.ones(e, device=device), requires_grad=False
    )

    preprocess_gluon_mxfp4_gfx950_moe_weights({}, layer, preshuffle=False)

    assert not hasattr(layer, "w13_weight")
    assert not hasattr(layer, "w13_weight_scale")
    assert not hasattr(layer, "w2_weight")
    assert not hasattr(layer, "w2_weight_scale")
    assert hasattr(layer, "w13_weight_triton_tensor")
    assert hasattr(layer, "w2_weight_triton_tensor")
    assert layer.w13_precision_config.b_mx_scale is not None
    assert layer.w2_precision_config.b_mx_scale is not None


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


def _assert_gluon_route_matches_default(
    logits: torch.Tensor,
    topk: int,
    case_name: str,
) -> None:
    ragged_metadata, gather_indx, scatter_indx, gate_scal = gluon_moe.gluon_fused_route(
        logits,
        topk,
        dtype=logits.dtype,
    )
    ref_metadata, ref_gather, ref_scatter, ref_gate = default_route(
        logits,
        topk,
        dtype=logits.dtype,
    )

    torch.cuda.synchronize()

    assert int(ragged_metadata.slice_sizes.sum().item()) == logits.shape[0] * topk
    assert torch.all(gather_indx >= 0), case_name
    assert torch.all(gather_indx < logits.shape[0]), case_name
    assert torch.all(scatter_indx >= 0), case_name
    assert torch.all(scatter_indx < logits.shape[0] * topk), case_name
    torch.testing.assert_close(ragged_metadata.slice_sizes, ref_metadata.slice_sizes)
    torch.testing.assert_close(ragged_metadata.slice_offs, ref_metadata.slice_offs)
    torch.testing.assert_close(
        ragged_metadata.block_offs_data,
        ref_metadata.block_offs_data,
    )
    torch.testing.assert_close(
        ragged_metadata.block_schedule_data,
        ref_metadata.block_schedule_data,
    )
    torch.testing.assert_close(gather_indx, ref_gather)
    torch.testing.assert_close(scatter_indx, ref_scatter)
    torch.testing.assert_close(gate_scal, ref_gate, equal_nan=True)


@requires_gfx950
@pytest.mark.parametrize("topk", [1, 4])
def test_gluon_small_m_route_finite_logits_match_default_route(topk: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260707 + topk)
    logits = torch.randn(
        (4, E),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).to(torch.bfloat16)
    _assert_gluon_route_matches_default(logits, topk, "finite")


@requires_gfx950
@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize(
    ("case_name", "fill_value"),
    [
        ("all_nan", float("nan")),
        ("all_neg_inf", -float("inf")),
        ("all_pos_inf", float("inf")),
    ],
)
def test_gluon_small_m_route_nonfinite_logits_match_default_route(
    case_name: str,
    fill_value: float,
    topk: int,
) -> None:
    logits = torch.full((4, E), fill_value, device="cuda", dtype=torch.bfloat16)
    _assert_gluon_route_matches_default(logits, topk, case_name)


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
    return fp8_quantize(
        exact_values,
        scale=scale,
    )


def _swiglu_activation() -> FusedActivation:
    return FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (SWIGLU_ALPHA, SWIGLU_LIMIT),
    )


def _compact_ragged_scales(
    scales: torch.Tensor,
    expected_scale_shape: tuple[int, ...],
    ragged_metadata,
) -> torch.Tensor:
    rows, k_scale = expected_scale_shape
    padded_rows = int(scales.shape[1]) * 32
    linear = _cdna4_swizzled_scales_to_linear(scales, (padded_rows, k_scale))
    block_offs = ragged_metadata.block_offs(32).to(torch.int64).cpu().tolist()
    slice_sizes = ragged_metadata.slice_sizes.to(torch.int64).cpu().tolist()
    chunks = []
    for expert, size in enumerate(slice_sizes):
        if size == 0:
            continue
        start = int(block_offs[expert]) * 32
        chunks.append(linear[start : start + int(size)])
    if not chunks:
        return linear[:0]
    compact = torch.cat(chunks, dim=0)
    assert compact.shape == (rows, k_scale)
    return compact


def _mxfp4_dequant(
    packed: torch.Tensor,
    scales: torch.Tensor,
    ragged_metadata=None,
) -> torch.Tensor:
    expected_scale_shape = (*packed.shape[:-1], packed.shape[-1] * 2 // MXFP4_BLOCK)
    if ragged_metadata is not None:
        scales = _compact_ragged_scales(scales, expected_scale_shape, ragged_metadata)
    elif scales.shape != expected_scale_shape:
        scales = _cdna4_swizzled_scales_to_linear(scales, expected_scale_shape)
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


def _cdna4_swizzled_scales_to_linear(
    scales: torch.Tensor,
    linear_shape: tuple[int, ...],
) -> torch.Tensor:
    if len(linear_shape) != 2:
        raise ValueError(
            "test CDNA4 scale unswizzle only supports rank-2 scales, "
            f"got {linear_shape}"
        )
    rows, k_scale = linear_shape
    m = torch.arange(rows, device=scales.device)
    k = torch.arange(k_scale, device=scales.device)
    mm = m[:, None]
    kk = k[None, :]
    m_in_block = mm % 32
    m_hi = m_in_block // 16
    m_lo = m_in_block % 16
    k_block = kk // 8
    k_in_block = kk % 8
    k_hi = k_in_block // 4
    k_lo = k_in_block % 4
    swizzled_k = (((k_block * 4 + k_lo) * 16 + m_lo) * 2 + k_hi) * 2 + m_hi
    m_block = mm // 32
    return scales[swizzled_k, m_block].contiguous()


def _fp8_dequant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32) * scale.to(torch.float32).reshape(())


def _swiglu_reference(gate_up: torch.Tensor) -> torch.Tensor:
    gate, linear = gate_up.reshape(gate_up.shape[0], -1, 2).unbind(dim=-1)
    gate = torch.clamp(gate, max=SWIGLU_LIMIT)
    linear = torch.clamp(linear, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    sigmoid = 1.0 / (1.0 + torch.exp(-SWIGLU_ALPHA * gate))
    return (gate * sigmoid) * (linear + 1.0)


def _silu_gate_up_reference(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.float().chunk(2, dim=-1)
    sigmoid = 1.0 / (1.0 + torch.exp(-gate))
    return (gate * sigmoid) * up


def test_interleave_gate_up_rows_matches_even_odd_layout_gfx950() -> None:
    gate = torch.tensor([[1, 2, 3]], device="cuda")
    up = torch.tensor([[10, 20, 30]], device="cuda")
    concat = torch.cat((gate, up), dim=-1)

    actual = _interleave_gate_up_rows(concat, dim=-1)

    expected = torch.tensor([[1, 10, 2, 20, 3, 30]], device="cuda")
    torch.testing.assert_close(actual, expected)


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
    gather_indx: torch.Tensor | None,
) -> torch.Tensor:
    n_rows = gather_indx.numel() if gather_indx is not None else gemm1_input.shape[0]
    output = torch.empty(
        (n_rows, INTERMEDIATE_SIZE),
        device=gemm1_input.device,
        dtype=torch.bfloat16,
    )
    x = _mxfp4_dequant(gemm1_input, gemm1_scale)
    for expert, (start, end) in enumerate(_expert_ranges(ragged_metadata)):
        if start == end:
            continue
        w13 = _mxfp4_dequant(raw.w13_weight[expert], raw.w13_scale[expert])
        x_rows = (
            x[gather_indx[start:end].long()]
            if gather_indx is not None
            else x[start:end]
        )
        gate_up = x_rows @ w13.T
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


def _recover_topk_from_route(
    ragged_metadata: Any,
    scatter_indx: torch.Tensor,
    gate_scal: torch.Tensor,
    num_tokens: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = scatter_indx.device
    expert_ids = torch.repeat_interleave(
        torch.arange(
            ragged_metadata.slice_sizes.numel(),
            device=device,
            dtype=torch.int32,
        ),
        ragged_metadata.slice_sizes.to(torch.long),
    )
    flat_ids = torch.empty((num_tokens * topk,), device=device, dtype=torch.int32)
    flat_weights = torch.empty(
        (num_tokens * topk,), device=device, dtype=gate_scal.dtype
    )
    flat_ids[scatter_indx.long()] = expert_ids
    flat_weights[scatter_indx.long()] = gate_scal
    return flat_weights.view(num_tokens, topk), flat_ids.view(num_tokens, topk)


def test_gluon_dynamic_mxfp4_moe_small_matches_torch_gfx950() -> None:
    from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import (
        _quantize_mxfp4_activation,
    )

    torch.manual_seed(20260630)
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260630)
    m, e, h, i, topk = 4, 8, 512, 512, 2
    n_group, topk_group = 2, 1
    hidden = (
        torch.randn((m, h), device=device, dtype=torch.bfloat16) * 0.1
    ).contiguous()
    logits = torch.randn((m, e), device=device, dtype=torch.bfloat16)
    correction_bias = torch.zeros((e,), device=device, dtype=torch.float32)

    def quant_weight(
        shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        quant, scale = _make_random_mxfp4_quantized_tensor(
            shape,
            device=device,
            generator=generator,
        )
        return (
            quant,
            scale,
            gluon_moe._swizzle_scales_cdna4(scale),
            _make_k_packed_mxfp4_weight(quant),
        )

    w13_quant, w13_scale, w13_scale_swizzled, w13_weight = quant_weight((e, 2 * i, h))
    w2_quant, w2_scale, w2_scale_swizzled, w2_weight = quant_weight((e, h, i))

    out = gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        logits,
        w13_weight,
        w2_weight,
        w13_mx_scale=w13_scale_swizzled,
        w2_mx_scale=w2_scale_swizzled,
        top_k=topk,
        correction_bias=correction_bias,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )

    ragged, gather_indx, scatter_indx, gate_scal = gluon_biased_grouped_fused_route(
        logits,
        correction_bias,
        topk,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        dtype=logits.dtype,
    )

    hidden_quant, hidden_scale = _quantize_mxfp4_activation(
        hidden,
        gather_indx,
        ragged_metadata=ragged,
    )
    hidden_dequant = _mxfp4_dequant(hidden_quant, hidden_scale, ragged)
    intermediate = torch.empty((m * topk, i), device=device, dtype=torch.bfloat16)
    offset = 0
    for expert, size in enumerate(ragged.slice_sizes.to(torch.int64).cpu().tolist()):
        start, end = offset, offset + int(size)
        offset = end
        if start == end:
            continue
        w13 = _mxfp4_dequant(w13_quant[expert], w13_scale[expert])
        intermediate[start:end] = _swiglu_reference(
            hidden_dequant[start:end] @ w13.T
        ).to(torch.bfloat16)

    inter_quant, inter_scale = _quantize_mxfp4_activation(
        intermediate,
        ragged_metadata=ragged,
    )
    inter_dequant = _mxfp4_dequant(inter_quant, inter_scale, ragged)
    routed = torch.empty((m * topk, h), device=device, dtype=torch.bfloat16)
    offset = 0
    for expert, size in enumerate(ragged.slice_sizes.to(torch.int64).cpu().tolist()):
        start, end = offset, offset + int(size)
        offset = end
        if start == end:
            continue
        w2 = _mxfp4_dequant(w2_quant[expert], w2_scale[expert])
        expert_out = inter_dequant[start:end] @ w2.T
        expert_out = expert_out * gate_scal[start:end].to(torch.float32)[:, None]
        routed[scatter_indx[start:end].long()] = expert_out.to(torch.bfloat16)
    ref = routed.view(m, topk, h).sum(dim=1)

    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.to(torch.float32),
        ref.to(torch.float32),
        atol=3e-3,
        rtol=3e-2,
    )


def test_default_biased_route_handles_nongrouped_correction_bias_gfx950() -> None:
    device = "cuda"
    logits = torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.75, -1.0, 0.5, -0.25, 1.25],
            [-0.75, 0.5, 1.5, -0.25, 0.0, 1.0, -1.5, 0.25],
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    correction_bias = torch.tensor(
        [0.0, 0.2, -0.1, 0.3, -0.2, 0.1, 0.4, -0.3],
        device=device,
        dtype=torch.float32,
    )
    topk = 3
    scale = 1.75

    ragged, _, scatter, gate = default_biased_route(
        logits,
        correction_bias,
        topk,
        routed_scaling_factor=scale,
        normalize_topk_weights=True,
        dtype=logits.dtype,
    )
    actual_weights, actual_ids = _recover_topk_from_route(
        ragged, scatter, gate, logits.shape[0], topk
    )

    scores = torch.softmax(logits.float(), dim=-1)
    _, expected_ids = torch.topk(
        scores + correction_bias.unsqueeze(0),
        k=topk,
        dim=-1,
        sorted=True,
    )
    expected_weights = scores.gather(1, expected_ids)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    expected_weights = expected_weights * scale

    torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(
        actual_weights.float(),
        expected_weights,
        atol=5e-3,
        rtol=5e-3,
    )


def test_default_grouped_route_preserves_grouping_and_scaling_gfx950() -> None:
    device = "cuda"
    logits = torch.tensor(
        [
            [1.0, 0.75, -0.25, -0.5, 0.5, 0.25, -1.0, -0.75],
            [-0.25, -0.5, 0.5, 1.0, -0.75, 0.25, 0.75, -1.0],
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    topk = 2
    n_group = 4
    topk_group = 2
    scale = 2.0

    ragged, _, scatter, gate = default_grouped_route(
        logits,
        topk,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=scale,
        normalize_topk_weights=True,
        dtype=logits.dtype,
    )
    actual_weights, actual_ids = _recover_topk_from_route(
        ragged, scatter, gate, logits.shape[0], topk
    )

    scores = torch.softmax(logits.float(), dim=-1)
    num_tokens, num_experts = scores.shape
    group_scores = scores.view(num_tokens, n_group, -1).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, n_group, num_experts // n_group)
        .reshape(num_tokens, -1)
    )
    expected_weights, expected_ids = torch.topk(
        scores.masked_fill(~score_mask.bool(), 0.0),
        k=topk,
        dim=-1,
        sorted=False,
    )
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    expected_weights = expected_weights * scale

    actual_order = actual_ids.argsort(dim=-1)
    expected_order = expected_ids.argsort(dim=-1)
    torch.testing.assert_close(
        actual_ids.gather(1, actual_order),
        expected_ids.to(torch.int32).gather(1, expected_order),
    )
    torch.testing.assert_close(
        actual_weights.float().gather(1, actual_order),
        expected_weights.gather(1, expected_order),
        atol=5e-3,
        rtol=5e-3,
    )


@pytest.mark.parametrize(
    ("topk_weights", "normalize_topk_weights", "expected_weights"),
    [
        (
            [[0.7, 0.3], [0.6, 0.4]],
            True,
            [[0.7, 0.3], [0.6, 0.4]],
        ),
        (
            [[2.0, 0.5], [1.5, 0.25]],
            False,
            [[2.0, 0.5], [1.5, 0.25]],
        ),
        (
            [[2.0, 0.5], [1.5, 0.25]],
            True,
            [[0.8, 0.2], [0.85714287, 0.14285715]],
        ),
    ],
)
def test_renormalize_route_recovers_packed_topk_without_scaling_gfx950(
    topk_weights: list[list[float]],
    normalize_topk_weights: bool,
    expected_weights: list[list[float]],
) -> None:
    device = "cuda"
    topk_ids = torch.tensor(
        [[4, 1], [2, 7]],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.tensor(topk_weights, device=device, dtype=torch.float32)
    expected_weights = torch.tensor(
        expected_weights, device=device, dtype=torch.float32
    )
    router_logits = torch.full((2, 8), -1e20, device=device, dtype=torch.float32)
    router_logits.scatter_(1, topk_ids.long(), topk_weights.log())
    correction_bias = torch.linspace(-4.0, 4.0, 8, device=device, dtype=torch.float32)

    ragged, _, scatter, gate = _dynamic_mxfp4_route(
        router_logits,
        top_k=2,
        correction_bias=correction_bias,
        n_group=0,
        topk_group=0,
        routed_scaling_factor=3.0,
        normalize_topk_weights=normalize_topk_weights,
        routing_method_type=1,
        dtype=router_logits.dtype,
    )
    actual_weights, actual_ids = _recover_topk_from_route(
        ragged, scatter, gate, router_logits.shape[0], 2
    )

    torch.testing.assert_close(actual_ids, topk_ids)
    torch.testing.assert_close(actual_weights, expected_weights)


def test_dynamic_route_without_topk_normalization_uses_full_softmax_gfx950() -> None:
    device = "cuda"
    router_logits = torch.tensor(
        [[4.0, 3.0, 0.0, -2.0], [1.0, -1.0, 2.5, 0.5]],
        device=device,
        dtype=torch.float32,
    )

    ragged, _, scatter, gate = _dynamic_mxfp4_route(
        router_logits,
        top_k=2,
        correction_bias=None,
        n_group=0,
        topk_group=0,
        routed_scaling_factor=1.0,
        normalize_topk_weights=False,
        routing_method_type=0,
        dtype=router_logits.dtype,
    )
    actual_weights, actual_ids = _recover_topk_from_route(
        ragged, scatter, gate, router_logits.shape[0], 2
    )
    expected_weights, expected_ids = torch.softmax(router_logits, dim=-1).topk(
        2, dim=-1, sorted=True
    )

    torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(actual_weights, expected_weights)
    assert not torch.allclose(
        actual_weights.sum(dim=-1), torch.ones_like(actual_weights.sum(dim=-1))
    )


@pytest.mark.parametrize("intermediate_size", [256, 512])
def test_gluon_dynamic_mxfp4_moe_concatenated_silu_matches_torch_gfx950(
    intermediate_size: int,
) -> None:
    from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import (
        _quantize_mxfp4_activation,
    )

    torch.manual_seed(20260630)
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260631)
    m, e, h, i, topk = 4, 8, 512, intermediate_size, 2
    n_group, topk_group = 2, 1
    hidden = (
        torch.randn((m, h), device=device, dtype=torch.bfloat16) * 0.1
    ).contiguous()
    logits = torch.randn((m, e), device=device, dtype=torch.bfloat16)
    correction_bias = torch.zeros((e,), device=device, dtype=torch.float32)

    def quant_weight(
        shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _make_random_mxfp4_quantized_tensor(
            shape,
            device=device,
            generator=generator,
        )

    w13_quant, w13_scale = quant_weight((e, 2 * i, h))
    w2_quant, w2_scale = quant_weight((e, h, i))

    layer = torch.nn.Module()
    layer.quant_config = type("QuantConfig", (), {})()
    layer.quant_config.use_dynamic_mxfp4_activations = True
    layer.w13_input_layout = "concatenated"
    layer.w13_weight = torch.nn.Parameter(w13_quant.clone(), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale.clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2_quant.clone(), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale.clone(), requires_grad=False)
    layer.w13_weight_bias = torch.nn.Parameter(
        torch.zeros(e, 2 * i, device=device), requires_grad=False
    )
    layer.w2_weight_bias = torch.nn.Parameter(
        torch.zeros(e, h, device=device), requires_grad=False
    )
    preprocess_gluon_mxfp4_gfx950_moe_weights(
        {"internal_activation_dtype": "input"}, layer, preshuffle=True
    )

    out = gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        logits,
        layer.w13_weight_triton_tensor,
        layer.w2_weight_triton_tensor,
        w13_mx_scale=layer.w13_precision_config.b_mx_scale,
        w2_mx_scale=layer.w2_precision_config.b_mx_scale,
        top_k=topk,
        correction_bias=correction_bias,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        w13_bias=layer.w13_weight_bias,
        w2_bias=layer.w2_weight_bias,
        swiglu_alpha=1.0,
        swiglu_limit=0.0,
        swiglu_beta=0.0,
    )

    ragged, gather_indx, scatter_indx, gate_scal = gluon_biased_grouped_fused_route(
        logits,
        correction_bias,
        topk,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        dtype=logits.dtype,
    )

    hidden_quant, hidden_scale = _quantize_mxfp4_activation(
        hidden,
        gather_indx,
        ragged_metadata=ragged,
    )
    hidden_dequant = _mxfp4_dequant(hidden_quant, hidden_scale, ragged)
    intermediate = torch.empty((m * topk, i), device=device, dtype=torch.bfloat16)
    offset = 0
    for expert, size in enumerate(ragged.slice_sizes.to(torch.int64).cpu().tolist()):
        start, end = offset, offset + int(size)
        offset = end
        if start == end:
            continue
        w13 = _mxfp4_dequant(w13_quant[expert], w13_scale[expert])
        intermediate[start:end] = _silu_gate_up_reference(
            hidden_dequant[start:end] @ w13.T
        ).to(torch.bfloat16)

    inter_quant, inter_scale = _quantize_mxfp4_activation(
        intermediate,
        ragged_metadata=ragged,
    )
    inter_dequant = _mxfp4_dequant(inter_quant, inter_scale, ragged)
    routed = torch.empty((m * topk, h), device=device, dtype=torch.bfloat16)
    offset = 0
    for expert, size in enumerate(ragged.slice_sizes.to(torch.int64).cpu().tolist()):
        start, end = offset, offset + int(size)
        offset = end
        if start == end:
            continue
        w2 = _mxfp4_dequant(w2_quant[expert], w2_scale[expert])
        expert_out = inter_dequant[start:end] @ w2.T
        expert_out = expert_out * gate_scal[start:end].to(torch.float32)[:, None]
        routed[scatter_indx[start:end].long()] = expert_out.to(torch.bfloat16)
    ref = routed.view(m, topk, h).sum(dim=1)

    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.to(torch.float32),
        ref.to(torch.float32),
        atol=3e-3,
        rtol=3e-2,
    )


def test_gluon_mxfp4_dispatch_handles_large_expert_offsets_gfx950() -> None:
    from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import (
        _quantize_mxfp4_activation,
    )

    torch.manual_seed(20260630)
    device = "cuda"
    m, e, h, i, topk = 1, 65, 8192, 4096, 1
    selected_expert = e - 1

    hidden = (
        torch.randn((m, h), device=device, dtype=torch.bfloat16) * 0.03
    ).contiguous()
    logits = torch.full((m, e), -1000.0, device=device, dtype=torch.bfloat16)
    logits[:, selected_expert] = 1000.0
    ragged, gather_indx, _scatter_indx, _gate_scal = default_route(
        logits,
        topk,
        dtype=logits.dtype,
    )

    w13_quant = torch.empty((e, 2 * i, h // 2), device=device, dtype=torch.uint8)
    w13_scale = torch.empty(
        (e, 2 * i, h // MXFP4_BLOCK),
        device=device,
        dtype=torch.uint8,
    )
    w13_scale.fill_(120)
    w13_quant[selected_expert].random_(0, 256)
    w13_weight = torch.empty((e, h // 2, 2 * i), device=device, dtype=torch.uint8)
    w13_weight[selected_expert].copy_(w13_quant[selected_expert].transpose(-2, -1))
    w13_scale_swizzled = gluon_moe._swizzle_scales_cdna4(w13_scale)

    hidden_quant, hidden_scale = _quantize_mxfp4_activation(hidden, gather_indx)
    out = gluon_mxfp_ragged_matmul(
        hidden_quant,
        w13_weight,
        None,
        w_mx_scale=w13_scale_swizzled,
        x_mx_scale=hidden_scale,
        x_format="e2m1",
        out_dtype=torch.bfloat16,
        a_ragged_metadata=ragged,
        fused_activation=_swiglu_activation(),
    )

    hidden_dequant = _mxfp4_dequant(hidden_quant, hidden_scale)
    w13_dequant = _mxfp4_dequant(
        w13_quant[selected_expert],
        w13_scale[selected_expert],
    )
    ref = _swiglu_reference(hidden_dequant @ w13_dequant.T).to(torch.bfloat16)

    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.to(torch.float32),
        ref.to(torch.float32),
        atol=3e-3,
        rtol=3e-2,
    )


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
def test_gluon_moe_gemm1_dynamic_mxfp4_pregathered_scales_match_torch_gfx950(
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
    gemm1_input, gemm1_scale = gluon_moe._quantize_mxfp4_activation(
        hidden_states,
        gather_indx,
    )

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
            fused_activation=_swiglu_activation(),
        )
        expected = _compute_torch_gemm1_mxfp4_reference(
            mxfp4_weights.raw,
            gemm1_input,
            gemm1_scale,
            weights,
            ragged_metadata,
            None,
        )

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=GEMM_ATOL,
        rtol=RTOL,
    )
