# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import importlib

import pytest
import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip(
        "AMD GFX950 is required for Gluon warp-decode tests",
        allow_module_level=True,
    )

from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import (  # noqa: E402
    _gluon_mxfp4_fp8_warp_decode_moe,
    fp8_quantize,
    gluon_mxfp_fused_moe,
)

_gluon_mxfp4 = importlib.import_module("tokenspeed_kernel.ops.moe.gluon.mxfp4")
_triton_mxfp4 = importlib.import_module("tokenspeed_kernel.ops.moe.triton.mxfp4")
gluon_mxfp4_moe_process_weights = _gluon_mxfp4.gluon_mxfp4_moe_process_weights
triton_mxfp4_moe_process_weights = _triton_mxfp4.triton_mxfp4_moe_process_weights

WEIGHT_NIBBLES = (0, 1, 2, 9, 10)
WARP_DECODE_ATOL = 64.0


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


def _build_case(
    *,
    M: int,
    E: int,
    D: int,
    I: int,
    topk: int,
    use_bias: bool,
    preshuffle: bool,
    device: str = "cuda",
    seed: int = 123,
) -> dict:
    """Construct kernel inputs plus the raw weights kept for the reference."""
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden = (
        torch.randint(-4, 5, (M, D), device=device, generator=generator).to(
            torch.float32
        )
        / 16.0
    ).to(torch.bfloat16)
    router = torch.randn(
        (M, E), device=device, dtype=torch.float32, generator=generator
    )
    w13 = _make_mxfp4_weight_bytes(
        (E, 2 * I, D // 2), device=device, generator=generator
    )
    w2 = _make_mxfp4_weight_bytes((E, D, I // 2), device=device, generator=generator)
    s13 = torch.full((E, 2 * I, D // 32), 127, device=device, dtype=torch.uint8)
    s2 = torch.full((E, D, I // 32), 127, device=device, dtype=torch.uint8)
    w13_bias = torch.randn((E, 2 * I), device=device, dtype=torch.float32)
    w2_bias = torch.randn((E, D), device=device, dtype=torch.float32)

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(s13, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(s2, requires_grad=False)
    layer.w13_weight_bias = torch.nn.Parameter(w13_bias, requires_grad=False)
    layer.w2_weight_bias = torch.nn.Parameter(w2_bias, requires_grad=False)
    layer.w13_input_scale = torch.nn.Parameter(
        torch.ones((E,), device=device, dtype=torch.float32),
        requires_grad=False,
    )
    layer.w2_input_scale = torch.nn.Parameter(
        torch.ones((E,), device=device, dtype=torch.float32),
        requires_grad=False,
    )

    plan = {"internal_activation_dtype": "fp8"}
    if preshuffle:
        gluon_mxfp4_moe_process_weights(plan, layer)
    else:
        triton_mxfp4_moe_process_weights(plan, layer)

    return {
        "M": M,
        "topk": topk,
        "use_bias": use_bias,
        "hidden": hidden,
        "router": router,
        "w13_weight": layer.w13_weight_triton_tensor,
        "w2_weight": layer.w2_weight_triton_tensor,
        "w13_bias": layer.w13_weight_bias if use_bias else None,
        "w2_bias": layer.w2_weight_bias if use_bias else None,
        "pc1": layer.w13_precision_config,
        "pc2": layer.w2_precision_config,
        "scale1": layer.w13_act_scale,
        "scale2": layer.w2_act_scale,
    }


def _run_warp_decode(case: dict) -> torch.Tensor:
    x_fp8 = fp8_quantize(case["hidden"], case["scale1"])
    return _gluon_mxfp4_fp8_warp_decode_moe(
        x_fp8,
        case["router"],
        case["w13_weight"],
        case["w2_weight"],
        w13_bias=case["w13_bias"],
        w2_bias=case["w2_bias"],
        w13_precision_config=case["pc1"],
        w2_precision_config=case["pc2"],
        w13_act_scale=case["scale1"],
        w2_act_scale=case["scale2"],
        top_k=case["topk"],
    )


def _run_non_warp_reference(case: dict) -> torch.Tensor:
    return gluon_mxfp_fused_moe(
        case["hidden"],
        case["router"],
        case["w13_weight"],
        case["w2_weight"],
        w13_bias=case["w13_bias"],
        w2_bias=case["w2_bias"],
        w13_precision_config=case["pc1"],
        w2_precision_config=case["pc2"],
        w13_act_scale=case["scale1"],
        w2_act_scale=case["scale2"],
        top_k=case["topk"],
        enable_warp_decode=False,
    )


@pytest.mark.parametrize("preshuffle", [False, True])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("M", [1, 4, 16])
def test_fp8_mxfp4_warp_decode_moe(M: int, use_bias: bool, preshuffle: bool):
    # GPT-OSS-like K/N dimensions exercise the production warp-decode
    # specializations. M sweeps the supported decode range and its stage2
    # split-K transitions.
    case = _build_case(
        M=M,
        E=4,
        D=2880,
        I=2880,
        topk=2,
        use_bias=use_bias,
        preshuffle=preshuffle,
    )
    out = _run_warp_decode(case)
    assert out is not None
    torch.cuda.synchronize()
    ref = _run_non_warp_reference(case)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.float(), ref.float(), rtol=0.0, atol=WARP_DECODE_ATOL
    )
