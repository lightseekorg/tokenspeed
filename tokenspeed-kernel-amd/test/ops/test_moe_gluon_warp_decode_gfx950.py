# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

from dataclasses import dataclass

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
)

try:
    from tokenspeed.runtime.layers.moe.backends.mxfp4.triton_kernel import (
        swizzle_mxfp4,
    )
except ImportError as exc:
    pytest.skip(
        f"tokenspeed runtime MXFP4 swizzle helper is required: {exc}",
        allow_module_level=True,
    )

# Standard OCP MXFP4 (E2M1) value table; index is the 4-bit code.
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

_FP8_DTYPE = torch.float8_e4m3fn


@dataclass
class PrecisionConfig:
    b_mx_scale: torch.Tensor
    b_microblock_size: int = 32
    out_dtype: torch.dtype | None = None


def _mxfp4_dequant(packed: torch.Tensor) -> torch.Tensor:
    """Decode packed MXFP4 (two e2m1 codes per byte) to float32.

    Replaces aiter.utility.fp4_utils.mxfp4_to_f32 so the test carries no
    aiter dependency. The low nibble is the even element along the unpacked
    axis, the high nibble the odd element. Input (..., K // 2) uint8 maps to
    output (..., K) float32. All weight microscales in these cases are e8m0
    code 127, i.e. a unit scale, so no scale factor is applied here.
    """
    lut = torch.tensor(_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    lo = lut[(packed & 0x0F).long()]
    hi = lut[(packed >> 4).long()]
    return torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], -1)


def _build_case(
    *,
    M: int,
    E: int,
    D: int,
    I: int,
    topk: int,
    use_bias: bool,
    device: str = "cuda",
    seed: int = 123,
) -> dict:
    """Construct kernel inputs plus the raw weights kept for the reference."""
    torch.manual_seed(seed)
    hidden = torch.randn((M, D), device=device, dtype=torch.bfloat16)
    router = torch.randn((M, E), device=device, dtype=torch.float32)
    w13 = torch.randint(0, 256, (E, 2 * I, D // 2), device=device, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, D, I // 2), device=device, dtype=torch.uint8)
    s13 = torch.full((E, 2 * I, D // 32), 127, device=device, dtype=torch.uint8)
    s2 = torch.full((E, D, I // 32), 127, device=device, dtype=torch.uint8)
    w13_bias = (
        torch.randn((E, 2 * I), device=device, dtype=torch.float32)
        if use_bias
        else None
    )
    w2_bias = (
        torch.randn((E, D), device=device, dtype=torch.float32) if use_bias else None
    )

    wt13, _w13_flex, st13 = swizzle_mxfp4(w13, s13, 8)
    wt2, _w2_flex, st2 = swizzle_mxfp4(w2, s2, 8)
    scale1 = torch.ones((1,), device=device, dtype=torch.float32)
    scale2 = torch.ones((1,), device=device, dtype=torch.float32)
    pc1 = PrecisionConfig(b_mx_scale=st13, out_dtype=torch.bfloat16)
    pc2 = PrecisionConfig(b_mx_scale=st2, out_dtype=torch.bfloat16)
    return {
        "M": M,
        "E": E,
        "D": D,
        "I": I,
        "topk": topk,
        "use_bias": use_bias,
        "hidden": hidden,
        "router": router,
        "w13": w13,
        "w2": w2,
        "w13_bias": w13_bias,
        "w2_bias": w2_bias,
        "wt13": wt13,
        "wt2": wt2,
        "pc1": pc1,
        "pc2": pc2,
        "scale1": scale1,
        "scale2": scale2,
    }


def _quantize_fp8(
    x: torch.Tensor, *, scale: torch.Tensor, solution: str | None = None
) -> torch.Tensor:
    del scale, solution
    return x.to(_FP8_DTYPE)


def _run_kernel(case: dict) -> torch.Tensor:
    hidden_fp8 = _quantize_fp8(case["hidden"], scale=case["scale1"])
    return _gluon_mxfp4_fp8_warp_decode_moe(
        hidden_fp8,
        case["router"],
        case["wt13"],
        case["wt2"],
        w13_bias=case["w13_bias"],
        w2_bias=case["w2_bias"],
        w13_precision_config=case["pc1"],
        w2_precision_config=case["pc2"],
        w13_act_scale=case["scale1"],
        w2_act_scale=case["scale2"],
        top_k=case["topk"],
    )


def _reference(case: dict) -> torch.Tensor:
    """Pure-torch decode-MoE matching the warp kernel's swiglu + fp8 rounding."""
    M, D, I, topk = case["M"], case["D"], case["I"], case["topk"]
    device = case["hidden"].device
    use_bias = case["use_bias"]
    router, w13, w2 = case["router"], case["w13"], case["w2"]
    w13_bias, w2_bias = case["w13_bias"], case["w2_bias"]

    topk_vals, topk_ids = torch.topk(router, topk, dim=-1)
    topk_weights = torch.softmax(topk_vals, dim=-1)
    hidden_fp8 = case["hidden"].to(_FP8_DTYPE).to(torch.float32)
    seven = torch.tensor(7.0, device=device)

    # Dequant only the experts that are actually routed to, keeping memory
    # bounded for the larger decode shapes.
    deq_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _expert_weights(expert: int) -> tuple[torch.Tensor, torch.Tensor]:
        if expert not in deq_cache:
            deq_cache[expert] = (
                _mxfp4_dequant(w13[expert]),
                _mxfp4_dequant(w2[expert]),
            )
        return deq_cache[expert]

    ref = torch.zeros((M, D), device=device, dtype=torch.float32)
    for m in range(M):
        for slot in range(topk):
            expert = int(topk_ids[m, slot])
            w13_f, w2_f = _expert_weights(expert)
            gate_up = hidden_fp8[m : m + 1] @ w13_f.T
            if use_bias:
                gate_up = gate_up + w13_bias[expert][None, :]
            gate = torch.minimum(gate_up[:, :I], seven)
            linear = torch.clamp(gate_up[:, I:], -7.0, 7.0)
            inter = (gate / (1.0 + torch.exp(-1.702 * gate))) * (linear + 1.0)
            inter_fp8 = inter.to(_FP8_DTYPE).to(torch.float32)
            second = inter_fp8 @ w2_f.T
            if use_bias:
                second = second + w2_bias[expert][None, :]
            ref[m] += topk_weights[m, slot] * second.squeeze(0)
    return ref


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("M", [1, 2, 4])
def test_fp8_mxfp4_warp_decode_moe(M: int, use_bias: bool):
    # I = 256 > BLOCK_K (128) so stage2 split-K partitions the reduction across
    # real K slices. M sweeps the supported warp-decode range (M<=4)
    # and its tiling transitions (stage2 at M>1).
    case = _build_case(M=M, E=4, D=256, I=256, topk=2, use_bias=use_bias)
    out = _run_kernel(case)
    assert out is not None
    torch.cuda.synchronize()
    ref = _reference(case)
    torch.testing.assert_close(
        out.float(), ref.to(torch.bfloat16).float(), rtol=5e-2, atol=2.0
    )


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("M", [8, 16])
def test_fp8_mxfp4_warp_decode_rejects_larger_m(M: int, use_bias: bool):
    # M>=8 is deliberately handled by the medium-decode direct kernels in the
    # generic fused-MoE path, not by the warp-decode path.
    case = _build_case(M=M, E=4, D=256, I=256, topk=2, use_bias=use_bias)
    assert _run_kernel(case) is None
