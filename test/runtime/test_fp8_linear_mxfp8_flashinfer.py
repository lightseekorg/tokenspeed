"""Fp8LinearMethod MXFP8 (1,32) path: load-time scale swizzle + flashinfer pin."""

from __future__ import annotations

import pytest
import torch
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod, has_flashinfer_mxfp8
from tokenspeed.runtime.layers.quantization.fp8 import Mxfp8Config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or has_flashinfer_mxfp8 is None
    or not has_flashinfer_mxfp8(),
    reason="requires SM100/103 CUDA and flashinfer mm_mxfp8",
)


def _make_layer(n: int, k: int, device: str = "cuda") -> torch.nn.Module:
    torch.manual_seed(0)
    layer = torch.nn.Module()
    weight = (torch.randn(n, k, device=device) * 0.02).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 130, (n, k // 32), device=device, dtype=torch.uint8)
    layer.weight = Parameter(weight, requires_grad=False)
    layer.weight_scale_inv = Parameter(scales, requires_grad=False)
    return layer


def _method() -> Fp8LinearMethod:
    return Fp8LinearMethod(
        Mxfp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[1, 32],
            scale_fmt="ue8m0",
        )
    )


def test_process_weights_swizzles_and_pins_flashinfer() -> None:
    n, k = 256, 512
    layer = _make_layer(n, k)
    ref_weight = layer.weight.data.clone()
    ref_scales = layer.weight_scale_inv.data.clone()

    _method().process_weights_after_loading(layer)

    assert layer._use_flashinfer_mxfp8
    assert not layer._use_deep_gemm_fp8
    assert layer.weight_scale_inv.dim() == 1
    assert torch.equal(layer.weight.data, ref_weight)

    x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    out = _method().apply(layer, x)

    dequant = ref_weight.float() * torch.exp2(
        ref_scales.float() - 127.0
    ).repeat_interleave(32, dim=1)
    ref = x.float() @ dequant.t()
    rel = (torch.norm(out.float() - ref) / torch.norm(ref)).item()
    assert rel < 5e-2, f"rel_l2={rel}"


def test_small_layers_keep_triton_fallback() -> None:
    # N < 128 is below the flashinfer problem-size floor; the layer must
    # keep row-major scales so the Triton kernel stays selectable.
    layer = _make_layer(64, 512)
    _method().process_weights_after_loading(layer)

    assert not layer._use_flashinfer_mxfp8
    assert layer.weight_scale_inv.dim() == 2

    x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    out = _method().apply(layer, x)
    assert torch.isfinite(out).all()
