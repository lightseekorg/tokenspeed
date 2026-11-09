"""Fp8LinearMethod MXFP8 (1,32): canonical and prepacked weight scales."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.dense.fp8 import (
    Fp8LinearMethod,
    prepare_fp8_linear_weight_scales,
)
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.quantization.fp8 import Mxfp8Config
from tokenspeed.runtime.utils.env import global_server_args_dict

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _make_layer(n: int, k: int, device: str = "cuda") -> torch.nn.Module:
    torch.manual_seed(0)
    layer = torch.nn.Module()
    weight = (torch.randn(n, k, device=device) * 0.02).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 130, (n, k // 32), device=device, dtype=torch.uint8)
    layer.weight = Parameter(weight, requires_grad=False)
    layer.weight_scale_inv = Parameter(scales, requires_grad=False)
    return layer


def _config() -> Mxfp8Config:
    return Mxfp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[1, 32],
        scale_fmt="ue8m0",
    )


def _method() -> Fp8LinearMethod:
    return Fp8LinearMethod(_config())


def test_process_weights_preserves_canonical_scales() -> None:
    n, k = 256, 512
    layer = _make_layer(n, k)
    ref_weight = layer.weight.data.clone()
    ref_scales = layer.weight_scale_inv.data.clone()

    method = _method()
    method.process_weights_after_loading(layer)

    assert torch.equal(layer.weight_scale_inv, ref_scales)
    assert torch.equal(layer.weight.data, ref_weight)

    x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    out = method.apply(layer, x)

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

    assert layer.weight_scale_inv.dim() == 2

    x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    out = _method().apply(layer, x)
    assert torch.isfinite(out).all()


requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_blackwell_plus,
    reason="prepacked MXFP8 scales feed the Blackwell native GEMM",
)


@pytest.fixture
def prepacked_policy(monkeypatch):
    # prepare_mxfp8_weight_scales serves the native GEMM only with PDL on.
    monkeypatch.setitem(global_server_args_dict, "dense_gemm_backend", "auto")
    previous = pdl_enabled()
    pdl_enabled(True)
    yield
    pdl_enabled(previous)


def _loaded_linear(n: int, k: int) -> ReplicatedLinear:
    with torch.device("cuda"):
        layer = ReplicatedLinear(
            k, n, bias=False, params_dtype=torch.bfloat16, quant_config=_config()
        )
    generator = torch.Generator(device="cuda").manual_seed(n * k)
    weight = torch.randn(n, k, device="cuda", generator=generator) * 0.02
    layer.weight.data.copy_(weight.to(torch.float8_e4m3fn))
    layer.weight_scale_inv.data.copy_(
        torch.randint(
            118,
            132,
            (n, k // 32),
            device="cuda",
            dtype=torch.uint8,
            generator=generator,
        )
    )
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def _canonical(layer: ReplicatedLinear, x: torch.Tensor) -> torch.Tensor:
    """The layer's output with canonical scales, swizzled inside the GEMM call."""
    codes, scales = quantize_fp8(
        x, granularity="token_group", group_size=32, scale_encoding="ue8m0"
    )
    return mm(
        codes,
        layer.weight,
        A_scales=scales,
        B_scales=layer.weight_scale_inv,
        out_dtype=x.dtype,
        quant="mxfp8",
        block_size=[1, 32],
    )


@requires_blackwell
@torch.inference_mode()
def test_prepared_scales_match_the_canonical_path(prepacked_policy) -> None:
    layer = _loaded_linear(384, 576)
    canonical_scales = layer.weight_scale_inv.data.clone()
    # Preparation is opt-in: loading alone keeps the canonical path.
    assert layer.quant_method.prepared_weight_scales is None

    prepare_fp8_linear_weight_scales(torch.nn.Sequential(layer))

    assert layer.quant_method.prepared_weight_scales is not None
    assert torch.equal(layer.weight_scale_inv, canonical_scales)
    for rows in (1, 4, 129):
        x = torch.randn(rows, 576, device="cuda", dtype=torch.bfloat16)
        out, _ = layer(x)
        assert torch.equal(out, _canonical(layer, x))
        # Pre-quantized codes with canonical scales reuse the prepared scales.
        codes, scales = quantize_fp8(
            x, granularity="token_group", group_size=32, scale_encoding="ue8m0"
        )
        shared = layer.quant_method.apply(layer, codes, None, scales, x.dtype)
        assert torch.equal(shared, out)
