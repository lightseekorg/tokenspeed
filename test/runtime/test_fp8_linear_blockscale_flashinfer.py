"""Dense FP8 (128,128) load-time scale preparation for FlashInfer."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm.flashinfer import has_flashinfer_fp8_blockscale
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or has_flashinfer_fp8_blockscale is None
    or not has_flashinfer_fp8_blockscale(),
    reason="requires SM100 CUDA and FlashInfer FP8 block-scale GEMM",
)


def _make_layer(n: int, k: int) -> torch.nn.Module:
    torch.manual_seed(0)
    layer = torch.nn.Module()
    weight = (torch.randn(n, k, device="cuda") * 0.02).to(torch.float8_e4m3fn)
    scales = (
        torch.rand(n // 128, k // 128, device="cuda", dtype=torch.float32) * 0.02
        + 0.001
    )
    layer.weight = Parameter(weight, requires_grad=False)
    layer.weight_scale_inv = Parameter(scales, requires_grad=False)
    return layer


def _method() -> Fp8LinearMethod:
    return Fp8LinearMethod(
        Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
    )


@pytest.mark.parametrize("m", [1, 3, 4, 5])
def test_process_weights_prepares_and_uses_native_scales(m: int) -> None:
    n, k = 256, 512
    layer = _make_layer(n, k)
    canonical_scales = layer.weight_scale_inv.data.clone()
    method = _method()

    method.process_weights_after_loading(layer)

    assert method.prepared_linear_plan(layer) is not None
    assert torch.equal(layer.weight_scale_inv, canonical_scales)

    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    prepared = method.apply(layer, x)
    dequant = layer.weight.float() * canonical_scales.repeat_interleave(
        128, dim=0
    ).repeat_interleave(128, dim=1)
    reference = x.float() @ dequant.t()
    torch.testing.assert_close(prepared.float(), reference, atol=2e-1, rtol=5e-2)
