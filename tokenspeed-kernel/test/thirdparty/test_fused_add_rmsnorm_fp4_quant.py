# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Unit tests for the fused (residual+input) -> RMSNorm -> NVFP4 quantize kernel.

The kernel is a port of TRT-LLM's ``ws_layernorm`` (warp-specialized) variant.
For each (M, N, dtype) shape we compare against a reference pipeline composed of
PyTorch RMSNorm + ``flashinfer.fp4_quantize`` (swizzled SF layout):

* ``fp4_packed`` must be **bit-exact** vs. the reference (same packed FP4 bytes)
* ``sf_out`` must be **bit-exact** vs. the reference (same swizzled SF bytes)
* ``residual_out`` must equal ``hidden_states + residual`` within a small tol
* ``hp_norm`` must equal the reference RMSNorm output within ``1e-3``

Run::

    pytest tokenspeed-kernel/test/thirdparty/test_fused_add_rmsnorm_fp4_quant.py -v
"""

from __future__ import annotations

import pytest
import torch


def _has_sm90_or_newer() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9
    )


def _has_sm100() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
    )


# The fused kernel produces SF in TRT-LLM swizzled layout. We compare against
# flashinfer ``fp4_quantize`` with ``is_sf_swizzled_layout=True`` for bit-exact
# parity. NVFP4 quant requires Blackwell (SM100) on the reference side.
pytestmark = pytest.mark.skipif(
    not _has_sm100(),
    reason="fused_add_rmsnorm_fp4_quant requires Blackwell SM100",
)


def _rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm matching TRT-LLM ws_layernorm semantics (fp32 reduce)."""
    x32 = x.float()
    variance = x32.pow(2).mean(-1, keepdim=True)
    normed = x32 * torch.rsqrt(variance + eps)
    return (normed * weight.float()).to(x.dtype)


def _padded_sf_size(m: int, n: int) -> int:
    """Mirror ``computeSwizzledLayoutSFSize(m_padded, n // 16)`` exactly."""
    m_padded = (m + 31) // 32 * 32
    sf_rows = (m_padded + 127) // 128 * 128
    sf_cols = (n // 16 + 3) // 4 * 4
    return sf_rows * sf_cols


@pytest.mark.parametrize(
    ("m", "n"),
    [
        pytest.param(1, 4096, id="m1_n4096"),
        pytest.param(16, 5120, id="m16_n5120"),
        pytest.param(1024, 8192, id="m1024_n8192"),
        pytest.param(1, 16384, id="m1_n16384"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float16, id="fp16"),
    ],
)
def test_fused_add_rmsnorm_fp4_quant_matches_reference(
    m: int, n: int, dtype: torch.dtype
) -> None:
    """Fused kernel must match RMSNorm + flashinfer.fp4_quantize byte-for-byte."""
    from tokenspeed_kernel.thirdparty.cuda import fused_add_rmsnorm_fp4_quant
    from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize

    torch.manual_seed(0xC0FFEE ^ (m * 131 + n))
    device = torch.device("cuda")
    eps = 1e-6

    # Use modest activations so RMSNorm doesn't blow up; weight ~ 1.0.
    hidden_states = torch.randn(m, n, device=device, dtype=dtype) * 0.5
    residual = torch.randn(m, n, device=device, dtype=dtype) * 0.5
    weight = torch.randn(n, device=device, dtype=dtype) * 0.1 + 1.0

    # sf_scale = 1 / input_scale; flashinfer's fp4_quantize takes the same.
    pre_norm = (hidden_states.float() + residual.float())
    normed_ref_fp32 = (
        pre_norm * torch.rsqrt(pre_norm.pow(2).mean(-1, keepdim=True) + eps)
        * weight.float()
    )
    amax = normed_ref_fp32.abs().amax().clamp(min=1e-8)
    sf_scale = ((448.0 * 6.0) / amax).view(1).to(torch.float32)

    # ── Reference path: torch RMSNorm + flashinfer fp4_quantize (swizzled).
    pre_norm_ref = (hidden_states + residual).contiguous()
    normed_ref = _rmsnorm_ref(pre_norm_ref, weight, eps)
    fp4_ref, sf_ref = fp4_quantize(
        normed_ref,
        sf_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )

    # ── Fused path.
    fp4, residual_out, sf_out, hp_norm = fused_add_rmsnorm_fp4_quant(
        hidden_states,
        residual,
        weight,
        sf_scale,
        eps=eps,
        output_hp_norm=True,
    )

    # Shape & dtype contracts.
    assert fp4.shape == (m, n // 2), f"fp4 shape {tuple(fp4.shape)} != ({m}, {n // 2})"
    assert fp4.dtype == torch.uint8
    assert residual_out.shape == (m, n)
    assert residual_out.dtype == dtype
    assert hp_norm is not None
    assert hp_norm.shape == (m, n)
    assert hp_norm.dtype == dtype
    expected_sf_numel = _padded_sf_size(m, n)
    assert sf_out.numel() == expected_sf_numel
    assert sf_out.dtype == torch.uint8

    # ── 1) residual_out == hidden_states + residual (high precision).
    torch.testing.assert_close(
        residual_out.float(),
        pre_norm_ref.float(),
        atol=1e-3,
        rtol=1e-3,
        msg="residual_out mismatch (input + residual)",
    )

    # ── 2) hp_norm == ref RMSNorm output within 1e-3.
    torch.testing.assert_close(
        hp_norm.float(),
        normed_ref.float(),
        atol=1e-3,
        rtol=1e-3,
        msg="hp_norm mismatch vs torch RMSNorm",
    )

    # ── 3) FP4 packed bytes are bit-exact.
    # flashinfer ``fp4_quantize`` returns the FP4 buffer either as int64 packed
    # (cuda 13 path) or uint8 packed; normalize to uint8 [M, N//2] for compare.
    fp4_ref_u8 = fp4_ref.view(torch.uint8).reshape(m, n // 2)
    fp4_u8 = fp4.view(torch.uint8).reshape(m, n // 2)
    assert torch.equal(fp4_u8, fp4_ref_u8), (
        f"FP4 packed bytes differ for (m={m}, n={n}, dtype={dtype}); "
        f"first diff at {(fp4_u8 != fp4_ref_u8).nonzero()[:4].tolist()}"
    )

    # ── 4) Swizzled SF buffer is bit-exact.
    sf_ref_u8 = sf_ref.view(torch.uint8).reshape(-1)
    sf_out_u8 = sf_out.view(torch.uint8).reshape(-1)
    # Reference flashinfer may return only the live portion (M, N//16) in
    # swizzled layout; we compare the prefix that both kernels agree on.
    common = min(sf_ref_u8.numel(), sf_out_u8.numel())
    assert common > 0, "empty sf buffers"
    assert torch.equal(sf_out_u8[:common], sf_ref_u8[:common]), (
        f"SF bytes differ for (m={m}, n={n}, dtype={dtype}); "
        f"first diff at {(sf_out_u8[:common] != sf_ref_u8[:common]).nonzero()[:4].tolist()}"
    )


def test_fused_add_rmsnorm_fp4_quant_skip_hp_norm() -> None:
    """``output_hp_norm=False`` should skip allocating the hp branch."""
    from tokenspeed_kernel.thirdparty.cuda import fused_add_rmsnorm_fp4_quant

    torch.manual_seed(0)
    m, n = 8, 4096
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eps = 1e-6

    hidden_states = torch.randn(m, n, device=device, dtype=dtype) * 0.5
    residual = torch.randn(m, n, device=device, dtype=dtype) * 0.5
    weight = torch.ones(n, device=device, dtype=dtype)
    sf_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    fp4, residual_out, sf_out, hp_norm = fused_add_rmsnorm_fp4_quant(
        hidden_states,
        residual,
        weight,
        sf_scale,
        eps=eps,
        output_hp_norm=False,
    )

    assert hp_norm is None
    assert fp4.shape == (m, n // 2)
    assert residual_out.shape == (m, n)
    assert sf_out.dtype == torch.uint8


def test_fused_add_rmsnorm_fp4_quant_invalid_n_rejected() -> None:
    """Hidden dim outside [2048, 16384] or not %16 must be rejected up front."""
    from tokenspeed_kernel.thirdparty.cuda import fused_add_rmsnorm_fp4_quant

    device = torch.device("cuda")
    dtype = torch.bfloat16
    weight = torch.ones(1024, device=device, dtype=dtype)  # too small N
    hidden_states = torch.randn(4, 1024, device=device, dtype=dtype)
    residual = torch.randn(4, 1024, device=device, dtype=dtype)
    sf_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    with pytest.raises(ValueError, match="must be in"):
        fused_add_rmsnorm_fp4_quant(
            hidden_states, residual, weight, sf_scale, eps=1e-6, output_hp_norm=False,
        )
