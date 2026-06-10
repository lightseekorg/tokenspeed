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

The kernel is a port of TRT-LLM's ``ws_layernorm`` (warp-specialized) variant
which performs the (input + residual) accumulation, RMSNorm, gamma multiply and
NVFP4 block-scale quant entirely in fp32 and only rounds back to the input
dtype at the very last step. The natural reference therefore is also a
fp32-presum pipeline; comparing against a *bf16/fp16-presum* reference would
disagree by ~1 dtype ULP (≈8e-3 for bf16), which is below any meaningful
correctness threshold and dominated by ``rsqrt`` round-off, not algorithmic
drift.

We therefore validate against:

* a **fp32-presum** PyTorch RMSNorm reference (exactly what the kernel computes
  before the final dtype cast) for ``hp_norm`` and ``residual_out``;
* a **dequantized** comparison for the NVFP4 outputs — both the fused and the
  ``flashinfer.fp4_quantize`` reference are decoded back to fp32
  (``e2m1_value * fp8_e4m3(sf) / sf_scale``) and compared with a relative
  tolerance corresponding to ~1 e2m1 codestep. Byte-exact NVFP4 parity is not
  achievable across implementations because block-max reductions and the
  fp32→e4m3 SF rounding are sensitive to fp32 ULP-level differences in the
  upstream RMSNorm output, which legitimately translates into occasional
  +/-1 e2m1 codestep flips on quantization-boundary values.

Run::

    pytest tokenspeed-kernel/test/thirdparty/test_fused_add_rmsnorm_fp4_quant.py -v
"""

from __future__ import annotations

import pytest
import torch


# E2M1 absolute code table (signed bit handled separately). Values are exact
# representations of the 4-bit float; see TRT-LLM/flashinfer fp4 docs.
_E2M1_ABS_TABLE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


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


# The fused kernel writes SF in row-major LINEAR layout. We compare against
# flashinfer ``fp4_quantize(is_sf_swizzled_layout=False)`` for bit-exact parity.
# NVFP4 quant requires Blackwell (SM100) on the reference side.
pytestmark = pytest.mark.skipif(
    not _has_sm100(),
    reason="fused_add_rmsnorm_fp4_quant requires Blackwell SM100",
)


def _rmsnorm_ref_fp32(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference RMSNorm matching the kernel's fp32-presum semantics.

    The fused kernel widens both ``hidden_states`` and ``residual`` to fp32
    before the residual add, performs the variance / rsqrt / gamma multiply in
    fp32, and only casts back to the input dtype at the final store. We mirror
    that exactly here so the reference is comparable at fp32 precision (modulo
    a single final-cast ULP).

    Returns ``(normed, residual_out)``:
        * ``normed``: ``[M, N]`` ``out_dtype`` — the fused ``hp_norm`` target.
        * ``residual_out``: ``[M, N]`` ``out_dtype`` — the fused
          ``residual_out`` target (= ``out_dtype(fp32(hs) + fp32(residual))``).
    """
    pre_norm_fp32 = hidden_states.float() + residual.float()
    inv_rms = torch.rsqrt(pre_norm_fp32.pow(2).mean(-1, keepdim=True) + eps)
    normed_fp32 = pre_norm_fp32 * inv_rms * weight.float()
    return normed_fp32.to(out_dtype), pre_norm_fp32.to(out_dtype)


def _dequant_nvfp4(
    fp4_u8: torch.Tensor,
    sf_linear_u8: torch.Tensor,
    sf_scale: torch.Tensor,
    m: int,
    n: int,
) -> torch.Tensor:
    """Dequantize an NVFP4 (packed FP4 + linear e4m3 SF) buffer back to fp32.

    Args:
        fp4_u8: ``[M, N // 2]`` ``uint8`` packed FP4 (low nibble first).
        sf_linear_u8: ``[M, N // 16]`` ``uint8`` SF buffer in row-major LINEAR
            layout (the layout the fused kernel now writes directly).
        sf_scale: ``[1]`` ``float32`` global scale (= 1 / input_scale).
        m, n: original logical shape.

    Returns:
        ``[M, N]`` ``float32`` reconstruction
        ``= e2m1(fp4) * fp8_e4m3(sf) / sf_scale``.
    """
    device = fp4_u8.device

    # ---- Decode E2M1 nibbles -> fp32 values ([M, N]) ------------------------
    table = torch.tensor(_E2M1_ABS_TABLE, dtype=torch.float32, device=device)
    lo = (fp4_u8 & 0x07).long()  # absolute code for even lane
    hi = ((fp4_u8 >> 4) & 0x07).long()  # absolute code for odd lane
    sign_lo = torch.where((fp4_u8 & 0x08) != 0, -1.0, 1.0)
    sign_hi = torch.where((fp4_u8 & 0x80) != 0, -1.0, 1.0)
    vals_lo = table[lo] * sign_lo  # [M, N // 2]
    vals_hi = table[hi] * sign_hi
    e2m1 = torch.empty((m, n), dtype=torch.float32, device=device)
    e2m1[:, 0::2] = vals_lo
    e2m1[:, 1::2] = vals_hi

    # ---- LINEAR SF -> [M, N // 16] e4m3 codes -> fp32 SFValue --------------
    sf_fp32 = sf_linear_u8.contiguous().view(torch.float8_e4m3fn).float()

    # ---- Combine: each [M, k*16:(k+1)*16] block scales by sf_fp32[M, k] ----
    sf_per_elem = sf_fp32.repeat_interleave(16, dim=1)  # [M, N]
    return e2m1 * sf_per_elem / sf_scale.float().item()


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
    """Fused kernel must match a fp32-presum RMSNorm + NVFP4 reference.

    The kernel runs the entire (input + residual) → variance → rsqrt → gamma
    pipeline in fp32 and only casts back to ``dtype`` at the final store. We
    therefore compare against a fp32-presum reference (see ``_rmsnorm_ref_fp32``)
    rather than a ``dtype``-presum one — the latter would disagree by ~1 dtype
    ULP (≈8e-3 for bf16) without any algorithmic difference.

    For NVFP4 we dequantize both buffers back to fp32 and compare with a tight
    tolerance. Byte-exact NVFP4 parity across implementations is not achievable
    because the per-block max reduction and the fp32 → e4m3 SF rounding amplify
    fp32-ULP differences in the upstream RMSNorm output into occasional ±1
    e2m1 codestep flips on quantization-boundary values.
    """
    from tokenspeed_kernel.thirdparty.cuda import fused_add_rmsnorm_fp4_quant
    from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize

    torch.manual_seed(0xC0FFEE ^ (m * 131 + n))
    device = torch.device("cuda")
    eps = 1e-6

    # Use modest activations so RMSNorm doesn't blow up; weight ~ 1.0.
    hidden_states = torch.randn(m, n, device=device, dtype=dtype) * 0.5
    residual = torch.randn(m, n, device=device, dtype=dtype) * 0.5
    weight = torch.randn(n, device=device, dtype=dtype) * 0.1 + 1.0

    # ── Reference RMSNorm at the kernel's true precision (fp32-presum).
    normed_ref, residual_ref = _rmsnorm_ref_fp32(
        hidden_states, residual, weight, eps, dtype
    )

    # sf_scale = 448*6 / amax(normed_ref_fp32). Use the fp32 path so the global
    # scale matches what the fused kernel sees internally.
    pre_norm_fp32 = hidden_states.float() + residual.float()
    normed_ref_fp32 = (
        pre_norm_fp32
        * torch.rsqrt(pre_norm_fp32.pow(2).mean(-1, keepdim=True) + eps)
        * weight.float()
    )
    amax = normed_ref_fp32.abs().amax().clamp(min=1e-8)
    sf_scale = ((448.0 * 6.0) / amax).view(1).to(torch.float32)

    # ── Reference NVFP4 path: feed the *fp32-presum* normed (cast to dtype) into
    #    flashinfer.fp4_quantize so both implementations quantize the same input.
    fp4_ref, sf_ref = fp4_quantize(
        normed_ref,
        sf_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=False,
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
    assert sf_out.shape == (m, n // 16), (
        f"sf_out shape {tuple(sf_out.shape)} != ({m}, {n // 16})"
    )
    assert sf_out.dtype == torch.uint8

    # Per-dtype tolerance ≈ 1 ULP of the storage type. bf16 has 7 mantissa bits
    # (ULP ≈ 7.8e-3 around 1.0); fp16 has 10 mantissa bits (ULP ≈ 9.8e-4).
    if dtype is torch.bfloat16:
        norm_atol, norm_rtol = 1.0e-2, 1.0e-2
    else:
        norm_atol, norm_rtol = 2.0e-3, 2.0e-3

    # ── 1) residual_out == fp32(hs) + fp32(residual), cast to dtype.
    torch.testing.assert_close(
        residual_out.float(),
        residual_ref.float(),
        atol=norm_atol,
        rtol=norm_rtol,
        msg="residual_out mismatch (input + residual, fp32-presum)",
    )

    # ── 2) hp_norm == fp32-presum RMSNorm reference within ~1 dtype ULP.
    torch.testing.assert_close(
        hp_norm.float(),
        normed_ref.float(),
        atol=norm_atol,
        rtol=norm_rtol,
        msg="hp_norm mismatch vs fp32-presum RMSNorm",
    )

    # ── 3) NVFP4: dequantize both buffers and compare with relative tolerance.
    # flashinfer ``fp4_quantize`` may return FP4 packed as int64 (cuda 13 path)
    # or uint8; normalize to uint8 [M, N // 2] before decoding.
    fp4_ref_u8 = fp4_ref.view(torch.uint8).reshape(m, n // 2)
    fp4_u8 = fp4.view(torch.uint8).reshape(m, n // 2)
    sf_ref_u8 = sf_ref.view(torch.uint8).reshape(m, n // 16)
    sf_out_u8 = sf_out.view(torch.uint8).reshape(m, n // 16)

    deq_fused = _dequant_nvfp4(fp4_u8, sf_out_u8, sf_scale, m, n)
    deq_ref = _dequant_nvfp4(fp4_ref_u8, sf_ref_u8, sf_scale, m, n)

    # 1 e2m1 codestep at magnitude ~1 is 0.5; the typical SF ≈ amax_block/6 so
    # one codestep on the dequantized output is ~ amax_block/12. We compare
    # both buffers against the fp32 reference (to bound absolute error) and
    # against each other with a rel. tolerance of 1 e2m1 codestep + a small
    # SF rounding margin.
    quant_atol = 0.5 * (amax.item() / 6.0) + 1.0e-3  # ~ 1 e2m1 codestep
    quant_rtol = 0.20  # NVFP4 has ~3 effective bits of precision

    torch.testing.assert_close(
        deq_fused,
        deq_ref,
        atol=quant_atol,
        rtol=quant_rtol,
        msg=(
            "Dequantized NVFP4 (fused) disagrees with flashinfer reference "
            f"beyond ~1 e2m1 codestep for (m={m}, n={n}, dtype={dtype})"
        ),
    )

    # Also bound the absolute error of the fused output against the true
    # (fp32) RMSNorm result. This catches systematic SF / packing bugs that
    # would otherwise cancel out if both decoders agree on a wrong value.
    torch.testing.assert_close(
        deq_fused,
        normed_ref_fp32,
        atol=quant_atol,
        rtol=quant_rtol,
        msg=(
            "Dequantized NVFP4 (fused) disagrees with fp32 RMSNorm reference "
            f"beyond ~1 e2m1 codestep for (m={m}, n={n}, dtype={dtype})"
        ),
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


@pytest.mark.parametrize(
    ("m", "n"),
    [
        pytest.param(1, 4096, id="m1_n4096"),
        pytest.param(16, 5120, id="m16_n5120"),
        pytest.param(64, 8192, id="m64_n8192"),
    ],
)
def test_fused_add_rmsnorm_fp4_quant_sf_layout_valid(m: int, n: int) -> None:
    """SF output must be [M, N//16] uint8 (LINEAR row-major) without NaN/Inf."""
    from tokenspeed_kernel.thirdparty.cuda import fused_add_rmsnorm_fp4_quant

    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eps = 1e-6

    hidden_states = torch.randn(m, n, device=device, dtype=dtype) * 0.5
    residual = torch.randn(m, n, device=device, dtype=dtype) * 0.5
    weight = torch.randn(n, device=device, dtype=dtype) * 0.1 + 1.0
    sf_scale = torch.tensor([10.0], device=device, dtype=torch.float32)

    fp4, residual_out, sf_out, hp_norm = fused_add_rmsnorm_fp4_quant(
        hidden_states, residual, weight, sf_scale, eps=eps, output_hp_norm=True,
    )

    # Basic shape checks
    assert fp4.shape == (m, n // 2)
    assert residual_out.shape == (m, n)
    assert hp_norm.shape == (m, n)
    assert sf_out.dtype == torch.uint8
    assert sf_out.shape == (m, n // 16)

    # Interpret as e4m3 and check no NaN/Inf in the float conversion
    sf_fp32 = sf_out.contiguous().view(torch.float8_e4m3fn).float()
    assert not torch.isnan(sf_fp32).any(), "SF contains NaN after e4m3 decode"
    assert not torch.isinf(sf_fp32).any(), "SF contains Inf after e4m3 decode"

    # Dequantize FP4 and verify finite output
    deq = _dequant_nvfp4(
        fp4.view(torch.uint8).reshape(m, n // 2),
        sf_out.view(torch.uint8).reshape(m, n // 16),
        sf_scale,
        m, n,
    )
    assert not torch.isnan(deq).any(), "Dequantized FP4 contains NaN"
    assert not torch.isinf(deq).any(), "Dequantized FP4 contains Inf"
