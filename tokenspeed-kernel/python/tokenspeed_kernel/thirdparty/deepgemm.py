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

from __future__ import annotations

import torch


def prepare_block_fp8_weight(
    weight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize E4M3/FP32 block weights and pack UE8M0 scales once.

    Args:
        weight: E4M3 [N,K] weights, with K divisible by 128.
        scales: Canonical FP32 [ceil(N/128),K/128] dequantization scales.

    Returns:
        Requantized E4M3 weights and DeepGEMM int32-packed per-row scales.
        Superseded weight/scale allocations are not retained.
    """
    import deep_gemm

    n, k = weight.shape
    padded_n = (n + 127) // 128 * 128
    dequantized = torch.zeros((padded_n, k), device=weight.device, dtype=torch.float32)
    dequantized[:n] = weight.float() * scales.repeat_interleave(128, 0)[
        :n
    ].repeat_interleave(128, 1)
    blocks = dequantized.view(padded_n // 128, 128, k // 128, 128)
    raw_scales = blocks.abs().amax((1, 3)).clamp_min(1e-10) / 448.0
    bits = raw_scales.view(torch.int32)
    exponent = (((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0)).clamp(1, 254)
    ue8m0_scales = (exponent << 23).view(torch.float32)
    quantized = (blocks / ue8m0_scales[:, None, :, None]).clamp(-448, 448)
    quantized = quantized.to(torch.float8_e4m3fn).view(padded_n, k)[:n].contiguous()
    packed = deep_gemm.transform_sf_into_required_layout(ue8m0_scales, n, k, (128, 128))
    return quantized, packed


def block_fp8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    enable_pdl: bool,
) -> torch.Tensor:
    """Run Hopper FP32 scales or already-packed Blackwell UE8M0 scales."""
    import deep_gemm

    if a_scale.dtype == torch.int32:
        deep_gemm.set_pdl(enable_pdl)
        deep_gemm.fp8_gemm_nt(
            (a, a_scale),
            (b, b_scale),
            out,
            recipe_a=(1, 128),
            recipe_b=(1, 128),
        )
        return out

    m, k = a.shape
    n = b.shape[0]
    sa = deep_gemm.transform_sf_into_required_layout(
        a_scale, m, k, (1, 128), disable_ue8m0_cast=True
    )
    sb = deep_gemm.transform_sf_into_required_layout(
        b_scale, n, k, (128, 128), disable_ue8m0_cast=True
    )
    deep_gemm.fp8_gemm_nt(
        (a, sa),
        (b, sb),
        out,
        recipe=(1, 128, 128),
        disable_ue8m0_cast=True,
    )
    return out


def grouped_fp8_weight_scales(scales: torch.Tensor, groups: int) -> torch.Tensor:
    """Pack 1x32 UE8M0 weight scales ``[groups * n, k / 32]`` for ``grouped_fp8_mm``."""
    import deep_gemm

    rows, blocks = scales.shape
    exponents = scales.to(torch.int32) << 23
    return deep_gemm.transform_sf_into_required_layout(
        exponents.view(torch.float32).view(groups, rows // groups, blocks),
        rows // groups,
        blocks * 32,
        (1, 1, 32),
        num_groups=groups,
        is_sfa=False,
    )


def grouped_fp8_mm(
    x: torch.Tensor,
    x_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    out: torch.Tensor,
    out_scales: torch.Tensor | None,
) -> torch.Tensor:
    """``einsum('tgk,gnk->tgn')`` of E4M3 operands with 1x32 UE8M0 scales.

    With ``out_scales``, ``out`` receives E4M3 codes of the BF16-rounded result
    and ``out_scales`` their packed 1x32 UE8M0 scales over ``out.flatten(1)``.
    """
    import deep_gemm

    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd",
        (x, x_scales),
        (weight, weight_scales),
        out if out_scales is None else (out, out_scales),
        recipe=(1, 1, 32),
    )
    return out


def fp8_weight_scales(scales: torch.Tensor) -> torch.Tensor:
    """Pack 1x32 UE8M0 weight scales ``[n, k / 32]`` for ``fp8_mm``."""
    import deep_gemm

    rows, blocks = scales.shape
    exponents = scales.to(torch.int32) << 23
    return deep_gemm.transform_sf_into_required_layout(
        exponents.view(torch.float32), rows, blocks * 32, (1, 1, 32), is_sfa=False
    )


def fp8_mm(
    x: torch.Tensor,
    x_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """``x @ weight.T`` of E4M3 operands with 1x32 UE8M0 scales into BF16 ``out``."""
    import deep_gemm

    deep_gemm.fp8_gemm_nt(
        (x, x_scales), (weight, weight_scales), out, recipe=(1, 1, 32)
    )
    return out
