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


def block_fp8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Adapt canonical FP32 scales without rounding them to UE8M0."""
    import deep_gemm

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
