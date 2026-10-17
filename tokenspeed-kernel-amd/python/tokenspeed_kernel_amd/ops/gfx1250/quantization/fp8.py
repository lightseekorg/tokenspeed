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

"""GFX1250 group-32 FP8 activation quantization with UE8M0 scales."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

__all__ = [
    "gluon_quantize_fp8_group32_ue8m0_gfx1250",
    "launch_gluon_quantize_fp8_group32_ue8m0_gfx1250",
]


def _quantize_launch_metadata(grid, kernel, args):
    """Report the activation read and the E4M3 code and UE8M0 scale writes."""
    groups = args["total_groups"]
    return {
        "name": kernel.name,
        "bytes": groups * 32 * (args["x_ptr"].element_size() + 1) + groups,
    }


@gluon.jit(
    launch_metadata=_quantize_launch_metadata,
    do_not_specialize=["total_groups"],
)
def gluon_quantize_fp8_group32_ue8m0_gfx1250(
    x_ptr,
    q_ptr,
    s_ptr,
    total_groups,
    stride_xm,
    K: gl.constexpr,
    BLOCK_G: gl.constexpr,
):
    """Quantize BLOCK_G consecutive 32-element groups of a row-major [M, K].

    A group's scale is ``2 ** ceil(log2(max(amax, 1e-4) / 448))``, found from
    the FP32 bits of the quotient, and its codes are the correctly rounded
    ``x / scale`` cast to E4M3. Each thread loads eight elements, so four
    threads hold one group.
    """
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [gl.num_warps(), 1], [1, 0])
    groups_per_row: gl.constexpr = K // 32
    first = gl.program_id(0) * BLOCK_G
    first_row = first // groups_per_row
    groups = first + gl.arange(0, BLOCK_G, layout=gl.SliceLayout(1, layout))
    lanes = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    rows = groups // groups_per_row - first_row
    columns = (groups % groups_per_row) * 32
    live = groups < total_groups
    mask = live[:, None] & (lanes[None, :] < 32)

    values = gl.amd.cdna5.buffer_load(
        x_ptr + first_row.to(gl.int64) * stride_xm,
        (rows * stride_xm + columns)[:, None] + lanes[None, :],
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    amax = gl.maximum(gl.max(gl.abs(values), axis=1), 1.0e-4)
    raw = amax * (1.0 / 448.0)
    bits = raw.to(gl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(gl.int32)
    scale = (exponent << 23).to(gl.float32, bitcast=True)
    quantized = gl.div_rn(values, scale[:, None]).to(gl.float8e4nv)
    gl.amd.cdna5.buffer_store(
        quantized,
        q_ptr + first_row.to(gl.int64) * K,
        (rows * K + columns)[:, None] + lanes[None, :],
        mask=mask,
    )
    gl.amd.cdna5.buffer_store(exponent.to(gl.uint8), s_ptr, groups, mask=live)


def launch_gluon_quantize_fp8_group32_ue8m0_gfx1250(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` [M, K] to E4M3 codes [M, K] and UE8M0 scales [M, K / 32].

    ``x`` is BF16 or FP16 with a contiguous last axis and K divisible by 32.
    The scales are row-major uint8 biased exponents.
    """
    if (
        x.ndim != 2
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.shape[1] % 32
        or x.stride(1) != 1
    ):
        raise ValueError(
            "group-32 FP8 quantization requires BF16/FP16 [M, K] with contiguous "
            "K divisible by 32"
        )
    m, k = x.shape
    quantized = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((m, k // 32), dtype=torch.uint8, device=x.device)
    total_groups = m * (k // 32)
    if total_groups:
        block_g, num_warps = (256, 8) if total_groups >= 49152 else (32, 4)
        gluon_quantize_fp8_group32_ue8m0_gfx1250[(triton.cdiv(total_groups, block_g),)](
            x,
            quantized,
            scales,
            total_groups,
            x.stride(0),
            K=k,
            BLOCK_G=block_g,
            num_warps=num_warps,
        )
    return quantized, scales
