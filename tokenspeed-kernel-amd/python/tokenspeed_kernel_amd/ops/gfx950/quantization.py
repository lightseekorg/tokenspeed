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

"""GFX950 group-32 E4M3 activation quantization with UE8M0 scales."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

cdna4 = gl.amd.cdna4

# Groups per CTA tile. Each lane owns eight contiguous values (one 16-byte
# load) and four lanes share a group, so a four-wave CTA covers 64 groups per
# pass and four passes per tile.
_GROUPS_PER_CTA = 256
_NUM_WARPS = 4

__all__ = ["launch_gluon_quantize_fp8_group32_ue8m0_gfx950"]


def _quantize_metadata(grid, kernel, args):
    groups = args["num_groups"]
    return {
        "name": kernel.name,
        # BF16 in, E4M3 out, one UE8M0 byte per group.
        "bytes": groups * (32 * 2 + 32 + 1),
    }


@gluon.jit(launch_metadata=_quantize_metadata, do_not_specialize=("num_groups",))
def gluon_quantize_fp8_group32_ue8m0_gfx950(
    x,
    q,
    scales,
    num_groups,
    K: gl.constexpr,
    X_STRIDE: gl.constexpr,
    GROUPS_PER_CTA: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [NUM_WARPS, 1], [1, 0])
    group = gl.program_id(0) * GROUPS_PER_CTA + gl.arange(
        0, GROUPS_PER_CTA, layout=gl.SliceLayout(1, layout)
    )
    column = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    live = group < num_groups
    row = group // (K // 32)
    k = (group % (K // 32)) * 32
    values = cdna4.buffer_load(
        x,
        row[:, None] * X_STRIDE + k[:, None] + column[None, :],
        mask=live[:, None],
        other=0.0,
    ).to(gl.float32)
    amax = gl.maximum(gl.max(gl.abs(values), 1), 1.0e-4)
    # Round max(amax, 1e-4) / 448 up to a power of two by its FP32 bits; the
    # exponent byte is the UE8M0 scale. See ``v41_quantize_fp8``.
    bits = (amax * (1.0 / 448.0)).to(gl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(gl.int32)
    scale = (exponent << 23).to(gl.float32, bitcast=True)
    # A power-of-two scale has an exact reciprocal for every finite input, so
    # one multiply per value rounds exactly like the reference division.
    # Infinite inputs keep the reference's x / inf semantics through 1 / inf.
    inverse = 1.0 / scale
    codes = (values * inverse[:, None]).to(gl.float8e4nv)
    cdna4.buffer_store(
        codes,
        q,
        group[:, None] * 32 + column[None, :],
        mask=live[:, None],
    )
    cdna4.buffer_store(exponent.to(gl.uint8), scales, group, mask=live)


def launch_gluon_quantize_fp8_group32_ue8m0_gfx950(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M, K]`` BF16/FP16 rows to E4M3 with UE8M0 group-32 scales.

    Args:
        x: ``[M, K]`` input whose last dimension is contiguous and divisible
            by 32.

    Returns:
        Contiguous E4M3 values ``[M, K]`` and uint8 UE8M0 scales
        ``[M, K // 32]``.
    """
    if x.ndim != 2 or x.shape[1] % 32 or x.stride(1) != 1:
        raise ValueError(
            "group-32 quantization requires [M, K] with contiguous K % 32 == 0"
        )
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("group-32 quantization requires BF16 or FP16 input")
    rows, k = x.shape
    if max(rows * x.stride(0), rows * k) >= 2**31:
        raise ValueError("group-32 quantization exceeds 32-bit buffer offsets")
    q = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((rows, k // 32), dtype=torch.uint8, device=x.device)
    num_groups = rows * (k // 32)
    if num_groups:
        gluon_quantize_fp8_group32_ue8m0_gfx950[
            (triton.cdiv(num_groups, _GROUPS_PER_CTA),)
        ](
            x,
            q,
            scales,
            num_groups,
            K=k,
            X_STRIDE=x.stride(0),
            GROUPS_PER_CTA=_GROUPS_PER_CTA,
            NUM_WARPS=_NUM_WARPS,
            num_warps=_NUM_WARPS,
        )
    return q, scales
