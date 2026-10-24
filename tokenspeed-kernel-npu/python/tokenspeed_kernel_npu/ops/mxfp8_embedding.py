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


"""Device Engram gather and original group32 dequantization in one kernel."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops.dsv41_pointwise import _decode_e4m3_bits


@triton.jit
def _mxfp8_gather(
    W,
    S,
    I,
    O,
    COUNT,
    W0,
    S0,
    START,
    END,
    D: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    rr = tl.arange(0, R)
    d = tl.arange(0, B)
    for first in range(tl.program_id(0) * R, COUNT, tl.num_programs(0) * R):
        row = first + rr
        ix = tl.load(I + row, row < COUNT, 0).to(tl.int64)
        local = (row < COUNT) & (ix >= START) & (ix < END)
        ix = tl.where(local, ix - START, 0)
        code = tl.load(
            W + ix[:, None] * W0 + d[None, :], local[:, None] & (d[None, :] < D), 0
        ).to(tl.int32)
        scale = tl.load(
            S + ix[:, None] * S0 + (d[None, :] // 32),
            local[:, None] & (d[None, :] < D),
            127,
        ).to(tl.int32)
        bits = tl.where(scale == 0, 0x00400000, scale << 23)
        bits = tl.where(scale == 255, 0x7FC00000, bits)
        value = _decode_e4m3_bits(code) * bits.to(tl.float32, bitcast=True)
        tl.store(
            O + row[:, None].to(tl.int64) * D + d[None, :],
            tl.where(local[:, None], value, 0.0),
            (row[:, None] < COUNT) & (d[None, :] < D),
        )


def mxfp8_embedding(weight, scales, indices, row_start, row_end):
    """Decode owned E4M3/E8M0 rows directly to BF16, zeroing nonlocal IDs."""
    dim = weight.shape[1]
    out = torch.empty(
        (*indices.shape, dim), device=indices.device, dtype=torch.bfloat16
    )
    if not indices.numel():
        return out
    if not weight.numel() or row_start == row_end:
        return out.zero_()
    rows = 8 if dim <= 256 else 2
    _mxfp8_gather[(min(48, triton.cdiv(indices.numel(), rows)),)](
        weight.view(torch.uint8),
        scales,
        indices,
        out,
        indices.numel(),
        weight.stride(0),
        scales.stride(0),
        row_start,
        row_end,
        dim,
        rows,
        triton.next_power_of_2(dim),
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )
    return out
