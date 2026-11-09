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

"""Output gate/RMSNorm and DeepGEMM quant-pack in one launch."""

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit(do_not_specialize=["rows", "padded_rows", "stride_x", "stride_z"])
def _output_fp8_quant_pack_kernel(
    X,
    Z,
    W,
    Q,
    S,
    rows,
    padded_rows,
    stride_x,
    stride_z,
    WIDTH: tl.constexpr,
    HEAD: tl.constexpr,
    Z_HEAD_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    NORMALIZE: tl.constexpr,
    SIGMOID: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    head = tl.program_id(1)
    col = tl.arange(0, HEAD)
    mask = (row < rows)[:, None]
    if NORMALIZE:
        weight = tl.load(W + col).to(tl.float32)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x = tl.load(X + row[:, None] * stride_x + head * HEAD + col[None, :], mask, 0).to(
        tl.float32
    )
    z = tl.load(
        Z + row[:, None] * stride_z + head * Z_HEAD_STRIDE + col[None, :], mask, 0
    ).to(tl.float32)
    if NORMALIZE:
        # Same reduction, reciprocal sqrt and multiply order as gated_rmsnorm.
        variance = tl.sum(x * x, axis=1) / HEAD
        rstd = 1 / tl.sqrt(variance + EPS)
        value = x * rstd[:, None] * weight[None, :]
        value *= tl.sigmoid(z) if SIGMOID else z * tl.sigmoid(z)
    else:
        value = x * tl.sigmoid(z)
    # Preserve the materialized BF16 boundary before quantization.
    value = value.to(tl.bfloat16).to(tl.float32)
    blocks = tl.reshape(value, (BLOCK_M, HEAD // GROUP, GROUP))
    amax = tl.maximum(tl.max(tl.abs(blocks), axis=2), 1.0e-10)
    raw_scale = amax * (1.0 / 448.0)
    exponent = (raw_scale.to(tl.int32, bitcast=True) + 0x7FFFFF) >> 23
    inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    # CuTe's standalone quantizer multiplies BF16 by the power-of-two inverse.
    scaled = (blocks * inverse[:, :, None]).to(tl.bfloat16).to(tl.float32)
    codes = tl.reshape(tl.clamp(scaled, -448.0, 448.0), (BLOCK_M, HEAD))
    tl.store(Q + row[:, None] * WIDTH + head * HEAD + col[None, :], codes, mask)
    groups = head * (HEAD // GROUP) + tl.arange(0, HEAD // GROUP)
    offsets = (
        groups[None, :].to(tl.int64) // 4 * padded_rows + row[:, None]
    ) * 4 + groups[None, :] % 4
    # Disjoint byte stores: no read/modify/write between heads sharing a word.
    tl.store(
        S + offsets,
        tl.where(row[:, None] < rows, exponent, 0).to(tl.uint8),
        row[:, None] < padded_rows,
    )
    if (WIDTH // GROUP) % 4 != 0:
        if head == WIDTH // HEAD - 1:
            for tail in tl.static_range((WIDTH // GROUP) % 4, 4):
                offset = (
                    (WIDTH // GROUP) // 4 * padded_rows.to(tl.int64) + row
                ) * 4 + tail
                tl.store(S + offset, 0, row < padded_rows)


def launch(x, z, weight, *, head_dim, eps, sigmoid_gate, enable_pdl):
    rows, width = x.shape
    padded = (rows + 3) // 4 * 4
    output = torch.empty_like(
        x, dtype=torch.float8_e4m3fn, memory_format=torch.contiguous_format
    )
    storage = torch.empty(
        (triton.cdiv(width, 512), padded), device=x.device, dtype=torch.int32
    )
    scales = storage.t()[:rows]
    if rows:
        # Four rows and one warp per head-row, as in the unfused GDN norm.
        _output_fp8_quant_pack_kernel[(triton.cdiv(rows, 4), width // head_dim)](
            x,
            z,
            weight,
            output,
            storage.view(torch.uint8),
            rows,
            padded,
            x.stride(0),
            z.stride(0),
            WIDTH=width,
            HEAD=head_dim,
            Z_HEAD_STRIDE=z.stride(1) if z.ndim == 3 else head_dim,
            EPS=eps,
            NORMALIZE=weight is not None,
            SIGMOID=sigmoid_gate,
            GROUP=128,
            BLOCK_M=4,
            ENABLE_PDL=enable_pdl,
            num_warps=4,
            **({"launch_pdl": True} if enable_pdl else {}),
        )
    return output, scales
