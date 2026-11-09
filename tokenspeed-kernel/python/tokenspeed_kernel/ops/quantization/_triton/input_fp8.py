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

"""Residual add, GemmaRMSNorm and block FP8 with a BF16 side output."""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.layernorm.triton import _row_block


@triton.jit(do_not_specialize=["rows", "padded_rows", "stride_x", "stride_r"])
def _add_rmsnorm_fp8_block_kernel(
    X,
    R,
    W,
    Q,
    S,
    rows,
    padded_rows,
    stride_x,
    stride_r,
    WIDTH: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
    CHUNKS: tl.constexpr,
    GROUP: tl.constexpr,
    PACKED: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    if row >= rows:
        if ENABLE_PDL:
            tl.extra.cuda.gdc_wait()
        groups = tl.arange(
            0, triton.next_power_of_2(triton.cdiv(WIDTH // GROUP, 4) * 4)
        )
        offsets = (groups // 4).to(tl.int64) * padded_rows * 4 + row * 4 + groups % 4
        tl.store(S + offsets, 0, groups < triton.cdiv(WIDTH // GROUP, 4) * 4)
        return
    weights = ()
    for c in tl.static_range(CHUNKS):
        offsets = c * BLOCK + cols
        weight = tl.load(W + offsets, offsets < WIDTH, 0).to(tl.float32)
        weights = weights + (1.0 + weight,)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    totals = ()
    for c in tl.static_range(CHUNKS):
        offsets = c * BLOCK + cols
        x = tl.load(X + row * stride_x + offsets, offsets < WIDTH, 0).to(tl.float32)
        residual = tl.load(R + row * stride_r + offsets, offsets < WIDTH, 0).to(
            tl.float32
        )
        totals = totals + (x + residual,)
    squares = tl.zeros([BLOCK], tl.float32)
    for c in tl.static_range(CHUNKS):
        offsets = c * BLOCK + cols
        tl.store(R + row * stride_r + offsets, totals[c], offsets < WIDTH)
        squares += totals[c] * totals[c]
    variance = tl.sum(squares, 0) / WIDTH
    rstd = tl.rsqrt(variance + eps)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    for c in tl.static_range(CHUNKS):
        offsets = c * BLOCK + cols
        normed = (totals[c] * rstd * weights[c]).to(tl.bfloat16)
        tl.store(X + row * stride_x + offsets, normed, offsets < WIDTH)
        values = tl.reshape(normed.to(tl.float32), (BLOCK // GROUP, GROUP))
        valid = c * BLOCK + tl.arange(0, BLOCK // GROUP) * GROUP < WIDTH
        amax = tl.max(tl.abs(values), 1)
        if PACKED:
            raw = tl.maximum(amax, 1.0e-10) * (1.0 / 448.0)
            exponent = (raw.to(tl.int32, bitcast=True) + 0x7FFFFF) >> 23
            inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
            quant = (values * inverse[:, None]).to(tl.bfloat16).to(tl.float32)
            groups = c * (BLOCK // GROUP) + tl.arange(0, BLOCK // GROUP)
            scale_offsets = (
                (groups // 4).to(tl.int64) * padded_rows * 4 + row * 4 + groups % 4
            )
            tl.store(S + scale_offsets, exponent.to(tl.uint8), valid)
        else:
            floor = tl.full((), 1.0e-10, tl.float32).to(tl.bfloat16).to(tl.float32)
            inverse = tl.div_rn(448.0, tl.maximum(amax, floor))
            quant = values * inverse[:, None]
            groups = c * (BLOCK // GROUP) + tl.arange(0, BLOCK // GROUP)
            tl.store(
                S + row * (WIDTH // GROUP) + groups, tl.div_rn(1.0, inverse), valid
            )
        quant = tl.reshape(tl.clamp(quant, -448.0, 448.0), (BLOCK,))
        # Match the BF16 producer's vector width to preserve reduction order.
        code_cols = tl.max_contiguous(tl.arange(0, BLOCK), 8)
        code_offsets = c * BLOCK + code_cols
        tl.store(Q + row * WIDTH + code_offsets, quant, code_offsets < WIDTH)
    if PACKED and (WIDTH // GROUP) % 4 != 0:
        for tail in tl.static_range((WIDTH // GROUP) % 4, 4):
            offset = (
                ((WIDTH // GROUP) // 4) * padded_rows.to(tl.int64) * 4 + row * 4 + tail
            )
            tl.store(S + offset, 0)


def launch(x, residual, weight, *, eps, scale_encoding, enable_pdl):
    rows, width = x.shape
    packed = scale_encoding == "packed_ue8m0"
    padded = (rows + 3) // 4 * 4 if packed else rows
    codes = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=x.device)
    if packed:
        storage = torch.empty(
            (triton.cdiv(width, 512), padded), dtype=torch.int32, device=x.device
        )
        scales = storage.t()[:rows]
        scale_pointer = storage.view(torch.uint8)
    else:
        scales = torch.empty((rows, width // 128), dtype=torch.float32, device=x.device)
        scale_pointer = scales
    if rows:
        block = _row_block(width)
        _add_rmsnorm_fp8_block_kernel[(padded,)](
            x,
            residual,
            weight,
            codes,
            scale_pointer,
            rows,
            padded,
            x.stride(0),
            residual.stride(0),
            WIDTH=width,
            eps=eps,
            BLOCK=block,
            CHUNKS=triton.cdiv(width, block),
            GROUP=128,
            PACKED=packed,
            ENABLE_PDL=enable_pdl,
            num_warps=min(max(triton.next_power_of_2(width) // 256, 1), 8),
            **({"launch_pdl": True} if enable_pdl else {}),
        )
    return x, codes, scales, residual
