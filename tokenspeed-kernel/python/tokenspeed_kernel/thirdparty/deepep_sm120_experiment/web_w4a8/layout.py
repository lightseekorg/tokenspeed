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

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _unpack_kernel(
    packed_ptr,
    counts_ptr,
    offsets_ptr,
    out_ptr,
    out_stride_e,
    out_stride_m,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    expert = tl.program_id(0)
    column = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    count = tl.load(counts_ptr + expert)
    offset = tl.load(offsets_ptr + expert)
    for block in range(tl.cdiv(count, BLOCK_M)):
        row = block * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = (row[:, None] < count) & (column[None, :] < HIDDEN)
        values = tl.load(
            packed_ptr + (offset + row[:, None]) * HIDDEN + column[None, :],
            valid,
            other=0,
        )
        tl.store(
            out_ptr
            + expert * out_stride_e
            + row[:, None] * out_stride_m
            + column[None, :],
            values,
            valid,
        )


def unpack_recv_rows(
    packed: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    out: torch.Tensor,
    block_m: int,
) -> torch.Tensor:
    """Write valid compact outputs into a DeepEP combine-layout buffer.

    ``out`` is BF16 [local_experts, receive_capacity, hidden] with unit column
    stride and any expert/row strides, such as DeepEP's zero-copy combine send
    buffer. Returns ``out``; DeepEP combines only valid rows, so unused
    capacity is untouched.
    """
    experts, recv_m, hidden = out.shape
    if out.dtype != torch.bfloat16 or out.stride(2) != 1:
        raise ValueError("combine buffer must be BF16 with contiguous rows")
    _unpack_kernel[(experts, triton.cdiv(hidden, 256))](
        packed,
        counts,
        offsets,
        out,
        out.stride(0),
        out.stride(1),
        HIDDEN=hidden,
        BLOCK_M=block_m,
        BLOCK_H=256,
    )
    return out
