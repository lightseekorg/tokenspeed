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
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _pack_dcp_output_lse_kernel(
    output_ptr,
    lse_ptr,
    packed_ptr,
    output_stride_b,
    output_stride_h,
    output_stride_d,
    lse_stride_b,
    lse_stride_h,
    packed_stride_rank,
    packed_stride_b,
    packed_stride_h,
    packed_stride_d,
    world_size: tl.constexpr,
    heads_per_rank: tl.constexpr,
    head_dim: tl.constexpr,
    value_block: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    local_head = tl.program_id(1).to(tl.int64)
    value_offsets = tl.arange(0, value_block)
    value_mask = value_offsets < head_dim
    for destination in tl.static_range(world_size):
        source_head = destination * heads_per_rank + local_head
        packed_base = (
            destination * packed_stride_rank
            + batch_idx * packed_stride_b
            + local_head * packed_stride_h
        )
        values = tl.load(
            output_ptr
            + batch_idx * output_stride_b
            + source_head * output_stride_h
            + value_offsets * output_stride_d,
            mask=value_mask,
        )
        tl.store(
            packed_ptr + packed_base + value_offsets * packed_stride_d,
            values,
            mask=value_mask,
        )
        lse = tl.load(
            lse_ptr + batch_idx * lse_stride_b + source_head * lse_stride_h
        ).to(tl.float32)
        bits = lse.to(tl.uint32, bitcast=True)
        low = (bits & 0xFFFF).to(tl.uint16)
        high = ((bits >> 16) & 0xFFFF).to(tl.uint16)
        tl.store(
            packed_ptr + packed_base + head_dim * packed_stride_d,
            low.to(packed_ptr.dtype.element_ty, bitcast=True),
        )
        tl.store(
            packed_ptr + packed_base + (head_dim + 1) * packed_stride_d,
            high.to(packed_ptr.dtype.element_ty, bitcast=True),
        )


def _pack_dcp_output_lse(
    output: torch.Tensor,
    lse: torch.Tensor,
    packed: torch.Tensor,
    *,
    world_size: int,
    heads_per_rank: int,
    head_dim: int,
) -> None:
    batch = output.shape[0]
    _pack_dcp_output_lse_kernel[(batch, heads_per_rank)](
        output,
        lse,
        packed,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        packed.stride(3),
        world_size=world_size,
        heads_per_rank=heads_per_rank,
        head_dim=head_dim,
        value_block=triton.next_power_of_2(head_dim),
    )
