# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Packed output/LSE all-to-all combine for decode context parallelism."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel._triton import tl, triton

from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.distributed.mapping import Group
else:
    Group = tuple[int, ...]


@triton.jit
def _pack_dcp_partials_kernel(
    output_ptr,
    lse_ptr,
    send_ptr,
    output_stride_b,
    output_stride_h,
    output_stride_d,
    lse_stride_b,
    lse_stride_h,
    send_stride_rank,
    send_stride_b,
    send_stride_h,
    send_stride_d,
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
        send_base = (
            destination * send_stride_rank
            + batch_idx * send_stride_b
            + local_head * send_stride_h
        )
        values = tl.load(
            output_ptr
            + batch_idx * output_stride_b
            + source_head * output_stride_h
            + value_offsets * output_stride_d,
            mask=value_mask,
        )
        tl.store(
            send_ptr + send_base + value_offsets * send_stride_d,
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
            send_ptr + send_base + head_dim * send_stride_d,
            low.to(send_ptr.dtype.element_ty, bitcast=True),
        )
        tl.store(
            send_ptr + send_base + (head_dim + 1) * send_stride_d,
            high.to(send_ptr.dtype.element_ty, bitcast=True),
        )


def _merge_received_partials(
    recv: torch.Tensor,
    *,
    head_dim: int,
    lse_base: float,
) -> torch.Tensor:
    """Merge packed A2A partials with the AG-RS FP32 arithmetic contract."""
    words = recv.view(torch.uint16)[..., head_dim:]
    low = words[..., 0].to(torch.int32)
    high = words[..., 1].to(torch.int32)
    lse = (low | (high << 16)).contiguous().view(torch.float32)

    valid = torch.isfinite(lse)
    sanitized = torch.where(valid, lse, torch.full_like(lse, -torch.inf))
    maximum = sanitized.amax(dim=0)
    has_value = torch.isfinite(maximum)
    shifted = torch.where(
        valid & has_value.unsqueeze(0),
        sanitized - maximum.unsqueeze(0),
        torch.full_like(sanitized, -torch.inf),
    )
    if lse_base == 2.0:
        masses = torch.exp2(shifted)
    elif lse_base == math.e:
        masses = torch.exp(shifted)
    else:
        masses = torch.pow(
            torch.as_tensor(lse_base, device=shifted.device),
            shifted,
        )
    denominator = masses.sum(dim=0)
    weights = torch.where(
        denominator.unsqueeze(0) > 0,
        masses / denominator.clamp_min(torch.finfo(masses.dtype).tiny).unsqueeze(0),
        torch.zeros_like(masses),
    )
    partials = recv[..., :head_dim].float()
    safe_partials = torch.where(
        torch.isfinite(partials),
        partials,
        torch.zeros_like(partials),
    )
    return (safe_partials * weights.unsqueeze(-1)).sum(dim=0).to(recv.dtype)


def reconstruct_with_all_to_all(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    *,
    group: Group,
    lse_base: float = 2.0,
) -> torch.Tensor:
    """Exchange head slices once, then exactly merge every context partial."""
    world_size = len(group)
    if world_size == 1:
        return local_output
    if local_output.ndim < 3 or local_lse.shape != local_output.shape[:-1]:
        raise ValueError("DCP A2A output/LSE shapes disagree")
    if local_output.element_size() != 2:
        raise ValueError(
            "DCP A2A packs FP32 LSE into two 16-bit output elements; got "
            f"{local_output.dtype}"
        )
    leading_shape = local_output.shape[:-2]
    heads, head_dim = local_output.shape[-2:]
    flat_output = local_output.reshape(-1, heads, head_dim)
    flat_lse = local_lse.reshape(-1, heads)
    batch = flat_output.shape[0]
    if heads % world_size:
        raise ValueError(
            f"DCP A2A head count {heads} is not divisible by group size {world_size}"
        )
    heads_per_rank = heads // world_size
    packed_shape = (world_size, batch, heads_per_rank, head_dim + 2)
    send = torch.empty(
        packed_shape, dtype=local_output.dtype, device=local_output.device
    )
    recv = torch.empty_like(send)
    value_block = triton.next_power_of_2(head_dim)
    with nvtx_range("dcp_output_lse_pack", category="dcp"):
        _pack_dcp_partials_kernel[(batch, heads_per_rank)](
            flat_output,
            flat_lse,
            send,
            flat_output.stride(0),
            flat_output.stride(1),
            flat_output.stride(2),
            flat_lse.stride(0),
            flat_lse.stride(1),
            send.stride(0),
            send.stride(1),
            send.stride(2),
            send.stride(3),
            world_size=world_size,
            heads_per_rank=heads_per_rank,
            head_dim=head_dim,
            value_block=value_block,
        )

    from tokenspeed.runtime.distributed.comm_ops import all_to_all_single

    with nvtx_range("dcp_output_lse_all_to_all", category="dcp"):
        all_to_all_single(recv.view(-1), send.view(-1), group)
    with nvtx_range("dcp_output_lse_unpack_merge", category="dcp"):
        output = _merge_received_partials(
            recv,
            head_dim=head_dim,
            lse_base=lse_base,
        )
    return output.reshape(*leading_shape, heads_per_rank, head_dim)


__all__ = ["reconstruct_with_all_to_all"]
