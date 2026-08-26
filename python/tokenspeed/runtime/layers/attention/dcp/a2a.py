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


@triton.jit
def _unpack_and_merge_dcp_partials_kernel(
    recv_ptr,
    output_ptr,
    recv_stride_rank,
    recv_stride_b,
    recv_stride_h,
    recv_stride_d,
    output_stride_b,
    output_stride_h,
    output_stride_d,
    world_size: tl.constexpr,
    head_dim: tl.constexpr,
    lse_base_e: tl.constexpr,
    value_block: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    value_offsets = tl.arange(0, value_block)
    value_mask = value_offsets < head_dim

    maximum = -float("inf")
    for source in tl.static_range(world_size):
        recv_base = (
            source * recv_stride_rank
            + batch_idx * recv_stride_b
            + head_idx * recv_stride_h
        )
        low_raw = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d)
        high_raw = tl.load(recv_ptr + recv_base + (head_dim + 1) * recv_stride_d)
        low = low_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        high = high_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        lse = (low | (high << 16)).to(tl.float32, bitcast=True)
        lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
        maximum = tl.maximum(maximum, lse)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)

    denominator = 0.0
    for source in tl.static_range(world_size):
        recv_base = (
            source * recv_stride_rank
            + batch_idx * recv_stride_b
            + head_idx * recv_stride_h
        )
        low_raw = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d)
        high_raw = tl.load(recv_ptr + recv_base + (head_dim + 1) * recv_stride_d)
        low = low_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        high = high_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        lse = (low | (high << 16)).to(tl.float32, bitcast=True)
        lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
        if lse_base_e:
            denominator += tl.exp(lse - maximum)
        else:
            denominator += tl.exp2(lse - maximum)

    accumulator = tl.zeros([value_block], dtype=tl.float32)
    for source in tl.static_range(world_size):
        recv_base = (
            source * recv_stride_rank
            + batch_idx * recv_stride_b
            + head_idx * recv_stride_h
        )
        low_raw = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d)
        high_raw = tl.load(recv_ptr + recv_base + (head_dim + 1) * recv_stride_d)
        low = low_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        high = high_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
        lse = (low | (high << 16)).to(tl.float32, bitcast=True)
        lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
        if lse_base_e:
            mass = tl.exp(lse - maximum)
        else:
            mass = tl.exp2(lse - maximum)
        weight = tl.where(denominator > 0.0, mass / denominator, 0.0)
        partial = tl.load(
            recv_ptr + recv_base + value_offsets * recv_stride_d,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        partial = tl.where(weight == 0.0, 0.0, partial)
        accumulator += partial * weight

    tl.store(
        output_ptr
        + batch_idx * output_stride_b
        + head_idx * output_stride_h
        + value_offsets * output_stride_d,
        accumulator,
        mask=value_mask,
    )


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
    output = torch.empty(
        (batch, heads_per_rank, head_dim),
        dtype=local_output.dtype,
        device=local_output.device,
    )
    with nvtx_range("dcp_output_lse_unpack_merge", category="dcp"):
        _unpack_and_merge_dcp_partials_kernel[(batch, heads_per_rank)](
            recv,
            output,
            recv.stride(0),
            recv.stride(1),
            recv.stride(2),
            recv.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            world_size=world_size,
            head_dim=head_dim,
            lse_base_e=lse_base == math.e,
            value_block=value_block,
        )
    return output.reshape(*leading_shape, heads_per_rank, head_dim)


__all__ = ["reconstruct_with_all_to_all"]
