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

import logging

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel_nvidia.registration import Priority, register_kernel

logger = logging.getLogger(__name__)

try:
    from tokenspeed_kernel_nvidia.thirdparty.deep_ep import Buffer
except ImportError:
    Buffer = None  # type: ignore[assignment]


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


@triton.jit
def _silu_and_mul_post_quant_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    size_n,
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d
    output_scale_offs = (
        output_scale_ptr
        + expert_id * stride_output_scale_0
        + hidden_dim_block_index * stride_output_scale_2
    )

    for token_index in tl.range(
        token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE
    ):
        gate = tl.load(
            input_ptr_offs + token_index * stride_input_1,
            mask=offs_in_d < size_n,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr_offs + token_index * stride_input_1 + size_n,
            mask=offs_in_d < size_n,
            other=0.0,
        )
        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = up * gate
        _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        output_s = _absmax / fp8_max
        output_q = tl.clamp(gate_up / output_s, fp8_min, fp8_max).to(
            output_ptr.dtype.element_ty
        )
        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            output_q,
            mask=offs_in_d < size_n,
        )
        tl.store(
            output_scale_offs + token_index * stride_output_scale_1,
            output_s,
        )


def silu_and_mul_masked_post_quant_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
):
    """
    input shape [num_groups, token_num_padded, inter_size_x2]
    output shape [num_groups, token_num_padded, inter_size_x2 // 2], dtype fp8
    output_scale [num_groups token_num_padded, inter_size_x2 // 2 // 128] dtype float32
    quant_group_size  int,
    masked_m shape [num_groups],
    """

    assert input.is_contiguous()
    assert output.dtype == torch.float8_e4m3fn
    assert output.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0

    size_n = input.shape[-1] // 2
    assert size_n % quant_group_size == 0

    num_groups = len(masked_m)

    BLOCK_NUM_PER_EXPERT = 64 if num_groups < 4 else 32

    BLOCK_N = quant_group_size
    num_warps = 1
    NUM_STAGES = 6
    hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)

    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        num_groups,
    )

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    _silu_and_mul_post_quant_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        masked_m,
        size_n,
        fp8_max,
        fp8_min,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
    )
    return


@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)

    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)

    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)

    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)

    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        tl.store(
            m_indices_start_ptr + start_m + off_expert,
            cur_expert,
        )


@triton.jit
def _fwd_kernel_ep_scatter_2(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE

    offset_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_s = offset_in_s < SCALE_HIDDEN_SIZE

    for token_id_int32 in range(start_token_id, total_token_num, grid_num):
        token_id = token_id_int32.to(tl.int64)
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
        to_copy_s = tl.load(
            recv_x_scale + token_id * recv_x_scale_stride0 + offset_in_s, mask=mask_s
        )

        for topk_index_int32 in tl.range(0, topk_num, 1, num_stages=4):
            topk_index = topk_index_int32.to(tl.int64)
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)
            if expert_id >= 0:
                dest_token_index_int32 = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index = dest_token_index_int32.to(tl.int64)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_index,
                    dest_token_index_int32,
                )
                output_tensor_ptr = (
                    output_tensor + dest_token_index * output_tensor_stride0
                )
                output_tensor_scale_ptr = (
                    output_tensor_scale + dest_token_index * output_tensor_scale_stride0
                )
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)
                tl.store(output_tensor_scale_ptr + offset_in_s, to_copy_s, mask=mask_s)


@register_kernel(
    "moe",
    "dispatch",
    name="deepep_moe_scatter",
    solution="deep_ep",
    signatures=format_signatures("x", "dense", {torch.float8_e4m3fn, torch.bfloat16}),
    traits={
        "comm_strategy": frozenset({"deep_ep"}),
    },
    priority=Priority.SPECIALIZED,
    tags={"throughput", "latency"},
)
@torch.no_grad()
def ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
):
    BLOCK_E = 128  # token num of per expert is aligned to 128
    BLOCK_D = 128  # block size of quantization
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    # grid = (triton.cdiv(hidden_size, BLOCK_D), num_experts)
    grid = num_experts

    assert m_indices.shape[0] % BLOCK_E == 0

    _fwd_kernel_ep_scatter_1[(grid,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )

    grid = min(recv_topk.shape[0], 1024 * 8)

    _fwd_kernel_ep_scatter_2[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        SCALE_HIDDEN_SIZE=hidden_size // BLOCK_D,
        SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size // BLOCK_D),
    )
    return


@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block_int32 = tl.program_id(0)
    cur_block = cur_block_int32.to(tl.int64)
    start_cur_token = tl.program_id(1)
    grid_num = tl.num_programs(1)

    for cur_token_int32 in range(start_cur_token, total_token_num, grid_num):
        cur_token = cur_token_int32.to(tl.int64)
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index_int32 in range(0, topk_num):
            topk_index = topk_index_int32.to(tl.int64)
            expert_id = tl.load(
                recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index
            )
            if expert_id >= 0:
                source_token_index_int32 = tl.load(
                    input_index + cur_token * input_index_stride0 + topk_index
                )
                source_token_index = source_token_index_int32.to(tl.int64)
                acc_weight = tl.load(
                    recv_topk_weight + cur_token * recv_topk_weight_stride0 + topk_index
                )
                tmp = tl.load(
                    input_tensor
                    + source_token_index * input_tensor_stride0
                    + cur_block * BLOCK_D
                    + off_d
                )
                accumulator += tmp.to(tl.float32) * acc_weight

        tl.store(
            output_tensor
            + cur_token * output_tensor_stride0
            + cur_block * BLOCK_D
            + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )


@register_kernel(
    "moe",
    "combine",
    name="deepep_moe_gather",
    solution="deep_ep",
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "comm_strategy": frozenset({"deep_ep"}),
    },
    priority=Priority.SPECIALIZED,
    tags={"throughput", "latency"},
)
@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
):
    BLOCK_D = 1024  # block size of quantization
    num_warps = 2
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    assert hidden_size % BLOCK_D == 0
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))
    _fwd_kernel_ep_gather[grid](
        num_tokens,
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )
    return


def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Global memory address of TMA must be 16-byte aligned.
    Since we use column-major layout for the LHS scaling tensor,
        the M-axis of the LHS scaling tensor needs to be padded to a multiple of 16 bytes.

    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.

    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    """
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return ceil_div(x, alignment) * alignment


@triton.jit
def _tma_align_input_scale_kernel(
    input_scale_ptr,
    output_ptr,
    m,
    k_div_block_size,
    input_scale_stride_m,
    input_scale_stride_k,
    output_stride_m,
    output_stride_k,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    grid_m = tl.num_programs(0)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)

    for m_base in range(pid_m, m, grid_m):
        input_offset = (
            input_scale_ptr
            + m_base * input_scale_stride_m
            + k_offsets * input_scale_stride_k
        )
        input_data = tl.load(input_offset, mask=k_offsets < k_div_block_size)

        output_offset = (
            output_ptr + k_offsets * output_stride_k + m_base * output_stride_m
        )
        tl.store(output_offset, input_data, mask=k_offsets < k_div_block_size)


def tma_align_input_scale(input_scale: torch.Tensor):
    assert input_scale.dim() == 2
    m, k_div_block_size = input_scale.shape
    padd_m = get_tma_aligned_size(m, input_scale.element_size())
    output = torch.empty(
        (k_div_block_size, padd_m), dtype=input_scale.dtype, device=input_scale.device
    )

    grid_m = min(m, 8192)
    BLOCK_SIZE_K = triton.next_power_of_2(k_div_block_size)

    _tma_align_input_scale_kernel[(grid_m,)](
        input_scale_ptr=input_scale,
        output_ptr=output,
        m=m,
        k_div_block_size=k_div_block_size,
        input_scale_stride_m=input_scale.stride(0),
        input_scale_stride_k=input_scale.stride(1),
        output_stride_m=output.stride(1),  # Note: these are swapped
        output_stride_k=output.stride(0),  # for column-major
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return output.t()[:m]
