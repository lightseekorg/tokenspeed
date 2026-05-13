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

"""Fused LoRA-B expand for stacked Q/K/V projections.

The QKV linear is fused into a single matmul with output layout
``[q_per_tp, k_per_tp, v_per_tp]``.  This kernel packs the three B
projections into one launch: each program instance picks ``q``, ``k``, or
``v`` via ``program_id(1)`` and writes its tile into the matching slice of
the fused output.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tokenspeed.runtime.lora.triton_ops.kernel_utils import _resolve_token_positions


@triton.jit
def _qkv_lora_b_kernel(
    x,
    weights,
    output,
    K,  # max_rank
    max_qkv_out_dim,  # max(q_per_tp, kv_per_tp)
    x_stride_0,
    x_stride_1,
    w_stride_0,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    n_offs,  # (4,) cumulative offsets into the fused QKV output
    sorted_token_ids,
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    scalings,
):
    batch_id = tl.program_id(axis=2)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)
    if rank == 0:
        return

    qkv_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    n_start = tl.load(n_offs + qkv_id)
    n_size = tl.load(n_offs + qkv_id + 1) - n_start
    scaling = tl.load(scalings + w_index)
    K = tl.minimum(K, rank)

    num_pid_n = tl.cdiv(max_qkv_out_dim, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_ptrs = (
        x
        + (qkv_id * K) * x_stride_1
        + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )
    w_ptrs = (weights + w_index * w_stride_0 + n_start * w_stride_1) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < n_size),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = (
        output
        + n_start * output_stride_1
        + (s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1)
    )
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < n_size)
    partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def qkv_lora_b_fwd(
    x: torch.Tensor,
    qkv_lora_b: torch.Tensor,
    batch_info,
    output_offset: torch.Tensor,
    max_qkv_out_dim: int,
    base_output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply LoRA-B for the fused QKV linear, fused-add into ``base_output``.

    Args:
        x: ``(s, 3 * max_rank)`` from ``sgemm_lora_a_fwd(stack_num=3)``.
        qkv_lora_b: ``(num_lora, q_per_tp + 2 * kv_per_tp, max_rank)``.
        batch_info: :class:`LoraBatchInfo`.
        output_offset: ``(4,)`` cumulative offsets ``[0, q, q+kv, q+2*kv]``.
        max_qkv_out_dim: ``max(q_per_tp, kv_per_tp)`` — used to size the grid.
        base_output: ``(s, q_per_tp + 2 * kv_per_tp)`` to fuse-add into.
    """
    s = x.shape[0]
    input_dim = x.shape[1]
    r = qkv_lora_b.shape[-1]
    output_dim = qkv_lora_b.shape[-2]
    assert input_dim == 3 * r
    assert output_offset.shape[0] == 4

    BLOCK_S = 16
    BLOCK_R = 16
    BLOCK_OUT = 64

    grid_b = (
        triton.cdiv(batch_info.max_len, BLOCK_S)
        * triton.cdiv(max_qkv_out_dim, BLOCK_OUT),
        3,
        batch_info.bs,
    )

    if base_output is None:
        output = torch.zeros((s, output_dim), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    sorted_by_adapter = batch_info.permutation is not None
    _qkv_lora_b_kernel[grid_b](
        x,
        qkv_lora_b,
        output,
        r,
        max_qkv_out_dim,
        x.stride(0),
        x.stride(1),
        qkv_lora_b.stride(0),
        qkv_lora_b.stride(1),
        qkv_lora_b.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        output_offset,
        batch_info.permutation,
        sorted_by_adapter,
        BLOCK_S,
        BLOCK_OUT,
        BLOCK_R,
        batch_info.scalings,
    )
    return output
