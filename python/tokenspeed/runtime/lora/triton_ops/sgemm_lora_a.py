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

"""Segmented LoRA-A matmul (shrink: in_dim → r).

For each segment ``b`` in the batch the kernel computes
``output[seg_b] = x[seg_b] @ A[wi_b].T`` where ``A[wi_b]`` has shape
``(stack_num * r, in_dim)``.  Adapter ``slot 0`` is reserved for "no
adapter" (rank == 0); the kernel returns immediately for that slot, leaving
the output rows untouched.  Higher slots may have varying real ranks up to
``max_rank``; ``output[..., :rank * stack_num]`` stores the real product
and ``output[..., rank * stack_num:]`` is irrelevant — the consumer
(``sgemm_lora_b`` / ``qkv_lora_b``) reads only the first ``rank * stack_num``
columns.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tokenspeed.runtime.lora.triton_ops.kernel_utils import _resolve_token_positions


@triton.jit
def _sgemm_lora_a_kernel(
    x,
    weights,
    output,
    N,  # stack_num * max_rank
    K,  # in_dim
    stack_num,
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
    sorted_token_ids,
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)

    # rank == 0 ⇒ no-adapter slot.  Skip — the output is left untouched
    # (downstream sgemm_lora_b / qkv_lora_b is also a no-op for rank == 0
    # so the leftover values never feed into the base-output add).
    if rank == 0:
        return

    pid = tl.program_id(axis=0)
    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return

    # Cap N to the real ``stack_num * rank`` for this adapter.
    N = tl.minimum(N, rank * stack_num)

    num_pid_n = tl.cdiv(N, BLOCK_N)
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
    x_ptrs = x + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    w_ptrs = (weights + w_index * w_stride_0) + (
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
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < N),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < N)
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    tl.store(output_ptr, partial_sum, mask=output_mask)


def sgemm_lora_a_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info,
    stack_num: int = 1,
) -> torch.Tensor:
    """Run the LoRA-A shrink for an arbitrary batch.

    Args:
        x:        ``(s, in_dim)`` activations, contiguous.
        weights:  ``(num_lora, stack_num * max_rank, in_dim)``, contiguous.
        batch_info: :class:`LoraBatchInfo` describing the segment layout.
        stack_num: 1 for single projection, 3 for fused QKV, 2 for gate-up.

    Returns:
        ``(s, stack_num * max_rank)`` tensor.  Rows of segments whose adapter
        is the no-op slot are unwritten — callers must not consume them
        (the matching sgemm_lora_b kernel is also a no-op for those segments).
    """
    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert x.dim() == 2
    assert weights.dim() == 3

    S = x.shape[0]
    N = weights.shape[-2]
    K = weights.shape[-1]
    assert x.shape[-1] == K

    BLOCK_S = 16
    BLOCK_K = 256
    BLOCK_N = 16

    grid = (
        triton.cdiv(batch_info.max_len, BLOCK_S) * triton.cdiv(N, BLOCK_N),
        batch_info.bs,
    )

    sorted_by_adapter = batch_info.permutation is not None

    output = torch.empty((S, N), device=x.device, dtype=x.dtype)
    _sgemm_lora_a_kernel[grid](
        x,
        weights,
        output,
        N,
        K,
        stack_num,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        sorted_by_adapter,
        BLOCK_S,
        BLOCK_N,
        BLOCK_K,
    )
    return output
