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

Adapted from sglang ``python/sglang/srt/lora/triton_ops/qkv_lora_b.py``
(Apache-2.0): https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/triton_ops/qkv_lora_b.py.
Local changes: autotune + on-disk cache, constexpr ordering.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.lora.triton.kernel_utils import _resolve_token_positions
from tokenspeed_kernel.ops.lora.triton.tuning import load_kernel_cache

_QKV_EXPAND_CONFIGS = [
    triton.Config(
        {"BLOCK_S": s, "BLOCK_N": n, "BLOCK_K": k},
        num_warps=w,
        num_stages=stages,
        maxnreg=mr,
    )
    for s in (16, 32)
    for n in (32, 64, 128)
    for k in (16, 32, 64, 128)
    for w in (4, 8)
    for stages in (1, 2, 3)
    for mr in (None, 128, 160)
]


@triton.autotune(
    configs=_QKV_EXPAND_CONFIGS,
    key=["max_qkv_out_dim", "K"],
    restore_value=["output"],
)
@triton.jit
def _lora_qkv_expand_kernel(
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
    scalings,
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=2)
    w_index = tl.load(weight_indices + batch_id)
    if w_index < 0:
        return
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

    s_mask = s_offset[:, None] < seg_len
    n_mask = n_offset[None, :] < n_size
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x_tile = tl.load(
            x_ptrs,
            mask=s_mask & (k_offset[None, :] < k_remaining),
            other=0.0,
            eviction_policy="evict_first",
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < k_remaining) & n_mask,
            other=0.0,
            eviction_policy="evict_last",
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
    output_mask = s_mask & n_mask
    partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def lora_qkv_expand_fwd(
    x: torch.Tensor,
    qkv_lora_b: torch.Tensor,
    batch_info,
    output_offset: torch.Tensor,
    max_qkv_out_dim: int,
    base_output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply LoRA-B for the fused QKV linear, fused-add into ``base_output``.

    Args:
        x: ``(s, 3 * max_rank)`` from ``lora_shrink_fwd(stack_num=3)``.
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

    max_len = batch_info.max_len

    def grid(meta):
        return (
            triton.cdiv(max_len, meta["BLOCK_S"])
            * triton.cdiv(max_qkv_out_dim, meta["BLOCK_N"]),
            3,
            batch_info.bs,
        )

    if base_output is None:
        output = torch.zeros((s, output_dim), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    sorted_by_adapter = batch_info.permutation is not None
    _lora_qkv_expand_kernel[grid](
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
        batch_info.scalings,
        sorted_by_adapter,
    )
    return output


load_kernel_cache(_lora_qkv_expand_kernel)
