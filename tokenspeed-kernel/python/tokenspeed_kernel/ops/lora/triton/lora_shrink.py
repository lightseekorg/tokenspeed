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
``(stack_num * r, in_dim)``.  No-adapter segments use a negative slot
sentinel; the kernel returns immediately for that slot, leaving the output
rows untouched.  Real slots may have varying real ranks up to
``max_rank``; ``output[..., :rank * stack_num]`` stores the real product
and ``output[..., rank * stack_num:]`` is irrelevant — the consumer
(``lora_expand`` / ``lora_qkv_expand``) reads only the first ``rank * stack_num``
columns.

Adapted from sglang ``python/sglang/srt/lora/triton_ops/sgemm_lora_a.py``
(Apache-2.0): https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/triton_ops/sgemm_lora_a.py.
sglang's kernel is in turn descended from the Punica S-LoRA design
(https://github.com/punica-ai/punica).  Local changes: ported to
``tokenspeed_kernel._triton``, added ``@triton.autotune`` over the
``(N, K)`` shape with an on-disk config cache, and reshuffled the
constexpr params so block sizes come last.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.lora.triton.kernel_utils import _resolve_token_positions
from tokenspeed_kernel.ops.lora.triton.tuning import load_kernel_cache

# Shrink kernel: N = stack_num * rank (tiny, 16–192), K = in_dim (large,
# 4096+).  Decode-step segments are short (S = 1–32 per segment), so the
# right tile shape is "small N, large K, small S".  Sweep matches the
# sglang csgmv-shrink space (PR sgl-project/sglang#20391) plus a BLOCK_S
# axis since our kernel exposes it.  72 configs.
_SHRINK_CONFIGS = [
    triton.Config(
        {"BLOCK_S": s, "BLOCK_N": n, "BLOCK_K": k}, num_warps=w, num_stages=stages
    )
    for s in (8, 16, 32)
    for n in (16, 32, 64)
    for k in (64, 128, 256)
    for w in (4, 8)
    for stages in (2, 3, 4)
]


@triton.autotune(configs=_SHRINK_CONFIGS, key=["N", "K"])
@triton.jit
def _lora_shrink_kernel(
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
    if w_index < 0:
        return
    rank = tl.load(lora_ranks + w_index)

    # rank == 0 is defensive: skip and leave the output untouched
    # (downstream lora_expand / lora_qkv_expand is also a no-op for rank == 0
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
    k_offset = tl.max_contiguous(tl.arange(0, BLOCK_K), BLOCK_K)
    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_ptrs = x + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Hoist loop-invariant masks — s_mask and n_mask don't change across K
    # iterations so computing them once saves instructions in the hot loop.
    s_mask = s_offset[:, None] < seg_len  # (BLOCK_S, 1)
    n_mask = n_offset[None, :] < N  # (1, BLOCK_N)

    K = tl.multiple_of(K, BLOCK_K)
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, K // BLOCK_K):
        x_tile = tl.load(
            x_ptrs,
            mask=s_mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        w_tile = tl.load(
            w_ptrs,
            mask=n_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_mask = s_mask & n_mask
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    tl.store(output_ptr, partial_sum, mask=output_mask)


def lora_shrink_fwd(
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
        (the matching lora_expand kernel is also a no-op for those segments).
    """
    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert x.dim() == 2
    assert weights.dim() == 3

    S = x.shape[0]
    N = weights.shape[-2]
    K = weights.shape[-1]
    assert x.shape[-1] == K

    max_len = batch_info.max_len

    def grid(meta):
        return (
            triton.cdiv(max_len, meta["BLOCK_S"]) * triton.cdiv(N, meta["BLOCK_N"]),
            batch_info.bs,
        )

    sorted_by_adapter = batch_info.permutation is not None

    output = torch.empty((S, N), device=x.device, dtype=x.dtype)
    _lora_shrink_kernel[grid](
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
    )
    return output


# Eager pre-population from disk happens lazily inside the autotuner cache
# (see `tokenspeed_kernel.ops.lora.triton.__init__`).
load_kernel_cache(_lora_shrink_kernel)
