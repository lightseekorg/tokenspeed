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

"""Prefill-optimised LoRA-A matmul (shrink: in_dim → r).

Drop-in replacement for :func:`lora_shrink_fwd` on prefill batches
(``max_len > 32``).  Identical algorithm; the structural difference is that
``K`` (= in_dim, 4096+), ``N`` (= stack_num * max_rank), and all strides are
**constexpr** — the compiler specialises the K-loop trip count at compile
time and eliminates all stride multiplications.

Benchmarked gain on H100 vs the decode shrink kernel at s=512, rank=64:
  QKV  stack=3  (K=4096, N=192): 23 µs → 17 µs  (1.3×)
  g/up stack=2  (K=4096, N=128): 19 µs → 16 µs  (1.2×)
  single        (K=4096, N=64):  18 µs → 17 µs  (~1.0×)

Adapted from sglang ``python/sglang/srt/lora/triton_ops/chunked_sgmv_shrink.py``
(Apache-2.0): https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/triton_ops/chunked_sgmv_shrink.py.
Local changes: kept SORTED_BY_ADAPTER + S-tiling from our decode kernel
(``lora_shrink.py``), replaced fixed configs with ``@triton.autotune`` +
on-disk cache.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.lora.triton.kernel_utils import _resolve_token_positions
from tokenspeed_kernel.ops.lora.triton.tuning import load_kernel_cache

# Same config space as the decode shrink kernel.
_PREFILL_SHRINK_CONFIGS = [
    triton.Config(
        {"BLOCK_S": s, "BLOCK_N": n, "BLOCK_K": k}, num_warps=w, num_stages=stages
    )
    for s in (16, 32)
    for n in (16, 32, 64)
    for k in (64, 128, 256)
    for w in (4, 8)
    for stages in (2, 3, 4)
]


@triton.autotune(configs=_PREFILL_SHRINK_CONFIGS, key=["N", "K", "NUM_SLICES"])
@triton.jit
def _lora_shrink_prefill_kernel(
    x,
    weights,
    output,
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,
    N: tl.constexpr,  # stack_num * max_rank
    K: tl.constexpr,  # in_dim
    NUM_SLICES: tl.constexpr,  # stack_num
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Constexpr strides — compiler eliminates all stride multiplications.
    x_stride_0: tl.constexpr = K
    x_stride_1: tl.constexpr = 1
    w_stride_0: tl.constexpr = N * K
    w_stride_1: tl.constexpr = K  # row stride of the (N, K) weight matrix
    w_stride_2: tl.constexpr = 1
    output_stride_0: tl.constexpr = N
    output_stride_1: tl.constexpr = 1

    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    if w_index < 0:
        return
    rank = tl.load(lora_ranks + w_index)
    if rank == 0:
        return

    pid = tl.program_id(axis=0)
    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return

    cur_n = tl.minimum(N, rank * NUM_SLICES)

    num_pid_n = tl.cdiv(cur_n, BLOCK_N)
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

    s_mask = s_offset[:, None] < seg_len
    n_mask = n_offset[None, :] < cur_n
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


def lora_shrink_prefill_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info,
    stack_num: int = 1,
) -> torch.Tensor:
    """Prefill-optimised LoRA-A shrink.  Same signature as :func:`lora_shrink_fwd`.

    Args:
        x:        ``(s, in_dim)`` activations, contiguous.
        weights:  ``(num_lora, stack_num * max_rank, in_dim)``, contiguous.
        batch_info: :class:`LoraBatchInfo`.
        stack_num: 1 for single projection, 3 for fused QKV, 2 for gate-up.

    Returns:
        ``(s, stack_num * max_rank)`` tensor.
    """
    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert x.dim() == 2
    assert weights.dim() == 3

    S = x.shape[0]
    N = weights.shape[-2]  # stack_num * max_rank
    K = weights.shape[-1]  # in_dim
    assert x.shape[-1] == K

    max_len = batch_info.max_len
    sorted_by_adapter = batch_info.permutation is not None

    def grid(meta):
        return (
            triton.cdiv(max_len, meta["BLOCK_S"]) * triton.cdiv(N, meta["BLOCK_N"]),
            batch_info.bs,
        )

    output = torch.empty((S, N), device=x.device, dtype=x.dtype)
    _lora_shrink_prefill_kernel[grid](
        x,
        weights,
        output,
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        N=N,
        K=K,
        NUM_SLICES=stack_num,
        SORTED_BY_ADAPTER=sorted_by_adapter,
    )
    return output


load_kernel_cache(_lora_shrink_prefill_kernel)
