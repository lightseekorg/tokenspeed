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

"""Segmented LoRA-B matmul (expand: r → out_dim) with fused scale + add.

Adapted from sglang ``python/sglang/srt/lora/triton_ops/sgemm_lora_b.py``
(Apache-2.0): https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/triton_ops/sgemm_lora_b.py.
sglang's kernel is descended from the Punica S-LoRA design
(https://github.com/punica-ai/punica).  Local changes mirror those in
``lora_shrink.py`` (autotune + on-disk cache, constexpr ordering).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.lora.triton.kernel_utils import _resolve_token_positions
from tokenspeed_kernel.ops.lora.triton.tuning import load_kernel_cache

# Expand kernel: N = out_dim (large, 4096+), K = max_rank (tiny, 16–128).
# Tile space targets "large N, small K, small S".  Mirrors sglang's
# csgmv-expand grid (PR #20391); maxnreg helped with occupancy there.
#
# Profiling (2026-05-19) showed the kernel is instruction/overhead-bound
# (0% memory bandwidth utilisation).  Two improvements over the original
# k ∈ {16, 32} space:
#  • k=64, 128: when BLOCK_K == rank the inner K-loop runs exactly once,
#    eliminating loop overhead and the k-mask predicate entirely.
#  • BLOCK_N=128 with num_warps=4: halves CTA count vs BLOCK_N=64, which
#    amortises per-CTA fixed cost without increasing register pressure.
_EXPAND_CONFIGS = [
    triton.Config(
        {"BLOCK_S": s, "BLOCK_N": n, "BLOCK_K": k},
        num_warps=w,
        num_stages=stages,
        maxnreg=mr,
    )
    for s in (8, 16, 32)
    for n in (32, 64, 128)
    for k in (16, 32, 64, 128)
    for w in (4, 8)
    for stages in (1, 2, 3)
    for mr in (None, 128, 160)
]


@triton.autotune(configs=_EXPAND_CONFIGS, key=["N", "K"], restore_value=["output"])
@triton.jit
def _lora_expand_kernel(
    x,
    weights,
    output,
    N,  # out_dim
    K,  # max_rank
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
    scalings,
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

    # rank == 0 is defensive: leave the base output unchanged.
    if rank == 0:
        return

    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    scaling = tl.load(scalings + w_index)
    K = tl.minimum(K, rank)

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

    s_mask = s_offset[:, None] < seg_len  # hoisted: loop-invariant
    n_mask = n_offset[None, :] < N  # hoisted: loop-invariant (already was)
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
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    output_mask = s_mask & n_mask
    partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def lora_expand_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info,
    base_output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the LoRA-B expand and fuse-add into ``base_output``.

    Args:
        x: ``(s, max_rank)`` activations from lora_shrink.
        weights: ``(num_lora, out_dim, max_rank)``, contiguous.
        batch_info: :class:`LoraBatchInfo` describing the segment layout.
        base_output: optional ``(s, out_dim)`` to add into.  When ``None``,
            allocates a fresh zero-filled output.

    Returns:
        ``(s, out_dim)`` (same buffer as ``base_output`` when supplied).
    """
    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert x.dim() == 2
    assert weights.dim() == 3

    S = x.shape[0]
    N = weights.shape[-2]
    R = weights.shape[-1]
    assert x.shape[-1] == R

    max_len = batch_info.max_len

    def grid(meta):
        return (
            triton.cdiv(max_len, meta["BLOCK_S"]) * triton.cdiv(N, meta["BLOCK_N"]),
            batch_info.bs,
        )

    if base_output is None:
        output = torch.zeros((S, N), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    sorted_by_adapter = batch_info.permutation is not None
    _lora_expand_kernel[grid](
        x,
        weights,
        output,
        N,
        R,
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
        batch_info.scalings,
        sorted_by_adapter,
    )
    return output


load_kernel_cache(_lora_expand_kernel)
