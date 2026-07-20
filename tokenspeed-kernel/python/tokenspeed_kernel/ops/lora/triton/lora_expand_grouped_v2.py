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

"""Adapter-grouped LoRA-B expand without gather/scatter overhead.

Adapts vLLM's token-sorted dispatch pattern (PR vllm-project/vllm#...,
Apache-2.0) to our kernel infrastructure.

This kernel reads ``x`` and writes ``output`` directly at the original
  (unsorted) token positions using ``token_indices`` loaded inside the kernel.
  No gather/scatter needed — only a cheap pointer indirection per tile.

Grid: ``(cdiv(N, BLOCK_N), num_groups)``  — axis 1 = unique adapter count.
Within each CTA, groups of ``BLOCK_S`` tokens are processed; each group loads
``BLOCK_S`` scattered rows from ``x`` via ``token_indices``.

Adapted from vLLM ``vllm/lora/ops/triton_ops/lora_expand_op.py`` (Apache-2.0):
https://github.com/vllm-project/vllm/blob/main/vllm/lora/ops/triton_ops/lora_expand_op.py
Local changes: removed SPLIT_K / PDL / CAST_TYPE / multi-slice indirection;
added BLOCK_K ∈ {16,32,64,128} + tl.multiple_of EVEN_K; adopted our
eviction-policy hints and autotune + on-disk cache infrastructure.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.lora.triton.tuning import load_kernel_cache

_GROUPED_V2_CONFIGS = [
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
    configs=_GROUPED_V2_CONFIGS,
    key=["N", "MAX_RANK"],
    restore_value=["output"],
)
@triton.jit(do_not_specialize=["output_stride_0", "output_stride_1"])
def _lora_expand_grouped_v2_kernel(
    x,  # (M, MAX_RANK)  original unsorted token order
    weights,  # (n_slots, N, MAX_RANK)
    output,  # (M, N)  written at original token positions
    group_slots,  # (num_groups,) int32 — weight-slot index per group
    group_starts,  # (num_groups,) int32 — start in token_indices
    group_sizes,  # (num_groups,) int32 — tokens per group
    token_indices,  # (M,) int32  — token positions sorted by adapter
    scalings,  # (n_slots,) float32
    lora_ranks,  # (n_slots,) int32
    output_stride_0,
    output_stride_1,
    N: tl.constexpr,
    MAX_RANK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Constexpr strides — x and weights are always contiguous.
    x_stride_0: tl.constexpr = MAX_RANK
    x_stride_1: tl.constexpr = 1
    w_stride_0: tl.constexpr = N * MAX_RANK
    w_stride_1: tl.constexpr = MAX_RANK  # row stride inside (N, MAX_RANK) slice
    w_stride_2: tl.constexpr = 1

    group_id = tl.program_id(axis=1)
    # axis=0 encodes both the within-group M-tile and the N-tile.
    # Grid: (cdiv(M, BLOCK_S) * cdiv(N, BLOCK_N), num_groups) — mirrors vLLM's
    # (M_tiles × N_tiles, num_active_loras) layout.  CTAs whose M-tile exceeds
    # the group size exit immediately (same early-exit pattern as vLLM).
    pid_flat = tl.program_id(axis=0)
    cta_n_num = tl.cdiv(N, BLOCK_N)
    pid_m = pid_flat // cta_n_num
    pid_n = pid_flat % cta_n_num

    w_index = tl.load(group_slots + group_id)
    if w_index < 0:
        return
    g_size = tl.load(group_sizes + group_id)
    if g_size == 0:
        return
    rank = tl.load(lora_ranks + w_index)
    if rank == 0:
        return

    m_off = pid_m * BLOCK_S
    if m_off >= g_size:
        return  # early exit for M-tiles beyond this group's token count

    g_start = tl.load(group_starts + group_id)
    scaling = tl.load(scalings + w_index)
    K = tl.minimum(MAX_RANK, rank)

    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.max_contiguous(tl.arange(0, BLOCK_K), BLOCK_K)
    n_mask = n_offset[None, :] < N

    # Load physical token positions for this M-tile.
    s_offset = tl.arange(0, BLOCK_S)
    m_valid = s_offset < g_size - m_off
    tok_ptrs = token_indices + g_start + m_off + s_offset
    ram = tl.load(tok_ptrs, mask=m_valid, other=0)
    s_valid = m_valid[:, None]

    # Scattered read of x — no pre-gather needed.
    x_ptrs = x + ram[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    partial = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        x_tile = tl.load(
            x_ptrs,
            mask=s_valid & (k_offset[None, :] < k_rem),
            other=0.0,
            eviction_policy="evict_first",
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < k_rem) & n_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        partial += tl.dot(x_tile, w_tile)
        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    partial *= scaling
    partial = partial.to(x.dtype.element_ty)

    # Scattered write — no post-scatter needed.
    out_ptrs = (
        output + ram[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    out_mask = s_valid & n_mask
    partial += tl.load(out_ptrs, mask=out_mask, other=0.0)
    tl.store(out_ptrs, partial, mask=out_mask)


def lora_expand_grouped_v2_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info,
    base_output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Adapter-grouped expand without gather/scatter.

    Reads ``x`` and writes ``output`` at original token positions using
    ``batch_info.token_indices`` (sorted by adapter).  Requires batch_info to
    have the adapter-group metadata populated by ``prepare_loras``:
    ``token_indices``, ``group_slots``, ``group_starts``, ``group_sizes``,
    ``num_groups``.

    Drops in for :func:`lora_expand_fwd` when ``batch_info.num_groups > 0``
    and ``batch_info.bs // batch_info.num_groups >= 8``.
    """
    assert x.is_contiguous()
    assert weights.is_contiguous()

    S, R = x.shape
    N = weights.shape[-2]
    dev, dt = x.device, x.dtype

    num_groups = batch_info.num_groups

    # Use the largest group size for the M dimension, not the total batch size.
    # This makes the grid tight for both extremes:
    #   • n_unique = n  (all different): max_group_size = 1
    #     → grid = (1 × cdiv(N,BLOCK_N), n) ≡ segmented layout, zero wasted CTAs
    #   • n_unique = 1  (all same):      max_group_size = n
    #     → grid = (n/BLOCK_S × cdiv(N,BLOCK_N), 1) ≡ grpv2 layout
    # max_group_size is pre-computed on CPU in prepare_loras — no GPU sync here.
    max_group_size = batch_info.max_group_size

    def grid(meta):
        return (
            triton.cdiv(max_group_size, meta["BLOCK_S"])
            * triton.cdiv(N, meta["BLOCK_N"]),
            num_groups,
        )

    output = (
        torch.zeros((S, N), device=dev, dtype=dt)
        if base_output is None
        else base_output
    )

    _lora_expand_grouped_v2_kernel[grid](
        x,
        weights,
        output,
        batch_info.group_slots[:num_groups],
        batch_info.group_starts[:num_groups],
        batch_info.group_sizes[:num_groups],
        batch_info.sort_order[: batch_info.bs],  # token_indices sorted by adapter
        batch_info.scalings,
        batch_info.lora_ranks,
        output.stride(0),
        output.stride(1),
        N=N,
        MAX_RANK=R,
    )
    return output


load_kernel_cache(_lora_expand_grouped_v2_kernel)
