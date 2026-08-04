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

import torch
from tokenspeed_kernel._triton import tl, triton


def _autotune_configs():
    # BLOCK_SIZE is always next_power_of_2(qkv_dim) — only num_warps/num_stages tuned.
    return [
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [4, 8]
        for ns in [2, 3, 4]
    ]


@triton.autotune(configs=_autotune_configs(), key=["qkv_dim"])
@triton.jit
def _fused_qkv_split_kernel(
    q,
    k,
    v,
    mixed_qkv,
    stride_t: tl.constexpr,
    stride_d: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,
    HEAD_V: tl.constexpr,
    qkv_dim,
    BLOCK_SIZE: tl.constexpr,
):
    i_t = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    q_dim: tl.constexpr = NUM_Q_HEADS * HEAD_Q
    k_dim: tl.constexpr = NUM_K_HEADS * HEAD_K
    v_dim: tl.constexpr = NUM_V_HEADS * HEAD_V
    qk_dim: tl.constexpr = q_dim + k_dim

    mask = offsets < qkv_dim
    values = tl.load(
        mixed_qkv + i_t * stride_t + offsets * stride_d,
        mask=mask,
    )

    tl.store(q + i_t * q_dim + offsets, values, mask=offsets < q_dim)

    k_offsets = offsets - q_dim
    tl.store(
        k + i_t * k_dim + k_offsets,
        values,
        mask=(offsets >= q_dim) & (offsets < qk_dim),
    )

    v_offsets = offsets - qk_dim
    tl.store(
        v + i_t * v_dim + v_offsets,
        values,
        mask=(offsets >= qk_dim) & (offsets < qkv_dim),
    )


@triton.autotune(configs=_autotune_configs(), key=["qkv_dim"])
@triton.jit
def _fused_qkv_split_l2norm_kernel(  # noqa: E501
    q,
    k,
    v,
    mixed_qkv,
    stride_t: tl.constexpr,
    stride_d: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,
    HEAD_V: tl.constexpr,
    qkv_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Split + per-head L2 normalisation of Q and K in one pass.

    Used only when the sm100 GDN fast-path is active — the caller omits the
    separate l2norm_fwd(query) / l2norm_fwd(key) passes.  V is written as-is.
    One program per token; per-head reduction is done inside BLOCK_SIZE.
    HEAD_Q must fit within BLOCK_SIZE (true for all Qwen3.5 configs).
    """
    i_t = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    q_dim: tl.constexpr = NUM_Q_HEADS * HEAD_Q
    k_dim: tl.constexpr = NUM_K_HEADS * HEAD_K
    v_dim: tl.constexpr = NUM_V_HEADS * HEAD_V
    qk_dim: tl.constexpr = q_dim + k_dim

    mask = offsets < qkv_dim
    values = tl.load(
        mixed_qkv + i_t * stride_t + offsets * stride_d,
        mask=mask,
        other=0.0,
    )

    # ── Q: per-head l2norm ──
    for h in tl.static_range(NUM_Q_HEADS):
        h_start = h * HEAD_Q
        h_end = h_start + HEAD_Q
        h_mask = (offsets >= h_start) & (offsets < h_end)
        q_vals = tl.where(h_mask, values, 0.0).to(tl.float32)
        norm = tl.sqrt(tl.sum(q_vals * q_vals) + 1e-6)
        q_vals_normed = (q_vals / norm).to(values.dtype)
        tl.store(q + i_t * q_dim + offsets, q_vals_normed, mask=h_mask)

    # ── K: per-head l2norm ──
    for h in tl.static_range(NUM_K_HEADS):
        h_start = q_dim + h * HEAD_K
        h_end = h_start + HEAD_K
        h_mask = (offsets >= h_start) & (offsets < h_end)
        k_vals = tl.where(h_mask, values, 0.0).to(tl.float32)
        norm = tl.sqrt(tl.sum(k_vals * k_vals) + 1e-6)
        k_vals_normed = (k_vals / norm).to(values.dtype)
        tl.store(k + i_t * k_dim + (offsets - q_dim), k_vals_normed, mask=h_mask)

    # ── V: passthrough ──
    v_offsets = offsets - qk_dim
    tl.store(
        v + i_t * v_dim + v_offsets,
        values,
        mask=(offsets >= qk_dim) & (offsets < qkv_dim),
    )


def fused_qkv_split_gdn_prefill(
    mixed_qkv: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_q: int,
    head_k: int,
    head_v: int,
    fuse_l2norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed post-conv GDN QKV into contiguous FLA prefill tensors.

    Replaces ``torch.split + view`` with a single Triton launch.
    Strided inputs are forced contiguous before the kernel (b3).

    Args:
        mixed_qkv: ``[T, qkv_dim]``, possibly strided.
        fuse_l2norm: when True, Q and K are L2-normalised per head inside the
            kernel (sm100 fast-path only — caller must not call l2norm_fwd
            separately).
    Returns:
        (q, k, v) each shaped ``[1, T, H, D]``.
    """
    if not mixed_qkv.is_contiguous():
        mixed_qkv = mixed_qkv.contiguous()

    seq_len = mixed_qkv.shape[0]
    q = torch.empty(
        (1, seq_len, num_q_heads, head_q),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    k = torch.empty(
        (1, seq_len, num_k_heads, head_k),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    v = torch.empty(
        (1, seq_len, num_v_heads, head_v),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )

    qkv_dim = num_q_heads * head_q + num_k_heads * head_k + num_v_heads * head_v
    block_size = triton.next_power_of_2(qkv_dim)
    kernel = _fused_qkv_split_l2norm_kernel if fuse_l2norm else _fused_qkv_split_kernel
    kernel[(seq_len,)](
        q,
        k,
        v,
        mixed_qkv,
        mixed_qkv.stride(0),
        mixed_qkv.stride(1),
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_q,
        head_k,
        head_v,
        qkv_dim,
        BLOCK_SIZE=block_size,
    )
    return q, k, v
