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

import numpy as np
import torch
from tokenspeed_kernel._triton import tl, triton

_PAD_SLOT_ID = -1
_CONV1D_BLOCK_M = 8
_CONV1D_BLOCK_N = 256



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


# ---------------------------------------------------------------------------
# Fused causal_conv1d + QKV split
# ---------------------------------------------------------------------------
# Kernel body mirrors _causal_conv1d_fwd_kernel from causal_conv1d.py.
# Only the output write is changed: instead of a single merged buffer,
# each program routes to q_ptr / k_ptr / v_ptr based on its channel block.
# BLOCK_N=256 divides Q_DIM / K_DIM / V_DIM exactly for all Qwen3.5 configs,
# so every program falls cleanly into one output region.


@triton.jit()
def _causal_conv1d_qkv_split_fwd_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    initial_states_ptr,
    cache_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    batch: tl.int32,
    dim: tl.constexpr,
    seqlen: tl.int32,
    num_cache_lines: tl.constexpr,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1

    idx_seq = tl.load(batch_ptr + tl.program_id(0))
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))

    pid_n = tl.program_id(1)
    idx_feats = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim

    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            return
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )

    w_base = w_ptr + (idx_feats * stride_w_dim)

    if chunk_offset == 0:
        load_init_state = False
        if HAS_INITIAL_STATES:
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            if KERNEL_WIDTH >= 2:
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            conv_states_ptrs_target = (
                conv_states_base[None, :]
                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
        else:
            if load_init_state:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)
                VAL = state_len - seqlen
                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                tl.debug_barrier()
                new_conv_state = tl.where(mask, conv_state, loaded_x)
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
    else:
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            conv_states_ptrs = prior_tokens
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")  # noqa: F841
            conv_states_ptrs = prior_tokens - 1 * stride_x_token
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 3 * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token

    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))

        mask_1d = (idx_token < segment_len) & (idx_feats < dim)
        g = sequence_start_index + token_offset + idx_token
        if pid_n < Q_DIM // BLOCK_N:
            o_ptrs = q_ptr + g * Q_DIM + idx_feats
        elif pid_n < (Q_DIM + K_DIM) // BLOCK_N:
            o_ptrs = k_ptr + g * K_DIM + (idx_feats - Q_DIM)
        else:
            o_ptrs = v_ptr + g * V_DIM + (idx_feats - Q_DIM - K_DIM)
        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_qkv_split_gdn_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_q: int,
    head_k: int,
    head_v: int,
    total_seq_len: int,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = _PAD_SLOT_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused causal_conv1d + QKV split for GDN prefill.

    Replaces the two-kernel sequence (causal_conv1d_fn → fused_qkv_split_gdn_prefill)
    with a single launch, eliminating the intermediate conv_out staging buffer.

    Args:
        x: (dim, cu_seq_len), channel-last (stride(0)=1).
        weight: (dim, conv_width).
        total_seq_len: actual token count to allocate; caller may pass seq_len
            to avoid over-allocation when cu_seq_len includes padding.
    Returns:
        (q, k, v) each shaped (1, total_seq_len, H, D).
    """
    Q_DIM = num_q_heads * head_q
    K_DIM = num_k_heads * head_k
    V_DIM = num_v_heads * head_v
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)
    padded_batch = query_start_loc.size(0) - 1

    q = torch.empty((1, total_seq_len, num_q_heads, head_q), dtype=x.dtype, device=x.device)
    k = torch.empty((1, total_seq_len, num_k_heads, head_k), dtype=x.dtype, device=x.device)
    v = torch.empty((1, total_seq_len, num_v_heads, head_v), dtype=x.dtype, device=x.device)

    num_cache_lines = 0
    stride_istate_seq = stride_istate_dim = stride_istate_token = 0
    if conv_states is not None:
        num_cache_lines = conv_states.size(0)
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)

    seqlens_np = np.asarray(seq_lens_cpu)
    nums = (seqlens_np + _CONV1D_BLOCK_M - 1) // _CONV1D_BLOCK_M
    tot = int(nums.sum())
    if tot == 0:
        return q, k, v

    mlist = np.repeat(np.arange(len(nums)), nums)
    offsetlist = np.arange(tot) - np.repeat(np.cumsum(nums) - nums, nums)
    combined = np.stack([mlist, offsetlist]).astype(np.int32, copy=False)
    combined_cpu = torch.from_numpy(combined).pin_memory()

    batch_ptr = torch.full((tot + 1,), pad_slot_id, dtype=torch.int32, device=x.device)
    token_chunk_offset_ptr = torch.full((tot + 1,), pad_slot_id, dtype=torch.int32, device=x.device)
    batch_ptr[:tot].copy_(combined_cpu[0], non_blocking=True)
    token_chunk_offset_ptr[:tot].copy_(combined_cpu[1], non_blocking=True)

    grid = (tot, triton.cdiv(dim, _CONV1D_BLOCK_N))

    _causal_conv1d_qkv_split_fwd_kernel[grid](
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        token_chunk_offset_ptr,
        q,
        k,
        v,
        padded_batch,
        dim,
        cu_seqlen,
        num_cache_lines,
        Q_DIM,
        K_DIM,
        V_DIM,
        0,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        HAS_INITIAL_STATES=has_initial_state is not None,
        HAS_CACHE=conv_states is not None,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        BLOCK_M=_CONV1D_BLOCK_M,
        BLOCK_N=_CONV1D_BLOCK_N,
        num_stages=2,
    )
    return q, k, v
