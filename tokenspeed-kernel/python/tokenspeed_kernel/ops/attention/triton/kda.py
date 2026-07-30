# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""AMD Triton primitives for Kimi Delta Attention.

The recurrence matches the vendored flash-linear-attention KDA implementation.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.triton.linear.l2norm import l2norm_fwd

__all__ = [
    "kda_recurrent",
    "kda_recurrent_decode",
    "kda_state_scatter",
]


@triton.jit
def _kda_prepare_gate_beta_kernel(
    raw_g,
    raw_beta,
    a_log,
    dt_bias,
    gate,
    beta,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_LOWER_BOUND: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    linear = (token_idx * H + head_idx) * D + offsets
    x = tl.load(raw_g + linear, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(
        dt_bias + head_idx * D + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x += bias
    a = tl.load(a_log + head_idx).to(tl.float32)
    if HAS_LOWER_BOUND:
        g = LOWER_BOUND * tl.sigmoid(tl.exp(a) * x)
    else:
        softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
        g = -tl.exp(a) * softplus
    tl.store(gate + linear, g, mask=mask)

    raw_b = tl.load(raw_beta + token_idx * H + head_idx).to(tl.float32)
    tl.store(beta + token_idx * H + head_idx, tl.sigmoid(raw_b))


@triton.heuristics(
    {
        "HAS_CU_SEQLENS": lambda args: args["cu_seqlens"] is not None,
        "HAS_STATE_INDICES": lambda args: args["state_indices"] is not None,
    }
)
@triton.jit(do_not_specialize=["TOTAL_TOKENS", "N"])
def _kda_recurrent_kernel(
    q,
    k,
    v,
    gate,
    beta,
    initial_state,
    final_state,
    output,
    cu_seqlens,
    state_indices,
    TOTAL_TOKENS: tl.int64,
    N: tl.int64,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_CU_SEQLENS: tl.constexpr,
    HAS_STATE_INDICES: tl.constexpr,
    QK_PRENORMALIZED: tl.constexpr,
):
    value_block = tl.program_id(0)
    sequence_head = tl.program_id(1)
    sequence_idx = sequence_head // H
    head_idx = sequence_head % H

    if HAS_CU_SEQLENS:
        begin = tl.load(cu_seqlens + sequence_idx).to(tl.int64)
        end = tl.load(cu_seqlens + sequence_idx + 1).to(tl.int64)
    else:
        tokens_per_sequence = TOTAL_TOKENS // N
        begin = sequence_idx * tokens_per_sequence
        end = begin + tokens_per_sequence

    value_offsets = value_block * BV + tl.arange(0, BV)
    value_mask = value_offsets < V
    if begin == end:
        # CUDA-graph padding represents trailing dummy requests as empty
        # sequences while q/k/v retain one physical row per captured request.
        # Define that otherwise-unwritten row so garbage activations cannot
        # enter later residual/MoE layers. The final packed offset distinguishes
        # trailing physical padding from an ordinary empty packed sequence.
        if HAS_CU_SEQLENS:
            packed_tokens = tl.load(cu_seqlens + N).to(tl.int64)
            if TOTAL_TOKENS == N and sequence_idx >= packed_tokens:
                tl.store(
                    output + (sequence_idx * H + head_idx) * V + value_offsets,
                    0.0,
                    mask=value_mask,
                )
        return

    if HAS_STATE_INDICES:
        state_idx = tl.load(state_indices + sequence_idx).to(tl.int64)
    else:
        state_idx = sequence_idx
    if state_idx < 0:
        # Graph capture binds the persistent buffers with invalid state IDs.
        # Keep the captured forward numerically defined without touching state.
        for token_idx in range(begin, end):
            tl.store(
                output + (token_idx * H + head_idx) * V + value_offsets,
                0.0,
                mask=value_mask,
            )
        return

    key_offsets = tl.arange(0, BK)
    key_mask = key_offsets < K
    state_mask = value_mask[:, None] & key_mask[None, :]
    state_base = (state_idx * H + head_idx) * V * K
    state_ptrs = (
        initial_state + state_base + value_offsets[:, None] * K + key_offsets[None, :]
    )
    running = tl.load(state_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    scale: tl.constexpr = K**-0.5

    for token_idx in range(begin, end):
        q_ptrs = q + (token_idx * H + head_idx) * K + key_offsets
        k_ptrs = k + (token_idx * H + head_idx) * K + key_offsets
        v_ptrs = v + (token_idx * H + head_idx) * V + value_offsets
        gate_ptrs = gate + (token_idx * H + head_idx) * K + key_offsets
        q_value = tl.load(q_ptrs, mask=key_mask, other=0.0).to(tl.float32)
        k_value = tl.load(k_ptrs, mask=key_mask, other=0.0).to(tl.float32)
        v_value = tl.load(v_ptrs, mask=value_mask, other=0.0).to(tl.float32)
        log_decay = tl.load(
            gate_ptrs,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        beta_value = tl.load(beta + token_idx * H + head_idx).to(tl.float32)

        if QK_PRENORMALIZED:
            q_value *= scale
        else:
            q_value *= tl.rsqrt(tl.sum(q_value * q_value) + 1e-6) * scale
            k_value *= tl.rsqrt(tl.sum(k_value * k_value) + 1e-6)
        running *= tl.exp(log_decay)[None, :]
        prediction = tl.sum(running * k_value[None, :], axis=1)
        delta = beta_value * (v_value - prediction)
        running += delta[:, None] * k_value[None, :]
        out_value = tl.sum(running * q_value[None, :], axis=1)
        tl.store(
            output + (token_idx * H + head_idx) * V + value_offsets,
            out_value,
            mask=value_mask,
        )

    final_ptrs = (
        final_state + state_base + value_offsets[:, None] * K + key_offsets[None, :]
    )
    tl.store(final_ptrs, running, mask=state_mask)


def kda_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float | None = -5.0,
    cu_seqlens: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the KDA recurrence for packed prefill or recurrent decode.

    Args:
        q: Query tensor ``[total_tokens, heads, key_dim]``.
        k: Key tensor with the same shape as ``q``.
        v: Value tensor ``[total_tokens, heads, value_dim]``.
        raw_g: Raw key-wise gate tensor with the same shape as ``q``.
        beta: Raw head-wise beta tensor ``[total_tokens, heads]``.
        state: State pool ``[slots, heads, value_dim, key_dim]`` or a single
            sequence state ``[heads, value_dim, key_dim]``.
        a_log: Head-wise FP32 decay parameter ``[heads]``.
        dt_bias: Key-wise FP32 bias ``[heads, key_dim]``.
        lower_bound: Optional safe lower bound for the log decay.
        cu_seqlens: Optional packed sequence boundaries. Trailing empty decode
            sequences represent physical CUDA-graph padding rows.
        state_indices: Optional state-pool slot per sequence. Negative slots
            skip state access and produce zero output for their token span.
        inplace: Update ``state`` directly instead of cloning it.

    Returns:
        Output tensor and updated state/state pool.
    """

    if not q.is_cuda:
        raise ValueError("KDA Triton recurrence requires GPU tensors")
    if q.shape != k.shape or q.shape != raw_g.shape:
        raise ValueError("q, k, and raw_g must have identical shapes")
    if q.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, v, and raw_g must be [tokens, heads, dim]")
    total_tokens, heads, key_dim = q.shape
    if v.shape[:2] != (total_tokens, heads):
        raise ValueError("v leading dimensions must match q")
    value_dim = v.shape[-1]
    if beta.shape != (total_tokens, heads):
        raise ValueError("beta must be [tokens, heads]")
    if a_log.shape != (heads,) or dt_bias.shape != (heads, key_dim):
        raise ValueError("invalid KDA gate parameter shapes")

    single_state = state.ndim == 3
    if single_state:
        state = state.unsqueeze(0)
    if state.ndim != 4 or state.shape[1:] != (heads, value_dim, key_dim):
        raise ValueError("state must be [slots, heads, value_dim, key_dim]")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    raw_g = raw_g.contiguous()
    beta = beta.contiguous()
    a_log = a_log.contiguous()
    dt_bias = dt_bias.contiguous()
    state = state.contiguous()
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        num_sequences = cu_seqlens.numel() - 1
    else:
        num_sequences = state_indices.numel() if state_indices is not None else 1
        if total_tokens % num_sequences:
            raise ValueError("tokens must divide evenly without cu_seqlens")
    if state_indices is not None:
        state_indices = state_indices.to(
            device=q.device,
            dtype=torch.int32,
        ).contiguous()
        if state_indices.numel() != num_sequences:
            raise ValueError("state_indices must have one entry per sequence")

    # Reuse the shared GDN/linear-attention L2 kernel for prefill. Decode keeps
    # normalization fused into the recurrent scan to avoid two extra launches.
    qk_prenormalized = total_tokens > num_sequences
    if qk_prenormalized:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    gate = torch.empty_like(raw_g, dtype=torch.float32)
    beta_sigmoid = torch.empty_like(beta, dtype=torch.float32)
    block_dim = triton.next_power_of_2(key_dim)
    _kda_prepare_gate_beta_kernel[(total_tokens, heads)](
        raw_g,
        beta,
        a_log,
        dt_bias,
        gate,
        beta_sigmoid,
        H=heads,
        D=key_dim,
        BLOCK_D=block_dim,
        HAS_LOWER_BOUND=lower_bound is not None,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        num_warps=min(max(block_dim // 32, 1), 8),
    )

    final_state = state if inplace else state.clone()
    output = torch.empty_like(v)
    block_key = triton.next_power_of_2(key_dim)
    block_value = min(triton.next_power_of_2(value_dim), 16)
    _kda_recurrent_kernel[(triton.cdiv(value_dim, block_value), num_sequences * heads)](
        q,
        k,
        v,
        gate,
        beta_sigmoid,
        state,
        final_state,
        output,
        cu_seqlens,
        state_indices,
        TOTAL_TOKENS=total_tokens,
        N=num_sequences,
        H=heads,
        K=key_dim,
        V=value_dim,
        BK=block_key,
        BV=block_value,
        QK_PRENORMALIZED=qk_prenormalized,
        num_warps=1,
        num_stages=3,
    )
    if single_state:
        final_state = final_state.squeeze(0)
    return output, final_state


@triton.jit
def _kda_recurrent_decode_kernel(
    q,
    k,
    v,
    raw_g,
    raw_beta,
    state_pool,
    read_indices,
    write_indices,
    output,
    cu_seqlens,
    a_log,
    dt_bias,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    STATE_PAGE_STRIDE: tl.constexpr,
    HAS_LOWER_BOUND: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
):
    """One-token KDA decode with direct indexed state-pool IO."""

    value_block = tl.program_id(0)
    sequence_head = tl.program_id(1)
    sequence_idx = sequence_head // H
    head_idx = sequence_head % H

    begin = tl.load(cu_seqlens + sequence_idx).to(tl.int64)
    end = tl.load(cu_seqlens + sequence_idx + 1).to(tl.int64)
    value_offsets = value_block * BV + tl.arange(0, BV)
    value_mask = value_offsets < V
    if begin == end:
        tl.store(
            output + (sequence_idx * H + head_idx) * V + value_offsets,
            0.0,
            mask=value_mask,
        )
        return

    read_idx = tl.load(read_indices + sequence_idx).to(tl.int64)
    write_idx = tl.load(write_indices + sequence_idx).to(tl.int64)
    valid_read = (read_idx >= 0) & (read_idx < NUM_SLOTS)
    if not valid_read:
        tl.store(
            output + (sequence_idx * H + head_idx) * V + value_offsets,
            0.0,
            mask=value_mask,
        )
        return

    key_offsets = tl.arange(0, BK)
    key_mask = key_offsets < K
    state_mask = value_mask[:, None] & key_mask[None, :]
    read_base = read_idx * STATE_PAGE_STRIDE + head_idx * V * K
    read_ptrs = (
        state_pool + read_base + value_offsets[:, None] * K + key_offsets[None, :]
    )
    running = tl.load(read_ptrs, mask=state_mask, other=0.0).to(tl.float32)

    token_idx = begin
    q_ptrs = q + (token_idx * H + head_idx) * K + key_offsets
    k_ptrs = k + (token_idx * H + head_idx) * K + key_offsets
    v_ptrs = v + (token_idx * H + head_idx) * V + value_offsets
    gate_ptrs = raw_g + (token_idx * H + head_idx) * K + key_offsets

    q_value = tl.load(q_ptrs, mask=key_mask, other=0.0).to(tl.float32)
    k_value = tl.load(k_ptrs, mask=key_mask, other=0.0).to(tl.float32)
    v_value = tl.load(v_ptrs, mask=value_mask, other=0.0).to(tl.float32)
    gate_value = tl.load(gate_ptrs, mask=key_mask, other=0.0).to(tl.float32)
    gate_value += tl.load(
        dt_bias + head_idx * K + key_offsets, mask=key_mask, other=0.0
    ).to(tl.float32)
    a = tl.load(a_log + head_idx).to(tl.float32)
    if HAS_LOWER_BOUND:
        log_decay = LOWER_BOUND * tl.sigmoid(tl.exp(a) * gate_value)
    else:
        softplus = tl.maximum(gate_value, 0.0) + tl.log(
            1.0 + tl.exp(-tl.abs(gate_value))
        )
        log_decay = -tl.exp(a) * softplus
    beta_value = tl.sigmoid(tl.load(raw_beta + token_idx * H + head_idx).to(tl.float32))

    scale: tl.constexpr = K**-0.5
    q_value *= tl.rsqrt(tl.sum(q_value * q_value) + 1e-6) * scale
    k_value *= tl.rsqrt(tl.sum(k_value * k_value) + 1e-6)
    running *= tl.exp(log_decay)[None, :]
    prediction = tl.sum(running * k_value[None, :], axis=1)
    delta = beta_value * (v_value - prediction)
    running += delta[:, None] * k_value[None, :]
    out_value = tl.sum(running * q_value[None, :], axis=1)
    tl.store(
        output + (sequence_idx * H + head_idx) * V + value_offsets,
        out_value,
        mask=value_mask,
    )

    valid_write = (write_idx >= 0) & (write_idx < NUM_SLOTS)
    safe_write_idx = tl.where(valid_write, write_idx, 0)
    write_base = safe_write_idx * STATE_PAGE_STRIDE + head_idx * V * K
    write_ptrs = (
        state_pool + write_base + value_offsets[:, None] * K + key_offsets[None, :]
    )
    tl.store(write_ptrs, running, mask=valid_write & state_mask)


def kda_recurrent_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    lower_bound: float | None = -5.0,
    block_value: int = 8,
) -> torch.Tensor:
    """Run fused one-token KDA decode directly against an indexed state pool."""

    if not q.is_cuda:
        raise ValueError("KDA Triton decode requires GPU tensors")
    if q.shape != k.shape or q.shape != raw_g.shape:
        raise ValueError("q, k, and raw_g must have identical shapes")
    if q.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, v, and raw_g must be [tokens, heads, dim]")
    total_tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if total_tokens != read_indices.numel() or total_tokens != write_indices.numel():
        raise ValueError("one-token decode requires one state index per token")
    if cu_seqlens.numel() != total_tokens + 1:
        raise ValueError("cu_seqlens must contain one boundary per decode row")
    if beta.shape != (total_tokens, heads):
        raise ValueError("beta must be [tokens, heads]")
    if state_pool.ndim != 4 or state_pool.shape[1:] != (
        heads,
        value_dim,
        key_dim,
    ):
        raise ValueError("state_pool must be [slots, heads, value_dim, key_dim]")
    expected_inner_strides = (value_dim * key_dim, key_dim, 1)
    if state_pool.stride()[1:] != expected_inner_strides:
        raise ValueError(
            "state_pool inner [heads, value_dim, key_dim] dimensions must be "
            "contiguous"
        )
    if state_pool.stride(0) < heads * value_dim * key_dim:
        raise ValueError("state_pool pages must not overlap")
    if a_log.shape != (heads,) or dt_bias.shape != (heads, key_dim):
        raise ValueError("invalid KDA gate parameter shapes")
    if block_value <= 0 or triton.next_power_of_2(block_value) != block_value:
        raise ValueError("block_value must be a positive power of two")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    raw_g = raw_g.contiguous()
    beta = beta.contiguous()
    a_log = a_log.contiguous()
    dt_bias = dt_bias.contiguous()
    cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    read_indices = read_indices.to(device=q.device, dtype=torch.int32).contiguous()
    write_indices = write_indices.to(device=q.device, dtype=torch.int32).contiguous()

    output = torch.empty_like(v)
    block_key = triton.next_power_of_2(key_dim)
    block_value = min(block_value, triton.next_power_of_2(value_dim))
    _kda_recurrent_decode_kernel[
        (triton.cdiv(value_dim, block_value), total_tokens * heads)
    ](
        q,
        k,
        v,
        raw_g,
        beta,
        state_pool,
        read_indices,
        write_indices,
        output,
        cu_seqlens,
        a_log,
        dt_bias,
        H=heads,
        K=key_dim,
        V=value_dim,
        BK=block_key,
        BV=block_value,
        NUM_SLOTS=state_pool.shape[0],
        STATE_PAGE_STRIDE=state_pool.stride(0),
        HAS_LOWER_BOUND=lower_bound is not None,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        num_warps=1,
        num_stages=3,
    )
    return output


@triton.jit
def _kda_state_scatter_kernel(
    state_pool,
    updates,
    indices,
    state_pool_stride,
    updates_stride,
    NUM_SLOTS: tl.constexpr,
    ELEMENTS_PER_STATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Write one recurrent state per sequence, skipping invalid state IDs."""

    element_block = tl.program_id(0)
    sequence_idx = tl.program_id(1)
    state_idx = tl.load(indices + sequence_idx).to(tl.int64)
    offsets = element_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_state = (state_idx >= 0) & (state_idx < NUM_SLOTS)
    safe_state_idx = tl.where(valid_state, state_idx, 0)
    mask = valid_state & (offsets < ELEMENTS_PER_STATE)

    values = tl.load(updates + sequence_idx * updates_stride + offsets, mask=mask)
    tl.store(
        state_pool + safe_state_idx * state_pool_stride + offsets,
        values,
        mask=mask,
    )


def kda_state_scatter(
    state_pool: torch.Tensor,
    updates: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """Write KDA recurrent states to indexed pool slots.

    Args:
        state_pool: Destination recurrent-state pool with shape
            ``[num_slots, num_heads, value_dim, key_dim]``.
        updates: Per-sequence final recurrent states with shape
            ``[num_sequences, num_heads, value_dim, key_dim]``.
        indices: Destination slot IDs with shape ``[num_sequences]``. Negative
            or out-of-range IDs are skipped without mutating the pool.

    Returns:
        ``None``. The destination pool is updated in place.
    """

    if not (state_pool.is_cuda and updates.is_cuda and indices.is_cuda):
        raise ValueError("kda_state_scatter requires GPU tensors")
    if state_pool.ndim != 4 or updates.ndim != 4:
        raise ValueError(
            "state_pool and updates must be [slots/sequences, heads, value_dim, key_dim]"
        )
    if state_pool.shape[1:] != updates.shape[1:]:
        raise ValueError(
            "state_pool and updates must agree on [heads, value_dim, key_dim]"
        )
    if indices.ndim != 1 or indices.numel() != updates.shape[0]:
        raise ValueError("indices must contain one destination ID per update")
    if not state_pool.is_contiguous() or not updates.is_contiguous():
        raise ValueError("state_pool and updates must be contiguous")

    num_sequences = updates.shape[0]
    if num_sequences == 0:
        return

    elements_per_state = updates[0].numel()
    block_size = 256
    grid = (triton.cdiv(elements_per_state, block_size), num_sequences)
    _kda_state_scatter_kernel[grid](
        state_pool,
        updates,
        indices,
        state_pool.stride(0),
        updates.stride(0),
        NUM_SLOTS=state_pool.shape[0],
        ELEMENTS_PER_STATE=elements_per_state,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
