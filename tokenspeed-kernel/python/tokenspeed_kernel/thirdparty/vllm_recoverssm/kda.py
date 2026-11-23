# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 RecoverSSM verify with stamped retained KDA records.

Based on vLLM c77b3faa1cc2c39edbe82365f28ce96981af1438. Retained records keep
FP32 U, normalized K, multiplicative decay and the accepted length in resident
request slots. Verify reads each live page's dense base plus the slot's
accepted records and writes this round's candidate records and stamps; it
retains the upstream recurrence, static token loop, and one-warp launch. The
batched prepare before verify and the batched commit after it are TokenSpeed's
own kernels in ``ops/attention/kda``.
"""

from typing import Any

import torch
from tokenspeed_kernel._triton import tl, triton

NULL_BLOCK_ID = 0


@triton.jit
def _kda_gate(
    raw_g,
    dt_bias,
    A,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
):
    gate_input = raw_g + dt_bias
    if USE_LOWER_BOUND:
        return lower_bound * tl.sigmoid(A * gate_input)
    softplus_gate = tl.where(
        gate_input > 20.0,
        gate_input,
        tl.log(1.0 + tl.exp(gate_input)),
    )
    return -A * softplus_gate


@triton.jit
def _kda_recurrent_step(
    state,
    k,
    v,
    raw_g,
    raw_beta,
    dt_bias,
    A,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
):
    normalized_k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
    gate = _kda_gate(
        raw_g,
        dt_bias,
        A,
        lower_bound,
        USE_LOWER_BOUND,
    )

    decay = tl.exp(gate)
    state *= decay[None, :]
    correction = v - tl.sum(state * normalized_k[None, :], axis=1)
    correction *= tl.sigmoid(raw_beta)
    return (
        state + correction[:, None] * normalized_k[None, :],
        correction,
        normalized_k,
        decay,
    )


@triton.heuristics(
    {
        "BV": lambda args: min(
            triton.next_power_of_2(args["V"]),
            4 if args["BATCH"] <= 4 else 8 if args["BATCH"] <= 8 else 16,
        )
    }
)
@triton.jit(
    do_not_specialize=[
        "RECORD_SLOTS",
        "stride_history_length",
        "stride_live_indices",
        "stride_record_slots",
        "payload_slots",
        "decay_strides",
        "pages",
    ]
)
def _kda_recoverssm_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    raw_g_ptr,
    raw_beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    correction_cache_ptr,
    key_cache_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    record_metadata_ptr,
    record_stamps_ptr,
    history_length_ptr,
    live_indices_ptr,
    slots_ptr,
    decay_cache_ptr,
    stride_history_length,
    stride_live_indices,
    stride_record_slots,
    payload_slots,
    decay_strides,
    HISTORY_CAPACITY: tl.constexpr,
    RECORD_CAPACITY: tl.constexpr,
    RECORD_SLOTS,
    lower_bound,
    null_block_id,
    pages,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_g_token,
    stride_beta_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_block,
    stride_correction_head,
    stride_correction_pos,
    stride_correction_dim,
    stride_key_block,
    stride_key_head,
    stride_key_pos,
    stride_key_dim,
    stride_out_token,
    stride_query_start_loc,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    BATCH: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)
    pid_h = tl.program_id(2)

    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    valid_source = (state_idx > null_block_id) & (state_idx < pages)
    state_idx = tl.load(live_indices_ptr + pid_b * stride_live_indices).to(tl.int64)
    payload_slot = tl.load(slots_ptr + pid_b * stride_record_slots).to(tl.int64)
    if (
        not valid_source
        or state_idx <= null_block_id
        or state_idx >= pages
        or payload_slot < 0
        or payload_slot >= payload_slots
    ):
        for token_offset in tl.static_range(SPEC_QUERY_LEN):
            token_valid = token_offset < query_len
            tl.store(
                out_ptr + (bos + token_offset) * stride_out_token + pid_h * V + offs_v,
                tl.zeros([BV], dtype=tl.float32),
                mask=token_valid & mask_v,
            )
        return

    if query_len == 0:
        return

    record_slot = pid_b.to(tl.int64)
    record_start = record_slot * 0
    record_generation = record_start
    if record_metadata_ptr is not None:
        record_slot = tl.load(record_metadata_ptr + pid_b * 3)
        record_generation = tl.load(record_metadata_ptr + pid_b * 3 + 1)
        record_start = tl.load(record_metadata_ptr + pid_b * 3 + 2)
        slot_valid = (record_slot >= 0) & (record_slot < RECORD_SLOTS)
        tl.device_assert(slot_valid, "invalid replay slot")
        if not slot_valid:
            return
        tl.device_assert(
            (record_generation >= 0) & (record_start >= 0), "invalid replay metadata"
        )
    # Commit runs after verify completes; stamps identify the slot generation
    # and absolute token position before the payload can be consumed.
    if record_stamps_ptr is not None:
        if pid_v == 0 and pid_h == 0:
            for token_offset in tl.static_range(SPEC_QUERY_LEN):
                position = record_start + token_offset
                stamp = (
                    record_stamps_ptr
                    + (record_slot * RECORD_CAPACITY + position % RECORD_CAPACITY) * 2
                )
                tl.store(stamp, position, mask=token_offset < query_len)
                tl.store(stamp + 1, record_generation, mask=token_offset < query_len)
    history_length = tl.load(history_length_ptr + payload_slot * stride_history_length)
    valid_history = (history_length >= 0) & (
        history_length + query_len <= HISTORY_CAPACITY
    )
    tl.device_assert(valid_history, "retained candidate window exceeds capacity")
    if not valid_history:
        return
    state_ptrs = (
        state_ptr
        + state_idx * stride_state_block
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    A = tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))
    dt_bias = tl.load(
        dt_bias_ptr + pid_h * K + offs_k,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    state = tl.load(state_ptrs, mask=mask_state, other=0.0).to(tl.float32)

    for position in range(history_length):
        u = tl.load(
            correction_cache_ptr
            + payload_slot * stride_correction_block
            + pid_h * stride_correction_head
            + position * stride_correction_pos
            + offs_v * stride_correction_dim,
            mask=mask_v,
            other=0.0,
        )
        key = tl.load(
            key_cache_ptr
            + payload_slot * stride_key_block
            + pid_h * stride_key_head
            + position * stride_key_pos
            + offs_k * stride_key_dim,
            mask=mask_k,
            other=0.0,
        )
        decay = tl.load(
            decay_cache_ptr
            + payload_slot * decay_strides[0]
            + pid_h * decay_strides[1]
            + position * decay_strides[2]
            + offs_k * decay_strides[3],
            mask=mask_k,
            other=0.0,
        )
        state = state * decay[None, :] + u[:, None] * key[None, :]

    # Load the candidate window before the recurrence stores its outputs and
    # replay records. This lets independent token loads overlap instead of
    # waiting behind each preceding token's stores.
    inputs = ()
    for token_offset in tl.static_range(SPEC_QUERY_LEN):
        token_valid = token_offset < query_len
        token = bos + token_offset
        q = tl.load(
            q_ptr + token * stride_q_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        k = tl.load(
            k_ptr + token * stride_k_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        v = tl.load(
            v_ptr + token * stride_v_token + pid_h * V + offs_v,
            mask=token_valid & mask_v,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        raw_g = tl.load(
            raw_g_ptr + token * stride_g_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        raw_beta = tl.load(
            raw_beta_ptr + token * stride_beta_token + pid_h,
            mask=token_valid,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        inputs += ((q, k, v, raw_g, raw_beta),)

    for token_offset in tl.static_range(SPEC_QUERY_LEN):
        token_valid = token_offset < query_len
        token = bos + token_offset
        q, k, v, raw_g, raw_beta = inputs[token_offset]
        q *= tl.rsqrt(tl.sum(q * q) + 1e-6) * (K**-0.5)
        updated_state, correction, normalized_k, decay = _kda_recurrent_step(
            state,
            k,
            v,
            raw_g,
            raw_beta,
            dt_bias,
            A,
            lower_bound,
            USE_LOWER_BOUND,
        )
        if SPEC_QUERY_LEN == 4 and K == 128 and V == 128 and BV == 16:
            # Valid tokens form a prefix, and verify never publishes this state.
            # Masked suffix tokens cannot write outputs or records, so keeping
            # the old state live only increases register pressure. Other tiles
            # retain their measured predicated-update schedule.
            state = updated_state
        else:
            state = tl.where(token_valid, updated_state, state)

        out = tl.sum(state * q[None, :], axis=1)
        tl.store(
            out_ptr + token * stride_out_token + pid_h * V + offs_v,
            out,
            mask=token_valid & mask_v,
            eviction_policy="evict_first",
        )

        payload_position = history_length + token_offset
        correction_ptr = (
            correction_cache_ptr
            + payload_slot * stride_correction_block
            + pid_h * stride_correction_head
            + payload_position * stride_correction_pos
        )
        tl.store(
            correction_ptr + offs_v * stride_correction_dim,
            correction,
            mask=token_valid & mask_v,
        )
        if pid_v == 0:
            key_ptr = (
                key_cache_ptr
                + payload_slot * stride_key_block
                + pid_h * stride_key_head
                + payload_position * stride_key_pos
            )
            tl.store(
                key_ptr + offs_k * stride_key_dim,
                normalized_k,
                mask=token_valid & mask_k,
            )
            tl.store(
                decay_cache_ptr
                + payload_slot * decay_strides[0]
                + pid_h * decay_strides[1]
                + payload_position * decay_strides[2]
                + offs_k * decay_strides[3],
                decay,
                mask=token_valid & mask_k,
            )


def kda_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    checkpoint_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    out: torch.Tensor,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
    retained_history: Any,
) -> torch.Tensor:
    """Verify a KDA speculative window without modifying its checkpoint."""
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("KDA RecoverSSM q must have shape [1, tokens, heads, dim]")
    _, total_tokens, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if k.shape != q.shape or v.shape != (1, total_tokens, num_heads, value_dim):
        raise ValueError("KDA RecoverSSM q, k, and v shapes are incompatible")
    if raw_g.shape != q.shape or raw_beta.shape != (1, total_tokens, num_heads):
        raise ValueError("KDA RecoverSSM gate or beta shape is incompatible")
    if any((tensor.stride()[2:] != (key_dim, 1) for tensor in (q, k, raw_g))):
        raise ValueError("KDA RecoverSSM q, k, and gate heads must be contiguous")
    if v.stride()[2:] != (value_dim, 1) or raw_beta.stride(2) != 1:
        raise ValueError("KDA RecoverSSM v and beta heads must be contiguous")
    if checkpoint_state.shape[1:] != (num_heads, value_dim, key_dim):
        raise ValueError("KDA RecoverSSM checkpoint shape is incompatible")
    correction_cache = retained_history.correction
    key_cache = retained_history.key
    if A_log.shape != (num_heads,) or dt_bias.numel() != num_heads * key_dim:
        raise ValueError("KDA RecoverSSM gate parameters are incompatible")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("KDA RecoverSSM gate parameters must be contiguous")
    batch = state_indices.shape[0]
    capacity = correction_cache.shape[2]
    if total_tokens > batch * spec_query_len:
        raise ValueError(
            "KDA RecoverSSM speculative decode input exceeds its activation capacity"
        )
    if out.shape != v.shape:
        raise ValueError("KDA RecoverSSM output shape is incompatible")
    if out.stride()[2:] != (value_dim, 1):
        raise ValueError("KDA RecoverSSM output heads must be contiguous")
    device = q.device
    input_tensors = (k, v, raw_g, raw_beta, A_log, dt_bias, checkpoint_state, out)
    if any(tensor.device != device for tensor in input_tensors):
        raise ValueError("KDA RecoverSSM inputs must be on the same device")
    if total_tokens == 0:
        return out
    block_k = triton.next_power_of_2(key_dim)
    grid = lambda meta: (triton.cdiv(value_dim, meta["BV"]), batch, num_heads * 1)
    value_block = min(
        triton.next_power_of_2(value_dim),
        4 if batch <= 4 else 8 if batch <= 8 else 16,
    )
    stamp_capacity = record_stamps.shape[1] if record_stamps is not None else capacity
    stamp_slots = (
        record_stamps.shape[0]
        if record_stamps is not None
        else correction_cache.shape[0]
    )
    _kda_recoverssm_verify_kernel[grid](
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        checkpoint_state,
        correction_cache,
        key_cache,
        out,
        query_start_loc,
        state_indices,
        record_metadata,
        record_stamps,
        retained_history.length,
        retained_history.live_state_indices,
        retained_history.record_slots,
        retained_history.decay,
        retained_history.length.stride(0),
        retained_history.live_state_indices.stride(0),
        retained_history.record_slots.stride(0),
        retained_history.correction.shape[0],
        retained_history.decay.stride(),
        capacity,
        stamp_capacity,
        stamp_slots,
        lower_bound,
        NULL_BLOCK_ID,
        checkpoint_state.shape[0],
        q.stride(1),
        k.stride(1),
        v.stride(1),
        raw_g.stride(1),
        raw_beta.stride(1),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        checkpoint_state.stride(3),
        correction_cache.stride(0),
        correction_cache.stride(1),
        correction_cache.stride(2),
        correction_cache.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        out.stride(1),
        query_start_loc.stride(0),
        state_indices.stride(0),
        K=key_dim,
        V=value_dim,
        BK=block_k,
        SPEC_QUERY_LEN=spec_query_len,
        USE_LOWER_BOUND=lower_bound is not None,
        BATCH=triton.next_power_of_2(batch),
        num_warps=1,
        debug=True,
        sanitize_overflow=False,
        # The four-token, 16-row tile no longer keeps the old state live, so its
        # compiler-selected registers avoid spills. Other shapes retain the
        # measured cap after preloading their inputs.
        maxnreg=(
            None
            if batch > 8 and spec_query_len == 4 and key_dim == value_dim == 128
            else 168
        ),
        num_stages=2,
    )
    return out
