# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 RecoverSSM kernels with stamped raw and retained KDA records.

Based on vLLM c77b3faa1cc2c39edbe82365f28ce96981af1438. Legacy raw records
keep FP32 U and activation-dtype K/raw gate in request slots; commit recovers
dense state with CUDA cores or TF32x3 Tensor Cores. Retained records keep FP32
U, normalized K, and multiplicative decay in cache pages. Ordered consolidation,
reset, and verify prepare private live pages; commit validates candidate stamps
and publishes only accepted length. Verify retains the upstream recurrence,
static token loop, and one-warp launch.
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


@triton.jit(
    do_not_specialize=[
        "state_strides",
        "correction_strides",
        "key_strides",
        "decay_strides",
        "length_stride",
        "source_stride",
        "live_stride",
        "query_stride",
    ]
)
def _consolidate_kda_history_kernel(
    state_ptr,
    correction_ptr,
    key_ptr,
    decay_ptr,
    length_ptr,
    source_ptr,
    live_ptr,
    query_ptr,
    state_strides,
    correction_strides,
    key_strides,
    decay_strides,
    length_stride,
    source_stride,
    live_stride,
    query_stride,
    CAPACITY: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    row = tl.program_id(1)
    head = tl.program_id(2)
    source = tl.load(source_ptr + row * source_stride).to(tl.int64)
    live = tl.load(live_ptr + row * live_stride).to(tl.int64)
    begin = tl.load(query_ptr + row * query_stride).to(tl.int64)
    end = tl.load(query_ptr + (row + 1) * query_stride).to(tl.int64)
    tokens = end - begin
    if source <= 0 or live <= 0 or tokens == 0:
        return
    length = tl.load(length_ptr + source * length_stride)
    valid = (length >= 0) & (length <= CAPACITY) & (tokens > 0) & (tokens <= CAPACITY)
    tl.device_assert(valid, "invalid retained history interval")
    if not valid:
        return
    if source == live and length + tokens <= CAPACITY:
        return
    keys = tl.arange(0, BK)
    values = tl.program_id(0) * BV + tl.arange(0, BV)
    mask = (values[:, None] < V) & (keys[None, :] < K)
    tile = (
        head * state_strides[1]
        + values[:, None] * state_strides[2]
        + keys[None, :] * state_strides[3]
    )
    state = tl.load(
        state_ptr + source * state_strides[0] + tile, mask=mask, other=0.0
    ).to(tl.float32)
    for position in range(length):
        u = tl.load(
            correction_ptr
            + source * correction_strides[0]
            + head * correction_strides[1]
            + position * correction_strides[2]
            + values * correction_strides[3],
            mask=values < V,
            other=0.0,
        )
        key = tl.load(
            key_ptr
            + source * key_strides[0]
            + head * key_strides[1]
            + position * key_strides[2]
            + keys * key_strides[3],
            mask=keys < K,
            other=0.0,
        )
        decay = tl.load(
            decay_ptr
            + source * decay_strides[0]
            + head * decay_strides[1]
            + position * decay_strides[2]
            + keys * decay_strides[3],
            mask=keys < K,
            other=0.0,
        )
        state = state * decay[None, :] + u[:, None] * key[None, :]
    # Each CTA owns its V tile. Length and records remain untouched until all
    # these stores finish; a separate ordered kernel resets the live length.
    tl.store(state_ptr + live * state_strides[0] + tile, state, mask=mask)


@triton.jit(
    do_not_specialize=["length_stride", "source_stride", "live_stride", "query_stride"]
)
def _reset_kda_history_kernel(
    length_ptr,
    source_ptr,
    live_ptr,
    query_ptr,
    length_stride,
    source_stride,
    live_stride,
    query_stride,
    CAPACITY: tl.constexpr,
):
    row = tl.program_id(0)
    source = tl.load(source_ptr + row * source_stride).to(tl.int64)
    live = tl.load(live_ptr + row * live_stride).to(tl.int64)
    begin = tl.load(query_ptr + row * query_stride).to(tl.int64)
    end = tl.load(query_ptr + (row + 1) * query_stride).to(tl.int64)
    tokens = end - begin
    if source <= 0 or live <= 0 or tokens == 0:
        return
    length = tl.load(length_ptr + source * length_stride)
    valid = (length >= 0) & (length <= CAPACITY) & (tokens > 0) & (tokens <= CAPACITY)
    tl.device_assert(valid, "invalid retained history interval")
    if not valid:
        return
    if source != live or length + tokens > CAPACITY:
        tl.store(length_ptr + live * length_stride, 0)


@triton.jit(
    do_not_specialize=[
        "length_stride",
        "source_stride",
        "live_stride",
        "query_stride",
        "accepted_stride",
        "STAMP_SLOTS",
    ]
)
def _publish_kda_history_kernel(
    length_ptr,
    source_ptr,
    live_ptr,
    query_ptr,
    accepted_ptr,
    metadata_ptr,
    stamps_ptr,
    length_stride,
    source_stride,
    live_stride,
    query_stride,
    accepted_stride,
    STAMP_SLOTS,
    STAMP_CAPACITY: tl.constexpr,
    CAPACITY: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
):
    row = tl.program_id(0)
    source = tl.load(source_ptr + row * source_stride).to(tl.int64)
    live = tl.load(live_ptr + row * live_stride).to(tl.int64)
    begin = tl.load(query_ptr + row * query_stride).to(tl.int64)
    end = tl.load(query_ptr + (row + 1) * query_stride).to(tl.int64)
    tokens = end - begin
    if source <= 0 or live <= 0 or tokens == 0:
        return
    accepted = tl.load(accepted_ptr + row * accepted_stride)
    accepted = tl.minimum(tl.minimum(tl.maximum(accepted, 0), tokens), SPEC_QUERY_LEN)
    length = tl.load(length_ptr + live * length_stride)
    valid = (length >= 0) & (length + accepted <= CAPACITY)
    tl.device_assert(valid, "invalid retained publication interval")
    if not valid:
        return
    if metadata_ptr is not None:
        slot = tl.load(metadata_ptr + row * 3)
        generation = tl.load(metadata_ptr + row * 3 + 1)
        start = tl.load(metadata_ptr + row * 3 + 2)
        valid = (
            (slot >= 0)
            & (slot < STAMP_SLOTS)
            & (generation >= 0)
            & (start >= 0)
            & (accepted <= STAMP_CAPACITY)
        )
        tl.device_assert(valid, "invalid retained replay metadata")
        if not valid:
            return
        for offset in range(accepted):
            position = start + offset
            stamp = stamps_ptr + (slot * STAMP_CAPACITY + position % STAMP_CAPACITY) * 2
            valid = (
                valid
                & (tl.load(stamp) == position)
                & (tl.load(stamp + 1) == generation)
            )
        tl.device_assert(valid, "stale retained replay record")
        if not valid:
            return
    tl.store(length_ptr + live * length_stride, (length + accepted).to(tl.int32))


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
        "decay_strides",
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
    kg_cache_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    record_metadata_ptr,
    record_stamps_ptr,
    history_length_ptr,
    live_indices_ptr,
    decay_cache_ptr,
    stride_history_length,
    stride_live_indices,
    decay_strides,
    RETAINED_HISTORY: tl.constexpr,
    HISTORY_CAPACITY: tl.constexpr,
    RECORD_CAPACITY: tl.constexpr,
    RECORD_SLOTS,
    lower_bound,
    null_block_id,
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
    stride_kg_block,
    stride_kg_head,
    stride_kg_pos,
    stride_kg_dim,
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
    pid_b = tl.program_id(1)
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

    valid_source = state_idx > null_block_id
    if RETAINED_HISTORY:
        state_idx = tl.load(live_indices_ptr + pid_b * stride_live_indices).to(tl.int64)
    if not valid_source or state_idx <= null_block_id:
        for token_offset in tl.static_range(SPEC_QUERY_LEN):
            token_valid = token_offset < query_len
            tl.store(
                out_ptr + (bos + token_offset) * stride_out_token + pid_h * V + offs_v,
                tl.zeros([BV], dtype=tl.float32),
                mask=token_valid & mask_v,
            )
        return

    if RETAINED_HISTORY and query_len == 0:
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
        if RETAINED_HISTORY and not slot_valid:
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
    history_length = 0
    if RETAINED_HISTORY:
        history_length = tl.load(history_length_ptr + state_idx * stride_history_length)
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

    if RETAINED_HISTORY:
        for position in range(history_length):
            u = tl.load(
                correction_cache_ptr
                + state_idx * stride_correction_block
                + pid_h * stride_correction_head
                + position * stride_correction_pos
                + offs_v * stride_correction_dim,
                mask=mask_v,
                other=0.0,
            )
            key = tl.load(
                kg_cache_ptr
                + state_idx * stride_kg_block
                + pid_h * stride_kg_head
                + position * stride_kg_pos
                + offs_k * stride_kg_dim,
                mask=mask_k,
                other=0.0,
            )
            decay = tl.load(
                decay_cache_ptr
                + state_idx * decay_strides[0]
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

        payload_slot = record_slot
        payload_position = (record_start + token_offset) % RECORD_CAPACITY
        if RETAINED_HISTORY:
            payload_slot = state_idx
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
            kg_ptr = (
                kg_cache_ptr
                + payload_slot * stride_kg_block
                + pid_h * stride_kg_head
                + payload_position * stride_kg_pos
            )
            if RETAINED_HISTORY:
                tl.store(
                    kg_ptr + offs_k * stride_kg_dim,
                    normalized_k,
                    mask=token_valid & mask_k,
                )
                tl.store(
                    decay_cache_ptr
                    + state_idx * decay_strides[0]
                    + pid_h * decay_strides[1]
                    + payload_position * decay_strides[2]
                    + offs_k * decay_strides[3],
                    decay,
                    mask=token_valid & mask_k,
                )
            else:
                tl.store(kg_ptr + offs_k * stride_kg_dim, k, mask=token_valid & mask_k)
                tl.store(
                    kg_ptr + (K + offs_k) * stride_kg_dim,
                    raw_g,
                    mask=token_valid & mask_k,
                )


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "ALIGN_MODE": lambda args: args["block_table_ptr"] is not None,
    }
)
@triton.jit
def _prepare_commit_plan_kernel(
    num_accepted_ptr,
    request_indices_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    num_computed_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    mamba_block_size,
    block_table_width,
    stride_num_accepted,
    stride_request_indices,
    stride_state_indices,
    stride_query_start_loc,
    stride_block_table_row,
    stride_block_table_col,
    stride_num_computed,
    SPEC_QUERY_LEN: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    spec_idx = tl.program_id(0)
    source_state_idx = tl.load(state_indices_ptr + spec_idx * stride_state_indices).to(
        tl.int64
    )
    request_idx = spec_idx
    if HAS_REQUEST_INDICES:
        request_idx = tl.load(
            request_indices_ptr + spec_idx * stride_request_indices
        ).to(tl.int64)
    num_accepted = tl.load(num_accepted_ptr + request_idx * stride_num_accepted).to(
        tl.int32
    )
    bos = tl.load(query_start_loc_ptr + spec_idx * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (spec_idx + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = (eos - bos).to(tl.int32)
    commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)
    final_state_idx = source_state_idx
    boundary_state_idx = null_block_id
    boundary_recovery_len = 0
    if ALIGN_MODE:
        num_computed = tl.load(num_computed_ptr + request_idx * stride_num_computed).to(
            tl.int32
        )
        final_num_computed = num_computed + commit_len
        final_state_col = tl.minimum(
            final_num_computed // mamba_block_size, block_table_width - 1
        )
        final_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + final_state_col * stride_block_table_col
        ).to(tl.int64)
        next_boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
        crosses_boundary = final_num_computed >= next_boundary
        boundary_recovery_len = next_boundary - num_computed
        boundary_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + (next_boundary // mamba_block_size - 1) * stride_block_table_col,
            mask=crosses_boundary,
            other=null_block_id,
        ).to(tl.int64)
    valid = (source_state_idx > null_block_id) & (commit_len > 0)
    tl.store(commit_lens_ptr + spec_idx, tl.where(valid, commit_len, 0))
    tl.store(
        final_state_indices_ptr + spec_idx,
        tl.where(valid, final_state_idx, null_block_id),
    )
    tl.store(
        boundary_state_indices_ptr + spec_idx,
        tl.where(valid, boundary_state_idx, null_block_id),
    )
    tl.store(
        boundary_recovery_lens_ptr + spec_idx, tl.where(valid, boundary_recovery_len, 0)
    )


def _commit_kda_value_block(args: dict[str, Any]) -> int:
    if args["USE_TENSOR_CORES"] and not args["ALIGN_MODE"]:
        return 64 if args["BT"] <= 8 else 128
    batch = args["BATCH"]
    if not args["ALIGN_MODE"]:
        return 4 if batch <= 2 else 16
    if batch == 2:
        return 16
    return 32 if batch == 4 else 8


def _commit_kda_num_warps(args: dict[str, Any]) -> int:
    if args["USE_TENSOR_CORES"] and not args["ALIGN_MODE"]:
        return 2 if args["BT"] <= 8 else 4
    if not args["ALIGN_MODE"]:
        return 1
    return 4 if args["BATCH"] == 1 else 2 if args["BATCH"] == 2 else 1


@triton.jit
def _recoverssm_decay_product(left, right):
    return left * right


@triton.heuristics({"BV": _commit_kda_value_block, "num_warps": _commit_kda_num_warps})
@triton.jit(do_not_specialize=["BATCH", "RECORD_SLOTS"])
def _commit_kda_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_stride: tl.constexpr,
    correction_cache_ref_ptr,
    correction_cache_base_addrs_ptr,
    correction_cache_block_stride: tl.constexpr,
    kg_cache_ref_ptr,
    kg_cache_base_addrs_ptr,
    kg_cache_block_stride: tl.constexpr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    stride_state_head: tl.constexpr,
    stride_state_v: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_correction_cache_head: tl.constexpr,
    stride_correction_cache_pos: tl.constexpr,
    stride_correction_cache_dim: tl.constexpr,
    stride_kg_cache_head: tl.constexpr,
    stride_kg_cache_pos: tl.constexpr,
    stride_kg_cache_dim: tl.constexpr,
    stride_state_indices: tl.constexpr,
    group_indices_ptr,
    stride_state_group,
    stride_pointer_layer: tl.constexpr,
    record_metadata_ptr,
    record_stamps_ptr,
    stamp_layer_stride: tl.constexpr,
    RECORD_CAPACITY: tl.constexpr,
    RECORD_SLOTS,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    USE_TENSOR_CORES: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    BATCH,
    LAUNCH_DEPENDENT_KERNELS: tl.constexpr,
    a_ptr,
    bias_ptr,
    USE_GATE_BASE_ADDRESSES: tl.constexpr,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
):
    if LAUNCH_DEPENDENT_KERNELS:
        tl.extra.cuda.gdc_launch_dependents()
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    pid_l = pid_lh // NUM_HEADS
    pid_h = pid_lh % NUM_HEADS
    group = 0
    if group_indices_ptr is not None:
        group = tl.load(group_indices_ptr + pid_l).to(tl.int64)
    plan_row = group * BATCH + pid_b
    source_state_idx = tl.load(
        state_indices_ptr + group * stride_state_group + pid_b * stride_state_indices
    ).to(tl.int64)
    if source_state_idx <= null_block_id:
        return
    commit_len = tl.load(commit_lens_ptr + pid_b)
    final_state_idx = tl.load(final_state_indices_ptr + plan_row).to(tl.int64)
    if ALIGN_MODE:
        boundary_state_idx = tl.load(boundary_state_indices_ptr + plan_row).to(tl.int64)
        boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + plan_row)
    if final_state_idx <= null_block_id:
        return
    record_slot = pid_b
    record_start = 0
    if record_metadata_ptr is not None:
        record_slot = tl.load(record_metadata_ptr + pid_b * 3)
        generation = tl.load(record_metadata_ptr + pid_b * 3 + 1)
        record_start = tl.load(record_metadata_ptr + pid_b * 3 + 2)
        slot_valid = (record_slot >= 0) & (record_slot < RECORD_SLOTS)
        interval_valid = (
            (commit_len >= 0)
            & (commit_len <= RECORD_CAPACITY)
            & (record_start >= 0)
            & (generation >= 0)
        )
        tl.device_assert(slot_valid & interval_valid, "invalid replay interval")
        if not slot_valid & interval_valid:
            return
        offsets = tl.arange(0, triton.next_power_of_2(RECORD_CAPACITY))
        positions = record_start + offsets
        stamps = (
            record_stamps_ptr
            + pid_l * stamp_layer_stride
            + (record_slot * RECORD_CAPACITY + positions % RECORD_CAPACITY) * 2
        )
        stamped_position = tl.load(stamps, mask=offsets < commit_len, other=-1)
        stamped_generation = tl.load(stamps + 1, mask=offsets < commit_len, other=-1)
        valid = (
            tl.sum(
                (
                    (offsets < commit_len)
                    & (
                        (stamped_position != positions)
                        | (stamped_generation != generation)
                    )
                ).to(tl.int32),
                axis=0,
            )
            == 0
        )
        tl.device_assert(valid, "stale replay record")
        if not valid:
            return
    state_base_addr = tl.load(state_base_addrs_ptr + pid_l * stride_pointer_layer)
    state_ptr = state_base_addr.to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    source_state_ptr = (
        state_ptr + source_state_idx * state_block_stride + pid_h * stride_state_head
    )
    correction_cache_base_addr = tl.load(
        correction_cache_base_addrs_ptr + pid_l * stride_pointer_layer
    )
    correction_cache_ptr = correction_cache_base_addr.to(
        tl.pointer_type(correction_cache_ref_ptr.dtype.element_ty)
    )
    correction_cache_ptr += (
        record_slot * correction_cache_block_stride
        + pid_h * stride_correction_cache_head
    )
    kg_cache_base_addr = tl.load(kg_cache_base_addrs_ptr + pid_l * stride_pointer_layer)
    kg_cache_ptr = kg_cache_base_addr.to(
        tl.pointer_type(kg_cache_ref_ptr.dtype.element_ty)
    )
    kg_cache_ptr += record_slot * kg_cache_block_stride + pid_h * stride_kg_cache_head
    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    state_ptrs = (
        source_state_ptr
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    if ALIGN_MODE or not USE_TENSOR_CORES:
        initial_state = tl.load(state_ptrs, mask=mask_state, other=0.0).to(tl.float32)
    # Batched recovery reads per-layer gate addresses from the descriptors.
    # Single-layer recovery passes the gate tensors directly.
    if USE_GATE_BASE_ADDRESSES:
        a_ptr = tl.load(a_ptr + pid_l * stride_pointer_layer).to(
            tl.pointer_type(tl.float32)
        )
        bias_ptr = tl.load(bias_ptr + pid_l * stride_pointer_layer).to(
            tl.pointer_type(tl.float32)
        )
    A = tl.exp(tl.load(a_ptr + pid_h))
    dt_bias = tl.load(bias_ptr + pid_h * K + offs_k, mask=mask_k, other=0.0)
    if ALIGN_MODE:
        boundary_ptrs = (
            state_ptr
            + boundary_state_idx * state_block_stride
            + pid_h * stride_state_head
            + offs_v[:, None] * stride_state_v
            + offs_k[None, :] * stride_state_k
        )
    # Aligned checkpoints use recurrence to write boundary and final states
    # in one pass. Tensor Cores only recover the final state.
    if USE_TENSOR_CORES and not ALIGN_MODE:
        # Recover S_n = S_0 * product(d_j) + U @ K_weighted. Each key row
        # includes only the decays after its update. The reverse product scan
        # avoids dividing by prefix products, which can underflow for long windows.
        # Zero padding supplies the Tensor Core reduction tile for short windows.
        tokens = tl.arange(0, BT)
        live = tokens < commit_len
        slots = (record_start + tokens) % RECORD_CAPACITY
        key_ptrs = (
            kg_cache_ptr
            + slots[:, None] * stride_kg_cache_pos
            + offs_k[None, :] * stride_kg_cache_dim
        )
        keys = tl.load(key_ptrs, mask=live[:, None] & mask_k[None, :], other=0.0).to(
            tl.float32
        )
        keys *= tl.rsqrt(tl.sum(keys * keys, axis=1) + 1e-06)[:, None]
        raw_g = tl.load(
            key_ptrs + K * stride_kg_cache_dim,
            mask=live[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        gates = _kda_gate(raw_g, dt_bias[None, :], A, lower_bound, USE_LOWER_BOUND)
        decay = tl.where(live[:, None], tl.exp(gates), 1.0)
        next_row = tl.broadcast_to(tl.minimum(tokens + 1, BT - 1)[:, None], (BT, BK))
        decay_after = tl.gather(decay, next_row, axis=0)
        decay_after = tl.where((tokens + 1 < commit_len)[:, None], decay_after, 1.0)
        suffix = tl.associative_scan(
            decay_after, axis=0, combine_fn=_recoverssm_decay_product, reverse=True
        )
        weighted_keys = keys * suffix
        total_decay = tl.sum(
            tl.where(tokens[:, None] == 0, suffix * decay, 0.0), axis=0
        )
        corrections = tl.load(
            correction_cache_ptr
            + slots[None, :] * stride_correction_cache_pos
            + offs_v[:, None] * stride_correction_cache_dim,
            mask=mask_v[:, None] & live[None, :],
            other=0.0,
        ).to(tl.float32)
        # U and weighted K stay FP32. TF32x3 uses Tensor Cores with FP32
        # accumulation; the checkpoint is scaled and added after the product.
        update = tl.dot(corrections, weighted_keys, input_precision="tf32x3")
        initial_state = tl.load(state_ptrs, mask=mask_state, other=0.0).to(tl.float32)
        state = tl.fma(initial_state, total_decay[None, :], update)
    else:
        state = initial_state
        for token_offset in range(commit_len):
            correction_ptr = (
                correction_cache_ptr
                + (record_start + token_offset)
                % RECORD_CAPACITY
                * stride_correction_cache_pos
            )
            kg_ptr = (
                kg_cache_ptr
                + (record_start + token_offset) % RECORD_CAPACITY * stride_kg_cache_pos
            )
            k = tl.load(
                kg_ptr + offs_k * stride_kg_cache_dim, mask=mask_k, other=0.0
            ).to(tl.float32)
            correction = tl.load(
                correction_ptr + offs_v * stride_correction_cache_dim,
                mask=mask_v,
                other=0.0,
            ).to(tl.float32)
            raw_g = tl.load(
                kg_ptr + (K + offs_k) * stride_kg_cache_dim, mask=mask_k, other=0.0
            ).to(tl.float32)
            normalized_k = k * tl.rsqrt(tl.sum(k * k) + 1e-06)
            gate = _kda_gate(raw_g, dt_bias, A, lower_bound, USE_LOWER_BOUND)
            decay = tl.exp(gate)
            update = correction[:, None] * normalized_k[None, :]
            state = state * decay[None, :] + update
            if ALIGN_MODE:
                if token_offset == boundary_recovery_len - 1:
                    tl.store(
                        boundary_ptrs,
                        state,
                        mask=mask_state & (boundary_state_idx > null_block_id),
                    )
    final_ptrs = (
        state_ptr
        + final_state_idx * state_block_stride
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    tl.store(final_ptrs, state, mask=mask_state)


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
    correction_cache: torch.Tensor | None,
    kg_cache: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    out: torch.Tensor,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
    retained_history: Any | None,
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
    retained = retained_history is not None
    if retained:
        correction_cache = retained_history.correction
        kg_cache = retained_history.key
    else:
        if correction_cache is None or kg_cache is None:
            raise ValueError("raw KDA recovery requires correction and key/gate caches")
        expected_correction_shape = (
            correction_cache.shape[0],
            num_heads,
            correction_cache.shape[2],
            value_dim,
        )
        if correction_cache.shape != expected_correction_shape:
            raise ValueError(
                f"KDA RecoverSSM correction buffer needs shape {expected_correction_shape}"
            )
        expected_kg_shape = (
            correction_cache.shape[0],
            num_heads,
            correction_cache.shape[2],
            2 * key_dim,
        )
        if kg_cache.shape != expected_kg_shape:
            raise ValueError(
                f"KDA RecoverSSM key/gate buffer needs shape {expected_kg_shape}"
            )
        if correction_cache.dtype != torch.float32:
            raise ValueError("KDA RecoverSSM correction buffer must use float32")
        if kg_cache.dtype != k.dtype:
            raise ValueError("KDA RecoverSSM record dtype does not match its format")
    if A_log.shape != (num_heads,) or dt_bias.numel() != num_heads * key_dim:
        raise ValueError("KDA RecoverSSM gate parameters are incompatible")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("KDA RecoverSSM gate parameters must be contiguous")
    batch = state_indices.shape[0]
    capacity = correction_cache.shape[2]
    if not retained and capacity < spec_query_len:
        raise ValueError("replay capacity must cover the candidate window")
    if not retained:
        if (record_metadata is None) != (record_stamps is None):
            raise ValueError("record metadata and stamps must be supplied together")
        if record_metadata is not None:
            if record_metadata.shape != (batch, 3) or record_stamps.shape != (
                correction_cache.shape[0],
                capacity,
                2,
            ):
                raise ValueError(
                    "replay metadata needs [batch, 3] and stamps [slots, capacity, 2]"
                )
            if any(
                (
                    x.dtype != torch.int64
                    or not x.is_contiguous()
                    or x.device != q.device
                    for x in (record_metadata, record_stamps)
                )
            ):
                raise ValueError(
                    "replay metadata and stamps must be contiguous INT64 on the input device"
                )
        if correction_cache.shape[0] < batch:
            raise ValueError("record capacity is smaller than the runtime batch")
    if not retained and query_start_loc.shape[0] != batch + 1:
        raise ValueError("KDA RecoverSSM query metadata is incompatible")
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
    if not retained:
        input_tensors += (
            correction_cache,
            kg_cache,
            query_start_loc,
            state_indices,
        )
    if any(tensor.device != device for tensor in input_tensors):
        raise ValueError("KDA RecoverSSM inputs must be on the same device")
    if total_tokens == 0:
        return out
    block_k = triton.next_power_of_2(key_dim)
    grid = lambda meta: (triton.cdiv(value_dim, meta["BV"]), batch, num_heads * 1)
    if retained:
        value_block = min(
            triton.next_power_of_2(value_dim),
            4 if batch <= 4 else 8 if batch <= 8 else 16,
        )
        _consolidate_kda_history_kernel[
            (triton.cdiv(value_dim, value_block), batch, num_heads)
        ](
            checkpoint_state,
            correction_cache,
            kg_cache,
            retained_history.decay,
            retained_history.length,
            state_indices,
            retained_history.live_state_indices,
            query_start_loc,
            checkpoint_state.stride(),
            correction_cache.stride(),
            kg_cache.stride(),
            retained_history.decay.stride(),
            retained_history.length.stride(0),
            state_indices.stride(0),
            retained_history.live_state_indices.stride(0),
            query_start_loc.stride(0),
            CAPACITY=capacity,
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=value_block,
            num_warps=1,
            debug=True,
        )
        _reset_kda_history_kernel[batch,](
            retained_history.length,
            state_indices,
            retained_history.live_state_indices,
            query_start_loc,
            retained_history.length.stride(0),
            state_indices.stride(0),
            retained_history.live_state_indices.stride(0),
            query_start_loc.stride(0),
            CAPACITY=capacity,
            num_warps=1,
            debug=True,
        )
    stamp_capacity = (
        record_stamps.shape[1] if retained and record_stamps is not None else capacity
    )
    stamp_slots = (
        record_stamps.shape[0]
        if retained and record_stamps is not None
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
        kg_cache,
        out,
        query_start_loc,
        state_indices,
        record_metadata,
        record_stamps,
        retained_history.length if retained else None,
        retained_history.live_state_indices if retained else None,
        retained_history.decay if retained else None,
        retained_history.length.stride(0) if retained else 0,
        retained_history.live_state_indices.stride(0) if retained else 0,
        retained_history.decay.stride() if retained else (0, 0, 0, 0),
        retained,
        capacity if retained else 0,
        stamp_capacity,
        stamp_slots,
        lower_bound,
        NULL_BLOCK_ID,
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
        kg_cache.stride(0),
        kg_cache.stride(1),
        kg_cache.stride(2),
        kg_cache.stride(3),
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
        debug=retained,
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
