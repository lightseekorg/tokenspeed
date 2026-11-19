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

"""GFX950 Gluon KDA decode kernels using V-major recurrent state."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops._kda_records import (
    kda_records_launch_metadata,
    kda_retained_panel,
    launch_kda_retained_commit,
    prepare_kda_retained,
    validate_kda_retained_verify,
)

cdna4 = gl.amd.cdna4

# MI355 gate projection crosses over between 64 and 128 rows at T1 and T4.
_INLINE_GATE_MAX_ROWS = 64


def _kda_value_splits(batch: int) -> int:
    return max(1, min(8, 16 // triton.next_power_of_2(batch)))


@gluon.jit
def _kda_conv_step(
    history0, history1, history2, current, weight0, weight1, weight2, weight3
):
    """Apply one four-tap depthwise convolution step and SiLU."""
    value = (
        history0 * weight0 + history1 * weight1 + history2 * weight2 + current * weight3
    ).to(gl.float32)
    return value * (1.0 / (1.0 + gl.exp(-value)))


@gluon.jit
def _kda_decay(raw_gate, dt_bias, a_value, HAS_LOWER_BOUND: gl.constexpr, LOWER_BOUND):
    """Convert the projected raw gate to the recurrent multiplicative decay."""
    gate = raw_gate + dt_bias
    if HAS_LOWER_BOUND:
        log_decay = LOWER_BOUND / (1.0 + gl.exp(-(a_value * gate)))
    else:
        softplus = gl.maximum(gate, 0.0) + gl.log(1.0 + gl.exp(-gl.abs(gate)))
        log_decay = -a_value * softplus
    return gl.exp(log_decay)


@gluon.jit
def _kda_recurrent_step(running, key, query, value, beta, key_query):
    """Apply one delta-rule update and return the updated state and output."""
    prediction = gl.sum(running * key[None, :], axis=1)
    prior_output = gl.sum(running * query[None, :], axis=1)
    delta = beta * (value - prediction)
    running += delta[:, None] * key[None, :]
    return running, prior_output + delta * key_query, delta


@gluon.jit
def _kda_qkvd_indices(head_idx, H: gl.constexpr, D: gl.constexpr):
    """Map Q/K/V/decay lanes to their input channels."""
    layout: gl.constexpr = gl.BlockedLayout([2], [64], [4], [0])
    offsets = gl.arange(0, 4 * D, layout=layout)
    slot_id = offsets // D
    local_offset = offsets - slot_id * D
    projection_width: gl.constexpr = H * D
    qkv_channel = (
        gl.where(
            slot_id == 1,
            projection_width,
            gl.where(slot_id == 2, 2 * projection_width, 0),
        )
        + head_idx * D
        + local_offset
    )
    return qkv_channel, slot_id, local_offset, slot_id == 3


@gluon.jit
def _kda_value_panels(vectors, D: gl.constexpr, VALUE_LAYOUT: gl.constexpr):
    """Load the eight 16-wide value panels from a packed Q/K/V/decay row."""
    return (
        vectors.slice(2 * D + 0, 16, dim=0).load(VALUE_LAYOUT),
        vectors.slice(2 * D + 16, 16, dim=0).load(VALUE_LAYOUT),
        vectors.slice(2 * D + 32, 16, dim=0).load(VALUE_LAYOUT),
        vectors.slice(2 * D + 48, 16, dim=0).load(VALUE_LAYOUT),
        vectors.slice(2 * D + 64, 16, dim=0).load(VALUE_LAYOUT),
        vectors.slice(2 * D + 80, 16, dim=0).load(VALUE_LAYOUT),
        vectors.slice(2 * D + 96, 16, dim=0).load(VALUE_LAYOUT),
        vectors.slice(2 * D + 112, 16, dim=0).load(VALUE_LAYOUT),
    )


@gluon.jit(
    launch_metadata=kda_records_launch_metadata,
    do_not_specialize=[
        "MIXED_ROW_STRIDE",
        "ORIGINAL_QKV_ROW_STRIDE",
        "CONV_POOL_PAGE_STRIDE",
        "GATE_ROW_STRIDE",
        "F_A_ROW_STRIDE",
        "BETA_ROW_STRIDE",
        "STATE_POOL_PAGE_STRIDE",
        "NUM_POOL_SLOTS",
        "LOWER_BOUND",
        "u_slot_stride",
        "u_head_stride",
        "u_pos_stride",
        "kg_slot_stride",
        "kg_head_stride",
        "kg_pos_stride",
        "read_stride",
        "query_stride",
        "record_slots",
        "stamp_slots",
        "u_channel_stride",
        "k_channel_stride",
        "d_page_stride",
        "d_head_stride",
        "d_pos_stride",
        "d_channel_stride",
        "length_stride",
        "live_stride",
    ],
)
def gluon_kda_fused_paged_verify_nostore_vmajor_gfx950(
    mixed_qkv: tl.const,
    original_qkv,
    conv_weights: tl.const,
    conv_pool: tl.const,
    raw_g: tl.const,
    f_a_out: tl.const,
    f_b_weight: tl.const,
    beta_logits: tl.const,
    state_pool: tl.const,
    read_indices: tl.const,
    output,
    correction,
    metadata,
    stamps,
    query_start_loc,
    retained_key,
    retained_decay,
    retained_length,
    live_indices,
    u_channel_stride,
    k_channel_stride,
    d_page_stride,
    d_head_stride,
    d_pos_stride,
    d_channel_stride,
    length_stride,
    live_stride,
    u_slot_stride,
    u_head_stride,
    u_pos_stride,
    kg_slot_stride,
    kg_head_stride,
    kg_pos_stride,
    read_stride,
    query_stride,
    record_slots,
    stamp_slots,
    a_log: tl.const,
    dt_bias: tl.const,
    H: gl.constexpr,
    D: gl.constexpr,
    F_A: gl.constexpr,
    TOKENS_PER_SEQUENCE: gl.constexpr,
    TOKEN_WIDTH: gl.constexpr,
    BATCH_TOKENS: gl.constexpr,
    VALUE_SPLITS: gl.constexpr,
    MIXED_ROW_STRIDE,
    ORIGINAL_QKV_ROW_STRIDE,
    CONV_WEIGHT_ROW_STRIDE: gl.constexpr,
    CONV_WEIGHT_COL_STRIDE: gl.constexpr,
    CONV_POOL_PAGE_STRIDE,
    CONV_POOL_CHANNEL_STRIDE: gl.constexpr,
    CONV_POOL_HISTORY_STRIDE: gl.constexpr,
    GATE_ROW_STRIDE,
    F_A_ROW_STRIDE,
    BETA_ROW_STRIDE,
    CAPACITY: gl.constexpr,
    STAMP_CAPACITY: gl.constexpr,
    STATE_POOL_PAGE_STRIDE,
    NUM_POOL_SLOTS,
    HAS_LOWER_BOUND: gl.constexpr,
    LOWER_BOUND,
):
    """Verify candidates into page-owned U, normalized K and decay history.

    BATCH_TOKENS projects inline gates with one MFMA tile and normalizes Q/K
    across the window. It supports up to 16 tokens.
    """
    head_program_idx = gl.program_id(0)
    head_idx = head_program_idx // VALUE_SPLITS
    value_split_idx = head_program_idx % VALUE_SPLITS
    sequence_idx = gl.program_id(1).to(gl.int64)

    # Physical state tiles are [V, K].
    state_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [4, 16],
        [4, 1],
        [1, 0],
    )
    key_layout: gl.constexpr = gl.SliceLayout(0, state_layout)
    value_layout: gl.constexpr = gl.SliceLayout(1, state_layout)
    key_offsets = gl.arange(0, D, layout=key_layout)
    compact_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    output_offsets = gl.arange(0, D, layout=compact_layout)
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    # The same unswizzled rows, viewed whole for the batched token passes.
    rows_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    shared_vectors = gl.allocate_shared_memory(
        gl.float32, [TOKEN_WIDTH, 4 * D], shared_layout
    )
    if raw_g is None:
        if BATCH_TOKENS:
            # Tokens are the tile's rows and this head's gate channels its
            # columns; rows past the window only feed unused LDS rows.
            gate_layout: gl.constexpr = gl.amd.AMDMFMALayout(
                version=4,
                instr_shape=[16, 16, 32],
                transposed=True,
                warps_per_cta=[1, 4],
            )
            gate_lhs: gl.constexpr = gl.DotOperandLayout(0, gate_layout, k_width=8)
            gate_rhs: gl.constexpr = gl.DotOperandLayout(1, gate_layout, k_width=8)
            rhs_k = gl.arange(0, F_A, layout=gl.SliceLayout(1, gate_rhs))
            rhs_n = gl.arange(0, D, layout=gl.SliceLayout(0, gate_rhs))
            f_b_tile = gl.load(
                f_b_weight + (head_idx * D + rhs_n)[None, :] * F_A + rhs_k[:, None]
            )
            gate_rows = gl.arange(0, D, layout=gl.SliceLayout(0, gate_layout))
            projected_decay = gl.allocate_shared_memory(
                gl.float32, [16, D], shared_layout
            )
        else:
            projection_layout: gl.constexpr = gl.BlockedLayout(
                [1, 8], [16, 4], [4, 1], [1, 0]
            )
            gate_rows = gl.arange(0, D, layout=gl.SliceLayout(1, projection_layout))
            f_a_offsets = gl.arange(0, F_A, layout=gl.SliceLayout(0, projection_layout))
            f_b_tile = gl.load(
                f_b_weight
                + (head_idx * D + gate_rows)[:, None] * F_A
                + f_a_offsets[None, :]
            )
            projected_decay = gl.allocate_shared_memory(
                gl.float32, [TOKENS_PER_SEQUENCE, D], shared_layout
            )
        gate_bias = gl.load(dt_bias + head_idx * D + gate_rows).to(gl.float32)
    token_base = gl.load(query_start_loc + sequence_idx * query_stride).to(gl.int64)
    token_count = (
        gl.load(query_start_loc + (sequence_idx + 1) * query_stride).to(gl.int64)
        - token_base
    )
    output_base = (token_base * H + head_idx) * D

    read_idx = gl.load(read_indices + sequence_idx * read_stride)
    # Issue the record metadata with the read index so the state loads wait on
    # one round trip rather than two.
    stamp_slot = sequence_idx.to(gl.int64)
    generation = stamp_slot * 0
    stamp_start = generation
    if metadata is not None:
        stamp_slot = gl.load(metadata + sequence_idx * 3)
        generation = gl.load(metadata + sequence_idx * 3 + 1)
        stamp_start = gl.load(metadata + sequence_idx * 3 + 2)
    recurrent_idx = gl.load(live_indices + sequence_idx.to(gl.int64) * live_stride)
    slot = recurrent_idx.to(gl.int64)
    start = gl.load(
        retained_length + slot * length_stride,
        mask=(slot > 0) & (slot < NUM_POOL_SLOTS),
        other=-1,
    ).to(gl.int64)
    valid_read = (
        (read_idx > 0)
        & (read_idx < NUM_POOL_SLOTS)
        & (recurrent_idx > 0)
        & (recurrent_idx < NUM_POOL_SLOTS)
    )
    if not valid_read:
        if value_split_idx == 0:
            for token_offset in gl.static_range(TOKENS_PER_SEQUENCE):
                gl.store(
                    output + output_base + token_offset * H * D + output_offsets,
                    0.0,
                    mask=token_offset < token_count,
                )
        return

    valid = (
        (slot >= 0)
        & (slot < record_slots)
        & (stamp_slot >= 0)
        & (stamp_slot < stamp_slots)
        & (stamp_start >= 0)
        & (generation >= 0)
        & (start >= 0)
        & (token_count >= 0)
        & (token_count <= TOKENS_PER_SEQUENCE)
        & (start + token_count <= CAPACITY)
    )
    gl.device_assert(valid, "invalid replay metadata")
    if not valid:
        return
    if token_count == 0:
        return
    # Rows past the window reload the last live row instead of masking the
    # loads: a uniform mask becomes a branch with a memory wait per load. Only
    # masked stores consume those rows.
    last_row = token_base + token_count - 1

    read_page_offset = read_idx.to(gl.int64)
    read_base = recurrent_idx.to(gl.int64) * STATE_POOL_PAGE_STRIDE + head_idx * D * D
    # Widen bounded one-token panels while keeping the same prefetch window.
    panel_rows: gl.constexpr = (
        32 if TOKENS_PER_SEQUENCE == 1 and VALUE_SPLITS == 1 and HAS_LOWER_BOUND else 16
    )
    panel_count: gl.constexpr = D // panel_rows
    panel_value_offsets = gl.arange(0, panel_rows, layout=value_layout)

    panels_per_split: gl.constexpr = panel_count // VALUE_SPLITS

    # Start the first state load before preparing the token vectors.
    off_pipe = ()
    raw_pipe = ()
    if VALUE_SPLITS == 1:
        _off = panel_value_offsets[:, None] * D + key_offsets[None, :]
        _raw = gl.load(state_pool + read_base + _off, cache_modifier=".cg").to(
            gl.float32
        )
        off_pipe = (_off,)
        raw_pipe = (_raw,)
    else:
        first_offsets = (
            panel_value_offsets[:, None]
            + value_split_idx * panels_per_split * panel_rows
        ) * D + key_offsets[None, :]
        first_raw = gl.load(
            state_pool + read_base + first_offsets, cache_modifier=".cg"
        ).to(gl.float32)

    # Process Q/K/V/decay together so Q/K/V share convolution loads.
    # Decay lanes use Q as an in-bounds dummy address and skip state writes.
    qkv_channel, slot_id, local_offset, is_decay = _kda_qkvd_indices(head_idx, H, D)

    qkv_history0 = gl.load(
        conv_pool
        + read_page_offset * CONV_POOL_PAGE_STRIDE
        + qkv_channel * CONV_POOL_CHANNEL_STRIDE
    ).to(gl.float32)
    qkv_history1 = gl.load(
        conv_pool
        + read_page_offset * CONV_POOL_PAGE_STRIDE
        + qkv_channel * CONV_POOL_CHANNEL_STRIDE
        + CONV_POOL_HISTORY_STRIDE
    ).to(gl.float32)
    qkv_history2 = gl.load(
        conv_pool
        + read_page_offset * CONV_POOL_PAGE_STRIDE
        + qkv_channel * CONV_POOL_CHANNEL_STRIDE
        + 2 * CONV_POOL_HISTORY_STRIDE
    ).to(gl.float32)
    qkv_weight_base = conv_weights + qkv_channel * CONV_WEIGHT_ROW_STRIDE
    weight0 = gl.load(qkv_weight_base)
    weight1 = gl.load(qkv_weight_base + CONV_WEIGHT_COL_STRIDE)
    weight2 = gl.load(qkv_weight_base + 2 * CONV_WEIGHT_COL_STRIDE)
    weight3 = gl.load(qkv_weight_base + 3 * CONV_WEIGHT_COL_STRIDE)
    decay_channel = head_idx * D + local_offset
    a_value = gl.exp(gl.load(a_log + head_idx).to(gl.float32))

    # Issue every token's inputs before the first record store: CDNA retires
    # loads and stores through one in-order counter, so a load issued after a
    # store also waits for that store.
    qkv_inputs = ()
    f_a_values = ()
    raw_gate_values = ()
    beta_inputs = ()
    for token_offset in gl.static_range(TOKENS_PER_SEQUENCE):
        load_idx = gl.minimum(token_base + token_offset, last_row)
        qkv_inputs += (gl.load(mixed_qkv + load_idx * MIXED_ROW_STRIDE + qkv_channel),)
        if raw_g is not None:
            raw_gate_values += (
                gl.load(raw_g + load_idx * GATE_ROW_STRIDE + decay_channel).to(
                    gl.float32
                ),
            )
        elif not BATCH_TOKENS:
            f_a_values += (gl.load(f_a_out + load_idx * F_A_ROW_STRIDE + f_a_offsets),)
        beta_inputs += (gl.load(beta_logits + load_idx * BETA_ROW_STRIDE + head_idx),)
    if raw_g is not None:
        decay_bias = gl.load(dt_bias + decay_channel).to(gl.float32)
    elif BATCH_TOKENS:
        lhs_tokens = gl.arange(0, 16, layout=gl.SliceLayout(1, gate_lhs))
        lhs_k = gl.arange(0, F_A, layout=gl.SliceLayout(0, gate_lhs))
        lhs_rows = gl.minimum(token_base + lhs_tokens, last_row)
        f_a_tile = gl.load(
            f_a_out + lhs_rows[:, None] * F_A_ROW_STRIDE + lhs_k[None, :]
        )
        projected_gates = (
            cdna4.mfma(f_a_tile, f_b_tile, gl.zeros([16, D], gl.float32, gate_layout))
            .to(mixed_qkv.dtype.element_ty)
            .to(gl.float32)
        )
        projected_decay.reinterpret(gl.float32, [16, D], rows_layout).store(
            _kda_decay(
                projected_gates,
                gate_bias[None, :],
                a_value,
                HAS_LOWER_BOUND,
                LOWER_BOUND,
            )
        )

    for token_offset in gl.static_range(TOKENS_PER_SEQUENCE):
        token_idx = token_base + token_offset
        qkv_input_raw = qkv_inputs[token_offset]
        if original_qkv is not None:
            # One value split retains the pre-convolution bytes for commit.
            gl.store(
                original_qkv + token_idx * ORIGINAL_QKV_ROW_STRIDE + qkv_channel,
                qkv_input_raw,
                mask=(value_split_idx == 0)
                & (slot_id != 3)
                & (token_offset < token_count),
            )
        qkv_input = qkv_input_raw.to(gl.float32)
        qkv_value = _kda_conv_step(
            qkv_history0,
            qkv_history1,
            qkv_history2,
            qkv_input,
            weight0,
            weight1,
            weight2,
            weight3,
        )

        # Verify and recovery must normalize the same activation-dtype K.
        qkv_value = qkv_value.to(mixed_qkv.dtype.element_ty).to(gl.float32)

        if raw_g is not None:
            raw_gate_value = raw_gate_values[token_offset]
        elif not BATCH_TOKENS:
            f_a_value = f_a_values[token_offset]
            projected_gate = (
                gl.sum(
                    f_b_tile.to(gl.float32) * f_a_value.to(gl.float32)[None, :], axis=1
                )
                .to(mixed_qkv.dtype.element_ty)
                .to(gl.float32)
            )
        if raw_g is None:
            if not BATCH_TOKENS:
                decay_value = _kda_decay(
                    projected_gate, gate_bias, a_value, HAS_LOWER_BOUND, LOWER_BOUND
                )
                projected_decay.index(token_offset).store(decay_value)
            shared_vectors.index(token_offset).store(qkv_value)
        else:
            decay_value = _kda_decay(
                raw_gate_value,
                decay_bias,
                a_value,
                HAS_LOWER_BOUND,
                LOWER_BOUND,
            )
            combined = gl.where(is_decay, decay_value, qkv_value)
            shared_vectors.index(token_offset).store(combined)

        qkv_history0 = qkv_history1
        qkv_history1 = qkv_history2
        qkv_history2 = qkv_input

    # Stamp the whole window at once; a guarded pair per token costs a branch
    # and its memory wait each.
    if stamps is not None and head_idx == 0 and value_split_idx == 0:
        stamp_offsets = gl.arange(0, TOKEN_WIDTH, layout=compact_layout)
        stamp_positions = stamp_start + stamp_offsets
        stamp = (
            stamps
            + (stamp_slot * STAMP_CAPACITY + stamp_positions % STAMP_CAPACITY) * 2
        )
        stamp_live = stamp_offsets < token_count
        gl.store(stamp, stamp_positions, mask=stamp_live)
        gl.store(stamp + 1, stamp_positions * 0 + generation, mask=stamp_live)

    if BATCH_TOKENS:
        # One lane row per token instead of every lane row and warp repeating
        # the window; the per-token path's in-lane order and lane tree remain.
        norm_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [2, 2], [1, 0])
        token_rows = shared_vectors.reinterpret(
            gl.float32, [TOKEN_WIDTH, 4 * D], rows_layout
        )
        q_rows = token_rows.slice(0, D, dim=1)
        k_rows = token_rows.slice(D, D, dim=1)
        q_tile = q_rows.load(norm_layout)
        k_tile = k_rows.load(norm_layout)
        q_tile *= gl.rsqrt(gl.sum(q_tile * q_tile, axis=1) + 1e-6)[:, None] * (D**-0.5)
        k_tile *= gl.rsqrt(gl.sum(k_tile * k_tile, axis=1) + 1e-6)[:, None]
        # Warps share rows: every read lands before any normalized write.
        gl.barrier()
        q_rows.store(q_tile)
        k_rows.store(k_tile)
        gl.barrier()

    q_values = ()
    k_values = ()
    decay_values = ()
    beta_values = ()
    key_queries = ()
    value_panels_by_token = ()
    for token_offset in gl.static_range(TOKENS_PER_SEQUENCE):
        vectors = shared_vectors.index(token_offset)
        q_value = vectors.slice(0, D, dim=0).load(key_layout)
        k_value = vectors.slice(D, D, dim=0).load(key_layout)
        if not BATCH_TOKENS:
            q_value *= gl.rsqrt(gl.sum(q_value * q_value, axis=0) + 1e-6) * (D**-0.5)
            k_value *= gl.rsqrt(gl.sum(k_value * k_value, axis=0) + 1e-6)
        if raw_g is None:
            decay = projected_decay.index(token_offset).load(key_layout)
        else:
            decay = vectors.slice(3 * D, D, dim=0).load(key_layout)
        owner = (value_split_idx == 0) & (token_offset < token_count)
        gl.store(
            retained_key
            + slot * kg_slot_stride
            + head_idx.to(gl.int64) * kg_head_stride
            + (start + token_offset) * kg_pos_stride
            + key_offsets.to(gl.int64) * k_channel_stride,
            k_value,
            mask=owner,
        )
        gl.store(
            retained_decay
            + slot * d_page_stride
            + head_idx.to(gl.int64) * d_head_stride
            + (start + token_offset) * d_pos_stride
            + key_offsets.to(gl.int64) * d_channel_stride,
            decay,
            mask=owner,
        )
        beta_value = beta_inputs[token_offset].to(gl.float32)
        beta_value = 1.0 / (1.0 + gl.exp(-beta_value))
        if panel_rows == 16:
            value_panels = _kda_value_panels(vectors, D, value_layout)
        else:
            value_panels = ()
            for vector_panel_idx in gl.static_range(panel_count):
                value_panels += (
                    vectors.slice(
                        2 * D + vector_panel_idx * panel_rows, panel_rows, dim=0
                    ).load(value_layout),
                )
        q_values = q_values + (q_value,)
        k_values = k_values + (k_value,)
        decay_values = decay_values + (decay,)
        beta_values = beta_values + (beta_value,)
        key_queries = key_queries + (gl.sum(k_value * q_value, axis=0),)
        value_panels_by_token = value_panels_by_token + (value_panels,)

    # Assign each panel to one value split. With one split, the same equation
    # owns every panel and the pipeline window carries loads between iterations.
    pipeline_depth: gl.constexpr = (
        64 // panel_rows if VALUE_SPLITS == 1 and TOKENS_PER_SEQUENCE == 1 else 1
    )
    # Issue more state loads after gate preparation releases the weight registers.
    if VALUE_SPLITS == 1:
        for prefetch_idx in gl.static_range(1, pipeline_depth):
            prefetch_offsets = (
                panel_value_offsets[:, None] + prefetch_idx * panel_rows
            ) * D + key_offsets[None, :]
            prefetch_raw = gl.load(
                state_pool + read_base + prefetch_offsets, cache_modifier=".cg"
            ).to(gl.float32)
            off_pipe = off_pipe + (prefetch_offsets,)
            raw_pipe = raw_pipe + (prefetch_raw,)
    for panel_idx in gl.static_range(panel_count):
        if value_split_idx == panel_idx // panels_per_split:
            if VALUE_SPLITS == 1:
                panel_offsets = off_pipe[0]
                running = raw_pipe[0]
                # Issue the next panel before this panel's stores; see above.
                next_idx = panel_idx + pipeline_depth
                if next_idx < panel_count:
                    next_offsets = (
                        panel_value_offsets[:, None] + next_idx * panel_rows
                    ) * D + key_offsets[None, :]
                    next_raw = gl.load(
                        state_pool + read_base + next_offsets, cache_modifier=".cg"
                    ).to(gl.float32)
                    off_pipe = off_pipe[1:] + (next_offsets,)
                    raw_pipe = raw_pipe[1:] + (next_raw,)
                else:
                    off_pipe = off_pipe[1:] + (off_pipe[-1],)
                    raw_pipe = raw_pipe[1:] + (raw_pipe[-1],)
            elif panel_idx % panels_per_split == 0:
                panel_offsets = first_offsets
                running = first_raw
            else:
                panel_offsets = (
                    panel_value_offsets[:, None] + panel_idx * panel_rows
                ) * D + key_offsets[None, :]
                running = gl.load(
                    state_pool + read_base + panel_offsets, cache_modifier=".cg"
                ).to(gl.float32)

            running = kda_retained_panel(
                running,
                correction,
                retained_key,
                retained_decay,
                slot,
                head_idx,
                start,
                u_slot_stride,
                u_head_stride,
                u_pos_stride,
                u_channel_stride,
                kg_slot_stride,
                kg_head_stride,
                kg_pos_stride,
                k_channel_stride,
                d_page_stride,
                d_head_stride,
                d_pos_stride,
                d_channel_stride,
                panel_idx * panel_rows + panel_value_offsets,
                key_offsets,
            )

            for token_offset in gl.static_range(TOKENS_PER_SEQUENCE):
                q_value = q_values[token_offset]
                k_value = k_values[token_offset]
                decay = decay_values[token_offset]
                value_panel = value_panels_by_token[token_offset][panel_idx]
                beta_value = beta_values[token_offset]

                running *= decay[None, :]
                running, panel_out, delta = _kda_recurrent_step(
                    running,
                    k_value,
                    q_value,
                    value_panel,
                    beta_value,
                    key_queries[token_offset],
                )
                gl.store(
                    output
                    + output_base
                    + token_offset * H * D
                    + panel_idx * panel_rows
                    + panel_value_offsets,
                    panel_out.to(output.dtype.element_ty),
                    mask=token_offset < token_count,
                )
                gl.store(
                    correction
                    + slot * u_slot_stride
                    + head_idx.to(gl.int64) * u_head_stride
                    + ((start + token_offset) % CAPACITY) * u_pos_stride
                    + (panel_idx * panel_rows + panel_value_offsets).to(gl.int64)
                    * u_channel_stride,
                    delta,
                    mask=token_offset < token_count,
                )


def _launch_kda_verify_gfx950(
    *,
    mixed_qkv: torch.Tensor,
    original_qkv: torch.Tensor | None,
    conv_weights: torch.Tensor,
    conv_pool: torch.Tensor,
    raw_g: torch.Tensor | None,
    f_a_out: torch.Tensor | None,
    f_b_weight: torch.Tensor | None,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    correction: torch.Tensor,
    record_metadata: torch.Tensor,
    record_stamps: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
    retained_key,
    retained_decay,
    retained_length,
    live_indices,
) -> torch.Tensor:
    batch = read_indices.numel()
    output = torch.empty(
        (1, mixed_qkv.shape[0], num_heads, head_dim),
        device=mixed_qkv.device,
        dtype=mixed_qkv.dtype,
    )
    if not batch:
        return output
    if (
        conv_weights.shape != (3 * num_heads * head_dim, 4)
        or conv_pool.ndim != 3
        or conv_pool.shape[1] < 3 * num_heads * head_dim
        or conv_pool.shape[2] != 3
        or conv_pool.stride()[1:] != (3, 1)
        or state_pool.stride()[1:] != (head_dim * head_dim, head_dim, 1)
    ):
        raise ValueError(
            "fused KDA verify requires four-tap convolution and V-major state"
        )
    if raw_g is None and mixed_qkv.shape[0] > _INLINE_GATE_MAX_ROWS:
        raw_g = torch.nn.functional.linear(f_a_out, f_b_weight)
        f_a_out = None
        f_b_weight = None
    value_splits = _kda_value_splits(batch)
    gluon_kda_fused_paged_verify_nostore_vmajor_gfx950[
        (num_heads * value_splits, batch)
    ](
        mixed_qkv,
        original_qkv,
        conv_weights,
        conv_pool,
        raw_g,
        f_a_out,
        f_b_weight,
        beta_logits,
        state_pool,
        read_indices,
        output,
        correction,
        record_metadata,
        record_stamps,
        query_start_loc,
        retained_key,
        retained_decay,
        retained_length,
        live_indices,
        correction.stride(3),
        retained_key.stride(3),
        retained_decay.stride(0),
        retained_decay.stride(1),
        retained_decay.stride(2),
        retained_decay.stride(3),
        retained_length.stride(0),
        live_indices.stride(0),
        correction.stride(0),
        correction.stride(1),
        correction.stride(2),
        retained_key.stride(0),
        retained_key.stride(1),
        retained_key.stride(2),
        read_indices.stride(0),
        query_start_loc.stride(0),
        correction.shape[0],
        read_indices.numel() if record_stamps is None else record_stamps.shape[0],
        A_log,
        dt_bias,
        H=num_heads,
        D=head_dim,
        F_A=head_dim,
        TOKENS_PER_SEQUENCE=draft_token_num,
        TOKEN_WIDTH=triton.next_power_of_2(draft_token_num),
        # The batched preparation takes at most 16 tokens.
        BATCH_TOKENS=4 <= draft_token_num <= 16,
        VALUE_SPLITS=value_splits,
        MIXED_ROW_STRIDE=mixed_qkv.stride(0),
        ORIGINAL_QKV_ROW_STRIDE=(0 if original_qkv is None else original_qkv.stride(0)),
        CONV_WEIGHT_ROW_STRIDE=conv_weights.stride(0),
        CONV_WEIGHT_COL_STRIDE=conv_weights.stride(1),
        CONV_POOL_PAGE_STRIDE=conv_pool.stride(0),
        CONV_POOL_CHANNEL_STRIDE=conv_pool.stride(1),
        CONV_POOL_HISTORY_STRIDE=conv_pool.stride(2),
        GATE_ROW_STRIDE=0 if raw_g is None else raw_g.stride(0),
        F_A_ROW_STRIDE=0 if f_a_out is None else f_a_out.stride(0),
        BETA_ROW_STRIDE=beta_logits.stride(0),
        CAPACITY=correction.shape[2],
        STAMP_CAPACITY=(
            correction.shape[2] if record_stamps is None else record_stamps.shape[1]
        ),
        STATE_POOL_PAGE_STRIDE=state_pool.stride(0),
        NUM_POOL_SLOTS=state_pool.shape[0],
        HAS_LOWER_BOUND=lower_bound is not None,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        num_warps=4,
        num_stages=2,
        debug=False,
        sanitize_overflow=False,
    )
    return output


__all__ = [
    "launch_gluon_kda_retained_verify_gfx950",
    "launch_gluon_kda_retained_commit_gfx950",
]


def launch_gluon_kda_retained_verify_gfx950(
    *,
    mixed_qkv,
    original_qkv,
    conv_weights,
    conv_pool,
    raw_g,
    f_a_out,
    f_b_weight,
    beta_logits,
    A_log,
    dt_bias,
    state_pool,
    read_indices,
    query_start_loc,
    correction,
    retained_key,
    retained_decay,
    retained_length,
    live_indices,
    record_metadata,
    record_stamps,
    num_heads,
    head_dim,
    draft_token_num,
    lower_bound,
):
    """Verify from prepared live history while reading convolution at source."""
    validate_kda_retained_verify(
        mixed_qkv,
        original_qkv,
        conv_weights,
        conv_pool,
        raw_g,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool,
        read_indices,
        query_start_loc,
        correction,
        retained_key,
        retained_decay,
        retained_length,
        live_indices,
        record_metadata,
        record_stamps,
        num_heads,
        head_dim,
        draft_token_num,
    )
    prepare_kda_retained(
        state_pool,
        correction,
        retained_key,
        retained_decay,
        retained_length,
        read_indices,
        live_indices,
        query_start_loc,
        heads=num_heads,
        dim=head_dim,
        window=draft_token_num,
        wave=64,
    )
    return _launch_kda_verify_gfx950(
        mixed_qkv=mixed_qkv,
        original_qkv=original_qkv,
        conv_weights=conv_weights,
        conv_pool=conv_pool,
        raw_g=raw_g,
        f_a_out=f_a_out,
        f_b_weight=f_b_weight,
        beta_logits=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        query_start_loc=query_start_loc,
        correction=correction,
        record_metadata=record_metadata,
        record_stamps=record_stamps,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        retained_key=retained_key,
        retained_decay=retained_decay,
        retained_length=retained_length,
        live_indices=live_indices,
    )


def launch_gluon_kda_retained_commit_gfx950(
    *,
    original_qkv,
    conv_pool,
    read_indices,
    live_indices,
    query_start_loc,
    accepted_length,
    retained_length,
    history_capacity,
    record_metadata,
    record_stamps,
    num_heads,
    head_dim,
    draft_token_num,
):
    """Update accepted convolution taps, then publish retained length."""
    launch_kda_retained_commit(
        original_qkv=original_qkv,
        conv_pool=conv_pool,
        read_indices=read_indices,
        live_indices=live_indices,
        query_start_loc=query_start_loc,
        accepted_length=accepted_length,
        retained_length=retained_length,
        history_capacity=history_capacity,
        record_metadata=record_metadata,
        record_stamps=record_stamps,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        wave=64,
    )
