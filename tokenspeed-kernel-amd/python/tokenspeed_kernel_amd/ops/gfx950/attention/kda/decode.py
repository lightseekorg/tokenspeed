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
    kda_record_interval,
    kda_records_launch_metadata,
    validate_kda_records_commit,
    validate_kda_records_verify,
)

cdna4 = gl.amd.cdna4

_CDNA4_NUM_CUS = 256
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


@gluon.jit
def gluon_kda_paged_decode_gfx950(
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
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    Q_TOKEN_STRIDE: gl.constexpr,
    K_TOKEN_STRIDE: gl.constexpr,
    V_TOKEN_STRIDE: gl.constexpr,
    G_TOKEN_STRIDE: gl.constexpr,
    BETA_TOKEN_STRIDE: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    NUM_SLOTS: gl.constexpr,
    STATE_PAGE_STRIDE: gl.constexpr,
    HAS_LOWER_BOUND: gl.constexpr,
    LOWER_BOUND: gl.constexpr,
):
    """One-token KDA decode with direct V-major indexed state-pool IO."""
    value_block = gl.program_id(0)
    sequence_head = gl.program_id(1)
    sequence_idx = sequence_head // H
    head_idx = sequence_head % H

    if BK == 128 and BV == 32:
        state_layout: gl.constexpr = gl.BlockedLayout(
            [4, 16],
            [8, 8],
            [1, 1],
            [1, 0],
        )
    elif BK == 128 and BV == 8:
        state_layout: gl.constexpr = gl.BlockedLayout(
            [1, 8],
            [8, 8],
            [1, 1],
            [1, 0],
        )
    else:
        state_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1],
            [64, 1],
            [gl.num_warps(), 1],
            [1, 0],
        )
    value_layout: gl.constexpr = gl.SliceLayout(1, state_layout)
    key_layout: gl.constexpr = gl.SliceLayout(0, state_layout)

    begin = gl.load(cu_seqlens + sequence_idx)
    end = gl.load(cu_seqlens + sequence_idx + 1)
    value_offsets = value_block * BV + gl.arange(0, BV, layout=value_layout)
    value_mask = value_offsets < V
    if begin == end:
        gl.store(
            output + (sequence_idx * H + head_idx) * V + value_offsets,
            0.0,
            mask=value_mask,
        )
        return

    read_idx = gl.load(read_indices + sequence_idx)
    write_idx = gl.load(write_indices + sequence_idx)
    valid_read = (read_idx >= 0) & (read_idx < NUM_SLOTS)
    if not valid_read:
        gl.store(
            output + (sequence_idx * H + head_idx) * V + value_offsets,
            0.0,
            mask=value_mask,
        )
        return

    key_offsets = gl.arange(0, BK, layout=key_layout)
    key_mask = key_offsets < K
    read_base = read_idx * STATE_PAGE_STRIDE + head_idx * K * V

    token_idx = begin
    q_value = gl.load(
        q + token_idx * Q_TOKEN_STRIDE + head_idx * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(gl.float32)
    k_value = gl.load(
        k + token_idx * K_TOKEN_STRIDE + head_idx * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(gl.float32)
    gate_value = gl.load(
        raw_g + token_idx * G_TOKEN_STRIDE + head_idx * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(gl.float32)
    gate_value += gl.load(
        dt_bias + head_idx * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(gl.float32)
    a_value = gl.exp(gl.load(a_log + head_idx).to(gl.float32))
    if HAS_LOWER_BOUND:
        log_decay = LOWER_BOUND / (1.0 + gl.exp(-(a_value * gate_value)))
    else:
        softplus = gl.maximum(gate_value, 0.0) + gl.log(
            1.0 + gl.exp(-gl.abs(gate_value))
        )
        log_decay = -a_value * softplus
    beta_value = gl.load(raw_beta + token_idx * BETA_TOKEN_STRIDE + head_idx).to(
        gl.float32
    )
    beta_value = 1.0 / (1.0 + gl.exp(-beta_value))

    scale: gl.constexpr = K**-0.5
    q_value *= gl.rsqrt(gl.sum(q_value * q_value, axis=0) + 1e-6) * scale
    k_value *= gl.rsqrt(gl.sum(k_value * k_value, axis=0) + 1e-6)
    valid_write = (write_idx >= 0) & (write_idx < NUM_SLOTS)
    safe_write_idx = gl.where(valid_write, write_idx, 0)
    write_base = safe_write_idx * STATE_PAGE_STRIDE + head_idx * K * V
    decay = gl.exp(log_decay)
    state_mask = value_mask[:, None] & key_mask[None, :]
    read_offsets = value_offsets[:, None] * K + key_offsets[None, :]
    running = cdna4.buffer_load(
        state_pool + read_base,
        read_offsets.to(gl.int32),
        mask=state_mask,
        other=0.0,
    ).to(gl.float32)
    v_value = gl.load(
        v + token_idx * V_TOKEN_STRIDE + head_idx * V + value_offsets,
        mask=value_mask,
        other=0.0,
    ).to(gl.float32)
    running *= decay[None, :]
    prediction = gl.sum(running * k_value[None, :], axis=1)
    delta = beta_value * (v_value - prediction)
    running += delta[:, None] * k_value[None, :]
    out_value = gl.sum(running * q_value[None, :], axis=1)
    gl.store(
        output + (sequence_idx * H + head_idx) * V + value_offsets,
        out_value.to(output.dtype.element_ty),
        mask=value_mask,
    )
    write_offsets = value_offsets[:, None] * K + key_offsets[None, :]
    cdna4.buffer_store(
        running,
        state_pool + write_base,
        write_offsets.to(gl.int32),
        mask=valid_write & state_mask,
    )


@gluon.jit
def gluon_kda_fused_paged_decode_vmajor_gfx950(
    mixed_qkv,
    conv_weights,
    conv_states,
    f_a_out,
    f_b_weight,
    beta_logits,
    state_pool,
    read_indices,
    write_indices,
    output,
    cu_seqlens,
    a_log,
    dt_bias,
    H: gl.constexpr,
    D: gl.constexpr,
    F_A: gl.constexpr,
    MIXED_ROW_STRIDE: gl.constexpr,
    CONV_WEIGHT_ROW_STRIDE: gl.constexpr,
    CONV_WEIGHT_COL_STRIDE: gl.constexpr,
    CONV_PAGE_STRIDE: gl.constexpr,
    CONV_CHANNEL_STRIDE: gl.constexpr,
    CONV_HISTORY_STRIDE: gl.constexpr,
    F_A_ROW_STRIDE: gl.constexpr,
    BETA_ROW_STRIDE: gl.constexpr,
    STATE_PAGE_STRIDE: gl.constexpr,
    NUM_SLOTS: gl.constexpr,
    HAS_LOWER_BOUND: gl.constexpr,
    LOWER_BOUND: gl.constexpr,
    PIPELINE_DEPTH: gl.constexpr,
):
    """Fuse the K3 decode convolution, decay projection, and recurrence."""
    head_idx = gl.program_id(0)
    sequence_idx = gl.program_id(1)

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
    # Pack Q, K, and V into one row (the decay lanes go unused); decay and
    # output each move between layouts through their own D-wide allocation.
    shared_vectors = gl.allocate_shared_memory(gl.float32, [1, 4 * D], shared_layout)
    decay_alloc = gl.allocate_shared_memory(gl.float32, [1, D], shared_layout)
    output_alloc = gl.allocate_shared_memory(gl.float32, [1, D], shared_layout)

    # Loads retire in order, so every small load is issued ahead of the state
    # panels, and the per-head parameters ahead of the padding-row checks.
    # The decay projection takes this head's [D, F_A] slice of f_b_weight.
    # Four lanes share a weight row, so the reduction leaves each thread only
    # two decay channels to finish instead of the state tile's eight.
    projection_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [16, 4],
        [4, 1],
        [1, 0],
    )
    gate_rows = gl.arange(0, D, layout=gl.SliceLayout(1, projection_layout))
    f_a_offsets = gl.arange(0, F_A, layout=gl.SliceLayout(0, projection_layout))
    f_b_tile = gl.load(
        f_b_weight + (head_idx * D + gate_rows)[:, None] * F_A + f_a_offsets[None, :]
    )
    gate_bias = gl.load(dt_bias + head_idx * D + gate_rows)
    a_value = gl.load(a_log + head_idx)
    # Process Q/K/V together so they share convolution loads. The fourth
    # (decay) lanes use Q as an in-bounds dummy address and skip state writes.
    qkv_channel, slot_id, _, _ = _kda_qkvd_indices(head_idx, H, D)
    qkv_weight_base = conv_weights + qkv_channel * CONV_WEIGHT_ROW_STRIDE
    qkv_weight0 = gl.load(qkv_weight_base)
    qkv_weight1 = gl.load(qkv_weight_base + CONV_WEIGHT_COL_STRIDE)
    qkv_weight2 = gl.load(qkv_weight_base + 2 * CONV_WEIGHT_COL_STRIDE)
    qkv_weight3 = gl.load(qkv_weight_base + 3 * CONV_WEIGHT_COL_STRIDE)

    begin = gl.load(cu_seqlens + sequence_idx)
    end = gl.load(cu_seqlens + sequence_idx + 1)
    read_idx = gl.load(read_indices + sequence_idx)
    write_idx = gl.load(write_indices + sequence_idx)
    output_base = (sequence_idx * H + head_idx) * D
    if begin == end:
        gl.store(output + output_base + output_offsets, 0.0)
        return

    valid_read = (read_idx >= 0) & (read_idx < NUM_SLOTS)
    if not valid_read:
        gl.store(output + output_base + output_offsets, 0.0)
        return

    valid_write = (write_idx >= 0) & (write_idx < NUM_SLOTS)
    safe_write_idx = gl.where(valid_write, write_idx, 0)
    read_page_offset = read_idx.to(gl.int64)
    write_page_offset = safe_write_idx.to(gl.int64)
    read_base = read_page_offset * STATE_PAGE_STRIDE + head_idx * D * D
    panel_value_offsets = gl.arange(0, 16, layout=value_layout)

    token_idx = begin
    f_a_value = gl.load(f_a_out + token_idx * F_A_ROW_STRIDE + f_a_offsets)
    beta_value = gl.load(beta_logits + token_idx * BETA_ROW_STRIDE + head_idx)
    qkv_input = gl.load(mixed_qkv + token_idx * MIXED_ROW_STRIDE + qkv_channel)
    qkv_read_base = (
        conv_states
        + read_page_offset * CONV_PAGE_STRIDE
        + qkv_channel * CONV_CHANNEL_STRIDE
    )
    qkv_history0 = gl.load(qkv_read_base)
    qkv_history1 = gl.load(qkv_read_base + CONV_HISTORY_STRIDE)
    qkv_history2 = gl.load(qkv_read_base + 2 * CONV_HISTORY_STRIDE)

    # Keep PIPELINE_DEPTH state panels in flight to balance latency and VGPR use.
    # static_range permits the unrolled tuple window to change length.
    off_pipe = ()
    raw_pipe = ()
    for _pd in gl.static_range(PIPELINE_DEPTH):
        _off_pd = (panel_value_offsets[:, None] + _pd * 16) * D + key_offsets[None, :]
        _raw_pd = gl.load(state_pool + read_base + _off_pd).to(gl.float32)
        off_pipe = off_pipe + (_off_pd,)
        raw_pipe = raw_pipe + (_raw_pd,)

    qkv_value = _kda_conv_step(
        qkv_history0.to(gl.float32),
        qkv_history1.to(gl.float32),
        qkv_history2.to(gl.float32),
        qkv_input.to(gl.float32),
        qkv_weight0,
        qkv_weight1,
        qkv_weight2,
        qkv_weight3,
    )
    qkv_write_base = (
        conv_states
        + write_page_offset * CONV_PAGE_STRIDE
        + qkv_channel * CONV_CHANNEL_STRIDE
    )
    write_mask = valid_write & (slot_id != 3)
    gl.store(qkv_write_base, qkv_history1, mask=write_mask)
    gl.store(qkv_write_base + CONV_HISTORY_STRIDE, qkv_history2, mask=write_mask)
    gl.store(qkv_write_base + 2 * CONV_HISTORY_STRIDE, qkv_input, mask=write_mask)

    # Round like the BF16 GEMM that prefill and verify use for the same gate.
    raw_gate = gl.sum(
        f_b_tile.to(gl.float32) * f_a_value.to(gl.float32)[None, :], axis=1
    )
    raw_gate = raw_gate.to(f_a_out.dtype.element_ty).to(gl.float32)
    decay_value = _kda_decay(
        raw_gate,
        gate_bias.to(gl.float32),
        gl.exp(a_value.to(gl.float32)),
        HAS_LOWER_BOUND,
        LOWER_BOUND,
    )
    beta_value = 1.0 / (1.0 + gl.exp(-beta_value.to(gl.float32)))

    shared_vectors.index(0).store(qkv_value)
    decay_alloc.index(0).store(decay_value)
    q_value = shared_vectors.index(0).slice(0, D, dim=0).load(key_layout)
    k_value = shared_vectors.index(0).slice(D, D, dim=0).load(key_layout)
    decay = decay_alloc.index(0).load(key_layout)
    q_square = gl.sum(q_value * q_value, axis=0)
    k_square = gl.sum(k_value * k_value, axis=0)
    raw_key_query = gl.sum(q_value * k_value, axis=0)
    q_scale = gl.rsqrt(q_square + 1e-6) * (D**-0.5)
    k_scale = gl.rsqrt(k_square + 1e-6)
    q_value *= q_scale
    k_value *= k_scale
    key_query = raw_key_query * q_scale * k_scale
    write_base = write_page_offset * STATE_PAGE_STRIDE + head_idx * D * D
    output_shared = output_alloc.index(0)
    value_panels = _kda_value_panels(shared_vectors.index(0), D, value_layout)
    output_panels = (
        output_shared.slice(0, 16, dim=0),
        output_shared.slice(16, 16, dim=0),
        output_shared.slice(32, 16, dim=0),
        output_shared.slice(48, 16, dim=0),
        output_shared.slice(64, 16, dim=0),
        output_shared.slice(80, 16, dim=0),
        output_shared.slice(96, 16, dim=0),
        output_shared.slice(112, 16, dim=0),
    )
    for panel_idx in gl.static_range(8):
        cur_off = off_pipe[0]
        cur_raw = raw_pipe[0]
        running = cur_raw * decay[None, :]
        prediction = gl.sum(running * k_value[None, :], axis=1)
        prior_output = gl.sum(running * q_value[None, :], axis=1)
        delta = beta_value * (value_panels[panel_idx] - prediction)
        running += delta[:, None] * k_value[None, :]
        panel_out = prior_output + delta * key_query
        gl.store(
            state_pool + write_base + cur_off,
            running,
            mask=valid_write,
            cache_modifier=".cs",
        )
        output_panels[panel_idx].store(panel_out)

        next_idx = panel_idx + PIPELINE_DEPTH
        if next_idx < 8:
            next_off = (panel_value_offsets[:, None] + next_idx * 16) * D + key_offsets[
                None, :
            ]
            next_raw = gl.load(state_pool + read_base + next_off).to(gl.float32)
            off_pipe = off_pipe[1:] + (next_off,)
            raw_pipe = raw_pipe[1:] + (next_raw,)
        else:
            off_pipe = off_pipe[1:] + (off_pipe[-1],)
            raw_pipe = raw_pipe[1:] + (raw_pipe[-1],)
    out_value = output_shared.load(compact_layout)
    gl.store(
        output + output_base + output_offsets,
        out_value.to(output.dtype.element_ty),
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
    key_gate,
    metadata,
    stamps,
    query_start_loc,
    u_slot_stride,
    u_head_stride,
    u_pos_stride,
    kg_slot_stride,
    kg_head_stride,
    kg_pos_stride,
    read_stride,
    query_stride,
    record_slots,
    a_log: tl.const,
    dt_bias: tl.const,
    H: gl.constexpr,
    D: gl.constexpr,
    F_A: gl.constexpr,
    TOKENS_PER_SEQUENCE: gl.constexpr,
    TOKEN_WIDTH: gl.constexpr,
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
    STATE_POOL_PAGE_STRIDE,
    NUM_POOL_SLOTS,
    HAS_LOWER_BOUND: gl.constexpr,
    LOWER_BOUND,
):
    """Run fused no-store verify and save U plus K/raw-g recovery records."""
    head_program_idx = gl.program_id(0)
    head_idx = head_program_idx // VALUE_SPLITS
    value_split_idx = head_program_idx % VALUE_SPLITS
    sequence_idx = gl.program_id(1)

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
    shared_vectors = gl.allocate_shared_memory(
        gl.float32, [TOKENS_PER_SEQUENCE, 4 * D], shared_layout
    )
    if raw_g is None:
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
        gate_bias = gl.load(dt_bias + head_idx * D + gate_rows).to(gl.float32)
        projected_decay = gl.allocate_shared_memory(
            gl.float32, [TOKENS_PER_SEQUENCE, D], shared_layout
        )
    token_base = gl.load(query_start_loc + sequence_idx * query_stride).to(gl.int64)
    token_count = (
        gl.load(query_start_loc + (sequence_idx + 1) * query_stride).to(gl.int64)
        - token_base
    )
    output_base = (token_base * H + head_idx) * D

    read_idx = gl.load(read_indices + sequence_idx * read_stride)
    # Issue the record metadata with the read index so the state loads wait on
    # one round trip rather than two.
    slot = gl.load(metadata + sequence_idx * 3)
    generation = gl.load(metadata + sequence_idx * 3 + 1)
    start = gl.load(metadata + sequence_idx * 3 + 2)
    valid_read = (read_idx > 0) & (read_idx < NUM_POOL_SLOTS)
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
        & (generation >= 0)
        & (start >= 0)
        & (token_count >= 0)
        & (token_count <= TOKENS_PER_SEQUENCE)
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
    read_base = read_page_offset * STATE_POOL_PAGE_STRIDE + head_idx * D * D
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
        if raw_g is None:
            f_a_values += (gl.load(f_a_out + load_idx * F_A_ROW_STRIDE + f_a_offsets),)
        else:
            raw_gate_values += (
                gl.load(raw_g + load_idx * GATE_ROW_STRIDE + decay_channel).to(
                    gl.float32
                ),
            )
        beta_inputs += (gl.load(beta_logits + load_idx * BETA_ROW_STRIDE + head_idx),)
    if raw_g is not None:
        decay_bias = gl.load(dt_bias + decay_channel).to(gl.float32)

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

        if raw_g is None:
            f_a_value = f_a_values[token_offset]
            projected_gate = (
                gl.sum(
                    f_b_tile.to(gl.float32) * f_a_value.to(gl.float32)[None, :], axis=1
                )
                .to(mixed_qkv.dtype.element_ty)
                .to(gl.float32)
            )
        else:
            raw_gate_value = raw_gate_values[token_offset]
        position = start + token_offset
        ring = position % CAPACITY
        record_base = (
            key_gate
            + slot * kg_slot_stride
            + head_idx * kg_head_stride
            + ring * kg_pos_stride
            + local_offset
        )
        owner = (value_split_idx == 0) & (token_offset < token_count)
        gl.store(record_base, qkv_value, mask=owner & (slot_id == 1))
        if raw_g is None:
            projected_record = (
                key_gate
                + slot * kg_slot_stride
                + head_idx * kg_head_stride
                + ring * kg_pos_stride
                + D
                + gate_rows
            )
            gl.store(projected_record, projected_gate, mask=owner)
        else:
            gl.store(record_base + D, raw_gate_value, mask=owner & is_decay)
        if raw_g is None:
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
    if head_idx == 0 and value_split_idx == 0:
        stamp_offsets = gl.arange(0, TOKEN_WIDTH, layout=compact_layout)
        stamp_positions = start + stamp_offsets
        stamp = stamps + (slot * CAPACITY + stamp_positions % CAPACITY) * 2
        stamp_live = stamp_offsets < token_count
        gl.store(stamp, stamp_positions, mask=stamp_live)
        gl.store(stamp + 1, stamp_positions * 0 + generation, mask=stamp_live)

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
        q_value *= gl.rsqrt(gl.sum(q_value * q_value, axis=0) + 1e-6) * (D**-0.5)
        k_value *= gl.rsqrt(gl.sum(k_value * k_value, axis=0) + 1e-6)
        if raw_g is None:
            decay = projected_decay.index(token_offset).load(key_layout)
        else:
            decay = vectors.slice(3 * D, D, dim=0).load(key_layout)
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
                    + head_idx * u_head_stride
                    + ((start + token_offset) % CAPACITY) * u_pos_stride
                    + panel_idx * panel_rows
                    + panel_value_offsets,
                    delta,
                    mask=token_offset < token_count,
                )


@gluon.jit(
    launch_metadata=kda_records_launch_metadata,
    do_not_specialize=[
        "BATCH",
        "qkv_stride",
        "conv_stride",
        "state_stride",
        "u_stride",
        "kg_stride",
        "record_slots",
        "stamp_layer_stride",
        "lower_bound",
    ],
)
def gluon_kda_fused_replay_gfx950(
    descriptors,
    group_indices,
    reads,
    writes,
    accepted,
    metadata,
    stamps,
    activation_ref,
    history_ref,
    BATCH,
    qkv_stride,
    conv_stride,
    state_stride,
    u_stride,
    kg_stride,
    record_slots,
    stamp_layer_stride,
    lower_bound,
    H: gl.constexpr,
    D: gl.constexpr,
    T: gl.constexpr,
    CAPACITY: gl.constexpr,
    BOUNDED: gl.constexpr,
):
    """Recover accepted records with one program per head, request and layer."""
    head_idx = gl.program_id(0)
    sequence_idx = gl.program_id(1)
    layer_idx = gl.program_id(2)
    # Descriptors and step counts depend only on the program, so they share the
    # group-index round trip instead of waiting behind the record validation.
    descriptor_base = layer_idx * 10
    mixed_qkv = gl.load(descriptors + descriptor_base).to(
        gl.pointer_type(activation_ref.dtype.element_ty)
    )
    conv_pool = gl.load(descriptors + descriptor_base + 2).to(
        gl.pointer_type(history_ref.dtype.element_ty)
    )
    correction = gl.load(descriptors + descriptor_base + 3).to(
        gl.pointer_type(gl.float32)
    )
    key_gate = gl.load(descriptors + descriptor_base + 5).to(
        gl.pointer_type(activation_ref.dtype.element_ty)
    )
    a_log = gl.load(descriptors + descriptor_base + 6).to(gl.pointer_type(gl.float32))
    dt_bias = gl.load(descriptors + descriptor_base + 7).to(gl.pointer_type(gl.float32))
    state_pool = gl.load(descriptors + descriptor_base + 8).to(
        gl.pointer_type(gl.float32)
    )
    group_idx = gl.load(group_indices + layer_idx).to(gl.int64)
    replay_steps = gl.load(accepted + sequence_idx)
    read_idx = gl.load(reads + group_idx * BATCH + sequence_idx).to(gl.int64)
    write_idx = gl.load(writes + group_idx * BATCH + sequence_idx).to(gl.int64)
    if read_idx <= 0 or write_idx <= 0:
        return

    state_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [4, 16],
        [4, 1],
        [1, 0],
    )
    key_layout: gl.constexpr = gl.SliceLayout(0, state_layout)
    value_layout: gl.constexpr = gl.SliceLayout(1, state_layout)
    key_offsets = gl.arange(0, D, layout=key_layout)

    read_page_offset = read_idx.to(gl.int64)
    read_base = read_page_offset * state_stride + head_idx * D * D
    panel_rows: gl.constexpr = 64 if T == 1 and BOUNDED else 16
    panel_count: gl.constexpr = D // panel_rows
    panel_value_offsets = gl.arange(0, panel_rows, layout=value_layout)
    _off = panel_value_offsets[:, None] * D + key_offsets[None, :]
    _raw = gl.load(state_pool + read_base + _off, cache_modifier=".cg").to(gl.float32)
    off_pipe = (_off,)
    raw_pipe = (_raw,)

    # One head owns its Q/K/V history, so load all taps before an in-place write.
    qkv_channel, slot_id, _, _ = _kda_qkvd_indices(head_idx, H, D)
    qkv_read_base = conv_pool + read_idx * conv_stride + qkv_channel * 3
    qkv_history0 = gl.load(qkv_read_base).to(gl.float32)
    qkv_history1 = gl.load(qkv_read_base + 1).to(gl.float32)
    qkv_history2 = gl.load(qkv_read_base + 2).to(gl.float32)
    for token_offset in gl.static_range(T):
        token_active = token_offset < replay_steps
        qkv_input = gl.load(
            mixed_qkv
            + (sequence_idx.to(gl.int64) * T + token_offset) * qkv_stride
            + qkv_channel,
            mask=token_active,
            other=0.0,
        ).to(gl.float32)
        qkv_history0 = gl.where(token_active, qkv_history1, qkv_history0)
        qkv_history1 = gl.where(token_active, qkv_history2, qkv_history1)
        qkv_history2 = gl.where(token_active, qkv_input, qkv_history2)
    slot, start, valid = kda_record_interval(
        metadata,
        stamps,
        sequence_idx,
        layer_idx,
        replay_steps,
        record_slots,
        CAPACITY,
        stamp_layer_stride,
        T,
        64,
    )
    if not valid:
        return
    qkv_write_base = conv_pool + write_idx * conv_stride + qkv_channel * 3
    gl.store(qkv_write_base, qkv_history0, mask=slot_id != 3)
    gl.store(qkv_write_base + 1, qkv_history1, mask=slot_id != 3)
    gl.store(qkv_write_base + 2, qkv_history2, mask=slot_id != 3)

    a_value = gl.exp(gl.load(a_log + head_idx).to(gl.float32))
    bias = gl.load(dt_bias + head_idx * D + key_offsets).to(gl.float32)
    # Stage accepted records once for key normalization and state updates.
    record_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    # Rows past the accepted prefix reload the last live record instead of
    # masking: a uniform mask becomes a branch with a memory wait per load.
    last_step = gl.maximum(replay_steps - 1, 0)
    correction_offsets = gl.arange(0, D, layout=record_layout)
    key_gate_offsets = gl.arange(0, 2 * D, layout=record_layout)
    correction_shared = gl.allocate_shared_memory(
        gl.float32, [T, D], gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    )
    key_gate_shared = gl.allocate_shared_memory(
        activation_ref.dtype.element_ty,
        [T, 2 * D],
        gl.SwizzledSharedLayout(1, 1, 1, order=[0]),
    )
    for token_offset in gl.static_range(T):
        record_row = (
            slot * CAPACITY + (start + gl.minimum(token_offset, last_step)) % CAPACITY
        )
        correction_value = gl.load(
            correction + record_row * u_stride + head_idx * D + correction_offsets
        )
        correction_shared.index(token_offset).store(correction_value)
        record_base = key_gate + record_row * kg_stride + head_idx * 2 * D
        key_gate_value = gl.load(record_base + key_gate_offsets)
        key_gate_shared.index(token_offset).store(key_gate_value)

    k_values = ()
    decay_values = ()
    for token_offset in gl.static_range(T):
        vectors = key_gate_shared.index(token_offset)
        k_value = vectors.slice(0, D, dim=0).load(key_layout).to(gl.float32)
        raw_gate = vectors.slice(D, D, dim=0).load(key_layout).to(gl.float32)
        k_value *= gl.rsqrt(gl.sum(k_value * k_value, axis=0) + 1e-6)
        decay = _kda_decay(raw_gate, bias, a_value, BOUNDED, lower_bound)
        k_values += (k_value,)
        decay_values += (decay,)

    # Bounded T1 uses two wide panels. Other windows retain their load pipeline.
    pipeline_depth: gl.constexpr = 2 if T == 1 and not BOUNDED else 1
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
        panel_offsets = off_pipe[0]
        running = raw_pipe[0]
        for token_offset in gl.static_range(T):
            # Inactive rows are initialized; unconditional reads share one barrier.
            delta = (
                correction_shared.index(token_offset)
                .slice(panel_idx * panel_rows, panel_rows, dim=0)
                .load(value_layout)
            )
            if token_offset < replay_steps:
                running *= decay_values[token_offset][None, :]
                running += delta[:, None] * k_values[token_offset][None, :]
        write_base = write_idx * state_stride + head_idx * D * D
        gl.store(state_pool + write_base + panel_offsets, running)

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


def gluon_kda_recurrent_decode_gfx950(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Run one-token indexed KDA decode on a V-major recurrent-state pool.

    Args:
        q, k, g_raw: Packed ``[1, batch, heads, key_dim]`` tensors.
        v: Packed ``[1, batch, heads, value_dim]`` tensor.
        beta_logits: Packed ``[1, batch, heads]`` beta logits.
        A_log: Per-head FP32 decay parameter.
        dt_bias: Per-head, per-key FP32 decay bias.
        state_pool: Persistent FP32 state, physical shape
            ``[pages, heads, value_dim, key_dim]`` (V-major).
        read_indices: Source page per batch row.
        write_indices: Destination page per batch row.
        cu_seqlens: Packed row boundaries. Each active row contains one token.
        lower_bound: Optional safe lower bound for log decay.

    Returns:
        KDA output with the same shape and dtype as ``v``.
    """
    tensors = (q, k, v, g_raw, beta_logits, A_log, dt_bias, state_pool)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("gfx950 Gluon KDA decode requires GPU tensors")
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("q must have shape [1, batch, heads, key_dim]")
    if read_indices.ndim != 1 or write_indices.shape != read_indices.shape:
        raise ValueError("read_indices and write_indices must be matching vectors")
    if q.shape[1] != read_indices.numel():
        raise ValueError("gfx950 Gluon KDA decode requires one token per sequence")
    if q.shape != k.shape or q.shape != g_raw.shape:
        raise ValueError("q, k, and raw_g must have identical shapes")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("v must match q through the head dimension")

    _, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if beta_logits.shape != (1, tokens, heads):
        raise ValueError("beta_logits must have shape [1, batch, heads]")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() != tokens + 1:
        raise ValueError("cu_seqlens must contain one boundary per decode row")
    expected_tail = (heads, value_dim, key_dim)
    if state_pool.ndim != 4 or state_pool.shape[1:] != expected_tail:
        raise ValueError(
            f"state_pool must have physical shape [pages, H, V, K] with tail {expected_tail}"
        )
    expected_strides = (value_dim * key_dim, key_dim, 1)
    if state_pool.stride()[1:] != expected_strides:
        raise ValueError("state_pool inner [H, V, K] dimensions must be contiguous")
    if state_pool.stride(0) < heads * key_dim * value_dim:
        raise ValueError("state_pool pages must not overlap")
    if A_log.shape != (heads,) or dt_bias.numel() != heads * key_dim:
        raise ValueError("invalid KDA gate parameter shapes")
    expected_inner_strides = (
        (q, key_dim),
        (k, key_dim),
        (g_raw, key_dim),
        (v, value_dim),
    )
    if any(
        tensor.stride(-1) != 1 or tensor.stride(-2) != width
        for tensor, width in expected_inner_strides
    ):
        raise ValueError("KDA inputs must have contiguous head vectors")
    if beta_logits.stride(-1) != 1:
        raise ValueError("KDA beta logits must have contiguous heads")

    A_log = A_log.contiguous()
    dt_bias = dt_bias.view(heads, key_dim).contiguous()
    read_indices = read_indices.to(device=q.device, dtype=torch.int32).contiguous()
    write_indices = write_indices.to(device=q.device, dtype=torch.int32).contiguous()
    cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()

    output = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    block_key = triton.next_power_of_2(key_dim)
    block_value = min(32, triton.next_power_of_2(value_dim))
    gluon_kda_paged_decode_gfx950[
        (triton.cdiv(value_dim, block_value), tokens * heads)
    ](
        q,
        k,
        v,
        g_raw,
        beta_logits,
        state_pool,
        read_indices,
        write_indices,
        output,
        cu_seqlens,
        A_log,
        dt_bias,
        H=heads,
        K=key_dim,
        V=value_dim,
        Q_TOKEN_STRIDE=q.stride(1),
        K_TOKEN_STRIDE=k.stride(1),
        V_TOKEN_STRIDE=v.stride(1),
        G_TOKEN_STRIDE=g_raw.stride(1),
        BETA_TOKEN_STRIDE=beta_logits.stride(1),
        BK=block_key,
        BV=block_value,
        NUM_SLOTS=state_pool.shape[0],
        STATE_PAGE_STRIDE=state_pool.stride(0),
        HAS_LOWER_BOUND=lower_bound is not None,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        num_warps=1,
        num_stages=2,
    )
    return output


def gluon_kda_fused_decode_gfx950(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Run fused single-token K3 decode against paged convolution/KDA state.

    Args:
        mixed_qkv: BF16 pre-convolution Q/K/V rows with shape
            ``[batch, 3 * num_heads * head_dim]``.
        conv_weights: Contiguous four-tap depthwise convolution weights with
            shape ``[3 * num_heads * head_dim, 4]``.
        conv_states: Mutable BF16 convolution state pool with shape
            ``[pages, 3 * num_heads * head_dim, 3]``.
        f_a_out: BF16 low-rank decay-gate activations with shape
            ``[batch, head_dim]``.
        f_b_weight: Contiguous BF16 decay-gate up projection with shape
            ``[num_heads * head_dim, head_dim]``. The kernel computes the raw
            decay gate ``f_a_out @ f_b_weight.T`` with FP32 accumulation and
            rounds it to BF16.
        beta_logits: Per-token, per-head update logits with shape
            ``[batch, num_heads]``.
        A_log: Per-head FP32 decay parameters with shape ``[num_heads]``.
        dt_bias: Per-head, per-channel FP32 decay bias with shape
            ``[num_heads * head_dim]``.
        state_pool: Mutable FP32 recurrent state, physical shape
            ``[pages, num_heads, head_dim(V), head_dim(K)]`` (V-major).
        read_indices: Source state page per batch row. Negative entries mark
            graph-padding rows.
        write_indices: Destination state page per batch row. Negative entries
            suppress state updates.
        num_heads: Number of local KDA heads.
        head_dim: Per-head key/value width; this specialization requires 128.
        cu_seqlens: Packed row boundaries with shape ``[batch + 1]``. Active
            rows contain one token and graph-padding rows contain zero tokens.
        lower_bound: Optional safe lower bound for the log-decay gate.

    Returns:
        Raw KDA output with shape
        ``[1, batch, num_heads, head_dim]`` and ``mixed_qkv`` dtype. The
        convolution and recurrent states at valid ``write_indices`` are
        updated as part of the call.
    """
    tensors = (
        mixed_qkv,
        conv_weights,
        conv_states,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool,
        read_indices,
        write_indices,
        cu_seqlens,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("gfx950 fused KDA decode requires GPU tensors")
    if head_dim != 128:
        raise ValueError("gfx950 fused KDA decode requires heads of width 128")

    tokens = read_indices.numel()
    projection_width = num_heads * head_dim
    if mixed_qkv.ndim != 2 or mixed_qkv.shape != (tokens, 3 * projection_width):
        raise ValueError("mixed_qkv must have shape [batch, 3 * heads * head_dim]")
    if mixed_qkv.stride(1) != 1:
        raise ValueError("mixed_qkv channels must be contiguous")
    if (
        conv_weights.shape != (3 * projection_width, 4)
        or not conv_weights.is_contiguous()
    ):
        raise ValueError("conv_weights must be contiguous [3 * heads * head_dim, 4]")
    if conv_states.ndim != 3 or conv_states.shape[1:] != (3 * projection_width, 3):
        raise ValueError("conv_states must have shape [pages, 3 * heads * head_dim, 3]")
    if conv_states.stride()[1:] != (3, 1):
        raise ValueError(
            "conv_states inner channel/history dimensions must be contiguous"
        )
    if f_a_out.shape != (tokens, head_dim) or f_a_out.stride(1) != 1:
        raise ValueError("f_a_out must have shape [batch, head_dim]")
    if (
        f_b_weight.shape != (projection_width, head_dim)
        or f_b_weight.dtype != f_a_out.dtype
        or not f_b_weight.is_contiguous()
    ):
        raise ValueError(
            "f_b_weight must be contiguous [heads * head_dim, head_dim] in "
            "f_a_out dtype"
        )
    if beta_logits.shape != (tokens, num_heads) or beta_logits.stride(1) != 1:
        raise ValueError("beta_logits must have shape [batch, heads]")
    if A_log.shape != (num_heads,) or not A_log.is_contiguous():
        raise ValueError("A_log must have shape [heads]")
    if dt_bias.shape != (projection_width,) or not dt_bias.is_contiguous():
        raise ValueError("dt_bias must have shape [heads * head_dim]")
    if state_pool.ndim != 4 or state_pool.shape[1:] != (
        num_heads,
        head_dim,
        head_dim,
    ):
        raise ValueError("state_pool must have physical shape [pages, heads, V, K]")
    if state_pool.stride()[1:] != (head_dim * head_dim, head_dim, 1):
        raise ValueError("state_pool inner dimensions must be contiguous")
    if conv_states.shape[0] != state_pool.shape[0]:
        raise ValueError(
            "convolution and recurrent state pools must have equal capacity"
        )
    if read_indices.shape != (tokens,) or write_indices.shape != (tokens,):
        raise ValueError("read_indices and write_indices must match the decode batch")
    if read_indices.dtype != torch.int32 or write_indices.dtype != torch.int32:
        raise ValueError("read_indices and write_indices must be int32")
    if not read_indices.is_contiguous() or not write_indices.is_contiguous():
        raise ValueError("read_indices and write_indices must be contiguous")
    if cu_seqlens.shape != (tokens + 1,) or cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be an int32 boundary vector")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")

    output = torch.empty(
        (1, tokens, num_heads, head_dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    # Up to one CTA per CU, keep every state panel in flight. Up to two, trade
    # the panel window for occupancy. Beyond that the grid has enough CTAs to
    # keep memory busy, and a four-panel window measured fastest.
    programs = num_heads * tokens
    if programs <= _CDNA4_NUM_CUS:
        pipeline_depth = 8
    elif programs <= 2 * _CDNA4_NUM_CUS:
        pipeline_depth = 1
    else:
        pipeline_depth = 4
    gluon_kda_fused_paged_decode_vmajor_gfx950[(num_heads, tokens)](
        mixed_qkv,
        conv_weights,
        conv_states,
        f_a_out,
        f_b_weight,
        beta_logits,
        state_pool,
        read_indices,
        write_indices,
        output,
        cu_seqlens,
        A_log,
        dt_bias,
        H=num_heads,
        D=head_dim,
        F_A=f_a_out.shape[1],
        MIXED_ROW_STRIDE=mixed_qkv.stride(0),
        CONV_WEIGHT_ROW_STRIDE=conv_weights.stride(0),
        CONV_WEIGHT_COL_STRIDE=conv_weights.stride(1),
        CONV_PAGE_STRIDE=conv_states.stride(0),
        CONV_CHANNEL_STRIDE=conv_states.stride(1),
        CONV_HISTORY_STRIDE=conv_states.stride(2),
        F_A_ROW_STRIDE=f_a_out.stride(0),
        BETA_ROW_STRIDE=beta_logits.stride(0),
        STATE_PAGE_STRIDE=state_pool.stride(0),
        NUM_SLOTS=state_pool.shape[0],
        HAS_LOWER_BOUND=lower_bound is not None,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        PIPELINE_DEPTH=pipeline_depth,
        num_warps=4,
        num_stages=2,
    )
    return output


def gluon_kda_fused_verify_gfx950(
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
    key_gate: torch.Tensor,
    record_metadata: torch.Tensor,
    record_stamps: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
) -> torch.Tensor:
    """Produce attention output and records without changing committed state.

    original_qkv is an optional separate destination for live pre-convolution
    rows, in packed input order. None leaves retention to the caller. Commit
    uses fixed-width windows, so serving supplies dense query boundaries.
    mixed_qkv contains activation-dtype pre-convolution Q/K/V rows; raw_g and
    beta_logits are projected gate inputs. A supplied raw_g requires both
    projection inputs to be None. Otherwise f_a_out and f_b_weight produce the
    gate inline for small row counts and with GEMM for larger inputs.
    state_pool is FP32 [page, head, V, K].
    read_indices selects checkpoint pages; nonpositive pages are padding.
    correction is FP32 [slot, head, capacity, V]. key_gate has the same layout
    with activation-dtype unnormalized K followed by raw gate in its last axis.
    record_metadata is INT64 [batch, 3] (slot, generation, absolute start), and
    record_stamps is INT64 [slots, capacity, 2] (position, generation).
    query_start_loc supplies packed token boundaries. draft_token_num bounds
    the candidate window; ordinary decode uses one token. lower_bound selects
    the bounded gate, or None selects softplus. Return [1, tokens, heads, dim].
    """
    validate_kda_records_verify(
        mixed_qkv,
        original_qkv,
        raw_g,
        beta_logits,
        state_pool,
        read_indices,
        correction,
        key_gate,
        record_metadata,
        record_stamps,
        query_start_loc,
        A_log,
        dt_bias,
        num_heads,
        head_dim,
        draft_token_num,
        f_a_out=f_a_out,
        f_b_weight=f_b_weight,
    )
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
        key_gate,
        record_metadata,
        record_stamps,
        query_start_loc,
        correction.stride(0),
        correction.stride(1),
        correction.stride(2),
        key_gate.stride(0),
        key_gate.stride(1),
        key_gate.stride(2),
        read_indices.stride(0),
        query_start_loc.stride(0),
        correction.shape[0],
        A_log,
        dt_bias,
        H=num_heads,
        D=head_dim,
        F_A=head_dim,
        TOKENS_PER_SEQUENCE=draft_token_num,
        TOKEN_WIDTH=triton.next_power_of_2(draft_token_num),
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


def launch_gluon_kda_fused_replay_gfx950(
    descriptors: torch.Tensor,
    group_indices: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    *,
    draft_token_num: int,
    num_heads: int,
    head_dim: int,
    qkv_stride: int,
    conv_stride: int,
    correction_stride: int,
    key_gate_stride: int,
    state_stride: int,
    conv_width: int,
    activation_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
    lower_bound: float | None,
) -> None:
    """Recover all layers from saved U/K/raw-g records, then commit conv history.

    Args:
        descriptors: Contiguous uint64 [layers, 10] pointer table. Columns
            0/2/8 hold original QKV, conv history and recurrent state; 3/5
            hold FP32 U and activation-dtype K/raw-g; 6/7 hold A_log/dt_bias.
            Columns 1/4/9 are reserved and are not read.
        group_indices: INT32 cache-group row for each layer.
        read_indices: Contiguous INT32 [groups, batch] checkpoint pages.
        write_indices: Matching destination pages. Nonpositive source or
            destination pages mark padding and leave both pools unchanged.
        accepted_length: Contiguous INT32 [batch] counts in [0, draft_token_num].
            Zero copies the checkpoint and conv history without reading records.
        draft_token_num: Candidate window bound; standard decode uses 1.
        num_heads: Local head count.
        head_dim: Key/value width, currently 128.
        qkv_stride: Original QKV row stride in elements.
        conv_stride: Conv-history page stride in elements.
        correction_stride: FP32 U token stride in elements.
        key_gate_stride: K/raw-g token stride in elements.
        state_stride: FP32 recurrent page stride in elements.
        conv_width: Convolution taps, currently 4.
        activation_dtype: Original QKV and K/raw-g storage dtype.
        conv_state_dtype: Persistent conv-history dtype, independently of inputs.
        record_metadata: INT64 [batch, 3] (slot, generation, absolute start).
        record_stamps: INT64 [layers, slots, capacity, 2] position/generation.
            Both metadata tensors can be None for standalone unstamped probes.
        lower_bound: Bounded log-decay, or None for softplus.

    Returns:
        None. Updates accepted destination states and history. Source and
        destination may alias within a request; writable pages cannot overlap
        between requests. Recovery never recomputes the state-dependent U.
    """
    if head_dim != 128 or conv_width != 4:
        raise ValueError("gfx950 records commit requires D=128 and W=4")
    validate_kda_records_commit(
        descriptors,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        record_metadata,
        record_stamps,
        draft_token_num,
    )
    batch, layers = accepted_length.numel(), descriptors.shape[0]
    if not batch:
        return
    capacity = draft_token_num if record_stamps is None else record_stamps.shape[2]
    slots = batch if record_stamps is None else record_stamps.shape[1]
    stamp_stride = 0 if record_stamps is None else record_stamps.stride(0)
    activation_ref = torch.empty(0, dtype=activation_dtype, device=descriptors.device)
    history_ref = torch.empty(0, dtype=conv_state_dtype, device=descriptors.device)
    gluon_kda_fused_replay_gfx950[(num_heads, batch, layers)](
        descriptors,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        record_metadata,
        record_stamps,
        activation_ref,
        history_ref,
        batch,
        qkv_stride,
        conv_stride,
        state_stride,
        correction_stride,
        key_gate_stride,
        slots,
        stamp_stride,
        0.0 if lower_bound is None else lower_bound,
        H=num_heads,
        D=head_dim,
        T=draft_token_num,
        CAPACITY=capacity,
        BOUNDED=lower_bound is not None,
        num_warps=4,
        num_stages=2,
        debug=False,
        sanitize_overflow=False,
    )


__all__ = [
    "gluon_kda_fused_decode_gfx950",
    "launch_gluon_kda_fused_replay_gfx950",
    "gluon_kda_fused_verify_gfx950",
    "gluon_kda_recurrent_decode_gfx950",
]
