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

"""GFX1250 Gluon KDA recurrent decode kernel."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

gfx1250 = gl.amd.gfx1250


@gluon.jit
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
    """One-token KDA decode with direct indexed state-pool IO."""
    value_block = gl.program_id(0)
    sequence_head = gl.program_id(1)
    sequence_idx = sequence_head // H
    head_idx = sequence_head % H

    if BK == 128 and BV == 32:
        state_layout: gl.constexpr = gl.BlockedLayout(
            [8, 4],
            [8, 4],
            [2, 2],
            [1, 0],
        )
    elif BK == 128 and BV == 8:
        state_layout: gl.constexpr = gl.BlockedLayout(
            [8, 1],
            [8, 4],
            [1, 2],
            [1, 0],
        )
    else:
        state_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1],
            [1, 32],
            [1, gl.num_warps()],
            [1, 0],
        )
    key_layout: gl.constexpr = gl.SliceLayout(1, state_layout)
    value_layout: gl.constexpr = gl.SliceLayout(0, state_layout)

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
    state_mask = key_mask[:, None] & value_mask[None, :]
    read_offsets = key_offsets[:, None] * V + value_offsets[None, :]
    running = gfx1250.buffer_load(
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
    running *= decay[:, None]
    prediction = gl.sum(running * k_value[:, None], axis=0)
    delta = beta_value * (v_value - prediction)
    running += k_value[:, None] * delta[None, :]
    out_value = gl.sum(running * q_value[:, None], axis=0)
    gl.store(
        output + (sequence_idx * H + head_idx) * V + value_offsets,
        out_value.to(output.dtype.element_ty),
        mask=value_mask,
    )
    write_offsets = key_offsets[:, None] * V + value_offsets[None, :]
    gfx1250.buffer_store(
        running,
        state_pool + write_base,
        write_offsets.to(gl.int32),
        mask=valid_write & state_mask,
    )


@gluon.jit
def _kda_fused_decode_kernel(
    mixed_qkv,
    conv_weights,
    conv_states,
    raw_g,
    beta_logits,
    output_gate,
    norm_weight,
    state_pool,
    read_indices,
    write_indices,
    output,
    cu_seqlens,
    a_log,
    dt_bias,
    H: gl.constexpr,
    D: gl.constexpr,
    MIXED_ROW_STRIDE: gl.constexpr,
    CONV_WEIGHT_ROW_STRIDE: gl.constexpr,
    CONV_WEIGHT_COL_STRIDE: gl.constexpr,
    CONV_PAGE_STRIDE: gl.constexpr,
    CONV_CHANNEL_STRIDE: gl.constexpr,
    CONV_HISTORY_STRIDE: gl.constexpr,
    GATE_ROW_STRIDE: gl.constexpr,
    BETA_ROW_STRIDE: gl.constexpr,
    OUTPUT_GATE_ROW_STRIDE: gl.constexpr,
    STATE_PAGE_STRIDE: gl.constexpr,
    NUM_SLOTS: gl.constexpr,
    HAS_LOWER_BOUND: gl.constexpr,
    LOWER_BOUND: gl.constexpr,
    NORM_EPS: gl.constexpr,
):
    """Fuse the K3 decode convolution, recurrence, and gated RMSNorm."""
    sequence_head = gl.program_id(0)
    sequence_idx = sequence_head // H
    head_idx = sequence_head % H

    state_layout: gl.constexpr = gl.BlockedLayout(
        [8, 8],
        [8, 4],
        [2, 4],
        [1, 0],
    )
    key_layout: gl.constexpr = gl.SliceLayout(1, state_layout)
    value_layout: gl.constexpr = gl.SliceLayout(0, state_layout)
    key_offsets = gl.arange(0, D, layout=key_layout)
    value_offsets = gl.arange(0, D, layout=value_layout)

    begin = gl.load(cu_seqlens + sequence_idx)
    end = gl.load(cu_seqlens + sequence_idx + 1)
    output_offsets = (sequence_idx * H + head_idx) * D + value_offsets
    if begin == end:
        gl.store(output + output_offsets, 0.0)
        return

    read_idx = gl.load(read_indices + sequence_idx)
    write_idx = gl.load(write_indices + sequence_idx)
    valid_read = (read_idx >= 0) & (read_idx < NUM_SLOTS)
    if not valid_read:
        gl.store(output + output_offsets, 0.0)
        return
    read_page_offset = read_idx.to(gl.int64)

    token_idx = begin
    projection_width: gl.constexpr = H * D
    q_channel = head_idx * D + key_offsets
    k_channel = projection_width + q_channel
    v_channel = 2 * projection_width + head_idx * D + value_offsets
    q_input = gl.load(mixed_qkv + token_idx * MIXED_ROW_STRIDE + q_channel).to(
        gl.float32
    )
    k_input = gl.load(mixed_qkv + token_idx * MIXED_ROW_STRIDE + k_channel).to(
        gl.float32
    )
    v_input = gl.load(mixed_qkv + token_idx * MIXED_ROW_STRIDE + v_channel).to(
        gl.float32
    )

    q_history0 = gl.load(
        conv_states
        + read_page_offset * CONV_PAGE_STRIDE
        + q_channel * CONV_CHANNEL_STRIDE
    ).to(gl.float32)
    q_history1 = gl.load(
        conv_states
        + read_page_offset * CONV_PAGE_STRIDE
        + q_channel * CONV_CHANNEL_STRIDE
        + CONV_HISTORY_STRIDE
    ).to(gl.float32)
    q_history2 = gl.load(
        conv_states
        + read_page_offset * CONV_PAGE_STRIDE
        + q_channel * CONV_CHANNEL_STRIDE
        + 2 * CONV_HISTORY_STRIDE
    ).to(gl.float32)
    k_history0 = gl.load(
        conv_states
        + read_page_offset * CONV_PAGE_STRIDE
        + k_channel * CONV_CHANNEL_STRIDE
    ).to(gl.float32)
    k_history1 = gl.load(
        conv_states
        + read_page_offset * CONV_PAGE_STRIDE
        + k_channel * CONV_CHANNEL_STRIDE
        + CONV_HISTORY_STRIDE
    ).to(gl.float32)
    k_history2 = gl.load(
        conv_states
        + read_page_offset * CONV_PAGE_STRIDE
        + k_channel * CONV_CHANNEL_STRIDE
        + 2 * CONV_HISTORY_STRIDE
    ).to(gl.float32)
    v_history0 = gl.load(
        conv_states
        + read_page_offset * CONV_PAGE_STRIDE
        + v_channel * CONV_CHANNEL_STRIDE
    ).to(gl.float32)
    v_history1 = gl.load(
        conv_states
        + read_page_offset * CONV_PAGE_STRIDE
        + v_channel * CONV_CHANNEL_STRIDE
        + CONV_HISTORY_STRIDE
    ).to(gl.float32)
    v_history2 = gl.load(
        conv_states
        + read_page_offset * CONV_PAGE_STRIDE
        + v_channel * CONV_CHANNEL_STRIDE
        + 2 * CONV_HISTORY_STRIDE
    ).to(gl.float32)

    q_weight_base = conv_weights + q_channel * CONV_WEIGHT_ROW_STRIDE
    k_weight_base = conv_weights + k_channel * CONV_WEIGHT_ROW_STRIDE
    v_weight_base = conv_weights + v_channel * CONV_WEIGHT_ROW_STRIDE
    q_value = (
        q_history0 * gl.load(q_weight_base)
        + q_history1 * gl.load(q_weight_base + CONV_WEIGHT_COL_STRIDE)
        + q_history2 * gl.load(q_weight_base + 2 * CONV_WEIGHT_COL_STRIDE)
        + q_input * gl.load(q_weight_base + 3 * CONV_WEIGHT_COL_STRIDE)
    ).to(gl.float32)
    k_value = (
        k_history0 * gl.load(k_weight_base)
        + k_history1 * gl.load(k_weight_base + CONV_WEIGHT_COL_STRIDE)
        + k_history2 * gl.load(k_weight_base + 2 * CONV_WEIGHT_COL_STRIDE)
        + k_input * gl.load(k_weight_base + 3 * CONV_WEIGHT_COL_STRIDE)
    ).to(gl.float32)
    v_value = (
        v_history0 * gl.load(v_weight_base)
        + v_history1 * gl.load(v_weight_base + CONV_WEIGHT_COL_STRIDE)
        + v_history2 * gl.load(v_weight_base + 2 * CONV_WEIGHT_COL_STRIDE)
        + v_input * gl.load(v_weight_base + 3 * CONV_WEIGHT_COL_STRIDE)
    ).to(gl.float32)
    q_value *= 1.0 / (1.0 + gl.exp(-q_value))
    k_value *= 1.0 / (1.0 + gl.exp(-k_value))
    v_value *= 1.0 / (1.0 + gl.exp(-v_value))

    valid_write = (write_idx >= 0) & (write_idx < NUM_SLOTS)
    safe_write_idx = gl.where(valid_write, write_idx, 0)
    write_page_offset = safe_write_idx.to(gl.int64)
    q_write_base = (
        conv_states
        + write_page_offset * CONV_PAGE_STRIDE
        + q_channel * CONV_CHANNEL_STRIDE
    )
    k_write_base = (
        conv_states
        + write_page_offset * CONV_PAGE_STRIDE
        + k_channel * CONV_CHANNEL_STRIDE
    )
    v_write_base = (
        conv_states
        + write_page_offset * CONV_PAGE_STRIDE
        + v_channel * CONV_CHANNEL_STRIDE
    )
    gl.store(q_write_base, q_history1, mask=valid_write)
    gl.store(q_write_base + CONV_HISTORY_STRIDE, q_history2, mask=valid_write)
    gl.store(q_write_base + 2 * CONV_HISTORY_STRIDE, q_input, mask=valid_write)
    gl.store(k_write_base, k_history1, mask=valid_write)
    gl.store(k_write_base + CONV_HISTORY_STRIDE, k_history2, mask=valid_write)
    gl.store(k_write_base + 2 * CONV_HISTORY_STRIDE, k_input, mask=valid_write)
    gl.store(v_write_base, v_history1, mask=valid_write)
    gl.store(v_write_base + CONV_HISTORY_STRIDE, v_history2, mask=valid_write)
    gl.store(v_write_base + 2 * CONV_HISTORY_STRIDE, v_input, mask=valid_write)

    gate_value = gl.load(
        raw_g + token_idx * GATE_ROW_STRIDE + head_idx * D + key_offsets
    ).to(gl.float32)
    gate_value += gl.load(dt_bias + head_idx * D + key_offsets).to(gl.float32)
    a_value = gl.exp(gl.load(a_log + head_idx).to(gl.float32))
    if HAS_LOWER_BOUND:
        log_decay = LOWER_BOUND / (1.0 + gl.exp(-(a_value * gate_value)))
    else:
        softplus = gl.maximum(gate_value, 0.0) + gl.log(
            1.0 + gl.exp(-gl.abs(gate_value))
        )
        log_decay = -a_value * softplus
    beta_value = gl.load(beta_logits + token_idx * BETA_ROW_STRIDE + head_idx).to(
        gl.float32
    )
    beta_value = 1.0 / (1.0 + gl.exp(-beta_value))

    q_value *= gl.rsqrt(gl.sum(q_value * q_value, axis=0) + 1e-6) * (D**-0.5)
    k_value *= gl.rsqrt(gl.sum(k_value * k_value, axis=0) + 1e-6)
    state_offsets = key_offsets[:, None] * D + value_offsets[None, :]
    read_base = read_page_offset * STATE_PAGE_STRIDE + head_idx * D * D
    running = gfx1250.buffer_load(
        state_pool + read_base,
        state_offsets.to(gl.int32),
    ).to(gl.float32)
    running *= gl.exp(log_decay)[:, None]
    prediction = gl.sum(running * k_value[:, None], axis=0)
    prior_output = gl.sum(running * q_value[:, None], axis=0)
    key_query = gl.sum(k_value * q_value, axis=0)
    delta = beta_value * (v_value - prediction)
    running += k_value[:, None] * delta[None, :]
    out_value = prior_output + delta * key_query

    write_base = write_page_offset * STATE_PAGE_STRIDE + head_idx * D * D
    gfx1250.buffer_store(
        running,
        state_pool + write_base,
        state_offsets.to(gl.int32),
        mask=valid_write,
    )

    inverse_rms = gl.rsqrt(gl.sum(out_value * out_value, axis=0) / D + NORM_EPS)
    gate = gl.load(
        output_gate + token_idx * OUTPUT_GATE_ROW_STRIDE + head_idx * D + value_offsets
    ).to(gl.float32)
    weight = gl.load(norm_weight + value_offsets).to(gl.float32)
    out_value *= inverse_rms * weight * (1.0 / (1.0 + gl.exp(-gate)))
    gl.store(output + output_offsets, out_value.to(output.dtype.element_ty))


def gluon_kda_recurrent_decode_gfx1250(
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
    """Run one-token indexed KDA decode on a K-major recurrent-state pool."""
    tensors = (q, k, v, g_raw, beta_logits, A_log, dt_bias, state_pool)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("gfx1250 Gluon KDA decode requires GPU tensors")
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("q must have shape [1, batch, heads, key_dim]")
    if read_indices.ndim != 1 or write_indices.shape != read_indices.shape:
        raise ValueError("read_indices and write_indices must be matching vectors")
    if q.shape[1] != read_indices.numel():
        raise ValueError("gfx1250 Gluon KDA decode requires one token per sequence")
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
    expected_tail = (heads, key_dim, value_dim)
    if state_pool.ndim != 4 or state_pool.shape[1:] != expected_tail:
        raise ValueError(
            f"state_pool must have shape [pages, H, K, V] with tail {expected_tail}"
        )
    expected_strides = (key_dim * value_dim, value_dim, 1)
    if state_pool.stride()[1:] != expected_strides:
        raise ValueError("state_pool inner [H, K, V] dimensions must be contiguous")
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
    _kda_recurrent_decode_kernel[(triton.cdiv(value_dim, block_value), tokens * heads)](
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
        num_warps=4 if block_key == 128 and block_value == 32 else 2,
        num_stages=2,
    )
    return output


def gluon_kda_fused_decode_gfx1250(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    raw_g: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
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
        raw_g: Projected decay-gate values with shape
            ``[batch, num_heads * head_dim]``.
        beta_logits: Per-token, per-head update logits with shape
            ``[batch, num_heads]``.
        A_log: Per-head FP32 decay parameters with shape ``[num_heads]``.
        dt_bias: Per-head, per-channel FP32 bias with shape
            ``[num_heads * head_dim]``.
        output_gate: Gated-RMSNorm logits with shape
            ``[batch, num_heads * head_dim]``.
        norm_weight: Gated-RMSNorm weights with shape ``[head_dim]``.
        norm_eps: Gated-RMSNorm epsilon.
        state_pool: Mutable FP32 recurrent state with shape
            ``[pages, num_heads, head_dim, head_dim]``.
        read_indices: Source state page per batch row.
        write_indices: Destination state page per batch row.
        num_heads: Number of local heads; this specialization requires 12.
        head_dim: Per-head width; this specialization requires 128.
        cu_seqlens: Packed row boundaries with shape ``[batch + 1]``.
        lower_bound: Optional safe lower bound for the log-decay gate.

    Returns:
        Gated-RMSNorm output with shape ``[1, batch, 12, 128]``. Valid
        destination convolution and recurrent states are updated in place.
    """
    tensors = (
        mixed_qkv,
        conv_weights,
        conv_states,
        raw_g,
        beta_logits,
        A_log,
        dt_bias,
        output_gate,
        norm_weight,
        state_pool,
        read_indices,
        write_indices,
        cu_seqlens,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("gfx1250 fused KDA decode requires GPU tensors")
    if num_heads != 12 or head_dim != 128:
        raise ValueError("gfx1250 fused KDA decode requires 12 heads of width 128")

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
    if raw_g.shape != (tokens, projection_width) or raw_g.stride(1) != 1:
        raise ValueError("raw_g must have shape [batch, heads * head_dim]")
    if beta_logits.shape != (tokens, num_heads) or beta_logits.stride(1) != 1:
        raise ValueError("beta_logits must have shape [batch, heads]")
    if output_gate.shape != (tokens, projection_width) or output_gate.stride(1) != 1:
        raise ValueError("output_gate must have shape [batch, heads * head_dim]")
    if A_log.shape != (num_heads,) or not A_log.is_contiguous():
        raise ValueError("A_log must have shape [heads]")
    if dt_bias.shape != (projection_width,) or not dt_bias.is_contiguous():
        raise ValueError("dt_bias must have shape [heads * head_dim]")
    if norm_weight.shape != (head_dim,) or not norm_weight.is_contiguous():
        raise ValueError("norm_weight must have shape [head_dim]")
    if state_pool.ndim != 4 or state_pool.shape[1:] != (
        num_heads,
        head_dim,
        head_dim,
    ):
        raise ValueError(
            "state_pool must have shape [pages, heads, head_dim, head_dim]"
        )
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
    _kda_fused_decode_kernel[(tokens * num_heads,)](
        mixed_qkv,
        conv_weights,
        conv_states,
        raw_g,
        beta_logits,
        output_gate,
        norm_weight,
        state_pool,
        read_indices,
        write_indices,
        output,
        cu_seqlens,
        A_log,
        dt_bias,
        H=num_heads,
        D=head_dim,
        MIXED_ROW_STRIDE=mixed_qkv.stride(0),
        CONV_WEIGHT_ROW_STRIDE=conv_weights.stride(0),
        CONV_WEIGHT_COL_STRIDE=conv_weights.stride(1),
        CONV_PAGE_STRIDE=conv_states.stride(0),
        CONV_CHANNEL_STRIDE=conv_states.stride(1),
        CONV_HISTORY_STRIDE=conv_states.stride(2),
        GATE_ROW_STRIDE=raw_g.stride(0),
        BETA_ROW_STRIDE=beta_logits.stride(0),
        OUTPUT_GATE_ROW_STRIDE=output_gate.stride(0),
        STATE_PAGE_STRIDE=state_pool.stride(0),
        NUM_SLOTS=state_pool.shape[0],
        HAS_LOWER_BOUND=lower_bound is not None,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        NORM_EPS=norm_eps,
        num_warps=8,
        num_stages=2,
    )
    return output


__all__ = [
    "gluon_kda_fused_decode_gfx1250",
    "gluon_kda_recurrent_decode_gfx1250",
]
