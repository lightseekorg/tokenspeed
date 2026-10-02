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

"""Triton Mamba2 scans: chunked SSD prefill, paged decode, speculative verify and replay."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton
from tokenspeed_kernel.ops.attention.mamba2 import Mamba2ChunkMetadata
from tokenspeed_kernel.platform import CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_SIGNATURES = format_signatures(("x",), "dense", {torch.bfloat16})
# Small tiles keep more states in flight in the latency-bound verify and replay loops.
_SPEC_BLOCK_M, _SPEC_NUM_WARPS = 16, 2
_LOG2E = tl.constexpr(1.4426950408889634)
# Prefill launch shapes, tuned on GB300 at Nemotron-3 Super geometry.
_STATE_PASS_BLOCK = 1024
_SSD_BLOCK_M, _SSD_BLOCK_K = 64, 32
_SSD_STATE_WARPS, _SSD_SCAN_WARPS = 4, 4
_SSD_STATE_STAGES, _SSD_SCAN_STAGES = 3, 1
# CUDA caps grid axis 1; scan chunks and verify requests sit there, the fastest order.
_MAX_GRID_Y = 65535


def _dot_block(size: int) -> int:
    """A tensor-core operand dimension: a power of two of at least 16."""
    return max(16, triton.next_power_of_2(size))


@triton.jit
def _softplus(x):
    return tl.where(x <= 20.0, libdevice.log1p(tl.math.exp(x)), x)


@triton.jit
def _exp(x):
    return tl.math.exp2(_LOG2E * x)


@triton.jit
def _log_decays(dt_row, stride_dt_t, rows, mask, bias, A, low, high):
    """Processed step sizes of a token block and their per-token log decays."""
    raw = tl.load(dt_row + rows.to(tl.int64) * stride_dt_t, mask=mask, other=0.0)
    step = tl.minimum(tl.maximum(_softplus(raw.to(tl.float32) + bias), low), high)
    step = tl.where(mask, step, 0.0)
    return step, step * A


@triton.jit
def _ssd_chunk_state_kernel(
    x,
    dt,
    B,
    A_log,
    dt_bias,
    cu_chunk_seqlens,
    states,
    chunk_decay,
    steps,
    decays,
    low,
    high,
    stride_x_t,
    stride_x_h,
    stride_dt_t,
    stride_dt_h,
    stride_B_t,
    stride_B_g,
    CHUNK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D_STATE: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One chunk's own contribution to the state at its end, from a zero start.

    Also records each token's step size and in-chunk cumulative log decay.
    """
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    num_heads = tl.num_programs(1)
    start = tl.load(cu_chunk_seqlens + chunk)
    length = tl.load(cu_chunk_seqlens + chunk + 1) - start
    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    A = -tl.exp(tl.load(A_log + head).to(tl.float32))
    bias = tl.load(dt_bias + head).to(tl.float32)
    dt_row = dt + head * stride_dt_h
    offs_l = tl.arange(0, CHUNK)
    _, chunk_log_decay = _log_decays(
        dt_row, stride_dt_t, start + offs_l, offs_l < length, bias, A, low, high
    )
    total = tl.sum(chunk_log_decay, axis=0)
    group = head // HEADS_PER_GROUP
    x_head = x + head * stride_x_h
    B_group = B + group * stride_B_g
    acc = tl.zeros([BLOCK_P, BLOCK_N], dtype=tl.float32)
    prefix = 0.0
    for k0 in range(0, length, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < length
        rows = (start + offs_k).to(tl.int64)
        step, log_decay = _log_decays(
            dt_row, stride_dt_t, rows, mask_k, bias, A, low, high
        )
        decay = prefix + tl.cumsum(log_decay, axis=0)
        prefix += tl.sum(log_decay, axis=0)
        token = (chunk.to(tl.int64) * num_heads + head) * CHUNK + offs_k
        tl.store(steps + token, step, mask=mask_k)
        tl.store(decays + token, decay, mask=mask_k)
        # Decay from each token to the chunk end, times its step size.
        weight = _exp(total - decay) * step
        xs_t = tl.load(
            x_head + rows[None, :] * stride_x_t + offs_p[:, None],
            mask=mask_k[None, :] & (offs_p[:, None] < HEAD_DIM),
            other=0.0,
        )
        bs = tl.load(
            B_group + rows[:, None] * stride_B_t + offs_n[None, :],
            mask=mask_k[:, None] & (offs_n[None, :] < D_STATE),
            other=0.0,
        )
        # The weighted operand is split into two BF16 halves to keep FP32-like precision.
        weighted = bs.to(tl.float32) * weight[:, None]
        upper = weighted.to(xs_t.dtype)
        lower = (weighted - upper.to(tl.float32)).to(xs_t.dtype)
        acc = tl.dot(xs_t, upper, acc)
        acc = tl.dot(xs_t, lower, acc)
    tile = (chunk.to(tl.int64) * num_heads + head) * HEAD_DIM * D_STATE
    tl.store(
        states + tile + offs_p[:, None] * D_STATE + offs_n[None, :],
        acc,
        mask=(offs_p[:, None] < HEAD_DIM) & (offs_n[None, :] < D_STATE),
    )
    tl.store(chunk_decay + chunk * num_heads + head, _exp(total))


@triton.jit
def _ssd_state_passing_kernel(
    states,
    incoming,
    chunk_decay,
    initial_states,
    final_states,
    last_chunk_indices,
    stride_initial_seq,
    stride_initial_head,
    stride_final_seq,
    stride_final_head,
    STATE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Carry each sequence's state across its chunks, recording each chunk's incoming state."""
    seq = tl.program_id(0)
    head = tl.program_id(1)
    num_heads = tl.num_programs(1)
    offs = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < STATE_SIZE
    first = tl.load(last_chunk_indices + seq - 1, mask=seq > 0, other=-1) + 1
    last = tl.load(last_chunk_indices + seq)
    state = tl.load(
        initial_states
        + seq.to(tl.int64) * stride_initial_seq
        + head * stride_initial_head
        + offs,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tile = (first.to(tl.int64) * num_heads + head) * STATE_SIZE + offs
    local = tl.load(states + tile, mask=mask, other=0.0)
    decay = tl.load(chunk_decay + first * num_heads + head)
    for chunk in range(first, last + 1):
        # The next chunk's loads issue before this chunk's store.
        more = chunk < last
        next_tile = tile + num_heads * STATE_SIZE
        next_local = tl.load(states + next_tile, mask=mask & more, other=0.0)
        next_decay = tl.load(
            chunk_decay + (chunk + 1) * num_heads + head, mask=more, other=0.0
        )
        tl.store(incoming + tile, state.to(incoming.dtype.element_ty), mask=mask)
        state = decay * state + local
        tile, local, decay = next_tile, next_local, next_decay
    tl.store(
        final_states
        + seq.to(tl.int64) * stride_final_seq
        + head * stride_final_head
        + offs,
        state.to(final_states.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _ssd_chunk_cb_kernel(
    B,
    C,
    cu_chunk_seqlens,
    cb,
    stride_B_t,
    stride_B_g,
    stride_C_t,
    stride_C_g,
    CHUNK: tl.constexpr,
    D_STATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``C_t . B_s`` for one row block of a chunk's token pairs, shared by a group's heads."""
    chunk = tl.program_id(0)
    m0 = tl.program_id(1) * BLOCK_M
    group = tl.program_id(2)
    num_groups = tl.num_programs(2)
    start = tl.load(cu_chunk_seqlens + chunk)
    length = tl.load(cu_chunk_seqlens + chunk + 1) - start
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_l = tl.arange(0, CHUNK)
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < D_STATE
    rows_m = (start + offs_m).to(tl.int64)
    rows_l = (start + offs_l).to(tl.int64)
    cs = tl.load(
        C + rows_m[:, None] * stride_C_t + group * stride_C_g + offs_n[None, :],
        mask=(offs_m < length)[:, None] & mask_n[None, :],
        other=0.0,
    )
    bs_t = tl.load(
        B + rows_l[None, :] * stride_B_t + group * stride_B_g + offs_n[:, None],
        mask=(offs_l < length)[None, :] & mask_n[:, None],
        other=0.0,
    )
    tile = (chunk.to(tl.int64) * num_groups + group) * CHUNK * CHUNK
    tl.store(
        cb + tile + offs_m[:, None] * CHUNK + offs_l[None, :],
        tl.dot(cs, bs_t),
        mask=(offs_m < CHUNK)[:, None],
    )


@triton.jit
def _ssd_chunk_scan_kernel(
    x,
    C,
    D,
    cu_chunk_seqlens,
    cb,
    incoming,
    steps,
    decays,
    out,
    stride_x_t,
    stride_x_h,
    stride_C_t,
    stride_C_g,
    stride_out_t,
    stride_out_h,
    CHUNK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D_STATE: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Outputs of one row block of a chunk: causal in-chunk part plus the incoming state's."""
    m0 = tl.program_id(0) * BLOCK_M
    chunk = tl.program_id(1)
    head = tl.program_id(2)
    num_heads = tl.num_programs(2)
    start = tl.load(cu_chunk_seqlens + chunk)
    length = tl.load(cu_chunk_seqlens + chunk + 1) - start
    if m0 >= length:
        return
    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    mask_p = offs_p < HEAD_DIM
    mask_n = offs_n < D_STATE
    group = head // HEADS_PER_GROUP
    x_head = x + head * stride_x_h
    C_group = C + group * stride_C_g
    cb_rows = (
        cb
        + (chunk.to(tl.int64) * (num_heads // HEADS_PER_GROUP) + group) * CHUNK * CHUNK
    )
    tokens = (chunk.to(tl.int64) * num_heads + head) * CHUNK
    offs_m = m0 + tl.arange(0, BLOCK_M)
    mask_m = offs_m < length
    rows_m = (start + offs_m).to(tl.int64)
    decay_m = tl.load(decays + tokens + offs_m, mask=mask_m, other=0.0)
    cs = tl.load(
        C_group + rows_m[:, None] * stride_C_t + offs_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )
    tile = (chunk.to(tl.int64) * num_heads + head) * HEAD_DIM * D_STATE
    incoming_t = tl.load(
        incoming + tile + offs_n[:, None] + offs_p[None, :] * D_STATE,
        mask=mask_n[:, None] & mask_p[None, :],
        other=0.0,
    )
    acc = tl.dot(cs, incoming_t) * _exp(decay_m)[:, None]
    for k0 in range(0, m0 + BLOCK_M, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < length
        rows_k = (start + offs_k).to(tl.int64)
        step_k = tl.load(steps + tokens + offs_k, mask=mask_k, other=0.0)
        decay_k = tl.load(decays + tokens + offs_k, mask=mask_k, other=0.0)
        pair = tl.load(
            cb_rows + offs_m[:, None] * CHUNK + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        xs = tl.load(
            x_head + rows_k[:, None] * stride_x_t + offs_p[None, :],
            mask=mask_k[:, None] & mask_p[None, :],
            other=0.0,
        )
        causal = offs_m[:, None] >= offs_k[None, :]
        gaps = tl.where(causal, decay_m[:, None] - decay_k[None, :], float("-inf"))
        scores = pair * _exp(gaps) * step_k[None, :]
        acc = tl.dot(scores.to(xs.dtype), xs, acc)
    xm = tl.load(
        x_head + rows_m[:, None] * stride_x_t + offs_p[None, :],
        mask=mask_m[:, None] & mask_p[None, :],
        other=0.0,
    )
    acc += tl.load(D + head).to(tl.float32) * xm.to(tl.float32)
    tl.store(
        out + rows_m[:, None] * stride_out_t + head * stride_out_h + offs_p[None, :],
        acc.to(out.dtype.element_ty),
        mask=mask_m[:, None] & mask_p[None, :],
    )


@register_kernel(
    "attention",
    "mamba2_chunk_scan",
    name="triton_mamba2_chunk_scan",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_SIGNATURES,
    priority=Priority.PORTABLE,
)
def triton_mamba2_chunk_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    dt_limit: tuple[float, float],
    initial_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_metadata: Mamba2ChunkMetadata,
    out: torch.Tensor,
) -> torch.Tensor:
    num_seqs = cu_seqlens.shape[0] - 1
    _, num_heads, head_dim = x.shape
    d_state = B.shape[-1]
    num_chunks = chunk_metadata.seq_idx.shape[0]
    if num_chunks > _MAX_GRID_Y:
        raise ValueError(
            f"{num_chunks} chunks exceed the {_MAX_GRID_Y} the scan grid holds per launch"
        )
    final_states = torch.empty_like(initial_states)
    if num_chunks == 0:
        return final_states.copy_(initial_states)
    chunk = chunk_metadata.chunk_size
    states = torch.empty(
        num_chunks, num_heads, head_dim, d_state, device=x.device, dtype=torch.float32
    )
    chunk_decay = torch.empty(
        num_chunks, num_heads, device=x.device, dtype=torch.float32
    )
    incoming = torch.empty_like(states, dtype=x.dtype)
    steps = torch.empty(
        num_chunks, num_heads, chunk, device=x.device, dtype=torch.float32
    )
    decays = torch.empty_like(steps)
    num_groups = B.shape[1]
    cb = torch.empty(
        num_chunks, num_groups, chunk, chunk, device=x.device, dtype=torch.float32
    )
    low, high = dt_limit
    cu_chunks = chunk_metadata.cu_chunk_seqlens
    blocks = dict(
        HEAD_DIM=head_dim,
        D_STATE=d_state,
        HEADS_PER_GROUP=num_heads // B.shape[1],
        BLOCK_P=_dot_block(head_dim),
        BLOCK_N=_dot_block(d_state),
        BLOCK_K=_SSD_BLOCK_K,
    )
    block_m = min(_SSD_BLOCK_M, chunk)
    _ssd_chunk_state_kernel[(num_chunks, num_heads)](
        x,
        dt,
        B,
        A_log,
        dt_bias,
        cu_chunks,
        states,
        chunk_decay,
        steps,
        decays,
        low,
        high,
        x.stride(0),
        x.stride(1),
        dt.stride(0),
        dt.stride(1),
        B.stride(0),
        B.stride(1),
        CHUNK=chunk,
        **blocks,
        num_warps=_SSD_STATE_WARPS,
        num_stages=_SSD_STATE_STAGES,
    )
    state_size = head_dim * d_state
    _ssd_state_passing_kernel[
        (num_seqs, num_heads, triton.cdiv(state_size, _STATE_PASS_BLOCK))
    ](
        states,
        incoming,
        chunk_decay,
        initial_states,
        final_states,
        chunk_metadata.last_chunk_indices,
        initial_states.stride(0),
        initial_states.stride(1),
        final_states.stride(0),
        final_states.stride(1),
        STATE_SIZE=state_size,
        BLOCK=_STATE_PASS_BLOCK,
        num_warps=4,
    )
    _ssd_chunk_cb_kernel[(num_chunks, triton.cdiv(chunk, block_m), num_groups)](
        B,
        C,
        cu_chunks,
        cb,
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        CHUNK=chunk,
        D_STATE=d_state,
        BLOCK_M=block_m,
        BLOCK_N=_dot_block(d_state),
        num_warps=8,
    )
    _ssd_chunk_scan_kernel[(triton.cdiv(chunk, block_m), num_chunks, num_heads)](
        x,
        C,
        D,
        cu_chunks,
        cb,
        incoming,
        steps,
        decays,
        out,
        x.stride(0),
        x.stride(1),
        C.stride(0),
        C.stride(1),
        out.stride(0),
        out.stride(1),
        CHUNK=chunk,
        BLOCK_M=block_m,
        **blocks,
        num_warps=_SSD_SCAN_WARPS,
        num_stages=_SSD_SCAN_STAGES,
    )
    return final_states


@register_kernel(
    "attention",
    "mamba2_state_update",
    name="triton_mamba2_state_update",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_SIGNATURES,
    priority=Priority.PORTABLE,
)
def triton_mamba2_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_indices: torch.Tensor,
    dst_state_indices: torch.Tensor,
    null_slot: int,
    out: torch.Tensor,
) -> None:
    # Decode is a one-token verify that writes its state.
    triton_mamba2_verify_scan(
        state,
        x[:, None],
        dt[:, None],
        A_log,
        B[:, None],
        C[:, None],
        D,
        dt_bias,
        state_indices=state_indices,
        dst_state_indices=dst_state_indices[:, None],
        null_slot=null_slot,
        out=out[:, None],
    )


@triton.jit(do_not_specialize=["null_slot", "has_dst"])
def _mamba2_verify_scan_kernel(
    state,
    x,
    dt,
    A_log,
    B,
    C,
    D,
    dt_bias,
    state_indices,
    dst_state_indices,
    out,
    null_slot,
    has_dst,
    stride_state_slot,
    stride_state_head,
    stride_state_dim,
    stride_x_batch,
    stride_x_t,
    stride_x_head,
    stride_dt_batch,
    stride_dt_t,
    stride_B_batch,
    stride_B_t,
    stride_B_group,
    stride_C_batch,
    stride_C_t,
    stride_C_group,
    stride_out_batch,
    stride_out_t,
    stride_out_head,
    stride_dst_batch,
    T: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D_STATE: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    # has_dst is a runtime flag: one binary for both modes keeps their arithmetic identical.
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)
    pid_h = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < HEAD_DIM
    mask_n = offs_n < D_STATE
    mask = mask_m[:, None] & mask_n[None, :]
    A = -tl.exp(tl.load(A_log + pid_h).to(tl.float32))
    bias = tl.load(dt_bias + pid_h).to(tl.float32)
    skip = tl.load(D + pid_h).to(tl.float32)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    read = tl.load(state_indices + pid_b).to(tl.int64)
    tile = (
        pid_h * stride_state_head + offs_m[:, None] * stride_state_dim + offs_n[None, :]
    )
    h = tl.load(
        state + read * stride_state_slot + tile,
        mask=mask & (read != null_slot),
        other=0.0,
    ).to(tl.float32)
    group = pid_h // HEADS_PER_GROUP
    x_row = x + pid_b * stride_x_batch + pid_h * stride_x_head + offs_m
    dt_row = dt + pid_b * stride_dt_batch + pid_h
    B_row = B + pid_b * stride_B_batch + group * stride_B_group + offs_n
    C_row = C + pid_b * stride_C_batch + group * stride_C_group + offs_n
    out_row = out + pid_b * stride_out_batch + pid_h * stride_out_head + offs_m
    # Token t + 1's inputs are in flight while token t computes.
    x_next = tl.load(x_row, mask=mask_m, other=0.0)
    dt_next = tl.load(dt_row)
    B_next = tl.load(B_row, mask=mask_n, other=0.0)
    C_next = tl.load(C_row, mask=mask_n, other=0.0)
    for t in tl.static_range(T):
        xv = x_next.to(tl.float32)
        dt_t = dt_next.to(tl.float32)
        Bv = B_next.to(tl.float32)
        Cv = C_next.to(tl.float32)
        if t + 1 < T:
            x_next = tl.load(x_row + (t + 1) * stride_x_t, mask=mask_m, other=0.0)
            dt_next = tl.load(dt_row + (t + 1) * stride_dt_t)
            B_next = tl.load(B_row + (t + 1) * stride_B_t, mask=mask_n, other=0.0)
            C_next = tl.load(C_row + (t + 1) * stride_C_t, mask=mask_n, other=0.0)
        step = _softplus(dt_t + bias)
        dA = _exp(A * step)
        dB = Bv * step
        h = h * dA + dB * xv[:, None]
        if has_dst != 0:
            dst = tl.load(dst_state_indices + pid_b * stride_dst_batch + t).to(tl.int64)
            if dst != null_slot:
                tl.store(
                    state + dst * stride_state_slot + tile,
                    h.to(state.dtype.element_ty),
                    mask=mask,
                )
        y = tl.sum(h * Cv[None, :], axis=1)
        y += xv * skip
        tl.store(out_row + t * stride_out_t, y, mask=mask_m)
        # Decode reloads the stored state for its next token.
        h = h.to(state.dtype.element_ty).to(tl.float32)


@register_kernel(
    "attention",
    "mamba2_verify_scan",
    name="triton_mamba2_verify_scan",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_SIGNATURES,
    priority=Priority.PORTABLE,
)
def triton_mamba2_verify_scan(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_indices: torch.Tensor,
    dst_state_indices: torch.Tensor | None,
    null_slot: int,
    out: torch.Tensor,
) -> None:
    batch, T, num_heads, head_dim = x.shape
    if batch > _MAX_GRID_Y:
        raise ValueError(
            f"{batch} requests exceed the {_MAX_GRID_Y} the verify grid holds per launch"
        )
    d_state = state.shape[-1]
    block_m, num_warps = _SPEC_BLOCK_M, _SPEC_NUM_WARPS
    enable_pdl = pdl_enabled()
    _mamba2_verify_scan_kernel[(triton.cdiv(head_dim, block_m), batch, num_heads)](
        state,
        x,
        dt,
        A_log,
        B,
        C,
        D,
        dt_bias,
        state_indices,
        state_indices if dst_state_indices is None else dst_state_indices,
        out,
        null_slot,
        int(dst_state_indices is not None),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        dt.stride(0),
        dt.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        0 if dst_state_indices is None else dst_state_indices.stride(0),
        T=T,
        HEAD_DIM=head_dim,
        D_STATE=d_state,
        HEADS_PER_GROUP=num_heads // B.shape[2],
        BLOCK_M=block_m,
        BLOCK_N=triton.next_power_of_2(d_state),
        ENABLE_PDL=enable_pdl,
        num_warps=num_warps,
        **({"launch_pdl": True} if enable_pdl else {}),
    )


@triton.jit(do_not_specialize=["B", "PAYLOAD_LAYER_STRIDE"])
def _mamba2_replay_commit_kernel(
    payload,
    parameters,
    state_addresses,
    state_row_strides,
    read_indices,
    write_indices,
    accepted_length,
    B,
    PAYLOAD_LAYER_STRIDE,
    T: tl.constexpr,
    T_PAD: tl.constexpr,
    GROUPS: tl.constexpr,
    HEADS: tl.constexpr,
    D_STATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STATE_DTYPE_CODE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i_lnh = tl.program_id(0)
    pid_m = tl.program_id(1)
    i_h = i_lnh % HEADS
    i_ln = i_lnh // HEADS
    i_n = i_ln % B
    i_l = i_ln // B
    group = i_h // (HEADS // GROUPS)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < HEAD_DIM
    mask_n = offs_n < D_STATE
    mask = mask_m[:, None] & mask_n[None, :]

    key_width: tl.constexpr = GROUPS * D_STATE
    value_width: tl.constexpr = HEADS * HEAD_DIM
    payload_width: tl.constexpr = key_width + value_width + 2 * HEADS
    row = (
        payload
        + i_l.to(tl.int64) * PAYLOAD_LAYER_STRIDE
        + (i_n * T).to(tl.int64) * payload_width
    )
    p_B = row + group * D_STATE + offs_n
    p_x = row + key_width + i_h * HEAD_DIM + offs_m
    p_dt = row + key_width + value_width + i_h

    layer_request = i_l * B + i_n
    read_idx = tl.load(read_indices + layer_request).to(tl.int64)
    steps = tl.minimum(tl.maximum(tl.load(accepted_length + i_n).to(tl.int32), 0), T)
    state_address = tl.load(state_addresses + i_l)
    if STATE_DTYPE_CODE == 0:
        state_pool = state_address.to(tl.pointer_type(tl.bfloat16))
    elif STATE_DTYPE_CODE == 1:
        state_pool = state_address.to(tl.pointer_type(tl.float16))
    else:
        state_pool = state_address.to(tl.pointer_type(tl.float32))
    state_row_stride = tl.load(state_row_strides + i_l).to(tl.int64)
    tile = i_h * HEAD_DIM * D_STATE + offs_m[:, None] * D_STATE + offs_n[None, :]
    h = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    if read_idx >= 0:
        h += tl.load(
            state_pool + read_idx * state_row_stride + tile, mask=mask, other=0.0
        ).to(tl.float32)

    # All accepted tokens' inputs load at once; the sequential steps then wait on no memory.
    offs_t = tl.arange(0, T_PAD)
    mask_t = offs_t < steps
    xs = tl.load(
        p_x[None, :] + offs_t[:, None] * payload_width,
        mask=mask_t[:, None] & mask_m[None, :],
        other=0.0,
    ).to(tl.float32)
    dts = tl.load(p_dt + offs_t * payload_width, mask=mask_t, other=0.0).to(tl.float32)
    Bs = tl.load(
        p_B[None, :] + offs_t[:, None] * payload_width,
        mask=mask_t[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)

    parameter_base = i_l * 2 * HEADS + i_h
    A = -tl.exp(tl.load(parameters + parameter_base).to(tl.float32))
    bias = tl.load(parameters + parameter_base + HEADS).to(tl.float32)
    for t in tl.static_range(T):
        if t < steps:
            here = offs_t == t
            xv = tl.sum(tl.where(here[:, None], xs, 0.0), axis=0)
            step = _softplus(tl.sum(tl.where(here, dts, 0.0), axis=0) + bias)
            dA = _exp(A * step)
            Bv = tl.sum(tl.where(here[:, None], Bs, 0.0), axis=0)
            dB = Bv * step
            h = h * dA + dB * xv[:, None]
            h = h.to(state_pool.dtype.element_ty).to(tl.float32)

    write_idx = tl.load(write_indices + layer_request).to(tl.int64)
    if write_idx >= 0:
        p_out = state_pool + write_idx * state_row_stride + tile
        tl.store(p_out, h.to(p_out.dtype.element_ty), mask=mask)


@register_kernel(
    "attention",
    "mamba2_replay_commit",
    name="triton_mamba2_replay_commit",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures(("payload",), "dense", {torch.bfloat16}),
    priority=Priority.PORTABLE,
)
def triton_mamba2_replay_commit(
    payload: torch.Tensor,
    parameters: torch.Tensor,
    *,
    state_addresses: torch.Tensor,
    state_row_strides: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    draft_token_num: int,
    geometry: tuple[int, int, int, int],
    state_dtype: torch.dtype,
) -> None:
    groups, heads, d_state, head_dim = geometry
    num_layers = payload.shape[0]
    batch_size = accepted_length.numel()
    block_m, num_warps = _SPEC_BLOCK_M, _SPEC_NUM_WARPS
    enable_pdl = pdl_enabled()
    # Layers x requests x heads exceeds the 65535 cap of grid axes 1 and 2.
    grid = (num_layers * batch_size * heads, triton.cdiv(head_dim, block_m))
    _mamba2_replay_commit_kernel[grid](
        payload,
        parameters,
        state_addresses,
        state_row_strides,
        read_indices,
        write_indices,
        accepted_length,
        batch_size,
        payload.stride(0),
        T=draft_token_num,
        T_PAD=triton.next_power_of_2(draft_token_num),
        GROUPS=groups,
        HEADS=heads,
        D_STATE=d_state,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=triton.next_power_of_2(d_state),
        STATE_DTYPE_CODE={torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}[
            state_dtype
        ],
        ENABLE_PDL=enable_pdl,
        num_warps=num_warps,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
