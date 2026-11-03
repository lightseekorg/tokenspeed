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
# Original recurrence: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li.


"""Frozen pre-RecoverSSM all-layer oracle, excluded from the runtime package.

Keep this arithmetic independent of the candidate's FP32 record producer.
It supplies the main-branch accepted-state reference for differential tests.
"""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
    batched_kda_commit_conv_window_kernel,
)

_SM_COUNT: dict[int, int] = {}


@triton.jit
def _kda_window_step(
    qkv_raw,
    beta,
    tok,
    stride_raw_tok,
    stride_beta_tok,
    i_hv,
    qf,
    kf,
    vf,
    mask_k,
    mask_v,
    w_q0,
    w_q1,
    w_q2,
    w_q3,
    w_k0,
    w_k1,
    w_k2,
    w_k3,
    w_v0,
    w_v1,
    w_v2,
    w_v3,
    s_q0,
    s_q1,
    s_q2,
    s_k0,
    s_k1,
    s_k2,
    s_v0,
    s_v1,
    s_v2,
    b_h,
    g_at,
    scale,
    COMPUTE_OUT: tl.constexpr,
):
    """One KDA token step: conv(+silu), f_b gate GEMV, delta-rule update.

    The single source of the per-token math. Every consumer -- plain verify,
    standalone replay, and the replay prefix fused into the next verify --
    inlines this same body, so their states cannot drift apart numerically.

    Returns the rolled conv window registers, the updated state, and the
    attention output (zeros unless ``COMPUTE_OUT``; replayed positions never
    need it, and skipping it drops the output dot from the critical path).
    """
    x_q = tl.load(qkv_raw + tok * stride_raw_tok + qf, mask=mask_k, other=0.0).to(
        tl.float32
    )
    x_k = tl.load(qkv_raw + tok * stride_raw_tok + kf, mask=mask_k, other=0.0).to(
        tl.float32
    )
    x_v = tl.load(qkv_raw + tok * stride_raw_tok + vf, mask=mask_v, other=0.0).to(
        tl.float32
    )

    acc_q = x_q * w_q3 + s_q0 * w_q0 + s_q1 * w_q1 + s_q2 * w_q2
    acc_k = x_k * w_k3 + s_k0 * w_k0 + s_k1 * w_k1 + s_k2 * w_k2
    acc_v = x_v * w_v3 + s_v0 * w_v0 + s_v1 * w_v1 + s_v2 * w_v2
    b_q = acc_q * tl.sigmoid(acc_q)
    b_k = acc_k * tl.sigmoid(acc_k)
    b_v = acc_v * tl.sigmoid(acc_v)

    # roll the window: after consuming x_t the taps are (x_{t-2}, x_{t-1}, x_t)
    s_q0, s_q1, s_q2 = s_q1, s_q2, x_q
    s_k0, s_k1, s_k2 = s_k1, s_k2, x_k
    s_v0, s_v1, s_v2 = s_v1, s_v2, x_v

    # f_b gate for this token: read it, or run the GEMV inline. Reading wins
    # whenever the caller can amortize the [K, D_FA] weight tile -- held live
    # across the recurrence it costs 128 registers per thread and spills.
    b_gk = tl.load(g_at, mask=mask_k, other=0.0)

    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    b_h *= tl.exp(b_gk)[None, :]
    b_v = b_v - tl.sum(b_h * b_k[None, :], axis=1)
    b_beta = tl.load(beta + tok * stride_beta_tok + i_hv).to(tl.float32)
    b_beta = tl.sigmoid(b_beta)
    b_v *= b_beta
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.zeros_like(b_v)
    if COMPUTE_OUT:
        b_o = tl.sum(b_h * b_q[None, :], axis=1)
    return s_q0, s_q1, s_q2, s_k0, s_k1, s_k2, s_v0, s_v1, s_v2, b_h, b_o


def _sm_count(device) -> int:
    """Multiprocessor count for the device, memoized per index."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    sms = _SM_COUNT.get(index)
    if sms is None:
        sms = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNT[index] = sms
    return sms


def _gate_tiling(rows: int, num_heads: int, head_dim: int, device, layers: int = 1):
    """Rows and gate channels per program: the widest tile the grid can afford.

    Both knobs only decide which rows a program owns -- every row still
    reduces over the whole D_FA axis -- but the tile shape does pick the
    reduction tree, so two tilings agree to a few ulp rather than bit for bit.
    What the choice really moves is the ``[BK, D_FA]`` weight traffic, which
    scales with ``cdiv(rows, block_t)``: a narrower row block buys parallelism
    by reading that tile again in every program, so taking one the grid did
    not need is pure amplification.

    ``layers`` is the batched caller's outer grid dimension. Omitting it made
    the estimate short by that factor and drove the all-layer launch to the
    narrowest tiling at every batch size.
    """
    sms = _sm_count(device)
    for block_t, block_k in ((8, 32), (4, 32), (2, 16), (1, 16)):
        if block_t > rows:
            continue
        programs = (
            layers
            * num_heads
            * triton.cdiv(rows, block_t)
            * triton.cdiv(head_dim, block_k)
        )
        if programs >= 4 * sms:
            return block_t, min(block_k, triton.next_power_of_2(head_dim))
    return 1, min(16, triton.next_power_of_2(head_dim))


def _gate_tiling_dot(rows: int, head_dim: int):
    """Tiling from shape alone, so looped and batched launches match bitwise."""
    return 32 if rows >= 32 else 16, min(32, triton.next_power_of_2(head_dim))


@triton.jit
def batched_kda_gate_precompute_kernel(
    addresses,
    rows,
    stride_fa: tl.constexpr,
    stride_gate: tl.constexpr,
    lower_bound: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    D_FA: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    """Descriptor-table form of the established KDA gate precompute."""
    i_lh, i_tb, i_kb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_l, i_hv = i_lh // HV, i_lh % HV
    ab = i_l * 10
    f_a = tl.load(addresses + ab + 3).to(tl.pointer_type(tl.bfloat16))
    f_b = tl.load(addresses + ab + 4).to(tl.pointer_type(tl.bfloat16))
    A_log = tl.load(addresses + ab + 6).to(tl.pointer_type(tl.float32))
    dt_bias = tl.load(addresses + ab + 7).to(tl.pointer_type(tl.float32))
    gate_scratch = tl.load(addresses + ab + 9).to(tl.pointer_type(tl.float32))
    o_k = i_kb * BK + tl.arange(0, BK)
    o_fa = tl.arange(0, D_FA)
    mask_k = o_k < K
    gc = i_hv * K + o_k
    wfb = tl.load(
        f_b + gc[:, None] * D_FA + o_fa[None, :],
        mask=mask_k[:, None],
        other=0.0,
    ).to(tl.float32)
    b_A = tl.load(A_log + i_hv).to(tl.float32)
    b_bias = tl.load(dt_bias + gc, mask=mask_k, other=0.0).to(tl.float32)
    for j in range(BT):
        row = i_tb * BT + j
        if row < rows:
            fa = tl.load(f_a + row * stride_fa + o_fa).to(tl.float32)
            gate = tl.sum(wfb * fa[None, :], axis=1) + b_bias
            gate = lower_bound * tl.sigmoid(tl.exp(b_A) * gate)
            tl.store(gate_scratch + row * stride_gate + gc, gate, mask=mask_k)


@triton.jit
def batched_kda_gate_precompute_dot_kernel(
    addresses,
    rows,
    stride_fa: tl.constexpr,
    stride_gate: tl.constexpr,
    lower_bound: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    D_FA: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    """Tensor-core form of the batched KDA gate precompute."""
    i_lh, i_tb, i_kb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_l, i_hv = i_lh // HV, i_lh % HV
    ab = i_l * 10
    f_a = tl.load(addresses + ab + 3).to(tl.pointer_type(tl.bfloat16))
    f_b = tl.load(addresses + ab + 4).to(tl.pointer_type(tl.bfloat16))
    A_log = tl.load(addresses + ab + 6).to(tl.pointer_type(tl.float32))
    dt_bias = tl.load(addresses + ab + 7).to(tl.pointer_type(tl.float32))
    gate_scratch = tl.load(addresses + ab + 9).to(tl.pointer_type(tl.float32))
    o_k = i_kb * BK + tl.arange(0, BK)
    o_fa = tl.arange(0, D_FA)
    o_t = tl.arange(0, BT)
    mask_k = o_k < K
    gc = i_hv * K + o_k
    row = i_tb * BT + o_t
    mask_t = row < rows
    # Left as bf16: tl.dot accumulates in fp32, and the product of two bf16
    # values is exact there, so upcasting first would only cost bandwidth.
    wfb = tl.load(
        f_b + gc[:, None] * D_FA + o_fa[None, :],
        mask=mask_k[:, None],
        other=0.0,
    )
    fa = tl.load(
        f_a + row[:, None] * stride_fa + o_fa[None, :],
        mask=mask_t[:, None],
        other=0.0,
    )
    b_A = tl.load(A_log + i_hv).to(tl.float32)
    b_bias = tl.load(dt_bias + gc, mask=mask_k, other=0.0).to(tl.float32)
    gate = tl.dot(fa, tl.trans(wfb)) + b_bias[None, :]
    gate = lower_bound * tl.sigmoid(tl.exp(b_A) * gate)
    tl.store(
        gate_scratch + row[:, None] * stride_gate + gc[None, :],
        gate,
        mask=mask_t[:, None] & mask_k[None, :],
    )


@triton.jit
def batched_recurrent_kda_replay_commit_kernel(
    addresses,
    group_indices,
    read_indices,
    write_indices,
    accepted_length,
    B,
    T: tl.constexpr,
    STRIDE_QKV: tl.constexpr,
    STRIDE_CONV: tl.constexpr,
    STRIDE_BETA: tl.constexpr,
    STRIDE_STATE: tl.constexpr,
    STRIDE_GATE: tl.constexpr,
    CONV_WIDTH: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    D_FA: tl.constexpr,
    BV: tl.constexpr,
):
    """Replay one KDA layer/request/head per program via pointer descriptors."""
    i_l, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # address columns: qkv, conv_w, conv_pool, f_a, f_b, beta, A, dt, state, gate
    ab = i_l * 10
    qkv = tl.load(addresses + ab + 0).to(tl.pointer_type(tl.bfloat16))
    conv_w = tl.load(addresses + ab + 1).to(tl.pointer_type(tl.bfloat16))
    conv_pool = tl.load(addresses + ab + 2).to(tl.pointer_type(tl.bfloat16))
    beta = tl.load(addresses + ab + 5).to(tl.pointer_type(tl.bfloat16))
    state = tl.load(addresses + ab + 8).to(tl.pointer_type(tl.float32))
    gate_scratch = tl.load(addresses + ab + 9).to(tl.pointer_type(tl.float32))
    group = tl.load(group_indices + i_l).to(tl.int64)

    read_page = tl.load(read_indices + group * B + i_n).to(tl.int64)
    write_page = tl.load(write_indices + group * B + i_n).to(tl.int64)
    steps = tl.load(accepted_length + i_n).to(tl.int32)
    steps = tl.minimum(tl.maximum(steps, 0), T)
    read_ok, write_ok = read_page >= 0, write_page >= 0
    P: tl.constexpr = HV * K
    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    qf = i_hv * K + o_k
    kf = P + qf

    w_q0 = tl.load(conv_w + qf * CONV_WIDTH + 0, mask=mask_k, other=0.0).to(tl.float32)
    w_q1 = tl.load(conv_w + qf * CONV_WIDTH + 1, mask=mask_k, other=0.0).to(tl.float32)
    w_q2 = tl.load(conv_w + qf * CONV_WIDTH + 2, mask=mask_k, other=0.0).to(tl.float32)
    w_q3 = tl.load(conv_w + qf * CONV_WIDTH + 3, mask=mask_k, other=0.0).to(tl.float32)
    w_k0 = tl.load(conv_w + kf * CONV_WIDTH + 0, mask=mask_k, other=0.0).to(tl.float32)
    w_k1 = tl.load(conv_w + kf * CONV_WIDTH + 1, mask=mask_k, other=0.0).to(tl.float32)
    w_k2 = tl.load(conv_w + kf * CONV_WIDTH + 2, mask=mask_k, other=0.0).to(tl.float32)
    w_k3 = tl.load(conv_w + kf * CONV_WIDTH + 3, mask=mask_k, other=0.0).to(tl.float32)
    cp = conv_pool + read_page * STRIDE_CONV
    s_q0 = tl.load(cp + qf * (CONV_WIDTH - 1) + 0, mask=mask_k & read_ok, other=0.0).to(
        tl.float32
    )
    s_q1 = tl.load(cp + qf * (CONV_WIDTH - 1) + 1, mask=mask_k & read_ok, other=0.0).to(
        tl.float32
    )
    s_q2 = tl.load(cp + qf * (CONV_WIDTH - 1) + 2, mask=mask_k & read_ok, other=0.0).to(
        tl.float32
    )
    s_k0 = tl.load(cp + kf * (CONV_WIDTH - 1) + 0, mask=mask_k & read_ok, other=0.0).to(
        tl.float32
    )
    s_k1 = tl.load(cp + kf * (CONV_WIDTH - 1) + 1, mask=mask_k & read_ok, other=0.0).to(
        tl.float32
    )
    s_k2 = tl.load(cp + kf * (CONV_WIDTH - 1) + 2, mask=mask_k & read_ok, other=0.0).to(
        tl.float32
    )

    # Each program walks every V tile; conv publication is a later launch.
    for i_v in tl.range(0, K, BV, loop_unroll_factor=1):
        o_v = i_v + tl.arange(0, BV)
        mask_v = o_v < K
        mask_h = mask_v[:, None] & mask_k[None, :]
        vf = 2 * P + i_hv * K + o_v
        w_v0 = tl.load(conv_w + vf * CONV_WIDTH + 0, mask=mask_v, other=0.0).to(
            tl.float32
        )
        w_v1 = tl.load(conv_w + vf * CONV_WIDTH + 1, mask=mask_v, other=0.0).to(
            tl.float32
        )
        w_v2 = tl.load(conv_w + vf * CONV_WIDTH + 2, mask=mask_v, other=0.0).to(
            tl.float32
        )
        w_v3 = tl.load(conv_w + vf * CONV_WIDTH + 3, mask=mask_v, other=0.0).to(
            tl.float32
        )
        s_v0 = tl.load(
            cp + vf * (CONV_WIDTH - 1) + 0, mask=mask_v & read_ok, other=0.0
        ).to(tl.float32)
        s_v1 = tl.load(
            cp + vf * (CONV_WIDTH - 1) + 1, mask=mask_v & read_ok, other=0.0
        ).to(tl.float32)
        s_v2 = tl.load(
            cp + vf * (CONV_WIDTH - 1) + 2, mask=mask_v & read_ok, other=0.0
        ).to(tl.float32)
        state_off = o_v[:, None] * K + o_k[None, :]
        ph = state + read_page * STRIDE_STATE + i_hv * K * K + state_off
        # Matches the window kernel's init; fp32 sums reorder, so close-not-exact.
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
        b_h += tl.load(
            ph, mask=mask_h & read_ok, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)

        tq0, tq1, tq2 = s_q0, s_q1, s_q2
        tk0, tk1, tk2 = s_k0, s_k1, s_k2
        for i_t in range(steps):
            tok = i_n * T + i_t
            tq0, tq1, tq2, tk0, tk1, tk2, s_v0, s_v1, s_v2, b_h, _ = _kda_window_step(
                qkv,
                beta,
                tok,
                STRIDE_QKV,
                STRIDE_BETA,
                i_hv,
                qf,
                kf,
                vf,
                mask_k,
                mask_v,
                w_q0,
                w_q1,
                w_q2,
                w_q3,
                w_k0,
                w_k1,
                w_k2,
                w_k3,
                w_v0,
                w_v1,
                w_v2,
                w_v3,
                tq0,
                tq1,
                tq2,
                tk0,
                tk1,
                tk2,
                s_v0,
                s_v1,
                s_v2,
                b_h,
                gate_scratch + tok * STRIDE_GATE + qf,
                K**-0.5,
                COMPUTE_OUT=False,
            )
        po = state + write_page * STRIDE_STATE + i_hv * K * K + state_off
        tl.store(po, b_h, mask=mask_h & write_ok, eviction_policy="evict_first")


def batched_recurrent_kda_replay_commit(
    addresses: torch.Tensor,
    group_indices: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    *,
    draft_token_num: int,
    num_heads: int,
    head_dim: int,
    f_a_dim: int,
    qkv_stride: int,
    conv_stride: int,
    f_a_stride: int,
    beta_stride: int,
    state_stride: int,
    gate_stride: int,
    conv_width: int,
    lower_bound: float,
) -> None:
    """Commit descriptor-table KDA layers using an explicit cache-group map."""
    if head_dim != 128:
        raise ValueError("batched KDA replay currently requires head_dim=128")
    layers = addresses.shape[0]
    if group_indices.shape != (layers,):
        raise ValueError(
            f"group_indices must have shape ({layers},), got {group_indices.shape}"
        )
    if group_indices.dtype != torch.int32:
        raise TypeError("group_indices must use torch.int32")
    if group_indices.device != addresses.device:
        raise ValueError("group_indices and addresses must be on the same device")
    if not group_indices.is_contiguous():
        raise ValueError("group_indices must be contiguous")
    batch = accepted_length.numel()
    rows = batch * draft_token_num
    # tl.dot needs 16 rows; tensor cores run this gate 5x faster than tl.sum.
    if rows >= 16:
        block_t, block_k = _gate_tiling_dot(rows, head_dim)
        gate_kernel = batched_kda_gate_precompute_dot_kernel
        gate_warps = 1
    else:
        block_t, block_k = _gate_tiling(
            rows, num_heads, head_dim, addresses.device, layers=layers
        )
        gate_kernel = batched_kda_gate_precompute_kernel
        gate_warps = 1 if block_t >= 4 else 2
    gate_kernel[
        (
            layers * num_heads,
            triton.cdiv(rows, block_t),
            triton.cdiv(head_dim, block_k),
        )
    ](
        addresses,
        rows,
        stride_fa=f_a_stride,
        stride_gate=gate_stride,
        lower_bound=lower_bound,
        HV=num_heads,
        K=head_dim,
        D_FA=f_a_dim,
        BK=block_k,
        BT=block_t,
        num_warps=gate_warps,
    )
    batched_recurrent_kda_replay_commit_kernel[(layers, batch, num_heads)](
        addresses,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        batch,
        T=draft_token_num,
        STRIDE_QKV=qkv_stride,
        STRIDE_CONV=conv_stride,
        STRIDE_BETA=beta_stride,
        STRIDE_STATE=state_stride,
        STRIDE_GATE=gate_stride,
        CONV_WIDTH=conv_width,
        HV=num_heads,
        K=head_dim,
        BK=triton.next_power_of_2(head_dim),
        D_FA=f_a_dim,
        BV=32,
        num_warps=1,
        num_stages=2,
    )
    # Swept 8192..64 at every batch size: 128 wins from batch 8 up (3x the
    # whole-row block at 32) and ties the best within noise at batch 1, while
    # 64 falls off a cliff. A fixed block beats sizing to the grid, which
    # stopped narrowing while the wins were still there.
    conv_dim = 3 * num_heads * head_dim
    conv_block = 128
    typed = torch.empty(0, dtype=torch.bfloat16, device=addresses.device)
    batched_kda_commit_conv_window_kernel[
        (layers, batch, triton.cdiv(conv_dim, conv_block))
    ](
        addresses,
        typed,
        typed,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        batch,
        T=draft_token_num,
        STRIDE_QKV=qkv_stride,
        STRIDE_CONV=conv_stride,
        CONV_DIM=conv_dim,
        BLOCK=conv_block,
        num_warps=1,
    )
