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

"""Fused Triton kernels for MoE LoRA applied to sorted expert outputs.

Targets the sglang_shared adapter format (shared outer A, per-expert inner B
for gate/up; per-expert A, shared outer B for down), operating directly on the
sorted token-expert buffers produced by the MoE dispatcher.

Gate/up expand replaces: all-experts B GEMM (m×R × R×E·I) + candidates.gather +
_add_route_delta with a single per-sorted-position GEMV kernel.

Down shrink replaces: _route_rows_from_cache + _select_expert_weights + einsum
with a per-sorted-position GEMV kernel; the caller then runs one shared-B GEMM
and scatter_add_ to accumulate into the token-ordered down output.

Both kernels tile over the rank dimension in BLOCK_R chunks so that register
pressure stays bounded regardless of adapter rank (r=16 to r=256).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ── Gate/Up Expand ───────────────────────────────────────────────────────────
#
# For each sorted position s:
#   exp    = safe_ids[flat_j // K, flat_j % K]  where flat_j = sorted_token_ids[s]
#   delta  = lora_a_m[flat_j // K, :] @ w13_B[exp, offs_i, :].T * scaling
#   gate_up_output[s, offs_i] += delta
#
# Rank dimension is reduced in BLOCK_R tiles to bound register usage.
# Grid: (cdiv(I2, BLOCK_I), padded)


@triton.jit
def _sorted_gate_up_b_expand_kernel(
    lora_a_m,  # (m, MAX_R)
    w13_B,  # (E, I2, MAX_R) — contiguous
    safe_ids,  # (m, K) int64
    sorted_token_ids,  # (padded,) int64  — sorted pos → flat pair
    gate_up_output,  # output — in-place add
    scaling_ptr,  # float32 scalar on device
    route_count,  # int32 — m*K
    K,  # int32
    I2: tl.constexpr,
    MAX_R: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SCATTER: tl.constexpr,  # True: write to flat_j (flat-pair output); False: write to pid_s (sorted output)
):
    pid_i = tl.program_id(0)
    pid_s = tl.program_id(1)

    flat_j = tl.load(sorted_token_ids + pid_s)
    if flat_j < 0:
        return
    if flat_j >= route_count:
        return

    tok = flat_j // K
    topk_v = flat_j % K
    exp = tl.load(safe_ids + tok * K + topk_v).to(tl.int32)

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = offs_i < I2

    scaling = tl.load(scaling_ptr).to(tl.float32)
    acc = tl.zeros((BLOCK_I,), dtype=tl.float32)

    for r_start in range(0, MAX_R, BLOCK_R):
        kr = r_start + tl.arange(0, BLOCK_R)
        la = tl.load(lora_a_m + tok * MAX_R + kr).to(tl.float32)  # (BLOCK_R,)
        B_ptr = w13_B + (exp * I2 + offs_i[:, None]) * MAX_R + kr[None, :]
        B = tl.load(B_ptr, mask=i_mask[:, None], other=0.0).to(
            tl.float32
        )  # (BLOCK_I, BLOCK_R)
        acc += tl.sum(B * la[None, :], axis=1)

    # SCATTER=True: write to flat-pair position flat_j (non-TMA, flat-pair output).
    # SCATTER=False: write to sorted position pid_s (TMA sorted output).
    out_row = flat_j if SCATTER else pid_s
    out_ptr = gate_up_output + out_row * I2 + offs_i
    old = tl.load(out_ptr, mask=i_mask, other=0.0).to(tl.float32)
    tl.store(out_ptr, old + acc * scaling, mask=i_mask)


def _choose_block_r(max_r: int) -> int:
    """Largest power-of-2 ≤ 32 that divides max_r."""
    block_r = min(32, max_r)
    while max_r % block_r != 0:
        block_r //= 2
    return max(block_r, 1)


def sorted_gate_up_b_expand(
    lora_a_m: torch.Tensor,  # (m, R) — already computed
    w13_B: torch.Tensor,  # (E, I2, R) — per-expert B, contiguous
    safe_ids: torch.Tensor,  # (m, K) int64
    sorted_token_ids: torch.Tensor,  # (padded,) int64
    gate_up_output: torch.Tensor,  # (padded, I2) — in-place add
    scaling: torch.Tensor,  # () or (1,) float32 device tensor
    route_count: int,  # = m*K
    K: int,
    BLOCK_I: int = 64,
) -> None:
    """Fused gate/up expand: lora_a_m @ B[expert].T, add directly to sorted output.

    For TMA-sorted dispatch: output is in sorted expert order (SCATTER=False).
    """
    padded, I2 = gate_up_output.shape
    MAX_R = w13_B.shape[2]
    BLOCK_R = _choose_block_r(MAX_R)
    assert w13_B.is_contiguous(), "w13_B must be contiguous for fused kernel"

    grid = (triton.cdiv(I2, BLOCK_I), padded)
    _sorted_gate_up_b_expand_kernel[grid](
        lora_a_m,
        w13_B,
        safe_ids.to(torch.int64),
        sorted_token_ids.to(torch.int64),
        gate_up_output,
        scaling,
        route_count,
        K,
        I2=I2,
        MAX_R=MAX_R,
        BLOCK_I=BLOCK_I,
        BLOCK_R=BLOCK_R,
        SCATTER=False,
        num_warps=4,
        num_stages=3,
    )


# ── Flat Gate/Up Expand (decode path) ────────────────────────────────────────
#
# No sorted_token_ids needed — computes tok = pid_s // K inside the kernel.
# One block per flat-pair position, processes all m*K positions directly.
# Replaces: all-experts B GEMM + candidates.gather + route_delta (3 → 1 kernel).
# Active-expert reads: only the ~51 unique experts' B rows, not all 128.


@triton.jit
def _gate_up_b_expand_kernel(
    lora_a_m,  # (m, MAX_R)
    w13_B_buffer,  # full buffer: n_slots × E × I2 × MAX_R (contiguous)
    slot_ptr,  # (1,) int32 — GPU scalar, dynamic at CUDA-graph replay
    n_slot_stride,  # int — E × I2 × MAX_R (stride between slots)
    safe_ids,  # (m, K) int64
    gate_up_output,  # (m*K, I2) — flat-pair order, in-place add
    scaling_ptr,  # float32 scalar on device
    K,  # int32 — topk count
    I2: tl.constexpr,
    MAX_R: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_s = tl.program_id(1)  # flat-pair index [0 .. m*K-1]

    tok = pid_s // K
    topk_v = pid_s % K
    exp = tl.load(safe_ids + tok * K + topk_v).to(tl.int32)

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = offs_i < I2

    # Load slot index dynamically (changes at CUDA-graph replay without re-capture).
    slot = tl.load(slot_ptr).to(tl.int32)
    # Load scaling from buffer at [slot] — avoids a separate scalings[slot_idx] gather.
    scaling = tl.load(scaling_ptr + slot).to(tl.float32)
    acc = tl.zeros((BLOCK_I,), dtype=tl.float32)

    for r_start in range(0, MAX_R, BLOCK_R):
        kr = r_start + tl.arange(0, BLOCK_R)
        la = tl.load(lora_a_m + tok * MAX_R + kr).to(tl.float32)
        # Compute B pointer directly into the full buffer using the slot offset,
        # avoiding a separate gather copy: buffer[slot, exp, offs_i, kr].
        B_ptr = (
            w13_B_buffer
            + slot * n_slot_stride
            + (exp * I2 + offs_i[:, None]) * MAX_R
            + kr[None, :]
        )
        B = tl.load(B_ptr, mask=i_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(B * la[None, :], axis=1)

    out_ptr = gate_up_output + pid_s * I2 + offs_i
    old = tl.load(out_ptr, mask=i_mask, other=0.0).to(tl.float32)
    tl.store(out_ptr, old + acc * scaling, mask=i_mask)


def gate_up_b_expand(
    lora_a_m: torch.Tensor,  # (m, R) — already computed
    w13_B_buffer: torch.Tensor,  # (n_slots, E, I2, R) — full buffer, contiguous
    slot_idx: torch.Tensor,  # (1,) int32 — GPU tensor; dynamic at CUDA-graph replay
    safe_ids: torch.Tensor,  # (m, K) int64 — expert assignments
    gate_up_output: torch.Tensor,  # (m*K, I2) — flat-pair order, in-place add
    scalings: torch.Tensor,  # (n_slots,) float32 — full scalings buffer; kernel loads [slot]
    BLOCK_I: int = 64,
) -> None:
    """Flat per-expert GEMV for decode (no TMA, no sorted_token_ids needed).

    Accepts the FULL (n_slots, E, I2, R) buffer, slot_idx, and the full scalings
    buffer — the kernel loads both w13_B and scalings via the slot offset, eliminating
    the separate w13_B gather (~38 µs) and scalings gather (~19 µs) per layer.

    One block per flat-pair position; computes tok = pid_s // K directly.
    Replaces: all-experts B GEMM + candidates.gather + route_delta (3 → 1 kernel).
    """
    m_k, I2 = gate_up_output.shape
    K = safe_ids.shape[1]
    # Buffer layout: (n_slots, E, I2, MAX_R).
    _n_slots, E, _I2, MAX_R = w13_B_buffer.shape
    n_slot_stride = E * I2 * MAX_R  # elements between consecutive slots
    BLOCK_R = _choose_block_r(MAX_R)
    assert (
        w13_B_buffer.is_contiguous()
    ), "w13_B_buffer must be contiguous for fused kernel"

    grid = (triton.cdiv(I2, BLOCK_I), m_k)
    _gate_up_b_expand_kernel[grid](
        lora_a_m,
        w13_B_buffer,
        slot_idx.to(torch.int32),
        n_slot_stride,
        safe_ids.to(torch.int64),
        gate_up_output,
        scalings,
        K,
        I2=I2,
        MAX_R=MAX_R,
        BLOCK_I=BLOCK_I,
        BLOCK_R=BLOCK_R,
        num_warps=4,
        num_stages=3,
    )


# ── Down Shrink ───────────────────────────────────────────────────────────────
#
# For each sorted position s, for each rank tile pid_r:
#   exp           = safe_ids[flat_j // K, flat_j % K]
#   lora_a_out[s, pid_r*BLOCK_R : (pid_r+1)*BLOCK_R]
#     = intermediate[s, :] @ down_A[exp, pid_r*BLOCK_R : ..., :].T
#
# Grid: (padded, cdiv(MAX_R, BLOCK_R))
# Splitting over rank tiles keeps (BLOCK_R × BLOCK_H) loads bounded in size.


@triton.jit
def _sorted_a_down_shrink_kernel(
    intermediate,  # (padded, INTER)
    down_A,  # (E, MAX_R, INTER) — per-expert A, contiguous
    safe_ids,  # (m, K) int64
    sorted_token_ids,  # (padded,) int64
    lora_a_out,  # (padded, MAX_R)
    route_count,  # int32
    K,  # int32
    INTER: tl.constexpr,
    MAX_R: tl.constexpr,
    BLOCK_R: tl.constexpr,  # rank output tile; MAX_R divisible by BLOCK_R
    BLOCK_H: tl.constexpr,  # INTER tile; INTER divisible by BLOCK_H
):
    pid_s = tl.program_id(0)
    pid_r = tl.program_id(1)

    flat_j = tl.load(sorted_token_ids + pid_s)
    valid = (flat_j >= 0) & (flat_j < route_count)

    kr = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)

    if not valid:
        tl.store(
            lora_a_out + pid_s * MAX_R + kr,
            tl.zeros((BLOCK_R,), dtype=intermediate.dtype.element_ty),
        )
        return

    tok = flat_j // K
    topk_v = flat_j % K
    exp = tl.load(safe_ids + tok * K + topk_v).to(tl.int32)

    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)

    for h_start in range(0, INTER, BLOCK_H):
        kh = h_start + tl.arange(0, BLOCK_H)
        x = tl.load(intermediate + pid_s * INTER + kh).to(tl.float32)  # (BLOCK_H,)
        A_ptr = down_A + (exp * MAX_R + kr[:, None]) * INTER + kh[None, :]
        A = tl.load(A_ptr).to(tl.float32)  # (BLOCK_R, BLOCK_H)
        acc += tl.sum(A * x[None, :], axis=1)

    tl.store(
        lora_a_out + pid_s * MAX_R + kr,
        acc.to(intermediate.dtype.element_ty),
    )


def _choose_block_h(inter: int) -> int:
    """Largest power-of-2 ≤ 128 that divides inter."""
    block_h = min(128, inter)
    while inter % block_h != 0:
        block_h //= 2
    return max(block_h, 1)


def sorted_a_down_shrink(
    intermediate: torch.Tensor,  # (padded, INTER)
    down_A: torch.Tensor,  # (E, MAX_R, INTER)
    safe_ids: torch.Tensor,  # (m, K) int64
    sorted_token_ids: torch.Tensor,  # (padded,) int64
    route_count: int,
    K: int,
) -> torch.Tensor:
    """Fused down shrink: intermediate[s] @ down_A[expert].T for each sorted pos."""
    padded, INTER = intermediate.shape
    MAX_R = down_A.shape[1]
    BLOCK_R = _choose_block_r(MAX_R)
    BLOCK_H = _choose_block_h(INTER)
    assert down_A.is_contiguous(), "down_A must be contiguous for fused kernel"

    lora_a = torch.empty(
        (padded, MAX_R), dtype=intermediate.dtype, device=intermediate.device
    )
    grid = (padded, MAX_R // BLOCK_R)
    _sorted_a_down_shrink_kernel[grid](
        intermediate,
        down_A,
        safe_ids.to(torch.int64),
        sorted_token_ids.to(torch.int64),
        lora_a,
        route_count,
        K,
        INTER=INTER,
        MAX_R=MAX_R,
        BLOCK_R=BLOCK_R,
        BLOCK_H=BLOCK_H,
        num_warps=4,
        num_stages=2,
    )
    return lora_a


# ── Flat Down Shrink (decode path) ────────────────────────────────────────────
#
# No sorted_token_ids needed — computes tok = pid_s // K inside the kernel.
# One block per (flat-pair, rank-tile), replaces: select_A gather + einsum.
# Avoids the (m*K, r, INTER) intermediate created by _select_expert_weights.
# Grid: (m*K, MAX_R // BLOCK_R)


@triton.jit
def _per_expert_a_shrink_kernel(
    route_input,  # (m*K, INTER)
    down_A_buffer,  # full buffer: n_slots × E × MAX_R × INTER (contiguous)
    slot_ptr,  # (1,) int32 — GPU scalar, dynamic at CUDA-graph replay
    n_slot_stride,  # int — E × MAX_R × INTER (stride between slots)
    safe_ids,  # (m, K) int64
    lora_a_out,  # (m*K, MAX_R)
    K,  # int32
    INTER: tl.constexpr,
    MAX_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_s = tl.program_id(0)  # flat-pair index
    pid_r = tl.program_id(1)  # rank tile

    tok = pid_s // K
    topk_v = pid_s % K
    exp = tl.load(safe_ids + tok * K + topk_v).to(tl.int32)

    # Load slot index dynamically (changes at CUDA-graph replay without re-capture).
    slot = tl.load(slot_ptr).to(tl.int32)

    kr = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)

    for h_start in range(0, INTER, BLOCK_H):
        kh = h_start + tl.arange(0, BLOCK_H)
        x = tl.load(route_input + pid_s * INTER + kh).to(tl.float32)
        # Compute A pointer directly into the full buffer using the slot offset,
        # avoiding a separate gather copy: buffer[slot, exp, kr, kh].
        A_ptr = (
            down_A_buffer
            + slot * n_slot_stride
            + (exp * MAX_R + kr[:, None]) * INTER
            + kh[None, :]
        )
        A = tl.load(A_ptr).to(tl.float32)
        acc += tl.sum(A * x[None, :], axis=1)

    tl.store(lora_a_out + pid_s * MAX_R + kr, acc.to(route_input.dtype.element_ty))


def per_expert_a_shrink(
    route_input: torch.Tensor,  # (m*K, INTER) — flat-pair intermediate
    down_A_buffer: torch.Tensor,  # (n_slots, E, MAX_R, INTER) — full buffer, contiguous
    slot_idx: torch.Tensor,  # (1,) int32 — GPU tensor; dynamic at CUDA-graph replay
    safe_ids: torch.Tensor,  # (m, K) int64
    out: torch.Tensor | None = None,  # optional pre-allocated (m*K, MAX_R) output
) -> torch.Tensor:
    """Flat per-expert shrink for decode: route_input[j] @ down_A_buffer[slot, exp[j]].T.

    Accepts the FULL (n_slots, E, MAX_R, INTER) buffer and a GPU scalar slot_idx,
    computing the slot offset inside the kernel.  This eliminates the separate
    gather copy ``down_A = buffer[slot_idx].squeeze(0)`` (saves ~64 µs/layer).

    Replaces _select_expert_weights gather + einsum without any sorted_token_ids.
    Returns lora_a (m*K, MAX_R) for the subsequent shared-B GEMM or shared_b_down_expand.
    """
    m_k, INTER = route_input.shape
    # Buffer layout: (n_slots, E, MAX_R, INTER).
    _n_slots, E, MAX_R, _INTER = down_A_buffer.shape
    n_slot_stride = E * MAX_R * INTER  # elements between consecutive slots
    BLOCK_R = _choose_block_r(MAX_R)
    BLOCK_H = _choose_block_h(INTER)
    assert down_A_buffer.is_contiguous(), "down_A_buffer must be contiguous"

    if out is None:
        lora_a = torch.empty(
            (m_k, MAX_R), dtype=route_input.dtype, device=route_input.device
        )
    else:
        lora_a = out
    grid = (m_k, MAX_R // BLOCK_R)
    _per_expert_a_shrink_kernel[grid](
        route_input,
        down_A_buffer,
        slot_idx.to(torch.int32),
        n_slot_stride,
        safe_ids.to(torch.int64),
        lora_a,
        safe_ids.shape[1],
        INTER=INTER,
        MAX_R=MAX_R,
        BLOCK_R=BLOCK_R,
        BLOCK_H=BLOCK_H,
        num_warps=4,
        num_stages=2,
    )
    return lora_a


# ── Flat Down Expand (decode path) ────────────────────────────────────────────
#
# Fused kernel that takes the lora_a output from per_expert_a_shrink and performs
# the shared-B GEMM + topk scaling + accumulation in a single pass.
# Avoids: separate down_B gather copy + standalone GEMM + scale + add.
#
# For each (token, topk_v) pair and each hidden chunk:
#   lora_a_row  = lora_a[tok*K + topk_v, :]               — (MAX_R,)
#   B_row       = down_B_buffer[slot, 0, offs_h, :]        — (BLOCK_H, MAX_R)
#   delta_h     = lora_a_row @ B_row.T                     — (BLOCK_H,)
#   out[tok, topk_v, offs_h] += delta_h * topk_weights[tok, topk_v] * scaling
#
# Grid: (m*K, cdiv(H, BLOCK_H))


@triton.jit
def _shared_b_down_expand_kernel(
    lora_a,  # (m*K, MAX_R)
    down_B_buffer,  # full buffer: n_slots × 1 × H × MAX_R (contiguous)
    slot_ptr,  # (1,) int32 — GPU scalar, dynamic at CUDA-graph replay
    n_slot_stride_B,  # int — H × MAX_R (stride between slots; shared-B has dim0=1)
    topk_weights,  # (m, K) — topk routing weights
    scaling_ptr,  # float32 scalar on device
    down_output,  # (m, K, H) — in-place add
    K,  # int32 — topk count
    H: tl.constexpr,  # hidden dimension (constexpr for tl.arange)
    MAX_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_s = tl.program_id(0)  # flat-pair index [0 .. m*K-1]
    pid_h = tl.program_id(1)  # hidden chunk index

    tok = pid_s // K
    topk_v = pid_s % K

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H

    # Load slot index dynamically (changes at CUDA-graph replay without re-capture).
    slot = tl.load(slot_ptr).to(tl.int32)
    # Load scaling from buffer at [slot] — avoids a separate scalings[slot_idx] gather.
    scaling = tl.load(scaling_ptr + slot).to(tl.float32)
    weight = tl.load(topk_weights + tok * K + topk_v).to(tl.float32)

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for r_start in range(0, MAX_R, BLOCK_R):
        kr = r_start + tl.arange(0, BLOCK_R)
        # Load lora_a row tile: lora_a[pid_s, kr].
        la = tl.load(lora_a + pid_s * MAX_R + kr).to(tl.float32)  # (BLOCK_R,)
        # Load B tile directly from buffer: buffer[slot, 0, offs_h, kr].
        # n_slot_stride_B = H × MAX_R (shared-B has expert-dim=1 so no expert offset).
        B_ptr = (
            down_B_buffer
            + slot * n_slot_stride_B
            + offs_h[:, None] * MAX_R
            + kr[None, :]
        )
        B = tl.load(B_ptr, mask=h_mask[:, None], other=0.0).to(
            tl.float32
        )  # (BLOCK_H, BLOCK_R)
        # delta_h += B @ la  (contract over rank dimension)
        acc += tl.sum(B * la[None, :], axis=1)

    # Scale by topk weight and adapter scaling, then accumulate.
    out_ptr = down_output + (tok * K + topk_v) * H + offs_h
    old = tl.load(out_ptr, mask=h_mask, other=0.0).to(tl.float32)
    tl.store(out_ptr, old + acc * weight * scaling, mask=h_mask)


def _choose_block_h_expand(h: int) -> int:
    """Largest power-of-2 ≤ 64 that divides h (or is the largest divisor ≤ 64)."""
    block_h = min(64, h)
    while h % block_h != 0:
        block_h //= 2
    return max(block_h, 1)


def shared_b_down_expand(
    lora_a: torch.Tensor,  # (m*K, MAX_R) — output of per_expert_a_shrink
    down_B_buffer: torch.Tensor,  # (n_slots, 1, H, MAX_R) — full buffer, contiguous
    slot_idx: torch.Tensor,  # (1,) int32 — GPU tensor; dynamic at CUDA-graph replay
    down_output: torch.Tensor,  # (m, K, H) or (m*K, H) — in-place add
    topk_weights: torch.Tensor,  # (m, K) routing weights
    scalings: torch.Tensor,  # (n_slots,) float32 — full scalings buffer; kernel loads [slot]
    K: int,
) -> None:
    """Fused down expand for decode: lora_a @ down_B[slot, 0].T × weight × scaling.

    Accepts the FULL (n_slots, 1, H, MAX_R) buffer, slot_idx, and the full scalings
    buffer — eliminates the separate down_B gather and scalings gather per layer.

    Performs the shared-B GEMM, topk-weight scaling, and accumulation into
    down_output in a single fused kernel.
    """
    m_k, MAX_R = lora_a.shape
    # Buffer layout: (n_slots, 1, H, MAX_R).
    _n_slots, _one, H, _MAX_R = down_B_buffer.shape
    # Stride between slots: only 1 expert-slot for shared B, so stride = 1 × H × MAX_R.
    n_slot_stride_B = H * MAX_R
    BLOCK_H = _choose_block_h_expand(H)
    BLOCK_R = _choose_block_r(MAX_R)
    assert (
        down_B_buffer.is_contiguous()
    ), "down_B_buffer must be contiguous for fused kernel"

    # Reshape output to (m*K, H) so the kernel can use a flat pid_s index.
    out_flat = down_output.view(m_k, H)

    grid = (m_k, triton.cdiv(H, BLOCK_H))
    _shared_b_down_expand_kernel[grid](
        lora_a,
        down_B_buffer,
        slot_idx.to(torch.int32),
        n_slot_stride_B,
        topk_weights,
        scalings,
        out_flat,
        K,
        H=H,
        MAX_R=MAX_R,
        BLOCK_H=BLOCK_H,
        BLOCK_R=BLOCK_R,
        num_warps=4,
        num_stages=3,
    )


# ── Flat A GEMM (decode path) ─────────────────────────────────────────────────
#
# Computes lora_a_m = hidden @ w13_A[slot, 0, :, :].T for each token,
# reading directly from the buffer without a prior gather copy.
# Replaces: w13_A gather (22 µs) + cuBLAS GEMM (25 µs) → ~5-8 µs per layer.
#
# Grid: (m, MAX_R // BLOCK_R)  — one block per (token, rank-tile)


@triton.jit
def _shared_a_shrink_kernel(
    hidden,  # (m, H)
    w13_A_buffer,  # full buffer: n_slots × 1 × MAX_R × H (contiguous)
    slot_ptr,  # (1,) int32 — GPU scalar, dynamic at CUDA-graph replay
    n_slot_stride_A,  # int — MAX_R × H (stride between slots; shared outer has 1 row)
    lora_a_out,  # (m, MAX_R)
    H: tl.constexpr,
    MAX_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)  # token index
    pid_r = tl.program_id(1)  # rank tile

    slot = tl.load(slot_ptr).to(tl.int32)
    kr = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)

    for h_start in range(0, H, BLOCK_H):
        kh = h_start + tl.arange(0, BLOCK_H)
        x = tl.load(hidden + pid_m * H + kh).to(tl.float32)  # (BLOCK_H,)
        # buffer[slot, 0, kr, kh]: stride = slot * n_slot_stride_A + kr * H + kh
        A_ptr = w13_A_buffer + slot * n_slot_stride_A + kr[:, None] * H + kh[None, :]
        A = tl.load(A_ptr).to(tl.float32)  # (BLOCK_R, BLOCK_H)
        acc += tl.sum(A * x[None, :], axis=1)

    tl.store(lora_a_out + pid_m * MAX_R + kr, acc.to(hidden.dtype.element_ty))


def shared_a_shrink(
    hidden: torch.Tensor,  # (m, H)
    w13_A_buffer: torch.Tensor,  # (n_slots, 1, MAX_R, H) — full buffer
    slot_idx: torch.Tensor,  # (1,) int32 GPU tensor
    BLOCK_H: int = 128,
) -> torch.Tensor:
    """Compute lora_a_m = hidden @ w13_A_buffer[slot, 0, :, :].T without gather.

    Replaces: w13_A gather (22 µs) + cuBLAS GEMM (25 µs) = 47 µs per layer
    With: single Triton kernel (~5-8 µs), saving ~40 µs × 48 = 1.9 ms.
    """
    m, H = hidden.shape
    _n_slots, _one, MAX_R, _H = w13_A_buffer.shape
    n_slot_stride_A = MAX_R * H  # stride between slots (1 × MAX_R × H)
    BLOCK_R = _choose_block_r(MAX_R)

    lora_a = torch.empty((m, MAX_R), dtype=hidden.dtype, device=hidden.device)
    grid = (m, MAX_R // BLOCK_R)
    _shared_a_shrink_kernel[grid](
        hidden,
        w13_A_buffer,
        slot_idx.to(torch.int32),
        n_slot_stride_A,
        lora_a,
        H=H,
        MAX_R=MAX_R,
        BLOCK_R=BLOCK_R,
        BLOCK_H=BLOCK_H,
        num_warps=4,
        num_stages=2,
    )
    return lora_a


# ── Per-Expert Gate/Up Expand ─────────────────────────────────────────────────
#
# Like gate_up_b_expand but reads lora_a_flat[pid_s] (per flat-pair position)
# instead of lora_a_m[tok] (shared per token).  Required for per_expert adapters
# where each expert has its own A matrix → lora_a differs per (token, topk_v) pair.
#
# Grid: (cdiv(I2, BLOCK_I), m*K)


@triton.jit
def _per_expert_gate_up_b_expand_kernel(
    lora_a_flat,  # (m*K, MAX_R) — per flat-pair lora_a (from per_expert_a_shrink w/ hidden)
    w13_B_buffer,  # full buffer: n_slots × E × I2 × MAX_R (contiguous)
    slot_ptr,  # (1,) int32
    n_slot_stride,  # E × I2 × MAX_R
    safe_ids,  # (m, K) int64
    gate_up_output,  # (m*K, I2) — in-place add
    scaling_ptr,  # (n_slots,) float32
    K,
    I2: tl.constexpr,
    MAX_R: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_s = tl.program_id(1)  # flat-pair index [0 .. m*K-1]

    tok = pid_s // K
    topk_v = pid_s % K
    exp = tl.load(safe_ids + tok * K + topk_v).to(tl.int32)

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = offs_i < I2

    slot = tl.load(slot_ptr).to(tl.int32)
    scaling = tl.load(scaling_ptr + slot).to(tl.float32)
    acc = tl.zeros((BLOCK_I,), dtype=tl.float32)

    for r_start in range(0, MAX_R, BLOCK_R):
        kr = r_start + tl.arange(0, BLOCK_R)
        # Per-position lora_a: lora_a_flat[pid_s] instead of lora_a_m[tok]
        la = tl.load(lora_a_flat + pid_s * MAX_R + kr).to(tl.float32)
        B_ptr = (
            w13_B_buffer
            + slot * n_slot_stride
            + (exp * I2 + offs_i[:, None]) * MAX_R
            + kr[None, :]
        )
        B = tl.load(B_ptr, mask=i_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(B * la[None, :], axis=1)

    out_ptr = gate_up_output + pid_s * I2 + offs_i
    old = tl.load(out_ptr, mask=i_mask, other=0.0).to(tl.float32)
    tl.store(out_ptr, old + acc * scaling, mask=i_mask)


def per_expert_gate_up_b_expand(
    lora_a_flat: torch.Tensor,  # (m*K, MAX_R) — from per_expert_a_shrink(hidden_flat, w13_A_buf, ...)
    w13_B_buffer: torch.Tensor,  # (n_slots, E, I2, MAX_R) — full buffer
    slot_idx: torch.Tensor,  # (1,) int32 GPU tensor
    safe_ids: torch.Tensor,  # (m, K) int64
    gate_up_output: torch.Tensor,  # (m*K, I2) — in-place add
    scalings: torch.Tensor,  # (n_slots,) float32
    BLOCK_I: int = 64,
) -> None:
    """Per-expert gate/up expand for decode: lora_a_flat[j] @ w13_B[slot, e_j].T.

    Replaces the gather-then-einsum path for per_expert adapters.  Accepts the FULL
    (n_slots, E, I2, MAX_R) buffer and reads the expert offset directly using safe_ids,
    eliminating the two gather copies (w13_B gather + expert-select gather).
    """
    m_k, MAX_R = lora_a_flat.shape
    _n_slots, E, I2, _MAX_R = w13_B_buffer.shape
    n_slot_stride = E * I2 * MAX_R
    BLOCK_R = _choose_block_r(MAX_R)
    K = safe_ids.shape[1]
    assert w13_B_buffer.is_contiguous(), "w13_B_buffer must be contiguous"

    grid = (triton.cdiv(I2, BLOCK_I), m_k)
    _per_expert_gate_up_b_expand_kernel[grid](
        lora_a_flat,
        w13_B_buffer,
        slot_idx.to(torch.int32),
        n_slot_stride,
        safe_ids.to(torch.int64),
        gate_up_output,
        scalings,
        K,
        I2=I2,
        MAX_R=MAX_R,
        BLOCK_I=BLOCK_I,
        BLOCK_R=BLOCK_R,
        num_warps=4,
        num_stages=2,
    )


# ── Per-Expert Down Expand ────────────────────────────────────────────────────
#
# Like shared_b_down_expand but reads per-expert B: down_B_buffer[slot, e_j, offs_h, :].
# Required for per_expert adapters where down_B is per-expert (not shared).
# Eliminates the two gather copies (down_B buffer copy + expert select gather).
#
# Grid: (m*K, cdiv(H, BLOCK_H))


@triton.jit
def _per_expert_b_down_expand_kernel(
    lora_a,  # (m*K, MAX_R)
    down_B_buffer,  # full buffer: n_slots × E × H × MAX_R (contiguous)
    slot_ptr,  # (1,) int32
    n_slot_stride_B,  # E × H × MAX_R
    safe_ids,  # (m, K) int64
    topk_weights,  # (m, K)
    scaling_ptr,  # (n_slots,) float32
    down_output,  # (m, K, H) — in-place add
    K,
    H: tl.constexpr,
    MAX_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)

    tok = pid_s // K
    topk_v = pid_s % K
    exp = tl.load(safe_ids + tok * K + topk_v).to(tl.int32)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H

    slot = tl.load(slot_ptr).to(tl.int32)
    scaling = tl.load(scaling_ptr + slot).to(tl.float32)
    weight = tl.load(topk_weights + tok * K + topk_v).to(tl.float32)

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for r_start in range(0, MAX_R, BLOCK_R):
        kr = r_start + tl.arange(0, BLOCK_R)
        la = tl.load(lora_a + pid_s * MAX_R + kr).to(tl.float32)
        # Per-expert B: buffer[slot, exp, offs_h, kr]
        B_ptr = (
            down_B_buffer
            + slot * n_slot_stride_B
            + (exp * H + offs_h[:, None]) * MAX_R
            + kr[None, :]
        )
        B = tl.load(B_ptr, mask=h_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(B * la[None, :], axis=1)

    out_ptr = down_output + (tok * K + topk_v) * H + offs_h
    old = tl.load(out_ptr, mask=h_mask, other=0.0).to(tl.float32)
    tl.store(out_ptr, old + acc * weight * scaling, mask=h_mask)


def per_expert_b_down_expand(
    lora_a: torch.Tensor,  # (m*K, MAX_R) — from per_expert_a_shrink
    down_B_buffer: torch.Tensor,  # (n_slots, E, H, MAX_R) — full buffer
    slot_idx: torch.Tensor,  # (1,) int32 GPU tensor
    safe_ids: torch.Tensor,  # (m, K) int64
    down_output: torch.Tensor,  # (m, K, H) or (m*K, H) — in-place add
    topk_weights: torch.Tensor,  # (m, K)
    scalings: torch.Tensor,  # (n_slots,) float32
    K: int,
) -> None:
    """Per-expert down expand for decode: lora_a[j] @ down_B[slot, e_j].T × weight.

    Eliminates the two gather copies (down_B buffer copy + expert select gather)
    for per_expert adapters where down_B is per-expert (not shared).
    """
    m_k, MAX_R = lora_a.shape
    _n_slots, E, H, _MAX_R = down_B_buffer.shape
    n_slot_stride_B = E * H * MAX_R
    BLOCK_H = _choose_block_h_expand(H)
    BLOCK_R = _choose_block_r(MAX_R)
    assert down_B_buffer.is_contiguous(), "down_B_buffer must be contiguous"

    out_flat = down_output.view(m_k, H)
    grid = (m_k, triton.cdiv(H, BLOCK_H))
    _per_expert_b_down_expand_kernel[grid](
        lora_a,
        down_B_buffer,
        slot_idx.to(torch.int32),
        n_slot_stride_B,
        safe_ids.to(torch.int64),
        topk_weights,
        scalings,
        out_flat,
        K,
        H=H,
        MAX_R=MAX_R,
        BLOCK_H=BLOCK_H,
        BLOCK_R=BLOCK_R,
        num_warps=4,
        num_stages=2,
    )


# ── Fused A+B Gate/Up (eliminates shared_a_shrink + gate_up_b_expand) ──────
#
# Combines hidden @ w13_A + lora_a @ w13_B in one kernel, removing a separate
# shared_a_shrink launch.  Lora_a is computed per flat-pair block (redundant for
# k>1 per token) but w13_A fits in L1 so cache hits make this negligible.
# Grid: (cdiv(I2, BLOCK_I), m*K)


@triton.jit
def _fused_shared_a_b_gate_up_kernel(
    hidden,  # (m, H)
    w13_A_buffer,  # (n_slots, 1, MAX_R, H) — contiguous
    w13_B_buffer,  # (n_slots, E, I2, MAX_R) — contiguous
    safe_ids,  # (m, K) int64
    gate_up_output,  # (m*K, I2) — in-place add
    scalings,  # (n_slots,) float32
    slot_ptr,  # (1,) int32
    K,
    n_A_stride,  # = MAX_R * H
    n_B_stride,  # = E * I2 * MAX_R
    H: tl.constexpr,
    I2: tl.constexpr,
    MAX_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_i = tl.program_id(0)  # I2 chunk
    pid_s = tl.program_id(1)  # flat-pair index

    slot = tl.load(slot_ptr).to(tl.int32)
    tok = pid_s // K
    topk_v = pid_s % K
    exp = tl.load(safe_ids + tok * K + topk_v).to(tl.int32)
    scaling = tl.load(scalings + slot).to(tl.float32)

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = offs_i < I2
    acc = tl.zeros((BLOCK_I,), dtype=tl.float32)

    # Outer loop over BLOCK_R chunks of rank: compute lora_a[r:r+BLOCK_R] then expand.
    # This avoids storing the full lora_a vector when BLOCK_R < MAX_R.
    for r_start in range(0, MAX_R, BLOCK_R):
        kr = r_start + tl.arange(0, BLOCK_R)

        # Phase 1 (for this rank chunk): la = hidden[tok] @ w13_A[slot, 0, kr, :].T
        la = tl.zeros((BLOCK_R,), dtype=tl.float32)
        for h_start in range(0, H, BLOCK_H):
            kh = h_start + tl.arange(0, BLOCK_H)
            x = tl.load(hidden + tok * H + kh).to(tl.float32)
            A_ptr = w13_A_buffer + slot * n_A_stride + kr[:, None] * H + kh[None, :]
            A = tl.load(A_ptr).to(tl.float32)
            la += tl.sum(A * x[None, :], axis=1)

        # Phase 2 (for this rank chunk): acc += la @ w13_B[slot, exp, offs_i, kr].T
        B_ptr = (
            w13_B_buffer
            + slot * n_B_stride
            + (exp * I2 + offs_i[:, None]) * MAX_R
            + kr[None, :]
        )
        B = tl.load(B_ptr, mask=i_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(B * la[None, :], axis=1)

    out_ptr = gate_up_output + pid_s * I2 + offs_i
    old = tl.load(out_ptr, mask=i_mask, other=0.0).to(tl.float32)
    tl.store(out_ptr, old + acc * scaling, mask=i_mask)


def fused_shared_a_b_gate_up_expand(
    hidden: torch.Tensor,  # (m, H)
    w13_A_buffer: torch.Tensor,  # (n_slots, 1, MAX_R, H)
    w13_B_buffer: torch.Tensor,  # (n_slots, E, I2, MAX_R)
    safe_ids: torch.Tensor,  # (m, K) int64
    gate_up_output: torch.Tensor,  # (m*K, I2) — in-place add
    scalings: torch.Tensor,  # (n_slots,) float32
    slot_idx: torch.Tensor,  # (1,) int32
    BLOCK_I: int = 64,
    BLOCK_H: int = 128,
) -> None:
    """Fused A+B gate/up: eliminates the separate shared_a_shrink kernel launch."""
    m_k, I2 = gate_up_output.shape
    m, H = hidden.shape
    K = safe_ids.shape[1]
    _ns, _one, MAX_R, _H = w13_A_buffer.shape
    _ns2, E, _I2, _MAX_R = w13_B_buffer.shape
    n_A_stride = MAX_R * H
    n_B_stride = E * I2 * MAX_R
    BLOCK_R = _choose_block_r(MAX_R)
    assert w13_A_buffer.is_contiguous() and w13_B_buffer.is_contiguous()

    grid = (triton.cdiv(I2, BLOCK_I), m_k)
    _fused_shared_a_b_gate_up_kernel[grid](
        hidden,
        w13_A_buffer,
        w13_B_buffer,
        safe_ids.to(torch.int64),
        gate_up_output,
        scalings,
        slot_idx.to(torch.int32),
        K,
        n_A_stride,
        n_B_stride,
        H=H,
        I2=I2,
        MAX_R=MAX_R,
        BLOCK_H=BLOCK_H,
        BLOCK_I=BLOCK_I,
        BLOCK_R=BLOCK_R,
        num_warps=4,
        num_stages=2,
    )


# ── Fused Shrink+Expand Down (eliminates per_expert_a_shrink + shared_b_down_expand) ─
#
# Combines ri @ down_A + lora_a @ down_B in one kernel per (flat-pair, H-chunk).
# Grid: (m*K, cdiv(H, BLOCK_H))


@triton.jit
def _fused_a_b_down_expand_kernel(
    route_input,  # (m*K, INTER)
    down_A_buffer,  # (n_slots, E, MAX_R, INTER) — contiguous
    down_B_buffer,  # (n_slots, 1, H, MAX_R) — contiguous
    safe_ids,  # (m, K) int64
    topk_weights,  # (m, K)
    scalings,  # (n_slots,) float32
    slot_ptr,  # (1,) int32
    down_output,  # (m*K, H) — in-place add
    K,
    n_A_stride,  # = E * MAX_R * INTER
    n_B_stride,  # = H * MAX_R
    INTER: tl.constexpr,
    H: tl.constexpr,
    MAX_R: tl.constexpr,
    BLOCK_H_S: tl.constexpr,  # shrink tile over INTER
    BLOCK_H_E: tl.constexpr,  # expand tile over H
):
    pid_s = tl.program_id(0)  # flat-pair index
    pid_h = tl.program_id(1)  # H chunk

    slot = tl.load(slot_ptr).to(tl.int32)
    tok = pid_s // K
    topk_v = pid_s % K
    exp = tl.load(safe_ids + tok * K + topk_v).to(tl.int32)
    weight = tl.load(topk_weights + tok * K + topk_v).to(tl.float32)
    scaling = tl.load(scalings + slot).to(tl.float32)

    offs_h = pid_h * BLOCK_H_E + tl.arange(0, BLOCK_H_E)
    h_mask = offs_h < H
    kr = tl.arange(0, MAX_R)

    # Phase 1: lora_a = ri[pid_s] @ down_A[slot, exp, :, :].T
    lora_a = tl.zeros((MAX_R,), dtype=tl.float32)
    for h_start in range(0, INTER, BLOCK_H_S):
        kh = h_start + tl.arange(0, BLOCK_H_S)
        x = tl.load(route_input + pid_s * INTER + kh).to(tl.float32)
        A_ptr = (
            down_A_buffer
            + slot * n_A_stride
            + (exp * MAX_R + kr[:, None]) * INTER
            + kh[None, :]
        )
        A = tl.load(A_ptr).to(tl.float32)
        lora_a += tl.sum(A * x[None, :], axis=1)

    # Phase 2: delta = lora_a @ down_B[slot, 0, offs_h, :].T * weight * scaling
    B_ptr = down_B_buffer + slot * n_B_stride + offs_h[:, None] * MAX_R + kr[None, :]
    B = tl.load(B_ptr, mask=h_mask[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(B * lora_a[None, :], axis=1) * weight * scaling

    out_ptr = down_output + pid_s * H + offs_h
    old = tl.load(out_ptr, mask=h_mask, other=0.0).to(tl.float32)
    tl.store(out_ptr, old + delta, mask=h_mask)


def fused_a_b_down_expand(
    route_input: torch.Tensor,  # (m*K, INTER)
    down_A_buffer: torch.Tensor,  # (n_slots, E, MAX_R, INTER)
    down_B_buffer: torch.Tensor,  # (n_slots, 1, H, MAX_R)
    safe_ids: torch.Tensor,  # (m, K) int64
    topk_weights: torch.Tensor,  # (m, K)
    scalings: torch.Tensor,  # (n_slots,) float32
    slot_idx: torch.Tensor,  # (1,) int32
    down_output: torch.Tensor,  # (m*K, H) or (m, K, H) — in-place add
    BLOCK_H_E: int = 64,
) -> None:
    """Fused shrink+expand down: eliminates per_expert_a_shrink + shared_b_down_expand launches."""
    m_k, INTER = route_input.shape
    _ns, E, MAX_R, _INTER = down_A_buffer.shape
    _ns2, _one, H, _MAX_R = down_B_buffer.shape
    K = safe_ids.shape[1]
    n_A_stride = E * MAX_R * INTER
    n_B_stride = H * MAX_R
    BLOCK_H_S = _choose_block_h(INTER)
    assert down_A_buffer.is_contiguous() and down_B_buffer.is_contiguous()

    out_flat = down_output.view(m_k, H)
    grid = (m_k, triton.cdiv(H, BLOCK_H_E))
    _fused_a_b_down_expand_kernel[grid](
        route_input,
        down_A_buffer,
        down_B_buffer,
        safe_ids.to(torch.int64),
        topk_weights,
        scalings,
        slot_idx.to(torch.int32),
        out_flat,
        K,
        n_A_stride,
        n_B_stride,
        INTER=INTER,
        H=H,
        MAX_R=MAX_R,
        BLOCK_H_S=BLOCK_H_S,
        BLOCK_H_E=BLOCK_H_E,
        num_warps=4,
        num_stages=2,
    )
