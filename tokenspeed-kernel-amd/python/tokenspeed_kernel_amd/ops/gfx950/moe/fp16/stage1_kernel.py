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

"""Gluon bf16 MoE stage 1: gate + up GEMM with fused SwiGLU (gfx950).

For each MoE expert ``e`` and every token routed to it::

    inter = silu(hidden @ w1_e[:I, :].T) * (hidden @ w1_e[I:, :].T)

Unquantized bf16 stage-1: bf16 activations, bf16 weights, fp32 MFMA
accumulate, SwiGLU epilogue.

The exact block-FP8 decode path copies compact weight bytes directly into LDS,
then scales and rounds them to bf16 in registers immediately before the same
bf16 MFMA. This preserves the materialized-bf16 arithmetic while reducing the
weight pipeline's memory traffic.

GEMM hot loop follows the ROCm/gfx950-gluon-tutorials **a16w16 v4/v9**
design: a 2-stage double-buffered async prefetch pipeline
(``buffer_load_to_shared`` into a ping-pong LDS buffer, consume the
previous buffer while the next fills, ``wait_group(1)``), plus v9's
XCD-aware PID remapping + ``GROUP_SIZE_M`` swizzling for L2 locality
(``_grid.get_pids``). MFMA idiom: ``AMDMFMALayout(version=4,
instr_shape=[16,16,32], k_width=8)`` + ``gl.amd.cdna3.mfma``.

Layout contract:
  hidden_states  (num_tokens, D)            bf16      (A operand, gathered)
  w1             (E, 2 * I, D)              bf16      (gate rows [0:I], up [I:2I])
  out            (num_tokens * topk, I)     bf16      inter_row = token*topk + slot
  sorted_token_ids   (EM,)             int32   packed (slot<<24 | token)
  sorted_expert_ids  (EM // BLOCK_M,)  int32   -1 marks an all-padding block
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import cdna4_async_copy, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16._grid import get_pids


@gluon.jit
def gluon_bf16_moe_stage1_kernel(
    a_ptr,  # hidden_states  (num_tokens, D)  bf16
    b_ptr,  # w1             (E, 2*I, D)      bf16
    b_scale_ptr,
    c_ptr,  # out            (num_tokens*topk, I) bf16
    sorted_token_ids_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    N,  # = 2 * I  (gate||up column count of w1)
    K,  # = D
    EM,
    num_tokens,
    top_k,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_bse,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    GRID_MN,
    I_r: gl.constexpr,  # SwiGLU output width (half of N)
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    WEIGHT_FP8: gl.constexpr,
    FP8_ASYNC: gl.constexpr,
):
    num_pid_m = gl.cdiv(EM, BLOCK_M)
    # Grid covers the SwiGLU output columns ``I_r`` (not the un-fused
    # gate||up width ``N``); each CTA emits both gate and up for its slab.
    num_pid_n = gl.cdiv(I_r, BLOCK_N)
    pid_m, pid_n = get_pids(num_pid_m, num_pid_n, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    num_valid = gl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_M >= num_valid:
        return

    off_experts = gl.load(sorted_expert_ids_ptr + pid_m)
    if off_experts == -1:
        return

    # ---- MFMA + operand layouts (bf16: instr [16,16,32], k_width=8) ----
    # A 16-row sparse tile owns exactly one MFMA row. Keep all four waves on
    # N instead of allocating a second, empty MFMA row as the dense tile does.
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=([1, NUM_WARPS] if BLOCK_M == 16 else [2, NUM_WARPS // 2]),
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )

    # ---- Global-load blocked layouts + swizzled LDS (a16w16 v3/v4) -----
    gload_a: gl.constexpr = gl.BlockedLayout(
        [1, 8], [512 // BLOCK_K, BLOCK_K // 8], [NUM_WARPS, 1], [1, 0]
    )
    if WEIGHT_FP8 and FP8_ASYNC:
        # Direct-to-LDS byte loads require a full 16-byte vector per thread.
        gload_b: gl.constexpr = gl.BlockedLayout(
            [16, 1], [BLOCK_K // 16, 1024 // BLOCK_K], [1, NUM_WARPS], [0, 1]
        )
    else:
        gload_b: gl.constexpr = gl.BlockedLayout(
            [8, 1], [BLOCK_K // 8, 512 // BLOCK_K], [1, NUM_WARPS], [0, 1]
        )
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
    if WEIGHT_FP8 and FP8_ASYNC:
        shared_b: gl.constexpr = gl.SwizzledSharedLayout(16, 2, 8, order=[0, 1])
    else:
        shared_b: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])

    NBUF: gl.constexpr = 2
    smem_a = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NBUF, BLOCK_M, BLOCK_K], shared_a
    )
    if WEIGHT_FP8 and FP8_ASYNC:
        smem_bg = gl.allocate_shared_memory(
            b_ptr.dtype.element_ty, [NBUF, BLOCK_K, BLOCK_N], shared_b
        )
        smem_bu = gl.allocate_shared_memory(
            b_ptr.dtype.element_ty, [NBUF, BLOCK_K, BLOCK_N], shared_b
        )
    else:
        smem_bg = gl.allocate_shared_memory(
            gl.bfloat16, [NBUF, BLOCK_K, BLOCK_N], shared_b
        )
        smem_bu = gl.allocate_shared_memory(
            gl.bfloat16, [NBUF, BLOCK_K, BLOCK_N], shared_b
        )

    # ---- Per-token A-row gather offsets --------------------------------
    am_layout: gl.constexpr = gl.SliceLayout(1, gload_a)
    ak_layout: gl.constexpr = gl.SliceLayout(0, gload_a)
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=am_layout)
    packed = gl.load(sorted_token_ids_ptr + offs_m, mask=offs_m < EM, other=num_tokens)
    a_row = packed & 0xFFFFFF
    token_mask = a_row < num_tokens
    offs_ak = gl.arange(0, BLOCK_K, layout=ak_layout)
    a_offsets = (a_row[:, None] * stride_am + offs_ak[None, :] * stride_ak).to(gl.int32)

    # ---- B (gate + up) offsets: element [k, n] = w1[e][n, k] -----------
    bk_layout: gl.constexpr = gl.SliceLayout(1, gload_b)
    bn_layout: gl.constexpr = gl.SliceLayout(0, gload_b)
    offs_bk = gl.arange(0, BLOCK_K, layout=bk_layout)
    offs_bn_gate = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=bn_layout)
    offs_bn_up = I_r + pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=bn_layout)
    b_off_gate = (offs_bk[:, None] * stride_bk + offs_bn_gate[None, :] * stride_bn).to(
        gl.int32
    )
    b_off_up = (offs_bk[:, None] * stride_bk + offs_bn_up[None, :] * stride_bn).to(
        gl.int32
    )

    a_base = a_ptr
    b_base = b_ptr + off_experts.to(gl.int64) * stride_be

    gate_acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma_layout)
    up_acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma_layout)

    num_k = gl.cdiv(K, BLOCK_K)
    gl.assume(num_k > 1)

    # ---- Prologue: prefetch tile 0 into buffer 0 -----------------------
    cdna4_async_copy.buffer_load_to_shared(
        smem_a.index(0), a_base, a_offsets, mask=token_mask[:, None]
    )
    if WEIGHT_FP8:
        if FP8_ASYNC:
            cdna4_async_copy.buffer_load_to_shared(smem_bg.index(0), b_base, b_off_gate)
            cdna4_async_copy.buffer_load_to_shared(smem_bu.index(0), b_base, b_off_up)
        else:
            scale_n = (pid_n * BLOCK_N) // 128
            scale_bg = gl.load(
                b_scale_ptr + off_experts * stride_bse + scale_n * stride_bsn
            )
            scale_bu = gl.load(
                b_scale_ptr
                + off_experts * stride_bse
                + (I_r // 128 + scale_n) * stride_bsn
            )
            bg_fp8 = gl.load(b_base + b_off_gate).to(gl.float8e4nv, bitcast=True)
            bu_fp8 = gl.load(b_base + b_off_up).to(gl.float8e4nv, bitcast=True)
            smem_bg.index(0).store((bg_fp8.to(gl.float32) * scale_bg).to(gl.bfloat16))
            smem_bu.index(0).store((bu_fp8.to(gl.float32) * scale_bu).to(gl.bfloat16))
    else:
        cdna4_async_copy.buffer_load_to_shared(smem_bg.index(0), b_base, b_off_gate)
        cdna4_async_copy.buffer_load_to_shared(smem_bu.index(0), b_base, b_off_up)
    cdna4_async_copy.commit_group()
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    # ---- Steady state: prefetch k+1 into g_idx, consume k from l_idx ---
    for k in range(num_k - 1):
        l_idx = k % 2
        g_idx = 1 - l_idx
        cdna4_async_copy.buffer_load_to_shared(
            smem_a.index(g_idx), a_base, a_offsets, mask=token_mask[:, None]
        )
        if WEIGHT_FP8:
            if FP8_ASYNC:
                cdna4_async_copy.buffer_load_to_shared(
                    smem_bg.index(g_idx), b_base, b_off_gate
                )
                cdna4_async_copy.buffer_load_to_shared(
                    smem_bu.index(g_idx), b_base, b_off_up
                )
            else:
                scale_k = ((k + 1) * BLOCK_K) // 128
                scale_n = (pid_n * BLOCK_N) // 128
                scale_bg = gl.load(
                    b_scale_ptr
                    + off_experts * stride_bse
                    + scale_n * stride_bsn
                    + scale_k * stride_bsk
                )
                scale_bu = gl.load(
                    b_scale_ptr
                    + off_experts * stride_bse
                    + (I_r // 128 + scale_n) * stride_bsn
                    + scale_k * stride_bsk
                )
                bg_fp8 = gl.load(b_base + b_off_gate).to(gl.float8e4nv, bitcast=True)
                bu_fp8 = gl.load(b_base + b_off_up).to(gl.float8e4nv, bitcast=True)
                smem_bg.index(g_idx).store(
                    (bg_fp8.to(gl.float32) * scale_bg).to(gl.bfloat16)
                )
                smem_bu.index(g_idx).store(
                    (bu_fp8.to(gl.float32) * scale_bu).to(gl.bfloat16)
                )
        else:
            cdna4_async_copy.buffer_load_to_shared(
                smem_bg.index(g_idx), b_base, b_off_gate
            )
            cdna4_async_copy.buffer_load_to_shared(
                smem_bu.index(g_idx), b_base, b_off_up
            )
        cdna4_async_copy.commit_group()
        cdna4_async_copy.wait_group(1)

        a = smem_a.index(l_idx).load(dot_a_layout)
        if WEIGHT_FP8 and FP8_ASYNC:
            scale_k = (k * BLOCK_K) // 128
            scale_n = (pid_n * BLOCK_N) // 128
            scale_bg = gl.load(
                b_scale_ptr
                + off_experts * stride_bse
                + scale_n * stride_bsn
                + scale_k * stride_bsk
            )
            scale_bu = gl.load(
                b_scale_ptr
                + off_experts * stride_bse
                + (I_r // 128 + scale_n) * stride_bsn
                + scale_k * stride_bsk
            )
            bg_fp8 = (
                smem_bg.index(l_idx).load(dot_b_layout).to(gl.float8e4nv, bitcast=True)
            )
            bu_fp8 = (
                smem_bu.index(l_idx).load(dot_b_layout).to(gl.float8e4nv, bitcast=True)
            )
            bg = (bg_fp8.to(gl.float32) * scale_bg).to(gl.bfloat16)
            bu = (bu_fp8.to(gl.float32) * scale_bu).to(gl.bfloat16)
        else:
            bg = smem_bg.index(l_idx).load(dot_b_layout)
            bu = smem_bu.index(l_idx).load(dot_b_layout)
        gate_acc = gl.amd.cdna3.mfma(a, bg, gate_acc)
        up_acc = gl.amd.cdna3.mfma(a, bu, up_acc)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    # ---- Epilogue: drain the last buffer -------------------------------
    cdna4_async_copy.wait_group(0)
    l_idx = (num_k - 1) % 2
    a = smem_a.index(l_idx).load(dot_a_layout)
    if WEIGHT_FP8 and FP8_ASYNC:
        scale_k = ((num_k - 1) * BLOCK_K) // 128
        scale_n = (pid_n * BLOCK_N) // 128
        scale_bg = gl.load(
            b_scale_ptr
            + off_experts * stride_bse
            + scale_n * stride_bsn
            + scale_k * stride_bsk
        )
        scale_bu = gl.load(
            b_scale_ptr
            + off_experts * stride_bse
            + (I_r // 128 + scale_n) * stride_bsn
            + scale_k * stride_bsk
        )
        bg_fp8 = smem_bg.index(l_idx).load(dot_b_layout).to(gl.float8e4nv, bitcast=True)
        bu_fp8 = smem_bu.index(l_idx).load(dot_b_layout).to(gl.float8e4nv, bitcast=True)
        bg = (bg_fp8.to(gl.float32) * scale_bg).to(gl.bfloat16)
        bu = (bu_fp8.to(gl.float32) * scale_bu).to(gl.bfloat16)
    else:
        bg = smem_bg.index(l_idx).load(dot_b_layout)
        bu = smem_bu.index(l_idx).load(dot_b_layout)
    gate_acc = gl.amd.cdna3.mfma(a, bg, gate_acc)
    up_acc = gl.amd.cdna3.mfma(a, bu, up_acc)

    # ---- SwiGLU: silu(gate) * up  (silu(x) = x * sigmoid(x)) -----------
    sig = 1.0 / (1.0 + gl.exp(-gate_acc))
    inter = gate_acc * sig * up_acc
    c_val = inter.to(c_ptr.type.element_ty)

    # ---- Scatter store to inter_row = token * topk + slot --------------
    cm_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)
    cn_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=cm_layout)
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=cn_layout)
    packed_c = gl.load(
        sorted_token_ids_ptr + offs_cm, mask=offs_cm < EM, other=num_tokens
    )
    token_c = packed_c & 0xFFFFFF
    slot_c = packed_c >> 24
    dst_row = token_c * top_k + slot_c
    c_ptrs = (
        c_ptr
        + dst_row[:, None].to(gl.int64) * stride_cm
        + offs_cn[None, :].to(gl.int64) * stride_cn
    )
    c_mask = (token_c[:, None] < num_tokens) & (offs_cn[None, :] < I_r)
    gl.store(c_ptrs, c_val, mask=c_mask)


def invoke_stage1(
    hidden_states: torch.Tensor,  # (num_tokens, D) bf16
    w1: torch.Tensor,  # (E, 2*I, D)     bf16
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: torch.Tensor,  # (num_tokens*topk, I) bf16
    topk: int,
    BLOCK_M: int = 64,
    BLOCK_N: int = 128,
    BLOCK_K: int | None = None,
    num_warps: int = 4,
    num_xcds: int = 8,
    group_size_m: int = 8,
    split_k: int | None = None,
    w1_scale: torch.Tensor | None = None,
    fp8_async: bool | None = None,
):
    """Launch bf16 MoE stage 1 (gate + up + SwiGLU).

    ``split_k`` controls the decode split-K path (see ``stage1_splitk_kernel``):
      * ``None`` (default) -> auto-pick from the batch size (``auto_split_k``);
        big split at small M, ``1`` (this single-launch path) at large M.
      * ``1``            -> force this single-launch pipelined kernel.
      * ``> 1``          -> force split-K with that factor.

    The default exact-FP8 16x64 decode tile uses a 128-wide compact-weight
    pipeline. Passing ``fp8_async=False`` retains the legacy load path for
    matched benchmarking.
    """
    assert hidden_states.dtype == torch.bfloat16
    weight_fp8 = w1_scale is not None
    if BLOCK_K is None:
        # The compact FP8 path uses one complete 128-wide scale block per
        # reduction tile. Other stage-1 shapes retain their established tile.
        BLOCK_K = 128 if weight_fp8 and BLOCK_M == 16 and BLOCK_N == 64 else 64
    if fp8_async is None:
        fp8_async = weight_fp8 and BLOCK_M == 16 and BLOCK_N == 64 and BLOCK_K == 128
    if weight_fp8:
        assert w1.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        assert w1_scale.dtype == torch.float32
        assert BLOCK_N <= 128 and 128 % BLOCK_N == 0
        assert BLOCK_K <= 128 and 128 % BLOCK_K == 0
    else:
        assert w1.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert hidden_states.dim() == 2 and w1.dim() == 3

    num_tokens, D = hidden_states.shape
    _, two_I, Dw = w1.shape
    assert Dw == D, f"w1 K dim {Dw} != hidden D {D}"
    assert two_I % 2 == 0
    I_r = two_I // 2
    N = two_I
    EM = sorted_token_ids.shape[0]
    assert out.shape == (
        num_tokens * topk,
        I_r,
    ), f"out {tuple(out.shape)} != {(num_tokens * topk, I_r)}"
    assert D % BLOCK_K == 0, f"D ({D}) must be a multiple of BLOCK_K ({BLOCK_K})"
    assert I_r % BLOCK_N == 0, f"I ({I_r}) must be a multiple of BLOCK_N ({BLOCK_N})"

    # Decode split-K dispatch.
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.stage1_splitk_kernel import (
        auto_split_k,
        invoke_stage1_splitk,
    )

    if split_k is None:
        split_k = auto_split_k(num_tokens, D // BLOCK_K)
    if weight_fp8 and split_k > 1:
        raise ValueError("exact block-FP8 stage 1 does not support split-K")
    if split_k > 1:
        return invoke_stage1_splitk(
            hidden_states,
            w1,
            sorted_token_ids,
            sorted_expert_ids,
            num_valid_ids,
            out,
            topk,
            split_k,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
        )

    grid_mn = triton.cdiv(EM, BLOCK_M) * triton.cdiv(I_r, BLOCK_N)
    scale = w1 if w1_scale is None else w1_scale
    scale_strides = (0, 0, 0) if w1_scale is None else w1_scale.stride()
    weight = w1.view(torch.uint8) if weight_fp8 else w1
    gluon_bf16_moe_stage1_kernel[(grid_mn,)](
        hidden_states,
        weight,
        scale,
        out,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        N,
        D,
        EM,
        num_tokens,
        topk,
        hidden_states.stride(0),
        hidden_states.stride(1),
        w1.stride(0),
        w1.stride(1),
        w1.stride(2),
        scale_strides[0],
        scale_strides[1],
        scale_strides[2],
        out.stride(0),
        out.stride(1),
        grid_mn,
        I_r=I_r,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_WARPS=num_warps,
        NUM_XCDS=num_xcds,
        GROUP_SIZE_M=group_size_m,
        WEIGHT_FP8=weight_fp8,
        FP8_ASYNC=fp8_async,
        num_warps=num_warps,
    )
    return out
