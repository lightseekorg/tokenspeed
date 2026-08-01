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

"""Split-K stage 1 for the bf16 Gluon MoE -- a decode-oriented path.

At decode (small M) the normal stage 1 is occupancy-starved: with E=256
experts and topk=8, M=1 routes only ~8 experts, so the grid is ~16 CTAs on a
256-CU GPU (~6% occupancy). Each CTA then serially walks the full
``K = D = 7168`` contraction, so per-CTA memory latency is fully exposed.

Split-K fixes this by partitioning the K contraction across ``SPLIT_K`` CTAs
per output tile, multiplying the grid by ``SPLIT_K`` (M=1, SPLIT_K=16 -> 256
CTAs -> full occupancy). Because stage 1's SwiGLU epilogue is **nonlinear**
(``silu(gate) * up``), the split partials must be the *raw* gate/up GEMM
sums; SwiGLU is deferred to the reduce kernel which runs after the K-sum.

Two kernels:
  1. ``gluon_bf16_moe_stage1_splitk_gemm_kernel``: each (tile, split) writes
     its fp32 gate/up partial into ``partial[inter_row, split, 2*I]``
     (gate in ``[0:I]``, up in ``[I:2*I]``). No SwiGLU. Atomic-free: every
     ``(inter_row, split, col)`` is written by exactly one CTA.
  2. ``gluon_bf16_moe_stage1_reduce_swiglu_kernel``: sums the partial over the
     ``SPLIT_K`` dim (fp32), applies ``silu(gate) * up``, writes ``inter``.

The partial buffer is ``[num_tokens*topk, SPLIT_K, 2*I]`` fp32, which is tiny
at decode (M=1, SPLIT_K=16 -> 256 KB) but would be huge at large M -- hence
split-K is auto-selected only for small M (see ``auto_split_k``).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import cdna4_async_copy, gl, gluon, triton


# Valid SPLIT_K values must divide the K-tile count (K // BLOCK_K = 112 for
# the DSv3 shape at BLOCK_K=64): {2,4,7,8,14,16,28,56,112}.
def auto_split_k(num_tokens: int, num_k_tiles: int) -> int:
    """Heuristic decode split-K factor from the batch size.

    Targets enough CTAs to fill the machine at small M while disabling
    split-K (and its per-M partial buffer) once M is large enough to fill
    the GPU on its own. Clamped to a divisor of ``num_k_tiles``.
    """
    # Tuned on MI355X, DSv3 TP=8 (see perf_report.md): the split-K win is
    # real only for M <= 16; at M >= 32 it is flat-to-negative (reduce +
    # partial-buffer overhead), so fall back to the single-launch path.
    if num_tokens <= 4:
        want = 8
    elif num_tokens <= 8:
        want = 4
    elif num_tokens <= 16:
        want = 2
    else:
        return 1
    # Clamp down to the largest allowed divisor <= want.
    for sk in (want, 8, 7, 4, 2, 1):
        if sk <= want and num_k_tiles % sk == 0:
            return sk
    return 1


@gluon.jit
def gluon_bf16_moe_stage1_splitk_gemm_kernel(
    a_ptr,  # hidden_states  (num_tokens, D)  bf16
    b_ptr,  # w1             (E, 2*I, D)      bf16
    p_ptr,  # partial (num_tokens*topk, SPLIT_K, 2*I) fp32
    c_ptr,  # inter (num_tokens*topk, I) bf16 -- fused-epilogue target
    sorted_token_ids_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    K,  # = D
    EM,
    num_tokens,
    top_k,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_pr,  # partial row  stride (= SPLIT_K * 2*I)
    stride_psk,  # partial split stride (= 2*I)
    stride_pc,  # partial col  stride (= 1)
    stride_cr,  # inter row stride
    stride_cc,  # inter col stride
    I_r: gl.constexpr,  # SwiGLU width (half of N = w1.shape[1])
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    FUSE: gl.constexpr,  # SPLIT_K==1: apply SwiGLU in-epilogue, write inter directly
    NUM_WARPS: gl.constexpr,
    NBUF: gl.constexpr,  # LDS buffers: 1 = single-buffer, 2 = ping-pong
):
    pid = gl.program_id(axis=0)
    pid_sk = pid % SPLIT_K
    pid_mn = pid // SPLIT_K
    num_pid_n = gl.cdiv(I_r, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    num_valid = gl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_M >= num_valid:
        return

    off_experts = gl.load(sorted_expert_ids_ptr + pid_m)
    if off_experts == -1:
        return

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, NUM_WARPS // 2],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )

    gload_a: gl.constexpr = gl.BlockedLayout(
        [1, 8], [512 // BLOCK_K, BLOCK_K // 8], [NUM_WARPS, 1], [1, 0]
    )
    gload_b: gl.constexpr = gl.BlockedLayout(
        [8, 1], [BLOCK_K // 8, 512 // BLOCK_K], [1, NUM_WARPS], [0, 1]
    )
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])

    smem_a = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NBUF, BLOCK_M, BLOCK_K], shared_a
    )
    smem_bg = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NBUF, BLOCK_K, BLOCK_N], shared_b
    )
    smem_bu = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NBUF, BLOCK_K, BLOCK_N], shared_b
    )

    # ---- Per-token A-row gather ----------------------------------------
    am_layout: gl.constexpr = gl.SliceLayout(1, gload_a)
    ak_layout: gl.constexpr = gl.SliceLayout(0, gload_a)
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=am_layout)
    packed = gl.load(sorted_token_ids_ptr + offs_m, mask=offs_m < EM, other=num_tokens)
    a_row = packed & 0xFFFFFF
    token_mask = a_row < num_tokens
    offs_ak = gl.arange(0, BLOCK_K, layout=ak_layout)
    a_offsets = (a_row[:, None] * stride_am + offs_ak[None, :] * stride_ak).to(gl.int32)

    # ---- B (gate + up) offsets -----------------------------------------
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

    # ---- Advance A/B base pointers to this split's K-slice -------------
    num_k_total = gl.cdiv(K, BLOCK_K)
    tps = num_k_total // SPLIT_K
    k_off = pid_sk * tps * BLOCK_K
    a_base = a_ptr + k_off * stride_ak
    b_base = b_ptr + off_experts.to(gl.int64) * stride_be + k_off * stride_bk

    gate_acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma_layout)
    up_acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma_layout)

    gl.assume(tps > 1)

    if NBUF == 1:
        # ---- single LDS buffer (min LDS -> max occupancy) --
        # load(k) -> wait -> MFMA(k); no ping-pong. Relies on many WGPs/CU to
        # hide latency rather than deep pipelining.
        for k in range(0, tps):
            cdna4_async_copy.buffer_load_to_shared(
                smem_a.index(0), a_base, a_offsets, mask=token_mask[:, None]
            )
            cdna4_async_copy.buffer_load_to_shared(smem_bg.index(0), b_base, b_off_gate)
            cdna4_async_copy.buffer_load_to_shared(smem_bu.index(0), b_base, b_off_up)
            cdna4_async_copy.commit_group()
            cdna4_async_copy.wait_group(0)
            a = smem_a.index(0).load(dot_a_layout)
            bg = smem_bg.index(0).load(dot_b_layout)
            bu = smem_bu.index(0).load(dot_b_layout)
            gate_acc = gl.amd.cdna3.mfma(a, bg, gate_acc)
            up_acc = gl.amd.cdna3.mfma(a, bu, up_acc)
            a_base += BLOCK_K * stride_ak
            b_base += BLOCK_K * stride_bk
    else:
        # ---- Double-buffered ping-pong (1 global-prefetch stage) ----------
        cdna4_async_copy.buffer_load_to_shared(
            smem_a.index(0), a_base, a_offsets, mask=token_mask[:, None]
        )
        cdna4_async_copy.buffer_load_to_shared(smem_bg.index(0), b_base, b_off_gate)
        cdna4_async_copy.buffer_load_to_shared(smem_bu.index(0), b_base, b_off_up)
        cdna4_async_copy.commit_group()
        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk
        for k in range(0, tps - 1):
            l_idx = k % 2
            g_idx = 1 - l_idx
            cdna4_async_copy.buffer_load_to_shared(
                smem_a.index(g_idx), a_base, a_offsets, mask=token_mask[:, None]
            )
            cdna4_async_copy.buffer_load_to_shared(
                smem_bg.index(g_idx), b_base, b_off_gate
            )
            cdna4_async_copy.buffer_load_to_shared(
                smem_bu.index(g_idx), b_base, b_off_up
            )
            cdna4_async_copy.commit_group()
            cdna4_async_copy.wait_group(1)
            a = smem_a.index(l_idx).load(dot_a_layout)
            bg = smem_bg.index(l_idx).load(dot_b_layout)
            bu = smem_bu.index(l_idx).load(dot_b_layout)
            gate_acc = gl.amd.cdna3.mfma(a, bg, gate_acc)
            up_acc = gl.amd.cdna3.mfma(a, bu, up_acc)
            a_base += BLOCK_K * stride_ak
            b_base += BLOCK_K * stride_bk
        cdna4_async_copy.wait_group(0)
        l_idx = (tps - 1) % 2
        a = smem_a.index(l_idx).load(dot_a_layout)
        bg = smem_bg.index(l_idx).load(dot_b_layout)
        bu = smem_bu.index(l_idx).load(dot_b_layout)
        gate_acc = gl.amd.cdna3.mfma(a, bg, gate_acc)
        up_acc = gl.amd.cdna3.mfma(a, bu, up_acc)

    # ---- Write raw fp32 gate/up partials (NO SwiGLU) -------------------
    # partial[inter_row, pid_sk, gate/up col]; inter_row = token*topk + slot.
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

    store_mask = (token_c[:, None] < num_tokens) & (offs_cn[None, :] < I_r)
    if FUSE:
        # SPLIT_K==1: no K-partials to sum -> apply SwiGLU here, write inter.
        sig = 1.0 / (1.0 + gl.exp(-gate_acc))
        inter_val = gate_acc * sig * up_acc
        c_ptrs = (
            c_ptr
            + dst_row[:, None].to(gl.int64) * stride_cr
            + offs_cn[None, :].to(gl.int64) * stride_cc
        )
        gl.store(c_ptrs, inter_val.to(c_ptr.type.element_ty), mask=store_mask)
    else:
        row_base = (
            p_ptr + dst_row[:, None].to(gl.int64) * stride_pr + pid_sk * stride_psk
        )
        g_ptrs = row_base + offs_cn[None, :].to(gl.int64) * stride_pc
        u_ptrs = row_base + (I_r + offs_cn[None, :]).to(gl.int64) * stride_pc
        gl.store(g_ptrs, gate_acc, mask=store_mask)
        gl.store(u_ptrs, up_acc, mask=store_mask)


@gluon.jit
def gluon_bf16_moe_stage1_reduce_swiglu_kernel(
    p_ptr,  # partial (num_tokens*topk, SPLIT_K, 2*I) fp32
    c_ptr,  # inter   (num_tokens*topk, I)            bf16
    n_rows,  # = num_tokens * topk
    stride_pr,
    stride_psk,
    stride_pc,
    stride_cr,
    stride_cc,
    I_r: gl.constexpr,
    SPLIT_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """Sum gate/up partials over SPLIT_K (fp32), apply SwiGLU, write inter."""
    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(I_r, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    blk: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 4], [1, 0])
    rm_layout: gl.constexpr = gl.SliceLayout(1, blk)
    cn_layout: gl.constexpr = gl.SliceLayout(0, blk)
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=rm_layout)
    offs_i = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=cn_layout)
    mask = (offs_m[:, None] < n_rows) & (offs_i[None, :] < I_r)

    gate = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=blk)
    up = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=blk)
    g_base = (
        p_ptr
        + offs_m[:, None].to(gl.int64) * stride_pr
        + offs_i[None, :].to(gl.int64) * stride_pc
    )
    u_base = (
        p_ptr
        + offs_m[:, None].to(gl.int64) * stride_pr
        + (I_r + offs_i)[None, :].to(gl.int64) * stride_pc
    )
    for sk in gl.static_range(0, SPLIT_K):
        gate += gl.load(g_base + sk * stride_psk, mask=mask, other=0.0)
        up += gl.load(u_base + sk * stride_psk, mask=mask, other=0.0)

    sig = 1.0 / (1.0 + gl.exp(-gate))
    inter = gate * sig * up
    out_ptrs = (
        c_ptr
        + offs_m[:, None].to(gl.int64) * stride_cr
        + offs_i[None, :].to(gl.int64) * stride_cc
    )
    gl.store(out_ptrs, inter.to(c_ptr.type.element_ty), mask=mask)


def invoke_stage1_splitk(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: torch.Tensor,  # (num_tokens*topk, I) bf16 (inter)
    topk: int,
    split_k: int,
    BLOCK_M: int = 64,
    BLOCK_N: int = 128,
    BLOCK_K: int = 64,
    num_warps: int = 4,
    nbuf: int = 2,
):
    """Split-K stage 1 (gate+up partials) + fused reduce/SwiGLU."""
    assert hidden_states.dtype == torch.bfloat16 and w1.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    num_tokens, D = hidden_states.shape
    E, two_I, Dw = w1.shape
    assert Dw == D and two_I % 2 == 0
    I_r = two_I // 2
    EM = sorted_token_ids.shape[0]
    n_rows = num_tokens * topk
    assert out.shape == (n_rows, I_r)
    assert D % BLOCK_K == 0 and I_r % BLOCK_N == 0
    num_k_tiles = D // BLOCK_K
    assert (
        num_k_tiles % split_k == 0
    ), f"split_k ({split_k}) must divide K-tile count ({num_k_tiles})"

    # split_k==1: no K-partials -> fuse SwiGLU into the GEMM epilogue and write
    # inter directly (one kernel, no partial buffer, no reduce). split_k>1: raw
    # fp32 gate/up partials [n_rows, SPLIT_K, 2*I] then reduce+SwiGLU.
    fuse = split_k == 1
    if fuse:
        partial = torch.empty(
            1, dtype=torch.float32, device=out.device
        )  # dummy (unused)
        p_sr = p_ssk = p_sc = 1
    else:
        partial = torch.empty(
            (n_rows, split_k, two_I), dtype=torch.float32, device=out.device
        )
        p_sr, p_ssk, p_sc = partial.stride(0), partial.stride(1), partial.stride(2)

    grid_mn = triton.cdiv(EM, BLOCK_M) * triton.cdiv(I_r, BLOCK_N)
    gluon_bf16_moe_stage1_splitk_gemm_kernel[(grid_mn * split_k,)](
        hidden_states,
        w1,
        partial,
        out,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        D,
        EM,
        num_tokens,
        topk,
        hidden_states.stride(0),
        hidden_states.stride(1),
        w1.stride(0),
        w1.stride(1),
        w1.stride(2),
        p_sr,
        p_ssk,
        p_sc,
        out.stride(0),
        out.stride(1),
        I_r=I_r,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=split_k,
        FUSE=fuse,
        NUM_WARPS=num_warps,
        NBUF=nbuf,
        num_warps=num_warps,
    )
    if fuse:
        return out

    R_BLOCK_M = 64
    R_BLOCK_N = 128
    rgrid = (triton.cdiv(n_rows, R_BLOCK_M) * triton.cdiv(I_r, R_BLOCK_N),)
    gluon_bf16_moe_stage1_reduce_swiglu_kernel[rgrid](
        partial,
        out,
        n_rows,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        out.stride(0),
        out.stride(1),
        I_r=I_r,
        SPLIT_K=split_k,
        BLOCK_M=R_BLOCK_M,
        BLOCK_N=R_BLOCK_N,
        num_warps=4,
    )
    return out
