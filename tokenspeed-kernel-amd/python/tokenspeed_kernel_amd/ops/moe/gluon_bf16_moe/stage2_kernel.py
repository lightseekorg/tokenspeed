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

"""Gluon bf16 MoE stage 2: down GEMM + routed-weight scale + reduce (gfx950).

For each routed slot::

    out_partial[token, slot, :] = moe_weight[slot] * (inter_slot @ w2_e.T)

then a second kernel reduces over the ``topk`` slot dim::

    out[token, :] = sum_slot out_partial[token, slot, :]

Unquantized bf16 stage-2 (down GEMM + routed-weight scale). The
``[num_tokens, topk, D]`` scratch + fp32-accumulate reduce (rather than
atomics) avoids the ``topk``-way atomic contention on overlapping output
rows -- same reduce-mode design as the mxfp4 sibling package.

GEMM idiom is the same a16w16 MFMA setup as stage 1 (``AMDMFMALayout(
version=4, instr_shape=[16,16,32], k_width=8)`` + ``gl.amd.cdna3.mfma``),
but the hot loop is deliberately **single-buffered** here: stage 2's K-loop
is short (``K = I = 256`` -> 4 tiles at ``BLOCK_K=64``) and the kernel is
bound by the wide ``[num_tokens, topk, D]`` partials write, so the v4/v9
double-buffered prefetch pipeline (which wins on stage 1's long ``K = D``
loop) measured *slower* here -- the prologue/epilogue and extra LDS don't
amortize over 4 iterations. XCD PID remap likewise gave no stage-2 benefit,
so stage 2 uses the plain ``pid`` mapping. See ``perf_report.md``.

Layout contract:
  inter (A)   (num_tokens * topk, I)   bf16   inter_row = token*topk + slot
  w2 (B)      (E, D, I)                bf16   element [k, n] = w2[e][n, k]
  partials    (num_tokens, topk, D)    bf16   scratch (written here)
  out         (num_tokens, D)          bf16   final (written by reduce)
  sorted_token_ids  (EM,)  int32  packed (slot<<24 | token)
  sorted_expert_ids (EM//BLOCK_M,) int32
  sorted_weights    (EM,)  float32  router weight per sorted slot
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import cdna4_async_copy, gl, gluon, triton


@gluon.jit
def gluon_bf16_moe_stage2_kernel(
    a_ptr,  # inter    (num_tokens*topk, I)   bf16
    b_ptr,  # w2       (E, D, I)              bf16
    c_ptr,  # partials (num_tokens, topk, D)  bf16
    sorted_token_ids_ptr,
    sorted_expert_ids_ptr,
    sorted_weights_ptr,
    num_valid_ids_ptr,
    N,  # = D
    K,  # = I
    EM,
    num_tokens,
    top_k,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_pt,  # partials token stride  (= topk * D)
    stride_ps,  # partials slot  stride  (= D)
    stride_pn,  # partials n     stride  (= 1)
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

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
        a_ptr.dtype.element_ty, [1, BLOCK_M, BLOCK_K], shared_a
    )
    smem_b = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [1, BLOCK_K, BLOCK_N], shared_b
    )

    # ---- Per-slot A-row gather: inter_row = token * topk + slot --------
    am_layout: gl.constexpr = gl.SliceLayout(1, gload_a)
    ak_layout: gl.constexpr = gl.SliceLayout(0, gload_a)
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=am_layout)
    packed = gl.load(sorted_token_ids_ptr + offs_m, mask=offs_m < EM, other=num_tokens)
    token_a = packed & 0xFFFFFF
    slot_a = packed >> 24
    inter_row = token_a * top_k + slot_a
    token_mask = token_a < num_tokens
    offs_ak = gl.arange(0, BLOCK_K, layout=ak_layout)
    a_offsets = (inter_row[:, None] * stride_am + offs_ak[None, :] * stride_ak).to(
        gl.int32
    )

    # ---- B offsets: element [k, n] = w2[e][n, k] -----------------------
    bk_layout: gl.constexpr = gl.SliceLayout(1, gload_b)
    bn_layout: gl.constexpr = gl.SliceLayout(0, gload_b)
    offs_bk = gl.arange(0, BLOCK_K, layout=bk_layout)
    offs_bn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=bn_layout)
    b_off = (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn).to(gl.int32)

    a_base = a_ptr
    b_base = b_ptr + off_experts.to(gl.int64) * stride_be

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma_layout)
    A_K_STEP = BLOCK_K * stride_ak
    B_K_STEP = BLOCK_K * stride_bk

    # ---- Single-buffered K-loop (short K; memory-bound epilogue) -------
    num_k = gl.cdiv(K, BLOCK_K)
    for k in range(num_k):
        cdna4_async_copy.buffer_load_to_shared(
            smem_a.index(0), a_base, a_offsets + k * A_K_STEP, mask=token_mask[:, None]
        )
        cdna4_async_copy.buffer_load_to_shared(
            smem_b.index(0), b_base, b_off + k * B_K_STEP
        )
        cdna4_async_copy.commit_group()
        cdna4_async_copy.wait_group(0)
        a = smem_a.index(0).load(dot_a_layout)
        b = smem_b.index(0).load(dot_b_layout)
        acc = gl.amd.cdna3.mfma(a, b, acc)

    # ---- Routed-weight scale + scatter to partials[token, slot, n] -----
    cm_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)
    cn_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=cm_layout)
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=cn_layout)
    moe_weight = gl.load(sorted_weights_ptr + offs_cm, mask=offs_cm < EM, other=0.0)
    packed_c = gl.load(
        sorted_token_ids_ptr + offs_cm, mask=offs_cm < EM, other=num_tokens
    )
    token_c = packed_c & 0xFFFFFF
    slot_c = packed_c >> 24

    c_val = (acc * moe_weight[:, None]).to(c_ptr.type.element_ty)
    c_ptrs = (
        c_ptr
        + token_c[:, None].to(gl.int64) * stride_pt
        + slot_c[:, None].to(gl.int64) * stride_ps
        + offs_cn[None, :].to(gl.int64) * stride_pn
    )
    c_mask = (token_c[:, None] < num_tokens) & (offs_cn[None, :] < N)
    gl.store(c_ptrs, c_val, mask=c_mask)


@gluon.jit
def gluon_bf16_moe_reduce_kernel(
    partials_ptr,  # (num_tokens, topk, D) bf16
    out_ptr,  # (num_tokens, D)       bf16
    num_tokens,
    N,
    stride_pt,
    stride_ps,
    stride_pn,
    stride_ot,
    stride_on,
    TOP_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """Sum the per-(token, slot) partials over the topk dim (fp32 accumulate)."""
    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    blk: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 4], [1, 0])
    rm_layout: gl.constexpr = gl.SliceLayout(1, blk)
    cn_layout: gl.constexpr = gl.SliceLayout(0, blk)
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=rm_layout)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=cn_layout)
    mask = (offs_m[:, None] < num_tokens) & (offs_n[None, :] < N)

    acc = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=blk)
    base = (
        partials_ptr
        + offs_m[:, None].to(gl.int64) * stride_pt
        + offs_n[None, :].to(gl.int64) * stride_pn
    )
    for s in gl.static_range(0, TOP_K):
        v = gl.load(base + s * stride_ps, mask=mask, other=0.0)
        acc += v.to(gl.float32)

    out_ptrs = (
        out_ptr
        + offs_m[:, None].to(gl.int64) * stride_ot
        + offs_n[None, :].to(gl.int64) * stride_on
    )
    gl.store(out_ptrs, acc.to(out_ptr.type.element_ty), mask=mask)


def invoke_stage2(
    inter_states: torch.Tensor,  # (num_tokens*topk, I) bf16
    w2: torch.Tensor,  # (E, D, I)            bf16
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: torch.Tensor,  # (num_tokens, D)      bf16
    topk: int,
    BLOCK_M: int = 64,
    BLOCK_N: int = 256,
    BLOCK_K: int = 64,
    num_warps: int = 4,
    atomic: bool | None = None,
):
    """Launch bf16 MoE stage 2 (down GEMM + routed weight) then reduce over topk.

    ``atomic`` selects the decode atomic-accumulate path (``stage2_decode_kernel``),
    which drops the partials scratch + reduce by ``buffer_atomic_add``-ing each
    slot straight into the output. ``None`` (default) -> auto: on at
    ``num_tokens <= 1`` (a ~1.22x stage-2 win at M=1; beyond M=1 the topk-way
    atomic contention on shared output rows makes it slower).
    """
    assert inter_states.dtype == torch.bfloat16
    assert w2.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert sorted_weights.dtype == torch.float32

    E, D, I_r = w2.shape
    num_tokens = out.shape[0]
    assert out.shape == (num_tokens, D)

    if atomic is None:
        atomic = num_tokens <= 1
    if atomic:
        from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe.stage2_decode_kernel import (
            invoke_stage2_decode,
        )

        return invoke_stage2_decode(
            inter_states,
            w2,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            out,
            topk,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
        )

    assert inter_states.shape == (
        num_tokens * topk,
        I_r,
    ), f"inter {tuple(inter_states.shape)} != {(num_tokens * topk, I_r)}"
    EM = sorted_token_ids.shape[0]
    assert I_r % BLOCK_K == 0, f"I ({I_r}) must be a multiple of BLOCK_K ({BLOCK_K})"
    assert D % BLOCK_N == 0, f"D ({D}) must be a multiple of BLOCK_N ({BLOCK_N})"

    # empty() is safe: every (token, slot) pair appears exactly once in the
    # sorted list, so the GEMM writes all [num_tokens, topk, D] slots before
    # the reduce reads them (avoids a per-call fill kernel).
    partials = torch.empty(
        (num_tokens, topk, D), dtype=torch.bfloat16, device=out.device
    )

    grid_mn = triton.cdiv(EM, BLOCK_M) * triton.cdiv(D, BLOCK_N)
    gluon_bf16_moe_stage2_kernel[(grid_mn,)](  # noqa: E501
        inter_states,
        w2,
        partials,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        D,
        I_r,
        EM,
        num_tokens,
        topk,
        inter_states.stride(0),
        inter_states.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )

    R_BLOCK_M = 64
    R_BLOCK_N = 256
    rgrid = (triton.cdiv(num_tokens, R_BLOCK_M) * triton.cdiv(D, R_BLOCK_N),)
    gluon_bf16_moe_reduce_kernel[rgrid](
        partials,
        out,
        num_tokens,
        D,
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        out.stride(0),
        out.stride(1),
        TOP_K=topk,
        BLOCK_M=R_BLOCK_M,
        BLOCK_N=R_BLOCK_N,
        num_warps=4,
    )
    return out
