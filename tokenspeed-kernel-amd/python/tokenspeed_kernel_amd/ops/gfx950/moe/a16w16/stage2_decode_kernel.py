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

"""Decode-oriented stage 2 for the bf16 Gluon MoE: atomic-accumulate path.

The default stage 2 writes each routed slot's down-projection to a
``[num_tokens, topk, D]`` scratch and then runs a separate reduce kernel over
the topk dim. At large M that reduce is amortized, but at **decode** (small M)
the reduce kernel is ~30% of stage-2 GPU time (pure launch/overhead), and the
GEMM itself is already weight-bandwidth-bound (each touched expert's ``w2``
must be read once regardless of tiling, so padding/tiling changes don't help).

This path removes the reduce entirely: every routed slot ``buffer_atomic_add``s
its weighted down-projection **directly** into an fp32 output accumulator
``out_fp32[token, :]``. All topk slots of a token target the same row, so the
atomic sum is the MoE reduction. Contention is low at decode (token_num small).
A final cast writes bf16 ``out``. The GEMM is double-buffered (a small decode
win at M<=4). Selects the atomic vs reduce stage-2 path by token_num.

fp32 accumulator (not bf16 atomics) keeps the topk sum accurate and lowers to
plain ``global_atomic_add_f32``.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import cdna4_async_copy, gl, gluon, triton


@gluon.jit
def gluon_bf16_moe_stage2_atomic_kernel(
    a_ptr,  # inter    (num_tokens*topk, I)  bf16
    b_ptr,  # w2       (E, D, I)             bf16
    o_ptr,  # out_fp32 (num_tokens, D)       fp32 (accumulator)
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
    stride_ot,  # out_fp32 token stride (= D)
    stride_on,  # out_fp32 n     stride (= 1)
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

    NBUF: gl.constexpr = 2
    smem_a = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NBUF, BLOCK_M, BLOCK_K], shared_a
    )
    smem_b = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NBUF, BLOCK_K, BLOCK_N], shared_b
    )

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

    bk_layout: gl.constexpr = gl.SliceLayout(1, gload_b)
    bn_layout: gl.constexpr = gl.SliceLayout(0, gload_b)
    offs_bk = gl.arange(0, BLOCK_K, layout=bk_layout)
    offs_bn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=bn_layout)
    b_off = (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn).to(gl.int32)

    a_base = a_ptr
    b_base = b_ptr + off_experts.to(gl.int64) * stride_be

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma_layout)
    num_k = gl.cdiv(K, BLOCK_K)
    gl.assume(num_k > 1)

    # ---- Double-buffered K-loop (short K, but hides load latency at decode)
    cdna4_async_copy.buffer_load_to_shared(
        smem_a.index(0), a_base, a_offsets, mask=token_mask[:, None]
    )
    cdna4_async_copy.buffer_load_to_shared(smem_b.index(0), b_base, b_off)
    cdna4_async_copy.commit_group()
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk
    for k in range(0, num_k - 1):
        l_idx = k % 2
        g_idx = 1 - l_idx
        cdna4_async_copy.buffer_load_to_shared(
            smem_a.index(g_idx), a_base, a_offsets, mask=token_mask[:, None]
        )
        cdna4_async_copy.buffer_load_to_shared(smem_b.index(g_idx), b_base, b_off)
        cdna4_async_copy.commit_group()
        cdna4_async_copy.wait_group(1)
        a = smem_a.index(l_idx).load(dot_a_layout)
        b = smem_b.index(l_idx).load(dot_b_layout)
        acc = gl.amd.cdna3.mfma(a, b, acc)
        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk
    cdna4_async_copy.wait_group(0)
    l_idx = (num_k - 1) % 2
    a = smem_a.index(l_idx).load(dot_a_layout)
    b = smem_b.index(l_idx).load(dot_b_layout)
    acc = gl.amd.cdna3.mfma(a, b, acc)

    # ---- Routed-weight scale + atomic-accumulate into out_fp32[token, :] ----
    cm_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)
    cn_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=cm_layout)
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=cn_layout)
    moe_weight = gl.load(sorted_weights_ptr + offs_cm, mask=offs_cm < EM, other=0.0)
    packed_c = gl.load(
        sorted_token_ids_ptr + offs_cm, mask=offs_cm < EM, other=num_tokens
    )
    token_c = packed_c & 0xFFFFFF

    c_val = acc * moe_weight[:, None]
    o_offs = (token_c[:, None] * stride_ot + offs_cn[None, :] * stride_on).to(gl.int32)
    o_mask = (token_c[:, None] < num_tokens) & (offs_cn[None, :] < N)
    gl.amd.cdna4.buffer_atomic_add(o_ptr, o_offs, c_val, mask=o_mask)


def invoke_stage2_decode(
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
):
    """Atomic-accumulate stage 2 (decode): no partials scratch, no reduce."""
    assert inter_states.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16 and sorted_weights.dtype == torch.float32
    E, D, I_r = w2.shape
    num_tokens = out.shape[0]
    assert out.shape == (num_tokens, D)
    assert inter_states.shape == (num_tokens * topk, I_r)
    EM = sorted_token_ids.shape[0]
    assert I_r % BLOCK_K == 0 and D % BLOCK_N == 0

    out_fp32 = torch.zeros((num_tokens, D), dtype=torch.float32, device=out.device)
    grid = (triton.cdiv(EM, BLOCK_M) * triton.cdiv(D, BLOCK_N),)
    gluon_bf16_moe_stage2_atomic_kernel[grid](
        inter_states,
        w2,
        out_fp32,
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
        out_fp32.stride(0),
        out_fp32.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    out.copy_(out_fp32)
    return out
