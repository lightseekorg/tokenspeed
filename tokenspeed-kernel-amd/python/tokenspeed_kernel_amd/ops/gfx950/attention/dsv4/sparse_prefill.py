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

"""CDNA4 sparse multi-head attention for gfx950.

This production path builds on the ROCm/AITER PR #3456 (MIT):
https://github.com/ROCm/aiter/pull/3456

This version adds TokenSpeed's selected-attention ABI, topk_lens support,
model-scale masking/addressing fixes, and the BLOCK_K=32 specialization used for
validated DeepSeek-V4 prefill shapes: B=1, D=512, H in {64, 128}, topk>=128.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import cdna4_async_copy as cdna4_async
from tokenspeed_kernel_amd._triton import (
    gl,
    gluon,
    tl,
    triton,
)

__all__ = ["gluon_dsv4_sparse_prefill_gfx950"]


# Production H64/D512 sparse-attention path builds on the ROCm/AITER PR #3456
# pipeline (MIT) with TokenSpeed ABI, lens, masking, and BLOCK_K=32 support.
@gluon.jit
def _sparse_attn_k64_kernel(
    q,
    kv,
    o,
    attn_sink,
    topk_idxs,
    topk_lens,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvn,
    stride_kvd,
    stride_om,
    stride_oh,
    stride_od,
    stride_topk_m,
    stride_topk_k,
    stride_lens_m,
    num_queries,
    num_kv_rows,
    num_iters,
    scale,
    BLOCK_H: gl.constexpr,
    BLOCK_D: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    ASSUME_COMPACT_INDICES: gl.constexpr,
    num_warps: gl.constexpr,
):
    BLOCK_K: gl.constexpr = 64
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    qk_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    qk_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    store_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0])

    q_load_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [1, 64], [num_warps, 1], [1, 0]
    )
    kv_load_layout: gl.constexpr = gl.BlockedLayout(
        [8, 64 // num_warps], [64, 1], [1, num_warps], [0, 1]
    )
    slot_load_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[],
        lane_bases=[[1], [2], [4], [8], [16], [32]],
        warp_bases=[[0], [0]],
        block_bases=[],
        shape=[BLOCK_K],
    )
    gl.static_assert(num_warps == 4)
    gl.static_assert(BLOCK_H == 64)
    gl.static_assert(BLOCK_D == 512)

    q_smem_layout: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [0, 128],
            [0, 256],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
        ],
        cga_layout=[],
        shape=[BLOCK_H, BLOCK_D],
    )
    kv_smem_layout: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [64, 0],
            [128, 0],
            [256, 0],
            [0, 1],
            [0, 2],
            [0, 8],
            [0, 4],
            [0, 16],
            [0, 32],
        ],
        cga_layout=[],
        shape=[BLOCK_D, BLOCK_K],
    )
    slot_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    sl_h_q: gl.constexpr = gl.SliceLayout(1, q_load_layout)
    sl_d_q: gl.constexpr = gl.SliceLayout(0, q_load_layout)
    sl_d_kv: gl.constexpr = gl.SliceLayout(1, kv_load_layout)
    sl_k_kv: gl.constexpr = gl.SliceLayout(0, kv_load_layout)
    sl_h_mma: gl.constexpr = gl.SliceLayout(1, mma)
    sl_k_mma: gl.constexpr = gl.SliceLayout(0, mma)

    query_idx = gl.program_id(0) + gl.program_id(2) * NUM_XCDS
    head_block_idx = gl.program_id(1)
    if query_idx >= num_queries:
        return
    active_topk_len = gl.load(topk_lens + query_idx * stride_lens_m).to(gl.int32)

    head_off = head_block_idx * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_q)
    dim_off = gl.arange(0, BLOCK_D, layout=sl_d_q)
    q_off = head_off[:, None] * stride_qh + dim_off[None, :] * stride_qd
    q_smem = gl.allocate_shared_memory(
        q.dtype.element_ty, [BLOCK_H, BLOCK_D], q_smem_layout
    )
    cdna4_async.buffer_load_to_shared(
        q_smem,
        q + query_idx * stride_qm,
        q_off,
        cache_modifier=".cg",
    )
    cdna4_async.commit_group()

    LOG2E: gl.constexpr = 1.4426950408889634
    qk_scale = scale * LOG2E
    sink_head = head_block_idx * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_mma)
    running_max = (
        gl.load(
            attn_sink + sink_head,
        ).to(gl.float32)
        * LOG2E
    )
    running_sum = gl.full([BLOCK_H], 1.0, gl.float32, sl_h_mma)
    acc = gl.zeros([BLOCK_H, BLOCK_D], gl.float32, mma)

    k_pos = gl.arange(0, BLOCK_K, layout=sl_k_kv)
    k_pos_mfma = gl.arange(0, BLOCK_K, layout=sl_k_mma)
    slot_off = gl.arange(0, BLOCK_K, layout=slot_load_layout)
    dim_kv = gl.arange(0, BLOCK_D, layout=sl_d_kv)
    topk_base = topk_idxs + query_idx * stride_topk_m

    index_smem = gl.allocate_shared_memory(
        topk_idxs.dtype.element_ty, [2, BLOCK_K], slot_smem_layout
    )
    cdna4_async.buffer_load_to_shared(
        index_smem.index(0),
        topk_base,
        slot_off * stride_topk_k,
    )
    cdna4_async.commit_group()
    cdna4_async.buffer_load_to_shared(
        index_smem.index(1),
        topk_base,
        (BLOCK_K + slot_off) * stride_topk_k,
    )
    cdna4_async.commit_group()
    cdna4_async.wait_group(2)
    q_dot = cdna4_async.load_shared_relaxed(q_smem, qk_a)

    kv_smem = gl.allocate_shared_memory(
        kv.dtype.element_ty, [2, BLOCK_D, BLOCK_K], kv_smem_layout
    )

    cdna4_async.wait_group(1)
    index0 = cdna4_async.load_shared_relaxed(index_smem.index(0), sl_k_kv)
    index0_mfma = cdna4_async.load_shared_relaxed(index_smem.index(0), sl_k_mma)
    valid0_pos = k_pos < active_topk_len
    if ASSUME_COMPACT_INDICES:
        valid0 = valid0_pos
    else:
        valid0 = valid0_pos & (index0 >= 0) & (index0.to(tl.int64) < num_kv_rows)
    kv_off0 = (
        dim_kv[:, None] * stride_kvd + gl.where(valid0, index0, 0)[None, :] * stride_kvn
    )
    cdna4_async.buffer_load_to_shared(
        kv_smem.index(0), kv, kv_off0, mask=valid0[None, :]
    )
    cdna4_async.commit_group()
    valid_mfma_pos = k_pos_mfma < active_topk_len
    if ASSUME_COMPACT_INDICES:
        valid_mfma = valid_mfma_pos
    else:
        valid_mfma = (
            valid_mfma_pos
            & (index0_mfma >= 0)
            & (index0_mfma.to(tl.int64) < num_kv_rows)
        )

    # Stage indices two tiles ahead and KV one tile ahead.
    for i in tl.range(0, num_iters - 2):
        future_index_pos = (i + 2) * BLOCK_K + slot_off
        cdna4_async.buffer_load_to_shared(
            index_smem.index(i % 2),
            topk_base,
            future_index_pos * stride_topk_k,
        )
        cdna4_async.commit_group()
        cdna4_async.wait_group(1)
        current_buffer = i % 2
        k_dot = cdna4_async.load_shared_relaxed(kv_smem.index(current_buffer), qk_b)
        scores = gl.zeros([BLOCK_H, BLOCK_K], gl.float32, mma)
        scores = gl.amd.cdna4.mfma(q_dot, k_dot, scores)

        next_buffer = (i + 1) % 2
        cdna4_async.wait_group(2)
        next_index = cdna4_async.load_shared_relaxed(
            index_smem.index(next_buffer), sl_k_kv
        )
        next_pos = (i + 1) * BLOCK_K + k_pos
        next_mfma_pos = (i + 1) * BLOCK_K + gl.arange(0, BLOCK_K, layout=sl_k_mma)
        next_index_mfma = cdna4_async.load_shared_relaxed(
            index_smem.index(next_buffer), sl_k_mma
        )
        next_valid_pos = next_pos < active_topk_len
        if ASSUME_COMPACT_INDICES:
            next_valid = next_valid_pos
        else:
            next_valid = (
                next_valid_pos
                & (next_index >= 0)
                & (next_index.to(tl.int64) < num_kv_rows)
            )
        next_valid_mfma_pos = next_mfma_pos < active_topk_len
        if ASSUME_COMPACT_INDICES:
            next_valid_mfma = next_valid_mfma_pos
        else:
            next_valid_mfma = (
                next_valid_mfma_pos
                & (next_index_mfma >= 0)
                & (next_index_mfma.to(tl.int64) < num_kv_rows)
            )
        next_kv_off = (
            dim_kv[:, None] * stride_kvd
            + gl.where(next_valid, next_index, 0)[None, :] * stride_kvn
        )
        cdna4_async.buffer_load_to_shared(
            kv_smem.index(next_buffer), kv, next_kv_off, mask=next_valid[None, :]
        )
        cdna4_async.commit_group()

        current_valid = valid_mfma
        scores *= qk_scale
        scores = gl.where(current_valid[None, :], scores, float("-inf"))
        new_max = gl.maximum(running_max, gl.max(scores, axis=1))
        alpha = gl.exp2(running_max - new_max)
        p = gl.exp2(scores - new_max[:, None])
        p = gl.where(current_valid[None, :], p, 0.0)
        running_sum = running_sum * alpha + gl.sum(p, axis=1)
        running_max = new_max
        v_dot = cdna4_async.load_shared_relaxed(
            kv_smem.index(current_buffer).permute([1, 0]), qk_b
        )
        p_dot = gl.convert_layout(p.to(kv.dtype.element_ty), qk_a)
        acc *= alpha[:, None]
        acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)
        valid_mfma = next_valid_mfma

    # Load the final KV tile, then drain the final two tiles.
    final_buffer = (num_iters - 1) % 2
    final_pos = (num_iters - 1) * BLOCK_K + k_pos
    cdna4_async.wait_group(1)
    final_index = cdna4_async.load_shared_relaxed(
        index_smem.index(final_buffer), sl_k_kv
    )
    final_load_pos_valid = final_pos < active_topk_len
    if ASSUME_COMPACT_INDICES:
        final_load_valid = final_load_pos_valid
    else:
        final_load_valid = (
            final_load_pos_valid
            & (final_index >= 0)
            & (final_index.to(tl.int64) < num_kv_rows)
        )
    final_kv_off = (
        dim_kv[:, None] * stride_kvd
        + gl.where(final_load_valid, final_index, 0)[None, :] * stride_kvn
    )
    cdna4_async.buffer_load_to_shared(
        kv_smem.index(final_buffer), kv, final_kv_off, mask=final_load_valid[None, :]
    )
    cdna4_async.commit_group()

    cdna4_async.wait_group(1)
    penultimate_tile = num_iters - 2
    penultimate_buffer = penultimate_tile % 2
    penultimate_pos = penultimate_tile * BLOCK_K + gl.arange(
        0, BLOCK_K, layout=sl_k_mma
    )
    penultimate_index = cdna4_async.load_shared_relaxed(
        index_smem.index(penultimate_buffer), sl_k_mma
    )
    penultimate_pos_valid = penultimate_pos < active_topk_len
    if ASSUME_COMPACT_INDICES:
        penultimate_valid = penultimate_pos_valid
    else:
        penultimate_valid = (
            penultimate_pos_valid
            & (penultimate_index >= 0)
            & (penultimate_index.to(tl.int64) < num_kv_rows)
        )
    k_dot = cdna4_async.load_shared_relaxed(kv_smem.index(penultimate_buffer), qk_b)
    scores = gl.zeros([BLOCK_H, BLOCK_K], gl.float32, mma)
    scores = gl.amd.cdna4.mfma(q_dot, k_dot, scores)
    scores = gl.where(penultimate_valid[None, :], scores, float("-inf"))
    scores *= qk_scale
    new_max = gl.maximum(running_max, gl.max(scores, axis=1))
    alpha = gl.exp2(running_max - new_max)
    p = gl.exp2(scores - new_max[:, None])
    p = gl.where(penultimate_valid[None, :], p, 0.0)
    running_sum = running_sum * alpha + gl.sum(p, axis=1)
    running_max = new_max
    v_dot = cdna4_async.load_shared_relaxed(
        kv_smem.index(penultimate_buffer).permute([1, 0]), qk_b
    )
    p_dot = gl.convert_layout(p.to(kv.dtype.element_ty), qk_a)
    acc *= alpha[:, None]
    acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)

    cdna4_async.wait_group(0)
    final_mfma_pos = (num_iters - 1) * BLOCK_K + gl.arange(0, BLOCK_K, layout=sl_k_mma)
    final_mfma_index = cdna4_async.load_shared_relaxed(
        index_smem.index(final_buffer), sl_k_mma
    )
    final_mfma_pos_valid = final_mfma_pos < active_topk_len
    if ASSUME_COMPACT_INDICES:
        final_valid = final_mfma_pos_valid
    else:
        final_valid = (
            final_mfma_pos_valid
            & (final_mfma_index >= 0)
            & (final_mfma_index.to(tl.int64) < num_kv_rows)
        )
    k_dot = cdna4_async.load_shared_relaxed(kv_smem.index(final_buffer), qk_b)
    scores = gl.zeros([BLOCK_H, BLOCK_K], gl.float32, mma)
    scores = gl.amd.cdna4.mfma(q_dot, k_dot, scores)
    scores = gl.where(final_valid[None, :], scores, float("-inf"))
    scores *= qk_scale
    new_max = gl.maximum(running_max, gl.max(scores, axis=1))
    alpha = gl.exp2(running_max - new_max)
    p = gl.exp2(scores - new_max[:, None])
    p = gl.where(final_valid[None, :], p, 0.0)
    running_sum = running_sum * alpha + gl.sum(p, axis=1)
    running_max = new_max
    v_dot = cdna4_async.load_shared_relaxed(
        kv_smem.index(final_buffer).permute([1, 0]), qk_b
    )
    p_dot = gl.convert_layout(p.to(kv.dtype.element_ty), qk_a)
    acc *= alpha[:, None]
    acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)

    final_sum = running_sum
    output_scale = 1.0 / gl.maximum(final_sum, 1.0e-30)
    output = gl.where(
        (final_sum > 0.0)[:, None],
        acc * output_scale[:, None],
        0.0,
    )

    # Store the first BF16 half while the second half changes layout.
    output_bf16 = output.to(o.dtype.element_ty)
    output_lo, output_hi = (
        output_bf16.reshape([BLOCK_H, 2, BLOCK_D // 2]).permute([0, 2, 1]).split()
    )
    out_head = head_block_idx * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, store_layout)
    )
    out_dim = gl.arange(0, BLOCK_D // 2, layout=gl.SliceLayout(0, store_layout))
    output_lo = gl.convert_layout(output_lo, store_layout)
    out_off = out_head[:, None] * stride_oh + out_dim[None, :] * stride_od
    gl.store(
        o + query_idx * stride_om + out_off,
        output_lo,
    )
    output_hi = gl.convert_layout(output_hi, store_layout)
    out_off = (
        out_head[:, None] * stride_oh + (BLOCK_D // 2 + out_dim[None, :]) * stride_od
    )
    gl.store(
        o + query_idx * stride_om + out_off,
        output_hi,
    )


@gluon.jit
def _sparse_attn_k32_kernel(
    q,
    kv,
    o,
    attn_sink,
    topk_idxs,
    topk_lens,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvn,
    stride_kvd,
    stride_om,
    stride_oh,
    stride_od,
    stride_topk_m,
    stride_topk_k,
    stride_lens_m,
    num_queries,
    num_kv_rows,
    num_iters,
    scale,
    BLOCK_H: gl.constexpr,
    BLOCK_D: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    ASSUME_COMPACT_INDICES: gl.constexpr,
    num_warps: gl.constexpr,
):
    BLOCK_K: gl.constexpr = 32
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    qk_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    qk_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    store_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0])

    q_load_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [1, 64], [num_warps, 1], [1, 0]
    )
    kv_load_layout: gl.constexpr = gl.BlockedLayout(
        [8, 32 // num_warps], [64, 1], [1, num_warps], [0, 1]
    )
    # The 32-wide async index layout does not lower cleanly. K32 loads
    # indices directly into the KV-column distribution instead.
    slot_load_layout: gl.constexpr = gl.SliceLayout(0, kv_load_layout)

    gl.static_assert(num_warps == 4)
    gl.static_assert(BLOCK_H == 64)
    gl.static_assert(BLOCK_D == 512)

    q_smem_layout: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [0, 128],
            [0, 256],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
        ],
        cga_layout=[],
        shape=[BLOCK_H, BLOCK_D],
    )
    kv_smem_layout: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [64, 0],
            [128, 0],
            [256, 0],
            [0, 1],
            [0, 2],
            [0, 8],
            [0, 4],
            [0, 16],
        ],
        cga_layout=[],
        shape=[BLOCK_D, BLOCK_K],
    )
    slot_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    sl_h_q: gl.constexpr = gl.SliceLayout(1, q_load_layout)
    sl_d_q: gl.constexpr = gl.SliceLayout(0, q_load_layout)
    sl_d_kv: gl.constexpr = gl.SliceLayout(1, kv_load_layout)
    sl_k_kv: gl.constexpr = gl.SliceLayout(0, kv_load_layout)
    sl_h_mma: gl.constexpr = gl.SliceLayout(1, mma)
    sl_k_mma: gl.constexpr = gl.SliceLayout(0, mma)

    query_idx = gl.program_id(0) + gl.program_id(2) * NUM_XCDS
    head_block_idx = gl.program_id(1)
    if query_idx >= num_queries:
        return
    active_topk_len = gl.load(topk_lens + query_idx * stride_lens_m).to(gl.int32)

    head_off = head_block_idx * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_q)
    dim_off = gl.arange(0, BLOCK_D, layout=sl_d_q)
    q_off = head_off[:, None] * stride_qh + dim_off[None, :] * stride_qd
    q_smem = gl.allocate_shared_memory(
        q.dtype.element_ty, [BLOCK_H, BLOCK_D], q_smem_layout
    )
    cdna4_async.buffer_load_to_shared(
        q_smem,
        q + query_idx * stride_qm,
        q_off,
        cache_modifier=".cg",
    )
    cdna4_async.commit_group()

    LOG2E: gl.constexpr = 1.4426950408889634
    qk_scale = scale * LOG2E
    sink_head = head_block_idx * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_mma)
    running_max = (
        gl.load(
            attn_sink + sink_head,
        ).to(gl.float32)
        * LOG2E
    )
    running_sum = gl.full([BLOCK_H], 1.0, gl.float32, sl_h_mma)
    acc = gl.zeros([BLOCK_H, BLOCK_D], gl.float32, mma)

    k_pos = gl.arange(0, BLOCK_K, layout=sl_k_kv)
    k_pos_mfma = gl.arange(0, BLOCK_K, layout=sl_k_mma)
    slot_off = gl.arange(0, BLOCK_K, layout=slot_load_layout)
    dim_kv = gl.arange(0, BLOCK_D, layout=sl_d_kv)
    topk_base = topk_idxs + query_idx * stride_topk_m

    index_smem = gl.allocate_shared_memory(
        topk_idxs.dtype.element_ty, [2, BLOCK_K], slot_smem_layout
    )
    cdna4_async.wait_group(0)
    q_dot = cdna4_async.load_shared_relaxed(q_smem, qk_a)

    kv_smem = gl.allocate_shared_memory(
        kv.dtype.element_ty, [2, BLOCK_D, BLOCK_K], kv_smem_layout
    )

    index0 = gl.load(
        topk_base + k_pos * stride_topk_k,
    )
    index0_mfma = gl.load(
        topk_base + k_pos_mfma * stride_topk_k,
    )
    valid0_pos = k_pos < active_topk_len
    if ASSUME_COMPACT_INDICES:
        valid0 = valid0_pos
    else:
        valid0 = valid0_pos & (index0 >= 0) & (index0.to(tl.int64) < num_kv_rows)
    kv_off0 = (
        dim_kv[:, None] * stride_kvd + gl.where(valid0, index0, 0)[None, :] * stride_kvn
    )
    cdna4_async.buffer_load_to_shared(
        kv_smem.index(0), kv, kv_off0, mask=valid0[None, :]
    )
    cdna4_async.commit_group()
    valid_mfma_pos = k_pos_mfma < active_topk_len
    if ASSUME_COMPACT_INDICES:
        valid_mfma = valid_mfma_pos
    else:
        valid_mfma = (
            valid_mfma_pos
            & (index0_mfma >= 0)
            & (index0_mfma.to(tl.int64) < num_kv_rows)
        )

    # Stage indices two tiles ahead and KV one tile ahead.
    for i in tl.range(0, num_iters - 2):
        cdna4_async.wait_group(0)
        current_buffer = i % 2
        k_dot = cdna4_async.load_shared_relaxed(kv_smem.index(current_buffer), qk_b)
        scores = gl.zeros([BLOCK_H, BLOCK_K], gl.float32, mma)
        scores = gl.amd.cdna4.mfma(q_dot, k_dot, scores)

        next_buffer = (i + 1) % 2
        next_pos = (i + 1) * BLOCK_K + k_pos
        next_index = gl.load(
            topk_base + next_pos * stride_topk_k,
        )
        next_pos = (i + 1) * BLOCK_K + k_pos
        next_mfma_pos = (i + 1) * BLOCK_K + gl.arange(0, BLOCK_K, layout=sl_k_mma)
        next_index_mfma = gl.load(
            topk_base + next_mfma_pos * stride_topk_k,
        )
        next_valid_pos = next_pos < active_topk_len
        if ASSUME_COMPACT_INDICES:
            next_valid = next_valid_pos
        else:
            next_valid = (
                next_valid_pos
                & (next_index >= 0)
                & (next_index.to(tl.int64) < num_kv_rows)
            )
        next_valid_mfma_pos = next_mfma_pos < active_topk_len
        if ASSUME_COMPACT_INDICES:
            next_valid_mfma = next_valid_mfma_pos
        else:
            next_valid_mfma = (
                next_valid_mfma_pos
                & (next_index_mfma >= 0)
                & (next_index_mfma.to(tl.int64) < num_kv_rows)
            )
        next_kv_off = (
            dim_kv[:, None] * stride_kvd
            + gl.where(next_valid, next_index, 0)[None, :] * stride_kvn
        )
        cdna4_async.buffer_load_to_shared(
            kv_smem.index(next_buffer), kv, next_kv_off, mask=next_valid[None, :]
        )
        cdna4_async.commit_group()

        current_valid = valid_mfma
        scores *= qk_scale
        scores = gl.where(current_valid[None, :], scores, float("-inf"))
        new_max = gl.maximum(running_max, gl.max(scores, axis=1))
        alpha = gl.exp2(running_max - new_max)
        p = gl.exp2(scores - new_max[:, None])
        p = gl.where(current_valid[None, :], p, 0.0)
        running_sum = running_sum * alpha + gl.sum(p, axis=1)
        running_max = new_max
        v_dot = cdna4_async.load_shared_relaxed(
            kv_smem.index(current_buffer).permute([1, 0]), qk_b
        )
        p_dot = gl.convert_layout(p.to(kv.dtype.element_ty), qk_a)
        acc *= alpha[:, None]
        acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)
        valid_mfma = next_valid_mfma

    # Load the final KV tile, then drain the final two tiles.
    final_buffer = (num_iters - 1) % 2
    final_pos = (num_iters - 1) * BLOCK_K + k_pos
    final_index = gl.load(
        topk_base + final_pos * stride_topk_k,
    )
    final_load_pos_valid = final_pos < active_topk_len
    if ASSUME_COMPACT_INDICES:
        final_load_valid = final_load_pos_valid
    else:
        final_load_valid = (
            final_load_pos_valid
            & (final_index >= 0)
            & (final_index.to(tl.int64) < num_kv_rows)
        )
    final_kv_off = (
        dim_kv[:, None] * stride_kvd
        + gl.where(final_load_valid, final_index, 0)[None, :] * stride_kvn
    )
    cdna4_async.buffer_load_to_shared(
        kv_smem.index(final_buffer), kv, final_kv_off, mask=final_load_valid[None, :]
    )
    cdna4_async.commit_group()

    cdna4_async.wait_group(1)
    penultimate_tile = num_iters - 2
    penultimate_buffer = penultimate_tile % 2
    penultimate_pos = penultimate_tile * BLOCK_K + gl.arange(
        0, BLOCK_K, layout=sl_k_mma
    )
    penultimate_index = gl.load(
        topk_base + penultimate_pos * stride_topk_k,
    )
    penultimate_pos_valid = penultimate_pos < active_topk_len
    if ASSUME_COMPACT_INDICES:
        penultimate_valid = penultimate_pos_valid
    else:
        penultimate_valid = (
            penultimate_pos_valid
            & (penultimate_index >= 0)
            & (penultimate_index.to(tl.int64) < num_kv_rows)
        )
    k_dot = cdna4_async.load_shared_relaxed(kv_smem.index(penultimate_buffer), qk_b)
    scores = gl.zeros([BLOCK_H, BLOCK_K], gl.float32, mma)
    scores = gl.amd.cdna4.mfma(q_dot, k_dot, scores)
    scores = gl.where(penultimate_valid[None, :], scores, float("-inf"))
    scores *= qk_scale
    new_max = gl.maximum(running_max, gl.max(scores, axis=1))
    alpha = gl.exp2(running_max - new_max)
    p = gl.exp2(scores - new_max[:, None])
    p = gl.where(penultimate_valid[None, :], p, 0.0)
    running_sum = running_sum * alpha + gl.sum(p, axis=1)
    running_max = new_max
    v_dot = cdna4_async.load_shared_relaxed(
        kv_smem.index(penultimate_buffer).permute([1, 0]), qk_b
    )
    p_dot = gl.convert_layout(p.to(kv.dtype.element_ty), qk_a)
    acc *= alpha[:, None]
    acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)

    cdna4_async.wait_group(0)
    final_mfma_pos = (num_iters - 1) * BLOCK_K + gl.arange(0, BLOCK_K, layout=sl_k_mma)
    final_mfma_index = gl.load(
        topk_base + final_mfma_pos * stride_topk_k,
    )
    final_mfma_pos_valid = final_mfma_pos < active_topk_len
    if ASSUME_COMPACT_INDICES:
        final_valid = final_mfma_pos_valid
    else:
        final_valid = (
            final_mfma_pos_valid
            & (final_mfma_index >= 0)
            & (final_mfma_index.to(tl.int64) < num_kv_rows)
        )
    k_dot = cdna4_async.load_shared_relaxed(kv_smem.index(final_buffer), qk_b)
    scores = gl.zeros([BLOCK_H, BLOCK_K], gl.float32, mma)
    scores = gl.amd.cdna4.mfma(q_dot, k_dot, scores)
    scores = gl.where(final_valid[None, :], scores, float("-inf"))
    scores *= qk_scale
    new_max = gl.maximum(running_max, gl.max(scores, axis=1))
    alpha = gl.exp2(running_max - new_max)
    p = gl.exp2(scores - new_max[:, None])
    p = gl.where(final_valid[None, :], p, 0.0)
    running_sum = running_sum * alpha + gl.sum(p, axis=1)
    running_max = new_max
    v_dot = cdna4_async.load_shared_relaxed(
        kv_smem.index(final_buffer).permute([1, 0]), qk_b
    )
    p_dot = gl.convert_layout(p.to(kv.dtype.element_ty), qk_a)
    acc *= alpha[:, None]
    acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)

    final_sum = running_sum
    output_scale = 1.0 / gl.maximum(final_sum, 1.0e-30)
    output = gl.where(
        (final_sum > 0.0)[:, None],
        acc * output_scale[:, None],
        0.0,
    )

    # Store the first BF16 half while the second half changes layout.
    output_bf16 = output.to(o.dtype.element_ty)
    output_lo, output_hi = (
        output_bf16.reshape([BLOCK_H, 2, BLOCK_D // 2]).permute([0, 2, 1]).split()
    )
    out_head = head_block_idx * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, store_layout)
    )
    out_dim = gl.arange(0, BLOCK_D // 2, layout=gl.SliceLayout(0, store_layout))
    output_lo = gl.convert_layout(output_lo, store_layout)
    out_off = out_head[:, None] * stride_oh + out_dim[None, :] * stride_od
    gl.store(
        o + query_idx * stride_om + out_off,
        output_lo,
    )
    output_hi = gl.convert_layout(output_hi, store_layout)
    out_off = (
        out_head[:, None] * stride_oh + (BLOCK_D // 2 + out_dim[None, :]) * stride_od
    )
    gl.store(
        o + query_idx * stride_om + out_off,
        output_hi,
    )


def _select_block_k(num_queries: int, num_heads: int) -> int:
    # K64 is best while its one-CTA/CU footprint can cover the 256-CU device
    # in one scheduling wave.  Above that point K32's two-CTA/CU residency
    # hides enough latency to outweigh its doubled loop/index overhead.  Count
    # head blocks as independent CTAs so H128 crosses over at half the query
    # length of H64.
    num_ctas = num_queries * triton.cdiv(num_heads, 64)
    return 32 if num_ctas > 256 else 64


def gluon_dsv4_sparse_prefill_gfx950(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    block_k: int | None = None,
    assume_compact_indices: bool = True,
) -> torch.Tensor:
    """Launch the moved H64/D512 sparse-attention kernel for DeepSeek V4.

    This internal AMD implementation uses the public ``dsv4_prefill`` selected
    attention ABI: ``q`` is ``[tokens, heads, 512]``, ``kv`` is reshapeable to
    rows of 512 values, ``indices`` is ``[tokens, selected_width]``, and
    ``lens`` is ``[tokens]``.
    """

    s, h, d = q.shape
    assert d == 512
    assert h in (64, 128)
    assert indices.shape[0] == s
    assert indices.stride(1) == 1
    assert indices.size(1) >= 128
    assert lens.shape == (s,)

    q4 = q.unsqueeze(0).contiguous()
    kv3 = kv.reshape(1, -1, 512).contiguous()
    topk3 = indices.unsqueeze(0).contiguous()
    lens_1d = lens.contiguous()
    if out is None:
        output = torch.empty_like(q)
    else:
        output = out
    o4 = output.unsqueeze(0)

    if softmax_scale is None:
        softmax_scale = d**-0.5
    if block_k is None:
        block_k = _select_block_k(s, h)
    assert block_k in (32, 64)
    assert topk3.size(2) % block_k == 0

    num_xcds = 8
    grid = (num_xcds, triton.cdiv(h, 64), triton.cdiv(s, num_xcds))
    kernel = _sparse_attn_k64_kernel if block_k == 64 else _sparse_attn_k32_kernel
    kernel[grid](
        q4,
        kv3,
        o4,
        attn_sink.reshape(-1),
        topk3,
        lens_1d,
        q4.stride(1),
        q4.stride(2),
        q4.stride(3),
        kv3.stride(1),
        kv3.stride(2),
        o4.stride(1),
        o4.stride(2),
        o4.stride(3),
        topk3.stride(1),
        topk3.stride(2),
        lens_1d.stride(0),
        s,
        kv3.shape[1],
        topk3.size(2) // block_k,
        float(softmax_scale),
        BLOCK_H=64,
        BLOCK_D=512,
        NUM_XCDS=num_xcds,
        ASSUME_COMPACT_INDICES=assume_compact_indices,
        num_warps=4,
    )
    return output
