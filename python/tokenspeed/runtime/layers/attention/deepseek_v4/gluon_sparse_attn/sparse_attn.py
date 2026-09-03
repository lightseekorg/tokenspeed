"""CDNA4 sparse multi-head attention for gfx950.

This production path builds on the ROCm/AITER PR #3456 (MIT):
https://github.com/ROCm/aiter/pull/3456

This version adds TokenSpeed's selected-attention ABI, topk_lens support,
model-scale masking/addressing fixes, and the BLOCK_K=32 specialization used for
validated DeepSeek-V4 prefill shapes: B=1, D=512, H in {64, 128}, topk>=128.
"""


import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async


# Production H64/D512 sparse-attention path adapted from ROCm/AITER PR #3456 (MIT).
@gluon.jit
def _sparse_attn_kernel(
    q, kv, o, attn_sink, topk_idxs, topk_lens,
    stride_qm, stride_qh, stride_qd,
    stride_kvn, stride_kvd,
    stride_om, stride_oh, stride_od,
    stride_topk_m, stride_topk_k,
    stride_lens_m,
    num_queries, num_heads, topk_len, num_iters, scale,
    BLOCK_H: gl.constexpr,
    BLOCK_D: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    HAS_LENS: gl.constexpr,
    num_warps: gl.constexpr,
):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    qk_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    qk_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    store_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0])

    q_load_layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [num_warps, 1], [1, 0])
    kv_load_layout: gl.constexpr = gl.BlockedLayout(
        [8, BLOCK_K // num_warps], [64, 1], [1, num_warps], [0, 1])
    if BLOCK_K == 64:
        slot_load_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[],
            lane_bases=[[1], [2], [4], [8], [16], [32]],
            warp_bases=[[0], [0]],
            block_bases=[],
            shape=[BLOCK_K],
        )
    else:
        # The 32-wide async index layout does not lower cleanly. K32 loads
        # indices directly into the KV-column distribution instead.
        slot_load_layout: gl.constexpr = gl.SliceLayout(0, kv_load_layout)

    gl.static_assert(num_warps == 4)
    gl.static_assert(BLOCK_H == 64)
    gl.static_assert(BLOCK_D == 512)
    gl.static_assert(BLOCK_K == 32 or BLOCK_K == 64)

    q_smem_layout: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[
            [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32],
            [0, 64], [0, 128], [0, 256],
            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
        ],
        cga_layout=[],
        shape=[BLOCK_H, BLOCK_D],
    )
    if BLOCK_K == 64:
        kv_smem_layout: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                [64, 0], [128, 0], [256, 0],
                [0, 1], [0, 2], [0, 8], [0, 4], [0, 16], [0, 32],
            ],
            cga_layout=[],
            shape=[BLOCK_D, BLOCK_K],
        )
    else:
        kv_smem_layout: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                [64, 0], [128, 0], [256, 0],
                [0, 1], [0, 2], [0, 8], [0, 4], [0, 16],
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
    active_topk_len = gl.full((), topk_len, gl.int32)
    if HAS_LENS:
        active_topk_len = gl.load(topk_lens + query_idx * stride_lens_m).to(gl.int32)

    head_off = head_block_idx * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_q)
    dim_off = gl.arange(0, BLOCK_D, layout=sl_d_q)
    q_off = head_off[:, None] * stride_qh + dim_off[None, :] * stride_qd
    q_smem = gl.allocate_shared_memory(q.dtype.element_ty, [BLOCK_H, BLOCK_D], q_smem_layout)
    cdna4_async.buffer_load_to_shared(
        q_smem,
        q + query_idx * stride_qm,
        q_off,
        mask=(head_off < num_heads)[:, None],
        cache_modifier=".cg",
    )
    cdna4_async.commit_group()

    LOG2E: gl.constexpr = 1.4426950408889634
    qk_scale = scale * LOG2E
    sink_head = head_block_idx * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_mma)
    running_max = gl.load(
        attn_sink + sink_head,
        mask=sink_head < num_heads,
        other=float("-inf"),
    ).to(gl.float32) * LOG2E
    running_sum = gl.full([BLOCK_H], 1.0, gl.float32, sl_h_mma)
    acc = gl.zeros([BLOCK_H, BLOCK_D], gl.float32, mma)

    k_pos = gl.arange(0, BLOCK_K, layout=sl_k_kv)
    slot_off = gl.arange(0, BLOCK_K, layout=slot_load_layout)
    dim_kv = gl.arange(0, BLOCK_D, layout=sl_d_kv)
    topk_base = topk_idxs + query_idx * stride_topk_m

    index_smem = gl.allocate_shared_memory(topk_idxs.dtype.element_ty, [2, BLOCK_K], slot_smem_layout)
    if BLOCK_K == 64:
        cdna4_async.buffer_load_to_shared(
            index_smem.index(0),
            topk_base,
            slot_off * stride_topk_k,
            mask=(slot_off < topk_len) & (slot_off < active_topk_len),
        )
        cdna4_async.commit_group()
        cdna4_async.buffer_load_to_shared(
            index_smem.index(1),
            topk_base,
            (BLOCK_K + slot_off) * stride_topk_k,
            mask=((BLOCK_K + slot_off) < topk_len) & ((BLOCK_K + slot_off) < active_topk_len),
        )
        cdna4_async.commit_group()
        cdna4_async.wait_group(2)
    else:
        cdna4_async.wait_group(0)
    q_dot = cdna4_async.load_shared_relaxed(q_smem, qk_a)

    kv_smem = gl.allocate_shared_memory(kv.dtype.element_ty, [2, BLOCK_D, BLOCK_K], kv_smem_layout)

    if BLOCK_K == 64:
        cdna4_async.wait_group(1)
        index0 = cdna4_async.load_shared_relaxed(index_smem.index(0), sl_k_kv)
    else:
        index0 = gl.load(
            topk_base + k_pos * stride_topk_k,
            mask=(k_pos < topk_len) & (k_pos < active_topk_len),
            other=0,
        )
    valid0 = (k_pos < topk_len) & (k_pos < active_topk_len)
    kv_off0 = (
        dim_kv[:, None] * stride_kvd
        + gl.where(valid0, index0, 0)[None, :] * stride_kvn)
    cdna4_async.buffer_load_to_shared(kv_smem.index(0), kv, kv_off0)
    cdna4_async.commit_group()

    # Stage indices two tiles ahead and KV one tile ahead.
    for i in tl.range(0, num_iters - 2):
        if BLOCK_K == 64:
            future_index_pos = (i + 2) * BLOCK_K + slot_off
            cdna4_async.buffer_load_to_shared(
                index_smem.index(i % 2),
                topk_base,
                future_index_pos * stride_topk_k,
                mask=(future_index_pos < topk_len) & (future_index_pos < active_topk_len),
            )
            cdna4_async.commit_group()
            cdna4_async.wait_group(1)
        else:
            cdna4_async.wait_group(0)
        current_buffer = i % 2
        k_dot = cdna4_async.load_shared_relaxed(
            kv_smem.index(current_buffer), qk_b)
        scores = gl.zeros([BLOCK_H, BLOCK_K], gl.float32, mma)
        scores = gl.amd.cdna4.mfma(q_dot, k_dot, scores)

        next_buffer = (i + 1) % 2
        if BLOCK_K == 64:
            cdna4_async.wait_group(2)
            next_index = cdna4_async.load_shared_relaxed(
                index_smem.index(next_buffer), sl_k_kv)
        else:
            next_pos = (i + 1) * BLOCK_K + k_pos
            next_index = gl.load(
                topk_base + next_pos * stride_topk_k,
                mask=(next_pos < topk_len) & (next_pos < active_topk_len),
                other=0,
            )
        next_pos = (i + 1) * BLOCK_K + k_pos
        next_valid = (next_pos < topk_len) & (next_pos < active_topk_len)
        next_kv_off = (
            dim_kv[:, None] * stride_kvd
            + gl.where(next_valid, next_index, 0)[None, :] * stride_kvn)
        cdna4_async.buffer_load_to_shared(
            kv_smem.index(next_buffer), kv, next_kv_off, mask=next_valid[None, :])
        cdna4_async.commit_group()

        current_pos = i * BLOCK_K + gl.arange(0, BLOCK_K, layout=sl_k_mma)
        current_valid = (current_pos < topk_len) & (current_pos < active_topk_len)
        scores *= qk_scale
        scores = gl.where(current_valid[None, :], scores, float("-inf"))
        new_max = gl.maximum(running_max, gl.max(scores, axis=1))
        alpha = gl.exp2(running_max - new_max)
        p = gl.exp2(scores - new_max[:, None])
        p = gl.where(current_valid[None, :], p, 0.0)
        running_sum = running_sum * alpha + gl.sum(p, axis=1)
        running_max = new_max
        v_dot = cdna4_async.load_shared_relaxed(
            kv_smem.index(current_buffer).permute([1, 0]), qk_b)
        p_dot = gl.convert_layout(p.to(kv.dtype.element_ty), qk_a)
        acc *= alpha[:, None]
        acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)

    # Load the final KV tile, then drain the final two tiles.
    final_buffer = (num_iters - 1) % 2
    final_pos = (num_iters - 1) * BLOCK_K + k_pos
    if BLOCK_K == 64:
        cdna4_async.wait_group(1)
        final_index = cdna4_async.load_shared_relaxed(
            index_smem.index(final_buffer), sl_k_kv)
    else:
        final_index = gl.load(
            topk_base + final_pos * stride_topk_k,
            mask=(final_pos < topk_len) & (final_pos < active_topk_len),
            other=0,
        )
    final_load_valid = (final_pos < topk_len) & (final_pos < active_topk_len)
    final_kv_off = (
        dim_kv[:, None] * stride_kvd
        + gl.where(final_load_valid, final_index, 0)[None, :] * stride_kvn)
    cdna4_async.buffer_load_to_shared(
        kv_smem.index(final_buffer), kv, final_kv_off, mask=final_load_valid[None, :])
    cdna4_async.commit_group()

    cdna4_async.wait_group(1)
    penultimate_tile = num_iters - 2
    penultimate_buffer = penultimate_tile % 2
    penultimate_valid = (
        penultimate_tile * BLOCK_K
        + gl.arange(0, BLOCK_K, layout=sl_k_mma)) < topk_len
    penultimate_valid = penultimate_valid & ((
        penultimate_tile * BLOCK_K
        + gl.arange(0, BLOCK_K, layout=sl_k_mma)) < active_topk_len)
    k_dot = cdna4_async.load_shared_relaxed(
        kv_smem.index(penultimate_buffer), qk_b)
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
        kv_smem.index(penultimate_buffer).permute([1, 0]), qk_b)
    p_dot = gl.convert_layout(p.to(kv.dtype.element_ty), qk_a)
    acc *= alpha[:, None]
    acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)

    cdna4_async.wait_group(0)
    final_valid = (
        (num_iters - 1) * BLOCK_K
        + gl.arange(0, BLOCK_K, layout=sl_k_mma)) < topk_len
    final_valid = final_valid & ((
        (num_iters - 1) * BLOCK_K
        + gl.arange(0, BLOCK_K, layout=sl_k_mma)) < active_topk_len)
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
        kv_smem.index(final_buffer).permute([1, 0]), qk_b)
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
    output_lo, output_hi = output_bf16.reshape(
        [BLOCK_H, 2, BLOCK_D // 2]).permute([0, 2, 1]).split()
    out_head = head_block_idx * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, store_layout))
    out_dim = gl.arange(
        0, BLOCK_D // 2, layout=gl.SliceLayout(0, store_layout))
    output_lo = gl.convert_layout(output_lo, store_layout)
    out_off = out_head[:, None] * stride_oh + out_dim[None, :] * stride_od
    gl.store(
        o + query_idx * stride_om + out_off, output_lo,
        mask=(out_head < num_heads)[:, None],
    )
    output_hi = gl.convert_layout(output_hi, store_layout)
    out_off = (
        out_head[:, None] * stride_oh
        + (BLOCK_D // 2 + out_dim[None, :]) * stride_od)
    gl.store(
        o + query_idx * stride_om + out_off, output_hi,
        mask=(out_head < num_heads)[:, None],
    )


def _select_block_k(num_queries: int, num_heads: int) -> int:
    # K64 is best while its one-CTA/CU footprint can cover the 256-CU device
    # in one scheduling wave.  Above that point K32's two-CTA/CU residency
    # hides enough latency to outweigh its doubled loop/index overhead.  Count
    # head blocks as independent CTAs so H128 crosses over at half the query
    # length of H64.
    num_ctas = num_queries * triton.cdiv(num_heads, 64)
    return 32 if num_ctas > 256 else 64


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    topk_lens: torch.Tensor | None = None,
    block_k: int = None,
) -> torch.Tensor:
    """Launch the production H64/D512 sparse-attention kernel."""
    b, s, h, d = q.shape
    assert b == 1
    assert d == 512
    assert h in (64, 128)
    assert topk_idxs.shape[:2] == (b, s)
    assert topk_idxs.stride(2) == 1
    assert topk_idxs.size(2) >= 128
    has_lens = topk_lens is not None
    if has_lens:
        if topk_lens.dim() == 1:
            assert tuple(topk_lens.shape) == (s,)
            lens_1d = topk_lens
        elif topk_lens.dim() == 2:
            assert tuple(topk_lens.shape) == (b, s)
            lens_1d = topk_lens.reshape(-1)
        else:
            raise AssertionError("topk_lens must have shape [s] or [b, s]")
        if lens_1d.device != q.device:
            lens_1d = lens_1d.to(device=q.device)
        if lens_1d.dtype != torch.int32:
            lens_1d = lens_1d.to(torch.int32)
        lens_1d = lens_1d.contiguous()
        stride_lens_m = lens_1d.stride(0)
    else:
        lens_1d = topk_idxs
        stride_lens_m = 0
    if softmax_scale is None:
        softmax_scale = d**-0.5
    if block_k is None:
        block_k = _select_block_k(s, h)
    assert block_k in (32, 64)

    o = torch.empty_like(q)
    num_xcds = 8
    grid = (num_xcds, triton.cdiv(h, 64), triton.cdiv(s, num_xcds))
    _sparse_attn_kernel[grid](
        q, kv, o, attn_sink, topk_idxs, lens_1d,
        q.stride(1), q.stride(2), q.stride(3),
        kv.stride(1), kv.stride(2),
        o.stride(1), o.stride(2), o.stride(3),
        topk_idxs.stride(1), topk_idxs.stride(2),
        stride_lens_m,
        s, h, topk_idxs.size(2), triton.cdiv(topk_idxs.size(2), block_k),
        float(softmax_scale),
        BLOCK_H=64,
        BLOCK_D=512,
        BLOCK_K=block_k,
        NUM_XCDS=num_xcds,
        HAS_LENS=has_lens,
        num_warps=4,
    )
    return o
