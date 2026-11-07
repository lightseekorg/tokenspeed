# Copyright (c) 2026 LightSeek Foundation
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
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

"""GFX950 DSA logits kernels for TokenSpeed's standard cache.

The scheduling and FP8 matrix structure follow the corresponding AITER scorer
design. This implementation directly supports TokenSpeed's standard page-64
block-split cache, resolves prefill workspace indirection in the key loader,
adds native BF16 matrix computation, and evaluates 64 heads as two 32-head
reductions over one resident key tile.

The ``*_bf16_*`` scorers read an unquantized bf16 ``[slots, head_dim]`` plane
instead. They score a block of query rows per program so every gathered key
tile is shared by the block's rows (see ``_multi_row_index_k_scores_body``);
the ``*_fp8_rows_*`` scorers do the same over the standard fp8_scaled pages.
"""

from tokenspeed_kernel_amd._triton import gl, gluon, tl

__all__ = [
    "gluon_kpool_prefill_topk_fp8_gfx950",
    "gluon_kpool_prefill_topk_fp8_plan_gfx950",
    "gluon_dsa_decode_topk_bf16_gfx950",
    "gluon_dsa_decode_topk_fp8_rows_gfx950",
    "gluon_dsa_decode_topk_standard_gfx950",
    "gluon_dsa_prefill_topk_bf16_gfx950",
    "gluon_dsa_prefill_topk_fp8_rows_gfx950",
    "gluon_dsa_prefill_topk_standard_gfx950",
]


@gluon.jit
def _relu_f32(value):
    return gl.maximum(value, 0.0, propagate_nan=tl.PropagateNan.ALL)


@gluon.constexpr_function
def _make_key_blocked_layout(
    HEAD_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    vector_width: gl.constexpr = 16
    head_threads: gl.constexpr = HEAD_DIM // vector_width
    return gl.BlockedLayout(
        size_per_thread=[vector_width, 1],
        threads_per_warp=[head_threads, 64 // head_threads],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )


@gluon.constexpr_function
def _make_key_shared_layout():
    return gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0, 1])


@gluon.constexpr_function
def _make_mfma_layout(
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    Q_IS_FP8: gl.constexpr,
):
    if Q_IS_FP8:
        mfma_layout = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[32, 32, 64],
            transposed=False,
            warps_per_cta=[1, NUM_WARPS],
        )
    else:
        mfma_layout = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[32, 32, 16],
            transposed=False,
            warps_per_cta=[1, NUM_WARPS],
        )
    return mfma_layout


@gluon.constexpr_function
def _make_dot_layout(
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    Q_IS_FP8: gl.constexpr,
    operand_index: gl.constexpr,
):
    mfma_layout = _make_mfma_layout(BLOCK_N, NUM_WARPS, Q_IS_FP8)
    k_width: gl.constexpr = 16 if Q_IS_FP8 else 8
    return gl.DotOperandLayout(
        operand_index=operand_index,
        parent=mfma_layout,
        k_width=k_width,
    )


@gluon.jit
def _candidate_slots(
    positions,
    valid,
    kv_workspace_slots,
    block_table,
    request_id,
    block_table_stride,
    PAGE_SIZE: gl.constexpr,
    IS_PREFILL: gl.constexpr,
):
    if IS_PREFILL:
        return gl.amd.cdna4.buffer_load(
            ptr=kv_workspace_slots,
            offsets=positions.to(gl.int32),
            mask=valid,
            other=0,
        ).to(gl.int64)
    page_indices = positions // PAGE_SIZE
    pages = gl.amd.cdna4.buffer_load(
        ptr=block_table,
        offsets=(request_id * block_table_stride + page_indices).to(gl.int32),
        mask=valid,
        other=0,
    ).to(gl.int64)
    return pages * PAGE_SIZE + positions % PAGE_SIZE


@gluon.jit
def _load_key_tile_to_shared(
    index_k_fp8,
    key_shared,
    buffer_id,
    candidate_start,
    candidate_end,
    kv_workspace_slots,
    block_table,
    request_id,
    block_table_stride,
    PAGE_SIZE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    IS_PREFILL: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
):
    blocked: gl.constexpr = _make_key_blocked_layout(HEAD_DIM, BLOCK_N, NUM_WARPS)
    dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(1, blocked))[:, None]
    columns = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked))[None, :]
    positions = candidate_start + columns
    valid = positions < candidate_end
    slots = _candidate_slots(
        positions,
        valid,
        kv_workspace_slots,
        block_table,
        request_id,
        block_table_stride,
        PAGE_SIZE,
        IS_PREFILL,
    )
    pages = slots // PAGE_SIZE
    page_rows = slots - pages * PAGE_SIZE
    byte_offsets = pages * PAGE_STRIDE_BYTES + page_rows * HEAD_DIM + dims
    if USE_BUFFER_LOAD:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            key_shared.index(buffer_id),
            index_k_fp8,
            byte_offsets.to(gl.int32),
            mask=valid,
        )
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(
            key_shared.index(buffer_id),
            index_k_fp8 + byte_offsets,
            mask=valid,
        )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _load_key_scales(
    index_k_scale,
    candidate_start,
    candidate_end,
    kv_workspace_slots,
    block_table,
    request_id,
    block_table_stride,
    output_layout: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_PREFILL: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
):
    columns = gl.arange(0, BLOCK_N, layout=output_layout)
    positions = candidate_start + columns
    valid = positions < candidate_end
    slots = _candidate_slots(
        positions,
        valid,
        kv_workspace_slots,
        block_table,
        request_id,
        block_table_stride,
        PAGE_SIZE,
        IS_PREFILL,
    )
    pages = slots // PAGE_SIZE
    page_rows = slots - pages * PAGE_SIZE
    scale_offsets = (
        pages * (PAGE_STRIDE_BYTES // 4) + (PAGE_SIZE * HEAD_DIM) // 4 + page_rows
    )
    if USE_BUFFER_LOAD:
        scales = gl.amd.cdna4.buffer_load(
            ptr=index_k_scale,
            offsets=scale_offsets.to(gl.int32),
            mask=valid,
            other=0.0,
        ).to(gl.float32)
    else:
        scales = gl.load(
            index_k_scale + scale_offsets,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
    return scales, valid


@gluon.jit
def _load_query_tile(
    q,
    weights,
    q_scales,
    row_id,
    head_offset: gl.constexpr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_w_row,
    stride_w_head,
    stride_qs_row,
    stride_qs_head,
    layout_q: gl.constexpr,
    mfma_layout: gl.constexpr,
    dot_a_layout: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    Q_IS_FP8: gl.constexpr,
):
    heads = gl.arange(0, 32, layout=gl.SliceLayout(1, layout_q))[:, None]
    dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, layout_q))[None, :]
    q_values = gl.amd.cdna4.buffer_load(
        ptr=q,
        offsets=(
            row_id * stride_q_row
            + (head_offset + heads) * stride_q_head
            + dims * stride_q_dim
        ).to(gl.int32),
        cache=".cg",
    )
    weight_heads = gl.arange(0, 32, layout=gl.SliceLayout(1, mfma_layout))
    head_weights = gl.amd.cdna4.buffer_load(
        ptr=weights,
        offsets=(
            row_id * stride_w_row + (head_offset + weight_heads) * stride_w_head
        ).to(gl.int32),
        cache=".cg",
    ).to(gl.float32)
    if Q_IS_FP8:
        query_scales = gl.amd.cdna4.buffer_load(
            ptr=q_scales,
            offsets=(
                row_id * stride_qs_row + (head_offset + weight_heads) * stride_qs_head
            ).to(gl.int32),
            cache=".cg",
        ).to(gl.float32)
        head_weights *= query_scales
    return gl.convert_layout(q_values, dot_a_layout), head_weights


@gluon.jit
def _score_head_tile(
    query,
    key,
    head_weights,
    mfma_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    Q_IS_FP8: gl.constexpr,
):
    accumulator = gl.zeros([32, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
    if Q_IS_FP8:
        head_scores = gl.amd.cdna4.mfma_scaled(
            a=query,
            a_scale=None,
            a_format="e4m3",
            b=key,
            b_scale=None,
            b_format="e4m3",
            acc=accumulator,
        )
    else:
        head_scores = gl.amd.cdna4.mfma(query, key, accumulator)
    head_scores = _relu_f32(head_scores)
    return gl.sum(head_scores * head_weights[:, None], axis=0)


@gluon.jit
def _fold_weighted_heads_in_logical_order(
    head_scores,
    head_weights,
    mfma_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    contributions = head_scores * head_weights[:, None]
    output_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    gather_indices = gl.zeros(
        [BLOCK_N],
        dtype=gl.int32,
        layout=output_layout,
    )[None, :]
    scores = gl.zeros([BLOCK_N], dtype=gl.float32, layout=output_layout)
    for head in gl.static_range(0, 32):
        contribution = gl.gather(contributions, gather_indices + head, axis=0)
        contribution = gl.convert_layout(
            gl.reshape(contribution, [BLOCK_N]),
            output_layout,
        )
        scores += contribution
    return scores


@gluon.jit
def _score_head_tile_ordered_head_fold(
    query,
    key,
    head_weights,
    mfma_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    Q_IS_FP8: gl.constexpr,
):
    accumulator = gl.zeros([32, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
    if Q_IS_FP8:
        head_scores = gl.amd.cdna4.mfma_scaled(
            a=query,
            a_scale=None,
            a_format="e4m3",
            b=key,
            b_scale=None,
            b_format="e4m3",
            acc=accumulator,
        )
    else:
        head_scores = gl.amd.cdna4.mfma(query, key, accumulator)
    head_scores = _relu_f32(head_scores)
    return _fold_weighted_heads_in_logical_order(
        head_scores,
        head_weights,
        mfma_layout,
        BLOCK_N,
    )


@gluon.jit
def _score_key_tile(
    query_0,
    weight_0,
    query_1,
    weight_1,
    raw_key,
    key_scales,
    model_scale,
    mfma_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    Q_IS_FP8: gl.constexpr,
):
    # Change layout while the key is still FP8: it moves half the bytes through
    # LDS and keeps fewer registers live than converting BF16.
    key = gl.convert_layout(raw_key, dot_b_layout)
    if not Q_IS_FP8:
        key = key.to(gl.bfloat16)
    scores = _score_head_tile(
        query_0,
        key,
        weight_0,
        mfma_layout,
        BLOCK_N,
        Q_IS_FP8,
    )
    if NUM_HEADS == 64:
        scores += _score_head_tile(
            query_1,
            key,
            weight_1,
            mfma_layout,
            BLOCK_N,
            Q_IS_FP8,
        )
    return scores * key_scales * model_scale


@gluon.jit
def _score_key_tile_ordered_head_fold(
    query_0,
    weight_0,
    query_1,
    weight_1,
    raw_key,
    key_scales,
    model_scale,
    mfma_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    Q_IS_FP8: gl.constexpr,
):
    # Change layout while the key is still FP8: it moves half the bytes through
    # LDS and keeps fewer registers live than converting BF16.
    key = gl.convert_layout(raw_key, dot_b_layout)
    if not Q_IS_FP8:
        key = key.to(gl.bfloat16)
    scores = _score_head_tile_ordered_head_fold(
        query_0,
        key,
        weight_0,
        mfma_layout,
        BLOCK_N,
        Q_IS_FP8,
    )
    if NUM_HEADS == 64:
        scores += _score_head_tile_ordered_head_fold(
            query_1,
            key,
            weight_1,
            mfma_layout,
            BLOCK_N,
            Q_IS_FP8,
        )
    return scores * key_scales * model_scale


@gluon.jit
def _standard_cache_logits_body(
    q,
    q_scales,
    index_k_fp8,
    index_k_scale,
    weights,
    kv_workspace_slots,
    row_starts,
    row_ends,
    seq_lens,
    req_ids,
    block_table,
    logits,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_qs_row,
    stride_qs_head,
    stride_w_row,
    stride_w_head,
    block_table_stride,
    logits_stride,
    model_scale,
    max_candidates,
    q_len_per_req,
    pool_offset,
    PAGE_SIZE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    POOL_SIZE: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    Q_IS_FP8: gl.constexpr,
    IS_PREFILL: gl.constexpr,
    IS_KPOOL: gl.constexpr,
    ORDERED_HEAD_FOLD: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    row_id = gl.program_id(0)
    split_id = gl.program_id(1)
    if IS_KPOOL:
        if IS_PREFILL:
            request_id = 0
            workspace_start = gl.maximum(gl.load(row_starts + row_id), 0)
            workspace_end = gl.minimum(gl.load(row_ends + row_id), max_candidates)
            output_base = workspace_start + pool_offset
            candidate_start = gl.minimum(output_base, workspace_end)
            candidate_end = gl.minimum(output_base + CHUNK_N, workspace_end)
        else:
            request_id = gl.load(req_ids + row_id).to(gl.int32)
            num_pools = (
                gl.maximum(gl.load(seq_lens + row_id).to(gl.int32), 0) // POOL_SIZE
            )
            output_base = pool_offset
            candidate_start = pool_offset
            candidate_end = gl.minimum(num_pools, pool_offset + CHUNK_N)
            candidate_end = gl.minimum(candidate_end, max_candidates)
    elif IS_PREFILL:
        request_id = row_id
        output_base = 0
        candidate_start = gl.maximum(gl.load(row_starts + row_id), 0)
        candidate_end = gl.minimum(gl.load(row_ends + row_id), max_candidates)
    else:
        request_id = row_id // q_len_per_req
        output_base = 0
        q_offset = row_id - request_id * q_len_per_req
        candidate_start = split_id * CHUNK_N
        candidate_end = gl.load(seq_lens + request_id).to(gl.int32)
        if q_len_per_req != 1:
            candidate_end = candidate_end - (q_len_per_req - 1) + q_offset
        candidate_end = gl.minimum(candidate_end, candidate_start + CHUNK_N)
        candidate_end = gl.minimum(candidate_end, max_candidates)
    candidate_start = gl.minimum(candidate_start, candidate_end)

    blocked: gl.constexpr = _make_key_blocked_layout(HEAD_DIM, BLOCK_N, NUM_WARPS)
    shared_layout: gl.constexpr = _make_key_shared_layout()
    mfma_layout: gl.constexpr = _make_mfma_layout(BLOCK_N, NUM_WARPS, Q_IS_FP8)
    dot_a_layout: gl.constexpr = _make_dot_layout(BLOCK_N, NUM_WARPS, Q_IS_FP8, 0)
    dot_b_layout: gl.constexpr = _make_dot_layout(BLOCK_N, NUM_WARPS, Q_IS_FP8, 1)
    key_shared = gl.allocate_shared_memory(
        index_k_fp8.type.element_ty,
        [2, HEAD_DIM, BLOCK_N],
        layout=shared_layout,
    )
    q_groups: gl.constexpr = HEAD_DIM // 16
    layout_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[64 // q_groups, q_groups],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    query_0, weight_0 = _load_query_tile(
        q,
        weights,
        q_scales,
        row_id,
        0,
        stride_q_row,
        stride_q_head,
        stride_q_dim,
        stride_w_row,
        stride_w_head,
        stride_qs_row,
        stride_qs_head,
        layout_q,
        mfma_layout,
        dot_a_layout,
        HEAD_DIM,
        Q_IS_FP8,
    )
    if NUM_HEADS == 64:
        query_1, weight_1 = _load_query_tile(
            q,
            weights,
            q_scales,
            row_id,
            32,
            stride_q_row,
            stride_q_head,
            stride_q_dim,
            stride_w_row,
            stride_w_head,
            stride_qs_row,
            stride_qs_head,
            layout_q,
            mfma_layout,
            dot_a_layout,
            HEAD_DIM,
            Q_IS_FP8,
        )
    else:
        query_1 = query_0
        weight_1 = weight_0

    tile_count = tl.cdiv(candidate_end - candidate_start, BLOCK_N)
    _load_key_tile_to_shared(
        index_k_fp8,
        key_shared,
        0,
        candidate_start,
        candidate_end,
        kv_workspace_slots,
        block_table,
        request_id,
        block_table_stride,
        PAGE_SIZE,
        PAGE_STRIDE_BYTES,
        HEAD_DIM,
        BLOCK_N,
        NUM_WARPS,
        IS_PREFILL,
        USE_BUFFER_LOAD,
    )
    _load_key_tile_to_shared(
        index_k_fp8,
        key_shared,
        1,
        candidate_start + BLOCK_N,
        candidate_end,
        kv_workspace_slots,
        block_table,
        request_id,
        block_table_stride,
        PAGE_SIZE,
        PAGE_STRIDE_BYTES,
        HEAD_DIM,
        BLOCK_N,
        NUM_WARPS,
        IS_PREFILL,
        USE_BUFFER_LOAD,
    )

    output_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    output_columns = gl.arange(0, BLOCK_N, layout=output_layout)
    current_buffer: gl.int32 = 0
    for tile_id in tl.range(0, tile_count):
        tile_start = candidate_start + tile_id * BLOCK_N
        key_scales, valid = _load_key_scales(
            index_k_scale,
            tile_start,
            candidate_end,
            kv_workspace_slots,
            block_table,
            request_id,
            block_table_stride,
            output_layout,
            PAGE_SIZE,
            PAGE_STRIDE_BYTES,
            HEAD_DIM,
            BLOCK_N,
            IS_PREFILL,
            USE_BUFFER_LOAD,
        )
        if tile_id + 1 < tile_count:
            gl.amd.cdna4.async_copy.wait_group(1)
        else:
            # The final tile must be complete before it is read. This also
            # drains the speculative masked preload for a one-tile span.
            gl.amd.cdna4.async_copy.wait_group(0)
        raw_key = key_shared.index(current_buffer).load(layout=blocked)
        if tile_id + 2 < tile_count:
            _load_key_tile_to_shared(
                index_k_fp8,
                key_shared,
                current_buffer,
                tile_start + 2 * BLOCK_N,
                candidate_end,
                kv_workspace_slots,
                block_table,
                request_id,
                block_table_stride,
                PAGE_SIZE,
                PAGE_STRIDE_BYTES,
                HEAD_DIM,
                BLOCK_N,
                NUM_WARPS,
                IS_PREFILL,
                USE_BUFFER_LOAD,
            )
        if ORDERED_HEAD_FOLD:
            scores = _score_key_tile_ordered_head_fold(
                query_0,
                weight_0,
                query_1,
                weight_1,
                raw_key,
                key_scales,
                model_scale,
                mfma_layout,
                dot_b_layout,
                NUM_HEADS,
                BLOCK_N,
                Q_IS_FP8,
            )
        else:
            scores = _score_key_tile(
                query_0,
                weight_0,
                query_1,
                weight_1,
                raw_key,
                key_scales,
                model_scale,
                mfma_layout,
                dot_b_layout,
                NUM_HEADS,
                BLOCK_N,
                Q_IS_FP8,
            )
        output_column = tile_start - output_base if IS_KPOOL else tile_start
        output_offsets = (
            row_id.to(gl.int64) * logits_stride
            + output_column.to(gl.int64)
            + output_columns.to(gl.int64)
        )
        if USE_BUFFER_STORE:
            gl.amd.cdna4.buffer_store(
                scores,
                ptr=logits,
                offsets=output_offsets.to(gl.int32),
                mask=valid,
            )
        else:
            gl.store(logits + output_offsets, scores, mask=valid)
        current_buffer = 1 - current_buffer
    # Empty spans still issued two fully masked speculative preloads.
    gl.amd.cdna4.async_copy.wait_group(0)


@gluon.jit
def gluon_dsa_prefill_topk_standard_gfx950(
    q,
    q_scales,
    index_k_fp8,
    index_k_scale,
    weights,
    kv_workspace_slots,
    row_starts,
    row_ends,
    block_table,
    logits,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_qs_row,
    stride_qs_head,
    stride_w_row,
    stride_w_head,
    logits_stride,
    model_scale,
    workspace_rows,
    PAGE_SIZE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    Q_IS_FP8: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    _standard_cache_logits_body(
        q,
        q_scales,
        index_k_fp8,
        index_k_scale,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        row_ends,
        row_ends,
        block_table,
        logits,
        stride_q_row,
        stride_q_head,
        stride_q_dim,
        stride_qs_row,
        stride_qs_head,
        stride_w_row,
        stride_w_head,
        0,
        logits_stride,
        model_scale,
        workspace_rows,
        1,
        0,
        PAGE_SIZE,
        PAGE_STRIDE_BYTES,
        1,
        NUM_HEADS,
        HEAD_DIM,
        BLOCK_N,
        BLOCK_N,
        NUM_WARPS,
        Q_IS_FP8,
        True,
        False,
        False,
        USE_BUFFER_LOAD,
        USE_BUFFER_STORE,
    )


@gluon.jit
def gluon_kpool_prefill_topk_fp8_gfx950(
    q,
    q_scales,
    index_k_fp8,
    index_k_scale,
    weights,
    causal_lens,
    req_ids,
    block_table,
    logits,
    row_ends,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_qs_row,
    stride_qs_head,
    stride_w_row,
    stride_w_head,
    block_table_stride,
    logits_stride,
    model_scale,
    max_candidates,
    pool_offset,
    PAGE_SIZE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    POOL_SIZE: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    WINDOW_COLS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    ORDERED_HEAD_FOLD: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    row_id = gl.program_id(0)
    num_pools = gl.maximum(gl.load(causal_lens + row_id).to(gl.int32), 0) // POOL_SIZE
    window_end = gl.minimum(num_pools, pool_offset + WINDOW_COLS)
    window_end = gl.minimum(window_end, max_candidates)
    local_end = gl.maximum(window_end - pool_offset, 0)
    gl.store(row_ends + row_id, local_end)

    _standard_cache_logits_body(
        q,
        q_scales,
        index_k_fp8,
        index_k_scale,
        weights,
        block_table,
        causal_lens,
        row_ends,
        causal_lens,
        req_ids,
        block_table,
        logits,
        stride_q_row,
        stride_q_head,
        stride_q_dim,
        stride_qs_row,
        stride_qs_head,
        stride_w_row,
        stride_w_head,
        block_table_stride,
        logits_stride,
        model_scale,
        max_candidates,
        1,
        pool_offset,
        PAGE_SIZE,
        PAGE_STRIDE_BYTES,
        POOL_SIZE,
        NUM_HEADS,
        HEAD_DIM,
        BLOCK_N,
        WINDOW_COLS,
        NUM_WARPS,
        False,
        False,
        True,
        ORDERED_HEAD_FOLD,
        USE_BUFFER_LOAD,
        USE_BUFFER_STORE,
    )


@gluon.jit
def gluon_kpool_prefill_topk_fp8_plan_gfx950(
    q,
    q_scales,
    index_k_fp8,
    index_k_scale,
    weights,
    pool_workspace_slots,
    row_starts,
    row_ends,
    logits,
    local_row_ends,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_qs_row,
    stride_qs_head,
    stride_w_row,
    stride_w_head,
    logits_stride,
    model_scale,
    workspace_rows,
    pool_offset,
    PAGE_SIZE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    POOL_SIZE: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    WINDOW_COLS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    ORDERED_HEAD_FOLD: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    row_id = gl.program_id(0)
    workspace_start = gl.maximum(gl.load(row_starts + row_id), 0)
    workspace_end = gl.minimum(gl.load(row_ends + row_id), workspace_rows)
    row_pools = gl.maximum(workspace_end - workspace_start, 0)
    local_end = gl.minimum(gl.maximum(row_pools - pool_offset, 0), WINDOW_COLS)
    gl.store(local_row_ends + row_id, local_end)

    _standard_cache_logits_body(
        q,
        q_scales,
        index_k_fp8,
        index_k_scale,
        weights,
        pool_workspace_slots,
        row_starts,
        row_ends,
        row_ends,
        row_ends,
        row_ends,
        logits,
        stride_q_row,
        stride_q_head,
        stride_q_dim,
        stride_qs_row,
        stride_qs_head,
        stride_w_row,
        stride_w_head,
        0,
        logits_stride,
        model_scale,
        workspace_rows,
        1,
        pool_offset,
        PAGE_SIZE,
        PAGE_STRIDE_BYTES,
        POOL_SIZE,
        NUM_HEADS,
        HEAD_DIM,
        BLOCK_N,
        WINDOW_COLS,
        NUM_WARPS,
        False,
        True,
        True,
        ORDERED_HEAD_FOLD,
        USE_BUFFER_LOAD,
        USE_BUFFER_STORE,
    )


@gluon.jit
def gluon_dsa_decode_topk_standard_gfx950(
    q,
    q_scales,
    index_k_fp8,
    index_k_scale,
    weights,
    seq_lens,
    block_table,
    logits,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_qs_row,
    stride_qs_head,
    stride_w_row,
    stride_w_head,
    block_table_stride,
    logits_stride,
    model_scale,
    max_candidates,
    q_len_per_req,
    PAGE_SIZE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    Q_IS_FP8: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    _standard_cache_logits_body(
        q,
        q_scales,
        index_k_fp8,
        index_k_scale,
        weights,
        block_table,
        seq_lens,
        seq_lens,
        seq_lens,
        seq_lens,
        block_table,
        logits,
        stride_q_row,
        stride_q_head,
        stride_q_dim,
        stride_qs_row,
        stride_qs_head,
        stride_w_row,
        stride_w_head,
        block_table_stride,
        logits_stride,
        model_scale,
        max_candidates,
        q_len_per_req,
        0,
        PAGE_SIZE,
        PAGE_STRIDE_BYTES,
        1,
        NUM_HEADS,
        HEAD_DIM,
        BLOCK_N,
        CHUNK_N,
        NUM_WARPS,
        Q_IS_FP8,
        False,
        False,
        False,
        USE_BUFFER_LOAD,
        USE_BUFFER_STORE,
    )


def _bf16_index_k_launch_metadata(grid, kernel, args):
    rows = args["num_rows"]
    heads = args["NUM_HEADS"]
    head_dim = args["HEAD_DIM"]
    cols = args["num_candidates"]
    # Upper bound: every row scores every candidate.
    return {
        "name": kernel.name,
        "flops16": 2 * rows * heads * head_dim * cols,
        "bytes": grid[0] * cols * head_dim * 2
        + grid[1] * rows * heads * head_dim * 2
        + rows * cols * 4,
    }


@gluon.jit
def _load_query_group(
    q,
    weights,
    q_scales,
    m_rows,
    m_valid,
    head_offset: gl.constexpr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_w_row,
    stride_w_head,
    stride_qs_row,
    stride_qs_head,
    mfma_layout: gl.constexpr,
    dot_a_layout: gl.constexpr,
    ROWS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    Q_FP8: gl.constexpr,
):
    # Row m = query m // 32, head head_offset + m % 32; FP8 q scales (> 0) fold into w.
    layout_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[64 // (HEAD_DIM // 16), HEAD_DIM // 16],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    m = gl.arange(0, ROWS * 32, layout=gl.SliceLayout(1, layout_q))
    dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, layout_q))
    q_rows = gl.convert_layout(m_rows, gl.SliceLayout(1, layout_q))
    q_valid = gl.convert_layout(m_valid, gl.SliceLayout(1, layout_q))
    q_values = gl.amd.cdna4.buffer_load(
        ptr=q,
        offsets=(
            q_rows[:, None] * stride_q_row
            + (head_offset + m[:, None] % 32) * stride_q_head
            + dims[None, :] * stride_q_dim
        ).to(gl.int32),
        mask=q_valid[:, None],
        other=0.0,
        cache=".cg",
    )
    w_m = gl.arange(0, ROWS * 32, layout=gl.SliceLayout(1, mfma_layout))
    head_weights = gl.amd.cdna4.buffer_load(
        ptr=weights,
        offsets=(m_rows * stride_w_row + (head_offset + w_m % 32) * stride_w_head).to(
            gl.int32
        ),
        mask=m_valid,
        other=0.0,
        cache=".cg",
    ).to(gl.float32)
    if Q_FP8:
        head_weights *= gl.amd.cdna4.buffer_load(
            ptr=q_scales,
            offsets=(
                m_rows * stride_qs_row + (head_offset + w_m % 32) * stride_qs_head
            ).to(gl.int32),
            mask=m_valid,
            other=0.0,
            cache=".cg",
        )
    return gl.convert_layout(q_values, dot_a_layout), head_weights


@gluon.jit
def _copy_key_tile(
    index_k,
    key_shared,
    buffer_id,
    slot_shared,
    tile_id,
    tile_start,
    candidate_end,
    HEAD_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    KEY_FP8: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
):
    # A lane copies 16 bytes of a key row; fp8_scaled pages put scales after the keys.
    vector: gl.constexpr = 16 if KEY_FP8 else 8
    blocked: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[vector, 1],
        threads_per_warp=[HEAD_DIM // vector, 64 // (HEAD_DIM // vector)],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )
    dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(1, blocked))[:, None]
    columns = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked))
    valid = (tile_start + columns < candidate_end)[None, :]
    # Copies past the chunk are masked; clamp their slot reads into the slot buffer.
    slot_tile = gl.minimum(tile_id, CHUNK_N // BLOCK_N - 1)
    slots = slot_shared.gather(slot_tile * BLOCK_N + columns, axis=0)[None, :]
    if not USE_BUFFER_LOAD:
        slots = slots.to(gl.int64)
    if KEY_FP8:
        pages = slots // PAGE_SIZE
        rows = pages * PAGE_STRIDE_BYTES + (slots - pages * PAGE_SIZE) * HEAD_DIM
    else:
        rows = slots * HEAD_DIM
    if USE_BUFFER_LOAD:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            key_shared.index(buffer_id),
            index_k,
            rows + dims,
            mask=valid,
        )
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(
            key_shared.index(buffer_id),
            index_k + (rows + dims),
            mask=valid,
        )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _score_rows_key_tile(
    query,
    head_weights,
    key,
    mfma_layout: gl.constexpr,
    ROWS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    Q_FP8: gl.constexpr,
):
    # A 32-row MFMA block per query row: scores bitwise independent of ROWS and warps.
    accumulator = gl.zeros([ROWS * 32, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
    if Q_FP8:
        head_scores = gl.amd.cdna4.mfma_scaled(
            a=query,
            a_scale=None,
            a_format="e4m3",
            b=key,
            b_scale=None,
            b_format="e4m3",
            acc=accumulator,
        )
    else:
        head_scores = gl.amd.cdna4.mfma(query, key, accumulator)
    head_scores = _relu_f32(head_scores)
    contributions = head_scores * head_weights[:, None]
    return gl.sum(gl.reshape(contributions, [ROWS, 32, BLOCK_N]), axis=1)


@gluon.jit
def _score_candidate_chunk(
    index_k,
    index_k_scale,
    kv_workspace_slots,
    block_table,
    logits,
    query_0,
    weight_0,
    query_1,
    weight_1,
    starts,
    ends,
    row_offsets,
    request_id,
    block_table_stride,
    model_scale,
    candidate_start,
    candidate_end,
    mfma_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    out_layout: gl.constexpr,
    col_layout: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    ROWS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    IS_DECODE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    KEY_FP8: gl.constexpr,
    Q_FP8: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
    NONFINITE_TO_NEG_INF: gl.constexpr = False,
):
    # Slots stay in LDS so slot reads never queue behind key copies on the shared vmcnt.
    slot_positions = candidate_start + gl.arange(
        0, CHUNK_N, layout=gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    )
    slot_valid = slot_positions < candidate_end
    if IS_DECODE:
        pages = gl.amd.cdna4.buffer_load(
            ptr=block_table,
            offsets=request_id * block_table_stride + slot_positions // PAGE_SIZE,
            mask=slot_valid,
            other=0,
        )
        slot_values = pages * PAGE_SIZE + slot_positions % PAGE_SIZE
    else:
        slot_values = gl.amd.cdna4.buffer_load(
            ptr=kv_workspace_slots,
            offsets=slot_positions,
            mask=slot_valid,
            other=0,
        ).to(gl.int32)
    slot_shared = gl.allocate_shared_memory(
        gl.int32,
        [CHUNK_N],
        layout=gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0]),
        value=slot_values,
    )
    if KEY_FP8:
        # A page's scales follow its PAGE_SIZE FP8 rows; the chunk's scales stay in LDS.
        scale_slots = slot_values
        if not USE_BUFFER_LOAD:
            scale_slots = scale_slots.to(gl.int64)
        scale_pages = scale_slots // PAGE_SIZE
        scale_offsets = (
            scale_pages * (PAGE_STRIDE_BYTES // 4)
            + (PAGE_SIZE * HEAD_DIM) // 4
            + (scale_slots - scale_pages * PAGE_SIZE)
        )
        if USE_BUFFER_LOAD:
            scale_values = gl.amd.cdna4.buffer_load(
                ptr=index_k_scale, offsets=scale_offsets, mask=slot_valid, other=0.0
            )
        else:
            scale_values = gl.load(
                index_k_scale + scale_offsets, mask=slot_valid, other=0.0
            )
        scale_shared = gl.allocate_shared_memory(
            gl.float32,
            [CHUNK_N],
            layout=gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0]),
            value=scale_values,
        )
        # XOR-swizzled 16-byte chunks spread 16 consecutive keys over all LDS banks.
        key_layout: gl.constexpr = gl.SwizzledSharedLayout(
            vec=16, per_phase=1, max_phase=HEAD_DIM // 16, order=[0, 1]
        )
    else:
        key_layout: gl.constexpr = gl.SwizzledSharedLayout(
            vec=8, per_phase=1, max_phase=HEAD_DIM // 8, order=[0, 1]
        )
    key_shared = gl.allocate_shared_memory(
        index_k.type.element_ty,
        [NUM_STAGES, HEAD_DIM, BLOCK_N],
        layout=key_layout,
    )
    tile_count = tl.cdiv(candidate_end - candidate_start, BLOCK_N)
    for stage in gl.static_range(NUM_STAGES - 1):
        _copy_key_tile(
            index_k,
            key_shared,
            stage,
            slot_shared,
            stage,
            candidate_start + stage * BLOCK_N,
            candidate_end,
            HEAD_DIM,
            BLOCK_N,
            CHUNK_N,
            NUM_WARPS,
            PAGE_SIZE,
            PAGE_STRIDE_BYTES,
            KEY_FP8,
            USE_BUFFER_LOAD,
        )

    columns = gl.arange(0, BLOCK_N, layout=col_layout)
    for tile_id in tl.range(0, tile_count):
        tile_start = candidate_start + tile_id * BLOCK_N
        # One copy group per iteration, masked or not, so NUM_STAGES - 2 stay in flight.
        gl.amd.cdna4.async_copy.wait_group(NUM_STAGES - 2)
        key = key_shared.index(tile_id % NUM_STAGES).load(layout=dot_b_layout)
        if KEY_FP8 and not Q_FP8:
            key = key.to(gl.bfloat16)
        _copy_key_tile(
            index_k,
            key_shared,
            (tile_id + NUM_STAGES - 1) % NUM_STAGES,
            slot_shared,
            tile_id + NUM_STAGES - 1,
            tile_start + (NUM_STAGES - 1) * BLOCK_N,
            candidate_end,
            HEAD_DIM,
            BLOCK_N,
            CHUNK_N,
            NUM_WARPS,
            PAGE_SIZE,
            PAGE_STRIDE_BYTES,
            KEY_FP8,
            USE_BUFFER_LOAD,
        )
        scores = _score_rows_key_tile(
            query_0, weight_0, key, mfma_layout, ROWS, BLOCK_N, Q_FP8
        )
        if NUM_HEADS == 64:
            scores += _score_rows_key_tile(
                query_1, weight_1, key, mfma_layout, ROWS, BLOCK_N, Q_FP8
            )
        scores = gl.convert_layout(scores, out_layout) * model_scale
        if KEY_FP8:
            key_scales = scale_shared.gather(tile_id * BLOCK_N + columns, axis=0)
            scores = scores * key_scales[None, :]
        if NONFINITE_TO_NEG_INF:
            finite = (scores == scores) & (gl.abs(scores) != float("inf"))
            scores = gl.where(finite, scores, -float("inf"))
        positions = tile_start + columns
        # Columns past the chunk belong to the next program.
        valid = (
            (positions[None, :] >= starts[:, None])
            & (positions[None, :] < ends[:, None])
            & (positions < candidate_end)[None, :]
        )
        output_offsets = row_offsets[:, None] + positions[None, :].to(gl.int64)
        if USE_BUFFER_STORE:
            gl.amd.cdna4.buffer_store(
                scores,
                ptr=logits,
                offsets=output_offsets.to(gl.int32),
                mask=valid,
            )
        else:
            gl.store(logits + output_offsets, scores, mask=valid)
    # Drain the masked copies issued past the span.
    gl.amd.cdna4.async_copy.wait_group(0)


@gluon.jit
def _multi_row_index_k_scores_body(
    q,
    q_scales,
    index_k,
    index_k_scale,
    weights,
    kv_workspace_slots,
    row_starts,
    row_ends,
    seq_lens,
    block_table,
    logits,
    num_rows,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_w_row,
    stride_w_head,
    stride_qs_row,
    stride_qs_head,
    block_table_stride,
    logits_stride,
    model_scale,
    num_candidates,
    q_len_per_req,
    PAGE_SIZE: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    ROWS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    IS_DECODE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    KEY_FP8: gl.constexpr,
    Q_FP8: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
    CHUNK_LOOP: gl.constexpr,
    EXPLICIT_ROW_LENS: gl.constexpr = False,
    NONFINITE_TO_NEG_INF: gl.constexpr = False,
    ROW_RUNS: gl.constexpr = False,
):
    # Program (i, j): rows i on chunk j; positive key scales apply after the head sum.
    # NONFINITE_TO_NEG_INF stores NaN and infinite scores as -inf.
    # ROW_RUNS (prefill rows): row_starts holds each row's request and
    # block_table each row's compact pages, which start at column 0 and are a
    # prefix of a later row's of its request. Each request's rows in the block
    # score their last row's pages.
    NUM_WARPS: gl.constexpr = WARPS_M * WARPS_N
    row_block = gl.program_id(0)
    split = gl.program_id(1)

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64] if Q_FP8 else [32, 32, 16],
        transposed=False,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    k_width: gl.constexpr = 16 if Q_FP8 else 8
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=k_width
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=k_width
    )
    out_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[64 // (BLOCK_N // 4), BLOCK_N // 4],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, out_layout)
    col_layout: gl.constexpr = gl.SliceLayout(0, out_layout)
    m_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)

    local_rows = gl.arange(0, ROWS, layout=row_layout)
    m_local = gl.arange(0, ROWS * 32, layout=m_layout) // 32
    if IS_DECODE:
        request_id = row_block
        rows = request_id * q_len_per_req + local_rows
        row_valid = local_rows < q_len_per_req
        m_rows = request_id * q_len_per_req + m_local
        m_valid = m_local < q_len_per_req
        if EXPLICIT_ROW_LENS:
            # Compact owned contexts do not grow on a rank when the newly
            # appended token belongs to another rank. Preserve every row's
            # actual local end rather than subtracting global draft offsets.
            ends = gl.amd.cdna4.buffer_load(
                ptr=row_ends, offsets=rows, mask=row_valid, other=0
            ).to(gl.int32)
            ends = gl.minimum(gl.maximum(ends, 0), num_candidates)
        else:
            # Row j sees seq_len - (q_len - 1) + j keys in a full context.
            seq_len = gl.load(seq_lens + request_id).to(gl.int32)
            ends = gl.minimum(
                seq_len - (q_len_per_req - 1) + local_rows, num_candidates
            )
        starts = gl.zeros([ROWS], gl.int32, layout=row_layout)
    else:
        request_id = 0
        rows = row_block * ROWS + local_rows
        row_valid = rows < num_rows
        m_rows = row_block * ROWS + m_local
        m_valid = m_rows < num_rows
        if ROW_RUNS:
            row_requests = gl.amd.cdna4.buffer_load(
                ptr=row_starts, offsets=rows, mask=row_valid, other=0
            )
            starts = gl.zeros([ROWS], gl.int32, layout=row_layout)
        else:
            starts = gl.maximum(
                gl.amd.cdna4.buffer_load(
                    ptr=row_starts, offsets=rows, mask=row_valid, other=0
                ),
                0,
            )
        ends = gl.minimum(
            gl.amd.cdna4.buffer_load(
                ptr=row_ends, offsets=rows, mask=row_valid, other=0
            ),
            num_candidates,
        )
    ends = gl.where(row_valid, ends, 0)
    starts = gl.where(row_valid, starts, num_candidates)
    span_end = gl.max(ends, axis=0)
    if CHUNK_LOOP:
        # Program j scans chunk run j, so the grid tracks the GPU, not max-model-len.
        run = tl.cdiv(tl.cdiv(span_end, CHUNK_N), gl.num_programs(1)) * CHUNK_N
    else:
        run = CHUNK_N
    # Scan the union of the rows' spans inside this program's chunks.
    candidate_start = gl.maximum(gl.min(starts, axis=0), split * run)
    candidate_end = gl.minimum(span_end, (split + 1) * run)
    candidate_start = gl.minimum(candidate_start, candidate_end)
    # Chunks past every span store nothing; skip the query and slot loads.
    if candidate_start == candidate_end:
        return

    query_0, weight_0 = _load_query_group(
        q,
        weights,
        q_scales,
        m_rows,
        m_valid,
        0,
        stride_q_row,
        stride_q_head,
        stride_q_dim,
        stride_w_row,
        stride_w_head,
        stride_qs_row,
        stride_qs_head,
        mfma_layout,
        dot_a_layout,
        ROWS,
        HEAD_DIM,
        NUM_WARPS,
        Q_FP8,
    )
    if NUM_HEADS == 64:
        query_1, weight_1 = _load_query_group(
            q,
            weights,
            q_scales,
            m_rows,
            m_valid,
            32,
            stride_q_row,
            stride_q_head,
            stride_q_dim,
            stride_w_row,
            stride_w_head,
            stride_qs_row,
            stride_qs_head,
            mfma_layout,
            dot_a_layout,
            ROWS,
            HEAD_DIM,
            NUM_WARPS,
            Q_FP8,
        )
    else:
        query_1 = query_0
        weight_1 = weight_0

    row_offsets = rows.to(gl.int64) * logits_stride
    if ROW_RUNS:
        tl.static_assert(not CHUNK_LOOP and not IS_DECODE)
        no_request = 2147483647
        previous = gl.min(gl.where(row_valid, row_requests, no_request), axis=0) - 1
        # One request per pass, in ascending order; a block holds at most ROWS.
        for _ in range(0, ROWS):
            request = gl.min(
                gl.where(
                    row_valid & (row_requests > previous), row_requests, no_request
                ),
                axis=0,
            )
            in_run = row_valid & (row_requests == request)
            run_starts = gl.where(in_run, starts, num_candidates)
            run_ends = gl.where(in_run, ends, 0)
            run_start = gl.maximum(gl.min(run_starts, axis=0), candidate_start)
            run_end = gl.minimum(gl.max(run_ends, axis=0), candidate_end)
            if run_start < run_end:
                _score_candidate_chunk(
                    index_k,
                    index_k_scale,
                    kv_workspace_slots,
                    block_table,
                    logits,
                    query_0,
                    weight_0,
                    query_1,
                    weight_1,
                    run_starts,
                    run_ends,
                    row_offsets,
                    gl.max(gl.where(in_run, rows, 0), axis=0),
                    block_table_stride,
                    model_scale,
                    run_start,
                    run_end,
                    mfma_layout,
                    dot_b_layout,
                    out_layout,
                    col_layout,
                    PAGE_SIZE,
                    NUM_HEADS,
                    HEAD_DIM,
                    ROWS,
                    BLOCK_N,
                    CHUNK_N,
                    NUM_WARPS,
                    NUM_STAGES,
                    True,
                    PAGE_STRIDE_BYTES,
                    KEY_FP8,
                    Q_FP8,
                    USE_BUFFER_LOAD,
                    USE_BUFFER_STORE,
                    NONFINITE_TO_NEG_INF,
                )
            previous = request
    elif CHUNK_LOOP:
        for chunk in range(candidate_start // CHUNK_N, tl.cdiv(candidate_end, CHUNK_N)):
            _score_candidate_chunk(
                index_k,
                index_k_scale,
                kv_workspace_slots,
                block_table,
                logits,
                query_0,
                weight_0,
                query_1,
                weight_1,
                starts,
                ends,
                row_offsets,
                request_id,
                block_table_stride,
                model_scale,
                gl.maximum(candidate_start, chunk * CHUNK_N),
                gl.minimum(candidate_end, (chunk + 1) * CHUNK_N),
                mfma_layout,
                dot_b_layout,
                out_layout,
                col_layout,
                PAGE_SIZE,
                NUM_HEADS,
                HEAD_DIM,
                ROWS,
                BLOCK_N,
                CHUNK_N,
                NUM_WARPS,
                NUM_STAGES,
                IS_DECODE,
                PAGE_STRIDE_BYTES,
                KEY_FP8,
                Q_FP8,
                USE_BUFFER_LOAD,
                USE_BUFFER_STORE,
                NONFINITE_TO_NEG_INF,
            )
    else:
        _score_candidate_chunk(
            index_k,
            index_k_scale,
            kv_workspace_slots,
            block_table,
            logits,
            query_0,
            weight_0,
            query_1,
            weight_1,
            starts,
            ends,
            row_offsets,
            request_id,
            block_table_stride,
            model_scale,
            candidate_start,
            candidate_end,
            mfma_layout,
            dot_b_layout,
            out_layout,
            col_layout,
            PAGE_SIZE,
            NUM_HEADS,
            HEAD_DIM,
            ROWS,
            BLOCK_N,
            CHUNK_N,
            NUM_WARPS,
            NUM_STAGES,
            IS_DECODE,
            PAGE_STRIDE_BYTES,
            KEY_FP8,
            Q_FP8,
            USE_BUFFER_LOAD,
            USE_BUFFER_STORE,
            NONFINITE_TO_NEG_INF,
        )


@gluon.jit(
    launch_metadata=_bf16_index_k_launch_metadata,
    do_not_specialize=("num_rows", "num_candidates"),
)
def gluon_dsa_prefill_topk_bf16_gfx950(
    q,
    index_k,
    weights,
    kv_workspace_slots,
    row_starts,
    row_ends,
    logits,
    num_rows,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_w_row,
    stride_w_head,
    logits_stride,
    model_scale,
    num_candidates,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    ROWS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    """Prefill scorer over a bf16 ``[slots, HEAD_DIM]`` plane and BF16 q.

    Launch on ``(cdiv(num_rows, ROWS), cdiv(num_candidates, CHUNK_N))``.
    Program ``(i, j)`` scores query rows ``[i * ROWS, (i + 1) * ROWS)``
    against workspace rows ``[j * CHUNK_N, (j + 1) * CHUNK_N)``; row ``t``
    writes ``logits[t, s]`` for ``row_starts[t] <= s < row_ends[t]`` only.
    Workspace rows resolve to plane slots through ``kv_workspace_slots``.
    """
    _multi_row_index_k_scores_body(
        q,
        q,
        index_k,
        index_k,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        row_ends,
        row_starts,
        logits,
        num_rows,
        stride_q_row,
        stride_q_head,
        stride_q_dim,
        stride_w_row,
        stride_w_head,
        0,
        0,
        0,
        logits_stride,
        model_scale,
        num_candidates,
        1,
        1,
        NUM_HEADS,
        HEAD_DIM,
        ROWS,
        BLOCK_N,
        CHUNK_N,
        WARPS_M,
        WARPS_N,
        NUM_STAGES,
        False,
        0,
        False,
        False,
        USE_BUFFER_LOAD,
        USE_BUFFER_STORE,
        False,
    )


@gluon.jit(
    launch_metadata=_bf16_index_k_launch_metadata,
    do_not_specialize=("num_rows", "num_candidates", "q_len_per_req"),
)
def gluon_dsa_decode_topk_bf16_gfx950(
    q,
    index_k,
    weights,
    seq_lens,
    block_table,
    logits,
    num_rows,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_w_row,
    stride_w_head,
    block_table_stride,
    logits_stride,
    model_scale,
    num_candidates,
    q_len_per_req,
    PAGE_SIZE: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    ROWS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    """Decode scorer over a bf16 ``[slots, HEAD_DIM]`` plane and BF16 q.

    Launch on ``(requests, cdiv(num_candidates, CHUNK_N))`` with
    ``ROWS >= q_len_per_req``. Program ``(i, j)`` scores request ``i``'s
    ``q_len_per_req`` query rows against positions
    ``[j * CHUNK_N, (j + 1) * CHUNK_N)``; ``block_table`` maps each
    request's ``PAGE_SIZE``-row pages to slots of the plane. Row ``r`` of a
    request writes its first ``seq_len - (q_len_per_req - 1) + r`` positions.
    """
    _multi_row_index_k_scores_body(
        q,
        q,
        index_k,
        index_k,
        weights,
        block_table,
        seq_lens,
        seq_lens,
        seq_lens,
        block_table,
        logits,
        num_rows,
        stride_q_row,
        stride_q_head,
        stride_q_dim,
        stride_w_row,
        stride_w_head,
        0,
        0,
        block_table_stride,
        logits_stride,
        model_scale,
        num_candidates,
        q_len_per_req,
        PAGE_SIZE,
        NUM_HEADS,
        HEAD_DIM,
        ROWS,
        BLOCK_N,
        CHUNK_N,
        WARPS_M,
        WARPS_N,
        NUM_STAGES,
        True,
        0,
        False,
        False,
        USE_BUFFER_LOAD,
        USE_BUFFER_STORE,
        False,
    )


def _fp8_index_k_launch_metadata(grid, kernel, args):
    rows = args["num_rows"]
    heads = args["NUM_HEADS"]
    head_dim = args["HEAD_DIM"]
    cols = args["num_candidates"]
    q_bytes = 1 if args["Q_FP8"] else 2
    # Upper bound: every row scores every candidate.
    return {
        "name": kernel.name,
        "flops8" if args["Q_FP8"] else "flops16": 2 * rows * heads * head_dim * cols,
        "bytes": grid[0] * cols * (head_dim + 4)
        + grid[1] * rows * heads * head_dim * q_bytes
        + rows * cols * 4,
    }


@gluon.jit(
    launch_metadata=_fp8_index_k_launch_metadata,
    do_not_specialize=("num_rows", "num_candidates"),
)
def gluon_dsa_prefill_topk_fp8_rows_gfx950(
    q,
    q_scales,
    index_k_fp8,
    index_k_scale,
    weights,
    kv_workspace_slots,
    row_starts,
    row_ends,
    logits,
    num_rows,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_w_row,
    stride_w_head,
    stride_qs_row,
    stride_qs_head,
    logits_stride,
    model_scale,
    num_candidates,
    PAGE_SIZE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    ROWS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    Q_FP8: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    """Prefill scorer over an fp8_scaled paged plane and BF16 or FP8 q.

    The grid and spans are ``gluon_dsa_prefill_topk_bf16_gfx950``'s;
    ``kv_workspace_slots`` holds plane slots of ``PAGE_SIZE``-row pages
    ``PAGE_STRIDE_BYTES`` apart, each its FP8 rows then their fp32 scales.

    With ``Q_FP8``, ``q`` is E4M3 with FP32 ``q_scales[row, head]`` and the
    keys multiply as E4M3; otherwise ``q`` is BF16 and ``q_scales`` unused.
    """
    _multi_row_index_k_scores_body(
        q,
        q_scales,
        index_k_fp8,
        index_k_scale,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        row_ends,
        row_starts,
        logits,
        num_rows,
        stride_q_row,
        stride_q_head,
        stride_q_dim,
        stride_w_row,
        stride_w_head,
        stride_qs_row,
        stride_qs_head,
        0,
        logits_stride,
        model_scale,
        num_candidates,
        1,
        PAGE_SIZE,
        NUM_HEADS,
        HEAD_DIM,
        ROWS,
        BLOCK_N,
        CHUNK_N,
        WARPS_M,
        WARPS_N,
        NUM_STAGES,
        False,
        PAGE_STRIDE_BYTES,
        True,
        Q_FP8,
        USE_BUFFER_LOAD,
        USE_BUFFER_STORE,
        False,
    )


@gluon.jit(
    launch_metadata=_fp8_index_k_launch_metadata,
    do_not_specialize=(
        "num_rows",
        "num_candidates",
        "block_table_stride",
        "logits_stride",
    ),
    do_not_specialize_on_alignment=("row_requests", "row_lens", "block_table"),
)
def gluon_dsa_owned_prefill_topk_fp8_rows_gfx950(
    q,
    q_scales,
    index_k_fp8,
    index_k_scale,
    weights,
    row_requests,
    row_lens,
    block_table,
    logits,
    num_rows,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_w_row,
    stride_w_head,
    stride_qs_row,
    stride_qs_head,
    block_table_stride,
    logits_stride,
    model_scale,
    num_candidates,
    PAGE_SIZE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    ROWS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    Q_FP8: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    """Prefill scorer over compact owned pages of an fp8_scaled paged plane.

    The grid and spans are ``gluon_dsa_prefill_topk_fp8_rows_gfx950``'s;
    row ``i`` (of request ``row_requests[i]``) scores its first
    ``row_lens[i]`` slots of ``block_table[i]``'s pages, as
    ``gluon_dsa_decode_topk_fp8_rows_gfx950`` scores a one-row group, with
    the same arithmetic. NaN and infinite scores are stored as -inf.
    """
    _multi_row_index_k_scores_body(
        q,
        q_scales,
        index_k_fp8,
        index_k_scale,
        weights,
        block_table,
        row_requests,
        row_lens,
        row_lens,
        block_table,
        logits,
        num_rows,
        stride_q_row,
        stride_q_head,
        stride_q_dim,
        stride_w_row,
        stride_w_head,
        stride_qs_row,
        stride_qs_head,
        block_table_stride,
        logits_stride,
        model_scale,
        num_candidates,
        1,
        PAGE_SIZE,
        NUM_HEADS,
        HEAD_DIM,
        ROWS,
        BLOCK_N,
        CHUNK_N,
        WARPS_M,
        WARPS_N,
        NUM_STAGES,
        False,
        PAGE_STRIDE_BYTES,
        True,
        Q_FP8,
        USE_BUFFER_LOAD,
        USE_BUFFER_STORE,
        False,
        False,
        True,
        True,
    )


@gluon.jit(
    launch_metadata=_fp8_index_k_launch_metadata,
    do_not_specialize=(
        "num_rows",
        "num_candidates",
        "q_len_per_req",
        "block_table_stride",
        "logits_stride",
    ),
    do_not_specialize_on_alignment=("seq_lens", "block_table", "explicit_row_lens"),
)
def gluon_dsa_decode_topk_fp8_rows_gfx950(
    q,
    q_scales,
    index_k_fp8,
    index_k_scale,
    weights,
    seq_lens,
    block_table,
    logits,
    num_rows,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_w_row,
    stride_w_head,
    stride_qs_row,
    stride_qs_head,
    block_table_stride,
    logits_stride,
    model_scale,
    num_candidates,
    q_len_per_req,
    PAGE_SIZE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    ROWS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    Q_FP8: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
    CHUNK_LOOP: gl.constexpr,
    explicit_row_lens=None,
    NONFINITE_TO_NEG_INF: gl.constexpr = False,
):
    """Decode scorer over an fp8_scaled paged plane and BF16 or FP8 q.

    The grid and spans are ``gluon_dsa_decode_topk_bf16_gfx950``'s; pages
    are ``PAGE_STRIDE_BYTES`` apart, each its FP8 rows then their fp32
    scales. With ``CHUNK_LOOP`` the grid may have fewer programs per request
    than chunks: program ``(i, j)`` then scores the ``j``-th equal run of the
    chunks request ``i``'s rows fill.

    With ``Q_FP8``, ``q`` is E4M3 with FP32 ``q_scales[row, head]`` and the
    keys multiply as E4M3; otherwise ``q`` is BF16 and ``q_scales`` unused.
    With ``NONFINITE_TO_NEG_INF``, NaN and infinite scores are stored as
    -inf.
    """
    _multi_row_index_k_scores_body(
        q,
        q_scales,
        index_k_fp8,
        index_k_scale,
        weights,
        block_table,
        seq_lens,
        seq_lens if explicit_row_lens is None else explicit_row_lens,
        seq_lens,
        block_table,
        logits,
        num_rows,
        stride_q_row,
        stride_q_head,
        stride_q_dim,
        stride_w_row,
        stride_w_head,
        stride_qs_row,
        stride_qs_head,
        block_table_stride,
        logits_stride,
        model_scale,
        num_candidates,
        q_len_per_req,
        PAGE_SIZE,
        NUM_HEADS,
        HEAD_DIM,
        ROWS,
        BLOCK_N,
        CHUNK_N,
        WARPS_M,
        WARPS_N,
        NUM_STAGES,
        True,
        PAGE_STRIDE_BYTES,
        True,
        Q_FP8,
        USE_BUFFER_LOAD,
        USE_BUFFER_STORE,
        CHUNK_LOOP,
        explicit_row_lens is not None,
        NONFINITE_TO_NEG_INF,
    )
