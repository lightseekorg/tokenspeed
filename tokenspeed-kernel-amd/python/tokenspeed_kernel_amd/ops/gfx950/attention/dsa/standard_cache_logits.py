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
"""

from tokenspeed_kernel_amd._triton import gl, gluon, tl

__all__ = [
    "_dsa_standard_decode_logits_kernel",
    "_dsa_standard_prefill_logits_kernel",
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
    ROW_BYTES: gl.constexpr,
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
    ROW_BYTES: gl.constexpr,
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
    if Q_IS_FP8:
        key = raw_key
    else:
        key = raw_key.to(gl.bfloat16)
    key = gl.convert_layout(key, dot_b_layout)
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
    ROW_BYTES: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    Q_IS_FP8: gl.constexpr,
    IS_PREFILL: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    row_id = gl.program_id(0)
    split_id = gl.program_id(1)
    if IS_PREFILL:
        request_id = row_id
        candidate_start = gl.maximum(gl.load(row_starts + row_id), 0)
        candidate_end = gl.minimum(gl.load(row_ends + row_id), max_candidates)
    else:
        request_id = row_id // q_len_per_req
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
        ROW_BYTES,
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
        ROW_BYTES,
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
            ROW_BYTES,
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
                ROW_BYTES,
                PAGE_STRIDE_BYTES,
                HEAD_DIM,
                BLOCK_N,
                NUM_WARPS,
                IS_PREFILL,
                USE_BUFFER_LOAD,
            )
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
        output_offsets = (
            row_id.to(gl.int64) * logits_stride
            + tile_start.to(gl.int64)
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
def _dsa_standard_prefill_logits_kernel(
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
    ROW_BYTES: gl.constexpr,
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
        PAGE_SIZE,
        ROW_BYTES,
        PAGE_STRIDE_BYTES,
        NUM_HEADS,
        HEAD_DIM,
        BLOCK_N,
        BLOCK_N,
        NUM_WARPS,
        Q_IS_FP8,
        True,
        USE_BUFFER_LOAD,
        USE_BUFFER_STORE,
    )


@gluon.jit
def _dsa_standard_decode_logits_kernel(
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
    ROW_BYTES: gl.constexpr,
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
        PAGE_SIZE,
        ROW_BYTES,
        PAGE_STRIDE_BYTES,
        NUM_HEADS,
        HEAD_DIM,
        BLOCK_N,
        CHUNK_N,
        NUM_WARPS,
        Q_IS_FP8,
        False,
        USE_BUFFER_LOAD,
        USE_BUFFER_STORE,
    )
