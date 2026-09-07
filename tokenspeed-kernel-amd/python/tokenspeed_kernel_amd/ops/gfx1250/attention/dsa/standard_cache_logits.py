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

"""GFX1250 matrix-core DSA logits kernels for the standard paged cache.

The scorer supports 32 or 64 index heads, BF16 or scaled FP8 queries, and both
packed-slot and page-planar FP8 key-cache storage.
"""

from tokenspeed_kernel_amd._triton import gl, gluon, tl

__all__ = [
    "_dsa_standard_decode_logits_kernel",
    "_dsa_standard_prefill_logits_kernel",
]


@gluon.constexpr_function
def _make_wmma_layout(
    NUM_WARPS: gl.constexpr,
    Q_IS_FP8: gl.constexpr,
):
    """Map Wave32 workgroups over 32 scoring heads and candidate columns."""
    if NUM_WARPS == 2:
        warp_bases = [[1, 0]]
    elif NUM_WARPS == 4:
        warp_bases = [[1, 0], [0, 1]]
    elif NUM_WARPS == 8:
        warp_bases = [[1, 0], [0, 1], [0, 2]]
    else:
        warp_bases = []
    return gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases,
        reg_bases=[],
        instr_shape=[16, 16, 64 if Q_IS_FP8 else 32],
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
        return gl.amd.cdna5.buffer_load(
            kv_workspace_slots,
            positions.to(gl.int32),
            mask=valid,
            other=0,
        ).to(gl.int64)
    page_indices = positions // PAGE_SIZE
    pages = gl.amd.cdna5.buffer_load(
        block_table,
        (request_id * block_table_stride + page_indices).to(gl.int32),
        mask=valid,
        other=0,
    ).to(gl.int64)
    return pages * PAGE_SIZE + positions % PAGE_SIZE


@gluon.jit
def _load_query(
    q,
    q_scales,
    weights,
    row_id,
    head_offset: gl.constexpr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_qs_row,
    stride_qs_head,
    stride_w_row,
    stride_w_head,
    q_load_layout: gl.constexpr,
    q_dot_layout: gl.constexpr,
    wmma_layout: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    Q_IS_FP8: gl.constexpr,
):
    heads = gl.arange(0, 32, layout=gl.SliceLayout(1, q_load_layout))[:, None]
    dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, q_load_layout))[None, :]
    query = gl.amd.cdna5.buffer_load(
        q,
        (
            row_id * stride_q_row
            + (head_offset + heads) * stride_q_head
            + dims * stride_q_dim
        ).to(gl.int32),
    )
    weight_heads = gl.arange(0, 32, layout=gl.SliceLayout(1, wmma_layout))
    head_weights = gl.amd.cdna5.buffer_load(
        weights,
        (row_id * stride_w_row + (head_offset + weight_heads) * stride_w_head).to(
            gl.int32
        ),
    ).to(gl.float32)
    if Q_IS_FP8:
        query_scales = gl.amd.cdna5.buffer_load(
            q_scales,
            (row_id * stride_qs_row + (head_offset + weight_heads) * stride_qs_head).to(
                gl.int32
            ),
        ).to(gl.float32)
        head_weights *= query_scales
    return gl.convert_layout(query, q_dot_layout), head_weights


@gluon.jit
def _score_head_tile(
    query,
    key,
    head_weights,
    wmma_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    accumulator = gl.zeros([32, BLOCK_N], gl.float32, layout=wmma_layout)
    head_scores = gl.amd.cdna5.wmma(query, key, accumulator)
    head_scores = gl.maximum(
        head_scores,
        0.0,
        propagate_nan=tl.PropagateNan.ALL,
    )
    return gl.sum(head_scores * head_weights[:, None], axis=0)


@gluon.jit
def _score_key_tile(
    query_0,
    weight_0,
    query_1,
    weight_1,
    index_k_fp8,
    index_k_scale,
    kv_workspace_slots,
    block_table,
    request_id,
    block_table_stride,
    candidate_start,
    candidate_end,
    model_scale,
    wmma_layout: gl.constexpr,
    k_dot_layout: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    Q_IS_FP8: gl.constexpr,
    IS_PREFILL: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
):
    dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(1, k_dot_layout))[:, None]
    columns = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, k_dot_layout))[None, :]
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
    key_offsets = pages * PAGE_STRIDE_BYTES + page_rows * HEAD_DIM + dims
    if USE_BUFFER_LOAD:
        raw_key = gl.amd.cdna5.buffer_load(
            index_k_fp8,
            key_offsets.to(gl.int32),
            mask=valid,
            other=0.0,
        )
    else:
        raw_key = gl.load(
            index_k_fp8 + key_offsets,
            mask=valid,
            other=0.0,
        )
    key = raw_key if Q_IS_FP8 else raw_key.to(gl.bfloat16)
    scores = _score_head_tile(
        query_0,
        key,
        weight_0,
        wmma_layout,
        BLOCK_N,
    )
    if NUM_HEADS == 64:
        scores += _score_head_tile(
            query_1,
            key,
            weight_1,
            wmma_layout,
            BLOCK_N,
        )

    output_layout: gl.constexpr = gl.SliceLayout(0, wmma_layout)
    scale_positions = candidate_start + gl.arange(0, BLOCK_N, layout=output_layout)
    scale_valid = scale_positions < candidate_end
    scale_slots = _candidate_slots(
        scale_positions,
        scale_valid,
        kv_workspace_slots,
        block_table,
        request_id,
        block_table_stride,
        PAGE_SIZE,
        IS_PREFILL,
    )
    scale_pages = scale_slots // PAGE_SIZE
    scale_rows = scale_slots - scale_pages * PAGE_SIZE
    scale_offsets = (
        scale_pages * (PAGE_STRIDE_BYTES // 4)
        + (PAGE_SIZE * HEAD_DIM) // 4
        + scale_rows
    )
    if USE_BUFFER_LOAD:
        key_scales = gl.amd.cdna5.buffer_load(
            index_k_scale,
            scale_offsets.to(gl.int32),
            mask=scale_valid,
            other=0.0,
        ).to(gl.float32)
    else:
        key_scales = gl.load(
            index_k_scale + scale_offsets,
            mask=scale_valid,
            other=0.0,
        ).to(gl.float32)
    return scores * key_scales * model_scale, scale_positions, scale_valid


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

    wmma_layout: gl.constexpr = _make_wmma_layout(NUM_WARPS, Q_IS_FP8)
    k_width: gl.constexpr = 16 if Q_IS_FP8 else 8
    q_dot_layout: gl.constexpr = gl.DotOperandLayout(
        0,
        wmma_layout,
        k_width=k_width,
    )
    k_dot_layout: gl.constexpr = gl.DotOperandLayout(
        1,
        wmma_layout,
        k_width=k_width,
    )
    q_load_layout: gl.constexpr = gl.BlockedLayout(
        [1, k_width],
        [4, 8],
        [NUM_WARPS, 1],
        [1, 0],
    )
    query_0, weight_0 = _load_query(
        q,
        q_scales,
        weights,
        row_id,
        0,
        stride_q_row,
        stride_q_head,
        stride_q_dim,
        stride_qs_row,
        stride_qs_head,
        stride_w_row,
        stride_w_head,
        q_load_layout,
        q_dot_layout,
        wmma_layout,
        HEAD_DIM,
        Q_IS_FP8,
    )
    if NUM_HEADS == 64:
        query_1, weight_1 = _load_query(
            q,
            q_scales,
            weights,
            row_id,
            32,
            stride_q_row,
            stride_q_head,
            stride_q_dim,
            stride_qs_row,
            stride_qs_head,
            stride_w_row,
            stride_w_head,
            q_load_layout,
            q_dot_layout,
            wmma_layout,
            HEAD_DIM,
            Q_IS_FP8,
        )
    else:
        query_1 = query_0
        weight_1 = weight_0

    tile_count = tl.cdiv(candidate_end - candidate_start, BLOCK_N)
    for tile_id in tl.range(0, tile_count):
        tile_start = candidate_start + tile_id * BLOCK_N
        scores, positions, valid = _score_key_tile(
            query_0,
            weight_0,
            query_1,
            weight_1,
            index_k_fp8,
            index_k_scale,
            kv_workspace_slots,
            block_table,
            request_id,
            block_table_stride,
            tile_start,
            candidate_end,
            model_scale,
            wmma_layout,
            k_dot_layout,
            PAGE_SIZE,
            PAGE_STRIDE_BYTES,
            HEAD_DIM,
            BLOCK_N,
            NUM_HEADS,
            Q_IS_FP8,
            IS_PREFILL,
            USE_BUFFER_LOAD,
        )
        if USE_BUFFER_STORE:
            gl.amd.cdna5.buffer_store(
                scores,
                logits,
                (row_id * logits_stride + positions).to(gl.int32),
                mask=valid,
            )
        else:
            output_offsets = row_id.to(gl.int64) * logits_stride + positions.to(
                gl.int64
            )
            gl.store(logits + output_offsets, scores, mask=valid)


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
