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

"""Use the regular FP8 matrix scorer over compact owned Index-K pages."""

import torch
from tokenspeed_kernel_amd._triton import triton
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
    _BF16_DECODE_CHUNK_N,
    _BF16_DECODE_NUM_WARPS,
    _BF16_PREFILL_BLOCK_N,
    _BF16_PREFILL_CHUNK_N,
    _BF16_PREFILL_NUM_STAGES,
    _BF16_PREFILL_NUM_WARPS,
    _BF16_PREFILL_ROWS,
    _BF16_PREFILL_WAVES_PER_EU,
    _FP8_ROWS_DECODE_NUM_STAGES,
    _FP8_ROWS_PREFILL_TILES,
    _check_fp8_rows_scorer_inputs,
    _check_standard_decode_metadata,
    _fp8_rows_buffer_load,
    _fp8_rows_decode_shape,
    _fp8_rows_decode_splits,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.standard_cache_logits import (
    gluon_dsa_decode_topk_fp8_rows_gfx950,
    gluon_dsa_owned_prefill_topk_fp8_rows_gfx950,
)

# Below this many rows a launch is decode-sized; the one-row scorer keeps it.
_PREFILL_MIN_ROWS = 64


def _scores_prefill_rows(rows: int, requests: int | None, heads: int) -> bool:
    """Whether one-row groups score through the multi-row prefill scorer.

    It shares each key tile across a block of rows, so it pays off once the
    launch's requests average at least a block of rows each; decode rows,
    one per request, keep the one-row scorer.
    """
    if requests is None or rows < _PREFILL_MIN_ROWS:
        return False
    return rows >= _BF16_PREFILL_ROWS[heads] * max(requests, 1)


def _launch_prefill_rows(
    q,
    q_scales,
    weights,
    index_k_cache,
    pages,
    lengths,
    query_requests,
    logits,
    page_size,
    softmax_scale,
    page_stride,
):
    rows, width = logits.shape
    if query_requests.shape != (rows,) or query_requests.dtype != torch.int32:
        raise ValueError("owned-page prefill rows need one int32 request per row")
    if pages.nbytes >= 2**31:
        raise ValueError("owned-page prefill pages must fit int32 offsets")
    rows_per_program = _BF16_PREFILL_ROWS[q.shape[1]]
    block_n, chunk_n = _FP8_ROWS_PREFILL_TILES.get(
        q.shape[1], (_BF16_PREFILL_BLOCK_N, _BF16_PREFILL_CHUNK_N)
    )
    grid = (triton.cdiv(rows, rows_per_program), triton.cdiv(width, chunk_n))
    gluon_dsa_owned_prefill_topk_fp8_rows_gfx950[grid](
        q,
        q_scales,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        query_requests,
        lengths,
        pages,
        logits,
        rows,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        weights.stride(0),
        weights.stride(1),
        q_scales.stride(0),
        q_scales.stride(1),
        pages.stride(0),
        logits.stride(0),
        float(softmax_scale),
        width,
        PAGE_SIZE=page_size,
        PAGE_STRIDE_BYTES=page_stride,
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        ROWS=rows_per_program,
        BLOCK_N=block_n,
        CHUNK_N=chunk_n,
        WARPS_M=_BF16_PREFILL_NUM_WARPS,
        WARPS_N=1,
        NUM_STAGES=_BF16_PREFILL_NUM_STAGES,
        Q_FP8=True,
        USE_BUFFER_LOAD=_fp8_rows_buffer_load(index_k_cache),
        USE_BUFFER_STORE=logits.nbytes < 2**31,
        num_warps=_BF16_PREFILL_NUM_WARPS,
        waves_per_eu=_BF16_PREFILL_WAVES_PER_EU,
    )
    return logits


def _owned_query_group_metadata(q, pages, lengths, query_group_size):
    """Validate the dense plan before forming its strided per-request view:
    every request's page list, its last row's."""
    if query_group_size not in (1, 2, 3, 4, 5, 6):
        raise ValueError("owned-page query groups must contain one to six verify rows")
    if q.shape[0] % query_group_size:
        raise ValueError("owned-page query groups must be complete")
    if pages.shape[0] != q.shape[0] or lengths.numel() != q.shape[0]:
        raise ValueError("owned-page metadata must cover every query row")
    _check_standard_decode_metadata(q, lengths, pages, 1)
    return pages[query_group_size - 1 :: query_group_size]


def launch_gluon_dsa_index_candidates_gfx950(
    q: torch.Tensor,
    q_scales: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    pages: torch.Tensor,
    lengths: torch.Tensor,
    *,
    page_size: int,
    softmax_scale: float,
    query_group_size: int,
    query_requests: torch.Tensor | None,
    request_count: int | None,
) -> torch.Tensor:
    """Score compact pages with the exact regular-serving native arithmetic.

    Pages and visible lengths are per query. Consecutive query groups belong
    to the same request and their compact page lists are increasing prefixes
    of the final row's list. That last list is the group's union; every row
    retains its own runtime visible length. Thus appended foreign tokens do
    not remove owned keys, and four verify rows can share each key tile.
    Only each row's valid prefix is written, NaN and infinite scores as
    -inf. The tail is uninitialized.

    ``query_requests`` (each row's request, int32) and ``request_count``
    (how many the rows may belong to) let prefill-sized one-row groups share
    key tiles across a block of rows: every row's list is a prefix of a later
    row's of its request, as causal windows' are. The scores are the same.
    """
    page_stride = _check_fp8_rows_scorer_inputs(
        q, q_scales, weights, index_k_cache, page_size
    )
    if q.dtype != torch.float8_e4m3fn:
        raise TypeError(
            "owned-page native scorer requires FP8 queries and explicit scales"
        )
    table = _owned_query_group_metadata(q, pages, lengths, query_group_size)
    requests = table.shape[0]
    if any(value.nbytes >= 2**31 for value in (q, q_scales, weights, lengths)):
        raise ValueError("owned-page query buffers must fit int32 offsets")
    width = pages.shape[1] * page_size
    if width < 1:
        raise ValueError("owned-page metadata needs positive column capacity")
    logits = torch.empty((q.shape[0], width), dtype=torch.float32, device=q.device)
    if not q.shape[0]:
        return logits
    if (
        query_requests is not None
        and query_group_size == 1
        and _scores_prefill_rows(q.shape[0], request_count, q.shape[1])
    ):
        return _launch_prefill_rows(
            q,
            q_scales,
            weights,
            index_k_cache,
            pages,
            lengths,
            query_requests,
            logits,
            page_size,
            softmax_scale,
            page_stride,
        )
    rows = triton.next_power_of_2(query_group_size)
    warps_m, warps_n, block_n, waves = _fp8_rows_decode_shape(
        rows, q.shape[1], q_fp8=True
    )
    chunks = triton.cdiv(width, _BF16_DECODE_CHUNK_N)
    splits = _fp8_rows_decode_splits(requests, chunks, waves, q.device)
    gluon_dsa_decode_topk_fp8_rows_gfx950[(requests, splits)](
        q,
        q_scales,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        # Request lengths, unread: explicit_row_lens bounds every row.
        lengths,
        table,
        logits,
        q.shape[0],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        weights.stride(0),
        weights.stride(1),
        q_scales.stride(0),
        q_scales.stride(1),
        table.stride(0),
        logits.stride(0),
        float(softmax_scale),
        width,
        query_group_size,
        PAGE_SIZE=page_size,
        PAGE_STRIDE_BYTES=page_stride,
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        ROWS=rows,
        BLOCK_N=block_n,
        CHUNK_N=_BF16_DECODE_CHUNK_N,
        WARPS_M=warps_m,
        WARPS_N=warps_n,
        NUM_STAGES=_FP8_ROWS_DECODE_NUM_STAGES,
        Q_FP8=True,
        USE_BUFFER_LOAD=_fp8_rows_buffer_load(index_k_cache),
        USE_BUFFER_STORE=logits.nbytes < 2**31,
        CHUNK_LOOP=splits < chunks,
        explicit_row_lens=lengths,
        NONFINITE_TO_NEG_INF=True,
        num_warps=_BF16_DECODE_NUM_WARPS,
        waves_per_eu=waves,
    )
    return logits
