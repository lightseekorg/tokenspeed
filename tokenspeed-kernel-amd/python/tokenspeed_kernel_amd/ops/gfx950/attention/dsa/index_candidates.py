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
    _BF16_DECODE_BLOCK_N,
    _BF16_DECODE_CHUNK_N,
    _BF16_DECODE_NUM_WARPS,
    _FP8_ROWS_DECODE_NUM_STAGES,
    _FP8_ROWS_DECODE_WAVES_PER_EU,
    _check_fp8_rows_scorer_inputs,
    _check_standard_decode_metadata,
    _fp8_rows_buffer_load,
    _fp8_rows_decode_splits,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.standard_cache_logits import (
    gluon_dsa_decode_topk_fp8_rows_gfx950,
)


def _owned_query_group_metadata(q, pages, lengths, query_group_size):
    """Validate the dense plan before forming its strided per-request view."""
    if query_group_size not in (1, 2, 3, 4, 5, 6):
        raise ValueError("owned-page query groups must contain one to six verify rows")
    if q.shape[0] % query_group_size:
        raise ValueError("owned-page query groups must be complete")
    if pages.shape[0] != q.shape[0] or lengths.numel() != q.shape[0]:
        raise ValueError("owned-page metadata must cover every query row")
    _check_standard_decode_metadata(q, lengths, pages, 1)
    table = pages[query_group_size - 1 :: query_group_size]
    seq_lens = lengths[query_group_size - 1 :: query_group_size].contiguous()
    return table, seq_lens


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
    query_group_size: int = 1,
) -> torch.Tensor:
    """Score compact pages with the exact regular-serving native arithmetic.

    Pages and visible lengths are per query. Consecutive query groups belong
    to the same request and their compact page lists are increasing prefixes
    of the final row's list. That last list is the group's union; every row
    retains its own runtime visible length. Thus appended foreign tokens do
    not remove owned keys, and four verify rows can share each key tile.
    Only each row's valid prefix is written. The tail is uninitialized.
    """
    page_stride = _check_fp8_rows_scorer_inputs(
        q, q_scales, weights, index_k_cache, page_size
    )
    if q.dtype != torch.float8_e4m3fn:
        raise TypeError(
            "owned-page native scorer requires FP8 queries and explicit scales"
        )
    table, seq_lens = _owned_query_group_metadata(q, pages, lengths, query_group_size)
    if any(value.nbytes >= 2**31 for value in (q, q_scales, weights, lengths)):
        raise ValueError("owned-page query buffers must fit int32 offsets")
    width = pages.shape[1] * page_size
    if width < 1:
        raise ValueError("owned-page metadata needs positive column capacity")
    logits = torch.empty((q.shape[0], width), dtype=torch.float32, device=q.device)
    if not q.shape[0]:
        return logits
    rows = triton.next_power_of_2(query_group_size)
    warps_m = min(rows, _BF16_DECODE_NUM_WARPS)
    warps_n = _BF16_DECODE_NUM_WARPS // warps_m
    block_n = max(_BF16_DECODE_BLOCK_N, 32 * warps_n)
    waves = 1 if rows * q.shape[1] > 256 else 2
    if q.shape[1] == 32 and rows <= 4:
        waves = _FP8_ROWS_DECODE_WAVES_PER_EU
    chunks = triton.cdiv(width, _BF16_DECODE_CHUNK_N)
    splits = _fp8_rows_decode_splits(seq_lens.numel(), chunks, waves, q.device)
    gluon_dsa_decode_topk_fp8_rows_gfx950[(seq_lens.numel(), splits)](
        q,
        q_scales,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        seq_lens,
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
        num_warps=_BF16_DECODE_NUM_WARPS,
        waves_per_eu=waves,
    )
    return logits
