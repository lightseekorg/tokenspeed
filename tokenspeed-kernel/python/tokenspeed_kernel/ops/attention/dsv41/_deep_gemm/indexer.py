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

"""Packed index-cache preparation and native sparse scoring for DeepGEMM."""

from __future__ import annotations

from functools import lru_cache

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsv41._metadata import IndexGeometry, IndexMetadata
from tokenspeed_kernel.thirdparty.deep_select import deep_select_alignment


@triton.jit(
    do_not_specialize=["X0", "L0", "T0", "T1", "TC", "NP", "CAP", "STRIDE"],
)
def _clean_logits_block_max(
    X,
    L,
    T,
    O,
    S,
    E,
    X0,
    L0,
    T0,
    T1,
    TC,
    NP,
    CAP,
    STRIDE,
    PAGE: tl.constexpr,
    PAGED: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    end = tl.minimum(tl.maximum(tl.load(L + row * L0), 0), CAP)
    block = tile * BLOCKS + tl.arange(0, BLOCKS)
    position = block[:, None] * 8 + tl.arange(0, 8)[None, :]
    valid = (position < CAP) & (position < end)
    if PAGED:
        page = tl.load(
            T + row * T0 + (position // PAGE) * T1,
            valid & (position // PAGE < TC),
            other=0,
        ).to(tl.int64)
        valid &= (page >= 0) & (page < NP) & (position // PAGE < TC)
    values = tl.load(X + row * X0 + position, valid, other=-float("inf"))
    tl.store(O + row * CAP + position, values, position < CAP)
    maxima = tl.max(values, axis=1)
    maxima = tl.where(
        (end > 0) & (block == (end - 1) // 8) & (maxima > -float("inf")),
        float("inf"),
        maxima,
    )
    tl.store(S + row * STRIDE + block, maxima, block < STRIDE)
    if tile == 0:
        tl.store(E + row, tl.cdiv(end, 8))


def clean_logits_and_block_max(
    values, lengths, table, pages, capacity, page_size, input_alignment
):
    """Return fully cleaned logits, aligned eight-row maxima and block ends.

    FP32 values and integer lengths follow the internal clean_logits contract;
    table is the original logical page table (None for unpaged input). Invalid
    pages and future positions are -inf. Both output matrices, including their
    alignment tails, are completely initialized. The newest visible block is
    pinned to +inf only when its maximum is greater than -inf.

    input_alignment is the native selector's input row alignment in BYTES.
    The returned contiguous int32 ends are ceil(clamp(length, 0, capacity) / 8),
    for the block selector only; row Top-K must use the original token ends.
    """
    if (
        values.ndim != 2
        or values.shape[0] != lengths.numel()
        or values.shape[1] < capacity
        or values.stride(1) != 1
        or values.dtype != torch.float32
    ):
        raise RuntimeError("DeepGEMM returned incompatible FP32 score geometry")
    alignment = input_alignment // 4
    stride = triton.cdiv(triton.cdiv(capacity, 8), alignment) * alignment
    cleaned = torch.empty(
        (values.shape[0], capacity), device=values.device, dtype=torch.float32
    )
    blocks = torch.empty(
        (values.shape[0], stride), device=values.device, dtype=torch.float32
    )
    ends = torch.empty(values.shape[0], device=values.device, dtype=torch.int32)
    if values.shape[0] and capacity:
        # Alignment can extend block scores beyond the final logits tile.
        # Those programs initialize the entire native-prefetch padding to -inf.
        tiles = max(triton.cdiv(capacity, 1024), triton.cdiv(stride, 128))
        _clean_logits_block_max[(values.shape[0], tiles)](
            values,
            lengths,
            table if table is not None else lengths,
            cleaned,
            blocks,
            ends,
            values.stride(0),
            lengths.stride(0),
            table.stride(0) if table is not None else 0,
            table.stride(1) if table is not None else 0,
            table.shape[1] if table is not None else 0,
            pages,
            capacity,
            stride,
            page_size,
            table is not None,
            128,
            num_warps=4,
        )
    elif values.shape[0]:
        ends.zero_()
    return cleaned, blocks, ends


@triton.jit(
    do_not_specialize=["C0", "T0", "N", "NP"],
)
def _gather_index_pages(C, T, V, F, C0, T0, N, NP, P: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    page = tl.load(T + (row // P) * T0, row < N, other=-1).to(tl.int64)
    valid = (row < N) & (page >= 0) & (page < NP)
    offset = row % P
    d = tl.arange(0, 64)
    data = tl.load(
        C + page[:, None] * C0 + offset[:, None] * 64 + d[None, :],
        valid[:, None],
        other=0,
    )
    tl.store(V + row[:, None] * 64 + d[None, :], data, (row < N)[:, None])
    b = tl.arange(0, 4)
    sf = tl.load(
        C + page[:, None] * C0 + P * 64 + offset[:, None] * 4 + b[None, :],
        valid[:, None],
        other=127,
    ).to(tl.uint32)
    word = tl.sum(sf << (b[None, :] * 8), 1)
    tl.store(F + row, word.to(tl.int32), row < N)


def gather_index_pages(
    cache: torch.Tensor, pages: torch.Tensor, page_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather validated FP4 pages into uint8 [rows,64] and int32 [rows] scales.

    The caller supplies uint8 [pages,4352] cache with contiguous page bytes
    and an integer [logical_pages] table on the same GPU; page_size is 64.
    Page zero is valid. Invalid pages yield zero payload and 127 scale bytes.
    Each call reads current cache bytes, including padded or strided pages.
    """
    count = pages.numel() * page_size
    data = torch.empty((count, 64), dtype=torch.uint8, device=cache.device)
    scales = torch.empty((count,), dtype=torch.int32, device=cache.device)
    if count:
        _gather_index_pages[(triton.cdiv(count, 32),)](
            cache,
            pages,
            data,
            scales,
            cache.stride(0),
            pages.stride(0),
            count,
            cache.shape[0],
            page_size,
            32,
            num_warps=4,
        )
    return data, scales


@triton.jit(
    do_not_specialize=["C0", "C1", "L0", "T0", "T1", "PAGES", "NC", "CAP", "WIDTH"],
)
def _normalize_candidate_blocks(
    C,
    L,
    TABLE,
    NORMALIZED,
    SPARSE,
    CONTEXT,
    END,
    C0,
    C1,
    L0,
    T0,
    T1,
    PAGES,
    NC,
    CAP,
    WIDTH,
    SKIP_SCORE_TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, BLOCK)
    end = tl.minimum(tl.maximum(tl.load(L + row * L0), 0), CAP)
    candidate = tl.load(C + row * C0 + i * C1, i < NC, -1)
    valid = (candidate >= 0) & (candidate < tl.cdiv(end, 8))
    page = tl.load(TABLE + row * T0 + (candidate // 8).to(tl.int64) * T1, valid, -1)
    candidate = tl.where(
        valid & (page >= 0) & (page < PAGES), candidate, 2147483647
    ).to(tl.int32)
    previous = tl.gather(candidate, tl.maximum(i - 1, 0), axis=0)
    if tl.sum((candidate < previous).to(tl.int32), 0) != 0:
        candidate = tl.sort(candidate, descending=False)
    previous = tl.gather(candidate, tl.maximum(i - 1, 0), axis=0)
    valid = (candidate != 2147483647) & ((i == 0) | (candidate != previous))
    rank = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32), 0)
    last = tl.max(tl.where(valid, candidate, 0), 0)
    tl.store(NORMALIZED + row * WIDTH + rank, candidate, valid)
    tl.store(NORMALIZED + row * WIDTH + i, -1, (i >= count) & (i < WIDTH))
    # Native metadata expects sorted block IDs, with the last block repeated
    # in vector-load padding. Its context counts candidate rows, not history.
    tl.store(SPARSE + row * WIDTH + rank, candidate, valid)
    tl.store(SPARSE + row * WIDTH + i, last, (i >= count) & (i < WIDTH))
    tail = tl.maximum(last * 8 + 8 - end, 0)
    candidate_end = tl.where(count > 0, count * 8 - tail, 0)
    # Every valid candidate is selected when it fits Top-K. Keep the true
    # selection length, but submit no scoring work for this query. The native
    # schedule already handles zero-length rows (including padded requests).
    tl.store(CONTEXT + row, tl.where(candidate_end <= SKIP_SCORE_TOPK, 0, count * 8))
    tl.store(END + row, candidate_end)


@triton.jit(
    do_not_specialize=["S0", "WIDTH"],
)
def _mask_candidate_logits(
    SCORES,
    END,
    S0,
    WIDTH,
    SKIP_SCORE_TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    end = tl.load(END + row)
    valid = (i < WIDTH) & (i < end)
    if end <= SKIP_SCORE_TOPK:
        # The scorer left these rows unwritten. Equal finite scores select
        # the entire normalized pool without reading uninitialized logits.
        score = tl.where(valid, 0.0, -float("inf"))
    else:
        score = tl.load(SCORES + row * S0 + i, valid, -float("inf")).to(tl.float32)
    tl.store(SCORES + row * S0 + i, score, i < WIDTH)


@lru_cache(maxsize=32)
def _warmup_sparse_mqa(api, blocks, device, enable_pdl):
    """Compile sparse metadata/scoring before capture; retain no GPU tensors."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up sparse indexer kernels before graph capture")
    api.set_pdl(enable_pdl)
    q = torch.zeros((32, 1, 32, 64), dtype=torch.int8, device=device)
    scales = torch.full((32, 1, 32), 0x7F7F7F7F, dtype=torch.int32, device=device)
    cache = torch.zeros((1, 4608), dtype=torch.uint8, device=device)
    cache[:, 4096:4352] = 127
    table = torch.zeros((32, 1), dtype=torch.int32, device=device)
    context = torch.ones(32, dtype=torch.int32, device=device)
    requests = torch.arange(32, dtype=torch.int32, device=device)
    candidates = torch.zeros((32, blocks), dtype=torch.int32, device=device)
    weights = torch.zeros((32, 32), dtype=torch.bfloat16, device=device)
    metadata = api.get_paged_sparse_mqa_logits_metadata(
        context, table, requests, 64, candidates, torch.int8, 8
    )
    api.fp8_fp4_paged_sparse_mqa_logits(
        (q, scales),
        cache.as_strided((1, 64, 1, 68), (4608, 68, 68, 1)),
        weights,
        metadata,
        blocks,
        8,
    )


def _sparse_candidate_scores(
    queries: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cache: torch.Tensor,
    page_table: torch.Tensor,
    visible: torch.Tensor,
    candidates: torch.Tensor,
    api: object,
    *,
    skip_score_topk: int,
    index_metadata: IndexMetadata | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return BF16 logits, candidate-row ends and sorted unique 8-row block IDs.

    index_topk supplies a query tile, 1..2048 candidate blocks, BF16 weights
    and FP4 cache pages with 512-byte aligned sub-2GiB strides. Native scoring
    reads K in place. index_metadata shares immutable candidate geometry and
    schedules within one invocation; queries, weights, KV values and logits
    are never cached.
    Invalid pages, duplicates and future blocks are removed; page zero is valid.

    skip_score_topk is the selection width, or zero for all scores. Pools that
    fit return zero valid logits and submit no scoring work, using live GPU
    lengths. Only the selected set is meaningful for those rows.
    """
    tokens, count = candidates.shape
    capacity = page_table.shape[1] * 64
    alignment = deep_select_alignment()[0] // 2
    blocks = triton.cdiv(count, alignment // 8) * (alignment // 8)
    width = blocks * 8
    _warmup_sparse_mqa(api, blocks, cache.device, api.get_pdl())
    key = (
        cache.shape[0],
        capacity,
        blocks,
        skip_score_topk,
        api.get_num_sms(),
        api.get_pdl(),
        tuple(
            (t.data_ptr(), t.shape, t.stride(), t.dtype)
            for t in (page_table, visible, candidates)
        ),
    )
    geometry = index_metadata.sparse.get(key) if index_metadata is not None else None
    if geometry is None:
        normalized = torch.empty(
            (tokens, blocks), dtype=torch.int32, device=cache.device
        )
        ends = torch.empty(tokens, dtype=torch.int32, device=cache.device)
        sparse = torch.empty_like(normalized)
        context = torch.empty_like(ends)
        _normalize_candidate_blocks[(tokens,)](
            candidates,
            visible,
            page_table,
            normalized,
            sparse,
            context,
            ends,
            *candidates.stride(),
            visible.stride(0),
            *page_table.stride(),
            cache.shape[0],
            count,
            capacity,
            blocks,
            skip_score_topk,
            triton.next_power_of_2(blocks),
            num_warps=4,
        )
        # A broadcast table proves request identity and permits native query-pair
        # reuse. Otherwise keep rows distinct, including reordered/strided views.
        requests = (
            torch.zeros(tokens, dtype=torch.int32, device=cache.device)
            if page_table.stride(0) == 0
            else torch.arange(tokens, dtype=torch.int32, device=cache.device)
        )
        table = page_table.to(torch.int32).contiguous()
        if table.stride(0) < table.shape[1]:
            # A one-row broadcast slice is "contiguous" to Torch despite its zero
            # row stride. Native metadata also uses that stride as its page bound.
            table = table.clone(memory_format=torch.contiguous_format)
        schedule = api.get_paged_sparse_mqa_logits_metadata(
            context, table, requests, 64, sparse, torch.int8, 8
        )
        if index_metadata is not None:
            # Retain results and source views to keep pointer-based keys valid.
            # The schedule embeds the page mapping, so build scratch can expire.
            index_metadata.sparse[key] = IndexGeometry(
                (
                    normalized,
                    ends,
                    schedule,
                    page_table,
                    visible,
                    candidates,
                )
            )
    else:
        normalized, ends, schedule, *_ = geometry.acquire()
    scores = api.fp8_fp4_paged_sparse_mqa_logits(
        q=(
            queries[0].contiguous().view(torch.int8).unsqueeze(1),
            queries[1].contiguous().unsqueeze(1),
        ),
        kv_cache=cache.as_strided(
            (cache.shape[0], 64, 1, 68), (cache.stride(0), 68, 68, 1)
        ),
        weights=weights.contiguous(),
        metadata=schedule,
        num_max_sparse_blocks=blocks,
        sparse_block_kv=8,
    )
    _mask_candidate_logits[(tokens, triton.cdiv(width, 1024))](
        scores,
        ends,
        scores.stride(0),
        width,
        skip_score_topk,
        1024,
    )
    return scores, ends, normalized
