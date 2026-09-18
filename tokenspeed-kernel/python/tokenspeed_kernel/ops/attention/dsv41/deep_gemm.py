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

"""Bounded packed CSA2 scoring and DeepSelect through existing TS integrations."""

from functools import lru_cache

import torch
from tokenspeed_kernel.ops.attention.dsv4.deep_gemm import (
    _mxfp4_cache_view,
    warmup_mqa_logits,
)
from tokenspeed_kernel.ops.attention.dsv41.deep_select import (
    select_candidates,
    select_topk,
)
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _LAYOUTS,
    _finish_topk,
    _index_topk_outputs,
    clean_logits,
    dense_ranges,
    gather_index_cache,
    pack_index_queries,
    safe_metadata,
    write_selection,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
    prepare_cuda_toolkit_env,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

platform = current_platform()

if platform.is_blackwell:
    prepare_cuda_toolkit_env()
    import deep_gemm
    from tokenspeed_kernel.ops._deep_gemm.mega_moe_bf16 import (
        prepare_mega_moe_bf16_jit,
    )

    prepare_mega_moe_bf16_jit()

if platform.is_hopper:
    prepare_cuda_toolkit_env()
    import deep_gemm
    import flashinfer


def is_native_indexer_available() -> bool:
    """Whether the current platform supports packed DeepGEMM MQA scoring."""
    return platform.is_blackwell


@lru_cache(maxsize=32)
def _warmup_indexer(heads: int, device: torch.device, enable_pdl: bool) -> None:
    """Compile packed MQA kernels before capture; retain no tensors."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up packed indexer kernels before graph capture")
    deep_gemm.set_pdl(enable_pdl)
    warmup_mqa_logits(
        num_heads=heads,
        index_head_dim=128,
        cache_block_size=64,
        max_decode_tokens=32,
        device=device,
    )


def _api(queries):
    enabled = pdl_enabled()
    _warmup_indexer(queries[0].shape[1], queries[0].device, enabled)
    if deep_gemm.get_pdl() != enabled:
        deep_gemm.set_pdl(enabled)
    return deep_gemm


def _paged_scores(
    queries, cache, weights, block_table, valid_lengths, capacity, page_size
):
    if cache.stride(0) >= 2**31 or cache.stride(0) % 16 or cache.data_ptr() % 16:
        raise ValueError(
            "DeepGEMM packed pages require aligned sub-2GiB strides; use the portable solution"
        )
    if queries[0].shape[0] == 0:
        return torch.zeros((0, capacity), dtype=torch.float32, device=cache.device)
    api = _api(queries)
    # Native TMA loads have no page upper-bound mask. Sanitize a scratch table
    # before launching, leaving the scheduler's LCM tables unchanged.
    table, lengths = safe_metadata(
        block_table, valid_lengths, cache.shape[0], capacity, page_size
    )
    context_lens = lengths[:, None]
    schedule = api.get_paged_mqa_logits_metadata(
        context_lens, page_size, api.get_num_sms()
    )
    logits = api.fp8_fp4_paged_mqa_logits(
        q=(
            queries[0].contiguous().view(torch.int8).unsqueeze(1),
            queries[1].contiguous().unsqueeze(1),
        ),
        kv_cache=_mxfp4_cache_view(cache, page_size),
        weights=weights.float().contiguous(),
        context_lens=context_lens,
        block_table=table,
        schedule_meta=schedule,
        max_context_len=capacity,
        clean_logits=False,
        logits_dtype=torch.float32,
    )
    return clean_logits(
        logits, lengths, block_table, cache.shape[0], capacity, page_size
    )


def _dense_scores(queries, keys, weights, lengths, table, pages, capacity):
    api = _api(queries)
    starts, ends = dense_ranges(lengths, capacity)
    logits = api.fp8_fp4_mqa_logits(
        q=(queries[0].view(torch.int8), queries[1]),
        kv=(keys[0].view(torch.int8), keys[1]),
        weights=weights,
        cu_seq_len_k_start=starts,
        cu_seq_len_k_end=ends,
        clean_logits=False,
        max_seqlen_k=capacity,
        logits_dtype=torch.float32,
    )
    return clean_logits(logits, ends, table, pages, capacity, 64)


@register_kernel(
    "attention",
    "dsv41_index_topk",
    name="deep_gemm_dsv41_index_topk",
    solution="deep_gemm",
    signatures=[format_signature(x=dense_tensor_format(torch.bfloat16))],
    traits={"native_indexer": frozenset({True})},
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    priority=Priority.SPECIALIZED,
)
def index_topk(
    index_q,
    weights,
    index_cache,
    page_table,
    visible_lens,
    candidate_blocks,
    topk,
    candidate_topk,
    candidate_block_size,
    query_chunk_size,
    score_chunk_size,
    process_group,
    out,
):
    """Score bounded query tiles; cache and selections remain caller owned."""
    if process_group is not None or candidate_block_size != 8:
        raise ValueError(
            "Native CSA2 selection requires replicated heads and 8-row blocks"
        )
    out = _index_topk_outputs(
        index_q,
        weights,
        index_cache,
        page_table,
        visible_lens,
        candidate_blocks,
        topk,
        candidate_topk,
        candidate_block_size,
        query_chunk_size,
        score_chunk_size,
        process_group,
        out,
    )
    n = index_q.shape[0]
    rows, lengths, blocks, block_lengths = out
    capacity = page_table.shape[1] * 64
    if capacity == 0:
        rows.fill_(-1)
        lengths.zero_()
        blocks.fill_(-1)
        block_lengths.zero_()
        return out
    # Scores and selection scratch are transient and independent of configured
    # request count. Never materialize the full prefill-by-history matrix.
    dense = page_table.stride(0) == 0 and n > 1
    budget = (128 if dense else 32) << 20
    tile = min(query_chunk_size, max(1, budget // (capacity * 4)))
    cache = index_cache.as_strided(
        (index_cache.shape[0], 64 * 68), (index_cache.stride(0), 1)
    )
    queries = pack_index_queries(index_q)
    weights = weights.float().contiguous()
    keys = None
    if dense:
        # A zero row stride is a proof that all queries address one request's
        # history. Gather once for this call; never retain payload across writes.
        logical = torch.arange(capacity, dtype=torch.int64, device=index_q.device)
        pages = page_table[0, logical // 64]
        slots = (pages.to(torch.int64) * 64 + logical % 64).masked_fill(
            (pages < 0) | (pages >= index_cache.shape[0]), -1
        )
        keys = gather_index_cache(cache, slots, 64)
    for begin in range(0, n, tile):
        end = min(begin + tile, n)
        visible = visible_lens[begin:end].clamp(0, capacity).to(torch.int32)
        packed = (queries[0][begin:end], queries[1][begin:end])
        if dense:
            logits = _dense_scores(
                packed,
                keys,
                weights[begin:end],
                visible,
                page_table[begin:end],
                index_cache.shape[0],
                capacity,
            )
        else:
            logits = _paged_scores(
                packed,
                cache,
                weights[begin:end],
                page_table[begin:end],
                visible,
                capacity,
                64,
            )
        candidates = None if candidate_blocks is None else candidate_blocks[begin:end]
        selected = select_topk(logits, visible, candidates, topk, 8)
        chosen = (
            select_candidates(logits, visible, candidate_topk, 8)
            if candidate_topk
            else None
        )
        write_selection(
            selected,
            chosen,
            tuple(tensor[begin:end] for tensor in out),
        )
    return out


# ---------------------------------------------------------------------------
# Hopper: FP8 index rows and DeepGEMM's FP8 MQA logits
#
# sm90 has no FP4 tensor cores, so the packed CSA2 path above cannot run there.
# It does carry DeepGEMM's DeepSeek-V3.2 logits kernels, which read index rows
# as 128 E4M3 values followed by one FP32 scale, page-planar. The "v4" cache
# format stores exactly those rows, so scoring moves onto tensor cores; the
# selection that follows runs on FlashInfer's radix top-k, DeepSelect being
# sm100-only.
# ---------------------------------------------------------------------------

_INDEX_ROW_BYTES = _LAYOUTS["index_v4"][3]
_INDEX_VALUE_BYTES = _LAYOUTS["index_v4"][2]


def is_hopper_indexer_available() -> bool:
    """Whether this platform scores FP8 index rows with DeepGEMM."""
    return platform.is_hopper


@lru_cache(maxsize=8)
def _warmup_hopper_indexer(heads: int, device: torch.device, enable_pdl: bool) -> None:
    """JIT the FP8 logits kernels before capture; retain no tensors."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up the FP8 indexer kernels before graph capture")
    deep_gemm.set_pdl(enable_pdl)
    q = torch.zeros((1, heads, 128), dtype=torch.float8_e4m3fn, device=device)
    weights = torch.zeros((1, heads), dtype=torch.float32, device=device)
    keys = torch.zeros((64, 128), dtype=torch.float8_e4m3fn, device=device)
    scales = torch.zeros((64,), dtype=torch.float32, device=device)
    bounds = torch.zeros((1,), dtype=torch.int32, device=device)
    deep_gemm.fp8_mqa_logits(
        q=q,
        kv=(keys, scales),
        weights=weights,
        cu_seq_len_k_start=bounds,
        cu_seq_len_k_end=bounds + 64,
        clean_logits=False,
        max_seqlen_k=64,
    )
    cache = torch.zeros((1, 64, 1, _INDEX_ROW_BYTES), dtype=torch.uint8, device=device)
    context = torch.full((1, 1), 64, dtype=torch.int32, device=device)
    table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    deep_gemm.fp8_paged_mqa_logits(
        q.unsqueeze(1),
        cache,
        weights,
        context,
        table,
        deep_gemm.get_paged_mqa_logits_metadata(context, 64, deep_gemm.get_num_sms()),
        64,
        clean_logits=False,
    )
    # The selector JITs and autotunes on its first call. dsa_graph_safe only
    # makes the kernel replayable; it does not make that first call safe, so
    # force it here rather than inside a capture.
    for columns in (64, 2048):
        _select(
            torch.zeros((1, columns), dtype=torch.float32, device=device),
            min(64, columns),
            True,
            torch.full((1, min(64, columns)), -1, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )


def _hopper_api(queries):
    enabled = pdl_enabled()
    _warmup_hopper_indexer(queries.shape[1], queries.device, enabled)
    if deep_gemm.get_pdl() != enabled:
        deep_gemm.set_pdl(enabled)
    return deep_gemm


def _quantize_index_queries(index_q):
    """Return E4M3 queries and the per-(token, head) scale to fold into weights.

    Mirrors the ``index_v4`` cache codec so queries and keys are quantized the
    same way. DeepGEMM's FP8 logits kernels take no query scale of their own.
    """
    values = index_q.float()
    scale = values.abs().amax(-1, keepdim=True).clamp_min(1.0e-6) / 448.0
    quantized = (values / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized.contiguous(), scale.squeeze(-1)


def _index_pages(cache):
    """View a strided [pages, 64, 132] index field as its contiguous page bytes."""
    return cache.as_strided(
        (cache.shape[0], 64 * _INDEX_ROW_BYTES), (cache.stride(0), 1)
    )


def _index_planes(cache):
    """Split a page-planar FP8 index field into its value and scale planes."""
    pages = cache.shape[0]
    flat = _index_pages(cache)
    values = flat[:, : 64 * _INDEX_VALUE_BYTES]
    scales = flat[:, 64 * _INDEX_VALUE_BYTES :]
    return (
        values.reshape(pages, 64, _INDEX_VALUE_BYTES),
        scales.reshape(pages, 64 * 4).view(torch.float32),
    )


def _gather_index_fp8(cache, slots):
    """Read the selected physical slots as (E4M3 values, FP32 scales)."""
    values, scales = _index_planes(cache)
    live = (slots >= 0) & (slots < cache.shape[0] * 64)
    safe = slots.clamp_min(0)
    pages, rows = safe // 64, safe % 64
    gathered = values[pages, rows]
    gathered = torch.where(live[:, None], gathered, torch.zeros_like(gathered))
    row_scales = torch.where(live, scales[pages, rows], torch.zeros_like(live).float())
    return gathered.view(torch.float8_e4m3fn).contiguous(), row_scales.contiguous()


def _hopper_dense_scores(queries, keys, weights, lengths, table, pages, capacity):
    starts, ends = dense_ranges(lengths, capacity)
    logits = _hopper_api(queries[0]).fp8_mqa_logits(
        q=queries[0],
        kv=keys,
        weights=weights,
        cu_seq_len_k_start=starts,
        cu_seq_len_k_end=ends,
        clean_logits=False,
        max_seqlen_k=capacity,
    )
    return clean_logits(logits, ends, table, pages, capacity, 64)


def _hopper_paged_scores(queries, cache, weights, block_table, valid_lengths, capacity):
    if cache.stride(0) >= 2**31 or cache.stride(0) % 16 or cache.data_ptr() % 16:
        raise ValueError(
            "DeepGEMM packed pages require aligned sub-2GiB strides; use the portable solution"
        )
    if queries[0].shape[0] == 0:
        return torch.zeros((0, capacity), dtype=torch.float32, device=cache.device)
    api = _hopper_api(queries[0])
    table, lengths = safe_metadata(
        block_table, valid_lengths, cache.shape[0], capacity, 64
    )
    # DeepGEMM's schedule generator faults when the batch holds no keys at all
    # (every context length 0), which a padded warmup batch is. Score block 0
    # for such rows -- safe_metadata made every page valid -- and let
    # clean_logits erase it below with the true lengths. No sync, so this
    # holds under graph capture too.
    context_lens = lengths.clamp_min(1)[:, None]
    schedule = api.get_paged_mqa_logits_metadata(context_lens, 64, api.get_num_sms())
    # DeepGEMM reads the fused [pages, block, 1, values + scale] tensor.
    flat = _index_pages(cache)
    view = flat.as_strided(
        (cache.shape[0], 64, 1, _INDEX_ROW_BYTES),
        (flat.stride(0), _INDEX_ROW_BYTES, _INDEX_ROW_BYTES, 1),
    )
    logits = api.fp8_paged_mqa_logits(
        queries[0].unsqueeze(1),
        view,
        weights,
        context_lens,
        table,
        schedule,
        capacity,
        clean_logits=False,
    )
    return clean_logits(
        logits.reshape(queries[0].shape[0], -1)[:, :capacity],
        lengths,
        block_table,
        cache.shape[0],
        capacity,
        64,
    )


def _restrict_to_candidates(logits, candidates):
    """Mask every row outside the candidate blocks, keeping the row addressing.

    Null blocks are routed to a sentinel column instead of being scattered as
    False: a scatter writes duplicate indices in an undefined order, so a null
    block sharing columns with a live one could otherwise erase it.
    """
    tokens, width = logits.shape
    blocks = candidates.to(torch.int64)
    columns = (
        blocks.clamp_min(0)[:, :, None] * 8
        + torch.arange(8, device=logits.device, dtype=torch.int64)
    ).reshape(tokens, -1)
    live = (blocks >= 0)[:, :, None].expand(-1, -1, 8).reshape(tokens, -1)
    columns = torch.where(live & (columns < width), columns, width)
    allowed = torch.zeros((tokens, width + 1), dtype=torch.bool, device=logits.device)
    allowed.scatter_(1, columns, True)
    return logits.masked_fill(~allowed[:, :width], -float("inf"))


def _block_maxima(logits, visible):
    """Reduce rows to their 8-row block maximum, pinning the newest block."""
    tokens, width = logits.shape
    padded = (width + 7) // 8 * 8
    if padded != width:
        logits = torch.nn.functional.pad(
            logits, (0, padded - width), value=-float("inf")
        )
    blocks = logits.reshape(tokens, -1, 8).amax(-1)
    newest = ((visible.to(torch.int64) - 1) // 8).clamp(0, blocks.shape[1] - 1)
    rows = torch.arange(tokens, device=blocks.device)
    current = blocks[rows, newest]
    blocks[rows, newest] = torch.where(
        current > -float("inf"), torch.full_like(current, float("inf")), current
    )
    return blocks


def _select(scores, k, graph_safe, destination, lengths):
    """Take the k best columns per row with FlashInfer's radix selector.

    ``tie_break`` matches the portable path's packed ordering, which breaks
    equal scores toward the smaller row id, and costs nothing. ``graph_safe``
    does cost: it is ~1.7x slower on prefill-sized rows, so only the captured
    decode path asks for it. V4.1 never captures a prefill graph.
    """
    width = min(k, scores.shape[1])
    values, indices = flashinfer.top_k(
        scores,
        width,
        sorted=False,
        deterministic=False,
        tie_break=flashinfer.TopKTieBreak.SMALL,
        dsa_graph_safe=graph_safe,
    )
    _finish_topk(values, indices.to(torch.int64), destination, lengths)


def _flashinfer_select(
    logits, visible, candidates, topk, candidate_topk, graph_safe, out
):
    rows, lengths, blocks, block_lengths = out
    _select(
        logits if candidates is None else _restrict_to_candidates(logits, candidates),
        topk,
        graph_safe,
        rows,
        lengths,
    )
    if candidate_topk:
        _select(
            _block_maxima(logits, visible),
            candidate_topk,
            graph_safe,
            blocks,
            block_lengths,
        )


@register_kernel(
    "attention",
    "dsv41_index_topk",
    name="deep_gemm_hopper_dsv41_index_topk",
    solution="deep_gemm",
    signatures=[format_signature(x=dense_tensor_format(torch.bfloat16))],
    traits={"native_indexer": frozenset({True})},
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(9, 0),
        vendors=frozenset({"nvidia"}),
    ),
    priority=Priority.SPECIALIZED,
)
def hopper_index_topk(
    index_q,
    weights,
    index_cache,
    page_table,
    visible_lens,
    candidate_blocks,
    topk,
    candidate_topk,
    candidate_block_size,
    query_chunk_size,
    score_chunk_size,
    process_group,
    out,
):
    """Score FP8 index rows on tensor cores; selection and outputs stay caller owned."""
    if process_group is not None or candidate_block_size != 8:
        raise ValueError(
            "Native CSA2 selection requires replicated heads and 8-row blocks"
        )
    out = _index_topk_outputs(
        index_q,
        weights,
        index_cache,
        page_table,
        visible_lens,
        candidate_blocks,
        topk,
        candidate_topk,
        candidate_block_size,
        query_chunk_size,
        score_chunk_size,
        process_group,
        out,
    )
    n = index_q.shape[0]
    rows, lengths, blocks, block_lengths = out
    capacity = page_table.shape[1] * 64
    if capacity == 0:
        rows.fill_(-1)
        lengths.zero_()
        blocks.fill_(-1)
        block_lengths.zero_()
        return out
    dense = page_table.stride(0) == 0 and n > 1
    budget = (128 if dense else 32) << 20
    tile = min(query_chunk_size, max(1, budget // (capacity * 4)))
    queries, query_scales = _quantize_index_queries(index_q)
    # The logits kernels apply no query scale, so it rides in the weights.
    folded = (weights.float() * query_scales).contiguous()
    keys = None
    if dense:
        logical = torch.arange(capacity, dtype=torch.int64, device=index_q.device)
        pages = page_table[0, logical // 64]
        slots = (pages.to(torch.int64) * 64 + logical % 64).masked_fill(
            (pages < 0) | (pages >= index_cache.shape[0]), -1
        )
        keys = _gather_index_fp8(index_cache, slots)
    for begin in range(0, n, tile):
        end = min(begin + tile, n)
        visible = visible_lens[begin:end].clamp(0, capacity).to(torch.int32)
        packed = (queries[begin:end],)
        if dense:
            logits = _hopper_dense_scores(
                packed,
                keys,
                folded[begin:end],
                visible,
                page_table[begin:end],
                index_cache.shape[0],
                capacity,
            )
        else:
            logits = _hopper_paged_scores(
                packed,
                index_cache,
                folded[begin:end],
                page_table[begin:end],
                visible,
                capacity,
            )
        candidates = None if candidate_blocks is None else candidate_blocks[begin:end]
        # DeepSelect ships no sm90 cubin, so selection runs on FlashInfer's
        # radix Top-K -- the same backend SGLang's DSA indexer uses.
        _flashinfer_select(
            logits,
            visible,
            candidates,
            topk,
            candidate_topk,
            not dense,
            tuple(tensor[begin:end] for tensor in out),
        )
    return out
