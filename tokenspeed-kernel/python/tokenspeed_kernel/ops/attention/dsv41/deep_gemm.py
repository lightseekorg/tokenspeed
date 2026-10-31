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

import importlib.util
from functools import lru_cache

import torch
from tokenspeed_kernel.ops.attention.dsv4.deep_gemm import (
    _mxfp4_cache_view,
    warmup_mqa_logits,
)
from tokenspeed_kernel.ops.attention.dsv41._deep_gemm.indexer import (
    _sparse_candidate_scores,
    clean_logits_and_block_max,
    gather_index_pages,
)
from tokenspeed_kernel.ops.attention.dsv41._inputs import IndexQueryInput
from tokenspeed_kernel.ops.attention.dsv41._metadata import IndexGeometry
from tokenspeed_kernel.ops.attention.dsv41._triton.rope import (
    triton_dsv41_index_query_rope_quant,
)
from tokenspeed_kernel.ops.attention.dsv41.cute_dsl import (
    sparse_index_scores,
    sparse_index_scores_supported,
)
from tokenspeed_kernel.ops.attention.dsv41.deep_select import (
    _native_indices,
    _select_topk_prepared,
    select_candidates,
    select_topk,
)
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _LAYOUTS,
    _finish_topk,
    _index_topk_outputs,
    block_maxima_with_lengths,
    candidate_scores,
    clean_block_maxima,
    clean_logits,
    clean_logits_for_selection,
    dense_ranges,
    finish_selection,
    pack_index_inputs,
    quantize_index_queries,
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
from tokenspeed_kernel.thirdparty.deep_select import deep_select_alignment

platform = current_platform()

if platform.is_blackwell and importlib.util.find_spec("deep_gemm") is not None:
    prepare_cuda_toolkit_env()
    import deep_gemm

if platform.is_hopper:
    prepare_cuda_toolkit_env()
    if importlib.util.find_spec("deep_gemm") is not None:
        import deep_gemm

    # The selector compiles with nvidia-cutlass-dsl, a CUDA-only dependency.
    from tokenspeed_kernel.ops.attention.dsa._cute_dsl import deep_select as row_select


def is_native_indexer_available() -> bool:
    """Whether the current platform supports packed DeepGEMM MQA scoring."""
    return platform.is_blackwell and importlib.util.find_spec("deep_gemm") is not None


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
    queries,
    cache,
    weights,
    block_table,
    valid_lengths,
    capacity,
    page_size,
    index_metadata,
    tile_begin,
    tile_end,
    prepare_blocks,
):
    if cache.stride(0) >= 2**31 or cache.stride(0) % 16 or cache.data_ptr() % 16:
        raise ValueError(
            "DeepGEMM packed pages require aligned sub-2GiB strides; use the portable solution"
        )
    if queries[0].shape[0] == 0:
        return (
            torch.zeros((0, capacity), dtype=torch.float32, device=cache.device),
            valid_lengths,
            None,
        )
    api = _api(queries)
    # Native TMA loads have no page upper-bound mask. Sanitize a scratch table
    # before launching, leaving the scheduler's LCM tables unchanged.
    key = (tile_begin, tile_end, cache.shape[0], capacity, page_size, api.get_num_sms())
    geometry = index_metadata.paged.get(key) if index_metadata is not None else None
    if geometry is None:
        table, lengths = safe_metadata(
            block_table, valid_lengths, cache.shape[0], capacity, page_size
        )
        context_lens = lengths[:, None]
        schedule = api.get_paged_mqa_logits_metadata(
            context_lens, page_size, api.get_num_sms()
        )
        if index_metadata is not None:
            geometry = IndexGeometry((table, lengths, context_lens, schedule))
            index_metadata.paged[key] = geometry
    else:
        table, lengths, context_lens, schedule = geometry.acquire()
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
    if prepare_blocks:
        input_alignment, _ = deep_select_alignment()
        cleaned, block_scores, block_ends = clean_logits_and_block_max(
            logits,
            lengths,
            block_table,
            cache.shape[0],
            capacity,
            page_size,
            input_alignment,
        )
        return cleaned, lengths, (block_scores, block_ends)
    return (
        clean_logits_for_selection(
            logits, lengths, block_table, cache.shape[0], capacity, page_size
        ),
        lengths,
        None,
    )


def _dense_scores(
    queries, keys, weights, lengths, table, pages, capacity, prepare_blocks
):
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
    if prepare_blocks:
        input_alignment, _ = deep_select_alignment()
        cleaned, block_scores, block_ends = clean_logits_and_block_max(
            logits, ends, table, pages, capacity, 64, input_alignment
        )
        return cleaned, (block_scores, block_ends)
    return clean_logits(logits, ends, table, pages, capacity, 64), None


@register_kernel(
    "attention",
    "dsv41_index_topk",
    name="deep_gemm_dsv41_index_topk",
    solution="deep_gemm",
    signatures=[format_signature(x=dense_tensor_format(torch.bfloat16))],
    # Fuses RoPE and query quantization when index_q is an IndexQueryInput.
    traits={
        "native_indexer": frozenset({True}),
        "supports_index_query_input": frozenset({True}),
    },
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
    index_metadata,
):
    """Score bounded query tiles; cache and selections remain caller owned."""
    if process_group is not None or candidate_block_size != 8:
        raise ValueError(
            "Native CSA2 selection requires replicated heads and 8-row blocks"
        )
    query_values = index_q.values if isinstance(index_q, IndexQueryInput) else index_q
    out = _index_topk_outputs(
        query_values,
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
    n = query_values.shape[0]
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
    sparse = _use_sparse_candidate_scores(index_cache, candidate_blocks, candidate_topk)
    dense = not sparse and page_table.stride(0) == 0 and n > 1
    # Wide histories benefit from skipping wholly future cleaning tiles;
    # combine full-width block preparation only for smaller capacities.
    prepare_blocks = candidate_topk > 0 and capacity <= 65536
    # Sparse width is bounded by candidates; use the caller's query tile.
    tile = query_chunk_size
    if not sparse:
        budget = (128 if dense else 32) << 20
        tile = min(tile, max(1, budget // (capacity * 4)))
    cache = index_cache.as_strided(
        (index_cache.shape[0], 64 * 68), (index_cache.stride(0), 1)
    )
    weight_dtype = torch.bfloat16 if sparse else torch.float32
    if isinstance(index_q, IndexQueryInput):
        queries, weights = triton_dsv41_index_query_rope_quant(
            index_q.values,
            weights,
            index_q.positions,
            index_q.cos_sin_cache,
            index_q.weight_scale,
            weight_dtype,
        )
    else:
        queries, weights = pack_index_inputs(index_q, weights, weight_dtype)
    keys = None
    if dense:
        # A zero row stride is a proof that all queries address one request's
        # history. Gather once for this call; never retain payload across writes.
        keys = gather_index_pages(cache, page_table[0], 64)
    for begin in range(0, n, tile):
        end = min(begin + tile, n)
        visible = visible_lens[begin:end]
        packed = (queries[0][begin:end], queries[1][begin:end])
        candidates = None if candidate_blocks is None else candidate_blocks[begin:end]
        if sparse:
            logits, ends, normalized = _sparse_candidate_scores(
                packed,
                weights[begin:end],
                cache,
                page_table[begin:end],
                visible,
                candidates,
                _api(packed),
                skip_score_topk=topk,
                index_metadata=index_metadata,
            )
            indices = _native_indices(logits, ends, topk)
            selected = finish_selection(
                indices, logits, ends, normalized, capacity, 8, True
            )
            write_selection(selected, None, tuple(tensor[begin:end] for tensor in out))
            continue
        if dense:
            visible = visible.clamp(0, capacity).to(torch.int32)
            logits, prepared_blocks = _dense_scores(
                packed,
                keys,
                weights[begin:end],
                visible,
                page_table[begin:end],
                index_cache.shape[0],
                capacity,
                prepare_blocks,
            )
        else:
            logits, visible, prepared_blocks = _paged_scores(
                packed,
                cache,
                weights[begin:end],
                page_table[begin:end],
                visible,
                capacity,
                64,
                index_metadata,
                begin,
                end,
                prepare_blocks,
            )
        selected = (
            _select_topk_prepared(logits, visible, topk)
            if candidates is None
            else select_topk(logits, visible, candidates, topk, 8)
        )
        chosen = None
        if candidate_topk and prepared_blocks is None:
            chosen = select_candidates(logits, visible, candidate_topk, 8)
        elif prepared_blocks is not None:
            block_scores, block_ends = prepared_blocks
            block_indices = _native_indices(block_scores, block_ends, candidate_topk)
            chosen = finish_selection(
                block_indices,
                block_scores,
                block_ends,
                None,
                block_scores.shape[1],
                8,
                False,
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
# selection that follows runs on the in-tree CuTe DSL DeepSelect kernel, the
# upstream DeepSelect package being sm100-only.
# ---------------------------------------------------------------------------

_INDEX_ROW_BYTES = _LAYOUTS["index_v4"][3]
_INDEX_VALUE_BYTES = _LAYOUTS["index_v4"][2]


def is_hopper_indexer_available() -> bool:
    """Whether this platform scores FP8 index rows with DeepGEMM."""
    return platform.is_hopper and importlib.util.find_spec("deep_gemm") is not None


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
    # The selector compiles on first use, which cannot happen inside a capture:
    # compile every cluster variant of both selection capacities up front.
    row_select.warmup((_ROW_CAPACITY, _BLOCK_CAPACITY), device)


def _hopper_api(queries):
    enabled = pdl_enabled()
    _warmup_hopper_indexer(queries.shape[1], queries.device, enabled)
    if deep_gemm.get_pdl() != enabled:
        deep_gemm.set_pdl(enabled)
    return deep_gemm


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


def _hopper_dense_scores(
    queries, keys, weights, lengths, table, pages, capacity, *, clean
):
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
    return clean_logits(logits, ends, table, pages, capacity, 64) if clean else logits


def _hopper_paged_scores(
    queries, cache, weights, block_table, valid_lengths, capacity, *, clean
):
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
    if not clean:
        return logits.reshape(queries[0].shape[0], -1)[:, :capacity]
    return clean_logits(
        logits.reshape(queries[0].shape[0], -1)[:, :capacity],
        lengths,
        block_table,
        cache.shape[0],
        capacity,
        64,
    )


# V4.1 selects at most 512 rows and at most 2048 candidate blocks per query;
# each selection binds to one compiled survivor capacity so that warmup knows
# exactly which variants a capture may replay.
_ROW_CAPACITY = 512
_BLOCK_CAPACITY = 2048


def _select(scores, k, capacity, destination, lengths, candidates=None, ends=None):
    """Take the k best columns per row with the CuTe DSL DeepSelect kernel.

    ``ends`` bounds each row (default: the whole row); a row shorter than
    ``k`` is taken whole. ``candidates`` marks ``scores`` as
    candidate-compacted and maps the winning columns back to row ids. Ties at
    the k-th score break arbitrarily but deterministically; the op guarantees
    no particular tie order.
    """
    width = min(k, scores.shape[1])
    if width == 0:
        destination.fill_(-1)
        lengths.zero_()
        return
    if ends is None:
        ends = torch.full(
            (scores.shape[0],), scores.shape[1], dtype=torch.int32, device=scores.device
        )
    cluster_size = row_select.choose_cluster_size(
        scores.shape[0], scores.shape[1], capacity, platform.sm_count
    )
    indices, values = row_select.deepselect_topk(
        scores, ends, width, capacity=capacity, cluster_size=cluster_size
    )
    # Null blocks resolve to negative ids; their -inf score drops them below.
    _finish_topk(values, indices, destination, lengths, candidates)


def _native_select(
    logits, visible, candidates, topk, candidate_topk, out, scores, block_stats=None
):
    """Select rows, and blocks when this pass sources the candidate pool.

    ``scores`` is the candidate-compacted score matrix when it was produced
    directly, or None to gather it here from the dense ``logits``. A pass that
    sources candidates always needs the dense row, so it never supplies it.
    ``block_stats`` optionally supplies maxima and valid block counts already
    produced while cleaning logits; their unwritten tails are never selected.
    """
    rows, lengths, blocks, block_lengths = out
    if candidates is None:
        # Columns past a query's visible length are -inf; skipping them keeps
        # short contexts in a wide capacity cheap.
        _select(logits, topk, _ROW_CAPACITY, rows, lengths, ends=visible)
    else:
        if scores is None:
            scores = candidate_scores(logits, candidates)
        _select(scores, topk, _ROW_CAPACITY, rows, lengths, candidates)
    if candidate_topk:
        maxima, block_ends = (
            block_maxima_with_lengths(logits, visible)
            if block_stats is None
            else block_stats
        )
        _select(
            maxima,
            candidate_topk,
            _BLOCK_CAPACITY,
            blocks,
            block_lengths,
            ends=block_ends,
        )
    else:
        # Outputs may be uninitialized; a pass that sources no candidates still
        # owes the caller the empty pool every other solution writes.
        blocks.fill_(-1)
        block_lengths.zero_()


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
    index_metadata,
):
    """Score FP8 index rows on tensor cores, select with CuTe DSL DeepSelect."""
    del index_metadata
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
    # The logits kernels apply no query scale, so it rides in the weights.
    queries, folded = quantize_index_queries(index_q, weights)
    keys = None
    if dense:
        logical = torch.arange(capacity, dtype=torch.int64, device=index_q.device)
        pages = page_table[0, logical // 64]
        slots = (pages.to(torch.int64) * 64 + logical % 64).masked_fill(
            (pages < 0) | (pages >= index_cache.shape[0]), -1
        )
        keys = _gather_index_fp8(index_cache, slots)
    # A Reindex pass reads only its candidate pool, so scoring the whole
    # history and discarding the rest is pure waste. DeepGEMM has no sm90
    # sparse kernel, so the pool is scored by the CuTe DSL one. A pass that
    # also sources candidates still needs every row for the block maxima.
    sparse = (
        candidate_blocks is not None
        and not candidate_topk
        and not dense
        and sparse_index_scores_supported(queries, folded, page_table, candidate_blocks)
    )
    planes = None
    if sparse:
        values, scales = _index_planes(index_cache)
        planes = (values.view(torch.float8_e4m3fn), scales)
    for begin in range(0, n, tile):
        end = min(begin + tile, n)
        visible = visible_lens[begin:end].clamp(0, capacity).to(torch.int32)
        packed = (queries[begin:end],)
        if sparse:
            _native_select(
                None,
                visible,
                candidate_blocks[begin:end],
                topk,
                candidate_topk,
                tuple(tensor[begin:end] for tensor in out),
                sparse_index_scores(
                    queries[begin:end],
                    folded[begin:end],
                    planes[0],
                    planes[1],
                    page_table[begin:end],
                    visible,
                    candidate_blocks[begin:end],
                    pdl_enabled(),
                ),
            )
            continue
        # Candidate gathering can read past visible into a partial block. Keep
        # its fully initialized scores; only bounded selectors consume the
        # partially initialized output of the fused cleaner.
        bounded = bool(candidate_topk) and candidate_blocks is None
        if dense:
            logits = _hopper_dense_scores(
                packed,
                keys,
                folded[begin:end],
                visible,
                page_table[begin:end],
                index_cache.shape[0],
                capacity,
                clean=not bounded,
            )
        else:
            logits = _hopper_paged_scores(
                packed,
                index_cache,
                folded[begin:end],
                page_table[begin:end],
                visible,
                capacity,
                clean=not bounded,
            )
        block_stats = None
        if bounded:
            logits, maxima, block_ends = clean_block_maxima(
                logits, visible, page_table[begin:end], index_cache.shape[0]
            )
            block_stats = (maxima, block_ends)
        candidates = None if candidate_blocks is None else candidate_blocks[begin:end]
        _native_select(
            logits,
            visible,
            candidates,
            topk,
            candidate_topk,
            tuple(tensor[begin:end] for tensor in out),
            None,
            block_stats=block_stats,
        )
    return out


def _use_sparse_candidate_scores(cache, candidates, candidate_topk):
    """Native sparse reads require aligned pages and a candidate consumer."""
    return (
        candidates is not None
        and candidates.shape[1] > 0
        and candidate_topk == 0
        and cache.stride(0) % 512 == 0
        and cache.stride(0) < 2**31
        and cache.data_ptr() % 16 == 0
    )


def _candidate_topk(scores, topk):
    ends = torch.full(
        (scores.shape[0],), scores.shape[1], dtype=torch.int32, device=scores.device
    )
    cluster = row_select.choose_cluster_size(
        scores.shape[0], scores.shape[1], _ROW_CAPACITY, platform.sm_count
    )
    indices, values = row_select.deepselect_topk(
        scores.contiguous(),
        ends,
        min(topk, scores.shape[1]),
        capacity=_ROW_CAPACITY,
        cluster_size=cluster,
    )
    if indices.shape[1] < topk:
        padding = (0, topk - indices.shape[1])
        indices = torch.nn.functional.pad(indices, padding, value=-1)
        values = torch.nn.functional.pad(values, padding, value=-float("inf"))
    return indices.long(), values


@register_kernel(
    "attention",
    "dsv41_index_candidates",
    name="deep_gemm_hopper_dsv41_index_candidates",
    solution="deep_gemm",
    signatures=[format_signature(x=dense_tensor_format(torch.bfloat16))],
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(9, 0),
        vendors=frozenset({"nvidia"}),
    ),
    priority=Priority.SPECIALIZED,
)
def index_candidates(index_q, weights, index_cache, table, visible, candidates, topk):
    queries, folded = quantize_index_queries(index_q, weights)
    _hopper_api(queries)
    if candidates is None:
        from tokenspeed_kernel.ops.attention.dsa._triton.index_candidates import (
            compact_index_pages,
            gather_index_candidates,
        )

        requests = torch.arange(table.shape[0], device=table.device, dtype=torch.int64)
        pages, positions, lengths = compact_index_pages(table, requests, visible, 64)
        scores = _hopper_paged_scores(
            (queries,),
            index_cache,
            folded,
            pages,
            lengths[:, 0],
            table.shape[1] * 64,
            clean=True,
        )
        offsets, _ = _candidate_topk(scores, topk)
        logical, selected = gather_index_candidates(offsets, scores, positions, 64)
    else:
        values, scales = _index_planes(index_cache)
        scores = sparse_index_scores(
            queries,
            folded,
            values.view(torch.float8_e4m3fn),
            scales,
            table,
            visible,
            candidates,
            pdl_enabled(),
        )
        offsets, selected = _candidate_topk(scores, topk)
        logical = (
            candidates.gather(1, offsets.clamp_min(0) // 8).long() * 8 + offsets % 8
        )
    logical = logical.to(torch.int32).masked_fill(selected == -float("inf"), -1)
    return torch.stack((logical, selected.contiguous().view(torch.int32)), dim=-1)


@register_kernel(
    "attention",
    "dsv41_merge_index_candidates",
    name="deep_gemm_hopper_dsv41_merge_index_candidates",
    solution="deep_gemm",
    signatures=[format_signature(x=dense_tensor_format(torch.int32))],
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(9, 0),
        vendors=frozenset({"nvidia"}),
    ),
    priority=Priority.SPECIALIZED,
)
def merge_index_candidates(packed, rows, lengths):
    scores = packed[:, :, 1].contiguous().view(torch.float32)
    selected, values = _candidate_topk(scores, rows.shape[1])
    logical = packed[:, :, 0].gather(1, selected.clamp_min(0))
    _finish_topk(values, logical, rows, lengths)
    return rows, lengths
