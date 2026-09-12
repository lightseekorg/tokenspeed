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
from tokenspeed_kernel.ops.attention.deep_gemm.dsv4 import _mxfp4_cache_view
from tokenspeed_kernel.ops.attention.deep_select import (
    select_candidates,
    select_topk,
)
from tokenspeed_kernel.ops.attention.triton.dsv41 import (
    _index_topk_outputs,
    clean_logits,
    dense_ranges,
    gather_index_cache,
    pack_index_queries,
    safe_metadata,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@lru_cache(maxsize=1)
def is_native_indexer_available() -> bool:
    """Whether the existing DeepGEMM dependency provides packed MQA scoring."""
    try:
        from tokenspeed_kernel.thirdparty import deep_gemm  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


@lru_cache(maxsize=32)
def _warmup_indexer(heads: int, device: torch.device, enable_pdl: bool) -> None:
    """Compile packed MQA kernels before capture; retain no tensors."""
    from tokenspeed_kernel.thirdparty import deep_gemm

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up packed indexer kernels before graph capture")
    deep_gemm.set_pdl(enable_pdl)
    deep_gemm.warmup_mqa_logits(
        num_heads=heads,
        index_head_dim=128,
        cache_block_size=64,
        max_decode_tokens=32,
        device=device,
    )


def _api(queries):
    from tokenspeed_kernel.thirdparty import deep_gemm as api

    enabled = pdl_enabled()
    _warmup_indexer(queries[0].shape[1], queries[0].device, enabled)
    if api.get_pdl() != enabled:
        api.set_pdl(enabled)
    return api


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
        rows[begin:end].copy_(selected)
        lengths[begin:end].copy_((selected >= 0).sum(-1, dtype=torch.int32))
        if candidate_topk:
            chosen = select_candidates(logits, visible, candidate_topk, 8)
            # Candidate IDs have no score ordering contract. Publish positions
            # sorted with invalid IDs in the trailing suffix, as the public API.
            chosen = chosen.masked_fill(chosen < 0, torch.iinfo(torch.int32).max)
            chosen = chosen.sort(dim=-1).values
            chosen.masked_fill_(chosen == torch.iinfo(torch.int32).max, -1)
            blocks[begin:end].copy_(chosen)
            block_lengths[begin:end].copy_((chosen >= 0).sum(-1, dtype=torch.int32))
        else:
            block_lengths[begin:end].zero_()
    return out
