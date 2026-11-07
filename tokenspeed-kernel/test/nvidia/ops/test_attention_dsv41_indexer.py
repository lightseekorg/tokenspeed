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

"""Deferred indexer preprocessing through native and portable selection."""

from unittest.mock import Mock

import pytest
import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.numerics.torch.attention.dsv41 import (
    torch_dsv41_sparse_scores,
)
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41 import _metadata as dsv41_metadata
from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm
from tokenspeed_kernel.ops.attention.dsv41._deep_gemm.indexer import (
    _sparse_candidate_scores,
    clean_logits_and_block_max,
    gather_index_pages,
)
from tokenspeed_kernel.ops.attention.dsv41.deep_select import (
    _native_indices,
    _select_topk_prepared,
    select_candidates,
    select_topk,
)
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _clean_logits,
    _mask_invalid_pages,
    clean_logits,
    clean_logits_for_selection,
    finish_selection,
    mask_invalid_pages,
    pack_index_inputs,
    pack_index_queries,
    prepare_scores,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.deep_select import deep_select_alignment
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform().is_blackwell
    or current_platform().arch_version.minor > 3,
    reason="requires SM100-SM103 native indexer",
)


def _sparse_inputs(tokens, count, strided):
    torch.manual_seed(917 + tokens + count)
    capacity = 65536
    pages = tokens * (capacity // 64) + 1
    storage = torch.empty(
        (pages, 9216 if strided else 4608), dtype=torch.uint8, device="cuda"
    )
    cache = storage[:, :4352]
    cache[:, :4096].random_(0, 256)
    cache[:, 4096:].random_(123, 127)
    q = torch.randn((tokens, 32, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((tokens, 32), device="cuda", dtype=torch.bfloat16)
    table = torch.arange(1, pages, device="cuda", dtype=torch.int32).reshape(tokens, -1)
    lengths = torch.full((tokens,), capacity - 3, device="cuda", dtype=torch.int32)
    candidates = torch.randint(
        -5, capacity // 8 + 5, (tokens, count), device="cuda", dtype=torch.int32
    )
    if tokens > 1:
        lengths[0] = 0
        lengths[1] = 81
    weights[-1] = -weights[-1].abs()
    table[:, 1] = -1
    table[:, 3] = pages + 7
    candidates[:, :5] = torch.tensor([0, 7, 7, -1, 8192], device="cuda")
    candidates[:, -1] = (lengths - 1).clamp_min(0) // 8
    if strided:

        def expand(tensor):
            out = torch.empty(
                (*tensor.shape[:-1], tensor.shape[-1] * 2),
                device="cuda",
                dtype=tensor.dtype,
            )[..., ::2]
            out.copy_(tensor)
            return out

        table, lengths, candidates, weights = map(
            expand, (table, lengths, candidates, weights)
        )
    queries = pack_index_queries(q)
    api = deep_gemm._api(queries)
    return queries, weights, cache, table, lengths, candidates, api


@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("candidate_pool", [False, True])
def test_deferred_query_preprocessing_matches_explicit_rope(broadcast, candidate_pool):
    from tokenspeed_kernel.ops.attention.dsv41.triton import rope_inplace

    _, weights, cache, table, lengths, candidates, _ = _sparse_inputs(4, 128, True)
    q = torch.randn(4, 32, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([-1, 1, 127, 255], device="cuda")
    angles = (
        torch.arange(256, device="cuda")[:, None]
        * torch.linspace(0.001, 1, 32, device="cuda")[None, :]
    )
    rotary = torch.cat((angles.cos(), angles.sin()), -1)
    if broadcast:
        table = table[-1:].expand(4, -1)
    paged_cache = cache.as_strided((cache.shape[0], 64, 68), (cache.stride(0), 68, 1))
    deferred = dsv41.IndexQueryInput(q, positions, rotary, 0.023456789)
    original = q.clone()
    chosen = candidates if candidate_pool else None

    def select(query, head_weights, solution):
        return dsv41.index_topk(
            query,
            head_weights,
            paged_cache,
            table,
            lengths,
            chosen,
            64,
            0,
            8,
            32,
            256,
            None,
            None,
            solution,
        )

    for solution in ("deep_gemm", "triton"):
        expected = select(
            rope_inplace(q.clone(), positions, rotary),
            weights * deferred.weight_scale,
            solution,
        )
        actual = select(deferred, weights, solution)
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
    torch.testing.assert_close(q, original, rtol=0, atol=0)


def _reference_sparse_scores(queries, weights, cache, table, lengths, candidates):
    """Independent address oracle plus DeepGEMM's documented BF16 head arithmetic."""
    return torch_dsv41_sparse_scores(
        queries,
        weights,
        cache,
        table,
        lengths,
        candidates,
        alignment=deep_select_alignment()[0] // 2,
    )


def _check_sparse_selection(selected, reference):
    """Allow equal-score Top-K ties while requiring unique, causal logical IDs."""
    values = reference.gather(1, selected.long().clamp_min(0)).masked_fill(
        selected < 0, -torch.inf
    )
    expected = reference.topk(min(selected.shape[1], reference.shape[1])).values
    torch.testing.assert_close(
        values.sort().values, expected.sort().values, rtol=0, atol=0
    )
    assert torch.equal((selected >= 0).sum(1), (expected > -torch.inf).sum(1))
    assert not ((selected[:, 1:] <= selected[:, :-1]) & (selected[:, 1:] >= 0)).any()


@pytest.mark.parametrize(
    "tokens,count,strided", [(1, 17, False), (4, 128, True), (32, 2048, False)]
)
def test_sparse_candidate_scores_match_bf16_reference(tokens, count, strided):
    args = _sparse_inputs(tokens, count, strided)
    queries, weights, cache, table, lengths, candidates, api = args
    before = candidates.clone()
    actual = _sparse_candidate_scores(*args, skip_score_topk=0)
    expected = _reference_sparse_scores(*args[:-1])
    for got, want in zip(actual, expected[:3], strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    torch.testing.assert_close(candidates, before, rtol=0, atol=0)
    scores, ends, normalized = actual
    selected = finish_selection(
        _native_indices(scores, ends, 512),
        scores,
        ends,
        normalized,
        table.shape[1] * 64,
        8,
        True,
    )
    _check_sparse_selection(selected, expected[3])


@pytest.mark.parametrize("malformed_candidates", [False, True])
def test_sparse_scoring_skip_uses_live_pool_count_in_one_graph(
    monkeypatch, malformed_candidates
):
    args = _sparse_inputs(4, 128, True)
    queries, weights, cache, table, lengths, candidates, api = args
    table.copy_(torch.arange(1, cache.shape[0], device="cuda").view_as(table))
    candidates.copy_(torch.arange(128, device="cuda").expand_as(candidates))
    if malformed_candidates:
        table[:, 1] = -1
        table[:, 3] = cache.shape[0]
        candidates[:, :5] = torch.tensor([9, 9, -1, 8192, 2], device="cuda")
    _sparse_candidate_scores(*args, skip_score_topk=0)
    contexts = []
    native_metadata = api.get_paged_sparse_mqa_logits_metadata

    def record_context(context, *rest):
        contexts.append(context)
        return native_metadata(context, *rest)

    monkeypatch.setattr(api, "get_paged_sparse_mqa_logits_metadata", record_context)

    def run():
        scores, ends, normalized = _sparse_candidate_scores(*args, skip_score_topk=512)
        selected = finish_selection(
            _native_indices(scores, ends, 512),
            scores,
            ends,
            normalized,
            table.shape[1] * 64,
            8,
            True,
        )
        return scores, ends, normalized, selected

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            scores, ends, normalized, selected = run()
    torch.cuda.current_stream().wait_stream(stream)
    for live_lengths in (
        [0, 0, 0, 0],
        [1, 8, 511, 512],
        [511, 512, 513, 1023],
        [513, 511, 0, 512],
    ):
        lengths.copy_(torch.tensor(live_lengths, device="cuda"))
        weights.normal_()
        graph.replay()
        expected_scores, expected_ends, expected_blocks, full = (
            _reference_sparse_scores(*args[:-1])
        )
        torch.testing.assert_close(ends, expected_ends, rtol=0, atol=0)
        torch.testing.assert_close(normalized, expected_blocks, rtol=0, atol=0)
        skip = expected_ends <= 512
        torch.testing.assert_close(
            contexts[-1],
            torch.where(skip, 0, (expected_ends + 7) // 8 * 8),
            rtol=0,
            atol=0,
        )
        valid = torch.arange(scores.shape[1], device="cuda")[None, :] < ends[:, None]
        expected_scores = torch.where(skip[:, None] & valid, 0, expected_scores)
        torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
        _check_sparse_selection(selected, full)


@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("query_chunk_size", [32, 256])
def test_reindex_chunked_graph_matches_sparse_reference(
    broadcast, query_chunk_size, monkeypatch
):
    """Small/partial query tiles use sparse scoring, including shared prefill tables."""
    _, weights, cache, table, lengths, candidates, api = _sparse_inputs(33, 2048, True)
    if broadcast:
        table = table[-1:].expand(33, -1)
    q = torch.randn(33, 32, 128, device="cuda", dtype=torch.bfloat16)
    paged_cache = cache.as_strided((cache.shape[0], 64, 68), (cache.stride(0), 68, 1))

    def forbid_full_scores(*args, **kwargs):
        raise AssertionError("Reindex must read candidates with native sparse scoring")

    monkeypatch.setattr(deep_gemm, "_paged_scores", forbid_full_scores)
    monkeypatch.setattr(deep_gemm, "_dense_scores", forbid_full_scores)
    monkeypatch.setattr(deep_gemm, "gather_index_pages", forbid_full_scores)

    native = api.fp8_fp4_paged_sparse_mqa_logits

    def checked_native(**kwargs):
        assert kwargs["kv_cache"].data_ptr() == cache.data_ptr()
        assert kwargs["kv_cache"].stride(0) == cache.stride(0)
        assert kwargs["weights"].dtype == torch.bfloat16
        return native(**kwargs)

    def run():
        return dsv41.index_topk(
            q,
            weights,
            paged_cache,
            table,
            lengths,
            candidates,
            512,
            0,
            8,
            query_chunk_size,
            256,
            None,
            None,
            solution="deep_gemm",
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        # Install after warmup, whose synthetic cache is intentionally separate.
        monkeypatch.setattr(api, "fp8_fp4_paged_sparse_mqa_logits", checked_native)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = run()
    torch.cuda.current_stream().wait_stream(stream)
    for live in (0, 1, 17, 33):
        q.normal_()
        weights.normal_()
        cache[:, :4096].random_(0, 256)
        cache[:, 4096:].random_(123, 127)
        table[0, 2] = 0 if live % 2 else -1
        lengths.fill_(-1)
        lengths[:live] = (
            torch.arange(live, device="cuda", dtype=torch.int32) * 1000 + 81
        )
        candidates.random_(-7, 8196)
        candidates[live:] = -1
        if live:
            candidates[:live, -1] = (lengths[:live] - 1) // 8
        original_candidates = candidates.clone()
        graph.replay()
        expected = _reference_sparse_scores(
            pack_index_queries(q), weights, cache, table, lengths, candidates
        )
        actual, actual_lengths, blocks, block_lengths = outputs
        _check_sparse_selection(actual, expected[3])
        torch.testing.assert_close(actual_lengths, (actual >= 0).sum(1).to(torch.int32))
        torch.testing.assert_close(candidates, original_candidates, rtol=0, atol=0)
        assert blocks.numel() == 0 and not block_lengths.any()


def test_reindex_unaligned_pages_keep_candidate_filtering(monkeypatch):
    queries, weights, cache, table, lengths, candidates, _ = _sparse_inputs(
        1, 128, False
    )
    cache = cache.contiguous().view(-1, 64, 68)
    q = torch.ones(1, 32, 128, device="cuda", dtype=torch.bfloat16)

    def forbid_sparse(*args, **kwargs):
        raise AssertionError("Unaligned pages must not enter the native sparse kernel")

    monkeypatch.setattr(deep_gemm, "_sparse_candidate_scores", forbid_sparse)
    ids, counts, _, _ = dsv41.index_topk(
        q,
        weights,
        cache,
        table,
        lengths,
        candidates,
        512,
        0,
        8,
        32,
        256,
        None,
        None,
        solution="deep_gemm",
    )
    selected = ids[0, : counts[0]].long()
    assert selected.numel() > 0
    assert torch.isin(selected // 8, candidates[0]).all()
    assert (selected < lengths[0]).all()
    assert (selected[1:] > selected[:-1]).all()
    pages = table[0, selected // 64]
    assert ((pages >= 0) & (pages < cache.shape[0])).all()


def _check_selection(scores, ends, selected):
    width = scores.shape[1]
    k = selected.shape[1]
    positions = torch.arange(width, device=scores.device)[None, :]
    masked = scores.masked_fill(positions >= ends[:, None], -torch.inf)
    expected = masked.topk(min(k, width), dim=1).values.sort().values
    actual = (
        masked.gather(1, selected.long().clamp_min(0))
        .masked_fill(selected < 0, -torch.inf)
        .sort()
        .values
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    valid_count = torch.minimum(
        (masked > -torch.inf).sum(1), torch.full_like(ends, k, dtype=torch.int64)
    )
    torch.testing.assert_close((selected >= 0).sum(1), valid_count, rtol=0, atol=0)
    assert not ((selected[:, 1:] <= selected[:, :-1]) & (selected[:, 1:] >= 0)).any()
    assert not ((selected >= ends[:, None]) & (selected >= 0)).any()


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("prepare_blocks", [False, True])
def test_full_score_bf16_producer_masking_and_selection(paged, prepare_blocks):
    queries, weights, cache, table, lengths, _, api = _sparse_inputs(4, 128, True)
    weights = weights.float().contiguous()
    capacity = table.shape[1] * 64
    ends = lengths.clamp(0, capacity).to(torch.int32)
    if paged:
        safe_table = table.masked_fill((table < 0) | (table >= cache.shape[0]), 0)
        context_lens = ends[:, None]
        schedule = api.get_paged_mqa_logits_metadata(
            context_lens, 64, api.get_num_sms()
        )
        reference = api.fp8_fp4_paged_mqa_logits(
            q=(queries[0].view(torch.int8).unsqueeze(1), queries[1].unsqueeze(1)),
            kv_cache=cache.as_strided(
                (cache.shape[0], 64, 1, 68), (cache.stride(0), 68, 68, 1)
            ),
            weights=weights,
            context_lens=context_lens,
            block_table=safe_table,
            schedule_meta=schedule,
            max_context_len=capacity,
            clean_logits=False,
            logits_dtype=torch.float32,
        )
        scores, actual_ends, blocks = deep_gemm._paged_scores(
            queries,
            cache,
            weights,
            table,
            lengths,
            capacity,
            64,
            None,
            0,
            4,
            prepare_blocks,
        )
        torch.testing.assert_close(actual_ends, ends, rtol=0, atol=0)
    else:
        table = table[-1:].expand(4, -1)
        keys = gather_index_pages(cache, table[0], 64)
        reference = api.fp8_fp4_mqa_logits(
            q=(queries[0].view(torch.int8), queries[1]),
            kv=(keys[0].view(torch.int8), keys[1]),
            weights=weights,
            cu_seq_len_k_start=torch.zeros_like(ends),
            cu_seq_len_k_end=ends,
            clean_logits=False,
            max_seqlen_k=capacity,
            logits_dtype=torch.float32,
        )
        # Dense scoring masks the MQA logits in place whether or not block
        # candidates follow; select_candidates then reads the same view.
        scores = deep_gemm._dense_scores(
            queries,
            keys,
            weights,
            ends,
            cache.shape[0],
            capacity,
            torch.zeros_like(ends),
            table[0],
        )
        blocks = None
    assert scores.dtype == torch.bfloat16
    position = torch.arange(capacity, device="cuda")[None, :]
    physical_pages = table[:, position[0] // 64]
    valid = (
        (position < ends[:, None])
        & (physical_pages >= 0)
        & (physical_pages < cache.shape[0])
    )
    expected = (
        reference[:, :capacity].to(torch.bfloat16).masked_fill(~valid, -torch.inf)
    )
    if prepare_blocks and paged:
        torch.testing.assert_close(scores, expected, rtol=0, atol=0)
    else:
        # Selection reads [0, end) only, so later columns are unspecified.
        # Invalid pages inside that range must already be -inf.
        within = position < ends[:, None]
        torch.testing.assert_close(
            scores[:, :capacity][within], expected[within], rtol=0, atol=0
        )
        selected = _select_topk_prepared(scores, ends, 512)
        _check_selection(expected, ends, selected)
        if not prepare_blocks:
            return
    if blocks is not None:
        assert blocks[0].dtype == torch.float32
    expected_blocks = expected.float().reshape(4, -1, 8).amax(dim=-1)
    block_ends = (ends + 7) // 8
    latest = (block_ends - 1).clamp_min(0)
    rows = torch.arange(4, device="cuda")
    pin = (block_ends > 0) & (expected_blocks[rows, latest] > -torch.inf)
    expected_blocks[rows[pin], latest[pin]] = torch.inf
    if blocks is None:
        chosen = select_candidates(scores, ends, 64, 8)
    else:
        torch.testing.assert_close(blocks[0], expected_blocks, rtol=0, atol=0)
        torch.testing.assert_close(blocks[1], block_ends, rtol=0, atol=0)
        chosen = finish_selection(
            _native_indices(*blocks, 64),
            *blocks,
            None,
            expected_blocks.shape[1],
            8,
            False,
        )
    selected_block_scores = expected_blocks.gather(1, chosen.long().clamp_min(0))
    selected_block_scores.masked_fill_(chosen < 0, -torch.inf)
    torch.testing.assert_close(
        selected_block_scores.sort().values,
        expected_blocks.topk(64).values.sort().values,
        rtol=0,
        atol=0,
    )
    selected = _select_topk_prepared(scores, ends, 512)
    _check_selection(expected, ends, selected)


def test_mask_invalid_pages_in_place_without_recompiling():
    pages = 16

    def run(histories, rows_per_history, width, stride):
        history_table = torch.randint(
            -1, pages + 2, (histories, width), device="cuda", dtype=torch.int32
        )
        rows = histories * rows_per_history
        history = torch.arange(histories, device="cuda").repeat_interleave(
            rows_per_history
        )
        starts = (history * width * 64).to(torch.int32)
        lengths = torch.randint(0, width * 64 + 1, (rows,), device="cuda")
        lengths = lengths.to(torch.int32)
        storage = torch.randn((rows, stride), device="cuda", dtype=torch.bfloat16)
        values = storage[:, : width * 64]
        before = storage.clone()
        position = torch.arange(width * 64, device="cuda")[None, :]
        page = history_table[history][:, position[0] // 64]
        hole = (position < lengths[:, None]) & ((page < 0) | (page >= pages))
        out = mask_invalid_pages(
            values, lengths, history_table.reshape(-1), starts, pages, 64
        )
        assert out.data_ptr() == values.data_ptr()
        torch.testing.assert_close(
            values,
            before[:, : width * 64].masked_fill(hole, -torch.inf),
            rtol=0,
            atol=0,
        )
        # Padding beyond the logical width is never written.
        assert torch.equal(storage[:, width * 64 :], before[:, width * 64 :])

    run(1, 3, 5, 5 * 64 + 32)
    # A single history may be a strided page-table row.
    table = torch.randint(-1, pages + 2, (5, 2), device="cuda", dtype=torch.int32)
    lengths = torch.full((3,), 5 * 64, device="cuda", dtype=torch.int32)
    values = torch.zeros((3, 5 * 64), device="cuda", dtype=torch.bfloat16)
    mask_invalid_pages(
        values, lengths, table[:, 1], torch.zeros_like(lengths), pages, 64
    )
    hole = ((table[:, 1] < 0) | (table[:, 1] >= pages)).repeat_interleave(64)
    assert torch.equal(torch.isinf(values), hole[None, :].expand(3, -1))
    with assert_no_triton_compile(_mask_invalid_pages):
        for histories, rows, width in ((3, 17, 70), (1, 1, 1), (2, 33, 600)):
            run(histories, rows, width, width * 64 + 64)


@pytest.mark.parametrize("history_rows", [(5,), (2, 3), (1, 1, 3)])
def test_packed_dense_histories_match_paged_scoring(history_rows):
    _, _, cache, pool_table, _, _, _ = _sparse_inputs(len(history_rows), 8, False)
    paged = cache.as_strided((cache.shape[0], 64, 68), (cache.stride(0), 68, 1))
    tokens = sum(history_rows)
    # Rows of one run share a history, as a batched prefill window does.
    table = torch.cat(
        [pool_table[i : i + 1].expand(rows, -1) for i, rows in enumerate(history_rows)]
    )
    lengths = torch.randint(
        1, table.shape[1] * 64, (tokens,), device="cuda", dtype=torch.int32
    )
    q = torch.randn((tokens, 32, 128), device="cuda", dtype=torch.bfloat16)
    head_weights = torch.randn((tokens, 32), device="cuda", dtype=torch.bfloat16)

    def run(rows):
        return dsv41.index_topk(
            q,
            head_weights,
            paged,
            table,
            lengths,
            None,
            32,
            0,
            8,
            tokens,
            256,
            None,
            None,
            solution="deep_gemm",
            index_metadata=dsv41.new_index_metadata(
                table, lengths, None, history_rows=rows
            ),
        )[:2]

    paged_ids, paged_counts = run(None)
    dense_ids, dense_counts = run(history_rows)
    torch.testing.assert_close(dense_counts, paged_counts, rtol=0, atol=0)
    for row in range(tokens):
        assert set(dense_ids[row].tolist()) == set(paged_ids[row].tolist())


def test_dense_scoring_rejects_windows_over_the_logits_budget(monkeypatch):
    _, _, cache, pool_table, _, _, _ = _sparse_inputs(1, 8, False)
    paged = cache.as_strided((cache.shape[0], 64, 68), (cache.stride(0), 68, 1))
    table = pool_table[:1].expand(4, -1)
    lengths = torch.full((4,), 64, device="cuda", dtype=torch.int32)
    q = torch.randn((4, 32, 128), device="cuda", dtype=torch.bfloat16)
    head_weights = torch.randn((4, 32), device="cuda", dtype=torch.bfloat16)
    # Three rows fit the budget; the runtime, not scoring, splits the window.
    capacity = table.shape[1] * 64
    monkeypatch.setattr(dsv41_metadata, "INDEX_LOGIT_BYTES", 3 * capacity * 2)
    with pytest.raises(ValueError, match="TOKENSPEED_PREFILL_INDEX_LOGITS_MB"):
        dsv41.index_topk(
            q,
            head_weights,
            paged,
            table,
            lengths,
            None,
            32,
            0,
            8,
            4,
            256,
            None,
            None,
            solution="deep_gemm",
            index_metadata=dsv41.new_index_metadata(
                table, lengths, None, history_rows=(4,)
            ),
        )


@pytest.mark.parametrize("candidate_topk", [0, 32])
def test_packed_dense_batch_beyond_int32_logit_offsets(candidate_topk):
    # Two request histories in one dense call; the last BF16 row starts beyond
    # 2**31 elements. Physical storage and live lengths stay small.
    history_rows = (1024, 1025)
    tokens = sum(history_rows)
    capacity = (1 << 20) + 64
    assert (tokens - 1) * capacity >= 2**31
    cache = torch.empty((2, 4352), device="cuda", dtype=torch.uint8)
    cache[:, :4096].random_(0, 256)
    cache[:, 4096:].random_(123, 127)
    paged_cache = cache.view(2, 64, 68)
    pages = torch.arange(capacity // 64, device="cuda", dtype=torch.int32)
    histories = torch.stack([pages.remainder(2), (pages + 1).remainder(2)])
    # The second history has an invalid page inside every live range.
    histories[1, 1] = -1
    table = torch.cat(
        [histories[i : i + 1].expand(rows, -1) for i, rows in enumerate(history_rows)]
    )
    lengths = torch.tensor(
        ([1, 7, 64, 65, 127, 128, 129, 191, 192] * ((tokens + 8) // 9))[:tokens],
        device="cuda",
        dtype=torch.int32,
    )
    q = torch.randn((tokens, 32, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((tokens, 32), device="cuda", dtype=torch.bfloat16)

    def run(rows):
        return dsv41.index_topk(
            q,
            weights,
            paged_cache,
            table,
            lengths,
            None,
            32,
            candidate_topk,
            8,
            tokens,
            256,
            None,
            None,
            solution="deep_gemm",
            index_metadata=dsv41.new_index_metadata(
                table, lengths, None, history_rows=rows
            ),
        )

    paged = run(None)
    dense = run(history_rows)
    torch.testing.assert_close(dense[1], paged[1], rtol=0, atol=0)
    torch.testing.assert_close(dense[3], paged[3], rtol=0, atol=0)
    for row in range(tokens):
        assert set(dense[0][row].tolist()) == set(paged[0][row].tolist()), row
        if candidate_topk:
            assert set(dense[2][row].tolist()) == set(paged[2][row].tolist()), row
    # Invalid page 1 (columns 64..127) is never selected.
    second = dense[0][history_rows[0] :]
    assert not ((second >= 64) & (second < 128)).any()


@pytest.mark.parametrize("tokens", [33, 2049])
def test_full_paged_query_tiles_match_with_large_capacity_in_graph(tokens):
    # With 2049 rows the final BF16 row starts beyond 2**31 elements. Keep
    # physical storage and live histories small while exercising that address.
    capacity = (1 << 20) + 64
    page_count = capacity // 64
    cache = torch.empty((2, 4352), device="cuda", dtype=torch.uint8)
    cache[:, :4096].random_(0, 256)
    cache[:, 4096:].random_(123, 127)
    table = torch.arange(page_count, device="cuda", dtype=torch.int32)[None, :]
    table = table.remainder(2).expand(tokens, -1).clone()
    lengths = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    q = torch.randn((tokens, 32, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((tokens, 32), device="cuda", dtype=torch.bfloat16)
    paged_cache = cache.view(2, 64, 68)

    def run(query_chunk_size):
        return dsv41.index_topk(
            q,
            weights,
            paged_cache,
            table,
            lengths,
            None,
            32,
            32,
            8,
            query_chunk_size,
            256,
            None,
            None,
            solution="deep_gemm",
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs = []
    with torch.cuda.stream(stream):
        for query_chunk_size in ((1, tokens) if tokens == 33 else (tokens,)):
            run(query_chunk_size)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = run(query_chunk_size)
            graphs.append((graph, outputs))
    torch.cuda.current_stream().wait_stream(stream)

    for live_lengths, invalid_first_page in (
        (([0, 1, 7, 8, 63, 64, 65, 127, 128] * ((tokens + 8) // 9))[:tokens], False),
        (([128, 127, 65, 64, 63, 8, 7, 1, 0] * ((tokens + 8) // 9))[:tokens], True),
    ):
        q.normal_()
        weights.normal_()
        lengths.copy_(torch.tensor(live_lengths, device="cuda", dtype=torch.int32))
        table[:, 0].copy_(torch.arange(tokens, device="cuda") % 2)
        table[:, 1].copy_((torch.arange(tokens, device="cuda") + 1) % 2)
        if invalid_first_page:
            table[::3, 0] = -1
            table[1::3, 1] = cache.shape[0]
        packed_q, packed_weights = pack_index_inputs(q, weights, torch.float32)
        api = deep_gemm._api(packed_q)
        reference_ends = lengths.clamp(0, 128).to(torch.int32)
        context_lens = reference_ends[:, None]
        safe_table = table[:, :2].masked_fill(
            (table[:, :2] < 0) | (table[:, :2] >= cache.shape[0]), 0
        )
        schedule = api.get_paged_mqa_logits_metadata(
            context_lens, 64, api.get_num_sms()
        )
        reference = api.fp8_fp4_paged_mqa_logits(
            q=(
                packed_q[0].view(torch.int8).unsqueeze(1),
                packed_q[1].unsqueeze(1),
            ),
            kv_cache=cache.view(2, 64, 1, 68),
            weights=packed_weights,
            context_lens=context_lens,
            block_table=safe_table,
            schedule_meta=schedule,
            max_context_len=128,
            clean_logits=False,
            logits_dtype=torch.float32,
        )
        positions = torch.arange(128, device="cuda")[None, :]
        physical_pages = table[:, :2][:, positions[0] // 64]
        valid = (
            (positions < reference_ends[:, None])
            & (physical_pages >= 0)
            & (physical_pages < cache.shape[0])
        )
        expected = reference[:, :128].to(torch.bfloat16).masked_fill(~valid, -torch.inf)

        for graph, outputs in graphs:
            graph.replay()
            rows, row_lengths, blocks, block_lengths = outputs
            _check_selection(expected, reference_ends, rows)
            torch.testing.assert_close(
                row_lengths, (rows >= 0).sum(1).to(torch.int32), rtol=0, atol=0
            )

            block_scores = expected.float().reshape(tokens, 16, 8).amax(-1)
            block_scores = torch.cat(
                (
                    block_scores,
                    torch.full((tokens, 16), -torch.inf, device="cuda"),
                ),
                dim=1,
            )
            block_ends = (reference_ends + 7) // 8
            latest = (block_ends - 1).clamp_min(0)
            rows_idx = torch.arange(tokens, device="cuda")
            pin = (block_ends > 0) & (block_scores[rows_idx, latest] > -torch.inf)
            block_scores[rows_idx[pin], latest[pin]] = torch.inf
            selected_scores = block_scores.gather(1, blocks.long().clamp_min(0))
            selected_scores.masked_fill_(blocks < 0, -torch.inf)
            expected_block_scores = block_scores.topk(32, dim=1).values
            torch.testing.assert_close(
                selected_scores.sort().values,
                expected_block_scores.sort().values,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                block_lengths, (blocks >= 0).sum(1).to(torch.int32), rtol=0, atol=0
            )


@pytest.mark.parametrize("count", [128, 513])
def test_bf16_candidate_gather_selection_matches_score_multiset(count):
    torch.manual_seed(317)
    width = 4103
    scores = torch.randn(4, width, device="cuda", dtype=torch.bfloat16)
    ends = torch.tensor([0, 1, 1330, width], device="cuda", dtype=torch.int32)
    positions = torch.arange(width, device="cuda")
    scores.masked_fill_(positions[None, :] >= ends[:, None], torch.nan)
    candidates = torch.randint(-3, 516, (4, count), device="cuda", dtype=torch.int32)
    candidates[:, :5] = torch.tensor([0, 0, -1, 512, 9999], device="cuda")
    candidates[:, -1] = (ends - 1).clamp_min(0) // 8
    reference = scores.clone()
    for row in range(4):
        valid = torch.isin(positions // 8, candidates[row]) & (positions < ends[row])
        reference[row].masked_fill_(~valid, -torch.inf)
    selected = select_topk(scores, ends, candidates, 512, 8)
    _check_selection(reference, ends, selected)


def test_bf16_prepared_topk_graph_refresh_ignores_poisoned_future():
    original = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
    scores = original.clone()
    ends = torch.full((4,), 4096, device="cuda", dtype=torch.int32)
    positions = torch.arange(4096, device="cuda")[None, :]
    _select_topk_prepared(scores, ends, 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _select_topk_prepared(scores, ends, 512)
    for values in ([0, 1, 513, 4095], [4096, 0, 1025, 1024]):
        ends.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
        scores.copy_(original).masked_fill_(positions >= ends[:, None], torch.nan)
        graph.replay()
        _check_selection(scores, ends, selected)


@pytest.mark.parametrize("width", [16384, 16320])
@pytest.mark.parametrize("topk", [1, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prepared_ends_topk_layouts_and_poisoned_future(width, topk, dtype):
    torch.manual_seed(317)
    ends = torch.tensor(
        [0, 1, 511, 512, 513, 1023, 1024, 1025, width - 1, width],
        dtype=torch.int32,
        device="cuda",
    )
    scores = torch.randn(ends.numel(), width, device="cuda", dtype=dtype)
    scores[:, ::17] = -torch.inf
    scores.masked_fill_(
        torch.arange(width, device="cuda")[None, :] >= ends[:, None], torch.nan
    )
    original_scores, original_ends = scores.clone(), ends.clone()
    actual = _select_topk_prepared(scores, ends, topk)
    _check_selection(scores, ends, actual)
    torch.testing.assert_close(scores, original_scores, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(ends, original_ends, rtol=0, atol=0)


@pytest.mark.parametrize(
    "capacity,paged,alignment",
    [
        (1, False, 16),
        (63, True, 2048),
        (64, "broadcast", 16),
        (1025, True, 16),
        (16384, False, 2048),
        (16384, "broadcast", 2048),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_clean_block_max_matches_separate_full_buffers(
    capacity, paged, alignment, dtype
):
    torch.manual_seed(193)
    values = torch.randn(4, capacity + 7, device="cuda", dtype=dtype)[:, :capacity]
    values.sub_(10)
    lengths = torch.tensor([-1, 0, min(67, capacity), capacity + 3], device="cuda")
    lengths_storage = torch.empty(8, dtype=torch.int64, device="cuda")
    lengths_storage[::2] = lengths
    lengths = lengths_storage[::2]
    columns = (capacity + 63) // 64
    table = (
        torch.arange(columns, device="cuda", dtype=torch.int32)[None, :]
        .expand(4, -1)
        .clone()
    )
    table[0].fill_(-1)
    table[2, -1] = columns + 10
    if columns > 1:
        table[3, 1] = -1
    if not paged:
        table = None
    elif paged == "broadcast":
        table = table[3:4].expand(4, -1)
    expected_clean = clean_logits(values, lengths, table, columns, capacity, 64)
    expected_blocks, expected_ends = prepare_scores(
        expected_clean, lengths, 8, alignment // 4, True
    )
    cleaned, blocks, ends = clean_logits_and_block_max(
        values, lengths, table, columns, capacity, 64, alignment
    )
    torch.testing.assert_close(cleaned, expected_clean, rtol=0, atol=0)
    torch.testing.assert_close(blocks, expected_blocks, rtol=0, atol=0)
    torch.testing.assert_close(ends, expected_ends, rtol=0, atol=0)
    assert cleaned.is_contiguous() and blocks.is_contiguous() and ends.is_contiguous()
    assert cleaned.dtype == dtype and blocks.dtype == torch.float32
    assert blocks.stride(0) * blocks.element_size() % alignment == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_clean_block_max_replay_changes_lengths_pages_and_scores(dtype):
    capacity = 16384
    values = torch.randn(4, capacity, device="cuda", dtype=dtype)
    lengths = torch.zeros(4, dtype=torch.int32, device="cuda")
    table = torch.zeros(4, capacity // 64, dtype=torch.int32, device="cuda")
    clean_logits_and_block_max(values, lengths, table, 2, capacity, 64, 2048)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = clean_logits_and_block_max(
            values, lengths, table, 2, capacity, 64, 2048
        )
    for current in ([0, 0, 0, 0], [1, 65, 1025, capacity], [capacity, 0, 17, 0]):
        lengths.copy_(torch.tensor(current, dtype=torch.int32, device="cuda"))
        values.normal_().sub_(10)
        table[:, :3] = -1
        table[:, 3:] = 1
        for output in outputs:
            output.fill_(123456)
        graph.replay()
        expected_clean = clean_logits(values, lengths, table, 2, capacity, 64)
        expected_blocks, expected_ends = prepare_scores(
            expected_clean, lengths, 8, 512, True
        )
        for actual, expected in zip(
            outputs, (expected_clean, expected_blocks, expected_ends)
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _live_logits_inputs():
    capacity = 4160
    values = -torch.arange(
        1, 2 * 4 * capacity + 1, device="cuda", dtype=torch.float32
    ).reshape(8, capacity)[::2]
    table = torch.arange(capacity // 64 * 2, device="cuda", dtype=torch.int64)[
        ::2
    ].repeat(4, 1)
    table[:, 1::3] = -1
    table[:, 2::5] = 99999
    lengths = torch.zeros(8, device="cuda", dtype=torch.int64)[::2]
    return values, lengths, table, capacity


@pytest.mark.parametrize("poison", [float("nan"), 1e30])
def test_live_logits_masks_prefetched_tail_for_selection(poison):
    values, lengths, table, capacity = _live_logits_inputs()
    output = torch.empty_like(values)
    candidates = torch.tensor(
        [8, 0, 8, -1, 16, 64, 2, 99999], device="cuda", dtype=torch.int32
    ).expand(4, -1)

    def select(scores):
        return (
            select_topk(scores, lengths, None, 512, 8),
            select_topk(scores, lengths, candidates, 512, 8),
            select_candidates(scores, lengths, 128, 8),
        )

    def run():
        _clean_logits[(4, triton.cdiv(capacity, 1024))](
            values,
            lengths,
            table,
            output,
            values.stride(0),
            lengths.stride(0),
            *table.stride(),
            table.shape[1],
            70,
            capacity,
            output.stride(0),
            64,
            True,
            True,
            1024,
            num_warps=4,
        )
        return select(output)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for visible in ([-7, 0, 1, 1023], [1024, 1025, 4160, 9000], [4097, 513, 512, 0]):
        lengths.copy_(torch.tensor(visible, device="cuda"))
        values.normal_().sub_(1000)
        table[:, 0] = 0 if visible[0] < 0 else -1
        output.fill_(poison)
        graph.replay()
        expected = clean_logits(values, lengths, table, 70, capacity, 64)
        written = ((lengths.clamp(0, capacity) + 1023) // 1024) * 1024
        live = torch.arange(capacity, device="cuda")[None, :] < written[:, None]
        torch.testing.assert_close(output[live], expected[live], rtol=0, atol=0)
        for actual, wanted in zip(result, select(expected), strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


def test_aligned_clean_logits_for_selection_graph_replay():
    capacity, page_size = 4163, 64
    pages = triton.cdiv(capacity, page_size)
    alignment = deep_select_alignment()[0] // 4
    values = torch.randn((4, capacity), device="cuda")
    lengths = torch.tensor([0, 63, 64, 4097], device="cuda", dtype=torch.int32)
    table = torch.arange(pages, device="cuda", dtype=torch.int32).repeat(4, 1)
    table[:, 1::3] = -1
    table[:, 2::5] = pages + 3

    def run():
        scores = clean_logits_for_selection(
            values, lengths, table, pages, capacity, page_size, alignment
        )
        return scores, select_topk(scores, lengths, None, 512, 1)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    expected_width = triton.cdiv(capacity, alignment) * alignment
    assert actual[0].shape == (4, expected_width)

    for visible in ([0, 1, 63, 64], [4097, 4163, 1024, 0], [17, 65, 1025, 4163]):
        lengths.copy_(torch.tensor(visible, device="cuda", dtype=torch.int32))
        values.normal_().sub_(1000)
        table[:, 1::3] = -1 if visible[0] == 0 else 1
        actual[0].fill_(float("nan"))
        actual[1].fill_(123456)
        graph.replay()

        expected_scores = clean_logits(
            values, lengths, table, pages, capacity, page_size
        )
        columns = torch.arange(capacity, device="cuda")[None, :]
        initialized = columns < ((lengths[:, None] + 1023) // 1024) * 1024
        torch.testing.assert_close(
            actual[0][:, :capacity][initialized],
            expected_scores[initialized],
            rtol=0,
            atol=0,
        )
        if visible[3] >= capacity:
            torch.testing.assert_close(
                actual[0][3, capacity:],
                torch.full_like(actual[0][3, capacity:], -float("inf")),
                rtol=0,
                atol=0,
            )
        _check_selection(expected_scores, lengths, actual[1])


@pytest.mark.parametrize("changed", ["candidates", "table", "lengths", "topk", "pages"])
def test_sparse_metadata_reuses_only_matching_input_geometry(monkeypatch, changed):
    args = list(_sparse_inputs(4, 128, True))
    _sparse_candidate_scores(*args, skip_score_topk=0)
    api = args[-1]
    schedule = Mock(wraps=api.get_paged_sparse_mqa_logits_metadata)
    monkeypatch.setattr(api, "get_paged_sparse_mqa_logits_metadata", schedule)
    metadata = dsv41.new_index_metadata(args[3], args[4], None, history_rows=None)
    baseline = _sparse_candidate_scores(
        *args, skip_score_topk=0, index_metadata=metadata
    )
    assert schedule.call_count == 1
    # New views of the same immutable candidate pool share the producer.
    args[5] = args[5].view_as(args[5])
    args[1].neg_()
    reused = _sparse_candidate_scores(*args, skip_score_topk=0, index_metadata=metadata)
    assert schedule.call_count == 1
    assert reused[1].data_ptr() == baseline[1].data_ptr()
    assert reused[2].data_ptr() == baseline[2].data_ptr()
    topk = 0
    if changed == "candidates":
        args[5] = args[5].clone()
        args[5][:, 0] = 10
    elif changed == "table":
        args[3] = args[3].clone()
    elif changed == "lengths":
        args[4] = args[4].clone()
    elif changed == "topk":
        topk = 512
    else:
        args[2] = args[2][:-1]
    _sparse_candidate_scores(*args, skip_score_topk=topk, index_metadata=metadata)
    assert schedule.call_count == 2


def test_sparse_metadata_reuse_across_streams_and_graph_replay(monkeypatch):
    args = _sparse_inputs(4, 128, True)
    queries, weights, cache, table, lengths, candidates, api = args
    _sparse_candidate_scores(*args, skip_score_topk=0)
    second_weights = -weights.contiguous()
    second_args = (queries, second_weights, cache, table, lengths, candidates, api)
    schedule = Mock(wraps=api.get_paged_sparse_mqa_logits_metadata)
    monkeypatch.setattr(api, "get_paged_sparse_mqa_logits_metadata", schedule)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    current = torch.cuda.current_stream()
    producer.wait_stream(current)
    consumer.wait_stream(current)

    def run():
        metadata = dsv41.new_index_metadata(table, lengths, None, history_rows=None)
        first = _sparse_candidate_scores(
            *args, skip_score_topk=0, index_metadata=metadata
        )
        with torch.cuda.stream(consumer):
            second = _sparse_candidate_scores(
                *second_args, skip_score_topk=0, index_metadata=metadata
            )
        producer.wait_stream(consumer)
        assert first[1].data_ptr() == second[1].data_ptr()
        return first, second

    with torch.cuda.stream(producer):
        run()
        schedule.reset_mock()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=producer):
            captured = run()
        assert schedule.call_count == 1
    current.wait_stream(producer)
    for end in (0, 511, 4097):
        lengths.fill_(end)
        candidates.random_(-1, max(1, end // 8 + 1))
        cache[:, :4096].random_(0, 256)
        weights.normal_()
        second_weights.normal_()
        producer.wait_stream(current)
        before = schedule.call_count
        with torch.cuda.stream(producer):
            eager = run()
        current.wait_stream(producer)
        assert schedule.call_count == before + 1
        graph.replay()
        assert schedule.call_count == before + 1
        for actual, replayed, inputs in zip(
            eager, captured, (args, second_args), strict=True
        ):
            expected = _reference_sparse_scores(*inputs[:-1])[:3]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(replayed, expected, rtol=0, atol=0)


def test_full_metadata_reuses_schedule_and_refreshes_live_inputs(monkeypatch):
    queries, weights, cache, table, lengths, _, api = _sparse_inputs(4, 128, False)
    head_weights = (weights.float(), -weights.float())
    capacity = table.shape[1] * 64
    schedule = Mock(wraps=api.get_paged_mqa_logits_metadata)
    monkeypatch.setattr(api, "get_paged_mqa_logits_metadata", schedule)

    def run(shared):
        metadata = (
            dsv41.new_index_metadata(table, lengths, None, history_rows=None)
            if shared
            else None
        )
        return tuple(
            deep_gemm._paged_scores(
                queries,
                cache,
                current_weights,
                table,
                lengths,
                capacity,
                64,
                metadata,
                0,
                table.shape[0],
                True,
            )
            for current_weights in head_weights
        )

    run(True)
    schedule.reset_mock()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(True)
    assert schedule.call_count == 1
    for visible in ([0, 81, 4097, capacity], [511, 0, 16384, 1025]):
        lengths.copy_(torch.tensor(visible, dtype=torch.int32, device="cuda"))
        table.copy_(table.roll(1, dims=1))
        for weights in head_weights:
            weights.normal_()
        expected = run(False)
        before = schedule.call_count
        eager = run(True)
        assert schedule.call_count == before + 1
        graph.replay()
        assert schedule.call_count == before + 1
        torch.testing.assert_close(eager, expected, rtol=0, atol=0)
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def _reference_clean(values, lengths, table, pages, capacity, width, page_size):
    """Torch oracle: valid logits, -inf for future columns and invalid pages."""
    columns = torch.arange(width, device=values.device)
    valid = (columns[None, :] < capacity) & (
        columns[None, :] < lengths.clamp(0, capacity)[:, None]
    )
    page_index = columns // page_size
    page = table[:, page_index.clamp(max=table.shape[1] - 1)]
    valid &= (page_index < table.shape[1])[None, :] & (page >= 0) & (page < pages)
    padded = torch.full(
        (values.shape[0], width), -float("inf"), dtype=values.dtype, device="cuda"
    )
    padded[:, :capacity] = values[:, :capacity]
    return torch.where(valid, padded, -float("inf"))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_wide_clean_logits_strides_each_row_over_its_tiles(dtype):
    # 256 rows of a 40-tile capacity launch 16 programs per row, each looping.
    rows, capacity, page_size = 256, 40 * 1024 - 37, 64
    pages = triton.cdiv(capacity, page_size)
    generator = torch.Generator(device="cuda").manual_seed(11)
    values = torch.randn((rows, capacity + 64), device="cuda", generator=generator).to(
        dtype
    )
    lengths = torch.randint(
        -3, capacity + 500, (rows,), device="cuda", generator=generator
    )
    lengths[:4] = torch.tensor([0, 1, capacity, 1024], device="cuda")
    table = torch.arange(pages, device="cuda", dtype=torch.int32).repeat(rows, 1)
    table[::3, 5::7] = -1
    table[1::4, 9::11] = pages + 2
    alignment = deep_select_alignment()[0] // values.element_size()
    width = triton.cdiv(capacity, alignment) * alignment
    expected = _reference_clean(
        values, lengths, table, pages, capacity, width, page_size
    )
    full = clean_logits(values, lengths, table, pages, capacity, page_size)
    torch.testing.assert_close(full, expected[:, :capacity], rtol=0, atol=0)
    live = clean_logits_for_selection(
        values, lengths, table, pages, capacity, page_size, alignment
    )
    assert live.shape == (rows, width)
    written = (lengths.clamp(0, capacity) + 1023) // 1024 * 1024
    initialized = torch.arange(width, device="cuda")[None, :] < written[:, None]
    torch.testing.assert_close(live[initialized], expected[initialized], rtol=0, atol=0)
    # Row counts and lengths change the grid and the loop, not the program.
    with assert_no_triton_compile(_clean_logits):
        for count in (1, 3, 64, 255):
            clean_logits_for_selection(
                values[:count],
                lengths[:count],
                table[:count],
                pages,
                capacity,
                page_size,
                alignment,
            )
            clean_logits(
                values[:count],
                lengths[:count],
                table[:count],
                pages,
                capacity,
                page_size,
            )
    torch.cuda.synchronize()


def _full_inputs():
    torch.manual_seed(229)
    capacity, page_size, pages = 4096, 64, 17
    q = torch.randn((4, 32, 128), device="cuda", dtype=torch.bfloat16)
    queries = pack_index_queries(q)
    weights = torch.randn((4, 32), device="cuda", dtype=torch.bfloat16)
    cache = torch.empty((pages, 4608), dtype=torch.uint8, device="cuda")[:, :4352]
    cache[:, :4096].random_(0, 256)
    cache[:, 4096:].fill_(126)
    table = (
        torch.arange(1, 65, device="cuda", dtype=torch.int32)
        .remainder_(pages - 1)
        .add_(1)
        .expand(4, -1)
        .clone()
    )
    table[:, 1] = -1
    table[:, 3] = pages + 2
    lengths = torch.tensor([0, 65, 129, 193], device="cuda", dtype=torch.int32)
    return queries, weights, cache, table, lengths, capacity, page_size


def _run_full_scores(
    queries, weights, cache, table, lengths, capacity, page_size, metadata
):
    return deep_gemm._paged_scores(
        queries,
        cache,
        weights,
        table,
        lengths,
        capacity,
        page_size,
        metadata,
        0,
        table.shape[0],
        False,
    )[0]


def _check_full_live_scores(actual, expected, lengths):
    columns = torch.arange(actual.shape[1], device="cuda")[None, :]
    initialized = columns < ((lengths[:, None] + 1023) // 1024) * 1024
    torch.testing.assert_close(
        actual[initialized], expected[initialized], rtol=0, atol=0
    )


def test_full_grouped_scores_match_independent_native_rows():
    queries, weights, cache, table, lengths, capacity, page_size = _full_inputs()
    # Each consecutive group shares a page table and has nondecreasing lengths.
    request_indices = torch.tensor([11, 11, 23, 23], device="cuda", dtype=torch.int32)
    expected = _run_full_scores(
        queries, weights, cache, table, lengths, capacity, page_size, None
    )
    metadata = dsv41.new_index_metadata(
        table, lengths, request_indices, history_rows=None
    )
    actual = _run_full_scores(
        queries, weights, cache, table, lengths, capacity, page_size, metadata
    )
    _check_full_live_scores(actual, expected, lengths)

    consumer = torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(consumer):
        shared = _run_full_scores(
            queries, weights, cache, table, lengths, capacity, page_size, metadata
        )
    torch.cuda.current_stream().wait_stream(consumer)
    _check_full_live_scores(shared, expected, lengths)


def test_full_grouped_metadata_graph_replay_changes_ids_lengths_and_weights():
    queries, weights, cache, table, lengths, capacity, page_size = _full_inputs()
    request_indices = torch.full((4,), 7, device="cuda", dtype=torch.int32)

    def run():
        metadata = dsv41.new_index_metadata(
            table, lengths, request_indices, history_rows=None
        )
        return _run_full_scores(
            queries, weights, cache, table, lengths, capacity, page_size, metadata
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()

    for ids, visible in (
        ([7, 7, 8, 8], [0, 65, 129, 193]),
        ([7, 7, 7, 7], [63, 64, 4095, 4096]),
        ([7, 7, 8, 8], [63, 64, 4095, 4096]),
        ([7, 7, 7, 7], [63, 64, 65, 0]),
    ):
        request_indices.copy_(torch.tensor(ids, device="cuda", dtype=torch.int32))
        lengths.copy_(torch.tensor(visible, device="cuda", dtype=torch.int32))
        weights.normal_()
        expected = _run_full_scores(
            queries, weights, cache, table, lengths, capacity, page_size, None
        )
        eager = run()
        graph.replay()
        _check_full_live_scores(eager, expected, lengths)
        _check_full_live_scores(captured, expected, lengths)


def test_native_candidate_scores_match_full_preparation_after_replay():
    block_size = 8
    logits = torch.randn((4, 35), dtype=torch.bfloat16, device="cuda")
    lengths = torch.tensor([0, 9, 17, 35], dtype=torch.int32, device="cuda")
    alignment = deep_select_alignment()[0] // 4

    def run():
        return select_candidates(logits, lengths, 4, block_size)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for visible in ([0, 9, 17, 35], [35, 0, 1, 8], [8, 17, 33, 35]):
        lengths.copy_(torch.tensor(visible, dtype=torch.int32, device="cuda"))
        logits.normal_()
        columns = torch.arange(35, device="cuda")[None, :]
        logits.masked_fill_(columns >= lengths[:, None], torch.nan)
        graph.replay()
        scores, ends = prepare_scores(logits, lengths, block_size, alignment, True)
        _check_selection(scores, ends, actual)


def test_full_grouped_metadata_slices_ids_across_request_boundaries():
    queries, weights, cache, table, lengths, capacity, page_size = _full_inputs()
    queries = tuple(
        tensor.repeat(2, 1, 1) if tensor.ndim == 3 else tensor.repeat(2, 1)
        for tensor in queries
    )
    weights = weights.repeat(2, 1)
    table = torch.cat((table, table.roll(1, dims=1)), dim=0)
    lengths = torch.tensor(
        [0, 61, 62, 63, 127, 128, 129, 130], device="cuda", dtype=torch.int32
    )
    request_indices = torch.tensor(
        [5, 5, 5, 5, 7, 7, 7, 7], device="cuda", dtype=torch.int32
    )
    metadata = dsv41.new_index_metadata(
        table, lengths, request_indices, history_rows=None
    )
    begin, end = 2, 6
    packed = tuple(tensor[begin:end] for tensor in queries)
    args = (
        packed,
        cache,
        weights[begin:end],
        table[begin:end],
        lengths[begin:end],
        capacity,
        page_size,
    )
    expected = deep_gemm._paged_scores(*args, None, 0, end - begin, False)[0]
    actual = deep_gemm._paged_scores(*args, metadata, begin, end, False)[0]
    _check_full_live_scores(actual, expected, lengths[begin:end])


@pytest.mark.parametrize("bad_indices", ["dtype", "shape", "stride", "device"])
def test_index_metadata_rejects_incompatible_request_indices(bad_indices):
    table = torch.zeros((4, 2), dtype=torch.int32, device="cuda")
    lengths = torch.ones(4, dtype=torch.int32, device="cuda")
    indices = torch.zeros(4, dtype=torch.int32, device="cuda")
    if bad_indices == "dtype":
        indices = indices.long()
    elif bad_indices == "shape":
        indices = indices[:3]
    elif bad_indices == "stride":
        indices = torch.zeros(8, dtype=torch.int32, device="cuda")[::2]
    else:
        indices = indices.cpu()
    with pytest.raises(ValueError, match="request_indices"):
        dsv41.new_index_metadata(table, lengths, indices, history_rows=None)
