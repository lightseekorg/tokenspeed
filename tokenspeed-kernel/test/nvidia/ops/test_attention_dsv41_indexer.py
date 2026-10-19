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
from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm
from tokenspeed_kernel.ops.attention.dsv41._deep_gemm.indexer import (
    _sparse_candidate_scores,
    clean_logits_and_block_max,
)
from tokenspeed_kernel.ops.attention.dsv41.deep_select import (
    _native_indices,
    _select_topk_prepared,
    select_candidates,
    select_topk,
)
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _clean_logits,
    clean_logits,
    finish_selection,
    pack_index_queries,
    prepare_scores,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.deep_select import deep_select_alignment

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


@pytest.mark.parametrize("width", [16384, 16320])
@pytest.mark.parametrize("topk", [1, 512])
def test_prepared_ends_topk_layouts_and_poisoned_future(width, topk):
    torch.manual_seed(317)
    ends = torch.tensor(
        [0, 1, 511, 512, 513, 1023, 1024, 1025, width - 1, width],
        dtype=torch.int32,
        device="cuda",
    )
    scores = torch.randn(ends.numel(), width, device="cuda")
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
def test_clean_block_max_matches_separate_full_buffers(capacity, paged, alignment):
    torch.manual_seed(193)
    values = torch.randn(4, capacity + 7, device="cuda")[:, :capacity]
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
    assert blocks.stride(0) * blocks.element_size() % alignment == 0


def test_clean_block_max_replay_changes_lengths_pages_and_scores():
    capacity = 16384
    values = torch.randn(4, capacity, device="cuda")
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


@pytest.mark.parametrize("changed", ["candidates", "table", "lengths", "topk", "pages"])
def test_sparse_metadata_reuses_only_matching_input_geometry(monkeypatch, changed):
    args = list(_sparse_inputs(4, 128, True))
    _sparse_candidate_scores(*args, skip_score_topk=0)
    api = args[-1]
    schedule = Mock(wraps=api.get_paged_sparse_mqa_logits_metadata)
    monkeypatch.setattr(api, "get_paged_sparse_mqa_logits_metadata", schedule)
    metadata = dsv41.new_index_metadata(args[3], args[4])
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
        metadata = dsv41.new_index_metadata(table, lengths)
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
        metadata = dsv41.new_index_metadata(table, lengths) if shared else None
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
