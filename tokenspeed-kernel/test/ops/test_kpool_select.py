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

"""KPool scoring, selection, and FlatKV handoff tests."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel import kpool_decode_topk, kpool_prefill_topk
from tokenspeed_kernel.ops.attention.triton.kpool_expand import (
    expand_kpool_to_flat_kv,
)
from tokenspeed_kernel.ops.attention.triton.kpool_score import score_kpool_dense
from tokenspeed_kernel.ops.attention.triton.kpool_select import (
    _prepare_kpool_decode_metadata,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_FP8_MAX = 448.0
_POOL, _PAGE, _KV_PAGE, _HEADS, _DIM = 4, 16, 64, 32, 128


def _build_cache(num_pages: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    rows = num_pages * _PAGE
    values = torch.randn(rows, _DIM, device="cuda", generator=generator) * 0.2
    scales = values.abs().amax(-1, keepdim=True).clamp(min=1e-4) / _FP8_MAX
    quantized = (values / scales).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    cache = torch.zeros(num_pages, _PAGE * (_DIM + 4), dtype=torch.uint8, device="cuda")
    cache[:, : _PAGE * _DIM] = quantized.view(torch.uint8).view(num_pages, -1)
    cache[:, _PAGE * _DIM :] = scales.contiguous().view(torch.uint8).view(num_pages, -1)
    return cache, quantized.float() * scales


def _build_monotonic_cache(num_rows: int) -> torch.Tensor:
    num_pages = (num_rows + _PAGE - 1) // _PAGE
    rows = num_pages * _PAGE
    values = torch.zeros((rows, _DIM), dtype=torch.float32, device="cuda")
    values[:, 0] = torch.arange(1, rows + 1, dtype=torch.float32, device="cuda")
    scales = values.abs().amax(-1, keepdim=True).clamp(min=1e-4) / _FP8_MAX
    quantized = (values / scales).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    cache = torch.zeros(num_pages, _PAGE * (_DIM + 4), dtype=torch.uint8, device="cuda")
    cache[:, : _PAGE * _DIM] = quantized.view(torch.uint8).view(num_pages, -1)
    cache[:, _PAGE * _DIM :] = scales.view(torch.uint8).view(num_pages, -1)
    return cache


def _setup(
    seq_lens: list[int], *, q_len_per_req: int = 1, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    requests = len(seq_lens)
    max_seq_len = max(seq_lens)
    index_pages = (max_seq_len // _POOL + _PAGE - 1) // _PAGE + 1
    kv_pages = (max_seq_len + _KV_PAGE - 1) // _KV_PAGE + 1
    cache, dequantized = _build_cache(index_pages * requests, seed + 1)
    index_table = torch.arange(
        index_pages * requests, device="cuda", dtype=torch.int32
    ).view(requests, index_pages)
    kv_table = torch.arange(kv_pages * requests, device="cuda", dtype=torch.int32).view(
        requests, kv_pages
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tokens = requests * q_len_per_req
    q = (
        torch.randn(tokens, _HEADS, _DIM, device="cuda", generator=generator) * 0.3
    ).to(torch.bfloat16)
    weights = torch.randn(tokens, _HEADS, device="cuda", generator=generator).float()
    return (
        q,
        cache,
        dequantized,
        weights,
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        index_table,
        kv_table,
    )


def _reference(
    q: torch.Tensor,
    pooled_k: torch.Tensor,
    weights: torch.Tensor,
    causal_lens: torch.Tensor,
    req_ids: torch.Tensor,
    index_table: torch.Tensor,
    kv_table: torch.Tensor,
    topk: int,
    *,
    apply_relu: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = topk * _POOL + _POOL - 1
    slots = torch.full((q.shape[0], width), -1, dtype=torch.int32, device="cuda")
    lens = torch.empty(q.shape[0], dtype=torch.int32, device="cuda")
    pooled_k = pooled_k.view(-1, _PAGE, _DIM)

    for token in range(q.shape[0]):
        req = int(req_ids[token])
        causal = max(int(causal_lens[token]), 0)
        num_pools = causal // _POOL
        if num_pools <= topk:
            selected = torch.arange(num_pools, device="cuda")
        else:
            pool_ids = torch.arange(num_pools, device="cuda")
            pages = index_table[req, pool_ids // _PAGE].long()
            keys = pooled_k[pages, pool_ids % _PAGE]
            head_scores = torch.einsum("hd,pd->hp", q[token].float(), keys).mul(
                _DIM**-0.5
            )
            if apply_relu:
                head_scores.relu_()
            scores = (head_scores * weights[token, :, None]).sum(0)
            selected = torch.topk(scores, k=topk, sorted=False).indices
        raw_slots = (
            selected[:, None] * _POOL + torch.arange(_POOL, device="cuda")
        ).flatten()
        raw_slots = torch.cat(
            (raw_slots, torch.arange(num_pools * _POOL, causal, device="cuda"))
        ).long()
        pages = kv_table[req, raw_slots // _KV_PAGE].long()
        global_slots = pages * _KV_PAGE + raw_slots % _KV_PAGE
        slots[token, : raw_slots.numel()] = global_slots.to(torch.int32)
        lens[token] = raw_slots.numel()
    return slots, lens


def _assert_selection(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    assert torch.equal(actual[1], expected[1])
    for row, length in enumerate(actual[1].tolist()):
        assert sorted(actual[0][row, :length].tolist()) == sorted(
            expected[0][row, :length].tolist()
        )


def _reference_expand_kpool(
    pool_indices: torch.Tensor,
    causal_lens: torch.Tensor,
    req_ids: torch.Tensor,
    block_table: torch.Tensor,
    *,
    pool_size: int,
    block_size: int,
    append_tail: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, topk_pools = pool_indices.shape
    width = topk_pools * pool_size + (pool_size - 1 if append_tail else 0)
    slots = torch.full(
        (num_tokens, width), -1, dtype=torch.int32, device=pool_indices.device
    )
    lens = torch.zeros(num_tokens, dtype=torch.int32, device=pool_indices.device)

    for token in range(num_tokens):
        seq_len = max(int(causal_lens[token]), 0)
        num_pools = seq_len // pool_size
        history_len = min(num_pools * pool_size, topk_pools * pool_size)
        selected = (
            list(range(num_pools))
            if num_pools <= topk_pools
            else pool_indices[token].tolist()
        )
        raw_slots = []
        for pool in selected[:topk_pools]:
            if len(raw_slots) >= history_len:
                break
            if 0 <= pool < num_pools:
                raw_slots.extend(range(pool * pool_size, (pool + 1) * pool_size))
            else:
                raw_slots.extend([-1] * pool_size)
        if append_tail:
            raw_slots.extend(range(num_pools * pool_size, seq_len))

        count = 0
        req = int(req_ids[token])
        for column, raw_slot in enumerate(raw_slots[:width]):
            if not 0 <= raw_slot < seq_len:
                continue
            page_idx = raw_slot // block_size
            if page_idx >= block_table.shape[1]:
                continue
            page = int(block_table[req, page_idx])
            slots[token, column] = page * block_size + raw_slot % block_size
            count += 1
        lens[token] = count
    return slots, lens


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_dense_scores_match_reference(weight_dtype: torch.dtype) -> None:
    q, cache, pooled_k, weights, seq_lens, index_table, _ = _setup(
        [2051, 4101], seed=47
    )
    req_ids = torch.arange(q.shape[0], dtype=torch.int32, device="cuda")
    weights = weights.to(weight_dtype)
    pool_lens = seq_lens // _POOL
    max_num_pools = int(pool_lens.max())
    actual = score_kpool_dense(
        q,
        cache,
        weights,
        seq_lens,
        req_ids,
        index_table,
        pool_size=_POOL,
        page_size=_PAGE,
        softmax_scale=_DIM**-0.5,
        apply_relu=True,
        max_num_pools=max_num_pools,
    )

    expected = torch.full_like(actual, -float("inf"))
    pooled_k = pooled_k.view(-1, _PAGE, _DIM)
    for token, num_pools in enumerate(pool_lens.tolist()):
        pool_ids = torch.arange(num_pools, device="cuda")
        pages = index_table[token, pool_ids // _PAGE].long()
        keys = pooled_k[pages, pool_ids % _PAGE]
        scores = torch.einsum("hd,pd->hp", q[token].float(), keys)
        expected[token, :num_pools] = (
            scores.mul(_DIM**-0.5).relu() * weights[token, :, None]
        ).sum(0)

    torch.testing.assert_close(actual, expected, rtol=0, atol=5e-7)


@pytest.mark.parametrize("q_len_per_req", [1, 2, 4])
@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64])
def test_decode_metadata_matches_torch_chain(
    q_len_per_req: int,
    seq_dtype: torch.dtype,
) -> None:
    seq_lens = torch.tensor([0, 2, 7], dtype=seq_dtype, device="cuda")
    num_tokens = seq_lens.numel() * q_len_per_req
    actual_req_ids, actual_causal_lens = _prepare_kpool_decode_metadata(
        seq_lens,
        num_tokens,
        q_len_per_req,
    )

    token_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    expected_req_ids = token_ids // q_len_per_req
    expected_causal_lens = (
        seq_lens.to(torch.int32).index_select(0, expected_req_ids)
        - (q_len_per_req - 1)
        + token_ids % q_len_per_req
    ).clamp_min(0)

    assert torch.equal(actual_req_ids, expected_req_ids)
    assert torch.equal(actual_causal_lens, expected_causal_lens)


@pytest.mark.parametrize("table_cols", [1, 2, 5, 9, 16, 31])
def test_expand_kpool_tracks_runtime_page_table_width(table_cols: int) -> None:
    pool_indices = torch.tensor(
        [[0, 1, 2], [0, -1, 5], [0, 2, 4], [0, 16, 32]],
        dtype=torch.int32,
        device="cuda",
    )
    causal_lens = torch.tensor(
        [0, 3, min(table_cols * _KV_PAGE, 17), table_cols * _KV_PAGE + 3],
        dtype=torch.int32,
        device="cuda",
    )
    req_ids = torch.arange(4, dtype=torch.int32, device="cuda")
    table_storage = torch.arange(
        4 * (table_cols + 3), dtype=torch.int32, device="cuda"
    ).view(4, table_cols + 3)
    block_table = table_storage[:, :table_cols]
    width = pool_indices.shape[1] * _POOL + _POOL - 1
    out_storage = torch.empty((4, width + 2), dtype=torch.int32, device="cuda")
    out = out_storage[:, :width]
    lens_out = torch.empty(4, dtype=torch.int32, device="cuda")
    assert not block_table.is_contiguous()
    assert not out.is_contiguous()

    expected = _reference_expand_kpool(
        pool_indices,
        causal_lens,
        req_ids,
        block_table,
        pool_size=_POOL,
        block_size=_KV_PAGE,
        append_tail=True,
    )
    actual = expand_kpool_to_flat_kv(
        pool_indices,
        causal_lens,
        req_ids,
        block_table,
        pool_size=_POOL,
        kv_page_size=_KV_PAGE,
        append_tail=True,
        out=out,
        lens_out=lens_out,
    )

    assert actual[0] is out
    assert actual[1] is lens_out
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_expand_kpool_empty_input_and_invalid_table_contract() -> None:
    pool_indices = torch.empty((0, 3), dtype=torch.int32, device="cuda")
    causal_lens = torch.empty(0, dtype=torch.int32, device="cuda")
    req_ids = torch.empty(0, dtype=torch.int32, device="cuda")
    block_table = torch.empty((1, 2), dtype=torch.int32, device="cuda")

    slots, lens = expand_kpool_to_flat_kv(
        pool_indices,
        causal_lens,
        req_ids,
        block_table,
        pool_size=_POOL,
        kv_page_size=_KV_PAGE,
        append_tail=True,
    )

    assert slots.shape == (0, 15)
    assert lens.shape == (0,)
    with pytest.raises(ValueError, match="at least one page"):
        expand_kpool_to_flat_kv(
            pool_indices,
            causal_lens,
            req_ids,
            block_table[:, :0],
            pool_size=_POOL,
            kv_page_size=_KV_PAGE,
            append_tail=True,
        )


@pytest.mark.parametrize(
    ("topk", "seq_lens"),
    [
        pytest.param(64, [2048, 4099], id="portable"),
        pytest.param(512, [63, 4101], id="production"),
        pytest.param(512, [63, 1029, 2051, 4101], id="production-16-rows"),
    ],
)
@pytest.mark.parametrize("q_len_per_req", [1, 2, 4])
def test_decode_matches_reference(
    topk: int, seq_lens: list[int], q_len_per_req: int
) -> None:
    q, cache, pooled_k, weights, lengths, index_table, kv_table = _setup(
        seq_lens, q_len_per_req=q_len_per_req, seed=2 + topk
    )
    token_ids = torch.arange(q.shape[0], dtype=torch.int32, device="cuda")
    req_ids = token_ids // q_len_per_req
    assert req_ids.dtype == torch.int32
    causal_lens = (
        lengths.index_select(0, req_ids)
        - (q_len_per_req - 1)
        + token_ids % q_len_per_req
    ).clamp_min(0)
    expected = _reference(
        q, pooled_k, weights, causal_lens, req_ids, index_table, kv_table, topk
    )
    out = torch.empty(
        (q.shape[0], topk * _POOL + _POOL - 1), dtype=torch.int32, device="cuda"
    )
    lens_out = torch.empty(q.shape[0], dtype=torch.int32, device="cuda")

    actual = kpool_decode_topk(
        q,
        cache,
        weights,
        lengths,
        index_table,
        kv_table,
        pool_size=_POOL,
        page_size=_PAGE,
        kv_page_size=_KV_PAGE,
        topk_pools=topk,
        softmax_scale=_DIM**-0.5,
        q_len_per_req=q_len_per_req,
        chunk_pools=1024,
        max_seq_len=8192,
        out=out,
        lens_out=lens_out,
    )

    assert actual[0] is out
    assert actual[1] is lens_out
    _assert_selection(actual, expected)


@pytest.mark.parametrize("chunk_pools", [1024, 2048])
@pytest.mark.parametrize("apply_relu", [False, True])
def test_ragged_prefill_parallel_scoring_and_device_sort_match_reference(
    apply_relu: bool,
    chunk_pools: int,
    monkeypatch,
) -> None:
    query_lens = [3, 17]
    starts = [0, chunk_pools * _POOL]
    q, cache, pooled_k, weights, _, index_table, kv_table = _setup(
        [3, starts[1] + query_lens[1]], seed=12
    )
    q = q[:1].expand(sum(query_lens), -1, -1).contiguous()
    signed_weights = torch.linspace(-1.0, 1.0, _HEADS, device="cuda")
    weights = signed_weights.unsqueeze(0).expand(sum(query_lens), -1).contiguous()
    positions = torch.cat(
        [
            torch.arange(start, start + length, dtype=torch.int32, device="cuda")
            for start, length in zip(starts, query_lens, strict=True)
        ]
    )
    query_start_loc = torch.tensor(
        [0, query_lens[0], sum(query_lens)], dtype=torch.int32, device="cuda"
    )
    req_ids = torch.repeat_interleave(
        torch.arange(2, dtype=torch.int32, device="cuda"),
        torch.tensor(query_lens, device="cuda"),
    )
    expected = _reference(
        q,
        pooled_k,
        weights,
        positions + 1,
        req_ids,
        index_table,
        kv_table,
        topk=64,
        apply_relu=apply_relu,
    )

    framework_topk_widths = []
    torch_topk = torch.topk

    def record_framework_topk(input_tensor, *args, **kwargs):
        framework_topk_widths.append(input_tensor.shape[-1])
        return torch_topk(input_tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "topk", record_framework_topk)

    actual = kpool_prefill_topk(
        q,
        cache,
        weights,
        positions,
        query_start_loc,
        index_table,
        kv_table,
        pool_size=_POOL,
        page_size=_PAGE,
        kv_page_size=_KV_PAGE,
        topk_pools=64,
        softmax_scale=_DIM**-0.5,
        apply_relu=apply_relu,
        chunk_pools=chunk_pools,
    )

    assert framework_topk_widths
    assert chunk_pools not in framework_topk_widths
    _assert_selection(actual, expected)


def test_ragged_prefill_workspace_budget_tiles_query_rows(monkeypatch) -> None:
    query_lens = [2, 3]
    starts = [0, 1024]
    q, cache, pooled_k, weights, _, index_table, kv_table = _setup([2, 1027], seed=29)
    q = q[:1].expand(sum(query_lens), -1, -1).contiguous()
    weights = weights[:1].expand(sum(query_lens), -1).contiguous()
    positions = torch.cat(
        [
            torch.arange(start, start + length, dtype=torch.int32, device="cuda")
            for start, length in zip(starts, query_lens, strict=True)
        ]
    )
    query_start_loc = torch.tensor(
        [0, query_lens[0], sum(query_lens)], dtype=torch.int32, device="cuda"
    )
    req_ids = torch.repeat_interleave(
        torch.arange(2, dtype=torch.int32, device="cuda"),
        torch.tensor(query_lens, device="cuda"),
    )
    expected = _reference(
        q, pooled_k, weights, positions + 1, req_ids, index_table, kv_table, topk=64
    )

    allocation_shapes = []
    torch_empty = torch.empty

    def record_empty(*args, **kwargs):
        if args and isinstance(args[0], (tuple, list, torch.Size)):
            allocation_shapes.append(tuple(args[0]))
        return torch_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", record_empty)
    chunk_pools = 128
    workspace_row_bytes = (_HEADS + 3) * chunk_pools * torch.float32.itemsize
    actual = kpool_prefill_topk(
        q,
        cache,
        weights,
        positions,
        query_start_loc,
        index_table,
        kv_table,
        pool_size=_POOL,
        page_size=_PAGE,
        kv_page_size=_KV_PAGE,
        topk_pools=64,
        softmax_scale=_DIM**-0.5,
        chunk_pools=chunk_pools,
        max_logits_bytes=workspace_row_bytes,
    )

    assert (1, _HEADS, chunk_pools) in allocation_shapes
    _assert_selection(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires Blackwell DeepGEMM",
)
def test_planned_prefill_expands_flat_kv() -> None:
    causal_lens = torch.tensor([2051, 2052], dtype=torch.int32, device="cuda")
    num_pools = int(causal_lens[-1]) // _POOL
    cache = _build_monotonic_cache(num_pools)
    q = torch.zeros((2, _HEADS, _DIM), dtype=torch.bfloat16, device="cuda")
    q[:, :, 0] = 1
    weights = torch.ones((2, _HEADS), dtype=torch.float32, device="cuda")
    positions = causal_lens - 1
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    req_ids = torch.zeros(2, dtype=torch.int32, device="cuda")
    pool_workspace_slots = torch.arange(num_pools, dtype=torch.int64, device="cuda")
    row_starts = torch.zeros(2, dtype=torch.int32, device="cuda")
    row_ends = causal_lens // _POOL
    index_table = torch.arange(
        (num_pools + _PAGE - 1) // _PAGE, dtype=torch.int32, device="cuda"
    ).unsqueeze(0)
    kv_table = torch.arange(
        (int(causal_lens[-1]) + _KV_PAGE - 1) // _KV_PAGE,
        dtype=torch.int32,
        device="cuda",
    ).unsqueeze(0)

    slots, lens = kpool_prefill_topk(
        q,
        cache,
        weights,
        positions,
        query_start_loc,
        index_table,
        kv_table,
        pool_size=_POOL,
        page_size=_PAGE,
        kv_page_size=_KV_PAGE,
        topk_pools=512,
        softmax_scale=_DIM**-0.5,
        req_ids=req_ids,
        causal_lens=causal_lens,
        pool_workspace_slots=pool_workspace_slots,
        row_starts=row_starts,
        row_ends=row_ends,
        max_num_pools=num_pools,
    )

    assert lens.tolist() == [2051, 2048]
    assert set(slots[0, :2051].tolist()) == set(range(2051))
    assert set(slots[1, :2048].tolist()) == set(range(4, 2052))


def test_decode_cuda_graph_tracks_dynamic_lengths() -> None:
    q, cache, pooled_k, weights, seq_lens, index_table, kv_table = _setup(
        [4101], seed=41
    )
    seq_lens.fill_(63)
    out = torch.empty((1, 2051), dtype=torch.int32, device="cuda")
    lens_out = torch.empty(1, dtype=torch.int32, device="cuda")

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        return kpool_decode_topk(
            q,
            cache,
            weights,
            seq_lens,
            index_table,
            kv_table,
            pool_size=_POOL,
            page_size=_PAGE,
            kv_page_size=_KV_PAGE,
            topk_pools=512,
            softmax_scale=_DIM**-0.5,
            out=out,
            lens_out=lens_out,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()

    seq_lens.fill_(4101)
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference(
        q,
        pooled_k,
        weights,
        seq_lens,
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        index_table,
        kv_table,
        topk=512,
    )

    assert captured[0] is out
    assert captured[1] is lens_out
    _assert_selection(captured, expected)
