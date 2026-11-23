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

"""Reusable GPU numerical and graph contracts for dense native FP4 M0."""

import os

import pytest
import torch
from tokenspeed_kernel.numerics.torch.quantization import dequantize_mxfp4
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm as impl
from tokenspeed_kernel.ops.attention.dsv41 import flashinfer_index_sm120  # noqa: F401
from tokenspeed_kernel.ops.attention.dsv41._deep_gemm.indexer import gather_index_pages
from tokenspeed_kernel.ops.attention.dsv41.triton import pack_index_inputs
from tokenspeed_kernel.thirdparty.flashinfer.topk import has_exact_topk

pytestmark = pytest.mark.skipif(
    os.environ.get("TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER") != "1"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not has_exact_topk(),
    reason="requires SM120, exact FI TopK and explicit native FP4 opt-in",
)


def _scale_bytes(words):
    shifts = torch.arange(4, device=words.device) * 8
    return ((words.long()[..., None] >> shifts) & 255).to(torch.uint8)


def _dense_reference(q, weights, cache, table, visible):
    packed, folded = pack_index_inputs(q, weights, torch.float32)
    flat = cache.as_strided((cache.shape[0], 64 * 68), (cache.stride(0), 1))
    keys = gather_index_pages(flat, table[0], 64)
    q_values = dequantize_mxfp4(packed[0], _scale_bytes(packed[1])).float()
    k_values = dequantize_mxfp4(keys[0], _scale_bytes(keys[1])).float()
    dots = torch.einsum("thc,kc->thk", q_values, k_values).relu()
    scores = (dots * folded.float().unsqueeze(-1)).sum(dim=1)
    capacity = table.shape[1] * 64
    positions = torch.arange(capacity, device=q.device)
    valid_pages = ((table[0] >= 0) & (table[0] < cache.shape[0])).repeat_interleave(64)
    for row in range(q.shape[0]):
        end = min(max(int(visible[row]), 0), capacity)
        valid = (positions < end) & valid_pages
        scores[row].masked_fill_(~valid, -torch.inf)
    return packed, folded, keys, scores


def _mixed_case(tokens=9, history=4160, seed=271):
    torch.manual_seed(seed)
    q = torch.randn(tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(tokens, 32, device="cuda", dtype=torch.float32)
    keys = torch.randn(history, 128, device="cuda", dtype=torch.bfloat16)
    pages = history // 64
    cache = torch.zeros(pages, 64, 68, device="cuda", dtype=torch.uint8)
    table = torch.arange(pages, device="cuda", dtype=torch.int32)
    slots = table.long().repeat_interleave(64) * 64
    slots += torch.arange(history, device="cuda") % 64
    dsv41.cache_scatter(keys, cache, slots, "index")
    # Page zero carries real data; the extra invalid page checks tail masking.
    table = torch.cat((table, torch.full((1,), -1, device="cuda", dtype=torch.int32)))
    table = table[None, :].expand(tokens, -1)
    visible = torch.tensor(
        [0, 1, 17, 511, 512, 513, history - 1, history, history + 64],
        device="cuda",
        dtype=torch.int32,
    )
    return q, weights, cache, table, visible


def test_native_dense_scores_match_explicit_fp32_reference():
    q, weights, cache, table, visible = _mixed_case()
    packed, folded, keys, expected = _dense_reference(q, weights, cache, table, visible)
    actual = impl._dense_scores_sm120(
        packed,
        keys,
        folded,
        visible,
        table,
        cache.shape[0],
        table.shape[1] * 64,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-3)


def test_native_full_topk_and_all_fallbacks():
    q, weights, cache, table, visible = _mixed_case()
    *_, scores = _dense_reference(q, weights, cache, table, visible)
    args = (q, weights, cache, table, visible, None, 512, 0, 8, 256, 256, None, None)
    rows, lengths, blocks, block_lengths = dsv41.index_topk(*args, solution="deep_gemm")
    assert blocks.shape[1] == 0 and block_lengths.count_nonzero() == 0
    # Reused caller buffers must also clear the unused candidate lengths.
    reused = tuple(
        torch.full_like(tensor, 73) for tensor in (rows, lengths, blocks, block_lengths)
    )
    actual = dsv41.index_topk(*args[:-1], reused, solution="deep_gemm")
    for got, wanted in zip(actual, (rows, lengths, blocks, block_lengths), strict=True):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    for row in range(q.shape[0]):
        count = min(max(int(visible[row]), 0), 512)
        assert int(lengths[row]) == count
        if count:
            expected = (
                torch.topk(scores[row], count).indices.sort().values.to(rows.dtype)
            )
            torch.testing.assert_close(rows[row, :count], expected, rtol=0, atol=0)
        assert torch.all(rows[row, count:] == -1)

    # Candidate production and Reindex must be bitwise BF16 fallbacks.
    candidate_args = (
        q,
        weights,
        cache,
        table,
        visible,
        None,
        512,
        17,
        8,
        256,
        256,
        None,
        None,
    )
    expected = dsv41.index_topk(*candidate_args, solution="flashinfer_sm120")
    actual = dsv41.index_topk(*candidate_args, solution="deep_gemm")
    for got, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    reindex_args = (
        q,
        weights,
        cache,
        table,
        visible,
        expected[2],
        512,
        0,
        8,
        256,
        256,
        None,
        None,
    )
    expected_reindex = dsv41.index_topk(*reindex_args, solution="flashinfer_sm120")
    actual_reindex = dsv41.index_topk(*reindex_args, solution="deep_gemm")
    for got, wanted in zip(actual_reindex, expected_reindex, strict=True):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)

    # A contiguous table models paged/Decode and must also remain BF16.
    paged_args = (
        q,
        weights,
        cache,
        table.clone(),
        visible,
        None,
        512,
        0,
        8,
        256,
        256,
        None,
        None,
    )
    expected = dsv41.index_topk(*paged_args, solution="flashinfer_sm120")
    actual = dsv41.index_topk(*paged_args, solution="deep_gemm")
    for got, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)

    empty_args = (
        q[:0],
        weights[:0],
        cache,
        table[:0],
        visible[:0],
        None,
        512,
        0,
        8,
        256,
        256,
        None,
        None,
    )
    expected = dsv41.index_topk(*empty_args, solution="flashinfer_sm120")
    actual = dsv41.index_topk(*empty_args, solution="deep_gemm")
    for got, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)


def test_native_dense_graph_refreshes_q_weights_and_visible():
    torch.manual_seed(31)
    tokens, history = 256, 8192
    cache = torch.zeros(history // 64, 64, 68, device="cuda", dtype=torch.uint8)
    dsv41.cache_scatter(
        torch.randn(history, 128, device="cuda", dtype=torch.bfloat16),
        cache,
        torch.arange(history, device="cuda"),
        "index",
    )
    table = torch.arange(history // 64, device="cuda", dtype=torch.int32)
    table = table[None, :].expand(tokens, -1)
    q = torch.randn(tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(tokens, 32, device="cuda", dtype=torch.float32)
    visible = torch.full((tokens,), history, device="cuda", dtype=torch.int32)
    args = (q, weights, cache, table, visible, None, 512, 0, 8, 256, 256, None, None)
    dsv41.index_topk(*args, solution="deep_gemm")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = dsv41.index_topk(*args, solution="deep_gemm")
    for shift in (3, 5):
        q.copy_(torch.randn_like(q))
        weights.copy_(torch.randn_like(weights))
        visible.copy_(torch.arange(tokens, device="cuda", dtype=torch.int32) + shift)
        visible[:7] = torch.tensor([0, 1, 17, 511, 512, 513, history], device="cuda")
        graph.replay()
        expected = dsv41.index_topk(*args, solution="deep_gemm")
        for got, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize(
    "tokens,history,candidate_topk", [(9, 4160, 17), (128, 131072, 2048)]
)
def test_native_candidate_blocks_match_same_score_reference(
    monkeypatch, ties, tokens, history, candidate_topk
):
    monkeypatch.setattr(impl, "_SM120_NATIVE_CANDIDATES", True)
    q, weights, cache, table, visible = _mixed_case(history=history)
    indices = torch.arange(tokens, device="cuda") % q.shape[0]
    q, weights, visible = (x.index_select(0, indices) for x in (q, weights, visible))
    table = table[:1].expand(tokens, -1)
    if ties:
        weights.zero_()
    # Use the native scores as the selection oracle: FP4 scores are not
    # bitwise equivalent to the BF16 score path.
    packed, folded = pack_index_inputs(q, weights, torch.float32)
    flat = cache.as_strided((cache.shape[0], 64 * 68), (cache.stride(0), 1))
    keys = gather_index_pages(flat, table[0], 64)
    capacity = table.shape[1] * 64
    scores = impl._dense_scores_sm120(
        packed, keys, folded, visible, table, cache.shape[0], capacity
    )
    args = (
        q,
        weights,
        cache,
        table,
        visible,
        None,
        512,
        candidate_topk,
        8,
        256,
        256,
        None,
        None,
    )
    result = dsv41.index_topk(*args, solution="deep_gemm")
    for row in range(q.shape[0]):
        score = scores[row]
        chosen = torch.argsort(score, descending=True, stable=True)[:512]
        chosen = chosen[torch.isfinite(score[chosen])].sort().values
        assert result[1][row].item() == chosen.numel()
        assert torch.equal(result[0][row, : chosen.numel()].long(), chosen)
        assert (result[0][row, chosen.numel() :] == -1).all()
        maxima = score.reshape(-1, 8).amax(-1).clone()
        end = min(max(int(visible[row]), 0), capacity)
        if end and torch.isfinite(maxima[(end - 1) // 8]):
            maxima[(end - 1) // 8] = torch.inf
        selected = torch.argsort(maxima, descending=True, stable=True)[:candidate_topk]
        selected = selected[maxima[selected] > -torch.inf].sort().values
        assert result[3][row].item() == selected.numel()
        assert torch.equal(result[2][row, : selected.numel()].long(), selected)
        assert (result[2][row, selected.numel() :] == -1).all()
    for tensor in result:
        tensor.fill_(77)
    refreshed = dsv41.index_topk(*args[:-1], result, solution="deep_gemm")
    for got, expected in zip(
        refreshed, dsv41.index_topk(*args, solution="deep_gemm"), strict=True
    ):
        assert torch.equal(got, expected)


def test_default_native_selection_survives_bf16_fallback_registration():
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    q, weights, cache, table, visible = _mixed_case()
    # Importing this adapter registers the higher-priority BF16 fast path.
    # A native opt-in must remain selected for eligible dense Full requests.
    from tokenspeed_kernel.ops.attention.dsv41 import (  # noqa: F401
        flashinfer_index_sm120,
    )

    traits = dsv41._index_topk_traits(q.shape[1], q.is_cuda, cache, None)
    selected = select_kernel(
        "attention",
        "dsv41_index_topk",
        format_signature(x=dense_tensor_format(q.dtype)),
        features=None,
        platform=None,
        traits=traits,
        solution=None,
        override=None,
    )
    assert selected.name == "deep_gemm_dsv41_index_topk"
    # A short query actually executes the BF16 fallback, then the next dense
    # request must return to native FP4, rather than staying on the fallback.
    dsv41.index_topk(
        q[:1],
        weights[:1],
        cache,
        table[:1],
        visible[:1],
        None,
        512,
        0,
        8,
        256,
        256,
        None,
        None,
        solution=None,
    )
    selected_after = select_kernel(
        "attention",
        "dsv41_index_topk",
        format_signature(x=dense_tensor_format(q.dtype)),
        features=None,
        platform=None,
        traits=traits,
        solution=None,
        override=None,
    )
    assert selected_after == selected
    args = (q, weights, cache, table, visible, None, 512, 0, 8, 256, 256, None, None)
    automatic = dsv41.index_topk(*args, solution=None)
    explicit = dsv41.index_topk(*args, solution="deep_gemm")
    for actual, expected in zip(automatic, explicit, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
