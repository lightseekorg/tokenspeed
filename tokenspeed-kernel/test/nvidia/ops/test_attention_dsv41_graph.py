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

from unittest.mock import patch

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires an NVIDIA GPU",
)


def _make_index_cache(x):
    cache = torch.zeros(
        ((x.shape[0] + 63) // 64, 64, 68), dtype=torch.uint8, device=x.device
    )
    dsv41.cache_scatter(x, cache, torch.arange(x.shape[0], device=x.device), "index")
    return cache


def test_index_topk_graph_full_candidates_and_reindex():
    device = torch.device("cuda:0")
    torch.manual_seed(43)
    cache = _make_index_cache(
        torch.randn(256, 128, device=device, dtype=torch.bfloat16)
    )
    q = torch.randn(2, 2, 128, device=device, dtype=torch.bfloat16)
    weights = torch.rand(2, 2, device=device, dtype=torch.bfloat16)
    table = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], device=device, dtype=torch.int32)
    visible = torch.tensor([0, 0], device=device, dtype=torch.int32)

    def run():
        full = dsv41.index_topk(
            q, weights, cache, table, visible, None, 16, 4, 8, 2, 64, None, None, None
        )
        reindex = dsv41.index_topk(
            q,
            weights,
            cache,
            table,
            visible,
            full[2],
            16,
            0,
            8,
            2,
            64,
            None,
            None,
            None,
        )
        return full + reindex

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = run()
    torch.cuda.current_stream().wait_stream(stream)
    for lengths in ([17, 130], [256, 9], [0, 0], [65, 255]):
        q.normal_()
        weights.uniform_()
        visible.copy_(torch.tensor(lengths, device=device, dtype=torch.int32))
        table.copy_(table.flip(1))
        expected = run()
        graph.replay()
        for got, want in zip(output, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


def _hopper_index_case(device, tokens, blocks, pages, table_width, seed):
    """An FP8 index cache, quantized queries and a candidate pool for the Hopper scorers.

    The page table is a permutation with a null and an out-of-range page, so a
    scorer that addressed pages by block id instead of through the table, or
    scored an unmapped page, would not match. Candidates cover the whole cache,
    so a pool wider than the table has blocks past it, and every fifth
    candidate of the second row is null. Visibility ends inside a block for
    the first row and at zero for the last.
    """
    from tokenspeed_kernel.ops.attention.dsv41.triton import quantize_index_queries

    torch.manual_seed(seed)
    rows = pages * 64
    # 132-byte rows: 128 E4M3 values then the FP32 scale, the format the
    # Hopper scorers read. The shared helper only knows the shorter layouts.
    cache = torch.zeros((pages, 64, 132), dtype=torch.uint8, device=device)
    dsv41.cache_scatter(
        torch.randn((rows, 128), dtype=torch.bfloat16, device=device),
        cache,
        torch.arange(rows, device=device),
        "index_v4",
    )
    q = torch.randn((tokens, 64, 128), dtype=torch.bfloat16, device=device)
    weights = torch.rand((tokens, 64), dtype=torch.float32, device=device)
    table = torch.stack(
        [torch.randperm(pages, device=device)[:table_width] for _ in range(tokens)]
    ).to(torch.int32)
    table[:, 1] = -1
    table[:, 2] = pages
    capacity = table_width * 64
    visible = torch.randint(0, capacity + 1, (tokens,), device=device).to(torch.int32)
    visible[0] = capacity - 11
    visible[tokens - 1] = 0
    candidates = torch.randint(
        0, rows // 8, (tokens, blocks), dtype=torch.int32, device=device
    )
    candidates[1 % tokens, ::5] = -1
    queries, folded = quantize_index_queries(q, weights)
    return cache, q, weights, queries, folded, table, visible, candidates, capacity


@pytest.mark.parametrize(
    "blocks,pages,table_width", [(16, 64, 64), (48, 64, 32), (2048, 300, 256)]
)
def test_sparse_index_scores_match_the_dense_scorer(blocks, pages, table_width):
    """The pool scorer reproduces the dense score's masks exactly and its values closely.

    Masks (null candidates, blocks past the table, unmapped pages, rows past
    the visibility bound) must agree bit for bit; values follow the same FP8
    tensor-core arithmetic in a different reduction order, which lands well
    inside 1e-3 of the scores' RMS. 48 blocks give a CTA three tiles.
    """
    from tokenspeed_kernel.ops.attention.dsv41 import cute_dsl, deep_gemm
    from tokenspeed_kernel.ops.attention.dsv41.triton import candidate_scores
    from tokenspeed_kernel.platform import pdl_enabled

    if not deep_gemm.is_hopper_indexer_available():
        pytest.skip("requires the Hopper FP8 indexer")
    device = torch.device("cuda:0")
    tokens = 5
    cache, _, _, queries, folded, table, visible, candidates, capacity = (
        _hopper_index_case(device, tokens, blocks, pages, table_width, 53)
    )
    assert cute_dsl.sparse_index_scores_supported(queries, candidates)

    def dense():
        logits = deep_gemm._hopper_paged_scores(
            (queries,), cache, folded, table, visible, capacity
        )
        return candidate_scores(logits, candidates)

    def sparse(field):
        values, scales = deep_gemm._index_planes(field)
        return cute_dsl.sparse_index_scores(
            queries,
            folded,
            values.view(torch.float8_e4m3fn),
            scales,
            table,
            visible,
            candidates,
            pdl_enabled(),
        )

    expected = dense()
    actual = sparse(cache)
    assert actual.shape == (tokens, blocks * 8)
    torch.testing.assert_close(
        torch.isinf(actual), torch.isinf(expected), rtol=0, atol=0
    )
    assert torch.isinf(expected[tokens - 1]).all()
    finite = ~torch.isinf(expected)
    assert finite.any()
    assert (actual[finite] - expected[finite]).norm() <= 1e-3 * expected[finite].norm()
    # The same launch is deterministic, as graph replay parity relies on.
    torch.testing.assert_close(sparse(cache), actual, rtol=0, atol=0)
    # A field with the same page count but another page stride is a different
    # kernel: strides are compiled in, so the cache must not hand out the first.
    padded = torch.zeros((pages, 2, 64, 132), dtype=torch.uint8, device=device)[:, 0]
    padded.copy_(cache)
    torch.testing.assert_close(sparse(padded), actual, rtol=0, atol=0)
    # A ragged pool would read past the candidate buffer; it is refused.
    values, scales = deep_gemm._index_planes(cache)
    with pytest.raises(ValueError, match="16-block tiles"):
        cute_dsl.sparse_index_scores(
            queries,
            folded,
            values.view(torch.float8_e4m3fn),
            scales,
            table,
            visible,
            candidates[:, :8],
            pdl_enabled(),
        )


# 255 pages make an int32 table row 1020 bytes, so every chunk after the first
# lands on an offset that is not 16-byte aligned.
@pytest.mark.parametrize("pages", [256, 255])
@pytest.mark.parametrize("blocks", [16, 64, 2048])
def test_sparse_reindex_selects_what_the_dense_score_selects(blocks, pages):
    """index_topk picks the same rows with the pool scorer on and off.

    Existing reindex cases use pools narrower than one gather tile, so they
    never reach the sparse kernel; these widths do, and 2048 blocks give a CTA
    several tiles so the gather pipeline rotates stages and flips parity.
    """
    from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm

    if not deep_gemm.is_hopper_indexer_available():
        pytest.skip("requires the Hopper FP8 indexer")
    device = torch.device("cuda:0")
    tokens = 4
    cache, q, weights, _, _, table, visible, candidates, _ = _hopper_index_case(
        device, tokens, blocks, pages, pages, 52
    )

    def select():
        return dsv41.index_topk(
            q,
            weights,
            cache,
            table,
            visible,
            candidates,
            512,
            0,
            8,
            1,
            256,
            None,
            None,
            None,
        )

    sparse, sparse_lengths = (tensor.clone() for tensor in select()[:2])
    with patch.object(deep_gemm, "sparse_index_scores_supported", return_value=False):
        dense, dense_lengths = (tensor.clone() for tensor in select()[:2])
    # The two scorers reduce over heads in a different order, so agreement is
    # on the selection, which is what the pass exists to produce.
    torch.testing.assert_close(sparse_lengths, dense_lengths, rtol=0, atol=0)
    torch.testing.assert_close(sparse, dense, rtol=0, atol=0)
