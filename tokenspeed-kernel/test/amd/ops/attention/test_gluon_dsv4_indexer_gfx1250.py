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

from __future__ import annotations

import pytest
import torch
from kimi3_reference import dequantize_mxfp4
from utils import is_cdna5

if not is_cdna5():
    pytest.skip("GFX1250 is required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.dsv4 import (
    dsv4_decode_topk,
    dsv4_plan,
    dsv4_prefill_topk,
)
from tokenspeed_kernel.ops.attention.dsv4._triton.indexer import _indexer_logits
from tokenspeed_kernel_amd.ops.gfx1250.attention.dsv4.indexer import (
    _TDM_MIN_CANDIDATES,
    _dsv4_mxfp4_logits,
)

_PAGE_SIZE = 64


def _inputs(tokens: int, heads: int, pages: int):
    generator = torch.Generator(device="cuda").manual_seed(2026)
    q = torch.randint(
        0,
        256,
        (tokens, heads, 64),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    q_scales = torch.randint(
        123,
        128,
        (tokens, heads, 4),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    weights = torch.randn(
        (tokens, heads), device="cuda", dtype=torch.float32, generator=generator
    )
    keys = torch.randint(
        0,
        256,
        (pages, _PAGE_SIZE, 64),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    key_scales = torch.randint(
        123,
        128,
        (pages, _PAGE_SIZE, 4),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    storage = torch.full(
        (pages, _PAGE_SIZE * 68 + 128),
        255,
        device="cuda",
        dtype=torch.uint8,
    )
    storage[:, : _PAGE_SIZE * 64] = keys.reshape(pages, -1)
    storage[:, _PAGE_SIZE * 64 : _PAGE_SIZE * 68] = key_scales.reshape(pages, -1)
    cache = storage[:, : _PAGE_SIZE * 68]
    index_q = (q, q_scales.view(torch.int32).squeeze(-1))
    return (
        index_q,
        weights,
        cache,
        dequantize_mxfp4(q, q_scales, group_size=32),
        dequantize_mxfp4(keys, key_scales, group_size=32),
    )


def _topk_set(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sort(tensor, dim=-1).values


@pytest.mark.parametrize("heads,topk", [(32, 512), (64, 1024), (64, 2048)])
def test_decode_logits_and_topk(heads: int, topk: int) -> None:
    context = topk + 137
    pages_per_request = (context + _PAGE_SIZE - 1) // _PAGE_SIZE
    index_q, weights, cache, _, _ = _inputs(3, heads, 3 * pages_per_request)
    table_storage = torch.full(
        (3, pages_per_request + 3), -1, device="cuda", dtype=torch.int32
    )
    block_table = table_storage[:, :pages_per_request]
    block_table.copy_(
        torch.arange(3 * pages_per_request, device="cuda", dtype=torch.int32).reshape(
            3, pages_per_request
        )
    )
    lengths = torch.tensor(
        [context + 99, topk + 1, 0], device="cuda", dtype=torch.int32
    ).clamp(max=context)

    actual_logits = _dsv4_mxfp4_logits(
        index_q,
        weights,
        cache,
        lengths,
        block_table,
        page_size=_PAGE_SIZE,
        max_candidates=context,
        cu_seq_lens=None,
        cu_seqlen_k_start=None,
    )
    expected_logits, _ = _indexer_logits(
        index_q,
        weights,
        cache,
        lengths,
        block_table,
        page_size=_PAGE_SIZE,
        max_candidates=context,
        cu_seq_lens=None,
        starts=None,
    )
    torch.testing.assert_close(actual_logits, expected_logits, atol=2e-3, rtol=2e-5)

    out = torch.full((4, topk), 77, device="cuda", dtype=torch.int32)
    actual = dsv4_decode_topk(
        index_q,
        weights,
        cache,
        lengths[:, None],
        block_table,
        page_size=_PAGE_SIZE,
        topk=topk,
        max_context_len=context,
        plan=None,
        index_k_format="mxfp4",
        block_table_base_offsets=None,
        out=out,
        persistent_topk_workspace=None,
        override=None,
        solution="gluon",
    )
    expected = dsv4_decode_topk(
        index_q,
        weights,
        cache,
        lengths[:, None],
        block_table,
        page_size=_PAGE_SIZE,
        topk=topk,
        max_context_len=context,
        plan=None,
        index_k_format="mxfp4",
        block_table_base_offsets=None,
        out=None,
        persistent_topk_workspace=None,
        override=None,
        solution="triton",
    )
    assert actual.data_ptr() == out.data_ptr()
    assert (out[-1] == 77).all()
    torch.testing.assert_close(_topk_set(actual), _topk_set(expected))


def test_prefill_ragged_candidates_and_int64_base_offsets() -> None:
    index_q, weights, cache, _, _ = _inputs(4, 32, 30)
    table_storage = torch.full((2, 16), -1, device="cuda", dtype=torch.int32)
    block_table = table_storage[:, :13]
    block_table.copy_(
        torch.stack(
            (
                torch.arange(1, 14, device="cuda", dtype=torch.int32),
                torch.arange(25, 12, -1, device="cuda", dtype=torch.int32),
            )
        )
    )
    cu_seq_lens = torch.tensor([0, 800, 1570], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 3, 800, 800], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([900, 60, 513, 0], device="cuda", dtype=torch.int32)
    ends = starts + lengths
    base_offsets = torch.tensor([7, 19], device="cuda", dtype=torch.int64)

    actual, gathered = dsv4_prefill_topk(
        index_q,
        weights,
        cache,
        block_table,
        cu_seq_lens,
        starts,
        ends,
        lengths,
        page_size=_PAGE_SIZE,
        topk=512,
        max_seqlen_k=704,
        index_k_format="mxfp4",
        block_table_base_offsets=base_offsets,
        gathered_k=None,
        gather_workspace=None,
        out=None,
        override=None,
        solution="gluon",
    )
    expected, _ = dsv4_prefill_topk(
        index_q,
        weights,
        cache,
        block_table,
        cu_seq_lens,
        starts,
        ends,
        lengths.clamp(max=704),
        page_size=_PAGE_SIZE,
        topk=512,
        max_seqlen_k=704,
        index_k_format="mxfp4",
        block_table_base_offsets=base_offsets,
        gathered_k=None,
        gather_workspace=None,
        out=None,
        override=None,
        solution="triton",
    )
    assert gathered is None
    torch.testing.assert_close(_topk_set(actual), _topk_set(expected))


def test_decode_clamps_lengths_and_masks_unaddressable_candidates() -> None:
    index_q, weights, cache, _, _ = _inputs(2, 32, 10)
    block_table = torch.tensor(
        [[0, 1, -1, 3], [9, 10, 7, 6]], device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor([[999], [-3]], device="cuda", dtype=torch.int32)
    actual = dsv4_decode_topk(
        index_q,
        weights,
        cache,
        lengths,
        block_table,
        page_size=_PAGE_SIZE,
        topk=512,
        max_context_len=256,
        plan=None,
        index_k_format="mxfp4",
        block_table_base_offsets=None,
        out=None,
        persistent_topk_workspace=None,
        override=None,
        solution="gluon",
    )
    assert int((actual[0] >= 0).sum()) == 192
    assert not (((actual[0] >= 128) & (actual[0] < 192)).any())
    assert (actual[1] == -1).all()

    empty_table = torch.empty((2, 0), device="cuda", dtype=torch.int32)
    empty = dsv4_decode_topk(
        index_q,
        weights,
        cache,
        torch.ones_like(lengths),
        empty_table,
        page_size=_PAGE_SIZE,
        topk=512,
        max_context_len=64,
        plan=None,
        index_k_format="mxfp4",
        block_table_base_offsets=None,
        out=None,
        persistent_topk_workspace=None,
        override=None,
        solution="gluon",
    )
    assert (empty == -1).all()


def test_decode_tdm_threshold_logits() -> None:
    # tokens * max_candidates reaches the TDM threshold while every context
    # stays short: the scorer must take the TDM path over the pages it needs
    # and fill the chunks past each length with -inf without touching the cache.
    tokens = 32
    max_candidates = 32 * 1024
    assert tokens * max_candidates >= _TDM_MIN_CANDIDATES
    context = 1024 + 37
    pages = (context + _PAGE_SIZE - 1) // _PAGE_SIZE
    index_q, weights, cache, _, _ = _inputs(tokens, 32, tokens * pages)
    block_table = torch.arange(
        tokens * pages - 1, -1, -1, device="cuda", dtype=torch.int32
    ).reshape(tokens, pages)
    lengths = torch.tensor(
        [context, 0, 512, 513, 64, 65, 1, 100] * 4, device="cuda", dtype=torch.int32
    )

    actual = _dsv4_mxfp4_logits(
        index_q,
        weights,
        cache,
        lengths,
        block_table,
        page_size=_PAGE_SIZE,
        max_candidates=max_candidates,
        cu_seq_lens=None,
        cu_seqlen_k_start=None,
    )
    expected, _ = _indexer_logits(
        index_q,
        weights,
        cache,
        lengths,
        block_table,
        page_size=_PAGE_SIZE,
        max_candidates=context,
        cu_seq_lens=None,
        starts=None,
    )
    torch.testing.assert_close(actual[:, :context], expected, atol=2e-3, rtol=2e-5)
    assert torch.isneginf(actual[:, context:]).all()


def test_decode_graph_refresh_and_plan_aliasing() -> None:
    from torch.utils._python_dispatch import TorchDispatchMode

    class RejectClone(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func != torch.ops.aten.clone.default
            return func(*args, **(kwargs or {}))

    index_q, weights, cache, _, _ = _inputs(2, 32, 24)
    block_table = torch.arange(24, device="cuda", dtype=torch.int32).reshape(2, 12)
    lengths = torch.tensor([[720], [0]], device="cuda", dtype=torch.int32)
    base_offsets = torch.tensor([4, 20], device="cuda", dtype=torch.int64)
    out = torch.empty((2, 512), device="cuda", dtype=torch.int32)
    with torch.inference_mode():
        plan = dsv4_plan(
            page_size=_PAGE_SIZE,
            seq_lens_2d=lengths,
            out=None,
            override=None,
            solution="gluon",
        )
    assert plan.data_ptr() != lengths.data_ptr()
    with RejectClone():
        refreshed = dsv4_plan(
            page_size=_PAGE_SIZE,
            seq_lens_2d=lengths,
            out=plan,
            override=None,
            solution="gluon",
        )
    assert refreshed.data_ptr() == plan.data_ptr()

    def run() -> torch.Tensor:
        refreshed = dsv4_plan(
            page_size=_PAGE_SIZE,
            seq_lens_2d=lengths,
            out=plan,
            override=None,
            solution="gluon",
        )
        assert refreshed.data_ptr() == plan.data_ptr()
        return dsv4_decode_topk(
            index_q,
            weights,
            cache,
            lengths,
            block_table,
            page_size=_PAGE_SIZE,
            topk=512,
            max_context_len=768,
            plan=plan,
            index_k_format="mxfp4",
            block_table_base_offsets=base_offsets,
            out=out,
            persistent_topk_workspace=None,
            override=None,
            solution="gluon",
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    lengths.copy_(torch.tensor([[17], [641]], device="cuda", dtype=torch.int32))
    block_table.copy_(block_table.flip(1))
    base_offsets.add_(3)
    graph.replay()
    expected = dsv4_decode_topk(
        index_q,
        weights,
        cache,
        lengths,
        block_table,
        page_size=_PAGE_SIZE,
        topk=512,
        max_context_len=768,
        plan=None,
        index_k_format="mxfp4",
        block_table_base_offsets=base_offsets,
        out=None,
        persistent_topk_workspace=None,
        override=None,
        solution="triton",
    )
    torch.testing.assert_close(_topk_set(out), _topk_set(expected))
    torch.testing.assert_close(plan, lengths)


def test_zero_token_indexer() -> None:
    index_q, weights, cache, _, _ = _inputs(0, 32, 1)
    block_table = torch.empty((0, 0), device="cuda", dtype=torch.int32)
    lengths = torch.empty((0,), device="cuda", dtype=torch.int32)
    base_offsets = torch.empty((0,), device="cuda", dtype=torch.int64)
    actual = dsv4_decode_topk(
        index_q,
        weights,
        cache,
        lengths[:, None],
        block_table,
        page_size=_PAGE_SIZE,
        topk=512,
        max_context_len=64,
        plan=None,
        index_k_format="mxfp4",
        block_table_base_offsets=base_offsets,
        out=None,
        persistent_topk_workspace=None,
        override=None,
        solution="gluon",
    )
    assert actual.shape == (0, 512)
