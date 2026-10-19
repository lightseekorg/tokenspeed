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

"""GFX950 Gluon QSA indexer scoring and top-k selection checks."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.qsa import (
    torch_qsa_paged_block_scores,
)
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel._triton import triton  # noqa: E402
from tokenspeed_kernel.ops.attention.dsa.triton import (  # noqa: E402
    triton_topk_from_logits,
)
from tokenspeed_kernel.ops.attention.qsa import (  # noqa: E402
    qsa_block_scores,
    qsa_block_topk_select,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.qsa.indexer import (  # noqa: E402
    gluon_qsa_block_scores_gfx950,
    gluon_qsa_block_topk_select_gfx950,
    launch_gluon_qsa_block_scores_gfx950,
    launch_gluon_qsa_block_topk_select_gfx950,
)

_PAGE_SIZE = 64


def _inputs(widths, contexts, heads=4, head_dim=128, seed=0):
    """Rows of each request are consecutive positions ending at its context."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pages = max(contexts) // 4 // _PAGE_SIZE + 2
    table = torch.randperm(len(widths) * pages, generator=generator) + 1
    table = table.view(len(widths), pages).int()
    requests, complete = [], []
    for request, (width, context) in enumerate(zip(widths, contexts)):
        for position in range(context - width, context):
            requests.append(request)
            complete.append((position + 1) // 4)
    rows = len(requests)
    # A strided query view, like the fused projection output.
    wide = torch.randn((rows, heads + 1, head_dim), generator=generator)
    key_cache = torch.randn(((len(widths) * pages + 1) * _PAGE_SIZE, 1, head_dim))
    return (
        wide.bfloat16().cuda()[:, :heads],
        key_cache.bfloat16().cuda(),
        table.cuda(),
        torch.tensor(requests, device="cuda"),
        torch.tensor(complete, dtype=torch.int32, device="cuda"),
    )


def _reference_scores(query, key_cache, table, requests, complete):
    return torch_qsa_paged_block_scores(
        query, key_cache, table, requests, complete, page_size=_PAGE_SIZE
    )


# Prefill-sized single requests, MTP verify runs of four rows and one-row
# drafts (every score tile spans several requests).
@pytest.mark.parametrize(
    ("widths", "contexts"),
    [
        ([700], [4000]),
        ([4] * 6, [300, 900, 2100, 2500, 4000, 64]),
        ([1] * 5, [5, 99, 2050, 4000, 9]),
    ],
)
@pytest.mark.parametrize("metadata_dtype", [torch.int32, torch.int64])
def test_gluon_qsa_block_scores_match_reference(
    widths, contexts, metadata_dtype
) -> None:
    query, key_cache, table, requests, complete = _inputs(widths, contexts)
    inputs = (
        query,
        key_cache,
        table.to(metadata_dtype),
        requests.to(metadata_dtype),
        complete.to(metadata_dtype),
    )
    scores = launch_gluon_qsa_block_scores_gfx950(*inputs, page_size=_PAGE_SIZE)
    expected = _reference_scores(*inputs)
    valid = torch.arange(scores.shape[1], device="cuda")[None, :] < inputs[-1][:, None]
    torch.testing.assert_close(scores[valid], expected[valid], atol=2e-3, rtol=1e-4)


def test_gluon_qsa_block_scores_read_caches_beyond_2gib() -> None:
    # Serving caches exceed the 32-bit byte reach of buffer loads; place the
    # requests' pages at the end of a > 2 GiB compressed cache.
    query, small_cache, table, requests, complete = _inputs([4] * 3, [900, 3000, 64])
    pages = small_cache.shape[0] // _PAGE_SIZE
    total_pages = (2**31 // (_PAGE_SIZE * 128 * 2)) + 2 * pages
    key_cache = torch.empty(
        (total_pages * _PAGE_SIZE, 1, 128), dtype=torch.bfloat16, device="cuda"
    )
    assert key_cache.numel() * key_cache.element_size() > 2**31
    key_cache[-small_cache.shape[0] :] = small_cache
    table = table + (total_pages - pages)
    scores = launch_gluon_qsa_block_scores_gfx950(
        query, key_cache, table, requests, complete, page_size=_PAGE_SIZE
    )
    expected = _reference_scores(query, key_cache, table, requests, complete)
    valid = torch.arange(scores.shape[1], device="cuda")[None, :] < complete[:, None]
    torch.testing.assert_close(scores[valid], expected[valid], atol=2e-3, rtol=1e-4)


def test_gluon_qsa_scores_refresh_frontiers_without_writing_tails() -> None:
    inputs = _inputs([4] * 3, [900, 3000, 64])
    query, key_cache, table, requests, complete = inputs
    rows, num_blocks = query.shape[0], table.shape[1] * _PAGE_SIZE
    logits = torch.full((rows, num_blocks), float("nan"), device="cuda")

    def forward():
        gluon_qsa_block_scores_gfx950[(triton.cdiv(rows, 4), 2)](
            query,
            key_cache,
            table,
            requests,
            complete,
            logits,
            rows,
            num_blocks,
            _PAGE_SIZE,
            2,
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            table.stride(0),
            logits.stride(0),
            NUM_HEADS=4,
            HEADS_PAD=4,
            HEAD_DIM=128,
            HEAD_DIM_PAD=128,
            BLOCK_Q=4,
            BLOCK_N=128,
            num_warps=4,
        )

    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    for frontier in (num_blocks, 129, -rows, 17, num_blocks):
        query.normal_()
        complete.copy_(
            (torch.arange(rows, device="cuda", dtype=torch.int32) + frontier).clamp(
                0, num_blocks
            )
        )
        logits.fill_(float("nan"))
        graph.replay()
        expected = _reference_scores(*inputs)
        valid = torch.arange(num_blocks, device="cuda")[None, :] < complete[:, None]
        torch.testing.assert_close(logits[valid], expected[valid], atol=2e-3, rtol=1e-4)
        assert torch.isnan(logits[~valid]).all()


@pytest.mark.parametrize("cols", [300, 2048, 2100, 9000])
@pytest.mark.parametrize("use_lengths", [False, True])
def test_gluon_qsa_block_topk_select_matches_portable_selection(
    cols, use_lengths
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(cols)
    rows, topk = 9, 512
    logits = torch.randn((rows, cols), generator=generator)
    # Coarse values create ties at the threshold; zeros mimic ReLU'd scores.
    logits[1] = torch.randint(0, 5, (cols,), generator=generator).float()
    logits[2] = 0.0
    logits[3, 200:] = -torch.inf
    logits[4] = -torch.inf
    logits[5, :] = torch.linspace(-1, 1, cols)
    logits = logits.cuda()
    lengths = None
    if use_lengths:
        storage = torch.zeros(rows * 2, device="cuda", dtype=torch.int64)
        lengths = storage[::2]
        lengths.copy_(
            torch.tensor([cols, 129, 0, 200, 0, cols + 23, -1, 513, 17], device="cuda")
        )
        valid = torch.arange(cols, device="cuda")[None, :] < lengths[:, None]
        logits.masked_fill_(~valid, float("nan"))
    selected = launch_gluon_qsa_block_topk_select_gfx950(logits, topk, lengths=lengths)
    expected = triton_topk_from_logits(logits, topk, seq_lens=lengths, enable_pdl=False)
    for row in range(rows):
        got = selected[row].tolist()
        assert sorted(got) == sorted(expected[row].tolist())
        valid = [index for index in got if index >= 0]
        assert valid == sorted(valid)


def test_qsa_indexer_public_ops_match_portable_solution() -> None:
    query, key_cache, table, requests, complete = _inputs([4, 4], [3000, 6000])
    kwargs = dict(
        page_size=_PAGE_SIZE,
        queries_per_request=4,
        enable_pdl=False,
    )
    gluon = qsa_block_scores(
        query,
        key_cache,
        table,
        requests,
        complete,
        **kwargs,
        override=None,
        solution=None,
    )
    portable = qsa_block_scores(
        query,
        key_cache,
        table,
        requests,
        complete,
        **kwargs,
        override=None,
        solution="triton",
    )
    valid = torch.arange(portable.shape[1], device="cuda")[None, :] < complete[:, None]
    torch.testing.assert_close(gluon[valid], portable[valid], rtol=1e-4, atol=1e-3)
    portable.masked_fill_(~valid, float("nan"))
    select_kwargs = dict(
        lengths=complete,
        enable_pdl=False,
    )
    ids = qsa_block_topk_select(
        portable, 256, **select_kwargs, override=None, solution=None
    )
    expected = qsa_block_topk_select(
        portable, 256, **select_kwargs, override=None, solution="triton"
    )
    # Ids are an unordered set per row.
    assert torch.equal(ids.sort(dim=1).values, expected.sort(dim=1).values)


def test_gluon_qsa_indexer_does_not_recompile_per_batch() -> None:
    def run(width, requests, length_stride):
        inputs = _inputs([width] * requests, [3000 + 7 * r for r in range(requests)])
        scores = launch_gluon_qsa_block_scores_gfx950(*inputs, page_size=_PAGE_SIZE)
        lengths = torch.empty(
            inputs[-1].numel() * length_stride, device="cuda", dtype=torch.int32
        )[::length_stride]
        lengths.copy_(inputs[-1])
        launch_gluon_qsa_block_topk_select_gfx950(scores, 512, lengths=lengths)

    # Warm the decode and prefill score tiles.
    run(4, 2, 1)
    run(600, 1, 1)
    with assert_no_triton_compile(
        gluon_qsa_block_scores_gfx950, gluon_qsa_block_topk_select_gfx950
    ):
        for width, requests, stride in (
            (1, 3, 1),
            (4, 5, 2),
            (1, 16, 3),
            (777, 1, 2),
            (520, 2, 1),
        ):
            run(width, requests, stride)
