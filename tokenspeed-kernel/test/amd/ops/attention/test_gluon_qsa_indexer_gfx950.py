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
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

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
    num_blocks = table.shape[1] * _PAGE_SIZE
    blocks = torch.arange(num_blocks, device=query.device)
    slots = table[requests].long()[:, blocks // _PAGE_SIZE] * _PAGE_SIZE + (
        blocks % _PAGE_SIZE
    )
    keys = key_cache[slots, 0].float()
    scores = torch.relu(torch.einsum("rhd,rnd->rhn", query.float(), keys)).sum(1)
    return torch.where(blocks[None, :] < complete[:, None], scores, -torch.inf)


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
def test_gluon_qsa_block_scores_match_reference(widths, contexts) -> None:
    inputs = _inputs(widths, contexts)
    scores = launch_gluon_qsa_block_scores_gfx950(*inputs, page_size=_PAGE_SIZE)
    expected = _reference_scores(*inputs)
    torch.testing.assert_close(scores, expected, atol=2e-3, rtol=1e-4)


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
    torch.testing.assert_close(scores, expected, atol=2e-3, rtol=1e-4)


@pytest.mark.parametrize("cols", [300, 2048, 2100, 9000])
def test_gluon_qsa_block_topk_select_matches_portable_selection(cols) -> None:
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
    selected = launch_gluon_qsa_block_topk_select_gfx950(logits, topk)
    expected = triton_topk_from_logits(logits, topk)
    for row in range(rows):
        got = selected[row].tolist()
        assert sorted(got) == sorted(expected[row].tolist())
        valid = [index for index in got if index >= 0]
        assert valid == sorted(valid)


def test_qsa_indexer_public_ops_match_portable_solution() -> None:
    query, key_cache, table, requests, complete = _inputs([4, 4], [3000, 6000])
    kwargs = dict(page_size=_PAGE_SIZE, queries_per_request=4, enable_pdl=False)
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
    torch.testing.assert_close(gluon, portable, rtol=1e-4, atol=1e-3)
    select_kwargs = dict(lengths=complete, persistent_workspace=None, enable_pdl=False)
    ids = qsa_block_topk_select(
        portable, 256, **select_kwargs, override=None, solution=None
    )
    expected = qsa_block_topk_select(
        portable, 256, **select_kwargs, override=None, solution="triton"
    )
    # Ids are an unordered set per row.
    assert torch.equal(ids.sort(dim=1).values, expected.sort(dim=1).values)


def test_gluon_qsa_indexer_does_not_recompile_per_batch() -> None:
    def run(width, requests):
        inputs = _inputs([width] * requests, [3000 + 7 * r for r in range(requests)])
        scores = launch_gluon_qsa_block_scores_gfx950(*inputs, page_size=_PAGE_SIZE)
        launch_gluon_qsa_block_topk_select_gfx950(scores, 512)

    # Warm the decode and prefill score tiles.
    run(4, 2)
    run(600, 1)
    with assert_no_triton_compile(
        gluon_qsa_block_scores_gfx950, gluon_qsa_block_topk_select_gfx950
    ):
        for width, requests in ((1, 3), (4, 5), (1, 16), (777, 1), (520, 2)):
            run(width, requests)
