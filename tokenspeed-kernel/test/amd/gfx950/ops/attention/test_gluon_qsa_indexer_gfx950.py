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
    qsa_block_topk_stream,
    qwen4_exp_qsa_block_topk,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.qsa.indexer import (  # noqa: E402
    gluon_qsa_block_scores_gfx950,
    gluon_qsa_block_topk_select_gfx950,
    launch_gluon_qsa_block_scores_gfx950,
    launch_gluon_qsa_block_topk_select_gfx950,
    launch_gluon_qsa_block_topk_stream_gfx950,
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


def _reference_topk(logits, topk, lengths):
    """Each row's topk largest scores below its length, as increasing ids
    padded with -1: -inf and NaN never count, and of equal scores the
    smaller id wins. Scores order by their FP32 bits, so +0 beats -0."""
    rows, cols = logits.shape
    host = logits.cpu()
    ends = [cols] * rows if lengths is None else lengths.cpu().tolist()
    out = torch.full((rows, topk), -1, dtype=torch.int32)
    for row in range(rows):
        values = host[row, : min(max(int(ends[row]), 0), cols)]
        ids = torch.nonzero(values >= torch.finfo(torch.float32).min).flatten()
        bits = values[ids].view(torch.int32).long() & 0xFFFFFFFF
        # Larger floats give smaller keys; a stable sort keeps id order.
        keys = torch.where(bits < 1 << 31, 0x7FFFFFFF - bits, bits)
        keep = ids[torch.sort(keys, stable=True).indices[:topk]].sort().values
        out[row, : keep.numel()] = keep.int()
    return out.cuda()


# Prefill-sized single requests, MTP verify runs of four rows and one-row
# drafts, in one-row tiles or (ragged) four-row tiles spanning requests.
@pytest.mark.parametrize(
    ("widths", "contexts", "queries_per_request"),
    [
        ([700], [4000], None),
        ([4] * 6, [300, 900, 2100, 2500, 4000, 64], 4),
        ([1] * 5, [5, 99, 2050, 4000, 9], 1),
        ([1] * 5, [5, 99, 2050, 4000, 9], None),
    ],
)
@pytest.mark.parametrize("metadata_dtype", [torch.int32, torch.int64])
def test_gluon_qsa_block_scores_match_reference(
    widths, contexts, queries_per_request, metadata_dtype
) -> None:
    query, key_cache, table, requests, complete = _inputs(widths, contexts)
    inputs = (
        query,
        key_cache,
        table.to(metadata_dtype),
        requests.to(metadata_dtype),
        complete.to(metadata_dtype),
    )
    scores = launch_gluon_qsa_block_scores_gfx950(
        *inputs, page_size=_PAGE_SIZE, queries_per_request=queries_per_request
    )
    expected = _reference_scores(*inputs)
    valid = torch.arange(scores.shape[1], device="cuda")[None, :] < inputs[-1][:, None]
    torch.testing.assert_close(scores[valid], expected[valid], atol=2e-3, rtol=1e-4)


def test_gluon_qsa_block_scores_do_not_depend_on_the_tile() -> None:
    # A row scores bit-identically in one-row decode tiles, four-row tiles
    # spanning requests and the 16-row prefill tiles of a large batch.
    query, key_cache, table, requests, complete = _inputs([600], [9000])
    large = launch_gluon_qsa_block_scores_gfx950(
        query,
        key_cache,
        table,
        requests,
        complete,
        page_size=_PAGE_SIZE,
        queries_per_request=None,
    )
    rows = torch.arange(0, 600, 75, device="cuda")
    # The picked rows as eight one-row requests over the same pages.
    small_inputs = (
        query[rows],
        key_cache,
        table.expand(rows.numel(), -1).contiguous(),
        torch.arange(rows.numel(), device="cuda"),
        complete[rows],
    )
    valid = (
        torch.arange(large.shape[1], device="cuda")[None, :] < complete[rows][:, None]
    )
    for queries_per_request in (1, None):
        small = launch_gluon_qsa_block_scores_gfx950(
            *small_inputs,
            page_size=_PAGE_SIZE,
            queries_per_request=queries_per_request,
        )
        assert torch.equal(small[valid], large[rows][valid])


def test_gluon_qsa_block_scores_read_padded_key_rows() -> None:
    # Key rows 16-byte aligned but not 32-byte: a 136-element row pitch.
    query, packed, table, requests, complete = _inputs([4] * 3, [900, 3000, 64])
    padded = torch.zeros((packed.shape[0], 1, 136), dtype=torch.bfloat16, device="cuda")
    padded[:, :, :128] = packed
    key_cache = padded[:, :, :128]
    scores = launch_gluon_qsa_block_scores_gfx950(
        query,
        key_cache,
        table,
        requests,
        complete,
        page_size=_PAGE_SIZE,
        queries_per_request=4,
    )
    expected = _reference_scores(query, packed, table, requests, complete)
    valid = torch.arange(scores.shape[1], device="cuda")[None, :] < complete[:, None]
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
        query,
        key_cache,
        table,
        requests,
        complete,
        page_size=_PAGE_SIZE,
        queries_per_request=4,
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
        # Eight splits: one per XCD.
        gluon_qsa_block_scores_gfx950[(triton.cdiv(rows, 4) * 8,)](
            query,
            key_cache,
            table,
            requests,
            complete,
            logits,
            rows,
            num_blocks,
            _PAGE_SIZE,
            8,
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
            NUM_STAGES=2,
            CHUNK_TILES=4,
            NUM_XCDS=8,
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
        scores = launch_gluon_qsa_block_scores_gfx950(
            *inputs, page_size=_PAGE_SIZE, queries_per_request=width
        )
        lengths = torch.empty(
            inputs[-1].numel() * length_stride, device="cuda", dtype=torch.int32
        )[::length_stride]
        lengths.copy_(inputs[-1])
        launch_gluon_qsa_block_topk_select_gfx950(scores, 512, lengths=lengths)

    # Warm the decode (four- and one-row) and prefill score tiles, and the
    # small- and large-batch selections.
    run(4, 2, 1)
    run(1, 2, 1)
    run(600, 1, 1)
    run(600, 2, 1)
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


def _selection_rows(cols, generator):
    """Score rows exercising the selection's cases at one width."""
    rows = [
        torch.randn(cols, generator=generator),
        # Heavy ties at the threshold, still listed.
        torch.randint(0, 8, (cols,), generator=generator).float(),
        # One value everywhere: the candidates overflow the list.
        torch.full((cols,), 3.0),
        torch.zeros(cols),
        torch.linspace(-1, 1, cols),
        # ReLU-summed magnitudes like the QSA scores.
        torch.randn(cols, 4, generator=generator).clamp_min(0).sum(1) * 7,
        # Distinct scores crowded into a few percent of an octave, like
        # scores of random-token prompts.
        100.0 + torch.rand(cols, generator=generator) * 4.0,
        # Positive scores above a negative majority.
        torch.where(
            torch.rand(cols, generator=generator) < 0.05,
            torch.rand(cols, generator=generator),
            -torch.rand(cols, generator=generator),
        ),
    ]
    signed_zeros = torch.zeros(cols)
    signed_zeros[::3] = -0.0
    signed_zeros[cols // 2 :: 97] = 1.0
    rows.append(signed_zeros)
    holes = torch.randn(cols, generator=generator)
    holes[::5] = float("-inf")
    holes[1::7] = float("nan")
    rows.append(holes)
    return torch.stack(rows)


@pytest.mark.parametrize("cols", [700, 4096, 9000, 22528, 65536])
@pytest.mark.parametrize("topk", [512, 2048])
def test_gluon_qsa_block_topk_select_matches_reference(cols, topk) -> None:
    generator = torch.Generator(device="cpu").manual_seed(cols + topk)
    logits = _selection_rows(cols, generator).cuda()
    rows = logits.shape[0]
    # Live lengths below and at the width; the tail holds garbage.
    lengths = torch.tensor(
        [cols - 3 * row for row in range(rows)], dtype=torch.int32, device="cuda"
    )
    lengths[0] = cols
    tail = torch.arange(cols, device="cuda")[None, :] >= lengths[:, None]
    logits.masked_fill_(tail, float("inf"))
    selected = launch_gluon_qsa_block_topk_select_gfx950(logits, topk, lengths=lengths)
    torch.testing.assert_close(
        selected, _reference_topk(logits, topk, lengths), rtol=0, atol=0
    )


def test_gluon_qsa_block_topk_select_matches_reference_in_large_batches() -> None:
    # Large batches run the selection with fewer warps per row.
    generator = torch.Generator(device="cpu").manual_seed(11)
    cols = 9000
    rows = _selection_rows(cols, generator)
    logits = rows.repeat(1100 // rows.shape[0] + 1, 1)[:1100].cuda()
    lengths = torch.randint(
        1, cols + 1, (logits.shape[0],), generator=generator, dtype=torch.int32
    ).cuda()
    selected = launch_gluon_qsa_block_topk_select_gfx950(logits, 512, lengths=lengths)
    torch.testing.assert_close(
        selected, _reference_topk(logits, 512, lengths), rtol=0, atol=0
    )


def test_gluon_qsa_block_topk_select_rows_beyond_2gib() -> None:
    # Row bases are 64-bit: the second row starts 2 GiB into the scores.
    stride = 1 << 29
    cols = 9000
    storage = torch.empty(stride + cols, dtype=torch.float32, device="cuda")
    logits = storage.as_strided((2, cols), (stride, 1))
    generator = torch.Generator(device="cpu").manual_seed(7)
    logits.copy_(torch.randn((2, cols), generator=generator).cuda())
    assert logits.stride(0) * logits.element_size() >= 1 << 31
    selected = launch_gluon_qsa_block_topk_select_gfx950(logits, 512, lengths=None)
    torch.testing.assert_close(
        selected, _reference_topk(logits, 512, None), rtol=0, atol=0
    )


@pytest.mark.parametrize("budget_rows", [1, 5, 64])
def test_gluon_qsa_block_topk_stream_matches_logits_path(budget_rows) -> None:
    # Ragged requests: a chunk boundary may split a request's rows.
    inputs = _inputs([700, 4, 1, 37], [6000, 2500, 9000, 1500])
    num_blocks = inputs[2].shape[1] * _PAGE_SIZE
    selected = launch_gluon_qsa_block_topk_stream_gfx950(
        *inputs,
        page_size=_PAGE_SIZE,
        block_topk=512,
        max_partial_bytes=budget_rows * num_blocks * 4,
    )
    scores = launch_gluon_qsa_block_scores_gfx950(
        *inputs, page_size=_PAGE_SIZE, queries_per_request=None
    )
    expected = launch_gluon_qsa_block_topk_select_gfx950(
        scores, 512, lengths=inputs[-1]
    )
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)


def test_gluon_qsa_block_topk_stream_rejects_a_budget_below_one_row() -> None:
    inputs = _inputs([4], [3000])
    num_blocks = inputs[2].shape[1] * _PAGE_SIZE
    with pytest.raises(ValueError, match="cannot hold one row"):
        launch_gluon_qsa_block_topk_stream_gfx950(
            *inputs,
            page_size=_PAGE_SIZE,
            block_topk=512,
            max_partial_bytes=num_blocks * 4 - 1,
        )


def test_qsa_stream_and_logits_paths_select_alike_on_gfx950() -> None:
    query, key_cache, table, requests, complete = _inputs([300], [9000])
    common = dict(
        page_size=_PAGE_SIZE,
        block_topk=512,
        queries_per_request=None,
        max_partial_bytes=64 * table.shape[1] * _PAGE_SIZE * 4,
        enable_pdl=False,
        solution=None,
        override=None,
    )
    stream = qsa_block_topk_stream(
        query, key_cache, table, requests, complete, **common
    )
    by_path = {
        path: qwen4_exp_qsa_block_topk(
            query,
            key_cache,
            table,
            requests,
            complete,
            **common,
            max_logits_bytes=1 << 30,
            path=path,
        )
        for path in ("stream", "logits")
    }
    torch.testing.assert_close(stream, by_path["stream"], rtol=0, atol=0)
    torch.testing.assert_close(by_path["stream"], by_path["logits"], rtol=0, atol=0)
