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

"""GFX950 Gluon scorer for DCP-owned Index-K pages: default selection and
agreement with the torch index-score reference."""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD GFX950 is required for the Gluon DCP index scorer",
        allow_module_level=True,
    )

from ops.test_attention_dsa import _pack_index_k_cache  # isort: skip
from tokenspeed_kernel.numerics.torch.attention.dsa import (  # isort: skip
    dsa_index_scores,
)
from tokenspeed_kernel.ops.attention.dsa import (  # isort: skip
    dsa_index_candidates,
    plan_dsa_index_candidates,
)

PAGE_SIZE = 64
HEAD_DIM = 128
GLUON = "gluon_dsa_index_candidates_gfx950"


def _fp8_query(rows: int, heads: int, device: str):
    """A random query's FP8 heads and their FP32 per-head scales."""
    q = torch.randn(rows, heads, HEAD_DIM, device=device)
    scales = q.abs().amax(dim=-1).clamp_min(1.0e-6) / 448.0
    return (q / scales[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scales


def _rows(steps, heads: int, device: str):
    """Query rows of ``(request, causal length, count)`` steps: ``count``
    consecutive rows each, from that length on, with FP8 queries and head
    weights."""
    requests = torch.cat(
        [torch.full((count,), r, dtype=torch.int32) for r, _, count in steps]
    ).to(device)
    lens = torch.cat(
        [
            torch.arange(first, first + count, dtype=torch.int32)
            for _, first, count in steps
        ]
    ).to(device)
    q, q_scales = _fp8_query(requests.numel(), heads, device)
    weights = torch.randn(requests.numel(), heads, device=device)
    return requests, lens, q, q_scales, weights


def _inputs(heads: int, group: int, device: str):
    """Two requests whose owned pages are scattered and partly absent (other
    ranks own them), ``group`` consecutive verify rows per request step."""
    torch.manual_seed(1907 + heads + group)
    table = torch.tensor(
        [
            [12, -1, 3, 40, -1, 7, 25, 1, -1, 33, 18, 9, 44, -1, 30, 5],
            [21, 2, -1, 37, 14, -1, 28, 11, 46, 6, -1, 35, 16, 42, -1, 23],
        ],
        dtype=torch.int32,
        device=device,
    )
    steps = [(0, 1000, group), (1, 977, group), (0, 523, group), (1, 64, group)]
    rows = _rows(steps, heads, device)
    cache, keys = _pack_index_k_cache(
        torch.randn(48 * PAGE_SIZE, HEAD_DIM, device=device), PAGE_SIZE
    )
    return table, *rows, cache, keys


def _reference(table, requests, lens, q_fp8, q_scales, weights, keys, scale):
    """Per row: the visible owned positions and their reference scores."""
    out = []
    q = q_fp8.float() * q_scales[..., None]
    for row in range(requests.numel()):
        pages = table[int(requests[row])]
        positions = torch.arange(int(lens[row]), device=table.device)
        page = pages[positions // PAGE_SIZE]
        owned = page >= 0
        positions = positions[owned]
        slots = page[owned].long() * PAGE_SIZE + positions % PAGE_SIZE
        scores = dsa_index_scores(q[row], weights[row], keys[slots], scale)
        out.append((positions, scores))
    return out


def _set_key_scale(cache: torch.Tensor, slot: int, value: float) -> None:
    page, row = divmod(slot, PAGE_SIZE)
    page_floats = PAGE_SIZE * (HEAD_DIM + 4) // 4
    scales = page * page_floats + PAGE_SIZE * HEAD_DIM // 4
    cache.reshape(-1).view(torch.float32)[scales + row] = value


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("group", [1, 4])
def test_gluon_owned_page_scorer_is_default_and_matches_reference(heads, group, device):
    table, requests, lens, q, q_scales, weights, cache, keys = _inputs(
        heads, group, device
    )
    scale, topk = 0.1, 256
    # Non-finite key scales on request 0's logical tokens 5 and 130 (physical
    # pages 12 and 3): never selected.
    poisoned = {5: float("nan"), 130: float("inf")}
    for position, value in poisoned.items():
        page = int(table[0, position // PAGE_SIZE])
        _set_key_scale(cache, page * PAGE_SIZE + position % PAGE_SIZE, value)

    pages = plan_dsa_index_candidates(
        q, weights, cache, table, requests, lens, page_size=PAGE_SIZE
    )
    assert pages is not None and pages.kernel == GLUON
    # A BF16 query is outside the Gluon leaf's contract and keeps the
    # portable scorer, which plans no pages.
    assert (
        plan_dsa_index_candidates(
            q.to(torch.bfloat16),
            weights,
            cache,
            table,
            requests,
            lens,
            page_size=PAGE_SIZE,
        )
        is None
    )

    kwargs = dict(
        page_size=PAGE_SIZE,
        topk=topk,
        softmax_scale=scale,
        initial_tokens=0,
        local_tokens=0,
        solution=None,
        pages=pages,
        q_scales=q_scales,
        query_group_size=group,
    )
    rows = requests.numel()
    # The head weights as a slice of a fused key and head-weight projection.
    fused = torch.randn(rows, HEAD_DIM + heads, device=device)
    fused[:, HEAD_DIM:] = weights
    strided = fused[:, HEAD_DIM:]
    assert not strided.is_contiguous()
    # The scores' share of a wider send buffer, as a collective packs it.
    buffer = torch.full((rows * topk + 64,), float("nan"), device=device)
    scores_out = buffer[: rows * topk].view(rows, topk)
    candidates = [
        dsa_index_candidates(q, weights, cache, table, requests, lens, **kwargs),
        dsa_index_candidates(
            q, strided, cache, table, requests, lens, scores_out=scores_out, **kwargs
        ),
    ]
    sent_offsets, sent = candidates[1]
    assert sent.data_ptr() == scores_out.data_ptr()
    assert sent_offsets.is_contiguous()
    assert buffer[rows * topk :].isnan().all()

    reference = _reference(table, requests, lens, q, q_scales, weights, keys, scale)
    for offsets, scores in candidates:
        for row, (positions, expected) in enumerate(reference):
            if int(requests[row]) == 0:
                clean = ~torch.isin(
                    positions, torch.tensor(list(poisoned), device=device)
                )
                positions, expected = positions[clean], expected[clean]
            valid = offsets[row] >= 0
            got = offsets[row][valid].long()
            assert got.numel() == min(topk, positions.numel())
            assert got.unique().numel() == got.numel()
            assert scores[row][valid].isfinite().all()
            assert (scores[row][~valid] == float("-inf")).all()
            by_position = torch.full(
                (int(lens[row]),), float("nan"), device=device, dtype=torch.float32
            )
            by_position[positions] = expected
            chosen = by_position[got]
            assert (
                not chosen.isnan().any()
            ), f"row {row} selected a non-owned or poisoned token"
            tol = 1.0e-4 * float(expected.abs().max()) + 1.0e-5
            torch.testing.assert_close(scores[row][valid], chosen, rtol=0, atol=tol)
            cutoff = float(torch.topk(expected, got.numel()).values[-1])
            assert (
                float(chosen.min()) >= cutoff - tol
            ), f"row {row} missed a better token"


def _prefill_inputs(heads: int, device: str):
    """Prefill rows of three requests over a 96-page table, several key chunks
    wide: a long history whose block of rows ends mid-block, five rows, and a
    fresh prompt whose first page another rank owns (its early rows own
    nothing)."""
    torch.manual_seed(2311 + heads)
    columns = 96
    table = torch.randperm(320, device=device)[: 3 * columns].reshape(3, columns)
    table = torch.where(torch.rand(table.shape, device=device) < 0.4, -1, table).to(
        torch.int32
    )
    table[2, 0] = -1
    rows = _rows([(0, 5000, 77), (1, 64, 5), (2, 1, 130)], heads, device)
    cache, _ = _pack_index_k_cache(
        torch.randn(320 * PAGE_SIZE, HEAD_DIM, device=device), PAGE_SIZE
    )
    # A non-finite key scale: both scorers store its score as -inf.
    _set_key_scale(cache, int(table[0][table[0] >= 0][3]) * PAGE_SIZE + 7, float("nan"))
    return table, *rows, cache


@pytest.mark.parametrize("heads", [32, 64])
def test_gluon_owned_page_prefill_rows_score_like_one_row_groups(heads, device):
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa import index_candidates

    table, requests, lens, q, q_scales, weights, cache = _prefill_inputs(heads, device)
    pages = plan_dsa_index_candidates(
        q, weights, cache, table, requests, lens, page_size=PAGE_SIZE
    )
    lengths = pages.lengths.reshape(-1)
    assert pages.pages.shape[1] * PAGE_SIZE > 4096
    assert (lengths == 0).any()
    launch = index_candidates.launch_gluon_dsa_index_candidates_gfx950
    args = (q, q_scales, weights, cache, pages.pages, lengths)

    one_row = launch(
        *args,
        query_group_size=1,
        query_requests=None,
        request_count=None,
        page_size=PAGE_SIZE,
        softmax_scale=0.1,
    )
    blocks = launch(
        *args,
        query_group_size=1,
        query_requests=requests,
        request_count=table.shape[0],
        page_size=PAGE_SIZE,
        softmax_scale=0.1,
    )
    columns = torch.arange(one_row.shape[1], device=device)
    valid = columns[None, :] < lengths[:, None]
    assert (one_row[valid] == float("-inf")).any()
    torch.testing.assert_close(blocks[valid], one_row[valid], rtol=0, atol=0)


def _owned_batch(requests: int, rows: int, columns: int, device: str):
    """``requests`` requests of ``rows`` consecutive rows each over a
    ``columns``-page table of 320 cache pages, a third of them unowned."""
    torch.manual_seed(97 * requests + 13 * rows + columns)
    table = torch.randint(0, 320, (requests, columns), device=device)
    table = torch.where(torch.rand(table.shape, device=device) < 0.3, -1, table)
    table = table.to(torch.int32)
    first = torch.randint(1, columns * PAGE_SIZE - rows, (requests,), device=device)
    steps = [(r, length, rows) for r, length in enumerate(first.tolist())]
    return table, *_rows(steps, 32, device)


def test_gluon_owned_page_scorer_compiles_once_across_batches(device):
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.index_candidates import (
        launch_gluon_dsa_index_candidates_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.standard_cache_logits import (
        gluon_dsa_decode_topk_fp8_rows_gfx950,
        gluon_dsa_owned_prefill_topk_fp8_rows_gfx950,
    )

    cache, _ = _pack_index_k_cache(
        torch.randn(320 * PAGE_SIZE, HEAD_DIM, device=device), PAGE_SIZE
    )

    def score(requests, rows, group, columns):
        table, request_ids, lens, q, q_scales, weights = _owned_batch(
            requests, rows, columns, device
        )
        pages = plan_dsa_index_candidates(
            q, weights, cache, table, request_ids, lens, page_size=PAGE_SIZE
        )
        return launch_gluon_dsa_index_candidates_gfx950(
            q,
            q_scales,
            weights,
            cache,
            pages.pages,
            pages.lengths.reshape(-1),
            page_size=PAGE_SIZE,
            softmax_scale=0.1,
            query_group_size=group,
            query_requests=request_ids,
            request_count=requests,
        )

    # Decode (one row per request), verify (four) and prefill (a block of
    # rows per request) launches; warm Triton's integer classes (1, a
    # multiple of 16, neither) and the decode scorer's chunk loop.
    batches = [(1, 1, 1), (16, 1, 1), (3, 1, 1), (16, 4, 4), (3, 4, 4)]
    batches += [(1, 64, 1), (1, 80, 1), (3, 70, 1)]
    for columns in (16, 17, 96):
        for batch in batches:
            score(*batch, columns)
    # Decode at its full static width splits spans over the batch, one binary
    # per bucket of request counts.
    for requests in (5, 32):
        score(requests, 1, 1, 4096)
        score(requests, 4, 4, 4096)
    with assert_no_triton_compile(
        gluon_dsa_decode_topk_fp8_rows_gfx950,
        gluon_dsa_owned_prefill_topk_fp8_rows_gfx950,
    ):
        for columns in (33, 48):
            for requests in (5, 32):
                score(requests, 1, 1, columns)
                score(requests, 4, 4, columns)
            for requests, rows in ((1, 65), (2, 64), (2, 100), (3, 130)):
                score(requests, rows, 1, columns)
        for requests in (12, 40):
            score(requests, 1, 1, 4100)
            score(requests, 4, 4, 4100)


@pytest.mark.parametrize("group", [4, 5])
def test_gluon_owned_page_decode_runs_over_the_batch_score_alike(
    group, device, monkeypatch
):
    """At decode's full static width, several requests' spans are cut into
    runs of one length over the batch: a long request, short ones and an
    almost idle graph row score exactly as with runs per request."""
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa import index_candidates

    torch.manual_seed(41 + group)
    columns = 4096
    firsts = [150_000, 3000, 0, 9000, 700, 64, 20_000, 1]
    table = torch.randint(0, 320, (len(firsts), columns), device=device)
    table = torch.where(torch.rand(table.shape, device=device) < 0.3, -1, table)
    table = table.to(torch.int32)
    steps = [(r, first, group) for r, first in enumerate(firsts)]
    requests, lens, q, q_scales, weights = _rows(steps, 32, device)
    cache, _ = _pack_index_k_cache(
        torch.randn(320 * PAGE_SIZE, HEAD_DIM, device=device), PAGE_SIZE
    )
    pages = plan_dsa_index_candidates(
        q, weights, cache, table, requests, lens, page_size=PAGE_SIZE
    )
    lengths = pages.lengths.reshape(-1)

    def score():
        return index_candidates.launch_gluon_dsa_index_candidates_gfx950(
            q,
            q_scales,
            weights,
            cache,
            pages.pages,
            lengths,
            page_size=PAGE_SIZE,
            softmax_scale=0.1,
            query_group_size=group,
            query_requests=requests,
            request_count=len(firsts),
        )

    batch_runs = score()
    monkeypatch.setattr(index_candidates, "_BALANCE_MIN_REQUESTS", len(firsts) + 1)
    request_runs = score()
    valid = torch.arange(batch_runs.shape[1], device=device)[None, :] < lengths[:, None]
    assert int(lengths.max()) > 64 * int(lengths.float().median())
    torch.testing.assert_close(
        batch_runs[valid], request_runs[valid], rtol=0, atol=0, equal_nan=True
    )


def test_gluon_owned_page_prefill_rows_store_past_int32_offsets(device):
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.index_candidates import (
        launch_gluon_dsa_index_candidates_gfx950,
    )

    # A prefill chunk over a 1M-token table: the last rows' scores start past
    # 2**31 logits elements. Short rows keep the scoring cheap.
    rows, columns, tail = 2176, 16384, 64
    torch.manual_seed(31)
    cache, _ = _pack_index_k_cache(
        torch.randn(320 * PAGE_SIZE, HEAD_DIM, device=device), PAGE_SIZE
    )
    pages = torch.randint(0, 320, (1, columns), device=device, dtype=torch.int32)
    pages = pages.expand(rows, columns).contiguous()
    lengths = 70 + torch.arange(rows, device=device, dtype=torch.int32) % 300
    q, q_scales = _fp8_query(rows, 32, device)
    weights = torch.randn(rows, 32, device=device)
    requests = torch.zeros(rows, dtype=torch.int32, device=device)

    def score(rows_slice):
        return launch_gluon_dsa_index_candidates_gfx950(
            q[rows_slice],
            q_scales[rows_slice],
            weights[rows_slice],
            cache,
            pages[rows_slice],
            lengths[rows_slice],
            page_size=PAGE_SIZE,
            softmax_scale=0.1,
            query_group_size=1,
            query_requests=requests[rows_slice],
            request_count=1,
        )

    full = score(slice(None))
    assert (rows - tail) * full.stride(0) > 2**31
    last = score(slice(rows - tail, None))
    positions = torch.arange(400, device=device)
    valid = positions[None, :] < lengths[-tail:, None]
    torch.testing.assert_close(
        full[-tail:, :400][valid], last[:, :400][valid], rtol=0, atol=0
    )


def test_gluon_owned_page_scorer_forces_mandatory_windows(device):
    table, requests, lens, q, q_scales, weights, cache, _ = _inputs(32, 4, device)
    initial, local, topk = 4, 64, 256

    offsets, scores = dsa_index_candidates(
        q,
        weights,
        cache,
        table,
        requests,
        lens,
        page_size=PAGE_SIZE,
        topk=topk,
        softmax_scale=0.1,
        initial_tokens=initial,
        local_tokens=local,
        solution=None,
        q_scales=q_scales,
        query_group_size=4,
    )

    for row in range(requests.numel()):
        pages = table[int(requests[row])]
        positions = torch.arange(int(lens[row]), device=device)
        owned = positions[pages[positions // PAGE_SIZE] >= 0]
        forced = owned[(owned < initial) | (owned >= int(lens[row]) - local)]
        got = offsets[row][offsets[row] >= 0]
        assert forced.numel() > 0
        assert torch.isin(forced, got).all(), f"row {row} dropped a mandatory token"
        assert (scores[row][torch.isin(offsets[row], forced)] == float("inf")).all()
