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

from tokenspeed_kernel.numerics.torch.attention.dsa import (  # isort: skip
    dsa_index_scores,
)
from tokenspeed_kernel.ops.attention.dsa import (  # isort: skip
    dsa_index_candidates,
    plan_dsa_index_candidates,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.index_candidates import (  # isort: skip
    _owned_query_group_metadata,
)

PAGE_SIZE = 64
HEAD_DIM = 128
GLUON = "gluon_dsa_index_candidates_gfx950"


def _pack_index_k_cache(index_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The packed FP8 Index-K cache and the key values it dequantizes to."""
    num_slots = index_k.shape[0]
    num_pages = num_slots // PAGE_SIZE
    row_bytes = HEAD_DIM + 4
    packed = torch.empty(
        (num_slots, row_bytes), device=index_k.device, dtype=torch.uint8
    )
    x = index_k.float()
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-6) / 448.0
    x_fp8 = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    flat = packed.reshape(-1)
    page_bytes = PAGE_SIZE * row_bytes
    torch.as_strided(
        flat.view(torch.float8_e4m3fn),
        (num_pages, PAGE_SIZE, HEAD_DIM),
        (page_bytes, HEAD_DIM, 1),
    ).copy_(x_fp8.reshape(num_pages, PAGE_SIZE, HEAD_DIM))
    torch.as_strided(
        flat.view(torch.float32),
        (num_pages, PAGE_SIZE, 1),
        (page_bytes // 4, 1, 1),
        (PAGE_SIZE * HEAD_DIM) // 4,
    ).copy_(scale.reshape(num_pages, PAGE_SIZE, 1))
    return packed, x_fp8.float() * scale


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
    steps = [(0, 1000), (1, 977), (0, 523), (1, 64)]
    requests, lens = [], []
    for request, length in steps:
        for row in range(group):
            requests.append(request)
            lens.append(length + row)
    requests = torch.tensor(requests, dtype=torch.int32, device=device)
    lens = torch.tensor(lens, dtype=torch.int32, device=device)
    rows = requests.numel()

    q = torch.randn(rows, heads, HEAD_DIM, device=device)
    q_scales = q.abs().amax(dim=-1).clamp_min(1.0e-6) / 448.0
    q_fp8 = (q / q_scales[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    weights = torch.randn(rows, heads, device=device)
    cache, keys = _pack_index_k_cache(
        torch.randn(48 * PAGE_SIZE, HEAD_DIM, device=device)
    )
    return table, requests, lens, q_fp8, q_scales.float(), weights, cache, keys


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


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("group", [1, 4])
def test_gluon_owned_page_scorer_is_default_and_matches_reference(heads, group, device):
    table, requests, lens, q, q_scales, weights, cache, keys = _inputs(
        heads, group, device
    )
    scale, topk = 0.1, 256

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

    offsets, scores = dsa_index_candidates(
        q,
        weights,
        cache,
        table,
        requests,
        lens,
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
    reference = _reference(table, requests, lens, q, q_scales, weights, keys, scale)
    for row, (positions, expected) in enumerate(reference):
        valid = offsets[row] >= 0
        got = offsets[row][valid].long()
        assert got.numel() == min(topk, positions.numel())
        assert got.unique().numel() == got.numel()
        by_position = torch.full(
            (int(lens[row]),), float("nan"), device=device, dtype=torch.float32
        )
        by_position[positions] = expected
        chosen = by_position[got]
        assert not chosen.isnan().any(), f"row {row} selected a non-owned token"
        tol = 1.0e-4 * float(expected.abs().max()) + 1.0e-5
        torch.testing.assert_close(scores[row][valid], chosen, rtol=0, atol=tol)
        cutoff = float(torch.topk(expected, got.numel()).values[-1])
        assert float(chosen.min()) >= cutoff - tol, f"row {row} missed a better token"


@pytest.mark.parametrize("heads", [32, 64])
def test_gluon_owned_page_scorer_grouped_rows_match_single_rows(heads, device):
    table, requests, lens, q, q_scales, weights, cache, _ = _inputs(heads, 4, device)
    kwargs = dict(
        page_size=PAGE_SIZE,
        topk=256,
        softmax_scale=0.1,
        initial_tokens=0,
        local_tokens=0,
        solution=None,
        q_scales=q_scales,
    )
    single = dsa_index_candidates(
        q, weights, cache, table, requests, lens, query_group_size=1, **kwargs
    )
    grouped = dsa_index_candidates(
        q, weights, cache, table, requests, lens, query_group_size=4, **kwargs
    )
    for actual, expected in zip(_by_offset(*grouped), _by_offset(*single), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _by_offset(offsets, scores):
    """Candidates ordered by offset: order within a row is unspecified, and
    -1 padding scores -inf."""
    order = offsets.argsort(dim=1, stable=True)
    return offsets.gather(1, order), scores.gather(1, order)


def test_gluon_owned_page_scorer_takes_strided_head_weight_rows(device):
    # The head weights as a slice of a fused key and head-weight projection.
    table, requests, lens, q, q_scales, weights, cache, _ = _inputs(32, 4, device)
    fused = torch.randn(weights.shape[0], HEAD_DIM + 32, device=device)
    fused[:, HEAD_DIM:] = weights
    strided = fused[:, HEAD_DIM:]
    assert not strided.is_contiguous()
    kwargs = dict(
        page_size=PAGE_SIZE,
        topk=256,
        softmax_scale=0.1,
        initial_tokens=0,
        local_tokens=0,
        solution=None,
        q_scales=q_scales,
        query_group_size=4,
    )
    expected = dsa_index_candidates(q, weights, cache, table, requests, lens, **kwargs)
    actual = dsa_index_candidates(q, strided, cache, table, requests, lens, **kwargs)
    for got, want in zip(_by_offset(*actual), _by_offset(*expected), strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_gluon_owned_page_scorer_writes_scores_into_a_send_buffer(device):
    table, requests, lens, q, q_scales, weights, cache, _ = _inputs(32, 4, device)
    rows, topk = requests.numel(), 256
    kwargs = dict(
        page_size=PAGE_SIZE,
        topk=topk,
        softmax_scale=0.1,
        initial_tokens=0,
        local_tokens=0,
        solution=None,
        q_scales=q_scales,
        query_group_size=4,
    )
    offsets, scores = dsa_index_candidates(
        q, weights, cache, table, requests, lens, **kwargs
    )
    # The scores' share of a wider send buffer, as a collective packs it.
    buffer = torch.full((rows * topk + 64,), float("nan"), device=device)
    scores_out = buffer[: rows * topk].view(rows, topk)

    positions, written = dsa_index_candidates(
        q, weights, cache, table, requests, lens, scores_out=scores_out, **kwargs
    )

    assert written.data_ptr() == scores_out.data_ptr()
    assert positions.is_contiguous()
    for got, want in zip(
        _by_offset(positions, written), _by_offset(offsets, scores), strict=True
    ):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert buffer[rows * topk :].isnan().all()


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
    steps = [(0, 5000, 77), (1, 64, 5), (2, 1, 130)]
    requests = torch.cat(
        [torch.full((count,), r, dtype=torch.int32) for r, _, count in steps]
    ).to(device)
    lens = torch.cat(
        [
            torch.arange(first, first + count, dtype=torch.int32)
            for _, first, count in steps
        ]
    ).to(device)
    rows = requests.numel()
    q = torch.randn(rows, heads, HEAD_DIM, device=device)
    q_scales = q.abs().amax(dim=-1).clamp_min(1.0e-6) / 448.0
    q_fp8 = (q / q_scales[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    weights = torch.randn(rows, heads, device=device)
    cache, _ = _pack_index_k_cache(
        torch.randn(320 * PAGE_SIZE, HEAD_DIM, device=device)
    )
    # A non-finite key scale: both scorers store its score as -inf.
    _set_key_scale(cache, int(table[0][table[0] >= 0][3]) * PAGE_SIZE + 7, float("nan"))
    return table, requests, lens, q_fp8, q_scales.float(), weights, cache


def _count_prefill_launches(monkeypatch) -> list[int]:
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa import index_candidates

    launches = []
    launch = index_candidates._launch_prefill_rows

    def counted(q, *args, **kwargs):
        launches.append(q.shape[0])
        return launch(q, *args, **kwargs)

    monkeypatch.setattr(index_candidates, "_launch_prefill_rows", counted)
    return launches


@pytest.mark.parametrize("heads", [32, 64])
def test_gluon_owned_page_prefill_rows_score_like_one_row_groups(
    heads, device, monkeypatch
):
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.index_candidates import (
        launch_gluon_dsa_index_candidates_gfx950,
    )

    table, requests, lens, q, q_scales, weights, cache = _prefill_inputs(heads, device)
    pages = plan_dsa_index_candidates(
        q, weights, cache, table, requests, lens, page_size=PAGE_SIZE
    )
    lengths = pages.lengths.reshape(-1)
    assert pages.pages.shape[1] * PAGE_SIZE > 4096
    assert (lengths == 0).any()
    launches = _count_prefill_launches(monkeypatch)
    kwargs = dict(page_size=PAGE_SIZE, softmax_scale=0.1)
    args = (q, q_scales, weights, cache, pages.pages, lengths)

    one_row = launch_gluon_dsa_index_candidates_gfx950(*args, **kwargs)
    assert launches == []
    blocks = launch_gluon_dsa_index_candidates_gfx950(
        *args, query_requests=requests, request_count=table.shape[0], **kwargs
    )
    assert launches == [requests.numel()]

    columns = torch.arange(one_row.shape[1], device=device)
    valid = columns[None, :] < lengths[:, None]
    assert (one_row[valid] == float("-inf")).any()
    torch.testing.assert_close(blocks[valid], one_row[valid], rtol=0, atol=0)


def test_gluon_owned_page_scorer_keeps_one_row_groups_below_prefill_size(device):
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.index_candidates import (
        _scores_prefill_rows,
    )

    assert _scores_prefill_rows(4096, 1, 32)
    assert _scores_prefill_rows(212, 3, 32)
    # Decode: a row per request, at any batch size.
    assert not _scores_prefill_rows(256, 256, 32)
    assert not _scores_prefill_rows(63, 1, 32)
    assert not _scores_prefill_rows(4096, None, 32)
    assert not _scores_prefill_rows(100, 26, 64)


def test_gluon_owned_page_candidates_select_alike_through_prefill_rows(
    device, monkeypatch
):
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa import index_candidates

    table, requests, lens, q, q_scales, weights, cache = _prefill_inputs(32, device)
    kwargs = dict(
        page_size=PAGE_SIZE,
        topk=512,
        softmax_scale=0.1,
        initial_tokens=0,
        local_tokens=0,
        solution=None,
        q_scales=q_scales,
    )
    launches = _count_prefill_launches(monkeypatch)
    blocks = dsa_index_candidates(q, weights, cache, table, requests, lens, **kwargs)
    assert launches == [requests.numel()]
    monkeypatch.setattr(index_candidates, "_scores_prefill_rows", lambda *_: False)
    one_row = dsa_index_candidates(q, weights, cache, table, requests, lens, **kwargs)
    assert launches == [requests.numel()]
    for got, want in zip(_by_offset(*blocks), _by_offset(*one_row), strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def _owned_batch(requests: int, rows: int, columns: int, device: str):
    """``requests`` requests of ``rows`` consecutive rows each over a
    ``columns``-page table of 320 cache pages, a third of them unowned."""
    torch.manual_seed(97 * requests + 13 * rows + columns)
    table = torch.randint(0, 320, (requests, columns), device=device)
    table = torch.where(torch.rand(table.shape, device=device) < 0.3, -1, table)
    table = table.to(torch.int32)
    request_ids = torch.arange(requests, device=device, dtype=torch.int32)
    request_ids = request_ids.repeat_interleave(rows)
    first = torch.randint(1, columns * PAGE_SIZE - rows, (requests,), device=device)
    lens = (first[:, None] + torch.arange(rows, device=device)[None, :]).reshape(-1)
    q = torch.randn(requests * rows, 32, HEAD_DIM, device=device)
    q_scales = q.abs().amax(dim=-1).clamp_min(1.0e-6) / 448.0
    q_fp8 = (q / q_scales[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    weights = torch.randn(requests * rows, 32, device=device)
    return table, request_ids, lens.to(torch.int32), q_fp8, q_scales.float(), weights


def test_gluon_owned_page_scorer_compiles_once_across_batches(device):
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.index_candidates import (
        launch_gluon_dsa_index_candidates_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.standard_cache_logits import (
        gluon_dsa_decode_topk_fp8_rows_gfx950,
        gluon_dsa_owned_prefill_topk_fp8_rows_gfx950,
    )

    cache, _ = _pack_index_k_cache(
        torch.randn(320 * PAGE_SIZE, HEAD_DIM, device=device)
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
    with assert_no_triton_compile(
        gluon_dsa_decode_topk_fp8_rows_gfx950,
        gluon_dsa_owned_prefill_topk_fp8_rows_gfx950,
    ):
        for columns in (18, 33, 48, 95):
            for requests in (2, 5, 32, 37):
                score(requests, 1, 1, columns)
                score(requests, 4, 4, columns)
            for requests, rows in ((1, 65), (2, 64), (2, 100), (3, 130)):
                score(requests, rows, 1, columns)


def test_gluon_owned_page_prefill_rows_store_past_int32_offsets(device):
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.index_candidates import (
        launch_gluon_dsa_index_candidates_gfx950,
    )

    # A prefill chunk over a 1M-token table: the last rows' scores start past
    # 2**31 logits elements. Short rows keep the scoring cheap.
    rows, columns, tail = 2176, 16384, 64
    torch.manual_seed(31)
    cache, _ = _pack_index_k_cache(
        torch.randn(320 * PAGE_SIZE, HEAD_DIM, device=device)
    )
    pages = torch.randint(0, 320, (1, columns), device=device, dtype=torch.int32)
    pages = pages.expand(rows, columns).contiguous()
    lengths = 70 + torch.arange(rows, device=device, dtype=torch.int32) % 300
    q = torch.randn(rows, 32, HEAD_DIM, device=device)
    q_scales = q.abs().amax(dim=-1).clamp_min(1.0e-6) / 448.0
    q = (q / q_scales[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    weights = torch.randn(rows, 32, device=device)
    requests = torch.zeros(rows, dtype=torch.int32, device=device)

    def score(rows_slice):
        return launch_gluon_dsa_index_candidates_gfx950(
            q[rows_slice],
            q_scales.float()[rows_slice],
            weights[rows_slice],
            cache,
            pages[rows_slice],
            lengths[rows_slice],
            page_size=PAGE_SIZE,
            softmax_scale=0.1,
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


def _set_key_scale(cache: torch.Tensor, slot: int, value: float) -> None:
    page, row = divmod(slot, PAGE_SIZE)
    page_floats = PAGE_SIZE * (HEAD_DIM + 4) // 4
    scales = page * page_floats + PAGE_SIZE * HEAD_DIM // 4
    cache.reshape(-1).view(torch.float32)[scales + row] = value


def _count_mask_launches(monkeypatch) -> list[int]:
    from tokenspeed_kernel.ops.attention.dsa._triton import index_candidates

    launches = []
    mask = index_candidates.mask_index_scores

    def counted(*args, **kwargs):
        launches.append(kwargs.get("mask_tail"))
        return mask(*args, **kwargs)

    monkeypatch.setattr(index_candidates, "mask_index_scores", counted)
    return launches


def test_gluon_owned_page_scorer_never_selects_non_finite_scores(device, monkeypatch):
    table, requests, lens, q, q_scales, weights, cache, keys = _inputs(32, 4, device)
    scale, topk = 0.1, 256
    # Request 0's logical tokens 5 and 130 sit on physical pages 12 and 3.
    poisoned = {5: float("nan"), 130: float("inf")}
    for position, value in poisoned.items():
        page = int(table[0, position // PAGE_SIZE])
        _set_key_scale(cache, page * PAGE_SIZE + position % PAGE_SIZE, value)
    launches = _count_mask_launches(monkeypatch)

    offsets, scores = dsa_index_candidates(
        q,
        weights,
        cache,
        table,
        requests,
        lens,
        page_size=PAGE_SIZE,
        topk=topk,
        softmax_scale=scale,
        initial_tokens=0,
        local_tokens=0,
        solution=None,
        pages=plan_dsa_index_candidates(
            q, weights, cache, table, requests, lens, page_size=PAGE_SIZE
        ),
        q_scales=q_scales,
        query_group_size=4,
    )

    assert launches == []
    reference = _reference(table, requests, lens, q, q_scales, weights, keys, scale)
    for row, (positions, expected) in enumerate(reference):
        if int(requests[row]) == 0:
            clean = ~torch.isin(positions, torch.tensor(list(poisoned), device=device))
            positions, expected = positions[clean], expected[clean]
        valid = offsets[row] >= 0
        got = offsets[row][valid].long()
        assert got.numel() == min(topk, positions.numel())
        assert scores[row][valid].isfinite().all()
        by_position = torch.full(
            (int(lens[row]),), float("nan"), device=device, dtype=torch.float32
        )
        by_position[positions] = expected
        chosen = by_position[got]
        assert not chosen.isnan().any(), f"row {row} selected a poisoned token"
        tol = 1.0e-4 * float(expected.abs().max()) + 1.0e-5
        torch.testing.assert_close(scores[row][valid], chosen, rtol=0, atol=tol)
        cutoff = float(torch.topk(expected, got.numel()).values[-1])
        assert float(chosen.min()) >= cutoff - tol, f"row {row} missed a better token"


def test_gluon_owned_page_scorer_forces_mandatory_windows(device, monkeypatch):
    table, requests, lens, q, q_scales, weights, cache, _ = _inputs(32, 4, device)
    initial, local, topk = 4, 64, 256
    launches = _count_mask_launches(monkeypatch)

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

    assert launches == [False]
    for row in range(requests.numel()):
        pages = table[int(requests[row])]
        positions = torch.arange(int(lens[row]), device=device)
        owned = positions[pages[positions // PAGE_SIZE] >= 0]
        forced = owned[(owned < initial) | (owned >= int(lens[row]) - local)]
        got = offsets[row][offsets[row] >= 0]
        assert forced.numel() > 0
        assert torch.isin(forced, got).all(), f"row {row} dropped a mandatory token"
        assert (scores[row][torch.isin(offsets[row], forced)] == float("inf")).all()


@pytest.mark.parametrize("group", [1, 4])
def test_grouped_owned_pages_are_a_live_view_of_each_groups_last_row(group):
    q = torch.empty(8, 32, 128, dtype=torch.float8_e4m3fn)
    pages = torch.arange(24, dtype=torch.int32).view(8, 3)
    lengths = torch.tensor([64, 64, 64, 65, 66, 66, 66, 66], dtype=torch.int32)
    table = _owned_query_group_metadata(q, pages, lengths, group)
    torch.testing.assert_close(table, pages[group - 1 :: group], atol=0, rtol=0)
    assert table.stride(0) == group * pages.stride(0)
    pages[group - 1, 0] = 987
    assert table[0, 0] == 987


def test_grouped_owned_pages_reject_incomplete_groups_and_nondense_plans():
    q = torch.empty(8, 32, 128, dtype=torch.float8_e4m3fn)
    pages = torch.empty(8, 4, dtype=torch.int32)
    lengths = torch.empty(8, dtype=torch.int32)
    with pytest.raises(ValueError, match="complete"):
        _owned_query_group_metadata(q[:7], pages[:7], lengths[:7], 4)
    with pytest.raises(ValueError, match="contiguous"):
        _owned_query_group_metadata(q, pages[:, ::2], lengths, 4)
