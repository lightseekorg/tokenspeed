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
from utils import is_cdna4

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
    # Candidate order within a row is unspecified; -1 padding scores -inf.
    for offsets, scores in (single, grouped):
        order = offsets.argsort(dim=1, stable=True)
        offsets.copy_(offsets.gather(1, order))
        scores.copy_(scores.gather(1, order))
    for actual, expected in zip(grouped, single, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
