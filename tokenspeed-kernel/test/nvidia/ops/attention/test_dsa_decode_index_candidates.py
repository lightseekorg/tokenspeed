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

"""Check the CuTe DSL KVP decode candidates against the DeepGEMM candidates.

Both score the same FP32 logits over a rank's compact owned pages, so every
row's sorted candidate scores must match bit for bit, and so must the
positions above the row's k-th score; ties at it may resolve differently, in
equal number. Rows of 1, 4 and 8 per request, skewed ownership, short and
empty rows, and a send buffer's rows as the scores' storage.
"""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_cute_compile, is_nvidia

if not is_nvidia():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)
pytest.importorskip("cutlass")
from tokenspeed_kernel import platform  # noqa: E402

if (
    not platform.ArchVersion(10, 0)
    <= platform.current_platform().arch_version
    <= platform.ArchVersion(10, 3)
):
    pytest.skip("requires SM100 or SM103", allow_module_level=True)

from tokenspeed_kernel.ops.attention.dsa import (  # noqa: E402
    dsa_decode_index_candidates,
    dsa_decode_index_candidates_supported,
    dsa_index_candidates,
    plan_dsa_index_candidates,
)

PAGE, HEADS, DIM, TOPK = 64, 32, 128, 2048
SOFTMAX_SCALE = DIM**-0.5 * HEADS**-0.5


def _inputs(requests, group, lengths, degree, rank, seed):
    """Build a rank's plane, local page table and decode rows.

    ``requests`` requests' last rows see ``lengths`` tokens; page column ``v``
    lives on rank ``v % degree``, as in a cyclic KVP placement.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    columns = max(-(-length // PAGE) for length in lengths)
    local_pages = requests * columns + 1
    plane = torch.randint(
        0, 0x7E, (local_pages * PAGE, DIM + 4), dtype=torch.uint8, device="cuda",
        generator=generator,
    )  # fmt: skip
    signs = torch.randint(0, 2, plane.shape, device="cuda", generator=generator)
    plane |= (signs * 0x80).to(torch.uint8)
    # UE8M0 scale bytes of 2**0 for every key.
    plane.view(local_pages, -1)[:, PAGE * DIM :] = 127
    table = torch.full((requests, columns), -1, dtype=torch.int32, device="cuda")
    free = 1
    for request, length in enumerate(lengths):
        for column in range(-(-length // PAGE)):
            if column % degree == rank:
                table[request, column] = free
                free += 1
    rows = requests * group
    row_requests = torch.arange(rows, device="cuda", dtype=torch.int32) // group
    causal = (
        torch.tensor(lengths, device="cuda", dtype=torch.int32)[row_requests.long()]
        - (group - 1)
        + torch.arange(rows, device="cuda", dtype=torch.int32) % group
    )
    causal = causal.clamp_min(0).int()
    q = torch.randn((rows, HEADS, DIM), device="cuda", generator=generator)
    weights = torch.randn((rows, HEADS), device="cuda", generator=generator)
    return plane, table, row_requests, causal, q.bfloat16(), weights.bfloat16()


def _check_rows(positions, scores, ref_positions, ref_scores):
    for row in range(positions.shape[0]):
        got, order = scores[row].sort(descending=True)
        want, ref_order = ref_scores[row].sort(descending=True)
        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), row
        live = want > -float("inf")
        assert torch.equal(
            positions[row][order][~live], ref_positions[row][ref_order][~live]
        )
        if not live.any():
            continue
        threshold = want[live][-1]
        above = got > threshold
        assert set(positions[row][order][above].tolist()) == set(
            ref_positions[row][ref_order][want > threshold].tolist()
        ), row
        assert (positions[row][order][live] >= 0).all()


@pytest.mark.parametrize("group", [1, 4, 8])
@pytest.mark.parametrize("degree,rank", [(1, 0), (8, 3)])
def test_kvp_decode_candidates_match_the_deep_gemm_candidates(group, degree, rank):
    # Long, past-topk, short (fewer keys than topk on this rank) and empty
    # requests; a window past a page edge.
    lengths = [40000, 2048 * degree + 77, 900, group, 130 * PAGE + 3]
    plane, table, requests, causal, q, weights = _inputs(
        len(lengths), group, lengths, degree, rank, seed=group * 10 + degree
    )
    pages = plan_dsa_index_candidates(
        q, weights, plane, table, requests, causal, page_size=PAGE,
        solution="deep_gemm",
    )  # fmt: skip
    assert pages is not None
    assert dsa_decode_index_candidates_supported(
        index_heads=HEADS,
        head_dim=DIM,
        page_size=PAGE,
        topk=TOPK,
        query_group_size=group,
        index_k_format="fp8_scaled",
        solution="cute_dsl",
    )
    ref_positions, ref_scores = dsa_index_candidates(
        q, weights, plane, table, requests, causal, page_size=PAGE, topk=TOPK,
        softmax_scale=SOFTMAX_SCALE, initial_tokens=0, local_tokens=0,
        solution="deep_gemm", pages=pages,
    )  # fmt: skip
    buffer = torch.full((q.shape[0], TOPK + 4), 11.0, device="cuda")
    positions, scores = dsa_decode_index_candidates(
        q,
        weights,
        plane,
        pages,
        query_group_size=group,
        page_size=PAGE,
        topk=TOPK,
        softmax_scale=SOFTMAX_SCALE,
        index_k_format="fp8_scaled",
        solution="cute_dsl",
        scores_out=buffer[:, 2:-2],
    )
    assert scores.data_ptr() == buffer[:, 2:-2].data_ptr()
    assert (buffer[:, :2] == 11.0).all() and (buffer[:, -2:] == 11.0).all()
    _check_rows(positions, scores, ref_positions, ref_scores)


def test_kvp_decode_candidates_compile_once_across_batches():
    def run(requests, seed):
        lengths = [3000 + 997 * r for r in range(requests)]
        plane, table, row_requests, causal, q, weights = _inputs(
            requests, 8, lengths, 8, 2, seed
        )
        pages = plan_dsa_index_candidates(
            q, weights, plane, table, row_requests, causal, page_size=PAGE,
            solution="deep_gemm",
        )  # fmt: skip
        dsa_decode_index_candidates(
            q,
            weights,
            plane,
            pages,
            query_group_size=8,
            page_size=PAGE,
            topk=TOPK,
            softmax_scale=SOFTMAX_SCALE,
            index_k_format="fp8_scaled",
            solution="cute_dsl",
            scores_out=None,
        )

    run(1, 1)
    with assert_no_cute_compile():
        for requests in (2, 5, 16):
            run(requests, requests)


def test_kvp_decode_candidates_refuse_other_sizes():
    sizes = dict(
        index_heads=HEADS,
        head_dim=DIM,
        page_size=PAGE,
        topk=TOPK,
        query_group_size=8,
        index_k_format="fp8_scaled",
        solution=None,
    )
    assert dsa_decode_index_candidates_supported(**sizes)
    for name, value in (
        ("query_group_size", 9),
        ("topk", 512),
        ("index_heads", 64),
        ("index_k_format", "mxfp4"),
    ):
        assert not dsa_decode_index_candidates_supported(**{**sizes, name: value})
