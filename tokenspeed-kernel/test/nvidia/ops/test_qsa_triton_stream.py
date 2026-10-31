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

"""Numerics, graph refresh and workspace contracts for Triton QSA streams."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.qsa import qsa_block_topk_stream
from tokenspeed_kernel.ops.attention.qsa import triton as implementation
from tokenspeed_kernel.ops.attention.qsa._triton import nvidia_gluon as native
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="NVIDIA Triton stream validation"
)


@pytest.fixture(autouse=True)
def fp32_reference():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def _inputs(lengths, seq_lens, pages, page_size, heads, dim, dtype, request_dtype):
    torch.manual_seed(307)
    rows = sum(lengths)
    query = torch.randn(rows, heads + 1, dim, device="cuda", dtype=dtype)[:, :heads]
    keys = torch.randn((pages + 1) * page_size, 1, dim, device="cuda", dtype=dtype)
    keys[:page_size].zero_()
    table = torch.empty(len(lengths), pages + 3, device="cuda", dtype=torch.int32)[
        :, :pages
    ]
    table.copy_(
        torch.stack([torch.randperm(pages, device="cuda").int() + 1 for _ in lengths])
    )
    requests = torch.arange(
        len(lengths), device="cuda", dtype=request_dtype
    ).repeat_interleave(torch.tensor(lengths, device="cuda"))
    complete = torch.cat(
        [
            (length - qlen + torch.arange(qlen, device="cuda", dtype=torch.int32) + 1)
            // 4
            for qlen, length in zip(lengths, seq_lens, strict=True)
        ]
    )
    return query, keys, table, requests, complete


def _run(x, page_size, topk, width, budget, pdl):
    return qsa_block_topk_stream(
        *x,
        page_size=page_size,
        block_topk=topk,
        queries_per_request=width,
        max_partial_bytes=budget,
        enable_pdl=pdl,
        solution="triton",
        override="triton",
    )


def _check(x, selected, page_size, topk):
    q, keys, table, requests, complete = x
    limit = complete.clamp(0, table.shape[1] * page_size)
    torch.testing.assert_close(
        (selected >= 0).sum(1), limit.clamp_max(topk).long(), rtol=0, atol=0
    )
    assert ((selected == -1) | ((selected >= 0) & (selected < limit[:, None]))).all()
    sorted_ids = selected.sort(1).values
    assert ((sorted_ids[:, 1:] != sorted_ids[:, :-1]) | (sorted_ids[:, 1:] == -1)).all()
    boundary = torch.where(requests[1:] != requests[:-1])[0]
    sample = torch.cat(
        [
            torch.linspace(0, q.shape[0] - 1, min(8, q.shape[0]), device="cuda").long(),
            boundary,
            boundary + 1,
        ]
    ).unique()
    slots = (
        table[requests[sample].long()].long()[:, :, None] * page_size
        + torch.arange(page_size, device="cuda")
    ).flatten(1)
    values = (
        torch.einsum("rhd,rnd->rhn", q[sample].float(), keys[slots, 0].float())
        .clamp_min_(0)
        .sum(1)
    )
    values.masked_fill_(
        torch.arange(values.shape[1], device="cuda")[None] >= limit[sample, None],
        -float("inf"),
    )
    actual = values.gather(1, selected[sample].clamp_min(0).long()).masked_fill(
        selected[sample] < 0, -float("inf")
    )
    if values.shape[1] < topk:
        values = torch.nn.functional.pad(
            values, (0, topk - values.shape[1]), value=-float("inf")
        )
    torch.testing.assert_close(
        actual.sort(1, descending=True).values,
        values.topk(topk, dim=1).values,
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("request_dtype", [torch.int32, torch.int64])
def test_large_ragged_graph_refresh(prefix, pdl, request_dtype):
    seq_lens = [258433, 262143] if prefix else [8883, 14971]
    x = _inputs(
        [8883, 14971], seq_lens, 1024, 64, 4, 128, torch.bfloat16, request_dtype
    )
    _check(x, _run(x, 64, 512, None, 32 << 20, pdl), 64, 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _run(x, 64, 512, None, 32 << 20, pdl)
    q, keys, table, requests, complete = x
    counts = torch.tensor(
        [-3, 0, 1, 511, 512, 513, 65535, 65543], device="cuda", dtype=torch.int32
    )
    for shift in (0, 3, 5):
        complete.copy_(
            counts[(torch.arange(q.shape[0], device="cuda") + shift) % counts.numel()]
        )
        if shift == 3:
            q.zero_()
            table.copy_(table.flip(1).clone())
        elif shift == 5:
            q.normal_()
            requests.copy_(requests.flip(0).clone())
        graph.replay()
        _check(x, selected, 64, 512)


@pytest.mark.parametrize("budget", [0, 32 << 20])
def test_full_logical_ids_above_65535(budget):
    x = _inputs(
        [65, 72], [524287, 524288], 2048, 64, 4, 128, torch.bfloat16, torch.int32
    )
    q, keys, table, requests, complete = x
    q.fill_(1)
    table.copy_(torch.arange(1, 2049, device="cuda", dtype=torch.int32)[None])
    keys.zero_()
    keys[64 + 65536 :].fill_(1)
    selected = _run(x, 64, 512, None, budget, False)
    assert (selected > 65535).all()
    _check(x, selected, 64, 512)


@pytest.mark.parametrize(
    "dtype,heads,dim,page_size,topk",
    [
        (torch.float32, 3, 96, 16, 64),
        (torch.float16, 2, 80, 32, 128),
        (torch.bfloat16, 4, 128, 64, 512),
    ],
)
def test_portable_model_geometry(dtype, heads, dim, page_size, topk):
    x = _inputs(
        [2, 2],
        [4095, 4096],
        1024 // page_size,
        page_size,
        heads,
        dim,
        dtype,
        torch.int64,
    )
    _check(x, _run(x, page_size, topk, 2, 0, False), page_size, topk)


def test_zero_scratch_allocation_budget():
    x = _inputs([131, 133], [4095, 4096], 128, 64, 4, 128, torch.bfloat16, torch.int32)
    warm = _run(x, 64, 512, None, 0, False)
    del warm
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    selected = _run(x, 64, 512, None, 0, False)
    torch.cuda.synchronize()
    assert (
        torch.cuda.max_memory_allocated() - before
        == selected.numel() * selected.element_size()
    )
    _check(x, selected, 64, 512)


def test_runtime_shape_changes_do_not_compile():
    cases = [
        _inputs(lengths, [12000, 16000], pages, 64, 4, 128, torch.bfloat16, torch.int32)
        for lengths, pages in [([131, 133], 127), ([137, 200], 128), ([255, 257], 129)]
    ]
    _run(cases[0], 64, 512, None, 0, False)
    with assert_no_triton_compile(
        native._qsa_bounded_stream_kernel, native._qsa_warp_stream_kernel
    ):
        for x in cases[1:]:
            _check(x, _run(x, 64, 512, None, 0, False), 64, 512)


@pytest.mark.parametrize("late_boundary", [False, True])
@pytest.mark.parametrize("group_rows", [272, 4112])
@pytest.mark.parametrize("budget", [0, 32 << 20])
def test_proxy_boundary_requires_fp32_refinement(late_boundary, group_rows, budget):
    x = _inputs(
        [group_rows // 2, group_rows // 2],
        [32767, 32768],
        128,
        64,
        4,
        128,
        torch.bfloat16,
        torch.int32,
    )
    q, keys, table, requests, complete = x
    complete.fill_(8192)
    table.copy_(torch.arange(1, 129, device="cuda", dtype=torch.int32)[None])
    q.zero_()
    q[:, 0, 0] = 1
    q[:, 0, 1] = 1 / 128
    keys.zero_()
    keys[:, 0, 0] = 1
    keys[64 : 64 + 500, 0, 1] = 1
    keys[64 + 500 : 64 + 1000, 0, 1] = 0.25
    high_start = 7000 if late_boundary else 1000
    keys[64 + high_start : 64 + high_start + 500, 0, 1] = 0.28125
    selected = _run(x, 64, 512, None, budget, False)
    assert (((selected >= 0) & (selected < 500)).sum(1) == 500).all()
    assert (
        ((selected >= high_start) & (selected < high_start + 500)).sum(1) == 12
    ).all()
    _check(x, selected, 64, 512)


@pytest.mark.parametrize("pdl", [False, True])
def test_winners_concentrated_in_one_lane(pdl):
    x = _inputs(
        [131, 133], [262143, 262144], 1024, 64, 4, 128, torch.bfloat16, torch.int32
    )
    q, keys, table, requests, complete = x
    complete.fill_(65536)
    q.zero_()
    q[:, 0, 0] = 1
    table.copy_(torch.arange(1, 1025, device="cuda", dtype=torch.int32)[None])
    keys.zero_()
    keys[64 + torch.arange(7, 65536, 128, device="cuda"), 0, 0] = 1
    selected = _run(x, 64, 512, None, 0, pdl)
    assert (selected % 128 == 7).all()
    _check(x, selected, 64, 512)


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("budget", [0, 32 << 20])
def test_large_ragged_all_rows_against_fp32(prefix, budget):
    lengths = [8883, 14971]
    seq_lens = [258433, 262143] if prefix else lengths
    x = _inputs(lengths, seq_lens, 1024, 64, 4, 128, torch.bfloat16, torch.int32)
    q, keys, table, requests, complete = x
    selected = _run(x, 64, 512, None, budget, False)
    _check(x, selected, 64, 512)
    begin = 0
    for request, (length, seq_len) in enumerate(zip(lengths, seq_lens, strict=True)):
        columns = min(65536, seq_len // 4)
        logical = torch.arange(columns, device="cuda")
        slots = table[request, logical // 64].long() * 64 + logical % 64
        reference_keys = keys[slots, 0].float().T
        for first in range(begin, begin + length, 512):
            last = min(first + 512, begin + length)
            scores = torch.zeros(last - first, columns, device="cuda")
            for head in range(4):
                scores.add_(
                    (q[first:last, head].float() @ reference_keys).clamp_min_(0)
                )
            scores.masked_fill_(
                logical[None] >= complete[first:last, None], -float("inf")
            )
            actual_ids = selected[first:last]
            actual = scores.gather(1, actual_ids.clamp_min(0).long())
            actual.masked_fill_(actual_ids < 0, -float("inf"))
            torch.testing.assert_close(
                actual.sort(1, descending=True).values,
                scores.topk(512, dim=1).values,
                rtol=1e-5,
                atol=1e-4,
            )
        begin += length


@pytest.mark.parametrize("pdl", [False, True])
def test_uniform_four_queries_per_request(pdl):
    lengths = [4] * 33
    x = _inputs(
        lengths,
        [8191 - 3 * i for i in range(33)],
        32,
        64,
        4,
        128,
        torch.bfloat16,
        torch.int64,
    )
    _check(x, _run(x, 64, 512, 4, 0, pdl), 64, 512)


@pytest.mark.parametrize("pdl", [False, True])
def test_dense_pair_after_coarse_retention(pdl):
    x = _inputs(
        [131, 133], [32767, 32768], 128, 64, 4, 128, torch.bfloat16, torch.int32
    )
    q, keys, table, requests, complete = x
    complete.fill_(8192)
    q.zero_()
    q[:, 0, 0] = 1
    table.copy_(torch.arange(1, 129, device="cuda", dtype=torch.int32)[None])
    keys.zero_()
    keys[64 : 64 + 600, 0, 0] = 1
    keys[64 + 1024 : 64 + 1536, 0, 0] = 2
    selected = _run(x, 64, 512, None, 0, pdl)
    assert ((selected >= 1024) & (selected < 1536)).all()
    _check(x, selected, 64, 512)


@pytest.mark.parametrize("pdl", [False, True])
def test_all_winners_in_one_partial_range(pdl):
    x = _inputs(
        [131, 133], [262143, 262144], 1024, 64, 4, 128, torch.bfloat16, torch.int32
    )
    q, keys, table, requests, complete = x
    complete.fill_(65536)
    q.zero_()
    q[:, 0, 0] = 1
    table.copy_(torch.arange(1, 1025, device="cuda", dtype=torch.int32)[None])
    keys.zero_()
    keys[64 + 20000 : 64 + 20512, 0, 0] = 1
    selected = _run(x, 64, 512, None, 32 << 20, pdl)
    assert ((selected >= 20000) & (selected < 20512)).all()
    _check(x, selected, 64, 512)


def test_small_scratch_budget_with_query_tails():
    x = _inputs(
        [131, 133], [262143, 262144], 1024, 64, 4, 128, torch.bfloat16, torch.int32
    )
    budget = 256 << 10
    warm = _run(x, 64, 512, None, budget, False)
    del warm
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    selected = _run(x, 64, 512, None, budget, False)
    peak = torch.cuda.max_memory_allocated()
    # CUDA allocations are rounded; the budget bounds logical scratch bytes.
    assert peak - before <= selected.numel() * selected.element_size() + budget + 1024
    _check(x, selected, 64, 512)


@pytest.mark.parametrize(
    "layout", ["strided-dimension", "strided-rows", "misaligned-base"]
)
def test_key_layouts_preserve_portable_fallback(layout):
    x = _inputs([131, 133], [8191, 8192], 32, 64, 4, 128, torch.bfloat16, torch.int32)
    q, keys, table, requests, complete = x
    count = keys.shape[0]
    if layout == "strided-dimension":
        replacement = torch.empty(count, 1, 256, device="cuda", dtype=keys.dtype)[
            :, :, ::2
        ]
    elif layout == "strided-rows":
        replacement = torch.empty(count, 1, 129, device="cuda", dtype=keys.dtype)[
            :, :, :128
        ]
    else:
        replacement = torch.empty(count * 128 + 1, device="cuda", dtype=keys.dtype)[
            1:
        ].view(count, 1, 128)
    replacement.copy_(keys)
    inputs = q, replacement, table, requests, complete
    _check(inputs, _run(inputs, 64, 512, None, 32 << 20, False), 64, 512)


@pytest.mark.parametrize("pdl", [False, True])
def test_soa_ids_above_65535(pdl):
    x = _inputs(
        [2049, 2065], [524287, 524288], 2048, 64, 4, 128, torch.bfloat16, torch.int32
    )
    q, keys, table, requests, complete = x
    q.fill_(1)
    table.copy_(torch.arange(1, 2049, device="cuda", dtype=torch.int32)[None])
    keys.zero_()
    keys[64 + 65536 :].fill_(1)
    selected = _run(x, 64, 512, None, 0, pdl)
    assert (selected > 65535).all()
    _check(x, selected, 64, 512)


def test_soa_runtime_shapes_do_not_compile():
    cases = [
        _inputs(lengths, [32767, 32768], pages, 64, 4, 128, torch.bfloat16, torch.int32)
        for lengths, pages in [
            ([2049, 2063], 127),
            ([2057, 2065], 128),
            ([2129, 2143], 129),
        ]
    ]
    _run(cases[0], 64, 512, None, 0, False)
    with assert_no_triton_compile(native._qsa_soa_stream_kernel):
        for x in cases[1:]:
            _check(x, _run(x, 64, 512, None, 0, False), 64, 512)


def test_soa_zero_scratch_budget():
    x = _inputs(
        [2049, 2063], [32767, 32768], 128, 64, 4, 128, torch.bfloat16, torch.int32
    )
    warm = _run(x, 64, 512, None, 0, False)
    del warm
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    selected = _run(x, 64, 512, None, 0, False)
    torch.cuda.synchronize()
    assert (
        torch.cuda.max_memory_allocated() - before
        == selected.numel() * selected.element_size()
    )
    _check(x, selected, 64, 512)


@pytest.mark.parametrize("budget", [0, 32 << 20])
@pytest.mark.parametrize("pdl", [False, True])
def test_two_decode_requests_large_capacity(budget, pdl):
    x = _inputs([1, 1], [8883, 14971], 1024, 64, 4, 128, torch.bfloat16, torch.int64)
    _check(x, _run(x, 64, 512, 1, budget, pdl), 64, 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _run(x, 64, 512, 1, budget, pdl)
    x[-1].copy_(torch.tensor([0, 65536], device="cuda", dtype=torch.int32))
    x[2].copy_(x[2].flip(1).clone())
    graph.replay()
    _check(x, selected, 64, 512)


@pytest.mark.parametrize("positive_blocks", [64, 512])
def test_decode_discarded_candidates_require_full_selection(positive_blocks):
    x = _inputs([1, 1], [32767, 32768], 128, 64, 4, 128, torch.bfloat16, torch.int32)
    query, keys, table, requests, complete = x
    query.fill_(1)
    keys.zero_()
    table[1].copy_(table[0])
    logical = torch.arange(positive_blocks, device="cuda")
    for request in range(2):
        slots = table[request, logical // 64].long() * 64 + logical % 64
        keys[slots] = 1
    selected = _run(x, 64, 512, 1, 32 << 20, False)
    for row in range(2):
        assert torch.isin(logical, selected[row]).all()
    _check(x, selected, 64, 512)


def test_decode_runtime_shapes_do_not_compile():
    cases = [
        _inputs(lengths, [32767, 32768], pages, 64, 4, 128, torch.bfloat16, torch.int32)
        for lengths, pages in [([1, 1], 129), ([2, 1], 137), ([3, 2], 200)]
    ]
    _run(cases[0], 64, 512, None, 32 << 20, False)
    with assert_no_triton_compile(
        native._qsa_native_score_candidates_kernel,
        native._qsa_native_pair_score_candidates_kernel,
        native._qsa_coarse_merge_candidates_kernel,
        implementation._qsa_stream_score_candidates_kernel,
        implementation._qsa_stream_merge_candidates_kernel,
        implementation._qwen4_exp_qsa_stream_block_topk_kernel,
    ):
        for x in cases[1:]:
            _check(x, _run(x, 64, 512, None, 32 << 20, False), 64, 512)


@pytest.mark.parametrize(
    "rows,pages,width",
    [
        (1, 64, 1),
        (8, 128, 1),
        (8, 128, 8),
        (64, 128, 1),
        (128, 1024, 1),
        (2, 2048, None),
    ],
)
@pytest.mark.parametrize("pdl", [False, True])
def test_native_decode_sizes_and_graph(rows, pages, width, pdl):
    lengths = [width] * (rows // width) if width is not None else [1, 1]
    x = _inputs(
        lengths,
        [pages * 64 * 4] * len(lengths),
        pages,
        64,
        4,
        128,
        torch.bfloat16,
        torch.int32,
    )
    _check(x, _run(x, 64, 512, width, 32 << 20, pdl), 64, 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _run(x, 64, 512, width, 32 << 20, pdl)
    boundaries = torch.tensor(
        [-3, 0, 1, 511, 512, 513, pages * 64, pages * 64 + 27],
        device="cuda",
        dtype=torch.int32,
    )
    for shift in (0, 3, 5):
        x[-1].copy_(boundaries[(torch.arange(rows, device="cuda") + shift) % 8])
        if shift == 3:
            x[0].zero_()
            x[2].copy_(x[2].flip(1).clone())
        if shift == 5:
            x[0].normal_()
            x[3].copy_(x[3].flip(0).clone())
        graph.replay()
        _check(x, selected, 64, 512)


@pytest.mark.parametrize(
    "dtype,heads,dim,topk",
    [
        (torch.float16, 2, 64, 64),
        (torch.float32, 2, 64, 64),
        (torch.bfloat16, 4, 128, 1024),
    ],
)
def test_portable_geometry_inline_fallback(dtype, heads, dim, topk):
    x = _inputs([1, 1], [32767, 32768], 128, 64, heads, dim, dtype, torch.int32)
    q, k, table, requests, complete = x
    q.fill_(1)
    k.zero_()
    table[1].copy_(table[0])
    logical = torch.arange(topk, device="cuda")
    slots = table[0, logical // 64].long() * 64 + logical % 64
    k[slots] = 1
    selected = _run(x, 64, topk, 1, 32 << 20, False)
    assert torch.isin(logical, selected[0]).all()
    assert torch.isin(logical, selected[1]).all()
    _check(x, selected, 64, topk)


@pytest.mark.parametrize("pages", [128, 129, 1024])
def test_runtime_frontier_ranges_and_graph(pages):
    x = _inputs([1, 1], [8883, 14971], pages, 64, 4, 128, torch.bfloat16, torch.int64)
    _run(x, 64, 512, 1, 32 << 20, False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _run(x, 64, 512, 1, 32 << 20, False)
    with assert_no_triton_compile(
        native._qsa_native_score_candidates_kernel,
        native._qsa_native_pair_score_candidates_kernel,
        native._qsa_coarse_merge_candidates_kernel,
        implementation._qsa_stream_merge_candidates_kernel,
        implementation._qsa_stream_merge_short_candidates_kernel,
    ):
        for bounds in [
            (2220, 3742),
            (4095, 4096),
            (4097, 8191),
            (8192, 8193),
            (65535, 65536),
            (0, 513),
            (3742, 2220),
        ]:
            x[-1].copy_(torch.tensor(bounds, device="cuda", dtype=torch.int32))
            x[3].copy_(x[3].flip(0).clone())
            eager = _run(x, 64, 512, 1, 32 << 20, False)
            graph.replay()
            torch.testing.assert_close(selected, eager, rtol=0, atol=0)
            _check(x, selected, 64, 512)


@pytest.mark.parametrize("budget", [266247, 266248, 32 << 20])
def test_short_frontier_concentrated_winners_and_budget(budget):
    x = _inputs([1, 1], [8883, 14971], 1024, 64, 4, 128, torch.bfloat16, torch.int32)
    q, k, table, requests, complete = x
    q.fill_(1)
    k.zero_()
    table[1].copy_(table[0])
    logical = torch.arange(512, device="cuda")
    slots = table[0, logical // 64].long() * 64 + logical % 64
    k[slots] = 1
    selected = _run(x, 64, 512, 1, budget, False)
    assert torch.isin(logical, selected[0]).all()
    assert torch.isin(logical, selected[1]).all()
    _check(x, selected, 64, 512)


@pytest.mark.parametrize("pdl", [False, True])
def test_long_frontier_concentrated_winners(pdl):
    x = _inputs([1, 1], [262143, 262144], 1024, 64, 4, 128, torch.bfloat16, torch.int32)
    q, k, table, requests, complete = x
    q.fill_(1)
    k.zero_()
    table[1].copy_(table[0])
    logical = torch.arange(512, device="cuda")
    slots = table[0, logical // 64].long() * 64 + logical % 64
    k[slots] = 1
    selected = _run(x, 64, 512, 1, 32 << 20, pdl)
    assert torch.isin(logical, selected[0]).all()
    assert torch.isin(logical, selected[1]).all()
    _check(x, selected, 64, 512)


@pytest.mark.parametrize("rows", [2, 8])
def test_decode_zero_scratch_peak_budget(rows):
    lengths = [1] * rows
    seq_lens = [8883, 14971] if rows == 2 else [262144] * rows
    x = _inputs(lengths, seq_lens, 1024, 64, 4, 128, torch.bfloat16, torch.int32)
    warm = _run(x, 64, 512, 1, 0, False)
    del warm
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    selected = _run(x, 64, 512, 1, 0, False)
    torch.cuda.synchronize()
    assert (
        torch.cuda.max_memory_allocated() - before
        == selected.numel() * selected.element_size()
    )
    _check(x, selected, 64, 512)


@pytest.mark.parametrize("pdl", [False, True])
def test_dense_fp32_boundary_remains_exact(pdl):
    x = _inputs([1, 1], [262143, 262144], 1024, 64, 4, 128, torch.bfloat16, torch.int32)
    query, keys, table, requests, complete = x
    query.fill_(1)
    keys.fill_(1)
    table[1].copy_(table[0])
    logical = torch.arange(0, 65536, 16, device="cuda")
    slots = table[0, logical // 64].long() * 64 + logical % 64
    keys[slots, 0, 0] = 1.0078125
    selected = _run(x, 64, 512, 1, 32 << 20, pdl)
    assert (selected % 16 == 0).all()
    _check(x, selected, 64, 512)


@pytest.mark.parametrize("pdl", [False, True])
def test_paired_scorer_odd_rows_and_independent_pages(pdl):
    x = _inputs(
        [3, 2, 4], [32765, 32768, 33000], 129, 64, 4, 128, torch.bfloat16, torch.int64
    )
    _check(x, _run(x, 64, 512, None, 32 << 20, pdl), 64, 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _run(x, 64, 512, None, 32 << 20, pdl)
    x[2][1, 0].copy_(x[2][0, 0])
    x[-1].copy_(
        torch.tensor(
            [513, 576, 577, 640, 641, 4096, 4097, 8192, 8256],
            device="cuda",
            dtype=torch.int32,
        )
    )
    graph.replay()
    _check(x, selected, 64, 512)


def test_paired_scorer_runtime_rows_do_not_compile():
    cases = [
        _inputs(
            lengths,
            [32768] * len(lengths),
            pages,
            64,
            4,
            128,
            torch.bfloat16,
            torch.int32,
        )
        for lengths, pages in [([3, 2, 4], 129), ([7, 3, 7], 137), ([8, 9, 8], 200)]
    ]
    _run(cases[0], 64, 512, None, 32 << 20, False)
    with assert_no_triton_compile(
        native._qsa_native_pair_score_candidates_kernel,
        implementation._qsa_stream_merge_candidates_kernel,
        implementation._qsa_stream_merge_short_candidates_kernel,
    ):
        for x in cases[1:]:
            _check(x, _run(x, 64, 512, None, 32 << 20, False), 64, 512)


@pytest.mark.parametrize("topk", [64, 1024])
def test_paired_scorer_preserves_other_topk(topk):
    x = _inputs(
        [3, 5, 1], [32767, 32768, 32769], 128, 64, 4, 128, torch.bfloat16, torch.int32
    )
    _check(x, _run(x, 64, topk, None, 32 << 20, False), 64, topk)


@pytest.mark.parametrize("pdl", [False, True])
def test_hierarchical_frontiers_and_graph(pdl):
    x = _inputs([1, 1], [524287, 524288], 2048, 64, 4, 128, torch.bfloat16, torch.int64)
    _check(x, _run(x, 64, 512, 1, 32 << 20, pdl), 64, 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _run(x, 64, 512, 1, 32 << 20, pdl)
    with assert_no_triton_compile(
        native._qsa_native_score_candidates_kernel,
        native._qsa_partial_merge_candidates_kernel,
        native._qsa_coarse_merge_candidates_kernel,
    ):
        for counts in [
            (8192, 8193),
            (65535, 65536),
            (98303, 98304),
            (100003, 131071),
            (131072, 131071),
            (2220, 3742),
            (0, 513),
        ]:
            x[-1].copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
            x[2].copy_(x[2].flip(1).clone())
            eager = _run(x, 64, 512, 1, 32 << 20, pdl)
            graph.replay()
            torch.testing.assert_close(selected, eager, rtol=0, atol=0)
            _check(x, selected, 64, 512)


@pytest.mark.parametrize("pdl", [False, True])
def test_hierarchical_partial_discard_requires_full_selection(pdl):
    x = _inputs([1, 1], [524287, 524288], 2048, 64, 4, 128, torch.bfloat16, torch.int32)
    query, keys, table, requests, complete = x
    query.fill_(1)
    keys.zero_()
    table[1].copy_(table[0])
    logical = torch.arange(512, device="cuda") * 16
    slots = table[0, logical // 64].long() * 64 + logical % 64
    keys[slots] = 1
    selected = _run(x, 64, 512, 1, 32 << 20, pdl)
    for row in range(2):
        assert torch.isin(logical, selected[row]).all()
    _check(x, selected, 64, 512)


@pytest.mark.parametrize("budget", [0, 532488, 598023, 598024, 32 << 20])
def test_hierarchical_workspace_budget(budget):
    x = _inputs([1, 1], [524287, 524288], 2048, 64, 4, 128, torch.bfloat16, torch.int32)
    warm = _run(x, 64, 512, 1, budget, False)
    del warm
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_stats()["requested_bytes.all.current"]
    selected = _run(x, 64, 512, 1, budget, False)
    torch.cuda.synchronize()
    peak = torch.cuda.memory_stats()["requested_bytes.all.peak"]
    assert peak - before <= selected.numel() * selected.element_size() + budget
    _check(x, selected, 64, 512)


def test_hierarchical_runtime_shapes_do_not_compile():
    cases = [
        _inputs(
            lengths,
            [500000, 510000, 520000],
            pages,
            64,
            4,
            128,
            torch.bfloat16,
            torch.int32,
        )
        for lengths, pages in [([3, 2, 4], 2100), ([5, 6, 6], 2500), ([8, 9, 8], 3000)]
    ]
    _run(cases[0], 64, 512, None, 32 << 20, False)
    with assert_no_triton_compile(
        native._qsa_native_pair_score_candidates_kernel,
        native._qsa_partial_merge_candidates_kernel,
        native._qsa_coarse_merge_candidates_kernel,
    ):
        for x in cases[1:]:
            _check(x, _run(x, 64, 512, None, 32 << 20, False), 64, 512)


@pytest.mark.parametrize("boundary_count", [32, 33])
def test_small_exact_boundary_and_graph_refresh(boundary_count):
    x = _inputs([1, 1], [32767, 32768], 128, 64, 4, 128, torch.bfloat16, torch.int32)
    query, keys, table, requests, complete = x
    query.fill_(1)
    keys.zero_()
    table[1].copy_(table[0])
    higher = torch.arange(480, device="cuda") * 16
    equal = torch.arange(boundary_count, device="cuda") * 16 + 1
    higher_slots = table[0, higher // 64].long() * 64 + higher % 64
    equal_slots = table[0, equal // 64].long() * 64 + equal % 64
    keys[higher_slots] = 2
    keys[equal_slots] = 1
    _check(x, _run(x, 64, 512, 1, 32 << 20, False), 64, 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _run(x, 64, 512, 1, 32 << 20, False)
    last = torch.tensor([32 * 16 + 1], device="cuda")
    last_slot = table[0, last // 64].long() * 64 + last % 64
    keys[last_slot] = 1 if boundary_count == 32 else 0
    with assert_no_triton_compile(
        native._qsa_native_score_candidates_kernel,
        native._qsa_coarse_merge_candidates_kernel,
    ):
        graph.replay()
        _check(x, selected, 64, 512)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("rows", [272, 4112])
def test_tma_frontiers_owners_null_page_and_graph(rows, pdl):
    lengths = [rows // 2 - 3, rows // 2 + 3]
    x = _inputs(lengths, [32768, 32768], 128, 64, 4, 128, torch.bfloat16, torch.int64)
    x[1][:64].fill_(float("nan"))
    _check(x, _run(x, 64, 512, None, 0, pdl), 64, 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _run(x, 64, 512, None, 0, pdl)
    with assert_no_triton_compile(
        native._qsa_warp_stream_kernel, native._qsa_soa_stream_kernel
    ):
        for offset in range(3):
            limits = torch.tensor(
                [0, 1, 511, 512, 513, 574, 4095, 4096, 8191, 8192],
                device="cuda",
                dtype=torch.int32,
            )
            x[-1].copy_(limits[(torch.arange(rows, device="cuda") + offset) % 10])
            x[3].copy_((torch.arange(rows, device="cuda") + offset) % 2)
            x[2].copy_(x[2].flip(1).clone())
            if offset == 1:
                x[0].zero_()
            else:
                x[0].normal_()
            eager = _run(x, 64, 512, None, 0, pdl)
            graph.replay()
            torch.testing.assert_close(selected, eager, rtol=0, atol=0)
            _check(x, selected, 64, 512)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("rows", [256, 4096])
def test_tma_partial_page_nan_padding(rows, pdl):
    x = _inputs([rows], [32768], 128, 64, 4, 128, torch.bfloat16, torch.int32)
    x[-1].fill_(513)
    x[1][:64].fill_(float("nan"))
    last_page = x[2][0, 8].long()
    tail = last_page * 64 + torch.arange(1, 64, device="cuda")
    x[1][tail] = float("nan")
    _check(x, _run(x, 64, 512, rows, 0, pdl), 64, 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _run(x, 64, 512, rows, 0, pdl)
    x[0].normal_()
    graph.replay()
    eager = _run(x, 64, 512, rows, 0, pdl)
    torch.testing.assert_close(selected, eager, rtol=0, atol=0)
    _check(x, selected, 64, 512)


@pytest.mark.parametrize("base_rows", [272, 4112])
def test_tma_descriptor_runtime_shapes_do_not_compile(base_rows):
    cases = []
    for offset, pages in [(0, 65), (16, 73), (32, 127)]:
        rows = base_rows + offset
        lengths = [rows // 2 - 3, rows // 2 + 3]
        x = _inputs(
            lengths, [32768, 32768], pages, 64, 4, 128, torch.bfloat16, torch.int32
        )
        if offset:
            query = torch.randn(rows, 6, 128, device="cuda", dtype=torch.bfloat16)[
                :, :4
            ]
            x = (query, *x[1:])
        cases.append(x)
    _run(cases[0], 64, 512, None, 0, False)
    with assert_no_triton_compile(
        native._qsa_warp_stream_kernel, native._qsa_soa_stream_kernel
    ):
        for x in cases[1:]:
            _check(x, _run(x, 64, 512, None, 0, False), 64, 512)


@pytest.mark.parametrize("rows", [272, 4112])
def test_tma_output_only_allocation(rows):
    x = _inputs([rows], [32768], 128, 64, 4, 128, torch.bfloat16, torch.int32)
    warm = _run(x, 64, 512, rows, 0, False)
    del warm
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_stats()["requested_bytes.all.current"]
    selected = _run(x, 64, 512, rows, 0, False)
    torch.cuda.synchronize()
    peak = torch.cuda.memory_stats()["requested_bytes.all.peak"]
    assert peak - before == selected.numel() * selected.element_size()
    _check(x, selected, 64, 512)


def test_tma_other_page_size_preserves_results():
    x = _inputs(
        [137, 135], [32768, 32768], 259, 32, 4, 128, torch.bfloat16, torch.int32
    )
    _check(x, _run(x, 32, 512, None, 0, False), 32, 512)


@pytest.mark.parametrize("rows", [272, 4112])
@pytest.mark.parametrize("pdl", [False, True])
def test_tma_stage_reuse_sparse_groups(rows, pdl):
    x = _inputs(
        [rows // 2 - 3, rows // 2 + 3],
        [32768, 32768],
        128,
        64,
        4,
        128,
        torch.bfloat16,
        torch.int32,
    )
    active = torch.arange(128, 160, device="cuda")
    x[-1].zero_()
    x[-1][active] = 8192
    x[3][active] = torch.arange(32, device="cuda", dtype=torch.int32) % 2
    x[1][:64].fill_(float("nan"))
    _run(x, 64, 512, None, 0, pdl)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _run(x, 64, 512, None, 0, pdl)
    for index, limit in enumerate([513, 8191, 4097, 8192]):
        x[-1][active] = limit
        x[2].copy_(x[2].flip(1).clone())
        if index == 2:
            x[0][active] = 0
        else:
            x[0][active] = torch.randn(32, 4, 128, device="cuda", dtype=torch.bfloat16)
        eager = _run(x, 64, 512, None, 0, pdl)
        graph.replay()
        torch.testing.assert_close(selected, eager, rtol=0, atol=0)
        probe = x[0][active], x[1], x[2], x[3][active], x[-1][active]
        _check(probe, selected[active], 64, 512)
        assert (selected[:128] == -1).all()
        assert (selected[160:] == -1).all()


@pytest.mark.parametrize(
    "rows,page_size,width", [(272, 32, None), (4112, 32, None), (272, 64, 4)]
)
@pytest.mark.parametrize("pdl", [False, True])
def test_cp_async_stage_reuse_sparse_groups(rows, page_size, width, pdl):
    x = _inputs(
        [rows // 2 - 3, rows // 2 + 3],
        [32768, 32768],
        8192 // page_size,
        page_size,
        4,
        128,
        torch.bfloat16,
        torch.int32,
    )
    active = torch.arange(128, 160, device="cuda")
    x[-1].zero_()
    x[-1][active] = 8192
    x[3][active] = torch.arange(32, device="cuda", dtype=torch.int32) % 2
    x[1][:page_size].fill_(float("nan"))
    _run(x, page_size, 512, width, 0, pdl)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _run(x, page_size, 512, width, 0, pdl)
    for index, limit in enumerate([513, 8191, 4097, 8192]):
        x[-1][active] = limit
        x[2].copy_(x[2].flip(1).clone())
        if index == 2:
            x[0][active] = 0
        else:
            x[0][active] = torch.randn(32, 4, 128, device="cuda", dtype=torch.bfloat16)
        eager = _run(x, page_size, 512, width, 0, pdl)
        graph.replay()
        torch.testing.assert_close(selected, eager, rtol=0, atol=0)
        probe = x[0][active], x[1], x[2], x[3][active], x[-1][active]
        _check(probe, selected[active], page_size, 512)
        assert (selected[:128] == -1).all()
        assert (selected[160:] == -1).all()


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("rows", [272, 4112])
def test_tma_cache_extent_uses_64bit_offsets(pdl, rows):
    x = list(
        _inputs(
            [rows // 2 - 3, rows // 2 + 3],
            [32768, 32768],
            128,
            64,
            4,
            128,
            torch.bfloat16,
            torch.int32,
        )
    )
    slots = x[1].shape[0]
    row_stride = 1 << 18
    backing = torch.empty(
        (slots - 1) * row_stride + 128, device="cuda", dtype=torch.bfloat16
    )
    strided = backing.as_strided((slots, 1, 128), (row_stride, 128, 1))
    strided.copy_(x[1])
    x[1] = strided
    eager = _run(x, 64, 512, None, 0, pdl)
    _check(x, eager, 64, 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _run(x, 64, 512, None, 0, pdl)
    x[2].copy_(x[2].flip(1).clone())
    x[-1].fill_(8191)
    graph.replay()
    torch.testing.assert_close(selected, _run(x, 64, 512, None, 0, pdl), rtol=0, atol=0)
    _check(x, selected, 64, 512)


@pytest.mark.parametrize("rows", [272, 4112])
@pytest.mark.parametrize("pdl", [False, True])
def test_integer_proxy_clamped_boundary_and_late_winners(rows, pdl):
    x = _inputs(
        [rows // 2 - 3, rows // 2 + 3],
        [32768, 32768],
        128,
        64,
        4,
        128,
        torch.bfloat16,
        torch.int32,
    )
    active = torch.arange(128, 160, device="cuda")
    query, keys, table, requests, complete = x
    query.zero_()
    query[:, 0, 0] = 1
    keys.zero_()
    keys[:, 0, 0] = 1
    keys[:64] = float("nan")
    keys[64 : 64 + 256, 0, 0] = 32
    keys[64 + 6000 : 64 + 6512, 0, 0] = 2
    table[0].copy_(torch.arange(1, 129, device="cuda", dtype=torch.int32))
    table[1].copy_(table[0].flip(0))
    requests[active] = torch.arange(32, device="cuda", dtype=torch.int32) % 2
    complete.zero_()
    complete[active] = 8192
    eager = _run(x, 64, 512, None, 0, pdl)
    ids = eager[active].long()
    slots = table[requests[active].long()[:, None], ids // 64].long() * 64 + ids % 64
    chosen = keys[slots, 0, 0]
    # The first pool spans more than the proxy's256-bin window. Its boundary
    # is clamped; later higher scores must still replace the smaller values.
    assert ((chosen == 32).sum(1) == 256).all()
    assert ((chosen == 2).sum(1) == 256).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _run(x, 64, 512, None, 0, pdl)
    limits = torch.tensor(
        [1023, 1024, 1025, 2049, 8191, 8192], device="cuda", dtype=torch.int32
    )
    for offset in (0, 1, 3):
        complete[active] = limits[
            (torch.arange(32, device="cuda") + offset) % limits.numel()
        ]
        graph.replay()
        eager = _run(x, 64, 512, None, 0, pdl)
        torch.testing.assert_close(selected, eager, rtol=0, atol=0)
        probe = query[active], keys, table, requests[active], complete[active]
        _check(probe, selected[active], 64, 512)
        assert (selected[:128] == -1).all()
        assert (selected[160:] == -1).all()


@pytest.mark.parametrize("pdl", [False, True])
def test_soa_last_id_at_16bit_capacity(pdl):
    x = _inputs([4112], [262144], 1024, 64, 4, 128, torch.bfloat16, torch.int32)
    query, keys, table, requests, complete = x
    query.fill_(1)
    table.copy_(torch.arange(1, 1025, device="cuda", dtype=torch.int32)[None])
    keys.zero_()
    keys[64 + 65024 :].fill_(1)
    active = torch.arange(128, 160, device="cuda")
    complete.zero_()
    complete[active] = 65536
    _run(x, 64, 512, None, 0, pdl)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _run(x, 64, 512, None, 0, pdl)
    for limit in [65536, 65535, 65520]:
        complete[active] = limit
        graph.replay()
        probe = query[active], keys, table, requests[active], complete[active]
        _check(probe, selected[active], 64, 512)
        assert (selected[active] < limit).all()
        if limit == 65536:
            assert (selected[active] >= 65024).all()
            assert (selected[active] == 65535).any(1).all()
        assert (selected[:128] == -1).all()
        assert (selected[160:] == -1).all()


def test_soa_capacity_buckets_do_not_compile_within_bucket():
    cases = []
    active = torch.arange(128, 160, device="cuda")
    for pages in [1023, 1024, 1025, 1026]:
        x = _inputs(
            [2049, 2063],
            [32767, 32768],
            pages,
            64,
            4,
            128,
            torch.bfloat16,
            torch.int32,
        )
        x[-1].zero_()
        x[-1][active] = 8191
        cases.append(x)
    for x in [cases[0], cases[2]]:
        _run(x, 64, 512, None, 0, False)
    with assert_no_triton_compile(native._qsa_soa_stream_kernel):
        for x in [cases[1], cases[3]]:
            selected = _run(x, 64, 512, None, 0, False)
            probe = x[0][active], x[1], x[2], x[3][active], x[-1][active]
            _check(probe, selected[active], 64, 512)


@pytest.mark.parametrize("lengths", [(1, 1), (8883, 14971)])
def test_two_requests_with_fixed_global_cache_capacity(lengths):
    seq_lens = [8883, 14971]
    x = _inputs(lengths, seq_lens, 1024, 64, 4, 128, torch.bfloat16, torch.int32)
    table = x[2]
    table.zero_()
    physical_pages = torch.randperm(1024, device="cuda").int() + 1
    page_counts = [(length // 4 + 63) // 64 for length in seq_lens]
    offset = 0
    for request, pages in enumerate(page_counts):
        table[request, :pages].copy_(physical_pages[offset : offset + pages])
        offset += pages
    # Both requests draw disjoint live pages from one 256K-token cache.
    width = 1 if lengths == (1, 1) else None
    _run(x, 64, 512, width, 32 << 20, False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = _run(x, 64, 512, width, 32 << 20, False)
    original_counts = x[-1].clone()
    for drop in [1, 0]:
        x[-1].copy_((original_counts - drop).clamp_min(0))
        for request, pages in enumerate(page_counts):
            table[request, :pages].copy_(table[request, :pages].flip(0).clone())
        graph.replay()
        eager = _run(x, 64, 512, width, 32 << 20, False)
        torch.testing.assert_close(selected, eager, rtol=0, atol=0)
        _check(x, selected, 64, 512)
