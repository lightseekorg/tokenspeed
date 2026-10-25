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

from contextlib import nullcontext

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

if not current_platform().is_blackwell:
    pytest.skip(
        "CuTe DSL QSA stream top-k requires NVIDIA Blackwell",
        allow_module_level=True,
    )

import tokenspeed_kernel.thirdparty.cute_dsl.qsa_stream_topk as stream_topk
from tokenspeed_kernel.ops.attention.qsa import (
    qsa_block_topk_stream,
    qwen4_exp_qsa_block_topk,
)
from tokenspeed_kernel.ops.attention.qsa.cute_dsl import (
    can_use_cute_dsl_qwen4_exp_qsa_block_topk,
)
from tokenspeed_kernel.thirdparty.cute_dsl.qsa_stream_topk import (
    kernel as cute_dsl_qsa_stream_topk,
)


@pytest.mark.parametrize(
    ("pages", "rows", "max_partial_bytes", "expected"),
    [
        pytest.param(80, 8, 32 * 1024 * 1024, True, id="small-partial-second-tile"),
        pytest.param(81, 8, 32 * 1024 * 1024, True, id="long-small-grid"),
        pytest.param(128, 64, 32 * 1024 * 1024, True, id="long-populated-grid"),
        pytest.param(
            128,
            64,
            64 * 512 * 8,
            True,
            id="long-with-small-partial-budget",
        ),
        pytest.param(1024, 16384, 32 * 1024 * 1024, True, id="prefill-256k-capacity"),
        pytest.param(1024, 16384, 0, True, id="prefill-without-partial-budget"),
        pytest.param(1025, 16384, 0, True, id="prefill-with-mtp-padding"),
        pytest.param(1025, 4, 32 * 1024 * 1024, True, id="decode-with-mtp-padding"),
        pytest.param(2048, 16384, 0, True, id="prefill-512k-capacity"),
        pytest.param(2049, 16384, 0, False, id="unsupported-capacity"),
    ],
)
def test_cute_dsl_qsa_stream_topk_dispatch_boundary(
    device: str,
    pages: int,
    rows: int,
    max_partial_bytes: int,
    expected: bool,
) -> None:
    query = torch.empty(rows, 4, 128, device=device, dtype=torch.bfloat16)
    key_cache = torch.empty(pages * 64, 1, 128, device=device, dtype=torch.bfloat16)
    page_table = torch.arange(pages, device=device, dtype=torch.int32).unsqueeze(0)
    request_indices = torch.zeros(rows, device=device, dtype=torch.int32)
    complete_blocks = torch.full((rows,), pages * 64, device=device, dtype=torch.int32)

    assert (
        can_use_cute_dsl_qwen4_exp_qsa_block_topk(
            query,
            key_cache,
            page_table,
            request_indices,
            complete_blocks,
            page_size=64,
            block_topk=512,
            queries_per_request=rows,
            max_partial_bytes=max_partial_bytes,
        )
        is expected
    )


def _reference_scores(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    offsets = torch.arange(page_size, device=query.device)
    slots = (
        page_table[request_indices.long()].long().unsqueeze(2) * page_size
        + offsets.view(1, 1, -1)
    ).flatten(1)
    keys = key_cache[slots, 0].float()
    scores = torch.einsum("rhd,rnd->rhn", query.float(), keys)
    scores = scores.clamp_min_(0).sum(dim=1)
    logical_blocks = torch.arange(scores.shape[1], device=query.device)
    return scores.masked_fill(
        logical_blocks.unsqueeze(0) >= complete_blocks.unsqueeze(1),
        -float("inf"),
    )


def _assert_matches_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    selected: torch.Tensor,
    page_size: int,
    block_topk: int,
) -> None:
    scores = _reference_scores(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size,
    )
    selected_scores = torch.gather(scores, 1, selected.clamp_min(0).long())
    selected_scores.masked_fill_(selected < 0, -float("inf"))
    expected_scores = torch.full_like(selected_scores, -float("inf"))
    count = min(block_topk, scores.shape[1])
    expected_scores[:, :count] = torch.topk(scores, count, dim=1).values
    torch.testing.assert_close(
        selected_scores.sort(dim=1, descending=True).values,
        expected_scores,
        rtol=1e-5,
        atol=1e-4,
    )
    expected_count = complete_blocks.clamp(0, min(scores.shape[1], block_topk)).long()
    torch.testing.assert_close((selected >= 0).sum(dim=1), expected_count)
    assert bool((selected >= -1).all())
    assert bool(((selected < complete_blocks[:, None]) | (selected < 0)).all())
    for row, count in zip(selected, expected_count.tolist(), strict=True):
        assert torch.unique(row[row >= 0]).numel() == count


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("request_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    ("lengths", "pages", "uniform", "algorithm"),
    [
        pytest.param([129], 9, True, "prefill", id="uniform-prefill-tail"),
        pytest.param([65, 65], 64, True, "dense", id="uniform-dense-tails"),
        pytest.param([67, 67], 129, True, "overlap", id="uniform-overlap-tails"),
        pytest.param([1] * 129, 9, True, "prefill", id="uniform-singletons"),
        pytest.param([3, 61, 67, 7, 1], 9, False, "prefill", id="ragged-prefill"),
        pytest.param([65, 72, 9], 64, False, "dense", id="ragged-dense"),
        pytest.param([127, 1, 9], 129, False, "overlap", id="ragged-overlap"),
        pytest.param(
            [3, 2, 1], 1024, False, "overlap", id="ragged-decode-zero-scratch"
        ),
    ],
)
def test_stream_topk_request_tails_and_ragged_graph_replay(
    device, lengths, pages, uniform, algorithm, request_dtype, enable_pdl
):
    torch.manual_seed(163)
    rows = sum(lengths)
    capacity = pages * 64
    query = torch.randn(rows, 5, 128, device=device, dtype=torch.bfloat16)[:, :4]
    keys = torch.randn(
        (len(lengths) * pages + 1) * 64, 1, 128, device=device, dtype=torch.bfloat16
    )
    keys[:64].fill_(float("nan"))
    table = torch.empty((len(lengths), pages + 3), device=device, dtype=torch.int32)
    table = table[:, :pages]
    table.copy_(
        torch.randperm(len(lengths) * pages, device=device).reshape(len(lengths), pages)
        + 1
    )
    original_requests = torch.arange(
        len(lengths), device=device, dtype=request_dtype
    ).repeat_interleave(torch.tensor(lengths, device=device))
    requests = original_requests.clone()
    complete = torch.full((rows,), capacity, device=device, dtype=torch.int32)

    def invoke():
        return qsa_block_topk_stream(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=lengths[0] if uniform else None,
            max_partial_bytes=0,
            enable_pdl=enable_pdl,
            solution="cute_dsl",
            override=None,
        )

    selected = invoke()
    config = stream_topk._configuration(
        rows, capacity, lengths[0] if uniform else None, 0
    )
    assert config is not None
    assert config.algorithm == algorithm
    assert config.group_layout == ("uniform" if uniform else "ragged")
    assert (
        query.device.index,
        request_dtype,
        enable_pdl,
        config,
    ) in stream_topk._COMPILED
    _assert_matches_reference(query, keys, table, requests, complete, selected, 64, 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = invoke()
    row_ids = torch.arange(rows, device=device)
    frontiers = torch.tensor(
        [0, 1, 63, 511, 512, 513, capacity - 1, capacity],
        device=device,
        dtype=torch.int32,
    )
    for step in range(3):
        # A ragged graph must refresh ownership boundaries as well as lengths.
        # Rolling also gives a request multiple non-adjacent runs.
        if not uniform:
            requests.copy_(original_requests.roll(step + 1))
        complete.copy_(frontiers[(row_ids + step) % frontiers.numel()])
        query.normal_()
        graph.replay()
        _assert_matches_reference(
            query, keys, table, requests, complete, selected, 64, 512
        )


@pytest.mark.parametrize("uniform", [False, True])
def test_stream_topk_grouped_tails_reuse_compilation(device, monkeypatch, uniform):
    monkeypatch.setattr(stream_topk, "_COMPILED", {})
    compile_count = 0
    original_compile = stream_topk._compile

    def counted_compile(*args):
        nonlocal compile_count
        compile_count += 1
        return original_compile(*args)

    monkeypatch.setattr(stream_topk, "_compile", counted_compile)
    layouts = (
        [([65, 65], 65), ([129], 80), ([67, 67, 67], 100)]
        if uniform
        else [([64, 66], 65), ([8, 129, 1], 80), ([129, 5, 67], 100)]
    )
    torch.manual_seed(167)
    for lengths, pages in layouts:
        rows = sum(lengths)
        query = torch.randn(rows, 5, 256, device=device, dtype=torch.bfloat16)[
            :, :4, :128
        ]
        keys = torch.randn(pages * 64, 1, 128, device=device, dtype=torch.bfloat16)
        table = torch.arange(pages, device=device, dtype=torch.int32)[None].repeat(
            len(lengths), 1
        )
        requests = torch.arange(
            len(lengths), device=device, dtype=torch.int32
        ).repeat_interleave(torch.tensor(lengths, device=device))
        complete = torch.linspace(0, pages * 64, rows, device=device).int()
        selected = cute_dsl_qsa_stream_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=lengths[0] if uniform else None,
            max_partial_bytes=0,
            enable_pdl=False,
        )
        _assert_matches_reference(
            query, keys, table, requests, complete, selected, 64, 512
        )
        configs = [key[-1] for key in stream_topk._COMPILED]
        assert len(configs) == 1
        assert configs[0].algorithm == "overlap"
        assert configs[0].group_layout == ("uniform" if uniform else "ragged")
    assert compile_count == 1


@pytest.mark.parametrize("request_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize(
    ("page_counts", "small_grid", "algorithm"),
    [
        pytest.param((9, 16, 24, 32), False, "prefill", id="prefill"),
        pytest.param((33, 40, 56, 64), False, "dense", id="dense"),
        pytest.param((65, 80, 100, 128), False, "overlap", id="overlap"),
        pytest.param((129, 160, 200, 256), True, "split", id="split"),
    ],
)
def test_cute_dsl_qsa_stream_topk_reuses_compile_across_layouts(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    request_dtype: torch.dtype,
    enable_pdl: bool,
    page_counts: tuple[int, ...],
    small_grid: bool,
    algorithm: str,
) -> None:
    torch.manual_seed(89)
    monkeypatch.setattr(stream_topk, "_COMPILED", {})
    compile_count = 0
    original_compile = stream_topk._compile

    def counted_compile(*args):
        nonlocal compile_count
        compile_count += 1
        return original_compile(*args)

    monkeypatch.setattr(stream_topk, "_compile", counted_compile)
    # Widths share two compiled variants: aligned and partial final groups.
    # Request counts, page-table widths and Q strides stay dynamic in both.
    layouts = (
        (256, 8, 4, 128),
        (512, 8, 5, 128),
        (384, 16, 5, 256),
        (640, 32, 4, 128),
        (272, 17, 5, 128),
        (288, 18, 5, 256),
        (285, 19, 5, 256),
        (300, 20, 4, 128),
        (315, 21, 5, 128),
        (330, 22, 5, 256),
        (322, 23, 4, 128),
        (384, 16, 5, 128),
        (285, 19, 4, 128),
    )
    for i, (rows, width, storage_heads, storage_dim) in enumerate(layouts):
        pages = page_counts[i % len(page_counts)]
        if small_grid:
            rows = width * (2 + i % 3)
        query = torch.randn(
            rows, storage_heads, storage_dim, device=device, dtype=torch.bfloat16
        )[:, :4, :128]
        keys = torch.randn(pages * 64, 1, 128, device=device, dtype=torch.bfloat16)
        table_storage = torch.stack(
            [torch.randperm(pages, device=device) for _ in range(rows // width)]
        ).to(torch.int32)
        if rows != 256:
            # Eager batched prefill shortens columns without compacting the
            # persistent allocation, including a row stride unaligned to 16B.
            table = torch.empty(
                (rows // width, 1025), device=device, dtype=torch.int32
            )[:, :pages]
            table.copy_(table_storage)
        else:
            table = table_storage
        requests = torch.arange(
            rows // width, device=device, dtype=request_dtype
        ).repeat_interleave(width)
        complete = torch.linspace(0, pages * 64, rows, device=device).to(torch.int32)
        assert can_use_cute_dsl_qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=width,
            max_partial_bytes=32 * 1024 * 1024,
        )
        selected = cute_dsl_qsa_stream_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=width,
            max_partial_bytes=32 * 1024 * 1024,
            enable_pdl=enable_pdl,
        )
        _assert_matches_reference(
            query, keys, table, requests, complete, selected, 64, 512
        )
        assert compile_count == (1 if i < 4 else 2)
        configs = [key[-1] for key in stream_topk._COMPILED]
        assert {config.algorithm for config in configs} == {algorithm}
        assert {config.group_layout for config in configs} == (
            {"aligned"} if i < 4 else {"aligned", "uniform"}
        )


@pytest.mark.parametrize(
    ("pages", "enable_pdl", "complete_counts", "direct"),
    [
        pytest.param(
            9,
            False,
            [0] * 8,
            False,
            id="empty-query-group",
        ),
        pytest.param(
            9,
            True,
            [512] * 8,
            False,
            id="all-blocks-fit-topk",
        ),
        pytest.param(
            9,
            False,
            [0, 1, 63, 511, 512, 513, 575, 576],
            False,
            id="partial-mma-tile",
        ),
        pytest.param(
            64,
            False,
            [0, 1, 511, 512, 513, 1024, 3073, 4096],
            False,
            id="one-stream-tile",
        ),
        pytest.param(
            64,
            True,
            [0, 1, 511, 512, 513, 1024, 3073, 4096],
            False,
            id="one-stream-tile-pdl",
        ),
        pytest.param(
            65,
            False,
            [0, 511, 512, 513, 4095, 4096, 4097, 4160],
            False,
            id="partial-second-stream-tile",
        ),
        pytest.param(
            65,
            True,
            [0, 511, 512, 513, 4095, 4096, 4097, 4160],
            False,
            id="partial-second-stream-tile-pdl",
        ),
        pytest.param(
            128,
            False,
            [511, 512, 513, 4095, 4096, 4097, 8191, 8192],
            True,
            id="two-stream-tiles",
        ),
        pytest.param(
            129,
            False,
            [4095, 4096, 4097, 8191, 8192, 8193, 8255, 8256],
            True,
            id="three-stream-tiles",
        ),
        pytest.param(
            193,
            False,
            [4095, 4096, 4097, 8191, 8192, 8193, 12351, 12352],
            True,
            id="four-stream-tiles",
        ),
    ],
)
def test_cute_dsl_qsa_stream_topk_matches_reference(
    device: str,
    pages: int,
    enable_pdl: bool,
    complete_counts: list[int],
    direct: bool,
) -> None:
    torch.manual_seed(83)
    rows, heads, head_dim = 8, 4, 128
    page_size, block_topk = 64, 512
    query_storage = torch.randn(
        rows, heads + 1, head_dim, device=device, dtype=torch.bfloat16
    )
    query = query_storage[:, :heads]
    key_cache = torch.randn(
        pages * page_size, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    page_table = torch.randperm(pages, device=device).to(torch.int32).unsqueeze(0)
    request_indices = torch.zeros(rows, device=device, dtype=torch.int64)
    complete_blocks = torch.tensor(complete_counts, device=device, dtype=torch.int32)

    if direct:
        selected = cute_dsl_qsa_stream_topk(
            query,
            key_cache,
            page_table,
            request_indices,
            complete_blocks,
            page_size=page_size,
            block_topk=block_topk,
            queries_per_request=rows,
            max_partial_bytes=32 * 1024 * 1024,
            enable_pdl=enable_pdl,
        )
    else:
        selected = qwen4_exp_qsa_block_topk(
            query,
            key_cache,
            page_table,
            request_indices,
            complete_blocks,
            page_size=page_size,
            block_topk=block_topk,
            queries_per_request=rows,
            max_partial_bytes=32 * 1024 * 1024,
            path="stream",
            max_logits_bytes=256 << 20,
            solution="cute_dsl",
            override=None,
            enable_pdl=enable_pdl,
        )
    _assert_matches_reference(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        selected,
        page_size,
        block_topk,
    )


@pytest.mark.parametrize(
    ("queries_per_request", "pages"), [(8, 65), (136, 32), (136, 64), (136, 128)]
)
def test_cute_dsl_qsa_stream_topk_keeps_groups_within_requests(
    device: str,
    queries_per_request: int,
    pages: int,
) -> None:
    torch.manual_seed(97)
    rows = 2 * queries_per_request
    heads, head_dim = 4, 128
    page_size, block_topk = 64, 512
    query_storage = torch.randn(
        rows, heads + 1, head_dim, device=device, dtype=torch.bfloat16
    )
    query = query_storage[:, :heads]
    key_cache = torch.randn(
        2 * pages * page_size,
        1,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    first_pages = torch.arange(pages, device=device, dtype=torch.int32)
    second_pages = torch.arange(
        2 * pages - 1, pages - 1, -1, device=device, dtype=torch.int32
    )
    page_table = torch.empty((2, 1025), device=device, dtype=torch.int32)[:, :pages]
    page_table.copy_(torch.stack((first_pages, second_pages)))
    request_indices = torch.arange(
        2, device=device, dtype=torch.int32
    ).repeat_interleave(queries_per_request)
    complete_counts = [0, 511, 512, 513, 4095, 4096, 4097, 4160]
    complete_blocks = torch.tensor(
        [complete_counts[row % 8] for row in range(rows)],
        device=device,
        dtype=torch.int32,
    )

    selected = qwen4_exp_qsa_block_topk(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        queries_per_request=queries_per_request,
        max_partial_bytes=32 * 1024 * 1024,
        path="stream",
        max_logits_bytes=256 << 20,
        solution="cute_dsl",
        override=None,
        enable_pdl=False,
    )
    _assert_matches_reference(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        selected,
        page_size,
        block_topk,
    )


def test_cute_dsl_qsa_stream_topk_multitile_ties_are_unique(
    device: str,
) -> None:
    """Exercise the full-radix fallback while merging four stream tiles."""

    rows, heads, head_dim = 8, 4, 128
    page_size, pages, block_topk = 64, 193, 512
    query = torch.zeros(rows, heads, head_dim, device=device, dtype=torch.bfloat16)
    key_cache = torch.zeros(
        pages * page_size, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    page_table = torch.randperm(pages, device=device).to(torch.int32).unsqueeze(0)
    request_indices = torch.zeros(rows, device=device, dtype=torch.int64)
    complete_blocks = torch.full(
        (rows,), pages * page_size, device=device, dtype=torch.int32
    )

    tied = cute_dsl_qsa_stream_topk(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        queries_per_request=rows,
        max_partial_bytes=32 * 1024 * 1024,
        enable_pdl=False,
    )
    assert bool(((tied >= 0) & (tied < pages * page_size)).all())
    for row in tied:
        assert torch.unique(row).numel() == block_topk


def test_cute_dsl_qsa_stream_topk_mixes_compact_and_full_refinement(
    device: str,
) -> None:
    """Keep row-local radix scratch isolated when selector paths diverge."""

    torch.manual_seed(109)
    rows, heads, head_dim = 8, 4, 128
    page_size, pages, block_topk = 64, 128, 512
    query = torch.randn(rows, heads, head_dim, device=device, dtype=torch.bfloat16)
    query[0].zero_()
    key_cache = torch.randn(
        pages * page_size, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    page_table = torch.randperm(pages, device=device).to(torch.int32).unsqueeze(0)
    request_indices = torch.zeros(rows, device=device, dtype=torch.int32)
    complete_blocks = torch.full(
        (rows,), pages * page_size, device=device, dtype=torch.int32
    )

    selected = cute_dsl_qsa_stream_topk(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        queries_per_request=rows,
        max_partial_bytes=32 * 1024 * 1024,
        enable_pdl=False,
    )
    _assert_matches_reference(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        selected,
        page_size,
        block_topk,
    )
    for row in selected:
        assert torch.unique(row).numel() == block_topk


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_cute_dsl_qsa_stream_topk_graph_refreshes_mixed_frontiers(
    device: str,
    enable_pdl: bool,
) -> None:
    """Replay mixed selector paths as groups grow, shrink, and become empty."""

    torch.manual_seed(127)
    rows, width, pages = 16, 8, 129
    query = torch.randn(rows, 5, 128, device=device, dtype=torch.bfloat16)[:, :4]
    query[::3].zero_()
    keys = torch.randn(2 * pages * 64, 1, 128, device=device, dtype=torch.bfloat16)
    table = torch.stack(
        (
            torch.randperm(pages, device=device),
            torch.randperm(pages, device=device) + pages,
        )
    ).to(torch.int32)
    requests = torch.arange(2, device=device, dtype=torch.int32).repeat_interleave(
        width
    )
    complete = torch.full((rows,), pages * 64, device=device, dtype=torch.int32)

    def invoke() -> torch.Tensor:
        return cute_dsl_qsa_stream_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=width,
            max_partial_bytes=32 * 1024 * 1024,
            enable_pdl=enable_pdl,
        )

    invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = invoke()

    for counts in (
        [0, 1, 511, 512, 513, 4096, 4097, 8256],
        [8256, 4097, 4096, 513, 512, 511, 1, 0],
        [0, 1, 63, 64, 255, 511, 512, 0],
        [0] * width,
    ):
        complete.copy_(torch.tensor(counts * 2, device=device, dtype=torch.int32))
        graph.replay()
        _assert_matches_reference(
            query, keys, table, requests, complete, selected, 64, 512
        )


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_cute_dsl_qsa_stream_topk_overlap_graph_refreshes_candidates(
    device: str,
    enable_pdl: bool,
) -> None:
    """Refresh frontiers while the two producer halves progress independently."""

    torch.manual_seed(131)
    rows, width, pages = 136, 8, 129
    query = torch.randn(rows, 5, 128, device=device, dtype=torch.bfloat16)[:, :4]
    query[::3].zero_()
    requests_count = rows // width
    keys = torch.randn(
        requests_count * pages * 64,
        1,
        128,
        device=device,
        dtype=torch.bfloat16,
    )
    table = torch.stack(
        [
            torch.randperm(pages, device=device) + request * pages
            for request in range(requests_count)
        ]
    ).to(torch.int32)
    requests = torch.arange(
        requests_count, device=device, dtype=torch.int32
    ).repeat_interleave(width)
    complete = torch.full((rows,), 8192, device=device, dtype=torch.int32)

    def invoke() -> torch.Tensor:
        return cute_dsl_qsa_stream_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=width,
            max_partial_bytes=0,
            enable_pdl=enable_pdl,
        )

    # More than 128 rows selects the grouped overlap implementation.
    invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = invoke()

    frontiers = [511, 512, 513, 1023, 1024, 1025, 6143, 8192]
    for counts in (
        frontiers,
        list(reversed(frontiers)),
        [0] * 4 + [8192] * 4,
        [8192] * 4 + [0] * 4,
        [512] * 4 + [8192] * 4,
        [8192] * 4 + [512] * 4,
        [0] * width,
    ):
        complete.copy_(
            torch.tensor(counts * requests_count, device=device, dtype=torch.int32)
        )
        graph.replay()
        _assert_matches_reference(
            query, keys, table, requests, complete, selected, 64, 512
        )


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_cute_dsl_qsa_stream_topk_prefill_graph_without_global_partial(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    enable_pdl: bool,
) -> None:
    """Keep 16K-query selection in the CTA under a fixed 256K table capacity."""

    torch.manual_seed(131)
    rows, capacity_pages, page_size, block_topk = 16384, 1024, 64, 512
    query = torch.randn(rows, 5, 128, device=device, dtype=torch.bfloat16)[:, :4]
    query[::257].zero_()
    keys = torch.randn(
        capacity_pages * page_size, 1, 128, device=device, dtype=torch.bfloat16
    )
    table = torch.zeros((1, capacity_pages), device=device, dtype=torch.int32)
    table[0, :128] = torch.arange(128, device=device, dtype=torch.int32)
    requests = torch.zeros(rows, device=device, dtype=torch.int32)
    positions = torch.arange(rows, device=device, dtype=torch.int32)
    complete = 4096 + (positions + 1) // 4
    sampled_rows = torch.cat(
        [
            torch.arange(start, start + 8, device=device)
            for start in (0, 4096, 8192, rows - 8)
        ]
    )

    def invoke() -> torch.Tensor:
        return qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=page_size,
            block_topk=block_topk,
            queries_per_request=rows,
            max_partial_bytes=0,
            path="stream",
            max_logits_bytes=256 << 20,
            solution="cute_dsl",
            override=None,
            enable_pdl=enable_pdl,
        )

    invoke()
    torch.cuda.synchronize()
    allocations = []
    original_empty = torch.empty

    def record_empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        allocations.append((tuple(result.shape), result.dtype))
        return result

    monkeypatch.setattr(torch, "empty", record_empty)
    complete.fill_(capacity_pages * page_size)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = invoke()
    # The output is the only Torch allocation. In particular, a large table
    # must not allocate the former [2, rows, topk] packed partial workspace.
    assert allocations == [((rows, block_topk), torch.int32)]

    for kv_tokens in (32768, 16384, 49152, 262144, 32768):
        live_pages = kv_tokens // (4 * page_size)
        table[0, :live_pages] = torch.arange(
            live_pages, device=device, dtype=torch.int32
        )
        table[0, live_pages:] = 0
        complete.copy_((kv_tokens - rows) // 4 + (positions + 1) // 4)
        graph.replay()
        _assert_matches_reference(
            query[sampled_rows],
            keys,
            table[:, :live_pages],
            requests[sampled_rows],
            complete[sampled_rows],
            selected[sampled_rows],
            page_size,
            block_topk,
        )
        torch.testing.assert_close(
            (selected >= 0).sum(1), complete.clamp_max(block_topk).long()
        )
        assert bool(((selected < complete[:, None]) | (selected < 0)).all())
        ordered = selected.sort(dim=1).values
        assert bool(((ordered[:, 1:] != ordered[:, :-1]) | (ordered[:, 1:] < 0)).all())


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("request_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    ("rows", "pages", "width"),
    [
        pytest.param(8, 64, 8, id="native-linear"),
        pytest.param(1, 128, 1, id="guarded-single"),
        pytest.param(8, 128, None, id="guarded-ragged"),
        pytest.param(2, 1024, 1, id="guarded-long"),
        pytest.param(2, 1025, 1, id="guarded-mtp-padding"),
        pytest.param(64, 128, 1, id="native-many-rows"),
        pytest.param(8, 1024, 8, id="split-prefill"),
        pytest.param(256, 32, 256, id="grouped-prefill"),
        pytest.param(256, 64, 256, id="dense-prefill"),
        pytest.param(256, 128, 256, id="overlap-prefill"),
    ],
)
def test_stream_topk_graph_refreshes_pruning_and_full_refinement(
    device: str,
    rows: int,
    pages: int,
    width: int | None,
    request_dtype: torch.dtype,
    enable_pdl: bool,
) -> None:
    """Retained candidates and full-score fallbacks stay exact after replay."""

    torch.manual_seed(149)
    capacity = pages * 64
    query = torch.randn(rows, 5, 128, device=device, dtype=torch.bfloat16)[:, :4]
    keys = torch.randn((pages + 1) * 64, 1, 128, device=device, dtype=torch.bfloat16)
    table = torch.arange(1, pages + 1, device=device, dtype=torch.int32)[None]
    requests = torch.zeros(rows, device=device, dtype=request_dtype)
    complete = torch.full((rows,), capacity, device=device, dtype=torch.int32)

    def invoke() -> torch.Tensor:
        return qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=width,
            max_partial_bytes=32 * 1024 * 1024,
            path="stream",
            max_logits_bytes=256 << 20,
            solution="cute_dsl",
            override=None,
            enable_pdl=enable_pdl,
        )

    invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = invoke()
    samples = torch.linspace(0, rows - 1, min(rows, 8), device=device).long()
    frontiers = torch.tensor(
        [-1, 0, 1, 511, 512, 513, 2048, 2049, capacity, capacity + 1],
        device=device,
        dtype=torch.int32,
    )
    row_ids = torch.arange(rows, device=device)
    for distribution in ("random", "ties", "concentrated", "tiny", "large"):
        query.normal_()
        keys.normal_()
        if distribution == "ties":
            query.zero_()
        elif distribution == "concentrated":
            # A guard retaining only a few keys per tile must reject this
            # candidate pool: all 512 winners occupy only eight key tiles.
            query.fill_(1)
            keys.zero_()
            keys[64:576].fill_(1)
        elif distribution == "tiny":
            query.mul_(0.001)
            keys.mul_(0.001)
        elif distribution == "large":
            query.mul_(1000)
            keys.mul_(1000)
        table.copy_(table.flip(1))
        for step in range(frontiers.numel()):
            complete.copy_(frontiers[(row_ids + step) % frontiers.numel()])
            graph.replay()
            _assert_matches_reference(
                query[samples],
                keys,
                table,
                requests[samples],
                complete[samples],
                selected[samples],
                64,
                512,
            )
            torch.testing.assert_close(
                (selected >= 0).sum(1), complete.clamp(0, 512).long()
            )
            assert bool(((selected < complete[:, None]) | (selected == -1)).all())
            ordered = selected.sort(dim=1).values
            assert bool(
                ((ordered[:, 1:] != ordered[:, :-1]) | (ordered[:, 1:] == -1)).all()
            )


@pytest.mark.parametrize("guarded", [False, True])
def test_stream_topk_small_shapes_reuse_capacity_buckets(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    guarded: bool,
) -> None:
    torch.manual_seed(151)
    monkeypatch.setattr(stream_topk, "_COMPILED", {})
    compile_count = 0
    original_compile = stream_topk._compile

    def counted_compile(*args):
        nonlocal compile_count
        compile_count += 1
        return original_compile(*args)

    monkeypatch.setattr(stream_topk, "_compile", counted_compile)
    layouts = (
        ((1, 129), (2, 160), (3, 255), (7, 256))
        if guarded
        else (
            (9, 33),
            (10, 40),
            (12, 63),
            (16, 64),
        )
    )
    for rows, pages in layouts:
        query = torch.randn(rows, 5, 128, device=device, dtype=torch.bfloat16)[:, :4]
        keys = torch.randn(pages * 64, 1, 128, device=device, dtype=torch.bfloat16)
        table = torch.randperm(pages, device=device).to(torch.int32)[None]
        requests = torch.zeros(rows, device=device, dtype=torch.int32)
        complete = torch.linspace(513, pages * 64, rows, device=device).to(torch.int32)
        selected = cute_dsl_qsa_stream_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=None,
            max_partial_bytes=32 * 1024 * 1024,
            enable_pdl=True,
        )
        _assert_matches_reference(
            query, keys, table, requests, complete, selected, 64, 512
        )
    assert compile_count == 1


@pytest.mark.parametrize(
    ("max_partial_bytes", "startup_rows", "startup_path", "max_logits_bytes"),
    [
        pytest.param(0, 1, "auto", 256 << 20, id="auto-logits-zero-scratch"),
        pytest.param(0, 8, "auto", 0, id="auto-stream-zero-scratch"),
        pytest.param(0, 8, "stream", 256 << 20, id="stream-zero-scratch"),
        pytest.param(32 << 20, 1, "stream", 256 << 20, id="stream-scratch"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_stream_topk_startup_warms_algorithm_and_capacity_buckets(
    device,
    monkeypatch,
    max_partial_bytes,
    startup_rows,
    startup_path,
    max_logits_bytes,
    enable_pdl,
):
    from tokenspeed_kernel.ops.tuning import kernel_warmup
    from tokenspeed_kernel.selection import kernel_override

    monkeypatch.setattr(stream_topk, "_COMPILED", {})
    monkeypatch.setattr(stream_topk, "_WARMED", set())
    torch.manual_seed(157)
    keys = torch.randn(260 * 64, 1, 128, device=device, dtype=torch.bfloat16)
    table_storage = (
        torch.arange(260, device=device, dtype=torch.int32)[None]
        + torch.arange(256, device=device, dtype=torch.int32)[:, None]
    ) % 260

    def inputs(rows, pages, width):
        query = torch.randn(rows, 4, 128, device=device, dtype=torch.bfloat16)
        if width is None:
            requests = torch.zeros(rows, device=device, dtype=torch.int32)
            request_count = 1
        else:
            requests = torch.arange(rows, device=device, dtype=torch.int32) // width
            request_count = rows // width
        complete = torch.linspace(0, pages * 64, rows, device=device).int()
        if rows == 1:
            complete.fill_(pages * 64)
        return query, table_storage[:request_count, :pages], requests, complete

    def forward(values, width, *, path, solution):
        query, table, requests, complete = values
        return qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=width,
            max_partial_bytes=max_partial_bytes,
            path=path,
            max_logits_bytes=max_logits_bytes,
            solution=solution,
            override=None,
            enable_pdl=enable_pdl,
        )

    # A tiny auto-path startup can use logits even though later grouped
    # prefill calls use stream kernels without any global scratch.
    # The logits startup uses its portable selector; this test targets CuTe warmup.
    with (
        kernel_warmup(enabled=True),
        kernel_override("attention", "qsa_block_topk_select", "triton"),
    ):
        forward(
            inputs(startup_rows, 16, startup_rows),
            startup_rows,
            path=startup_path,
            solution=None if startup_path == "auto" else "cute_dsl",
        )
    compiled = dict(stream_topk._COMPILED)
    # Scan all table widths and legal request layouts within this row budget,
    # independently of the representative shapes chosen by warmup.
    layouts = [(rows, None) for rows in range(1, 257)]
    layouts.extend(
        (rows, width) for width in range(1, 257) for rows in range(width, 257, width)
    )
    configurations = {}
    for pages in range(1, table_storage.shape[1] + 1):
        for rows, width in layouts:
            config = stream_topk._configuration(
                rows, pages * 64, width, max_partial_bytes
            )
            if config is not None:
                configurations.setdefault(config, (rows, pages, width))
    for rows, pages, width in configurations.values():
        values = inputs(rows, pages, width)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            selected = forward(values, width, path="stream", solution="cute_dsl")
        assert stream_topk._COMPILED == compiled
        graph.replay()
        query, table, requests, complete = values
        row_ids = [0, rows // 2, rows - 1]
        if width is not None and rows > width:
            row_ids.extend((width - 1, width))
        _assert_matches_reference(
            query[row_ids],
            keys,
            table,
            requests[row_ids],
            complete[row_ids],
            selected[row_ids],
            64,
            512,
        )
    assert stream_topk._COMPILED == compiled


@pytest.mark.parametrize("override_source", ["context", "environment"])
@pytest.mark.parametrize("entry", ["auto-logits", "auto-stream", "stream", "direct"])
def test_stream_topk_startup_respects_kernel_override(
    device, monkeypatch, override_source, entry
):
    from tokenspeed_kernel.ops.tuning import kernel_warmup
    from tokenspeed_kernel.selection import kernel_override

    monkeypatch.setattr(stream_topk, "_COMPILED", {})
    monkeypatch.setattr(stream_topk, "_WARMED", set())
    torch.manual_seed(181)
    rows = 1 if entry == "auto-logits" else 8
    query = torch.randn(rows, 4, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn(1024, 1, 128, device=device, dtype=torch.bfloat16)
    table = torch.arange(16, device=device, dtype=torch.int32)[None]
    requests = torch.zeros(rows, device=device, dtype=torch.int32)
    complete = torch.full((rows,), 1024, device=device, dtype=torch.int32)
    kwargs = dict(
        page_size=64,
        block_topk=512,
        queries_per_request=rows,
        max_partial_bytes=32 << 20,
        enable_pdl=False,
        override="cute_dsl",
        solution=None,
    )
    if override_source == "environment":
        monkeypatch.setenv(
            "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_QSA_BLOCK_TOPK_STREAM", "triton"
        )
        context = nullcontext()
    else:
        context = kernel_override("attention", "qsa_block_topk_stream", "triton")
    with (
        context,
        kernel_override("attention", "qsa_block_scores", "triton"),
        kernel_override("attention", "qsa_block_topk_select", "triton"),
        kernel_warmup(enabled=True),
    ):
        if entry == "direct":
            selected = qsa_block_topk_stream(
                query, keys, table, requests, complete, **kwargs
            )
        else:
            selected = qwen4_exp_qsa_block_topk(
                query,
                keys,
                table,
                requests,
                complete,
                path="stream" if entry == "stream" else "auto",
                max_logits_bytes=0 if entry == "auto-stream" else 256 << 20,
                **kwargs,
            )
    _assert_matches_reference(query, keys, table, requests, complete, selected, 64, 512)
    assert not stream_topk._COMPILED
    assert not stream_topk._WARMED


@pytest.mark.parametrize("override_source", ["context", "environment"])
@pytest.mark.parametrize("entry", ["direct", "auto-logits"])
def test_stream_topk_startup_warms_override_with_conflicting_solution(
    device, monkeypatch, override_source, entry
):
    from tokenspeed_kernel.ops.tuning import kernel_warmup
    from tokenspeed_kernel.selection import kernel_override

    monkeypatch.setattr(stream_topk, "_COMPILED", {})
    monkeypatch.setattr(stream_topk, "_WARMED", set())
    env_key = "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_QSA_BLOCK_TOPK_STREAM"
    monkeypatch.delenv(env_key, raising=False)
    forced = "cute_dsl_qwen4_exp_qsa_block_topk"
    if override_source == "environment":
        monkeypatch.setenv(env_key, forced)
        context = nullcontext()
    else:
        context = kernel_override("attention", "qsa_block_topk_stream", forced)

    torch.manual_seed(197)
    keys = torch.randn(1024, 1, 128, device=device, dtype=torch.bfloat16)
    table = torch.arange(16, device=device, dtype=torch.int32)[None]

    def inputs(rows):
        query = torch.randn(rows, 4, 128, device=device, dtype=torch.bfloat16)
        requests = torch.zeros(rows, device=device, dtype=torch.int32)
        complete = torch.full((rows,), 1024, device=device, dtype=torch.int32)
        return query, requests, complete

    def forward(query, requests, complete, *, startup):
        kwargs = dict(
            page_size=64,
            block_topk=512,
            queries_per_request=query.shape[0],
            max_partial_bytes=32 << 20,
            enable_pdl=False,
            override=None,
            solution="triton",
        )
        if startup and entry == "auto-logits":
            return qwen4_exp_qsa_block_topk(
                query,
                keys,
                table,
                requests,
                complete,
                path="auto",
                max_logits_bytes=256 << 20,
                **kwargs,
            )
        return qsa_block_topk_stream(query, keys, table, requests, complete, **kwargs)

    with context:
        with kernel_warmup(enabled=True):
            first = inputs(1)
            selected = forward(*first, startup=True)
        _assert_matches_reference(
            first[0], keys, table, first[1], first[2], selected, 64, 512
        )
        compiled = dict(stream_topk._COMPILED)
        assert compiled
        for rows in (8, 136):
            query, requests, complete = inputs(rows)
            forward(query, requests, complete, startup=False)
            assert stream_topk._COMPILED == compiled
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                selected = forward(query, requests, complete, startup=False)
            query.mul_(0.75)
            graph.replay()
            row_ids = [0, rows - 1]
            _assert_matches_reference(
                query[row_ids],
                keys,
                table,
                requests[row_ids],
                complete[row_ids],
                selected[row_ids],
                64,
                512,
            )
            assert stream_topk._COMPILED == compiled


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize(
    ("rows", "width", "pages"),
    [
        pytest.param(2, 1, 1025, id="compact-decode"),
        pytest.param(16, 1, 1025, id="independent-decode"),
        pytest.param(8, 8, 1025, id="split-prefill"),
        pytest.param(136, 136, 1025, id="overlap-prefill"),
        pytest.param(136, 136, 2048, id="overlap-second-window"),
        pytest.param(129, 129, 2048, id="overlap-second-window-tail"),
        pytest.param(137, None, 2048, id="overlap-second-window-ragged"),
        pytest.param(2, 1, 2049, id="portable-long-capacity"),
    ],
)
def test_stream_topk_graph_preserves_ids_above_65535(
    device: str,
    rows: int,
    width: int | None,
    pages: int,
    enable_pdl: bool,
) -> None:
    """MTP padding must neither wrap high ids nor hide them after graph capture."""

    capacity = pages * 64
    query = torch.ones(rows, 4, 128, device=device, dtype=torch.bfloat16)
    keys = torch.zeros(capacity, 1, 128, device=device, dtype=torch.bfloat16)
    table = torch.randperm(pages, device=device).to(torch.int32)[None]
    requests = torch.zeros(rows, device=device, dtype=torch.int64)
    complete = torch.full((rows,), capacity, device=device, dtype=torch.int32)
    winners = torch.arange(capacity - 512, capacity, device=device)
    slots = table[0, winners // 64].long() * 64 + winners % 64
    keys[slots, 0, 0] = torch.arange(1, 513, device=device).to(torch.bfloat16)

    def invoke() -> torch.Tensor:
        return qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=width,
            max_partial_bytes=32 * 1024 * 1024,
            path="stream",
            max_logits_bytes=256 << 20,
            solution="triton" if pages > 2048 else "cute_dsl",
            override=None,
            enable_pdl=enable_pdl,
        )

    invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = invoke()
    for frontier in (capacity, 65536, 65537, capacity - 1, 512, 0, capacity):
        complete.fill_(frontier)
        graph.replay()
        _assert_matches_reference(
            query[:2], keys, table, requests[:2], complete[:2], selected[:2], 64, 512
        )
        torch.testing.assert_close(
            (selected >= 0).sum(1), complete.clamp(0, 512).long()
        )
        if frontier == capacity:
            torch.testing.assert_close(
                selected.sort(1).values, winners.int()[None].expand(rows, -1)
            )
    if rows == 136:
        # One producer half crosses the offset window while the other finishes
        # earlier; replay must refresh both queues' independent stage counts.
        counts = [capacity] * 4 + [65537] * 4
        complete.copy_(
            torch.tensor(counts * (rows // 8), device=device, dtype=torch.int32)
        )
        graph.replay()
        sampled_rows = torch.tensor([0, 3, 4, 7, rows - 4, rows - 1], device=device)
        _assert_matches_reference(
            query[sampled_rows],
            keys,
            table,
            requests[sampled_rows],
            complete[sampled_rows],
            selected[sampled_rows],
            64,
            512,
        )
        torch.testing.assert_close(
            (selected >= 0).sum(1), complete.clamp(0, 512).long()
        )
        complete.fill_(capacity)

    # Sparse winners across every split let guarded decode accept its compact
    # pool, while the concentrated tail above exercises its full-score fallback.
    keys.zero_()
    sparse_winners = torch.arange(512, device=device) * ((pages // 512) * 64)
    sparse_winners[-1] = capacity - 1
    slots = table[0, sparse_winners // 64].long() * 64 + sparse_winners % 64
    keys[slots, 0, 0] = 1
    graph.replay()
    torch.testing.assert_close(
        selected.sort(1).values,
        sparse_winners.int().sort().values[None].expand(rows, -1),
    )


@pytest.mark.parametrize(("rows", "pages", "width"), [(8, 128, 1), (64, 1024, 64)])
def test_stream_topk_respects_global_workspace_budget(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    pages: int,
    width: int,
) -> None:
    query = torch.zeros(rows, 4, 128, device=device, dtype=torch.bfloat16)
    keys = torch.zeros(pages * 64, 1, 128, device=device, dtype=torch.bfloat16)
    table = torch.arange(pages, device=device, dtype=torch.int32)[None]
    requests = torch.zeros(rows, device=device, dtype=torch.int32)
    complete = torch.full((rows,), pages * 64, device=device, dtype=torch.int32)
    config = stream_topk._configuration(rows, pages * 64, width, 32 * 1024 * 1024)
    required = stream_topk._workspace_bytes(rows, config)
    allocations = []
    original_empty = torch.empty
    original_empty_like = torch.empty_like

    def record_empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        allocations.append(result.numel() * result.element_size())
        return result

    def record_empty_like(*args, **kwargs):
        result = original_empty_like(*args, **kwargs)
        allocations.append(result.numel() * result.element_size())
        return result

    monkeypatch.setattr(torch, "empty", record_empty)
    monkeypatch.setattr(torch, "empty_like", record_empty_like)
    for budget in (required, required - 1):
        allocations.clear()
        supported = can_use_cute_dsl_qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=width,
            max_partial_bytes=budget,
        )
        assert supported
        selected = cute_dsl_qsa_stream_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=width,
            max_partial_bytes=budget,
            enable_pdl=False,
        )
        torch.cuda.synchronize()
        assert sum(allocations) - selected.numel() * selected.element_size() <= budget
        _assert_matches_reference(
            query, keys, table, requests, complete, selected, 64, 512
        )


@pytest.mark.parametrize("pages", [32, 64])
@pytest.mark.parametrize("rows", [136, 137])
@pytest.mark.parametrize("frontier", [0, 512])
def test_stream_topk_short_frontier_preserves_output_bounds(
    device: str, pages: int, rows: int, frontier: int
) -> None:
    query = torch.zeros(rows, 4, 128, device=device, dtype=torch.bfloat16)
    keys = torch.zeros(pages * 64, 1, 128, device=device, dtype=query.dtype)
    table = torch.arange(pages, device=device, dtype=torch.int32)[None]
    requests = torch.zeros(rows, device=device, dtype=torch.int32)
    # Back the views with padding so excess CTA threads visibly corrupt the
    # output guard instead of accessing an unrelated allocator block.
    complete_storage = torch.full(
        (rows + 8,), frontier, device=device, dtype=torch.int32
    )
    complete = complete_storage[:rows]
    output_storage = torch.full((rows + 8, 512), -2, device=device, dtype=torch.int32)
    selected = output_storage[:rows]
    config = stream_topk._configuration(rows, pages * 64, rows, 0)
    compiled = stream_topk._compile(
        query, keys, table, requests, complete, selected, [], rows, config, True
    )
    compiled[0](
        query,
        keys,
        table,
        requests,
        complete,
        selected,
        query.stride(0),
        query.stride(1),
        rows,
    )
    expected = torch.arange(512, device=device, dtype=torch.int32)
    expected.masked_fill_(expected >= frontier, -1)
    torch.testing.assert_close(selected.sort(1).values, expected.expand(rows, -1))
    assert bool((output_storage[rows:] == -2).all())


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("pages", [32, 64, 128, 574])
@pytest.mark.parametrize("remainder", range(1, 8))
def test_stream_topk_request_tails(
    device: str, pages: int, remainder: int, enable_pdl: bool
) -> None:
    """Partial groups must not read/write the following request or tensor end."""
    torch.manual_seed(317)
    width, requests_count = 16 + remainder, 9
    rows, capacity = width * requests_count, pages * 64
    query = torch.randn(rows, 5, 128, device=device, dtype=torch.bfloat16)[:, :4]
    keys = torch.randn(capacity, 1, 128, device=device, dtype=query.dtype)
    table = torch.stack(
        [torch.randperm(pages, device=device) for _ in range(requests_count)]
    ).to(torch.int32)
    requests = torch.arange(requests_count, device=device, dtype=torch.int32)
    requests = requests.repeat_interleave(width)
    frontiers = torch.tensor(
        [0, 1, 511, 512, 513, capacity - 1, capacity], device=device, dtype=torch.int32
    )
    complete = frontiers[torch.arange(rows, device=device) % frontiers.numel()]
    complete[:width] = 0
    complete[width : 2 * width] = 512
    query[:width] = float("nan")

    def invoke() -> torch.Tensor:
        return qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=width,
            max_partial_bytes=0,
            max_logits_bytes=0,
            path="auto",
            enable_pdl=enable_pdl,
            solution="cute_dsl",
            override=None,
        )

    invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = invoke()
    # Exercise both sides of every request boundary and the final partial group.
    sample = torch.cat(
        [
            torch.arange(requests_count, device=device) * width,
            torch.arange(1, requests_count + 1, device=device) * width - 1,
            torch.arange(rows - remainder, rows, device=device),
        ]
    ).unique()
    for step in range(2):
        if step:
            query[:width].zero_()
            complete.copy_(capacity - complete)
            table.copy_(table.roll(1, dims=0))
        graph.replay()
        torch.testing.assert_close(
            (selected >= 0).sum(1), complete.clamp_max(512).long()
        )
        assert bool(((selected < complete[:, None]) | (selected < 0)).all())
        ordered = selected.sort(1).values
        assert bool(
            ((ordered[:, 1:] != ordered[:, :-1]) | (ordered[:, :-1] == -1)).all()
        )
        _assert_matches_reference(
            query[sample],
            keys,
            table,
            requests[sample],
            complete[sample],
            selected[sample],
            64,
            512,
        )


@pytest.mark.parametrize("remainder", range(1, 8))
def test_stream_topk_split_request_tails(device: str, remainder: int) -> None:
    torch.manual_seed(319)
    width, pages = 8 + remainder, 256
    rows, capacity = width * 2, pages * 64
    query = torch.randn(rows, 4, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn(capacity, 1, 128, device=device, dtype=query.dtype)
    table = torch.stack(
        [torch.randperm(pages, device=device), torch.randperm(pages, device=device)]
    ).to(torch.int32)
    requests = torch.arange(2, device=device, dtype=torch.int64).repeat_interleave(
        width
    )
    complete = torch.linspace(0, capacity, rows, device=device).to(torch.int32)
    selected = cute_dsl_qsa_stream_topk(
        query,
        keys,
        table,
        requests,
        complete,
        page_size=64,
        block_topk=512,
        queries_per_request=width,
        max_partial_bytes=32 << 20,
        enable_pdl=True,
    )
    config = stream_topk._configuration(rows, capacity, width, 32 << 20)
    assert config is not None
    assert config.algorithm == "split"
    assert config.group_layout == "uniform"
    assert (query.device.index, requests.dtype, True, config) in stream_topk._COMPILED
    _assert_matches_reference(query, keys, table, requests, complete, selected, 64, 512)
