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
import tokenspeed_kernel.ops.attention.qsa.cute_dsl as cute_ops
import tokenspeed_kernel.ops.attention.qsa.triton as qsa_ops
import tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.general as general
import torch
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform().is_nvidia and torch.cuda.is_available()),
    reason="CuTe QSA streaming requires NVIDIA",
)


def _assert_selection(
    query: torch.Tensor,
    keys: torch.Tensor,
    table: torch.Tensor,
    requests: torch.Tensor,
    complete: torch.Tensor,
    selected: torch.Tensor,
    page_size: int,
    topk: int,
) -> None:
    capacity = table.shape[1] * page_size
    logical = torch.arange(capacity, device=query.device)
    slots = (
        table[requests.long()[:, None], logical[None] // page_size].long() * page_size
        + logical[None] % page_size
    )
    scores = (
        torch.einsum("rhd,rnd->rhn", query.float(), keys[slots, 0].float())
        .clamp_min_(0)
        .sum(1)
    )
    scores.masked_fill_(logical[None] >= complete[:, None], -float("inf"))
    expected = torch.topk(scores, min(topk, capacity), dim=1).values
    actual = scores.gather(1, selected.clamp_min(0).long()).masked_fill(
        selected < 0, -float("inf")
    )
    torch.testing.assert_close(
        actual.sort(1, descending=True).values[:, : expected.shape[1]],
        expected,
        rtol=1e-5,
        atol=1e-4,
    )
    torch.testing.assert_close(
        (selected >= 0).sum(1), complete.clamp(0, min(capacity, topk)).long()
    )
    assert bool(((selected < complete[:, None]) | (selected == -1)).all())
    ordered = selected.sort(1).values
    assert bool(((ordered[:, 1:] != ordered[:, :-1]) | (ordered[:, 1:] == -1)).all())


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_nvidia_stream_empty_queries(
    device: str, monkeypatch: pytest.MonkeyPatch, enable_pdl: bool
) -> None:
    monkeypatch.setattr(general, "_COMPILED", {})

    def unexpected_compile(*args, **kwargs):
        pytest.fail("Empty QSA queries must not compile or launch a kernel")

    monkeypatch.setattr(general.cute, "compile", unexpected_compile)
    query = torch.empty((0, 4, 128), device=device, dtype=torch.bfloat16)
    keys = torch.empty((64, 1, 128), device=device, dtype=query.dtype)
    table = torch.empty((1, 1), device=device, dtype=torch.int32)
    requests = torch.empty((0,), device=device, dtype=torch.int32)
    complete = torch.empty_like(requests)

    selected = cute_ops.cute_dsl_qwen4_exp_qsa_block_topk(
        query,
        keys,
        table,
        requests,
        complete,
        page_size=64,
        block_topk=512,
        queries_per_request=None,
        max_partial_bytes=0,
        enable_pdl=enable_pdl,
    )
    assert selected.shape == (0, 512)
    assert selected.dtype == torch.int32
    assert selected.device == query.device


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize(
    ("heads", "dim", "topk", "page_size", "pages", "rows", "feature_stride"),
    [
        (3, 16, 64, 64, 9, 9, 2),
        (6, 96, 128, 32, 19, 9, 2),
        (4, 128, 512, 64, 33, 136, 2),
        (4, 128, 512, 64, 33, 136, 1),
    ],
)
def test_nvidia_stream_general_graph_layouts(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    enable_pdl: bool,
    heads: int,
    dim: int,
    topk: int,
    page_size: int,
    pages: int,
    rows: int,
    feature_stride: int,
) -> None:
    torch.manual_seed(163)
    monkeypatch.setattr(
        qsa_ops,
        "_qwen4_exp_qsa_block_topk_stream",
        lambda *args, **kwargs: pytest.fail("NVIDIA stream must not call Triton"),
    )
    query = torch.randn(
        rows, heads + 1, feature_stride * dim, device=device, dtype=dtype
    )[:, :heads, ::feature_stride]
    keys = torch.randn(
        (3 * pages + 1) * page_size,
        1,
        feature_stride * dim,
        device=device,
        dtype=dtype,
    )[:, :, ::feature_stride]
    keys[:page_size].fill_(float("nan"))
    storage = torch.zeros((3, 2 * pages + 2), device=device, dtype=torch.int64)
    table = storage[:, 1:-1:2]
    table.copy_((torch.randperm(3 * pages, device=device) + 1).view(3, pages))
    requests = torch.arange(rows, device=device, dtype=torch.int64) % 3
    complete = torch.full((rows,), pages * page_size, device=device, dtype=torch.int64)

    def invoke() -> torch.Tensor:
        return qsa_ops.qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=page_size,
            block_topk=topk,
            queries_per_request=None,
            max_partial_bytes=0,
            solution="stream",
            persistent_topk_workspace=None,
            enable_pdl=enable_pdl,
        )

    invoke()
    allocations = []
    empty = torch.empty

    def record_empty(*args, **kwargs) -> torch.Tensor:
        result = empty(*args, **kwargs)
        allocations.append((tuple(result.shape), result.dtype))
        return result

    monkeypatch.setattr(torch, "empty", record_empty)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = invoke()
    assert allocations == [((rows, topk), torch.int32)]
    frontiers = torch.tensor(
        [-1, 0, 1, topk - 1, topk, topk + 1, pages * page_size, pages * page_size + 1],
        device=device,
        dtype=complete.dtype,
    )
    sample = torch.arange(min(rows, 9), device=device)
    for ties in (False, True):
        if ties:
            query.zero_()
        table.copy_(table.flip(1))
        for step in range(frontiers.numel()):
            complete.copy_(
                frontiers[
                    (torch.arange(rows, device=device) + step) % frontiers.numel()
                ]
            )
            graph.replay()
            _assert_selection(
                query[sample],
                keys,
                table,
                requests[sample],
                complete[sample],
                selected[sample],
                page_size,
                topk,
            )
            if ties:
                valid = complete.clamp(0, pages * page_size)
                expected = valid[:, None] - 1 - torch.arange(topk, device=device)[None]
                # Rows fitting top-k return all ids in unspecified order.
                expected = torch.where(expected >= 0, expected, -1).int()
                torch.testing.assert_close(
                    selected.sort(1).values, expected.sort(1).values
                )


def test_nvidia_stream_general_reuses_compile_for_strides_and_alignment(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(general, "_COMPILED", {})
    monkeypatch.setattr(
        qsa_ops,
        "_qwen4_exp_qsa_block_topk_stream",
        lambda *args, **kwargs: pytest.fail("NVIDIA stream must not call Triton"),
    )
    # Exercise the non-Blackwell dispatch even on a Blackwell test device.
    monkeypatch.setattr(cute_ops, "_IS_NVIDIA_BLACKWELL", False)
    compile_count = 0
    compile_op = general.cute.compile

    def counted_compile(*args, **kwargs):
        nonlocal compile_count
        compile_count += 1
        return compile_op(*args, **kwargs)

    monkeypatch.setattr(general.cute, "compile", counted_compile)
    for rows, pages, stride, offset in (
        (3, 9, 1, 0),
        (6, 11, 2, 0),
        (4, 13, 3, 0),
        (5, 10, 1, 1),
    ):
        # The offset also changes base-pointer alignment while retaining
        # contiguous features, so vector-copy eligibility must stay dynamic.
        query = torch.randn(
            rows, 4, 128 * stride + offset, device=device, dtype=torch.bfloat16
        )[:, :, offset::stride]
        keys = torch.randn(
            pages * 64, 1, 128 * stride + offset, device=device, dtype=query.dtype
        )[:, :, offset::stride]
        table = torch.zeros(
            (rows, pages * stride + offset), device=device, dtype=torch.int32
        )[:, offset::stride]
        table.copy_(torch.arange(pages, device=device, dtype=torch.int32)[None])
        requests = torch.zeros(
            rows * stride + offset, device=device, dtype=torch.int32
        )[offset::stride]
        complete = torch.full(
            (rows * stride + offset,), pages * 64, device=device, dtype=torch.int32
        )[offset::stride]
        selected = qsa_ops.qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=1,
            max_partial_bytes=32 * 1024 * 1024,
            solution="stream",
            persistent_topk_workspace=None,
            enable_pdl=False,
        )
        _assert_selection(query, keys, table, requests, complete, selected, 64, 512)
    assert compile_count == 1
