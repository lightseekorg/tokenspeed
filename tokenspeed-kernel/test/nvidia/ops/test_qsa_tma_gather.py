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

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="TMA gather4 sparse attention requires SM107",
)


def _plan(page, split, pdl):
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_tma_gather import (
        TmaGatherSparseAttention,
    )

    return TmaGatherSparseAttention(
        page_size=page, split_kv=split, enable_pdl=pdl, device=torch.device("cuda")
    )


def _case(batch, queries, length, page, *, strided):
    generator = torch.Generator().manual_seed(719 + batch + queries + length)
    width = math.ceil(length / page)
    table = (
        (torch.randperm(batch * width, generator=generator) + 1)
        .to(torch.int32)
        .view(batch, width)
    )
    slots = (batch * width + 1) * page
    k = torch.randn(slots, 2, 256, generator=generator, dtype=torch.bfloat16).cuda()
    v = torch.randn(slots, 2, 256, generator=generator, dtype=torch.bfloat16).cuda()
    q = torch.randn(
        batch,
        queries,
        48 if strided else 24,
        256,
        generator=generator,
        dtype=torch.bfloat16,
    ).cuda()[:, :, :24]
    ids = torch.full((batch, queries, 512), -1, dtype=torch.int32)
    for b in range(batch):
        for j in range(queries):
            complete = (length - queries + j + 1) // 4
            chosen = torch.randperm(complete, generator=generator)[:512]
            ids[b, j, : len(chosen)] = chosen.int()
    table = table.cuda()
    if strided:
        padded = torch.full((batch, width + 7), -123, dtype=torch.int32, device="cuda")
        padded[:, :width].copy_(table)
        table = padded[:, :width]
    return (
        q,
        k,
        v,
        ids.cuda(),
        torch.full((batch,), length, dtype=torch.int32, device="cuda"),
        table,
    )


def _reference(case, page):
    q, k, v, ids, lengths, table = (x.cpu() for x in case)
    result = torch.zeros(q.shape, dtype=torch.float64)
    batch, queries, heads, _ = q.shape
    for b in range(batch):
        for j in range(queries):
            pos = int(lengths[b]) - queries + j
            if pos < 0:
                continue
            complete = (pos + 1) // 4
            logical = [
                int(block) * 4 + i
                for block in ids[b, j, : min(512, complete)]
                if int(block) >= 0
                for i in range(4)
            ]
            logical += list(range(complete * 4, pos + 1))
            slots = []
            for token in logical:
                if token > pos or token // page >= table.shape[1]:
                    continue
                physical = int(table[b, token // page])
                if 0 < physical < k.shape[0] // page:
                    slots.append(physical * page + token % page)
            if not slots:
                continue
            for h in range(heads):
                scores = (k[slots, h // 12].double() @ q[b, j, h].double()) / 16
                result[b, j, h] = scores.softmax(0) @ v[slots, h // 12].double()
    return result


def _buffers(plan, q):
    output = torch.empty_like(q)
    count = plan.workspace_bytes(
        batch_size=q.shape[0], query_length=q.shape[1], num_query_heads=q.shape[2]
    )
    return output, (
        torch.empty(count, dtype=torch.uint8, device="cuda") if count else None
    )


def _run(plan, case, output, workspace):
    return plan.execute(*case, scale=1 / 16, output=output, workspace=workspace)


@pytest.mark.parametrize("split", [1, 4, 8, 16])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize(
    "batch,queries,length,page,strided", [(1, 1, 129, 4, False), (2, 4, 4099, 64, True)]
)
def test_sm107_sparse_matches_reference(
    split, pdl, batch, queries, length, page, strided
):
    case = _case(batch, queries, length, page, strided=strided)
    plan = _plan(page, split, pdl)
    output, workspace = _buffers(plan, case[0])
    _run(plan, case, output, workspace)
    torch.testing.assert_close(
        output.cpu().double(), _reference(case, page), atol=2e-3, rtol=2e-2
    )


@pytest.mark.parametrize("pdl", [False, True])
def test_prepared_tactics_can_coexist(pdl):
    prepared = []
    for page, split in ((4, 1), (64, 4), (4, 8), (64, 16)):
        case = _case(2, 4, 131, page, strided=True)
        plan = _plan(page, split, pdl)
        output, workspace = _buffers(plan, case[0])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _run(plan, case, output, workspace)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run(plan, case, output, workspace)
        prepared.append((plan, case, output, workspace, graph, page))

    # Creating another tactic must not change an earlier plan's page mapping,
    # split workspace geometry or captured execution. Refresh values as well.
    for plan, case, output, workspace, graph, page in reversed(prepared):
        case[2].mul_(0.5)
        graph.replay()
        torch.cuda.synchronize()
        captured = output.clone()
        _run(plan, case, output, workspace)
        torch.testing.assert_close(output, captured, atol=0, rtol=0)
        torch.testing.assert_close(
            output.cpu().double(), _reference(case, page), atol=2e-3, rtol=2e-2
        )


@pytest.mark.parametrize("split", [1, 4, 8, 16])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("length", [7, 8])
@pytest.mark.parametrize("invalid_page", [0, -1, 3])
def test_sm107_missing_pages_have_no_softmax_weight(split, pdl, length, invalid_page):
    q = torch.zeros((1, 1, 24, 256), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros((12, 2, 256), dtype=torch.bfloat16, device="cuda")
    v = torch.ones_like(k)
    k[:4].fill_(float("nan"))
    v[:4].fill_(float("nan"))
    ids = torch.full((1, 1, 512), -1, dtype=torch.int32, device="cuda")
    ids[0, 0, :2] = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    case = (
        q,
        k,
        v,
        ids,
        torch.tensor([length], device="cuda", dtype=torch.int32),
        torch.tensor([[2, invalid_page]], device="cuda", dtype=torch.int32),
    )
    plan = _plan(4, split, pdl)
    output, workspace = _buffers(plan, q)
    _run(plan, case, output, workspace)
    # Four valid values equal one. Invalid selected blocks or the implicit tail
    # must contribute neither values nor softmax weight, regardless of split.
    torch.testing.assert_close(output, torch.ones_like(output), atol=0, rtol=0)


@pytest.mark.parametrize("split", [1, 4, 8, 16])
@pytest.mark.parametrize("pdl", [False, True])
def test_sm107_graph_refresh_and_poisoned_padding(split, pdl):
    case = list(_case(2, 4, 131, 4, strided=True))
    plan = _plan(4, split, pdl)
    output, workspace = _buffers(plan, case[0])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(plan, case, output, workspace)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(plan, case, output, workspace)
    # Padded graph requests have positive clamped lengths and a zero page table.
    # Poison Q and the null cache page to catch a masked-read/NaN leak.
    case[0][1].fill_(float("nan"))
    case[1][:4].fill_(float("nan"))
    case[2][:4].fill_(float("nan"))
    case[5][1].zero_()
    case[4][0] = 129
    case[3][0, :, 0] = -1
    graph.replay()
    torch.cuda.synchronize()
    captured = output.clone()
    _run(plan, case, output, workspace)
    torch.testing.assert_close(output, captured, atol=0, rtol=0)
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]), atol=0, rtol=0)
    torch.testing.assert_close(
        output.cpu().double(), _reference(case, 4), atol=2e-3, rtol=2e-2
    )
    case[4].zero_()
    case[0].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)


@pytest.mark.parametrize("pdl", [False, True])
def test_sm107_pipeline_wraps_across_work_items(pdl):
    # Oversubscribe the SMs so CLC can reuse a CTA for requests with different
    # tile counts. Long requests wrap the three K/V slots within one item.
    batch = 2 * torch.cuda.get_device_properties("cuda").multi_processor_count
    case = list(_case(batch, 1, 1024, 64, strided=True))
    q, k, v, ids, lengths, table = case
    q.zero_()
    k.zero_()
    lengths.copy_(
        torch.tensor([16, 64, 256, 1024], device="cuda").repeat(math.ceil(batch / 4))[
            :batch
        ]
    )
    ids.copy_(torch.arange(512, dtype=torch.int32, device="cuda"))

    # Uniform attention over power-of-two lengths has exactly representable
    # probabilities. Distinct request/head/channel values expose stale slots;
    # the token pattern also distinguishes the short and long selections.
    values = (
        (torch.arange(batch, device="cuda") % 16)[:, None, None] / 8
        + torch.arange(2, device="cuda")[None, :, None] / 4
        + (torch.arange(256, device="cuda") % 4)[None, None, :] / 8
    )
    tokens = (torch.arange(64, device="cuda") // 4 % 8) / 32
    pages = v.view(-1, 64, 2, 256)
    pages[table.long()] = (
        values[:, None, None, :, :] + tokens[None, None, :, None, None]
    ).to(v.dtype)
    mean_token = torch.where(lengths == 16, 1.5 / 32, 3.5 / 32)
    expected = (values + mean_token[:, None, None]).repeat_interleave(12, dim=1)
    expected = expected[:, None].to(q.dtype)

    plan = _plan(64, 1, pdl)
    output, workspace = _buffers(plan, q)
    _run(plan, case, output, workspace)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(plan, case, output, workspace)
    for _ in range(2):
        v.neg_()
        expected.neg_()
        graph.replay()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_sm107_prepared_kernel_reuses_runtime_shapes():
    plan = _plan(64, 4, False)
    for b, s, length in [(1, 1, 129), (2, 4, 259), (3, 1, 1027)]:
        case = _case(b, s, length, 64, strided=True)
        out, ws = _buffers(plan, case[0])
        _run(plan, case, out, ws)
        torch.testing.assert_close(
            out.cpu().double(), _reference(case, 64), atol=2e-3, rtol=2e-2
        )


def test_sm107_rejects_unsupported_inputs():
    with pytest.raises(ValueError, match="page_size"):
        _plan(3, 1, False)
    with pytest.raises(ValueError, match="split_kv"):
        _plan(4, 2, False)
    case = list(_case(1, 1, 129, 4, strided=False))
    plan = _plan(4, 4, False)
    out, ws = _buffers(plan, case[0])
    with pytest.raises(ValueError, match="workspace"):
        _run(plan, case, out, ws[:16])
    case[1] = case[1].to(torch.float16)
    with pytest.raises(ValueError, match="BF16"):
        _run(plan, case, out, ws)
