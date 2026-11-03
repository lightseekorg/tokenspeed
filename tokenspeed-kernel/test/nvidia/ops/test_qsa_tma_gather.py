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


def _plan(page, split, pdl, *, query_group_size):
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_tma_gather import (
        TmaGatherSparseAttention,
    )

    return TmaGatherSparseAttention(
        page_size=page,
        split_kv=split,
        enable_pdl=pdl,
        device=torch.device("cuda"),
        query_group_size=query_group_size,
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


def _reference(case, page, *, probability_dtype):
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
                if probability_dtype is None:
                    result[b, j, h] = scores.softmax(0) @ v[slots, h // 12].double()
                else:
                    # Only for one-KV-tile cases: preserve the kernel's BF16 P
                    # boundary while retaining the unrounded denominator.
                    weights = (scores - scores.max()).exp()
                    result[b, j, h] = (
                        weights.to(probability_dtype).double()
                        @ v[slots, h // 12].double()
                    ) / weights.sum()
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
    plan = _plan(page, split, pdl, query_group_size=1)
    output, workspace = _buffers(plan, case[0])
    _run(plan, case, output, workspace)
    torch.testing.assert_close(
        output.cpu().double(),
        _reference(case, page, probability_dtype=None),
        atol=2e-3,
        rtol=2e-2,
    )


@pytest.mark.parametrize("pdl", [False, True])
def test_prepared_tactics_can_coexist(pdl):
    prepared = []
    for page, split in ((4, 1), (64, 4), (4, 8), (64, 16)):
        case = _case(2, 4, 131, page, strided=True)
        plan = _plan(page, split, pdl, query_group_size=1)
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
            output.cpu().double(),
            _reference(case, page, probability_dtype=None),
            atol=2e-3,
            rtol=2e-2,
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
    plan = _plan(4, split, pdl, query_group_size=1)
    output, workspace = _buffers(plan, q)
    _run(plan, case, output, workspace)
    # Four valid values equal one. Invalid selected blocks or the implicit tail
    # must contribute neither values nor softmax weight, regardless of split.
    torch.testing.assert_close(output, torch.ones_like(output), atol=0, rtol=0)


@pytest.mark.parametrize("split", [1, 4, 8, 16])
@pytest.mark.parametrize("pdl", [False, True])
def test_sm107_graph_refresh_and_poisoned_padding(split, pdl):
    case = list(_case(2, 4, 131, 4, strided=True))
    plan = _plan(4, split, pdl, query_group_size=1)
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
        output.cpu().double(),
        _reference(case, 4, probability_dtype=None),
        atol=2e-3,
        rtol=2e-2,
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

    plan = _plan(64, 1, pdl, query_group_size=1)
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
    plan = _plan(64, 4, False, query_group_size=1)
    for b, s, length in [(1, 1, 129), (2, 4, 259), (3, 1, 1027)]:
        case = _case(b, s, length, 64, strided=True)
        out, ws = _buffers(plan, case[0])
        _run(plan, case, out, ws)
        torch.testing.assert_close(
            out.cpu().double(),
            _reference(case, 64, probability_dtype=None),
            atol=2e-3,
            rtol=2e-2,
        )


def test_sm107_rejects_unsupported_inputs():
    with pytest.raises(ValueError, match="page_size"):
        _plan(3, 1, False, query_group_size=1)
    with pytest.raises(ValueError, match="split_kv"):
        _plan(4, 2, False, query_group_size=1)
    case = list(_case(1, 1, 129, 4, strided=False))
    plan = _plan(4, 4, False, query_group_size=1)
    out, ws = _buffers(plan, case[0])
    with pytest.raises(ValueError, match="workspace"):
        _run(plan, case, out, ws[:16])
    case[1] = case[1].to(torch.float16)
    with pytest.raises(ValueError, match="BF16"):
        _run(plan, case, out, ws)


@pytest.mark.parametrize("split", [1, 4, 8, 16])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("length", [7, 8, 9, 10])
def test_paired_queries_keep_independent_causal_tails(split, pdl, length):
    # Cover every alignment of a pair against four-token block boundaries.
    case = list(_case(2, 2, length, 4, strided=True))
    case[0].mul_(0.25)
    case[3].copy_(torch.arange(512, dtype=torch.int32, device="cuda"))
    case[5][1, 0] = 0
    plan = _plan(4, split, pdl, query_group_size=2)
    output, workspace = _buffers(plan, case[0])
    _run(plan, case, output, workspace)
    torch.testing.assert_close(
        output.cpu().double(),
        _reference(case, 4, probability_dtype=torch.bfloat16),
        atol=2e-3,
        rtol=2e-2,
    )


@pytest.mark.parametrize("split", [1, 4, 8, 16])
@pytest.mark.parametrize("pdl", [False, True])
def test_paired_queries_wrap_and_refresh_graph(split, pdl):
    batch = torch.cuda.get_device_properties("cuda").multi_processor_count
    case = list(_case(batch, 4, 1027, 64, strided=True))
    q, k, v, ids, lengths, table = case
    q.zero_()
    k.zero_()
    ids.copy_(torch.arange(512, dtype=torch.int32, device="cuda"))
    lengths.copy_(
        torch.tensor([7, 8, 129, 1027], device="cuda").repeat(math.ceil(batch / 4))[
            :batch
        ]
    )
    v.fill_(0.5)
    v[:64].fill_(float("nan"))
    table[1::4].zero_()
    expected = torch.full_like(q, 0.5)
    expected[1::4].zero_()
    plan = _plan(64, split, pdl, query_group_size=2)
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
    lengths.zero_()
    q.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)


@pytest.mark.parametrize("queries", [2, 4])
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("pdl", [False, True])
def test_paired_queries_keep_head_columns_separate(queries, split, pdl):
    # Random query/head/channel values expose mixed-up half-head groups.
    # One K/V tile allows an independent reference with BF16 probabilities.
    case = list(_case(3, queries, 31, 64, strided=True))
    case[0].mul_(0.25)
    case[3].copy_(torch.arange(512, dtype=torch.int32, device="cuda"))
    plan = _plan(64, split, pdl, query_group_size=2)
    output, workspace = _buffers(plan, case[0])
    _run(plan, case, output, workspace)
    torch.testing.assert_close(
        output.cpu().double(),
        _reference(case, 64, probability_dtype=torch.bfloat16),
        atol=2e-3,
        rtol=2e-2,
    )


def test_paired_plan_requires_even_query_length():
    plan = _plan(64, 1, False, query_group_size=2)
    for queries in (1, 3):
        case = _case(1, queries, 129, 64, strided=False)
        output, workspace = _buffers(plan, case[0])
        with pytest.raises(ValueError, match="divisible"):
            _run(plan, case, output, workspace)


@pytest.mark.parametrize(
    "query_group_size,queries", [(1, 5), (1, 257), (2, 6), (2, 258)]
)
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("pdl", [False, True])
def test_long_queries_match_short_chunks(query_group_size, queries, split, pdl):
    case = list(_case(2, queries, queries + 131, 64, strided=True))
    q, k, v, ids, lengths, table = case
    if query_group_size == 2:
        ids.copy_(torch.arange(512, dtype=torch.int32, device="cuda"))
    plan = _plan(64, split, pdl, query_group_size=query_group_size)
    output, workspace = _buffers(plan, q)
    _run(plan, case, output, workspace)

    # The existing short-query contract independently establishes each chunk's
    # bottom-right causal position, without changing its selection order.
    expected = torch.empty_like(q)
    for start in range(0, queries, 4):
        end = min(start + 4, queries)
        chunk = (
            q[:, start:end],
            k,
            v,
            ids[:, start:end].contiguous(),
            lengths - queries + end,
            table,
        )
        _run(plan, chunk, expected[:, start:end], workspace)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(plan, case, output, workspace)
    v.neg_()
    graph.replay()
    torch.testing.assert_close(output, -expected, atol=0, rtol=0)


def test_long_queries_reject_list_index_overflow():
    case = list(_case(1, 1, 1, 4, strided=False))
    plan = _plan(4, 1, False, query_group_size=1)
    output = torch.empty_like(case[0])
    # Expanded storage lets the host reject an oversized shape before launch.
    case[0] = case[0].expand(1, 2**22, 24, 256)
    with pytest.raises(ValueError, match="Int32 index range"):
        _run(plan, case, output, None)


@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_prefill_registry_ragged_and_live_graph_metadata(position_dtype, enable_pdl):
    from tokenspeed_kernel.ops.attention.qsa import (
        QSASparseMetadata,
        QSAWorkspace,
        qsa_sparse_attention,
        select_qsa_sparse_attention_kernel,
    )
    from tokenspeed_kernel.ops.attention.qsa.triton import qwen4_exp_qsa_selected_slots
    from tokenspeed_kernel.platform import pdl_enabled
    from tokenspeed_kernel.workspace import capture_session

    case = _case(3, 6, 140, 64, strided=True)
    q, k, v, ids, lengths, table = case
    flat = torch.cat((q[0, :5], q[2]))
    blocks = torch.cat((ids[0, :5], ids[2]))
    positions = torch.cat((torch.arange(134, 139), torch.arange(134, 140))).to(
        device="cuda", dtype=position_dtype
    )
    seq_lens = torch.tensor([139, 0, 140], dtype=torch.int32, device="cuda")
    metadata = QSASparseMetadata(
        blocks,
        table,
        torch.tensor([0] * 5 + [2] * 6, device="cuda", dtype=torch.int32),
        positions,
        (0, 5, 5, 11),
        64,
        4,
        QSAWorkspace(),
        seq_lens=seq_lens,
    )
    slots = qwen4_exp_qsa_selected_slots(
        blocks,
        ((positions + 1) // 4).to(torch.int32),
        positions,
        metadata.token_to_request,
        table,
        64,
        4,
        2048,
        enable_pdl=False,
    )
    kwargs = dict(
        max_seqlen_q=None,
        k_scale=None,
        v_scale=None,
        sparse_metadata=metadata,
        override=None,
        solution=None,
    )
    selected = select_qsa_sparse_attention_kernel(flat, k, v, slots, **kwargs)
    assert selected.name == "cute_dsl_tma_gather_qsa_prefill"

    def call():
        return qsa_sparse_attention(
            flat, k, v, slots, scale=1 / 16, metadata_capacity_rows=None, **kwargs
        )

    def reference():
        expected = _reference(case, 64, probability_dtype=None)
        return torch.cat((expected[0, :5], expected[2])).cuda().to(torch.bfloat16)

    previous_pdl = pdl_enabled()
    pdl_enabled(enable_pdl)
    try:
        torch.testing.assert_close(call(), reference(), atol=2e-3, rtol=2e-2)
        graph = torch.cuda.CUDAGraph()
        with capture_session(graph), torch.cuda.graph(graph):
            output = call()
        # Lengths belong to the forward layout and are refreshed in place.
        # Captured attention must consume them along with the updated page table.
        lengths.add_(4)
        seq_lens.add_(4)
        positions.add_(4)
        table.copy_(table.flip(0))
        v.neg_()
        graph.replay()
        torch.testing.assert_close(output, reference(), atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(output, call(), atol=0, rtol=0)
    finally:
        pdl_enabled(previous_pdl)


@pytest.mark.parametrize(
    "unsupported", ["metadata", "heads", "blocks_stride", "scale", "fp8", "decode"]
)
def test_prefill_registry_preserves_fallbacks(unsupported):
    from dataclasses import replace

    from tokenspeed_kernel.numerics.torch.attention.qsa import (
        torch_qsa_selected_attention,
    )
    from tokenspeed_kernel.ops.attention.qsa import (
        QSASparseMetadata,
        QSAWorkspace,
        qsa_sparse_attention,
        select_qsa_sparse_attention_kernel,
    )
    from tokenspeed_kernel.ops.attention.qsa.triton import qwen4_exp_qsa_selected_slots

    q, k, v, ids, lengths, table = _case(1, 1, 140, 64, strided=False)
    q, blocks = q[0], ids[0]
    positions = lengths - 1
    requests = torch.zeros((1,), dtype=torch.int32, device="cuda")
    metadata = QSASparseMetadata(
        blocks,
        table,
        requests,
        positions,
        (0, 1),
        64,
        4,
        QSAWorkspace(),
        seq_lens=lengths,
    )
    slots = qwen4_exp_qsa_selected_slots(
        blocks,
        lengths // 4,
        positions,
        requests,
        table,
        64,
        4,
        2048,
        enable_pdl=False,
    )
    key_scale = None
    value_scale = None
    width = None
    if unsupported == "metadata":
        metadata = None
    elif unsupported == "heads":
        q = q[:, :12].contiguous()
    elif unsupported == "blocks_stride":
        padded = torch.empty((1, 1024), dtype=torch.int32, device="cuda")
        padded[:, ::2].copy_(blocks)
        metadata = replace(metadata, selected_blocks=padded[:, ::2])
    elif unsupported == "scale":
        # BF16 has no FP8 descale. An explicitly supplied unit scale still
        # selects the existing fallback instead of being discarded by TMA.
        key_scale = 1.0
    elif unsupported == "fp8":
        k, v = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
        # The runtime's FP8 cache path supplies explicit unit descales.
        key_scale = value_scale = 1.0
    else:
        width = 1
    kwargs = dict(
        max_seqlen_q=width,
        k_scale=key_scale,
        v_scale=value_scale,
        sparse_metadata=metadata,
        override=None,
        solution=None,
    )
    selected = select_qsa_sparse_attention_kernel(q, k, v, slots, **kwargs)
    expected_kernel = (
        "cute_dsl_blackwell_qsa_sparse_attention"
        if width is not None
        else (
            "flashinfer_fa2_fp8_qsa_sparse_attention"
            if unsupported == "fp8"
            else "flashinfer_fa2_qsa_sparse_attention"
        )
    )
    assert selected.name == expected_kernel
    actual = qsa_sparse_attention(
        q, k, v, slots, scale=1 / 16, metadata_capacity_rows=None, **kwargs
    )
    expected = torch_qsa_selected_attention(
        q,
        k,
        v,
        slots,
        1 / 16,
        1.0 if key_scale is None else key_scale,
        1.0 if value_scale is None else value_scale,
    )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("query_width", [1, 4])
@pytest.mark.parametrize("override_source", ["callsite", "context", "environment"])
def test_cute_dsl_solution_override_preserves_decode(
    query_width, override_source, monkeypatch
):
    from contextlib import nullcontext

    from tokenspeed_kernel.ops.attention.qsa import qsa_sparse_attention
    from tokenspeed_kernel.ops.attention.qsa.triton import qwen4_exp_qsa_selected_slots
    from tokenspeed_kernel.selection import kernel_override

    case = _case(2, query_width, 140, 64, strided=False)
    q, k, v, ids, lengths, table = case
    q.mul_(0.25)
    positions = torch.arange(140 - query_width, 140, device="cuda").repeat(2)
    requests = torch.arange(2, dtype=torch.int32, device="cuda").repeat_interleave(
        query_width
    )
    slots = qwen4_exp_qsa_selected_slots(
        ids.flatten(0, 1),
        ((positions + 1) // 4).int(),
        positions,
        requests,
        table,
        64,
        4,
        2048,
        enable_pdl=False,
    )
    env_key = "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_QSA_SPARSE_ATTENTION"
    monkeypatch.delenv(env_key, raising=False)
    if override_source == "environment":
        monkeypatch.setenv(env_key, "cute_dsl")
    context = (
        kernel_override("attention", "qsa_sparse_attention", "cute_dsl")
        if override_source == "context"
        else nullcontext()
    )
    with context:
        output = qsa_sparse_attention(
            q.flatten(0, 1),
            k,
            v,
            slots,
            scale=1 / 16,
            max_seqlen_q=query_width,
            metadata_capacity_rows=None,
            k_scale=None,
            v_scale=None,
            override="cute_dsl" if override_source == "callsite" else None,
            solution=None,
            sparse_metadata=None,
        )
    torch.testing.assert_close(
        output.cpu().double(),
        _reference(case, 64, probability_dtype=None).flatten(0, 1),
        atol=2e-3,
        rtol=2e-2,
    )


@pytest.mark.parametrize(
    "invalid", ["q_alignment", "kv_stride", "blocks_stride", "table_stride"]
)
def test_dispatch_and_prepared_entry_reject_the_same_tma_layout(invalid):
    from tokenspeed_kernel.ops.attention.qsa import (
        QSASparseMetadata,
        QSAWorkspace,
        select_qsa_sparse_attention_kernel,
    )

    case = list(_case(2, 1, 140, 64, strided=False))
    q, k, v, ids, lengths, table = case
    if invalid == "q_alignment":
        case[0] = torch.empty((*q.shape[:-1], 264), dtype=q.dtype, device=q.device)[
            ..., 1:257
        ]
    elif invalid == "kv_stride":
        case[1] = torch.empty(
            (k.shape[0] * 2, *k.shape[1:]), dtype=k.dtype, device=k.device
        )[::2]
    elif invalid == "blocks_stride":
        case[3] = torch.empty((2, 1, 1024), dtype=ids.dtype, device=ids.device)[
            ..., ::2
        ]
    else:
        case[5] = torch.empty(
            (2, table.shape[1] * 2), dtype=table.dtype, device=table.device
        )[:, ::2]
    q, k, v, ids, lengths, table = case
    metadata = QSASparseMetadata(
        ids[:, 0],
        table,
        torch.arange(2, dtype=torch.int32, device="cuda"),
        lengths - 1,
        (0, 1, 2),
        64,
        4,
        QSAWorkspace(),
        seq_lens=lengths,
    )
    selected = select_qsa_sparse_attention_kernel(
        q[:, 0],
        k,
        v,
        torch.empty((2, 2051), dtype=torch.int32, device="cuda"),
        max_seqlen_q=None,
        k_scale=None,
        v_scale=None,
        sparse_metadata=metadata,
        override=None,
        solution=None,
    )
    assert selected.name == "flashinfer_fa2_qsa_sparse_attention"
    plan = _plan(64, 1, False, query_group_size=1)
    output, workspace = _buffers(plan, q)
    with pytest.raises(ValueError, match="aligned|contiguous"):
        _run(plan, case, output, workspace)
