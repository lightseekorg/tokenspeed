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

"""PDL dependency tests with producers that publish inputs after early release."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsa.cuda import (
    has_ragged_decode_topk,
    ragged_decode_topk,
)
from tokenspeed_kernel.ops.attention.qsa import (
    qsa_sparse_attention,
    qwen4_exp_qsa_block_topk,
)
from tokenspeed_kernel.ops.attention.qsa.triton import (
    qwen4_exp_qsa_compress_and_store,
    qwen4_exp_qsa_prepare_metadata,
    qwen4_exp_qsa_recent_write,
    qwen4_exp_qsa_selected_slots,
)
from tokenspeed_kernel.platform import current_platform, pdl_enabled

pytestmark = pytest.mark.skipif(
    not current_platform().is_hopper_plus, reason="PDL requires NVIDIA SM90+"
)


@triton.jit
def _publish_inputs(sources, targets, sizes: tl.constexpr, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    start = tl.inline_asm_elementwise(
        "mov.u64 $0, %clock64;", "=l", [], dtype=tl.uint64, is_pure=False, pack=1
    )
    now = start
    while now - start < 200000:
        now = tl.inline_asm_elementwise(
            "mov.u64 $0, %clock64;", "=l", [], dtype=tl.uint64, is_pure=False, pack=1
        )
    for tensor in tl.static_range(len(sizes)):
        for base in range(
            tl.program_id(0) * BLOCK, sizes[tensor], tl.num_programs(0) * BLOCK
        ):
            offsets = base + tl.arange(0, BLOCK)
            values = tl.load(
                sources[tensor] + offsets, offsets < sizes[tensor], other=0
            )
            tl.store(targets[tensor] + offsets, values, offsets < sizes[tensor])


def _publish(sources, targets):
    _publish_inputs[(8,)](
        tuple(t.view(torch.uint8) for t in sources),
        tuple(t.view(torch.uint8) for t in targets),
        tuple(t.numel() * t.element_size() for t in sources),
        BLOCK=256,
        launch_pdl=True,
    )


@pytest.fixture
def restore_pdl():
    previous = pdl_enabled()
    yield
    torch.cuda.synchronize()
    pdl_enabled(previous)


def _check_graph_edges(graph, enabled):
    from cuda.bindings import driver as cu

    def checked(result):
        assert result[0] == cu.CUresult.CUDA_SUCCESS, result
        return result[1:]

    raw = cu.CUgraph(graph.raw_cuda_graph())
    _, _, _, count = checked(cu.cuGraphGetEdges(raw, 0))
    _, targets, edges, _ = checked(cu.cuGraphGetEdges(raw, count))
    tested = 0
    for target, edge in zip(targets, edges):
        (kind,) = checked(cu.cuGraphNodeGetType(target))
        if kind == cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            (params,) = checked(cu.cuGraphKernelNodeGetParams(target))
            (name,) = checked(cu.cuFuncGetName(params.func))
            assert int(edge.type) == int(enabled), (name, edge.type, enabled)
            tested += 1
    assert tested > 0


def _check_replays(forward, sources, targets, enabled):
    # Explicit kernel arguments must take precedence over the global setting.
    pdl_enabled(not enabled)
    forward(enabled)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        actual = forward(enabled)
    _check_graph_edges(graph, enabled)
    # A captured graph must retain its PDL setting after the global toggle.
    for _ in range(3):
        for source in sources:
            if source.dtype in (torch.bfloat16, torch.float32):
                source.normal_()
        pdl_enabled(False)
        expected = tuple(t.clone() for t in forward(False))
        for target in targets:
            target.fill_(float("nan") if target.is_floating_point() else 0)
        pdl_enabled(not enabled)
        graph.replay()
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("queries_per_request", [1, 3, 4])
@pytest.mark.parametrize(
    ("algorithm", "blocks", "topk"),
    [
        ("stream", 1024, 64),
        ("stream", 16384, 512),
        ("logits", 1024, 64),
        ("logits", 32768, 512),
    ],
)
def test_qsa_selection_waits_for_query_cache_and_metadata(
    enabled, queries_per_request, algorithm, blocks, topk, restore_pdl
):
    torch.manual_seed(311)
    batch, heads, dim, page_size, ratio = 3, 4, 32, 64, 4
    rows = batch * queries_per_request
    pages = blocks // page_size
    query = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(blocks + page_size, 1, dim, device="cuda", dtype=query.dtype)
    table = torch.arange(1, pages + 1, device="cuda", dtype=torch.int32).repeat(
        batch, 1
    )
    complete = torch.tensor([0, 35, blocks], device="cuda", dtype=torch.int32)
    sources = (query, cache, table, complete * ratio + 2)
    targets = tuple(torch.empty_like(t) for t in sources)
    full_table = torch.arange(
        1, pages * ratio + 2, device="cuda", dtype=torch.int32
    ).repeat(batch, 1)

    def forward(enable_pdl):
        _publish(sources, targets)
        logical, requests, _, _, complete_blocks = qwen4_exp_qsa_prepare_metadata(
            targets[3],
            queries_per_request,
            rows,
            full_table,
            page_size * ratio,
            full_table,
            page_size,
            ratio,
            enable_pdl=enable_pdl,
            draft_logical_positions=None,
        )
        selected = qwen4_exp_qsa_block_topk(
            *targets[:3],
            requests,
            complete_blocks,
            page_size=page_size,
            block_topk=topk,
            queries_per_request=queries_per_request,
            max_partial_bytes=32 * 1024 * 1024,
            algorithm=algorithm,
            persistent_topk_workspace=None,
            enable_pdl=enable_pdl,
        )
        slots = qwen4_exp_qsa_selected_slots(
            selected,
            complete_blocks,
            logical,
            requests,
            full_table,
            page_size,
            ratio,
            topk * ratio,
            enable_pdl=enable_pdl,
        )
        return selected, slots

    _check_replays(forward, sources, targets, enabled)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("write_rows", [False, True])
def test_qsa_compression_orders_ring_reuse(enabled, write_rows, restore_pdl):
    torch.manual_seed(313)
    rows, heads, dim, ratio = 3, 4, 32, 4
    logical = torch.tensor([10, 11, -1], device="cuda", dtype=torch.int64)
    requests = torch.zeros(rows, device="cuda", dtype=torch.int64)
    recent = torch.tensor([66, 67, 0], device="cuda", dtype=torch.int32)
    locations = torch.tensor([0, 67, 0], device="cuda", dtype=torch.int32)
    write_mask = (
        None if write_rows else torch.zeros(rows, device="cuda", dtype=torch.bool)
    )
    positions = torch.tensor([[10, 10, 10], [11, 11, 11], [0, 0, 0]], device="cuda")
    keys = torch.randn(rows, 1, dim, device="cuda", dtype=torch.bfloat16)
    query = torch.randn(rows, heads * dim, device="cuda", dtype=keys.dtype)
    initial_raw = torch.randn(2, ratio, 1, dim, device="cuda", dtype=keys.dtype)
    initial_positions = torch.tensor([[0, 0, 0], [8, 8, 8]], device="cuda")
    weights = torch.randn(dim, device="cuda")
    query_weights = torch.randn(dim, device="cuda")
    rope = torch.randn(32, 16, device="cuda")
    sources = (keys, query, initial_raw, weights, query_weights, initial_positions)
    targets = tuple(torch.empty_like(t) for t in sources)
    compressed = torch.zeros(2, 16, 1, dim, device="cuda", dtype=keys.dtype)

    def forward(enable_pdl):
        _publish(sources, targets)
        key, q, raw, weight, q_weight, pos_cache = targets
        out = qwen4_exp_qsa_compress_and_store(
            key,
            logical,
            requests,
            recent,
            raw,
            positions,
            pos_cache,
            weight,
            1e-6,
            rope,
            locations,
            compressed,
            64,
            ratio,
            64,
            sections=None,
            interleaved=False,
            write_mask=None,
            draft_raw_cache=None,
            draft_logical_positions=None,
            draft_position_cache=None,
            enable_pdl=enable_pdl,
            query=q,
            query_norm_weight=q_weight,
            query_norm_epsilon=1e-6,
            num_query_heads=heads,
            stage_verify_buffers=None,
            stage_draft=False,
        )
        qwen4_exp_qsa_recent_write(
            key,
            logical,
            requests,
            recent,
            positions,
            raw,
            pos_cache,
            64,
            ratio,
            write_mask=write_mask,
            request_limit=None,
            enable_pdl=enable_pdl,
        )
        return out, compressed, raw, pos_cache

    _check_replays(forward, sources, targets, enabled)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("solution", ["cute_dsl", "flashinfer"])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_qsa_attention_waits_for_slots_and_kv(
    enabled, solution, cache_dtype, restore_pdl
):
    if solution == "cute_dsl" and not current_platform().is_blackwell:
        pytest.skip("CuTe QSA requires Blackwell")
    torch.manual_seed(317)
    rows, heads, dim = 4, 6, 256
    query = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    keys, values = [
        torch.randn(4096, 1, dim, device="cuda").to(cache_dtype) for _ in range(2)
    ]
    slots = torch.randint(1, 4096, (rows, 2051), device="cuda", dtype=torch.int32)
    slots[0, :] = -1
    slots[1, 20:] = -1
    sources = (query, keys, values, slots)
    targets = tuple(torch.empty_like(t) for t in sources)

    # FA2 plans are stream-private; initialize once on the capture stream via forward.
    def forward(enable_pdl):
        pdl_enabled(enable_pdl)
        _publish(sources, targets)
        output = qsa_sparse_attention(
            *targets,
            scale=dim**-0.5,
            max_seqlen_q=4,
            metadata_capacity_rows=None,
            k_scale=1.0,
            v_scale=1.0,
            override=None,
            solution=solution,
        )
        return (output,)

    _check_replays(forward, sources, targets, enabled)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
def test_qsa_selection_refreshes_block_frontiers(enabled, persistent, restore_pdl):
    if persistent and not has_ragged_decode_topk():
        pytest.skip("CUDA persistent top-k is unavailable")
    batch, width, heads, dim, page_size, blocks = 3, 4, 4, 128, 64, 17408
    rows = batch * width
    query = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(
        (batch * blocks + page_size, 1, dim), device="cuda", dtype=query.dtype
    )
    sources = (query, cache)
    targets = tuple(torch.empty_like(t) for t in sources)
    table = torch.arange(
        1, batch * blocks // page_size + 1, device="cuda", dtype=torch.int32
    ).reshape(batch, -1)
    requests = torch.arange(batch, device="cuda").repeat_interleave(width)
    complete = torch.zeros(rows, device="cuda", dtype=torch.int32)
    workspace = (
        torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)
        if persistent
        else None
    )

    def forward(enable_pdl):
        _publish(sources, targets)
        return qwen4_exp_qsa_block_topk(
            targets[0],
            targets[1],
            table,
            requests,
            complete,
            page_size=page_size,
            block_topk=512,
            queries_per_request=width,
            max_partial_bytes=32 * 1024 * 1024,
            algorithm="logits",
            persistent_topk_workspace=workspace,
            enable_pdl=enable_pdl,
        )

    forward(enabled)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        actual = forward(enabled)
    _check_graph_edges(graph, enabled)
    for frontiers in ((0, 511, blocks), (1, 8193, 16384), (512, 513, blocks)):
        complete.copy_(
            torch.tensor(frontiers, device="cuda", dtype=torch.int32).repeat_interleave(
                width
            )
        )
        for source in sources:
            source.normal_()
        expected = forward(False).sort(dim=1).values.clone()
        for target in targets:
            target.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(actual.sort(dim=1).values, expected, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "length_dtype, length_stride",
    [(torch.int32, 1), (torch.int64, 1), (torch.int32, 2)],
)
@pytest.mark.parametrize("q_len_per_req", [1, 4])
@pytest.mark.parametrize(
    "rows, columns",
    [
        (4, 4096),
        (36, 16384),
        (36, 65536),
        (4, 65536),
        (4, 131072),
        (4, 262144),
    ],
)
def test_persistent_topk_waits_for_producers(
    enabled, length_dtype, length_stride, q_len_per_req, rows, columns, restore_pdl
):
    if not has_ragged_decode_topk():
        pytest.skip("CUDA persistent top-k is unavailable")
    logits = torch.randn(rows, columns, device="cuda")
    length_values = torch.tensor(
        [0, 1, 511, 512, 513, 8192, 8193, columns], device="cuda", dtype=length_dtype
    ).clamp_max(columns)
    length_storage = torch.zeros(
        rows // q_len_per_req * length_stride, device="cuda", dtype=length_dtype
    )
    lengths = length_storage[::length_stride]
    lengths.copy_(
        length_values[
            torch.arange(rows // q_len_per_req, device="cuda") % length_values.numel()
        ]
    )
    lengths[-1] = columns
    target_logits = torch.empty_like(logits)
    target_length_storage = torch.empty_like(length_storage)
    target_lengths = target_length_storage[::length_stride]
    output = torch.empty(rows, 512, device="cuda", dtype=torch.int32)
    workspace = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)

    def forward(enable_pdl):
        _publish((logits, length_storage), (target_logits, target_length_storage))
        ragged_decode_topk(
            target_logits,
            output,
            512,
            lengths=target_lengths,
            workspace=workspace,
            max_seq_len=columns,
            q_len_per_req=q_len_per_req,
            enable_pdl=enable_pdl,
        )
        return output

    forward(enabled)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        actual = forward(enabled)
    # Length conversions add ordinary edges; the identity case checks PDL edges.
    if length_dtype == torch.int32 and length_stride == 1:
        _check_graph_edges(graph, enabled)
    for step in range(3):
        logits.normal_()
        lengths.copy_(
            length_values[
                (torch.arange(lengths.numel(), device="cuda") + step)
                % length_values.numel()
            ]
        )
        lengths[-1] = columns
        expected = forward(False).sort(dim=1).values.clone()
        target_logits.fill_(float("nan"))
        target_lengths.zero_()
        graph.replay()
        torch.testing.assert_close(actual.sort(dim=1).values, expected, rtol=0, atol=0)


def _check_native_topk_edge(graph, enabled, rows, columns):
    from cuda.bindings import driver as cu

    # Identify the shared CUDA selector through a real reference launch,
    # independently of its compiled symbol name or the model's other kernels.
    logits = torch.zeros(rows, columns, device="cuda")
    lengths = torch.full((rows,), columns, device="cuda", dtype=torch.int32)
    output = torch.empty(rows, 512, device="cuda", dtype=torch.int32)
    workspace = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)
    reference = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(reference):
        ragged_decode_topk(
            logits,
            output,
            512,
            lengths=lengths,
            workspace=workspace,
            max_seq_len=columns,
            q_len_per_req=1,
            enable_pdl=enabled,
        )
    error, nodes, count = cu.cuGraphGetNodes(cu.CUgraph(reference.raw_cuda_graph()), 1)
    assert error == cu.CUresult.CUDA_SUCCESS
    assert count == 1
    error, reference_params = cu.cuGraphKernelNodeGetParams(nodes[0])
    assert error == cu.CUresult.CUDA_SUCCESS
    expected_function = int(reference_params.func)

    raw = cu.CUgraph(graph.raw_cuda_graph())
    error, _, _, _, count = cu.cuGraphGetEdges(raw, 0)
    assert error == cu.CUresult.CUDA_SUCCESS
    error, _, targets, edges, _ = cu.cuGraphGetEdges(raw, count)
    assert error == cu.CUresult.CUDA_SUCCESS
    checked = 0
    for target, edge in zip(targets, edges):
        error, kind = cu.cuGraphNodeGetType(target)
        assert error == cu.CUresult.CUDA_SUCCESS
        if kind != cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        error, params = cu.cuGraphKernelNodeGetParams(target)
        assert error == cu.CUresult.CUDA_SUCCESS
        if int(params.func) == expected_function:
            assert int(edge.type) == int(enabled)
            checked += 1
    assert checked == 1


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rows, columns", [(4, 8192), (36, 16384), (4, 65536)])
def test_dsa_persistent_decode_preserves_global_pdl(
    enabled, rows, columns, restore_pdl, monkeypatch
):
    for package in ("deep_gemm", "deep_ep", "trtllm_kernel"):
        pytest.importorskip(package)
    from tokenspeed_kernel.ops.attention.dsa import deep_gemm as dsa

    if not has_ragged_decode_topk():
        pytest.skip("CUDA persistent top-k is unavailable")
    # Exercise the supported TS_DSA_DECODE_TOPK_CUTEDSL=0 configuration.
    monkeypatch.setattr(dsa, "_CUTE_DSL_DECODE_TOPK_ENABLED", False)
    torch.manual_seed(397)
    page, heads, dim, topk = 64, 32, 128, 512
    query = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(rows, heads, device="cuda")
    lengths = torch.full((rows,), columns, device="cuda", dtype=torch.int32)
    table = torch.arange(rows * columns // page, device="cuda", dtype=torch.int32)
    table = table.reshape(rows, columns // page)
    # Zero keys make all valid scores equal; selected global slots must still
    # belong to the request and be distinct, under either PDL setting.
    cache = torch.zeros(rows * columns, dim + 4, device="cuda", dtype=torch.uint8)
    lengths_2d = lengths[:, None].contiguous()
    plan = dsa.deep_gemm_dsa_plan(page_size=page, seq_lens_2d=lengths_2d)
    output = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    counts = torch.empty(rows, device="cuda", dtype=torch.int32)

    def forward():
        return dsa.deep_gemm_dsa_decode_topk(
            query,
            weights,
            lengths,
            table,
            page_size=page,
            topk=topk,
            softmax_scale=1.0,
            q_len_per_req=1,
            index_k_cache=cache,
            seq_lens_2d=lengths_2d,
            plan=plan,
            out=output,
            lens_out=counts,
            initial_tokens=0,
            local_tokens=0,
            batch_invariant=False,
        )

    pdl_enabled(enabled)
    forward()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        actual, actual_counts = forward()
    _check_native_topk_edge(graph, enabled, rows, columns)
    for _ in range(3):
        query.normal_()
        pdl_enabled(not enabled)
        graph.replay()
        start = torch.arange(rows, device="cuda")[:, None] * columns
        assert ((actual >= start) & (actual < start + columns)).all()
        selected = actual.sort(dim=1).values
        assert (selected[:, 1:] > selected[:, :-1]).all()
        torch.testing.assert_close(actual_counts, torch.full_like(actual_counts, topk))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("large_grid", [False, True])
@pytest.mark.parametrize("columns", [8192, 16384, 65536])
def test_persistent_topk_pdl_register_budget(enabled, large_grid, columns, restore_pdl):
    from cuda.bindings import driver as cu

    if not has_ragged_decode_topk():
        pytest.skip("CUDA persistent top-k is unavailable")
    sms = current_platform().sm_count
    rows = sms + 4 if large_grid else max(36, sms - 4)
    logits = torch.randn(rows, columns, device="cuda")
    target = torch.empty_like(logits)
    lengths = torch.full((rows,), min(16384, columns), device="cuda", dtype=torch.int32)
    output = torch.empty(rows, 512, device="cuda", dtype=torch.int32)
    workspace = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)

    def forward():
        _publish((logits,), (target,))
        ragged_decode_topk(
            target,
            output,
            512,
            lengths=lengths,
            workspace=workspace,
            max_seq_len=columns,
            enable_pdl=enabled,
            q_len_per_req=1,
        )

    forward()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        forward()
    _check_graph_edges(graph, enabled)
    error, _, _, _, count = cu.cuGraphGetEdges(cu.CUgraph(graph.raw_cuda_graph()), 0)
    assert error == cu.CUresult.CUDA_SUCCESS
    error, _, nodes, _, _ = cu.cuGraphGetEdges(
        cu.CUgraph(graph.raw_cuda_graph()), count
    )
    assert error == cu.CUresult.CUDA_SUCCESS
    assert len(nodes) == 1
    error, params = cu.cuGraphKernelNodeGetParams(nodes[0])
    assert error == cu.CUresult.CUDA_SUCCESS
    expected_budget = 1 if enabled and not large_grid else 2
    error, occupancy = cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        params.func, 1024, 0
    )
    assert error == cu.CUresult.CUDA_SUCCESS
    # Register allocation varies by architecture/toolchain; guard the measured
    # Blackwell configuration used to select this policy.
    if torch.cuda.get_device_capability() == (10, 3):
        assert occupancy == expected_budget
