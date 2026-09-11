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
from tokenspeed_kernel.ops.attention import qsa_sparse_attention
from tokenspeed_kernel.ops.attention.triton.qwen4_exp_qsa import (
    qwen4_exp_qsa_block_topk,
    qwen4_exp_qsa_compress_and_store,
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
    pdl_enabled(enabled)
    forward()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        actual = forward()
    _check_graph_edges(graph, enabled)
    # A captured graph must retain its PDL setting after the global toggle.
    for _ in range(3):
        for source in sources:
            if source.dtype in (torch.bfloat16, torch.float32):
                source.normal_()
        pdl_enabled(False)
        expected = tuple(t.clone() for t in forward())
        for target in targets:
            target.fill_(float("nan") if target.is_floating_point() else 0)
        pdl_enabled(not enabled)
        graph.replay()
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    ("solution", "blocks", "topk"),
    [
        ("stream", 1024, 64),
        ("stream", 16384, 512),
        ("logits", 1024, 64),
    ],
)
def test_qsa_selection_waits_for_query_cache_and_metadata(
    enabled, solution, blocks, topk, restore_pdl
):
    torch.manual_seed(311)
    rows, heads, dim, page_size, ratio = 3, 4, 32, 64, 4
    pages = blocks // page_size
    query = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(blocks + page_size, 1, dim, device="cuda", dtype=query.dtype)
    table = torch.arange(1, pages + 1, device="cuda", dtype=torch.int32).repeat(rows, 1)
    requests = torch.arange(rows, device="cuda", dtype=torch.int64)
    complete = torch.tensor([0, 35, blocks], device="cuda", dtype=torch.int32)
    sources = (query, cache, table, requests, complete)
    targets = tuple(torch.empty_like(t) for t in sources)
    logical = complete.long() * ratio + 1
    full_table = torch.arange(
        1, pages * ratio + 2, device="cuda", dtype=torch.int32
    ).repeat(rows, 1)

    def forward():
        _publish(sources, targets)
        selected = qwen4_exp_qsa_block_topk(
            *targets,
            page_size=page_size,
            block_topk=topk,
            page_expansion=1,
            max_partial_bytes=32 * 1024 * 1024,
            solution=solution,
            persistent_topk_workspace=None,
            enable_pdl=pdl_enabled(),
        )
        slots = qwen4_exp_qsa_selected_slots(
            selected,
            targets[4],
            logical,
            targets[3],
            full_table,
            page_size,
            ratio,
            topk * ratio,
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

    def forward():
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
            enable_pdl=pdl_enabled(),
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
            enable_pdl=pdl_enabled(),
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
    def forward():
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
