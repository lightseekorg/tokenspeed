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

import ctypes

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsa.cuda import (
    has_ragged_decode_topk,
    ragged_decode_topk,
)
from tokenspeed_kernel.ops.attention.qsa import (
    qsa_block_topk_select,
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
    # Keep loop bounds and offsets in 64 bits: byte counts can exceed 2 GiB.
    first = tl.program_id(0).to(tl.int64) * BLOCK
    step = tl.num_programs(0).to(tl.int64) * BLOCK
    for tensor in tl.static_range(len(sizes)):
        limit = tl.full((), sizes[tensor], tl.int64)
        for base in range(first, limit, step):
            offsets = base + tl.arange(0, BLOCK)
            values = tl.load(sources[tensor] + offsets, offsets < limit, other=0)
            tl.store(targets[tensor] + offsets, values, offsets < limit)


def _publish(sources, targets):
    _publish_inputs[(8,)](
        tuple(t.view(torch.uint8) for t in sources),
        tuple(t.view(torch.uint8) for t in targets),
        tuple(t.numel() * t.element_size() for t in sources),
        BLOCK=256,
        launch_pdl=True,
    )


def test_qsa_publish_inputs_large_byte_offsets():
    """The delayed producer must copy beyond the signed int32 byte boundary."""
    size = (1 << 31) + 256
    if torch.cuda.mem_get_info()[0] < 2 * size + (1 << 30):
        pytest.skip("large producer regression needs two buffers above 2 GiB")
    sources = (
        torch.arange(16, device="cuda", dtype=torch.uint8),
        torch.full((size,), 91, device="cuda", dtype=torch.uint8),
    )
    targets = tuple(torch.zeros_like(source) for source in sources)
    _publish(sources, targets)
    assert all(torch.equal(source, target) for source, target in zip(sources, targets))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _publish(sources, targets)
    sources[1].fill_(37)
    targets[1].zero_()
    graph.replay()
    assert all(torch.equal(source, target) for source, target in zip(sources, targets))


@pytest.fixture
def restore_pdl():
    previous = pdl_enabled()
    yield
    torch.cuda.synchronize()
    pdl_enabled(previous)


def _graph_edges(graph):
    """Read edge data into owned buffers before inspecting graph nodes."""
    from cuda.bindings import driver as cu

    class EdgeData(ctypes.Structure):
        _fields_ = [
            ("from_port", ctypes.c_ubyte),
            ("to_port", ctypes.c_ubyte),
            ("type", ctypes.c_ubyte),
            ("reserved", ctypes.c_ubyte * 5),
        ]

    # cuda-python's returned edge structs can reference released array storage.
    # Keep the CUDA driver's output arrays alive until their values are copied.
    driver = ctypes.CDLL("libcuda.so.1")
    get_edges = driver.cuGraphGetEdges_v2
    get_edges.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(EdgeData),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    get_edges.restype = ctypes.c_int
    raw = ctypes.c_void_p(graph.raw_cuda_graph())
    count = ctypes.c_size_t()
    assert get_edges(raw, None, None, None, ctypes.byref(count)) == 0
    sources = (ctypes.c_void_p * count.value)()
    targets = (ctypes.c_void_p * count.value)()
    edges = (EdgeData * count.value)()
    assert get_edges(raw, sources, targets, edges, ctypes.byref(count)) == 0
    return (
        [cu.CUgraphNode(node) for node in sources[: count.value]],
        [cu.CUgraphNode(node) for node in targets[: count.value]],
        [int(edge.type) for edge in edges[: count.value]],
    )


def _check_graph_edges(graph, enabled):
    from cuda.bindings import driver as cu

    def checked(result):
        assert result[0] == cu.CUresult.CUDA_SUCCESS, result
        return result[1:]

    _, targets, edge_types = _graph_edges(graph)
    tested = 0
    for target, edge_type in zip(targets, edge_types):
        (kind,) = checked(cu.cuGraphNodeGetType(target))
        if kind == cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            (params,) = checked(cu.cuGraphKernelNodeGetParams(target))
            (name,) = checked(cu.cuFuncGetName(params.func))
            assert edge_type == int(enabled), (name, edge_type, enabled)
            tested += 1
    assert tested > 0


def _check_replays(
    forward, sources, targets, enabled, *, unordered_outputs, compare_outputs=None
):
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
        if compare_outputs is None:
            for result, reference in zip(actual, expected):
                if unordered_outputs:
                    result = result.sort(dim=-1).values
                    reference = reference.sort(dim=-1).values
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
        else:
            compare_outputs(actual, expected)


@pytest.mark.skipif(
    not current_platform().is_blackwell, reason="CuTe QSA requires Blackwell"
)
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("pages", [9, 64, 129])
def test_cute_grouped_topk_waits_for_query_and_ownership(
    enabled, uniform, pages, restore_pdl
):
    """Grouping and scoring must wait for an early-releasing input producer."""
    torch.manual_seed(337)
    lengths = [65, 65] if uniform else [64, 66]
    rows = sum(lengths)
    capacity = pages * 64
    query = torch.randn(rows, 4, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn((pages + 1) * 64, 1, 128, device="cuda", dtype=torch.bfloat16)
    table = torch.arange(1, pages + 1, device="cuda", dtype=torch.int32).repeat(2, 1)
    requests = torch.arange(2, device="cuda", dtype=torch.int64).repeat_interleave(
        torch.tensor(lengths, device="cuda")
    )
    complete = torch.linspace(0, capacity, rows, device="cuda").to(torch.int32)
    sources = query, cache, table, requests, complete
    targets = tuple(torch.empty_like(source) for source in sources)

    def forward(enable_pdl):
        _publish(sources, targets)
        selected = qwen4_exp_qsa_block_topk(
            *targets,
            page_size=64,
            block_topk=512,
            queries_per_request=lengths[0] if uniform else None,
            max_partial_bytes=0,
            max_logits_bytes=256 << 20,
            path="stream",
            enable_pdl=enable_pdl,
            solution="cute_dsl",
            override=None,
        )
        return (selected,)

    _check_replays(forward, sources, targets, enabled, unordered_outputs=True)


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
            path=algorithm,
            max_logits_bytes=256 << 20,
            solution=None,
            override=None,
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

    _check_replays(forward, sources, targets, enabled, unordered_outputs=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="CuTe DSL QSA stream top-k requires NVIDIA Blackwell",
)
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    ("rows", "width", "pages"),
    [
        (16, 8, 65),
        (1, 1, 128),
        (2, 1, 1024),
        (8, 8, 1024),
        (256, 256, 32),
        (256, 256, 64),
        (256, 256, 128),
        (9, 9, 1024),
        (153, 17, 32),
        (171, 19, 64),
        (207, 23, 128),
    ],
)
def test_cute_dsl_qsa_stream_waits_before_query_copy_and_frontiers(
    enabled,
    restore_pdl,
    rows,
    width,
    pages,
):
    torch.manual_seed(317)
    query = torch.randn(rows, 4, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn((pages + 1) * 64, 1, 128, device="cuda", dtype=torch.bfloat16)
    table = torch.arange(1, pages + 1, device="cuda", dtype=torch.int32).repeat(
        rows // width, 1
    )
    requests = torch.arange(
        rows // width, device="cuda", dtype=torch.int32
    ).repeat_interleave(width)
    capacity = pages * 64
    counts = [0, 1, 511, 512, 513, capacity // 2, capacity - 1, capacity]
    if rows < 8:
        counts = [capacity, 513][:rows]
    else:
        counts = (counts * ((rows + 7) // 8))[:rows]
    complete = torch.tensor(
        counts,
        device="cuda",
        dtype=torch.int32,
    )
    sources = query, cache, table, requests, complete
    targets = tuple(torch.empty_like(t) for t in sources)

    def forward(enable_pdl):
        _publish(sources, targets)
        selected = qwen4_exp_qsa_block_topk(
            *targets,
            page_size=64,
            block_topk=512,
            queries_per_request=width,
            max_partial_bytes=32 * 1024 * 1024,
            path="stream",
            max_logits_bytes=256 << 20,
            solution=None,
            override=None,
            enable_pdl=enable_pdl,
        )
        return (selected,)

    def compare_selected(actual, expected):
        query, cache, table, requests, complete = targets

        def selected_scores(selected):
            counts = complete.clamp(0, 512).long()
            torch.testing.assert_close((selected >= 0).sum(1), counts)
            assert bool(((selected < complete[:, None]) | (selected == -1)).all())
            ordered = selected.sort(dim=1).values
            assert bool(
                ((ordered[:, 1:] != ordered[:, :-1]) | (ordered[:, 1:] == -1)).all()
            )
            ids = selected.clamp_min(0).long()
            slots = table[requests.long()[:, None], ids // 64].long() * 64 + ids % 64
            scores = torch.einsum(
                "rhd,rnd->rhn", query.float(), cache[slots, 0].float()
            )
            scores = (
                scores.clamp_min_(0).sum(1).masked_fill_(selected < 0, -float("inf"))
            )
            return scores.sort(dim=1).values

        # Atomic candidate compaction leaves output order and boundary tie
        # ids unspecified. Compare FP32 selected scores, counts and uniqueness.
        torch.testing.assert_close(
            selected_scores(actual[0]),
            selected_scores(expected[0]),
            rtol=1e-5,
            atol=1e-4,
        )

    _check_replays(
        forward,
        sources,
        targets,
        enabled,
        unordered_outputs=False,
        compare_outputs=compare_selected,
    )


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

    _check_replays(forward, sources, targets, enabled, unordered_outputs=False)


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
            sparse_metadata=None,
        )
        return (output,)

    _check_replays(forward, sources, targets, enabled, unordered_outputs=False)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("topk", [64, 512])
def test_qsa_selection_refreshes_block_frontiers(enabled, topk, restore_pdl):
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

    def forward(enable_pdl):
        _publish(sources, targets)
        return qwen4_exp_qsa_block_topk(
            targets[0],
            targets[1],
            table,
            requests,
            complete,
            page_size=page_size,
            block_topk=topk,
            queries_per_request=width,
            max_partial_bytes=32 * 1024 * 1024,
            path="logits",
            max_logits_bytes=256 << 20,
            solution=None,
            override=None,
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

    def forward(enable_pdl):
        _publish((logits, length_storage), (target_logits, target_length_storage))
        ragged_decode_topk(
            target_logits,
            output,
            512,
            lengths=target_lengths,
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


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("column_stride", [1, 2])
@pytest.mark.parametrize("columns", [4096, 65536])
@pytest.mark.parametrize("topk", [512, 2048])
def test_qsa_cuda_selection_orders_strided_score_copies(
    enabled, column_stride, columns, topk, restore_pdl, require
):
    from cuda.bindings import driver as cu

    require("attention", "qsa_block_topk_select", "cuda", torch.float32, "logits")
    torch.manual_seed(449)
    rows = 3
    storage = torch.randn(rows, columns * column_stride, device="cuda")
    scores = storage[:, ::column_stride]
    lengths = torch.tensor([0, topk - 1, columns], device="cuda", dtype=torch.int32)
    target_storage = torch.empty_like(storage)
    target_scores = target_storage[:, ::column_stride]
    target_lengths = torch.empty_like(lengths)

    def forward():
        _publish((storage, lengths), (target_storage, target_lengths))
        return qsa_block_topk_select(
            target_scores,
            topk,
            lengths=target_lengths,
            enable_pdl=enabled,
            solution="cuda",
            override=None,
        )

    forward()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        selected = forward()
    _check_native_topk_edge(graph, enabled, rows, columns, topk)
    if column_stride != 1:
        _, _, edge_types = _graph_edges(graph)
        # The ordinary copy must wait for the producer's full completion.
        assert 0 in edge_types

    offsets = torch.arange(columns, device="cuda")
    for frontiers in ((0, topk - 1, columns), (1, topk, columns - 1), (0, 7, topk + 1)):
        storage.normal_()
        lengths.copy_(torch.tensor(frontiers, device="cuda", dtype=torch.int32))
        target_storage.fill_(float("nan"))
        target_lengths.zero_()
        graph.replay()
        live = selected >= 0
        torch.testing.assert_close(live.sum(dim=1), lengths.clamp_max(topk).long())
        assert torch.all((selected < lengths[:, None]) | ~live)
        actual = scores.gather(1, selected.clamp_min(0).long()).masked_fill(
            ~live, -float("inf")
        )
        expected = torch.topk(
            scores.masked_fill(offsets[None, :] >= lengths[:, None], -float("inf")),
            topk,
        ).values
        torch.testing.assert_close(
            actual.sort(dim=1).values, expected.sort(dim=1).values, rtol=0, atol=0
        )


def _check_native_topk_edge(graph, enabled, rows, columns, topk):
    from cuda.bindings import driver as cu

    # Identify the shared CUDA selector through a real reference launch,
    # independently of its compiled symbol name or the model's other kernels.
    logits = torch.zeros(rows, columns, device="cuda")
    lengths = torch.full((rows,), columns, device="cuda", dtype=torch.int32)
    output = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    reference = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(reference):
        ragged_decode_topk(
            logits,
            output,
            topk,
            lengths=lengths,
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

    _, targets, edge_types = _graph_edges(graph)
    checked = 0
    for target, edge_type in zip(targets, edge_types):
        error, kind = cu.cuGraphNodeGetType(target)
        assert error == cu.CUresult.CUDA_SUCCESS
        if kind != cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        error, params = cu.cuGraphKernelNodeGetParams(target)
        assert error == cu.CUresult.CUDA_SUCCESS
        if int(params.func) == expected_function:
            assert edge_type == int(enabled)
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
    _check_native_topk_edge(graph, enabled, rows, columns, topk)
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

    def forward():
        _publish((logits,), (target,))
        ragged_decode_topk(
            target,
            output,
            512,
            lengths=lengths,
            max_seq_len=columns,
            enable_pdl=enabled,
            q_len_per_req=1,
        )

    forward()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        forward()
    _check_graph_edges(graph, enabled)
    _, nodes, _ = _graph_edges(graph)
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
