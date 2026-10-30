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

"""Exact source edits and numerical isolation for the private SM90 MoE adapter."""

import ctypes
import difflib
import json
import os
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.ops.moe import prepare_sm90_fp8_moe
from tokenspeed_kernel.thirdparty.flashinfer.sm90_moe import (
    _apply_patch,
    _patched_files,
)


def test_private_patch_matches_installed_sources():
    flashinfer = pytest.importorskip("flashinfer")
    from flashinfer.jit import env as jit_env

    _patched_files(Path(flashinfer.__file__).parent, jit_env.FLASHINFER_CSRC_DIR)


@pytest.mark.parametrize("context", (0, 3))
def test_patch_preserves_unedited_lines_and_applies_multiple_hunks(context):
    original = "".join(f"line {i}\n" for i in range(30))
    edited = original.replace("line 2\n", "replacement\nextra\n").replace(
        "line 25\n", ""
    )
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            edited.splitlines(keepends=True),
            fromfile="a/input",
            tofile="b/input",
            n=context,
        )
    )
    assert _apply_patch(original, diff) == edited
    with pytest.raises(RuntimeError, match="context differs"):
        _apply_patch(original.replace("line 2\n", "upstream change\n"), diff)


@pytest.mark.parametrize(
    "original,edited",
    [("one\n", "new\none\n"), ("one\n", "one\nnew\n"), ("one\ntwo\n", "two\n")],
)
def test_patch_handles_insertions_and_deletions_at_file_boundaries(original, edited):
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            edited.splitlines(keepends=True),
            fromfile="a/input",
            tofile="b/input",
            n=0,
        )
    )
    assert _apply_patch(original, diff) == edited


@pytest.mark.parametrize("mode", ("auto", "local_mean", "", None, True))
def test_unknown_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="Unknown SM90 MoE mode"):
        prepare_sm90_fp8_moe(mode=mode)


def _require_private_sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Requires Hopper SM90")
    flashinfer = pytest.importorskip("flashinfer")
    if flashinfer.__version__ != "0.7.2.post1":
        pytest.skip("The private source patch targets FlashInfer 0.7.2.post1")


@pytest.fixture(scope="module")
def implementations():
    _require_private_sm90()
    # Build before warmup/capture, with independent stage configurations coexisting.
    return {
        "default": prepare_sm90_fp8_moe(mode="default"),
        "expected_m": prepare_sm90_fp8_moe(mode="expected_m"),
        "optimized": prepare_sm90_fp8_moe(mode="optimized"),
        "fc1_4": prepare_sm90_fp8_moe(mode="optimized", max_num_stages_fc1=4),
        "fc2_4": prepare_sm90_fp8_moe(mode="optimized", max_num_stages_fc2=4),
        "both_4": prepare_sm90_fp8_moe(
            mode="optimized", max_num_stages_fc1=4, max_num_stages_fc2=4
        ),
    }


# Synthetic dimensions for the integration tests, independent of model configuration.
EXPERTS = 288
TOP_K = 16
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 2048


@pytest.fixture(scope="module")
def expert_weights(implementations):
    return _make_expert_weights(EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)


def _make_expert_weights(experts, hidden_size, intermediate_size):
    generator = torch.Generator(device="cuda").manual_seed(1701)
    weights = []
    scales = []
    for rows, columns in (
        (2 * intermediate_size, hidden_size),
        (hidden_size, intermediate_size),
    ):
        weights.append(
            (
                torch.randn(
                    (experts, rows, columns),
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                * 8
            ).to(torch.float8_e4m3fn)
        )
        scales.append(
            torch.exp2(
                torch.randint(
                    -8,
                    -3,
                    (experts, rows // 128, columns // 128),
                    device="cuda",
                    generator=generator,
                ).float()
            )
        )
    return *weights, *scales


def _arguments(weights, tokens, experts, hidden_size, top_k, concentrated):
    w1, w2, s1, s2 = weights
    ids = torch.arange(tokens * top_k, device="cuda", dtype=torch.int32).view(
        tokens, top_k
    )
    ids.remainder_(top_k if concentrated else experts)
    return dict(
        input=torch.randn((tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
        * 0.1,
        token_selected_experts=ids,
        token_final_scales=torch.softmax(
            torch.randn((tokens, top_k), device="cuda"), dim=-1
        ),
        fc1_expert_weights=w1,
        fc2_expert_weights=w2,
        output_dtype=torch.bfloat16,
        quant_scales=[s1, s2],
        use_deepseek_fp8_block_scale=True,
        enable_pdl=False,
    )


def _capture_calls(calls):
    graphs = []
    for call in calls:
        for _ in range(3):
            call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call()[0]
        graphs.append((graph, output))
    return graphs


@pytest.mark.parametrize("tokens", (32, 128, 512, 1024))
@pytest.mark.parametrize("concentrated", (False, True))
@pytest.mark.parametrize("ep_size,ep_rank", ((1, 0), (8, 7)))
def test_private_modes_match_original_after_route_changes(
    implementations, expert_weights, tokens, concentrated, ep_size, ep_rank
):
    arguments = _arguments(
        expert_weights, tokens, EXPERTS, HIDDEN_SIZE, TOP_K, concentrated
    )
    arguments.update(ep_size=ep_size, ep_rank=ep_rank)
    ids = arguments["token_selected_experts"]
    ids.add_(ep_rank * EXPERTS)
    hint = max(1, (tokens * TOP_K + EXPERTS - 1) // EXPERTS)
    calls = [
        partial(
            call, **arguments, **({} if mode == "default" else {"expected_m": hint})
        )
        for mode, call in implementations.items()
    ]
    graphs = _capture_calls(calls)
    for shift in (0, EXPERTS // 2, EXPERTS - TOP_K):
        local = torch.arange(tokens * TOP_K, device="cuda").view(tokens, TOP_K)
        local = (
            (local % TOP_K + shift) % EXPERTS
            if concentrated
            else (local + shift) % EXPERTS
        )
        ids.copy_(local + ep_rank * EXPERTS)
        arguments["input"].copy_(torch.randn_like(arguments["input"]) * 0.1)
        arguments["token_final_scales"].copy_(
            torch.softmax(torch.randn_like(arguments["token_final_scales"]), dim=-1)
        )
        reference = calls[0]()[0].clone()
        for call, (graph, output) in reversed(list(zip(calls, graphs))):
            torch.testing.assert_close(call()[0], reference, atol=0, rtol=0)
            graph.replay()
            torch.testing.assert_close(output, reference, atol=0, rtol=0)
        for mode, call in implementations.items():
            # An explicit hint or a different module must not alter the default estimate.
            torch.testing.assert_close(call(**arguments)[0], reference, atol=0, rtol=0)


def test_direct_lookup_overflow_matches_original(implementations, expert_weights):
    # More than two maximum-size token tiles per expert exceeds the lookup capacity.
    tokens = 2 * EXPERTS * 128 // TOP_K + EXPERTS
    arguments = _arguments(expert_weights, tokens, EXPERTS, HIDDEN_SIZE, TOP_K, False)
    reference = implementations["default"](**arguments)[0].clone()
    for mode in ("optimized", "fc1_4", "fc2_4", "both_4"):
        actual = implementations[mode](**arguments, expected_m=1)[0]
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize("tokens", (32, 512))
def test_prefix_scan_with_more_experts_than_threads_matches_original(
    implementations, tokens
):
    # 512 experts take two scan chunks of the 384-thread SwapAB block.
    experts = 512
    weights = _make_expert_weights(experts, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    arguments = _arguments(weights, tokens, experts, HIDDEN_SIZE, TOP_K, False)
    reference = implementations["default"](**arguments)[0].clone()
    hint = max(1, (tokens * TOP_K + experts - 1) // experts)
    for mode in ("expected_m", "optimized", "fc1_4"):
        actual = implementations[mode](**arguments, expected_m=hint)[0]
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.fixture(scope="module")
def fusion_implementations():
    _require_private_sm90()
    # Selecting fusion tests must not build all six general-purpose modules.
    return (
        prepare_sm90_fp8_moe(mode="default"),
        prepare_sm90_fp8_moe(
            mode="optimized", max_num_stages_fc1=4, max_num_stages_fc2=-1
        ),
    )


@pytest.fixture(scope="module")
def fusion_weights(fusion_implementations):
    return _make_expert_weights(384, 512, 1024)


def _assert_bitwise(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    torch.testing.assert_close(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
        atol=0,
        rtol=0,
    )


def _check_fusion_routes(implementations, arguments, concentrated):
    reference, candidate = implementations
    tokens, top_k = arguments["token_selected_experts"].shape
    experts = arguments["fc1_expert_weights"].shape[0]
    hint = max(1, -(-tokens * top_k // experts))
    call = partial(candidate, **arguments, expected_m=hint)
    graph, output = _capture_calls([call])[0]
    for shift in (0, experts // 2, experts - top_k):
        ids = torch.arange(tokens * top_k, device="cuda").view(tokens, top_k)
        if concentrated:
            ids.remainder_(top_k)
        arguments["token_selected_experts"].copy_((ids + shift) % experts)
        arguments["input"].normal_(0, 0.1)
        if shift == experts // 2:
            arguments["input"][0].zero_()
        arguments["token_final_scales"].copy_(
            torch.softmax(torch.randn_like(arguments["token_final_scales"]), dim=-1)
        )
        expected = reference(**arguments)[0].clone()
        assert torch.isfinite(expected).all()
        if shift == experts // 2:
            assert torch.count_nonzero(expected[0]) == 0
        _assert_bitwise(call()[0], expected)
        for _ in range(2):
            graph.replay()
            _assert_bitwise(output, expected)


@pytest.mark.parametrize("tokens", (1, 8, 32, 128, 256))
@pytest.mark.parametrize("enable_pdl", (False, True))
@pytest.mark.parametrize("concentrated", (False, True))
def test_fused_gated_block_fp8_matches_upstream(
    fusion_implementations, fusion_weights, tokens, enable_pdl, concentrated
):
    """Fused FC1/quantized FC2 preserve bytes as graph inputs and routes change."""
    arguments = _arguments(fusion_weights, tokens, 384, 512, 12, concentrated)
    arguments["enable_pdl"] = enable_pdl
    _check_fusion_routes(fusion_implementations, arguments, concentrated)


@pytest.mark.parametrize("tokens,top_k", ((1, 1), (1, 2), (1, 3), (2, 1)))
@pytest.mark.parametrize("enable_pdl", (False, True))
def test_fused_gated_block_fp8_unaligned_rows(
    fusion_implementations, fusion_weights, tokens, top_k, enable_pdl
):
    """Short FP8 inputs must zero-fill tail loads within this call's row capacity."""
    arguments = _arguments(fusion_weights, tokens, 384, 512, top_k, False)
    arguments["enable_pdl"] = enable_pdl
    _check_fusion_routes(fusion_implementations, arguments, False)


@pytest.mark.parametrize("enable_pdl", (False, True))
def test_fused_gated_block_fp8_graph_lifecycle(
    fusion_implementations, fusion_weights, enable_pdl
):
    """Captured fusion survives nonfused calls, layout changes and empty EP work."""
    reference, candidate = fusion_implementations
    arguments = _arguments(fusion_weights, 8, 384, 512, 12, False)
    arguments.update(enable_pdl=enable_pdl, ep_size=8, ep_rank=7)
    arguments["token_selected_experts"].add_(7 * 384)
    call = partial(candidate, **arguments, expected_m=1)
    graph, output = _capture_calls([call])[0]
    # M512 disables fusion; hints 31/32 straddle SwapAB/regular GEMM selection.
    for tokens, hint in ((512, 16), (8, 31), (8, 32)):
        other = _arguments(fusion_weights, tokens, 384, 512, 12, False)
        other.update(enable_pdl=enable_pdl, ep_size=8, ep_rank=7)
        other["token_selected_experts"].add_(7 * 384)
        _assert_bitwise(
            candidate(**other, expected_m=hint)[0], reference(**other)[0].clone()
        )
        graph.replay()
        _assert_bitwise(output, reference(**arguments)[0].clone())

    for routing in ("local", "mixed", "no-local"):
        ids = torch.arange(8 * 12, device="cuda").view(8, 12) % 384
        if routing == "local":
            ids.add_(7 * 384)
        elif routing == "mixed":
            ids[:, :6].add_(7 * 384)
        arguments["token_selected_experts"].copy_(ids)
        arguments["input"].normal_(0, 0.1)
        arguments["input"][0].zero_()
        arguments["token_final_scales"].copy_(
            torch.softmax(torch.randn_like(arguments["token_final_scales"]), dim=-1)
        )
        expected = reference(**arguments)[0].clone()
        assert torch.isfinite(expected).all()
        if routing == "no-local":
            assert torch.count_nonzero(expected) == 0
        _assert_bitwise(call()[0], expected)
        for _ in range(2):
            graph.replay()
            _assert_bitwise(output, expected)


@pytest.mark.parametrize("enable_pdl", (False, True))
def test_fused_gated_block_fp8_multichunk_scan(fusion_implementations, enable_pdl):
    # 512 experts span two chunks of the 384-thread grouped scan.
    weights = _make_expert_weights(512, 512, 1024)
    for tokens, concentrated in ((8, False), (256, True)):
        arguments = _arguments(weights, tokens, 512, 512, 12, concentrated)
        arguments["enable_pdl"] = enable_pdl
        _check_fusion_routes(fusion_implementations, arguments, concentrated)


@pytest.mark.parametrize("enable_pdl", (False, True))
@pytest.mark.parametrize("use_graph", (False, True))
def test_fused_gated_block_fp8_launches(
    fusion_implementations, fusion_weights, enable_pdl, use_graph
):
    """Inspect executed kernels and require FC1 completion before FC2 without PDL."""
    arguments = _arguments(fusion_weights, 8, 384, 512, 12, False)
    arguments["enable_pdl"] = enable_pdl
    reference, candidate = fusion_implementations
    calls = [
        partial(reference, **arguments),
        partial(candidate, **arguments, expected_m=1),
    ]
    graphs = _capture_calls(calls)
    traces = [
        _kernel_events(graph.replay if use_graph else call)
        for call, (graph, _) in zip(calls, graphs)
    ]
    upstream, fused = traces
    activation = "doActivationKernel"
    quantization = "fp8_blockscale_gemm::scale_1x128_kernel<"
    assert any(activation in event["name"] for event in upstream)
    assert not any(activation in event["name"] for event in fused)
    assert sum(quantization in event["name"] for event in upstream) == 2
    assert sum(quantization in event["name"] for event in fused) == 1
    grouped = sorted(
        (
            event
            for event in fused
            if "fp8_gemm_kernel" in event["name"]
            and "GroupedWithOffset" in event["name"]
        ),
        key=lambda event: event["ts"],
    )
    assert len(grouped) == 2, [event["name"] for event in fused]
    if not enable_pdl:
        fc1, fc2 = grouped
        assert fc2["ts"] >= fc1["ts"] + fc1["dur"], grouped


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


def _grouped_gemm_edges(graph):
    """(source is a grouped GEMM, edge type) for each edge into a grouped GEMM."""
    from cuda.bindings import driver as cu

    def checked(result):
        assert result[0] == cu.CUresult.CUDA_SUCCESS, result
        return result[1:]

    def is_grouped_gemm(node):
        (kind,) = checked(cu.cuGraphNodeGetType(node))
        if kind != cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            return False
        (params,) = checked(cu.cuGraphKernelNodeGetParams(node))
        # Library kernels launched as cudaKernel_t may leave func null.
        if int(params.func):
            (name,) = checked(cu.cuFuncGetName(params.func))
        else:
            (name,) = checked(cu.cuKernelGetName(params.kern))
        name = name.decode() if isinstance(name, bytes) else name
        return "fp8_gemm_kernel" in name and "GroupedWithOffset" in name

    sources, targets, types = _graph_edges(graph)
    nodes = {int(node): node for node in sources + targets}
    grouped = {key for key, node in nodes.items() if is_grouped_gemm(node)}
    assert len(grouped) == 2, grouped  # FC1 and FC2
    return sorted(
        (int(source) in grouped, edge_type)
        for source, target, edge_type in zip(sources, targets, types)
        if int(target) in grouped
    )


@pytest.mark.parametrize(
    "tokens,enable_pdl,expected",
    (
        # 512 rows exceed the fusion cap but still run swapAB grouped GEMMs.
        (512, True, [(False, 0), (False, 0)]),
        (8, True, [(False, 1), (True, 1)]),
        (8, False, [(False, 0), (True, 0)]),
    ),
)
def test_only_fused_chain_launches_as_programmatic_dependent(
    fusion_implementations, fusion_weights, tokens, enable_pdl, expected
):
    """Only the fused FC1 -> FC2 chain is written to start before its
    predecessor finishes; other grouped GEMMs keep stream order with PDL on."""
    _, candidate = fusion_implementations
    arguments = _arguments(fusion_weights, tokens, 384, 512, 12, False)
    arguments["enable_pdl"] = enable_pdl
    call = partial(candidate, **arguments, expected_m=1 if tokens <= 256 else 16)
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        call()
    # Programmatic (type 1) edges come from launches with PDL enabled.
    assert _grouped_gemm_edges(graph) == expected


def _kernel_events(call):
    """Executed CUDA kernels, including device timestamps, from one call."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        call()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.json"
        profile.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
    return [event for event in events if event.get("cat") == "kernel"]


def _activation_quantization_launches(call):
    """Kernel names and grids of the grouped activation quantization in one call."""
    return [
        (event["name"], event["args"]["grid"])
        for event in _kernel_events(call)
        if "fp8_blockscale_gemm::scale_1x128_kernel<" in event["name"]
    ]


def test_small_call_quantization_follows_its_own_rows():
    """A decode-sized call quantizes as if no larger batch had run before it.

    The block-scale runner keeps the largest M it has seen for its workspace.
    Sizing the grouped activation quantization from that high-water mark gave
    every later small call the large batch's grid and linear expert search.
    """
    _require_private_sm90()
    # A private mode with an explicit FC1 stage bound; shared with other tests.
    moe = prepare_sm90_fp8_moe(mode="optimized", max_num_stages_fc1=4)
    weights = _make_expert_weights(EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    small = _arguments(weights, 1, EXPERTS, HIDDEN_SIZE, TOP_K, False)
    large = _arguments(weights, 1024, EXPERTS, HIDDEN_SIZE, TOP_K, False)
    reference = moe(**small, expected_m=1)[0].clone()
    moe(**large, expected_m=(1024 * TOP_K + EXPERTS - 1) // EXPERTS)

    launches = _activation_quantization_launches(lambda: moe(**small, expected_m=1))
    # FC1 quantizes [rows, HIDDEN_SIZE] and FC2 [rows, INTERMEDIATE_SIZE]. One warp
    # handles one 128-wide group and a block has eight warps; more blocks would idle.
    rows = TOP_K
    assert len(launches) == 2
    for (name, grid), width in zip(launches, (HIDDEN_SIZE, INTERMEDIATE_SIZE)):
        assert "scale_1x128_kernel<true," in name
        assert grid[0] <= -(-rows * (width // 128) // 8)
    torch.testing.assert_close(moe(**small, expected_m=1)[0], reference, atol=0, rtol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = moe(**small, expected_m=1)[0]
    graph.replay()
    torch.testing.assert_close(output, reference, atol=0, rtol=0)


@pytest.mark.parametrize("hint", (0, -1, True, 1.5, 2**32))
def test_invalid_hint_is_rejected_before_launch(implementations, hint):
    with pytest.raises(ValueError, match="expected_m must"):
        implementations["optimized"](
            input=torch.empty((1, 128), dtype=torch.bfloat16),
            token_selected_experts=torch.zeros((1, 1), dtype=torch.int32),
            token_final_scales=torch.ones((1, 1)),
            fc1_expert_weights=torch.empty((1, 256, 128), dtype=torch.float8_e4m3fn),
            fc2_expert_weights=torch.empty((1, 128, 128), dtype=torch.float8_e4m3fn),
            output_dtype=torch.bfloat16,
            quant_scales=[],
            use_deepseek_fp8_block_scale=True,
            expected_m=hint,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"enable_sm90_moe_optimizations": False},
        {"max_num_stages_fc1": 4},
        {"max_num_stages_fc2": 4},
    ),
)
def test_settings_cannot_change_on_a_prepared_callable(implementations, overrides):
    with pytest.raises(TypeError, match="during initialization"):
        implementations["optimized"](**overrides)


@pytest.mark.parametrize(
    "experts,hidden_size,intermediate_size",
    (
        (17, 256, 640),
        (33, 512, 512),
        (65, 640, 896),
        (127, 896, 1280),
        (256, 1280, 640),
        (512, 1536, 256),
        (1024, 256, 640),
    ),
)
@pytest.mark.parametrize("concentrated", (False, True))
@pytest.mark.parametrize("expected_m", (1, 128))
def test_scheduler_and_stage_bounds_across_shapes(
    implementations, experts, hidden_size, intermediate_size, concentrated, expected_m
):
    """Cover partial warps, multi-chunk scans, and both grouped GEMM layouts."""
    weights = _make_expert_weights(experts, hidden_size, intermediate_size)
    arguments = _arguments(weights, 32, experts, hidden_size, TOP_K, concentrated)
    calls = [
        partial(
            call,
            **arguments,
            **({} if mode == "default" else {"expected_m": expected_m}),
        )
        for mode, call in implementations.items()
    ]
    graphs = _capture_calls(calls)
    for _ in range(2):
        arguments["input"].copy_(torch.randn_like(arguments["input"]) * 0.1)
        arguments["token_selected_experts"].add_(TOP_K).remainder_(experts)
        reference = calls[0]()[0].clone()
        for call, (graph, output) in zip(calls, graphs):
            torch.testing.assert_close(call()[0], reference, atol=0, rtol=0)
            graph.replay()
            torch.testing.assert_close(output, reference, atol=0, rtol=0)


@pytest.mark.parametrize("stages", (-2, 0, 1, 2, 3, True, 4.5, "4", 2**31))
@pytest.mark.parametrize("parameter", ("max_num_stages_fc1", "max_num_stages_fc2"))
def test_invalid_stage_bound_is_rejected_before_initialization(stages, parameter):
    with pytest.raises(ValueError, match=f"{parameter} must"):
        prepare_sm90_fp8_moe(mode="optimized", **{parameter: stages})


@pytest.mark.parametrize("parameter", ("max_num_stages_fc1", "max_num_stages_fc2"))
def test_default_mode_rejects_an_explicit_stage_bound(parameter):
    with pytest.raises(ValueError, match="requires a private"):
        prepare_sm90_fp8_moe(mode="default", **{parameter: 4})


_COLD_COMPILE = """
import sys
import torch
sys.path.insert(0, {test_dir!r})
import test_sm90_moe as t
from tokenspeed_kernel.ops.moe import prepare_sm90_fp8_moe

default = prepare_sm90_fp8_moe(mode="default")
weights = t._make_expert_weights(384, 512, 1024)
default(**t._arguments(weights, 256, 384, 512, 12, False))
if {load_private}:
    prepare_sm90_fp8_moe(mode="optimized", max_num_stages_fc1=4, max_num_stages_fc2=-1)
torch.manual_seed(0)
output = default(**t._arguments(weights, 1, 384, 512, 12, False))[0]
torch.save(output.cpu(), {output!r})
"""


def test_private_module_leaves_original_cold_compiles_unchanged(tmp_path):
    """Loading a private module must not point the installed module's DeepGEMM
    JIT at patched headers: a shape it compiles afterwards matches a process
    that never loaded one."""
    _require_private_sm90()
    outputs = []
    for load_private in (False, True):
        # A fresh process and DeepGEMM cache make the M=1 shape compile cold.
        name = f"load_private_{load_private}"
        script = _COLD_COMPILE.format(
            test_dir=str(Path(__file__).resolve().parent),
            load_private=load_private,
            output=str(tmp_path / f"{name}.pt"),
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=dict(os.environ, TRTLLM_DG_CACHE_DIR=str(tmp_path / name)),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert result.returncode == 0, result.stderr[-4000:]
        outputs.append(torch.load(tmp_path / f"{name}.pt"))
    _assert_bitwise(outputs[1], outputs[0])
