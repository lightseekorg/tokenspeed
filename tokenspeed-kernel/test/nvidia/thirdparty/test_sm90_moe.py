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

import difflib
import json
import tempfile
from functools import partial
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.ops.moe import prepare_sm90_fp8_moe
from tokenspeed_kernel.thirdparty.flashinfer.sm90_moe import _apply_patch


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


def _activation_quantization_launches(call):
    """Kernel names and grids of the grouped activation quantization in one call."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        call()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.json"
        profile.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
    return [
        (event["name"], event["args"]["grid"])
        for event in events
        if event.get("cat") == "kernel"
        and "fp8_blockscale_gemm::scale_1x128_kernel<" in event["name"]
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
