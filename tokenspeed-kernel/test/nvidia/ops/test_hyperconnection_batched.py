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

"""Multi-token CTA scheduling, output coverage and workspace reuse."""

from contextlib import nullcontext
from unittest import mock

import pytest
import torch
from tokenspeed_kernel.numerics.torch.residual import torch_hyperconnection_mix
from tokenspeed_kernel.ops.residual import cute_fused, gated_residual_mix
from tokenspeed_kernel.platform import ArchVersion, current_platform, pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def restore_state():
    if not cute_fused.supports_fused_hc(torch.device("cuda")):
        pytest.skip("requires supported Blackwell and CuTe DSL")
    pdl = pdl_enabled()
    capture = ShapeCapture.get()
    capture_enabled = capture.enabled
    yield
    torch.cuda.synchronize()
    pdl_enabled(pdl)
    capture.enabled = capture_enabled
    capture.clear()


def _reset_caches(monkeypatch):
    for name in ("_PLANS", "_CAPACITIES", "_WORKSPACES"):
        monkeypatch.setattr(cute_fused, name, {})


def inputs(rows, dtype, has_inject):
    generator = torch.Generator(device="cuda").manual_seed(1729 + rows)
    return tuple(
        torch.randn(shape, dtype=dtype, device="cuda", generator=generator) * scale
        for shape, scale in (
            ((rows, 10240), 1.0),
            ((324 if has_inject else 320, 10240), 0.01),
            ((10240, 320), 0.01),
        )
    )


def reference(values, scale):
    x, w, u = (value.cpu() for value in values)
    return torch_hyperconnection_mix(x, w, u, 4, 2560, 320, w.shape[0], scale, True)


def check(result, expected, dtype):
    tolerance = 0.04 if dtype == torch.bfloat16 else 0.008
    for actual, wanted in zip(result, expected):
        if wanted is None:
            assert actual is None
        else:
            torch.testing.assert_close(
                actual.cpu().double(), wanted, rtol=tolerance, atol=tolerance
            )


def mix(values, scale, independent):
    return gated_residual_mix(
        *values,
        4,
        2560,
        320,
        projection_rows=320 if values[1].shape[0] == 320 else 324,
        projection_scale=scale,
        weights_independent=independent,
        override="cute_fused_hyperconnection_mix",
        solution=None,
        activation_quantization="none"
    )


# The boundary graph test below covers bf16 with PDL on, including the Triton rows.
@pytest.mark.parametrize(
    ("rows", "expected_kernel"),
    [
        (17, "cute_fused_hyperconnection_mix"),
        (32, "cute_fused_hyperconnection_mix"),
        (64, "cute_fused_hyperconnection_mix"),
        (128, "cute_fused_hyperconnection_mix"),
        (255, "cute_fused_hyperconnection_mix"),
        (256, "cute_fused_hyperconnection_mix"),
        (257, "triton_hyperconnection_mix"),
        (1024, "triton_hyperconnection_mix"),
    ],
)
@pytest.mark.parametrize(
    ("dtype", "enable_pdl"),
    [(torch.bfloat16, False), (torch.float16, False), (torch.float16, True)],
)
def test_batched_default_dispatch_respects_row_limit(
    rows, expected_kernel, dtype, enable_pdl
):
    values = inputs(rows, dtype, True)
    pdl_enabled(enable_pdl)
    capture = ShapeCapture.get()
    capture.enabled = True
    capture.clear()
    result = gated_residual_mix(
        *values,
        4,
        2560,
        320,
        projection_rows=324,
        projection_scale=0.25,
        weights_independent=True,
        override=None,
        solution=None,
        activation_quantization="none"
    )
    assert capture._records[-1].kernel_name == expected_kernel
    check(result, reference(values, 0.25), dtype)


@pytest.mark.parametrize(
    ("rows", "expected_kernel"),
    [
        (16, "cute_fused_hyperconnection_mix"),
        (32, "cute_fused_hyperconnection_mix"),
        (64, "cute_fused_hyperconnection_mix"),
        (65, "cute_fused_hyperconnection_mix"),
        (256, "cute_fused_hyperconnection_mix"),
        (257, "triton_hyperconnection_mix"),
        (1024, "triton_hyperconnection_mix"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_default_dispatch_boundary_graph_replays(rows, expected_kernel, dtype):
    values = inputs(rows, dtype, True)
    pdl_enabled(True)
    capture = ShapeCapture.get()
    capture.enabled = True
    capture.clear()

    def run():
        return gated_residual_mix(
            *values,
            4,
            2560,
            320,
            projection_rows=324,
            projection_scale=0.25,
            weights_independent=True,
            override=None,
            solution=None,
            activation_quantization="none"
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [run() for _ in range(3)]
    assert all(record.kernel_name == expected_kernel for record in capture._records)
    for factor in (0.75, -0.5):
        values[0].mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        expected = reference(values, 0.25)
        for output in outputs:
            check(output, expected, dtype)


@pytest.mark.parametrize(
    "rows",
    [
        17,
        24,
        31,
        32,
        33,
        48,
        63,
        64,
        65,
        96,
        97,
        127,
        128,
        129,
        160,
        192,
        193,
        207,
        223,
        224,
        225,
        255,
        256,
        257,
        384,
        385,
        511,
        512,
        513,
        1023,
        1024,
    ],
)
def test_batched_fp64_and_changed_input_graph(rows):
    dtype = torch.bfloat16
    values = inputs(rows, dtype, True)
    pdl_enabled(True)
    result = mix(values, 0.25, True)
    check(result, reference(values, 0.25), dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mix(values, 0.25, True)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [mix(values, 0.25, True) for _ in range(3)]
    for factor in (0.75, -0.5, 1.25):
        values[0].mul_(factor)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        expected = reference(values, 0.25)
        for output in outputs:
            check(output, expected, dtype)
        for output in outputs[1:]:
            for actual, first in zip(output, outputs[0]):
                assert torch.equal(actual, first)


@pytest.mark.parametrize("rows", [17, 32, 33, 64, 97, 193, 224, 225, 256, 385, 1024])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("has_inject", [False, True])
def test_fp16_optional_inject_and_pdl(rows, enable_pdl, has_inject):
    values = inputs(rows, torch.float16, has_inject)
    pdl_enabled(enable_pdl)
    check(mix(values, 1.0, False), reference(values, 1.0), torch.float16)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("has_inject", [False, True])
def test_up_activation_reuse_across_token_tiles(
    dtype, enable_pdl, has_inject, monkeypatch
):
    _reset_caches(monkeypatch)
    # Limit residency to one worker so its activation slot is overwritten for
    # five token tiles, including a tail, within each launch.
    monkeypatch.setattr(
        cute_fused,
        "_capacity",
        lambda compiled, kernel, device, stream: kernel.clusters,
    )
    values = inputs(129, dtype, has_inject)
    pdl_enabled(enable_pdl)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mix(values, 0.25, True)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [mix(values, 0.25, True) for _ in range(2)]
    atol, rtol = (0.002, 0.008) if dtype == torch.bfloat16 else (0.0005, 0.002)
    for factor in (0.75, -0.5):
        values[0].mul_(factor)
        for output in outputs:
            for tensor in output:
                if tensor is not None:
                    tensor.fill_(float("nan"))
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        expected = reference(values, 0.25)
        for output in outputs:
            for tensor, wanted in zip(output, expected):
                if wanted is None:
                    assert tensor is None
                else:
                    torch.testing.assert_close(
                        tensor.cpu().double(), wanted, atol=atol, rtol=rtol
                    )


@pytest.mark.parametrize(
    ("rows", "tile"),
    [
        (65, (1, 64, 16, 1, 2, 5, 32)),
        (65, (2, 128, 16, 1, 2, 2, 16)),
        (65, (4, 64, 16, 6, 2, 5, 32)),
        (256, (16, 64, 16, 6, 2, 5, 32)),
    ],
)
def test_projection_and_batch_loops(rows, tile, monkeypatch):
    monkeypatch.setattr(cute_fused, "_tactic", lambda rows, projection_rows: tile)
    _reset_caches(monkeypatch)
    values = inputs(rows, torch.bfloat16, True)
    pdl_enabled(True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mix(values, 1.0, True)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = mix(values, 1.0, True)
    for factor in (0.75, -0.5, 1.25):
        values[0].mul_(factor)
        for _ in range(4):
            graph.replay()
        torch.cuda.synchronize()
        check(result, reference(values, 1.0), torch.bfloat16)
    kernel = list(cute_fused._PLANS.values())[-1][0]
    assert kernel.batch_tiles == 2
    if tile[3] == 6:
        assert kernel.clusters == 1
        if rows == 256:
            # Exercise ordinary multi-wave cluster scheduling when the target
            # has fewer resident clusters than logical jobs.
            workspaces = list(cute_fused._WORKSPACES.values())
            assert any(
                epochs.numel() == 2 * ((rows + 31) // 32) for _, _, epochs in workspaces
            )


def test_uneven_down_stages_recycle_across_four_tiles(monkeypatch):
    monkeypatch.setattr(
        cute_fused,
        "_tactic",
        lambda rows, projection_rows: (16, 64, 8, 1, 2, 2, 32),
    )
    monkeypatch.setattr(
        cute_fused,
        "_capacity",
        lambda compiled, kernel, device, stream: kernel.clusters,
    )
    _reset_caches(monkeypatch)
    values = inputs(32, torch.bfloat16, True)
    pdl_enabled(True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mix(values, 1.0, True)
    stream.synchronize()
    kernel = list(cute_fused._PLANS.values())[-1][0]
    assert (kernel.k_tiles, kernel.down_stages) == (5, 2)
    assert (kernel.workers, kernel.rounds, kernel.batch_tiles) == (1, 2, 2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = mix(values, 1.0, True)
    for factor in (0.75, -0.5, 1.25):
        values[0].mul_(factor)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        check(result, reference(values, 1.0), torch.bfloat16)


def test_serialized_stream_graphs_share_plans_and_workspaces(monkeypatch):
    _reset_caches(monkeypatch)
    pdl_enabled(True)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    values = [inputs(97, torch.bfloat16, True) for _ in streams]
    streams[0].wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(streams[0]):
        mix(values[0], 0.25, True)
    streams[0].synchronize()
    plans = dict(cute_fused._PLANS)
    workspaces = dict(cute_fused._WORKSPACES)

    streams[1].wait_stream(torch.cuda.current_stream())
    with mock.patch.object(
        cute_fused.cute_ext,
        "compile",
        side_effect=AssertionError("equal capacities must share compiled plans"),
    ) as compile_kernel:
        with torch.cuda.stream(streams[1]):
            mix(values[1], 0.25, True)
        streams[1].synchronize()
    compile_kernel.assert_not_called()
    assert cute_fused._PLANS.keys() == plans.keys()
    assert cute_fused._WORKSPACES.keys() == workspaces.keys()
    assert all(cute_fused._PLANS[key] is value for key, value in plans.items())
    assert all(
        cute_fused._WORKSPACES[key] is value for key, value in workspaces.items()
    )

    states = []
    with (
        mock.patch.object(
            cute_fused.cute_ext,
            "compile",
            side_effect=AssertionError(
                "warmed stream graphs must share compiled plans"
            ),
        ),
        mock.patch.object(
            cute_fused,
            "_capacity",
            side_effect=AssertionError("warmed stream graphs must reuse occupancy"),
        ),
    ):
        for stream, stream_values in zip(streams, values):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = mix(stream_values, 0.25, True)
            stream.synchronize()
            states.append((stream, stream_values, graph, output))
    for storage, _, epochs in cute_fused._WORKSPACES.values():
        storage.fill_(0x7FC1)
        epochs.fill_(2**32 - 1)
    torch.cuda.synchronize()
    for factor in (-0.75, 1.25, -0.5):
        for stream, stream_values, graph, output in reversed(states):
            stream_values[0].mul_(factor)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            check(
                output,
                reference(stream_values, 0.25),
                torch.bfloat16,
            )


def test_stream_capacity_preserves_mutable_graph_replay(monkeypatch):
    from utils import assert_no_cute_compile

    _reset_caches(monkeypatch)
    pdl_enabled(True)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    resident_workers = {
        int(stream.cuda_stream): i + 1 for i, stream in enumerate(streams)
    }
    real_capacity = cute_fused._capacity

    def limited_capacity(compiled, kernel, device, stream):
        # Model two smaller contexts while executing real kernels on the GPU.
        return min(
            real_capacity(compiled, kernel, device, stream),
            kernel.physical_clusters * resident_workers[int(stream)],
        )

    monkeypatch.setattr(cute_fused, "_capacity", limited_capacity)
    values = [inputs(97, torch.bfloat16, True) for _ in streams]
    graphs = []
    outputs = []
    for stream, tensors in zip(streams, values, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            mix(tensors, 0.25, True)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs.append(mix(tensors, 0.25, True))
        graphs.append(graph)
    with assert_no_cute_compile():
        for graph, tensors, actual in zip(graphs, values, outputs, strict=True):
            tensors[0].mul_(-0.75)
            graph.replay()
            torch.cuda.synchronize()
            check(actual, reference(tensors, 0.25), torch.bfloat16)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("has_inject", [False, True])
def test_mid_batch_replay_with_limited_cluster_capacity(
    dtype, enable_pdl, has_inject, monkeypatch
):
    if current_platform().arch_version != ArchVersion(10, 7):
        pytest.skip("requires the SM107 medium-tile preference")
    _reset_caches(monkeypatch)
    pdl_enabled(enable_pdl)
    real_capacity = cute_fused._capacity

    def limited_capacity(compiled, kernel, device, stream):
        # Model a smaller GPU context without requiring a green context in CI.
        return min(real_capacity(compiled, kernel, device, stream), 36)

    monkeypatch.setattr(cute_fused, "_capacity", limited_capacity)
    values = [
        inputs(rows, dtype, has_inject) for rows in (193, 224, 225, 207, 223, 239)
    ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for value in values[:3]:
            mix(value, 0.25, True)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with (
        mock.patch.object(
            cute_fused.cute_ext,
            "compile",
            side_effect=AssertionError("warmed fallback must not compile new rows"),
        ),
        mock.patch.object(
            cute_fused,
            "_capacity",
            side_effect=AssertionError("capture must reuse the stream capacity"),
        ),
        torch.cuda.graph(graph, stream=stream),
    ):
        outputs = [mix(value, 0.25, True) for value in values[3:]]
    for factor in (0.75, -0.5):
        for value in values:
            value[0].mul_(factor)
        for output in outputs:
            for tensor in output:
                if tensor is not None:
                    tensor.fill_(float("nan"))
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        atol, rtol = (0.002, 0.008) if dtype == torch.bfloat16 else (0.0005, 0.002)
        for actual, value in zip(outputs, values[3:]):
            for tensor, expected in zip(actual, reference(value, 0.25)):
                if expected is None:
                    assert tensor is None
                else:
                    torch.testing.assert_close(
                        tensor.cpu().double(), expected, atol=atol, rtol=rtol
                    )


def test_compiled_capacity_guards_cooperative_grid(monkeypatch):
    _reset_caches(monkeypatch)
    values = inputs(33, torch.bfloat16, True)
    real_capacity = cute_fused._capacity
    seen = []

    def record(compiled, kernel, device, stream):
        result = real_capacity(compiled, kernel, device, stream)
        seen.append((kernel, result))
        return result

    with mock.patch.object(cute_fused, "_capacity", side_effect=record):
        mix(values, 1.0, True)
    assert seen
    for kernel, clusters in seen:
        assert kernel.clusters * kernel.workers <= clusters


def test_oversized_pipeline_rejects_shared_memory_budget():
    with pytest.raises(ValueError, match="shared|memory"):
        cute_fused.FusedGatedResidualKernel(
            hc_count=4,
            hidden_size=2560,
            lowrank=320,
            up_k=128,
            down_k_order="interleaved",
            down_token_parts=1,
            epilogue_threads=128,
            token_major_up=False,
            up_token_tail_overlap=False,
            up_output_parts=1,
            up_token_parts=1,
            up_stage_counts=None,
            rotate_up_k=False,
            partial_stride=352,
            reduction_groups=1,
            token_tile_batches=1,
            up_clusters=0,
            pin_up_activation=False,
            reuse_down_storage=False,
            evict_up_weights_first=True,
            evict_down_weights_first=False,
            pipelines_per_cta=1,
            up_columns=256,
            pipeline_shared_budget=1024,
            down_pair_reduction="auto",
            down_producer_mode="auto",
            up_activation_ready_mode="auto",
            up_l2_prefetch_k=0,
            up_sigmoid_mode="exp_rcp",
            down_x_cache_mode="native",
            projection_tile=64,
            token_tile=8,
            projection_rows=324,
            split_k=4,
            projection_tiles=1,
            batch_tiles=1,
            workers=1,
            stages=5,
            final_tile=32,
            rounds=1,
            use_pdl=True,
            scale=1.0,
            weights_independent=True,
            single_tile=True,
            full_tiles=True,
        )


def test_exact_override_rejects_unvalidated_row_count():
    values = inputs(1025, torch.bfloat16, True)
    with pytest.raises(ValueError, match="at most 1024 rows"):
        mix(values, 1.0, True)


# Plan reuse is dtype independent; the output-guard test builds the other fp16 plans.
@pytest.mark.parametrize(
    ("row_counts", "dtype"),
    [
        ((1, 3, 7), torch.bfloat16),
        ((9, 13, 15), torch.bfloat16),
        ((17, 24, 31), torch.bfloat16),
        ((33, 47, 63), torch.bfloat16),
        ((97, 111, 127), torch.bfloat16),
        ((193, 207, 223), torch.bfloat16),
        ((225, 239, 255), torch.bfloat16),
        ((9, 13, 15), torch.float16),
        ((97, 111, 127), torch.float16),
    ],
)
def test_dynamic_rows_reuse_plan_in_eager_and_capture(row_counts, dtype, monkeypatch):
    _reset_caches(monkeypatch)
    values = [inputs(rows, dtype, True) for rows in row_counts]
    pdl_enabled(True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mix(values[0], 0.25, True)
    stream.synchronize()
    plans = dict(cute_fused._PLANS)
    workspaces = dict(cute_fused._WORKSPACES)
    graph = torch.cuda.CUDAGraph()
    with (
        mock.patch.object(
            cute_fused.cute_ext,
            "compile",
            side_effect=AssertionError("row changes must reuse the compiled tactic"),
        ),
        mock.patch.object(
            cute_fused,
            "_capacity",
            side_effect=AssertionError("row changes must reuse the occupancy result"),
        ),
    ):
        # Capture unseen sizes, including a different active worker count
        # sharing the already warmed workspace.
        with torch.cuda.graph(graph, stream=stream):
            outputs = [mix(value, 0.25, True) for value in values[1:]]
        for factor in (0.75, -0.5):
            for value in values:
                value[0].mul_(factor)
            graph.replay()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                eager = [mix(value, 0.25, True) for value in reversed(values)]
            stream.synchronize()
            for actual, value in zip(outputs, values[1:]):
                check(actual, reference(value, 0.25), dtype)
            for actual, value in zip(eager, reversed(values)):
                check(actual, reference(value, 0.25), dtype)
    assert cute_fused._PLANS == plans
    assert cute_fused._WORKSPACES.keys() == workspaces.keys()
    assert all(
        cute_fused._WORKSPACES[key] is value for key, value in workspaces.items()
    )


def test_startup_warms_full_tail_tiles_and_rounds(monkeypatch):
    from tokenspeed_kernel.ops.tuning import kernel_warmup

    _reset_caches(monkeypatch)
    monkeypatch.setattr(cute_fused, "_WARMED", set())
    values = inputs(256, torch.bfloat16, True)
    pdl_enabled(True)

    def forward(rows):
        return gated_residual_mix(
            values[0][:rows],
            values[1],
            values[2],
            4,
            2560,
            320,
            projection_rows=324,
            projection_scale=0.25,
            weights_independent=True,
            override=None,
            solution=None,
            activation_quantization="none",
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with kernel_warmup(enabled=True):
            forward(1)
    stream.synchronize()
    plans = dict(cute_fused._PLANS)
    row_counts = (13, 16, 23, 32, 33, 64, 95, 96, 97, 128, 191, 192, 193, 224, 255, 256)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [forward(rows) for rows in row_counts]
    for factor in (0.75, -0.5):
        values[0].mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        expected = reference(values, 0.25)
        for rows, actual in zip(row_counts, outputs):
            check(actual, tuple(value[:rows] for value in expected), torch.bfloat16)
    assert cute_fused._PLANS == plans


# Both override sources hide the call-site override alike; pair each with one setting.
@pytest.mark.parametrize(
    ("override_source", "override", "solution"),
    [
        ("environment", None, None),
        ("context", "triton_hyperconnection_mix", "triton"),
    ],
    ids=["environment-default", "context-conflicting"],
)
def test_startup_warms_global_fused_override(
    monkeypatch, override_source, override, solution
):
    from tokenspeed_kernel.ops.tuning import kernel_warmup
    from tokenspeed_kernel.selection import kernel_override

    _reset_caches(monkeypatch)
    monkeypatch.setattr(cute_fused, "_WARMED", set())
    values = inputs(1024, torch.bfloat16, True)
    pdl_enabled(True)
    env_key = "TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_HYPERCONNECTION_MIX"
    forced = "cute_fused_hyperconnection_mix"
    monkeypatch.delenv(env_key, raising=False)
    if override_source == "environment":
        monkeypatch.setenv(env_key, forced)
        context = nullcontext()
    else:
        context = kernel_override("residual", "hyperconnection_mix", forced)

    def forward(rows):
        return gated_residual_mix(
            values[0][:rows],
            values[1],
            values[2],
            4,
            2560,
            320,
            projection_rows=324,
            projection_scale=0.25,
            weights_independent=True,
            override=override,
            solution=solution,
            activation_quantization="none",
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    row_counts = (520, 257, 385, 512, 768, 1024)
    with context:
        with torch.cuda.stream(stream), kernel_warmup(enabled=True):
            forward(1)
        stream.synchronize()
        plans = dict(cute_fused._PLANS)
        assert plans
        with torch.cuda.stream(stream):
            for rows in row_counts:
                forward(rows)
                assert cute_fused._PLANS == plans
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = [forward(rows) for rows in row_counts]
        for factor in (0.75, -0.5):
            values[0].mul_(factor)
            graph.replay()
            torch.cuda.synchronize()
            expected = reference(values, 0.25)
            for rows, actual in zip(row_counts, outputs):
                check(actual, tuple(value[:rows] for value in expected), torch.bfloat16)
        assert cute_fused._PLANS == plans


@pytest.mark.parametrize("override_source", ["environment", "context"])
def test_startup_skips_global_nonfused_override(monkeypatch, override_source):
    from tokenspeed_kernel.ops.tuning import kernel_warmup
    from tokenspeed_kernel.selection import kernel_override

    _reset_caches(monkeypatch)
    monkeypatch.setattr(cute_fused, "_WARMED", set())
    values = inputs(1, torch.bfloat16, True)
    env_key = "TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_HYPERCONNECTION_MIX"
    forced = "triton_hyperconnection_mix"
    monkeypatch.delenv(env_key, raising=False)
    if override_source == "environment":
        monkeypatch.setenv(env_key, forced)
        context = nullcontext()
    else:
        context = kernel_override("residual", "hyperconnection_mix", forced)

    with context, kernel_warmup(enabled=True):
        actual = gated_residual_mix(
            *values,
            4,
            2560,
            320,
            projection_rows=324,
            projection_scale=0.25,
            weights_independent=True,
            override="cute_fused_hyperconnection_mix",
            solution="cute_fused",
            activation_quantization="none"
        )
    check(actual, reference(values, 0.25), torch.bfloat16)
    assert not cute_fused._PLANS
    assert not cute_fused._WARMED


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty:UserWarning")
def test_dynamic_rows_reuse_round_bucket_and_workspace(dtype, enable_pdl, monkeypatch):
    _reset_caches(monkeypatch)
    pdl_enabled(enable_pdl)
    first = inputs(193, dtype, True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mix(first, 0.25, True)
    stream.synchronize()
    kernel = next(iter(cute_fused._PLANS.values()))[0]
    span = kernel.n * kernel.workers
    values = [inputs(rows, dtype, True) for rows in (span + 1, span + 7, 2 * span - 1)]
    # A new scheduling-round bucket still needs warmup before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        with pytest.raises(RuntimeError, match="kernel plan.*warm up"):
            mix(values[0], 0.25, True)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mix(values[0], 0.25, True)
    stream.synchronize()
    plans = dict(cute_fused._PLANS)
    workspace_keys = set(cute_fused._WORKSPACES)
    graph = torch.cuda.CUDAGraph()
    with mock.patch.object(
        cute_fused.cute_ext,
        "compile",
        side_effect=AssertionError(
            "rows within a warmed round bucket must not compile"
        ),
    ):
        with torch.cuda.graph(graph, stream=stream):
            outputs = [mix(value, 0.25, True) for value in values]
            one_round = mix(first, 0.25, True)
        for factor in (-0.5, 0.75):
            for value in values:
                value[0].mul_(factor)
            graph.replay()
            torch.cuda.synchronize()
            for actual, value in zip(outputs, values):
                check(actual, reference(value, 0.25), dtype)
            check(one_round, reference(first, 0.25), dtype)
    assert cute_fused._PLANS == plans
    assert set(cute_fused._WORKSPACES) == workspace_keys


@pytest.mark.parametrize("rows", [3, 17, 31, 33, 63, 193, 385, 513, 1023])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dynamic_tail_preserves_output_guards(rows, dtype):
    values = inputs(rows, dtype, True)
    pdl_enabled(True)
    mix(values, 0.25, True)
    empty = torch.empty
    guards = []

    def guarded_empty(shape, **kwargs):
        if shape in ((rows, 2560), (rows, 4)):
            # Two guard rows preserve the wrapper's 16-byte alignment promise,
            # including the four-column inject output.
            storage = empty((rows + 4, shape[1]), **kwargs).fill_(123)
            guards.append(storage)
            return storage[2:-2]
        return empty(shape, **kwargs)

    with mock.patch.object(cute_fused.torch, "empty", side_effect=guarded_empty):
        result = mix(values, 0.25, True)
    torch.cuda.synchronize()
    assert len(guards) == 2
    for storage in guards:
        assert torch.all(storage[:2] == 123)
        assert torch.all(storage[-2:] == 123)
    check(result, reference(values, 0.25), dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("has_inject", [False, True])
@torch.inference_mode()
def test_prefill_up_mix_dynamic_graph(dtype, enable_pdl, has_inject):
    from tokenspeed_kernel.ops.residual import (
        pad_gated_residual_projection_weight,
        prepare_gated_residual_up_weight,
        select_gated_residual_mix_kernel,
    )
    from tokenspeed_kernel.ops.residual.cute_dsl import supports_up_mix

    if not supports_up_mix(torch.device("cuda")):
        pytest.skip("requires SM107 and CuTe DSL")
    pdl_enabled(enable_pdl)
    x, projection, up = inputs(16385, dtype, has_inject)
    logical_rows = projection.shape[0]
    projection = pad_gated_residual_projection_weight(
        projection, projection_rows=logical_rows
    )
    packed = prepare_gated_residual_up_weight(up, 4, 2560, 320)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def invoke(value, override):
        return gated_residual_mix(
            value,
            projection,
            up,
            4,
            2560,
            320,
            projection_rows=logical_rows,
            weights_independent=True,
            projection_scale=0.25,
            packed_up_weight=packed,
            override=override,
            activation_quantization="none",
        )

    # Only warm the first large shape. Unseen tails must capture without JIT.
    with torch.cuda.stream(stream):
        invoke(x[:4096], None)
    stream.synchronize()
    for rows in (4095, 4096, 4097, 8192, 16385):
        value = x[:rows]
        kernel, _ = select_gated_residual_mix_kernel(
            value,
            projection,
            up,
            4,
            2560,
            320,
            weights_independent=True,
            packed_up_weight=packed,
            override=None,
            solution=None,
            activation_quantization="none",
        )
        expected_name = (
            "triton_hyperconnection_mix"
            if rows < 4096
            else "cute_dsl_prefill_hyperconnection_mix"
        )
        assert kernel.name == expected_name
        if rows < 4096:
            invoke(value, None)
            continue
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = invoke(value, None)
        for scale in (1.0, 8.0):
            value.normal_().mul_(scale)
            graph.replay()
            expected = invoke(value, "triton_hyperconnection_mix")
            torch.cuda.synchronize()
            for actual, wanted in zip(outputs, expected):
                if wanted is None:
                    assert actual is None
                else:
                    assert torch.isfinite(actual).all()
                    torch.testing.assert_close(actual, wanted, rtol=0.005, atol=0.002)
        graph.reset()


@torch.inference_mode()
def test_prefill_up_mix_fallback_contract():
    from tokenspeed_kernel.ops.residual import (
        prepare_gated_residual_up_weight,
        select_gated_residual_mix_kernel,
    )
    from tokenspeed_kernel.ops.residual.cute_dsl import supports_up_mix

    if not supports_up_mix(torch.device("cuda")):
        pytest.skip("requires SM107 and CuTe DSL")
    value, projection, up = inputs(4097, torch.bfloat16, True)
    packed = prepare_gated_residual_up_weight(up, 4, 2560, 320)
    storage = torch.empty(packed.numel() + 1, dtype=packed.dtype, device=packed.device)
    unaligned = storage[1:].reshape_as(packed).copy_(packed)
    strided = torch.empty(4097, 20480, device=value.device, dtype=value.dtype)[:, ::2]
    strided.copy_(value)
    expected = gated_residual_mix(
        value,
        projection,
        up,
        4,
        2560,
        320,
        projection_rows=324,
        weights_independent=True,
        override="triton_hyperconnection_mix",
        activation_quantization="none",
    )
    for x, prepared, independent in (
        (value, None, True),
        (value, packed, False),
        (value, unaligned, True),
        (strided, packed, True),
    ):
        kernel, _ = select_gated_residual_mix_kernel(
            x,
            projection,
            up,
            4,
            2560,
            320,
            weights_independent=independent,
            packed_up_weight=prepared,
            override=None,
            solution=None,
            activation_quantization="none",
        )
        assert kernel.name == "triton_hyperconnection_mix"
        actual = gated_residual_mix(
            x,
            projection,
            up,
            4,
            2560,
            320,
            projection_rows=324,
            weights_independent=independent,
            packed_up_weight=prepared,
            activation_quantization="none",
        )
        torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.002)
