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

from unittest import mock

import pytest
import torch
from tokenspeed_kernel import gated_residual_mix
from tokenspeed_kernel.ops.residual import cute_fused
from tokenspeed_kernel.platform import pdl_enabled
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
    x, w, u = (value.cpu().double() for value in values)
    down = (x @ w.T) * scale
    gates = torch.nn.functional.silu(down[:, :320]) @ u.T
    mixed = (gates.sigmoid() * x).reshape(-1, 4, 2560).mean(1)
    return mixed, down[:, 320:] if w.shape[0] == 324 else None


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
        projection_scale=scale,
        weights_independent=independent,
        override="cute_fused_hyperconnection_mix",
        solution=None,
    )


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
        (512, "triton_hyperconnection_mix"),
        (1024, "triton_hyperconnection_mix"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_pdl", [False, True])
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
        projection_scale=0.25,
        weights_independent=True,
        override=None,
        solution=None,
    )
    assert capture._records[-1].kernel_name == expected_kernel
    check(result, reference(values, 0.25), dtype)


@pytest.mark.parametrize(
    ("rows", "expected_kernel"),
    [
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
            projection_scale=0.25,
            weights_independent=True,
            override=None,
            solution=None,
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
        192,
        193,
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


@pytest.mark.parametrize("rows", [17, 33, 97, 193, 385, 1024])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("has_inject", [False, True])
def test_fp16_optional_inject_and_pdl(rows, enable_pdl, has_inject):
    values = inputs(rows, torch.float16, has_inject)
    pdl_enabled(enable_pdl)
    check(mix(values, 1.0, False), reference(values, 1.0), torch.float16)


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
                epochs.numel() == 2 * ((rows + 31) // 32) for _, epochs in workspaces
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
    for storage, epochs in cute_fused._WORKSPACES.values():
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


def test_stream_capacity_selects_worker_key_without_stream_key(monkeypatch):
    _reset_caches(monkeypatch)
    pdl_enabled(True)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    stream_ids = {int(stream.cuda_stream) for stream in streams}
    resident_workers = dict(zip(stream_ids, (1, 2)))

    def fake_compile(kernel, *operands):
        return mock.Mock()

    def fake_capacity(compiled, kernel, device, stream):
        stream_id = int(torch.cuda.current_stream(device).cuda_stream)
        return kernel.clusters * resident_workers[stream_id]

    values = inputs(97, torch.bfloat16, True)
    with (
        mock.patch.object(
            cute_fused.cute_ext, "compile", side_effect=fake_compile
        ) as compile_kernel,
        mock.patch.object(cute_fused, "_capacity", side_effect=fake_capacity),
    ):
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                mix(values, 0.25, True)
            stream.synchronize()

    assert compile_kernel.call_count == 4
    static_keys = {plan_key[:-2] for plan_key in cute_fused._PLANS}
    assert len(static_keys) == 1
    assert {(plan_key[-2], plan_key[-1]) for plan_key in cute_fused._PLANS} == {
        (1, 1),
        (1, 4),
        (2, 1),
        (2, 2),
    }
    assert {capacity_key[0] for capacity_key in cute_fused._CAPACITIES} == stream_ids
    assert {workspace_key[2] for workspace_key in cute_fused._WORKSPACES} == {1, 2}


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


def test_kernel_configuration_is_complete_at_construction():
    def make(token_tile):
        return cute_fused.FusedGatedResidualKernel(
            projection_tile=64,
            token_tile=token_tile,
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

    small = make(8)
    wide = make(64)
    assert (small.n, small.slot_rows, small.tmem_columns) == (8, 16, 32)
    assert (wide.n, wide.slot_rows, wide.tmem_columns) == (64, 64, 64)
    assert "configure" not in vars(type(small))
    assert "smem_bytes" not in vars(small)


def test_exact_override_rejects_unvalidated_row_count():
    values = inputs(1025, torch.bfloat16, True)
    with pytest.raises(ValueError, match="at most 1024 rows"):
        mix(values, 1.0, True)


@pytest.mark.parametrize(
    "row_counts",
    [
        (1, 3, 7),
        (9, 13, 15),
        (17, 24, 31),
        (33, 47, 63),
        (97, 111, 127),
        (193, 207, 223),
        (224, 256),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
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


@pytest.mark.parametrize("rows", [3, 193, 385, 513, 1023])
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
