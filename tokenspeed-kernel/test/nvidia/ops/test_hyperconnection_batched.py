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


@pytest.mark.parametrize("rows", [17, 32, 64, 128, 256, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_batched_default_dispatch_selects_cute(rows, dtype, enable_pdl):
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
    assert capture._records[-1].kernel_name == "cute_fused_hyperconnection_mix"
    check(result, reference(values, 0.25), dtype)


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
    monkeypatch.setattr(cute_fused, "_PLANS", {})
    monkeypatch.setattr(cute_fused, "_COMPILED", {})
    monkeypatch.setattr(cute_fused, "_CAPACITIES", {})
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
            assert kernel.workers == (rows + 31) // 32


def test_interleaved_graphs_streams_and_high_epochs():
    states = []
    for rows, dtype, has_inject in (
        (33, torch.bfloat16, True),
        (97, torch.float16, False),
        (385, torch.bfloat16, True),
    ):
        values = inputs(rows, dtype, has_inject)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            mix(values, 0.25, True)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = mix(values, 0.25, True)
        states.append((values, graph, output, dtype))
    for storage, epochs in cute_fused._WORKSPACES.values():
        storage.fill_(0x7FC1)
        epochs.fill_(2**32 - 1)
    torch.cuda.synchronize()
    for _ in range(5):
        for values, graph, output, dtype in reversed(states):
            values[0].mul_(-0.75)
            graph.replay()
        torch.cuda.synchronize()
        for values, graph, output, dtype in states:
            check(output, reference(values, 0.25), dtype)


def test_compiled_capacity_guards_cooperative_grid(monkeypatch):
    monkeypatch.setattr(cute_fused, "_PLANS", {})
    monkeypatch.setattr(cute_fused, "_COMPILED", {})
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
    for kernel, (blocks, clusters, registers, local_bytes) in seen:
        assert blocks >= 1
        assert registers > 0
        assert local_bytes >= 0
        assert kernel.clusters * kernel.workers <= clusters


def test_wider_native_tile_does_not_inherit_small_tile_smem_override():
    kernel = cute_fused.FusedGatedResidualKernel(8, 324, 4, True, 1.0, True)
    kernel.configure(64, 8, 1, 1, 1, 5, 32)
    assert kernel.smem_bytes == 227 * 1024
    # This oversized tuning configuration must advertise its actual storage
    # requirement so the compiler/driver rejects it instead of underallocating.
    kernel.configure(64, 64, 1, 1, 1, 5, 32)
    assert kernel.smem_bytes == 274 * 1024
