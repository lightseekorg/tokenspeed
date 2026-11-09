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

"""Fused mixer pipelines retain numeric and dynamic-row graph behavior."""

import math

import pytest
import torch
from cuda.bindings import driver
from tokenspeed_kernel.numerics.torch.residual import torch_hyperconnection_mix
from tokenspeed_kernel.ops.residual import cute_fused, gated_residual_mix
from tokenspeed_kernel.platform import pdl_enabled
from utils import assert_no_cute_compile


@pytest.fixture(autouse=True)
def fused_device():
    if not torch.cuda.is_available() or not cute_fused.supports_fused_hc(
        torch.device("cuda")
    ):
        pytest.skip("requires supported Blackwell and CuTe DSL")
    previous_pdl = pdl_enabled()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.cuda.synchronize()
    pdl_enabled(previous_pdl)
    torch.set_num_threads(previous_threads)


def _inputs(rows, dtype, inject, model):
    branches, hidden_size, lowrank = model
    width = branches * hidden_size
    generator = torch.Generator(device="cuda").manual_seed(1409 + rows)
    return tuple(
        torch.randn(shape, device="cuda", dtype=dtype, generator=generator) * scale
        for shape, scale in (
            ((rows, width), 1.0),
            (
                (lowrank + (branches if inject else 0), width),
                1.0 / math.sqrt(width) / branches,
            ),
            ((width, lowrank), 1.0 / math.sqrt(lowrank)),
        )
    )


def _mix(values, inject, weights_independent, model):
    return gated_residual_mix(
        *values,
        *model,
        projection_rows=model[2] + (model[0] if inject else 0),
        projection_scale=1.0,
        weights_independent=weights_independent,
        override="cute_fused_hyperconnection_mix",
        solution=None,
    )


def _check(actual, values, inject, model):
    expected = torch_hyperconnection_mix(
        *(value.cpu() for value in values),
        *model,
        model[2] + (model[0] if inject else 0),
        1.0,
        True,
    )
    atol, rtol = (0.004, 0.01) if values[0].dtype == torch.bfloat16 else (0.001, 0.002)
    for result, reference in zip(actual, expected, strict=True):
        if reference is None:
            assert result is None
        else:
            torch.testing.assert_close(
                result.cpu().double(), reference, atol=atol, rtol=rtol
            )


def _check_single_kernel(graph):
    handle = driver.CUgraph(graph.raw_cuda_graph())
    status, _, count = driver.cuGraphGetNodes(handle, 0)
    assert status == driver.CUresult.CUDA_SUCCESS
    status, nodes, count = driver.cuGraphGetNodes(handle, count)
    assert status == driver.CUresult.CUDA_SUCCESS
    assert count == 1
    status, kind = driver.cuGraphNodeGetType(nodes[0])
    assert status == driver.CUresult.CUDA_SUCCESS
    assert kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL


@pytest.mark.parametrize(
    "model", [(4, 2560, 320), (4, 8192, 1024)], ids=["legacy", "wide"]
)
@pytest.mark.parametrize(
    "rows,inject",
    [
        (rows, True)
        for rows in (
            1,
            4,
            8,
            16,
            24,
            32,
            63,
            64,
            65,
            96,
            128,
            192,
            256,
            95,
            127,
            191,
            255,
        )
    ]
    + [(rows, False) for rows in (64, 128, 256)],
)
@pytest.mark.parametrize(
    "dtype,enable_pdl,weights_independent",
    [(torch.bfloat16, False, True), (torch.float16, True, False)],
)
@torch.inference_mode()
def test_pipeline_graph_uses_current_inputs(
    rows, inject, dtype, enable_pdl, weights_independent, model
):
    values = _inputs(rows, dtype, inject, model)
    originals = tuple(value.clone() for value in values)
    pdl_enabled(enable_pdl)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = _mix(values, inject, weights_independent, model)
    stream.synchronize()
    _check(actual, values, inject, model)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        actual = _mix(values, inject, weights_independent, model)
    _check_single_kernel(graph)
    for changed, factor in enumerate((-0.75, -2.0, 4.0)):
        with torch.cuda.stream(stream):
            for value, original in zip(values, originals, strict=True):
                value.copy_(original)
            values[changed].mul_(factor)
            for _ in range(3):
                graph.replay()
        stream.synchronize()
        _check(actual, values, inject, model)


@pytest.mark.parametrize(
    "model", [(4, 2560, 320), (4, 8192, 1024)], ids=["legacy", "wide"]
)
@pytest.mark.parametrize(
    "rows",
    [
        (5, 6, 7),
        (17, 19, 23),
        (33, 39, 47),
        (65, 79, 95),
        (97, 113, 127),
        (129, 157, 191),
        (193, 221, 255),
    ],
)
@torch.inference_mode()
def test_pipeline_reuses_compilation_within_row_bucket(rows, model):
    values = _inputs(max(rows), torch.bfloat16, True, model)
    first = (values[0][: rows[0]], values[1], values[2])
    _mix(first, True, True, model)
    torch.cuda.synchronize()
    with assert_no_cute_compile():
        for count in rows[1:]:
            current = (values[0][:count], values[1], values[2])
            actual = _mix(current, True, True, model)
            torch.cuda.synchronize()
            _check(actual, current, True, model)
