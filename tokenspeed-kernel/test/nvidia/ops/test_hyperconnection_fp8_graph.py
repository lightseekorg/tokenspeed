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

"""Numerics and CUDA graph replay across FP8 mixer schedules."""

from dataclasses import replace

import pytest
import torch
from test_hyperconnection_fp8 import _check, _inputs, _mix, _offset_scale
from test_hyperconnection_fp8 import fp8_device as fp8_device
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.platform import pdl_enabled
from utils import assert_no_cute_compile, assert_no_triton_compile


@pytest.mark.parametrize("rows,tensor_scales", [(13, True), (31, False), (61, False)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_pdl", [False, True])
@torch.inference_mode()
def test_small_native_up_observes_each_updated_input(
    rows, tensor_scales, dtype, enable_pdl
):
    pdl_enabled(enable_pdl)
    model = (4, 8192, 1024)
    warm_values = _inputs(model, 1, dtype, True, tensor_scales)
    with kernel_warmup(enabled=True):
        warm_result = _mix(warm_values, model, "fp8", False)
    _check(warm_result, warm_values, model, "fp8")
    values = _inputs(model, rows, dtype, True, tensor_scales)
    tensors = values.tensors()
    originals = tuple(value.clone() for value in tensors)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with assert_no_cute_compile(), assert_no_triton_compile():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = _mix(values, model, "fp8", False)
        for index in range(len(tensors)):
            with torch.cuda.stream(stream):
                for target, original in zip(tensors, originals, strict=True):
                    target.copy_(original)
                factor = 1.5 if index >= 3 else -0.75
                tensors[index].copy_(
                    (originals[index].float() * factor).to(tensors[index].dtype)
                )
                for _ in range(3):
                    graph.replay()
            stream.synchronize()
            _check(actual, values, model, "fp8")


@pytest.mark.parametrize("rows", [33, 63, 65, 127, 129, 192, 193, 255, 257])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_pdl", [False, True])
@torch.inference_mode()
def test_native_fp8_graph_reuse_with_updated_inputs(rows, dtype, enable_pdl):
    pdl_enabled(enable_pdl)
    model = (4, 8192, 1024)
    values = _inputs(model, rows, dtype, True, rows == 63)
    with kernel_warmup(enabled=True):
        _mix(values.with_rows(1), model, "fp8", False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with assert_no_cute_compile(), assert_no_triton_compile():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = _mix(values, model, "fp8", False)
        with torch.cuda.stream(stream):
            for _ in range(3):
                values.x.mul_(0.875)
                values.down.copy_((values.down.float() * 0.875).to(values.down.dtype))
                values.up.copy_((values.up.float() * 0.875).to(values.up.dtype))
                values.down_scale.mul_(1.125)
                values.up_scale.mul_(1.125)
                graph.replay()
        stream.synchronize()
    _check(actual, values, model, "fp8")


@pytest.mark.parametrize(
    "model,rows",
    [((4, 2560, 320), rows) for rows in (112, 113, 128, 129)]
    + [((4, 8192, 1024), rows) for rows in (1, 16, 17, 32, 33, 64, 65, 128, 256)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("independent", [False, True])
@torch.inference_mode()
def test_native_fp8_reads_weights_produced_in_graph(
    model, rows, dtype, enable_pdl, independent
):
    pdl_enabled(enable_pdl)
    values = _inputs(model, rows, dtype, True, False)
    with kernel_warmup(enabled=True):
        _mix(values.with_rows(1), model, "fp8", independent)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with assert_no_cute_compile(), assert_no_triton_compile():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            # Dependent weights are produced inside the graph before the mixer.
            down = (values.down.float() * 0.875).to(values.down.dtype)
            up = (values.up.float() * 0.875).to(values.up.dtype)
            supplied = replace(values, down=down, up=up)
            actual = _mix(supplied, model, "fp8", independent)
        for _ in range(3):
            with torch.cuda.stream(stream):
                values.x.mul_(0.875)
                values.down.copy_((values.down.float() * 0.875).to(values.down.dtype))
                values.up.copy_((values.up.float() * 0.875).to(values.up.dtype))
                values.down_scale.mul_(1.125)
                values.up_scale.mul_(1.125)
                graph.replay()
            stream.synchronize()
            _check(actual, supplied, model, "fp8")


@pytest.mark.parametrize("enable_pdl", [False, True])
@torch.inference_mode()
def test_quantization_overlap_concurrent_graphs(enable_pdl):
    pdl_enabled(enable_pdl)
    model = (4, 8192, 1024)
    values = [_inputs(model, rows, torch.bfloat16, False, True) for rows in (64, 128)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    outputs = []
    for value, stream in zip(values, streams, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _mix(value, model, "fp8", False)
        stream.synchronize()
    for value, graph, stream in zip(values, graphs, streams, strict=True):
        with torch.cuda.graph(graph, stream=stream):
            outputs.append(_mix(value, model, "fp8", False))
    with assert_no_cute_compile(), assert_no_triton_compile():
        for _ in range(8):
            for value, graph, stream in zip(values, graphs, streams, strict=True):
                with torch.cuda.stream(stream):
                    value.x.mul_(0.875)
                    value.down_scale.mul_(1.03125)
                    value.up_scale.mul_(0.96875)
                    graph.replay()
        for stream in streams:
            stream.synchronize()
    for actual, value in zip(outputs, values, strict=True):
        _check(actual, value, model, "fp8")


@pytest.mark.parametrize("rows,dtype", [(33, torch.bfloat16), (63, torch.float16)])
@pytest.mark.parametrize("enable_pdl", [False, True])
@torch.inference_mode()
def test_native_hidden_tail_observes_updated_weights(rows, dtype, enable_pdl):
    pdl_enabled(enable_pdl)
    model = (4, 8192, 1024)
    values = _inputs(model, rows, dtype, True, False)
    with kernel_warmup(enabled=True):
        _mix(values, model, "fp8", False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with assert_no_cute_compile(), assert_no_triton_compile():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = _mix(values, model, "fp8", False)
        with torch.cuda.stream(stream):
            for _ in range(3):
                tail = values.up.reshape(model)[:, -16:, :]
                tail.copy_((-tail.float()).to(tail.dtype))
                values.up_scale.reshape(model[:2])[:, -16:].mul_(1.125)
                graph.replay()
        stream.synchronize()
    _check(actual, values, model, "fp8")


@pytest.mark.parametrize(
    "model,mode,rows,dtype",
    [
        ((4, 8192, 1024), "none", rows, dtype)
        for rows in (1, 15, 17, 31, 33, 63, 65, 127, 129, 255, 257)
        for dtype in (torch.bfloat16, torch.float16)
    ]
    + [
        (model, mode, rows, torch.float16)
        for model, mode, rows in (
            ((4, 2560, 320), "none", 65),
            ((4, 2560, 320), "none", 97),
            ((4, 8192, 1024), "fp8", 129),
            ((4, 8192, 1024), "fp8", 191),
        )
    ],
)
@pytest.mark.parametrize("tensor_scales", [False, True])
@torch.inference_mode()
def test_fp8_graph_with_offset_scales(model, mode, rows, dtype, tensor_scales):
    pdl_enabled(True)
    values = _inputs(model, rows, dtype, False, tensor_scales)
    values.down_scale = _offset_scale(values.down_scale)
    values.up_scale = _offset_scale(values.up_scale)
    with kernel_warmup(enabled=True):
        _mix(values.with_rows(1), model, mode, False)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with assert_no_cute_compile(), assert_no_triton_compile():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = _mix(values, model, mode, False)
        graph.replay()
        stream.synchronize()
    _check(actual, values, model, mode)


@pytest.mark.parametrize("rows", [64, 128, 256])
@pytest.mark.parametrize("use_pdl", [False, True])
@torch.inference_mode()
def test_projection_updates_in_captured_graph(rows, use_pdl):
    pdl_enabled(use_pdl)
    model = (4, 8192, 1024)
    values = _inputs(model, rows, torch.bfloat16, True, False)
    _mix(values, model, "none", True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = _mix(values, model, "none", True)
    for _ in range(4):
        with torch.cuda.stream(stream):
            values.x.mul_(-0.75)
            values.down.copy_((-values.down.float()).to(values.down.dtype))
            graph.replay()
        stream.synchronize()
        _check(actual, values, model, "none")


@pytest.mark.parametrize("rows", [128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_up_channel_scales_change_between_graph_replays(rows, dtype):
    pdl_enabled(True)
    model = (4, 8192, 1024)
    values = _inputs(model, rows, dtype, True, False)
    values.up_scale = _offset_scale(values.up_scale)
    base = values.up_scale.clone()
    factors = torch.exp2(
        (torch.arange(base.numel(), device="cuda") % 7 - 3).float()
    ).reshape_as(base)
    values.up_scale.copy_(base * factors)
    with kernel_warmup(enabled=True):
        _mix(values.with_rows(1), model, "none", False)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with assert_no_cute_compile(), assert_no_triton_compile():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = _mix(values, model, "none", False)
        for step in range(3):
            with torch.cuda.stream(stream):
                values.up_scale.copy_(base * factors.roll(step * 13 + 1))
                graph.replay()
            stream.synchronize()
            _check(actual, values, model, "none")
