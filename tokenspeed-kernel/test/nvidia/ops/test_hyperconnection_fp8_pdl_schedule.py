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

"""Exercise quantization handoffs through mutable inputs and concurrent graphs."""

import pytest
import torch
from test_hyperconnection_fp8 import _check, _inputs, _mix, _Values
from test_hyperconnection_fp8 import fp8_device as fp8_device
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.platform import pdl_enabled
from utils import assert_no_cute_compile, assert_no_triton_compile


@pytest.mark.parametrize("rows", [112, 113, 119, 127, 128, 129])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("independent", [False, True])
@torch.inference_mode()
def test_legacy_pdl_schedule_mutable_graph(rows, dtype, enable_pdl, independent):
    pdl_enabled(enable_pdl)
    model = (4, 2560, 320)
    values = _inputs(model, rows, dtype, True, False)
    warm = _Values(
        values.x[:1], values.down, values.up, values.down_scale, values.up_scale
    )
    with kernel_warmup(enabled=True):
        _mix(warm, model, "fp8", independent)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with assert_no_cute_compile(), assert_no_triton_compile():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            # Dependent weights are produced inside the graph before the mixer.
            down = (values.down.float() * 0.875).to(values.down.dtype)
            up = (values.up.float() * 0.875).to(values.up.dtype)
            supplied = _Values(values.x, down, up, values.down_scale, values.up_scale)
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
