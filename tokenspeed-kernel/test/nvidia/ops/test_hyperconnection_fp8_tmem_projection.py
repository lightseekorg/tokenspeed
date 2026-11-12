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

"""Projection TMEM rings retain tails, scale offsets and graph replay."""

import pytest
import torch
from test_hyperconnection_fp8 import (
    _check,
    _inputs,
    _mix,
    _quantize,
    _Values,
    fp8_device,
)
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.platform import pdl_enabled
from utils import assert_no_cute_compile, assert_no_triton_compile


@pytest.mark.parametrize("rows", [1, 7, 15, 17, 31, 33, 63, 65, 127, 129, 255])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tensor_scales", [False, True])
@torch.inference_mode()
def test_tmem_projection_tail_offset_scales_and_capture(rows, dtype, tensor_scales):
    pdl_enabled(True)
    model = (4, 8192, 1024)
    values = _inputs(model, rows, dtype, False, tensor_scales)
    for name in ("down_scale", "up_scale"):
        original = values.down_scale if name == "down_scale" else values.up_scale
        shifted = torch.empty(original.numel() + 1, device="cuda", dtype=torch.float32)[
            1:
        ].reshape_as(original)
        shifted.copy_(original)
        if name == "down_scale":
            values.down_scale = shifted
        else:
            values.up_scale = shifted
    first = _Values(
        values.x[:1], values.down, values.up, values.down_scale, values.up_scale
    )
    with kernel_warmup(enabled=True):
        _mix(first, model, "none", False)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with assert_no_cute_compile(), assert_no_triton_compile():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = _mix(values, model, "none", False)
        graph.replay()
        stream.synchronize()
    _check(actual, values, model, "none")


@pytest.mark.parametrize("rows", [128, 256])
@pytest.mark.parametrize("use_pdl", [False, True])
@torch.inference_mode()
def test_projection_updates_in_captured_graph(rows, use_pdl):
    pdl_enabled(use_pdl)
    model = (4, 8192, 1024)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(20261046)
        x = torch.randn(rows, 32768, device="cuda", dtype=torch.bfloat16)
        down16 = (
            torch.randn(1028, 32768, device="cuda", dtype=torch.bfloat16)
            / (32768**0.5)
            / 4
        ).contiguous()
        up16 = (
            torch.randn(32768, 1024, device="cuda", dtype=torch.bfloat16) / 32
        ).contiguous()
    down, ds = _quantize(down16, False)
    up, us = _quantize(up16, False)
    values = _Values(x, down, up, ds, us)
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
