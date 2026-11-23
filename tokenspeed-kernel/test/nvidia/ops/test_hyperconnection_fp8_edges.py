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

"""FP8 tails, scale offsets, and graph replay across tile boundaries."""

import pytest
import torch
from test_hyperconnection_fp8 import _check, _inputs, _mix, _offset_scale
from test_hyperconnection_fp8 import fp8_device as fp8_device
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.platform import pdl_enabled
from utils import assert_no_cute_compile, assert_no_triton_compile


@pytest.mark.parametrize("model", [(4, 2560, 320), (4, 8192, 1024)])
@pytest.mark.parametrize("mode", ["none", "fp8"])
@torch.inference_mode()
def test_unaligned_channel_scales_keep_graph_contract(model, mode):
    pdl_enabled(True)
    values = _inputs(model, 255, torch.bfloat16, True, False)
    values.down_scale = _offset_scale(values.down_scale)
    values.up_scale = _offset_scale(values.up_scale)
    with kernel_warmup(enabled=True):
        _mix(values.with_rows(1), model, mode, True)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with assert_no_cute_compile(), assert_no_triton_compile():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            batches = [values.with_rows(rows) for rows in (13, 31, 95, 127, 255)]
            results = [_mix(batch, model, mode, True) for batch in batches]
        graph.replay()
        torch.cuda.synchronize()
    for batch, result in zip(batches, results, strict=True):
        _check(result, batch, model, mode)
    with torch.cuda.stream(stream):
        values.down_scale.mul_(1.25)
        values.up_scale.mul_(0.75)
        graph.replay()
    stream.synchronize()
    for batch, result in zip(batches, results, strict=True):
        _check(result, batch, model, mode)


@torch.inference_mode()
def test_two_token_up_falls_back_above256_after_startup_warmup():
    pdl_enabled(False)
    model = (4, 8192, 1024)
    values = _inputs(model, 257, torch.bfloat16, True, False)
    with kernel_warmup(enabled=True):
        _mix(values.with_rows(1), model, "none", True)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with assert_no_cute_compile(), assert_no_triton_compile():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = _mix(values, model, "none", True)
        graph.replay()
        stream.synchronize()
    _check(actual, values, model, "none")
