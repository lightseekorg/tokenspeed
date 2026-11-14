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
from test_hyperconnection_fp8 import _check, _inputs, _mix, _Values
from test_hyperconnection_fp8 import fp8_device as fp8_device
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.platform import pdl_enabled
from utils import assert_no_cute_compile, assert_no_triton_compile


@pytest.mark.parametrize(
    ("model", "mode", "rows"),
    [
        ((4, 8192, 1024), "none", 33),
        ((4, 8192, 1024), "none", 63),
        ((4, 2560, 320), "none", 65),
        ((4, 2560, 320), "none", 97),
        ((4, 8192, 1024), "fp8", 129),
        ((4, 8192, 1024), "fp8", 191),
    ],
)
@pytest.mark.parametrize("tensor_scales", [False, True])
@torch.inference_mode()
def test_fp8_tail_rows_with_offset_scales_and_capture(model, mode, rows, tensor_scales):
    pdl_enabled(True)
    values = _inputs(model, rows, torch.float16, False, tensor_scales)
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
        _mix(first, model, mode, False)
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


@pytest.mark.parametrize("model", [(4, 2560, 320), (4, 8192, 1024)])
@pytest.mark.parametrize("mode", ["none", "fp8"])
@torch.inference_mode()
def test_unaligned_channel_scales_keep_graph_contract(model, mode):
    pdl_enabled(True)
    values = _inputs(model, 255, torch.bfloat16, True, False)
    down = torch.empty(
        values.down_scale.numel() + 1, device="cuda", dtype=torch.float32
    )[1:].reshape_as(values.down_scale)
    up = torch.empty(values.up_scale.numel() + 1, device="cuda", dtype=torch.float32)[
        1:
    ].reshape_as(values.up_scale)
    down.copy_(values.down_scale)
    up.copy_(values.up_scale)
    values.down_scale, values.up_scale = down, up
    first = _Values(values.x[:1], values.down, values.up, down, up)
    with kernel_warmup(enabled=True):
        _mix(first, model, mode, True)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with assert_no_cute_compile(), assert_no_triton_compile():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            results = []
            for rows in [13, 31, 95, 127, 255]:
                current = _Values(values.x[:rows], values.down, values.up, down, up)
                results.append(_mix(current, model, mode, True))
        graph.replay()
        torch.cuda.synchronize()
    for rows, result in zip([13, 31, 95, 127, 255], results, strict=True):
        _check(
            result,
            _Values(values.x[:rows], values.down, values.up, down, up),
            model,
            mode,
        )
    with torch.cuda.stream(stream):
        down.mul_(1.25)
        up.mul_(0.75)
        graph.replay()
    stream.synchronize()
    for rows, result in zip([13, 31, 95, 127, 255], results, strict=True):
        _check(
            result,
            _Values(values.x[:rows], values.down, values.up, down, up),
            model,
            mode,
        )


@torch.inference_mode()
def test_two_token_up_falls_back_above256_after_startup_warmup():
    pdl_enabled(False)
    model = (4, 8192, 1024)
    values = _inputs(model, 257, torch.bfloat16, True, False)
    first = _Values(
        values.x[:1], values.down, values.up, values.down_scale, values.up_scale
    )
    with kernel_warmup(enabled=True):
        _mix(first, model, "none", True)
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
