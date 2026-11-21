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

"""Every graph replay observes distinct channel scales, including offset storage."""

import pytest
import torch
from test_hyperconnection_fp8 import _check, _inputs, _mix, _Values, fp8_device
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.platform import pdl_enabled
from utils import assert_no_cute_compile, assert_no_triton_compile


@pytest.mark.parametrize("rows", [128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_up_channel_scales_change_between_graph_replays(rows, dtype):
    pdl_enabled(True)
    model = (4, 8192, 1024)
    values = _inputs(model, rows, dtype, True, False)
    shifted = torch.empty(
        values.up_scale.numel() + 1, device="cuda", dtype=torch.float32
    )[1:].reshape_as(values.up_scale)
    shifted.copy_(values.up_scale)
    values.up_scale = shifted
    base = values.up_scale.clone()
    factors = torch.exp2(
        (torch.arange(base.numel(), device="cuda") % 7 - 3).float()
    ).reshape_as(base)
    values.up_scale.copy_(base * factors)
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
        for step in range(3):
            with torch.cuda.stream(stream):
                values.up_scale.copy_(base * factors.roll(step * 13 + 1))
                graph.replay()
            stream.synchronize()
            _check(actual, values, model, "none")
