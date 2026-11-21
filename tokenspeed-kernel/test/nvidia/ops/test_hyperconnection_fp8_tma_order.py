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


"""Native TMA stages retain dependency ordering across captured input updates."""

import pytest
import torch
from test_hyperconnection_fp8 import _check, _inputs, _mix, _Values
from test_hyperconnection_fp8 import fp8_device as fp8_device
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.platform import pdl_enabled
from utils import assert_no_cute_compile, assert_no_triton_compile


@pytest.mark.parametrize("rows", [65, 127, 193, 257])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_pdl", [False, True])
@torch.inference_mode()
def test_native_tma_mutable_graph_dependencies(rows, dtype, enable_pdl):
    pdl_enabled(enable_pdl)
    model = (4, 8192, 1024)
    values = _inputs(model, rows, dtype, True, False)
    first = _Values(
        values.x[:1], values.down, values.up, values.down_scale, values.up_scale
    )
    with kernel_warmup(enabled=True):
        _mix(first, model, "fp8", False)
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
