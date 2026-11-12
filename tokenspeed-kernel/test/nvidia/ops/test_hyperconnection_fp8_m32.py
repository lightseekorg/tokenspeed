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

"""One M64 row bucket must retain fresh graph data without recompiling."""

import pytest
import torch
from test_hyperconnection_fp8 import _check, _inputs, _mix
from test_hyperconnection_fp8 import fp8_device as fp8_device
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.platform import pdl_enabled
from utils import assert_no_cute_compile, assert_no_triton_compile


@pytest.mark.parametrize("rows", [13, 31])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_pdl", [False, True])
@torch.inference_mode()
def test_m32_row_buckets_observe_updated_inputs(rows, dtype, enable_pdl):
    pdl_enabled(enable_pdl)
    model = (4, 8192, 1024)
    tensor_scales = rows == 13
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
