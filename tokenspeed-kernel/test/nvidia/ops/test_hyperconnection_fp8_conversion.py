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

"""Check every FP8 code through the actual shared conversion helper."""

import cutlass.cute as cute
import pytest
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack
from test_hyperconnection_fp8 import fp8_device as fp8_device
from tokenspeed_kernel.thirdparty.cute_dsl.hc_fp8 import _convert_fp8_weight


@cute.experimental.kernel
def _conversion_kernel(source, target):
    tid = cute.arch.thread_idx()[0]
    inputs = cute.make_rmem_tensor(8, source.element_type)
    outputs = cute.make_rmem_tensor(8, target.element_type)
    src = cute.make_tensor(
        cute.make_ptr(
            source.element_type,
            (source.iterator + tid * 8).toint(),
            cute.AddressSpace.gmem,
            assumed_align=8,
        ),
        cute.make_layout(8),
    )
    dst = cute.make_tensor(
        cute.make_ptr(
            target.element_type,
            (target.iterator + tid * 8).toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        cute.make_layout(8),
    )
    cute.copy(
        cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), source.element_type, num_bits_per_copy=64
        ),
        src,
        inputs,
    )
    _convert_fp8_weight(inputs, outputs)
    cute.copy(
        cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), target.element_type, num_bits_per_copy=128
        ),
        outputs,
        dst,
    )


@cute.experimental.jit
def _convert(source, target, stream: CUstream):
    _conversion_kernel(source, target).launch(
        grid=(1, 1, 1), block=(32, 1, 1), stream=stream
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_all_fp8_codes(dtype):
    source = (
        torch.arange(256, device="cuda", dtype=torch.int32)
        .to(torch.uint8)
        .view(torch.float8_e4m3fn)
    )
    actual = torch.empty((256,), device="cuda", dtype=dtype)
    operands = (
        from_dlpack(source, assumed_align=16),
        from_dlpack(actual, assumed_align=16),
    )
    stream = CUstream(int(torch.cuda.current_stream().cuda_stream))
    compiled = cute_ext.compile(_convert, *operands, stream)
    compiled(*operands, stream)
    expected = source.to(dtype)
    torch.cuda.synchronize()
    finite = torch.isfinite(expected)
    assert torch.equal(
        actual.view(torch.int16)[finite], expected.view(torch.int16)[finite]
    )
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
