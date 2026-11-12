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

"""Exercise the real projection partial helper beyond signed32bit offsets."""

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack
from test_hyperconnection_fp8 import fp8_device as fp8_device
from tokenspeed_kernel.thirdparty.cute_dsl.hc_fp8 import (
    Fp8GatedResidualProjectionEpilogue,
)
from utils import assert_no_cute_compile


class _TailPartials:
    def __init__(self, epilogue):
        self.epilogue = epilogue

    @cute.experimental.jit
    def __call__(self, partial, output, row: cutlass.Int32, stream: CUstream):
        self.kernel(partial, output, row).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream
        )

    @cute.experimental.kernel
    def kernel(self, partial, output, row: cutlass.Int32):
        if cute.arch.thread_idx()[0] == 0:
            values = self.epilogue._partials(partial, row, cutlass.Int32(0))
            output[0] = values[0]
            output[1] = values[1]


@torch.inference_mode()
def test_partial_reduction_uses_full_offset_width():
    splits, rows, columns = 8, 2**18 + 1, 1028
    required = splits * rows * columns * 4
    free, _ = torch.cuda.mem_get_info()
    if free < required + 2**30:
        pytest.skip("projection offset test requires about9GiB of free GPU memory")
    assert splits * rows * columns - 1 > 2**31
    partial = torch.empty((splits, rows, columns), device="cuda", dtype=torch.float32)
    # Only the eight far-end packets are initialized or read by this test.
    values = torch.arange(1, splits + 1, device="cuda", dtype=torch.float32)[:, None]
    values = values * torch.tensor([0.125, -0.25], device="cuda")
    partial[:, -1, :2].copy_(values)
    small = torch.zeros((splits, 1, columns), device="cuda", dtype=torch.float32)
    output = torch.empty((2,), device="cuda", dtype=torch.float32)
    epilogue = Fp8GatedResidualProjectionEpilogue(
        lowrank=1024,
        projection_rows=columns,
        hc_count=4,
        splits=splits,
        projection_scale=1.0,
        input_fp8=True,
        activation_fp8=True,
        weight_scale_per_channel=True,
        use_pdl=False,
        block_threads=512,
        early_up_signal=False,
        parallel_inject=False,
    )
    stream = CUstream(int(torch.cuda.current_stream().cuda_stream))
    view = lambda tensor: from_dlpack(tensor).mark_layout_dynamic(
        leading_dim=tensor.ndim - 1
    )
    compiled = cute_ext.compile(
        _TailPartials(epilogue), view(small), view(output), cutlass.Int32(0), stream
    )
    compiled(view(small), view(output), cutlass.Int32(0), stream)
    with assert_no_cute_compile():
        compiled(view(partial), view(output), cutlass.Int32(rows - 1), stream)
    torch.testing.assert_close(output.cpu(), torch.tensor([4.5, -9.0]), atol=0, rtol=0)
