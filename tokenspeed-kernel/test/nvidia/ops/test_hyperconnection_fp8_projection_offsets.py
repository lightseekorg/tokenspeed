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

"""Exercise mixer partial and residual loads beyond signed 32-bit offsets."""

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
    _load_up_residual_vector,
)
from utils import assert_no_cute_compile


def _view(value):
    return from_dlpack(value).mark_layout_dynamic(leading_dim=value.ndim - 1)


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
        pytest.skip("projection offset test requires about 9 GiB of free GPU memory")
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
    compiled = cute_ext.compile(
        _TailPartials(epilogue), _view(small), _view(output), cutlass.Int32(0), stream
    )
    compiled(_view(small), _view(output), cutlass.Int32(0), stream)
    with assert_no_cute_compile():
        compiled(_view(partial), _view(output), cutlass.Int32(rows - 1), stream)
    torch.testing.assert_close(output.cpu(), torch.tensor([4.5, -9.0]), atol=0, rtol=0)


@cute.experimental.kernel
def _read_residual_row(
    original,
    output,
    row: cutlass.Int32,
    width: cutlass.Constexpr,
    count: cutlass.Constexpr,
):
    if cute.arch.thread_idx()[0] == 0:
        values = cute.make_rmem_tensor((count, 1), original.element_type)
        _load_up_residual_vector(
            original, values, row, 0, 0, 0, width, width, count * 16
        )
        for i in cutlass.range_constexpr(count):
            output[i] = values[i, 0]


@cute.experimental.jit
def _launch_residual_row(
    original,
    output,
    row: cutlass.Int32,
    width: cutlass.Constexpr,
    count: cutlass.Constexpr,
    stream: CUstream,
):
    _read_residual_row(original, output, row, width, count).launch(
        grid=(1, 1, 1), block=(32, 1, 1), stream=stream
    )


@pytest.mark.parametrize("count", (2, 8))
@torch.inference_mode()
def test_up_residual_prefetch_beyond_int32(count):
    width = 16
    row = (1 << 31) // width + 1
    free, _ = torch.cuda.mem_get_info()
    if free < (row + 1) * width * 2 + (1 << 30):
        pytest.skip("residual offset test requires about 5 GiB of free GPU memory")
    original = torch.empty((row + 1, width), device="cuda", dtype=torch.bfloat16)
    expected = torch.arange(1, count + 1, device="cuda", dtype=torch.bfloat16)
    original[row, :count].copy_(expected)
    small = torch.zeros((1, width), device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(expected)
    stream = CUstream(int(torch.cuda.current_stream().cuda_stream))
    compiled = cute_ext.compile(
        _launch_residual_row,
        _view(small),
        _view(output),
        cutlass.Int32(0),
        width,
        count,
        stream,
    )
    with assert_no_cute_compile():
        compiled(_view(original), _view(output), cutlass.Int32(row), stream)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
