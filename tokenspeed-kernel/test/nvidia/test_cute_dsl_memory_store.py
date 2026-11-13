# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute.runtime import from_dlpack
from tokenspeed_kernel.thirdparty.cute_dsl.memory import store_global_f32x4_as_fp8_e4m3


@cute.kernel
def store_four_kernel(x: cute.Tensor, out: cute.Tensor, groups: cutlass.Int32):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * 128 + tid
    if index < groups:
        offset = cutlass.Int64(index) * 4
        store_global_f32x4_as_fp8_e4m3(
            out.iterator + offset,
            x[offset],
            x[offset + 1],
            x[offset + 2],
            x[offset + 3],
        )


@cute.jit
def launch_store_four(x, out, groups: cutlass.Int32, stream: cuda.CUstream):
    store_four_kernel(x, out, groups).launch(
        grid=[(groups + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
    )


@pytest.mark.parametrize("groups", (1, 3, 129))
def test_aligned_four_value_store(groups):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires SM90 or later")
    pattern = torch.tensor(
        [
            0.0,
            -0.0,
            1.0625,
            -1.1875,
            448.0,
            -448.0,
            1000.0,
            -1000.0,
            2**-10,
            -(2**-10),
            3 * 2**-10,
            -3 * 2**-10,
        ],
        device="cuda",
        dtype=torch.float32,
    )
    x = pattern.repeat((groups * 4 + 11) // 12)[: groups * 4].contiguous()
    backing = torch.full((x.numel() + 8,), 0x55, device="cuda", dtype=torch.uint8)
    out = backing[4:-4].view(torch.float8_e4m3fn)
    views = [
        from_dlpack(t, assumed_align=4).mark_compact_shape_dynamic(
            mode=0, stride_order=t.dim_order()
        )
        for t in (x, out)
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(launch_store_four, *views, cutlass.Int32(groups), stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        capture_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(*views, cutlass.Int32(groups), capture_stream)
    try:
        for sign in (1, -1):
            x.mul_(sign)
            graph.replay()
            expected = x.clamp(-448, 448).to(torch.float8_e4m3fn)
            assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
            assert (backing[:4] == 0x55).all() and (backing[-4:] == 0x55).all()
    finally:
        torch.cuda.synchronize()
        graph.reset()
