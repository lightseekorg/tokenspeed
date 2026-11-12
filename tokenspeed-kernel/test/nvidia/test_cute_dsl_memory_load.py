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
from tokenspeed_kernel.thirdparty.cute_dsl.memory import load_global_f16x4_as_f32


@cute.kernel
def load_four_kernel(
    x: cute.Tensor, out: cute.Tensor, groups: cutlass.Int32, bf16: cutlass.Constexpr
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * 128 + tid
    if index < groups:
        values = load_global_f16x4_as_f32(
            x.iterator.toint() + cutlass.Int64(index) * 8, bf16
        )
        for i in cutlass.range_constexpr(4):
            out[cutlass.Int64(index) * 4 + i] = values[i]


@cute.jit
def launch_load_four(
    x, out, groups: cutlass.Int32, bf16: cutlass.Constexpr, stream: cuda.CUstream
):
    load_four_kernel(x, out, groups, bf16).launch(
        grid=[(groups + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("groups", (1, 3, 129))
def test_aligned_four_value_load(dtype, groups):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires SM90 or later")
    info = torch.finfo(dtype)
    pattern = torch.tensor(
        [
            0.0,
            -0.0,
            info.tiny / 128,
            -info.tiny / 128,
            1.0625,
            -1.1875,
            info.max,
            -info.max,
        ],
        device="cuda",
        dtype=dtype,
    )
    x = pattern.repeat((groups * 4 + 7) // 8)[: groups * 4].contiguous()
    backing = torch.full(
        (x.numel() + 8,), float("nan"), device="cuda", dtype=torch.float32
    )
    out = backing[4:-4]
    views = [
        from_dlpack(t, assumed_align=8).mark_compact_shape_dynamic(
            mode=0, stride_order=t.dim_order()
        )
        for t in (x, out)
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        launch_load_four, *views, cutlass.Int32(groups), dtype == torch.bfloat16, stream
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        capture_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(*views, cutlass.Int32(groups), capture_stream)
    try:
        for sign in (1, -1):
            x.mul_(sign)
            graph.replay()
            assert torch.equal(out.view(torch.int32), x.float().view(torch.int32))
            assert torch.isnan(backing[:4]).all() and torch.isnan(backing[-4:]).all()
    finally:
        torch.cuda.synchronize()
        graph.reset()
