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

import pytest
import torch

cuda = pytest.importorskip("cuda.bindings.driver")
cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")

from cutlass.cute.runtime import from_dlpack
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap
from tokenspeed_kernel.thirdparty.cute_dsl.conversion import cvt_f16x2_f32
from tokenspeed_kernel.thirdparty.cute_dsl.tma import tma_cache_hint, tma_gather4
from tokenspeed_kernel.thirdparty.msa.cute.src.common.tma_utils import (
    tma_gather4_cached,
)


@cute.kernel
def _pack(source, packed, dtype: cutlass.Constexpr):
    i = cute.arch.thread_idx()[0]
    packed[i] = cvt_f16x2_f32(source[2 * i], source[2 * i + 1], dtype)


@cute.jit
def _pack_launch(source, packed, dtype: cutlass.Constexpr, stream: cuda.CUstream):
    _pack(source, packed, dtype).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_rounding_and_lane_order(dtype):
    values = torch.linspace(-3, 5, 64, dtype=torch.float32, device="cuda")
    # Include signed zero, infinities and halfway values for round-to-nearest-even.
    values[:8] = torch.tensor(
        [
            0.0,
            -0.0,
            float("inf"),
            -float("inf"),
            1.00390625,
            1.01171875,
            1.00048828125,
            1.00146484375,
        ],
        device="cuda",
    )
    packed = torch.empty(32, dtype=torch.int32, device="cuda")
    cute_dtype = cutlass.Float16 if dtype == torch.float16 else cutlass.BFloat16
    src, dst = from_dlpack(values), from_dlpack(packed)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(_pack_launch, src, dst, cute_dtype, stream)
    compiled(src, dst, stream)
    torch.testing.assert_close(
        packed, values.to(dtype).view(torch.int32), atol=0, rtol=0
    )


@cute.kernel
def _gather(
    descriptor: cutlass.GridConstant[tmap.TensorMap],
    rows,
    output,
    offset: cutlass.Constexpr[int],
    use_msa: cutlass.Constexpr[bool],
    hint_value: cutlass.Constexpr[int],
):
    storage = cutlass.Array(
        cutlass.BFloat16, 256 + offset, alignment=128, space=cutlass.AddressSpace.smem
    )
    barrier = cutlass.Array(
        cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem
    )
    lane = cute.arch.thread_idx()[0]
    if lane == 0:
        prims.mbarrier_init(barrier, 1)
    prims.fence_mbarrier_init()
    cute.arch.sync_threads()
    hint = tma_cache_hint(hint_value)
    if lane == 0:
        prims.mbarrier_arrive_expect_tx(barrier, 512)
        if cutlass.const_expr(use_msa):
            tma_gather4_cached(
                storage.data_ptr(),
                offset * 2,
                descriptor.get_ptr(),
                0,
                rows[0],
                rows[1],
                rows[2],
                rows[3],
                barrier.data_ptr(),
                cache_hint=hint_value,
            )
        else:
            tma_gather4(
                cutlass.Int32(storage.data_ptr().toint()) + offset * 2,
                descriptor.get_ptr().toint(),
                cutlass.Int32(barrier.data_ptr().toint()),
                0,
                rows[0],
                rows[1],
                rows[2],
                rows[3],
                cache_hint=hint,
            )
    while not prims.mbarrier_try_wait_parity(barrier, cutlass.Int32(0), time_limit=1):
        pass
    for i in cutlass.range_constexpr(8):
        output[lane + i * 32] = storage.load(offset + lane + i * 32)


@cute.jit
def _gather_launch(
    source,
    rows,
    output,
    offset: cutlass.Constexpr[int],
    use_msa: cutlass.Constexpr[bool],
    hint_value: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    desc = tmap.create_tensor_map_tiled(
        global_address=source.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=(64, 32),
        global_strides=(8,),
        box_dims=(64, 1),
        swizzle=tmap.TensorMapSwizzle.none,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    _gather(desc, rows, output, offset, use_msa, hint_value).launch(
        grid=(1, 1, 1), block=(32, 1, 1), stream=stream
    )


@pytest.mark.parametrize("use_msa", [False, True])
@pytest.mark.parametrize("offset", [0, 64])
@pytest.mark.parametrize("hint", [0x12F0000000000000, 0x14F0000000000000])
def test_gather_rows_offsets_and_cache_policy(use_msa, offset, hint):
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("TMA gather4 requires SM100 or later")
    source = (
        torch.arange(32 * 64, dtype=torch.float32, device="cuda")
        .reshape(32, 64)
        .to(torch.bfloat16)
    )
    rows = torch.tensor([7, 2, 31, -1], dtype=torch.int32, device="cuda")
    output = torch.full((256,), float("nan"), dtype=torch.bfloat16, device="cuda")
    args = tuple(from_dlpack(tensor) for tensor in (source, rows, output))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(_gather_launch, *args, offset, use_msa, hint, stream)
    compiled(*args, stream)
    expected = torch.cat(
        [source[7], source[2], source[31], torch.zeros_like(source[0])]
    )
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
