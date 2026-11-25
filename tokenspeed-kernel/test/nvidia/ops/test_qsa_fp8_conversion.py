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

"""Exact E4M3 conversion used by the QSA K/V producer."""

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")

from cuda.bindings import driver as cuda
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import target_version
from tokenspeed_kernel.thirdparty.cute_dsl.conversion import convert_e4m3_bf16
from utils import assert_no_cute_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="QSA packed conversion requires SM100+",
)


@cute.kernel
def _convert_pairs_kernel(
    src: cute.Tensor, dst: cute.Tensor, native: cutlass.Constexpr
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * 256 + thread
    inputs = cute.make_rmem_tensor(2, cutlass.Float8E4M3FN)
    outputs = cute.make_rmem_tensor(2, cutlass.BFloat16)
    cute.recast_tensor(inputs, cutlass.Int16)[0] = src[index]
    convert_e4m3_bf16(inputs, outputs, native=native)
    dst[index] = cute.recast_tensor(outputs, cutlass.Int32)[0]


@cute.jit
def _convert_pairs(
    src: cute.Tensor, dst: cute.Tensor, native: cutlass.Constexpr, stream: cuda.CUstream
):
    _convert_pairs_kernel(src, dst, native).launch(
        grid=(256, 1, 1), block=(256, 1, 1), stream=stream
    )


@pytest.mark.parametrize("native", [False, True])
def test_every_fp8_pair_converts_exactly_and_refreshes_in_graph(native):
    if native and not target_version(min_version="13.2"):
        pytest.skip("Native E4M3 to BF16 conversion requires CUDA 13.2/PTX 9.2")
    # All 65536 adjacent byte combinations, including signed zero, the
    # smallest subnormals, maximum finite values and either NaN encoding.
    src = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    dst = torch.empty(65536, device="cuda", dtype=torch.int32)
    src_meta, dst_meta = (from_dlpack(t, assumed_align=16) for t in (src, dst))

    def stream():
        return cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled = cute.compile(_convert_pairs, src_meta, dst_meta, native, stream())

    def check():
        expected = src.view(torch.float8_e4m3fn).to(torch.bfloat16)
        actual = dst.view(torch.bfloat16)
        finite = torch.isfinite(expected)
        # Compare bits to check that negative zero also survives conversion.
        assert torch.equal(
            actual.view(torch.int16)[finite], expected.view(torch.int16)[finite]
        )
        assert torch.equal(torch.isnan(actual), torch.isnan(expected))

    compiled(src_meta, dst_meta, stream())
    check()
    graph = torch.cuda.CUDAGraph()
    with assert_no_cute_compile(), torch.cuda.graph(graph):
        compiled(src_meta, dst_meta, stream())
    src.copy_(src.flip(0))
    with assert_no_cute_compile():
        graph.replay()
        check()
