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

"""Stream-ordered byte clearing for mapped pinned-host cache allocations."""

import cuda.bindings.driver as cuda
from cutlass import Int32, Int64, Uint8, cute


class ZeroHostByteRanges:
    @cute.jit
    def __call__(
        self,
        backing: cute.Pointer,
        ranges: cute.Pointer,
        count: Int32,
        tiles: Int32,
        stream: cuda.CUstream,
    ):
        payload = cute.make_tensor(backing, cute.make_layout(9223372036854775807))
        table = cute.make_tensor(ranges, cute.make_layout((count, 2), stride=(2, 1)))
        self.zero_host_byte_ranges(payload, table).launch(
            grid=(count, tiles, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def zero_host_byte_ranges(self, payload: cute.Tensor, table: cute.Tensor):
        row, tile, _ = cute.arch.block_idx()
        tid = cute.arch.thread_idx()[0]
        offset, size = table[row, 0], table[row, 1]
        # The bounded rectangle gives large ranges enough CTAs without launching
        # a full cache-sized grid for every small null/block range.
        byte = Int64(tile) * 256 + tid
        stride = Int64(cute.arch.grid_dim()[1]) * 256
        while byte < size:
            payload[offset + byte] = Uint8(0)
            byte += stride
