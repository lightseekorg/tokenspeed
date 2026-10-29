# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims


@cute.jit
def load_q(
    smem, descriptor, mbar, head, token, batch, *, boxes: int, box_elems: int, rows: int
):
    """Load Q's four column boxes into the current shared-memory slot."""
    for i in cutlass.range_constexpr(boxes):
        if prims.elect_sync():
            prims.cp_async_bulk_tensor_shared_cta_global(
                smem.subview(i * rows * box_elems),
                descriptor.get_ptr(),
                [cutlass.Int32(i * box_elems), head, token, batch],
                mbar,
            )


@cute.jit
def bulk_copy(smem_dst, gmem_src, n_bytes, mbar):
    if prims.elect_sync():
        prims.cp_async_bulk_shared_cluster_global(smem_dst, gmem_src, mbar, n_bytes)
