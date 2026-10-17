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

pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels


def test_full_scan_chunk_hints_are_cached_by_shape():
    kernels = AscendDSAKernels.__new__(AscendDSAKernels)
    kernels._full_scan_chunks = {}
    indices = torch.zeros(3, 1, 2048, dtype=torch.int32)

    first = kernels._full_scan_chunk_hints(indices)
    second = kernels._full_scan_chunk_hints(indices)

    assert first.data_ptr() == second.data_ptr()
    assert first.tolist() == [16, 16, 16]


def test_optional_cp_primitives_fail_at_device_boundary():
    kernels = AscendDSAKernels.__new__(AscendDSAKernels)
    kernels._select_local = None
    kernels._merge_partials = None

    with pytest.raises(RuntimeError, match="SelectLocalTopkIndices"):
        kernels.select_local(torch.empty(0), torch.empty(0), 0)
    with pytest.raises(RuntimeError, match="KvpAttentionMerge"):
        kernels.merge_partials(torch.empty(0), torch.empty(0))
