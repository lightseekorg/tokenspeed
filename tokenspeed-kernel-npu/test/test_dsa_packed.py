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

"""Packed single-rank SFAD preserves both public output layouts."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels


@pytest.mark.parametrize("heads", [8, 32, 64])
def test_packed_sfad_matches_split_query(heads):
    kernels = AscendDSAKernels()
    rows = 8
    q = torch.randn(rows, heads, 576, dtype=torch.bfloat16, device="npu") * 0.1
    cache = torch.randn(3, 128, 1, 576, dtype=torch.bfloat16, device="npu") * 0.1
    indices = torch.full((rows, 1, 2048), -1, dtype=torch.int32, device="npu")
    indices[:, :, :32] = torch.arange(32, dtype=torch.int32, device="npu")
    chunks = torch.full((rows,), 16, dtype=torch.int32, device="npu")
    ends = torch.arange(1, rows + 1, dtype=torch.int32, device="npu")
    lengths = torch.full((rows,), 32, dtype=torch.int32, device="npu")
    table = torch.ones(rows, 1, dtype=torch.int32, device="npu")
    padded = q.repeat(1, 32 // heads, 1) if heads < 32 else q
    reference = kernels.attention(
        padded[..., :512].contiguous(),
        padded[..., 512:].contiguous(),
        cache,
        indices,
        chunks,
        ends,
        lengths,
        table,
        192**-0.5,
    )[:, :heads]
    for major in (False, True):
        actual = kernels.attention_packed(
            q,
            cache,
            indices,
            chunks,
            ends,
            lengths,
            table,
            192**-0.5,
            head_major_output=major,
        )
        expected = (
            reference.transpose(0, 1).contiguous() if major else reference.flatten(1)
        )
        assert torch.equal(actual, expected)
