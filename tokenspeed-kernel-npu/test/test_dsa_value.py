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

"""Head-major value projection keeps token-major output order."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels


@pytest.mark.parametrize("rows,heads", [(1, 8), (64, 8), (64, 32), (64, 64)])
def test_value_projection_layout(rows, heads):
    x = torch.randn(heads, rows, 512, dtype=torch.bfloat16, device="npu")
    weight = torch.randn(heads, 512, 128, dtype=torch.bfloat16, device="npu")
    reference = torch.bmm(x, weight).transpose(0, 1).contiguous().flatten(1)
    actual = AscendDSAKernels.project_value(x, weight)
    error = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert error < 1e-3
    assert actual.shape == reference.shape
    assert actual.is_contiguous()
