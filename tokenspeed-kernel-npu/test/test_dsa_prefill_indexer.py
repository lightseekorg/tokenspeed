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

"""Prefill TopK retains score distinctions erased by a BF16 transport."""

import pytest
import torch

pytest.importorskip("torch_npu")
from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend NPU required"
)


@torch.inference_mode()
def test_prefill_topk_preserves_fp32_weight_boundary():
    torch.npu.set_device(0)
    query = torch.zeros((1, 64, 128), dtype=torch.bfloat16, device="npu")
    query[0, 0, 0] = 1
    query[0, 1, 1] = 1
    key = torch.zeros((4096, 1, 128), dtype=query.dtype, device=query.device)
    key[:2048, 0, 0] = 1
    key[2048:, 0, 1] = 1
    weights = torch.zeros((1, 64), dtype=torch.float32, device=query.device)
    weights[0, 0], weights[0, 1] = 1.001, 1.002
    # Both score classes collapse to 1.0 in BF16; FP32 must select the latter.
    assert torch.equal(
        weights[0, :2].bfloat16(),
        torch.ones(2, dtype=torch.bfloat16, device=query.device),
    )
    query_cu = torch.tensor([0, 1], dtype=torch.int64, device=query.device)
    key_cu = torch.tensor([0, 4096], dtype=torch.int64, device=query.device)
    table = torch.arange(32, dtype=torch.int32, device=query.device).unsqueeze(0)
    indices = AscendDSAKernels().prefill_index(
        query,
        key.view(32, 128, 1, 128),
        weights,
        query_cu,
        key_cu,
        table,
        2048,
        0,
        0,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        indices.cpu().flatten().sort().values,
        torch.arange(2048, 4096, dtype=indices.dtype),
        atol=0,
        rtol=0,
    )
