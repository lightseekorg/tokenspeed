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

import math

import deep_gemm
import pytest
import torch
from deep_gemm.testing import calc_diff
from deep_gemm.utils import (
    get_mn_major_tma_aligned_tensor,
    per_block_cast_to_fp8,
    per_token_cast_to_fp8,
)
from flash_mla import flash_mla_sparse_fwd


@pytest.fixture(scope="module", autouse=True)
def require_allocated_gpus() -> None:
    assert torch.cuda.device_count() == 4
    assert torch.cuda.get_device_capability(0) == (10, 7)


def test_deepgemm_fp8_gemm_matches_reference() -> None:
    torch.manual_seed(0)
    a = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16)
    expected = (a.float() @ b.float().T).to(torch.bfloat16)

    a_fp8, a_scales = per_token_cast_to_fp8(a, use_ue8m0=True, gran_k=128)
    b_fp8, b_scales = per_block_cast_to_fp8(b, use_ue8m0=True, gran_k=128)
    a_scales = get_mn_major_tma_aligned_tensor(a_scales)
    actual = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt((a_fp8, a_scales), (b_fp8, b_scales), actual)

    torch.cuda.synchronize()
    assert calc_diff(actual, expected) < 0.001


def test_flashmla_sparse_prefill_matches_reference() -> None:
    torch.manual_seed(12)
    q = torch.randn((1, 64, 512), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((128, 1, 512), device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 1, 128)
    scale = 1.0 / math.sqrt(512)

    out, max_logits, lse = flash_mla_sparse_fwd(q, kv, indices, sm_scale=scale, d_v=512)
    scores = torch.matmul(q[0].float(), kv[:, 0, :].float().T) * scale
    expected = torch.matmul(torch.softmax(scores, dim=-1), kv[:, 0, :].float())

    torch.testing.assert_close(out[0].float(), expected, atol=0.05, rtol=0.05)
    torch.testing.assert_close(
        max_logits[0], scores.max(dim=-1).values, atol=0.02, rtol=0.02
    )
    torch.testing.assert_close(
        lse[0], torch.logsumexp(scores, dim=-1), atol=0.02, rtol=0.02
    )
