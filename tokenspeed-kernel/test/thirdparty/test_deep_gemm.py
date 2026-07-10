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

from __future__ import annotations

import pytest
import torch

deep_gemm_testing = pytest.importorskip("deep_gemm.testing")
deep_gemm_utils = pytest.importorskip("deep_gemm.utils")

from tokenspeed_kernel.ops.gemm import deep_gemm as deep_gemm_ops
from tokenspeed_kernel.ops.gemm.fp8_utils import _per_token_group_quant_8bit_raw
from tokenspeed_kernel.ops.quantization.trtllm import trtllm_fp8_packed_ue8m0
from tokenspeed_kernel.platform import ArchVersion, current_platform

platform = current_platform()


@pytest.mark.skipif(not platform.is_nvidia, reason="Requires NVIDIA GPU")
def test_deep_gemm_mm_fp8_blockscale_matches_reference(device: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for DeepGEMM verification")

    kernel = getattr(deep_gemm_ops, "deep_gemm_mm_fp8_blockscale", None)
    if kernel is None:
        pytest.skip("DeepGEMM kernel is not available")

    torch.manual_seed(0)
    m, n, k = 128, 128, 256
    use_ue8m0 = torch.cuda.get_device_capability()[0] >= 10

    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b = torch.randn((n, k), device=device, dtype=torch.bfloat16)
    expected = (a.float() @ b.float().T).to(torch.bfloat16)

    a_fp8, a_scales = deep_gemm_utils.per_token_cast_to_fp8(
        a,
        use_ue8m0=use_ue8m0,
        gran_k=128,
    )
    b_fp8, b_scales = deep_gemm_utils.per_block_cast_to_fp8(
        b,
        use_ue8m0=use_ue8m0,
        gran_k=128,
    )

    actual = kernel(
        a_fp8,
        b_fp8,
        a_scales,
        b_scales,
        torch.bfloat16,
        block_size=[128, 128],
    )

    torch.cuda.synchronize()
    assert deep_gemm_testing.calc_diff(actual, expected) < 0.001


@pytest.mark.skipif(
    not (platform.is_nvidia and platform.arch_version == ArchVersion(10, 0)),
    reason="packed UE8M0 TRT-LLM quantization requires SM100",
)
def test_deep_gemm_consumes_trtllm_packed_ue8m0(device: str) -> None:
    kernel = getattr(deep_gemm_ops, "deep_gemm_mm_fp8_blockscale", None)
    if kernel is None:
        pytest.skip("DeepGEMM kernel is not available")

    torch.manual_seed(11)
    m, n, k = 3, 128, 2048
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b = torch.randn((n, k), device=device, dtype=torch.bfloat16)
    expected = (a.float() @ b.float().T).to(torch.bfloat16)

    candidate_a, candidate_scales = trtllm_fp8_packed_ue8m0(a)
    baseline_a, baseline_scales = _per_token_group_quant_8bit_raw(
        a,
        128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    b_fp8, b_scales = deep_gemm_utils.per_block_cast_to_fp8(
        b,
        use_ue8m0=True,
        gran_k=128,
    )
    candidate = kernel(
        candidate_a,
        b_fp8,
        candidate_scales,
        b_scales,
        torch.bfloat16,
        block_size=[128, 128],
    )
    baseline = kernel(
        baseline_a,
        b_fp8,
        baseline_scales,
        b_scales,
        torch.bfloat16,
        block_size=[128, 128],
    )

    torch.cuda.synchronize()
    assert torch.equal(candidate.view(torch.int16), baseline.view(torch.int16))
    assert deep_gemm_testing.calc_diff(candidate, expected) < 0.001
