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
from tokenspeed_kernel.ops.gemm import flashinfer as flashinfer_ops
from tokenspeed_kernel.platform import current_platform

flashinfer_testing_utils = pytest.importorskip("flashinfer.testing.utils")

platform = current_platform()


def _calc_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.double()
    expected = expected.double()
    denominator = (actual * actual + expected * expected).sum()
    if denominator == 0:
        return 0.0
    similarity = 2 * (actual * expected).sum() / denominator
    return float((1 - similarity).item())


@pytest.mark.skipif(
    not platform.is_hopper,
    reason="FlashInfer sm90 blockscale GEMM is only registered on Hopper.",
)
@pytest.mark.parametrize(
    "m,n,k",
    [
        (8, 1024, 3072),
        (32, 1024, 3072),
        (8, 3072, 768),
        (32, 3072, 768),
    ],
)
def test_flashinfer_sm90_mm_fp8_blockscale_matches_reference(
    device: str,
    m: int,
    n: int,
    k: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlashInfer GEMM verification")

    kernel = getattr(flashinfer_ops, "flashinfer_sm90_mm_fp8_blockscale", None)
    if kernel is None:
        pytest.skip("FlashInfer sm90 blockscale GEMM is not available")

    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b = torch.randn((n, k), device=device, dtype=torch.bfloat16)
    expected = (a.float() @ b.float().T).to(torch.bfloat16)

    b_fp8, b_scales = flashinfer_testing_utils.per_block_cast_to_fp8(b)
    actual = kernel(
        a,
        b_fp8,
        None,
        b_scales,
        torch.bfloat16,
        block_size=[128, 128],
    )

    torch.cuda.synchronize()
    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert _calc_diff(actual, expected) < 0.001
