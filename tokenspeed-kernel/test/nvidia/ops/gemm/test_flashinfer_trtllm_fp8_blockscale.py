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

"""The small-M TRT-LLM-Gen block-FP8 GEMM against the dequantized product.

Results are held to the FP32 product of the dequantized FP8 codes; the bound
covers BF16 output rounding plus FP32 accumulation over K.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm import mm, mm_fp8_blockscale_reads_mn_major_scales
from tokenspeed_kernel.ops.gemm.flashinfer import (
    FLASHINFER_TRTLLM_FP8_BLOCKSCALE_MAX_M,
    flashinfer_trtllm_fp8_blockscale_serves,
)
from tokenspeed_kernel.ops.quantization import quantize_fp8

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not flashinfer_trtllm_fp8_blockscale_serves(1, 256),
    reason="requires SM100/SM103 and FlashInfer's TRT-LLM-Gen FP8 GEMM",
)

_MN_MAJOR = frozenset({"mn_major_scales"})


def _operands(m: int, n: int, k: int, device: str):
    torch.manual_seed(m * 7 + n + k)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight = torch.randn(n, k, device=device).to(torch.float8_e4m3fn)
    # Per-block scales spread over [0.5, 1.5] / sqrt(K): outputs are O(1) and
    # a misread scale block moves an output by far more than the tolerance.
    weight_scales = (
        torch.rand(n // 128, k // 128, device=device, dtype=torch.float32) + 0.5
    ) / k**0.5
    return x, weight, weight_scales


def _dequantized_product(q, scales, weight, weight_scales) -> torch.Tensor:
    a = q.float() * scales.float().repeat_interleave(128, dim=1)
    w = weight.float() * weight_scales.repeat_interleave(128, dim=0).repeat_interleave(
        128, dim=1
    )
    return a @ w.t()


# 17..32 is the row range the CUTLASS kernel mis-reads; this kernel must not.
@pytest.mark.parametrize(
    "m", [1, 3, 4, 13, 17, 24, 32, FLASHINFER_TRTLLM_FP8_BLOCKSCALE_MAX_M]
)
@pytest.mark.parametrize("n,k", [(4096, 4096), (1024, 4096), (4096, 1536), (4096, 512)])
@pytest.mark.parametrize("mn_major", [True, False])
def test_small_m_gemm_matches_the_dequantized_product(
    device: str, m: int, n: int, k: int, mn_major: bool
) -> None:
    x, weight, weight_scales = _operands(m, n, k, device)
    q, scales = quantize_fp8(
        x,
        granularity="token_group",
        group_size=128,
        required_features=_MN_MAJOR if mn_major else None,
    )

    out = mm(
        q,
        weight,
        A_scales=scales,
        B_scales=weight_scales,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[128, 128],
        override="flashinfer_trtllm_mm_fp8_blockscale",
    )

    expected = _dequantized_product(q, scales, weight, weight_scales)
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)


def test_mn_major_scales_are_read_for_small_m_and_k_multiple_of_256() -> None:
    bound = FLASHINFER_TRTLLM_FP8_BLOCKSCALE_MAX_M
    assert mm_fp8_blockscale_reads_mn_major_scales(1, 4096)
    assert mm_fp8_blockscale_reads_mn_major_scales(bound, 512)
    assert not mm_fp8_blockscale_reads_mn_major_scales(bound + 1, 4096)
    assert not mm_fp8_blockscale_reads_mn_major_scales(1, 640)


@pytest.mark.parametrize(
    "m,k",
    [
        (FLASHINFER_TRTLLM_FP8_BLOCKSCALE_MAX_M, 4096),
        (FLASHINFER_TRTLLM_FP8_BLOCKSCALE_MAX_M + 1, 4096),
        (4, 640),
    ],
)
def test_auto_gemm_on_either_side_of_the_bound_matches(
    device: str, m: int, k: int
) -> None:
    n = 2048
    x, weight, weight_scales = _operands(m, n, k, device)
    q, scales = quantize_fp8(
        x,
        granularity="token_group",
        group_size=128,
        required_features=(
            _MN_MAJOR if mm_fp8_blockscale_reads_mn_major_scales(m, k) else None
        ),
    )
    out = torch.empty(m, n, device=device, dtype=torch.bfloat16)

    mm(
        q,
        weight,
        A_scales=scales,
        B_scales=weight_scales,
        out=out,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[128, 128],
    )

    expected = _dequantized_product(q, scales, weight, weight_scales)
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)


def test_rejects_rows_past_the_small_m_bound(device: str) -> None:
    m = FLASHINFER_TRTLLM_FP8_BLOCKSCALE_MAX_M + 1
    x, weight, weight_scales = _operands(m, 1024, 512, device)
    q, scales = quantize_fp8(x, granularity="token_group", group_size=128)
    with pytest.raises(ValueError, match="serves M <= .* K % 256"):
        mm(
            q,
            weight,
            A_scales=scales,
            B_scales=weight_scales,
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[128, 128],
            override="flashinfer_trtllm_mm_fp8_blockscale",
        )
