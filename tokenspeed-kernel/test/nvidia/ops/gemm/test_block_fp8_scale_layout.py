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

"""Block-FP8 activation scales reach the GEMM in the layout it reads."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import mm, mm_activation_scale_layout
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_blackwell, reason="Blackwell only"
)

N, K = 512, 1024
BF16 = torch.bfloat16


def _weights():
    torch.manual_seed(2427)
    b = (torch.randn(N, K, device="cuda") * 0.02).to(torch.float8_e4m3fn)
    sb = torch.rand(N // 128, K // 128, device="cuda") * 0.02 + 0.001
    return b, sb


def _quantize_like(x: torch.Tensor, layout: str):
    return quantize_fp8(
        x,
        granularity="token_group",
        group_size=128,
        override="trtllm_quantize_fp8_mn_major" if layout == "mn_major" else None,
    )


def _quantize(m: int, mn_major: bool):
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    return _quantize_like(x, "mn_major" if mn_major else "k_major")


@pytest.mark.parametrize("m", [1, 3, 17, 64, 257])
def test_cutedsl_reads_padded_scales_like_dense_ones(m: int) -> None:
    b, sb = _weights()
    a, dense = _quantize(m, mn_major=True)
    assert dense.t().is_contiguous()
    # Fused all-reduce quantization pads the token stride to four.
    storage = torch.full((K // 128, (m + 3) // 4 * 4), float("nan"), device="cuda")
    padded = storage[:, :m].copy_(dense.t()).t()
    kwargs = dict(B_scales=sb, quant="mxfp8", block_size=[128, 128], out_dtype=BF16)
    expected = mm(a, b, A_scales=dense, solution="trtllm_cutedsl", **kwargs)
    actual = mm(a, b, A_scales=padded, solution="trtllm_cutedsl", **kwargs)
    assert torch.equal(actual, expected)


def test_cutedsl_rejects_k_major_scales() -> None:
    b, sb = _weights()
    a, scales = _quantize(17, mn_major=False)
    with pytest.raises(ValueError, match="MN-major activation scales"):
        mm(
            a,
            b,
            A_scales=scales,
            B_scales=sb,
            out_dtype=BF16,
            quant="mxfp8",
            block_size=[128, 128],
            override="trtllm_cutedsl_mm_fp8_blockscale",
        )


@pytest.mark.parametrize("m", [1, 16, 17, 64])
def test_scales_in_the_reported_layout_give_canonical_results(m: int) -> None:
    b, sb = _weights()
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    layout = mm_activation_scale_layout(x, b, B_scales=sb, block_size=[128, 128])
    a, canonical = quantize_fp8(x, granularity="token_group", group_size=128)
    a_native, native = _quantize_like(x, layout)
    # Canonical scales as an MN-major kernel used to receive them: transposed.
    before = canonical.t().contiguous().t() if layout == "mn_major" else canonical
    kwargs = dict(B_scales=sb, quant="mxfp8", block_size=[128, 128], out_dtype=BF16)
    assert torch.equal(a_native.view(torch.uint8), a.view(torch.uint8))
    assert torch.equal(
        mm(a_native, b, A_scales=native, **kwargs),
        mm(a, b, A_scales=before, **kwargs),
    )


def test_online_quantization_feeds_the_cutedsl_kernel_mn_major_scales() -> None:
    b, sb = _weights()
    x = torch.randn(64, K, device="cuda", dtype=torch.bfloat16)
    a, scales = quantize_fp8(
        x, granularity="token_group", group_size=128, solution="triton"
    )
    kwargs = dict(B_scales=sb, quant="mxfp8", block_size=[128, 128], out_dtype=BF16)
    expected = mm(
        a, b, A_scales=scales.t().contiguous().t(), solution="trtllm_cutedsl", **kwargs
    )
    assert torch.equal(mm(x, b, solution="trtllm_cutedsl", **kwargs), expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
