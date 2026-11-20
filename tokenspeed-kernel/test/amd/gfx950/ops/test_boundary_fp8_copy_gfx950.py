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
"""The fused layer boundary's per-row FP8 copy of its next input on gfx950."""

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("gfx950 required", allow_module_level=True)

from tokenspeed_kernel.ops.residual import (  # noqa: E402
    post_norm_residual_pre_norm,
    post_norm_residual_pre_norm_bf16,
    post_norm_residual_pre_norm_fp8,
    residual_gate_coefficients,
)


def _quantize_rows_amax_div(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The registered amax-division token quantizer the FP8 copy matches."""
    from tokenspeed_kernel.ops.quantization import quantize_fp8

    codes, scales = quantize_fp8(
        x,
        granularity="token",
        scale_encoding="float32",
        enable_pdl=False,
        required_features=frozenset({"token_fp8_amax_div"}),
    )
    return codes, scales.reshape(-1)


@pytest.mark.parametrize("rows", [1, 17, 128, 172, 256])
def test_boundary_fp8_next_input_matches_the_row_reference(rows: int) -> None:
    """The FP8 next input is the per-row FP8 quantization of the BF16 next
    input (scale = max(row amax, 1e-20) / 448 within one FP32 ulp, codes =
    rows / scale), at the FP8 window's row counts and past it, and asking for it leaves
    the residual and BF16 outputs unchanged."""
    torch.manual_seed(43)
    stream = torch.randn(rows, 8192, device="cuda")
    branch = torch.randn_like(stream).bfloat16()
    post_weight = torch.randn(8192, device="cuda").bfloat16() * 0.1
    pre_weight = torch.randn_like(post_weight) * 0.1
    alpha, beta = residual_gate_coefficients(torch.randn_like(post_weight), 0.3, 1e-3)
    # A zero row takes the scale floor instead of dividing by zero.
    branch[0].zero_()
    stream[0].zero_()
    args = (stream, branch, post_weight, 1e-6, 1.0, alpha, beta, pre_weight,
            1e-6, 1.0, False, None, 1)  # fmt: skip
    stream_out, next_input, (codes, scales) = post_norm_residual_pre_norm_bf16(
        *args, return_pre_fp8=True
    )
    plain = post_norm_residual_pre_norm_bf16(*args, return_pre_fp8=False)
    assert plain[2] is None
    assert torch.equal(stream_out, plain[0]) and torch.equal(next_input, plain[1])
    # CPU reference: correctly rounded FP32 divisions, as the kernel's.
    values = next_input.float().cpu()
    reference = values.abs().amax(1).clamp_min(1e-20) / 448.0
    assert torch.equal(scales.reshape(-1).cpu(), reference)
    expected = (values / reference[:, None]).to(torch.float8_e4m3fn)
    assert torch.equal(codes.cpu().view(torch.uint8), expected.view(torch.uint8))
    assert not codes[0].float().any()


def test_boundary_fp8_copy_matches_boundary_then_row_quantize() -> None:
    """The Triton boundary's in-launch FP8 copy (post_norm_residual_pre_norm_fp8)
    equals post_norm_residual_pre_norm followed by the registered row
    quantizer: the stream, both pre-norm copies, and the FP8 codes and scales,
    at the decode row counts the FP8 projections take; the split-K GEMM's output from either copy is the same.
    """
    from tokenspeed_kernel.ops.gemm import mm, prepare_fp8_channel_weight

    torch.manual_seed(47)
    post_weight = torch.randn(8192, device="cuda").bfloat16() * 0.1
    pre_weight = torch.randn_like(post_weight) * 0.1
    alpha, beta = residual_gate_coefficients(torch.randn_like(post_weight), 0.3, 1e-3)
    fp8_weight, weight_scales = prepare_fp8_channel_weight(
        (torch.randn(1024, 8192, device="cuda") * 0.02).bfloat16()
    )

    def project(codes, scales):
        return mm(
            codes,
            fp8_weight.t(),
            A_scales=scales,
            B_scales=weight_scales,
            out_dtype=torch.bfloat16,
            quant="fp8",
            override="triton_mm_fp8_channel_splitk",
        )

    def check(rows: int) -> None:
        stream = torch.randn(rows, 8192, device="cuda")
        branch = torch.randn_like(stream).bfloat16()
        # A zero row takes the scale floor instead of dividing by zero.
        branch[0].zero_()
        stream[0].zero_()
        args = (stream, branch, post_weight, 1e-6, 1.0, alpha, beta, pre_weight,
                1e-6, 1.0, False, None, 1)  # fmt: skip
        fused_stream, fused_fp32, fused_bf16, (codes, scales) = (
            post_norm_residual_pre_norm_fp8(*args)
        )
        new_stream, pre_fp32, pre_bf16 = post_norm_residual_pre_norm(*args)
        expected_codes, expected_scales = _quantize_rows_amax_div(pre_bf16)
        assert torch.equal(fused_stream, new_stream)
        assert torch.equal(fused_fp32, pre_fp32)
        assert torch.equal(fused_bf16, pre_bf16)
        assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))
        assert torch.equal(scales, expected_scales)
        assert torch.equal(
            project(codes, scales), project(expected_codes, expected_scales)
        )

    for rows in (17, 24, 32, 48, 64, 96, 128, 172):
        check(rows)
