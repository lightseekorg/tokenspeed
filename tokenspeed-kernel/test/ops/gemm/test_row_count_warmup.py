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

"""Row-count warm-ups launch every kernel serving launches at a later count."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import (
    dsv4_linear_fp32,
    grouped_bf16_projection,
    grouped_bf16_projection_quantize_fp8,
    warmup_dsv4_linear_fp32,
    warmup_grouped_bf16_projection,
)
from utils import assert_warmup_launches_cover

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

MAX_TOKENS = 300
# Small-M kernel bounds and their neighbours, library GEMM tiles, the bound.
SWEEP = (1, 2, 4, 5, 16, 17, 32, 33, 100, 255, 256, 257, MAX_TOKENS)


@pytest.mark.parametrize("experts", [384, 1024])
def test_dsv4_linear_fp32_warmup_covers_every_count(experts):
    torch.manual_seed(experts)
    weight = torch.randn(experts, 5120, device="cuda", dtype=torch.bfloat16)
    inputs = {
        tokens: torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
        for tokens in SWEEP
    }
    assert_warmup_launches_cover(
        lambda: warmup_dsv4_linear_fp32(weight, torch.bfloat16, MAX_TOKENS),
        lambda tokens: dsv4_linear_fp32(inputs[tokens], weight, None, None),
        SWEEP,
    )


@pytest.mark.parametrize("quantize_fp8", [False, True])
@pytest.mark.parametrize("kernel_heads", [8, 16])
def test_grouped_bf16_projection_warmup_covers_every_count(quantize_fp8, kernel_heads):
    torch.manual_seed(kernel_heads)
    groups, live_heads, head_dim = 2, 8, 256
    weight = torch.randn(groups, 384, live_heads * head_dim // groups, device="cuda")
    weight = weight.to(torch.bfloat16)
    # A slice of the padded kernel heads, as the attention output reaches wo_a.
    inputs = {
        tokens: torch.randn(
            tokens, kernel_heads, head_dim, device="cuda", dtype=torch.bfloat16
        )[:, :live_heads].reshape(tokens, groups, -1)
        for tokens in SWEEP
    }
    project = (
        (lambda x: grouped_bf16_projection_quantize_fp8(x, weight))
        if quantize_fp8
        else (lambda x: grouped_bf16_projection(x, weight, None, None))
    )
    assert_warmup_launches_cover(
        lambda: warmup_grouped_bf16_projection(
            weight, kernel_heads * head_dim, quantize_fp8, MAX_TOKENS
        ),
        lambda tokens: project(inputs[tokens]),
        SWEEP,
    )


def test_grouped_bf16_projection_warmup_rejects_overlapping_tokens():
    weight = torch.zeros(2, 64, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="token_stride"):
        warmup_grouped_bf16_projection(weight, 128, False, 4)
