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

"""Numerics of the monokernel's post-MoE bodies on gfx950, each as a test
kernel (monokernel_body_helpers.py) against the regular kernels or FP32."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from monokernel_body_helpers import (  # noqa: E402
    column_relay_kernel,
    sender_sums_kernel,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (  # noqa: E402
    launch_gluon_mm_a16w16_column_shard_gfx950,
    supports_gluon_mm_a16w16_column_shard_gfx950,
)


@pytest.mark.parametrize("rows", [8, 12, 16])
def test_column_relay_equals_the_column_shard_bitwise(rows: int) -> None:
    """The post-expert sender's wave relay runs each output's MFMA chain in
    the column shard GEMM's K order, so its columns are that GEMM's bit for
    bit (the post-expert projection: 1024 columns of K 4096 per rank)."""
    torch.manual_seed(rows)
    n, k = 1024, 4096
    assert supports_gluon_mm_a16w16_column_shard_gfx950(rows, 8 * n, n, k)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    expected = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    launch_gluon_mm_a16w16_column_shard_gfx950(x, w, expected)
    actual = torch.full_like(expected, float("nan"))
    column_relay_kernel[(n // 16,)](x, w, actual, rows, K=k, num_warps=8)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_sender_sums_match_the_four_norm_sums() -> None:
    """Each sender's tagged words hold, per row, the FFN boundary's four sums
    over its 16 columns (sum b^2, B, A, C, with b the BF16 post-expert
    output times the post-norm weight) and its tag; the post-expert output is
    the column shard GEMM's, bit for bit."""
    torch.manual_seed(0)
    rows, n, k, tag = 8, 1024, 4096, 0x5A5A
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    h = torch.randn(rows, n, device="cuda")
    alpha = torch.rand(n, device="cuda") + 0.5
    beta = torch.rand(n, device="cuda") + 0.5
    post = (torch.rand(n, device="cuda") + 0.5).to(torch.bfloat16)
    sums = torch.zeros(n // 16, 16, 4, device="cuda", dtype=torch.int64)
    sender_sums_kernel[(n // 16,)](
        x, w, h, alpha, beta, post, sums, tag, rows, K=k, N=n, num_warps=8
    )
    y = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    launch_gluon_mm_a16w16_column_shard_gfx950(x, w, y)
    b = y.double() * post.double()
    ah = alpha.double() * h.double()
    bb = beta.double() * b
    terms = torch.stack([b * b, ah * bb, ah * ah, bb * bb], dim=-1)
    expected = terms.view(rows, n // 16, 16, 4).sum(dim=2).transpose(0, 1)
    words = sums[:, :rows]
    assert torch.all((words >> 32) == tag)
    actual = (words & 0xFFFFFFFF).to(torch.int32).view(torch.float32).double()
    # Each term is formed in FP32 (a few roundings) and 16 of them add in FP32,
    # so the error is below 2^-19 of the sum of the terms' magnitudes.
    bound = terms.abs().view(rows, n // 16, 16, 4).sum(dim=2).transpose(0, 1)
    assert torch.all((actual - expected).abs() <= bound * 2**-19)
