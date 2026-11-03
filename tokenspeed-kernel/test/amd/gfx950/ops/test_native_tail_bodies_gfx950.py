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

"""Numerics of the native decoder layer's tail bodies on gfx950, each as a
test kernel (native_body_helpers.py) against today's kernels or FP32."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from native_body_helpers import column_relay_kernel  # noqa: E402
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
