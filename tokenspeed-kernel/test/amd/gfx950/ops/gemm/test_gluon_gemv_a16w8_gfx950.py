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
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for the gfx950 FP8-weight GEMV tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.gemm import mv_fp8_weight  # noqa: E402
from tokenspeed_kernel.ops.gemm.native import torch_mv_fp8_weight  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.gemv import (  # noqa: E402
    gluon_gemv_a16w8_gfx950,
    launch_gluon_gemv_a16w8_gfx950,
)

# FP32 sums of exact products, one scale multiply and one rounding: within one
# output ULP of the FP64 reference; atol covers summation-order noise on
# near-zero outputs.
_RTOL = {torch.bfloat16: 2**-7, torch.float16: 2**-10}
_ATOL = 1e-3

# (N, K): a draft MLP's gate/up and down at TP8, an LM-head shard (several
# programs a CU), ragged N, a K tail inside the last iteration, and a tiny N.
_SHAPES = [
    (7168, 8192),
    (8192, 3584),
    (25256, 8192),
    (1000, 1040),
    (300, 8208),
    (8, 64),
]


def _quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = w.float().abs().amax(dim=1) / 448.0
    return (w.float() / scale[:, None]).to(torch.float8_e4m3fn), scale


def _reference(x, q, scale):
    exact = x.double() @ (q.double() * scale.double()[:, None]).T
    return exact.to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n,k", _SHAPES)
@pytest.mark.parametrize("m", [1, 7, 24, 64, 128])
def test_gemv_matches_reference(m, n, k, dtype):
    torch.manual_seed(n + k + m)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    q, scale = _quantize(torch.randn(n, k, device="cuda") / k**0.5)
    out = torch.full((m, n), float("nan"), device="cuda", dtype=dtype)
    launch_gluon_gemv_a16w8_gfx950(x, q, scale, out)
    torch.testing.assert_close(
        out, _reference(x, q, scale), rtol=_RTOL[dtype], atol=_ATOL
    )


def test_gemv_ignores_memory_past_k_and_n():
    # NaN right after each activation row and after the weights: a masked K
    # tail or ragged-N lane that reads them instead of zeros turns outputs
    # NaN. (1000, 2576) has both.
    m, n, k = 8, 1000, 2576
    torch.manual_seed(m)
    x_rows = torch.full((m, k + 512), float("nan"), device="cuda", dtype=torch.bfloat16)
    x = x_rows[:, :k]
    x.copy_(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
    q_src, scale = _quantize(torch.randn(n, k, device="cuda") / k**0.5)
    q_flat = torch.full(((n + 1) * k,), float("nan"), device="cuda").to(
        torch.float8_e4m3fn
    )
    q = q_flat[: n * k].view(n, k)
    q.copy_(q_src)
    out = x.new_empty(m, n)
    launch_gluon_gemv_a16w8_gfx950(x, q, scale, out)
    torch.testing.assert_close(
        out, _reference(x, q, scale), rtol=_RTOL[torch.bfloat16], atol=_ATOL
    )


def test_gemv_honors_activation_and_output_row_strides():
    m, n, k = 8, 7168, 8192
    x_rows = torch.randn(m, k + 64, device="cuda", dtype=torch.bfloat16)
    q, scale = _quantize(torch.randn(n, k, device="cuda") / k**0.5)
    out_rows = torch.zeros(m, n + 32, device="cuda", dtype=torch.bfloat16)
    launch_gluon_gemv_a16w8_gfx950(x_rows[:, :k], q, scale, out_rows[:, :n])
    expected = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    launch_gluon_gemv_a16w8_gfx950(x_rows[:, :k].contiguous(), q, scale, expected)
    assert torch.equal(out_rows[:, :n], expected)
    assert torch.count_nonzero(out_rows[:, n:]) == 0


def test_gemv_rows_are_independent_within_a_row_bucket():
    # The summation order depends only on the row bucket's constexprs, so a
    # row's bits do not depend on how many other rows share the launch.
    n, k = 7168, 8192
    q, scale = _quantize(torch.randn(n, k, device="cuda") / k**0.5)
    x = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
    full = x.new_empty(16, n)
    launch_gluon_gemv_a16w8_gfx950(x, q, scale, full)
    part = x.new_empty(9, n)
    launch_gluon_gemv_a16w8_gfx950(x[:9], q, scale, part)
    assert torch.equal(full[:9], part)


def test_gemv_rejects_unsupported_inputs():
    q = torch.zeros(64, 128, device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.ones(64, device="cuda")
    x = torch.empty(129, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="does not support"):
        launch_gluon_gemv_a16w8_gfx950(x, q, scale, x.new_empty(129, 64))
    x = torch.empty(4, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        launch_gluon_gemv_a16w8_gfx950(
            x, q.view(torch.uint8), scale, x.new_empty(4, 64)
        )
    with pytest.raises(ValueError, match="weight_scale"):
        launch_gluon_gemv_a16w8_gfx950(x, q, scale[:32], x.new_empty(4, 64))


def test_gemv_compiles_once_per_row_bucket():
    n, k = 8192, 3584
    q, scale = _quantize(torch.randn(n, k, device="cuda") / k**0.5)

    def run(m):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        launch_gluon_gemv_a16w8_gfx950(x, q, scale, x.new_empty(m, n))

    for m in (16, 32):
        run(m)
    with assert_no_triton_compile(gluon_gemv_a16w8_gfx950):
        for m in range(1, 33):
            run(m)


@pytest.mark.parametrize("m", [8, 129])
def test_mv_fp8_weight_matches_the_reference_at_any_row_count(m):
    # Decode rows take the GEMV; rows past it take the chunked Torch path.
    n, k = 2560, 1024
    torch.manual_seed(m)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    q, scale = _quantize(torch.randn(n, k, device="cuda") / k**0.5)
    out = mv_fp8_weight(x, q, scale)
    expected = torch_mv_fp8_weight(x, q, scale, x.new_empty(m, n))
    torch.testing.assert_close(out, expected, rtol=_RTOL[torch.bfloat16], atol=_ATOL)
    torch.testing.assert_close(
        out, _reference(x, q, scale), rtol=_RTOL[torch.bfloat16], atol=_ATOL
    )
