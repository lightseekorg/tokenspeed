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
from tokenspeed_kernel.numerics.torch.gemm import torch_mm
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for the gfx950 decode GEMV tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.gemm import _select_mv, mm  # noqa: E402
from tokenspeed_kernel.ops.gemm.gluon import (  # noqa: E402
    gluon_gemv_a16w16_gfx950 as registered_gemv,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.gemv import (  # noqa: E402
    _PREFERRED_BUCKETS,
    gluon_gemv_a16w16_gfx950,
    launch_gluon_gemv_a16w16_gfx950,
    prefers_gluon_gemv_a16w16_gfx950,
)

# The kernel accumulates in FP32 and rounds once, so it differs from the FP32
# reference by at most one output ULP; atol covers FP32 summation-order noise
# on near-zero outputs.
_RTOL = {torch.bfloat16: 2**-7, torch.float16: 2**-10}
_ATOL = 1e-3

# (N, K): the register FMA tile (one chunk and several), the LDS-staged FMA
# tile, ragged N, a tiny N below one row per CU, and a long row whose MFMA
# tile splits K over eight waves (with a K tail).
_SHAPES = [
    (2560, 320),
    (2560, 3072),
    (512, 2560),
    (6656, 2560),
    (1000, 1040),
    (8, 64),
    (2304, 8192),
    (300, 8208),
]


def _reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch_mm(x.double(), w.double(), None, None, x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n,k", _SHAPES)
@pytest.mark.parametrize("m", [1, 2, 5, 16])
def test_gemv_matches_reference(m, n, k, dtype):
    torch.manual_seed(n + k + m)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    w = torch.randn(n, k, device="cuda", dtype=dtype) / k**0.5
    out = torch.full((m, n), float("nan"), device="cuda", dtype=dtype)
    launch_gluon_gemv_a16w16_gfx950(x, w, out)
    torch.testing.assert_close(out, _reference(x, w), rtol=_RTOL[dtype], atol=_ATOL)


@pytest.mark.parametrize("k", [32768, 40960])
def test_gemv_long_k_fits_lds(k):
    # The LDS-staged tile holds the whole activation row next to its weight
    # ring; at long K the ring must shrink so both fit gfx950's LDS. On 256
    # CUs (two rows per wave), K = 32768 keeps the full ring (64 KB row +
    # 8 x 8 KB ring) and K = 40960 shrinks it to fill the LDS exactly
    # (128 KB row + 4 x 8 KB ring).
    m, n = 1, 1280
    torch.manual_seed(k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    out = x.new_empty(m, n)
    launch_gluon_gemv_a16w16_gfx950(x, w, out)
    torch.testing.assert_close(
        out, _reference(x, w), rtol=_RTOL[torch.bfloat16], atol=_ATOL
    )


@pytest.mark.parametrize("m", [1, 5])
def test_gemv_ignores_memory_past_k_and_n(m):
    # NaN right after each activation row and after the weights: a masked K
    # tail that reads them instead of zeros turns outputs NaN. (1000, 2576)
    # has both a K tail and ragged N on the LDS-staged tile.
    n, k = 1000, 2576
    torch.manual_seed(m)
    x_rows = torch.full((m, k + 512), float("nan"), device="cuda", dtype=torch.bfloat16)
    x = x_rows[:, :k]
    x.copy_(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
    w_flat = torch.full(
        ((n + 1) * k,), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    w = w_flat[: n * k].view(n, k)
    w.copy_(torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5)
    out = x.new_empty(m, n)
    launch_gluon_gemv_a16w16_gfx950(x, w, out)
    torch.testing.assert_close(
        out, _reference(x, w), rtol=_RTOL[torch.bfloat16], atol=_ATOL
    )


def test_gemv_honors_activation_and_output_row_strides():
    m, n, k = 4, 2560, 3072
    x_rows = torch.randn(m, k + 64, device="cuda", dtype=torch.bfloat16)
    x = x_rows[:, :k]
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    out_rows = torch.zeros(m, n + 32, device="cuda", dtype=torch.bfloat16)
    launch_gluon_gemv_a16w16_gfx950(x, w, out_rows[:, :n])
    expected = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    launch_gluon_gemv_a16w16_gfx950(x.contiguous(), w, expected)
    assert torch.equal(out_rows[:, :n], expected)
    assert torch.count_nonzero(out_rows[:, n:]) == 0


def test_gemv_rejects_unsupported_shapes():
    w = torch.empty(64, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="does not support"):
        x = torch.empty(17, 128, device="cuda", dtype=torch.bfloat16)
        launch_gluon_gemv_a16w16_gfx950(x, w, x.new_empty(17, 64))
    with pytest.raises(ValueError, match="contiguous"):
        x = torch.empty(1, 64, device="cuda", dtype=torch.bfloat16)
        launch_gluon_gemv_a16w16_gfx950(x, w[:, :64], x.new_empty(1, 64))


def test_gemv_compiles_once_per_row_bucket():
    n, k = 2560, 3072
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

    def run(m):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        launch_gluon_gemv_a16w16_gfx950(x, w, x.new_empty(m, n))

    # Warm the M = 1 and M = 2..16 buckets.
    run(1)
    run(2)
    with assert_no_triton_compile(gluon_gemv_a16w16_gfx950):
        for m in range(1, 17):
            run(m)


@pytest.mark.parametrize("m", [1, 4, 16])
def test_mm_takes_the_gemv_only_at_its_measured_buckets(m):
    for n, k in ((2560, 3072), (6144, 2560), (12288, 2560)):
        selected = _select_mv(m, n, k, True, torch.bfloat16, torch.bfloat16, False)
        assert (selected is registered_gemv) is prefers_gluon_gemv_a16w16_gfx950(
            m, n, k
        ), (m, n, k)


# Row counts each M bucket of _PREFERRED_BUCKETS covers.
_BUCKET_ROWS = {1: range(1, 2), 4: range(2, 5), 16: range(5, 17)}


def test_selection_takes_the_gemv_on_every_listed_bucket():
    # A listed entry the GEMV cannot serve (say a K that is not a multiple of
    # 16) or a bucket other than 1, 4 or 16 would never be selected, silently
    # keeping the slower default.
    for (n, k), buckets in _PREFERRED_BUCKETS.items():
        for bucket in buckets:
            assert bucket in _BUCKET_ROWS, ((n, k), bucket)
            for m in _BUCKET_ROWS[bucket]:
                selected = _select_mv(
                    m, n, k, True, torch.bfloat16, torch.bfloat16, False
                )
                assert selected is registered_gemv, (m, n, k)


@pytest.mark.parametrize("m", [1, 4, 16])
def test_mm_routes_a_measured_shape_through_the_gemv(m):
    n, k = 1280, 2560
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    expected = x.new_empty(m, n)
    launch_gluon_gemv_a16w16_gfx950(x, w, expected)
    assert torch.equal(mm(x, w), expected)
