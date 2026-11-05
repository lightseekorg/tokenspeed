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

"""Dense large-M warp pipeline projection coverage on gfx1250."""

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna5

if not is_cdna5():
    pytest.skip("AMD CDNA5 is required", allow_module_level=True)

from tokenspeed_kernel import compile_monitor
from tokenspeed_kernel.ops.gemm import _try_gluon_largem_gfx1250
from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
    _LARGEM_ROWS,
    _wmma_tdm_dense_kernel,
    _wmma_tdm_dense_largem_kernel,
    _wmma_tdm_dense_largem_warp_pipeline_kernel,
    gluon_mm_a16w16_gfx1250,
    gluon_mm_a16w16_largem_gfx1250,
    use_gluon_largem_gfx1250,
)


def _project(a, b):
    """Run into a NaN-filled buffer and check nothing past the output moved."""
    m, n = a.shape[0], b.shape[0]
    storage = torch.full((m * n + 4096,), float("nan"), device="cuda", dtype=a.dtype)
    out = storage[: m * n].view(m, n)
    actual = gluon_mm_a16w16_largem_gfx1250(a, b, out=out)
    assert actual.data_ptr() == out.data_ptr()
    assert torch.isnan(storage[m * n :]).all()
    return actual


# Ragged M and N edges (2112 = 8.25 tiles), an odd K-tile count (4224 / 128),
# and full tiles. The last case is a ragged M edge whose last row of 2x2
# clusters holds one valid row, leaving half of those CTAs without rows, at a
# size the MI450 simulator lane can run.
@pytest.mark.parametrize(
    "m,n,k",
    [(4097, 2112, 7168), (4352, 7168, 4224), (4096, 3072, 512), (4097, 3072, 512)],
)
def test_largem_warp_pipeline_matches_torch(m, n, k):
    torch.manual_seed(m + n + k)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    actual = _project(a, b)
    expected = (a.float() @ b.float().T).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_largem_warp_pipeline_row_counts_reuse_compilation():
    torch.manual_seed(1250)
    k, n = 1536, 2304
    a = torch.randn(8192, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    _project(a[:4096], b)
    with assert_no_triton_compile(_wmma_tdm_dense_largem_warp_pipeline_kernel):
        for rows in (4097, 6096, 8144, 8192):
            actual = _project(a[:rows], b)
            torch.testing.assert_close(actual, a[:rows] @ b.T, atol=1e-2, rtol=1e-2)


def test_largem_warp_pipeline_compiles_under_compile_monitor():
    # Serving installs the compile monitor, whose JIT hooks serialize every
    # launch argument; the kernel must compile with them in place.
    torch.manual_seed(7)
    a = torch.randn(4096, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(3072, 512, device="cuda", dtype=torch.bfloat16) / 512**0.5
    _wmma_tdm_dense_largem_warp_pipeline_kernel.device_caches.clear()
    compile_monitor.install_compile_monitor("warn")
    try:
        actual = _project(a, b)
    finally:
        compile_monitor.uninstall_compile_monitor()
    torch.testing.assert_close(actual, a @ b.T, atol=1e-2, rtol=1e-2)


# Serving projections (M, K, N): a K3 projection from its first row, a range's
# first row and the row before it, a closed range's last row and a row past
# it, a gap and a later range on one weight, and weights with no range.
@pytest.mark.parametrize(
    "m,k,n,selected",
    [
        (512, 1536, 7168, True),  # K3 o_proj
        (511, 1536, 7168, False),
        (848, 4096, 6416, True),  # GLM KDA q/k/v/beta/f_a/g_a
        (768, 4096, 6416, False),
        (8192, 4096, 288, True),  # GLM MoE router
        (12288, 4096, 288, False),
        (1793, 8192, 4096, False),
        (7424, 8192, 4096, True),
        (8192, 21504, 896, False),  # K3 EAGLE3 fc
        (8192, 4096 + 64, 4096, False),  # K must be a multiple of 128
    ],
)
def test_largem_selection_follows_row_ranges(m, k, n, selected):
    assert use_gluon_largem_gfx1250(m, k, n) is selected


def test_largem_table_is_well_formed():
    # The kernels need K in whole 128-wide tiles, and each weight's ranges
    # ascend without overlap.
    for (k, _), ranges in _LARGEM_ROWS.items():
        assert k % 128 == 0
        previous_high = 0
        for index, (low, high) in enumerate(ranges):
            assert previous_high < low
            if high is None:
                assert index == len(ranges) - 1
            else:
                assert low <= high
                previous_high = high


def _cold_kernel(kernel):
    kernel.device_caches.clear()
    return kernel


# Narrow outputs: one wave of 64 x 32 (router at 848 rows) or 64 x 64 (wk,
# KPool gate) dense tiles, and the 128 x 128 large-M tile when no dense tile
# divides N within one wave (router at 8192 rows).
@pytest.mark.parametrize(
    "m,n,k,kernel",
    [
        (848, 288, 4096, _wmma_tdm_dense_kernel),
        (5377, 128, 4096, _wmma_tdm_dense_kernel),
        (8192, 288, 4096, _wmma_tdm_dense_largem_kernel),
    ],
)
def test_largem_narrow_outputs_match_torch(m, n, k, kernel):
    torch.manual_seed(m + n)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    _cold_kernel(kernel)
    actual = _project(a, b)
    assert kernel.device_caches
    expected = (a.float() @ b.float().T).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_largem_warp_pipeline_peels_column_tail():
    # The columns past the last 512-wide cluster column would add a CU round
    # to the warp pipeline grid, so 128 x 64 large-M tiles cover them.
    m, n, k = 8192, 2304, 1536
    torch.manual_seed(n + k)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    _cold_kernel(_wmma_tdm_dense_largem_warp_pipeline_kernel)
    _cold_kernel(_wmma_tdm_dense_largem_kernel)
    actual = _project(a, b)
    assert _wmma_tdm_dense_largem_warp_pipeline_kernel.device_caches
    assert _wmma_tdm_dense_largem_kernel.device_caches
    expected = (a.float() @ b.float().T).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_largem_row_strided_a_matches_torch():
    # KDA f_b/g_b read their 128-wide input columns out of the fused
    # q/k/v/beta/f_a/g_a projection rows.
    m, k, n = 8192, 128, 2048
    torch.manual_seed(6416)
    projected = torch.randn(m, 6416, device="cuda", dtype=torch.bfloat16)
    g_a = projected[:, 6288:]
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    _cold_kernel(_wmma_tdm_dense_largem_warp_pipeline_kernel)
    actual = _try_gluon_largem_gfx1250(g_a, b)
    assert actual is not None
    assert _wmma_tdm_dense_largem_warp_pipeline_kernel.device_caches
    torch.testing.assert_close(actual, g_a @ b.T, atol=1e-2, rtol=1e-2)
    # A row stride or offset off 16 bytes stays on torch.
    assert _try_gluon_largem_gfx1250(projected[:, 6287:6415], b) is None
    odd = torch.empty(m, k + 4, device="cuda", dtype=torch.bfloat16)[:, :k]
    assert _try_gluon_largem_gfx1250(odd, b) is None
    with pytest.raises(ValueError, match="16-byte"):
        gluon_mm_a16w16_largem_gfx1250(odd, b)


def test_largem_warp_pipeline_fp32_output_row_strided_a():
    # The general BF16 GEMM reaches the warp pipeline kernel too: FP32 output and
    # activations with a row stride wider than K.
    m, n, k = 4160, 1536, 1024
    torch.manual_seed(4160)
    storage = torch.randn(m, 2 * k, device="cuda", dtype=torch.bfloat16)
    a = storage[:, k:]
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    _wmma_tdm_dense_largem_warp_pipeline_kernel.device_caches.clear()
    actual = gluon_mm_a16w16_gfx1250(a, b, torch.float32)
    # The FP32-output specialization was compiled, so this shape ran here.
    assert _wmma_tdm_dense_largem_warp_pipeline_kernel.device_caches
    assert actual.dtype == torch.float32
    expected = a.double() @ b.double().T
    torch.testing.assert_close(actual.double(), expected, atol=2e-4, rtol=2e-4)
