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

"""Once serving, the joint BF16 GEMM yields to the caller's other GEMM instead of compiling."""

from dataclasses import astuple

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel import compile_monitor
from tokenspeed_kernel.ops.gemm import flashinfer as fi
from tokenspeed_kernel.ops.gemm.triton_gemv import (
    _rowcta_gemv_kernel,
    decode_gemv,
    use_decode_gemv,
)
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not fi.has_flashinfer_cute_dsl_bf16(),
    reason="the joint BF16 GEMM needs FlashInfer's cute-dsl backend on SM100/SM103",
)

# A shape no other test compiles, so this test's first compile is its own.
N, K = 1056, 1152


@pytest.fixture
def warp_splitk(monkeypatch):
    """Pin FI's choice to warp split-K, which compiles once per exact M."""
    module = pytest.importorskip("flashinfer.gemm.kernels.dense_bf16_gemm_warp_splitk")
    tactic = astuple(module.autotune_tactics(fi.BF16_GEMM_MAX_M, N, K)[0])

    def choose_one(custom_op, runners, tuning_config, inputs, **kwargs):
        runner = next(
            (r for r in runners if type(r).__name__ == "CuteDSLWarpSplitKBf16Runner"),
            None,
        )
        if runner is None:
            pytest.skip("this FlashInfer build omits the warp split-K runner")
        return runner, tactic

    # On the singleton: undoing an earlier instance patch leaves a bound method that shadows the class.
    monkeypatch.setattr(fi._fi_gemm.AutoTuner.get(), "choose_one", choose_one)
    monkeypatch.setattr(compile_monitor, "_hooks", None)
    monkeypatch.setattr(compile_monitor, "_serving", False)
    return module


def test_serving_routes_new_row_counts_away_from_per_m_compiles(warp_splitk):
    weight = (torch.randn(N, K, device="cuda") * 0.05).to(torch.bfloat16)
    x = torch.randn(fi.BF16_GEMM_MAX_M, K, device="cuda").to(torch.bfloat16)

    misses = warp_splitk._compile.cache_info().misses
    decode_gemv(x[:24], weight)
    assert warp_splitk._compile.cache_info().misses == misses + 1

    compile_monitor.mark_serving()
    misses = warp_splitk._compile.cache_info().misses
    # Nor the registry's M == 1 Triton GEMV, which startup never compiled for this K.
    with assert_no_triton_compile(_rowcta_gemv_kernel):
        for m in (1, 5, 17, 23):
            rows = x[:m]
            assert not use_decode_gemv(rows, weight)
            expected = (rows.float() @ weight.float().T).to(torch.bfloat16)
            for got in (decode_gemv(rows, weight), tokenspeed_kernel.mm(rows, weight)):
                torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
    assert warp_splitk._compile.cache_info().misses == misses
