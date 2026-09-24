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
from tokenspeed_kernel import dsv4_linear_fp32
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.thirdparty.cute_dsl.ll_bf16 import ll_bf16_router


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA GPU")
@pytest.mark.parametrize(
    "tokens,hidden,experts",
    [
        (1, 3072, 256),
        (4, 5120, 384),
        (5, 5120, 384),
        (16, 5120, 384),
        (17, 5120, 384),
        (32, 7168, 896),
    ],
)
@pytest.mark.parametrize("pdl", [False, True])
def test_shared_router_accuracy_and_graph(tokens, hidden, experts, pdl):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires SM90 or newer")
    if not ll_bf16_router.is_available():
        pytest.skip("requires CuTe DSL")
    original_pdl = pdl_enabled()
    pdl_enabled(pdl)
    try:
        torch.manual_seed(31)
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        w = (torch.randn(experts, hidden, device="cuda") / hidden**0.5).bfloat16()
        kwargs = dict(override="cute_dsl_dsv4_linear_fp32", solution=None)
        actual = dsv4_linear_fp32(x, w, **kwargs)
        expected = (x.double() @ w.double().T).float()
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
        # Ordinary dispatch must reach the same implementation.
        torch.testing.assert_close(
            dsv4_linear_fp32(x, w, override=None, solution=None), actual, rtol=0, atol=0
        )
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            captured = dsv4_linear_fp32(x, w, **kwargs)
        x.normal_()
        g.replay()
        expected = (x.double() @ w.double().T).float()
        torch.testing.assert_close(captured, expected, rtol=3e-5, atol=3e-5)
    finally:
        pdl_enabled(original_pdl)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA GPU")
@pytest.mark.parametrize(
    "tokens,weight_dtype", [(33, torch.bfloat16), (8, torch.float32)]
)
def test_unsupported_inputs_keep_cuda_dispatch(tokens, weight_dtype):
    x = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(384, 5120, device="cuda", dtype=weight_dtype)
    expected = dsv4_linear_fp32(
        x, w, override="cuda_dsv3_dsv4_linear_fp32", solution=None
    )
    actual = dsv4_linear_fp32(x, w, override=None, solution=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA GPU")
def test_missing_cute_dependency_keeps_cuda_fallback(monkeypatch):
    x = torch.randn(16, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(384, 5120, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(ll_bf16_router, "is_available", lambda: False)
    expected = dsv4_linear_fp32(
        x, w, override="cuda_dsv3_dsv4_linear_fp32", solution=None
    )
    actual = dsv4_linear_fp32(x, w, override=None, solution=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
