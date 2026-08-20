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

"""The vendored CuTe dot-product router GEMM against the paths it displaces."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm.kimi3 import (
    KIMI3_HIDDEN_SIZE,
    KIMI3_ROUTER_SIZE,
    kimi3_router_projection,
)
from tokenspeed_kernel.ops.gemm.ll_bf16 import MAX_M, ll_bf16_router_supported
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cute_dsl.ll_bf16 import ll_bf16_router

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
# Split-K needs SM90 clusters; the "cuda" reference solution is Hopper-plus too.
if not (current_platform().is_nvidia and current_platform().is_hopper_plus):
    pytest.skip(
        "ll_bf16 router GEMM needs an NVIDIA Hopper-plus GPU", allow_module_level=True
    )
if not ll_bf16_router.is_available():
    pytest.skip("CuTe DSL not installed", allow_module_level=True)


def _inputs(m: int, seed: int = 0):
    torch.manual_seed(seed)
    a = torch.randn(m, KIMI3_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(
        KIMI3_ROUTER_SIZE, KIMI3_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    return a, b


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32])
def test_matches_the_cuda_router_kernel(m: int) -> None:
    """FP32 accumulation, so it must agree with the kernel it replaces."""
    a, b = _inputs(m)
    got = kimi3_router_projection(a, b, solution="ll_bf16")
    expected = kimi3_router_projection(a, b, solution="cuda")
    torch.testing.assert_close(got, expected, atol=2e-3, rtol=2e-3)
    reference = torch.nn.functional.linear(a.float(), b.float())
    torch.testing.assert_close(got, reference, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("m", [1, 8, 32])
def test_auto_selects_it_and_honours_out(m: int) -> None:
    a, b = _inputs(m, seed=3)
    out = torch.empty(m, KIMI3_ROUTER_SIZE, device="cuda", dtype=torch.float32)
    returned = kimi3_router_projection(a, b, out=out, solution="auto")
    assert returned.data_ptr() == out.data_ptr()
    assert torch.equal(out, kimi3_router_projection(a, b, solution="ll_bf16"))


def test_declines_above_its_measured_range() -> None:
    """Past MAX_M the driver must refuse rather than serve a slower path."""
    a, b = _inputs(MAX_M + 8)
    assert not ll_bf16_router_supported(a, b, a.shape[0])
    with pytest.raises(ValueError):
        kimi3_router_projection(a, b, solution="ll_bf16")
    # ``auto`` still answers, through cublas.
    kimi3_router_projection(a, b, solution="auto")


def test_declines_non_contiguous_and_odd_k() -> None:
    a, b = _inputs(1)
    assert not ll_bf16_router_supported(a.expand(2, -1), b, 2)
    odd = torch.randn(1, KIMI3_HIDDEN_SIZE + 1, device="cuda", dtype=torch.bfloat16)
    odd_w = torch.randn(
        KIMI3_ROUTER_SIZE, KIMI3_HIDDEN_SIZE + 1, device="cuda", dtype=torch.bfloat16
    )
    assert not ll_bf16_router_supported(odd, odd_w, 1)
