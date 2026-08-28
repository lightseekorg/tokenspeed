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

"""The vendored CuTe router GEMM, and its dense-linear form, against the paths
they displace."""

from typing import Literal

import pytest
import torch
from tokenspeed_kernel.ops.gemm import ll_bf16 as ll_bf16_ops
from tokenspeed_kernel.ops.gemm.kimi3 import (
    KIMI3_HIDDEN_SIZE,
    KIMI3_ROUTER_SIZE,
    kimi3_router_projection,
)
from tokenspeed_kernel.ops.gemm.ll_bf16 import (
    MAX_M,
    _declares_cute_dsl_backend,
    ll_bf16_mm,
    ll_bf16_mm_supported,
    ll_bf16_router_supported,
)
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


# Covers both backends: the dot product at M <= 4, split-K above it.
_MM_MS = [1, 4, 8, 32]


@pytest.mark.parametrize("m", _MM_MS)
def test_bf16_epilogue_rounds_once(m: int) -> None:
    """The epilogue converts from its FP32 accumulator, so no double rounding."""
    a, b = _inputs(m, seed=5)
    fp32 = ll_bf16_router(a, b)
    assert torch.equal(
        ll_bf16_router(a, b, out_dtype=torch.bfloat16), fp32.to(torch.bfloat16)
    )
    bias = torch.randn(KIMI3_ROUTER_SIZE, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(
        ll_bf16_router(a, b, bias=bias, out_dtype=torch.bfloat16),
        (fp32 + bias.float()).to(torch.bfloat16),
    )


@pytest.mark.parametrize("m", _MM_MS)
@pytest.mark.parametrize("with_bias", [False, True])
def test_mm_matches_linear(m: int, with_bias: bool) -> None:
    """Whichever path serves it: FlashInfer's backend or the vendored copy."""
    a, b = _inputs(m, seed=7)
    bias = (
        torch.randn(KIMI3_ROUTER_SIZE, device="cuda", dtype=torch.bfloat16)
        if with_bias
        else None
    )
    assert ll_bf16_mm_supported(a, b, bias)
    got = ll_bf16_mm(a, b, bias)
    assert got.dtype is torch.bfloat16
    torch.testing.assert_close(
        got.float(),
        torch.nn.functional.linear(
            a.float(), b.float(), None if bias is None else bias.float()
        ),
        atol=2e-2,
        rtol=2e-2,
    )


def test_flashinfer_probe_reads_the_declared_backends() -> None:
    """0.6.18 declares the backend; earlier wheels name every other one."""

    def upstreamed(backend: Literal["cudnn", "cute-dsl"] = "cudnn") -> None: ...

    def earlier(backend: Literal["cudnn", "tgv", "tinygemm"] = "cudnn") -> None: ...

    assert _declares_cute_dsl_backend(upstreamed)
    assert not _declares_cute_dsl_backend(earlier)
    assert not _declares_cute_dsl_backend(lambda: None)


def test_flashinfer_branch_meets_its_documented_requirement(monkeypatch) -> None:
    """Its guard wants row-major 2-D A, column-major B, and a contiguous [N] bias."""
    a, b = _inputs(4, seed=19)
    bias = torch.randn(KIMI3_ROUTER_SIZE, device="cuda", dtype=torch.bfloat16)

    def stub(a_arg, b_arg, *, bias, pdl, out, backend):
        assert backend == "cute-dsl"
        assert isinstance(pdl, bool)
        assert a_arg.ndim == 2 and a_arg.is_contiguous()
        assert tuple(b_arg.shape) == (KIMI3_HIDDEN_SIZE, KIMI3_ROUTER_SIZE)
        assert b_arg.T.is_contiguous()
        assert bias.shape == (KIMI3_ROUTER_SIZE,) and bias.is_contiguous()
        assert out is None or (out.dtype is torch.bfloat16 and out.is_contiguous())
        return ll_bf16_router(
            a_arg, b_arg.t(), out, bias=bias, out_dtype=torch.bfloat16
        )

    monkeypatch.setattr(ll_bf16_ops, "_flashinfer_mm_bf16", lambda: stub)
    torch.testing.assert_close(
        ll_bf16_mm(a, b, bias).float(),
        torch.nn.functional.linear(a.float(), b.float(), bias.float()),
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.parametrize(("lead", "m"), [((2, 2), 4), ((4, 8), 32)])
def test_flattens_leading_dims_and_honours_out(lead, m: int) -> None:
    a, b = _inputs(m, seed=11)
    x = a.view(*lead, KIMI3_HIDDEN_SIZE)
    out = torch.empty(*lead, KIMI3_ROUTER_SIZE, device="cuda", dtype=torch.bfloat16)
    returned = ll_bf16_mm(x, b, out=out)
    assert returned.shape == out.shape
    assert returned.data_ptr() == out.data_ptr()
    assert torch.equal(out.view(m, KIMI3_ROUTER_SIZE), ll_bf16_mm(a, b))


def test_mm_guard_declines_what_the_kernels_cannot_serve() -> None:
    a, b = _inputs(1, seed=13)
    # K must be a multiple of 128: both backends step K in those units.
    k_odd = KIMI3_HIDDEN_SIZE + 8
    assert not ll_bf16_mm_supported(
        torch.randn(1, k_odd, device="cuda", dtype=torch.bfloat16),
        torch.randn(KIMI3_ROUTER_SIZE, k_odd, device="cuda", dtype=torch.bfloat16),
    )
    big, big_w = _inputs(MAX_M + 1, seed=13)
    assert not ll_bf16_mm_supported(big, big_w)
    # Contiguous but offset 8 bytes into its storage, so vector loads misalign.
    buf = torch.randn(KIMI3_HIDDEN_SIZE + 16, device="cuda", dtype=torch.bfloat16)
    assert not ll_bf16_mm_supported(buf[4 : 4 + KIMI3_HIDDEN_SIZE].view(1, -1), b)
    assert not ll_bf16_mm_supported(a, b.unsqueeze(0))
    # Declined on the original tensor: a reshape would copy into something
    # contiguous and 32-byte aligned, voiding both checks.
    wide, wide_w = _inputs(2, seed=17)
    assert not ll_bf16_mm_supported(wide.t().contiguous().t(), wide_w)
    # Degenerate weight must be declined, not divided by.
    assert not ll_bf16_mm_supported(
        a, torch.zeros(KIMI3_ROUTER_SIZE, 0, device="cuda", dtype=torch.bfloat16)
    )
    assert not ll_bf16_mm_supported(
        a, b, torch.zeros(KIMI3_ROUTER_SIZE + 1, device="cuda", dtype=torch.bfloat16)
    )
    assert not ll_bf16_mm_supported(
        a, b, torch.zeros(KIMI3_ROUTER_SIZE, device="cuda", dtype=torch.float32)
    )
