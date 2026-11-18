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

"""Pinned small-N BF16 projection: dispatch, outputs and graph reuse."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import flashinfer as fi
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture
from utils import assert_no_cute_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or fi.platform.arch_version != fi.ArchVersion(10, 7),
    reason="small-N Split-K policy requires SM107",
)

NAME = "flashinfer_mm_bf16_small_n_splitk"


@pytest.fixture(autouse=True)
def pdl():
    previous = pdl_enabled()
    pdl_enabled(True)
    yield
    pdl_enabled(previous)


def inputs(m, n, k):
    generator = torch.Generator(device="cuda").manual_seed(39)
    x = torch.randn(m, k, device="cuda", generator=generator).bfloat16()
    w = (torch.randn(n, k, device="cuda", generator=generator) / k**0.5).bfloat16()
    return x, w


def routed(x, w, **kwargs):
    capture = ShapeCapture.get()
    previous = capture.enabled
    capture.clear()
    capture.enabled = True
    try:
        output = mm(x, w, **kwargs)
        return capture._records[-1].kernel_name, output
    finally:
        capture.enabled = previous
        capture.clear()


def check(output, x, w):
    # FP32 accumulation, one BF16 rounding; allow cancellation near zero.
    torch.testing.assert_close(
        output.double(), x.double() @ w.double().T, rtol=2**-7, atol=3e-5
    )


def test_automatic_dispatch_preserves_out_and_bias():
    x, w = inputs(4, 640, 2560)
    guard = torch.full((4 * 640 + 64,), 42, device="cuda", dtype=x.dtype)
    out = guard[: 4 * 640].view(4, 640)
    name, result = routed(x, w, out=out)
    assert name == NAME
    assert result is out
    check(result, x, w)
    assert torch.all(guard[4 * 640 :] == 42)
    bias = torch.linspace(-0.5, 0.5, 640, device="cuda", dtype=x.dtype)
    _, biased = routed(x, w, bias=bias)
    torch.testing.assert_close(biased, result + bias, atol=0, rtol=0)


@pytest.mark.parametrize("operand", ["x", "weight", "out"])
def test_unaligned_buffers_keep_the_existing_gemm(operand):
    x, w = inputs(4, 640, 2560)
    out = torch.empty((4, 640), device="cuda", dtype=x.dtype)

    def offset(tensor):
        storage = torch.empty(tensor.numel() + 1, device="cuda", dtype=tensor.dtype)
        view = storage[1:].view_as(tensor)
        view.copy_(tensor)
        return view

    if operand == "x":
        x = offset(x)
    elif operand == "weight":
        w = offset(w)
    else:
        out = offset(out)
    name, result = routed(x, w, out=out)
    assert name != NAME
    assert result is out
    # Compare the existing provider with the same out= alignment. cuBLAS may
    # choose different reduction tactics for an aligned, newly allocated result.
    expected = torch.empty_like(out)
    if operand == "out":
        expected = offset(expected)
    mm(x, w, out=expected, override="torch_mm")
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


@pytest.mark.parametrize("m,n,k", [(1, 640, 2560), (8, 640, 2560), (4, 768, 2048)])
def test_other_shapes_keep_existing_selection(m, n, k):
    x, w = inputs(m, n, k)
    name, result = routed(x, w)
    assert name != NAME
    torch.testing.assert_close(result, x @ w.T)


def test_explicit_existing_kernel_override_is_preserved():
    x, w = inputs(4, 640, 2560)
    name, result = routed(x, w, override="torch_mm")
    assert name == "torch_mm"
    torch.testing.assert_close(result, x @ w.T)


def test_graph_replay_reads_new_inputs():
    x, w = inputs(4, 640, 2560)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        for _ in range(3):
            mm(x, w)
        with torch.cuda.graph(graph, stream=stream):
            result = mm(x, w)
        with assert_no_cute_compile():
            for factor in (1.0, -0.5, 2.0):
                x.mul_(factor)
                graph.replay()
                check(result, x, w)
    stream.synchronize()


def test_provider_reuses_compilation_across_row_counts():
    # Registry admission is a performance policy; the implementation itself
    # supports other shapes and must not compile for each runtime row count.
    x, w = inputs(8, 768, 2048)
    kernel = fi.flashinfer_mm_bf16_small_n_splitk
    kernel(x, w, None, None, x.dtype, enable_pdl=True)
    with assert_no_cute_compile():
        for rows in (1, 2, 4, 8):
            result = kernel(x[:rows], w, None, None, x.dtype, enable_pdl=True)
            check(result, x[:rows], w)


def test_provider_rejects_unsupported_scaling():
    x, w = inputs(4, 640, 2560)
    with pytest.raises(ValueError, match="unscaled"):
        fi.flashinfer_mm_bf16_small_n_splitk(
            x,
            w,
            None,
            None,
            x.dtype,
            enable_pdl=True,
            alpha=torch.ones((), device="cuda"),
        )
