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

"""Each PDL-enabled kernel must call ``gdc_wait()`` before it reads anything an
upstream kernel wrote, since ``launch_pdl=True`` lets its CTAs be scheduled
before that upstream kernel retires.

Every case below races a synthetic slow producer against the kernel under
test on the same stream, with no host sync in between. Like the production
kernels, the producer releases its dependents at its start, then spins before
copying the real input over a poisoned buffer. The dependent therefore starts
while the producer is still spinning; only its own ``gdc_wait()``, which returns
after the producer has finished and its writes are visible, keeps it from
reading the poison.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.quantization.triton import _quantize_fp8_group32_swizzled
from tokenspeed_kernel.platform import pdl_enabled

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DEVICE = "cuda"
# Longer than the host takes to launch the kernel under test after the
# producer, including its wrapper's allocations, so a kernel missing its wait
# starts while the producer still spins and reads the poison.
_DELAY_NS = 500_000


@pytest.fixture(autouse=True)
def _require_pdl():
    if not pdl_enabled():
        pytest.skip("PDL is not enabled on this platform")


# ---------------------------------------------------------------------------
# Synthetic slow producer: release dependents, delay, then publish real bytes.
# ---------------------------------------------------------------------------


@triton.jit
def _pdl_spin_wait_ns(delay_ns: tl.constexpr):
    # A self-contained PTX busy loop on %globaltimer (nanoseconds since boot).
    # Written entirely in inline asm so the compiler cannot hoist, skip, or
    # shorten it: there is no data dependency for it to optimize away.
    tl.inline_asm_elementwise(
        f"""
        {{
            .reg .u64 %t0, %t1, %delta;
            .reg .pred %spin;
            mov.u64 %t0, %globaltimer;
        spin_loop:
            mov.u64 %t1, %globaltimer;
            sub.u64 %delta, %t1, %t0;
            setp.lt.u64 %spin, %delta, {delay_ns};
            @%spin bra spin_loop;
            mov.s32 $0, 0;
        }}
        """,
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _pdl_race_producer_kernel(
    src_ptr,
    dst_ptr,
    n_bytes,
    delay_ns: tl.constexpr,
    block: tl.constexpr,
):
    # The source comes from an ordinary kernel, so wait for it like any PDL
    # kernel, then let the kernel under test launch during the delay.
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n_bytes
    _pdl_spin_wait_ns(delay_ns)
    values = tl.load(src_ptr + offs, mask=mask, other=0)
    tl.store(dst_ptr + offs, values, mask=mask)


def _poison_like(real: torch.Tensor) -> torch.Tensor:
    """A same-shape buffer whose raw bytes differ from ``real``'s values."""
    poisoned = torch.empty_like(real)
    poisoned.view(torch.uint8).fill_(0xAB)
    return poisoned


def _publish_with_delay(real: torch.Tensor, dst: torch.Tensor, delay_ns: int) -> None:
    """Copy ``real``'s bytes into ``dst`` after a delay, letting the next
    kernel launch before the copy lands.
    """
    assert real.shape == dst.shape and real.dtype == dst.dtype
    assert real.is_contiguous() and dst.is_contiguous()
    src_bytes = real.view(torch.uint8).reshape(-1)
    dst_bytes = dst.view(torch.uint8).reshape(-1)
    n = src_bytes.numel()
    block = 1024
    grid = (triton.cdiv(n, block),)
    _pdl_race_producer_kernel[grid](
        src_bytes,
        dst_bytes,
        n,
        delay_ns,
        block,
        num_warps=4,
        launch_pdl=True,
    )


def _assert_tuples_close(actual, expected) -> None:
    if not isinstance(actual, tuple):
        actual = (actual,)
    if not isinstance(expected, tuple):
        expected = (expected,)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(
            a.view(torch.uint8), e.view(torch.uint8), rtol=0, atol=0
        )


# ---------------------------------------------------------------------------
# One case per PDL-patched kernel: (real tensor to race, run(raced) -> tuple).
# run() launches only the kernel under test, so nothing separates the producer
# from it.
# ---------------------------------------------------------------------------


def _case_quantize_fp8_group32_swizzled():
    torch.manual_seed(105)
    rows, hidden = 128, 5120
    real = torch.randn(rows, hidden, dtype=torch.bfloat16, device=DEVICE)

    def run(x):
        return _quantize_fp8_group32_swizzled(x)

    return real, run


_CASES = {
    "fp8_group32_ue8m0_quantize_swizzled": _case_quantize_fp8_group32_swizzled,
}


@pytest.mark.parametrize("case_name", list(_CASES))
def test_pdl_kernel_waits_for_upstream_write(case_name):
    real, run = _CASES[case_name]()

    # Reference first: it also compiles the kernel, so the raced launch below
    # reaches the GPU while the producer is still spinning.
    expected = run(real.clone())
    torch.cuda.synchronize()

    # Race path: no host sync between the producer and the kernel under test.
    race_buf = _poison_like(real)
    _publish_with_delay(real, race_buf, _DELAY_NS)
    actual = run(race_buf)
    torch.cuda.synchronize()

    _assert_tuples_close(actual, expected)


@pytest.mark.parametrize("case_name", list(_CASES))
def test_pdl_kernel_waits_for_upstream_write_under_cuda_graph(case_name):
    real, run = _CASES[case_name]()

    # CUDA graphs require a warm-up launch outside the capture on buffers
    # that are not reused by the capture itself.
    warmup_real = real.clone()
    warmup_buf = _poison_like(warmup_real)
    _publish_with_delay(warmup_real, warmup_buf, _DELAY_NS)
    run(warmup_buf)
    torch.cuda.synchronize()

    source = real.clone()
    race_buf = _poison_like(real)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        race_buf.view(torch.uint8).fill_(0xAB)
        _publish_with_delay(source, race_buf, _DELAY_NS)
        captured = run(race_buf)

    for _ in range(3):
        source.normal_()
        graph.replay()
        torch.cuda.synchronize()
        actual = tuple(tensor.clone() for tensor in captured)

        expected = run(source.clone())
        torch.cuda.synchronize()

        _assert_tuples_close(actual, expected)
