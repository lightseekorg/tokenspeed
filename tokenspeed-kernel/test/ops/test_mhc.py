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
from tokenspeed_kernel.numerics.torch.residual import torch_mhc_pre
from tokenspeed_kernel.ops.residual import mhc_init
from tokenspeed_kernel.ops.residual.triton import (
    _mhc_init_kernel,
    triton_mhc_pre,
)
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_tiled_hc4_prefill_matches_reference() -> None:
    generator = torch.Generator(device="cuda").manual_seed(123)
    residual = torch.randn(
        (257, 4, 64), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    fn = torch.randn((24, 256), device="cuda", dtype=torch.float32, generator=generator)
    hc_scale = torch.tensor([0.7, 1.1, 0.5], device="cuda", dtype=torch.float32)
    hc_base = torch.randn(24, device="cuda", dtype=torch.float32, generator=generator)
    args = (residual, fn, hc_scale, hc_base, 1e-6, 1e-5, 3)

    actual = triton_mhc_pre(*args, norm_weight=None, norm_eps=None)
    expected = torch_mhc_pre(*args, None, None)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor.float(), expected_tensor.float(), rtol=2e-2, atol=2e-2
        )


def _check_mhc_init(hidden, actual, hc_mult):
    residual, pre = actual
    expected = hidden[:, None, :].repeat(1, hc_mult, 1)
    expected_pre = torch.zeros(hidden.shape[0], hc_mult, device=hidden.device)
    expected_pre[:, 0] = 1
    torch.testing.assert_close(residual, expected, rtol=0, atol=0)
    torch.testing.assert_close(pre, expected_pre, rtol=0, atol=0)
    assert residual.is_contiguous() and pre.is_contiguous()
    # Streams must be independent, including when the source was an expanded view.
    if hidden.shape[0] and hc_mult > 1:
        untouched = residual[:, 1].clone()
        residual[:, 0].zero_()
        torch.testing.assert_close(residual[:, 1], untouched, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [0, 6, 12, 24, 48, 96, 192, 2049])
@pytest.mark.parametrize("width", [129, 5120])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("hc_mult", [1, 4])
def test_mhc_init(rows, width, dtype, hc_mult):
    hidden = torch.randn(rows, width, device="cuda", dtype=dtype)
    _check_mhc_init(hidden, mhc_init(hidden, hc_mult), hc_mult)


@pytest.mark.parametrize("layout", ["transpose", "slice", "expand"])
def test_mhc_init_strided(layout):
    if layout == "transpose":
        hidden = torch.randn(5120, 24, device="cuda", dtype=torch.bfloat16).T
    elif layout == "slice":
        hidden = torch.randn(48, 10240, device="cuda", dtype=torch.bfloat16)[::2, ::2]
    else:
        hidden = torch.randn(1, 5120, device="cuda", dtype=torch.bfloat16).expand(
            24, -1
        )
    _check_mhc_init(hidden, mhc_init(hidden, 4), 4)


def test_mhc_init_large_inner_stride():
    stride = 1 << 30
    elements = 4 * stride + 1
    needed = elements * torch.bfloat16.itemsize
    if torch.cuda.mem_get_info()[0] < needed + (256 << 20):
        pytest.skip("requires 8 GiB of free device memory")
    storage = torch.empty(elements, device="cuda", dtype=torch.bfloat16)
    # Keep wrapped addresses inside the allocation so overflow fails by value.
    storage[0] = -17
    hidden = storage.as_strided((1, 3), (0, stride), storage_offset=2 * stride)
    hidden.copy_(torch.tensor([[1, 2, 3]], device="cuda", dtype=hidden.dtype))
    _check_mhc_init(hidden, mhc_init(hidden, 4), 4)


def test_mhc_init_runtime_rows_and_strides_do_not_compile():
    # Warm once, then vary row count and both embedding strides.
    hidden = torch.randn(6, 5120, device="cuda", dtype=torch.bfloat16)
    mhc_init(hidden, 4)
    with assert_no_triton_compile(_mhc_init_kernel):
        for rows in [12, 24, 48, 96, 192]:
            hidden = torch.randn(rows * 2, 10240, device="cuda", dtype=torch.bfloat16)[
                ::2, ::2
            ]
            _check_mhc_init(hidden, mhc_init(hidden, 4), 4)


def test_mhc_init_graph_replay_then_eager():
    hidden = torch.randn(24, 5120, device="cuda", dtype=torch.bfloat16)
    mhc_init(hidden, 4)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mhc_init(hidden, 4)
    hidden.normal_()
    graph.replay()
    # Eager outputs survive another graph replay and own independent storage.
    eager = mhc_init(hidden, 4)
    expected = hidden.clone()
    hidden.normal_()
    graph.replay()
    _check_mhc_init(expected, eager, 4)
    _check_mhc_init(hidden, captured, 4)
