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

"""Row scheduling preserves the original reduction and native graph updates."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu._triton import triton
from tokenspeed_kernel_npu.ops.dsv41_pointwise import _reference_norm, reference_rmsnorm


def original(x, weight, eps):
    result = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    n = x.shape[-1]
    rows = x.reshape(-1, n) if x.ndim != 2 else x
    if rows.numel():
        _reference_norm[(rows.shape[0],)](
            rows,
            weight,
            result,
            rows.stride(0),
            n,
            n,
            eps,
            triton.next_power_of_2(n),
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return result


@pytest.mark.parametrize(
    "rows,width,strided",
    [
        (0, 512, False),
        (1, 5120, False),
        (8, 1280, True),
        (31, 512, True),
        (32, 512, True),
        (48, 512, True),
        (49, 512, True),
        (127, 512, False),
        (128, 512, False),
        (192, 512, False),
        (768, 512, False),
        (48, 1280, True),
        (49, 1280, True),
        (48, 5120, False),
        (33, 5120, True),
        (48, 1024, False),
    ],
)
def test_exact_replay(rows, width, strided):
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(112)
    backing = torch.randn(
        rows, width + (512 if strided else 0), device="npu", dtype=torch.bfloat16
    )
    x = backing[:, :width]
    weight = torch.randn(width, device="npu", dtype=torch.bfloat16)
    out = torch.empty((rows, width), device="npu", dtype=x.dtype)
    reference_rmsnorm(x, weight, 1e-6, out)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        actual = reference_rmsnorm(x, weight, 1e-6, out)
    assert actual.data_ptr() == out.data_ptr()
    for scale in (0.0, 1e-3, 1.0, 30.0):
        backing.copy_(torch.randn_like(backing) * scale)
        weight.copy_(torch.randn_like(weight))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, original(x, weight, 1e-6), atol=0, rtol=0)
        cpu = x.cpu().double()
        expected = (
            weight.cpu().double()
            * (cpu * torch.rsqrt(cpu.square().mean(-1, keepdim=True) + 1e-6))
        ).bfloat16()
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-7, rtol=0.0079)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adversarial_strides_and_rounding(dtype):
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    for seed in range(16):
        torch.manual_seed(seed)
        storage = torch.randn(193, 544, device="npu", dtype=dtype)
        x = storage[1:, 16:528]
        weight = torch.randn(512, device="npu", dtype=dtype)
        for variant in range(3):
            if variant == 1:
                x[:, 1::2].neg_()
                x[0].zero_()
            elif variant == 2:
                x[:, 0] = 64
            actual = reference_rmsnorm(x, weight, 1e-6, None)
            torch.testing.assert_close(
                actual, original(x, weight, 1e-6), atol=0, rtol=0
            )
