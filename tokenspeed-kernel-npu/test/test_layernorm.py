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

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.layernorm import rmsnorm

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend RMSNorm tests require an NPU"
)


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_rmsnorm(dtype: torch.dtype, weight_dtype: torch.dtype) -> None:
    x = torch.randn(7, 1024, device="npu", dtype=dtype)
    weight = torch.randn(1024, device="npu", dtype=weight_dtype)
    expected = _reference(x, weight, 1e-6)

    out = torch.empty_like(x)
    result = rmsnorm(x, weight, 1e-6, out=out)

    assert result is out
    assert result.dtype == dtype
    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_add_rmsnorm_updates_residual(
    dtype: torch.dtype, weight_dtype: torch.dtype
) -> None:
    x = torch.randn(7, 1024, device="npu", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(1024, device="npu", dtype=weight_dtype)
    expected_residual = x + residual
    expected = _reference(expected_residual, weight, 1e-6)

    result, residual_out = rmsnorm(x, weight, 1e-6, residual=residual)

    assert residual_out is residual
    assert result.dtype == dtype
    torch.testing.assert_close(residual, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


def test_bf16_output_retains_fp32_scale_until_multiplication():
    x = torch.tensor([1.0, 2.0], dtype=torch.bfloat16, device="npu").repeat(1, 512)
    scale = torch.full((1024,), 1.0038, dtype=torch.float32, device="npu")
    expected = _reference(x, scale, 1e-6)
    assert not torch.equal(expected, _reference(x, scale.bfloat16(), 1e-6))
    torch.testing.assert_close(rmsnorm(x, scale, 1e-6), expected, atol=0, rtol=0)


def test_fp32_scale_graph_replay() -> None:
    x = torch.randn(7, 1024, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(1024, device="npu", dtype=torch.float32)
    rmsnorm(x, weight, 1e-6)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = rmsnorm(x, weight, 1e-6)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.npu.synchronize()
        assert result.dtype == x.dtype
        torch.testing.assert_close(
            result, _reference(x, weight, 1e-6), atol=2e-2, rtol=2e-2
        )
