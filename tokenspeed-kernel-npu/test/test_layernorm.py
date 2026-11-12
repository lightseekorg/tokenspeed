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

from tokenspeed_kernel_npu.ops.layernorm import gated_rmsnorm, gemma_rmsnorm, rmsnorm

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend RMSNorm tests require an NPU"
)

EPS = 1e-6


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rmsnorm(dtype: torch.dtype) -> None:
    x = torch.randn(7, 1024, device="npu", dtype=dtype)
    weight = torch.randn(1024, device="npu", dtype=dtype)
    expected = _reference(x, weight, 1e-6)

    out = torch.empty_like(x)
    result = rmsnorm(x, weight, 1e-6, out=out)

    assert result is out
    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_add_rmsnorm_updates_residual(dtype: torch.dtype) -> None:
    x = torch.randn(7, 1024, device="npu", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(1024, device="npu", dtype=dtype)
    expected_residual = x + residual
    expected = _reference(expected_residual, weight, 1e-6)

    result, residual_out = rmsnorm(x, weight, 1e-6, residual=residual)

    assert residual_out is residual
    torch.testing.assert_close(residual, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


def _gemma_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    scale = 1.0 + weight.float()
    return (x_float * torch.rsqrt(variance + eps) * scale).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gemma_rmsnorm_scales_by_one_plus_weight(dtype: torch.dtype) -> None:
    x = torch.randn(7, 256, device="npu", dtype=dtype)
    weight = torch.randn(256, device="npu", dtype=dtype)
    expected = _gemma_reference(x, weight, EPS)

    result = gemma_rmsnorm(x, weight, EPS)

    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


def test_gemma_rmsnorm_updates_residual_and_normalizes_the_sum() -> None:
    x = torch.randn(7, 256, device="npu", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(256, device="npu", dtype=torch.bfloat16)
    expected_residual = x + residual
    expected = _gemma_reference(expected_residual, weight, EPS)

    result, residual_out = gemma_rmsnorm(x, weight, EPS, residual)

    assert residual_out is residual
    torch.testing.assert_close(residual, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


def _gated_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor | None,
    *,
    group_size: int,
    norm_before_gate: bool,
    sigmoid_gate: bool,
    eps: float,
) -> torch.Tensor:
    width = x.shape[-1]
    groups = width // group_size
    x_float = x.float().reshape(*x.shape[:-1], groups, group_size)
    gate = None
    if z is not None:
        gate_input = z.float().reshape(*x.shape[:-1], groups, group_size)
        gate = (
            torch.sigmoid(gate_input)
            if sigmoid_gate
            else gate_input * torch.sigmoid(gate_input)
        )
        if not norm_before_gate:
            x_float = x_float * gate
    variance = x_float.square().mean(dim=-1, keepdim=True)
    scale = weight.float().reshape(1 if weight.numel() == group_size else groups, -1)
    result = x_float * torch.rsqrt(variance + eps) * scale
    if gate is not None and norm_before_gate:
        result = result * gate
    return result.reshape(*x.shape).to(x.dtype)


@pytest.mark.parametrize("norm_before_gate", [True, False])
@pytest.mark.parametrize("sigmoid_gate", [True, False])
@pytest.mark.parametrize("group_size", [128, 64])
@pytest.mark.parametrize("shared_weight", [True, False])
def test_gated_rmsnorm_matches_the_gate_placement(
    group_size: int, shared_weight: bool, sigmoid_gate: bool, norm_before_gate: bool
) -> None:
    x = torch.randn(7, 128, device="npu", dtype=torch.bfloat16)
    z = torch.randn_like(x)
    weight = torch.randn(
        group_size if shared_weight else 128, device="npu", dtype=torch.bfloat16
    )
    expected = _gated_reference(
        x,
        weight,
        z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        sigmoid_gate=sigmoid_gate,
        eps=EPS,
    )

    result = gated_rmsnorm(
        x,
        weight,
        z,
        eps=EPS,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        sigmoid_gate=sigmoid_gate,
        fp8_scale=None,
        weights_independent=True,
    )

    torch.testing.assert_close(result, expected, atol=3e-2, rtol=3e-2)


def test_gated_rmsnorm_without_a_gate_is_a_grouped_norm() -> None:
    x = torch.randn(7, 128, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(128, device="npu", dtype=torch.bfloat16)
    expected = _gated_reference(
        x,
        weight,
        None,
        group_size=128,
        norm_before_gate=True,
        sigmoid_gate=False,
        eps=EPS,
    )

    result = gated_rmsnorm(
        x,
        weight,
        None,
        eps=EPS,
        group_size=128,
        norm_before_gate=True,
        sigmoid_gate=False,
        fp8_scale=None,
        weights_independent=True,
    )

    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


def test_gated_rmsnorm_rejects_an_fp8_output() -> None:
    x = torch.randn(7, 128, device="npu", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="FP8"):
        gated_rmsnorm(
            x,
            torch.ones(128, device="npu", dtype=torch.bfloat16),
            None,
            eps=EPS,
            group_size=128,
            norm_before_gate=True,
            sigmoid_gate=False,
            fp8_scale=torch.ones(1, device="npu"),
            weights_independent=True,
        )
