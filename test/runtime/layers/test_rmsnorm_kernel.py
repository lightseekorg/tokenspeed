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

"""The original RMSNorm module with an initialization-selected callable."""

import pytest
import torch
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.selection import kernel_override
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)


@pytest.fixture(autouse=True)
def norm_registry(monkeypatch):
    # Keep registrations local to these CPU module tests.
    monkeypatch.setattr(KernelRegistry, "_instance", KernelRegistry())


@pytest.mark.parametrize("rows", [0, 1, 17])
@pytest.mark.parametrize("has_residual", [False, True])
def test_rmsnorm_selected_kernel_preserves_arguments_and_output(rows, has_residual):
    from tokenspeed_kernel.selection import select_kernel

    from tokenspeed.runtime.layers.layernorm import RMSNorm

    def reference_norm(x, weight, eps, *, residual, out):
        merged = x if residual is None else x + residual
        result = torch.nn.functional.rms_norm(merged, weight.shape, weight, eps)
        if out is not None:
            out.copy_(result)
            result = out
        return result if residual is None else (result, merged)

    KernelRegistry.get().register(
        KernelSpec(
            name="reference_standalone_norm",
            family="layernorm",
            mode="rmsnorm",
            solution="reference",
            format_signatures=format_signatures(
                ("x", "weight"), "dense", {torch.float32}
            ),
        ),
        reference_norm,
    )
    kernel = select_kernel(
        "layernorm",
        "rmsnorm",
        format_signature(
            x=dense_tensor_format(torch.float32),
            weight=dense_tensor_format(torch.float32),
        ),
        solution="reference",
    )
    layer = RMSNorm(16, eps=1e-5, kernel=kernel)
    x = torch.randn(rows, 16)
    weight = torch.randn(16)
    layer.weight.data.copy_(weight)
    residual = torch.randn_like(x) if has_residual else None
    original_residual = residual.clone() if has_residual else None
    merged = x + residual if has_residual else x
    expected = torch.nn.functional.rms_norm(merged, (16,), weight, 1e-5)
    out = torch.empty_like(x)
    # A later override cannot replace a callable already selected for this module.
    with kernel_override("layernorm", "rmsnorm", "missing_norm"):
        actual = layer(x, residual=residual, out=out)
    if has_residual:
        actual, updated = actual
        torch.testing.assert_close(updated, merged, rtol=0, atol=0)
        torch.testing.assert_close(residual, original_residual, rtol=0, atol=0)
    assert actual is (out if rows else x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="all-reduce fusion"):
        layer.forward_with_allreduce_fusion(0, (0,), x, residual)
    with pytest.raises(ValueError, match="reduce-scatter fusion"):
        layer.forward_with_reducescatter_fusion(0, (0,), x, residual)
