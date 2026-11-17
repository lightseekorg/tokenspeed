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

"""The GDN prefill decay gate against its FP32 definition."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _reference(A_log, a, dt_bias, beta=1.0, threshold=20.0):
    x = a.float() + dt_bias.float()
    softplus = torch.where(
        beta * x <= threshold, torch.log(1 + torch.exp(beta * x)) / beta, x
    )
    return -torch.exp(A_log.float()) * softplus


def _inputs(rows, heads, seed, *, width=None, dtype=torch.bfloat16):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    width = heads if width is None else width
    # A wide projection: the gate reads its last ``heads`` columns as a view.
    projection = (torch.randn(rows, width, device="cuda", generator=generator) * 4).to(
        dtype
    )
    a = projection[:, width - heads :]
    A_log = torch.randn(heads, device="cuda", generator=generator)
    dt_bias = torch.randn(heads, device="cuda", generator=generator)
    return A_log, a, dt_bias


@pytest.mark.parametrize("rows", [1, 3, 16, 17, 1000, 16384])
@pytest.mark.parametrize("heads", [12, 48, 64])
def test_gate_matches_fp32_definition(rows, heads):
    from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating

    A_log, a, dt_bias = _inputs(rows, heads, 7 + rows + heads, width=4 * heads)
    g = fused_gdn_gating(A_log, a, dt_bias)
    assert g.shape == (rows, heads) and g.dtype == torch.float32
    assert g.is_contiguous()
    # The kernel's exp/log are approximate; softplus of very negative inputs
    # keeps ~1e-4 relative accuracy.
    torch.testing.assert_close(g, _reference(A_log, a, dt_bias), rtol=5e-4, atol=2e-6)


def test_large_inputs_take_the_linear_branch():
    from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating

    A_log, a, dt_bias = _inputs(64, 48, 3)
    a.fill_(40.0)
    g = fused_gdn_gating(A_log, a, dt_bias)
    expected = -torch.exp(A_log) * (40.0 + dt_bias)
    torch.testing.assert_close(g, expected.expand_as(g), rtol=1e-6, atol=0)


def test_strided_view_matches_contiguous_copy_bitwise():
    from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating

    A_log, a, dt_bias = _inputs(4099, 48, 11, width=6 * 48)
    assert not a.is_contiguous()
    torch.testing.assert_close(
        fused_gdn_gating(A_log, a, dt_bias),
        fused_gdn_gating(A_log, a.contiguous(), dt_bias),
        rtol=0,
        atol=0,
    )


def test_row_offsets_past_int32():
    from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating

    # Rows 2**20 elements apart: the stride fits int32, the last row's offset
    # (2**31) does not. Only the gated elements of the ~4 GiB buffer are
    # written, so the test stays cheap.
    heads, stride = 48, 2**20
    rows = 2**31 // stride + 1
    buffer = torch.empty(2**31 + heads, device="cuda", dtype=torch.bfloat16)
    a = buffer.as_strided((rows, heads), (stride, 1))
    A_log, values, dt_bias = _inputs(rows, heads, 13)
    a.copy_(values)
    torch.testing.assert_close(
        fused_gdn_gating(A_log, a, dt_bias),
        _reference(A_log, a, dt_bias),
        rtol=5e-4,
        atol=2e-6,
    )


def test_row_counts_compile_no_new_specialization():
    from tokenspeed.runtime.layers.attention.linear.gdn import (
        fused_gdn_gating,
        fused_gdn_gating_kernel,
    )

    kernel_test_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..",
        "tokenspeed-kernel",
        "test",
    )
    helper_spec = importlib.util.spec_from_file_location(
        "gdn_gating_kernel_test_utils", os.path.join(kernel_test_dir, "utils.py")
    )
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)

    A_log, a, dt_bias = _inputs(4096, 48, 5, width=2 * 48)
    fused_gdn_gating(A_log, a[:16], dt_bias)
    with helper.assert_no_triton_compile(fused_gdn_gating_kernel):
        for rows in (1, 17, 31, 32, 4095, 4096):
            fused_gdn_gating(A_log, a[:rows], dt_bias)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
