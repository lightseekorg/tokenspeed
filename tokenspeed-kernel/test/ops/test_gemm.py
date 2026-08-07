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
import tokenspeed_kernel
import torch


def test_mm_rejects_bad_out_layout() -> None:
    a = torch.empty((4, 8), dtype=torch.bfloat16)
    b = torch.empty((16, 8), dtype=torch.bfloat16)
    out = torch.empty((16, 4), dtype=torch.bfloat16).transpose(0, 1)

    with pytest.raises(ValueError, match=r"stride\(-1\) == 1"):
        tokenspeed_kernel.mm(a, b, out=out)


def test_mm_reference_rejects_out_dtype_mismatch() -> None:
    a = torch.empty((4, 8), dtype=torch.float32)
    b = torch.empty((16, 8), dtype=torch.float32)
    out = torch.empty((4, 16), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="torch_mm out= requires out_dtype"):
        tokenspeed_kernel.mm(a, b, out=out, override="torch_mm")


def test_bmm_rejects_batch_mismatch() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((3, 16, 8), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="batch mismatch"):
        tokenspeed_kernel.bmm(a, b)


def test_bmm_rejects_rank2_weights() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((16, 8), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=r"B with shape \[B, N, K\]"):
        tokenspeed_kernel.bmm(a, b)


def test_bmm_rejects_bad_out_layout() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((2, 16, 8), dtype=torch.bfloat16)
    out = torch.empty((2, 16, 4), dtype=torch.bfloat16).transpose(1, 2)

    with pytest.raises(ValueError, match=r"stride\(-1\) == 1"):
        tokenspeed_kernel.bmm(a, b, out=out)


def test_bmm_reference_rejects_out_dtype_mismatch() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.float32)
    b = torch.empty((2, 16, 8), dtype=torch.float32)
    out = torch.empty((2, 4, 16), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="torch_bmm out= requires out_dtype"):
        tokenspeed_kernel.bmm(a, b, out=out, override="torch_bmm")


def test_bmm_writes_head_major_strided_output(device: str) -> None:
    heads, tokens, k, n = 3, 1, 8, 16
    a = torch.randn(heads, tokens, k, device=device, dtype=torch.bfloat16)
    weight = torch.randn(heads, k, n, device=device, dtype=torch.bfloat16)
    backing = torch.empty(tokens, heads, n + 4, device=device, dtype=torch.bfloat16)
    out = backing[..., :n].transpose(0, 1)

    returned = tokenspeed_kernel.bmm(
        a,
        weight.transpose(1, 2),
        out=out,
        override="torch_bmm",
    )

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, torch.bmm(a, weight), atol=0, rtol=0)


def test_gluon_bmm_writes_head_major_strided_output(device: str, require) -> None:
    require("gemm", "bmm", "gluon", torch.bfloat16, "a")
    heads, tokens, k, n = 12, 1, 128, 512
    a_backing = torch.randn(tokens, heads, k, device=device, dtype=torch.bfloat16)
    a = a_backing.transpose(0, 1)
    weight = torch.randn(heads, k, n, device=device, dtype=torch.bfloat16)
    backing = torch.empty(tokens, heads, n + 64, device=device, dtype=torch.bfloat16)
    out = backing[..., :n].transpose(0, 1)

    returned = tokenspeed_kernel.bmm(
        a,
        weight.transpose(1, 2),
        out=out,
        override="gluon_bmm_a16w16_gfx950",
    )

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, torch.bmm(a, weight), atol=0.25, rtol=0.01)


def test_gluon_bmm_allocates_output(device: str, require) -> None:
    require("gemm", "bmm", "gluon", torch.bfloat16, "a")
    a = torch.randn(12, 1, 128, device=device, dtype=torch.bfloat16)
    weight = torch.randn(12, 128, 512, device=device, dtype=torch.bfloat16)

    output = tokenspeed_kernel.bmm(
        a,
        weight.transpose(1, 2),
        override="gluon_bmm_a16w16_gfx950",
    )

    torch.testing.assert_close(output, torch.bmm(a, weight), atol=0.25, rtol=0.01)


def test_gluon_bmm_falls_back_for_fp32_output(device: str, require) -> None:
    require("gemm", "bmm", "gluon", torch.bfloat16, "a")
    a = torch.randn(12, 1, 128, device=device, dtype=torch.bfloat16)
    weight = torch.randn(12, 128, 512, device=device, dtype=torch.bfloat16)

    output = tokenspeed_kernel.bmm(a, weight.transpose(1, 2), out_dtype=torch.float32)

    assert output.dtype == torch.float32


def test_decode_gemv_writes_preallocated_output() -> None:
    from tokenspeed_kernel.ops.gemm.triton_gemv import decode_gemv

    x = torch.randn(2, 8)
    weight = torch.randn(4, 8)
    out = torch.empty(2, 4)

    returned = decode_gemv(x, weight, out=out)

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, x @ weight.t())
