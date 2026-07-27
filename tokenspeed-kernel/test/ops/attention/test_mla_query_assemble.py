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

"""Parity for the fused NoPE MLA query assembly + fp8 quantization."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.triton.mla_query_assemble import (
    mla_nope_query_fp8,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize("H", [16, 128])
def test_matches_cat_cast(T, H):
    """Bitwise vs torch cat + cast; q_pe as the strided slice the absorb
    path produces (a column window of the packed q projection)."""
    torch.manual_seed(T * 100 + H)
    nope_dim, pe_dim = 512, 64
    q_nope = torch.randn(T, H, nope_dim, dtype=torch.bfloat16, device="cuda") * 8
    q_full = torch.randn(T, H, 128 + pe_dim, dtype=torch.bfloat16, device="cuda") * 8
    q_pe = q_full[..., 128:]  # strided slice, unit inner stride

    got = mla_nope_query_fp8(q_nope, q_pe)

    ref = torch.cat([q_nope, q_pe], dim=-1).to(torch.float8_e4m3fn)
    assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))


def test_writes_into_out():
    torch.manual_seed(0)
    q_nope = torch.randn(1, 16, 512, dtype=torch.bfloat16, device="cuda")
    q_pe = torch.randn(1, 16, 64, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(1, 16, 576, dtype=torch.float8_e4m3fn, device="cuda")
    got = mla_nope_query_fp8(q_nope, q_pe, out=out)
    assert got.data_ptr() == out.data_ptr()
    ref = torch.cat([q_nope, q_pe], dim=-1).to(torch.float8_e4m3fn)
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))
