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
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_cdna4,
    reason="gfx950 is required",
)


def _reference(k_nope, k_pe, v, k_scale_inv, v_scale_inv, fp8_dtype):
    if k_pe.ndim == 2:
        k_pe = k_pe.unsqueeze(1)
    k_pe = k_pe.expand(-1, k_nope.shape[1], -1)
    return (
        (torch.cat((k_nope, k_pe), dim=-1).float() * k_scale_inv).to(fp8_dtype),
        (v.float() * v_scale_inv).to(fp8_dtype),
    )


@pytest.mark.parametrize("seq_len", [0, 7, 256, 2048])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("k_pe_ndim", [2, 3])
def test_gluon_mla_kv_pack_quantize_fp8(
    seq_len: int, fp8_dtype: torch.dtype, k_pe_ndim: int
) -> None:
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.kv_pack import (
        gluon_mla_kv_pack_quantize_fp8_gfx950,
    )

    torch.manual_seed(0)
    heads, qk_nope, qk_rope, v_head = 4, 128, 64, 128
    packed = torch.randn(
        seq_len,
        heads,
        qk_nope + v_head,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_nope = packed[..., :qk_nope]
    v = packed[..., qk_nope:]
    k_pe = torch.randn(seq_len, 1, qk_rope, device="cuda", dtype=torch.bfloat16)
    if k_pe_ndim == 2:
        k_pe = k_pe.squeeze(1)
    k_scale_inv, v_scale_inv = 0.5, 1.7
    expected_k, expected_v = _reference(
        k_nope, k_pe, v, k_scale_inv, v_scale_inv, fp8_dtype
    )
    k_out = torch.empty_like(expected_k)
    v_out = torch.empty_like(expected_v)

    actual_k, actual_v = gluon_mla_kv_pack_quantize_fp8_gfx950(
        k_nope,
        k_pe,
        v,
        k_scale_inv=k_scale_inv,
        v_scale_inv=v_scale_inv,
        k_out=k_out,
        v_out=v_out,
        fp8_dtype=fp8_dtype,
    )
    torch.cuda.synchronize()

    assert actual_k.data_ptr() == k_out.data_ptr()
    assert actual_v.data_ptr() == v_out.data_ptr()
    assert torch.equal(actual_k.view(torch.uint8), expected_k.view(torch.uint8))
    assert torch.equal(actual_v.view(torch.uint8), expected_v.view(torch.uint8))


def test_gluon_mla_kv_pack_production_heads() -> None:
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.kv_pack import (
        gluon_mla_kv_pack_quantize_fp8_gfx950,
    )

    torch.manual_seed(1)
    seq_len, heads, qk_nope, qk_rope, v_head = 4096, 16, 128, 64, 128
    packed = torch.randn(
        seq_len,
        heads,
        qk_nope + v_head,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_nope = packed[..., :qk_nope]
    v = packed[..., qk_nope:]
    k_pe = torch.randn(seq_len, 1, qk_rope, device="cuda", dtype=torch.bfloat16)
    expected_k, expected_v = _reference(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn)

    actual_k, actual_v = gluon_mla_kv_pack_quantize_fp8_gfx950(k_nope, k_pe, v)
    torch.cuda.synchronize()

    assert torch.equal(actual_k.view(torch.uint8), expected_k.view(torch.uint8))
    assert torch.equal(actual_v.view(torch.uint8), expected_v.view(torch.uint8))
