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
from tokenspeed_kernel.ops.attention import tokenspeed_mla as kernel_mla
from tokenspeed_kernel.ops.attention.tokenspeed_mla.fallback import (
    mla_kv_pack_quantize_fp8,
)
from tokenspeed_kernel.platform import current_platform


@pytest.mark.parametrize("k_pe_ndim", [2, 3])
@pytest.mark.parametrize("preallocate", [False, True])
def test_mla_kv_pack_quantize_fp8_fallback(
    device: str, k_pe_ndim: int, preallocate: bool
) -> None:
    torch.manual_seed(0)
    seq_len, num_heads, qk_nope, qk_rope, v_head = 7, 4, 8, 4, 8
    packed_kv = torch.randn(
        seq_len,
        num_heads,
        qk_nope + v_head,
        dtype=torch.bfloat16,
        device=device,
    )
    k_nope = packed_kv[..., :qk_nope]
    v = packed_kv[..., qk_nope:]
    k_pe = torch.randn(seq_len, 1, qk_rope, dtype=torch.bfloat16, device=device)
    if k_pe_ndim == 2:
        k_pe = k_pe.squeeze(1)

    k_scale_inv, v_scale_inv = 0.5, 1.7
    expected_k_pe = k_pe.unsqueeze(1) if k_pe.ndim == 2 else k_pe
    expected_k_pe = expected_k_pe.expand(-1, num_heads, -1)
    expected_k = (
        torch.cat((k_nope, expected_k_pe), dim=-1)
        .float()
        .mul_(k_scale_inv)
        .to(torch.float8_e4m3fn)
    )
    expected_v = v.float().mul_(v_scale_inv).to(torch.float8_e4m3fn)
    k_out = torch.empty_like(expected_k) if preallocate else None
    v_out = torch.empty_like(expected_v) if preallocate else None

    actual_k, actual_v = mla_kv_pack_quantize_fp8(
        k_nope,
        k_pe,
        v,
        k_scale_inv=k_scale_inv,
        v_scale_inv=v_scale_inv,
        k_out=k_out,
        v_out=v_out,
        enable_pdl=True,
    )

    if preallocate:
        assert k_out is not None
        assert v_out is not None
        assert actual_k.data_ptr() == k_out.data_ptr()
        assert actual_v.data_ptr() == v_out.data_ptr()
    assert torch.equal(actual_k.view(torch.uint8), expected_k.view(torch.uint8))
    assert torch.equal(actual_v.view(torch.uint8), expected_v.view(torch.uint8))


def test_non_nvidia_public_api_uses_fallback() -> None:
    if current_platform().is_nvidia or current_platform().is_cdna4:
        pytest.skip("this platform uses an optimized MLA pack implementation")

    assert kernel_mla.mla_kv_pack_quantize_fp8 is mla_kv_pack_quantize_fp8


def test_gfx950_public_api_uses_gluon() -> None:
    if not current_platform().is_cdna4:
        pytest.skip("gfx950 is required")

    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.kv_pack import (
        gluon_mla_kv_pack_quantize_fp8_gfx950,
    )

    assert kernel_mla.mla_kv_pack_quantize_fp8 is gluon_mla_kv_pack_quantize_fp8_gfx950
