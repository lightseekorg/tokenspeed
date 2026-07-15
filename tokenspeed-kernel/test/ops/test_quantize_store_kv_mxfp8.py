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

"""Bit-parity of the fused MXFP8 quantize+store against the 5-launch path."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.triton import (
    quantize_store_kv_mxfp8,
    store_kv_cache,
    store_sf_interleaved,
)
from tokenspeed_kernel.ops.quantization import quantize_mxfp8

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU."
)


@pytest.mark.parametrize("nheads,page_tokens", [(4, 128), (2, 256)])
@pytest.mark.parametrize("t", [1, 7])
def test_bit_parity_with_split_path(nheads, page_tokens, t):
    torch.manual_seed(0)
    dev = "cuda"
    d = 128
    rows = 16 * page_tokens
    num_pages = rows // page_tokens
    k = torch.randn(t, nheads, d, device=dev, dtype=torch.bfloat16) * 4
    v = torch.randn(t, nheads, d, device=dev, dtype=torch.bfloat16) * 4
    k[0, 0, :32] = 0  # zero group: exponent must clamp, data must be 0
    loc = torch.randperm(rows, device=dev, dtype=torch.int64)[:t]

    def bufs():
        data = torch.zeros(rows, nheads, d, dtype=torch.float8_e4m3fn, device=dev)
        cpp = page_tokens // 128
        sf = torch.zeros(
            num_pages, nheads, cpp, 32, 4, 4, dtype=torch.float8_e8m0fnu, device=dev
        )
        return data, sf

    # Reference: the current 5-launch path.
    k_ref, k_sf_ref = bufs()
    v_ref, v_sf_ref = bufs()
    kq, ksf = quantize_mxfp8(k.reshape(t * nheads, d))
    vq, vsf = quantize_mxfp8(v.reshape(t * nheads, d))
    store_kv_cache(
        kq.view(t, -1).view(torch.uint8),
        vq.view(t, -1).view(torch.uint8),
        k_ref.view(rows, -1).view(torch.uint8),
        v_ref.view(rows, -1).view(torch.uint8),
        loc,
    )
    store_sf_interleaved(
        ksf.view(torch.float8_e8m0fnu).view(t, nheads, 4),
        k_sf_ref,
        loc,
        page_size=page_tokens,
    )
    store_sf_interleaved(
        vsf.view(torch.float8_e8m0fnu).view(t, nheads, 4),
        v_sf_ref,
        loc,
        page_size=page_tokens,
    )

    # Fused single launch.
    k_fus, k_sf_fus = bufs()
    v_fus, v_sf_fus = bufs()
    quantize_store_kv_mxfp8(
        k, v, k_fus, v_fus, k_sf_fus, v_sf_fus, loc, page_tokens=page_tokens
    )

    assert torch.equal(k_ref.view(torch.uint8), k_fus.view(torch.uint8))
    assert torch.equal(v_ref.view(torch.uint8), v_fus.view(torch.uint8))
    assert torch.equal(k_sf_ref.view(torch.uint8), k_sf_fus.view(torch.uint8))
    assert torch.equal(v_sf_ref.view(torch.uint8), v_sf_fus.view(torch.uint8))


@pytest.mark.parametrize("r", [1, 32, 500])
def test_rows_bit_parity_with_flashinfer(r):
    from tokenspeed_kernel.ops.kvcache.triton import quantize_mxfp8_rows

    torch.manual_seed(1)
    x = torch.randn(r, 128, device="cuda", dtype=torch.bfloat16) * 5
    if r > 1:
        x[1, :32] = 0
    ref_data, ref_sf = quantize_mxfp8(x)
    data, sf = quantize_mxfp8_rows(x)
    assert torch.equal(ref_data.view(torch.uint8), data.view(torch.uint8))
    assert torch.equal(ref_sf.view(torch.uint8).reshape(r, 4), sf.view(torch.uint8))
