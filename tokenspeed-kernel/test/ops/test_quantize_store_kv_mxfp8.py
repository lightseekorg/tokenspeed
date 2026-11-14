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
from utils import (
    assert_no_triton_compile,
    int_specialization_class,
    warm_specialization_classes,
)

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


def test_rejects_pages_that_are_not_a_multiple_of_128_tokens():
    """The scale slab is laid out per 128-token page; a 64-token page aliases scales."""
    rows = torch.zeros(4, 2, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(128, 2, 128, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.zeros(1, 2, 1, 32, 4, 4, device="cuda", dtype=torch.uint8)
    with pytest.raises(AssertionError):
        quantize_store_kv_mxfp8(
            rows, rows, cache, cache, scales, scales, torch.arange(4, device="cuda"), 64
        )


@pytest.mark.parametrize(
    "rows_shape,cache_shape",
    [
        ((4, 2, 64), (128, 2, 64)),
        ((4, 4, 128), (128, 2, 128)),
        ((4, 1, 256), (128, 2, 128)),
        ((4, 128), (128, 2, 64)),
        ((4, 512), (128, 2, 128)),
        ((4, 128), (128, 128)),
    ],
)
def test_rejects_rows_the_scale_layout_cannot_hold(rows_shape, cache_shape):
    """The scale-plane strides assume 128-wide heads, as many as the cache has."""
    rows = torch.zeros(rows_shape, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(cache_shape, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.zeros(1, 4, 1, 32, 4, 4, device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="128-wide heads"):
        quantize_store_kv_mxfp8(
            rows, rows, cache, cache, scales, scales, torch.arange(4, device="cuda")
        )


@pytest.mark.parametrize(
    "v_shape,v_cache_shape,v_cache_dtype",
    [
        ((4, 2, 128), (128, 1, 128), torch.float8_e4m3fn),
        ((4, 2, 128), (128, 2, 128), torch.uint8),
        ((4, 1, 128), (128, 2, 128), torch.float8_e4m3fn),
        ((4, 2, 128), (2, 2, 128), torch.float8_e4m3fn),
        ((2, 2, 128), (128, 2, 128), torch.float8_e4m3fn),
    ],
)
def test_rejects_value_rows_or_cache_unlike_the_keys(
    v_shape, v_cache_shape, v_cache_dtype
):
    """The kernel writes V with K's geometry."""
    k = torch.zeros(4, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros(v_shape, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.zeros(128, 2, 128, device="cuda", dtype=torch.float8_e4m3fn)
    v_cache = torch.zeros(v_cache_shape, device="cuda", dtype=v_cache_dtype)
    scales = torch.zeros(1, 2, 1, 32, 4, 4, device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="value rows and cache"):
        quantize_store_kv_mxfp8(
            k, v, k_cache, v_cache, scales, scales, torch.arange(4, device="cuda")
        )


def _dequant_reference(x: torch.Tensor) -> torch.Tensor:
    """BF16 value of each element after an MXFP8 round trip (flashinfer quantize)."""
    rows = x.reshape(-1, 128)
    data, sf = quantize_mxfp8(rows)
    exp = sf.view(torch.uint8).reshape(-1, 4).to(torch.float32) - 127.0
    scale = torch.exp2(exp).repeat_interleave(32, dim=1)
    return (
        (data.view(torch.float8_e4m3fn).float() * scale)
        .to(torch.bfloat16)
        .view(x.shape)
    )


@pytest.mark.parametrize(
    "nheads,page_tokens,kernel_page", [(2, 128, 128), (4, 256, 128), (2, 128, 64)]
)
def test_dequant_gather_matches_round_trip(nheads, page_tokens, kernel_page):
    from tokenspeed_kernel.ops.kvcache.triton import dequant_gather_kv_mxfp8

    torch.manual_seed(2)
    dev = "cuda"
    num_pages = 64  # kernel pages in the cache
    slots = num_pages * kernel_page
    data_shape = (slots, nheads, 128)
    sf_shape = (slots // page_tokens, nheads, page_tokens // 128, 32, 4, 4)
    k_cache = torch.full(data_shape, float("nan"), device=dev).to(torch.float8_e4m3fn)
    v_cache = torch.full(data_shape, float("nan"), device=dev).to(torch.float8_e4m3fn)
    k_sf = torch.full(sf_shape, 255, device=dev, dtype=torch.uint8)
    v_sf = torch.full(sf_shape, 255, device=dev, dtype=torch.uint8)

    seq_lens = [1, 3 * kernel_page + 5, 7 * kernel_page]
    width = 8
    page_table = torch.zeros(len(seq_lens), width, dtype=torch.int32, device=dev)
    perm = torch.randperm(num_pages, device=dev).to(torch.int32)
    k_ref = torch.zeros(len(seq_lens), width * kernel_page, nheads, 128, device=dev)
    v_ref = torch.zeros_like(k_ref)
    used = 0
    for b, n in enumerate(seq_lens):
        pages = -(-n // kernel_page)
        page_table[b, :pages] = perm[used : used + pages]
        used += pages
        k = torch.randn(n, nheads, 128, device=dev, dtype=torch.bfloat16) * 3
        v = torch.randn(n, nheads, 128, device=dev, dtype=torch.bfloat16) * 3
        k[0, 0, :32] = 0
        tok = torch.arange(n, device=dev)
        loc = page_table[b, tok // kernel_page].long() * kernel_page + tok % kernel_page
        quantize_store_kv_mxfp8(
            k, v, k_cache, v_cache, k_sf, v_sf, loc, page_tokens=page_tokens
        )
        k_ref[b, :n] = _dequant_reference(k).float()
        v_ref[b, :n] = _dequant_reference(v).float()

    # Request 1 reads only its last two pages (a sliding window); the others all.
    first_page = torch.tensor([0, 2, 0], dtype=torch.int32, device=dev)
    last_page = torch.tensor(
        [-(-n // kernel_page) for n in seq_lens], dtype=torch.int32, device=dev
    )
    num = last_page - first_page
    scratch = int(num.sum()) * kernel_page
    k_out = torch.full((scratch, nheads, 128), 7.0, device=dev, dtype=torch.bfloat16)
    v_out = torch.full_like(k_out, 7.0)
    out_pt = torch.full_like(page_table, -1)
    dequant_gather_kv_mxfp8(
        k_cache,
        v_cache,
        k_sf,
        v_sf,
        page_table,
        torch.tensor(seq_lens, dtype=torch.int32, device=dev),
        first_page,
        num,
        k_out,
        v_out,
        out_pt,
        kernel_page_size=kernel_page,
        page_tokens=page_tokens,
    )
    base = 0
    for b, n in enumerate(seq_lens):
        f, l = int(first_page[b]), int(last_page[b])
        expected = [base + i - f if f <= i < l else 0 for i in range(width)]
        assert out_pt[b].tolist() == expected
        base += l - f
        for i in range(f, l):
            got_k = k_out.view(-1, kernel_page, nheads, 128)[expected[i]]
            got_v = v_out.view(-1, kernel_page, nheads, 128)[expected[i]]
            rows = slice(i * kernel_page, (i + 1) * kernel_page)
            assert torch.equal(got_k.float(), k_ref[b, rows])
            assert torch.equal(got_v.float(), v_ref[b, rows])


def test_dequant_gather_compiles_once_per_layout():
    """Batch size, page-table width and scratch size are runtime arguments."""
    from tokenspeed_kernel.ops.kvcache.triton import (
        _dequant_gather_kv_mxfp8_kernel,
        dequant_gather_kv_mxfp8,
    )

    dev, nheads, page = "cuda", 2, 128
    slots = 64 * page
    k_cache = torch.zeros(slots, nheads, 128, device=dev, dtype=torch.float8_e4m3fn)
    v_cache = torch.zeros_like(k_cache)
    sf_shape = (slots // page, nheads, 1, 32, 4, 4)
    k_sf = torch.full(sf_shape, 127, device=dev, dtype=torch.uint8)
    v_sf = torch.full_like(k_sf, 127)

    def run(shape: tuple[int, int, int]) -> None:
        bs, width, capacity = shape
        page_table = torch.zeros(bs, width, dtype=torch.int32, device=dev)
        k_out = torch.empty(
            capacity * page, nheads, 128, device=dev, dtype=torch.bfloat16
        )
        dequant_gather_kv_mxfp8(
            k_cache,
            v_cache,
            k_sf,
            v_sf,
            page_table,
            torch.full((bs,), page, dtype=torch.int32, device=dev),
            torch.zeros(bs, dtype=torch.int32, device=dev),
            torch.ones(bs, dtype=torch.int32, device=dev),
            k_out,
            torch.empty_like(k_out),
            torch.empty_like(page_table),
            kernel_page_size=page,
            page_tokens=page,
        )

    # The width also sets both page-table strides.
    def key(shape):
        return tuple(int_specialization_class(value) for value in shape[1:])

    sweep = [
        (bs, width, capacity)
        for bs in (1, 3, 8)
        for width in (1, 5, 16, 33, 48)
        for capacity in (8, 16, 24, 32)
    ]
    warm_specialization_classes(run, key, sweep, pool=[])
    with assert_no_triton_compile(_dequant_gather_kv_mxfp8_kernel):
        for shape in sweep:
            run(shape)
