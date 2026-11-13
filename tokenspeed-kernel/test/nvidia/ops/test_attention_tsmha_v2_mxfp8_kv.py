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

"""FP32-output relative decode reading an MXFP8 K/V cache directly.

The FA4 decode kernel dequantizes the cache's FP8 pages to BF16 in shared
memory, so with a BF16 query its output and LSE must equal, bit for bit, the
BF16 kernel's on the same cache dequantized by ``dequant_gather_kv_mxfp8``.
Cache rows past each request's length hold NaN bytes and NaN scales; they must
not leak into the result.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.rmha import rel_mha_decode_with_kvcache
from tokenspeed_kernel.ops.kvcache.triton import (
    dequant_gather_kv_mxfp8,
    quantize_store_kv_mxfp8,
)
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import (
    MXFP8_BLOCK_SCALE,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

# Inkling 2 attention at TP4.
NUM_Q_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 128
PAGE = 128
EXTENT = 1024
SCALE = HEAD_DIM**-0.5


def _require_fa4(require) -> None:
    require("attention", "rel_mha_decode_with_kvcache", "fa4", torch.bfloat16, "q")


def _mxfp8_cache(kv_lens, device, seed):
    """MXFP8 pages for each request behind a shuffled page table.

    Every byte the requests do not write is a NaN: FP8 0x7F data and 0xFF
    scales, as stale pool memory may hold.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    pages = [-(-n // PAGE) for n in kv_lens]
    total = sum(pages) + 2
    data_shape = (total * PAGE, NUM_KV_HEADS, HEAD_DIM)
    k_cache = torch.full(data_shape, 0x7F, device=device, dtype=torch.uint8)
    v_cache = torch.full(data_shape, 0x7F, device=device, dtype=torch.uint8)
    k_cache, v_cache = (c.view(torch.float8_e4m3fn) for c in (k_cache, v_cache))
    sf_shape = (total, NUM_KV_HEADS, 1, 32, 4, 4)
    k_scale = torch.full(sf_shape, 0xFF, device=device, dtype=torch.uint8)
    v_scale = torch.full(sf_shape, 0xFF, device=device, dtype=torch.uint8)
    k_scale, v_scale = (s.view(torch.float8_e8m0fnu) for s in (k_scale, v_scale))
    table = torch.zeros(len(kv_lens), max(pages), device=device, dtype=torch.int32)
    order = torch.randperm(total, device=device, generator=gen).to(torch.int32)
    used = 0
    for b, n in enumerate(kv_lens):
        table[b, : pages[b]] = order[used : used + pages[b]]
        used += pages[b]
        shape = (n, NUM_KV_HEADS, HEAD_DIM)
        k = 3 * torch.randn(shape, device=device, generator=gen)
        v = 3 * torch.randn(shape, device=device, generator=gen)
        k[0, 0, :32] = 0  # a zero group: zero data and the smallest scale
        tok = torch.arange(n, device=device)
        loc = table[b, tok // PAGE].long() * PAGE + tok % PAGE
        quantize_store_kv_mxfp8(
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            loc,
            page_tokens=PAGE,
        )
    return k_cache, v_cache, k_scale, v_scale, table


def _first_pages(kv_lens, rows, window_left, device):
    """First page each request's decode reads: all, or from its window's start."""
    if window_left < 0:
        return torch.zeros(len(kv_lens), device=device, dtype=torch.int32)
    return torch.tensor(
        [max(n - rows - window_left, 0) // PAGE for n in kv_lens],
        device=device,
        dtype=torch.int32,
    )


def _dequantized(k_cache, v_cache, k_scale, v_scale, table, seqlens, first_page):
    """BF16 pages of what the decode reads, via the serving dequant fallback."""
    end_page = (seqlens + PAGE - 1) // PAGE
    num_pages = end_page - first_page
    scratch = int(num_pages.sum()) * PAGE
    k_out = torch.zeros(
        scratch, NUM_KV_HEADS, HEAD_DIM, device=k_cache.device, dtype=torch.bfloat16
    )
    v_out = torch.zeros_like(k_out)
    out_table = torch.zeros_like(table)
    dequant_gather_kv_mxfp8(
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        table,
        seqlens,
        first_page,
        num_pages,
        k_out,
        v_out,
        out_table,
        kernel_page_size=PAGE,
        page_tokens=PAGE,
    )
    shape = (-1, PAGE, NUM_KV_HEADS, HEAD_DIM)
    return k_out.view(shape), v_out.view(shape), out_table


def _queries(batch, rows, mode, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    total = batch * rows
    q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM, device=device, generator=gen)
    rel = 0.5 * torch.randn(total, NUM_Q_HEADS, EXTENT, device=device, generator=gen)
    tau = 1 + 0.25 * torch.rand(total, device=device, generator=gen)
    kwargs = dict(
        q=q.to(torch.bfloat16),
        rel_logits=rel.to(torch.bfloat16),
        tau=tau,
        softmax_scale=SCALE,
    )
    if mode == "fp32":
        kwargs.update(output_dtype=torch.float32, return_lse=True, gate=None, sink=None)
    else:
        # Rows of a wider projection, as the model slices them.
        width = NUM_Q_HEADS * HEAD_DIM
        projection = 3 * torch.randn(
            total, 8 + width + NUM_Q_HEADS, device=device, generator=gen
        )
        projection = projection.to(torch.bfloat16)
        kwargs.update(
            output_dtype=torch.bfloat16 if mode == "gated_bf16" else torch.float32,
            return_lse=mode == "gated_fp32",
            gate=projection[:, 8 : 8 + width],
            sink=projection[:, 8 + width :],
        )
    return kwargs


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def _assert_bit_equal(got, expected, return_lse: bool) -> None:
    if return_lse:
        (got, got_lse), (expected, expected_lse) = got, expected
        assert torch.equal(_bits(got_lse), _bits(expected_lse))
    assert got.dtype == expected.dtype
    assert not torch.isnan(got).any()
    assert torch.equal(_bits(got), _bits(expected))


def _check(kv_lens, rows, window_left, mode, device, seed=0) -> None:
    kv_lens = [n + rows for n in kv_lens]
    k_cache, v_cache, k_scale, v_scale, table = _mxfp8_cache(kv_lens, device, seed)
    seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    first_page = _first_pages(kv_lens, rows, window_left, device)
    k_bf16, v_bf16, bf16_table = _dequantized(
        k_cache, v_cache, k_scale, v_scale, table, seqlens, first_page
    )
    # Pages before a window's first read page may be released to the pool.
    for b in range(len(kv_lens)):
        table[b, : int(first_page[b])] = -1
    kwargs = _queries(len(kv_lens), rows, mode, device, seed + 1)
    kwargs.update(
        cache_seqlens=seqlens,
        max_seqlen_k=max(kv_lens),
        cu_seqlens_q=torch.arange(len(kv_lens) + 1, device=device, dtype=torch.int32)
        * rows,
        max_seqlen_q=rows,
        window_left=window_left,
    )
    expected = rel_mha_decode_with_kvcache(
        k_cache=k_bf16, v_cache=v_bf16, page_table=bf16_table, **kwargs
    )
    got = rel_mha_decode_with_kvcache(
        k_cache=k_cache.view(-1, PAGE, NUM_KV_HEADS, HEAD_DIM),
        v_cache=v_cache.view(-1, PAGE, NUM_KV_HEADS, HEAD_DIM),
        k_scale=k_scale,
        v_scale=v_scale,
        page_table=table,
        **kwargs,
    )
    _assert_bit_equal(got, expected, kwargs["return_lse"])


# Context lengths before the decode rows: empty, partial pages, past the window.
BATCHES = {
    "single": [2900],
    "mixed3": [0, 300, 2500],
    "mixed33": [(i * 997) % 3001 for i in range(33)],
}


def test_mxfp8_kv_selects_the_fa4_kernel(require) -> None:
    """A BF16 query over an MXFP8 cache selects the in-kernel dequant decode."""
    _require_fa4(require)
    mxfp8 = tensor_format("mxfp8", torch.float8_e4m3fn, scale=MXFP8_BLOCK_SCALE)
    signature = format_signature(
        q=dense_tensor_format(torch.bfloat16), k_cache=mxfp8, v_cache=mxfp8
    )
    for gated in (False, True):
        kernel = select_kernel(
            "attention",
            "rel_mha_decode_with_kvcache",
            signature,
            traits={
                "fp32_output": True,
                "gate_sink": gated,
                "head_dim": HEAD_DIM,
                "page_size": PAGE,
                "return_lse": True,
                "sliding_window": True,
            },
        )
        assert kernel.name == "fa4_rel_mha_decode_with_kvcache_fp32_mxfp8_kv"


@pytest.mark.parametrize("batch", list(BATCHES))
@pytest.mark.parametrize("mode", ["fp32", "gated_bf16"])
@pytest.mark.parametrize("window_left", [-1, EXTENT - 1])
@pytest.mark.parametrize("rows", [1, 8])
def test_mxfp8_kv_decode_equals_bf16_decode_of_dequantized_cache(
    device, require, rows, window_left, mode, batch
) -> None:
    _require_fa4(require)
    _check(BATCHES[batch], rows, window_left, mode, device)


def test_mxfp8_kv_gated_fp32_output(device, require) -> None:
    """The fused gate also stores an FP32 output, with the LSE."""
    _require_fa4(require)
    _check(BATCHES["mixed3"], 8, -1, "gated_fp32", device)


def test_mxfp8_kv_long_context(device, require) -> None:
    """Hundreds of tiles per KV split and a tail page of one token."""
    _require_fa4(require)
    _check([300 * PAGE, 5], 1, -1, "fp32", device)


def test_mxfp8_kv_cuda_graph_replay(device, require) -> None:
    """A captured MXFP8 decode replays new contents identically to eager."""
    _require_fa4(require)
    rows, window_left = 8, EXTENT - 1
    kv_lens = [n + rows for n in (300, 2600)]
    k_cache, v_cache, k_scale, v_scale, table = _mxfp8_cache(kv_lens, device, 3)
    seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    kwargs = _queries(len(kv_lens), rows, "gated_bf16", device, 4)
    kwargs.update(
        k_cache=k_cache.view(-1, PAGE, NUM_KV_HEADS, HEAD_DIM),
        v_cache=v_cache.view(-1, PAGE, NUM_KV_HEADS, HEAD_DIM),
        k_scale=k_scale,
        v_scale=v_scale,
        page_table=table,
        cache_seqlens=seqlens,
        max_seqlen_k=max(kv_lens),
        cu_seqlens_q=torch.arange(len(kv_lens) + 1, device=device, dtype=torch.int32)
        * rows,
        max_seqlen_q=rows,
        window_left=window_left,
    )
    rel_mha_decode_with_kvcache(**kwargs)  # eager compile and workspaces
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = rel_mha_decode_with_kvcache(**kwargs)
    # New queries, shorter requests and requantized pages.
    fresh = _mxfp8_cache(kv_lens, device, 5)
    for dst, src in zip((k_cache, v_cache, k_scale, v_scale, table), fresh):
        dst.copy_(src)
    for name in ("q", "rel_logits", "tau", "gate", "sink"):
        kwargs[name].copy_(_queries(len(kv_lens), rows, "gated_bf16", device, 6)[name])
    seqlens.copy_(torch.tensor([200, 1900], device=device, dtype=torch.int32))
    graph.replay()
    eager = rel_mha_decode_with_kvcache(**kwargs)
    _assert_bit_equal(out, eager, False)


def test_mxfp8_kv_batch_shapes_do_not_recompile(device, require) -> None:
    """Batch, length and page-count changes reuse the compiled kernel."""
    _require_fa4(require)
    from tokenspeed_kernel.ops.attention.rmha._cute_dsl import rel_decode_v2

    _check([400], 8, -1, "fp32", device)
    compiled = len(rel_decode_v2._DECODE._cache)
    _check([300, 1700, 2900], 8, -1, "fp32", device, seed=1)
    _check([5000, 129, 0, 7], 8, -1, "fp32", device, seed=2)
    assert len(rel_decode_v2._DECODE._cache) == compiled
