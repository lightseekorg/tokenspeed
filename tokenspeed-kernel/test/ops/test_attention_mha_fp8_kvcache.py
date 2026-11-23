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

"""Portable Triton MHA reads of a unit-scale FP8 E4M3 KV cache under a 16-bit
query: the kernels widen each K/V tile to the query dtype in registers, so the
result equals the same kernel over the dequantized (bf16) cache (bit for bit on
AMD, see ``_assert_matches_dequant``), and an fp32 reference over the
dequantized cache within the tolerance the bf16-cache tests of these kernels
use."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_fp8_kvcache_query_dtype,
)
from tokenspeed_kernel.ops.attention.mha._triton.decode import (
    _fwd_folded_kernel_stage1,
    _fwd_folded_kernel_stage2,
)
from tokenspeed_kernel.ops.attention.mha._triton.prefill import _fwd_kernel
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import assert_no_triton_compile

FP8 = torch.float8_e4m3fn
DECODE = "mha_decode_with_kvcache"
EXTEND = "mha_extend_with_kvcache"

# (num_q_heads, num_kv_heads, head_dim): Qwen3.8-27B's per-rank attention
# (hd256, GQA 6), a 128-wide GQA layer, and MHA.
SHAPES = pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim",
    [(24, 4, 256), (16, 4, 128), (4, 4, 256)],
    ids=["hd256-gqa6", "hd128-gqa4", "hd256-mha"],
)
PAGE_SIZES = pytest.mark.parametrize("page_size", [16, 64])


def _paged_cache(cache_lens, page_size, num_kv_heads, head_dim, gen):
    """An FP8 K/V cache over shuffled pages (null page 0 left unused) and the
    page table addressing it."""
    pages_per_req = [(n + page_size - 1) // page_size for n in cache_lens]
    total = sum(pages_per_req)
    order = (torch.randperm(total, generator=gen) + 1).tolist()
    table = torch.zeros(len(cache_lens), max(pages_per_req) + 1, dtype=torch.int32)
    start = 0
    for b, n in enumerate(pages_per_req):
        table[b, :n] = torch.tensor(order[start : start + n], dtype=torch.int32)
        start += n
    shape = (total + 1, page_size, num_kv_heads, head_dim)
    # Values beyond +-1 exercise e4m3's exponent range, not just its mantissa.
    k = (2 * torch.randn(shape, generator=gen)).to(FP8)
    v = (2 * torch.randn(shape, generator=gen)).to(FP8)
    return k, v, table


def _reference(q, k_cache, v_cache, table, cache_lens, q_lens, scale):
    """fp32 attention over the dequantized cache; query row t of a request
    with L keys and Q rows sees keys [0, L - Q + t] (causal suffix)."""
    num_q_heads = q.shape[1]
    page_size, num_kv_heads, head_dim = k_cache.shape[1:]
    group = num_q_heads // num_kv_heads
    out = torch.empty(q.shape, dtype=torch.float32)
    lse = torch.empty(q.shape[:2], dtype=torch.float32)
    row = 0
    for b, (length, q_len) in enumerate(zip(cache_lens, q_lens)):
        pos = torch.arange(length)
        pages = table[b, pos // page_size].long()
        kk = k_cache[pages, pos % page_size].float().repeat_interleave(group, 1)
        vv = v_cache[pages, pos % page_size].float().repeat_interleave(group, 1)
        for t in range(q_len):
            s = torch.einsum("hd,xhd->hx", q[row].float(), kk) * scale
            s[:, length - q_len + t + 1 :] = float("-inf")
            out[row] = torch.einsum("hx,xhd->hd", torch.softmax(s, -1), vv)
            lse[row] = torch.logsumexp(s, -1)
            row += 1
    return out, lse


def _assert_matches_dequant(actual, expected):
    """The FP8 read widens each K/V tile and then does the dequantized read's
    arithmetic. On AMD both compile to the same instructions and agree bit for
    bit. Triton's NVIDIA backend does not lower the dot over a widened FP8 load
    the same way (on B200 a few elements differ by an ulp): the fp32
    accumulators can differ in their last bits, which can flip the rounding of
    the probabilities to the query dtype before P.V and of the output. There a
    16-bit output may be two rounding steps off, and the float32 log-sum-exp is
    checked at torch's float32 default tolerance."""
    if current_platform().is_amd:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected)
    else:
        eps = torch.finfo(actual.dtype).eps
        torch.testing.assert_close(actual, expected, atol=2 * eps, rtol=2 * eps)


def _decode(q, k, v, table, lens, q_len, return_lse=False):
    return mha_decode_with_kvcache(
        q=q,
        k_cache=k,
        v_cache=v,
        page_table=table,
        cache_seqlens=lens,
        max_seqlen_k=int(lens.max()),
        max_seqlen_q=q_len,
        return_lse=return_lse,
        solution="triton",
        block_draft=False,
    )


def _extend(q, k, v, table, lens, q_lens, return_lse):
    cu_q = torch.nn.functional.pad(torch.cumsum(q_lens, 0, dtype=torch.int32), (1, 0))
    return mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=torch.nn.functional.pad(
            torch.cumsum(lens, 0, dtype=torch.int32), (1, 0)
        ),
        k_cache=k,
        v_cache=v,
        page_table=table,
        cache_seqlens=lens,
        max_seqlen_q=int(q_lens.max()),
        max_seqlen_k=int(lens.max()),
        is_causal=True,
        return_lse=return_lse,
        solution="triton",
    )


@SHAPES
@PAGE_SIZES
@pytest.mark.parametrize("q_len", [1, 4, 16], ids=["q1", "q4", "q16"])
@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float16])
def test_triton_decode_reads_fp8_cache(
    q_dtype, q_len, page_size, num_q_heads, num_kv_heads, head_dim
):
    """Decode (q_len 1), compact speculative verify (q_len > 1) and the tree
    cascade's causal prefix (return_lse; prefixes shorter than the rows), all
    through the folded decode."""
    gen = torch.Generator().manual_seed(head_dim + page_size + q_len)
    cache_lens = [q_len, 63 + q_len, 700, 1031]
    k, v, table = _paged_cache(cache_lens, page_size, num_kv_heads, head_dim, gen)
    q = torch.randn(len(cache_lens) * q_len, num_q_heads, head_dim, generator=gen).to(
        q_dtype
    )
    ref, ref_lse = _reference(
        q, k, v, table, cache_lens, [q_len] * len(cache_lens), head_dim**-0.5
    )
    q, k, v, table = q.cuda(), k.cuda(), v.cuda(), table.cuda()
    lens = torch.tensor(cache_lens, dtype=torch.int32, device="cuda")

    out = _decode(q, k, v, table, lens, q_len)
    dequant = _decode(q, k.to(q_dtype), v.to(q_dtype), table, lens, q_len)

    assert out.dtype == q_dtype
    _assert_matches_dequant(out, dequant)
    torch.testing.assert_close(out.float().cpu(), ref, atol=3e-2, rtol=3e-2)

    out, lse = _decode(q, k, v, table, lens, q_len, return_lse=True)
    dequant, dequant_lse = _decode(
        q, k.to(q_dtype), v.to(q_dtype), table, lens, q_len, return_lse=True
    )
    _assert_matches_dequant(out, dequant)
    _assert_matches_dequant(lse, dequant_lse)
    torch.testing.assert_close(out.float().cpu(), ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse.cpu(), ref_lse, atol=1e-3, rtol=1e-3)


@SHAPES
@PAGE_SIZES
@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float16])
def test_triton_extend_reads_fp8_cache(
    q_dtype, page_size, num_q_heads, num_kv_heads, head_dim
):
    """Chunked prefill over a cached prefix (prewrite extend) and the
    return_lse read (prefixes shorter than the rows)."""
    gen = torch.Generator().manual_seed(head_dim + page_size)
    q_lens_list = [1, 16, 37, 130]
    cache_lens = [5, 16, 300, 1100]
    k, v, table = _paged_cache(cache_lens, page_size, num_kv_heads, head_dim, gen)
    q = torch.randn(sum(q_lens_list), num_q_heads, head_dim, generator=gen).to(q_dtype)
    ref, ref_lse = _reference(q, k, v, table, cache_lens, q_lens_list, head_dim**-0.5)
    q, k, v, table = q.cuda(), k.cuda(), v.cuda(), table.cuda()
    lens = torch.tensor(cache_lens, dtype=torch.int32, device="cuda")
    q_lens = torch.tensor(q_lens_list, dtype=torch.int32, device="cuda")

    out, lse = _extend(q, k, v, table, lens, q_lens, return_lse=True)
    dequant, dequant_lse = _extend(
        q, k.to(q_dtype), v.to(q_dtype), table, lens, q_lens, return_lse=True
    )

    assert out.dtype == q_dtype
    _assert_matches_dequant(out, dequant)
    _assert_matches_dequant(lse, dequant_lse)
    torch.testing.assert_close(out.float().cpu(), ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse.cpu(), ref_lse, atol=1e-3, rtol=1e-3)


def test_triton_fp8_reads_do_not_recompile_per_batch():
    """Batch size, lengths and page-table width are runtime arguments."""
    gen = torch.Generator().manual_seed(0)
    heads, kv_heads, dim, page = 24, 4, 256, 64
    kernels = (_fwd_folded_kernel_stage1, _fwd_folded_kernel_stage2)

    def run(cache_lens, q_len):
        k, v, table = _paged_cache(cache_lens, page, kv_heads, dim, gen)
        q = torch.randn(len(cache_lens) * q_len, heads, dim, generator=gen)
        q, k, v, table = q.bfloat16().cuda(), k.cuda(), v.cuda(), table.cuda()
        lens = torch.tensor(cache_lens, dtype=torch.int32, device="cuda")
        # One row per request (plain decode) and all rows (speculative verify).
        _decode(q[q_len - 1 :: q_len].contiguous(), k, v, table, lens, 1)
        _decode(q, k, v, table, lens, q_len)
        q_lens = torch.full_like(lens, q_len)
        _extend(q, k, v, table, lens, q_lens, return_lse=True)

    # Warm both integer specialization classes (divisible by 16 or not) of the
    # page-table width (15 pages + the trailing null column = 16).
    for cache_lens in ([16, 32], [17, 33, 50], [15 * page]):
        run(cache_lens, 4)
    with assert_no_triton_compile(*kernels, _fwd_kernel):
        for cache_lens in ([20, 900, 64], [5, 6, 7, 8, 9], [3000]):
            run(cache_lens, 4)


@SHAPES
@PAGE_SIZES
@pytest.mark.parametrize("mode", [DECODE, EXTEND])
def test_fp8_query_dtype_names_a_servable_read(
    mode, page_size, num_q_heads, num_kv_heads, head_dim
):
    """The query dtype chosen for an FP8 cache read selects a kernel; under
    the portable solution it is the activation dtype (the kernel widens K/V)."""
    for solution in (None, "triton"):
        q_dtype = mha_fp8_kvcache_query_dtype(
            mode=mode,
            kv_cache_dtype=FP8,
            activation_dtype=torch.bfloat16,
            head_dim=head_dim,
            page_size=page_size,
            window_left=-1,
            logit_cap=0.0,
            sinks=False,
            solution=solution,
        )
        if solution == "triton":
            assert q_dtype == torch.bfloat16
        traits = {
            "head_dim": head_dim,
            "page_size": page_size,
            "logit_cap": False,
            "return_lse": False,
            "sinks": False,
            "sliding_window": False,
        }
        if mode == EXTEND:
            traits["is_causal"] = True
        else:
            traits["q_len"] = 1
        select_kernel(
            "attention",
            mode,
            format_signature(
                q=dense_tensor_format(q_dtype),
                k_cache=dense_tensor_format(FP8),
                v_cache=dense_tensor_format(FP8),
            ),
            traits=traits,
            solution=solution,
        )


@pytest.mark.skipif(not current_platform().is_cdna4, reason="gfx950 kernels")
@pytest.mark.parametrize(
    "mode,head_dim,page_size,expected",
    [
        # The Gluon decode reads FP8 queries at hd256 over 64-token pages.
        (DECODE, 256, 64, FP8),
        (DECODE, 128, 64, FP8),
        # It reads 64-token pages only; Triton widens the cache instead.
        (DECODE, 256, 16, torch.bfloat16),
        # The Gluon extend has no hd256 kernel.
        (EXTEND, 256, 64, torch.bfloat16),
    ],
)
def test_gfx950_fp8_query_dtype(mode, head_dim, page_size, expected):
    q_dtype = mha_fp8_kvcache_query_dtype(
        mode=mode,
        kv_cache_dtype=FP8,
        activation_dtype=torch.bfloat16,
        head_dim=head_dim,
        page_size=page_size,
        window_left=-1,
        logit_cap=0.0,
        sinks=None,
        solution=None,
    )
    assert q_dtype == expected


@pytest.mark.skipif(not current_platform().is_cdna4, reason="gfx950 kernels")
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(24, 4), (4, 4)])
def test_gfx950_fp8_query_decode_reads_hd256_fp8_cache(num_q_heads, num_kv_heads):
    """The hd256 decode the resolution picks on gfx950 (FP8 queries, page 64)
    against the fp32 reference over the FP8 query and the dequantized cache.

    That kernel rounds the probabilities to e4m3 (3 mantissa bits) before
    P.V, which is at most 2**-4 relative per key, so an output element may
    be off by up to 2**-4 * sum_x p_x |v_x| (the reference over |V|), plus
    the bf16 rounding of the output."""
    head_dim, page_size = 256, 64
    gen = torch.Generator().manual_seed(1)
    cache_lens = [1, 64, 700, 5000]
    k, v, table = _paged_cache(cache_lens, page_size, num_kv_heads, head_dim, gen)
    q = torch.randn(len(cache_lens), num_q_heads, head_dim, generator=gen).to(FP8)
    q_lens = [1] * len(cache_lens)
    ref, _ = _reference(q, k, v, table, cache_lens, q_lens, head_dim**-0.5)
    ref_abs_v, _ = _reference(
        q, k, v.float().abs().to(FP8), table, cache_lens, q_lens, head_dim**-0.5
    )
    lens = torch.tensor(cache_lens, dtype=torch.int32, device="cuda")
    out = mha_decode_with_kvcache(
        q=q.cuda(),
        k_cache=k.cuda(),
        v_cache=v.cuda(),
        page_table=table.cuda(),
        cache_seqlens=lens,
        max_seqlen_k=max(cache_lens),
        max_seqlen_q=1,
        block_draft=False,
    )
    error = (out.float().cpu() - ref).abs()
    assert (error <= 2**-4 * ref_abs_v + 2**-8 * ref.abs() + 1e-5).all()
