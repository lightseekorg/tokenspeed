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

"""Portable Triton MHA reads of an FP8 E4M3 KV cache under a 16-bit query: the
kernels widen each K/V tile to the query dtype in registers, so over a
unit-scale cache the result equals the same kernel over the dequantized (bf16)
cache bit for bit, and an fp32 reference over the dequantized cache within the
tolerance the bf16-cache tests of these kernels use. A cache at per-tensor
scales matches the reference over the cache dequantized with its scales."""

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
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    tensor_format,
)
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
# (k_scale, v_scale) of a scaled cache: powers of two, and scales that move
# every value off the e4m3 grid.
KV_SCALES = pytest.mark.parametrize(
    "kv_scales",
    [(0.5, 2.0), (2.0, 0.5), (0.0371, 1.73)],
    ids=["k0.5-v2", "k2-v0.5", "k0.0371-v1.73"],
)
SCALED_FP8_CACHE = tensor_format(
    "scaled-fp8",
    FP8,
    scale=ScaleFormat(storage_dtype=torch.float32, granularity="tensor"),
)


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


def _reference(
    q, k_cache, v_cache, table, cache_lens, q_lens, scale, kv_scales=(1.0, 1.0)
):
    """fp32 attention over the cache dequantized with ``kv_scales``; query row
    t of a request with L keys and Q rows sees keys [0, L - Q + t] (causal
    suffix)."""
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
        kk, vv = kk * kv_scales[0], vv * kv_scales[1]
        for t in range(q_len):
            s = torch.einsum("hd,xhd->hx", q[row].float(), kk) * scale
            s[:, length - q_len + t + 1 :] = float("-inf")
            out[row] = torch.einsum("hx,xhd->hd", torch.softmax(s, -1), vv)
            lse[row] = torch.logsumexp(s, -1)
            row += 1
    return out, lse


def _decode(q, k, v, table, lens, q_len, return_lse=False, fp8_kv_scales=None):
    return mha_decode_with_kvcache(
        q=q,
        k_cache=k,
        v_cache=v,
        page_table=table,
        cache_seqlens=lens,
        max_seqlen_k=int(lens.max()),
        max_seqlen_q=q_len,
        return_lse=return_lse,
        fp8_kv_scales=fp8_kv_scales,
        solution="triton",
        block_draft=False,
    )


def _extend(q, k, v, table, lens, q_lens, return_lse, fp8_kv_scales=None):
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
        fp8_kv_scales=fp8_kv_scales,
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
    torch.testing.assert_close(out, dequant, atol=0, rtol=0)
    torch.testing.assert_close(out.float().cpu(), ref, atol=3e-2, rtol=3e-2)

    out, lse = _decode(q, k, v, table, lens, q_len, return_lse=True)
    dequant, dequant_lse = _decode(
        q, k.to(q_dtype), v.to(q_dtype), table, lens, q_len, return_lse=True
    )
    torch.testing.assert_close(out, dequant, atol=0, rtol=0)
    torch.testing.assert_close(lse, dequant_lse, atol=0, rtol=0)
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
    torch.testing.assert_close(out, dequant, atol=0, rtol=0)
    torch.testing.assert_close(lse, dequant_lse, atol=0, rtol=0)
    torch.testing.assert_close(out.float().cpu(), ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse.cpu(), ref_lse, atol=1e-3, rtol=1e-3)


@SHAPES
@KV_SCALES
@pytest.mark.parametrize("q_len", [1, 16], ids=["q1", "q16"])
def test_triton_decode_reads_scaled_fp8_cache(
    q_len, kv_scales, num_q_heads, num_kv_heads, head_dim
):
    """A cache at per-tensor scales, for decode, verify and the tree cascade's
    causal prefix (return_lse): the K scale reaches the logits and the
    log-sum-exp, the V scale the output."""
    page_size = 64
    gen = torch.Generator().manual_seed(head_dim + q_len)
    cache_lens = [q_len, 63 + q_len, 700, 1031]
    k, v, table = _paged_cache(cache_lens, page_size, num_kv_heads, head_dim, gen)
    # Queries scaled so k_scale = 0.0371 still gives O(1) logits.
    q = (
        torch.randn(len(cache_lens) * q_len, num_q_heads, head_dim, generator=gen)
        / max(kv_scales[0], 0.5)
    ).bfloat16()
    ref, ref_lse = _reference(
        q,
        k,
        v,
        table,
        cache_lens,
        [q_len] * len(cache_lens),
        head_dim**-0.5,
        kv_scales,
    )
    q, k, v, table = q.cuda(), k.cuda(), v.cuda(), table.cuda()
    lens = torch.tensor(cache_lens, dtype=torch.int32, device="cuda")
    tol = 3e-2 * max(1.0, kv_scales[1])

    out = _decode(q, k, v, table, lens, q_len, fp8_kv_scales=kv_scales)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float().cpu(), ref, atol=tol, rtol=tol)

    out, lse = _decode(
        q, k, v, table, lens, q_len, return_lse=True, fp8_kv_scales=kv_scales
    )
    torch.testing.assert_close(out.float().cpu(), ref, atol=tol, rtol=tol)
    torch.testing.assert_close(lse.cpu(), ref_lse, atol=1e-3, rtol=1e-3)


@SHAPES
@KV_SCALES
def test_triton_extend_reads_scaled_fp8_cache(
    kv_scales, num_q_heads, num_kv_heads, head_dim
):
    """Chunked prefill over a prefix cached at per-tensor scales, with its
    log-sum-exp."""
    page_size = 64
    gen = torch.Generator().manual_seed(head_dim)
    q_lens_list = [1, 16, 37, 130]
    cache_lens = [5, 16, 300, 1100]
    k, v, table = _paged_cache(cache_lens, page_size, num_kv_heads, head_dim, gen)
    q = (
        torch.randn(sum(q_lens_list), num_q_heads, head_dim, generator=gen)
        / max(kv_scales[0], 0.5)
    ).bfloat16()
    ref, ref_lse = _reference(
        q, k, v, table, cache_lens, q_lens_list, head_dim**-0.5, kv_scales
    )
    q, k, v, table = q.cuda(), k.cuda(), v.cuda(), table.cuda()
    lens = torch.tensor(cache_lens, dtype=torch.int32, device="cuda")
    q_lens = torch.tensor(q_lens_list, dtype=torch.int32, device="cuda")

    out, lse = _extend(
        q, k, v, table, lens, q_lens, return_lse=True, fp8_kv_scales=kv_scales
    )
    tol = 3e-2 * max(1.0, kv_scales[1])
    torch.testing.assert_close(out.float().cpu(), ref, atol=tol, rtol=tol)
    torch.testing.assert_close(lse.cpu(), ref_lse, atol=1e-3, rtol=1e-3)


def test_triton_fp8_reads_do_not_recompile_per_batch():
    """Batch size, lengths and page-table width are runtime arguments."""
    gen = torch.Generator().manual_seed(0)
    heads, kv_heads, dim, page = 24, 4, 256, 64
    kernels = (_fwd_folded_kernel_stage1, _fwd_folded_kernel_stage2)

    def run(cache_lens, q_len, scales):
        k, v, table = _paged_cache(cache_lens, page, kv_heads, dim, gen)
        q = torch.randn(len(cache_lens) * q_len, heads, dim, generator=gen)
        q, k, v, table = q.bfloat16().cuda(), k.cuda(), v.cuda(), table.cuda()
        lens = torch.tensor(cache_lens, dtype=torch.int32, device="cuda")
        # One row per request (plain decode) and all rows (speculative verify).
        _decode(
            q[q_len - 1 :: q_len].contiguous(),
            k,
            v,
            table,
            lens,
            1,
            fp8_kv_scales=scales,
        )
        _decode(q, k, v, table, lens, q_len, fp8_kv_scales=scales)
        q_lens = torch.full_like(lens, q_len)
        _extend(q, k, v, table, lens, q_lens, return_lse=True, fp8_kv_scales=scales)

    # Warm both integer specialization classes (divisible by 16 or not) of the
    # page-table width (15 pages + the trailing null column = 16).
    for cache_lens in ([16, 32], [17, 33, 50], [15 * page]):
        run(cache_lens, 4, None)
    # Per-tensor scales are runtime arguments too: unit and scaled layers
    # share one binary.
    with assert_no_triton_compile(*kernels, _fwd_kernel):
        for cache_lens, scales in (
            ([20, 900, 64], None),
            ([5, 6, 7, 8, 9], (0.5, 2.0)),
            ([3000], (0.0371, 1.73)),
        ):
            run(cache_lens, 4, scales)


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
            fp8_kv_scaled=False,
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


@SHAPES
@pytest.mark.parametrize("mode", [DECODE, EXTEND])
def test_scaled_fp8_query_dtype_names_a_portable_read(
    mode, num_q_heads, num_kv_heads, head_dim
):
    """Under the portable solution a cache at per-tensor scales keeps the
    activation-dtype query, and that read selects a kernel."""
    q_dtype = mha_fp8_kvcache_query_dtype(
        mode=mode,
        kv_cache_dtype=FP8,
        activation_dtype=torch.bfloat16,
        head_dim=head_dim,
        page_size=64,
        window_left=-1,
        logit_cap=0.0,
        sinks=False,
        solution="triton",
        fp8_kv_scaled=True,
    )
    assert q_dtype == torch.bfloat16
    traits = {
        "head_dim": head_dim,
        "page_size": 64,
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
            k_cache=SCALED_FP8_CACHE,
            v_cache=SCALED_FP8_CACHE,
        ),
        traits=traits,
        solution="triton",
    )


@pytest.mark.skipif(
    not current_platform().is_amd, reason="gfx950 hd256 has no FP8-query kernel"
)
def test_fp8_query_dtype_keeps_hd256_queries_in_bf16_on_amd():
    q_dtype = mha_fp8_kvcache_query_dtype(
        mode=DECODE,
        kv_cache_dtype=FP8,
        activation_dtype=torch.bfloat16,
        head_dim=256,
        page_size=64,
        window_left=-1,
        logit_cap=0.0,
        sinks=None,
        solution=None,
        fp8_kv_scaled=False,
    )
    assert q_dtype == torch.bfloat16
