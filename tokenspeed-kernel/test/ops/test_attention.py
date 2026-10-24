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

import math

import pytest
import torch
from tokenspeed_kernel.ops.attention import attn_merge_state
from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
)
from tokenspeed_kernel.platform import current_platform

torch.manual_seed(42)

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz})


def _randn(shape: tuple[int, ...], *, device: str, dtype: torch.dtype) -> torch.Tensor:
    init_dtype = torch.bfloat16 if dtype in _FP8_DTYPES else dtype
    tensor = torch.randn(shape, device=device, dtype=init_dtype)
    if dtype != init_dtype:
        tensor = tensor.to(dtype)
    return tensor


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        pytest.param(torch.bfloat16, 64, 8, 2, id="bf16-d64"),
        pytest.param(torch.bfloat16, 128, 8, 2, id="bf16-d128"),
        pytest.param(torch.float8_e4m3fn, 64, 8, 2, id="e4m3-d64"),
        pytest.param(torch.float8_e4m3fn, 128, 8, 2, id="e4m3-d128"),
        pytest.param(torch.float8_e5m2, 64, 8, 2, id="e5m2-d64"),
        pytest.param(torch.float8_e5m2, 128, 8, 2, id="e5m2-d128"),
    ],
)
@pytest.mark.parametrize("solution", ["triton", "fa3", "fa4", "gluon"])
@pytest.mark.parametrize("has_sink", [False, True], ids=["no-sink", "sink"])
@pytest.mark.parametrize("is_sliding", [False, True], ids=["full", "sliding"])
def test_mha_prefill(
    device: str,
    solution: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    has_sink: bool,
    is_sliding: bool,
    require,
) -> None:
    require("attention", "mha_prefill", solution, dtype, "q")
    if solution == "fa4" and (has_sink or is_sliding):
        pytest.skip("FA4 MHA prefill does not support sinks or sliding window")

    seqlens_list = [851, 914, 1053]
    max_seqlen = max(seqlens_list)
    cu_seqlens_cpu = [0]
    for seqlen in seqlens_list:
        cu_seqlens_cpu.append(cu_seqlens_cpu[-1] + seqlen)
    seqlens = torch.tensor(seqlens_list, device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor(cu_seqlens_cpu, device=device, dtype=torch.int32)
    total_tokens = int(seqlens.sum().item())

    q = _randn((total_tokens, num_q_heads, head_dim), device=device, dtype=dtype)
    k = _randn((total_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)
    v = _randn((total_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)
    sinks = _randn((num_q_heads,), device=device, dtype=q.dtype) if has_sink else None
    window_left = 127 if is_sliding else -1

    out = mha_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        max_seqlen=max_seqlen,
        window_left=window_left,
        sinks=sinks,
        solution=solution,
    )

    assert out.shape == q.shape
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        pytest.param(torch.bfloat16, 64, 8, 2, id="bf16-d64"),
        pytest.param(torch.bfloat16, 128, 8, 2, id="bf16-d128"),
    ],
)
@pytest.mark.parametrize("solution", ["triton", "gluon"])
@pytest.mark.parametrize("window_left", [-1, 127], ids=["full", "sliding"])
def test_mha_prefill_lse(
    device: str,
    solution: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    window_left: int,
    require,
) -> None:
    require("attention", "mha_prefill", solution, dtype, "q")

    seqlens_list = [851, 914, 1053]
    max_seqlen = max(seqlens_list)
    cu_seqlens_cpu = [0]
    for seqlen in seqlens_list:
        cu_seqlens_cpu.append(cu_seqlens_cpu[-1] + seqlen)
    cu_seqlens = torch.tensor(cu_seqlens_cpu, device=device, dtype=torch.int32)
    total_tokens = cu_seqlens_cpu[-1]

    q = _randn((total_tokens, num_q_heads, head_dim), device=device, dtype=dtype)
    k = _randn((total_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)
    v = _randn((total_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)
    sm_scale = 1.0 / math.sqrt(head_dim)
    group = num_q_heads // num_kv_heads

    out, lse = mha_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        max_seqlen=max_seqlen,
        window_left=window_left,
        return_lse=True,
        solution=solution,
    )

    assert out.shape == q.shape
    assert lse.shape == (total_tokens, num_q_heads)

    # Reference: natural-log log-sum-exp over a causal MHA prefill.
    ref_outs = []
    ref_lses = []
    for start, end in zip(cu_seqlens_cpu[:-1], cu_seqlens_cpu[1:]):
        q_i = q[start:end].float()
        k_i = k[start:end].float()
        k_exp = k_i.repeat_interleave(group, dim=1)
        v_exp = v[start:end].float().repeat_interleave(group, dim=1)
        seq_len = end - start
        scores = torch.einsum("qhd,khd->hqk", q_i, k_exp) * sm_scale
        pos = torch.arange(seq_len, device=device)
        mask = pos[:, None] >= pos[None, :]
        if window_left >= 0:
            mask &= (pos[:, None] - pos[None, :]) <= window_left
        scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        ref_outs.append(torch.einsum("hqk,khd->qhd", probs, v_exp))
        ref_lses.append(torch.logsumexp(scores, dim=-1).transpose(0, 1))
    out_ref = torch.cat(ref_outs, dim=0)
    lse_ref = torch.cat(ref_lses, dim=0)

    torch.testing.assert_close(out.float(), out_ref, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=8e-2, atol=8e-2)


def _reference_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    window_left: int,
    sinks: torch.Tensor | None,
) -> torch.Tensor:
    """Causal GQA in fp32; a sink adds one logit per head to the softmax
    denominator and contributes no value."""
    group = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(group, dim=1)
    v = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) / math.sqrt(q.shape[-1])
    mask = q_pos[:, None] >= k_pos[None, :]
    if window_left >= 0:
        mask &= q_pos[:, None] - k_pos[None, :] <= window_left
    scores = scores.masked_fill(~mask[None], float("-inf"))
    if sinks is not None:
        sink = sinks.float()[:, None, None].expand(-1, scores.shape[1], 1)
        scores = torch.cat([scores, sink], dim=-1)
    probs = torch.softmax(scores, dim=-1)
    if sinks is not None:
        probs = probs[..., :-1]
    return torch.einsum("hqk,khd->qhd", probs, v)


def _paged_kv(
    kv_lens: list[int], page_size: int, rows: tuple[int, ...], device: str
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Scatter each sequence's rows over shuffled pages; returns the cache,
    the page table and the per-sequence dense rows."""
    pages_per_seq = [(n + page_size - 1) // page_size for n in kv_lens]
    num_pages = sum(pages_per_seq)
    order = torch.randperm(num_pages, device=device, dtype=torch.int32)
    cache = torch.zeros(
        (num_pages, page_size, *rows), device=device, dtype=torch.bfloat16
    )
    page_table = torch.zeros(
        (len(kv_lens), max(pages_per_seq)), device=device, dtype=torch.int32
    )
    dense, first = [], 0
    for seq, (kv_len, pages) in enumerate(zip(kv_lens, pages_per_seq)):
        table = order[first : first + pages]
        first += pages
        page_table[seq, :pages] = table
        rows_i = torch.randn((kv_len, *rows), device=device, dtype=torch.bfloat16)
        padded = torch.zeros(
            (pages * page_size, *rows), device=device, dtype=torch.bfloat16
        )
        padded[:kv_len] = rows_i
        cache[table.long()] = padded.view(pages, page_size, *rows)
        dense.append(rows_i)
    return cache, page_table, dense


# MiMo-V2 pairs 192-wide query/key heads with 128-wide value heads.
_DISTINCT_V_GEOMETRY = dict(
    num_q_heads=16, num_kv_heads=2, head_dim=192, v_head_dim=128
)


@pytest.mark.parametrize("solution", ["triton", "fa3"])
@pytest.mark.parametrize("has_sink", [False, True], ids=["no-sink", "sink"])
@pytest.mark.parametrize("window_left", [-1, 127], ids=["full", "sliding"])
def test_mha_prefill_distinct_v_head_dim(
    device: str, solution: str, has_sink: bool, window_left: int, require
) -> None:
    require("attention", "mha_prefill", solution, torch.bfloat16, "q")
    g = _DISTINCT_V_GEOMETRY
    seqlens = [300, 77, 512]
    cu_seqlens_cpu = [0]
    for seqlen in seqlens:
        cu_seqlens_cpu.append(cu_seqlens_cpu[-1] + seqlen)
    total = cu_seqlens_cpu[-1]
    q = torch.randn(total, g["num_q_heads"], g["head_dim"], device=device).bfloat16()
    k = torch.randn(total, g["num_kv_heads"], g["head_dim"], device=device).bfloat16()
    v = torch.randn(total, g["num_kv_heads"], g["v_head_dim"], device=device).bfloat16()
    sinks = (
        torch.randn(g["num_q_heads"], device=device).bfloat16() if has_sink else None
    )

    out = mha_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=torch.tensor(cu_seqlens_cpu, device=device, dtype=torch.int32),
        cu_seqlens_cpu=cu_seqlens_cpu,
        max_seqlen=max(seqlens),
        window_left=window_left,
        sinks=sinks,
        solution=solution,
    )

    assert out.shape == (total, g["num_q_heads"], g["v_head_dim"])
    expected = torch.cat(
        [
            _reference_gqa_attention(
                q[start:end],
                k[start:end],
                v[start:end],
                torch.arange(end - start, device=device),
                torch.arange(end - start, device=device),
                window_left,
                sinks,
            )
            for start, end in zip(cu_seqlens_cpu[:-1], cu_seqlens_cpu[1:])
        ]
    )
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("solution", ["triton", "fa3"])
@pytest.mark.parametrize("has_sink", [False, True], ids=["no-sink", "sink"])
@pytest.mark.parametrize("window_left", [-1, 127], ids=["full", "sliding"])
def test_mha_extend_with_kvcache_distinct_v_head_dim(
    device: str, solution: str, has_sink: bool, window_left: int, require
) -> None:
    require("attention", "mha_extend_with_kvcache", solution, torch.bfloat16, "q")
    g = _DISTINCT_V_GEOMETRY
    page_size = 64
    prefix_lens, query_lens = [0, 191, 300], [130, 9, 67]
    kv_lens = [p + n for p, n in zip(prefix_lens, query_lens)]
    k_cache, page_table, k_rows = _paged_kv(
        kv_lens, page_size, (g["num_kv_heads"], g["head_dim"]), device
    )
    v_cache, _, v_rows = _paged_kv(
        kv_lens, page_size, (g["num_kv_heads"], g["v_head_dim"]), device
    )
    # Both caches must share one page table: rewrite V at K's pages.
    v_cache.zero_()
    for seq, rows in enumerate(v_rows):
        pages = (kv_lens[seq] + page_size - 1) // page_size
        padded = torch.zeros(
            (pages * page_size, *rows.shape[1:]), device=device, dtype=rows.dtype
        )
        padded[: kv_lens[seq]] = rows
        v_cache[page_table[seq, :pages].long()] = padded.view(
            pages, page_size, *rows.shape[1:]
        )
    q = torch.randn(
        sum(query_lens), g["num_q_heads"], g["head_dim"], device=device
    ).bfloat16()
    sinks = (
        torch.randn(g["num_q_heads"], device=device).bfloat16() if has_sink else None
    )
    cu_q = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()])
    cu_kv = torch.tensor([0, *torch.tensor(kv_lens).cumsum(0).tolist()])

    out = mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=cu_q.to(device=device, dtype=torch.int32),
        cu_seqlens_kv=cu_kv.to(device=device, dtype=torch.int32),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=torch.tensor(kv_lens, device=device, dtype=torch.int32),
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(kv_lens),
        is_causal=True,
        window_left=window_left,
        sinks=sinks,
        solution=solution,
    )

    assert out.shape == (sum(query_lens), g["num_q_heads"], g["v_head_dim"])
    expected = torch.cat(
        [
            _reference_gqa_attention(
                q[cu_q[seq] : cu_q[seq + 1]],
                k_rows[seq],
                v_rows[seq],
                prefix_lens[seq] + torch.arange(query_lens[seq], device=device),
                torch.arange(kv_lens[seq], device=device),
                window_left,
                sinks,
            )
            for seq in range(len(kv_lens))
        ]
    )
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("solution", ["triton", "fa3"])
@pytest.mark.parametrize("has_sink", [False, True], ids=["no-sink", "sink"])
@pytest.mark.parametrize("window_left", [-1, 127], ids=["full", "sliding"])
@pytest.mark.parametrize("seqlen_q", [1, 4], ids=["q1", "q4"])
def test_mha_decode_with_kvcache_distinct_v_head_dim(
    device: str,
    solution: str,
    has_sink: bool,
    window_left: int,
    seqlen_q: int,
    require,
) -> None:
    require("attention", "mha_decode_with_kvcache", solution, torch.bfloat16, "q")
    g = _DISTINCT_V_GEOMETRY
    page_size = 64
    kv_lens = [seqlen_q + n for n in (63, 300, 17, 191)]
    k_cache, page_table, k_rows = _paged_kv(
        kv_lens, page_size, (g["num_kv_heads"], g["head_dim"]), device
    )
    v_cache = torch.zeros(
        (*k_cache.shape[:3], g["v_head_dim"]), device=device, dtype=torch.bfloat16
    )
    v_rows = []
    for seq, kv_len in enumerate(kv_lens):
        pages = (kv_len + page_size - 1) // page_size
        rows = torch.randn(
            kv_len, g["num_kv_heads"], g["v_head_dim"], device=device
        ).bfloat16()
        padded = torch.zeros(
            (pages * page_size, *rows.shape[1:]), device=device, dtype=rows.dtype
        )
        padded[:kv_len] = rows
        v_cache[page_table[seq, :pages].long()] = padded.view(
            pages, page_size, *rows.shape[1:]
        )
        v_rows.append(rows)
    q = torch.randn(
        len(kv_lens) * seqlen_q, g["num_q_heads"], g["head_dim"], device=device
    ).bfloat16()
    sinks = (
        torch.randn(g["num_q_heads"], device=device).bfloat16() if has_sink else None
    )

    out = mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=torch.tensor(kv_lens, device=device, dtype=torch.int32),
        max_seqlen_k=max(kv_lens),
        max_seqlen_q=seqlen_q,
        window_left=window_left,
        sinks=sinks,
        solution=solution,
    )

    assert out.shape == (q.shape[0], g["num_q_heads"], g["v_head_dim"])
    expected = torch.cat(
        [
            _reference_gqa_attention(
                q[seq * seqlen_q : (seq + 1) * seqlen_q],
                k_rows[seq],
                v_rows[seq],
                kv_len - seqlen_q + torch.arange(seqlen_q, device=device),
                torch.arange(kv_len, device=device),
                window_left,
                sinks,
            )
            for seq, kv_len in enumerate(kv_lens)
        ]
    )
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("op", ["prefill", "extend", "decode"])
def test_mha_distinct_v_head_dim_skips_equal_width_only_kernels(op: str) -> None:
    """A kernel that has not opted in never serves a narrower value head."""
    from tokenspeed_kernel.registry import KernelRegistry
    from tokenspeed_kernel.selection import spec_matches_traits

    mode = {
        "prefill": "mha_prefill",
        "extend": "mha_extend_with_kvcache",
        "decode": "mha_decode_with_kvcache",
    }[op]
    specs = KernelRegistry.get().get_for_operator("attention", mode)
    serving = {
        spec.solution
        for spec in specs
        if spec_matches_traits(spec, {"distinct_v_head_dim": True})
    }
    assert serving <= {"triton", "fa3"}
    assert all("distinct_v_head_dim" in spec.traits for spec in specs)


@pytest.mark.parametrize(
    "is_causal,use_custom_mask",
    [(True, False), (False, False), (True, True)],
    ids=["causal", "noncausal", "custom-future"],
)
def test_mha_prefill_triton_window_bounds(
    device: str, require, is_causal: bool, use_custom_mask: bool
) -> None:
    from tokenspeed_kernel.ops.attention.mha._triton.prefill import (
        prefill_attention_fwd,
    )

    require("attention", "mha_extend_with_kvcache", "triton", torch.bfloat16, "q")
    # A cached suffix with unaligned window boundaries and a partial query tile.
    query_len, prefix_len, window_left, page_size = 257, 577, 511, 64
    kv_len = prefix_len + query_len
    num_pages = (kv_len + page_size - 1) // page_size
    q = _randn((query_len, 4, 64), device=device, dtype=torch.bfloat16)
    k = _randn((num_pages * page_size, 2, 64), device=device, dtype=q.dtype)
    v = _randn(k.shape, device=device, dtype=q.dtype)
    out = torch.empty_like(q)
    page_table = torch.arange(num_pages - 1, -1, -1, device=device, dtype=torch.int32)
    q_pos = prefix_len + torch.arange(query_len, device=device)
    k_pos = torch.arange(kv_len, device=device)
    mask = q_pos[:, None] - k_pos[None, :] <= window_left
    if use_custom_mask:
        # The custom mask replaces causality and selects a future key tile.
        mask &= k_pos[None, :] == kv_len - 1
    elif is_causal:
        mask &= q_pos[:, None] >= k_pos[None, :]

    # Call the shared kernel entry point because the public API has no custom mask.
    prefill_attention_fwd(
        q_extend=q,
        k_extend=k,
        v_extend=v,
        o_extend=out,
        k_buffer=k.view(num_pages, page_size, 2, 64).flip(0).flatten(0, 1),
        v_buffer=v.view(num_pages, page_size, 2, 64).flip(0).flatten(0, 1),
        cu_seqlens_q=torch.tensor([0, query_len], device=device, dtype=torch.int32),
        cache_seqlens=torch.tensor([kv_len], device=device, dtype=torch.int32),
        custom_mask=mask if use_custom_mask else None,
        is_causal=is_causal,
        max_len_extend=query_len,
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        logit_cap=0.0,
        skip_prefix_custom_mask=False,
        sliding_window_size=window_left,
        sinks=None,
        page_table=page_table,
        page_table_stride_b=num_pages,
        page_size=page_size,
        has_kv_cache=True,
        lse_extend=None,
    )

    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(0, 1),
        k[:kv_len].float().repeat_interleave(2, dim=1).transpose(0, 1),
        v[:kv_len].float().repeat_interleave(2, dim=1).transpose(0, 1),
        attn_mask=mask,
    ).transpose(0, 1)
    torch.testing.assert_close(out.float(), expected, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        pytest.param(torch.bfloat16, 64, 8, 2, id="bf16"),
        pytest.param(torch.bfloat16, 128, 8, 2, id="bf16-d128"),
        pytest.param(torch.float8_e4m3fn, 64, 8, 2, id="fp8"),
        pytest.param(torch.float8_e4m3fn, 128, 8, 2, id="fp8-d128"),
        pytest.param(torch.float8_e5m2, 64, 8, 2, id="fp8-e5m2"),
        pytest.param(torch.float8_e5m2, 128, 8, 2, id="fp8-e5m2-d128"),
    ],
)
@pytest.mark.parametrize("solution", ["triton", "fa3", "fa4", "flashinfer", "gluon"])
def test_mha_extend_with_kvcache(
    device: str,
    solution: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    require,
) -> None:
    require("attention", "mha_extend_with_kvcache", solution, dtype, "q")

    batch_size = 4
    page_size = 64
    max_cache_seqlen = 256
    prefix_seqlens_list = [63, 48, 17, 80]
    query_seqlens_list = [3, 1, 2, 4]
    max_query_seqlen = max(query_seqlens_list)
    max_cache_seqlen_used = max(
        prefix_len + query_len
        for prefix_len, query_len in zip(prefix_seqlens_list, query_seqlens_list)
    )
    prefix_seqlens = torch.tensor(prefix_seqlens_list, device=device, dtype=torch.int32)
    query_seqlens = torch.tensor(query_seqlens_list, device=device, dtype=torch.int32)
    cache_seqlens = prefix_seqlens + query_seqlens
    num_blocks_per_seq = (cache_seqlens + page_size - 1) // page_size
    max_num_blocks_per_seq = (max_cache_seqlen + page_size - 1) // page_size
    total_num_blocks = int(num_blocks_per_seq.sum().item())
    total_q = int(query_seqlens.sum().item())

    q = _randn((total_q, num_q_heads, head_dim), device=device, dtype=dtype)
    cu_seqlens_q = torch.cumsum(query_seqlens, dim=0, dtype=torch.int32)
    cu_seqlens_q = torch.nn.functional.pad(cu_seqlens_q, (1, 0))
    cu_seqlens_kv = torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
    cu_seqlens_kv = torch.nn.functional.pad(cu_seqlens_kv, (1, 0))

    page_table = torch.zeros(
        batch_size,
        max_num_blocks_per_seq,
        device=device,
        dtype=torch.int32,
    )
    next_block = 0
    for batch_idx, num_blocks in enumerate(num_blocks_per_seq.tolist()):
        page_table[batch_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            device=device,
            dtype=torch.int32,
        )
        next_block += num_blocks

    k_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    for batch_idx, total_kv_len in enumerate(cache_seqlens.tolist()):
        num_blocks = int(num_blocks_per_seq[batch_idx].item())
        for block_idx in range(num_blocks):
            physical_block = int(page_table[batch_idx, block_idx].item())
            block_start = block_idx * page_size
            tokens_in_block = min(page_size, total_kv_len - block_start)
            if tokens_in_block > 0:
                k_cache[physical_block, :tokens_in_block] = torch.randn(
                    tokens_in_block,
                    num_kv_heads,
                    head_dim,
                    device=device,
                    dtype=torch.bfloat16 if dtype in _FP8_DTYPES else dtype,
                ).to(dtype)
                v_cache[physical_block, :tokens_in_block] = torch.randn(
                    tokens_in_block,
                    num_kv_heads,
                    head_dim,
                    device=device,
                    dtype=torch.bfloat16 if dtype in _FP8_DTYPES else dtype,
                ).to(dtype)

    out = mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=max_query_seqlen,
        max_seqlen_k=max_cache_seqlen_used,
        solution=solution,
    )

    assert out.shape == q.shape

    if solution in ("triton", "gluon"):
        lse_out, lse = mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=prefix_seqlens,
            max_seqlen_q=max_query_seqlen,
            max_seqlen_k=int(prefix_seqlens.max().item()),
            return_lse=True,
            solution=solution,
        )

        assert lse_out.shape == q.shape
        assert lse.shape == (q.shape[0], q.shape[1])


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        pytest.param(torch.bfloat16, 64, 8, 2, id="bf16"),
        pytest.param(torch.bfloat16, 128, 8, 2, id="bf16-d128"),
        pytest.param(torch.float8_e4m3fn, 64, 8, 2, id="fp8"),
        pytest.param(torch.float8_e4m3fn, 128, 8, 2, id="fp8-d128"),
        pytest.param(torch.float8_e5m2, 64, 8, 2, id="fp8-e5m2"),
        pytest.param(torch.float8_e5m2, 128, 8, 2, id="fp8-e5m2-d128"),
    ],
)
@pytest.mark.parametrize("solution", ["triton", "fa3", "fa4", "flashinfer", "gluon"])
@pytest.mark.parametrize("seqlen_q", [1, 4], ids=["q1", "q4"])
def test_mha_decode_with_kvcache(
    device: str,
    solution: str,
    seqlen_q: int,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    require,
) -> None:
    require("attention", "mha_decode_with_kvcache", solution, dtype, "q")
    if solution == "gluon" and current_platform().is_cdna5 and seqlen_q != 1:
        pytest.skip("GFX1250 Gluon decode currently supports one query token")

    batch_size = 4
    page_size = 64
    max_cache_seqlen = 256
    prefix_seqlens = torch.tensor([63, 129, 17, 191], dtype=torch.int32)
    cache_seqlens = prefix_seqlens + seqlen_q
    num_blocks_per_seq = (cache_seqlens + page_size - 1) // page_size
    max_num_blocks_per_seq = (max_cache_seqlen + page_size - 1) // page_size
    total_num_blocks = int(num_blocks_per_seq.sum().item())

    # Build inputs on CPU for the SDPA reference below.
    q = _randn(
        (batch_size * seqlen_q, num_q_heads, head_dim),
        device="cpu",
        dtype=dtype,
    )

    page_table = torch.zeros(
        batch_size,
        max_num_blocks_per_seq,
        dtype=torch.int32,
    )
    next_block = 0
    for batch_idx, num_blocks in enumerate(num_blocks_per_seq.tolist()):
        page_table[batch_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            dtype=torch.int32,
        )
        next_block += num_blocks

    k_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
    )
    v_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
    )
    for batch_idx, total_kv_len in enumerate(cache_seqlens.tolist()):
        num_blocks = int(num_blocks_per_seq[batch_idx].item())
        for block_idx in range(num_blocks):
            physical_block = int(page_table[batch_idx, block_idx].item())
            block_start = block_idx * page_size
            tokens_in_block = min(page_size, total_kv_len - block_start)
            if tokens_in_block > 0:
                k_cache[physical_block, :tokens_in_block] = torch.randn(
                    tokens_in_block,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.bfloat16 if dtype in _FP8_DTYPES else dtype,
                ).to(dtype)
                v_cache[physical_block, :tokens_in_block] = torch.randn(
                    tokens_in_block,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.bfloat16 if dtype in _FP8_DTYPES else dtype,
                ).to(dtype)

    expected_out = None
    if seqlen_q == 1:
        group_size = num_q_heads // num_kv_heads
        expected = []
        for batch_idx, cache_len in enumerate(cache_seqlens.tolist()):
            num_blocks = int(num_blocks_per_seq[batch_idx].item())
            physical_blocks = page_table[batch_idx, :num_blocks].long()
            k_i = k_cache[physical_blocks].reshape(-1, num_kv_heads, head_dim)
            v_i = v_cache[physical_blocks].reshape(-1, num_kv_heads, head_dim)
            k_i = k_i[:cache_len].repeat_interleave(group_size, dim=1)
            v_i = v_i[:cache_len].repeat_interleave(group_size, dim=1)
            expected.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q[batch_idx : batch_idx + 1].float().unsqueeze(2),
                    k_i.float().permute(1, 0, 2).unsqueeze(0),
                    v_i.float().permute(1, 0, 2).unsqueeze(0),
                ).squeeze(2)
            )
        expected_out = torch.cat(expected, dim=0)

    q = q.to(device)
    k_cache = k_cache.to(device)
    v_cache = v_cache.to(device)
    page_table = page_table.to(device)
    cache_seqlens = cache_seqlens.to(device)

    out = mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max_cache_seqlen,
        max_seqlen_q=seqlen_q,
        solution=solution,
    )

    assert out.shape == q.shape
    assert not torch.isnan(out).any()
    expected_dtype = torch.bfloat16 if dtype in _FP8_DTYPES else dtype
    assert out.dtype == expected_dtype
    if expected_out is not None:
        tol = 3e-1 if dtype in _FP8_DTYPES else 3e-2
        torch.testing.assert_close(out.float().cpu(), expected_out, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "cache_seqlen,adversarial_inputs",
    [
        pytest.param(9 * 64, False, id="uneven-peeled-split"),
        pytest.param(64, True, id="inactive-peeled-tiles"),
    ],
)
def test_mha_decode_with_kvcache_gluon_peeled_split(
    device: str,
    cache_seqlen: int,
    adversarial_inputs: bool,
    require,
) -> None:
    require(
        "attention",
        "mha_decode_with_kvcache",
        "gluon",
        torch.bfloat16,
        "q",
    )

    batch_size = 1
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 64
    page_size = 64
    num_pages = 9
    max_seqlen_k = num_pages * page_size

    if adversarial_inputs:
        q = torch.full(
            (batch_size, num_q_heads, head_dim),
            16.0,
            dtype=torch.bfloat16,
        )
        k_cache = torch.full(
            (num_pages, page_size, num_kv_heads, head_dim),
            -16.0,
            dtype=torch.bfloat16,
        )
    else:
        q = torch.randn(
            batch_size,
            num_q_heads,
            head_dim,
            dtype=torch.bfloat16,
        )
        k_cache = torch.randn(
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
        )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.arange(num_pages, dtype=torch.int32).reshape(1, num_pages)
    cache_seqlens = torch.full((batch_size,), cache_seqlen, dtype=torch.int32)

    group_size = num_q_heads // num_kv_heads
    k_ref = k_cache.reshape(max_seqlen_k, num_kv_heads, head_dim)[:cache_seqlen]
    v_ref = v_cache.reshape(max_seqlen_k, num_kv_heads, head_dim)[:cache_seqlen]
    k_ref = k_ref.repeat_interleave(group_size, dim=1)
    v_ref = v_ref.repeat_interleave(group_size, dim=1)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(2),
        k_ref.permute(1, 0, 2).unsqueeze(0),
        v_ref.permute(1, 0, 2).unsqueeze(0),
    ).squeeze(2)

    out = mha_decode_with_kvcache(
        q=q.to(device),
        k_cache=k_cache.to(device),
        v_cache=v_cache.to(device),
        page_table=page_table.to(device),
        cache_seqlens=cache_seqlens.to(device),
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=1,
        solution="gluon",
    )

    assert out.shape == q.shape
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out.cpu(), expected, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("op", ["decode", "extend"])
@pytest.mark.parametrize("seqlen_q", [1, 4], ids=["q1", "q4"])
def test_fa4_mha_fp8_kvcache_matches_bf16_on_dequant_inputs(
    device: str,
    op: str,
    seqlen_q: int,
    require,
) -> None:
    """Dense-FP8 fa4 path vs the BF16 kernel on the same representable values.

    q/k/v are quantized to e4m3 and the BF16 reference runs on the exact
    dequantized values (e4m3 -> bf16 is lossless), so the comparison isolates
    the kernel's in-fp8 compute error from input quantization error.
    """
    mode = f"mha_{op}_with_kvcache"
    require("attention", mode, "fa4", torch.float8_e4m3fn, "q")

    torch.manual_seed(20260722)
    batch_size = 2
    page_size = 128
    num_q_heads, num_kv_heads, head_dim = 16, 16, 128  # MHA draft shape
    prefix_seqlens = torch.tensor([2305, 2205], device=device, dtype=torch.int32)
    cache_seqlens = prefix_seqlens + seqlen_q
    max_seqlen_k = int(cache_seqlens.max().item())
    blocks_per_seq = (max_seqlen_k + page_size - 1) // page_size
    page_table = torch.arange(
        batch_size * blocks_per_seq, device=device, dtype=torch.int32
    ).view(batch_size, blocks_per_seq)

    q8 = _randn(
        (batch_size * seqlen_q, num_q_heads, head_dim),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    k8 = _randn(
        (batch_size * blocks_per_seq, page_size, num_kv_heads, head_dim),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    v8 = _randn(
        (batch_size * blocks_per_seq, page_size, num_kv_heads, head_dim),
        device=device,
        dtype=torch.float8_e4m3fn,
    )

    kwargs = dict(
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=max_seqlen_k,
        solution="fa4",
    )
    if op == "extend":
        query_seqlens = torch.full(
            (batch_size,), seqlen_q, device=device, dtype=torch.int32
        )
        cu_seqlens_q = torch.nn.functional.pad(
            torch.cumsum(query_seqlens, dim=0, dtype=torch.int32), (1, 0)
        )
        cu_seqlens_kv = torch.nn.functional.pad(
            torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
        )
        kwargs.update(
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, is_causal=True
        )
        run = mha_extend_with_kvcache
    else:
        run = mha_decode_with_kvcache

    out_fp8 = run(q=q8, k_cache=k8, v_cache=v8, **kwargs)
    out_ref = run(
        q=q8.to(torch.bfloat16),
        k_cache=k8.to(torch.bfloat16),
        v_cache=v8.to(torch.bfloat16),
        **kwargs,
    )

    assert out_fp8.dtype == torch.bfloat16
    torch.testing.assert_close(out_fp8.float(), out_ref.float(), atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize(
    "dtype,head_dim,num_heads",
    [(torch.bfloat16, 64, 8)],
)
@pytest.mark.parametrize(
    "solution",
    ["triton", "cuda"],
)
@pytest.mark.parametrize("inplace", [False, True], ids=["out-of-place", "in-place"])
def test_attn_merge_state(
    device: str,
    solution: str,
    inplace: bool,
    dtype: torch.dtype,
    head_dim: int,
    num_heads: int,
    require,
) -> None:
    require("attention", "attn_merge_state", solution, dtype, "out_a")

    total_q = 31
    out_a = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
    out_b = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
    lse_a = torch.randn(total_q, num_heads, device=device, dtype=torch.float32)
    lse_b = torch.randn(total_q, num_heads, device=device, dtype=torch.float32)
    out_a_ref_input = out_a.clone()
    lse_a_ref_input = lse_a.clone()

    out, lse = attn_merge_state(
        out_a,
        lse_a,
        out_b,
        lse_b,
        inplace=inplace,
        solution=solution,
    )

    lse_ref = torch.maximum(lse_a_ref_input, lse_b)
    weight_a = torch.exp(lse_a_ref_input - lse_ref)
    weight_b = torch.exp(lse_b - lse_ref)
    denom = weight_a + weight_b
    out_ref = (
        out_a_ref_input.float() * weight_a[..., None]
        + out_b.float() * weight_b[..., None]
    ) / denom[..., None]
    lse_ref = lse_ref + torch.log(denom)

    if inplace:
        assert out.data_ptr() == out_a.data_ptr()
        assert lse.data_ptr() == lse_a.data_ptr()
    else:
        torch.testing.assert_close(out_a, out_a_ref_input)
        torch.testing.assert_close(lse_a, lse_a_ref_input)

    assert out.shape == out_a.shape
    assert lse.shape == lse_a.shape
    torch.testing.assert_close(out.float(), out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-5, atol=1e-5)
