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

"""GFX950 query-row MLA decode: BF16 or FP8 queries over an FP8 cache, every
query row of a request with its own visible length (a KVP rank's local
lengths), against an FP32 softmax over the dequantized operands and against
the folded ``bh16bn128`` decode the KVP drafter ran before."""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required for Gluon MLA tests", allow_module_level=True)

from tokenspeed_kernel.ops.attention.mla import (  # noqa: E402
    mla_decode_query_rows,
    supports_mla_decode_query_rows,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.mla import decode  # noqa: E402

_PAGE = 64
_SCALE = 192**-0.5
_MAX_K = 1 << 20
_FP8 = torch.float8_e4m3fn
_DTYPES = [
    (torch.bfloat16, torch.float8_e4m3fn),
    (torch.bfloat16, torch.float8_e5m2),
    (_FP8, _FP8),
]


def _out_tol(q_dtype):
    # FP8 queries also round the probabilities to FP8 before the PV product.
    tol = 5e-2 if q_dtype == _FP8 else 2e-2
    return dict(atol=tol, rtol=tol)


def _inputs(lengths, heads, q_dtype=torch.bfloat16, kv_dtype=_FP8, seed=0):
    lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    batch, queries = lengths.shape
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages_per_request = max(1, -(-int(lengths.max()) // _PAGE)) + 1
    pages = batch * pages_per_request + 3
    kv = (torch.randn(pages, _PAGE, 1, 576, device="cuda", generator=g) * 0.5).to(
        kv_dtype
    )
    table = (
        torch.randperm(pages, device="cuda", generator=g)[: batch * pages_per_request]
        .int()
        .view(batch, pages_per_request)
        .contiguous()
    )
    q = torch.randn(
        batch, queries, heads, 576, device="cuda", dtype=torch.bfloat16, generator=g
    ).to(q_dtype)
    return q, kv, table, lengths.contiguous()


def _reference(q, kv, table, lengths):
    out = torch.zeros(*q.shape[:3], 512, device="cuda")
    lse = torch.full(q.shape[:3], -float("inf"), device="cuda")
    for b in range(q.shape[0]):
        for j in range(q.shape[1]):
            n = int(lengths[b, j])
            if n == 0:
                continue
            keys = kv[table[b, : -(-n // _PAGE)].long()].reshape(-1, 576)[:n].float()
            logits = q[b, j].float() @ keys.T * _SCALE
            lse[b, j] = torch.logsumexp(logits, -1)
            out[b, j] = torch.softmax(logits, -1) @ keys[:, :512]
    return out, lse


@pytest.mark.parametrize(
    "lengths,heads",
    [
        ([[3000, 3001, 3001, 3002], [64, 64, 65, 66], [1, 2, 3, 4]], 96),
        ([[2500], [7000]], 96),
        ([[0, 0, 0, 1], [129, 129, 129, 129]], 96),
        ([[5000, 5001, 5002], [300, 300, 301]], 32),
        ([[4096] * 8, [17] * 8], 16),
    ],
)
@pytest.mark.parametrize("q_dtype,kv_dtype", _DTYPES)
def test_matches_fp32_softmax(lengths, heads, q_dtype, kv_dtype):
    q, kv, table, row_lengths = _inputs(lengths, heads, q_dtype, kv_dtype)
    assert supports_mla_decode_query_rows(
        q_dtype=q.dtype,
        kv_dtype=kv.dtype,
        page_size=_PAGE,
        q_len=q.shape[1],
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        return_lse=True,
    )
    out, lse = mla_decode_query_rows(
        q, kv, table, row_lengths, _MAX_K, _SCALE, kv_lora_rank=512, return_lse=True
    )
    ref_out, ref_lse = _reference(q, kv, table, row_lengths)
    torch.testing.assert_close(out.float(), ref_out, **_out_tol(q_dtype))
    visible = row_lengths > 0
    torch.testing.assert_close(lse[visible], ref_lse[visible], atol=1e-3, rtol=1e-4)
    assert torch.isneginf(lse[~visible]).all()
    assert (out[~visible] == 0).all()


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, _FP8])
@pytest.mark.parametrize("splits", [1, 3, 7, 32])
def test_every_split_count_merges_alike(splits, q_dtype, monkeypatch):
    q, kv, table, row_lengths = _inputs([[9000, 9001], [70, 4000]], 96, q_dtype, seed=3)
    ref_out, ref_lse = _reference(q, kv, table, row_lengths)
    monkeypatch.setattr(decode, "_select_num_kv_splits_query_rows", lambda **_: splits)
    out, lse = decode.launch_gluon_mla_decode_query_rows_gfx950(
        q, kv, table, row_lengths, _MAX_K, _SCALE, return_lse=True
    )
    torch.testing.assert_close(out.float(), ref_out, **_out_tol(q_dtype))
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, _FP8])
def test_matches_the_folded_decode(q_dtype):
    """The rows the KVP drafter folded onto the batch axis, 16 heads each."""
    q, kv, table, row_lengths = _inputs(
        [[6000, 6001, 6002, 6003], [800, 800, 800, 801]], 96, q_dtype, seed=7
    )
    out, lse = mla_decode_query_rows(
        q, kv, table, row_lengths, _MAX_K, _SCALE, kv_lora_rank=512, return_lse=True
    )
    batch, queries, heads, _ = q.shape
    groups = heads // 16
    folded_decode = (
        decode.gluon_mla_decode_fp8xfp8_gfx950
        if q_dtype == _FP8
        else decode.gluon_mla_decode_bf16xfp8_gfx950
    )
    folded, folded_lse = folded_decode(
        q.reshape(batch * queries * groups, 1, 16, 576),
        kv,
        table.repeat_interleave(queries * groups, 0).contiguous(),
        row_lengths.reshape(-1).repeat_interleave(groups).contiguous(),
        _MAX_K,
        128,
        512,
        64,
        _SCALE,
        return_lse=True,
    )
    torch.testing.assert_close(
        out.float(),
        folded.reshape(batch, queries, heads, 512).float(),
        **(_out_tol(q_dtype) if q_dtype == _FP8 else dict(atol=1e-2, rtol=1e-2)),
    )
    torch.testing.assert_close(
        lse, folded_lse.reshape(batch, queries, heads), atol=1e-3, rtol=1e-4
    )


def test_split_count_fills_whole_waves():
    select = decode._select_num_kv_splits_query_rows
    assert select(batch=16, row_tiles=6, max_seqlen_k=_MAX_K) == 8
    assert select(batch=1, row_tiles=6, max_seqlen_k=_MAX_K) == 32
    assert select(batch=16, row_tiles=6, max_seqlen_k=1024) == 2
    assert select(batch=64, row_tiles=6, max_seqlen_k=_MAX_K) == 2


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, _FP8])
def test_large_pool_uses_safe_64_bit_addresses(q_dtype):
    """Pages past the first 2^31 bytes of the pool read the right rows."""
    q, kv, table, row_lengths = _inputs(
        [[2049, 2050, 2051], [130, 130, 131]], 96, q_dtype, seed=5
    )
    pool_pages = 0x80000000 // (kv[0].numel() * kv.element_size()) + 1
    large_kv = torch.empty((pool_pages, *kv.shape[1:]), dtype=kv.dtype, device="cuda")
    large_kv[-kv.shape[0] :].copy_(kv)
    high_table = table + (pool_pages - kv.shape[0])
    out, lse = mla_decode_query_rows(
        q,
        large_kv,
        high_table,
        row_lengths,
        _MAX_K,
        _SCALE,
        kv_lora_rank=512,
        return_lse=True,
    )
    ref_out, ref_lse = _reference(q, kv, table, row_lengths)
    torch.testing.assert_close(out.float(), ref_out, **_out_tol(q_dtype))
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, _FP8])
def test_batch_and_table_width_reuse_one_binary(q_dtype, monkeypatch):
    """Serving varies the batch and the page-table width at a fixed split count."""
    monkeypatch.setattr(decode, "_select_num_kv_splits_query_rows", lambda **_: 4)

    def run(batch, table_pages, seed):
        q, kv, table, row_lengths = _inputs(
            [[257 + 64 * i, 300 + 64 * i] for i in range(batch)], 32, q_dtype, seed=seed
        )
        wide = torch.zeros((batch, table_pages), dtype=torch.int32, device="cuda")
        wide[:, : table.shape[1]] = table
        decode.launch_gluon_mla_decode_query_rows_gfx950(
            q, kv, wide, row_lengths, _MAX_K, _SCALE, return_lse=True
        )

    run(2, 16, 0)
    torch.cuda.synchronize()
    with assert_no_triton_compile(
        decode.gluon_mla_decode_query_rows_gfx950,
        decode.gluon_mla_decode_query_rows_reduce_gfx950,
    ):
        for step, (batch, table_pages) in enumerate(
            ((1, 9), (3, 17), (8, 32), (5, 33))
        ):
            run(batch, table_pages, step + 1)
        torch.cuda.synchronize()
