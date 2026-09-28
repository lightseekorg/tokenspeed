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

"""Coverage for the gfx950 absorbed MLA extend kernel.

Calls name the kernel through ``override`` so every dtype pair runs at every
shape, including query lengths the registry would send to prefix replay.
"""

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.ops.attention.mla import mla_extend_with_kvcache
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile

platform = current_platform()
pytestmark = pytest.mark.skipif(not platform.is_cdna4, reason="gfx950 MLA extend")

_KV_LORA_RANK = 512
_ROPE_DIM = 64
_HEAD_DIM = _KV_LORA_RANK + _ROPE_DIM
_PAGE_SIZE = 64
_SCALE = 1.0 / math.sqrt(128 + _ROPE_DIM)
_FP8 = torch.float8_e4m3fn
# (q dtype, kv dtype) -> registered kernel.
_KERNELS = {
    (_FP8, _FP8): "gluon_mla_extend_gfx950",
    (torch.bfloat16, _FP8): "gluon_mla_extend_gfx950",
    (torch.bfloat16, torch.bfloat16): "gluon_mla_extend_bf16_gfx950",
}
_DTYPES = [
    pytest.param(dtypes, id=f"{str(dtypes[0])[6:]}x{str(dtypes[1])[6:]}")
    for dtypes in _KERNELS
]
# FP8 rounds P to E4M3 before the P @ V MFMA.
_ATOL = {_FP8: 6e-2, torch.bfloat16: 1e-2}


def _make_case(q_lens, prefix_lens, num_heads, q_dtype, kv_dtype, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cache_lens = [p + q for p, q in zip(prefix_lens, q_lens, strict=True)]
    pages = [math.ceil(c / _PAGE_SIZE) for c in cache_lens]
    num_pages = sum(pages) + 2
    # Half-scale samples keep the logits in the few-unit range of real models.
    kv = (
        torch.randn(num_pages, _PAGE_SIZE, 1, _HEAD_DIM, generator=gen, device="cuda")
        * 0.5
    ).to(kv_dtype)
    if kv_dtype == _FP8:
        # Slots past each cache end hold NaN; masking must never read them.
        kv.view(torch.uint8)[-1] = 0x7F
    # Shuffled pages so the kernel follows the page table.
    perm = torch.randperm(num_pages - 1, generator=gen, device="cuda")
    page_table = torch.full(
        (len(q_lens), max(pages)), num_pages - 1, dtype=torch.int32, device="cuda"
    )
    start = 0
    for b, n in enumerate(pages):
        page_table[b, :n] = perm[start : start + n].to(torch.int32)
        start += n
        tail = cache_lens[b] % _PAGE_SIZE
        if kv_dtype == _FP8 and tail:
            kv.view(torch.uint8)[page_table[b, n - 1], tail:] = 0x7F
    q = (
        torch.randn(sum(q_lens), num_heads, _HEAD_DIM, generator=gen, device="cuda")
        * 0.5
    )
    cu_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    cu_k = torch.tensor(
        [0, *torch.tensor(cache_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    return dict(
        q=q.to(q_dtype),
        kv_cache=kv,
        page_table=page_table,
        cache_seqlens=torch.tensor(cache_lens, dtype=torch.int32, device="cuda"),
        cu_seqlens_q=cu_q.cuda(),
        cu_seqlens_kv=cu_k.cuda(),
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(cache_lens),
    )


def _run(case, kernel, out=None):
    return mla_extend_with_kvcache(
        **case,
        qk_nope_head_dim=128,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_ROPE_DIM,
        softmax_scale=_SCALE,
        is_causal=True,
        out=out,
        override=kernel,
    )


def _reference(case):
    # FP8-cache kernels cast a BF16 query to the cache dtype on load.
    q = case["q"].to(case["kv_cache"].dtype).float()
    kv = case["kv_cache"].float()
    refs = []
    cu_q = case["cu_seqlens_q"].tolist()
    for b, cache_len in enumerate(case["cache_seqlens"].tolist()):
        n = math.ceil(cache_len / _PAGE_SIZE)
        k = kv[case["page_table"][b, :n].long()].reshape(-1, _HEAD_DIM)[:cache_len]
        q_b = q[cu_q[b] : cu_q[b + 1]]
        q_len = q_b.shape[0]
        scores = torch.einsum("qhd,kd->qhk", q_b, k) * _SCALE
        last = cache_len - q_len + torch.arange(q_len, device=q.device)
        visible = torch.arange(cache_len, device=q.device)[None, :] <= last[:, None]
        scores = scores.masked_fill(~visible[:, None, :], -float("inf"))
        refs.append(
            torch.einsum("qhk,kd->qhd", scores.softmax(-1), k[:, :_KV_LORA_RANK])
        )
    return torch.cat(refs)


@pytest.mark.parametrize("dtypes", _DTYPES)
@pytest.mark.parametrize(
    "q_lens,prefix_lens,num_heads",
    [
        # Initial prefill and a tiny prefix, packed into one program.
        ([3, 2], [0, 5], 12),
        # Decode-like chunk over a long prefix: the KV stream is split.
        ([16], [4000], 12),
        # Ragged batch with q blocks and caches ending mid-page.
        ([37, 5, 130], [100, 3000, 64], 12),
        # Kimi K3 at TP4 / TP2 / TP1 (24 / 48 / 96 local heads).
        ([64], [130], 24),
        ([10, 7], [8192, 0], 48),
        ([4], [2000], 96),
        # A head count without a large divisor falls back to small groups.
        ([200, 1], [1000, 70], 7),
    ],
)
def test_mla_extend_gfx950_matches_reference(
    device, dtypes, q_lens, prefix_lens, num_heads
):
    q_dtype, kv_dtype = dtypes
    case = _make_case(
        q_lens, prefix_lens, num_heads, q_dtype, kv_dtype, seed=len(q_lens)
    )
    out = _run(case, _KERNELS[dtypes])
    assert out.dtype == torch.bfloat16
    assert not out.isnan().any()
    torch.testing.assert_close(
        out.float(), _reference(case), atol=_ATOL[kv_dtype], rtol=0
    )


@pytest.mark.parametrize("dtypes", _DTYPES)
def test_mla_extend_gfx950_repeatable_and_writes_out(device, dtypes):
    q_dtype, kv_dtype = dtypes
    for seed in range(3):
        case = _make_case([48, 9], [700, 2500], 12, q_dtype, kv_dtype, seed=seed)
        out = torch.full(
            (57, 12, _KV_LORA_RANK), float("nan"), dtype=torch.bfloat16, device=device
        )
        first = _run(case, _KERNELS[dtypes], out=out)
        assert first.data_ptr() == out.data_ptr()
        torch.testing.assert_close(
            first.float(), _reference(case), atol=_ATOL[kv_dtype], rtol=0
        )
        again = _run(case, _KERNELS[dtypes])
        torch.testing.assert_close(again, first, atol=0, rtol=0)


def test_mla_extend_gfx950_no_recompile_across_shapes(device):
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.extend import (
        gluon_mla_extend_gfx950,
        gluon_mla_extend_reduce_gfx950,
    )

    kernel = _KERNELS[(_FP8, _FP8)]
    # Warm both split and unsplit launches, with runtime scalars divisible by
    # 16 and not.
    for q_lens, prefix_lens in (([16], [4000]), ([5, 3], [4000, 70]), ([300], [64])):
        _run(_make_case(q_lens, prefix_lens, 12, _FP8, _FP8, seed=0), kernel)
    with assert_no_triton_compile(gluon_mla_extend_gfx950), assert_no_triton_compile(
        gluon_mla_extend_reduce_gfx950
    ):
        for q_lens, prefix_lens in (
            ([7], [9000]),
            ([33, 2, 90], [1500, 64, 0]),
            ([256], [2048]),
            ([1, 1, 1, 1], [100, 200, 300, 400]),
        ):
            _run(_make_case(q_lens, prefix_lens, 12, _FP8, _FP8, seed=1), kernel)
