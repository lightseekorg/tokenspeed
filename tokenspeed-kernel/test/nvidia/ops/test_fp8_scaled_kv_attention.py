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

"""Paged attention over a per-tensor scaled FP8 KV cache.

The cache holds K / k_scale and V / v_scale. Reading it at its scales equals
reading the same bytes at unit scale with k_scale folded into the softmax scale
and the output multiplied by v_scale; power-of-two scales keep both exact.
"""

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import NoKernelFoundError

platform = current_platform()

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    pytest.mark.skipif(
        not platform.is_blackwell, reason="trtllm-gen attention requires SM10x"
    ),
]

PAGE = 64
HEAD_DIM = 128
HEADS_Q = 16
HEADS_KV = 2
FP8 = torch.float8_e4m3fn
K_SCALE, V_SCALE = 2.0**-4, 2.0**-3
UNIT_SOFTMAX_SCALE = K_SCALE / math.sqrt(HEAD_DIM)


def _paged_cache(seq_lens: list[int], seed: int) -> dict:
    """Per-request contiguous pages of FP8 rows."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = [math.ceil(s / PAGE) for s in seq_lens]
    shape = (sum(pages), PAGE, HEADS_KV, HEAD_DIM)
    out = {
        name: (torch.randn(shape, device="cuda", generator=g) * 64).to(FP8)
        for name in ("k_cache", "v_cache")
    }
    table = torch.zeros(len(seq_lens), max(pages), dtype=torch.int32, device="cuda")
    start = 0
    for b, n in enumerate(pages):
        table[b, :n] = torch.arange(start, start + n, device="cuda")
        start += n
    out["page_table"] = table
    out["cache_seqlens"] = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    return out


@pytest.mark.parametrize("q_len", [1, 4])
def test_decode_reads_a_scaled_fp8_cache(q_len: int):
    from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache

    seq_lens = [700, 129, 1500]
    g = torch.Generator(device="cuda").manual_seed(4)
    q = torch.randn(
        len(seq_lens) * q_len, HEADS_Q, HEAD_DIM, device="cuda", generator=g
    ).to(FP8)
    common = dict(
        q=q, max_seqlen_k=2048, max_seqlen_q=q_len, **_paged_cache(seq_lens, seed=3)
    )
    out = mha_decode_with_kvcache(fp8_kv_scales=(K_SCALE, V_SCALE), **common)
    unit = mha_decode_with_kvcache(
        softmax_scale=UNIT_SOFTMAX_SCALE, solution="flashinfer", **common
    )
    assert torch.equal(out, unit * V_SCALE)


def test_extend_reads_a_scaled_fp8_cache():
    from tokenspeed_kernel.ops.attention.mha import mha_extend_with_kvcache

    seq_lens = [300, 77, 1024]
    extend = [40, 77, 9]
    g = torch.Generator(device="cuda").manual_seed(6)
    q = torch.randn(sum(extend), HEADS_Q, HEAD_DIM, device="cuda", generator=g).to(FP8)
    cu_q = torch.tensor([0, *torch.tensor(extend).cumsum(0).tolist()], device="cuda")
    cu_kv = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], device="cuda")
    common = dict(
        q=q,
        cu_seqlens_q=cu_q.to(torch.int32),
        cu_seqlens_kv=cu_kv.to(torch.int32),
        max_seqlen_q=max(extend),
        max_seqlen_k=2048,
        is_causal=True,
        **_paged_cache(seq_lens, seed=5),
    )
    out = mha_extend_with_kvcache(fp8_kv_scales=(K_SCALE, V_SCALE), **common)
    unit = mha_extend_with_kvcache(
        softmax_scale=UNIT_SOFTMAX_SCALE, solution="flashinfer", **common
    )
    assert torch.equal(out, unit * V_SCALE)


def test_only_kernels_serving_scaled_caches_take_one():
    """FA4's public entry takes no value descale, so it declines a scaled cache."""
    from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache

    with pytest.raises(NoKernelFoundError):
        mha_decode_with_kvcache(
            q=torch.zeros(1, HEADS_Q, HEAD_DIM, device="cuda").to(FP8),
            max_seqlen_k=64,
            max_seqlen_q=1,
            fp8_kv_scales=(K_SCALE, V_SCALE),
            solution="fa4",
            **_paged_cache([64], seed=7),
        )
