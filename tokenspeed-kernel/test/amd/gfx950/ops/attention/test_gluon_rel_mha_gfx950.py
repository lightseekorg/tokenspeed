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

"""Large KV pools for relative-bias MHA decode and extend on gfx950."""

from __future__ import annotations

import pytest
import torch
from rel_mha_helpers import (
    DTYPE,
    HEAD_DIM,
    NUM_Q_HEADS,
    build_paged,
    cu_seqlens,
)
from utils import is_cdna4, move_pages_past_int32

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.rmha import (  # noqa: E402
    rel_mha_decode_with_kvcache,
    rel_mha_extend_with_kvcache,
)

PAGE = 256
REL_EXTENT = 512

# Each KV page base is computed in 64-bit. Pages past 2**31 elements must read
# exactly what the same pages read at the start of a small pool.


@pytest.mark.parametrize("window_left", [-1, 255])
def test_decode_reads_pages_past_int32_range(window_left: int) -> None:
    torch.manual_seed(20261029)
    kv_lens = [255, 256, 300, 511]
    batch = len(kv_lens)
    q = torch.randn(batch, NUM_Q_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE) * 0.5
    rel_logits = (
        torch.randn(batch, NUM_Q_HEADS, REL_EXTENT, device="cuda", dtype=DTYPE) * 0.5
    )
    cu_seqlens_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    cache_seqlens = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    k_cache, v_cache, page_table, _, _ = build_paged(kv_lens, "cuda", PAGE)

    def decode(k, v, table):
        return rel_mha_decode_with_kvcache(
            q=q,
            k_cache=k,
            v_cache=v,
            page_table=table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max(kv_lens),
            rel_logits=rel_logits,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            window_left=window_left,
            softmax_scale=1.0 / HEAD_DIM,
            solution="gluon",
        )

    expected = decode(k_cache, v_cache, page_table)
    (k_far, v_far), far_table = move_pages_past_int32((k_cache, v_cache), page_table)
    actual = decode(k_far, v_far, far_table)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("window_left", [-1, 255])
def test_extend_reads_pages_past_int32_range(window_left: int) -> None:
    torch.manual_seed(20261029)
    q_lens, kv_lens = [64, 32], [300, 511]
    total_q = sum(q_lens)
    q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE) * 0.5
    rel_logits = (
        torch.randn(total_q, NUM_Q_HEADS, REL_EXTENT, device="cuda", dtype=DTYPE)
        * 0.5
    )
    cu_q = cu_seqlens(q_lens, "cuda")
    cu_kv = cu_seqlens(kv_lens, "cuda")
    cache_seqlens = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    k_cache, v_cache, page_table, _, _ = build_paged(kv_lens, "cuda", PAGE)

    def extend(k, v, table):
        return rel_mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            k_cache=k,
            v_cache=v,
            page_table=table,
            cache_seqlens=cache_seqlens,
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(kv_lens),
            rel_logits=rel_logits,
            window_left=window_left,
            softmax_scale=1.0 / HEAD_DIM,
            solution="gluon",
        )

    expected = extend(k_cache, v_cache, page_table)
    (k_far, v_far), far_table = move_pages_past_int32((k_cache, v_cache), page_table)
    actual = extend(k_far, v_far, far_table)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
