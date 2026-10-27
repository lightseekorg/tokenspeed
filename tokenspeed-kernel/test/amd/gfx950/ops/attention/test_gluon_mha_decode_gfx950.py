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

"""Numerics and graph replay for split-K MHA decode on gfx950."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.mha import torch_dense_attention
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
    get_config,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (  # noqa: E402
    launch_gluon_mha_decode_gfx950 as gluon_mha_decode_gfx950,
)

HEAD_DIM = 64
Q_HEADS = 32
KV_HEADS = 4
PAGE_SIZE = 64
SEQUENCE_LENGTH = 512
WINDOW_LEFT = SEQUENCE_LENGTH - 1


def _reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    pages_per_request = SEQUENCE_LENGTH // PAGE_SIZE
    outputs = []
    for batch_index in range(q.shape[0]):
        pages = slice(
            batch_index * pages_per_request, (batch_index + 1) * pages_per_request
        )
        out, _ = torch_dense_attention(
            q[batch_index, None],
            k_cache[pages].flatten(0, 1),
            v_cache[pages].flatten(0, 1),
            scale=HEAD_DIM**-0.5,
            causal_offset=None,
            window_left=None,
            sinks=sinks,
        )
        outputs.append(out[0])
    return torch.stack(outputs).to(torch.bfloat16)


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
def test_split_sliding_decode_matches_reference_and_replays(batch: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260915 + batch)
    pages_per_request = SEQUENCE_LENGTH // PAGE_SIZE
    q = torch.randn(
        batch,
        Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k_cache = torch.randn(
        batch * pages_per_request,
        PAGE_SIZE,
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    v_cache = torch.randn(
        k_cache.shape,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    page_table = torch.arange(
        batch * pages_per_request,
        dtype=torch.int32,
        device="cuda",
    ).reshape(batch, pages_per_request)
    cache_seqlens = torch.full(
        (batch,),
        SEQUENCE_LENGTH,
        dtype=torch.int32,
        device="cuda",
    )
    sinks = torch.randn(
        Q_HEADS, dtype=torch.float32, device="cuda", generator=generator
    )
    config = get_config(
        q=q,
        k_cache=k_cache,
        max_seqlen_k=SEQUENCE_LENGTH,
        window_left=WINDOW_LEFT,
        softmax_scale=None,
    )
    expected_splits = 8 if batch <= 4 else 4
    assert config.num_kv_splits == expected_splits

    def invoke() -> torch.Tensor:
        return gluon_mha_decode_gfx950(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=SEQUENCE_LENGTH,
            max_seqlen_q=1,
            window_left=WINDOW_LEFT,
            logit_cap=0.0,
            sinks=sinks,
            return_lse=False,
            softmax_scale=None,
            q_scale=None,
            k_scale=None,
            v_scale=None,
        )

    expected = _reference(q, k_cache, v_cache, sinks)
    actual = invoke()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = invoke()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(replayed, actual, atol=0, rtol=0)


# Few KV heads per rank, so a small batch leaves the grid short of base CTAs.
FULL_Q_HEADS = 16
FULL_KV_HEADS = 2
# The paged backend passes the model's maximum context, not the batch maximum.
FULL_MAX_SEQLEN_K = 1 << 20


def test_full_decode_config_keeps_prefetch_within_vmcnt() -> None:
    """BF16 head dim 128 would prefetch more copies than s_waitcnt vmcnt counts."""
    q = torch.empty(1, FULL_Q_HEADS, 128, dtype=torch.bfloat16)
    k_cache = torch.empty(1, PAGE_SIZE, FULL_KV_HEADS, 128, dtype=torch.bfloat16)
    config = get_config(
        q=q,
        k_cache=k_cache,
        max_seqlen_k=FULL_MAX_SEQLEN_K,
        window_left=-1,
        softmax_scale=None,
    )
    assert (config.num_kv_splits, config.num_stages) == (64, 1)


@pytest.mark.parametrize(
    "cache_lens,num_stages",
    [
        # 0: a padded graph row; 1: one token; 129: a partial third page;
        # 5 * 64 and 7 * 64 + 1 straddle loop iterations; 300 * 64 + 17 spans
        # many splits. Six rows fill enough of the grid for the single buffer.
        pytest.param(
            [0, 1, 129, 5 * PAGE_SIZE, 7 * PAGE_SIZE + 1, 300 * PAGE_SIZE + 17],
            1,
            id="single-buffer",
        ),
        # Four rows leave the grid short, so long splits pipeline their page
        # loads: 256 pages give each split exactly one loop iteration of
        # pages, 301 pages repeat the last page in prefetches past the split,
        # and 1000 * 64 + 33 ends in a partial page.
        pytest.param(
            [0, 256 * PAGE_SIZE, 300 * PAGE_SIZE + 17, 1000 * PAGE_SIZE + 33],
            4,
            id="pipelined",
        ),
    ],
)
def test_full_decode_ragged_rows_match_reference(
    cache_lens: list[int], num_stages: int
) -> None:
    """Full-attention decode over empty, short, unaligned and long rows.

    Page-table entries past each request hold out-of-range page ids, so a copy
    that reads past a request's pages faults instead of passing silently.
    """
    generator = torch.Generator(device="cuda").manual_seed(20261025 + num_stages)
    pages = [-(-length // PAGE_SIZE) for length in cache_lens]
    table_width = max(pages)
    total_pages = sum(pages) + 1
    k_cache = torch.randn(
        total_pages,
        PAGE_SIZE,
        FULL_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    v_cache = torch.randn(
        k_cache.shape, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    page_table = torch.full(
        (len(cache_lens), table_width), 1 << 30, dtype=torch.int32, device="cuda"
    )
    # Shuffled physical pages; page 0 only backs the row without cached tokens.
    physical = torch.randperm(
        total_pages - 1, generator=torch.Generator().manual_seed(7)
    )
    physical = (physical + 1).to(torch.int32).cuda()
    start = 0
    for row, count in enumerate(pages):
        page_table[row, :count] = physical[start : start + count]
        start += count
    cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device="cuda")
    q = torch.randn(
        len(cache_lens),
        FULL_Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    config = get_config(
        q=q,
        k_cache=k_cache,
        max_seqlen_k=FULL_MAX_SEQLEN_K,
        window_left=-1,
        softmax_scale=None,
    )
    assert config.num_stages == num_stages

    def invoke() -> torch.Tensor:
        return gluon_mha_decode_gfx950(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=FULL_MAX_SEQLEN_K,
        )

    actual = invoke()
    torch.cuda.synchronize()
    for row, length in enumerate(cache_lens):
        if length == 0:
            continue
        row_pages = page_table[row, : pages[row]].long()
        expected, _ = torch_dense_attention(
            q[row, None],
            k_cache[row_pages].flatten(0, 1)[:length],
            v_cache[row_pages].flatten(0, 1)[:length],
            scale=HEAD_DIM**-0.5,
            causal_offset=None,
            window_left=None,
            sinks=None,
        )
        # FP32 accumulation and one BF16 rounding of the output.
        torch.testing.assert_close(
            actual[row].float(), expected[0].float(), atol=2e-3, rtol=2e-3
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = invoke()
    graph.replay()
    torch.cuda.synchronize()
    nonempty = cache_seqlens > 0
    torch.testing.assert_close(replayed[nonempty], actual[nonempty], atol=0, rtol=0)


# One TP8 rank of Muse Spark's attention: 16 query heads over 2 KV heads.
VERIFY_Q_HEADS = 16
VERIFY_KV_HEADS = 2
VERIFY_TOKENS = 8


@pytest.mark.parametrize("window_left", [2047, -1], ids=["sliding", "full"])
def test_verify_block_rows_are_causal_within_the_block(window_left: int) -> None:
    """Speculative verify: each request's last ``VERIFY_TOKENS`` cached tokens
    are queries, and query ``i`` sees the cache up to and including itself.

    Lengths cover a block shorter than the window, one whose window starts
    mid-page, and one that slides past many pages.
    """
    cache_lens = [VERIFY_TOKENS + 3, 2048 + VERIFY_TOKENS + 37, 300 * PAGE_SIZE + 5]
    generator = torch.Generator(device="cuda").manual_seed(20261027)
    pages = [-(-length // PAGE_SIZE) for length in cache_lens]
    k_cache = torch.randn(
        sum(pages),
        PAGE_SIZE,
        VERIFY_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    v_cache = torch.randn(
        k_cache.shape, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    page_table = torch.full(
        (len(cache_lens), max(pages)), 1 << 30, dtype=torch.int32, device="cuda"
    )
    physical = torch.randperm(
        sum(pages), generator=torch.Generator().manual_seed(11)
    ).to(dtype=torch.int32, device="cuda")
    start = 0
    for row, count in enumerate(pages):
        page_table[row, :count] = physical[start : start + count]
        start += count
    cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device="cuda")
    q = torch.randn(
        len(cache_lens) * VERIFY_TOKENS,
        VERIFY_Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    actual = gluon_mha_decode_gfx950(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=FULL_MAX_SEQLEN_K,
        max_seqlen_q=VERIFY_TOKENS,
        window_left=window_left,
    )
    torch.cuda.synchronize()
    for row, length in enumerate(cache_lens):
        row_pages = page_table[row, : pages[row]].long()
        expected, _ = torch_dense_attention(
            q[row * VERIFY_TOKENS : (row + 1) * VERIFY_TOKENS],
            k_cache[row_pages].flatten(0, 1)[:length],
            v_cache[row_pages].flatten(0, 1)[:length],
            scale=HEAD_DIM**-0.5,
            causal_offset=length - VERIFY_TOKENS,
            window_left=None if window_left < 0 else window_left,
            sinks=None,
        )
        # Short rows attend to few keys and reach |out| > 1, where one BF16
        # rounding of the FP32 result is up to one BF16 ULP (2**-7) relative.
        torch.testing.assert_close(
            actual[row * VERIFY_TOKENS : (row + 1) * VERIFY_TOKENS].float(),
            expected.float(),
            atol=2e-3,
            rtol=2**-7,
        )
