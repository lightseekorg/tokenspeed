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

"""Numerics, graph replay and large KV pools for MHA decode on gfx950."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.mha import torch_dense_attention
from utils import assert_no_triton_compile, is_cdna4, move_pages_past_int32

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.attention.mha import decode
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


def _make_inputs(batch: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
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
    return q, k_cache, v_cache, page_table, cache_seqlens, sinks


def _decode(q, k_cache, v_cache, page_table, cache_seqlens, sinks, window_left):
    return gluon_mha_decode_gfx950(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=SEQUENCE_LENGTH,
        max_seqlen_q=1,
        window_left=window_left,
        logit_cap=0.0,
        sinks=sinks,
        return_lse=False,
        softmax_scale=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
    )


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
def test_split_sliding_decode_matches_reference_and_replays(batch: int) -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, sinks = _make_inputs(
        batch, 20260915 + batch
    )

    def invoke() -> torch.Tensor:
        return _decode(
            q, k_cache, v_cache, page_table, cache_seqlens, sinks, WINDOW_LEFT
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


@pytest.mark.parametrize("window_left", [-1, WINDOW_LEFT])
def test_decode_reads_pages_past_int32_range(window_left: int) -> None:
    # Each KV page base is computed in 64-bit. Pages past 2**31 elements must
    # read exactly what the same pages read at the start of a small pool.
    q, k_cache, v_cache, page_table, cache_seqlens, sinks = _make_inputs(2, 20261029)
    expected = _decode(
        q, k_cache, v_cache, page_table, cache_seqlens, sinks, window_left
    )
    (k_far, v_far), far_table = move_pages_past_int32((k_cache, v_cache), page_table)
    actual = _decode(q, k_far, v_far, far_table, cache_seqlens, sinks, window_left)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


# Few KV heads per rank, so a small batch leaves the grid short of base CTAs.
FULL_Q_HEADS = 16
FULL_KV_HEADS = 2
# The paged backend passes the model's maximum context, not the batch maximum.
FULL_MAX_SEQLEN_K = 1 << 20


def test_full_decode_config_keeps_prefetch_within_vmcnt() -> None:
    """BF16 head dim 128 would prefetch more copies than s_waitcnt vmcnt counts."""
    q = torch.empty(1, FULL_Q_HEADS, 128, dtype=torch.bfloat16)
    k_cache = torch.empty(1, PAGE_SIZE, FULL_KV_HEADS, 128, dtype=torch.bfloat16)
    config = decode.get_config(
        q=q,
        k_cache=k_cache,
        max_seqlen_k=FULL_MAX_SEQLEN_K,
        max_seqlen_q=1,
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

    config = decode.get_config(
        q=q,
        k_cache=k_cache,
        max_seqlen_k=FULL_MAX_SEQLEN_K,
        max_seqlen_q=1,
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

    # The same pages past 2**31 elements of the pool read the same values.
    (k_far, v_far), far_table = move_pages_past_int32((k_cache, v_cache), page_table)
    far = gluon_mha_decode_gfx950(
        q=q,
        k_cache=k_far,
        v_cache=v_far,
        page_table=far_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=FULL_MAX_SEQLEN_K,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(far[nonempty], actual[nonempty], atol=0, rtol=0)


# 16 query heads over 2 KV heads, verifying 8 tokens per request.
VERIFY_Q_HEADS = 16
VERIFY_KV_HEADS = 2
VERIFY_TOKENS = 8


def _verify_inputs(
    cache_lens: list[int],
    max_seqlen_q: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    seed: int,
) -> dict:
    """Queries for each request's last max_seqlen_q cached tokens over a
    shuffled paged cache.

    Page-table entries past each request hold out-of-range page ids, so a copy
    that reads past a request's pages faults instead of passing silently.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    pages = [-(-length // PAGE_SIZE) for length in cache_lens]
    k_cache = torch.randn(
        sum(pages),
        PAGE_SIZE,
        kv_heads,
        head_dim,
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
        sum(pages), generator=torch.Generator().manual_seed(seed)
    ).to(dtype=torch.int32, device="cuda")
    start = 0
    for row, count in enumerate(pages):
        page_table[row, :count] = physical[start : start + count]
        start += count
    q = torch.randn(
        len(cache_lens) * max_seqlen_q,
        q_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "page_table": page_table,
        "cache_seqlens": torch.tensor(cache_lens, dtype=torch.int32, device="cuda"),
        "max_seqlen_k": FULL_MAX_SEQLEN_K,
        "max_seqlen_q": max_seqlen_q,
    }


def _verify_reference(inputs: dict, window_left: int, sinks) -> torch.Tensor:
    """FP32 dense attention where query i of a request sees its cache up to
    and including itself."""
    width = inputs["max_seqlen_q"]
    head_dim = inputs["q"].shape[2]
    expected = torch.empty(inputs["q"].shape, dtype=torch.float32, device="cuda")
    for row, length in enumerate(inputs["cache_seqlens"].tolist()):
        rows = slice(row * width, (row + 1) * width)
        pages = inputs["page_table"][row, : -(-length // PAGE_SIZE)].long()
        expected[rows], _ = torch_dense_attention(
            inputs["q"][rows],
            inputs["k_cache"][pages].flatten(0, 1)[:length],
            inputs["v_cache"][pages].flatten(0, 1)[:length],
            scale=head_dim**-0.5,
            causal_offset=length - width,
            window_left=None if window_left < 0 else window_left,
            sinks=sinks,
        )
    return expected


def _check_verify(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # Short rows attend to few keys and reach |out| > 1, where one BF16
    # rounding of the FP32 result is up to one BF16 ULP (2**-7) relative. P @ V
    # takes P rounded to BF16 (2**-9 relative), which over a handful of keys
    # cancelling to a small output leaves up to 2**-8 absolute.
    torch.testing.assert_close(actual.float(), expected, atol=2**-8, rtol=2**-7)


@pytest.mark.parametrize("with_sinks", [False, True], ids=["no-sinks", "sinks"])
@pytest.mark.parametrize(
    "window_left",
    [-1, 2047, 511, 255, 40],
    ids=["full", "window-2047", "window-511", "window-255", "window-40"],
)
@pytest.mark.parametrize(
    "max_seqlen_q,q_heads,kv_heads,head_dim",
    [
        # 8 tokens x 8 heads fill one 64-row, four-wave block.
        pytest.param(VERIFY_TOKENS, VERIFY_Q_HEADS, VERIFY_KV_HEADS, 64, id="64-rows"),
        # 2 x 8 rows fit a single wave.
        pytest.param(2, 16, 2, 64, id="16-rows"),
        # 3 x 6 rows pad a two-wave block; query positions straddle waves.
        pytest.param(3, 12, 2, 64, id="18-rows"),
        # 16 x 8 rows take two blocks per KV head.
        pytest.param(16, 8, 1, 64, id="128-rows"),
        pytest.param(VERIFY_TOKENS, 8, 1, 128, id="head-dim-128"),
        # 16 x 6 rows: under a short window, six one-wave blocks whose rows
        # straddle query positions.
        pytest.param(16, 6, 1, 128, id="96-rows"),
    ],
)
def test_verify_rows_match_reference_and_replay(
    max_seqlen_q: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    window_left: int,
    with_sinks: bool,
) -> None:
    """Speculative verify: each request's last ``max_seqlen_q`` cached tokens
    are queries, and query ``i`` sees the cache up to and including itself.

    Lengths cover the shortest verify (the block alone), a block whose first
    query's keys end on a page boundary (later queries alone see the last
    page, a split of its own under full attention), windows starting
    mid-page, and one sliding past many pages. A 40-key window starts later
    queries' windows a page past earlier ones'. The captured graph replays
    new lengths and queries.
    """
    cache_lens = [
        max_seqlen_q,
        PAGE_SIZE + max_seqlen_q - 1,
        2 * PAGE_SIZE + 37,
        2048 + max_seqlen_q + 37,
        300 * PAGE_SIZE + 5,
    ]
    inputs = _verify_inputs(cache_lens, max_seqlen_q, q_heads, kv_heads, head_dim, 5)
    sinks = (
        torch.randn(
            q_heads,
            dtype=torch.float32,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(13),
        )
        if with_sinks
        else None
    )

    def invoke() -> torch.Tensor:
        return gluon_mha_decode_gfx950(**inputs, window_left=window_left, sinks=sinks)

    _check_verify(invoke(), _verify_reference(inputs, window_left, sinks))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = invoke()
    # Serving refreshes the captured buffers in place. Every request keeps at
    # least its max_seqlen_q verified tokens.
    inputs["cache_seqlens"][1:].sub_(1)
    inputs["q"].mul_(-0.75)
    expected = _verify_reference(inputs, window_left, sinks)
    graph.replay()
    torch.cuda.synchronize()
    _check_verify(replayed, expected)
    _check_verify(invoke(), expected)


def test_short_window_verify_compiles_once_per_request_bucket() -> None:
    """A short window's split count changes with the request count only
    between at most 4 and more requests, so one warm-up on each side covers
    every batch size: 1 to 4 requests take the split kernel, more the
    unsplit one."""
    width, q_heads, kv_heads, dim, window_left = 16, 6, 1, 128, 511

    def run(batch: int) -> None:
        # Distinct lengths of 10 pages each keep the page-table stride fixed.
        lengths = [600 + slot for slot in range(batch)]
        inputs = _verify_inputs(lengths, width, q_heads, kv_heads, dim, batch)
        gluon_mha_decode_gfx950(**inputs, window_left=window_left)

    run(5)
    run(1)
    with assert_no_triton_compile(
        decode.gluon_mha_decode_sliding_gfx950,
        decode._mha_decode_sliding_split_atomic,
    ):
        for batch in [*range(6, 17), 2, 3, 4]:
            run(batch)


def test_verify_long_splits_pipeline_page_loads() -> None:
    """Two verifying requests leave the grid short, so each split of a long
    context streams its pages through the four-deep pipeline, the block's
    four waves sharing every page's copies."""
    cache_lens = [300 * PAGE_SIZE + 5, 1000 * PAGE_SIZE + 33]
    inputs = _verify_inputs(
        cache_lens, VERIFY_TOKENS, VERIFY_Q_HEADS, VERIFY_KV_HEADS, HEAD_DIM, 17
    )
    config = decode.get_config(
        q=inputs["q"],
        k_cache=inputs["k_cache"],
        max_seqlen_k=FULL_MAX_SEQLEN_K,
        max_seqlen_q=VERIFY_TOKENS,
        window_left=-1,
        softmax_scale=None,
    )
    assert (config.num_kv_splits, config.num_stages) == (64, 4)
    actual = gluon_mha_decode_gfx950(**inputs, window_left=-1)
    _check_verify(actual, _verify_reference(inputs, -1, None))


def test_pipelined_split_waits_for_prefetches_before_epilogue(monkeypatch) -> None:
    """The pipelined page loop ends with prefetched pages still copying into
    LDS, which the epilogue reuses for its layout conversions. A two-deep
    pipeline issues its last prefetch just before the final page's compute,
    so without a wait at loop exit the copy lands in the middle of the
    epilogue and corrupts rows (every run of this shape did). The shipped
    kernel selects four stages, where the race did not reproduce; two stages
    exposes it."""
    monkeypatch.setattr(decode, "_FULL_DECODE_STAGES", 2)
    inputs = _verify_inputs(
        [190000, 120000, 64000, 30000],
        1,
        VERIFY_Q_HEADS,
        VERIFY_KV_HEADS,
        HEAD_DIM,
        41,
    )
    config = decode.get_config(
        q=inputs["q"],
        k_cache=inputs["k_cache"],
        max_seqlen_k=FULL_MAX_SEQLEN_K,
        max_seqlen_q=1,
        window_left=-1,
        softmax_scale=None,
    )
    assert (config.num_kv_splits, config.num_stages) == (64, 2)
    expected = _verify_reference(inputs, -1, None)
    for _ in range(10):
        _check_verify(gluon_mha_decode_gfx950(**inputs, window_left=-1), expected)


# ===-----------------------------------------------------------------------===#
# Balanced split schedule
# ===-----------------------------------------------------------------------===#


def _balanced(inputs: dict, sinks=None, schedule=None) -> torch.Tensor:
    """The balanced decode, with a schedule of the current lengths unless one
    is given."""
    if schedule is None:
        schedule = decode.launch_gluon_mha_decode_schedule_gfx950(
            inputs["q"],
            inputs["k_cache"],
            inputs["cache_seqlens"],
            max_seqlen_q=inputs["max_seqlen_q"],
            window_left=-1,
            out=None,
        )
    return decode.launch_gluon_mha_decode_balanced_gfx950(
        **inputs, sinks=sinks, schedule=schedule
    )


@pytest.mark.parametrize("with_sinks", [False, True], ids=["no-sinks", "sinks"])
@pytest.mark.parametrize(
    "max_seqlen_q,q_heads,kv_heads,head_dim",
    [
        pytest.param(VERIFY_TOKENS, VERIFY_Q_HEADS, VERIFY_KV_HEADS, 64, id="64-rows"),
        pytest.param(1, VERIFY_Q_HEADS, VERIFY_KV_HEADS, 64, id="decode"),
        pytest.param(3, 12, 2, 64, id="18-rows"),
        pytest.param(16, 8, 1, 64, id="128-rows"),
        pytest.param(VERIFY_TOKENS, 8, 1, 128, id="head-dim-128"),
    ],
)
def test_balanced_rows_match_reference_and_replay(
    max_seqlen_q: int, q_heads: int, kv_heads: int, head_dim: int, with_sinks: bool
) -> None:
    """A few long requests among padding ones (one page each), as a padded
    graph batch holds them: the long ones take most splits, a split may end
    mid-page, and the last split runs to the request's end. The captured
    graph computes the schedule, so a replay balances its own lengths."""
    # Each request's pages cover both steps; the captured step and the
    # replayed one give different requests the long lengths.
    first = [1500 * PAGE_SIZE + 33, max_seqlen_q, max_seqlen_q, 300 * PAGE_SIZE + 5]
    first += [max_seqlen_q, PAGE_SIZE + max_seqlen_q - 1]
    second = [max_seqlen_q, 700 * PAGE_SIZE + 9, max_seqlen_q, max_seqlen_q]
    second += [1200 * PAGE_SIZE + 61, max_seqlen_q]
    inputs = _verify_inputs(
        [max(a, b) for a, b in zip(first, second)],
        max_seqlen_q,
        q_heads,
        kv_heads,
        head_dim,
        21,
    )
    inputs["cache_seqlens"].copy_(torch.tensor(first, dtype=torch.int32))
    sinks = (
        torch.randn(
            q_heads,
            dtype=torch.float32,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(17),
        )
        if with_sinks
        else None
    )
    _check_verify(_balanced(inputs, sinks), _verify_reference(inputs, -1, sinks))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = _balanced(inputs, sinks)
    inputs["cache_seqlens"].copy_(torch.tensor(second, dtype=torch.int32))
    inputs["q"].mul_(-0.75)
    graph.replay()
    torch.cuda.synchronize()
    _check_verify(replayed, _verify_reference(inputs, -1, sinks))


@pytest.mark.parametrize("max_seqlen_q", [1, VERIFY_TOKENS], ids=["decode", "verify"])
def test_balanced_stale_schedule_covers_grown_lengths(max_seqlen_q: int) -> None:
    """A drafter's steps grow the lengths after the step's schedule was
    computed; each split is bounded by the live lengths and a request's last
    split runs to its end, so the older schedule still covers every key."""
    grown = [1000 * PAGE_SIZE + 33, 3 * PAGE_SIZE + 9, 17 * PAGE_SIZE, 640 * PAGE_SIZE]
    inputs = _verify_inputs(
        grown, max_seqlen_q, VERIFY_Q_HEADS, VERIFY_KV_HEADS, HEAD_DIM, 43
    )
    shorter = torch.tensor(
        [max(max_seqlen_q, length - 500 * PAGE_SIZE) for length in grown],
        dtype=torch.int32,
        device="cuda",
    )
    stale = decode.launch_gluon_mha_decode_schedule_gfx950(
        inputs["q"],
        inputs["k_cache"],
        shorter,
        max_seqlen_q=max_seqlen_q,
        window_left=-1,
        out=None,
    )
    _check_verify(
        _balanced(inputs, schedule=stale), _verify_reference(inputs, -1, None)
    )


def test_balanced_batch_past_one_schedule_chunk() -> None:
    """Work-groups find their request 64 schedule rows per load; 80 requests
    of mixed lengths take two chunks."""
    cache_lens = [(i % 7) * 37 * PAGE_SIZE + 1 + i for i in range(80)]
    inputs = _verify_inputs(cache_lens, 1, VERIFY_Q_HEADS, VERIFY_KV_HEADS, 64, 3)
    _check_verify(_balanced(inputs), _verify_reference(inputs, -1, None))


def test_balanced_rows_without_keys_merge_to_the_sink() -> None:
    """A request without keys stores only empty partials (lse = -inf); its
    rows merge to the sink alone, a zero output, not NaN."""
    inputs = _verify_inputs(
        [0, 900 * PAGE_SIZE + 7], 1, VERIFY_Q_HEADS, VERIFY_KV_HEADS, 64, 29
    )
    sinks = torch.randn(VERIFY_Q_HEADS, dtype=torch.float32, device="cuda")
    out = _balanced(inputs, sinks).float()
    assert torch.equal(out[0], torch.zeros_like(out[0]))
    reference = _verify_reference(
        {
            **inputs,
            "q": inputs["q"][1:],
            "cache_seqlens": inputs["cache_seqlens"][1:],
            "page_table": inputs["page_table"][1:],
        },
        -1,
        sinks,
    )
    _check_verify(out[1:], reference)


def test_public_ops_select_the_balanced_path_only_with_a_schedule(tmp_path) -> None:
    """Verify calls take a schedule and plain decode and sliding windows none;
    mha_decode_with_kvcache keeps the split grid without a schedule and takes
    the balanced kernels with one."""
    import json

    from tokenspeed_kernel.ops.attention.mha import (
        mha_decode_schedule,
        mha_decode_takes_schedule,
        mha_decode_with_kvcache,
    )
    from tokenspeed_kernel.profiling import ShapeCapture

    plain = _verify_inputs(
        [900 * PAGE_SIZE + 3, 1, 1, 1], 1, VERIFY_Q_HEADS, VERIFY_KV_HEADS, 64, 37
    )
    plain_call = dict(q=plain["q"], k_cache=plain["k_cache"], v_cache=plain["v_cache"])
    assert not mha_decode_takes_schedule(max_seqlen_q=1, **plain_call)
    assert (
        mha_decode_schedule(
            cache_seqlens=plain["cache_seqlens"], max_seqlen_q=1, **plain_call
        )
        is None
    )

    inputs = _verify_inputs(
        [900 * PAGE_SIZE + 3, VERIFY_TOKENS, VERIFY_TOKENS, VERIFY_TOKENS],
        VERIFY_TOKENS,
        VERIFY_Q_HEADS,
        VERIFY_KV_HEADS,
        64,
        37,
    )
    call = dict(
        q=inputs["q"],
        k_cache=inputs["k_cache"],
        v_cache=inputs["v_cache"],
        max_seqlen_q=VERIFY_TOKENS,
    )
    assert mha_decode_takes_schedule(**call)
    assert not mha_decode_takes_schedule(window_left=511, **call)
    assert (
        mha_decode_schedule(
            cache_seqlens=inputs["cache_seqlens"], window_left=511, **call
        )
        is None
    )
    schedule = mha_decode_schedule(cache_seqlens=inputs["cache_seqlens"], **call)

    ShapeCapture.reset()
    ShapeCapture.get().enabled = True
    try:
        expected = _verify_reference(inputs, -1, None)
        for passed in (None, schedule):
            _check_verify(
                mha_decode_with_kvcache(
                    page_table=inputs["page_table"],
                    cache_seqlens=inputs["cache_seqlens"],
                    max_seqlen_k=1 << 20,
                    schedule=passed,
                    **call,
                ),
                expected,
            )
        ShapeCapture.get().dump(tmp_path / "shapes.json")
    finally:
        ShapeCapture.reset()
    names = [
        record["kernel_name"]
        for record in json.loads((tmp_path / "shapes.json").read_text())
        if record["mode"] == "mha_decode_with_kvcache"
    ]
    assert names == ["gluon_mha_decode_gfx950", "gluon_mha_decode_balanced_gfx950"]


def test_balanced_grid_past_512_items() -> None:
    """300 requests of two row blocks take more items than the 512 target:
    every (request, row block) still gets its split."""
    cache_lens = [(i % 5) * 9 * PAGE_SIZE + 1 + i for i in range(300)]
    inputs = _verify_inputs(cache_lens, 1, VERIFY_Q_HEADS, VERIFY_KV_HEADS, 64, 41)
    _check_verify(_balanced(inputs), _verify_reference(inputs, -1, None))


def test_balanced_schedule_buffer_at_capacity_ignores_its_stale_tail() -> None:
    """A persistent schedule buffer sized for the largest batch keeps the
    rows of earlier, larger batches past this batch's; the kernels read only
    this batch's rows. Padding requests (one page) take one split each."""
    cache_lens = [800 * PAGE_SIZE + 3, 1, 1, 1]
    inputs = _verify_inputs(cache_lens, 1, VERIFY_Q_HEADS, VERIFY_KV_HEADS, 64, 31)
    buffer = torch.full((33, 2), 1 << 20, dtype=torch.int32, device="cuda")
    schedule = decode.launch_gluon_mha_decode_schedule_gfx950(
        inputs["q"],
        inputs["k_cache"],
        inputs["cache_seqlens"],
        max_seqlen_q=1,
        window_left=-1,
        out=buffer,
    )
    assert schedule is buffer
    assert buffer[2:5, 1].tolist() == [1, 1, 1]
    assert buffer[5:].eq(1 << 20).all()
    _check_verify(
        _balanced(inputs, schedule=buffer), _verify_reference(inputs, -1, None)
    )
