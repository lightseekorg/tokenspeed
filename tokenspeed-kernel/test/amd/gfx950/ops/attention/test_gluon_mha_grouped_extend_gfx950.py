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

"""Non-power-of-two GQA groups, speculative rows, live paged metadata and large
KV pools."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import pytest
import torch
from utils import assert_no_triton_compile, is_amd, is_cdna4, move_pages_past_int32

if is_amd():
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.extend import (
        gluon_mha_extend_gfx950,
        gluon_mha_extend_split_gfx950,
    )


@pytest.fixture
def require_gfx950():
    from utils import is_cdna4

    if not is_cdna4():
        pytest.skip("AMD gfx950 is required for Gluon grouped MHA device tests")


@pytest.mark.parametrize(
    "group,query_rows,warp_pipeline",
    [
        (group, query_rows, False)
        for group in (1, 2, 4, 5, 6, 8, 12)
        for query_rows in (1, 16, 64, 848, 896, 8192)
    ]
    + [(6, 848, True), (8, 8192, True)],
)
def test_select_extend_tile(group, query_rows, warp_pipeline):
    # Tile selection is CPU-only; the AMD kernel package is optional on other hosts.
    if not is_amd():
        pytest.importorskip(
            "tokenspeed_kernel_amd", reason="AMD kernel package is optional"
        )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.extend import (
        _select_extend_tile,
    )

    block_m, block_q, block_n, warps = _select_extend_tile(
        query_rows, group, warp_pipeline
    )
    padded_group = 1 << (group - 1).bit_length()
    assert block_m & (block_m - 1) == 0
    assert block_q & (block_q - 1) == 0
    assert block_m == block_q * padded_group
    assert block_n == 64
    if warp_pipeline:
        # The warp-pipelined loop needs 8 waves of 32 rows each.
        assert (block_m, warps) == (256, 8)
    else:
        assert warps in (2, 4)
    # The tile must cover each real (position, head) exactly once.
    actual = [
        (row // padded_group, row % padded_group)
        for row in range(block_m)
        if row % padded_group < group
    ]
    assert actual == [(pos, head) for pos in range(block_q) for head in range(group)]


@pytest.mark.parametrize(
    "max_q,prefix,total_q,window,expected",
    [
        # Full attention: a cached prefix of 8192 keys, at any row count past
        # the 64-row tile.
        (128, 8192, 128, -1, True),
        (8192, 65536, 8192, -1, True),
        (128, 8191, 128, -1, False),
        (64, 65536, 64, -1, False),
        # Short prefixes, as in serving's fresh and short-prefix extends.
        (1024, 0, 1024, -1, False),
        (4096, 512, 8192, -1, False),
        # A 2047-key window: a prefix of one window and one to four rounds of
        # work-groups.
        (1024, 2047, 4 * 1024, 2047, True),
        (4096, 65536, 4 * 4096, 2047, True),
        (4096, 2046, 4096, 2047, False),
        (512, 65536, 512, 2047, False),
        (4096, 65536, 8 * 4096, 2047, False),
        # Narrower windows keep the 4-wave tiles.
        (4096, 65536, 4096, 511, False),
    ],
)
def test_warp_pipeline_selection(max_q, prefix, total_q, window, expected):
    # Tile selection is CPU-only; the AMD kernel package is optional on other hosts.
    if not is_amd():
        pytest.importorskip(
            "tokenspeed_kernel_amd", reason="AMD kernel package is optional"
        )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.extend import (
        _use_warp_pipeline,
    )

    assert (
        _use_warp_pipeline(
            max_seqlen_q=max_q,
            max_seqlen_k=max_q + prefix,
            total_q=total_q,
            group_size=8,
            n_kv_heads=2,
            head_dim=64,
            element_size=2,
            window_left=window,
        )
        == expected
    )


@pytest.mark.parametrize(
    "max_q,prefix,total_q,window,expected",
    [
        # FP8 K/V takes the 8-wave tile under the same bounds as 16-bit K/V,
        # at any query row count past the 64-row tile.
        (8192, 65536, 8192, -1, True),
        (150, 100000, 150, -1, True),
        (384, 512000, 384, -1, True),
        (64, 100000, 64, -1, False),
        (4096, 512, 8192, -1, False),
        (4096, 65536, 4 * 4096, 2047, True),
        (512, 65536, 512, 2047, False),
    ],
)
def test_fp8_warp_pipeline_selection(max_q, prefix, total_q, window, expected):
    if not is_amd():
        pytest.importorskip(
            "tokenspeed_kernel_amd", reason="AMD kernel package is optional"
        )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.extend import (
        _use_warp_pipeline,
    )

    assert (
        _use_warp_pipeline(
            max_seqlen_q=max_q,
            max_seqlen_k=max_q + prefix,
            total_q=total_q,
            group_size=8,
            n_kv_heads=2,
            head_dim=64,
            element_size=1,
            window_left=window,
        )
        == expected
    )


@dataclass
class PagedCase:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    table: torch.Tensor
    lengths: torch.Tensor
    cu_q: torch.Tensor
    cu_kv: torch.Tensor
    queries: list[int]
    max_q: int
    max_k: int
    window: int
    causal: bool
    sinks: torch.Tensor | None

    def invoke(self, solution: str, *, return_lse: bool):
        from tokenspeed_kernel.ops.attention.mha import mha_extend_with_kvcache

        return mha_extend_with_kvcache(
            q=self.q,
            cu_seqlens_q=self.cu_q,
            cu_seqlens_kv=self.cu_kv,
            k_cache=self.k,
            v_cache=self.v,
            page_table=self.table,
            cache_seqlens=self.lengths,
            is_causal=self.causal,
            window_left=self.window,
            sinks=self.sinks,
            return_lse=return_lse,
            max_seqlen_q=self.max_q,
            max_seqlen_k=self.max_k,
            solution=solution,
        )


def make_case(
    queries, prefixes, *, heads, kv_heads, window, causal, sinks, head_dim=128
):
    torch.manual_seed(1357)
    lengths = [q + p for q, p in zip(queries, prefixes, strict=True)]
    pages_per_request = math.ceil(max(lengths) / 64)
    page_count = len(queries) * pages_per_request + 1
    q = torch.randn(
        (sum(queries), heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        (page_count, 64, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    table = torch.arange(1, page_count, device="cuda", dtype=torch.int32).view(
        len(queries), pages_per_request
    )
    # Noncontiguous physical pages catch accidental use of logical token order.
    table = table.flip(1).contiguous()
    # Entries past a request's pages hold an out-of-range page id, so a copy
    # that reads past them faults instead of passing silently.
    for request, length in enumerate(lengths):
        table[request, math.ceil(length / 64) :] = 1 << 30
    cu_q = [0]
    cu_kv = [0]
    for query, length in zip(queries, lengths, strict=True):
        cu_q.append(cu_q[-1] + query)
        cu_kv.append(cu_kv[-1] + length)
    return PagedCase(
        q,
        k,
        v,
        table,
        torch.tensor(lengths, device="cuda", dtype=torch.int32),
        torch.tensor(cu_q, device="cuda", dtype=torch.int32),
        torch.tensor(cu_kv, device="cuda", dtype=torch.int32),
        list(queries),
        max(queries),
        max(lengths),
        window,
        causal,
        torch.randn(heads, device="cuda", dtype=torch.bfloat16) if sinks else None,
    )


def reference(case: PagedCase):
    """FP32 dense oracle, chunked only along independent query rows."""
    outputs, lses = [], []
    start = 0
    group = case.q.shape[1] // case.k.shape[2]
    for request, query_rows in enumerate(case.queries):
        length = int(case.lengths[request].item())
        pages = case.table[request, : math.ceil(length / 64)].long()
        k = case.k[pages].flatten(0, 1)[:length].float().repeat_interleave(group, 1)
        v = case.v[pages].flatten(0, 1)[:length].float().repeat_interleave(group, 1)
        key_pos = torch.arange(length, device="cuda")
        prefix = length - query_rows
        for offset in range(0, query_rows, 32):
            end = min(offset + 32, query_rows)
            q = case.q[start + offset : start + end].float()
            scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(q.shape[-1])
            positions = prefix + torch.arange(offset, end, device="cuda")
            mask = torch.ones((end - offset, length), device="cuda", dtype=torch.bool)
            if case.causal:
                mask &= key_pos[None, :] <= positions[:, None]
            if case.window >= 0:
                mask &= key_pos[None, :] >= positions[:, None] - case.window
            scores.masked_fill_(~mask[None], -float("inf"))
            if case.sinks is not None:
                sink_scores = case.sinks.float()[:, None, None].expand(
                    -1, end - offset, 1
                )
                scores = torch.cat((scores, sink_scores), dim=-1)
            lse = torch.logsumexp(scores, dim=-1)
            p = torch.softmax(scores, dim=-1)[..., :length]
            outputs.append(torch.einsum("hqk,khd->qhd", p, v))
            lses.append(lse.transpose(0, 1))
        start += query_rows
    return torch.cat(outputs), torch.cat(lses)


def assert_reference(case: PagedCase):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected, expected_lse = reference(case)
        actual, actual_lse = case.invoke("gluon", return_lse=True)
        ordinary = case.invoke("gluon", return_lse=False)
        # The output goes through BF16 probabilities and keeps the public MHA
        # tolerance. The LSE is summed in FP32 from FP32 scores and stays
        # within 2e-4 of the FP32 oracle even over 50k keys; a page of keys
        # wrongly in or out of a window of up to a few thousand keys moves it
        # further.
        torch.testing.assert_close(actual.float(), expected, rtol=8e-2, atol=8e-2)
        torch.testing.assert_close(ordinary.float(), expected, rtol=8e-2, atol=8e-2)
        torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=2e-4)
        assert torch.isfinite(actual).all() and torch.isfinite(actual_lse).all()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("window", [-1, 511])
@pytest.mark.usefixtures("require_gfx950")
def test_speculative_verify_rows(batch, window):
    case = make_case(
        [16] * batch,
        [50000 - 7 * i for i in range(batch)],
        heads=6,
        kv_heads=1,
        window=window,
        causal=True,
        sinks=False,
    )
    assert_reference(case)


@pytest.mark.parametrize("query_rows", [848, 896, 8192])
@pytest.mark.parametrize("window", [-1, 511])
@pytest.mark.usefixtures("require_gfx950")
def test_cached_prefill_chunks(query_rows, window):
    case = make_case(
        [query_rows],
        [50000 - query_rows],
        heads=6,
        kv_heads=1,
        window=window,
        causal=True,
        sinks=False,
    )
    assert_reference(case)


@pytest.mark.parametrize("window", [-1, 511])
@pytest.mark.usefixtures("require_gfx950")
def test_ragged_mixed_prefill_and_verify(window):
    case = make_case(
        [848, 16, 16],
        [49152, 49997, 50019],
        heads=6,
        kv_heads=1,
        window=window,
        causal=True,
        sinks=True,
    )
    assert_reference(case)


@pytest.mark.parametrize(
    "heads,kv_heads,causal,window", [(5, 1, False, 4095), (12, 2, True, 511)]
)
@pytest.mark.usefixtures("require_gfx950")
def test_draft_and_multiple_kv_head_groups(heads, kv_heads, causal, window):
    case = make_case(
        [16, 5],
        [127, 211],
        heads=heads,
        kv_heads=kv_heads,
        window=window,
        causal=causal,
        sinks=True,
    )
    assert_reference(case)


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected).norm() / expected.norm())


# Head dim 256 at Qwen3.8-27B's TP1 shape: 24 query heads over 4 KV heads (a
# group of 6). A chunk of a long prompt over its cached prefix, the first
# chunk, a ragged mix with draft-sized rows, and verify rows; full attention
# may split the KV stream, the window takes the single pass.
_HEAD_DIM_256_CASES = [
    pytest.param([2048], [6144], id="chunk"),
    pytest.param([1000], [0], id="first-chunk"),
    pytest.param([848, 16, 5], [4096, 9001, 127], id="ragged"),
    pytest.param([16] * 4, [9000, 9013, 9027, 9064], id="verify"),
]


@pytest.mark.parametrize("queries,prefixes", _HEAD_DIM_256_CASES)
@pytest.mark.parametrize("window", [-1, 511])
@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.usefixtures("require_gfx950")
def test_head_dim_256_grouped_queries(queries, prefixes, window, sinks):
    case = make_case(
        queries,
        prefixes,
        heads=24,
        kv_heads=4,
        window=window,
        causal=True,
        sinks=sinks,
        head_dim=256,
    )
    assert_reference(case)


@pytest.mark.parametrize("queries,prefixes", _HEAD_DIM_256_CASES)
@pytest.mark.parametrize("window", [-1, 511])
@pytest.mark.usefixtures("require_gfx950")
def test_head_dim_256_fp8_cache(queries, prefixes, window):
    # An FP8 E4M3 cache with FP8 queries (unit scales), against the FP32
    # oracle on the same FP8 inputs. The probabilities' FP8 rounding costs
    # 1.8-2.5% relative L2 over these thousands of nearly even keys, the same
    # as head dim 128 on the same inputs (BF16 on them: 0.2%; measured on
    # MI355X); the LSE is summed from FP32 scores.
    case = make_case(
        queries,
        prefixes,
        heads=24,
        kv_heads=4,
        window=window,
        causal=True,
        sinks=False,
        head_dim=256,
    )
    fp8 = torch.float8_e4m3fn
    case = dataclasses.replace(
        case, q=case.q.to(fp8), k=case.k.to(fp8), v=case.v.to(fp8)
    )
    expected, expected_lse = reference(case)
    actual, actual_lse = case.invoke("gluon", return_lse=True)
    assert _rel_l2(actual, expected) < 3e-2
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=2e-4)


@pytest.mark.parametrize(
    "queries,prefixes,window",
    [
        # Head dim 64 with 2 KV heads, so the pipeline's LDS buffers leave two
        # workgroups per CU. Up to 64 rows use the 64-row tile; 32 rows leave
        # the grid short, so 32 splits each pipeline about ten pages, and the
        # last one ends mid-page.
        pytest.param([32], [300 * 64 + 17], -1, id="split-pipelined"),
        # An empty prefix, a split with fewer pages than buffers and a long row.
        pytest.param([5, 64, 33], [0, 4 * 64, 1000 * 64 + 33], -1, id="split-ragged"),
        # 128-row tiles: 1000 rows take five splits of about 28 pages; a
        # prefix under 8192 keys keeps the 4-wave tile.
        pytest.param([1000], [8000 + 3], -1, id="split-pipelined-128"),
        # 4096 rows fill the grid unsplit; pipelining adds no round of
        # workgroups.
        pytest.param([4096], [7000 + 11], -1, id="pipelined"),
        # 6144 rows: pipelining would add a round of workgroups, so pages
        # load single-buffered.
        pytest.param([6144], [3000 + 7], -1, id="single-buffer"),
        # Sliding windows never split; a window spans many loop iterations.
        pytest.param([100, 37], [20000, 9000], 4095, id="window-pipelined"),
    ],
)
@pytest.mark.usefixtures("require_gfx950")
def test_kv_stream(queries, prefixes, window):
    case = make_case(
        queries,
        prefixes,
        heads=16,
        kv_heads=2,
        window=window,
        causal=True,
        sinks=window >= 0,
        head_dim=64,
    )
    assert_reference(case)


@pytest.mark.parametrize(
    "queries,prefixes",
    [
        # 63 tiles of 16 positions: the K/V page loads are pipelined.
        pytest.param([1000], [5000 + 13], id="pipelined"),
        # 768 work-groups: the page loads are single-buffered. A cached
        # prefix shorter than a 2047-key window keeps the 4-wave tile.
        pytest.param([6144], [1500 + 7], id="single-buffer"),
        # A tile straddling the prefix end, and windows longer than the context.
        pytest.param([37, 300], [9, 700], id="short-context"),
    ],
)
@pytest.mark.parametrize("window", [100, 511, 2047])
@pytest.mark.usefixtures("require_gfx950")
def test_sliding_window_sees_exactly_its_keys(queries, prefixes, window):
    # Only the pages that hold some row's window edge or its causal diagonal
    # are masked; the pages between are visible to every row of a tile.
    case = make_case(
        queries,
        prefixes,
        heads=16,
        kv_heads=2,
        window=window,
        causal=True,
        sinks=False,
        head_dim=64,
    )
    assert_reference(case)


def uses_warp_pipeline(case: PagedCase) -> bool:
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha import extend

    return extend._use_warp_pipeline(
        max_seqlen_q=case.max_q,
        max_seqlen_k=case.max_k,
        total_q=case.q.shape[0],
        group_size=case.q.shape[1] // case.k.shape[2],
        n_kv_heads=case.k.shape[2],
        head_dim=case.q.shape[2],
        element_size=case.k.element_size(),
        window_left=case.window,
    )


@pytest.mark.parametrize(
    "queries,prefixes,heads,kv_heads,window,sinks",
    [
        # 256-row tiles of 32 positions x 8 heads. 1024 rows split the KV
        # stream; the last page is partial.
        pytest.param([1024], [20000 + 3], 16, 2, -1, False, id="split"),
        # 4096 rows fill every CU unsplit; the last tile ends mid-page.
        pytest.param([4096], [10000 + 11], 16, 2, -1, False, id="unsplit"),
        # One request under a page past its rows, one with a prefix shorter
        # than the copy distance, one long; the short ones get empty splits.
        pytest.param(
            [65, 200, 900], [9000, 3 * 64, 700 * 64 + 5], 16, 2, -1, True, id="ragged"
        ),
        # A group of 6 heads padded to 8, and a group of 8 with one KV head.
        pytest.param([1024], [9000 + 7], 12, 2, -1, True, id="group-6"),
        pytest.param([2048], [30000 + 9], 8, 1, -1, True, id="group-8"),
        # Sliding windows: 4096 rows give a work-group to every CU. The pages
        # that hold some row's window edge or its diagonal are masked, the
        # pages between are not.
        pytest.param([4096], [5000 + 13], 16, 2, 2047, False, id="window"),
        pytest.param([8192], [6000 + 1], 16, 2, 4095, True, id="window-4095"),
        # A tile straddling the prefix end, and windows longer than the
        # context of the second request.
        pytest.param([4096, 37], [3000, 9], 16, 2, 2047, False, id="window-ragged"),
    ],
)
@pytest.mark.usefixtures("require_gfx950")
def test_warp_pipeline(queries, prefixes, heads, kv_heads, window, sinks):
    case = make_case(
        queries,
        prefixes,
        heads=heads,
        kv_heads=kv_heads,
        window=window,
        causal=True,
        sinks=sinks,
        head_dim=64,
    )
    assert uses_warp_pipeline(case)
    assert_reference(case)
    first, first_lse = case.invoke("gluon", return_lse=True)
    # Tighter than the public tolerance: guards P of up to 2**8 before BF16 rounding.
    expected, _ = reference(case)
    torch.testing.assert_close(first.float(), expected, rtol=2e-2, atol=1e-2)
    # Each work-group sums its pages in a fixed order: the result does not
    # change between launches.
    second, second_lse = case.invoke("gluon", return_lse=True)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_lse, second_lse, rtol=0, atol=0)


@pytest.mark.parametrize(
    "queries,prefixes,window,sinks",
    [
        # A cached-prefill chunk of 8192 rows over a short and a long prefix:
        # unsplit, as serving's chunks.
        pytest.param([8192], [8192], -1, False, id="chunk-8k-prefix"),
        pytest.param([8192], [100000], -1, False, id="chunk-100k-prefix"),
        # A short extend over a long prefix splits the KV stream: whole splits
        # sit above some rows' diagonal, the rows with no visible key. At 384
        # rows the 4-wave split count would leave a partial 8-wave round, so
        # the launch takes another split count.
        pytest.param([150], [100000 + 5], -1, False, id="split"),
        pytest.param([384], [100000 + 5], -1, False, id="split-whole-rounds"),
        # 2560 rows: more splits than the 4-wave launch, over two rounds.
        pytest.param([2560], [100000 + 5], -1, False, id="split-more-than-four-wave"),
        pytest.param([65, 900], [9000, 700 * 64 + 5], -1, True, id="ragged-sinks"),
        pytest.param([4096], [5000 + 13], 2047, False, id="window"),
    ],
)
@pytest.mark.usefixtures("require_gfx950")
def test_fp8_warp_pipeline_as_accurate_as_four_wave(
    queries, prefixes, window, sinks, monkeypatch
):
    """FP8 K/V on the 8-wave tile is as close to the FP32 oracle as the 4-wave
    tile, in output and LSE, and repeats bit for bit."""
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha import extend

    case = make_case(
        queries,
        prefixes,
        heads=16,
        kv_heads=2,
        window=window,
        causal=True,
        sinks=sinks,
        head_dim=64,
    )
    case = dataclasses.replace(
        case,
        q=case.q.to(torch.float8_e4m3fn),
        k=case.k.to(torch.float8_e4m3fn),
        v=case.v.to(torch.float8_e4m3fn),
    )
    assert uses_warp_pipeline(case)
    expected, expected_lse = reference(case)
    eight, eight_lse = case.invoke("gluon", return_lse=True)
    again, again_lse = case.invoke("gluon", return_lse=True)
    # The 4-wave tile is the launcher's choice whenever the 8-wave tile is not.
    monkeypatch.setattr(extend, "_use_warp_pipeline", lambda **kwargs: False)
    four, four_lse = case.invoke("gluon", return_lse=True)
    torch.testing.assert_close(eight, again, rtol=0, atol=0)
    torch.testing.assert_close(eight_lse, again_lse, rtol=0, atol=0)
    # Both tiles round FP8 probabilities the same way up to the order of FP32
    # sums, so the 8-wave error stays at the 4-wave tile's: within a tenth of
    # it, plus a few FP32 ulps of LSE.
    for actual, baseline, oracle, slack in (
        (eight.float(), four.float(), expected, 0.0),
        (eight_lse, four_lse, expected_lse, 1e-5),
    ):
        error = (actual - oracle).abs()
        baseline_error = (baseline - oracle).abs()
        assert error.max() <= 1.1 * baseline_error.max() + slack
        assert error.mean() <= 1.1 * baseline_error.mean() + slack


def fp8_case(queries, prefixes, *, heads, kv_heads):
    """A full-attention, causal, head-dim-64 case with FP8 Q / K / V."""
    case = make_case(
        queries,
        prefixes,
        heads=heads,
        kv_heads=kv_heads,
        window=-1,
        causal=True,
        sinks=False,
        head_dim=64,
    )
    return dataclasses.replace(
        case,
        q=case.q.to(torch.float8_e4m3fn),
        k=case.k.to(torch.float8_e4m3fn),
        v=case.v.to(torch.float8_e4m3fn),
    )


@pytest.mark.parametrize(
    "queries,prefixes",
    [
        # A served long-prefix chunk of one request (Muse Spark TP4: 32
        # query, 4 KV heads) at 156k cached keys.
        pytest.param([2500], [156000], id="long-prefix-chunk"),
        # A short extend over a long prefix takes the split-K kernel.
        pytest.param([150], [131072 + 5], id="split"),
    ],
)
@pytest.mark.usefixtures("require_gfx950")
def test_fp8_eight_wave_occupancy_keeps_every_bit(queries, prefixes, monkeypatch):
    """The FP8 full-attention 8-wave tile of one request, at four waves a
    SIMD, gives the same output and LSE bits as at the
    compiler's occupancy, and repeats bit for bit."""
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha import extend

    case = fp8_case(queries, prefixes, heads=32, kv_heads=4)
    assert uses_warp_pipeline(case)
    capped, capped_lse = case.invoke("gluon", return_lse=True)
    again, again_lse = case.invoke("gluon", return_lse=True)
    launch_options = extend._launch_options

    def compiler_occupancy(*args, **kwargs):
        options = launch_options(*args, **kwargs)
        options.pop("waves_per_eu")
        return options

    # The launch as it compiled before the occupancy cap.
    monkeypatch.setattr(extend, "_launch_options", compiler_occupancy)
    base, base_lse = case.invoke("gluon", return_lse=True)
    torch.testing.assert_close(capped, again, rtol=0, atol=0)
    torch.testing.assert_close(capped_lse, again_lse, rtol=0, atol=0)
    torch.testing.assert_close(capped, base, rtol=0, atol=0)
    torch.testing.assert_close(capped_lse, base_lse, rtol=0, atol=0)


@pytest.mark.parametrize(
    "num_warps,element_size,window,single_request,capped",
    [
        # Only the FP8 full-attention 8-wave tile of a one-request batch
        # takes the occupancy cap.
        (8, 1, -1, True, True),
        (8, 1, -1, False, False),
        (8, 1, 2047, True, False),
        (8, 2, -1, True, False),
        (4, 1, -1, True, False),
        (2, 1, -1, True, False),
    ],
)
def test_eight_wave_occupancy_selection(
    num_warps, element_size, window, single_request, capped
):
    if not is_amd():
        pytest.importorskip(
            "tokenspeed_kernel_amd", reason="AMD kernel package is optional"
        )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha import extend

    options = extend._launch_options(
        num_warps,
        element_size=element_size,
        window_left=window,
        single_request=single_request,
    )
    assert options["num_warps"] == num_warps
    assert options.get("waves_per_eu") == (4 if capped else None)


@pytest.mark.usefixtures("require_gfx950")
def test_fp8_request_count_keeps_bits_and_compiles_at_warmup():
    """A one-request FP8 batch runs the 8-wave tile at four waves a
    SIMD, a batch of several requests at the
    compiler's occupancy. A request gets the same output and LSE bits alone
    as beside a copy of itself, and after the warm-up neither launch
    compiles, unsplit or split."""
    from tokenspeed_kernel.ops.tuning import kernel_warmup
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.extend import (
        gluon_mha_extend_gfx950,
        gluon_mha_extend_reduce_gfx950,
        gluon_mha_extend_split_gfx950,
    )
    from utils import assert_no_triton_compile

    # 4096 rows over a 9000-key prefix fill the CUs unsplit, alone and
    # doubled; 150 rows over 20000 keys split the KV stream.
    for queries, prefix, unsplit in ((4096, 9000, True), (150, 20000, False)):
        two = fp8_case([queries, queries], [prefix, prefix], heads=16, kv_heads=2)
        # The second request repeats the first: same queries, same pages.
        two.q[queries:] = two.q[:queries]
        two.table[1] = two.table[0]
        one = fp8_case([queries], [prefix], heads=16, kv_heads=2)
        one = dataclasses.replace(
            one, q=two.q[:queries], k=two.k, v=two.v, table=two.table[:1]
        )
        assert uses_warp_pipeline(one) and uses_warp_pipeline(two)
        with kernel_warmup(enabled=True):
            one.invoke("gluon", return_lse=True)
        with assert_no_triton_compile(
            gluon_mha_extend_gfx950,
            gluon_mha_extend_split_gfx950,
            gluon_mha_extend_reduce_gfx950,
        ):
            single, single_lse = one.invoke("gluon", return_lse=True)
            double, double_lse = two.invoke("gluon", return_lse=True)
        torch.testing.assert_close(double[:queries], double[queries:], rtol=0, atol=0)
        if unsplit:
            # The same sums in the same order at either occupancy.
            torch.testing.assert_close(single, double[:queries], rtol=0, atol=0)
            torch.testing.assert_close(single_lse, double_lse[:queries], rtol=0, atol=0)
        expected, expected_lse = reference(one)
        torch.testing.assert_close(single.float(), expected, rtol=8e-2, atol=8e-2)
        torch.testing.assert_close(single_lse, expected_lse, rtol=0, atol=2e-4)


@pytest.mark.parametrize(
    "heads,kv_heads,queries,prefix",
    [
        # Muse Spark TP4 (32 query, 4 KV heads): 16 query blocks a head, so
        # each head runs on two XCDs; 2 KV heads take four each.
        pytest.param(32, 4, 4096, 20000, id="tp4"),
        pytest.param(16, 2, 4096, 9000, id="tp8"),
    ],
)
@pytest.mark.usefixtures("require_gfx950")
def test_xcd_head_order_keeps_every_bit(heads, kv_heads, queries, prefix):
    """The unsplit kernel places each KV head's workgroups on their own XCDs.
    Every workgroup still computes one (query block, head), so a request's
    output equals the oracle and repeats bit for bit beside another request."""
    two = fp8_case([queries, queries], [prefix, prefix], heads=heads, kv_heads=kv_heads)
    two.q[queries:] = two.q[:queries]
    two.table[1] = two.table[0]
    one = fp8_case([queries], [prefix], heads=heads, kv_heads=kv_heads)
    one = dataclasses.replace(
        one, q=two.q[:queries], k=two.k, v=two.v, table=two.table[:1]
    )
    single, single_lse = one.invoke("gluon", return_lse=True)
    double, double_lse = two.invoke("gluon", return_lse=True)
    torch.testing.assert_close(single, double[:queries], rtol=0, atol=0)
    torch.testing.assert_close(single, double[queries:], rtol=0, atol=0)
    torch.testing.assert_close(single_lse, double_lse[:queries], rtol=0, atol=0)
    expected, expected_lse = reference(one)
    torch.testing.assert_close(single.float(), expected, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(single_lse, expected_lse, rtol=0, atol=2e-4)


@pytest.mark.usefixtures("require_gfx950")
def test_extend_rejects_head_slices_of_the_cache():
    # The kernels read K / V pages as contiguous [pages, 64, kv_heads, dim];
    # a head slice of a larger cache has other strides and must not be read
    # as that layout.
    case = make_case(
        [256],
        [1000],
        heads=32,
        kv_heads=4,
        window=-1,
        causal=True,
        sinks=False,
        head_dim=64,
    )
    sliced = dataclasses.replace(
        case, q=case.q[:, :16], k=case.k[:, :, :2], v=case.v[:, :, :2]
    )
    with pytest.raises(ValueError, match="contiguous"):
        sliced.invoke("gluon", return_lse=False)


@pytest.mark.usefixtures("require_gfx950")
def test_fp8_eight_wave_reads_pages_past_int32_range():
    # The FP8 8-wave tile computes each KV page base in 64-bit too: pages past
    # 2**31 elements read exactly what the same pages read in a small pool.
    case = fp8_case([256], [20000], heads=32, kv_heads=4)
    assert uses_warp_pipeline(case)
    expected = case.invoke("gluon", return_lse=False)
    (k_far, v_far), far_table = move_pages_past_int32((case.k, case.v), case.table)
    far_case = dataclasses.replace(case, k=k_far, v=v_far, table=far_table)
    # The arena pointers are not specialized (serving sizes the arena after
    # startup's warm-up), so the small pool's binaries serve the large one.
    with assert_no_triton_compile(
        gluon_mha_extend_gfx950, gluon_mha_extend_split_gfx950
    ):
        actual = far_case.invoke("gluon", return_lse=False)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def assert_graph_replay_follows_metadata(case: PagedCase, updates):
    """Capture case once, then for each (queries, lengths) rewrite the
    metadata in place and check the replay against eager and the oracle,
    without compiling."""
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.extend import (
        gluon_mha_extend_gfx950,
        gluon_mha_extend_reduce_gfx950,
        gluon_mha_extend_split_gfx950,
    )
    from utils import assert_no_triton_compile

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            case.invoke("gluon", return_lse=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output, lse = case.invoke("gluon", return_lse=True)
    torch.cuda.synchronize()
    for queries, lengths in updates:
        case.queries = queries
        case.cu_q.copy_(
            torch.tensor(
                [0, queries[0], sum(queries)], device="cuda", dtype=torch.int32
            )
        )
        case.lengths.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        case.cu_kv.copy_(
            torch.tensor(
                [0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32
            )
        )
        case.table.copy_(case.table.flip(1))
        with assert_no_triton_compile(
            gluon_mha_extend_gfx950,
            gluon_mha_extend_split_gfx950,
            gluon_mha_extend_reduce_gfx950,
        ):
            graph.replay()
            torch.cuda.synchronize()
            eager, eager_lse = case.invoke("gluon", return_lse=True)
        torch.testing.assert_close(output, eager, rtol=0, atol=0)
        torch.testing.assert_close(lse, eager_lse, rtol=0, atol=0)
        assert_reference(case)
    graph.reset()


@pytest.mark.parametrize("window", [-1, 2047])
@pytest.mark.usefixtures("require_gfx950")
def test_warp_pipeline_graph_replay(window):
    # Full attention splits 1024 rows of two requests; a window gets 4096
    # rows so every CU has a work-group. Every length below needs all 146
    # pages of its table row, so the flipped table holds only valid pages.
    rows = 1024 if window < 0 else 4096
    case = make_case(
        [rows // 2, rows // 2],
        [9300 - rows // 2, 9340 - rows // 2],
        heads=16,
        kv_heads=2,
        window=window,
        causal=True,
        sinks=True,
        head_dim=64,
    )
    case.max_q = rows
    assert uses_warp_pipeline(case)
    assert_graph_replay_follows_metadata(
        case,
        [
            ([rows // 4 + 3, rows - rows // 4 - 3], [9330, 9290]),
            ([rows - 37, 37], [9344, 9281]),
        ],
    )


@pytest.mark.parametrize("window", [-1, 511])
@pytest.mark.parametrize(
    "heads,kv_heads,head_dim", [(6, 1, 128), (24, 4, 256)], ids=["hd128", "hd256"]
)
@pytest.mark.usefixtures("require_gfx950")
def test_changed_metadata_graph_replay_without_recompile(
    window, heads, kv_heads, head_dim
):
    case = make_case(
        [16, 16],
        [1001, 977],
        heads=heads,
        kv_heads=kv_heads,
        window=window,
        causal=True,
        sinks=True,
        head_dim=head_dim,
    )
    case.max_q = 32
    case.max_k = 2048
    assert_graph_replay_follows_metadata(
        case, [([9, 23], [1021, 997]), ([21, 11], [1007, 1013])]
    )


@pytest.mark.parametrize("query_rows", [848, 896, 8192])
@pytest.mark.parametrize("window", [-1, 511])
@pytest.mark.usefixtures("require_gfx950")
def test_initial_prefill_groups(query_rows, window):
    from tokenspeed_kernel.ops.attention.mha import mha_prefill

    case = make_case(
        [query_rows], [0], heads=6, kv_heads=1, window=window, causal=True, sinks=False
    )
    k = case.k[case.table[0].long()].flatten(0, 1)[:query_rows]
    v = case.v[case.table[0].long()].flatten(0, 1)[:query_rows]
    actual = mha_prefill(
        q=case.q,
        k=k,
        v=v,
        cu_seqlens=case.cu_q,
        cu_seqlens_cpu=[0, query_rows],
        max_seqlen=query_rows,
        window_left=window,
        solution="gluon",
    )
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected, _ = reference(case)
        torch.testing.assert_close(actual.float(), expected, rtol=8e-2, atol=8e-2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize(
    "queries,prefixes", [([64, 48], [200, 300]), ([16], [50000])], ids=["short", "long"]
)
# Full attention may take the split-K kernel; a window takes the single pass.
@pytest.mark.parametrize("window", [-1, 511])
# Head dim 64 pipelines the page loads; 128 and 256 stay single-buffered.
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.usefixtures("require_gfx950")
def test_extend_reads_pages_past_int32_range(queries, prefixes, window, head_dim):
    # Each KV page base is computed in 64-bit. Pages past 2**31 elements must
    # read exactly what the same pages read at the start of a small pool.
    case = make_case(
        queries,
        prefixes,
        heads=8,
        kv_heads=2,
        window=window,
        causal=True,
        sinks=False,
        head_dim=head_dim,
    )
    expected = case.invoke("gluon", return_lse=False)
    (k_far, v_far), far_table = move_pages_past_int32((case.k, case.v), case.table)
    far_case = dataclasses.replace(case, k=k_far, v=v_far, table=far_table)
    # The arena pointers are not specialized (serving sizes the arena after
    # startup's warm-up), so the small pool's binaries serve the large one.
    with assert_no_triton_compile(
        gluon_mha_extend_gfx950, gluon_mha_extend_split_gfx950
    ):
        actual = far_case.invoke("gluon", return_lse=False)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _one_request(case: PagedCase, request: int) -> PagedCase:
    """The same request launched alone: its own grid, block count and order."""
    start = int(case.cu_q[request])
    rows = case.queries[request]
    length = int(case.lengths[request])
    return dataclasses.replace(
        case,
        q=case.q[start : start + rows],
        table=case.table[request : request + 1],
        lengths=case.lengths[request : request + 1],
        cu_q=torch.tensor([0, rows], device="cuda", dtype=torch.int32),
        cu_kv=torch.tensor([0, length], device="cuda", dtype=torch.int32),
        queries=[rows],
        max_q=rows,
        max_k=length,
    )


# Batched and alone launches here pick the same tile and KV split count (a
# split count is part of the numerics, so cases keep it at one): under full
# attention every request has at least 4096 rows over a prefix of at least
# 8192 keys (8-wave tile, the grid fills the machine unsplit); under the
# window every launch takes the 8-wave tile's 1-4 rounds of work-groups.
_PLACEMENT_CASES = {
    -1: [
        # A rectangular launch whose blocks x KV heads leave a remainder on
        # 8 XCDs (129 blocks a request), alone and batched.
        pytest.param([4112] * 3, [9000 + 5] * 3, id="rect-tail"),
        pytest.param([4096] * 2, [12000 + 3] * 2, id="rect"),
        pytest.param([4100, 6000, 4500], [9000 + 7, 12000, 8500 + 1], id="ragged"),
    ],
    2047: [
        pytest.param([2100] * 2, [9000 + 5] * 2, id="rect"),
        pytest.param([2100, 2300], [9000 + 7, 3000 + 1], id="ragged"),
    ],
}


@pytest.mark.parametrize(
    "window,queries,prefixes",
    [
        pytest.param(window, *case.values, id=f"w{window}-{case.id}")
        for window, cases in _PLACEMENT_CASES.items()
        for case in cases
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.usefixtures("require_gfx950")
def test_work_group_placement_keeps_every_bit(window, queries, prefixes, dtype):
    """Where a (query block, KV head) runs and in which order is placement
    only: a batched launch writes exactly the bytes of each request launched
    alone, whose grids, remap tails and orders all differ."""
    case = make_case(
        queries,
        prefixes,
        heads=32,
        kv_heads=4,
        window=window,
        causal=True,
        sinks=False,
        head_dim=64,
    )
    if dtype == torch.bfloat16:
        # Every block runs once: a skipped or repeated block misses the oracle.
        assert_reference(case)
    else:
        case = dataclasses.replace(
            case, q=case.q.to(dtype), k=case.k.to(dtype), v=case.v.to(dtype)
        )
    alone_cases = [_one_request(case, request) for request in range(len(queries))]
    for single in alone_cases:
        assert uses_warp_pipeline(single) == uses_warp_pipeline(case)
    batched = case.invoke("gluon", return_lse=False)
    alone = torch.cat(
        [single.invoke("gluon", return_lse=False) for single in alone_cases]
    )
    assert torch.equal(batched.view(torch.int16), alone.view(torch.int16))


@pytest.mark.usefixtures("require_gfx950")
def test_work_group_placement_does_not_recompile():
    """The remap reads the grid at run time: once the launcher's warmup has
    compiled every variant for these heads, batch shapes that change the block
    count, the remap tail and the grid layout compile nothing new."""
    from tokenspeed_kernel.ops.tuning import kernel_warmup
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha import extend
    from utils import assert_no_triton_compile

    def run(queries, prefix):
        case = make_case(
            queries,
            [prefix] * len(queries),
            heads=32,
            kv_heads=4,
            window=-1,
            causal=True,
            sinks=False,
            head_dim=64,
        )
        # The page table's row stride is compiled in, as the served table's
        # fixed width is: pad every batch's table to one width.
        table = torch.full(
            (len(queries), 256), 1 << 30, device="cuda", dtype=torch.int32
        )
        table[:, : case.table.shape[1]] = case.table
        dataclasses.replace(case, table=table).invoke("gluon", return_lse=False)

    with kernel_warmup(enabled=True):
        run([100], 300)
    with assert_no_triton_compile(
        extend.gluon_mha_extend_gfx950,
        extend.gluon_mha_extend_split_gfx950,
        extend.gluon_mha_extend_reduce_gfx950,
    ):
        for queries in ([1056], [1040] * 2, [1040] * 3, [640, 1500], [900, 33, 1200]):
            for prefix in (300, 9000):
                run(queries, prefix)


def test_extend_refuses_an_arena_off_16_bytes(require_gfx950):
    """The kernels state the arena's 16-byte alignment instead of specializing
    its pointers; the direct-to-LDS page copy has no lowering without it, so
    the launcher refuses an unaligned arena rather than failing to compile."""
    case = make_case(
        [16, 40], [48, 100], heads=8, kv_heads=1, window=-1, causal=True, sinks=None
    )

    def shifted(cache):
        flat = torch.empty(cache.numel() + 1, dtype=cache.dtype, device=cache.device)
        flat[1:].copy_(cache.flatten())
        return flat[1:].view_as(cache)

    off = dataclasses.replace(case, k=shifted(case.k), v=shifted(case.v))
    with pytest.raises(ValueError, match="16-byte aligned"):
        off.invoke("gluon", return_lse=False)


if is_cdna4():
    from tokenspeed_kernel_amd._triton import gl, gluon
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.extend import (
        _remap_extend_program,
    )

    @gluon.jit
    def _remap_probe(out, RAGGED: gl.constexpr, N_KV_HEADS: gl.constexpr):
        """Stores the (query block, batch, KV head) that _remap_extend_program
        gives each program, at the program's index in launch order."""
        # Program ids as the unsplit extend kernel reads them.
        pid_q = gl.program_id(0)
        if RAGGED:
            kv_head = gl.program_id(1)
            launch = pid_q + gl.num_programs(0) * kv_head
        else:
            kv_head = gl.program_id(2)
            launch = pid_q + gl.num_programs(0) * (
                gl.program_id(1) + gl.num_programs(1) * kv_head
            )
        pid_batch = 0 if RAGGED else gl.program_id(1)
        pid_q, pid_batch, kv_head = _remap_extend_program(
            pid_q, pid_batch, kv_head, RAGGED, N_KV_HEADS
        )
        gl.store(out + launch * 3, pid_q)
        if not RAGGED:
            # A ragged launch has one batch: its slot keeps the host's zero.
            gl.store(out + (launch * 3 + 1), pid_batch)
        gl.store(out + (launch * 3 + 2), kv_head)


# Launch shapes without the KV-head dimension: ragged (blocks,) and rectangular
# (blocks_per_req, batch). With 8 XCDs and 1, 2, 4 or 8 heads they cover one
# block, launches below, at and at multiples of one round of XCDs, and
# remainders below and above one round.
_REMAP_RAGGED_GRIDS = [(blocks,) for blocks in (1, 3, 4, 7, 8, 9, 16, 21)]
_REMAP_RECTANGULAR_GRIDS = [
    (blocks_per_req, batch) for batch in (1, 2, 5) for blocks_per_req in (1, 3, 7)
] + [(4, 2), (8, 3)]
_REMAP_GRIDS = _REMAP_RAGGED_GRIDS + _REMAP_RECTANGULAR_GRIDS


def _remapped_programs(grid, kv_heads):
    """Launches the probe over ``grid + (kv_heads,)``; returns an int64
    [programs, 3] tensor of (query block, batch, KV head) in launch order."""
    programs = math.prod(grid) * kv_heads
    out = torch.zeros(programs * 3, device="cuda", dtype=torch.int32)
    _remap_probe[(*grid, kv_heads)](out, len(grid) == 1, kv_heads, num_warps=1)
    return out.view(programs, 3).cpu().long()


@pytest.mark.parametrize("kv_heads", [1, 2, 4, 8])
@pytest.mark.parametrize("grid", _REMAP_GRIDS, ids=lambda g: "x".join(map(str, g)))
@pytest.mark.usefixtures("require_gfx950")
def test_work_group_placement_runs_every_block_and_head_once(grid, kv_heads):
    """The remap is one to one for every launch shape, and each full round of
    XCDs gives every XCD a work-group of the KV head that owns it."""
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha import extend

    num_xcds = extend._NUM_XCDS.value
    # The extend kernel remaps only these head counts.
    assert num_xcds % kv_heads == 0
    blocks_per_req = grid[0]
    batch = 1 if len(grid) == 1 else grid[1]
    blocks = blocks_per_req * batch
    programs = blocks * kv_heads

    pid_q, pid_batch, kv_head = _remapped_programs(grid, kv_heads).unbind(1)
    assert pid_q.min() >= 0 and pid_q.max() < blocks_per_req
    assert pid_batch.min() >= 0 and pid_batch.max() < batch
    assert kv_head.min() >= 0 and kv_head.max() < kv_heads
    block = pid_batch * blocks_per_req + pid_q
    visits = torch.bincount(block * kv_heads + kv_head, minlength=programs)
    assert torch.equal(visits, torch.ones(programs, dtype=torch.long))

    # Work-group i of the launch runs on XCD i % num_xcds; the work-groups
    # past the last full round are not placed by head.
    placed = programs - programs % num_xcds
    owners = []
    for xcd in range(num_xcds if placed else 0):
        on_xcd = slice(xcd, placed, num_xcds)
        owner = kv_head[on_xcd].unique()
        assert owner.numel() == 1, f"XCD {xcd} runs KV heads {owner.tolist()}"
        owners.append(owner.item())
        # Last block first, so the longest causal work starts the launch.
        assert (block[on_xcd].diff() < 0).all()
    if placed:
        owned = torch.bincount(torch.tensor(owners), minlength=kv_heads)
        assert (owned == num_xcds // kv_heads).all()


@pytest.mark.parametrize("kv_heads", [1, 2, 4, 8])
@pytest.mark.parametrize("grids", [_REMAP_RAGGED_GRIDS, _REMAP_RECTANGULAR_GRIDS])
@pytest.mark.usefixtures("require_gfx950")
def test_remap_reads_the_grid_at_run_time(grids, kv_heads):
    """For one head count and grid layout, no launch size compiles the remap
    again."""
    from utils import assert_no_triton_compile

    _remapped_programs(grids[0], kv_heads)
    with assert_no_triton_compile(_remap_probe):
        for grid in grids:
            _remapped_programs(grid, kv_heads)
