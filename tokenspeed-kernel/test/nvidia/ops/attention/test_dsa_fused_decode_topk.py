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

"""The fused CuTe DSL decode top-k against DeepGEMM and a stable-sort oracle.

Logits must equal ``deep_gemm.fp8_paged_mqa_logits`` (after the non-finite
scrub) bit for bit on every row's causal window. Selections must equal the
head of each window's stable descending sort of those logits (ties to the
lowest positions), in ascending position, mapped through the block table, with
a ``-1`` tail and the counts; every output entry is checked against a planted
sentinel. Block tables are shuffled so slot order differs from position order.
"""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_cute_compile, is_nvidia, move_pages_past_int32

if not is_nvidia():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)
pytest.importorskip("cutlass")
from tokenspeed_kernel.platform import ArchVersion, current_platform  # noqa: E402

if not ArchVersion(10, 0) <= current_platform().arch_version <= ArchVersion(10, 3):
    pytest.skip("requires SM100 or SM103", allow_module_level=True)

from ops.test_attention_dsa import _pack_index_k_cache  # noqa: E402
from tokenspeed_kernel.ops.attention.dsa import cute_dsl as dsa_cute_dsl  # noqa: E402
from tokenspeed_kernel.ops.attention.dsa import deep_gemm as dsa_deep_gemm  # noqa: E402
from tokenspeed_kernel.ops.attention.dsa import dsa_decode_topk  # noqa: E402
from tokenspeed_kernel.ops.attention.dsa._triton.topk import (  # noqa: E402
    quantize_index_query_weights,
    scrub_nonfinite_logits,
)
from tokenspeed_kernel.selection import select_kernel  # noqa: E402
from tokenspeed_kernel.signature import (  # noqa: E402
    dense_tensor_format,
    format_signature,
)

if dsa_cute_dsl._fused_indexer is None:
    pytest.skip("the fused CuTe DSL decode top-k is not built", allow_module_level=True)
if not dsa_deep_gemm._HAS_DEEP_GEMM:
    pytest.skip("the DeepGEMM reference is not installed", allow_module_level=True)

PAGE = 64
HEADS = 32
DIM = 128
TOPK = 2048
SOFTMAX_SCALE = DIM**-0.5 * HEADS**-0.5
SENTINEL = 0x5A5A5A5A
LEAF = "cute_dsl_dsa_decode_topk"


def _case(lengths, q_len, *, seed, extra_pages=3, zero_keys=False):
    """Queries, BF16 weights, a packed plane (random keys, or all zero) with
    each request on shuffled pages, and a page table padded with unused
    columns."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    requests = len(lengths)
    pages_per = [max(1, -(-length // PAGE)) for length in lengths]
    width = max(pages_per) + extra_pages
    pages = sum(pages_per) + 1
    keys = torch.randn(
        pages * PAGE, DIM, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    if zero_keys:
        keys.zero_()
    cache, _ = _pack_index_k_cache(keys, PAGE)
    order = torch.randperm(pages, device="cuda", generator=generator).to(torch.int32)
    table = torch.zeros(requests, width, dtype=torch.int32, device="cuda")
    used = 0
    for r, count in enumerate(pages_per):
        table[r, :count] = order[used : used + count]
        used += count
    tokens = requests * q_len
    q = torch.randn(
        tokens, HEADS, DIM, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weights = torch.randn(
        tokens, HEADS, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return dict(
        q=q, weights=weights, seq_lens=seq_lens, table=table, cache=cache, q_len=q_len
    )


def _row_lengths(seq_lens, q_len, width):
    offsets = torch.arange(1 - q_len, 1, device=seq_lens.device, dtype=torch.int32)
    rows = (seq_lens[:, None] + offsets).reshape(-1)
    return rows.clamp(0, width * PAGE)


def _deep_gemm_logits(q_fp8, folded, seq_lens, table, cache, q_len):
    """DeepGEMM paged MQA logits as the DeepGEMM leaf runs them (one request
    row per token), scrubbed: ``[tokens, block-table capacity]``."""
    dg = dsa_deep_gemm.deep_gemm
    tokens = q_fp8.shape[0]
    lens_2d = seq_lens.repeat_interleave(q_len).unsqueeze(1).contiguous()
    plan = dg.get_paged_mqa_logits_metadata(lens_2d, PAGE, dg.get_num_sms())
    logits = dg.fp8_paged_mqa_logits(
        q_fp8.view(tokens, 1, HEADS, DIM),
        cache.view(-1, PAGE, 1, cache.shape[-1]),
        folded,
        lens_2d,
        table.repeat_interleave(q_len, dim=0).contiguous(),
        plan,
        table.shape[1] * PAGE,
        clean_logits=False,
    )
    scrub_nonfinite_logits(logits, seq_lens, next_n=q_len)
    return logits


def _fused(case, cache=None, table=None):
    """The fused kernels' slots, counts and request-compact logits."""
    q_fp8, folded = quantize_index_query_weights(
        case["q"], case["weights"], SOFTMAX_SCALE
    )
    tokens = q_fp8.shape[0]
    out = torch.full((tokens, TOPK), SENTINEL, dtype=torch.int32, device="cuda")
    lens = torch.full((tokens,), SENTINEL, dtype=torch.int32, device="cuda")
    logits, _ = dsa_cute_dsl.fused_decode_topk_scores(
        q_fp8,
        folded,
        case["seq_lens"],
        case["table"] if table is None else table,
        case["cache"] if cache is None else cache,
        q_len_per_req=case["q_len"],
        topk=TOPK,
        out=out,
        lens_out=lens,
    )
    return out, lens, logits, q_fp8, folded


def _scratch_rows(logits, seq_lens, q_len, width):
    """Each scored row's logits from the request-compact scratch, keyed by row."""
    ends = seq_lens.clamp(0, width * PAGE).tolist()
    row_lengths = _row_lengths(seq_lens, q_len, width).tolist()
    rows, start = {}, 0
    for r, end in enumerate(ends):
        tiles = -(-end // 128) if end > TOPK else 0
        for j in range(q_len):
            base = (start * q_len + j * tiles) * 128
            row = r * q_len + j
            rows[row] = logits[base : base + row_lengths[row]]
        start += tiles
    return rows


def _oracle(reference, seq_lens, table, q_len):
    """The head of each causal window's stable descending sort, in ascending
    position, as slots with a -1 tail, and the counts."""
    width = table.shape[1]
    row_lengths = _row_lengths(seq_lens, q_len, width).tolist()
    tokens = len(row_lengths)
    slots = torch.full((tokens, TOPK), -1, dtype=torch.int32, device="cuda")
    lens = torch.tensor(
        [min(length, TOPK) for length in row_lengths], dtype=torch.int32, device="cuda"
    )
    for row, length in enumerate(row_lengths):
        if length <= TOPK:
            positions = torch.arange(length, device="cuda")
        else:
            order = torch.sort(reference[row, :length], descending=True, stable=True)
            positions = order.indices[:TOPK].sort().values
        pages = table[row // q_len, positions // PAGE]
        slots[row, : positions.numel()] = (pages * PAGE + positions % PAGE).to(
            torch.int32
        )
    return slots, lens


def _check_selection(case, out, lens, reference, table=None):
    table = case["table"] if table is None else table
    expected, expected_lens = _oracle(reference, case["seq_lens"], table, case["q_len"])
    assert torch.equal(lens, expected_lens)
    assert torch.equal(out, expected)


def _check_logits(case, logits, reference):
    rows = _scratch_rows(
        logits, case["seq_lens"], case["q_len"], case["table"].shape[1]
    )
    ends = case["seq_lens"].clamp(0, case["table"].shape[1] * PAGE)
    for row, mine in rows.items():
        if int(ends[row // case["q_len"]]) <= TOPK:
            continue  # the request is not scored
        expected = reference[row, : mine.numel()].contiguous()
        assert torch.equal(mine.view(torch.int32), expected.view(torch.int32)), row


# Lengths across the bypass (<= 2048), the key bound (< 2 * 2048 vectors), the
# vector-maxima bound past it, and maxima past the shared-memory copy (131072).
_LENGTHS = {
    1: [0, 1, 300, 2048, 2049, 4111, 32767, 40000, 131077],
    2: [2049, 2048, 9000, 1],
    3: [2049, 7000, 2],
    4: [2050, 2051, 40001, 3],
    5: [2052, 4, 12000],
    6: [2053, 70000, 6],
    7: [2054, 0, 33000],
    8: [2055, 2048 + 7, 5000, 131080, 3],
}


@pytest.mark.parametrize("q_len", sorted(_LENGTHS))
def test_logits_are_bitwise_deep_gemm(q_len) -> None:
    case = _case(_LENGTHS[q_len], q_len, seed=10 + q_len)
    out, lens, logits, q_fp8, folded = _fused(case)
    reference = _deep_gemm_logits(
        q_fp8, folded, case["seq_lens"], case["table"], case["cache"], q_len
    )
    _check_logits(case, logits, reference)


@pytest.mark.parametrize("q_len", sorted(_LENGTHS))
def test_selection_matches_the_sorted_oracle(q_len) -> None:
    case = _case(_LENGTHS[q_len], q_len, seed=20 + q_len)
    out, lens, _, q_fp8, folded = _fused(case)
    reference = _deep_gemm_logits(
        q_fp8, folded, case["seq_lens"], case["table"], case["cache"], q_len
    )
    _check_selection(case, out, lens, reference)


@pytest.mark.parametrize("q_len", [1, 8])
def test_the_facade_pin_runs_the_fused_leaf(q_len) -> None:
    case = _case([2100, 37000, 5], q_len, seed=30 + q_len)
    tokens = case["q"].shape[0]
    out = torch.full((tokens, TOPK), SENTINEL, dtype=torch.int32, device="cuda")
    lens = torch.full((tokens,), SENTINEL, dtype=torch.int32, device="cuda")
    dsa_decode_topk(
        case["q"],
        case["weights"],
        case["seq_lens"],
        case["table"],
        page_size=PAGE,
        topk=TOPK,
        softmax_scale=SOFTMAX_SCALE,
        batch_invariant=False,
        q_len_per_req=q_len,
        index_k_cache=case["cache"],
        seq_lens_2d=None,
        plan=None,
        out=out,
        lens_out=lens,
        slot_order="sorted",
        solution="cute_dsl",
    )
    q_fp8, folded = quantize_index_query_weights(
        case["q"], case["weights"], SOFTMAX_SCALE
    )
    reference = _deep_gemm_logits(
        q_fp8, folded, case["seq_lens"], case["table"], case["cache"], q_len
    )
    _check_selection(case, out, lens, reference)


def test_selection_agrees_with_the_deep_gemm_batch_invariant_leaf() -> None:
    """The existing leaf's lowest-index-tie selection picks the same slots."""
    from tokenspeed_kernel.ops.attention.dsa import dsa_plan
    from tokenspeed_kernel.ops.attention.dsa.flashinfer import (
        has_deterministic_decode_topk,
    )

    if not has_deterministic_decode_topk():
        pytest.skip("needs the lowest-index tie-break top-k")
    q_len = 4
    case = _case([2052, 9000, 41000], q_len, seed=41)
    out, lens, *_ = _fused(case)
    lens_2d = case["seq_lens"].repeat_interleave(q_len).unsqueeze(1).contiguous()
    slots, counts = dsa_decode_topk(
        case["q"],
        case["weights"],
        case["seq_lens"],
        case["table"],
        page_size=PAGE,
        topk=TOPK,
        softmax_scale=SOFTMAX_SCALE,
        batch_invariant=True,
        q_len_per_req=q_len,
        index_k_cache=case["cache"],
        seq_lens_2d=lens_2d,
        plan=dsa_plan(page_size=PAGE, seq_lens_2d=lens_2d, solution="deep_gemm"),
        slot_order="selection",
        solution="deep_gemm",
    )
    assert torch.equal(lens, counts)
    assert torch.equal(out.sort(dim=1).values, slots.sort(dim=1).values)


def _planes(cache):
    """The packed plane's FP8 key bytes ``[pages, 64, 128]`` and FP32 scales
    ``[pages, 64]``, as views: each page holds its keys, then their scales."""
    pages = cache.view(-1, PAGE * (DIM + 4))
    keys = pages[:, : PAGE * DIM].view(-1, PAGE, DIM)
    scales = pages[:, PAGE * DIM :].view(torch.float32)
    return keys, scales


def _slot(table, request, position):
    return int(table[request, position // PAGE]) * PAGE + position % PAGE


@pytest.mark.parametrize("length", [2600, 9000, 45000])
def test_equal_logits_select_the_lowest_positions(length) -> None:
    """All-zero keys score every key 0: the selection is the first K keys,
    through the exact select over the whole row once candidates overflow."""
    case = _case([length, 2100], 2, seed=51, zero_keys=True)
    out, lens, _, q_fp8, folded = _fused(case)
    reference = _deep_gemm_logits(
        q_fp8, folded, case["seq_lens"], case["table"], case["cache"], 2
    )
    _check_selection(case, out, lens, reference)


@pytest.mark.parametrize("length", [6000, 50000])
def test_ties_at_the_threshold_go_to_the_lowest_positions(length) -> None:
    """A key copied to 300 positions scores identically everywhere; copies of
    the key ranked near K straddle the threshold, so only the lowest copies are
    taken."""
    generator = torch.Generator(device="cuda").manual_seed(61)
    case = _case([length], 1, seed=62)
    q_fp8, folded = quantize_index_query_weights(
        case["q"], case["weights"], SOFTMAX_SCALE
    )
    reference = _deep_gemm_logits(
        q_fp8, folded, case["seq_lens"], case["table"], case["cache"], 1
    )
    ranked = torch.sort(reference[0, :length], descending=True, stable=True).indices
    source = int(ranked[TOPK - 100])
    copies = torch.randperm(length, device="cuda", generator=generator)[:300]
    keys, scales = _planes(case["cache"])
    source_page, source_offset = divmod(_slot(case["table"], 0, source), PAGE)
    key_bytes = keys[source_page, source_offset].clone()
    scale = scales[source_page, source_offset].clone()
    for position in copies.tolist():
        page, offset = divmod(_slot(case["table"], 0, position), PAGE)
        keys[page, offset] = key_bytes
        scales[page, offset] = scale
    out, lens, logits, q_fp8, folded = _fused(case)
    reference = _deep_gemm_logits(
        q_fp8, folded, case["seq_lens"], case["table"], case["cache"], 1
    )
    tied = reference[0, :length] == reference[0, source]
    assert int(tied.sum()) >= 300
    _check_logits(case, logits, reference)
    _check_selection(case, out, lens, reference)


def test_nonfinite_logits_score_minus_infinity() -> None:
    """NaN keys and an overflowing head weight score NaN or +inf in DeepGEMM;
    both are -inf after the scrub, the fused scorer stores -inf, and rows with
    fewer than K finite keys take the lowest -inf positions."""
    generator = torch.Generator(device="cuda").manual_seed(71)
    case = _case([3000, 2600, 40000], 2, seed=72)
    keys, _ = _planes(case["cache"])
    # E4M3 NaN (0x7F) in the first byte of most of request 0's keys.
    pages = case["table"][0, : -(-3000 // PAGE)].long()
    nan_keys = torch.rand(pages.numel(), PAGE, device="cuda", generator=generator)
    first_bytes = keys[pages, :, 0]
    keys[pages, :, 0] = torch.where(
        nan_keys < 0.7, torch.full_like(first_bytes, 0x7F), first_bytes
    )
    # Request 2's first row: one huge head weight overflows its logits.
    case["weights"][4, 0] = 3.0e38
    out, lens, logits, q_fp8, folded = _fused(case)
    reference = _deep_gemm_logits(
        q_fp8, folded, case["seq_lens"], case["table"], case["cache"], 2
    )
    assert bool(torch.isneginf(reference[0, :2999]).any())
    _check_logits(case, logits, reference)
    _check_selection(case, out, lens, reference)


def test_far_pages_past_int32_offsets() -> None:
    """Pages placed past 2**31 bytes of the plane score and select as near."""
    q_len = 2
    case = _case([2300, 9000], q_len, seed=81)
    near_out, near_lens, near_logits, q_fp8, folded = _fused(case)
    pages = case["cache"].shape[0] // PAGE
    (pool,), table = move_pages_past_int32(
        (case["cache"].view(pages, -1),), case["table"]
    )
    far_cache = pool.view(-1, DIM + 4)
    assert int(table.min()) * PAGE * (DIM + 4) > 2**31
    out, lens, logits, *_ = _fused(case, cache=far_cache, table=table)
    reference = _deep_gemm_logits(
        q_fp8, folded, case["seq_lens"], case["table"], case["cache"], q_len
    )
    _check_logits(case, logits, reference)
    _check_selection(case, out, lens, reference, table=table)
    assert torch.equal(lens, near_lens)


def test_no_recompile_across_batch_shapes() -> None:
    for q_len in (1, 8):
        _fused(_case([2100, 30], q_len, seed=90 + q_len))
    with assert_no_cute_compile():
        for q_len, lengths, extra_pages in (
            (1, [5, 2049, 7000, 33000, 90], 1),
            (1, [2500], 40),
            (8, [2100, 3000, 12], 2),
            (8, [2055] * 9, 7),
        ):
            case = _case(lengths, q_len, seed=95, extra_pages=extra_pages)
            out, lens, _, q_fp8, folded = _fused(case)
            reference = _deep_gemm_logits(
                q_fp8, folded, case["seq_lens"], case["table"], case["cache"], q_len
            )
            _check_selection(case, out, lens, reference)


@pytest.mark.parametrize("q_len", [1, 8])
def test_graph_replay_follows_new_lengths_and_queries(q_len) -> None:
    case = _case([2200, 9000, 1500], q_len, seed=100 + q_len, extra_pages=40)
    tokens = case["q"].shape[0]
    out = torch.full((tokens, TOPK), SENTINEL, dtype=torch.int32, device="cuda")
    lens = torch.full((tokens,), SENTINEL, dtype=torch.int32, device="cuda")

    def run():
        dsa_decode_topk(
            case["q"],
            case["weights"],
            case["seq_lens"],
            case["table"],
            page_size=PAGE,
            topk=TOPK,
            softmax_scale=SOFTMAX_SCALE,
            batch_invariant=False,
            q_len_per_req=q_len,
            index_k_cache=case["cache"],
            seq_lens_2d=None,
            plan=None,
            out=out,
            lens_out=lens,
            slot_order="selection",
            solution="cute_dsl",
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # New lengths cross the bypass both ways and grow within the table.
    case["seq_lens"].copy_(
        torch.tensor([1900, 9500, 4000], dtype=torch.int32, device="cuda")
    )
    case["q"].mul_(-0.5)
    out.fill_(SENTINEL)
    lens.fill_(SENTINEL)
    graph.replay()
    torch.cuda.synchronize()
    q_fp8, folded = quantize_index_query_weights(
        case["q"], case["weights"], SOFTMAX_SCALE
    )
    reference = _deep_gemm_logits(
        q_fp8, folded, case["seq_lens"], case["table"], case["cache"], q_len
    )
    _check_selection(case, out, lens, reference)


def test_only_the_cute_dsl_pin_selects_the_fused_leaf() -> None:
    """Models with this indexer geometry keep their leaf unless they pin it."""
    signature = format_signature(
        q=dense_tensor_format(torch.bfloat16),
        weights=dense_tensor_format(torch.bfloat16),
    )
    traits = {
        "q_len": 8,
        "index_heads": HEADS,
        "head_dim": DIM,
        "page_size": PAGE,
        "topk": TOPK,
        "index_k_format": "fp8_scaled",
        "index_k_layout": "packed",
    }
    default = select_kernel("attention", "dsa_decode_topk", signature, traits=traits)
    assert default.name != LEAF
    pinned = select_kernel(
        "attention", "dsa_decode_topk", signature, traits=traits, solution="cute_dsl"
    )
    assert pinned.name == LEAF
