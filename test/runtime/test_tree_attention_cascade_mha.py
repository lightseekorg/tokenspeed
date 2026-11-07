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

"""Draft-tree attention on the portable MHA leaf (the ROCm default;
docs/design/tree-speculation.md): the Triton causal prefix extend plus
``tree_window_attention``, reached through the verify and lane dispatch,
against an fp32 reference."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

from tokenspeed.runtime.execution.drafter.eagle_harness.lanes import (  # noqa: E402
    Eagle4TreeLanes,
)
from tokenspeed.runtime.layers.attention.backends.paged.mha import (  # noqa: E402
    MHAAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (  # noqa: E402
    TreeDraftInputs,
    TreeVerifyInputs,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig  # noqa: E402
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

PAGE = 64
HQ, HKV = 16, 4


def _backend(num_draft_tokens: int, max_bs: int, is_draft: bool, dim: int):
    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=HQ,
        num_kv_heads=HKV,
        head_dim=dim,
        attn_tp_size=1,
    )
    cfg = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kernel_page_size=PAGE,
        context_len=2048,
        max_bs=max_bs,
        kv_cache_quant_method="none",
        speculative_num_steps=3,
        speculative_num_draft_tokens=num_draft_tokens,
        is_draft=is_draft,
        components=(spec,),
    )
    backend = MHAAttnBackend(cfg, spec, kernel_page_size=PAGE)
    backend.init_cuda_graph_state(max_bs)
    return backend


def _layer(dim: int, sliding_window_size: int = -1):
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=HQ,
        tp_k_head_num=HKV,
        tp_v_head_num=HKV,
        qk_head_dim=dim,
        v_head_dim=dim,
        scaling=dim**-0.5,
        logit_cap=0.0,
        sliding_window_size=sliding_window_size,
    )


def _problem(seq_lens, rows, window, max_pages, gen, dim, poison=None, window_left=-1):
    """Paged K/V rows for every request's keys, queries, window masks, and the
    fp32 reference: row r sees the prefix and window key j when bit j % 64 of
    its mask word j // 64 is set. With ``window_left`` the row decodes at the
    prefix end and sees only the last ``window_left + 1`` prefix keys."""
    bs = len(seq_lens)
    pages = (max(seq_lens) + PAGE - 1) // PAGE
    table = torch.zeros(bs, max_pages, dtype=torch.int32)
    table[:, :pages] = (torch.randperm(bs * pages, generator=gen) + 1).view(bs, pages)
    rows_shape = ((bs * pages + 1) * PAGE, HKV, dim)
    k_rows = torch.randn(rows_shape, generator=gen).bfloat16()
    v_rows = torch.randn(rows_shape, generator=gen).bfloat16()
    q = torch.randn(bs * rows, HQ, dim, generator=gen).bfloat16()
    words = (window + 63) // 64
    if words == 1:
        bits = torch.randint(0, 1 << 62, (bs * rows,), generator=gen)
        # With a poison, rows only see step 1's slots (later lane steps unwritten yet).
        seen_width = window if poison is None else rows
        if seen_width < 62:
            bits &= (1 << seen_width) - 1
    else:
        assert poison is None
        # Full 64-bit words (the sign bit included), the last one cut to the window.
        bits = torch.randint(
            -(1 << 63), (1 << 63) - 1, (bs * rows, words), generator=gen
        )
        if window % 64:
            bits[:, -1] &= (1 << (window % 64)) - 1
        bits = bits.flatten()
    # Each row's mask as one Python int: window slot j is bit j.
    row_bits = [
        sum((w & ((1 << 64) - 1)) << (64 * i) for i, w in enumerate(row))
        for row in bits.view(bs * rows, words).tolist()
    ]
    ref = torch.empty(bs * rows, HQ, dim)
    for b, length in enumerate(seq_lens):
        pos = torch.arange(length)
        slots = (table[b, pos // PAGE].long() * PAGE + pos % PAGE).tolist()
        kk = k_rows[slots].float().repeat_interleave(HQ // HKV, 1)
        vv = v_rows[slots].float().repeat_interleave(HQ // HKV, 1)
        prefix = length - window
        first = 0 if window_left < 0 else prefix - 1 - window_left
        for i in range(rows):
            vis = torch.tensor(
                [
                    first <= j < prefix
                    or (
                        j >= prefix
                        and bool((row_bits[b * rows + i] >> (j - prefix)) & 1)
                    )
                    for j in range(length)
                ]
            )
            s = torch.einsum("hd,xhd->hx", q[b * rows + i].float(), kk)
            s = (s * dim**-0.5).masked_fill(~vis[None], float("-inf"))
            ref[b * rows + i] = torch.einsum("hx,xhd->hd", torch.softmax(s, -1), vv)
    if poison is not None:
        # Window slots no row of the request sees get a non-finite V after the reference.
        for b, length in enumerate(seq_lens):
            seen = 0
            for i in range(rows):
                seen |= row_bits[b * rows + i]
            prefix = length - window
            for j in range(rows, window):
                if not (seen >> j) & 1:
                    pos = prefix + j
                    v_rows[int(table[b, pos // PAGE]) * PAGE + pos % PAGE] = poison
    k_rows, v_rows = k_rows.cuda(), v_rows.cuda()
    pool = SimpleNamespace(
        get_key_buffer=lambda _: k_rows, get_value_buffer=lambda _: v_rows
    )
    return table.cuda(), q.cuda(), bits.cuda(), pool, ref


@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("nodes", [4, 16, 64])
def test_tree_verify_matches_reference(nodes, dim):
    gen = torch.Generator().manual_seed(nodes + dim)
    # Long and short committed prefixes, including ones shorter than the window.
    committed = [700, 1, nodes - 2, 130]
    seq_lens = [c + nodes for c in committed]
    backend = _backend(nodes, max_bs=8, is_draft=False, dim=dim)
    table, q, bits, pool, ref = _problem(
        seq_lens, nodes, nodes, backend.max_num_pages, gen, dim
    )
    mask = torch.zeros(8 * nodes, dtype=torch.int64, device="cuda")
    mask[: len(seq_lens) * nodes] = bits
    backend.bind_tree_verify(
        TreeVerifyInputs(mask, nodes, torch.zeros(8, nodes, dtype=torch.int32))
    )
    bs = len(seq_lens)
    backend.refresh_decode_metadata(
        bs, bs, torch.tensor(seq_lens, dtype=torch.int32, device="cuda"), table
    )
    out = backend.forward_decode(q, None, None, _layer(dim), None, pool, bs)
    torch.testing.assert_close(
        out.view(-1, HQ, dim).float().cpu(), ref, atol=2e-2, rtol=2e-2
    )


@pytest.mark.parametrize("poison", [None, float("nan")])
@pytest.mark.parametrize("topk,steps", [(2, 7), (4, 4), (8, 5)])
def test_tree_lanes_match_reference(topk, steps, poison):
    dim = 128
    gen = torch.Generator().manual_seed(topk * 10 + steps)
    window = (steps - 1) * topk
    frontier = [700, 1, 3, 130]
    seq_lens = [f + window for f in frontier]
    backend = _backend(window, max_bs=8, is_draft=True, dim=dim)
    table, q, bits, pool, ref = _problem(
        seq_lens, topk, window, backend.max_num_pages, gen, dim, poison
    )
    lanes = TreeDraftInputs(topk, steps, 8, torch.device("cuda"))
    backend.bind_tree_draft(lanes)
    bs = len(frontier)
    lanes.set_frontier(bs, torch.tensor(frontier, dtype=torch.int32, device="cuda"))
    lanes.lane_mask[: bs * topk] = bits
    backend.refresh_decode_metadata(
        bs, bs, torch.tensor(frontier, dtype=torch.int32, device="cuda"), table
    )
    lanes.active = True
    out = backend.forward_decode(q, None, None, _layer(dim), None, pool, bs)
    lanes.active = False
    torch.testing.assert_close(
        out.view(-1, HQ, dim).float().cpu(), ref, atol=2e-2, rtol=2e-2
    )


# 80 slots: two mask words per row, the second partly used.
@pytest.mark.parametrize("topk,steps", [(4, 5), (8, 11)])
@pytest.mark.parametrize("window_left", [-1, 40])
def test_eagle4_lanes_match_reference(window_left, topk, steps):
    """EAGLE-4 lanes: a sliding-window draft layer decodes each lane row's
    prefix alone (per-row tables), then its window ancestors; windows past 64
    slots read the ancestors from the next mask word."""
    dim = 128
    gen = torch.Generator().manual_seed(window_left + topk * steps)
    window = (steps - 1) * topk
    # Prefixes past, at, and inside the sliding window.
    frontier = [700, 41, 3, 130]
    seq_lens = [f + window for f in frontier]
    backend = _backend(window, max_bs=8, is_draft=True, dim=dim)
    table, q, bits, pool, ref = _problem(
        seq_lens, topk, window, backend.max_num_pages, gen, dim, None, window_left
    )
    lanes = Eagle4TreeLanes(
        topk, steps, 8, torch.device("cuda"), slides=window_left >= 0
    )
    assert lanes.mask_words == (window + 63) // 64
    backend.bind_tree_draft(lanes)
    bs = len(frontier)
    lanes.set_frontier(bs, torch.tensor(frontier, dtype=torch.int32, device="cuda"))
    lanes.lane_mask[: bs * topk * lanes.mask_words] = bits
    backend.refresh_decode_metadata(
        bs, bs, torch.tensor(frontier, dtype=torch.int32, device="cuda"), table
    )
    lanes.active = True
    out = backend.forward_decode(
        q, None, None, _layer(dim, window_left), None, pool, bs
    )
    lanes.active = False
    torch.testing.assert_close(
        out.view(-1, HQ, dim).float().cpu(), ref, atol=2e-2, rtol=2e-2
    )


@pytest.mark.parametrize("lane_rows", [False, True], ids=["verify", "lanes"])
def test_tree_cascade_graph_replay_matches_eager(lane_rows):
    """One graph captured at the capture seeding replays prefixes of zero,
    short and long length and padding requests: every replay refreshes the
    cascade metadata (the verify prefix lengths, the lane frontiers) in place,
    and its live rows equal an eager forward of the same inputs."""
    dim, graph_bs = 128, 4
    if lane_rows:
        topk, steps = 4, 5
        rows, window = topk, (steps - 1) * topk
    else:
        rows = window = 16
    backend = _backend(window, max_bs=8, is_draft=lane_rows, dim=dim)
    pages = backend.max_num_pages
    lanes = None
    if lane_rows:
        lanes = Eagle4TreeLanes(topk, steps, 8, torch.device("cuda"), slides=False)
        backend.bind_tree_draft(lanes)
        mask = lanes.lane_mask
    else:
        mask = torch.zeros(8 * rows, dtype=torch.int64, device="cuda")
        backend.bind_tree_verify(
            TreeVerifyInputs(mask, rows, torch.zeros(8, rows, dtype=torch.int32))
        )
    # The graph reads these buffers; each replay copies a new problem in.
    rows_shape = ((graph_bs * pages + 1) * PAGE, HKV, dim)
    k_buf = torch.zeros(rows_shape, dtype=torch.bfloat16, device="cuda")
    v_buf = torch.zeros_like(k_buf)
    pool = SimpleNamespace(
        get_key_buffer=lambda _: k_buf, get_value_buffer=lambda _: v_buf
    )
    q_buf = torch.zeros(graph_bs * rows, HQ, dim, dtype=torch.bfloat16, device="cuda")
    layer = _layer(dim)

    def forward(bs):
        if lanes is not None:
            lanes.active = True
        out = backend.forward_decode(
            q_buf[: bs * rows], None, None, layer, None, pool, bs
        )
        if lanes is not None:
            lanes.active = False
        return out

    def set_frontier(frontier):
        if lanes is not None:
            lanes.set_frontier(
                len(frontier), torch.tensor(frontier, dtype=torch.int32, device="cuda")
            )

    # Capture seeding: every request is padding (seq_len 1, null pages).
    backend.init_forward_metadata_capture_cuda_graph(
        graph_bs,
        torch.ones(graph_bs, dtype=torch.int32, device="cuda"),
        torch.zeros(graph_bs, pages, dtype=torch.int32, device="cuda"),
    )
    set_frontier([0] * graph_bs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward(graph_bs)  # compile outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = forward(graph_bs)

    # Committed prefixes of the live requests; the rest of the batch is padding.
    for case, committed in enumerate([[0, 0, 130, 700], [3, 64, 1], [0]]):
        gen = torch.Generator().manual_seed(case)
        live = len(committed)
        seq_lens = [c + window for c in committed]
        table, q, bits, case_pool, ref = _problem(
            seq_lens, rows, window, pages, gen, dim
        )
        case_k = case_pool.get_key_buffer(0)
        k_buf.zero_()
        v_buf.zero_()
        k_buf[: case_k.shape[0]].copy_(case_k)
        v_buf[: case_k.shape[0]].copy_(case_pool.get_value_buffer(0))
        q_buf.zero_()
        q_buf[: live * rows].copy_(q)
        mask.zero_()
        mask[: live * rows].copy_(bits)
        # Lanes decode at the frontier; verify presents its keys window included.
        live_lens = committed if lane_rows else seq_lens
        padded_lens = torch.tensor(
            live_lens + [1] * (graph_bs - live), dtype=torch.int32, device="cuda"
        )
        padded_table = torch.cat([table, table.new_zeros(graph_bs - live, pages)])

        backend.refresh_decode_metadata(
            graph_bs, live, padded_lens, padded_table, for_graph_replay=True
        )
        set_frontier(committed + [0] * (graph_bs - live))
        graph.replay()
        replayed = graph_out[: live * rows].float().cpu()

        backend.refresh_decode_metadata(live, live, padded_lens, table)
        set_frontier(committed)
        eager = forward(live).float().cpu()
        torch.testing.assert_close(replayed, eager, atol=0, rtol=0)
        torch.testing.assert_close(
            replayed.view(-1, HQ, dim), ref, atol=2e-2, rtol=2e-2
        )


def test_tree_verify_refuses_a_sliding_window_layer():
    """Sliding windows are per layer, so the forward refuses them, not tree_support."""
    gen = torch.Generator().manual_seed(7)
    nodes, seq_lens, dim = 4, [40, 50], 128
    backend = _backend(nodes, max_bs=8, is_draft=False, dim=dim)
    table, q, _, pool, _ = _problem(
        seq_lens, nodes, nodes, backend.max_num_pages, gen, dim
    )
    backend.bind_tree_verify(
        TreeVerifyInputs(
            torch.zeros(8 * nodes, dtype=torch.int64, device="cuda"),
            nodes,
            torch.zeros(8, nodes, dtype=torch.int32),
        )
    )
    bs = len(seq_lens)
    backend.refresh_decode_metadata(
        bs, bs, torch.tensor(seq_lens, dtype=torch.int32, device="cuda"), table
    )
    with pytest.raises(NotImplementedError, match="sliding-window"):
        backend.forward_decode(q, None, None, _layer(dim, 16), None, pool, bs)


def test_tree_support_blocks_an_fp8_cache():
    """The window kernel reads the cache unscaled, so a scaled FP8 cache has no tree path."""
    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=HQ,
        num_kv_heads=HKV,
        head_dim=128,
        attn_tp_size=1,
    )
    cfg = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        kernel_page_size=PAGE,
        context_len=2048,
        max_bs=8,
        kv_cache_quant_method="none",
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        is_draft=False,
        components=(spec,),
    )
    support = MHAAttnBackend(cfg, spec, kernel_page_size=PAGE).tree_support()
    assert support.verify_blocker is not None and support.draft_blocker is not None
    assert _backend(4, 8, False, 128).tree_support().verify_blocker is None
