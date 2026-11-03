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

"""EAGLE-4's draft-tree lanes (``Eagle4TreeLanes``) on a real draft router:
each lane row decodes over a copy of its request's live table row, and on the
trtllm leaf the lanes' windows past 64 slots and sliding-window rows match an
fp32 reference."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402
from runtime.cache_pool_test_utils import make_mha_memory_plan, make_pool  # noqa: E402
from runtime.test_cache_group_router import FULL, _geometry, _StubLeaf  # noqa: E402
from runtime.test_tree_attention_cascade import (  # noqa: E402
    ATOL,
    HKV,
    HQ,
    KV_DTYPES,
    PAGE,
    D,
)
from runtime.test_tree_attention_cascade import _backend as _trtllm_leaf  # noqa: E402

register_cuda_ci(
    est_time=30,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="TRT-LLM tree attention requires an NVIDIA GPU.",
)

from tokenspeed.runtime.execution.drafter.eagle_harness.lanes import (  # noqa: E402
    Eagle4TreeLanes,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.layers.attention.backends.paged.cache_group_geometry import (  # noqa: E402
    CacheGroupGeometry,
)
from tokenspeed.runtime.layers.attention.backends.paged.router import (  # noqa: E402
    CacheGroupRouter,
)
from tokenspeed.runtime.layers.attention.kv_cache.mha import (  # noqa: E402
    MHATokenToKVPool,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention  # noqa: E402

requires_nvidia = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="needs an NVIDIA GPU",
)


def _attention(
    num_heads: int, num_kv_heads: int, head_dim: int, sliding_window_size: int = -1
) -> PagedAttention:
    """A draft attention layer on the full-history group."""
    layer = PagedAttention(
        num_heads,
        head_dim,
        head_dim**-0.5,
        num_kv_heads,
        0,
        sliding_window_size=sliding_window_size,
        rotary_emb=None,
        qk_norm=None,
    )
    layer.bind_cache_group(FULL)
    return layer


def _lane_router():
    """A single-group draft router with sliding-window EAGLE-4 lanes (K = 2, 3 steps) bound."""
    leaf = _StubLeaf(4, is_draft=True, spec=2)
    router = CacheGroupRouter(
        None, is_draft=True, spec_num_tokens=2, device="cpu", consumed_group_ids=None
    )
    router.bind(_geometry(), {FULL: leaf})
    lanes = Eagle4TreeLanes(2, 3, 4, torch.device("cpu"), slides=True)
    router.bind_tree_draft(lanes)
    router.init_cuda_graph_state(4)
    return router, leaf, lanes


def _tables(flip=False):
    table = torch.tensor([[5, 6, -1], [9, 0, 2]], dtype=torch.int32)
    return {FULL: table.flip(0) if flip else table}


def _refresh(router, *, flip=False):
    router.refresh_decode_metadata(
        2,
        2,
        torch.arange(2, dtype=torch.int32),
        torch.tensor([9, 4], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        block_tables=_tables(flip=flip),
    )


def test_lane_rows_decode_over_their_requests_live_table_rows():
    """Each lane row decodes over a copy of its request's table row from its
    request's frontier, following the router's refreshes; the lanes write
    through the router's published decode window, as plain lanes do."""
    router, leaf, lanes = _lane_router()
    _refresh(router)
    start = torch.tensor([5, 1], dtype=torch.int32)
    lanes.set_frontier(2, start)
    lanes.active = True
    lane_slots = router.publish_draft_step_locations(start, 2)
    assert lane_slots.tolist() == [25, 26, 37, 38]
    written = router.write_locations(_attention(8, 8, 16), ForwardMode.DECODE)
    assert written.tolist() == lane_slots.tolist()
    rows = lanes.cascade_rows(leaf, 2)
    assert rows.page_table[:, :2].tolist() == [[5, 6], [9, 0]]
    assert rows.row_page_table[:, :2].tolist() == [[5, 6], [5, 6], [9, 0], [9, 0]]
    assert rows.row_prefix_lens.tolist() == [5, 5, 1, 1]
    # The next round's refresh, with the requests reordered.
    _refresh(router, flip=True)
    rows = lanes.cascade_rows(leaf, 2)
    assert rows.row_page_table[:, :2].tolist() == [[9, 0], [9, 0], [5, 6], [5, 6]]
    lanes.active = False


def _window_masks(gen, num_rows, window, seen_width):
    """Random visibility over each row's first ``seen_width`` window keys: per
    row as an integer, and as the lanes' int64 words (bit j % 64 of word j // 64)."""
    words = (window + 63) // 64
    draws = torch.randint(0, 1 << 62, (num_rows, words), generator=gen).tolist()
    row_bits = [
        sum(x << (62 * c) for c, x in enumerate(row)) & ((1 << seen_width) - 1)
        for row in draws
    ]
    split = [(b >> (64 * w)) & ((1 << 64) - 1) for b in row_bits for w in range(words)]
    # Bit 63 of each word is the sign.
    return row_bits, torch.tensor([x - (1 << 64) if x >> 63 else x for x in split])


def _lane_problem(frontier, rows, window, max_pages, gen, kv_dtype, frozen, poison):
    """Paged K/V rows for every request's keys, lane queries, window masks and
    the fp32 reference: row r sees the prefix and window key j when bit j is
    set; a ``frozen`` sliding window of W keys limits it to the prefix's last
    W keys. K/V hold values exact in ``kv_dtype``; trtllm-gen sees the query
    cast to it over the keys its prefix decode covers (row r of a causal
    decode: the first ``prefix - rows + 1 + r``; a frozen row: the prefix).
    With a ``poison``, rows see only step 1's slots and the window slots no
    row sees (later lane steps, unwritten yet) get that V."""
    seq_lens = [f + window for f in frontier]
    bs = len(seq_lens)
    pages = (max(seq_lens) + PAGE - 1) // PAGE
    table = torch.zeros(bs, max_pages, dtype=torch.int32)
    table[:, :pages] = (torch.randperm(bs * pages, generator=gen) + 1).view(bs, pages)
    rows_shape = ((bs * pages + 1) * PAGE, HKV, D)
    k_rows = torch.randn(rows_shape, generator=gen).to(kv_dtype).bfloat16()
    v_rows = torch.randn(rows_shape, generator=gen).to(kv_dtype).bfloat16()
    q = torch.randn(bs * rows, HQ, D, generator=gen).bfloat16()
    q_prefix = q.to(kv_dtype).float()
    row_bits, bits = _window_masks(
        gen, bs * rows, window, window if poison is None else rows
    )
    ref = torch.empty(bs * rows, HQ, D)
    for b, length in enumerate(seq_lens):
        pos = torch.arange(length)
        slots = (table[b, pos // PAGE].long() * PAGE + pos % PAGE).tolist()
        kk = k_rows[slots].float().repeat_interleave(HQ // HKV, 1)
        vv = v_rows[slots].float().repeat_interleave(HQ // HKV, 1)
        prefix = frontier[b]
        for i in range(rows):
            bits_i = row_bits[b * rows + i]
            vis = torch.tensor(
                [
                    j < prefix or bool((bits_i >> (j - prefix)) & 1)
                    for j in range(length)
                ]
            )
            if frozen is not None:
                vis[: max(prefix - frozen, 0)] = False
            covered = pos < (prefix - rows + 1 + i if frozen is None else prefix)
            s = torch.where(
                covered[None],
                torch.einsum("hd,xhd->hx", q_prefix[b * rows + i], kk),
                torch.einsum("hd,xhd->hx", q[b * rows + i].float(), kk),
            )
            s = (s * D**-0.5).masked_fill(~vis[None], float("-inf"))
            ref[b * rows + i] = torch.einsum("hx,xhd->hd", torch.softmax(s, -1), vv)
    if poison is not None:
        for b, prefix in enumerate(frontier):
            # trtllm-gen loads the prefix's last page whole: poison later pages only.
            for j in range(rows, window):
                pos = prefix + j
                if pos // PAGE > (prefix - 1) // PAGE:
                    v_rows[int(table[b, pos // PAGE]) * PAGE + pos % PAGE] = poison
    plan = make_mha_memory_plan(
        size=bs * pages * PAGE,
        prefix_granularity=PAGE,
        layer_num=1,
        kv_heads=HKV,
        head_dim=D,
        dtype=kv_dtype,
    )
    _, pool = make_pool(
        MHATokenToKVPool,
        plan,
        device="cuda",
        dtype=kv_dtype,
        head_num=HKV,
        head_dim=D,
        layer_num=1,
        rank=0,
    )
    k_cache, v_cache = pool.get_kv_buffer(0)
    k_cache.copy_(k_rows.cuda().to(kv_dtype))
    v_cache.copy_(v_rows.cuda().to(kv_dtype))
    return table.cuda(), q.cuda(), bits.cuda(), pool, ref


@requires_nvidia
@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
@pytest.mark.parametrize(
    "frozen,poison", [(None, None), (None, float("nan")), (128, None)]
)
@pytest.mark.parametrize("topk,steps", [(2, 7), (4, 12), (8, 13)])
def test_trtllm_lanes_match_reference(topk, steps, frozen, poison, kv_dtype):
    """Lane rows over the prefix and their window bits, up to two mask words;
    a sliding-window draft layer sees the W prefix keys ending at the frontier
    (frozen for the round)."""
    gen = torch.Generator().manual_seed(topk * 10 + steps)
    window = (steps - 1) * topk
    frontier = [700, 1, 3, 130]
    bs = len(frontier)
    leaf = _trtllm_leaf(window, max_bs=8, is_draft=True, kv_dtype=kv_dtype)
    table, q, bits, pool, ref = _lane_problem(
        frontier, topk, window, leaf.max_num_pages, gen, kv_dtype, frozen, poison
    )
    router = CacheGroupRouter(
        None, is_draft=True, spec_num_tokens=1, device="cuda", consumed_group_ids=None
    )
    geometry = CacheGroupGeometry(
        granularities={FULL: PAGE},
        attn_kinds={FULL: "full_history"},
        full_history_group_id=FULL,
        row_geometry={FULL: (PAGE, 1)},
        windows={FULL: None},
    )
    router.bind(geometry, {FULL: leaf})
    router.init_cuda_graph_state(8)
    lanes = Eagle4TreeLanes(
        topk, steps, 8, torch.device("cuda"), slides=frozen is not None
    )
    router.bind_tree_draft(lanes)
    lengths = torch.tensor(frontier, dtype=torch.int32, device="cuda")
    router.refresh_decode_metadata(
        bs,
        bs,
        torch.arange(bs, dtype=torch.int32, device="cuda"),
        lengths,
        forward_mode=ForwardMode.DECODE,
        block_tables={FULL: table},
    )
    lanes.set_frontier(bs, lengths)
    lanes.lane_mask[: bits.numel()] = bits
    layer = _attention(HQ, HKV, D, -1 if frozen is None else frozen - 1)
    lanes.active = True
    out = leaf.forward_decode(q, None, None, layer, None, pool, bs)
    lanes.active = False
    torch.testing.assert_close(
        out.view(-1, HQ, D).float().cpu(), ref, atol=ATOL[kv_dtype], rtol=2e-2
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
