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

"""Tree speculative decoding kernels against plain references."""

import random

import pytest
import torch
from tokenspeed_kernel.ops.attention.tree import tree_window_attention
from tokenspeed_kernel.ops.kvcache.triton import compact_window_rows
from tokenspeed_kernel.ops.sampling.triton.logprob_topk import logprob_topk
from tokenspeed_kernel.ops.sampling.triton.tree_verify import verify_tree

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _last_dim_strided(t):
    """The same values viewed with stride 2 along the last dim."""
    backing = torch.zeros(
        *t.shape[:-1], 2 * t.shape[-1], dtype=t.dtype, device=t.device
    )
    backing[..., ::2] = t
    return backing[..., ::2]


def _reference_verify(cands, parents, target):
    bs, n = cands.shape
    predicts = torch.zeros(bs * n, dtype=torch.int32)
    lengths, paths = [], torch.full((bs, n), -1, dtype=torch.int32)
    for b in range(bs):
        cur, path = 0, [0]
        while True:
            pick = int(target[b * n + cur])
            kids = [
                j
                for j in range(n)
                if int(parents[b, j]) == cur and int(cands[b, j]) == pick
            ]
            if not kids:
                break
            predicts[b * n + len(path) - 1] = pick
            cur = kids[0]
            path.append(cur)
        predicts[b * n + len(path) - 1] = int(target[b * n + cur])
        lengths.append(len(path))
        paths[b, : len(path)] = torch.tensor(path, dtype=torch.int32)
    return predicts, torch.tensor(lengths, dtype=torch.int32), paths


def _random_tree(n, max_depth, rng):
    parents, depth = [-1], [0]
    for j in range(1, n):
        choices = [i for i in range(j) if depth[i] < max_depth]
        p = rng.choice(choices)
        parents.append(p)
        depth.append(depth[p] + 1)
    return parents


@pytest.mark.parametrize(
    "n,max_depth,vocab", [(8, 3, 3), (32, 6, 4), (64, 10, 5), (5, 4, 2)]
)
def test_verify_tree_matches_reference(n, max_depth, vocab):
    rng = random.Random(n)
    bs = 16
    parents = torch.tensor(
        [_random_tree(n, max_depth, rng) for _ in range(bs)], dtype=torch.int32
    )
    cands = torch.randint(0, vocab, (bs, n), dtype=torch.int32)
    target = torch.randint(0, vocab, (bs * n,), dtype=torch.int32)
    # A chain row deeper than max_depth and an all-accept row.
    parents[0] = torch.arange(-1, n - 1)
    cands[1, 1:] = 0
    target[n : 2 * n] = 0
    ref = _reference_verify(cands, parents, target)
    out = [
        torch.zeros(bs * n, dtype=torch.int32),
        torch.zeros(bs, dtype=torch.int32),
        torch.zeros(bs, n, dtype=torch.int32),
    ]
    out = [t.cuda() for t in out]
    verify_tree(*out, cands.cuda(), parents.cuda(), target.cuda())
    predicts, lengths, paths = [t.cpu() for t in out]
    assert torch.equal(lengths, ref[1])
    assert torch.equal(paths, ref[2])
    for b in range(bs):
        assert torch.equal(
            predicts[b * n : b * n + int(lengths[b])],
            ref[0][b * n : b * n + int(lengths[b])],
        )


def _paged_tree_problem(
    bs, n, prefix_lens, hq, hkv, d, page, gen, rows=None, seen_width=None
):
    """A paged HND cache holding each request's prefix + ``n``-key tree window,
    plus the reference. ``rows=None`` is verify (a row per node, ancestor
    masks); otherwise ``rows`` lane rows per request with random window masks."""
    r = n if rows is None else rows
    scale = d**-0.5
    pages_per_req = max((p + n + page - 1) // page for p in prefix_lens)
    k_cache = torch.zeros(bs * pages_per_req + 1, hkv, page, d, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    tables = torch.zeros(bs, pages_per_req, dtype=torch.int32)
    q = torch.randn(bs * r, hq, d, generator=gen).bfloat16()
    kt = torch.randn(bs * n, hkv, d, generator=gen).bfloat16()
    vt = torch.randn(bs * n, hkv, d, generator=gen).bfloat16()
    mask = torch.zeros(bs * r, dtype=torch.int64)
    refs = []
    for b, plen in enumerate(prefix_lens):
        tables[b] = torch.arange(pages_per_req) + 1 + b * pages_per_req
        kp = torch.randn(plen, hkv, d, generator=gen).bfloat16()
        vp = torch.randn(plen, hkv, d, generator=gen).bfloat16()
        keys = torch.cat([kp, kt[b * n : (b + 1) * n]])
        vals = torch.cat([vp, vt[b * n : (b + 1) * n]])
        for pos in range(plen + n):
            pg, off = tables[b, pos // page], pos % page
            k_cache[pg, :, off] = keys[pos]
            v_cache[pg, :, off] = vals[pos]
        parent = [-1] + [
            int(torch.randint(0, j, (1,), generator=gen)) for j in range(1, n)
        ]
        for i in range(r):
            if rows is None:
                bits, cur = 0, i
                while cur >= 0:
                    bits |= 1 << cur
                    cur = parent[cur]
            else:
                bits = int(torch.randint(0, 2**62, (1,), generator=gen)) & (
                    (1 << (n if seen_width is None else seen_width)) - 1
                )
            mask[b * r + i] = (
                bits - (1 << 64) if bits >> 63 else bits
            )  # bit 63 is the sign
        group = hq // hkv
        kk = keys.float().repeat_interleave(group, 1)
        vv = vals.float().repeat_interleave(group, 1)
        qq = q[b * r : (b + 1) * r].float()
        s = torch.einsum("nhd,xhd->hnx", qq, kk) * scale
        vis = torch.ones(r, plen + n, dtype=torch.bool)
        for i in range(r):
            for j in range(n):
                vis[i, plen + j] = bool((int(mask[b * r + i]) >> j) & 1)
        s = s.masked_fill(~vis[None], float("-inf"))
        refs.append(torch.einsum("hnx,xhd->nhd", torch.softmax(s, -1), vv))
    return q, kt, vt, mask, k_cache, v_cache, tables, torch.cat(refs), scale


@pytest.mark.parametrize("n", [8, 16, 4, 32, 64])
@pytest.mark.parametrize("prefix_lens", [[5, 70, 131], [1, 2, 64], [0, 3, 40]])
def test_tree_window_attention_matches_reference(n, prefix_lens, require):
    require("attention", "tree_window", "triton", torch.bfloat16, "q")
    gen = torch.Generator().manual_seed(n * 10 + prefix_lens[0])
    bs, hq, hkv, d, page = 3, 32, 8, 128, 32
    _check_tree_window(bs, n, prefix_lens, hq, hkv, d, page, gen)


@pytest.mark.parametrize(
    "hq,hkv,d,n",
    [
        (16, 2, 256, 64),
        (16, 2, 256, 17),
        (24, 4, 256, 64),
        (32, 4, 128, 64),
        (32, 2, 128, 64),
    ],
)
def test_tree_window_attention_large_query_tiles(hq, hkv, d, n, require):
    require("attention", "tree_window", "triton", torch.bfloat16, "q")
    """N x GQA group beyond one row tile (e.g. GQA 8 at head_dim 256) still matches."""
    gen = torch.Generator().manual_seed(hq * n + d)
    _check_tree_window(2, n, [40, 97], hq, hkv, d, 64, gen)


@pytest.mark.parametrize("lanes,window", [(2, 12), (4, 12), (8, 32), (16, 64), (1, 6)])
def test_tree_window_attention_lane_rows(lanes, window, require):
    require("attention", "tree_window", "triton", torch.bfloat16, "q")
    """Draft lanes: K rows per request over a wider (S - 1) * K lane window."""
    gen = torch.Generator().manual_seed(lanes * 100 + window)
    _check_tree_window(3, window, [5, 70, 1], 32, 8, 128, 32, gen, rows=lanes)


def test_tree_window_attention_last_dim_strided(require):
    require("attention", "tree_window", "triton", torch.bfloat16, "q")
    gen = torch.Generator().manual_seed(20260930)
    _check_tree_window(2, 16, [40, 97], 32, 8, 128, 32, gen, strided=True)


@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_tree_window_attention_ignores_nonfinite_v_in_unseen_slots(poison, require):
    require("attention", "tree_window", "triton", torch.bfloat16, "q")
    """Lane window slots no row sees (later steps, padding) stay out of P.V."""
    gen = torch.Generator().manual_seed(7)
    _check_tree_window(3, 12, [100, 3, 1], 16, 2, 128, 16, gen, rows=4, poison=poison)


def _causal_prefix_partial(q, k_rows, v_rows, tables, prefix_lens, r, page, scale):
    """What a causal ``q_len = r`` decode over each request's prefix returns: row
    ``i`` attends keys ``[0, P - r + 1 + i)``; bf16 output and base-2 LSE."""
    hq, hkv = q.shape[1], k_rows.shape[1]
    group = hq // hkv
    out = torch.zeros_like(q)
    lse = torch.full(q.shape[:2], float("-inf"))
    for b, plen in enumerate(prefix_lens):
        pos = torch.arange(plen)
        slots = (tables[b, pos // page].long() * page + pos % page).tolist()
        kk = k_rows[slots].float().repeat_interleave(group, 1)
        vv = v_rows[slots].float().repeat_interleave(group, 1)
        for i in range(r):
            covered = plen - r + 1 + i
            if covered <= 0:
                continue
            s = torch.einsum("hd,xhd->hx", q[b * r + i].float(), kk[:covered]) * scale
            out[b * r + i] = torch.einsum(
                "hx,xhd->hd", torch.softmax(s, -1), vv[:covered]
            ).bfloat16()
            lse[b * r + i] = torch.logsumexp(s, -1) / torch.log(torch.tensor(2.0))
    return out, lse


def _check_tree_window(
    bs, n, prefix_lens, hq, hkv, d, page, gen, strided=False, rows=None, poison=None
):
    q, _, _, mask, k_cache, v_cache, tables, ref, scale = _paged_tree_problem(
        bs,
        n,
        prefix_lens,
        hq,
        hkv,
        d,
        page,
        gen,
        rows,
        None if poison is None else rows,
    )
    r = n if rows is None else rows
    if poison is not None:
        for b, plen in enumerate(prefix_lens):
            seen = 0
            for i in range(r):
                seen |= int(mask[b * r + i]) & ((1 << n) - 1)
            for j in range(n):
                if not (seen >> j) & 1:
                    pos = plen + j
                    v_cache[tables[b, pos // page], :, pos % page] = poison
    k_rows = k_cache.permute(0, 2, 1, 3).reshape(-1, hkv, d)
    v_rows = v_cache.permute(0, 2, 1, 3).reshape(-1, hkv, d)
    prefix_out, prefix_lse = _causal_prefix_partial(
        q, k_rows, v_rows, tables, prefix_lens, r, page, scale
    )
    layout = _last_dim_strided if strided else (lambda t: t)
    out = tree_window_attention(
        layout(q.cuda()),
        layout(k_rows.cuda()),
        layout(v_rows.cuda()),
        tables.cuda(),
        torch.tensor(prefix_lens, dtype=torch.int32).cuda() + n,
        mask.cuda(),
        layout(prefix_out.cuda()),
        prefix_lse.cuda(),
        rows_per_req=r,
        window=n,
        page_size=page,
        sm_scale=scale,
    )
    torch.testing.assert_close(out.float().cpu(), ref, atol=2e-2, rtol=2e-2)


def test_logprob_topk_row_offset_beyond_int32():
    """rows x vocab past 2**31 elements must address rows in 64 bits."""
    vocab, rows = 248320, 8700
    logits = torch.zeros(rows, vocab, device="cuda", dtype=torch.bfloat16)
    logits[-1, 12345] = 30.0
    scores, ids = logprob_topk(logits, 2)
    assert int(ids[-1, 0]) == 12345
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize(
    "rows,vocab,k",
    [(1, 32000, 1), (32, 32000, 4), (64, 128256, 8), (5, 10, 8), (4, 248320, 4)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_logprob_topk_matches_torch(rows, vocab, k, dtype):
    gen = torch.Generator().manual_seed(rows + vocab + k)
    logits = (torch.randn(rows, vocab, generator=gen) * 4).to(dtype).cuda()
    scores, ids = logprob_topk(logits, k)
    ref_scores, _ = torch.topk(torch.log_softmax(logits.float(), -1), k)
    torch.testing.assert_close(scores, ref_scores, atol=1e-4, rtol=1e-5)
    picked = torch.log_softmax(logits.float(), -1).gather(1, ids)
    torch.testing.assert_close(picked, scores, atol=1e-4, rtol=1e-5)
    assert all(len(set(r)) == k for r in ids.tolist())


@pytest.mark.parametrize("nodes", [8, 64])
def test_compact_window_rows_moves_every_buffer(nodes):
    """Every buffer (bf16 and fp8 planes alike) packs each accepted path to the window front."""
    gen = torch.Generator().manual_seed(nodes)
    bs, slots, heads, dim = 3, 4096, 2, 128
    buffers = [
        torch.randn(slots, heads, dim, generator=gen).bfloat16().cuda()
        for _ in range(3)
    ] + [
        torch.randn(slots, heads, 2 * dim, generator=gen).to(torch.float8_e4m3fn).cuda()
    ]  # same bytes per token row as the bf16 planes
    locs = torch.randperm(slots, generator=gen)[: bs * nodes].int().cuda()
    paths = [[0, 2, 5], [0, 1, 2, 3], [0]]  # jump, identity, root only
    path = torch.full((bs, nodes), -1, dtype=torch.int32)
    for b, p in enumerate(paths):
        path[b, : len(p)] = torch.tensor(p, dtype=torch.int32)
    expected = [buf.view(torch.uint8).clone() for buf in buffers]
    for want in expected:
        for b, p in enumerate(paths):
            window = locs[b * nodes : (b + 1) * nodes].long()
            want[window[: len(p)]] = want[window[torch.tensor(p)]]

    addresses = torch.tensor(
        [buf.data_ptr() for buf in buffers], dtype=torch.int64, device="cuda"
    )
    compact_window_rows(addresses, locs, path.cuda(), row_bytes=heads * dim * 2)

    for got, want in zip(buffers, expected):
        assert torch.equal(got.view(torch.uint8), want)
