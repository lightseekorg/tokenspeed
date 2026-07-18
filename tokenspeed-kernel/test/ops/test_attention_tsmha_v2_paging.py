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

"""tsmha rel-decode v2 native-paging tests.

``rel_mha_decode_tsmha_v2`` consumes the page table at its native
granularity (64/128/256-token pages; no expansion prepass). Each case is
pinned against the torch reference with EXACT sliding-window semantics.
The serving-geometry cases (SWA at 128-token pages, full attention at
256-token hetero slots) are additionally cross-checked against the v1
route on identical inputs. Hole-punched (``-1``) tables must match the
unpunched output bit-for-bit: punched pages sit outside the window, the
page index is a bounds-checked TMA coordinate (zero-fill), and the
in-kernel window mask is index math.
"""

from __future__ import annotations

import pytest
import torch

torch.manual_seed(13)

DTYPE = torch.bfloat16
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 128
TOL = 2e-2
SCALE = 1.0 / HEAD_DIM


def _skip_unless_supported() -> None:
    pytest.importorskip(
        "tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.fmha_bias_helper"
    )
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tokenspeed-mha decode kernel is SM100-only")


def _ref_decode(q, k, v, rel_logits, rel_extent, window_left):
    """Single-request torch reference. q [H,D], k/v [Sk,KV,D], rel [H,E]."""
    Sk = k.shape[0]
    H = q.shape[0]
    k = k.repeat_interleave(H // NUM_KV_HEADS, dim=1)
    v = v.repeat_interleave(H // NUM_KV_HEADS, dim=1)
    logits = torch.einsum("hd,khd->hk", q.float(), k.float()) * SCALE
    dist = (Sk - 1) - torch.arange(Sk, device=q.device)
    in_range = (dist >= 0) & (dist < rel_extent)
    idx = dist.clamp(0, rel_extent - 1)
    bias = rel_logits.float().gather(-1, idx.unsqueeze(0).expand(H, Sk))
    logits = logits + bias.masked_fill(~in_range[None], 0.0)
    masked = dist < 0
    if window_left >= 0:
        masked = masked | (dist > window_left)
    logits = logits.masked_fill(masked[None], -torch.inf)
    return torch.einsum("hk,khd->hd", torch.softmax(logits, dim=-1), v.float())


def _build_paged(kv_lens, page, device):
    pages_per = [(L + page - 1) // page for L in kv_lens]
    k_cache = torch.zeros(
        sum(pages_per), page, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE
    )
    v_cache = torch.zeros_like(k_cache)
    table = torch.zeros(len(kv_lens), max(pages_per), device=device, dtype=torch.int32)
    ks, vs, nxt = [], [], 0
    for i, L in enumerate(kv_lens):
        ki = torch.randn(L, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
        vi = torch.randn(L, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
        ks.append(ki)
        vs.append(vi)
        for p in range(pages_per[i]):
            table[i, p] = nxt
            rows = min(page, L - p * page)
            k_cache[nxt, :rows] = ki[p * page : p * page + rows]
            v_cache[nxt, :rows] = vi[p * page : p * page + rows]
            nxt += 1
    return k_cache, v_cache, table, ks, vs


def _run_v2(q, k_cache, v_cache, table, kv_lens, rel, window_left):
    from tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.rel_decode_v2 import (
        rel_mha_decode_tsmha_v2,
    )

    seqlens = torch.tensor(kv_lens, device=q.device, dtype=torch.int32)
    out = rel_mha_decode_tsmha_v2(
        q, k_cache, v_cache, table, seqlens, rel, window_left, SCALE
    )
    return out.view(q.shape[0], NUM_Q_HEADS, HEAD_DIM)


# Serving geometry: swa = 128-token pages / window 511, full = 256-token
# hetero slots / no window. Page 64 keeps the pre-native path covered.
CASES = [
    pytest.param(128, 512, 511, id="swa-p128"),
    pytest.param(64, 512, 511, id="swa-p64"),
    pytest.param(256, 1024, -1, id="full-p256"),
    pytest.param(64, 1024, -1, id="full-p64"),
]


@pytest.mark.parametrize("page,rel_extent,window_left", CASES)
def test_v2_native_paging_vs_reference(page, rel_extent, window_left) -> None:
    _skip_unless_supported()
    device = "cuda"
    # Unaligned lengths on purpose; window straddles page boundaries.
    kv_lens = [1000, 517, 2048, page, 129, 3333, 64, 2000]
    B = len(kv_lens)
    q = torch.randn(B, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel = torch.randn(B, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE) * 0.5
    k_cache, v_cache, table, ks, vs = _build_paged(kv_lens, page, device)

    out = _run_v2(q, k_cache, v_cache, table, kv_lens, rel, window_left)
    for b in range(B):
        ref = _ref_decode(q[b], ks[b], vs[b], rel[b], rel_extent, window_left)
        torch.testing.assert_close(
            out[b].float(), ref, atol=TOL, rtol=0, msg=f"request {b}"
        )


@pytest.mark.parametrize(
    "page,rel_extent,window_left",
    [
        pytest.param(128, 512, 511, id="swa-p128"),
        pytest.param(256, 1024, -1, id="full-p256"),
    ],
)
def test_v2_native_paging_vs_v1_route(page, rel_extent, window_left) -> None:
    """Serving-geometry cross-check: v1 and v2 guard each other. Chunk-aligned
    lengths only — v1's tile-granular window over-attends on unaligned SWA."""
    _skip_unless_supported()
    from tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.rel_decode import (
        rel_mha_decode_tsmha,
    )

    device = "cuda"
    kv_lens = [1024, 2048, 128 * 5, 4096]
    B = len(kv_lens)
    q = torch.randn(B, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel = torch.randn(B, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE) * 0.5
    k_cache, v_cache, table, _, _ = _build_paged(kv_lens, page, device)
    seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    cu = torch.arange(B + 1, device=device, dtype=torch.int32)

    out2 = _run_v2(q, k_cache, v_cache, table, kv_lens, rel, window_left)
    out1 = rel_mha_decode_tsmha(
        q, k_cache, v_cache, table, seqlens, rel, cu, window_left, SCALE
    ).view(B, NUM_Q_HEADS, HEAD_DIM)
    torch.testing.assert_close(out2.float(), out1.float(), atol=TOL, rtol=0)


def test_v2_strided_table_view_bit_equal() -> None:
    """Serving passes a column-sliced VIEW of a wider static table (row
    stride > width); it must match the contiguous-table output exactly."""
    _skip_unless_supported()
    device = "cuda"
    page, rel_extent, window_left = 128, 512, 511
    kv_lens = [1000, 2048, 517, 3333]
    B = len(kv_lens)
    q = torch.randn(B, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel = torch.randn(B, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE) * 0.5
    k_cache, v_cache, table, _, _ = _build_paged(kv_lens, page, device)

    wide = torch.full((B, table.shape[1] + 37), -7, device=device, dtype=torch.int32)
    wide[:, : table.shape[1]] = table
    strided = wide[:, : table.shape[1]]
    assert not strided.is_contiguous()

    out_contig = _run_v2(q, k_cache, v_cache, table, kv_lens, rel, window_left)
    out_strided = _run_v2(q, k_cache, v_cache, strided, kv_lens, rel, window_left)
    assert torch.equal(out_contig, out_strided)


def test_v2_stale_workspace_immunity() -> None:
    """A huge-magnitude call between two identical calls must not change the
    second one: the shared per-B workspace carries no cross-launch state
    (regression for the stale-colmax NaN bug — the m workspace was a gmem
    atomic-fmax running max whose -inf identity only held on first use)."""
    _skip_unless_supported()
    device = "cuda"
    page, rel_extent, window_left = 128, 512, 511
    kv_lens = [2048]
    q = torch.randn(1, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel = torch.randn(1, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE) * 0.5
    k_cache, v_cache, table, _, _ = _build_paged(kv_lens, page, device)

    o_ref = _run_v2(q, k_cache, v_cache, table, kv_lens, rel, window_left).clone()
    poison_q = q * 80.0
    poison_rel = rel * 80.0
    _run_v2(poison_q, k_cache, v_cache, table, kv_lens, poison_rel, window_left)
    o_after = _run_v2(q, k_cache, v_cache, table, kv_lens, rel, window_left)
    assert not torch.isnan(o_after).any()
    assert torch.equal(o_after, o_ref)


def test_v2_hole_punched_table_bit_equal() -> None:
    """Punching fully-out-of-window pages to -1 must not change the output."""
    _skip_unless_supported()
    device = "cuda"
    page, rel_extent, window_left = 128, 512, 511
    # (s - 512) % 256 == 255 maximizes punched pages straddling the first
    # active 256-row sequence tile (the worst case for the -1 load path).
    kv_lens = [512 + 4 * 256 + 255, 2048, 1279, 517]
    B = len(kv_lens)
    q = torch.randn(B, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel = torch.randn(B, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE) * 0.5
    k_cache, v_cache, table, _, _ = _build_paged(kv_lens, page, device)

    out_full = _run_v2(q, k_cache, v_cache, table, kv_lens, rel, window_left)
    punched = table.clone()
    for b, s in enumerate(kv_lens):
        cutoff = (s - window_left - 1 - (page - 1)) // page
        if cutoff > 0:
            punched[b, :cutoff] = -1
    out_punched = _run_v2(q, k_cache, v_cache, punched, kv_lens, rel, window_left)
    assert torch.equal(out_full, out_punched)
