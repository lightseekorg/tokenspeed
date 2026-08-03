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

"""tokenspeed-mha varlen prefill/extend route tests.

With a 128-multiple extent the registered rel_mha prefill/extend ops route
to the tokenspeed-mha kernel compiled with ``is_varlen_q=True``, fed by the
ported ShearingBias prepass (``TSMHA_REL_EXTEND=1`` default). Each case is
pinned against the torch reference AND against the score_mod fallback route
on identical inputs (flag patched off), so the two implementations guard
each other. Unaligned ``seqlen_k - seqlen_q`` extends are covered
explicitly: the sheared -inf pattern is the causal mask on diagonal blocks,
and its tile-end anchor is the part a reimplementation gets wrong first.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel import rel_mha_extend_with_kvcache, rel_mha_prefill
from tokenspeed_kernel.ops.attention import flash_attn as fa_mod

torch.manual_seed(11)

DTYPE = torch.bfloat16
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 128
PAGE = 128
TOL = 2e-2
ROUTE_TOL = 2e-2


def _require_fa4(require) -> None:
    require("attention", "rel_mha_prefill", "fa4", DTYPE, "q")


def _skip_unless_routable() -> None:
    if not getattr(fa_mod, "_TSMHA_REL_EXTEND", False):
        pytest.skip("TSMHA_REL_EXTEND disabled in this environment")


def _ref_rel_attn(q, k, v, rel_logits, rel_extent, window_left, scale):
    """Per-sequence torch reference. q [Sq,H,D], k/v [Sk,KV,D], rel [Sq,H,E]."""
    Sq, H, _ = q.shape
    Sk, KV, _ = k.shape
    k = k.repeat_interleave(H // KV, dim=1)
    v = v.repeat_interleave(H // KV, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    q_pos = torch.arange(Sq, device=q.device) + (Sk - Sq)
    kv_pos = torch.arange(Sk, device=q.device)
    dist = q_pos[:, None] - kv_pos[None, :]
    in_range = (dist >= 0) & (dist < rel_extent)
    idx = dist.clamp(0, rel_extent - 1)
    bias = rel_logits.float().gather(-1, idx.unsqueeze(1).expand(Sq, H, Sk))
    logits = logits + bias.permute(1, 0, 2).masked_fill(~in_range[None], 0.0)
    masked = dist < 0
    if window_left >= 0:
        masked = masked | (dist > window_left)
    logits = logits.masked_fill(masked[None], -torch.inf)
    return torch.einsum("hqk,khd->qhd", torch.softmax(logits, dim=-1), v.float())


def _cu(lens, device):
    cpu = [0]
    for length in lens:
        cpu.append(cpu[-1] + length)
    return torch.tensor(cpu, device=device, dtype=torch.int32), cpu


def _build_paged(kv_lens, device):
    pages_per = [(L + PAGE - 1) // PAGE for L in kv_lens]
    k_cache = torch.zeros(
        sum(pages_per), PAGE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE
    )
    v_cache = torch.zeros_like(k_cache)
    page_table = torch.zeros(
        len(kv_lens), max(pages_per), device=device, dtype=torch.int32
    )
    ks, vs, nxt = [], [], 0
    for i, L in enumerate(kv_lens):
        ki = torch.randn(L, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
        vi = torch.randn(L, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
        ks.append(ki)
        vs.append(vi)
        for p in range(pages_per[i]):
            page_table[i, p] = nxt
            rows = min(PAGE, L - p * PAGE)
            k_cache[nxt, :rows] = ki[p * PAGE : p * PAGE + rows]
            v_cache[nxt, :rows] = vi[p * PAGE : p * PAGE + rows]
            nxt += 1
    return k_cache, v_cache, page_table, ks, vs


def _with_route(enabled: bool, fn):
    saved = fa_mod._TSMHA_REL_EXTEND
    fa_mod._TSMHA_REL_EXTEND = enabled
    try:
        return fn()
    finally:
        fa_mod._TSMHA_REL_EXTEND = saved


@pytest.mark.parametrize(
    "rel_extent,window_left",
    [(128, -1), (256, -1), (128, 127)],
    ids=["full-e128", "full-e256", "swa-127"],
)
def test_tsmha_varlen_prefill(
    device: str, require, rel_extent: int, window_left: int
) -> None:
    """Packed varlen prefill: tsmha route vs torch reference and score_mod."""
    _require_fa4(require)
    _skip_unless_routable()
    q_lens = [128, 200, 65]
    scale = 1.0 / HEAD_DIM
    total = sum(q_lens)
    q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel_logits = (
        torch.randn(total, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE) * 0.5
    )
    cu, cu_cpu = _cu(q_lens, device)

    def call():
        return rel_mha_prefill(
            q=q,
            k=k,
            v=v,
            rel_logits=rel_logits,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
            max_seqlen=max(q_lens),
            window_left=window_left,
            softmax_scale=scale,
        )

    out = _with_route(True, call)
    out_fallback = _with_route(False, call)
    route_err = (out.float() - out_fallback.float()).abs().max().item()
    assert route_err < ROUTE_TOL, f"tsmha vs score_mod: {route_err:.4e}"

    for i, length in enumerate(q_lens):
        s = cu_cpu[i]
        ref = _ref_rel_attn(
            q[s : s + length],
            k[s : s + length],
            v[s : s + length],
            rel_logits[s : s + length],
            rel_extent,
            window_left,
            scale,
        )
        err = (out[s : s + length].float() - ref).abs().max().item()
        assert err < TOL, f"seq {i}: max_err={err:.4e}"


@pytest.mark.parametrize(
    "rel_extent,window_left,kv_lens",
    [
        (128, -1, [192, 260, 139]),
        (128, 127, [192, 260, 139]),
        (256, -1, [613, 165, 900]),
    ],
    ids=["full-unaligned", "swa-unaligned", "full-e256-unaligned"],
)
def test_tsmha_varlen_extend_paged(
    device: str, require, rel_extent: int, window_left: int, kv_lens: list[int]
) -> None:
    """Paged extend with unaligned prefixes: tsmha vs reference and score_mod.

    ``(kv - q) % 128 != 0`` for every request here — serving's chunked
    prefill never produces this, so only these tests exercise the shear's
    tile-end diagonal anchor.
    """
    _require_fa4(require)
    _skip_unless_routable()
    q_lens = [64, 100, 1]
    scale = 1.0 / HEAD_DIM
    total = sum(q_lens)
    q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel_logits = (
        torch.randn(total, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE) * 0.5
    )
    cu, cu_cpu = _cu(q_lens, device)
    cache_seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    k_cache, v_cache, page_table, ks, vs = _build_paged(kv_lens, device)

    def call():
        return rel_mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=cu,
            cu_seqlens_kv=None,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(kv_lens),
            rel_logits=rel_logits,
            window_left=window_left,
            softmax_scale=scale,
        )

    out = _with_route(True, call)
    out_fallback = _with_route(False, call)
    route_err = (out.float() - out_fallback.float()).abs().max().item()
    assert route_err < ROUTE_TOL, f"tsmha vs score_mod: {route_err:.4e}"

    for i, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        s = cu_cpu[i]
        ref = _ref_rel_attn(
            q[s : s + ql],
            ks[i],
            vs[i],
            rel_logits[s : s + ql],
            rel_extent,
            window_left,
            scale,
        )
        err = (out[s : s + ql].float() - ref).abs().max().item()
        assert err < TOL, f"seq {i}: max_err={err:.4e}"


def test_tsmha_varlen_gate_falls_back(device: str, require) -> None:
    """Ineligible extents (not 128-multiples) must keep the score_mod route."""
    _require_fa4(require)
    from tokenspeed_kernel.ops.attention.cute_dsl.rel_mha import rel_extend

    q_lens = [70, 130]
    rel_extent = 64
    scale = 1.0 / HEAD_DIM
    total = sum(q_lens)
    q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel_logits = (
        torch.randn(total, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE) * 0.5
    )
    cu, cu_cpu = _cu(q_lens, device)
    keys_before = set(rel_extend._FWD_VARLEN)
    out = rel_mha_prefill(
        q=q,
        k=k,
        v=v,
        rel_logits=rel_logits,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        max_seqlen=max(q_lens),
        softmax_scale=scale,
    )
    assert set(rel_extend._FWD_VARLEN) == keys_before
    for i, length in enumerate(q_lens):
        s = cu_cpu[i]
        ref = _ref_rel_attn(
            q[s : s + length],
            k[s : s + length],
            v[s : s + length],
            rel_logits[s : s + length],
            rel_extent,
            -1,
            scale,
        )
        err = (out[s : s + length].float() - ref).abs().max().item()
        assert err < TOL, f"seq {i}: max_err={err:.4e}"
