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

"""Shared shapes and torch references for the rel_mha operator tests.

Both ``ops/test_attention_rel_mha.py`` and
``nvidia/ops/test_attention_rel_bias_fused.py`` pin the relative-attention
kernels against the same reference and paged-cache builder; keeping them here
lets the two files live in different vendor subtrees without importing each
other.
"""

from __future__ import annotations

import torch

DTYPE = torch.bfloat16
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 128
PAGE = 128


def ref_rel_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_logits: torch.Tensor | None,
    rel_extent: int,
    window_left: int,
    scale: float,
) -> torch.Tensor:
    """Per-sequence torch reference. q [Sq,H,D], k/v [Sk,KV,D], rel_logits [Sq,H,E]."""
    Sq, H, _ = q.shape
    Sk, KV, _ = k.shape
    rep = H // KV
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    q_pos = torch.arange(Sq, device=q.device) + (Sk - Sq)
    kv_pos = torch.arange(Sk, device=q.device)
    dist = q_pos[:, None] - kv_pos[None, :]  # [Sq, Sk]
    if rel_logits is not None:
        in_range = (dist >= 0) & (dist < rel_extent)
        idx = dist.clamp(0, rel_extent - 1)
        bias = rel_logits.float().gather(-1, idx.unsqueeze(1).expand(Sq, H, Sk))
        bias = torch.where(in_range.unsqueeze(1), bias, 0.0)
        logits = logits + bias.permute(1, 0, 2)
    mask = dist < 0  # causal
    if window_left >= 0:
        mask |= dist > window_left
    logits.masked_fill_(mask.unsqueeze(0), float("-inf"))
    return torch.einsum("hqk,khd->qhd", logits.softmax(-1), v.float()).to(q.dtype)


def cu_seqlens(lens: list[int], device: str) -> torch.Tensor:
    return torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0)), device=device, dtype=torch.int32
    )


def build_paged(
    kv_lens: list[int], device: str, page: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, list]:
    """Scatter per-sequence K/V into a paged cache; return caches and flat k/v."""
    batch = len(kv_lens)
    pages_per = [(length + page - 1) // page for length in kv_lens]
    total_pages = sum(pages_per) + 3  # spare pages
    k_cache = torch.zeros(
        total_pages, page, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE
    )
    v_cache = torch.zeros_like(k_cache)
    page_table = torch.zeros(batch, max(pages_per), device=device, dtype=torch.int32)
    ks, vs = [], []
    next_page = 1  # leave page 0 unused to catch indexing bugs
    for i, length in enumerate(kv_lens):
        k = torch.randn(length, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE)
        k *= 0.5
        v = torch.randn(length, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE)
        v *= 0.5
        ks.append(k)
        vs.append(v)
        for p in range(pages_per[i]):
            n = min(page, length - p * page)
            k_cache[next_page, :n] = k[p * page : p * page + n]
            v_cache[next_page, :n] = v[p * page : p * page + n]
            page_table[i, p] = next_page
            next_page += 1
    return k_cache, v_cache, page_table, ks, vs


def require_fa4(require) -> None:
    require("attention", "rel_mha_prefill", "fa4", DTYPE, "q")
