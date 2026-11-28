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

"""Shared Top-K / Top-P renormalization helpers for sampled DSpark proposals."""

from __future__ import annotations

from collections.abc import Callable

import torch
from tokenspeed_kernel.ops.sampling.cuda import (
    fused_topk_topp_available as _FUSED_TOPK_TOPP_AVAILABLE,
)
from tokenspeed_kernel.ops.sampling.cuda import (
    fused_topk_topp_renorm,
    fused_topk_topp_workspace_size,
)

# Minimum vocabulary width for the multi-block radix fused_topk_topp extension.
_FUSED_TOPK_TOPP_MIN_VOCAB = 1024
# Mirrors K_TOPK_MAX in fused_topk_topp.h: top_k > 128 signals disabled top-k.
_FUSED_TOPK_TOPP_MAX_K = 128


def dspark_topk_topp_workspace_size(
    max_rows: int,
    vocab_size: int,
    *,
    device: torch.device | str,
) -> int:
    """Return workspace bytes for in-place DSpark top-k/top-p renormalization."""
    if (
        _FUSED_TOPK_TOPP_AVAILABLE
        and vocab_size >= _FUSED_TOPK_TOPP_MIN_VOCAB
        and torch.device(device).type == "cuda"
    ):
        return fused_topk_topp_workspace_size(max_rows, vocab_size)
    return 0


def torch_small_vocab_topk_topp_renorm(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    """CUDA-graph-safe PyTorch top-k/top-p renormalization for small vocabularies.

    Matches ``fused_topk_topp_renorm(..., in_place_skip=True)``: greedy rows
    (``top_k == 1``) and uncut rows (``top_k > 128`` and ``top_p >= 1.0``) keep
    their existing contents unchanged; ties break toward smaller token indices.
    """
    if probs.shape[0] == 0:
        return probs
    vocab_size = probs.shape[1]
    top_k_col = top_ks.to(torch.int64).unsqueeze(1)
    top_p_col = top_ps.to(torch.float32).unsqueeze(1)
    skip = (top_k_col == 1) | (
        (top_k_col > _FUSED_TOPK_TOPP_MAX_K) & (top_p_col >= 1.0)
    )

    k_lim = top_k_col.clamp(min=1, max=vocab_size)
    vals, idx = probs.to(torch.float32).sort(dim=-1, descending=True, stable=True)
    ranks = torch.arange(vocab_size, device=probs.device, dtype=torch.int64).unsqueeze(
        0
    )
    keep_k = ranks < k_lim
    topk_vals = torch.where(keep_k, vals, 0.0)
    cumsum = topk_vals.cumsum(dim=-1)
    total = topk_vals.sum(dim=-1, keepdim=True)
    threshold = top_p_col * total
    keep = keep_k & (
        (top_p_col >= 1.0) | (ranks == 0) | ((cumsum - topk_vals) < threshold)
    )
    kept_vals = torch.where(keep, topk_vals, 0.0)
    denom = kept_vals.sum(dim=-1, keepdim=True).clamp_min(1e-38)
    renormed = kept_vals / denom
    filtered = torch.zeros_like(probs, dtype=torch.float32).scatter_(1, idx, renormed)
    probs.copy_(torch.where(skip, probs, filtered.to(probs.dtype)))
    return probs


def renorm_dspark_probabilities(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    *,
    workspace: torch.Tensor | None,
    small_vocab_renorm: (
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None
    ) = None,
) -> torch.Tensor:
    """In-place DSpark proposal probability top-k/top-p renormalization."""
    if (
        probs.is_cuda
        and probs.shape[1] >= _FUSED_TOPK_TOPP_MIN_VOCAB
        and _FUSED_TOPK_TOPP_AVAILABLE
    ):
        return fused_topk_topp_renorm(
            probs,
            top_ks,
            top_ps,
            workspace=workspace,
            out=probs,
            in_place_skip=True,
        )
    if small_vocab_renorm is not None:
        return small_vocab_renorm(probs, top_ks, top_ps)
    return torch_small_vocab_topk_topp_renorm(probs, top_ks, top_ps)
