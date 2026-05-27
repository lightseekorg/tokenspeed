# Copyright (c) 2026 LightSeek Foundation

"""Torch probability helpers for sampling verify paths.

These helpers intentionally live behind the ``tokenspeed-kernel`` package
boundary so runtime sampling code keeps all probability construction under the
kernel package interface.
"""

from __future__ import annotations

import torch


def softmax_temperature(
    logits: torch.Tensor,
    temperature: torch.Tensor | None = None,
    *,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Softmax with per-row temperature.

    ``enable_pdl`` is accepted for call-site compatibility with kernel helpers;
    this torch implementation has no PDL hook.
    """

    _ = enable_pdl
    if temperature is not None:
        logits = logits / temperature.to(torch.float32).view(-1, 1).clamp_min(1.0e-20)
    return torch.softmax(logits, dim=-1, dtype=torch.float32)


def top_k_top_p_renorm_prob(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    """Apply top-k then top-p filtering to probabilities and renormalize.

    This uses only torch ops and intentionally materializes sorted
    probabilities because speculative verify already consumes full
    ``target_probs``.
    """

    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got {probs.ndim}D")
    rows, vocab_size = probs.shape
    if rows == 0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    ranks = torch.arange(vocab_size, device=probs.device).view(1, vocab_size)

    top_ks = top_ks.to(torch.int64).clamp(min=1, max=vocab_size)
    top_k_keep = ranks < top_ks.view(-1, 1)
    top_k_probs = torch.where(top_k_keep, sorted_probs, torch.zeros_like(sorted_probs))
    top_k_denom = top_k_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
    top_k_probs = top_k_probs / top_k_denom

    cdf = torch.cumsum(top_k_probs, dim=-1)
    keep_counts = (cdf < top_ps.to(torch.float32).view(-1, 1)).sum(dim=-1) + 1
    keep_counts = keep_counts.clamp(max=vocab_size)
    top_p_keep = ranks < keep_counts.view(-1, 1)

    kept = torch.where(top_p_keep & top_k_keep, top_k_probs, torch.zeros_like(probs))
    kept = kept / kept.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)

    out = torch.zeros_like(probs)
    out.scatter_(1, sorted_indices, kept)
    return out


def build_top_k_top_p_probs_from_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    *,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Build verify target probabilities from logits."""

    probs = softmax_temperature(logits, temperatures, enable_pdl=enable_pdl)
    return top_k_top_p_renorm_prob(probs, top_ks, top_ps)


__all__ = [
    "build_top_k_top_p_probs_from_logits",
    "softmax_temperature",
    "top_k_top_p_renorm_prob",
]
