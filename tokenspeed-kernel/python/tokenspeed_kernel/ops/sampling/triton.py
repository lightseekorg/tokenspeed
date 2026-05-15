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

"""Triton sampling helper kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["gather_and_expand_scalars", "min_p_renorm_prob"]


@triton.jit
def _gather_and_expand_scalars_kernel(
    index_ptr,
    temperature_ptr,
    top_k_ptr,
    top_p_ptr,
    min_p_ptr,
    seed_ptr,
    offsets_ptr,
    out_temperature_ptr,
    out_top_k_ptr,
    out_top_p_ptr,
    out_min_p_ptr,
    out_seed_ptr,
    out_offsets_ptr,
    n: tl.constexpr,
    N_BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    # PDL: wait for producer (e.g., penalty kernel writing into pools) to drain.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    bi = tl.program_id(0)
    idx = tl.load(index_ptr + bi)

    t = tl.load(temperature_ptr + idx)
    k = tl.load(top_k_ptr + idx)
    p = tl.load(top_p_ptr + idx)
    if min_p_ptr is not None:
        mp = tl.load(min_p_ptr + idx)
    if seed_ptr is not None:
        s = tl.load(seed_ptr + idx)
    if offsets_ptr is not None:
        # Cast int32 valid_cache_lengths to int64 for flashinfer's offset arg.
        o = tl.load(offsets_ptr + idx).to(tl.int64)

    n_off = tl.arange(0, N_BLOCK)
    mask = n_off < n
    base = bi * n

    tl.store(out_temperature_ptr + base + n_off, t, mask=mask)
    tl.store(out_top_k_ptr + base + n_off, k, mask=mask)
    tl.store(out_top_p_ptr + base + n_off, p, mask=mask)
    if out_min_p_ptr is not None:
        tl.store(out_min_p_ptr + base + n_off, mp, mask=mask)
    if out_seed_ptr is not None:
        tl.store(out_seed_ptr + base + n_off, s, mask=mask)
    if out_offsets_ptr is not None:
        tl.store(out_offsets_ptr + base + n_off, o, mask=mask)

    # PDL: signal that dependents (e.g., flashinfer softmax) can begin preamble.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def gather_and_expand_scalars(
    index: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor | None = None,
    seed: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
    n: int = 1,
    enable_pdl: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Fused gather-and-broadcast for per-request sampling scalars.

    Replaces the pattern ``index_select(pool, index)`` followed by
    ``repeat_interleave(..., n)`` across up to six streams with one Triton
    launch. ``offsets`` (int32) is cast to int64 inside the kernel.

    Optional streams (min_p, seed, offsets) pass through as ``None`` — Triton
    specializes the kernel on pointer-None-ness at JIT time and the gated
    load/store paths are dead-code-eliminated.

    Args:
        ...
        enable_pdl: opt into Programmatic Dependent Launch (Hopper+). Lets the
            downstream flashinfer softmax/renorm kernels start their preamble
            while our writes drain.

    Returns ``(temperatures, top_ks, top_ps, min_ps_or_None, seeds_or_None,
    offsets_or_None)``, each shape ``[bs * n]`` (or ``None`` when the
    corresponding pool was omitted).
    """
    bs = index.size(0)
    total = bs * n
    device = index.device

    out_temperature = torch.empty(total, dtype=temperature.dtype, device=device)
    out_top_k = torch.empty(total, dtype=top_k.dtype, device=device)
    out_top_p = torch.empty(total, dtype=top_p.dtype, device=device)
    out_min_p = (
        torch.empty(total, dtype=min_p.dtype, device=device)
        if min_p is not None
        else None
    )
    out_seed = (
        torch.empty(total, dtype=seed.dtype, device=device)
        if seed is not None
        else None
    )
    out_offsets = (
        torch.empty(total, dtype=torch.int64, device=device)
        if offsets is not None
        else None
    )

    if bs == 0:
        return (
            out_temperature,
            out_top_k,
            out_top_p,
            out_min_p,
            out_seed,
            out_offsets,
        )

    extra_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _gather_and_expand_scalars_kernel[(bs,)](
        index,
        temperature,
        top_k,
        top_p,
        min_p,
        seed,
        offsets,
        out_temperature,
        out_top_k,
        out_top_p,
        out_min_p,
        out_seed,
        out_offsets,
        n=n,
        N_BLOCK=triton.next_power_of_2(max(n, 1)),
        ENABLE_PDL=enable_pdl,
        num_warps=1,
        **extra_kwargs,
    )

    return out_temperature, out_top_k, out_top_p, out_min_p, out_seed, out_offsets


@triton.jit
def _min_p_renorm_prob_kernel(
    probs_ptr,
    min_p_ptr,
    out_ptr,
    vocab_size: tl.constexpr,
    probs_row_stride: tl.constexpr,
    out_row_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    probs_row = probs_ptr + row * probs_row_stride
    out_row = out_ptr + row * out_row_stride

    max_prob = tl.full((), 0.0, tl.float32)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + offs
        mask = cols < vocab_size
        vals = tl.load(probs_row + cols, mask=mask, other=0.0).to(tl.float32)
        max_prob = tl.maximum(max_prob, tl.max(tl.where(mask, vals, 0.0), axis=0))

    threshold = max_prob * tl.load(min_p_ptr + row).to(tl.float32)
    denom = tl.full((), 0.0, tl.float32)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + offs
        mask = cols < vocab_size
        vals = tl.load(probs_row + cols, mask=mask, other=0.0).to(tl.float32)
        keep = mask & (vals >= threshold)
        denom += tl.sum(tl.where(keep, vals, 0.0), axis=0)

    inv_denom = 1.0 / tl.maximum(denom, 1.0e-20)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + offs
        mask = cols < vocab_size
        vals = tl.load(probs_row + cols, mask=mask, other=0.0).to(tl.float32)
        keep = mask & (vals >= threshold)
        out = tl.where(keep, vals * inv_denom, 0.0)
        tl.store(out_row + cols, out, mask=mask)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def min_p_renorm_prob(
    probs: torch.Tensor,
    min_p: torch.Tensor,
    *,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Renormalize probabilities after applying a per-row min-p cutoff.

    For each row, this computes ``threshold = min_p[row] * max(probs[row])``,
    zeros probabilities below the threshold, and renormalizes the surviving
    probabilities so the row sums to one.
    """
    if probs.ndim != 2:
        raise ValueError(f"min_p_renorm_prob expects 2D probs, got {probs.ndim}D")
    if min_p.ndim != 1:
        raise ValueError(f"min_p_renorm_prob expects 1D min_p, got {min_p.ndim}D")
    if min_p.shape[0] != probs.shape[0]:
        raise ValueError(
            "min_p length must match probs rows, "
            f"got {min_p.shape[0]} and {probs.shape[0]}"
        )
    if probs.device.type != "cuda" or min_p.device.type != "cuda":
        raise ValueError("min_p_renorm_prob requires CUDA tensors")
    if probs.stride(-1) != 1:
        raise ValueError(
            f"min_p_renorm_prob requires stride-1 vocab dimension, got stride={probs.stride()}"
        )
    if not min_p.is_contiguous():
        min_p = min_p.contiguous()

    out = torch.empty_like(probs)
    rows, vocab_size = probs.shape
    if rows == 0:
        return out

    block_size = min(4096, triton.next_power_of_2(vocab_size))
    num_warps = 4 if block_size <= 1024 else 8
    extra_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _min_p_renorm_prob_kernel[(rows,)](
        probs,
        min_p,
        out,
        vocab_size=vocab_size,
        probs_row_stride=probs.stride(0),
        out_row_stride=out.stride(0),
        BLOCK_SIZE=block_size,
        ENABLE_PDL=enable_pdl,
        num_warps=num_warps,
        num_stages=3,
        **extra_kwargs,
    )
    return out
