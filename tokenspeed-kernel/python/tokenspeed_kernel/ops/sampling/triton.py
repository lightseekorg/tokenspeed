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

"""Triton fused sampling-scalar gather + broadcast kernel."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["gather_and_expand_scalars"]


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
):
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
        num_warps=1,
    )

    return out_temperature, out_top_k, out_top_p, out_min_p, out_seed, out_offsets
