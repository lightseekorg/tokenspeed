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

"""Shared Triton sampling helpers and constants."""

# Contains TokenSpeed compatibility helpers plus constants used by the
# Qrita-style top-k/top-p sampler adapted from vLLM:
#   https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_triton.py
#   https://arxiv.org/abs/2602.01518

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

_GUMBEL_BLOCK_SIZE = 1024
_GUMBEL_COMPACT_BLOCK_SIZE = 2048
_MIN_P_GUMBEL_BLOCK_SIZE = 1024
_TOP_K_TOP_P_BLOCK_SIZE = 2048
_TOP_K_TOP_P_PAD = 128
_TOP_K_TOP_P_CANDIDATE_BLOCK_SIZE = 1024
_GENERIC_GUMBEL_BLOCK_SIZE = 2048
_GENERIC_GUMBEL_TOP_K_PAD = 128
_GENERIC_GUMBEL_NUM_ATTEMPTS = 8
_TOP_P_PARALLEL_BLOCK_SIZE = 1024
_TOP_P_PARALLEL_NUM_ATTEMPTS = 3
_TOP_P_REPAIR_NUM_ATTEMPTS = 8
_QRITA_BLOCK_SIZE = 8192
_QRITA_BLOCK_SIZE_TRUNC = 4096
_QRITA_NUM_WARPS = 16
_TOP_K_DISABLED = 1 << 30

_QRITA_PERCENTILE_TO_STD_TABLE = [
    2.576,
    2.319,
    2.178,
    2.064,
    1.968,
    1.892,
    1.819,
    1.757,
    1.708,
    1.659,
    1.616,
    1.568,
    1.526,
    1.492,
    1.456,
    1.420,
    1.382,
    1.342,
    1.309,
    1.280,
    1.249,
    1.221,
    1.193,
    1.169,
    1.145,
    1.121,
    1.095,
    1.073,
    1.050,
    1.030,
    1.008,
    0.987,
    0.966,
    0.945,
    0.926,
    0.910,
    0.891,
    0.871,
    0.854,
    0.837,
    0.819,
    0.803,
    0.784,
    0.767,
    0.753,
    0.734,
    0.719,
    0.702,
    0.690,
    0.675,
    0.658,
    0.640,
    0.625,
    0.609,
    0.595,
    0.578,
    0.564,
    0.550,
    0.537,
    0.521,
    0.509,
    0.495,
    0.481,
    0.466,
    0.453,
    0.439,
    0.424,
    0.410,
    0.397,
    0.383,
    0.370,
    0.356,
    0.343,
    0.330,
    0.316,
    0.302,
    0.289,
    0.274,
    0.261,
    0.247,
    0.235,
    0.223,
    0.209,
    0.196,
    0.184,
    0.172,
    0.159,
    0.149,
    0.137,
    0.124,
    0.112,
    0.100,
    0.086,
    0.074,
    0.062,
    0.050,
    0.035,
    0.023,
    0.009,
    -0.003,
    -0.015,
    -0.027,
    -0.039,
    -0.052,
    -0.063,
    -0.074,
    -0.085,
    -0.097,
    -0.109,
    -0.122,
    -0.134,
    -0.147,
    -0.158,
    -0.171,
    -0.184,
    -0.196,
    -0.210,
    -0.223,
    -0.235,
    -0.248,
    -0.261,
    -0.275,
    -0.289,
    -0.302,
    -0.317,
    -0.328,
    -0.341,
    -0.353,
    -0.368,
    -0.382,
    -0.396,
    -0.410,
    -0.426,
    -0.439,
    -0.452,
    -0.465,
    -0.480,
    -0.493,
    -0.507,
    -0.521,
    -0.537,
    -0.551,
    -0.568,
    -0.582,
    -0.597,
    -0.614,
    -0.628,
    -0.643,
    -0.658,
    -0.673,
    -0.691,
    -0.706,
    -0.721,
    -0.738,
    -0.754,
    -0.769,
    -0.789,
    -0.808,
    -0.824,
    -0.838,
    -0.857,
    -0.877,
    -0.893,
    -0.912,
    -0.929,
    -0.947,
    -0.965,
    -0.983,
    -1.003,
    -1.027,
    -1.050,
    -1.070,
    -1.092,
    -1.117,
    -1.139,
    -1.162,
    -1.189,
    -1.216,
    -1.241,
    -1.272,
    -1.300,
    -1.330,
    -1.367,
    -1.404,
    -1.441,
    -1.485,
    -1.523,
    -1.564,
    -1.607,
    -1.658,
    -1.710,
    -1.778,
    -1.832,
    -1.901,
    -1.978,
    -2.068,
    -2.174,
    -2.325,
    -2.577,
    -3.813,
]


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
