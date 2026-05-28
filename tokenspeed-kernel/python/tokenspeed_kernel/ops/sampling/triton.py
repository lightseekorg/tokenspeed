# Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

"""TokenSpeed-native Triton sampling kernels.

The algorithms adapt vLLM MRV2 sampler principles: sample from logits with
Gumbel-Max where possible, keep request indirection explicit, and avoid
materializing full probabilities in the normal sampling hot path. The code
keeps TokenSpeed's pool-state model rather than mirroring vLLM runner state.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = [
    "gather_and_expand_scalars",
    "gumbel_sample_from_pools_compact",
    "gumbel_sample_from_pools",
    "min_p_renorm_prob",
    "sample_top_p_rejection_from_pools",
    "sample_rejection_from_pools",
    "sample_rejection_min_p_from_pools",
    "sample_top_k_top_p_from_pools_compact",
    "sample_top_k_top_p_from_pools",
]

_GUMBEL_BLOCK_SIZE = 1024
_GUMBEL_COMPACT_BLOCK_SIZE = 2048
_TOP_K_FILTER_BLOCK_SIZE = 1024
_TOP_K_FILTER_MAX_K = 128
_TOP_K_TOP_P_DIRECT_MAX_K = 64
_TOP_K_TOP_P_DIRECT_BLOCK_SIZE = 8192
_TOP_K_TOP_P_COMPACT_BLOCK_SIZE = 2048
_TOP_P_REJECTION_TRIES = 4


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
        # Cast int32 valid_cache_lengths to int64 for offset arg.
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

    # PDL: signal that dependents can begin their preamble.
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
            downstream kernels start their preamble while our writes drain.

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
def _top_k_filter_stage1_kernel(
    logits_ptr,
    local_values_ptr,
    logits_row_stride: tl.constexpr,
    local_row_stride: tl.constexpr,
    local_block_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_SIZE)
    cols = block_idx * BLOCK_SIZE + offsets
    mask = cols < vocab_size

    vals = tl.load(
        logits_ptr + row * logits_row_stride + cols,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    top_vals = tl.topk(vals, TOP_K_PAD, descending=True)

    top_offsets = tl.arange(0, TOP_K_PAD)
    tl.store(
        local_values_ptr
        + row * local_row_stride
        + block_idx * local_block_stride
        + top_offsets,
        top_vals,
    )


@triton.jit
def _top_k_filter_stage2_kernel(
    local_values_ptr,
    local_row_stride: tl.constexpr,
    local_block_stride: tl.constexpr,
    num_blocks: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, TOP_K_PAD)
    top_vals = tl.full((TOP_K_PAD,), float("-inf"), tl.float32)

    for block_idx in tl.range(0, num_blocks, num_stages=3):
        block_vals = tl.load(
            local_values_ptr
            + row * local_row_stride
            + block_idx * local_block_stride
            + offsets
        ).to(tl.float32)
        top_vals = tl.topk(tl.cat(top_vals, block_vals), TOP_K_PAD, descending=True)

    tl.store(
        local_values_ptr + row * local_row_stride + offsets,
        top_vals,
    )


@triton.jit
def _top_k_top_p_value_sample_pool_kernel(
    local_values_ptr,
    req_pool_indices_ptr,
    top_k_pool_ptr,
    top_p_pool_ptr,
    temperature_pool_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    selected_values_ptr,
    local_values_row_stride: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    candidate_offsets = tl.arange(0, TOP_K_PAD)
    pool_idx = tl.load(req_pool_indices_ptr + row)

    top_k = tl.load(top_k_pool_ptr + pool_idx).to(tl.int32)
    top_k = tl.minimum(tl.maximum(top_k, 1), TOP_K_PAD)
    top_p = tl.load(top_p_pool_ptr + pool_idx).to(tl.float32)
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )

    vals = tl.load(
        local_values_ptr + row * local_values_row_stride + candidate_offsets
    ).to(tl.float32)
    keep_top_k = candidate_offsets < top_k
    vals = tl.where(keep_top_k, vals, float("-inf"))
    scaled_vals = vals / temperature

    max_val = tl.max(scaled_vals, axis=0)
    exp_vals = tl.where(keep_top_k, tl.exp(scaled_vals - max_val), 0.0)
    denom = tl.maximum(tl.sum(exp_vals, axis=0), 1.0e-20)
    probs = exp_vals / denom
    cdf = tl.cumsum(probs, 0)

    threshold_rank = tl.sum(tl.where((cdf < top_p) & keep_top_k, 1, 0), axis=0)
    threshold_rank = tl.minimum(threshold_rank, top_k - 1)
    keep = candidate_offsets <= threshold_rank

    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64)
    gumbel_seed = tl.randint(seed, offset)
    uniform = tl.maximum(tl.rand(gumbel_seed, candidate_offsets), 1.0e-7)
    gumbel = -tl.log(-tl.log(uniform))
    scores = tl.where(keep, scaled_vals + gumbel, float("-inf"))

    max_score = tl.max(scores, axis=0)
    selected_rank = tl.min(
        tl.where(scores == max_score, candidate_offsets, TOP_K_PAD), axis=0
    )
    selected_value = tl.max(
        tl.where(candidate_offsets == selected_rank, vals, float("-inf")), axis=0
    )
    tl.store(selected_values_ptr + row, selected_value)


@triton.jit
def _top_k_selected_value_to_token_id_kernel(
    logits_ptr,
    selected_values_ptr,
    out_ptr,
    logits_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    token_offsets = tl.arange(0, BLOCK_SIZE)
    selected_value = tl.load(selected_values_ptr + row).to(tl.float32)
    selected_id = tl.full((), 2147483647, tl.int32)

    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + token_offsets
        mask = cols < vocab_size
        vals = tl.load(
            logits_ptr + row * logits_row_stride + cols,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        block_id = tl.min(
            tl.where((vals == selected_value) & mask, cols, 2147483647),
            axis=0,
        )
        selected_id = tl.minimum(selected_id, block_id)

    tl.store(out_ptr + row, selected_id)


@triton.jit
def _top_k_top_p_value_sample_resolve_pool_kernel(
    logits_ptr,
    local_values_ptr,
    req_pool_indices_ptr,
    top_k_pool_ptr,
    top_p_pool_ptr,
    temperature_pool_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    out_ptr,
    logits_row_stride: tl.constexpr,
    local_values_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    candidate_offsets = tl.arange(0, TOP_K_PAD)
    pool_idx = tl.load(req_pool_indices_ptr + row)

    top_k = tl.load(top_k_pool_ptr + pool_idx).to(tl.int32)
    top_k = tl.minimum(tl.maximum(top_k, 1), TOP_K_PAD)
    top_p = tl.load(top_p_pool_ptr + pool_idx).to(tl.float32)
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )

    vals = tl.load(
        local_values_ptr + row * local_values_row_stride + candidate_offsets
    ).to(tl.float32)
    keep_top_k = candidate_offsets < top_k
    vals = tl.where(keep_top_k, vals, float("-inf"))
    scaled_vals = vals / temperature

    max_val = tl.max(scaled_vals, axis=0)
    exp_vals = tl.where(keep_top_k, tl.exp(scaled_vals - max_val), 0.0)
    denom = tl.maximum(tl.sum(exp_vals, axis=0), 1.0e-20)
    probs = exp_vals / denom
    cdf = tl.cumsum(probs, 0)

    threshold_rank = tl.sum(tl.where((cdf < top_p) & keep_top_k, 1, 0), axis=0)
    threshold_rank = tl.minimum(threshold_rank, top_k - 1)
    keep = candidate_offsets <= threshold_rank

    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64)
    gumbel_seed = tl.randint(seed, offset)
    uniform = tl.maximum(tl.rand(gumbel_seed, candidate_offsets), 1.0e-7)
    gumbel = -tl.log(-tl.log(uniform))
    scores = tl.where(keep, scaled_vals + gumbel, float("-inf"))

    max_score = tl.max(scores, axis=0)
    selected_rank = tl.min(
        tl.where(scores == max_score, candidate_offsets, TOP_K_PAD), axis=0
    )
    selected_value = tl.max(
        tl.where(candidate_offsets == selected_rank, vals, float("-inf")), axis=0
    )

    token_offsets = tl.arange(0, BLOCK_SIZE)
    selected_id = tl.full((), 2147483647, tl.int32)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + token_offsets
        mask = cols < vocab_size
        row_vals = tl.load(
            logits_ptr + row * logits_row_stride + cols,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        block_id = tl.min(
            tl.where((row_vals == selected_value) & mask, cols, 2147483647),
            axis=0,
        )
        selected_id = tl.minimum(selected_id, block_id)

    tl.store(out_ptr + row, selected_id)


def _prepare_top_k_values(
    logits: torch.Tensor,
    local_values: torch.Tensor,
    *,
    fn_name: str,
    top_k_pad: int = _TOP_K_FILTER_MAX_K,
    block_size: int = _TOP_K_FILTER_BLOCK_SIZE,
    stage1_num_warps: int = 4,
) -> tuple[int, int]:
    if logits.ndim != 2:
        raise ValueError(f"{fn_name} expects 2D logits, got {logits.ndim}D")
    if logits.device.type != "cuda":
        raise ValueError(f"{fn_name} requires CUDA logits")
    if logits.stride(-1) != 1:
        raise ValueError(
            f"{fn_name} requires stride-1 vocab dimension, "
            f"got stride={logits.stride()}"
        )

    rows, vocab_size = logits.shape
    if vocab_size <= 0:
        raise ValueError(f"{fn_name} requires non-empty vocab dimension")
    if local_values.ndim != 3:
        raise ValueError("local_values must be a 3D scratch tensor")
    if local_values.device.type != "cuda":
        raise ValueError("local_values scratch tensor must be CUDA")
    if local_values.dtype != torch.float32:
        raise ValueError(f"local_values must be float32, got {local_values.dtype}")
    if local_values.stride(-1) != 1:
        raise ValueError(
            "local_values requires contiguous candidate dimension, "
            f"got stride={local_values.stride()}"
        )

    num_blocks = triton.cdiv(vocab_size, block_size)
    if local_values.shape[0] < rows or local_values.shape[1] < num_blocks:
        raise ValueError(
            "local_values scratch is too small: "
            f"shape={tuple(local_values.shape)}, required rows/blocks="
            f"({rows}, {num_blocks})"
        )
    if local_values.shape[2] < top_k_pad:
        raise ValueError(
            "local_values scratch candidate dimension is too small: "
            f"{local_values.shape[2]} < {top_k_pad}"
        )
    if rows == 0:
        return rows, num_blocks

    _top_k_filter_stage1_kernel[(rows, num_blocks)](
        logits,
        local_values,
        logits_row_stride=logits.stride(0),
        local_row_stride=local_values.stride(0),
        local_block_stride=local_values.stride(1),
        vocab_size=vocab_size,
        BLOCK_SIZE=block_size,
        TOP_K_PAD=top_k_pad,
        num_warps=stage1_num_warps,
    )
    _top_k_filter_stage2_kernel[(rows,)](
        local_values,
        local_row_stride=local_values.stride(0),
        local_block_stride=local_values.stride(1),
        num_blocks=num_blocks,
        TOP_K_PAD=top_k_pad,
        num_warps=4,
        num_stages=3,
    )
    return rows, num_blocks


@triton.jit
def _gumbel_sample_pool_stage1_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    local_ids_ptr,
    local_scores_ptr,
    logits_row_stride: tl.constexpr,
    local_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    token_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_offsets < vocab_size
    pool_idx = tl.load(req_pool_indices_ptr + row)

    logits = tl.load(
        logits_ptr + row * logits_row_stride + token_offsets,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )

    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64)
    gumbel_seed = tl.randint(seed, offset)
    uniform = tl.maximum(tl.rand(gumbel_seed, token_offsets), 1.0e-7)
    gumbel = -tl.log(-tl.log(uniform))
    scores = tl.where(mask, logits / temperature + gumbel, float("-inf"))

    max_score = tl.max(scores, axis=0)
    token_id = tl.min(
        tl.where(scores == max_score, token_offsets, vocab_size + BLOCK_SIZE),
        axis=0,
    )

    tl.store(local_ids_ptr + row * local_row_stride + block_idx, token_id)
    tl.store(local_scores_ptr + row * local_row_stride + block_idx, max_score)


@triton.jit
def _gumbel_sample_stage2_kernel(
    local_ids_ptr,
    local_scores_ptr,
    out_ptr,
    local_row_stride: tl.constexpr,
    num_blocks: tl.constexpr,
    NUM_BLOCKS_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    block_offsets = tl.arange(0, NUM_BLOCKS_PAD)
    mask = block_offsets < num_blocks

    scores = tl.load(
        local_scores_ptr + row * local_row_stride + block_offsets,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    ids = tl.load(
        local_ids_ptr + row * local_row_stride + block_offsets,
        mask=mask,
        other=2147483647,
    )

    max_score = tl.max(scores, axis=0)
    token_id = tl.min(tl.where(scores == max_score, ids, 2147483647), axis=0)
    tl.store(out_ptr + row, token_id)


@triton.jit
def _gumbel_sample_compact_pool_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    out_ptr,
    logits_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    pool_idx = tl.load(req_pool_indices_ptr + row)
    token_offsets = tl.arange(0, BLOCK_SIZE)
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64)
    gumbel_seed = tl.randint(seed, offset)

    best_score = tl.full((), float("-inf"), tl.float32)
    best_id = tl.full((), 2147483647, tl.int32)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + token_offsets
        mask = cols < vocab_size
        logits = tl.load(
            logits_ptr + row * logits_row_stride + cols,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        uniform = tl.maximum(tl.rand(gumbel_seed, cols), 1.0e-7)
        gumbel = -tl.log(-tl.log(uniform))
        scores = tl.where(mask, logits / temperature + gumbel, float("-inf"))

        block_score = tl.max(scores, axis=0)
        block_id = tl.min(tl.where(scores == block_score, cols, 2147483647), axis=0)
        better = (block_score > best_score) | (
            (block_score == best_score) & (block_id < best_id)
        )
        best_score = tl.where(better, block_score, best_score)
        best_id = tl.where(better, block_id, best_id)

    tl.store(out_ptr + row, best_id)


@triton.jit
def _top_p_rejection_sample_stage1_pool_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    local_ids_ptr,
    local_scores_ptr,
    local_argmax_ids_ptr,
    local_argmax_scores_ptr,
    logits_row_stride: tl.constexpr,
    local_ids_row_stride: tl.constexpr,
    local_ids_try_stride: tl.constexpr,
    local_ids_block_stride: tl.constexpr,
    local_argmax_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RETRIES: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    token_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_offsets < vocab_size
    pool_idx = tl.load(req_pool_indices_ptr + row)

    logits = tl.load(
        logits_ptr + row * logits_row_stride + token_offsets,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    scaled_logits = logits / temperature

    max_scaled = tl.max(tl.where(mask, scaled_logits, float("-inf")), axis=0)
    argmax_id = tl.min(
        tl.where(scaled_logits == max_scaled, token_offsets, vocab_size + BLOCK_SIZE),
        axis=0,
    )
    tl.store(
        local_argmax_ids_ptr + row * local_argmax_row_stride + block_idx,
        argmax_id,
    )
    tl.store(
        local_argmax_scores_ptr + row * local_argmax_row_stride + block_idx,
        max_scaled,
    )

    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64)
    for retry in tl.static_range(0, RETRIES):
        gumbel_seed = tl.randint(seed, offset * RETRIES + retry)
        uniform = tl.maximum(tl.rand(gumbel_seed, token_offsets), 1.0e-7)
        gumbel = -tl.log(-tl.log(uniform))
        scores = tl.where(mask, scaled_logits + gumbel, float("-inf"))

        max_score = tl.max(scores, axis=0)
        token_id = tl.min(
            tl.where(scores == max_score, token_offsets, vocab_size + BLOCK_SIZE),
            axis=0,
        )
        tl.store(
            local_ids_ptr
            + row * local_ids_row_stride
            + retry * local_ids_try_stride
            + block_idx * local_ids_block_stride,
            token_id,
        )
        tl.store(
            local_scores_ptr
            + row * local_ids_row_stride
            + retry * local_ids_try_stride
            + block_idx * local_ids_block_stride,
            max_score,
        )


@triton.jit
def _top_p_rejection_sample_stage2_pool_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    local_ids_ptr,
    local_scores_ptr,
    local_argmax_ids_ptr,
    local_argmax_scores_ptr,
    candidate_ids_ptr,
    candidate_logits_ptr,
    logits_row_stride: tl.constexpr,
    local_ids_row_stride: tl.constexpr,
    local_ids_try_stride: tl.constexpr,
    local_ids_block_stride: tl.constexpr,
    local_argmax_row_stride: tl.constexpr,
    candidate_row_stride: tl.constexpr,
    num_blocks: tl.constexpr,
    NUM_BLOCKS_PAD: tl.constexpr,
    RETRIES: tl.constexpr,
):
    row = tl.program_id(0)
    block_offsets = tl.arange(0, NUM_BLOCKS_PAD)
    mask = block_offsets < num_blocks
    pool_idx = tl.load(req_pool_indices_ptr + row)
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )

    for retry in tl.static_range(0, RETRIES):
        scores = tl.load(
            local_scores_ptr
            + row * local_ids_row_stride
            + retry * local_ids_try_stride
            + block_offsets * local_ids_block_stride,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        ids = tl.load(
            local_ids_ptr
            + row * local_ids_row_stride
            + retry * local_ids_try_stride
            + block_offsets * local_ids_block_stride,
            mask=mask,
            other=2147483647,
        )

        max_score = tl.max(scores, axis=0)
        token_id = tl.min(tl.where(scores == max_score, ids, 2147483647), axis=0)
        token_logit = tl.load(logits_ptr + row * logits_row_stride + token_id).to(
            tl.float32
        )
        tl.store(
            candidate_ids_ptr + row * candidate_row_stride + retry,
            token_id,
        )
        tl.store(
            candidate_logits_ptr + row * candidate_row_stride + retry,
            token_logit / temperature,
        )

    argmax_scores = tl.load(
        local_argmax_scores_ptr + row * local_argmax_row_stride + block_offsets,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    argmax_ids = tl.load(
        local_argmax_ids_ptr + row * local_argmax_row_stride + block_offsets,
        mask=mask,
        other=2147483647,
    )
    max_scaled = tl.max(argmax_scores, axis=0)
    argmax_id = tl.min(
        tl.where(argmax_scores == max_scaled, argmax_ids, 2147483647), axis=0
    )
    tl.store(candidate_ids_ptr + row * candidate_row_stride + RETRIES, argmax_id)
    tl.store(candidate_logits_ptr + row * candidate_row_stride + RETRIES, max_scaled)


@triton.jit
def _rejection_sample_stage3_pool_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    candidate_ids_ptr,
    candidate_logits_ptr,
    local_total_probs_ptr,
    local_before_probs_ptr,
    local_before_counts_ptr,
    logits_row_stride: tl.constexpr,
    candidate_row_stride: tl.constexpr,
    local_total_row_stride: tl.constexpr,
    local_before_row_stride: tl.constexpr,
    local_before_try_stride: tl.constexpr,
    local_before_block_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RETRIES: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    token_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_offsets < vocab_size
    pool_idx = tl.load(req_pool_indices_ptr + row)

    logits = tl.load(
        logits_ptr + row * logits_row_stride + token_offsets,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    scaled_logits = logits / temperature
    global_max = tl.load(
        candidate_logits_ptr + row * candidate_row_stride + RETRIES
    ).to(tl.float32)
    probs = tl.where(mask, tl.exp(scaled_logits - global_max), 0.0)
    tl.store(
        local_total_probs_ptr + row * local_total_row_stride + block_idx,
        tl.sum(probs, axis=0),
    )

    for retry in tl.static_range(0, RETRIES):
        candidate_id = tl.load(candidate_ids_ptr + row * candidate_row_stride + retry)
        candidate_logit = tl.load(
            candidate_logits_ptr + row * candidate_row_stride + retry
        ).to(tl.float32)
        before = mask & (
            (scaled_logits > candidate_logit)
            | ((scaled_logits == candidate_logit) & (token_offsets < candidate_id))
        )
        tl.store(
            local_before_probs_ptr
            + row * local_before_row_stride
            + retry * local_before_try_stride
            + block_idx * local_before_block_stride,
            tl.sum(tl.where(before, probs, 0.0), axis=0),
        )
        tl.store(
            local_before_counts_ptr
            + row * local_before_row_stride
            + retry * local_before_try_stride
            + block_idx * local_before_block_stride,
            tl.sum(tl.where(before, 1, 0), axis=0),
        )


@triton.jit
def _top_p_rejection_sample_stage3_pool_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    candidate_ids_ptr,
    candidate_logits_ptr,
    local_total_probs_ptr,
    local_before_probs_ptr,
    logits_row_stride: tl.constexpr,
    candidate_row_stride: tl.constexpr,
    local_total_row_stride: tl.constexpr,
    local_before_row_stride: tl.constexpr,
    local_before_try_stride: tl.constexpr,
    local_before_block_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RETRIES: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    token_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_offsets < vocab_size
    pool_idx = tl.load(req_pool_indices_ptr + row)

    logits = tl.load(
        logits_ptr + row * logits_row_stride + token_offsets,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    scaled_logits = logits / temperature
    global_max = tl.load(
        candidate_logits_ptr + row * candidate_row_stride + RETRIES
    ).to(tl.float32)
    probs = tl.where(mask, tl.exp(scaled_logits - global_max), 0.0)
    tl.store(
        local_total_probs_ptr + row * local_total_row_stride + block_idx,
        tl.sum(probs, axis=0),
    )

    for retry in tl.static_range(0, RETRIES):
        candidate_id = tl.load(candidate_ids_ptr + row * candidate_row_stride + retry)
        candidate_logit = tl.load(
            candidate_logits_ptr + row * candidate_row_stride + retry
        ).to(tl.float32)
        before = mask & (
            (scaled_logits > candidate_logit)
            | ((scaled_logits == candidate_logit) & (token_offsets < candidate_id))
        )
        tl.store(
            local_before_probs_ptr
            + row * local_before_row_stride
            + retry * local_before_try_stride
            + block_idx * local_before_block_stride,
            tl.sum(tl.where(before, probs, 0.0), axis=0),
        )


@triton.jit
def _top_p_rejection_sample_stage4_pool_kernel(
    req_pool_indices_ptr,
    top_p_pool_ptr,
    candidate_ids_ptr,
    local_total_probs_ptr,
    local_before_probs_ptr,
    out_ptr,
    candidate_row_stride: tl.constexpr,
    local_total_row_stride: tl.constexpr,
    local_before_row_stride: tl.constexpr,
    local_before_try_stride: tl.constexpr,
    local_before_block_stride: tl.constexpr,
    num_blocks: tl.constexpr,
    NUM_BLOCKS_PAD: tl.constexpr,
    RETRIES: tl.constexpr,
):
    row = tl.program_id(0)
    block_offsets = tl.arange(0, NUM_BLOCKS_PAD)
    mask = block_offsets < num_blocks
    pool_idx = tl.load(req_pool_indices_ptr + row)

    total_probs = tl.load(
        local_total_probs_ptr + row * local_total_row_stride + block_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    total_prob = tl.sum(total_probs, axis=0)
    top_p = tl.load(top_p_pool_ptr + pool_idx).to(tl.float32)

    selected = tl.load(candidate_ids_ptr + row * candidate_row_stride + RETRIES)
    found = False
    for retry in tl.static_range(0, RETRIES):
        before_probs = tl.load(
            local_before_probs_ptr
            + row * local_before_row_stride
            + retry * local_before_try_stride
            + block_offsets * local_before_block_stride,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        before_prob = tl.sum(before_probs, axis=0)

        ok = (top_p >= 1.0 - 1.0e-6) | (before_prob < top_p * total_prob)
        candidate_id = tl.load(candidate_ids_ptr + row * candidate_row_stride + retry)
        selected = tl.where(ok & ~found, candidate_id, selected)
        found = found | ok

    tl.store(out_ptr + row, selected)


@triton.jit
def _rejection_sample_stage4_pool_kernel(
    req_pool_indices_ptr,
    top_k_pool_ptr,
    top_p_pool_ptr,
    candidate_ids_ptr,
    local_total_probs_ptr,
    top_k_total_probs_ptr,
    local_before_probs_ptr,
    local_before_counts_ptr,
    out_ptr,
    candidate_row_stride: tl.constexpr,
    local_total_row_stride: tl.constexpr,
    local_before_row_stride: tl.constexpr,
    local_before_try_stride: tl.constexpr,
    local_before_block_stride: tl.constexpr,
    num_blocks: tl.constexpr,
    NUM_BLOCKS_PAD: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    RETRIES: tl.constexpr,
):
    row = tl.program_id(0)
    block_offsets = tl.arange(0, NUM_BLOCKS_PAD)
    mask = block_offsets < num_blocks
    pool_idx = tl.load(req_pool_indices_ptr + row)

    total_probs = tl.load(
        local_total_probs_ptr + row * local_total_row_stride + block_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    total_prob = tl.sum(total_probs, axis=0)
    top_k = tl.load(top_k_pool_ptr + pool_idx).to(tl.int32)
    top_p = tl.load(top_p_pool_ptr + pool_idx).to(tl.float32)
    top_k_total_prob = tl.load(top_k_total_probs_ptr + row).to(tl.float32)
    finite_top_k = (top_k > 0) & (top_k <= TOP_K_PAD)
    top_p_total_prob = tl.where(finite_top_k, top_k_total_prob, total_prob)

    selected = tl.load(candidate_ids_ptr + row * candidate_row_stride + RETRIES)
    found = False
    for retry in tl.static_range(0, RETRIES):
        before_probs = tl.load(
            local_before_probs_ptr
            + row * local_before_row_stride
            + retry * local_before_try_stride
            + block_offsets * local_before_block_stride,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        before_counts = tl.load(
            local_before_counts_ptr
            + row * local_before_row_stride
            + retry * local_before_try_stride
            + block_offsets * local_before_block_stride,
            mask=mask,
            other=0,
        ).to(tl.int32)
        before_prob = tl.sum(before_probs, axis=0)
        before_count = tl.sum(before_counts, axis=0)

        top_k_ok = (top_k <= 0) | (before_count < top_k)
        top_p_ok = (top_p >= 1.0 - 1.0e-6) | (before_prob < top_p * top_p_total_prob)
        ok = top_k_ok & top_p_ok
        candidate_id = tl.load(candidate_ids_ptr + row * candidate_row_stride + retry)
        selected = tl.where(ok & ~found, candidate_id, selected)
        found = found | ok

    tl.store(out_ptr + row, selected)


@triton.jit
def _rejection_sample_min_p_stage4_pool_kernel(
    req_pool_indices_ptr,
    top_k_pool_ptr,
    top_p_pool_ptr,
    min_p_pool_ptr,
    candidate_ids_ptr,
    candidate_logits_ptr,
    local_total_probs_ptr,
    top_k_total_probs_ptr,
    local_before_probs_ptr,
    local_before_counts_ptr,
    out_ptr,
    candidate_row_stride: tl.constexpr,
    local_total_row_stride: tl.constexpr,
    local_before_row_stride: tl.constexpr,
    local_before_try_stride: tl.constexpr,
    local_before_block_stride: tl.constexpr,
    num_blocks: tl.constexpr,
    NUM_BLOCKS_PAD: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    RETRIES: tl.constexpr,
):
    row = tl.program_id(0)
    block_offsets = tl.arange(0, NUM_BLOCKS_PAD)
    mask = block_offsets < num_blocks
    pool_idx = tl.load(req_pool_indices_ptr + row)

    total_probs = tl.load(
        local_total_probs_ptr + row * local_total_row_stride + block_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    total_prob = tl.sum(total_probs, axis=0)
    top_k = tl.load(top_k_pool_ptr + pool_idx).to(tl.int32)
    top_p = tl.load(top_p_pool_ptr + pool_idx).to(tl.float32)
    min_p = tl.minimum(
        tl.maximum(tl.load(min_p_pool_ptr + pool_idx).to(tl.float32), 0.0), 1.0
    )
    top_k_total_prob = tl.load(top_k_total_probs_ptr + row).to(tl.float32)
    finite_top_k = (top_k > 0) & (top_k <= TOP_K_PAD)
    top_p_total_prob = tl.where(finite_top_k, top_k_total_prob, total_prob)
    max_scaled = tl.load(
        candidate_logits_ptr + row * candidate_row_stride + RETRIES
    ).to(tl.float32)

    selected = tl.load(candidate_ids_ptr + row * candidate_row_stride + RETRIES)
    found = False
    for retry in tl.static_range(0, RETRIES):
        before_probs = tl.load(
            local_before_probs_ptr
            + row * local_before_row_stride
            + retry * local_before_try_stride
            + block_offsets * local_before_block_stride,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        before_counts = tl.load(
            local_before_counts_ptr
            + row * local_before_row_stride
            + retry * local_before_try_stride
            + block_offsets * local_before_block_stride,
            mask=mask,
            other=0,
        ).to(tl.int32)
        before_prob = tl.sum(before_probs, axis=0)
        before_count = tl.sum(before_counts, axis=0)

        candidate_logit = tl.load(
            candidate_logits_ptr + row * candidate_row_stride + retry
        ).to(tl.float32)
        top_k_ok = (top_k <= 0) | (before_count < top_k)
        top_p_ok = (top_p >= 1.0 - 1.0e-6) | (before_prob < top_p * top_p_total_prob)
        min_p_ok = (min_p <= 0.0) | (tl.exp(candidate_logit - max_scaled) >= min_p)
        ok = top_k_ok & top_p_ok & min_p_ok
        candidate_id = tl.load(candidate_ids_ptr + row * candidate_row_stride + retry)
        selected = tl.where(ok & ~found, candidate_id, selected)
        found = found | ok

    tl.store(out_ptr + row, selected)


@triton.jit
def _top_k_total_prob_pool_kernel(
    req_pool_indices_ptr,
    top_k_pool_ptr,
    temperature_pool_ptr,
    candidate_logits_ptr,
    local_values_ptr,
    top_k_total_probs_ptr,
    candidate_row_stride: tl.constexpr,
    local_values_row_stride: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    RETRIES: tl.constexpr,
):
    row = tl.program_id(0)
    pool_idx = tl.load(req_pool_indices_ptr + row)
    top_k = tl.load(top_k_pool_ptr + pool_idx).to(tl.int32)
    top_k = tl.minimum(tl.maximum(top_k, 1), TOP_K_PAD)
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    global_max = tl.load(
        candidate_logits_ptr + row * candidate_row_stride + RETRIES
    ).to(tl.float32)

    offsets = tl.arange(0, TOP_K_PAD)
    vals = tl.load(local_values_ptr + row * local_values_row_stride + offsets).to(
        tl.float32
    )
    probs = tl.where(offsets < top_k, tl.exp(vals / temperature - global_max), 0.0)
    tl.store(top_k_total_probs_ptr + row, tl.sum(probs, axis=0))


def _check_cuda_vector(name: str, tensor: torch.Tensor, rows: int) -> None:
    if tensor.ndim != 1:
        raise ValueError(f"{name} expects 1D tensor, got {tensor.ndim}D")
    if tensor.shape[0] < rows:
        raise ValueError(f"{name} has {tensor.shape[0]} rows, needs at least {rows}")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} requires CUDA tensor")


def _check_gumbel_pool_inputs(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    local_ids: torch.Tensor,
    local_scores: torch.Tensor,
    out: torch.Tensor,
    *,
    fn_name: str,
) -> tuple[int, int, int]:
    if logits.ndim != 2:
        raise ValueError(f"{fn_name} expects 2D logits, got {logits.ndim}D")
    if logits.device.type != "cuda":
        raise ValueError(f"{fn_name} requires CUDA logits")
    if logits.stride(-1) != 1:
        raise ValueError(
            f"{fn_name} requires stride-1 vocab dimension, "
            f"got stride={logits.stride()}"
        )

    rows, vocab_size = logits.shape
    if vocab_size <= 0:
        raise ValueError(f"{fn_name} requires non-empty vocab dimension")
    _check_cuda_vector("req_pool_indices", req_pool_indices, rows)
    if req_pool_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "req_pool_indices must be int32 or int64, " f"got {req_pool_indices.dtype}"
        )
    for name, tensor in (
        ("temperature_pool", temperature_pool),
        ("seed_pool", seed_pool),
        ("offsets_pool", offsets_pool),
    ):
        if tensor.ndim != 1:
            raise ValueError(f"{name} expects 1D tensor, got {tensor.ndim}D")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} requires CUDA tensor")
    _check_cuda_vector("out", out, rows)
    if out.dtype != torch.int32:
        raise ValueError(f"out must be int32, got {out.dtype}")

    num_blocks = triton.cdiv(vocab_size, _GUMBEL_BLOCK_SIZE)
    if local_ids.ndim != 2 or local_scores.ndim != 2:
        raise ValueError("local_ids and local_scores must be 2D scratch tensors")
    if local_ids.device.type != "cuda" or local_scores.device.type != "cuda":
        raise ValueError("gumbel pool scratch tensors must be CUDA")
    if local_ids.dtype != torch.int32:
        raise ValueError(f"local_ids must be int32, got {local_ids.dtype}")
    if local_scores.dtype != torch.float32:
        raise ValueError(f"local_scores must be float32, got {local_scores.dtype}")
    if local_ids.shape[0] < rows or local_ids.shape[1] < num_blocks:
        raise ValueError(
            "local_ids scratch is too small: "
            f"shape={tuple(local_ids.shape)}, required=({rows}, {num_blocks})"
        )
    if local_scores.shape[0] < rows or local_scores.shape[1] < num_blocks:
        raise ValueError(
            "local_scores scratch is too small: "
            f"shape={tuple(local_scores.shape)}, required=({rows}, {num_blocks})"
        )
    return rows, vocab_size, num_blocks


def gumbel_sample_from_pools(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    local_ids: torch.Tensor,
    local_scores: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Pool-aware no-filter Gumbel-Max sampler.

    This graph-safe variant keeps the hot path allocation-free by gathering
    ``temperature``, ``seed``, and ``offset`` from TokenSpeed request-pool state
    inside the Triton kernel.
    """
    rows, vocab_size, num_blocks = _check_gumbel_pool_inputs(
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        local_ids,
        local_scores,
        out,
        fn_name="gumbel_sample_from_pools",
    )
    if rows == 0:
        return out[:0]

    _gumbel_sample_pool_stage1_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        local_ids,
        local_scores,
        logits_row_stride=logits.stride(0),
        local_row_stride=local_ids.stride(0),
        vocab_size=vocab_size,
        BLOCK_SIZE=_GUMBEL_BLOCK_SIZE,
        num_warps=4,
    )
    _gumbel_sample_stage2_kernel[(rows,)](
        local_ids,
        local_scores,
        out,
        local_row_stride=local_ids.stride(0),
        num_blocks=num_blocks,
        NUM_BLOCKS_PAD=triton.next_power_of_2(max(num_blocks, 1)),
        num_warps=8,
    )
    return out[:rows]


def gumbel_sample_from_pools_compact(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    out: torch.Tensor,
    *,
    block_size: int = _GUMBEL_COMPACT_BLOCK_SIZE,
) -> torch.Tensor:
    """Pool-aware single-kernel no-filter sampler for compact vocab sizes."""
    if logits.ndim != 2:
        raise ValueError(f"logits expects 2D tensor, got {logits.ndim}D")
    if logits.device.type != "cuda":
        raise ValueError("gumbel_sample_from_pools_compact requires CUDA logits")
    rows, vocab_size = logits.shape
    _check_cuda_vector("req_pool_indices", req_pool_indices, rows)
    _check_cuda_vector("out", out, rows)
    if req_pool_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "req_pool_indices must be int32 or int64, " f"got {req_pool_indices.dtype}"
        )
    for name, tensor in (
        ("temperature_pool", temperature_pool),
        ("seed_pool", seed_pool),
        ("offsets_pool", offsets_pool),
    ):
        if tensor.ndim != 1:
            raise ValueError(f"{name} expects 1D tensor, got {tensor.ndim}D")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} requires CUDA tensor")
    if temperature_pool.dtype != torch.float32:
        raise ValueError(
            f"temperature_pool must be float32, got {temperature_pool.dtype}"
        )
    if seed_pool.dtype != torch.int64:
        raise ValueError(f"seed_pool must be int64, got {seed_pool.dtype}")
    if offsets_pool.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"offsets_pool must be int32 or int64, got {offsets_pool.dtype}"
        )
    if out.dtype != torch.int32:
        raise ValueError(f"out must be int32, got {out.dtype}")
    if rows == 0:
        return out[:0]

    _gumbel_sample_compact_pool_kernel[(rows,)](
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        out,
        logits_row_stride=logits.stride(0),
        vocab_size=vocab_size,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return out[:rows]


def sample_top_k_top_p_from_pools(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_k_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    local_values: torch.Tensor,
    selected_values: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Pool-aware finite top-k/top-p direct sampler for CUDA graph replay."""
    rows, _ = _prepare_top_k_values(
        logits,
        local_values,
        fn_name="sample_top_k_top_p_from_pools",
        top_k_pad=_TOP_K_TOP_P_DIRECT_MAX_K,
        block_size=_TOP_K_TOP_P_DIRECT_BLOCK_SIZE,
        stage1_num_warps=8,
    )
    _check_cuda_vector("req_pool_indices", req_pool_indices, rows)
    if req_pool_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "req_pool_indices must be int32 or int64, " f"got {req_pool_indices.dtype}"
        )
    for name, tensor in (
        ("temperature_pool", temperature_pool),
        ("top_k_pool", top_k_pool),
        ("top_p_pool", top_p_pool),
        ("seed_pool", seed_pool),
        ("offsets_pool", offsets_pool),
    ):
        if tensor.ndim != 1:
            raise ValueError(f"{name} expects 1D tensor, got {tensor.ndim}D")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} requires CUDA tensor")
    if top_k_pool.dtype != torch.int32:
        raise ValueError(f"top_k_pool must be int32, got {top_k_pool.dtype}")
    if top_p_pool.dtype != torch.float32:
        raise ValueError(f"top_p_pool must be float32, got {top_p_pool.dtype}")
    if temperature_pool.dtype != torch.float32:
        raise ValueError(
            f"temperature_pool must be float32, got {temperature_pool.dtype}"
        )
    _check_cuda_vector("selected_values", selected_values, rows)
    _check_cuda_vector("out", out, rows)
    if selected_values.dtype != torch.float32:
        raise ValueError(
            f"selected_values must be float32, got {selected_values.dtype}"
        )
    if out.dtype != torch.int32:
        raise ValueError(f"out must be int32, got {out.dtype}")
    if rows == 0:
        return out[:0]

    _top_k_top_p_value_sample_pool_kernel[(rows,)](
        local_values,
        req_pool_indices,
        top_k_pool,
        top_p_pool,
        temperature_pool,
        seed_pool,
        offsets_pool,
        selected_values,
        local_values_row_stride=local_values.stride(0),
        TOP_K_PAD=_TOP_K_TOP_P_DIRECT_MAX_K,
        num_warps=4,
    )
    _top_k_selected_value_to_token_id_kernel[(rows,)](
        logits,
        selected_values,
        out,
        logits_row_stride=logits.stride(0),
        vocab_size=logits.shape[1],
        BLOCK_SIZE=_GUMBEL_BLOCK_SIZE,
        num_warps=4,
    )
    return out[:rows]


def sample_top_k_top_p_from_pools_compact(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_k_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    local_values: torch.Tensor,
    out: torch.Tensor,
    *,
    block_size: int = _TOP_K_TOP_P_COMPACT_BLOCK_SIZE,
) -> torch.Tensor:
    """Pool-aware compact-vocab finite top-k/top-p sampler."""
    rows, _ = _prepare_top_k_values(
        logits,
        local_values,
        fn_name="sample_top_k_top_p_from_pools_compact",
        top_k_pad=_TOP_K_TOP_P_DIRECT_MAX_K,
        block_size=block_size,
        stage1_num_warps=4,
    )
    _check_cuda_vector("req_pool_indices", req_pool_indices, rows)
    if req_pool_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "req_pool_indices must be int32 or int64, " f"got {req_pool_indices.dtype}"
        )
    for name, tensor in (
        ("temperature_pool", temperature_pool),
        ("top_k_pool", top_k_pool),
        ("top_p_pool", top_p_pool),
        ("seed_pool", seed_pool),
        ("offsets_pool", offsets_pool),
    ):
        if tensor.ndim != 1:
            raise ValueError(f"{name} expects 1D tensor, got {tensor.ndim}D")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} requires CUDA tensor")
    if top_k_pool.dtype != torch.int32:
        raise ValueError(f"top_k_pool must be int32, got {top_k_pool.dtype}")
    if top_p_pool.dtype != torch.float32:
        raise ValueError(f"top_p_pool must be float32, got {top_p_pool.dtype}")
    if temperature_pool.dtype != torch.float32:
        raise ValueError(
            f"temperature_pool must be float32, got {temperature_pool.dtype}"
        )
    _check_cuda_vector("out", out, rows)
    if out.dtype != torch.int32:
        raise ValueError(f"out must be int32, got {out.dtype}")
    if rows == 0:
        return out[:0]

    _top_k_top_p_value_sample_resolve_pool_kernel[(rows,)](
        logits,
        local_values,
        req_pool_indices,
        top_k_pool,
        top_p_pool,
        temperature_pool,
        seed_pool,
        offsets_pool,
        out,
        logits_row_stride=logits.stride(0),
        local_values_row_stride=local_values.stride(0),
        vocab_size=logits.shape[1],
        TOP_K_PAD=_TOP_K_TOP_P_DIRECT_MAX_K,
        BLOCK_SIZE=_GUMBEL_BLOCK_SIZE,
        num_warps=4,
    )
    return out[:rows]


def _check_rejection_scratch(
    rows: int,
    num_blocks: int,
    local_ids: torch.Tensor,
    local_scores: torch.Tensor,
    local_argmax_ids: torch.Tensor,
    local_argmax_scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
    local_total_probs: torch.Tensor,
    local_before_probs: torch.Tensor,
    local_before_counts: torch.Tensor | None,
) -> None:
    tensors_3d = (
        ("local_ids", local_ids),
        ("local_scores", local_scores),
        ("local_before_probs", local_before_probs),
    )
    if local_before_counts is not None:
        tensors_3d += (("local_before_counts", local_before_counts),)
    for name, tensor in tensors_3d:
        if tensor.ndim != 3:
            raise ValueError(f"{name} must be a 3D scratch tensor")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} scratch tensor must be CUDA")
        if tensor.shape[0] < rows:
            raise ValueError(f"{name} has {tensor.shape[0]} rows, needs {rows}")
        if tensor.shape[1] < _TOP_P_REJECTION_TRIES:
            raise ValueError(
                f"{name} has {tensor.shape[1]} retries, "
                f"needs {_TOP_P_REJECTION_TRIES}"
            )
        if tensor.shape[2] < num_blocks:
            raise ValueError(f"{name} has {tensor.shape[2]} blocks, needs {num_blocks}")

    for name, tensor in (
        ("local_argmax_ids", local_argmax_ids),
        ("local_argmax_scores", local_argmax_scores),
        ("local_total_probs", local_total_probs),
    ):
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be a 2D scratch tensor")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} scratch tensor must be CUDA")
        if tensor.shape[0] < rows or tensor.shape[1] < num_blocks:
            raise ValueError(
                f"{name} scratch is too small: shape={tuple(tensor.shape)}, "
                f"required=({rows}, {num_blocks})"
            )

    for name, tensor in (
        ("candidate_ids", candidate_ids),
        ("candidate_logits", candidate_logits),
    ):
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be a 2D scratch tensor")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} scratch tensor must be CUDA")
        if tensor.shape[0] < rows or tensor.shape[1] < _TOP_P_REJECTION_TRIES + 1:
            raise ValueError(
                f"{name} scratch is too small: shape={tuple(tensor.shape)}, "
                f"required=({rows}, {_TOP_P_REJECTION_TRIES + 1})"
            )

    if local_ids.dtype != torch.int32:
        raise ValueError(f"local_ids must be int32, got {local_ids.dtype}")
    if local_argmax_ids.dtype != torch.int32:
        raise ValueError(
            f"local_argmax_ids must be int32, got {local_argmax_ids.dtype}"
        )
    if candidate_ids.dtype != torch.int32:
        raise ValueError(f"candidate_ids must be int32, got {candidate_ids.dtype}")
    if local_before_counts is not None and local_before_counts.dtype != torch.int32:
        raise ValueError(
            "local_before_counts must be int32, " f"got {local_before_counts.dtype}"
        )
    for name, tensor in (
        ("local_scores", local_scores),
        ("local_argmax_scores", local_argmax_scores),
        ("candidate_logits", candidate_logits),
        ("local_total_probs", local_total_probs),
        ("local_before_probs", local_before_probs),
    ):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {tensor.dtype}")


def _check_top_p_pool_inputs(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    local_argmax_ids: torch.Tensor,
    local_argmax_scores: torch.Tensor,
    out: torch.Tensor,
    *,
    fn_name: str,
) -> tuple[int, int, int]:
    if logits.ndim != 2:
        raise ValueError(f"{fn_name} expects 2D logits, got {logits.ndim}D")
    if logits.device.type != "cuda":
        raise ValueError(f"{fn_name} requires CUDA logits")
    if logits.stride(-1) != 1:
        raise ValueError(
            f"{fn_name} requires stride-1 vocab dimension, "
            f"got stride={logits.stride()}"
        )
    rows, vocab_size = logits.shape
    if vocab_size <= 0:
        raise ValueError(f"{fn_name} requires non-empty vocab dimension")
    _check_cuda_vector("req_pool_indices", req_pool_indices, rows)
    if req_pool_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "req_pool_indices must be int32 or int64, " f"got {req_pool_indices.dtype}"
        )
    for name, tensor in (
        ("temperature_pool", temperature_pool),
        ("top_p_pool", top_p_pool),
        ("seed_pool", seed_pool),
        ("offsets_pool", offsets_pool),
    ):
        if tensor.ndim != 1:
            raise ValueError(f"{name} expects 1D tensor, got {tensor.ndim}D")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} requires CUDA tensor")
    _check_cuda_vector("out", out, rows)
    num_blocks = triton.cdiv(vocab_size, _GUMBEL_BLOCK_SIZE)
    for name, tensor, dtype in (
        ("local_argmax_ids", local_argmax_ids, torch.int32),
        ("local_argmax_scores", local_argmax_scores, torch.float32),
    ):
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be a 2D scratch tensor")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} scratch tensor must be CUDA")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
        if tensor.shape[0] < rows or tensor.shape[1] < num_blocks:
            raise ValueError(
                f"{name} scratch is too small: shape={tuple(tensor.shape)}, "
                f"required=({rows}, {num_blocks})"
            )
    if out.dtype != torch.int32:
        raise ValueError(f"out must be int32, got {out.dtype}")
    return rows, vocab_size, num_blocks


def _check_rejection_pool_inputs(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_k_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    local_argmax_ids: torch.Tensor,
    local_argmax_scores: torch.Tensor,
    out: torch.Tensor,
    *,
    fn_name: str,
) -> tuple[int, int, int]:
    rows, vocab_size, num_blocks = _check_top_p_pool_inputs(
        logits,
        req_pool_indices,
        temperature_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        local_argmax_ids,
        local_argmax_scores,
        out,
        fn_name=fn_name,
    )
    if top_k_pool.ndim != 1:
        raise ValueError(f"top_k_pool expects 1D tensor, got {top_k_pool.ndim}D")
    if top_k_pool.device.type != "cuda":
        raise ValueError("top_k_pool requires CUDA tensor")
    if top_k_pool.dtype != torch.int32:
        raise ValueError(f"top_k_pool must be int32, got {top_k_pool.dtype}")
    return rows, vocab_size, num_blocks


def sample_top_p_rejection_from_pools(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    local_ids: torch.Tensor,
    local_scores: torch.Tensor,
    local_argmax_ids: torch.Tensor,
    local_argmax_scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
    local_total_probs: torch.Tensor,
    local_before_probs: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Top-p-only bounded rejection sampler.

    This is the same pool-aware rejection algorithm as the generic sampler,
    but it deliberately skips finite top-k scratch prep, top-k mass, and
    rank-count bookkeeping. It is used only for ``top_k`` disabled rows.
    """
    rows, vocab_size, num_blocks = _check_top_p_pool_inputs(
        logits,
        req_pool_indices,
        temperature_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        local_argmax_ids,
        local_argmax_scores,
        out,
        fn_name="sample_top_p_rejection_from_pools",
    )
    _check_rejection_scratch(
        rows,
        num_blocks,
        local_ids,
        local_scores,
        local_argmax_ids,
        local_argmax_scores,
        candidate_ids,
        candidate_logits,
        local_total_probs,
        local_before_probs,
        None,
    )
    if rows == 0:
        return out[:0]

    _top_p_rejection_sample_stage1_pool_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        local_ids,
        local_scores,
        local_argmax_ids,
        local_argmax_scores,
        logits_row_stride=logits.stride(0),
        local_ids_row_stride=local_ids.stride(0),
        local_ids_try_stride=local_ids.stride(1),
        local_ids_block_stride=local_ids.stride(2),
        local_argmax_row_stride=local_argmax_ids.stride(0),
        vocab_size=vocab_size,
        BLOCK_SIZE=_GUMBEL_BLOCK_SIZE,
        RETRIES=_TOP_P_REJECTION_TRIES,
        num_warps=4,
    )
    _top_p_rejection_sample_stage2_pool_kernel[(rows,)](
        logits,
        req_pool_indices,
        temperature_pool,
        local_ids,
        local_scores,
        local_argmax_ids,
        local_argmax_scores,
        candidate_ids,
        candidate_logits,
        logits_row_stride=logits.stride(0),
        local_ids_row_stride=local_ids.stride(0),
        local_ids_try_stride=local_ids.stride(1),
        local_ids_block_stride=local_ids.stride(2),
        local_argmax_row_stride=local_argmax_ids.stride(0),
        candidate_row_stride=candidate_ids.stride(0),
        num_blocks=num_blocks,
        NUM_BLOCKS_PAD=triton.next_power_of_2(max(num_blocks, 1)),
        RETRIES=_TOP_P_REJECTION_TRIES,
        num_warps=8,
    )
    _top_p_rejection_sample_stage3_pool_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        temperature_pool,
        candidate_ids,
        candidate_logits,
        local_total_probs,
        local_before_probs,
        logits_row_stride=logits.stride(0),
        candidate_row_stride=candidate_ids.stride(0),
        local_total_row_stride=local_total_probs.stride(0),
        local_before_row_stride=local_before_probs.stride(0),
        local_before_try_stride=local_before_probs.stride(1),
        local_before_block_stride=local_before_probs.stride(2),
        vocab_size=vocab_size,
        BLOCK_SIZE=_GUMBEL_BLOCK_SIZE,
        RETRIES=_TOP_P_REJECTION_TRIES,
        num_warps=4,
    )
    _top_p_rejection_sample_stage4_pool_kernel[(rows,)](
        req_pool_indices,
        top_p_pool,
        candidate_ids,
        local_total_probs,
        local_before_probs,
        out,
        candidate_row_stride=candidate_ids.stride(0),
        local_total_row_stride=local_total_probs.stride(0),
        local_before_row_stride=local_before_probs.stride(0),
        local_before_try_stride=local_before_probs.stride(1),
        local_before_block_stride=local_before_probs.stride(2),
        num_blocks=num_blocks,
        NUM_BLOCKS_PAD=triton.next_power_of_2(max(num_blocks, 1)),
        RETRIES=_TOP_P_REJECTION_TRIES,
        num_warps=8,
    )
    return out[:rows]


def sample_rejection_from_pools(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_k_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    local_ids: torch.Tensor,
    local_scores: torch.Tensor,
    local_argmax_ids: torch.Tensor,
    local_argmax_scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
    local_top_k_values: torch.Tensor,
    top_k_total_probs: torch.Tensor,
    local_total_probs: torch.Tensor,
    local_before_probs: torch.Tensor,
    local_before_counts: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Graph-safe bounded rejection sampler that gathers pool scalars in-kernel."""
    rows, vocab_size, num_blocks = _check_rejection_pool_inputs(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        local_argmax_ids,
        local_argmax_scores,
        out,
        fn_name="sample_rejection_from_pools",
    )
    _check_rejection_scratch(
        rows,
        num_blocks,
        local_ids,
        local_scores,
        local_argmax_ids,
        local_argmax_scores,
        candidate_ids,
        candidate_logits,
        local_total_probs,
        local_before_probs,
        local_before_counts,
    )
    _check_cuda_vector("top_k_total_probs", top_k_total_probs, rows)
    if top_k_total_probs.dtype != torch.float32:
        raise ValueError(
            f"top_k_total_probs must be float32, got {top_k_total_probs.dtype}"
        )
    if rows == 0:
        return out[:0]

    _top_p_rejection_sample_stage1_pool_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        local_ids,
        local_scores,
        local_argmax_ids,
        local_argmax_scores,
        logits_row_stride=logits.stride(0),
        local_ids_row_stride=local_ids.stride(0),
        local_ids_try_stride=local_ids.stride(1),
        local_ids_block_stride=local_ids.stride(2),
        local_argmax_row_stride=local_argmax_ids.stride(0),
        vocab_size=vocab_size,
        BLOCK_SIZE=_GUMBEL_BLOCK_SIZE,
        RETRIES=_TOP_P_REJECTION_TRIES,
        num_warps=4,
    )
    _top_p_rejection_sample_stage2_pool_kernel[(rows,)](
        logits,
        req_pool_indices,
        temperature_pool,
        local_ids,
        local_scores,
        local_argmax_ids,
        local_argmax_scores,
        candidate_ids,
        candidate_logits,
        logits_row_stride=logits.stride(0),
        local_ids_row_stride=local_ids.stride(0),
        local_ids_try_stride=local_ids.stride(1),
        local_ids_block_stride=local_ids.stride(2),
        local_argmax_row_stride=local_argmax_ids.stride(0),
        candidate_row_stride=candidate_ids.stride(0),
        num_blocks=num_blocks,
        NUM_BLOCKS_PAD=triton.next_power_of_2(max(num_blocks, 1)),
        RETRIES=_TOP_P_REJECTION_TRIES,
        num_warps=8,
    )
    _prepare_top_k_values(
        logits,
        local_top_k_values,
        fn_name="sample_rejection_from_pools",
        top_k_pad=_TOP_K_FILTER_MAX_K,
    )
    _top_k_total_prob_pool_kernel[(rows,)](
        req_pool_indices,
        top_k_pool,
        temperature_pool,
        candidate_logits,
        local_top_k_values,
        top_k_total_probs,
        candidate_row_stride=candidate_ids.stride(0),
        local_values_row_stride=local_top_k_values.stride(0),
        TOP_K_PAD=_TOP_K_FILTER_MAX_K,
        RETRIES=_TOP_P_REJECTION_TRIES,
        num_warps=4,
    )
    _rejection_sample_stage3_pool_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        temperature_pool,
        candidate_ids,
        candidate_logits,
        local_total_probs,
        local_before_probs,
        local_before_counts,
        logits_row_stride=logits.stride(0),
        candidate_row_stride=candidate_ids.stride(0),
        local_total_row_stride=local_total_probs.stride(0),
        local_before_row_stride=local_before_probs.stride(0),
        local_before_try_stride=local_before_probs.stride(1),
        local_before_block_stride=local_before_probs.stride(2),
        vocab_size=vocab_size,
        BLOCK_SIZE=_GUMBEL_BLOCK_SIZE,
        RETRIES=_TOP_P_REJECTION_TRIES,
        num_warps=4,
    )
    _rejection_sample_stage4_pool_kernel[(rows,)](
        req_pool_indices,
        top_k_pool,
        top_p_pool,
        candidate_ids,
        local_total_probs,
        top_k_total_probs,
        local_before_probs,
        local_before_counts,
        out,
        candidate_row_stride=candidate_ids.stride(0),
        local_total_row_stride=local_total_probs.stride(0),
        local_before_row_stride=local_before_probs.stride(0),
        local_before_try_stride=local_before_probs.stride(1),
        local_before_block_stride=local_before_probs.stride(2),
        num_blocks=num_blocks,
        NUM_BLOCKS_PAD=triton.next_power_of_2(max(num_blocks, 1)),
        TOP_K_PAD=_TOP_K_FILTER_MAX_K,
        RETRIES=_TOP_P_REJECTION_TRIES,
        num_warps=8,
    )
    return out[:rows]


def sample_rejection_min_p_from_pools(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_k_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    min_p_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    local_ids: torch.Tensor,
    local_scores: torch.Tensor,
    local_argmax_ids: torch.Tensor,
    local_argmax_scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
    local_top_k_values: torch.Tensor,
    top_k_total_probs: torch.Tensor,
    local_total_probs: torch.Tensor,
    local_before_probs: torch.Tensor,
    local_before_counts: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Pool-aware bounded rejection sampler with top-k/top-p/min-p filters."""
    rows, vocab_size, num_blocks = _check_rejection_pool_inputs(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        local_argmax_ids,
        local_argmax_scores,
        out,
        fn_name="sample_rejection_min_p_from_pools",
    )
    if min_p_pool.ndim != 1:
        raise ValueError(f"min_p_pool expects 1D tensor, got {min_p_pool.ndim}D")
    if min_p_pool.device.type != "cuda":
        raise ValueError("min_p_pool requires CUDA tensor")
    if min_p_pool.dtype != torch.float32:
        raise ValueError(f"min_p_pool must be float32, got {min_p_pool.dtype}")
    _check_rejection_scratch(
        rows,
        num_blocks,
        local_ids,
        local_scores,
        local_argmax_ids,
        local_argmax_scores,
        candidate_ids,
        candidate_logits,
        local_total_probs,
        local_before_probs,
        local_before_counts,
    )
    _check_cuda_vector("top_k_total_probs", top_k_total_probs, rows)
    if top_k_total_probs.dtype != torch.float32:
        raise ValueError(
            f"top_k_total_probs must be float32, got {top_k_total_probs.dtype}"
        )
    if rows == 0:
        return out[:0]

    _top_p_rejection_sample_stage1_pool_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        local_ids,
        local_scores,
        local_argmax_ids,
        local_argmax_scores,
        logits_row_stride=logits.stride(0),
        local_ids_row_stride=local_ids.stride(0),
        local_ids_try_stride=local_ids.stride(1),
        local_ids_block_stride=local_ids.stride(2),
        local_argmax_row_stride=local_argmax_ids.stride(0),
        vocab_size=vocab_size,
        BLOCK_SIZE=_GUMBEL_BLOCK_SIZE,
        RETRIES=_TOP_P_REJECTION_TRIES,
        num_warps=4,
    )
    _top_p_rejection_sample_stage2_pool_kernel[(rows,)](
        logits,
        req_pool_indices,
        temperature_pool,
        local_ids,
        local_scores,
        local_argmax_ids,
        local_argmax_scores,
        candidate_ids,
        candidate_logits,
        logits_row_stride=logits.stride(0),
        local_ids_row_stride=local_ids.stride(0),
        local_ids_try_stride=local_ids.stride(1),
        local_ids_block_stride=local_ids.stride(2),
        local_argmax_row_stride=local_argmax_ids.stride(0),
        candidate_row_stride=candidate_ids.stride(0),
        num_blocks=num_blocks,
        NUM_BLOCKS_PAD=triton.next_power_of_2(max(num_blocks, 1)),
        RETRIES=_TOP_P_REJECTION_TRIES,
        num_warps=8,
    )
    _prepare_top_k_values(
        logits,
        local_top_k_values,
        fn_name="sample_rejection_min_p_from_pools",
        top_k_pad=_TOP_K_FILTER_MAX_K,
    )
    _top_k_total_prob_pool_kernel[(rows,)](
        req_pool_indices,
        top_k_pool,
        temperature_pool,
        candidate_logits,
        local_top_k_values,
        top_k_total_probs,
        candidate_row_stride=candidate_ids.stride(0),
        local_values_row_stride=local_top_k_values.stride(0),
        TOP_K_PAD=_TOP_K_FILTER_MAX_K,
        RETRIES=_TOP_P_REJECTION_TRIES,
        num_warps=4,
    )
    _rejection_sample_stage3_pool_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        temperature_pool,
        candidate_ids,
        candidate_logits,
        local_total_probs,
        local_before_probs,
        local_before_counts,
        logits_row_stride=logits.stride(0),
        candidate_row_stride=candidate_ids.stride(0),
        local_total_row_stride=local_total_probs.stride(0),
        local_before_row_stride=local_before_probs.stride(0),
        local_before_try_stride=local_before_probs.stride(1),
        local_before_block_stride=local_before_probs.stride(2),
        vocab_size=vocab_size,
        BLOCK_SIZE=_GUMBEL_BLOCK_SIZE,
        RETRIES=_TOP_P_REJECTION_TRIES,
        num_warps=4,
    )
    _rejection_sample_min_p_stage4_pool_kernel[(rows,)](
        req_pool_indices,
        top_k_pool,
        top_p_pool,
        min_p_pool,
        candidate_ids,
        candidate_logits,
        local_total_probs,
        top_k_total_probs,
        local_before_probs,
        local_before_counts,
        out,
        candidate_row_stride=candidate_ids.stride(0),
        local_total_row_stride=local_total_probs.stride(0),
        local_before_row_stride=local_before_probs.stride(0),
        local_before_try_stride=local_before_probs.stride(1),
        local_before_block_stride=local_before_probs.stride(2),
        num_blocks=num_blocks,
        NUM_BLOCKS_PAD=triton.next_power_of_2(max(num_blocks, 1)),
        TOP_K_PAD=_TOP_K_FILTER_MAX_K,
        RETRIES=_TOP_P_REJECTION_TRIES,
        num_warps=8,
    )
    return out[:rows]


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
