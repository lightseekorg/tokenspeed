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

"""Fused Triton kernels for sampled DSpark Markov proposal walks.

Each DSpark block step adds a rank-reduced bigram bias ``embedding[prev] @
projection.T`` to precomputed base logits, normalizes per-request temperature /
greedy probabilities into static workspace storage, and (after optional
in-place Top-K/Top-P renormalization) scatters normalized probabilities directly
into the verifier's slot pool while sampling the step token with
``Philox4x32-10`` and prefetching the next bigram embedding row.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature, format_signatures

__all__ = [
    "DsparkMarkovSampleWorkspace",
    "dspark_markov_sample_workspace",
    "triton_dspark_markov_softmax",
    "triton_dspark_probs_scatter_and_sample",
]

_LOG2E: tl.constexpr = tl.constexpr(1.4426950408889634)
_BLOCK_M: int = 16
_BLOCK_K: int = 32
_SUB_V: int = 64
_BLOCK_V: int = 1024
_COPY_K: int = 128

_PHILOX_M0: tl.constexpr = tl.constexpr(0xD2511F53)
_PHILOX_M1: tl.constexpr = tl.constexpr(0xCD9E8D57)
_PHILOX_W0: tl.constexpr = tl.constexpr(0x9E3779B9)
_PHILOX_W1: tl.constexpr = tl.constexpr(0xBB67AE85)


@dataclass(frozen=True)
class DsparkMarkovSampleWorkspace:
    """Static batch-sized buffers for sampled DSpark Markov generation."""

    markov_embed: torch.Tensor
    probs: torch.Tensor
    partial_max: torch.Tensor
    partial_sum: torch.Tensor
    partial_idx: torch.Tensor
    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor
    seeds: torch.Tensor
    vocab_size: int
    rank: int
    block_size_v: int
    block_p: int


def dspark_markov_sample_workspace(
    *,
    max_rows: int,
    vocab_size: int,
    rank: int,
    embed_dtype: torch.dtype,
    device: torch.device | str,
) -> DsparkMarkovSampleWorkspace:
    """Allocate persistent CUDA-graph-safe workspace for sampled DSpark."""
    if max_rows < 0 or vocab_size <= 0 or rank <= 0:
        raise ValueError(
            "dspark_markov_sample_workspace requires max_rows >= 0, vocab_size > 0, "
            f"rank > 0; got {max_rows=} {vocab_size=} {rank=}"
        )
    num_blocks = triton.cdiv(vocab_size, _BLOCK_V)
    block_p = triton.next_power_of_2(num_blocks)
    return DsparkMarkovSampleWorkspace(
        markov_embed=torch.empty((max_rows, rank), dtype=embed_dtype, device=device),
        probs=torch.empty((max_rows, vocab_size), dtype=torch.float32, device=device),
        partial_max=torch.empty((max_rows, block_p), dtype=torch.float32, device=device),
        partial_sum=torch.empty((max_rows, block_p), dtype=torch.float32, device=device),
        partial_idx=torch.empty((max_rows, block_p), dtype=torch.int32, device=device),
        temperatures=torch.empty((max_rows,), dtype=torch.float32, device=device),
        top_ks=torch.empty((max_rows,), dtype=torch.int32, device=device),
        top_ps=torch.empty((max_rows,), dtype=torch.float32, device=device),
        seeds=torch.empty((max_rows,), dtype=torch.int64, device=device),
        vocab_size=vocab_size,
        rank=rank,
        block_size_v=_BLOCK_V,
        block_p=block_p,
    )


@triton.jit
def _philox4x32_10_uniform(seed, position, rng_tag):
    """Philox4x32-10 matching CuTe ``packed_verify.uniform``."""
    bits = seed.to(tl.uint64)
    pos = position.to(tl.uint64)
    k0 = (bits & 0xFFFFFFFF).to(tl.uint32)
    k1 = (bits >> 32).to(tl.uint32)
    x0 = (pos & 0xFFFFFFFF).to(tl.uint32)
    x1 = (pos >> 32).to(tl.uint32)
    x2 = (tl.zeros((), dtype=tl.uint32) + rng_tag).to(tl.uint32)
    x3 = tl.zeros((), dtype=tl.uint32)
    m0 = tl.full((), _PHILOX_M0, tl.uint64)
    m1 = tl.full((), _PHILOX_M1, tl.uint64)
    w0 = tl.full((), _PHILOX_W0, tl.uint64).to(tl.uint32)
    w1 = tl.full((), _PHILOX_W1, tl.uint64).to(tl.uint32)
    for _ in tl.static_range(10):
        p0 = x0.to(tl.uint64) * m0
        p1 = x2.to(tl.uint64) * m1
        x0 = ((p1 >> 32).to(tl.uint32) ^ x1) ^ k0
        x1 = (p1 & 0xFFFFFFFF).to(tl.uint32)
        x2 = ((p0 >> 32).to(tl.uint32) ^ x3) ^ k1
        x3 = (p0 & 0xFFFFFFFF).to(tl.uint32)
        k0 = k0 + w0
        k1 = k1 + w1
    return (x0 >> 8).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit(
    do_not_specialize=[
        "rows",
        "embed_vocab",
        "rank",
        "anchor_stride",
        "slots_stride",
        "embed_row_stride",
        "out_embed_row_stride",
    ],
    do_not_specialize_on_alignment=[
        "anchor_ptr",
        "slots_ptr",
        "embedding_ptr",
    ],
)
def _dspark_markov_init_kernel(
    anchor_ptr,
    slots_ptr,
    temperature_pool_ptr,
    top_k_pool_ptr,
    top_p_pool_ptr,
    seed_pool_ptr,
    embedding_ptr,
    out_embed_ptr,
    out_temp_ptr,
    out_top_k_ptr,
    out_top_p_ptr,
    out_seed_ptr,
    rows,
    embed_vocab,
    rank,
    anchor_stride,
    slots_stride,
    embed_row_stride,
    out_embed_row_stride,
    COPY_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row >= rows:
        return
    zero64 = tl.zeros((), dtype=tl.int64)
    slot = tl.load(slots_ptr + row * (zero64 + slots_stride)).to(tl.int64)
    tl.store(out_temp_ptr + row, tl.load(temperature_pool_ptr + slot))
    tl.store(out_top_k_ptr + row, tl.load(top_k_pool_ptr + slot))
    tl.store(out_top_p_ptr + row, tl.load(top_p_pool_ptr + slot))
    tl.store(out_seed_ptr + row, tl.load(seed_pool_ptr + slot))

    anchor = tl.load(anchor_ptr + row * (zero64 + anchor_stride)).to(tl.int32)
    # Graph padding rows may carry out-of-range anchors; clamp before indexing.
    clamped = tl.minimum(tl.maximum(anchor, 0), embed_vocab - 1).to(tl.int64)
    src_base = embedding_ptr + clamped * (zero64 + embed_row_stride)
    dst_base = out_embed_ptr + row * (zero64 + out_embed_row_stride)
    for k_start in range(0, rank, COPY_K):
        k_offs = k_start + tl.arange(0, COPY_K)
        k_mask = k_offs < rank
        vals = tl.load(src_base + k_offs, mask=k_mask, other=0.0)
        tl.store(dst_base + k_offs, vals, mask=k_mask)


@triton.jit(
    do_not_specialize=[
        "rows",
        "step",
        "vocab_size",
        "rank",
        "base_row_stride",
        "base_step_stride",
        "embed_row_stride",
        "proj_row_stride",
        "probs_row_stride",
        "partial_stride",
    ],
    do_not_specialize_on_alignment=[
        "base_ptr",
        "embed_ptr",
        "proj_ptr",
        "probs_ptr",
    ],
)
def _dspark_markov_stage1_kernel(
    base_ptr,
    embed_ptr,
    proj_ptr,
    temps_ptr,
    top_ks_ptr,
    probs_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_idx_ptr,
    rows,
    step,
    vocab_size,
    rank,
    base_row_stride,
    base_step_stride,
    embed_row_stride,
    proj_row_stride,
    probs_row_stride,
    partial_stride,
    ROWS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    SUB_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_IEEE_DOT: tl.constexpr,
):
    block_id = tl.program_id(0)
    row_offs = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    row_mask = row_offs < rows
    zero64 = tl.zeros((), dtype=tl.int64)

    temp = tl.load(temps_ptr + row_offs, mask=row_mask, other=1.0).to(tl.float32)
    safe_temp = tl.maximum(temp, 1.0e-20)
    inv_temp_log2e = _LOG2E / safe_temp

    block_start = block_id * BLOCK_V
    base_step_ptr = base_ptr + (zero64 + step) * (zero64 + base_step_stride)
    embed_base = embed_ptr + row_offs[:, None].to(tl.int64) * (
        zero64 + embed_row_stride
    )
    probs_base = probs_ptr + row_offs[:, None].to(tl.int64) * (
        zero64 + probs_row_stride
    )
    row_base_ptr = base_step_ptr + row_offs[:, None].to(tl.int64) * (
        zero64 + base_row_stride
    )

    blk_max = tl.full((ROWS,), float("-inf"), dtype=tl.float32)
    blk_sum = tl.full((ROWS,), 0.0, dtype=tl.float32)
    blk_idx = tl.full((ROWS,), vocab_size, dtype=tl.int32)

    for sub in tl.static_range(0, BLOCK_V, SUB_V):
        if block_start + sub < vocab_size:
            col_offs = block_start + sub + tl.arange(0, SUB_V)
            col_mask = col_offs < vocab_size
            proj_base = proj_ptr + col_offs[:, None].to(tl.int64) * (
                zero64 + proj_row_stride
            )

            acc = tl.zeros((ROWS, SUB_V), dtype=tl.float32)
            for k_start in range(0, rank, BLOCK_K):
                k_offs = k_start + tl.arange(0, BLOCK_K)
                k_mask = k_offs < rank
                a = tl.load(
                    embed_base + k_offs[None, :],
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                b = tl.load(
                    proj_base + k_offs[None, :],
                    mask=col_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                if USE_IEEE_DOT:
                    acc = tl.dot(a, tl.trans(b), acc=acc, input_precision="ieee")
                else:
                    acc = tl.dot(a, tl.trans(b), acc=acc)

            base = tl.load(
                row_base_ptr + col_offs[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=float("-inf"),
            ).to(tl.float32)
            logits = base + acc
            tl.store(
                probs_base + col_offs[None, :],
                logits,
                mask=row_mask[:, None] & col_mask[None, :],
            )

            sub_logits = tl.where(col_mask[None, :], logits, float("-inf"))
            sub_max = tl.max(sub_logits, axis=1)
            new_max = tl.maximum(blk_max, sub_max)
            rescale = tl.where(
                blk_max == float("-inf"),
                0.0,
                tl.exp2((blk_max - new_max) * inv_temp_log2e),
            )
            sub_exp = tl.where(
                col_mask[None, :]
                & (sub_logits != float("-inf"))
                & (new_max[:, None] != float("-inf")),
                tl.exp2((sub_logits - new_max[:, None]) * inv_temp_log2e),
                0.0,
            )
            sub_sum = tl.sum(sub_exp, axis=1)
            blk_sum = blk_sum * rescale + sub_sum

            # Track exact lowest-index argmax per block for top_k == 1 rows.
            is_best = col_mask[None, :] & (sub_logits == sub_max[:, None])
            sub_idx = tl.min(tl.where(is_best, col_offs[None, :], vocab_size), axis=1)
            take_sub = (sub_max > blk_max) | (
                (sub_max == blk_max) & (sub_idx < blk_idx)
            )
            blk_idx = tl.where(take_sub, sub_idx, blk_idx)
            blk_max = new_max

    out_offs = row_offs.to(tl.int64) * (zero64 + partial_stride) + block_id
    tl.store(partial_max_ptr + out_offs, blk_max, mask=row_mask)
    tl.store(partial_sum_ptr + out_offs, blk_sum, mask=row_mask)
    tl.store(partial_idx_ptr + out_offs, blk_idx, mask=row_mask)


@triton.jit(
    do_not_specialize=[
        "vocab_size",
        "num_blocks",
        "probs_row_stride",
        "partial_stride",
    ],
    do_not_specialize_on_alignment=["probs_ptr"],
)
def _dspark_markov_stage2_norm_kernel(
    probs_ptr,
    temps_ptr,
    top_ks_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_idx_ptr,
    vocab_size,
    num_blocks,
    probs_row_stride,
    partial_stride,
    BLOCK_V: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    block_id = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    zero64 = tl.zeros((), dtype=tl.int64)

    p = tl.arange(0, BLOCK_P)
    p_mask = p < num_blocks
    p_base = row * (zero64 + partial_stride) + p
    p_max = tl.load(partial_max_ptr + p_base, mask=p_mask, other=float("-inf"))
    global_max = tl.max(p_max, axis=0)

    top_k = tl.load(top_ks_ptr + row)
    offs = block_id * BLOCK_V + tl.arange(0, BLOCK_V)
    n_mask = offs < vocab_size
    dst = probs_ptr + row * (zero64 + probs_row_stride) + offs

    if top_k == 1:
        p_idx = tl.load(partial_idx_ptr + p_base, mask=p_mask, other=vocab_size)
        win_idx = tl.min(
            tl.where(p_mask & (p_max == global_max), p_idx, vocab_size), axis=0
        )
        p_out = tl.where(n_mask & (offs == win_idx), 1.0, 0.0).to(tl.float32)
        tl.store(dst, p_out, mask=n_mask)
    else:
        temp = tl.maximum(tl.load(temps_ptr + row).to(tl.float32), 1.0e-20)
        inv_temp_log2e = _LOG2E / temp
        p_sum = tl.load(partial_sum_ptr + p_base, mask=p_mask, other=0.0)
        exp_diff = tl.where(
            p_mask & (p_max != float("-inf")) & (p_sum > 0.0),
            tl.exp2((p_max - global_max) * inv_temp_log2e),
            0.0,
        )
        total_sum = tl.sum(p_sum * exp_diff, axis=0)
        logits = tl.load(dst, mask=n_mask, other=float("-inf")).to(tl.float32)
        valid = (
            n_mask
            & (logits != float("-inf"))
            & (global_max != float("-inf"))
            & (total_sum > 0.0)
        )
        p_out = tl.where(
            valid, tl.exp2((logits - global_max) * inv_temp_log2e) / total_sum, 0.0
        )
        tl.store(dst, p_out, mask=n_mask)


@triton.jit(
    do_not_specialize=[
        "step",
        "vocab_size",
        "probs_row_stride",
        "draft_probs_slot_stride",
        "draft_probs_step_stride",
        "write_slots_stride",
        "partial_stride",
    ],
    do_not_specialize_on_alignment=[
        "probs_ptr",
        "write_slots_ptr",
        "draft_probs_ptr",
    ],
)
def _dspark_scatter_and_mass_kernel(
    probs_ptr,
    write_slots_ptr,
    draft_probs_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_idx_ptr,
    step,
    vocab_size,
    probs_row_stride,
    draft_probs_slot_stride,
    draft_probs_step_stride,
    write_slots_stride,
    partial_stride,
    BLOCK_V: tl.constexpr,
):
    block_id = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    zero64 = tl.zeros((), dtype=tl.int64)

    offs = block_id * BLOCK_V + tl.arange(0, BLOCK_V)
    n_mask = offs < vocab_size
    p = tl.load(
        probs_ptr + row * (zero64 + probs_row_stride) + offs, mask=n_mask, other=0.0
    ).to(tl.float32)

    write_slot = tl.load(write_slots_ptr + row * (zero64 + write_slots_stride)).to(
        tl.int64
    )
    dst_base = (
        draft_probs_ptr
        + write_slot * (zero64 + draft_probs_slot_stride)
        + (zero64 + step) * (zero64 + draft_probs_step_stride)
    )
    tl.store(dst_base + offs, p, mask=n_mask)

    blk_mass = tl.sum(tl.where(n_mask, p, 0.0), axis=0)
    blk_max = tl.max(tl.where(n_mask, p, -1.0), axis=0)
    blk_idx = tl.min(tl.where(n_mask & (p == blk_max), offs, vocab_size), axis=0)

    out_off = row * (zero64 + partial_stride) + block_id
    tl.store(partial_sum_ptr + out_off, blk_mass)
    tl.store(partial_max_ptr + out_off, blk_max)
    tl.store(partial_idx_ptr + out_off, blk_idx)


@triton.jit(
    do_not_specialize=[
        "step",
        "vocab_size",
        "embed_vocab",
        "rank",
        "num_blocks",
        "probs_row_stride",
        "partial_stride",
        "seeds_stride",
        "pos_stride",
        "position_delta",
        "rng_tag",
        "out_row_stride",
        "out_col_stride",
        "embed_row_stride",
        "out_embed_row_stride",
    ],
    do_not_specialize_on_alignment=[
        "probs_ptr",
        "seeds_ptr",
        "positions_ptr",
        "out_tokens_ptr",
        "embedding_ptr",
        "out_embed_ptr",
    ],
)
def _dspark_select_and_embed_kernel(
    probs_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_idx_ptr,
    seeds_ptr,
    positions_ptr,
    out_tokens_ptr,
    embedding_ptr,
    out_embed_ptr,
    step,
    vocab_size,
    embed_vocab,
    rank,
    num_blocks,
    probs_row_stride,
    partial_stride,
    seeds_stride,
    pos_stride,
    position_delta,
    rng_tag,
    out_row_stride,
    out_col_stride,
    embed_row_stride,
    out_embed_row_stride,
    BLOCK_V: tl.constexpr,
    BLOCK_P: tl.constexpr,
    COPY_K: tl.constexpr,
    GATHER_EMBED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    zero64 = tl.zeros((), dtype=tl.int64)

    p = tl.arange(0, BLOCK_P)
    p_mask = p < num_blocks
    p_base = row * (zero64 + partial_stride) + p

    masses = tl.load(partial_sum_ptr + p_base, mask=p_mask, other=0.0)
    total_mass = tl.sum(masses, axis=0)
    cum = tl.cumsum(masses, axis=0)

    seed = tl.load(seeds_ptr + row * (zero64 + seeds_stride)).to(tl.int64)
    position = tl.load(positions_ptr + row * (zero64 + pos_stride)).to(tl.int64) + (
        zero64 + position_delta
    )
    coin = _philox4x32_10_uniform(seed, position, rng_tag)
    threshold = coin * total_mass

    p_max = tl.load(partial_max_ptr + p_base, mask=p_mask, other=-1.0)
    p_idx = tl.load(partial_idx_ptr + p_base, mask=p_mask, other=vocab_size)
    g_max = tl.max(p_max, axis=0)
    fallback_tok = tl.min(
        tl.where(p_mask & (p_max == g_max), p_idx, vocab_size), axis=0
    )

    hit = p_mask & (cum > threshold)
    blk = tl.min(tl.where(hit, p, num_blocks), axis=0)
    sampled = fallback_tok
    if (total_mass > 0.0) & (blk < num_blocks):
        before = tl.sum(tl.where(p == blk, cum - masses, 0.0), axis=0)
        offs = blk * BLOCK_V + tl.arange(0, BLOCK_V)
        in_blk = offs < vocab_size
        vals = tl.load(
            probs_ptr + row * (zero64 + probs_row_stride) + offs,
            mask=in_blk,
            other=0.0,
        )
        prefix = before + tl.cumsum(vals, axis=0)
        cand = tl.min(
            tl.where(in_blk & (vals > 0.0) & (prefix > threshold), offs, vocab_size),
            axis=0,
        )
        sampled = tl.where(cand < vocab_size, cand, fallback_tok)

    sampled_i32 = sampled.to(tl.int32)
    tl.store(
        out_tokens_ptr
        + row * (zero64 + out_row_stride)
        + (zero64 + step) * (zero64 + out_col_stride),
        sampled_i32,
    )

    if GATHER_EMBED:
        clamped = tl.minimum(tl.maximum(sampled_i32, 0), embed_vocab - 1).to(tl.int64)
        src_base = embedding_ptr + clamped * (zero64 + embed_row_stride)
        dst_base = out_embed_ptr + row * (zero64 + out_embed_row_stride)
        for k_start in range(0, rank, COPY_K):
            k_offs = k_start + tl.arange(0, COPY_K)
            k_mask = k_offs < rank
            vals = tl.load(src_base + k_offs, mask=k_mask, other=0.0)
            tl.store(dst_base + k_offs, vals, mask=k_mask)


@register_kernel(
    "sampling",
    "dspark_markov_softmax",
    name="triton_dspark_markov_softmax",
    solution="triton",
    signatures=format_signatures(
        "base_logits", "dense", {torch.float32, torch.bfloat16, torch.float16}
    ),
    priority=Priority.PORTABLE,
)
def triton_dspark_markov_softmax(
    base_logits: torch.Tensor,
    step: int,
    anchor_ids: torch.Tensor,
    embedding: torch.Tensor,
    projection: torch.Tensor,
    slots: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_k_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    workspace: DsparkMarkovSampleWorkspace,
    vocab_size: int,
    *,
    prepare_step_inputs: bool,
) -> torch.Tensor:
    """Add Markov bigram bias to ``base_logits[:, step]`` and softmax into workspace."""
    if base_logits.ndim != 3 or base_logits.stride(2) != 1:
        raise ValueError("base_logits must be a 3D tensor with contiguous vocab axis")
    rows, block, local_vocab = base_logits.shape
    if not 0 <= step < block:
        raise ValueError(f"step {step} is outside the block of {block}")
    if vocab_size <= 0 or vocab_size > local_vocab:
        raise ValueError(
            f"vocab_size={vocab_size} must be in (0, local_vocab={local_vocab}]"
        )
    if embedding.ndim != 2 or projection.ndim != 2:
        raise ValueError("embedding and projection must be 2D tensors")
    if embedding.dtype != projection.dtype:
        raise ValueError("embedding and projection must share dtype")
    if embedding.stride(1) != 1 or projection.stride(1) != 1:
        raise ValueError("embedding and projection rank axes must be contiguous")
    rank = embedding.shape[1]
    if projection.shape[1] != rank or projection.shape[0] < vocab_size:
        raise ValueError("projection shape does not match rank and vocab_size")
    if (
        workspace.vocab_size != vocab_size
        or workspace.rank != rank
        or workspace.markov_embed.dtype != embedding.dtype
        or workspace.probs.shape[0] < rows
    ):
        raise ValueError("DsparkMarkovSampleWorkspace does not cover this batch")
    if rows == 0:
        return workspace.probs[:0]
    if slots.shape != (rows,) or anchor_ids.shape != (rows,):
        raise ValueError("slots and anchor_ids must have shape (rows,)")

    num_blocks = triton.cdiv(vocab_size, workspace.block_size_v)
    if prepare_step_inputs:
        _dspark_markov_init_kernel[(rows,)](
            anchor_ids,
            slots,
            temperature_pool,
            top_k_pool,
            top_p_pool,
            seed_pool,
            embedding,
            workspace.markov_embed,
            workspace.temperatures,
            workspace.top_ks,
            workspace.top_ps,
            workspace.seeds,
            rows,
            embedding.shape[0],
            rank,
            anchor_ids.stride(0),
            slots.stride(0),
            embedding.stride(0),
            workspace.markov_embed.stride(0),
            COPY_K=_COPY_K,
        )

    use_ieee_dot = embedding.dtype == torch.float32
    _dspark_markov_stage1_kernel[(num_blocks, triton.cdiv(rows, _BLOCK_M))](
        base_logits,
        workspace.markov_embed,
        projection,
        workspace.temperatures,
        workspace.top_ks,
        workspace.probs,
        workspace.partial_max,
        workspace.partial_sum,
        workspace.partial_idx,
        rows,
        step,
        vocab_size,
        rank,
        base_logits.stride(0),
        base_logits.stride(1),
        workspace.markov_embed.stride(0),
        projection.stride(0),
        workspace.probs.stride(0),
        workspace.partial_max.stride(0),
        ROWS=_BLOCK_M,
        BLOCK_V=workspace.block_size_v,
        SUB_V=_SUB_V,
        BLOCK_K=_BLOCK_K,
        USE_IEEE_DOT=use_ieee_dot,
        num_warps=4,
    )

    _dspark_markov_stage2_norm_kernel[(num_blocks, rows)](
        workspace.probs,
        workspace.temperatures,
        workspace.top_ks,
        workspace.partial_max,
        workspace.partial_sum,
        workspace.partial_idx,
        vocab_size,
        num_blocks,
        workspace.probs.stride(0),
        workspace.partial_max.stride(0),
        BLOCK_V=workspace.block_size_v,
        BLOCK_P=workspace.block_p,
        num_warps=4,
    )
    return workspace.probs[:rows]


@register_kernel(
    "sampling",
    "dspark_probs_scatter_and_sample",
    name="triton_dspark_probs_scatter_and_sample",
    solution="triton",
    signatures=frozenset(
        {format_signature(probs=dense_tensor_format(torch.float32))}
    ),
    priority=Priority.PORTABLE,
)
def triton_dspark_probs_scatter_and_sample(
    probs: torch.Tensor,
    write_slots: torch.Tensor,
    draft_probs: torch.Tensor,
    step: int,
    seeds: torch.Tensor,
    base_positions: torch.Tensor,
    position_delta: int,
    rng_tag: int,
    embedding: torch.Tensor,
    next_tokens_out: torch.Tensor,
    workspace: DsparkMarkovSampleWorkspace,
    vocab_size: int,
    *,
    gather_next_embed: bool,
) -> None:
    """Record ``probs`` into ``draft_probs`` at ``write_slots`` and sample step tokens."""
    if probs.ndim != 2 or probs.dtype != torch.float32 or probs.stride(1) != 1:
        raise ValueError("probs must be 2D FP32 with contiguous vocabulary axis")
    rows, width = probs.shape
    if width != vocab_size:
        raise ValueError(f"probs width {width} must equal vocab_size {vocab_size}")
    if draft_probs.ndim != 3 or draft_probs.dtype != torch.float32:
        raise ValueError("draft_probs must be a 3D FP32 tensor")
    if draft_probs.stride(2) != 1 or draft_probs.shape[2] < vocab_size:
        raise ValueError("draft_probs vocabulary axis must be contiguous and >= vocab")
    if not 0 <= step < draft_probs.shape[1]:
        raise ValueError(f"step {step} is outside draft_probs width {draft_probs.shape[1]}")
    if rows == 0:
        return
    if write_slots.shape != (rows,) or seeds.shape != (rows,) or base_positions.shape != (rows,):
        raise ValueError("write_slots, seeds, and base_positions must have shape (rows,)")
    if next_tokens_out.ndim != 2 or next_tokens_out.shape[0] != rows or next_tokens_out.shape[1] <= step:
        raise ValueError("next_tokens_out shape does not cover (rows, step)")
    if embedding.ndim != 2 or embedding.stride(1) != 1:
        raise ValueError("embedding must be 2D with contiguous rank axis")
    rank = embedding.shape[1]
    if workspace.rank != rank or workspace.markov_embed.dtype != embedding.dtype:
        raise ValueError("workspace embedding shape/dtype mismatch")

    num_blocks = triton.cdiv(vocab_size, workspace.block_size_v)
    _dspark_scatter_and_mass_kernel[(num_blocks, rows)](
        probs,
        write_slots,
        draft_probs,
        workspace.partial_max,
        workspace.partial_sum,
        workspace.partial_idx,
        step,
        vocab_size,
        probs.stride(0),
        draft_probs.stride(0),
        draft_probs.stride(1),
        write_slots.stride(0),
        workspace.partial_max.stride(0),
        BLOCK_V=workspace.block_size_v,
        num_warps=4,
    )

    _dspark_select_and_embed_kernel[(rows,)](
        probs,
        workspace.partial_max,
        workspace.partial_sum,
        workspace.partial_idx,
        seeds,
        base_positions,
        next_tokens_out,
        embedding,
        workspace.markov_embed,
        step,
        vocab_size,
        embedding.shape[0],
        rank,
        num_blocks,
        probs.stride(0),
        workspace.partial_max.stride(0),
        seeds.stride(0),
        base_positions.stride(0),
        position_delta,
        rng_tag,
        next_tokens_out.stride(0),
        next_tokens_out.stride(1),
        embedding.stride(0),
        workspace.markov_embed.stride(0),
        BLOCK_V=workspace.block_size_v,
        BLOCK_P=workspace.block_p,
        COPY_K=_COPY_K,
        GATHER_EMBED=gather_next_embed,
        num_warps=4,
    )
