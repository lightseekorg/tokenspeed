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

"""Dense target probabilities and draft-probability verify for the triton
sampling backend (``--enable-speculative-sampling``).

The target distribution ``p'`` of a verify row is the one the backend's own
route samples plain rows from: ``softmax(logits / T)`` on the no-filter
route; on the top-k routes the softmax over the row's kept tokens (its
``min(top_k, top_k_pad)`` best logits, lowest id first among equal logits,
after the top-p cut), zero elsewhere.

Coins follow the position-keyed noise of ``rejection.py``: a request with
seed ``s`` and verify offset ``o`` draws the uniform of verify row ``r`` from
``rand(randint(s, COIN_OFFSET_SALT + o + r), lane)``, lane 0 for a chain's
accept coin and 1 for its residual draw. A tree node at depth ``d`` is
verified on its parent's row ``max(d - 1, 0)`` and takes lane ``i`` (its node
index); the bonus takes row 0, lane ``N``. Verify offsets advance past every
row a round consulted, so no coin is reused.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.sampling.triton.rejection import COIN_OFFSET_SALT
from tokenspeed_kernel.ops.sampling.triton.softmax import softmax_rows
from tokenspeed_kernel.ops.sampling.triton.topk_topp import (
    _merge_top_k_candidates,
    _top_k_top_p_candidates_stage1_kernel,
)

__all__ = [
    "chain_verify_draft_probs",
    "target_probs_from_pools",
    "tree_verify_coins",
]

_ZERO_BLOCK = 4096
_RESIDUAL_BLOCK = 4096


@triton.jit(do_not_specialize=["num_tokens_per_req", "out_row_stride"])
def _target_probs_top_k_kernel(
    candidate_ids_ptr,
    candidate_logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    top_k_pool_ptr,
    top_p_pool_ptr,
    out_ptr,
    num_tokens_per_req,
    out_row_stride,
    vocab_size,
    candidate_row_stride: tl.constexpr,
    num_blocks: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    NUM_BLOCKS_PAD: tl.constexpr,
    ZERO_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    pool_idx = tl.load(req_pool_indices_ptr + row // num_tokens_per_req)
    # The plain top-k sampler's arithmetic (_top_k_top_p_gumbel_stage2_kernel).
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    top_k = tl.load(top_k_pool_ptr + pool_idx)
    top_k = tl.minimum(tl.maximum(top_k, 1), vocab_size)
    top_p = tl.load(top_p_pool_ptr + pool_idx).to(tl.float32)
    top_logits, top_ids = _merge_top_k_candidates(
        candidate_ids_ptr,
        candidate_logits_ptr,
        row,
        top_k,
        candidate_row_stride,
        num_blocks,
        TOPK_PAD,
        NUM_BLOCKS_PAD,
    )
    rank_offsets = tl.arange(0, TOPK_PAD)
    scaled_logits = top_logits / temperature
    max_logit = tl.max(scaled_logits, axis=0)
    weights = tl.exp(scaled_logits - max_logit)
    probs = weights / tl.maximum(tl.sum(weights, axis=0), 1.0e-20)
    cumulative_before = tl.cumsum(probs) - probs
    keep = (rank_offsets < top_k) & (cumulative_before < top_p)
    kept_weights = tl.where(keep, tl.exp(scaled_logits - max_logit), 0.0)
    lse = max_logit + tl.log(tl.sum(kept_weights, axis=0))
    kept_p = tl.where(keep, tl.exp(scaled_logits - lse), 0.0)

    out_row = out_ptr + row.to(tl.int64) * out_row_stride
    cols = tl.arange(0, ZERO_BLOCK)
    for start in range(0, vocab_size, ZERO_BLOCK):
        tl.store(out_row + start + cols, 0.0, mask=start + cols < vocab_size)
    # The kept entries overwrite zeros other threads stored.
    tl.debug_barrier()
    tl.store(out_row + top_ids, kept_p, mask=keep)


def target_probs_from_pools(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_k_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
    out: torch.Tensor,
    *,
    top_k_filter: bool,
    block_size: int,
    top_k_pad: int,
    num_tokens_per_req: int,
) -> torch.Tensor:
    """Each verify row's target distribution ``p'`` (see the module docstring).

    Args:
        logits: ``[bs * N, V]`` target logits, unit column stride.
        req_pool_indices: ``[bs]`` int32 pool slot of each request.
        temperature_pool: ``[pool]`` fp32 temperature.
        top_k_pool: ``[pool]`` int32 top-k.
        top_p_pool: ``[pool]`` fp32 top-p.
        candidate_ids: int32 ``[bs * N, cdiv(V, block_size) * top_k_pad]``
            stage-1 scratch of the top-k routes.
        candidate_logits: fp32 scratch shaped like ``candidate_ids``.
        out: ``[bs * N, V]`` fp32 output, unit column stride.
        top_k_filter: The top-k routes' ``p'`` (finite top-k, any top-p);
            False for the no-filter route's plain softmax.
        block_size: Stage-1 vocabulary block of the top-k routes.
        top_k_pad: Most kept tokens per row (power of two).
        num_tokens_per_req: Verify window width ``N``.

    Returns:
        ``out``.
    """
    rows, vocab_size = logits.shape
    if out.shape != logits.shape or out.dtype != torch.float32:
        raise ValueError(
            f"target_probs_from_pools: out must be fp32 {tuple(logits.shape)}, got "
            f"{out.dtype} {tuple(out.shape)}"
        )
    if req_pool_indices.dtype != torch.int32:
        raise ValueError("target_probs_from_pools: req_pool_indices must be int32")
    if rows != req_pool_indices.shape[0] * num_tokens_per_req:
        raise ValueError(
            f"target_probs_from_pools: {rows} rows for "
            f"{req_pool_indices.shape[0]} requests of {num_tokens_per_req}"
        )
    if rows == 0:
        return out
    if not top_k_filter:
        return softmax_rows(
            logits,
            temperature_pool,
            out,
            index=req_pool_indices,
            rows_per_index=num_tokens_per_req,
        )
    if top_k_pad & (top_k_pad - 1):
        raise ValueError("top_k_pad must be a power of two for tl.topk")
    num_blocks = triton.cdiv(vocab_size, block_size)
    if (
        candidate_ids.shape[0] < rows
        or candidate_ids.shape[1] < num_blocks * top_k_pad
        or candidate_logits.shape != candidate_ids.shape
        or candidate_ids.stride(-1) != 1
        or candidate_logits.stride(-1) != 1
    ):
        raise ValueError(
            "target_probs_from_pools: candidate scratch is too small or strided"
        )
    _top_k_top_p_candidates_stage1_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        top_k_pool,
        candidate_ids,
        candidate_logits,
        logits_row_stride=logits.stride(0),
        candidate_row_stride=candidate_ids.stride(0),
        vocab_size=vocab_size,
        BLOCK_SIZE=block_size,
        TOPK_PAD=top_k_pad,
        NUM_TOKENS_PER_REQ=num_tokens_per_req,
        num_warps=1 if block_size <= 1024 else 8,
    )
    _target_probs_top_k_kernel[(rows,)](
        candidate_ids,
        candidate_logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        out,
        num_tokens_per_req,
        out.stride(0),
        vocab_size,
        candidate_row_stride=candidate_ids.stride(0),
        num_blocks=num_blocks,
        TOPK_PAD=top_k_pad,
        NUM_BLOCKS_PAD=triton.next_power_of_2(num_blocks),
        ZERO_BLOCK=_ZERO_BLOCK,
        num_warps=4,
    )
    return out


@triton.jit
def _row_coin(seed, offset, lane, COIN_SALT: tl.constexpr):
    return tl.rand(tl.randint(seed, COIN_SALT + offset), lane)


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "probs_row_stride",
        "draft_probs_pool_stride",
        "draft_probs_step_stride",
        "candidates_row_stride",
        "accept_index_row_stride",
    ]
)
def _chain_verify_draft_probs_kernel(
    probs_ptr,
    draft_probs_ptr,
    candidates_ptr,
    req_pool_indices_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    predicts_ptr,
    accept_index_ptr,
    accept_length_ptr,
    reject_threshold,
    num_tokens,
    vocab_size,
    probs_row_stride,
    draft_probs_pool_stride,
    draft_probs_step_stride,
    candidates_row_stride,
    accept_index_row_stride,
    COIN_SALT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    pool_idx = tl.load(req_pool_indices_ptr + b).to(tl.int64)
    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64)
    row0 = b.to(tl.int64) * num_tokens
    q_base = draft_probs_ptr + pool_idx * draft_probs_pool_stride
    cand_row = candidates_ptr + b.to(tl.int64) * candidates_row_stride
    index_row = accept_index_ptr + b.to(tl.int64) * accept_index_row_stride
    tl.store(index_row, row0.to(tl.int32))

    # Walk the chain: row r = i - 1 verifies draft x_i, drawn from
    # draft_probs[pool, r]; stop at the first rejection.
    accepted = tl.full((), 0, tl.int32)
    going = tl.full((), 1, tl.int32) > 0
    i = tl.full((), 1, tl.int32)
    while going & (i < num_tokens):
        draft = tl.load(cand_row + i)
        in_vocab = (draft >= 0) & (draft < vocab_size)
        safe = tl.where(in_vocab, draft, 0)
        p = tl.load(probs_ptr + (row0 + i - 1) * probs_row_stride + safe)
        q = tl.load(q_base + (i - 1).to(tl.int64) * draft_probs_step_stride + safe)
        coin = _row_coin(seed, offset + i - 1, tl.full((), 0, tl.int32), COIN_SALT)
        # q above the threshold is the "no recorded proposal" sentinel.
        accept = in_vocab & (q <= reject_threshold) & (coin * q < p)
        if accept:
            tl.store(predicts_ptr + row0 + i - 1, draft)
            tl.store(index_row + i, (row0 + i).to(tl.int32))
            accepted += 1
            i += 1
        else:
            going = tl.full((), 0, tl.int32) > 0
    tl.store(accept_length_ptr + b, accepted)

    # The emitted token at row pos: a draw from norm(relu(p - q)) after a
    # rejection (a sentinel row subtracts nothing), from p after the last row.
    pos = accepted
    p_row = probs_ptr + (row0 + pos) * probs_row_stride
    q_row = q_base + pos.to(tl.int64) * draft_probs_step_stride
    # Columns of q to subtract: none for the last row (no draft follows it).
    q_cols = tl.where(pos < num_tokens - 1, vocab_size, 0)
    coin = _row_coin(seed, offset + pos, tl.full((), 1, tl.int32), COIN_SALT)
    cols = tl.arange(0, BLOCK)
    total = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, vocab_size, BLOCK):
        mask = start + cols < vocab_size
        p = tl.load(p_row + start + cols, mask=mask, other=0.0)
        q = tl.load(q_row + start + cols, mask=start + cols < q_cols, other=0.0)
        q = tl.where(q > reject_threshold, 0.0, q)
        total += tl.maximum(p - q, 0.0)
    target = coin * tl.sum(total, axis=0)

    # Inverse CDF: the first token whose running sum passes the target; the
    # last positive token when rounding leaves the target unreached.
    token = tl.full((), -1, tl.int32)
    last_positive = tl.full((), -1, tl.int32)
    running = tl.zeros((), tl.float32)
    start = tl.full((), 0, tl.int32)
    while (token < 0) & (start < vocab_size):
        mask = start + cols < vocab_size
        p = tl.load(p_row + start + cols, mask=mask, other=0.0)
        q = tl.load(q_row + start + cols, mask=start + cols < q_cols, other=0.0)
        q = tl.where(q > reject_threshold, 0.0, q)
        r = tl.maximum(p - q, 0.0)
        cum = running + tl.cumsum(r, axis=0)
        hit = (r > 0.0) & (cum > target)
        first_hit = tl.min(tl.where(hit, start + cols, 2147483647), axis=0)
        token = tl.where(first_hit < 2147483647, first_hit, token)
        last_positive = tl.maximum(
            last_positive, tl.max(tl.where(r > 0.0, start + cols, -1), axis=0)
        )
        running += tl.sum(r, axis=0)
        start += BLOCK
    token = tl.where(token < 0, last_positive, token)
    if token < 0:
        # p <= q everywhere only through rounding: take the target's mode.
        best = tl.full((), float("-inf"), tl.float32)
        best_id = tl.full((), 0, tl.int32)
        for start2 in range(0, vocab_size, BLOCK):
            mask = start2 + cols < vocab_size
            p = tl.load(p_row + start2 + cols, mask=mask, other=float("-inf"))
            block_best = tl.max(p, axis=0)
            block_id = tl.min(
                tl.where(p == block_best, start2 + cols, 2147483647), axis=0
            )
            better = block_best > best
            best_id = tl.where(better, block_id, best_id)
            best = tl.where(better, block_best, best)
        token = best_id
    tl.store(predicts_ptr + row0 + pos, token)


def chain_verify_draft_probs(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    candidates: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_length: torch.Tensor,
    *,
    reject_threshold: float,
) -> None:
    """Chain rejection sampling against recorded dense draft distributions.

    Request ``b`` accepts draft ``x_i = candidates[b, i]`` while ``coin *
    q(x_i) < p'(x_i)``, with ``p'`` row ``b * N + i - 1`` of ``target_probs``
    and ``q = draft_probs[pool, i - 1]``; a ``q`` entry above
    ``reject_threshold`` marks a slot without a recorded proposal and rejects.
    The token after the accepted prefix is drawn from ``norm(relu(p' - q))``
    at the rejected row (``p'`` alone for a sentinel row or after the last
    row). The same output law as the CUDA chain kernel's draft-prob rule.

    Args:
        target_probs: ``[bs * N, V]`` fp32 ``p'``, unit column stride.
        draft_probs: ``[pool, N, V]`` fp32 recorded draft distributions, unit
            column stride; step ``N - 1`` is never read.
        candidates: ``[bs, N]`` int32 chains, column 0 the verified token.
        req_pool_indices: ``[bs]`` int32 pool slot of each request.
        seed_pool: ``[pool]`` int64 seed.
        offsets_pool: ``[pool]`` int64 verify offset of each slot's row 0.
        predicts: ``[bs * N]`` int32 output: the accepted drafts, then the
            emitted token.
        accept_index: ``[bs, N]`` int32 output: ``b * N + j`` for the
            accepted positions ``j``; other entries are left as they are.
        accept_length: ``[bs]`` int32 output: accepted drafts (the emitted
            token excluded).
        reject_threshold: Sentinel bound of ``draft_probs``, at least 1.0.
    """
    bs, n = candidates.shape
    vocab_size = target_probs.shape[1]
    if target_probs.shape[0] != bs * n or target_probs.dtype != torch.float32:
        raise ValueError(
            f"chain_verify_draft_probs: target_probs must be fp32 [{bs * n}, V], "
            f"got {target_probs.dtype} {tuple(target_probs.shape)}"
        )
    if (
        draft_probs.dtype != torch.float32
        or draft_probs.ndim != 3
        or draft_probs.shape[1] != n
        or draft_probs.shape[2] != vocab_size
    ):
        raise ValueError(
            f"chain_verify_draft_probs: draft_probs must be fp32 [pool, {n}, "
            f"{vocab_size}], got {draft_probs.dtype} {tuple(draft_probs.shape)}"
        )
    if target_probs.stride(1) != 1 or draft_probs.stride(2) != 1:
        raise ValueError("chain_verify_draft_probs: vocab strides must be 1")
    if candidates.stride(1) != 1 or accept_index.stride(1) != 1:
        raise ValueError("chain_verify_draft_probs: column strides must be 1")
    for name, tensor in (
        ("candidates", candidates),
        ("req_pool_indices", req_pool_indices),
        ("predicts", predicts),
        ("accept_index", accept_index),
        ("accept_length", accept_length),
    ):
        if tensor.dtype != torch.int32:
            raise ValueError(f"chain_verify_draft_probs: {name} must be int32")
    if reject_threshold < 1.0:
        raise ValueError("chain_verify_draft_probs: reject_threshold must be >= 1")
    if bs == 0:
        return
    _chain_verify_draft_probs_kernel[(bs,)](
        target_probs,
        draft_probs,
        candidates,
        req_pool_indices,
        seed_pool,
        offsets_pool,
        predicts,
        accept_index,
        accept_length,
        float(reject_threshold),
        n,
        vocab_size,
        target_probs.stride(0),
        draft_probs.stride(0),
        draft_probs.stride(1),
        candidates.stride(0),
        accept_index.stride(0),
        COIN_SALT=COIN_OFFSET_SALT,
        BLOCK=_RESIDUAL_BLOCK,
        num_warps=8,
    )


@triton.jit(do_not_specialize=["num_nodes", "depths_row_stride", "coins_row_stride"])
def _tree_verify_coins_kernel(
    req_pool_indices_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    depths_ptr,
    coins_ptr,
    final_coins_ptr,
    num_nodes,
    depths_row_stride,
    coins_row_stride,
    COIN_SALT: tl.constexpr,
    N_PAD: tl.constexpr,
):
    b = tl.program_id(0)
    pool_idx = tl.load(req_pool_indices_ptr + b).to(tl.int64)
    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64)
    nodes = tl.arange(0, N_PAD)
    mask = nodes < num_nodes
    depth = tl.load(
        depths_ptr + b.to(tl.int64) * depths_row_stride + nodes, mask=mask, other=1
    )
    row = offset + tl.maximum(depth - 1, 0).to(tl.int64)
    coins = tl.rand(tl.randint(seed, COIN_SALT + row), nodes)
    tl.store(coins_ptr + b.to(tl.int64) * coins_row_stride + nodes, coins, mask=mask)
    final = tl.rand(tl.randint(seed, COIN_SALT + offset), num_nodes)
    tl.store(final_coins_ptr + b, final)


def tree_verify_coins(
    req_pool_indices: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    depths: torch.Tensor,
    coins: torch.Tensor,
    final_coins: torch.Tensor,
) -> None:
    """Position-keyed uniforms of a draft-tree verify (module docstring).

    Args:
        req_pool_indices: ``[bs]`` int32 pool slot of each request.
        seed_pool: ``[pool]`` int64 seed.
        offsets_pool: ``[pool]`` int64 verify offset of each slot's row 0.
        depths: ``[bs, N]`` int32 node depth, 0 for the root.
        coins: ``[bs, N]`` fp32 output, node ``i``'s accept uniform.
        final_coins: ``[bs]`` fp32 output, the bonus token's uniform.
    """
    bs, n = depths.shape
    if coins.shape[0] < bs or coins.shape[1] < n or coins.stride(1) != 1:
        raise ValueError(
            f"tree_verify_coins: coins {tuple(coins.shape)} cannot hold [{bs}, {n}]"
        )
    if depths.stride(1) != 1 or depths.dtype != torch.int32:
        raise ValueError("tree_verify_coins: depths must be int32, unit column stride")
    if coins.dtype != torch.float32 or final_coins.dtype != torch.float32:
        raise ValueError("tree_verify_coins: coins must be fp32")
    if bs == 0:
        return
    _tree_verify_coins_kernel[(bs,)](
        req_pool_indices,
        seed_pool,
        offsets_pool,
        depths,
        coins,
        final_coins,
        n,
        depths.stride(0),
        coins.stride(0),
        COIN_SALT=COIN_OFFSET_SALT,
        N_PAD=max(triton.next_power_of_2(n), 2),
    )
