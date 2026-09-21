# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Tensor-parallel DSpark Markov and confidence heads.

The Markov head keeps the checkpoint's BF16 dtypes: its bigram table is
replicated on every rank so a lookup is a plain gather, and its rank-``R``
projection is sharded over the vocabulary exactly like the LM head, so the
bias of a shard column lands on the rank that owns that column's base logit.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.sampling.triton import (
    dspark_block_candidate_tiles,
    dspark_block_greedy_resolve,
    dspark_block_greedy_step,
)
from torch import nn

from tokenspeed.runtime.distributed.comm_ops import all_gather_single
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)


class DSparkVanillaMarkov(nn.Module):
    """Low-rank token-bigram correction over a vocab-sharded output."""

    def __init__(
        self, embedding: VocabParallelEmbedding, projection: ParallelLMHead
    ) -> None:
        super().__init__()
        if embedding.tp_size != 1:
            raise ValueError("DSpark Markov bigram table must be replicated")
        if embedding.embedding_dim != projection.embedding_dim:
            raise ValueError("DSpark Markov table and projection ranks differ")
        self.embedding = embedding
        self.projection = projection


class DSparkConfidenceHead(nn.Module):
    """Per-position acceptance-confidence predictor.

    Week-0 keeps a fixed proposal width, but loads this checkpoint component so
    malformed DSpark heads fail during startup and future dynamic truncation can
    be added without changing the weight contract.
    """

    def __init__(self, projection: nn.Module) -> None:
        super().__init__()
        self.projection = projection

    def forward(
        self,
        hidden_states: torch.Tensor,
        previous_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [hidden_states, previous_embeddings.to(hidden_states.dtype)],
            dim=-1,
        )
        logits, _ = self.projection(features.float())
        return logits.squeeze(-1)


def dspark_greedy_workspace(
    tp_size: int, max_rows: int, local_vocab: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Allocate the candidate buffers ``sample_dspark_block_greedy`` gathers into.

    Args:
        tp_size: Ranks the LM head is sharded over.
        max_rows: Most rows one call will sample.
        local_vocab: Columns of this rank's head shard.
        device: Device of the draft.

    Returns:
        ``(candidates, partials)``: a ``[2, tp_size, max_rows, tiles]`` int64
        double buffer holding every rank's packed candidates -- consecutive
        steps alternate slots, so a step's programs never overwrite the
        candidates they are still resolving -- and, when the head spans
        several ranks, the ``[max_rows, tiles]`` buffer this rank scores into
        before the gather. A single rank scores straight into its slot, so
        ``partials`` is None.
    """
    tiles = dspark_block_candidate_tiles(local_vocab)
    candidates = torch.empty(
        (2, tp_size, max_rows, tiles), dtype=torch.int64, device=device
    )
    partials = (
        None
        if tp_size == 1
        else torch.empty((max_rows, tiles), dtype=torch.int64, device=device)
    )
    return candidates, partials


def sample_dspark_block_greedy(
    local_base_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    markov_head: DSparkVanillaMarkov,
    lm_head: ParallelLMHead,
    tp_group,
    candidates: torch.Tensor,
    partials: torch.Tensor | None,
    output: torch.Tensor,
) -> torch.Tensor:
    """Apply the trained Markov correction and greedily sample a fixed block.

    Args:
        local_base_logits: ``[rows, block, local_vocab]`` FP32 logits of this
            rank's head shard for every block position.
        bonus_token_ids: ``[rows]`` token each block continues from.
        markov_head: Bigram table and projection shard.
        lm_head: Head whose shard geometry the logits follow.
        tp_group: Tensor-parallel group of the head.
        candidates: Gathered-candidate double buffer from
            ``dspark_greedy_workspace``.
        partials: This rank's candidate buffer from
            ``dspark_greedy_workspace``; None on a single rank.
        output: ``[rows, block]`` int32 destination for the block tokens.

    Returns:
        ``output``, filled in place.
    """
    rows, block, local_vocab = local_base_logits.shape
    shard = lm_head.shard_indices
    if shard.num_added_elements != 0:
        raise ValueError("DSpark greedy sampling needs a head without added vocabulary")
    projection = markov_head.projection
    if (
        projection.shard_indices.org_vocab_start_index != shard.org_vocab_start_index
        or projection.shard_indices.num_org_elements != shard.num_org_elements
    ):
        raise ValueError("DSpark Markov projection is not sharded like the LM head")
    if candidates.ndim != 4 or candidates.shape[0] != 2:
        raise ValueError(
            "DSpark greedy candidates must be a [2, tp, rows, tiles] buffer"
        )
    _, tp_size, capacity, tiles = candidates.shape
    if capacity < rows or (partials is None) != (tp_size == 1):
        raise ValueError("DSpark greedy workspace does not cover the batch")
    if partials is not None and partials.shape != (capacity, tiles):
        raise ValueError("DSpark greedy workspace does not cover the batch")
    slots = [
        candidates[slot].view(-1)[: tp_size * rows * tiles].view(tp_size, rows, tiles)
        for slot in range(2)
    ]
    for step in range(block):
        previous, current = slots[(step - 1) % 2], slots[step % 2]
        scored = (
            current[0]
            if partials is None
            else partials.view(-1)[: rows * tiles].view(rows, tiles)
        )
        dspark_block_greedy_step(
            local_base_logits,
            step,
            bonus_token_ids,
            previous,
            markov_head.embedding.weight,
            projection.weight,
            shard.org_vocab_start_index,
            shard.num_org_elements,
            scored,
            output,
        )
        if partials is not None:
            all_gather_single(current.view(-1), scored.view(-1), tp_group)
    dspark_block_greedy_resolve(slots[(block - 1) % 2], output, block - 1)
    return output
