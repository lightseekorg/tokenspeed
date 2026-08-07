# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Tensor-parallel DSpark Markov and confidence heads."""

from __future__ import annotations

import torch
from torch import nn

from tokenspeed.runtime.distributed.comm_ops import all_gather_into_tensor


def _local_vocab_argmax(
    local_logits: torch.Tensor,
    lm_head: nn.Module,
    tp_group,
    gathered_values: torch.Tensor,
    gathered_ids: torch.Tensor,
) -> torch.Tensor:
    """Return global argmax IDs for vocab-sharded logits."""

    if (
        gathered_values.ndim != 2
        or gathered_ids.ndim != 2
        or gathered_values.shape != gathered_ids.shape
        or gathered_values.shape[1] < local_logits.shape[0]
    ):
        raise ValueError(
            "DSpark TP gather workspaces must be matching [tp_size, capacity] "
            "tensors with capacity for every active row."
        )
    if not gathered_values.is_contiguous() or not gathered_ids.is_contiguous():
        raise ValueError("DSpark TP gather workspaces must be contiguous.")

    shard = lm_head.shard_indices
    num_org = int(shard.num_org_elements)
    num_org_padded = int(shard.num_org_elements_padded)
    num_added = int(shard.num_added_elements)
    org_vocab_start = int(shard.org_vocab_start_index)
    added_vocab_start = int(shard.added_vocab_start_index)
    rows = local_logits.shape[0]

    if num_org > 0:
        local_max, local_arg = torch.max(local_logits[:, :num_org], dim=-1)
    else:
        local_max = torch.full(
            (rows,),
            torch.finfo(local_logits.dtype).min,
            dtype=local_logits.dtype,
            device=local_logits.device,
        )
        local_arg = torch.zeros(
            (rows,),
            dtype=torch.int64,
            device=local_logits.device,
        )

    if num_added > 0:
        added_logits = local_logits[
            :,
            num_org_padded : num_org_padded + num_added,
        ]
        added_max, added_arg = torch.max(added_logits, dim=-1)
        use_added = added_max > local_max
        local_max = torch.where(use_added, added_max, local_max)
        local_arg = torch.where(
            use_added,
            added_arg.to(local_arg.dtype) + num_org_padded,
            local_arg,
        )

    if num_added == 0:
        global_ids = local_arg + org_vocab_start
    else:
        global_ids = torch.empty_like(local_arg)
        is_base = local_arg < num_org
        global_ids[is_base] = org_vocab_start + local_arg[is_base]
        global_ids[~is_base] = added_vocab_start + (
            local_arg[~is_base] - num_org_padded
        )

    tp_size = gathered_values.shape[0]
    if tp_size == 1:
        return global_ids.to(torch.int32)

    flat_values = gathered_values.reshape(-1)[: tp_size * rows]
    flat_ids = gathered_ids.reshape(-1)[: tp_size * rows]
    all_gather_into_tensor(
        flat_values,
        local_max.contiguous(),
        tp_group,
    )
    all_gather_into_tensor(
        flat_ids,
        global_ids.contiguous(),
        tp_group,
    )
    values = flat_values.view(tp_size, rows)
    ids = flat_ids.view(tp_size, rows)
    best_rank = torch.argmax(values, dim=0).unsqueeze(0)
    return torch.gather(ids, 0, best_rank).squeeze(0).to(torch.int32)


class DSparkVanillaMarkov(nn.Module):
    """Low-rank token-bigram correction over a vocab-sharded output."""

    def __init__(self, embedding: nn.Module, projection: nn.Module) -> None:
        super().__init__()
        self.embedding = embedding
        self.projection = projection

    def local_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids.long())
        return torch.matmul(
            hidden.to(self.projection.weight.dtype),
            self.projection.weight.T,
        )


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


def sample_dspark_block_greedy(
    local_base_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    markov_head: DSparkVanillaMarkov,
    lm_head: nn.Module,
    tp_group,
    gathered_values: torch.Tensor,
    gathered_ids: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Apply the trained Markov correction and greedily sample a fixed block."""

    previous = bonus_token_ids.to(torch.int32)
    for step in range(local_base_logits.shape[1]):
        corrected = local_base_logits[:, step] + markov_head.local_bias(previous)
        previous = _local_vocab_argmax(
            corrected,
            lm_head,
            tp_group,
            gathered_values,
            gathered_ids,
        )
        output[:, step] = previous
    return output
