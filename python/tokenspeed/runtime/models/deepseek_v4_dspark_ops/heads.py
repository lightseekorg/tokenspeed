# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Tensor-parallel DSpark Markov and confidence heads."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.sampling.cute_dsl import (
    DistArgmaxState,
)
from tokenspeed_kernel.ops.sampling.cute_dsl import _invoke_kernel as _cute_local_argmax
from tokenspeed_kernel.ops.sampling.cute_dsl import (
    distributed_argmax,
)
from torch import nn

from tokenspeed.runtime.distributed.comm_ops import all_gather_into_tensor


def _local_vocab_argmax(
    local_logits: torch.Tensor,
    lm_head: nn.Module,
    tp_group,
    gathered_values: torch.Tensor,
    gathered_ids: torch.Tensor,
    dist_argmax_state: DistArgmaxState | None,
    dist_argmax_max: torch.Tensor | None,
    dist_argmax_ids: torch.Tensor | None,
    local_cute_argmax: bool,
    local_argmax_max: torch.Tensor | None,
    local_argmax_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Return global argmax IDs for vocab-sharded logits."""
    shard = lm_head.shard_indices
    num_org = int(shard.num_org_elements)
    num_org_padded = int(shard.num_org_elements_padded)
    num_added = int(shard.num_added_elements)
    org_vocab_start = int(shard.org_vocab_start_index)
    added_vocab_start = int(shard.added_vocab_start_index)
    rows = local_logits.shape[0]

    if dist_argmax_state is not None:
        if (
            num_added != 0
            or num_org_padded != num_org
            or local_logits.shape[1] != num_org
        ):
            raise ValueError(
                "DSpark distributed argmax requires an unpadded base-only "
                "local vocabulary shard."
            )
        if dist_argmax_max is None or dist_argmax_ids is None:
            raise ValueError(
                "DSpark distributed argmax requires preallocated max/id scratch."
            )
        max_out = dist_argmax_max[:rows]
        ids_out = dist_argmax_ids[:rows]
        _, global_ids = distributed_argmax(
            dist_argmax_state,
            local_logits.contiguous(),
            out_max=max_out,
            out_idx=ids_out,
        )
        return global_ids.to(torch.int32)

    if local_cute_argmax:
        if dist_argmax_state is not None:
            raise ValueError(
                "DSpark local and distributed CuTe argmax paths are mutually exclusive."
            )
        if (
            num_added != 0
            or num_org_padded != num_org
            or local_logits.shape[1] != num_org
        ):
            raise ValueError(
                "DSpark local CuTe argmax requires an unpadded base-only local "
                "vocabulary shard."
            )
        if local_argmax_max is None or local_argmax_ids is None:
            raise ValueError(
                "DSpark local CuTe argmax requires preallocated max/id scratch."
            )
        if local_argmax_max.dtype != torch.float32:
            raise ValueError("DSpark local CuTe argmax max scratch must be float32.")
        if local_argmax_ids.dtype != torch.int64:
            raise ValueError("DSpark local CuTe argmax ID scratch must be int64.")
        if not local_logits.is_contiguous():
            raise ValueError("DSpark local CuTe argmax logits must be contiguous.")
        local_max = local_argmax_max[:rows]
        local_arg = local_argmax_ids[:rows]
        _cute_local_argmax(local_logits, local_max, local_arg)
    elif num_org > 0:
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

    tp_size = int(getattr(lm_head, "tp_size", gathered_values.shape[0]))
    if tp_size == 1:
        return global_ids.to(torch.int32)

    if (
        gathered_values.ndim != 2
        or gathered_ids.ndim != 2
        or gathered_values.shape != gathered_ids.shape
        or gathered_values.shape[0] != tp_size
        or gathered_values.shape[1] < local_logits.shape[0]
    ):
        raise ValueError(
            "DSpark TP gather workspaces must be matching [tp_size, capacity] "
            "tensors with capacity for every active row."
        )
    if not gathered_values.is_contiguous() or not gathered_ids.is_contiguous():
        raise ValueError("DSpark TP gather workspaces must be contiguous.")

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
    dist_argmax_state: DistArgmaxState | None,
    dist_argmax_max: torch.Tensor | None,
    dist_argmax_ids: torch.Tensor | None,
    local_cute_argmax: bool,
    local_argmax_max: torch.Tensor | None,
    local_argmax_ids: torch.Tensor | None,
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
            dist_argmax_state,
            dist_argmax_max,
            dist_argmax_ids,
            local_cute_argmax,
            None if local_argmax_max is None else local_argmax_max[step],
            None if local_argmax_ids is None else local_argmax_ids[step],
        )
        output[:, step] = previous
    return output
