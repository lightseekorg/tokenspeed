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

"""Distributed sparse selection over sharded Index-K storage."""

import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_index_candidates

from tokenspeed.runtime.distributed.comm_ops import all_gather
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    resolve_cache_slots,
)


def merge_index_candidates(
    offsets: torch.Tensor,
    scores: torch.Tensor,
    *,
    topk: int,
    group: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge disjoint local candidates with one global Top-K selection.

    Communicates only K candidate positions and scores per query and rank.
    Returns global logical positions and valid counts, identical on every rank.
    Positive infinity marks mandatory candidates; negative infinity is padding.
    Equal-score candidates have no prescribed logical-position order.
    """
    if offsets.shape != scores.shape or offsets.ndim != 2 or topk <= 0:
        raise ValueError("Candidate offsets/scores must be matching matrices")
    if len(group) > 1:
        offsets = all_gather(offsets.contiguous(), group, dim=-1)
        scores = all_gather(scores.contiguous(), group, dim=-1)
    valid = (offsets >= 0) & ~torch.isnan(scores) & (scores > -float("inf"))
    scores = torch.where(valid, scores, -float("inf"))
    selected_scores, order = torch.topk(scores, topk, dim=-1)
    result = offsets.gather(1, order)
    valid = selected_scores > -float("inf")
    result = torch.where(valid, result, -1)
    return result, valid.sum(dim=-1, dtype=torch.int32)


def select_dsa_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_requests: torch.Tensor,
    causal_lens: torch.Tensor,
    *,
    placement: CachePlacement,
    page_size: int,
    topk: int,
    softmax_scale: float,
    initial_tokens: int,
    local_tokens: int,
    max_logits_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return request-relative global Top-K from local Index-K pages.

    Metadata has global positions throughout. Only cache addresses are local;
    global Top-K precedes the attention-side ownership mask and LSE reduction.
    Query tiling caps temporary logits for both prefill and decode.
    """
    if q.shape[0] == 0:
        return (
            torch.empty((0, topk), device=q.device, dtype=torch.int32),
            torch.empty(0, device=q.device, dtype=torch.int32),
        )
    slots, owned = resolve_cache_slots(page_table.long() * page_size, placement)
    local_table = torch.where(owned, slots // page_size, -1).to(torch.int32)
    columns = page_table.shape[1]
    tile = max(1, max_logits_bytes // (columns * page_size * 4))
    outputs, lengths = [], []
    for start in range(0, q.shape[0], tile):
        end = min(q.shape[0], start + tile)
        offsets, scores = dsa_index_candidates(
            q[start:end],
            weights[start:end],
            index_k_cache,
            local_table,
            query_requests[start:end],
            causal_lens[start:end],
            page_size=page_size,
            topk=topk,
            softmax_scale=softmax_scale,
            initial_tokens=initial_tokens,
            local_tokens=local_tokens,
            solution=None,
        )
        indices, lens = merge_index_candidates(
            offsets, scores, topk=topk, group=placement.group
        )
        outputs.append(indices)
        lengths.append(lens)
    return torch.cat(outputs), torch.cat(lengths)
