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

"""Dynamic draft trees (EAGLE-2 style) as fixed-shape device tensors.

Drafting keeps ``K`` lanes. Step 0 seeds them with the draft's top-``K`` at
the accepted frontier; every later step expands each lane to ``K`` children,
scores them by cumulative log-probability and keeps the best ``K`` of the
``K * K`` candidates as the next lanes. Every scored candidate is recorded,
``K + (S - 1) K^2`` entries in all. The tree keeps the best ``N - 1`` of them
under the root, numbered depth first with each node's best child first, so
the most likely path is nodes ``0, 1, ..`` and a one-wide tree is a chain.

Node tensors are ``[bs, N]``: ``tokens`` (node 0 the root), ``parent`` (``-1``
for the root), ``depth`` and a 64-bit ``mask`` whose bit ``j`` marks node
``j`` as an ancestor of, or equal to, the row's node.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.sampling.triton import (
    draft_tree_expand,
    draft_tree_finalize,
)

__all__ = ["DraftTree"]

# Bits per depth level in the depth-first sort key (sibling rank + 1 < 64).
_RANK_BITS = 6


class DraftTree:
    """Candidate record and lane state of one drafting round."""

    def __init__(
        self,
        max_bs: int,
        topk: int,
        num_steps: int,
        num_nodes: int,
        device: torch.device,
    ) -> None:
        if num_steps * _RANK_BITS > 62 or topk + 1 >= 1 << _RANK_BITS:
            raise ValueError(
                f"draft tree of depth {num_steps} and topk {topk} is too large"
            )
        if num_nodes - 1 > topk + (num_steps - 1) * topk * topk:
            raise ValueError(
                f"{num_nodes - 1} draft nodes exceed the {topk + (num_steps - 1) * topk * topk} "
                f"candidates of topk={topk} over {num_steps} steps"
            )
        self.topk = topk
        self.num_steps = num_steps
        self.num_nodes = num_nodes
        num_entries = topk + (num_steps - 1) * topk * topk
        self.entry_tokens = torch.zeros(
            (max_bs, num_entries), dtype=torch.int64, device=device
        )
        self.entry_scores = torch.zeros(
            (max_bs, num_entries), dtype=torch.float32, device=device
        )
        self.entry_parent = torch.full(
            (max_bs, num_entries), -1, dtype=torch.int64, device=device
        )
        self.entry_depth = torch.zeros(
            (max_bs, num_entries), dtype=torch.int64, device=device
        )
        self.lane_entry = torch.zeros((max_bs, topk), dtype=torch.int64, device=device)
        self.lane_scores = torch.zeros(
            (max_bs, topk), dtype=torch.float32, device=device
        )

    def seed(self, bs: int, scores: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Step 0: the drafter's best K candidates at the frontier become the first lanes.

        Args:
            scores: ``[bs, K]`` float32 log-probabilities, best first.
            tokens: ``[bs, K]`` int64 candidate tokens.

        Returns:
            ``[bs, K]`` int64 tokens the lanes forward next.
        """
        k = self.topk
        self.entry_tokens[:bs, :k] = tokens
        self.entry_scores[:bs, :k] = scores
        self.entry_parent[:bs, :k] = -1
        self.entry_depth[:bs, :k] = 1
        self.lane_entry[:bs] = torch.arange(k, device=tokens.device)
        self.lane_scores[:bs] = scores
        return tokens

    def expand(
        self,
        bs: int,
        step: int,
        scores: torch.Tensor,
        tokens: torch.Tensor,
        next_lanes: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Steps ``1 .. S - 1``: expand every lane, keep the best K children.

        Args:
            step: drafting step, from 1.
            scores: ``[bs * K, K]`` float32 log-probabilities of each lane's
                best K children, best first (at most 0: a child never scores
                above its parent).
            tokens: ``[bs * K, K]`` int64 child tokens.
            next_lanes: ``(lane_mask, hidden_src, hidden_dst)`` handed to
                ``draft_tree_expand`` to prepare the next step, or ``None``
                after the last step.

        Returns:
            ``[bs, K]`` int64 tokens the lanes forward next.
        """
        k = self.topk
        lane_tokens = draft_tree_expand(
            scores.view(bs, k * k),
            tokens.view(bs, k * k),
            self.lane_scores[:bs],
            self.lane_entry[:bs],
            self.entry_scores[:bs],
            self.entry_parent[:bs],
            self.entry_depth[:bs],
            self.entry_tokens[:bs],
            start=k + (step - 1) * k * k,
            depth=step + 1,
            next_lanes=next_lanes,
        )
        return lane_tokens

    def finalize(
        self, bs: int, root_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep the best ``N - 1`` candidates under the root, depth first.

        Args:
            root_tokens: ``[bs]`` last verified token of each request.

        Returns:
            ``(tokens, parent)``: ``[bs, N]`` int32 node tokens and parents.
        """
        return draft_tree_finalize(
            self.entry_scores[:bs],
            self.entry_parent[:bs],
            self.entry_depth[:bs],
            self.entry_tokens[:bs],
            root_tokens,
            num_nodes=self.num_nodes,
            max_depth=self.num_steps,
            rank_bits=_RANK_BITS,
        )
