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

"""Draft-tree verify by rejection sampling, with the EAGLE-Harness kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.sampling import argmax as sampling_argmax
from tokenspeed_kernel.ops.sampling.eagle_harness import (
    eagle_harness_residual_sample,
    eagle_harness_tree_reject_sample,
)

from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ, TreeVerifyBatch

__all__ = ["TreeRejectionSampler"]


class TreeRejectionSampler:
    """Accepts each request's draft tree by rejection sampling against the
    target p' its sampling backend computes (the distribution the backend
    samples plain rows from). Children drawn from the draft q
    (``tree.draft_q``) are accepted by recursive rejection without
    replacement, the others by their p' over the mass left; the bonus token
    comes from the last accepted node's residual. Exact for any draft.

    A greedy row (top-k 1) has a one-hot p' at the logits' lowest-index argmax
    (``sampling_argmax``): its first child holding the argmax is accepted and
    its bonus is the argmax, so it skips the residual scan and its row of p'
    is never read.
    """

    def __init__(self, max_bs: int, device: torch.device | str) -> None:
        # Each request's last accepted node's row of the target probs.
        self.last_row = torch.zeros((max_bs,), dtype=torch.int64, device=device)
        # Stands in for the draft q of trees without sampled nodes (never read).
        self.no_draft_q = DraftTreeQ.zeros(1, 1, 1, 1, device)

    def __call__(
        self,
        probs: torch.Tensor,
        logits: torch.Tensor,
        candidates: torch.Tensor,
        tree: TreeVerifyBatch,
        req_pool_indices: torch.Tensor,
        top_ks: torch.Tensor,
        *,
        coins: torch.Tensor,
        final_coins: torch.Tensor,
        path: torch.Tensor,
        predict: torch.Tensor,
        accept_length: torch.Tensor,
    ) -> None:
        """Verify ``candidates`` (``[bs, N]`` tree tokens) against the target
        ``probs`` (``[bs * N, V]`` fp32 p') of ``logits`` (``[bs * N, V]``),
        with ``top_ks`` (``[bs * N]`` int32, 1 for greedy rows), one uniform
        per child in ``coins`` (``[bs, N]``) and per bonus in ``final_coins``
        (``[bs]``); writes ``path`` (``[bs, N]``), ``predict`` (``[bs * N]``)
        and ``accept_length`` (``[bs]``) as ``SamplingBackend.verify`` returns
        them."""
        bs, n = candidates.shape
        last_row = self.last_row[:bs]
        argmax = sampling_argmax(logits)
        draft_q = tree.draft_q
        q = draft_q if draft_q is not None else self.no_draft_q
        eagle_harness_tree_reject_sample(
            probs,
            candidates.to(torch.int32),
            tree.parents,
            req_pool_indices if draft_q is not None else None,
            q.slot,
            q.pos,
            q.q,
            q.q_ids,
            coins,
            top_ks,
            argmax,
            path,
            predict,
            accept_length,
            last_row,
        )
        eagle_harness_residual_sample(
            probs, last_row, final_coins, top_ks, argmax, accept_length, predict, n
        )
