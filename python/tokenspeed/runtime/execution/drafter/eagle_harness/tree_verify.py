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
from tokenspeed_kernel.ops.sampling.cuda import (
    fused_topk_topp_available as _FUSED_TOPK_TOPP_AVAILABLE,
)
from tokenspeed_kernel.ops.sampling.cuda import (
    fused_topk_topp_renorm,
)
from tokenspeed_kernel.ops.sampling.flashinfer import (
    softmax,
    top_k_renorm_prob,
    top_p_renorm_prob,
)
from tokenspeed_kernel.ops.sampling.triton import gather_and_expand_scalars
from tokenspeed_kernel.thirdparty.cuda.eagle_harness import (
    eagle_harness_residual_sample,
    eagle_harness_tree_reject_sample,
)

from tokenspeed.runtime.sampling.backends.base import SpeculativeSamplingPools
from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ, TreeVerifyBatch

__all__ = ["TreeRejectionSampler"]


class TreeRejectionSampler:
    """Accepts each request's draft tree by rejection sampling against the
    target p' (the softmax at the request's temperature, cut to its top-k and
    top-p and renormalized; a row that cuts nothing keeps its softmax values).
    Children drawn from the draft q (``tree.draft_q``) are accepted by
    recursive rejection without replacement, the others by their p' over the
    mass left; the bonus token comes from the last accepted node's residual.
    Exact for any draft.

    A greedy row (top-k 1) has a one-hot p' at the logits' lowest-index argmax
    (``sampling_argmax``): its first child holding the argmax is accepted and
    its bonus is the argmax, so it skips the residual scan and, with the fused
    top-k / top-p kernel, the softmax and the cut.
    """

    def __init__(
        self, pools: SpeculativeSamplingPools, max_bs: int, device: torch.device | str
    ) -> None:
        # The verifier's per-request sampling scalars, indexed by pool slot.
        self.pools = pools
        # Each request's last accepted node's row of the target probs.
        self.last_row = torch.zeros((max_bs,), dtype=torch.int64, device=device)
        # Stands in for the draft q of trees without sampled nodes (never read).
        self.no_draft_q = DraftTreeQ.zeros(1, 1, 1, 1, device)

    def __call__(
        self,
        logits: torch.Tensor,
        candidates: torch.Tensor,
        tree: TreeVerifyBatch,
        req_pool_indices: torch.Tensor,
        *,
        coins: torch.Tensor,
        final_coins: torch.Tensor,
        path: torch.Tensor,
        predict: torch.Tensor,
        accept_length: torch.Tensor,
    ) -> None:
        """Verify ``candidates`` (``[bs, N]`` tree tokens) against ``logits``
        (``[bs * N, V]``) with one uniform per child in ``coins`` (``[bs, N]``)
        and per bonus in ``final_coins`` (``[bs]``); writes ``path``
        (``[bs, N]``), ``predict`` (``[bs * N]``) and ``accept_length``
        (``[bs]``) as ``SamplingBackend.verify`` returns them."""
        bs, n = candidates.shape
        last_row = self.last_row[:bs]
        temperatures, top_ks, top_ps, _, _, _ = gather_and_expand_scalars(
            req_pool_indices,
            temperature=self.pools.temperature,
            top_k=self.pools.top_k,
            top_p=self.pools.top_p,
            n=n,
        )
        # The unfused cut reads every row, so only the fused one lets greedy rows skip.
        probs = softmax(
            logits,
            temperature=temperatures,
            skip_greedy_top_ks=top_ks if _FUSED_TOPK_TOPP_AVAILABLE else None,
        )
        if _FUSED_TOPK_TOPP_AVAILABLE:
            probs = fused_topk_topp_renorm(probs, top_ks, top_ps, in_place_skip=True)
        else:
            cut = top_k_renorm_prob(probs, top_ks)
            cut = top_p_renorm_prob(cut, top_ps, is_deterministic=True)
            uncut = (top_ks >= probs.shape[1]) & (top_ps >= 1.0)
            probs = torch.where(uncut[:, None], probs, cut)
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
