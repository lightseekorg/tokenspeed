# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Static DSpark drafter.

This is the minimal static configuration: greedy sampling, vanilla Markov head,
fixed verify window (verify-all). No confidence head / SPS-STS / ragged verify.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.execution.drafter.dflash import DFlash
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.nvtx import nvtx_range

logger = get_colorful_logger(__name__)


class DSpark(DFlash):
    """DFlash block drafter + a Markov head (semi-autoregressive proposal)."""

    sample_from_anchor = True
    spec_algorithm = "DSPARK"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.markov_head = getattr(self.model, "markov_head", None)
        if self.markov_head is None:
            raise ValueError(
                "DSPARK requires the draft model to define a markov_head "
                "(use a DSparkDraftModel checkpoint with markov_rank > 0)."
            )

    @nvtx_range("dspark_sample_block", color="purple")
    def _sample_block(
        self,
        draft_hidden: torch.Tensor,
        block_ids: torch.Tensor,
        next_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Semi-autoregressive greedy proposal over the block positions."""
        next_tokens[:, 0] = block_ids[:, 0]
        # Only the bias follows the previous token; the projection does not.
        block_logits = self._block_base_logits(draft_hidden)
        for k in range(1, self.spec_num_tokens):
            # The Markov head embeds the previous token, so it must be in range
            # before this step, not after the loop: the anchor comes from the
            # target's last output (garbage during warmup) and each proposal
            # from a vocab-parallel argmax that can lose every shard. An
            # out-of-range id here indexes past the embedding table.
            bias_fn = self._make_step_bias_fn(next_tokens[:, k - 1])
            if block_logits is None:
                self._greedy_argmax_vocab_parallel(
                    draft_hidden[:, k - 1, :],
                    out=next_tokens[:, k],
                    bias_fn=bias_fn,
                )
            else:
                self._greedy_argmax_vocab_parallel(
                    draft_hidden[:, k - 1, :],
                    out=next_tokens[:, k],
                    bias_fn=bias_fn,
                    base_logits=block_logits[k - 1],
                )
        next_tokens.clamp_(min=0)
        return next_tokens

    def _block_base_logits(self, draft_hidden: torch.Tensor):
        """Project every block position's hidden state in one GEMM.

        Returns ``[positions, rows, org_vocab_shard]``, or None when the head
        has no shard metadata and the caller must take its own path.
        """
        head = getattr(self, "lm_head", None)
        if head is None:
            return None
        if not hasattr(head, "weight") or not hasattr(head, "shard_indices"):
            return None
        num_org = int(head.shard_indices.num_org_elements)
        if num_org <= 0:
            return None
        positions = int(draft_hidden.shape[1])
        rows = int(draft_hidden.shape[0])
        weight = head.weight
        flat = draft_hidden.reshape(rows * positions, -1).to(weight.dtype)
        logits = torch.matmul(flat, weight[:num_org].T)
        # draft_hidden is [rows, positions, H]; the walk indexes by position.
        # The walk slices one position at a time; transpose alone would hand
        # each slice a positions-strided row layout, which the distributed
        # argmax's compact-row kernel must not see.
        return logits.view(rows, positions, num_org).transpose(0, 1).contiguous()

    def _make_step_bias_fn(self, prev_tokens: torch.Tensor):
        """Build the per-position additive-bias hook for the Markov head.

        Returns a closure ``bias_fn(vocab_start, count) -> [rows, count]`` that
        supplies the Markov correction for a requested global vocab slice.

        The Markov head only spans the original vocab (``markov_w2`` has
        ``vocab_size`` rows). Any part of a requested slice that falls beyond
        that range -- i.e. the target LM head's *added* vocab shard -- gets a
        zero bias, so added tokens keep their plain base logit and only compete
        unbiased.
        """
        w2_weight = self.markov_head.markov_w2.weight
        vocab_size = int(w2_weight.shape[0])
        latent = self.markov_head.get_prev_latent(prev_tokens.clamp(0, vocab_size - 1))
        rows = int(latent.shape[0])

        def bias_fn(vocab_start: int, count: int) -> torch.Tensor:
            end = vocab_start + count
            # Fast path: slice fully inside the Markov (org) vocab.
            if vocab_start >= 0 and end <= vocab_size:
                w2_slice = w2_weight[vocab_start:end]
                return torch.matmul(latent.to(w2_slice.dtype), w2_slice.T)
            # Partial / added-vocab slice: bias only the org intersection,
            # zero elsewhere.
            bias = latent.new_zeros((rows, count))
            lo = max(vocab_start, 0)
            hi = min(end, vocab_size)
            if hi > lo:
                w2_slice = w2_weight[lo:hi]
                real = torch.matmul(latent.to(w2_slice.dtype), w2_slice.T)
                bias[:, lo - vocab_start : hi - vocab_start] = real.to(bias.dtype)
            return bias

        return bias_fn
