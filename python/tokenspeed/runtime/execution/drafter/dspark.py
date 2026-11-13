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

Static Markov head and fixed verify window (verify-all). Greedy by default;
explicit block rejection enables stochastic proposals through the shared sampler.
No confidence head / SPS-STS / ragged verify.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.distributed.comm_ops import all_gather_single
from tokenspeed.runtime.execution.drafter._dspark_walk import (
    dspark_walk_finish,
    dspark_walk_round,
)
from tokenspeed.runtime.execution.drafter.dflash import DFlash
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.models.dspark import VanillaMarkov
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.nvtx import nvtx_range

logger = get_colorful_logger(__name__)


class DSpark(DFlash):
    """DFlash block drafter + a Markov head (semi-autoregressive proposal)."""

    sample_from_anchor = True
    spec_algorithm = "DSPARK"
    supports_speculative_sampling = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.markov_head = getattr(self.model, "markov_head", None)
        if self.markov_head is None:
            raise ValueError(
                "DSPARK requires the draft model to define a markov_head "
                "(use a DSparkDraftModel checkpoint with markov_rank > 0)."
            )
        self._walk_words: torch.Tensor | None = None
        self._walk_gathered: torch.Tensor | None = None

    def require_full_draft_logits(self) -> None:
        """DSpark samples the target's shared head, not a draft-owned processor."""
        self.logits_processor.require_full_vocab_logits()

    @nvtx_range("dspark_sample_block", color="purple")
    def _sample_block(
        self,
        draft_hidden: torch.Tensor,
        block_ids: torch.Tensor,
        next_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Semi-autoregressive proposal using the configured shared draft sampler."""
        next_tokens[:, 0] = block_ids[:, 0]
        # Only the bias follows the previous token; the projection does not.
        block_logits = (
            self._block_base_logits(draft_hidden)
            if self.draft_sampler is None
            else None
        )
        if block_logits is not None and self._fused_walk_fits(block_logits):
            self._fused_greedy_walk(block_logits, next_tokens)
            return next_tokens
        for k in range(1, self.spec_num_tokens):
            # The Markov head embeds the previous token, so it must be in range
            # before this step, not after the loop: the anchor comes from the
            # target's last output (garbage during warmup) and each proposal
            # from a vocab-parallel argmax that can lose every shard. An
            # out-of-range id here indexes past the embedding table.
            bias_fn = self._make_step_bias_fn(next_tokens[:, k - 1])
            if self.draft_sampler is not None:
                # Full logits are necessary for the recorded proposal law.
                # require_full_vocab gives this step private TP gather storage;
                # successive Markov steps cannot alias a shared comm buffer.
                metadata = LogitsMetadata(
                    forward_mode=ForwardMode.DECODE, query_shard=None
                )
                logits = self.logits_processor._get_logits(
                    draft_hidden[:, k - 1, :],
                    self.lm_head,
                    metadata,
                    embedding_bias=None,
                    plan=None,
                    require_full_vocab=True,
                )
                logits = logits + bias_fn(0, logits.shape[-1]).to(logits.dtype)
                next_tokens[:, k].copy_(self.draft_sampler.propose(logits, step=k - 1))
            elif block_logits is None:
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
        head = self.lm_head
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

    def _fused_walk_fits(self, block_logits: torch.Tensor) -> bool:
        """Whether the greedy walk can take one launch and one gather per round.

        Without a distributed argmax state the generic walk takes several
        launches and two all-gathers per round. A state, where the
        platform has one, keeps the walk it already fuses.

        Every test must come out the same on every rank: ranks taking
        different walks would issue different collectives.
        """
        markov = self.markov_head
        if not block_logits.is_cuda or not isinstance(markov, VanillaMarkov):
            return False
        # The probe is collective, so no rank-local test may come before it.
        if (
            self._ensure_dist_argmax_state(
                self.lm_head.weight.dtype, block_logits.device
            )
            is not None
        ):
            return False
        head = self.lm_head
        org = int(head.org_vocab_size)
        tp = int(self.logits_processor.tp_size)
        latent_table = markov.markov_w1.weight
        bias_weight = markov.markov_w2.weight
        return (
            # No added vocab and equal unpadded shards: no rank is left with
            # an added lane or an empty shard, and shard ids are global ids.
            int(head.num_embeddings) == org
            and int(head.shard_indices.num_org_elements) * tp == org
            and org <= int(bias_weight.shape[0])
            and block_logits.dtype == torch.bfloat16
            and latent_table.dtype == bias_weight.dtype == torch.bfloat16
            and latent_table.is_contiguous()
            and bias_weight.is_contiguous()
        )

    def _ensure_walk_buffers(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """The walk's per-round words and their gathers, at max capacity.

        Allocated once, in warmup, for the same reason as the generic walk's
        gather scratch: graphs captured at small batch sizes must not record
        storage that a later, larger capture frees.
        """
        rounds = self.spec_num_tokens - 1
        max_bs = self.input_buffers.max_bs
        tp = int(self.logits_processor.tp_size)
        if self._walk_words is None or self._walk_words.device != device:
            self._walk_words = torch.zeros(
                (rounds, max_bs), dtype=torch.int64, device=device
            )
            self._walk_gathered = torch.zeros(
                (rounds, tp * max_bs), dtype=torch.int64, device=device
            )
        return self._walk_words, self._walk_gathered

    def _fused_greedy_walk(
        self, block_logits: torch.Tensor, next_tokens: torch.Tensor
    ) -> None:
        """Fill ``next_tokens[:, 1:]`` from the anchor in column 0."""
        rows = int(next_tokens.shape[0])
        rounds = self.spec_num_tokens - 1
        tp = int(self.logits_processor.tp_size)
        start = int(self.lm_head.shard_indices.org_vocab_start_index)
        num_org = int(block_logits.shape[-1])
        latent_table = self.markov_head.markov_w1.weight
        bias_weight = self.markov_head.markov_w2.weight[start : start + num_org]
        words, gathered = self._ensure_walk_buffers(block_logits.device)
        words = words[:, :rows]
        words.zero_()
        previous = None
        for k in range(1, rounds + 1):
            word = words[k - 1]
            dspark_walk_round(
                block_logits[k - 1],
                latent_table,
                bias_weight,
                next_tokens,
                word,
                previous,
                column=k,
                vocab_start=start,
                tp=tp,
            )
            if tp == 1:
                previous = word
            else:
                previous = gathered[k - 1, : tp * rows]
                all_gather_single(previous, word, self.logits_processor.tp_group)
        dspark_walk_finish(next_tokens, previous, column=rounds, tp=tp)

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
