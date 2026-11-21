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

"""Sampled rows' logits left as this rank's vocabulary shard."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch
from tokenspeed_kernel.ops.sampling import (
    vocab_shard_logprob_blocks,
    vocab_shard_logprob_layout,
    vocab_shard_logprobs,
    vocab_shard_topk,
    vocab_topk_merge,
)

from tokenspeed.runtime.distributed.comm_backend.registry import get_global_backend
from tokenspeed.runtime.distributed.comm_ops import all_gather_single


def gather_topk_pairs(
    staged: torch.Tensor,
    landing: torch.Tensor,
    tp_size: int,
    tp_group: tuple[int, ...],
) -> torch.Tensor:
    """Every rank's packed rows (the ``vocab_shard_topk`` pairs and what
    travels with them, or int64 argmax keys), as ``[R, rows, width]``.

    A column gather the communication backend prepared for this group (Iris
    on gfx950, the all-reduce workspace's lamport exchange on NVIDIA) moves
    the rows as raw 16-bit words and lands rank r's row in columns
    ``r * width`` onward of every row; otherwise they land rank-major in
    ``landing`` through ``all_gather_single``.

    Args:
        staged: This rank's contiguous fp32 or int64 ``[rows, width]`` rows.
        landing: Contiguous ``[tp_size * rows, width]`` destination of the
            fallback gather, of ``staged``'s dtype.
        tp_size: Ranks the vocabulary is sharded over.
        tp_group: Their global ranks.

    Returns:
        A ``[tp_size, rows, width]`` view of ``staged``'s dtype; with the prepared gather it
        borrows collective storage, which the caller must read before the
        group's next collective.
    """
    rows, width = staged.shape
    words = staged.view(torch.bfloat16)
    gather = get_global_backend().get_prepared_all_gather(tp_group, words.dtype)
    if gather is not None and gather.can_run(words):
        gathered = gather.run(words).view(staged.dtype)
        return gathered.as_strided(
            (tp_size, rows, width),
            (width, gathered.stride(0), 1),
            gathered.storage_offset(),
        )
    all_gather_single(landing, staged, tp_group)
    return landing.view(tp_size, rows, width)


@dataclasses.dataclass(frozen=True)
class SampledVocabShard:
    """This rank's vocabulary shard of the sampled rows' logits.

    A vocabulary-parallel LM head leaves each rank the logits of its own
    vocabulary columns. A sampler that reads only each row's best candidates
    selects them per shard and gathers just those (:meth:`top_k`), a few KB
    instead of the whole rows; any other reader gathers the whole rows
    (:meth:`full_logits`), exactly as the logits processor would have.

    Attributes:
        logits: ``[rows, shard]`` LM-head output, before the logit scale and
            the softcap.
        columns: The shard's real vocabulary columns, its first ones: fewer
            than ``shard`` where the vocabulary ends inside the padded head.
        vocab_start: Global token id of shard column 0.
        logit_scale: Multiplier the processor applies to the logits, or None.
        softcap: The processor's final logit softcapping, or None.
        softcap_form: How the processor computes it (``vocab_shard_topk``'s
            ``softcap_form``), None without one.
        tp_size: Ranks the vocabulary is sharded over.
        tp_group: Their global ranks.
        gather_full: The processor's own tail: whole-vocabulary rows,
            scaled and softcapped logits from shard logits.
    """

    logits: torch.Tensor
    columns: int
    vocab_start: int
    logit_scale: float | None
    softcap: float | None
    softcap_form: str | None
    tp_size: int
    tp_group: tuple[int, ...]
    gather_full: Callable[[torch.Tensor], torch.Tensor]

    def rows(self, index: slice | torch.Tensor) -> SampledVocabShard:
        """The shard of a subset of the rows."""
        return dataclasses.replace(self, logits=self.logits[index])

    def full_logits(self) -> torch.Tensor:
        """Whole rows, as the logits processor returns them without a shard.

        May scale ``logits`` in place (a scale without a softcap), so call it
        instead of :meth:`top_k`, not after it.
        """
        return self.gather_full(self.logits)

    def top_k(
        self,
        k: int,
        vocab_mask: torch.Tensor | None,
        staged: torch.Tensor,
        landing: torch.Tensor,
        ids_out: torch.Tensor,
        values_out: torch.Tensor,
        logprob_values: torch.Tensor | None,
    ) -> None:
        """Each row's ``k`` best logits over the whole vocabulary.

        The values are the scaled, softcapped (and masked) logits
        :meth:`full_logits` would hold, in descending order, lowest token id
        first among equal values: the same list a whole-row selection keeps.

        Args:
            k: Candidates per row.
            vocab_mask: int32 whole-vocabulary token bitmask (xgrammar
                layout, bit set when allowed) per row, or None.
            staged: Contiguous fp32 ``[rows, 2k + halo]`` scratch, ``halo``
                the :meth:`logprob_halo` columns with ``logprob_values``,
                else 0.
            landing: Contiguous fp32 ``[tp_size * rows, 2k + halo]`` scratch.
            ids_out: Contiguous int32 ``[rows, k]`` token ids.
            values_out: Contiguous fp32 ``[rows, k]`` values.
            logprob_values: Contiguous ``[rows, V]`` scratch of the logits'
                dtype for :meth:`token_logprobs`, or None: it receives the
                final logits of this shard's columns and of the next
                shard's first ``halo``.
        """
        width = 2 * k
        halo = None
        if logprob_values is not None:
            halo = staged[:, width:]
        vocab_shard_topk(
            self.logits[:, : self.columns],
            k,
            self.logit_scale,
            self.vocab_start,
            staged[:, :width],
            softcap=self.softcap,
            softcap_form=self.softcap_form,
            vocab_mask=vocab_mask,
            values_out=logprob_values,
            halo_out=halo,
        )
        gathered = gather_topk_pairs(staged, landing, self.tp_size, self.tp_group)
        # The values were capped before the selection; the merge only sorts.
        vocab_topk_merge(gathered[:, :, :width], k, None, ids_out, values_out)
        if logprob_values is not None:
            # The next shard's leading columns that this shard's last
            # denominator block reads.
            shard = self.logits.shape[1]
            nxt = self.vocab_start // shard + 1
            vocab = logprob_values.shape[1]
            reach = min(nxt * shard + halo.shape[1], vocab)
            if nxt < self.tp_size and reach > nxt * shard:
                logprob_values[:, nxt * shard : reach].copy_(
                    gathered[nxt, :, width : width + reach - nxt * shard]
                )

    def logprob_halo(self, vocab_size: int) -> int:
        """Columns of the next shard :meth:`top_k` gathers for
        :meth:`token_logprobs` over a ``vocab_size`` vocabulary."""
        return vocab_shard_logprob_layout(self.logits.shape[1], vocab_size)[0]

    def logprob_width(self, vocab_size: int) -> int:
        """Columns of :meth:`token_logprobs`'s ``staged`` rows."""
        return vocab_shard_logprob_layout(self.logits.shape[1], vocab_size)[1]

    def token_logprobs(
        self,
        logprob_values: torch.Tensor,
        row_max: torch.Tensor,
        rows: torch.Tensor,
        tokens: torch.Tensor,
        staged: torch.Tensor,
        landing: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Selected-token logprobs, bit for bit ``selected_token_logprobs``
        on the whole masked rows :meth:`full_logits` gives.

        Args:
            logprob_values: The :meth:`top_k` scratch of this step's rows.
            row_max: fp32 ``[S]`` best value of each row (``values_out[:,
                0]`` of :meth:`top_k`), any stride.
            rows: ``[M]`` row of each logprob (a tree's accepted path).
            tokens: ``[M]`` token each logprob scores.
            staged: Contiguous fp32 ``[M, logprob_width]`` scratch.
            landing: Contiguous fp32 ``[tp_size * M, logprob_width]``
                scratch.
            out: fp32 ``[>= M]`` logprobs.
        """
        shard = self.logits.shape[1]
        vocab_shard_logprob_blocks(
            logprob_values, rows, tokens, row_max, self.vocab_start, shard, staged
        )
        partials = gather_topk_pairs(staged, landing, self.tp_size, self.tp_group)
        vocab_shard_logprobs(
            partials, rows, tokens, row_max, shard, logprob_values.shape[1], out
        )
