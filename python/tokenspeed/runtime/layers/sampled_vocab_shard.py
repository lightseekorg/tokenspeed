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
from tokenspeed_kernel.ops.sampling import vocab_shard_topk, vocab_topk_merge

from tokenspeed.runtime.distributed.comm_backend.registry import get_global_backend
from tokenspeed.runtime.distributed.comm_ops import all_gather_single


def gather_topk_pairs(
    staged: torch.Tensor,
    landing: torch.Tensor,
    tp_size: int,
    tp_group: tuple[int, ...],
) -> torch.Tensor:
    """Every rank's packed ``vocab_shard_topk`` pairs, as ``[R, rows, 2k]``.

    A column gather the communication backend prepared for this group (Iris
    on gfx950) moves the fp32 pairs as raw 16-bit words and lands rank r's
    pairs in columns ``r * 2k`` onward of every row; otherwise they land
    rank-major in ``landing`` through ``all_gather_single``.

    Args:
        staged: This rank's contiguous fp32 ``[rows, 2k]`` pairs.
        landing: Contiguous fp32 ``[tp_size * rows, 2k]`` destination of the
            fallback gather.
        tp_size: Ranks the vocabulary is sharded over.
        tp_group: Their global ranks.

    Returns:
        An fp32 ``[tp_size, rows, 2k]`` view; with the prepared gather it
        borrows collective storage, which the caller must read before the
        group's next collective.
    """
    rows, width = staged.shape
    words = staged.view(torch.bfloat16)
    gather = get_global_backend().get_prepared_all_gather(tp_group, words.dtype)
    if gather is not None and gather.can_run(words):
        gathered = gather.run(words).view(torch.float32)
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
        vocab_start: Global token id of shard column 0.
        logit_scale: Multiplier the processor applies to the logits, or None.
        softcap: The processor's final logit softcapping, or None.
        tp_size: Ranks the vocabulary is sharded over.
        tp_group: Their global ranks.
        gather_full: The processor's own tail: whole-vocabulary rows,
            scaled and softcapped logits from shard logits.
    """

    logits: torch.Tensor
    vocab_start: int
    logit_scale: float | None
    softcap: float | None
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
    ) -> None:
        """Each row's ``k`` best logits over the whole vocabulary.

        The values are the scaled, softcapped (and masked) logits
        :meth:`full_logits` would hold, in descending order, lowest token id
        first among equal values: the same list a whole-row selection keeps.

        Args:
            k: Candidates per row.
            vocab_mask: int32 whole-vocabulary token bitmask (xgrammar
                layout, bit set when allowed) per row, or None.
            staged: Contiguous fp32 ``[rows, 2k]`` scratch.
            landing: Contiguous fp32 ``[tp_size * rows, 2k]`` scratch.
            ids_out: Contiguous int32 ``[rows, k]`` token ids.
            values_out: Contiguous fp32 ``[rows, k]`` values.
        """
        vocab_shard_topk(
            self.logits,
            k,
            self.logit_scale,
            self.vocab_start,
            staged,
            softcap=self.softcap,
            vocab_mask=vocab_mask,
        )
        pairs = gather_topk_pairs(staged, landing, self.tp_size, self.tp_group)
        # The values were capped before the selection; the merge only sorts.
        vocab_topk_merge(pairs, k, None, ids_out, values_out)
