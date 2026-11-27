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

"""Graph-capturable full-vocabulary head and tail log probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel.selection import KernelPlan, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _helper(mode: str, tensor: torch.Tensor):
    # CPU tensors are also used by the bit-pattern reference tests. Device
    # dispatch stays inside the kernel package, not in runtime call sites.
    return select_kernel(
        "sampling",
        mode,
        format_signature(values=dense_tensor_format(torch.float32)),
        solution="torch" if tensor.device.type == "cpu" else None,
    )


@dataclass(frozen=True)
class FullVocabScores:
    shifted: torch.Tensor
    log_z: torch.Tensor
    ids: torch.Tensor | None
    _head: torch.Tensor | None
    log_tail: torch.Tensor | None
    form: str
    vocab_block: int
    kernel_name: str
    source_logits: torch.Tensor | None = None
    pending_top_k: int = 0
    row_maximum: torch.Tensor | None = None

    @property
    def logprobs(self) -> torch.Tensor | None:
        """Materialize head scores on inspection; packing computes them in place."""
        if self._head is not None or self.ids is None:
            return self._head
        return _helper("full_vocab_head", self.shifted)(
            self.shifted, self.ids, self.log_z, form=self.form
        )

    def sampled_logprobs(
        self, tokens: torch.Tensor, *, prediction_rows: torch.Tensor | None
    ) -> torch.Tensor:
        """Gather scalar scores without copying full vocabulary rows for a tree."""
        target = tokens.long().clamp(0, self.shifted.shape[1] - 1)
        if prediction_rows is None:
            selected = self.shifted.gather(1, target[:, None]).squeeze(1)
            log_z = self.log_z
        else:
            selected = self.shifted[prediction_rows, target]
            log_z = self.log_z.index_select(0, prediction_rows)
        return selected - log_z if self.form == "direct" else -(log_z - selected)

    def rows(self, rows: slice) -> FullVocabScores:
        """View the same normalizer in one part of a mixed forward."""
        return FullVocabScores(
            self.shifted[rows],
            self.log_z[rows],
            None if self.ids is None else self.ids[rows],
            None if self._head is None else self._head[rows],
            None if self.log_tail is None else self.log_tail[rows],
            self.form,
            self.vocab_block,
            self.kernel_name,
            None if self.source_logits is None else self.source_logits[rows],
            self.pending_top_k,
            None if self.row_maximum is None else self.row_maximum[rows],
        )

    def complete(self, active_rows: torch.Tensor) -> FullVocabScores:
        """Score only GPU-selected source rows; keep the original row geometry.

        ``active_rows`` is a fixed-size device boolean mask. It never determines
        a host tensor size, and it is refreshed on every graph replay.
        """
        if self.ids is not None:
            return self
        if (
            self.source_logits is None
            or self.row_maximum is None
            or self.pending_top_k <= 0
        ):
            raise ValueError("pending full-vocabulary scores have no original logits")
        _, ids = _helper("full_vocab_topk", self.source_logits)(
            self.source_logits, k=self.pending_top_k, active_rows=active_rows
        )
        tail = _helper("full_vocab_tail", self.source_logits)(
            self.source_logits,
            ids,
            self.log_z,
            row_maximum=self.row_maximum,
            active_rows=active_rows,
        )
        return FullVocabScores(
            self.shifted,
            self.log_z,
            ids,
            None,
            tail,
            self.form,
            self.vocab_block,
            self.kernel_name,
        )

    def complete_accepted(
        self,
        accept_lengths: torch.Tensor,
        req_pool_indices: torch.Tensor,
        *,
        num_prefill_outputs: int,
        num_extends: int,
        decode_width: int,
        padding_index: int,
        prediction_rows: torch.Tensor | None,
    ) -> FullVocabScores:
        """Finish deferred rows after verification and stop trimming.

        A leaf that already completed its rows needs no acceptance-mask launch.
        """
        if self.ids is not None:
            return self
        active = full_vocab_active_rows(
            accept_lengths,
            req_pool_indices,
            num_prefill_outputs=num_prefill_outputs,
            num_extends=num_extends,
            decode_width=decode_width,
            padding_index=padding_index,
            prediction_rows=prediction_rows,
        )
        return self.complete(active)

    def pack(
        self, tokens: torch.Tensor, *, prediction_rows: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack target prediction rows in committed-token order.

        Args:
            tokens: Flat sampled token buffer, including unaccepted positions.
            prediction_rows: Tree path row indices, or None for chain order.

        Returns:
            Int32 rows [sampled ID, tail bits, K IDs, K logprob bits] and FP32
            sampled logprobs sharing the head's normalizer. Both are views of
            one [rows, 3 + 2*K] int32 allocation, whose private final word is the
            sampled logprob's bits. The caller removes unaccepted positions and
            graph padding before publishing rows.
        """
        if self.ids is None or self.log_tail is None:
            raise ValueError("complete full-vocabulary scores before packing")
        shifted, ids, head, tail = (
            self.shifted,
            self.ids,
            self._head,
            self.log_tail,
        )
        if tokens.ndim != 1 or tokens.numel() != shifted.shape[0]:
            raise ValueError(
                "sampled tokens must match the target prediction row layout"
            )
        if prediction_rows is not None and prediction_rows.shape != tokens.shape:
            raise ValueError(
                "accepted path must contain one prediction row per output slot"
            )
        kernel = _helper("full_vocab_pack", shifted)
        packed = kernel(
            shifted,
            self.log_z,
            ids,
            head,
            tail,
            tokens,
            prediction_rows=prediction_rows,
            form=self.form,
        )
        return packed[:, :-1], packed[:, -1].view(torch.float32)


class FullVocabTopK:
    """Select a fixed block-sum-exp plan once, then score original logits.

    The block width and selected solution must match the deployment's training
    logprob contract. Full vocabulary rows are required, including under TP.
    """

    def __init__(
        self, *, k: int, vocab_block: int, form: str, solution: str | None
    ) -> None:
        if not 1 <= k <= 128 or vocab_block <= 0:
            raise ValueError(
                "full-vocabulary top-k requires 1 <= k <= 128 and a positive block"
            )
        self.k = k
        self.vocab_block = vocab_block
        if form not in ("direct", "negative-cross-entropy"):
            raise ValueError(f"unknown full-vocabulary logprob form: {form!r}")
        self.form = form
        self.plan = KernelPlan.select(
            "sampling",
            "block_sumexp",
            format_signature(shifted=dense_tensor_format(torch.float32)),
            features=frozenset({"batch_invariant"}),
            static_traits={},
            variable_traits=frozenset(),
            declared_traits=frozenset(),
            solution=solution,
            ignore_layout=False,
        )

    def _validate_logits(self, logits: torch.Tensor) -> None:
        if logits.ndim != 2 or logits.shape[1] < self.k:
            raise ValueError(
                "full-vocabulary top-k requires [rows, vocab] logits with vocab >= k"
            )
        if logits.shape[1] % self.vocab_block:
            raise ValueError(
                "full vocabulary must be divisible by the logprob block width"
            )

    def _normalizer(self, logits: torch.Tensor, maximum: torch.Tensor):
        shifted = logits - maximum[:, None]
        return shifted, self._log_denominator(shifted)

    def _log_denominator(self, shifted: torch.Tensor) -> torch.Tensor:
        partials = self.plan.kernel(shifted, block_size=self.vocab_block)
        total = _helper("full_vocab_sum", partials)(partials)
        return total.log()

    def begin(self, logits: torch.Tensor, *, defer_head: bool) -> FullVocabScores:
        """Preserve original logits and normalize before sampling can mutate them.

        Leaves without GPU row selection finish immediately. A selected-row
        leaf retains the normalizer for ordinary sampled logprobs, and defers
        Top-K/tail until verification and stop trimming establish valid rows.
        """
        self._validate_logits(logits)
        if not defer_head or not _helper("full_vocab_row_selection_supports", logits)(
            logits
        ):
            return self.prepare(logits)
        logits = logits.float()
        maximum = logits.max(dim=-1).values
        source, shifted = _helper("full_vocab_source_shift", logits)(logits, maximum)
        log_z = self._log_denominator(shifted)
        return FullVocabScores(
            shifted,
            log_z,
            None,
            None,
            None,
            self.form,
            self.vocab_block,
            self.plan.kernel.name,
            source_logits=source,
            pending_top_k=self.k,
            row_maximum=maximum,
        )

    def prepare(self, logits: torch.Tensor) -> FullVocabScores:
        """Score every original row, reusing Top-K's maximum when exact.

        The selected provider's block tree, ordered FP32 fold and Torch log
        remain unchanged. Tail leaves preserve the original Torch reduction.
        """
        self._validate_logits(logits)
        logits = logits.float()
        values, ids = _helper("full_vocab_topk", logits)(
            logits, k=self.k, active_rows=None
        )
        maximum = _helper("full_vocab_maximum", logits)(logits, values)
        shifted, log_z = self._normalizer(logits, maximum)
        tail = _helper("full_vocab_tail", logits)(
            logits, ids, log_z, row_maximum=maximum, active_rows=None
        )
        return FullVocabScores(
            shifted,
            log_z,
            ids,
            None,
            tail,
            self.form,
            self.vocab_block,
            self.plan.kernel.name,
        )


def full_vocab_active_rows(
    accept_lengths: torch.Tensor,
    req_pool_indices: torch.Tensor,
    *,
    num_prefill_outputs: int,
    num_extends: int,
    decode_width: int,
    padding_index: int,
    prediction_rows: torch.Tensor | None,
) -> torch.Tensor:
    """Map accepted output positions to source-logit rows without a host read.

    Completing prefills form a request prefix. Decode positions use their
    fixed width and optional tree path. Padding slots never select a row.
    """
    return _helper("full_vocab_active_rows", accept_lengths)(
        accept_lengths,
        req_pool_indices,
        num_prefill_outputs=num_prefill_outputs,
        num_extends=num_extends,
        decode_width=decode_width,
        padding_index=padding_index,
        prediction_rows=prediction_rows,
    )
