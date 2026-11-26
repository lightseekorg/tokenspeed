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
from tokenspeed_kernel.selection import KernelPlan
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@dataclass(frozen=True)
class FullVocabScores:
    shifted: torch.Tensor
    log_z: torch.Tensor
    ids: torch.Tensor
    logprobs: torch.Tensor
    log_tail: torch.Tensor
    form: str

    def pack(
        self, tokens: torch.Tensor, *, prediction_rows: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack target prediction rows in committed-token order.

        Args:
            tokens: Flat sampled token buffer, including unaccepted positions.
            prediction_rows: Tree path row indices, or None for chain order.

        Returns:
            Int32 rows [sampled ID, tail bits, K IDs, K logprob bits] and FP32
            sampled logprobs sharing the head's normalizer. The caller removes
            unaccepted positions and graph padding before publishing rows.
        """
        shifted, log_z, ids, head, tail = (
            self.shifted,
            self.log_z,
            self.ids,
            self.logprobs,
            self.log_tail,
        )
        if tokens.ndim != 1 or tokens.numel() != shifted.shape[0]:
            raise ValueError(
                "sampled tokens must match the target prediction row layout"
            )
        if prediction_rows is not None:
            if prediction_rows.shape != tokens.shape:
                raise ValueError(
                    "accepted path must contain one prediction row per output slot"
                )
            shifted, log_z, ids, head, tail = (
                value.index_select(0, prediction_rows)
                for value in (shifted, log_z, ids, head, tail)
            )
        # Unaccepted slots can contain sentinels. Keep their original ID in
        # the packed row; clamp only the gather address to avoid a device fault.
        target = tokens.long().clamp(0, shifted.shape[1] - 1)
        selected = shifted.gather(1, target[:, None]).squeeze(1)
        sampled = selected - log_z if self.form == "direct" else -(log_z - selected)
        packed = torch.cat(
            (
                tokens.to(torch.int32)[:, None],
                tail.contiguous().view(torch.int32)[:, None],
                ids.to(torch.int32),
                head.contiguous().view(torch.int32),
            ),
            dim=1,
        )
        return packed, sampled


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

    def prepare(self, logits: torch.Tensor) -> FullVocabScores:
        """Compute a head and stable tail from unmodified [rows, vocab] logits.

        Returns device tensors using an ordered FP32 fold across vocabulary
        blocks. No host synchronization or request-dependent specialization.
        Tail roundoff is validated and clamped after accepted rows reach the CPU.
        """
        if logits.ndim != 2 or logits.shape[1] < self.k:
            raise ValueError(
                "full-vocabulary top-k requires [rows, vocab] logits with vocab >= k"
            )
        if logits.shape[1] % self.vocab_block:
            raise ValueError(
                "full vocabulary must be divisible by the logprob block width"
            )
        logits = logits.float()
        ids = torch.topk(logits, self.k, dim=-1).indices
        row_max = logits.max(dim=-1).values
        shifted = logits - row_max[:, None]
        partials = self.plan.kernel(shifted, block_size=self.vocab_block)
        total = partials[:, 0]
        for block in range(1, partials.shape[1]):
            total = total + partials[:, block]
        log_z = total.log()
        selected = shifted.gather(1, ids)
        # Preserve the trainer's subtraction order, including the sign of zero.
        head = (
            selected - log_z[:, None]
            if self.form == "direct"
            else -(log_z[:, None] - selected)
        )
        tail_logits = logits.scatter(1, ids, -torch.inf)
        tail = torch.logsumexp(tail_logits, dim=-1) - (row_max + log_z)
        return FullVocabScores(shifted, log_z, ids, head, tail, self.form)
