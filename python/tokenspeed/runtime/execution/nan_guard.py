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

"""Per-request NaN containment for the model executor.

Models and kernels cannot be assumed 100% NaN-free. This guard records
*which* requests produced NaN logits (or an out-of-vocab token id),
sanitizes the logits in place, and ships a per-request flag tensor to the
CPU where the output processor terminates the flagged requests with
``ABORT_CODE.NumericalError``.

Design constraints (all hold by construction):

- **Graph-safe / no sync.** Fixed-shape device ops on a persistent flag
  buffer; flags OR in-graph and are zeroed once per step outside it, so
  multi-cycle decode graphs accumulate correctly.
- **Near-zero cost.** Detection is one fused ``amax`` reduction over the
  logits (NaN propagates through ``amax``; no ``[rows, vocab]`` mask) plus
  ops on ``[bs]``-sized vectors; sanitize is one ``nan_to_num_``.
- **Rank-consistent.** OOV flags derive from already-broadcast token ids;
  logits flags rely on the bit-identical-logits-per-rank assumption the
  conditional sampling broadcast already depends on.
- **Zero branching at call sites.** ``NanGuard.create`` returns a no-op
  singleton when disabled.

Limitation: with Batch-DP spec-verify sampling the logits arrive sharded
(``logits_layout_plan is not None``), so logits attribution is skipped for
those steps; sanitize still applies and the OOV backstop still covers the
gathered full-batch ids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_context import ForwardContext
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput

# Replacement for NaN / +-inf logits (fp32). +-1e30 leaves headroom so a
# later temperature division cannot overflow back to inf; matches SGLang.
_NEG_SANITIZED = -1e30
_POS_SANITIZED = 1e30


class NanGuard:
    """Tracks per-request numerical corruption across one executor step.

    Lifecycle per ``execute_forward_op``::

        guard.reset(bs)                    # outside the graph
        ... per forward cycle (in-graph):
            guard.audit_logits(logits_output, ctx)     # pre-sampling
            guard.merge_oov(tokens, ctx, vocab_size)   # pre-clamp
        flags = guard.flags_cpu            # with the output D2H batch
    """

    def __init__(self, max_bs: int, device: torch.device | str) -> None:
        self.flags = torch.zeros((max_bs,), dtype=torch.int32, device=device)
        self._bs = 0

    @classmethod
    def create(cls, enabled: bool, max_bs: int, device) -> NanGuard:
        return cls(max_bs, device) if enabled else _DISABLED

    def reset(self, bs: int) -> None:
        """Zero the flags and pin this step's batch size; call outside the graph."""
        self._bs = bs
        self.flags.zero_()

    def audit_logits(
        self, logits_output: LogitsProcessorOutput, ctx: ForwardContext
    ) -> None:
        """Flag requests with NaN logits, then sanitize the logits in place.

        Must run before sampling and before grammar vocab masks / logit_bias,
        so their legitimate ``-inf`` entries survive sanitize.
        """
        logits = logits_output.next_token_logits
        if logits_output.logits_layout_plan is None:
            self._or_per_request(torch.isnan(logits.amax(dim=-1)), ctx)
        torch.nan_to_num_(
            logits, nan=_NEG_SANITIZED, posinf=_POS_SANITIZED, neginf=_NEG_SANITIZED
        )

    def merge_oov(
        self, output_tokens: torch.Tensor, ctx: ForwardContext, vocab_size: int
    ) -> None:
        """Backstop: flag requests whose sampled ids fall outside [0, vocab).

        Catches corruption past the logits (sampler/verify kernel output) and
        covers DP-sharded steps. Must run before the in-vocab clamps — they
        erase the evidence. Token ids are already rank-synced here.
        """
        self._or_per_request((output_tokens < 0) | (output_tokens >= vocab_size), ctx)

    @property
    def flags_cpu(self) -> torch.Tensor | None:
        """Async D2H of this step's flags (order with the copy event)."""
        return self.flags[: self._bs].to("cpu", non_blocking=True)

    def _or_per_request(self, rows: torch.Tensor, ctx: ForwardContext) -> None:
        """OR a per-row bool vector into per-request flags.

        Row layout mirrors ``_run_sampling``: ``[num_extends]`` extend rows,
        then ``num_decodes * n`` decode/verify rows.
        """
        ne = ctx.num_extends
        nd = ctx.bs - ne
        if ne > 0:
            self.flags[:ne] |= rows[:ne].to(torch.int32)
        if nd > 0:
            n = (rows.shape[0] - ne) // nd
            self.flags[ne : ctx.bs] |= rows[ne:].view(nd, n).any(dim=-1).to(torch.int32)


class _DisabledNanGuard(NanGuard):
    """No-op stand-in so call sites need no enabled-checks."""

    def __init__(self) -> None:  # no buffer
        pass

    def reset(self, bs: int) -> None:
        pass

    def audit_logits(self, logits_output, ctx) -> None:
        pass

    def merge_oov(self, output_tokens, ctx, vocab_size) -> None:
        pass

    @property
    def flags_cpu(self) -> None:
        return None


_DISABLED = _DisabledNanGuard()
