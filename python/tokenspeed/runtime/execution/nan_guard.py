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

Models and kernels cannot be assumed 100% NaN-free (driver bugs, numerical
edge cases, memory corruption). This guard keeps one corrupted request from
poisoning the rest of the batch: it records *which* requests produced NaN
logits (or an out-of-vocab token id, the symptom of a sampling kernel fed
non-finite input), sanitizes the logits in place, and ships a per-request
flag tensor to the CPU where the output processor terminates the flagged
requests with ``ABORT_CODE.NumericalError``.

Design constraints (all hold by construction):

- **Graph-safe / no sync.** Every device op is fixed-shape and the flag
  buffer is persistent, so the whole guard captures into CUDA graphs.
  Flags are OR-merged in-graph and zeroed once per executor step *outside*
  the graph, so multi-cycle decode graphs accumulate correctly.
- **Near-zero cost.** Detection is a single fused ``amax`` reduction over
  the logits (NaN propagates through ``amax``; no ``isnan`` intermediate
  the size of the logits is materialized) plus a handful of ops on
  ``[bs]``-sized vectors; sanitize is one elementwise ``nan_to_num_``.
- **Rank-consistent.** OOV flags derive from already-broadcast token ids;
  logits flags rely on the same bit-identical-logits-per-rank assumption
  the conditional sampling broadcast already depends on. All ranks
  therefore reach identical finish decisions — a rank-divergent finish
  would desync batch composition and wedge collectives.
- **Zero branching at call sites.** ``NanGuard.create`` returns a no-op
  singleton when disabled, so the executor's hot path has no ``if``\\ s.

Limitation: with Batch-DP spec-verify sampling the logits arrive
vocab/request-sharded per rank (``logits_layout_plan is not None``), so
per-request logits attribution is skipped for those steps; sanitize still
applies, and the OOV backstop still covers the gathered full-batch ids.

Legitimate NaN sources stay harmless: warmup/dummy batches read
uninitialized KV and may flag padding rows, but the CPU side only consults
flags for rids tracked in ``rid_to_state``, and the buffer is re-zeroed
every step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_context import ForwardContext
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput

# Replacement for NaN / -inf logits. Logits are fp32 here; +-1e30 (vs.
# finfo.max) leaves headroom so a later temperature division cannot
# overflow back to inf. Matches SGLang's sanitize constants.
_NEG_SANITIZED = -1e30
_POS_SANITIZED = 1e30


class NanGuard:
    """Tracks per-request numerical corruption across one executor step.

    Lifecycle per ``execute_forward_op``::

        guard.reset()                      # outside the graph
        ... per forward cycle (in-graph):
            guard.observe_logits(logits_output, ctx)   # pre-sampling
            guard.merge_oov(tokens, ctx, vocab_size)   # pre-clamp
        flags = guard.flags_to_cpu(bs)     # with the output D2H batch
    """

    def __init__(self, max_bs: int, device: torch.device | str) -> None:
        self.flags = torch.zeros((max_bs,), dtype=torch.int32, device=device)

    @classmethod
    def create(cls, enabled: bool, max_bs: int, device) -> NanGuard:
        return cls(max_bs, device) if enabled else _DISABLED

    def reset(self) -> None:
        """Zero the flags. Call once per executor step, outside the graph."""
        self.flags.zero_()

    def observe_logits(
        self, logits_output: LogitsProcessorOutput, ctx: ForwardContext
    ) -> None:
        """Flag requests with NaN logits, then sanitize the logits in place.

        Must run before grammar vocab masks / logit_bias are applied so
        their legitimate ``-inf`` entries survive sanitize, and before any
        sampling kernel sees the logits.
        """
        logits = logits_output.next_token_logits
        if logits_output.logits_layout_plan is None:
            # amax propagates NaN: one fused reduction gives per-row NaN
            # presence without materializing a [rows, vocab] mask.
            self._or_per_request(torch.isnan(logits.amax(dim=-1)), ctx)
        torch.nan_to_num_(
            logits, nan=_NEG_SANITIZED, posinf=_POS_SANITIZED, neginf=_NEG_SANITIZED
        )

    def merge_oov(
        self, output_tokens: torch.Tensor, ctx: ForwardContext, vocab_size: int
    ) -> None:
        """Backstop: flag requests whose sampled ids fall outside [0, vocab).

        Catches sampling-kernel misbehavior (the -1 argmax sentinel, or
        rejection-sampling UB on garbage input). Must run before the
        executor's in-vocab clamps — they erase the evidence. Token ids are
        already TP-rank-synced here, so the flags stay rank-consistent.
        """
        self._or_per_request((output_tokens < 0) | (output_tokens >= vocab_size), ctx)

    def flags_to_cpu(self, bs: int) -> torch.Tensor | None:
        """Async D2H of the first ``bs`` flags (order with the copy event)."""
        return self.flags[:bs].to("cpu", non_blocking=True)

    def _or_per_request(self, rows: torch.Tensor, ctx: ForwardContext) -> None:
        """OR a per-row bool vector into per-request flags.

        Row layout mirrors ``_run_sampling``: ``[num_extends]`` one row per
        prefill request, then ``num_decodes * n`` decode/verify rows.
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

    def reset(self) -> None:
        pass

    def observe_logits(self, logits_output, ctx) -> None:
        pass

    def merge_oov(self, output_tokens, ctx, vocab_size) -> None:
        pass

    def flags_to_cpu(self, bs: int) -> None:
        return None


_DISABLED = _DisabledNanGuard()
