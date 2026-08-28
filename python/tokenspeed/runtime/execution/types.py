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

"""Shared result and enum types for model execution."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from tokenspeed.runtime.grammar.capturable_grammar import (
        GrammarStepCompletion,
    )


@dataclass(frozen=True)
class DpForwardMetadata:
    """CPU-only DP metadata, gathered by the event loop before each forward.

    Every DP rank must enter the model's collectives with the same shapes,
    so the loop all-gathers these before any GPU work and hands the result
    to the executor as one unit.
    """

    global_num_tokens: list[int]
    global_batch_size: list[int]
    global_forward_mode: list[int]
    all_decode_or_idle: bool
    all_extend: bool
    need_idle_forward: bool


@dataclass(frozen=True)
class PlannedForward:
    """One round's planned work, as the device side needs to see it.

    Everything the data plane learns about a round arrives through here and
    is captured into the submitted closure, so every field must be something
    the control plane will not touch again once the round is dispatched (the
    capture contract in ``forward_thread.py``). A new field either satisfies
    that, or it is a new registered exception — not a comment.

    Attributes:
        forward_op: The scheduler's op for this round. A per-round value copy
            out of C++ (``rv_policy::copy`` on ``ExecutionPlan.forward``), so
            no later scheduler activity can reach it.
        sampling_params_list: Per-slot sampling params, gathered by the loop.
            Read-only for the request's lifetime.
        dp_metadata: CPU-gathered DP metadata, None outside DP. A frozen
            dataclass of plain lists.
        grammar_inputs: Per-batch grammar state, None when no request in the
            batch is constrained. A registered exception: these are the
            control plane's live matchers, see the contract's rule 5.
        multimodal_context: Per-batch multimodal state, None for text-only.
            Its ``mm_inputs`` are shallow copies taken at gather time; the
            items inside are the other registered exception.
    """

    forward_op: Any
    sampling_params_list: list
    dp_metadata: "DpForwardMetadata | None"
    grammar_inputs: Any
    multimodal_context: Any


@dataclass
class ModelExecutionResult:
    """
    Result of model execution returned to scheduler.

    This is the output from the Python executor back to the C++ scheduler.

    Attributes:
        output_tokens: Sampled token IDs
        output_lengths: Number of tokens generated per request (for spec decoding)
    """

    output_tokens: torch.Tensor
    copy_event: torch.cuda.Event | None = None
    output_lengths: torch.Tensor | None = None
    grammar_completion: GrammarStepCompletion | None = None
    # Per-position logprob of the sampled token, same layout as output_tokens.
    # Populated unconditionally by the sampling backend so it's always
    # available if any request asks for it.
    output_logprobs: torch.Tensor | None = None
    # P role, final chunk only: the sampled rows the commit path folds into
    # the ExtendResult as the bootstrap payload the peer's decode needs.
    next_input_ids: torch.Tensor | None = None
    # Per-request NaN-guard flags (int32, [bs]); None when the guard is disabled.
    output_nan_flags: torch.Tensor | None = None
    # Optional verify-input snapshot used by speculative diagnostics. Layout is
    # [batch, verify_width]: anchor followed by draft candidate token ids.
    spec_candidate_tokens: torch.Tensor | None = None
    _synced: bool = field(default=False, init=False, repr=False, compare=False)

    def sync(self) -> None:
        """Block until this result's D2H copy has landed.

        Called exactly once, by ``PendingExecution.result()`` — the single
        gate between the forward thread and the control plane. Consumers
        receive an already-synced result and must not call this again; a
        second call means a redundant sync path was reintroduced, so it
        raises instead of silently costing a no-op event join.
        """
        if self._synced:
            raise RuntimeError(
                "ModelExecutionResult is synced exactly once, by "
                "PendingExecution.result(); consumers get a synced result."
            )
        if self.copy_event is None:
            raise RuntimeError("copy_event is required before synchronizing results.")
        self.copy_event.synchronize()
        self._synced = True


@dataclass
class PendingExecution:
    """A forward submitted to the forward thread, awaiting its result.

    The control plane's ``in_flight`` queue holds these instead of raw
    ``ModelExecutionResult``s. ``result()`` joins the forward-thread future
    (all launches + D2H issued) and then syncs the copy event (D2H landed).
    Only commit blocks here — the dispatch path never waits, which is what
    keeps a backpressured stage's launches from stalling the control plane.

    Also the only place a ``ModelExecutionResult`` crosses back, which is
    what makes the sync exactly-once: the future is private and the result
    is memoized.
    """

    _future: Future
    _result: ModelExecutionResult | None = field(default=None, init=False, repr=False)

    def result(self) -> ModelExecutionResult:
        if self._result is None:
            results = self._future.result()
            results.sync()
            self._result = results
        return self._result
