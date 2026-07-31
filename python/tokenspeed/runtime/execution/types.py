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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tokenspeed.runtime.grammar.capturable_grammar import (
        GrammarStepCompletion,
    )


@dataclass
class LogprobDetails:
    """Additional prompt and distribution logprobs."""

    input_token_logprobs: torch.Tensor | None = None
    input_token_ids: torch.Tensor | None = None
    input_top_logprobs_val: list | None = None
    input_top_logprobs_idx: list | None = None
    input_token_ids_logprobs_val: list | None = None
    input_token_ids_logprobs_idx: list | None = None
    output_top_logprobs_val: torch.Tensor | None = None
    output_top_logprobs_idx: torch.Tensor | None = None
    output_token_ids_logprobs_val: torch.Tensor | None = None
    output_token_ids_logprobs_idx: torch.Tensor | None = None

    def to_cpu(self) -> LogprobDetails:
        """Enqueue non-blocking CPU copies for tensor-backed fields."""

        for field_name in (
            "input_token_logprobs",
            "input_token_ids",
            "output_top_logprobs_val",
            "output_top_logprobs_idx",
            "output_token_ids_logprobs_val",
            "output_token_ids_logprobs_idx",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, value.to("cpu", non_blocking=True))
        return self


@dataclass
class ModelExecutionResult:
    """
    Result of model execution returned to scheduler.

    This is the output from the Python executor back to the C++ scheduler.

    Attributes:
        output_tokens: Sampled token IDs
        output_logits: Output logits (if requested)
        output_lengths: Number of tokens generated per request (for spec decoding)
    """

    output_tokens: torch.Tensor
    copy_event: torch.cuda.Event | None = None
    output_logits: torch.Tensor | None = None
    output_lengths: torch.Tensor | None = None
    grammar_completion: GrammarStepCompletion | None = None
    # Per-position logprob of the sampled token, same layout as output_tokens.
    # Populated unconditionally by the sampling backend so it's always
    # available if any request asks for it.
    output_logprobs: torch.Tensor | None = None
    logprob_details: LogprobDetails | None = None
    # Optional next-round input rows captured for PD prefill data-plane handoff.
    next_input_ids: torch.Tensor | None = None
    # Per-request NaN-guard flags (int32, [bs]); None when the guard is disabled.
    output_nan_flags: torch.Tensor | None = None

    def sync(self) -> None:
        if self.copy_event is None:
            raise RuntimeError("copy_event is required before synchronizing results.")
        self.copy_event.synchronize()
