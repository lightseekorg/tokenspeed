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

"""Lightweight output values shared by logits processing and sampling."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tokenspeed.runtime.sampling.logits_layout import LogitsLayoutPlan


@dataclasses.dataclass
class LogitsProcessorOutput:
    ## Part 1: This part will be assigned in python/tokenspeed/runtime/layers/logits_processor.py::LogitsProcessor
    # The logits of the next tokens.       shape: [#seq, vocab_size]
    next_token_logits: torch.Tensor
    # Used when ``do_argmax=True``.   shape: [#seq]
    next_token_ids: torch.Tensor | None = None
    # Used by speculative decoding.
    # The last hidden layers
    hidden_states: torch.Tensor | None = None
    logits_layout_plan: LogitsLayoutPlan | None = None

    ## Part 2: Sampling output metadata.
    # Selected-token logprobs written by the active SamplingBackend. shape: [#seq]
    next_token_logprobs: torch.Tensor | None = None
    # Probability mass before the selected token, ordered by descending logit
    # and then ascending token ID. Written by the common sampling diagnostic
    # boundary in the native Engine path. Optional, [#seq].
    next_token_cumulative_probs: torch.Tensor | None = None
    # The logprobs and ids of the top-k tokens in output positions. shape: [#seq, k]
    next_token_top_logprobs_val: list | None = None
    next_token_top_logprobs_idx: list | None = None
    # The logprobs and ids of the requested token ids in output positions. shape: [#seq, n] (n is the number of requested token ids)
    next_token_token_ids_logprobs_val: list | None = None
    next_token_token_ids_logprobs_idx: list | None = None
    # Eager prefills only (``ForwardContext.sampled_top_k``): the most likely
    # tokens of each prefill row's sampled position, fp32 log-softmax values
    # and int64 ids, set by the executor after sampling. shape: [#prefill rows, k]
    sampled_top_logprobs_val: torch.Tensor | None = None
    sampled_top_logprobs_idx: torch.Tensor | None = None

    ## Part 3: Prefill-only. This part will be assigned in python/tokenspeed/runtime/layers/logits_processor.py::LogitsProcessor
    # The logprobs of the prompt rows the forward's ``InputLogprobPlan``
    # named, fp32 in plan order, whole on every rank.  shape: [#rows]
    input_token_logprobs: torch.Tensor | None = None
    # With the plan's ``top_k``: each row's most likely next tokens, fp32
    # values and int64 ids, most likely first.  shape: [#rows, k]
    input_top_logprobs_val: torch.Tensor | None = None
    input_top_logprobs_idx: torch.Tensor | None = None
