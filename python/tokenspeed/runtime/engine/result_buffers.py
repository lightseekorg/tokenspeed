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

"""Host result buffers consumed synchronously by the Rust completion query."""

from __future__ import annotations

import numpy as np
import torch

from tokenspeed.runtime.execution.types import ModelExecutionResult


def result_buffer(tensor: torch.Tensor | None) -> np.ndarray | None:
    """Export landed CPU storage without constructing per-element Python objects.

    The view retains the tensor's storage. Its caller must finish copying before
    releasing the result's reuse lease; retaining a view alone does not prevent
    an executor from overwriting a reusable host buffer. No device copy is
    performed here. Shape, strides, and dtype are preserved for Rust to validate.
    """
    if tensor is None:
        return None
    if tensor.device.type != "cpu":
        raise ValueError("result buffers must already be on the CPU")
    # ndarray supports both buffer exports and the legacy sequence reader.
    return tensor.numpy()


def completed_result(
    results: ModelExecutionResult,
    *,
    serving_compiles: tuple[int, float] | None,
    grammar_terminated_at: list[int] | None,
) -> tuple:
    """Export a landed forward in the Rust host's positional result layout.

    The caller waits for the copy event before entering here. Large numeric
    columns retain CPU storage as NumPy views, readable both through the buffer
    protocol and by older hosts using Python sequences. Rust copies before the
    caller releases the result's ownership/reuse lease.
    """

    def top(values, ids):
        return (
            (result_buffer(values), result_buffer(ids)) if values is not None else None
        )

    logprobs = None
    if (
        results.input_token_logprobs is not None
        or results.input_top_logprobs_val is not None
        or results.output_top_logprobs_val is not None
    ):
        logprobs = (
            result_buffer(results.input_token_logprobs),
            top(results.input_top_logprobs_val, results.input_top_logprobs_idx),
            top(results.output_top_logprobs_val, results.output_top_logprobs_idx),
        )
    return (
        results.output_tokens.tolist(),
        results.output_lengths.tolist(),
        result_buffer(results.output_logprobs),
        (
            results.output_nan_flags.tolist()
            if results.output_nan_flags is not None
            else None
        ),
        (
            [list(row) for row in results.next_input_ids.tolist()]
            if results.next_input_ids is not None
            else None
        ),
        serving_compiles,
        grammar_terminated_at,
        logprobs,
        result_buffer(results.output_token_cumulative_probs),
    )
