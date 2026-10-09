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

"""Helpers for label scoring (the Score API readout).

A score request reads, at the answer boundary (the last prefill position
of a request), the model's distribution restricted to a caller-declared
label token set. These helpers build the padded gather index used on the
forward thread and finalize the per-request row into the response values.
See docs/design/scoring.md for the contract.
"""

from __future__ import annotations

import math

import torch

from tokenspeed.runtime.sampling.logprobs import gather_token_logprobs
from tokenspeed.runtime.sampling.sampling_params import SamplingParams


def build_score_label_ids(
    sampling_params_list: list[SamplingParams],
    num_rows: int,
    device,
) -> torch.Tensor | None:
    """Pad per-request score label ids into a ``[num_rows, max_labels]`` index.

    Covers the first ``num_rows`` entries of ``sampling_params_list`` (the
    extend rows; score requests never decode). Rows without score labels
    are zero-filled and produce gather output the consumer must ignore.
    Returns ``None`` when no row carries score labels.
    """
    label_lists = [sp.score_label_token_ids for sp in sampling_params_list[:num_rows]]
    if not any(label_lists):
        return None
    max_labels = max(len(ids) for ids in label_lists if ids is not None)
    target_device = torch.device(device)
    label_ids = torch.zeros(
        num_rows,
        max_labels,
        dtype=torch.int64,
        device="cpu",
        pin_memory=target_device.type == "cuda",
    )
    for row, ids in enumerate(label_lists):
        if ids is not None:
            label_ids[row, : len(ids)] = torch.tensor(
                ids, dtype=torch.int64, device="cpu"
            )
    return label_ids.to(target_device, non_blocking=True)


def gather_score_logprobs(
    next_token_logits: torch.Tensor,
    score_label_ids: torch.Tensor,
    num_prefill_outputs: int,
    *,
    logprob_order: str,
) -> torch.Tensor | None:
    """Gather full-vocab logprobs at the label positions.

    ``next_token_logits`` is ``[rows, vocab]`` with one row per scored
    sequence at its answer boundary; ``score_label_ids`` is the padded
    ``[rows, max_labels]`` index from :func:`build_score_label_ids`.
    ``num_prefill_outputs`` selects the emitted request prefix; incomplete
    chunks and decode rows have no score readout. Zero returns ``None``.
    ``logprob_order`` uses the same resolved reduction as sampled and prompt
    logprobs, including the trainer's fixed vocab-block order.
    Returns raw logprobs ``[num_prefill_outputs, max_labels]`` — padded columns and rows
    without score labels hold values the consumer must ignore.
    """
    if num_prefill_outputs == 0:
        return None
    return gather_token_logprobs(
        next_token_logits[:num_prefill_outputs],
        score_label_ids[:num_prefill_outputs],
        logprob_order=logprob_order,
    )


def finalize_score_row(row_logprobs: list[float], apply_softmax: bool) -> list[float]:
    """Finalize one request's label scores for the response.

    With ``apply_softmax`` the row is normalized across the label set
    (label-restricted softmax); otherwise the raw logprobs are returned
    unchanged. Normalization is per candidate row, never across rows, and
    the result is not a calibrated correctness probability.
    """
    if not apply_softmax:
        return row_logprobs
    peak = max(row_logprobs)
    exps = [math.exp(v - peak) for v in row_logprobs]
    total = sum(exps)
    return [v / total for v in exps]
