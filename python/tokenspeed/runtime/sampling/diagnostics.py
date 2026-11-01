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

"""Optional diagnostics around the Engine's common sampling boundary."""

from collections.abc import Callable

import torch

from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.logits_processor_output import LogitsProcessorOutput
from tokenspeed.runtime.sampling.sampling_batch_info import (
    SamplingBatchInfo,
    request_vocab_mask,
)

# Elements of one row chunk's [rows, vocab] temporaries: about 11 bytes each
# (FP32 weights, two FP32 selections and a boolean order mask), so a chunk
# stays near 90 MB whatever the batch.
_CHUNK_ELEMENTS = 1 << 23


def _copy_score_logits(
    logits_output: LogitsProcessorOutput,
    sampling_info: SamplingBatchInfo,
    layout: ForwardOutputLayout,
) -> torch.Tensor:
    """Keep pre-penalty scores while samplers may change their input in place."""
    logits = logits_output.next_token_logits
    if logits.ndim != 2 or logits.shape[0] != layout.num_output_tokens:
        raise ValueError("sampling score rows must match the forward output layout")
    if logits_output.logits_layout_plan is not None:
        raise ValueError("cumulative probabilities require full-vocabulary score rows")
    scores = logits.clone()
    mask = sampling_info.vocab_mask
    if mask is None:
        return scores
    if sampling_info.apply_vocab_mask is None:
        raise ValueError("a vocabulary mask requires an apply_vocab_mask function")
    requests = layout.num_extends + layout.num_decodes
    if mask.ndim != 2 or mask.shape[0] % requests:
        raise ValueError("vocabulary mask rows must be request-indexed")
    width = mask.shape[0] // requests
    if width == 0 or (layout.num_decodes and width != layout.decode_width):
        raise ValueError("vocabulary mask width must match decode verification")
    # The score rows' masks, in output order, selected exactly as the sampler
    # selects them (request_vocab_mask): the prefills with an output row
    # (every extend, except where the layout omits incomplete prefills), then
    # the decodes' verify columns.
    parts = []
    if layout.num_prefill_outputs:
        parts.append(
            request_vocab_mask(
                mask, layout.prefill_slice, mask_width=width, prefill=True
            )
        )
    if layout.num_decodes:
        parts.append(
            request_vocab_mask(
                mask, layout.decode_request_slice, mask_width=width, prefill=False
            )
        )
    packed_mask = parts[0] if len(parts) == 1 else torch.cat(parts)
    sampling_info.apply_vocab_mask(logits=scores, vocab_mask=packed_mask.contiguous())
    return scores


def sample_with_diagnostics(
    logits_output: LogitsProcessorOutput,
    sampling_info: SamplingBatchInfo,
    layout: ForwardOutputLayout,
    sample: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    enabled: bool,
    finalize: Callable[[torch.Tensor, torch.Tensor], None] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run sampling once and optionally record selected-token cumulative mass.

    Args:
        logits_output: Model scores and the logprobs written by ``sample``.
        sampling_info: Request-indexed grammar masks and their in-place applier.
        layout: Packed output geometry for prefills and chain verify.
        sample: The existing sample/verify operation, including TP agreement.
            It must return flat tokens in ``layout`` order and request lengths.
        enabled: Whether to retain score logits and compute the diagnostic.
        finalize: Run on the sampled tokens and lengths before anything reads
            them -- a drafter's in-place rewrite of its verify outputs -- or
            None. The diagnostic scores the tokens it leaves.

    Returns:
        The operation's tokens and lengths, as ``finalize`` left them. When
        enabled and outputs exist, ``logits_output.next_token_cumulative_probs``
        receives one value per score row; otherwise the field is ``None``.

    The Engine calls this after its NaN guard and grammar-mask wait, before stop
    trimming. Keep that ordering: backend-local NaN recovery is not duplicated
    here. Scores are copied and grammar-masked before the operation, so in-place
    penalties cannot change the diagnostic's pre-penalty distribution. Only
    BF16/FP16/FP32 model scores are part of the Engine's numerical contract.
    The disabled path performs no tensor operations or additional random draws.
    """
    logits_output.next_token_cumulative_probs = None
    scores = (
        _copy_score_logits(logits_output, sampling_info, layout)
        if enabled and layout.num_output_tokens
        else None
    )
    result = sample()
    if finalize is not None:
        finalize(*result)
    if not enabled:
        return result
    tokens, _ = result
    if tokens.shape != (layout.num_output_tokens,):
        raise ValueError("sampled tokens must match the forward output layout")
    if scores is not None:
        logprobs = logits_output.next_token_logprobs
        if logprobs is None or logprobs.shape != tokens.shape:
            raise ValueError("cumulative probabilities require aligned output logprobs")
        logits_output.next_token_cumulative_probs = gather_token_cumulative_probs(
            scores, tokens
        )
    return result


def gather_token_cumulative_probs(
    logits: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    """Return the normalized probability mass preceding each selected token.

    Tokens are ordered by descending logit, with smaller token IDs first for
    exact ties. The selected token itself is excluded. The diagnostic evaluates
    the supplied score logits; it applies no temperature, penalty or filtering.

    Args:
        logits: Real floating-point ``[rows, vocab]`` score logits. The vocabulary
            must be nonempty. Masked entries may be negative infinity.
        tokens: ``[rows]`` int32/int64 selected token IDs on the logits' device.

    Returns:
        ``[rows]`` FP32 cumulative mass in ``[0, 1]`` for valid distributions.
        A selected negative-infinity logit has mass 1 when the row otherwise
        defines a distribution. All-masked, NaN and positive-infinity rows
        produce NaN, rather than being clamped into apparently valid metadata,
        and so does a token outside ``[0, vocab)``: a sampler can return one for
        an invalid row (an all-NaN row's argmax lands on padding), and the
        Engine clamps it only later.

    Normalization is independent of the reported logprob arithmetic and does
    not modify it. FP64 inputs retain their score precision for ordering and
    max-shifting; other inputs use FP32 scores. Weights and reductions are
    always FP32, so no FP64 device support is needed for ordinary model logits.
    Rounding can leave a small endpoint difference between this result and
    ``exp(reported_logprob)``. Rows are processed in chunks of at most
    ``_CHUNK_ELEMENTS`` scores, which bounds the temporaries whatever the
    batch. Ordinary Torch operations keep this optional reference diagnostic
    backend-neutral and graph-capturable; no device values are read on the
    host.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [rows, vocab], got {tuple(logits.shape)}")
    if not logits.is_floating_point():
        raise ValueError("logits must contain real floating-point values")
    rows, vocab = logits.shape
    if vocab == 0:
        raise ValueError("logits must have a nonempty vocabulary")
    if tokens.shape != (rows,):
        raise ValueError("tokens must have one entry per logits row")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"tokens must be int32 or int64, got {tokens.dtype}")
    if tokens.device != logits.device:
        raise ValueError("tokens and logits must be on the same device")

    if rows == 0:
        return torch.empty(0, dtype=torch.float32, device=logits.device)
    token_ids = tokens.to(torch.int64)
    in_vocab = (token_ids >= 0) & (token_ids < vocab)
    # Gather in range; the out-of-range rows' results are replaced below.
    token_ids = torch.where(in_vocab, token_ids, 0)
    columns = torch.arange(vocab, device=logits.device, dtype=torch.int64)
    chunk = max(1, _CHUNK_ELEMENTS // vocab)
    parts = []
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        block = logits[start:stop]
        values = block if block.dtype == torch.float64 else block.float()
        ids = token_ids[start:stop].unsqueeze(-1)
        selected = values.gather(-1, ids)
        preceding = (values > selected) | ((values == selected) & (columns < ids))
        weights = torch.exp(values - values.amax(dim=-1, keepdim=True)).float()
        before = torch.where(preceding, weights, 0.0).sum(dim=-1)
        remaining = torch.where(preceding, 0.0, weights).sum(dim=-1)
        # Reuse the same rounded numerator in the denominator: adding
        # nonnegative remaining mass cannot make it smaller. Independent
        # full/prefix reductions can otherwise round a near-one CDF above 1.
        # Invalid weights remain NaN.
        parts.append(before / (before + remaining))
    mass = parts[0] if len(parts) == 1 else torch.cat(parts)
    return torch.where(in_vocab, mass, torch.nan)
