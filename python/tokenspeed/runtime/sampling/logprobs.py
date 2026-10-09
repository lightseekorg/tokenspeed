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

"""Shared selected-token logprobs for sampled, prompt and Score readouts."""

from __future__ import annotations

import torch

from tokenspeed.runtime.configs.numerics import MEGATRON_VOCAB_BLOCK


def gather_token_logprobs_torch(
    logits: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Return the selected token's log probability for each logits row.

    The one logprob arithmetic for sampled and prompt rows (see
    ``docs/design/numerics.md``): an fp32 log-softmax over the row, gathered
    at the token. ``dtype=torch.float32`` converts bf16 logits inside the
    kernel (an exact widening) instead of materializing an fp32 copy of the
    ``[rows, vocab]`` tensor first. Both consumers call this function, so a
    token's prompt and output logprobs are the same number by construction.
    """
    raw_logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
    if tokens.ndim == 1:
        return raw_logprobs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return raw_logprobs.gather(-1, tokens)


def gather_token_logprobs(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    logprob_order: str,
) -> torch.Tensor:
    """Return each row's log probability of ``tokens`` in the launch's order.

    Args:
        logits: ``[rows, vocab]`` logits.
        tokens: ``[rows]`` or ``[rows, labels]`` integer token ids.
        logprob_order: ``"torch"`` for ``torch.log_softmax``; ``"megatron"``
            for the trainer's vocab-parallel cross-entropy order over fixed
            32768-wide vocab blocks (``--logprob-order``, validated by
            ServerArgs).

    Returns:
        fp32 log probabilities with the same shape as ``tokens``.
    """
    if logprob_order == "megatron":
        from tokenspeed_kernel.ops.sampling import vocab_parallel_logprobs

        return vocab_parallel_logprobs(
            logits, tokens.to(torch.int64), vocab_block=MEGATRON_VOCAB_BLOCK
        )
    return gather_token_logprobs_torch(logits, tokens)
