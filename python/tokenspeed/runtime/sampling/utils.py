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

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.sampling import vocab_parallel_logprobs

from tokenspeed.runtime.configs.numerics import MEGATRON_VOCAB_BLOCK
from tokenspeed.runtime.utils import crash_on_warnings, get_colorful_logger

logger = get_colorful_logger(__name__)

# Smallest positive value per dtype, used as the lower bound for `uniform_`
# draws that feed rejection-sampling kernels. A coin of exact 0 silently
# accepts a zero-probability draft in `chain_speculative_sampling_target_only`
# (the kernel condition `coin <= target_prob / threshold_acc` reduces to
# `0 <= 0`), so the coin must be strictly positive.
COIN_EPS = {
    torch.float32: torch.finfo(torch.float32).tiny,
    torch.bfloat16: torch.finfo(torch.bfloat16).tiny,
}


def coin_eps(dtype: torch.dtype) -> float:
    """Lower bound for uniform coin draws of the given dtype. See COIN_EPS."""
    return COIN_EPS[dtype]


def nan_guard_logits(
    logits: torch.Tensor,
    enable_nan_detection: bool,
) -> torch.Tensor:
    """Replace NaNs with -1e5 and optionally crash; no-op when detection is disabled."""
    if not enable_nan_detection:
        return logits

    if not torch.any(torch.isnan(logits)):
        return logits

    logger.warning("Detected errors during sampling! NaN in the logits.")
    logits = torch.where(torch.isnan(logits), torch.full_like(logits, -1e5), logits)
    if crash_on_warnings():
        raise ValueError("Detected errors during sampling! NaN in the logits.")
    return logits


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
    return raw_logprobs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def gather_token_logprobs(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    logprob_order: str,
) -> torch.Tensor:
    """Return each row's log probability of ``tokens`` in the launch's order.

    Args:
        logits: ``[rows, vocab]`` logits.
        tokens: ``[rows]`` integer token ids.
        logprob_order: ``"torch"`` for ``torch.log_softmax``; ``"megatron"``
            for the trainer's vocab-parallel cross-entropy order over fixed
            32768-wide vocab blocks (``--logprob-order``, validated by
            ServerArgs).

    Returns:
        ``[rows]`` fp32 log probabilities.
    """
    if logprob_order == "megatron":
        return vocab_parallel_logprobs(
            logits, tokens.to(torch.int64), vocab_block=MEGATRON_VOCAB_BLOCK
        )
    return gather_token_logprobs_torch(logits, tokens)
