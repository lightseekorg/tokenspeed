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

"""Portable sqrt-softplus top-k routing for DeepSeek V4.

V4 scores experts with ``sqrt(softplus(logits))``. The correction bias steers
selection only -- the route weights come from the unbiased score -- so the
reference does the whole thing as a chain of a dozen tensor ops. Without a
vendor routing library that chain is what runs, and at decode widths every one
of those launches costs more than the arithmetic in it.

This is the same routing in one kernel, one program per token.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["deepseek_v4_softplus_sqrt_topk"]

# torch.nn.functional.softplus switches to the identity above this input, and
# the kernel has to switch at the same place or the scores disagree.
_SOFTPLUS_THRESHOLD = 20.0
_FP32_TINY = torch.finfo(torch.float32).tiny


@triton.jit
def _ordered_key(value):
    """Map float32 to a uint32 whose unsigned order is the float order."""
    bits = value.to(tl.uint32, bitcast=True)
    sign = tl.full(bits.shape, 0x80000000, tl.uint32)
    full = tl.full(bits.shape, 0xFFFFFFFF, tl.uint32)
    return bits ^ tl.where((bits & sign) != 0, full, sign)


@triton.jit
def _softplus_sqrt(x, threshold: tl.constexpr):
    """``sqrt(softplus(x))`` to float32 precision across the whole range.

    Computing ``log(1 + exp(x))`` directly collapses to zero once ``exp(x)``
    falls below the float32 spacing at 1.0 -- around x = -16.6, well inside the
    range the reference still resolves through ``log1p``. Selection never picks
    those experts, but the score matrix goes back to the caller, so this uses
    the standard compensated ``log1p`` instead of a hand-picked crossover:
    where ``1 + z`` rounded back to 1 the addition lost ``z`` entirely and
    ``log1p(z) == z``, and elsewhere ``log(u) * z / (u - 1)`` corrects for
    whatever the addition did round off.
    """
    z = tl.exp(x)
    u = 1.0 + z
    log1p = tl.where(u == 1.0, z, tl.log(u) * z / (u - 1.0))
    return tl.sqrt(tl.where(x > threshold, x, log1p))


@triton.jit(do_not_specialize=["num_tokens"])
def _deepseek_v4_softplus_sqrt_topk_kernel(
    logits_ptr,
    bias_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    scores_ptr,
    num_tokens,
    NUM_EXPERTS: tl.constexpr,
    PADDED_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    SELECT_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    WRITE_SCORES: tl.constexpr,
    THRESHOLD: tl.constexpr,
    TINY: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_tokens:
        return

    expert = tl.arange(0, PADDED_EXPERTS)
    live = expert < NUM_EXPERTS
    logits = tl.load(
        logits_ptr + row * NUM_EXPERTS + expert, mask=live, other=-float("inf")
    ).to(tl.float32)
    scores = _softplus_sqrt(logits, THRESHOLD)
    if WRITE_SCORES:
        tl.store(scores_ptr + row * NUM_EXPERTS + expert, scores, mask=live)

    choice = scores
    if HAS_BIAS:
        choice = choice + tl.load(bias_ptr + expert, mask=live, other=0.0).to(
            tl.float32
        )
    # Padding lanes and NaN must not win a slot; the packed key sorts NaN high.
    choice = tl.where(live & (choice == choice), choice, -float("inf"))

    # One bitonic pass over a key that carries the score in the high half and
    # the inverted expert id in the low half gives descending score with the
    # lower id winning ties, which is what torch.topk(sorted=True) returns.
    packed = (_ordered_key(choice).to(tl.uint64) << 32) | (PADDED_EXPERTS - expert).to(
        tl.uint64
    )
    selected = tl.topk(packed, SELECT_K, dim=0)
    ids = (PADDED_EXPERTS - (selected & 0xFFFFFFFF).to(tl.int32)).to(tl.int32)

    slot = tl.arange(0, SELECT_K)
    keep = slot < TOPK
    # Weights come from the unbiased score, so recompute at the chosen experts
    # rather than reusing ``choice``.
    chosen = tl.load(logits_ptr + row * NUM_EXPERTS + ids, mask=keep, other=0.0).to(
        tl.float32
    )
    weights = tl.where(keep, _softplus_sqrt(chosen, THRESHOLD), 0.0)
    if RENORMALIZE:
        weights = weights / tl.maximum(tl.sum(weights, axis=0), TINY)

    tl.store(topk_ids_ptr + row * TOPK + slot, ids, mask=keep)
    tl.store(topk_weights_ptr + row * TOPK + slot, weights, mask=keep)


def deepseek_v4_softplus_sqrt_topk(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    *,
    correction_bias: torch.Tensor | None = None,
    return_scores: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Select each token's experts by ``sqrt(softplus(logits))`` in one launch.

    Args:
        router_logits: ``[num_tokens, num_experts]`` float32 router output.
        top_k: Experts to select per token; at most 32.
        renormalize: Divide the selected weights by their sum.
        correction_bias: Optional ``[num_experts]`` float32 bias that steers
            selection only -- the returned weights come from the unbiased
            score, matching the reference.
        return_scores: Also return the full ``[num_tokens, num_experts]``
            score matrix, which callers that need it would otherwise recompute.

    Returns:
        ``(topk_weights, topk_ids, scores)`` with float32 weights, int32 ids,
        and ``scores`` set only when ``return_scores``.
    """
    if router_logits.dim() != 2:
        raise ValueError(f"router_logits must be 2D, got {router_logits.shape}")
    if router_logits.dtype != torch.float32:
        raise TypeError(f"router_logits must be float32, got {router_logits.dtype}")
    num_tokens, num_experts = router_logits.shape
    if not 0 < top_k <= min(32, num_experts):
        raise ValueError(f"top_k {top_k} out of range for {num_experts} experts")

    logits = router_logits.contiguous()
    topk_ids = torch.empty((num_tokens, top_k), dtype=torch.int32, device=logits.device)
    topk_weights = torch.empty(
        (num_tokens, top_k), dtype=torch.float32, device=logits.device
    )
    scores = torch.empty_like(logits) if return_scores else logits.new_empty(0)
    if num_tokens == 0:
        return topk_weights, topk_ids, (scores if return_scores else None)

    bias = None
    if correction_bias is not None:
        bias = correction_bias.to(device=logits.device, dtype=torch.float32)
        bias = bias.reshape(-1).contiguous()
        if bias.numel() != num_experts:
            raise ValueError(
                f"correction_bias has {bias.numel()} entries, need {num_experts}"
            )

    _deepseek_v4_softplus_sqrt_topk_kernel[(num_tokens,)](
        logits,
        bias,
        topk_ids,
        topk_weights,
        scores,
        num_tokens,
        NUM_EXPERTS=num_experts,
        PADDED_EXPERTS=triton.next_power_of_2(num_experts),
        TOPK=top_k,
        SELECT_K=max(2, triton.next_power_of_2(top_k)),
        HAS_BIAS=bias is not None,
        RENORMALIZE=renormalize,
        WRITE_SCORES=return_scores,
        THRESHOLD=_SOFTPLUS_THRESHOLD,
        TINY=_FP32_TINY,
        num_warps=4,
    )
    return topk_weights, topk_ids, (scores if return_scores else None)
