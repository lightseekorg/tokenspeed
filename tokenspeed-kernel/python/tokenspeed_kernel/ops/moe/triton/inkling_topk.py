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

"""Fused Inkling MoE gate: sigmoid+bias top-k with shared-expert-sink weights.

One program per token, modeled on the vendored MiniMax biased grouped top-k
kernel (``thirdparty/triton``). Selection is ``sigmoid(routed logits) + bias``
with the reference's deterministic lowest-index tie-breaking; weights are the
raw-logit sigmoids of the selected routed experts jointly normalized with the
shared-expert sigmoids (the "sink"), scaled by ``route_scale * global_scale``.

Numerics: the normalization is computed in linear space,
``sigmoid(z) / sum(sigmoid(z))``, which is identical to the reference's
log-space form ``exp(logsigmoid(z) - logsumexp(logsigmoid(z)))``. Sigmoid is
bounded in (0, 1], so nothing can overflow, and the denominator only
underflows if every selected and shared logit is below ~-88 in fp32 — top-k
selection makes that unreachable. This matches the linear-space convention of
the DeepSeek-V3 / MiniMax production routing kernels.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["inkling_topk"]


@triton.jit
def inkling_topk_kernel(
    logits_ptr,
    bias_ptr,
    global_scale_ptr,
    weights_ptr,
    topk_ids_ptr,
    stride_lm,
    stride_le,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    n_routed: tl.constexpr,
    n_shared: tl.constexpr,
    route_scale: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_S: tl.constexpr,
    TOPK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    routed_mask = offs_e < n_routed

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    logits = tl.load(
        logits_ptr + token_id * stride_lm + offs_e * stride_le,
        mask=routed_mask,
        other=-float("inf"),
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + offs_e,
        mask=routed_mask,
        other=-float("inf"),
    ).to(tl.float32)
    scores = tl.sigmoid(logits)
    choice_scores = tl.where(routed_mask, scores + bias, -float("inf"))

    # Ties go to the lowest expert index (reference); staged sigmoids rescale once sink denom is known.
    denom = 0.0
    for k in tl.static_range(0, TOPK):
        best_choice_score = tl.max(choice_scores, axis=0)
        best_expert = tl.min(
            tl.where(choice_scores == best_choice_score, offs_e, BLOCK_E), axis=0
        )
        best_score = tl.max(tl.where(offs_e == best_expert, scores, 0.0), axis=0)
        denom += best_score
        tl.store(
            topk_ids_ptr + token_id * stride_im + k * stride_ik,
            best_expert.to(tl.int32),
        )
        tl.store(weights_ptr + token_id * stride_wm + k * stride_wk, best_score)
        choice_scores = tl.where(offs_e == best_expert, -float("inf"), choice_scores)

    # Shared-expert sink: columns [n_routed, n_routed+n_shared) join normalization but skip top-k.
    offs_s = tl.arange(0, BLOCK_S)
    shared_mask = offs_s < n_shared
    shared_logits = tl.load(
        logits_ptr + token_id * stride_lm + (n_routed + offs_s) * stride_le,
        mask=shared_mask,
        other=0.0,
    ).to(tl.float32)
    shared_scores = tl.where(shared_mask, tl.sigmoid(shared_logits), 0.0)
    denom += tl.sum(shared_scores, axis=0)

    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    denom = tl.where(denom != 0.0, denom, 1.0)
    scale = route_scale * global_scale / denom

    for k in tl.static_range(0, TOPK):
        w = tl.load(weights_ptr + token_id * stride_wm + k * stride_wk)
        tl.store(weights_ptr + token_id * stride_wm + k * stride_wk, w * scale)
    tl.store(
        weights_ptr + token_id * stride_wm + (TOPK + offs_s) * stride_wk,
        shared_scores * scale,
        mask=shared_mask,
    )

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def inkling_topk(
    logits: torch.Tensor,
    gate_bias: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    top_k: int,
    n_routed: int,
    route_scale: float,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Inkling gate: biased sigmoid top-k + sink-normalized weights.

    Args:
        logits: ``[num_tokens, n_routed + n_shared]`` router logits (fp32 or
            bf16, CUDA; strided views are fine). Columns ``[0, n_routed)`` are
            routed experts, the tail columns are the shared-expert sink.
        gate_bias: ``[n_routed]`` fp32 selection bias, added to the sigmoid
            scores for top-k selection only (never enters the weights).
        global_scale: ``[1]`` fp32 per-layer output multiplier.
        top_k: number of routed experts to select (``1 <= top_k <= n_routed``).
        n_routed: number of routed experts (``<= logits.shape[1]``).
        route_scale: static weight multiplier from the model config.
        enable_pdl: launch with Programmatic Dependent Launch (Hopper+):
            the kernel waits for its producer before the first load and
            signals dependents after its last store.

    Returns:
        ``(weights, topk_ids)`` where ``weights`` is ``[num_tokens,
        top_k + n_shared]`` fp32 — the selected-routed weights (first
        ``top_k`` columns, descending selection order, paired with
        ``topk_ids``) and the shared sink gammas (tail columns), jointly
        normalized to sum to ``route_scale * global_scale`` — and
        ``topk_ids`` is ``[num_tokens, top_k]`` int32.
    """
    if logits.ndim != 2 or gate_bias.ndim != 1:
        raise ValueError("logits must be [T, R+S] and gate_bias [R]")
    n_shared = logits.shape[1] - n_routed
    if n_shared < 0 or gate_bias.shape[0] != n_routed:
        raise ValueError(
            f"n_routed={n_routed} inconsistent with logits {tuple(logits.shape)} "
            f"/ gate_bias {tuple(gate_bias.shape)}"
        )
    if not 1 <= top_k <= n_routed:
        raise ValueError(f"top_k={top_k} out of range for n_routed={n_routed}")

    num_tokens = logits.shape[0]
    weights = torch.empty(
        (num_tokens, top_k + n_shared), dtype=torch.float32, device=logits.device
    )
    topk_ids = torch.empty((num_tokens, top_k), dtype=torch.int32, device=logits.device)
    if num_tokens == 0:
        return weights, topk_ids

    inkling_topk_kernel[(num_tokens,)](
        logits,
        gate_bias,
        global_scale,
        weights,
        topk_ids,
        logits.stride(0),
        logits.stride(1),
        weights.stride(0),
        weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        n_routed=n_routed,
        n_shared=n_shared,
        route_scale=float(route_scale),
        BLOCK_E=triton.next_power_of_2(n_routed),
        BLOCK_S=triton.next_power_of_2(max(n_shared, 1)),
        TOPK=top_k,
        ENABLE_PDL=enable_pdl,
        num_warps=1,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return weights, topk_ids
