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

"""Single-launch softmax top-k routing for NVIDIA GPUs."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement, Platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit
def _float32_to_ordered_key(value):
    bits = value.to(tl.uint32, bitcast=True)
    sign = tl.full(bits.shape, 0x80000000, tl.uint32)
    full = tl.full(bits.shape, 0xFFFFFFFF, tl.uint32)
    return bits ^ tl.where((bits & sign) != 0, full, sign)


@triton.jit
def _softmax_topk_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    stride_lm,
    stride_le,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    ROUTED_SCALING_FACTOR: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    token = tl.program_id(0)
    expert = tl.arange(0, BLOCK_EXPERTS)
    valid = expert < NUM_EXPERTS

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    logits = tl.load(
        logits_ptr + token * stride_lm + expert * stride_le,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    logits = tl.where(logits == 0.0, 0.0, logits)

    # Softmax preserves ordering, so route directly on logits. Packing the
    # ordered FP32 bits with the inverse expert id gives one bitonic top-k and
    # deterministic lowest-id tie breaking.
    packed = (_float32_to_ordered_key(logits).to(tl.uint64) << 32) | (
        BLOCK_EXPERTS - expert
    ).to(tl.uint64)
    packed = tl.where(valid, packed, 0)
    selected = tl.topk(packed, TOPK_PAD, dim=0)
    selected_ids = (BLOCK_EXPERTS - (selected & 0xFFFFFFFF).to(tl.int32)).to(tl.int32)

    rank = tl.arange(0, TOPK_PAD)
    selected_mask = rank < TOPK
    selected_logits = tl.load(
        logits_ptr + token * stride_lm + selected_ids * stride_le,
        mask=selected_mask,
        other=-float("inf"),
    ).to(tl.float32)

    if RENORMALIZE:
        # softmax(all)[topk] / sum(softmax(all)[topk]) is exactly a softmax
        # over the selected logits; the full-expert denominator cancels.
        max_logit = tl.max(selected_logits, axis=0)
        numerators = tl.where(selected_mask, tl.exp(selected_logits - max_logit), 0.0)
    else:
        max_logit = tl.max(logits, axis=0)
        all_exp = tl.where(valid, tl.exp(logits - max_logit), 0.0)
        numerators = tl.where(selected_mask, tl.exp(selected_logits - max_logit), 0.0)

    denominator = tl.sum(numerators, axis=0) if RENORMALIZE else tl.sum(all_exp, axis=0)
    weights = numerators / denominator
    weights *= ROUTED_SCALING_FACTOR

    tl.store(
        weights_ptr + token * stride_wm + rank * stride_wk,
        weights,
        mask=selected_mask,
    )
    tl.store(
        ids_ptr + token * stride_im + rank * stride_ik,
        selected_ids,
        mask=selected_mask,
    )

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@register_kernel(
    "moe",
    "softmax_topk",
    name="triton_softmax_topk",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures(
        "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    priority=Priority.PERFORMANT,
    tags={"nvidia", "cuda_graph", "latency"},
)
def triton_softmax_topk(
    *,
    router_logits: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused softmax, top-k selection, normalization, and route scaling."""
    if (
        router_logits.ndim != 2
        or not router_logits.is_cuda
        or router_logits.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or router_logits.stride(1) != 1
    ):
        raise ValueError(
            "Triton softmax-topk requires 2D CUDA FP16/BF16/FP32 logits "
            "with a contiguous expert dimension"
        )
    tokens, experts = router_logits.shape
    if not 0 < topk <= min(experts, 32):
        raise ValueError(f"topk must be in [1, min({experts}, 32)], got {topk}")
    if experts > 1024:
        raise ValueError(
            f"Triton softmax-topk supports at most 1024 experts, got {experts}"
        )

    weights = torch.empty(
        (tokens, topk), dtype=torch.float32, device=router_logits.device
    )
    ids = torch.empty((tokens, topk), dtype=torch.int64, device=router_logits.device)
    if tokens == 0:
        return weights, ids

    block_experts = triton.next_power_of_2(experts)
    topk_pad = triton.next_power_of_2(topk)
    use_pdl = bool(enable_pdl and Platform.get().is_hopper_plus)
    if block_experts <= 256 and tokens > 256:
        num_warps = 1
    elif block_experts <= 256 and tokens > 128:
        num_warps = 2
    else:
        num_warps = 4

    _softmax_topk_kernel[(tokens,)](
        router_logits,
        weights,
        ids,
        router_logits.stride(0),
        router_logits.stride(1),
        weights.stride(0),
        weights.stride(1),
        ids.stride(0),
        ids.stride(1),
        NUM_EXPERTS=experts,
        BLOCK_EXPERTS=block_experts,
        TOPK=topk,
        TOPK_PAD=topk_pad,
        RENORMALIZE=renormalize,
        ROUTED_SCALING_FACTOR=float(routed_scaling_factor),
        ENABLE_PDL=use_pdl,
        num_warps=num_warps,
        num_stages=1,
        **({"launch_pdl": True} if use_pdl else {}),
    )
    return weights, ids


__all__ = ["triton_softmax_topk"]
