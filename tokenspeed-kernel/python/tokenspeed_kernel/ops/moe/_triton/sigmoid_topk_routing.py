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

"""Biased-sigmoid top-k over a small batch that also emits trtllm-gen routing metadata."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform


@triton.jit
def _sigmoid_bias_topk_routing_kernel(
    gating_output_ptr,
    correction_bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    stride_gm,
    stride_ge,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    padded_ptr,
    expanded_ptr,
    permuted_ptr,
    counts_ptr,
    batch_ptr,
    limit_ptr,
    num_ctas_ptr,
    num_tokens,
    num_experts: tl.constexpr,
    routed_scaling_factor: tl.constexpr,
    renormalize: tl.constexpr,
    local_offset: tl.constexpr,
    num_local: tl.constexpr,
    tile: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_L: tl.constexpr,
    MAX_CTA: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Dependents may launch now; their own wait still covers this grid's completion.
        tl.extra.cuda.gdc_launch_dependents()
    offs_t = tl.arange(0, BLOCK_T)
    offs_e = tl.arange(0, BLOCK_E)
    token_mask = offs_t < num_tokens
    expert_mask = offs_e < num_experts
    # The selection below is the per-token minimax kernel, vectorized over the batch rows.
    logits = tl.load(
        gating_output_ptr + offs_t[:, None] * stride_gm + offs_e[None, :] * stride_ge,
        mask=token_mask[:, None] & expert_mask[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    bias = tl.load(
        correction_bias_ptr + offs_e, mask=expert_mask, other=-float("inf")
    ).to(tl.float32)
    scores = tl.sigmoid(logits)
    choice_scores = scores + bias[None, :]
    choice_scores = tl.where(
        expert_mask[None, :] & (choice_scores == choice_scores),
        choice_scores,
        -float("inf"),
    )
    weights_sum = tl.zeros([BLOCK_T], dtype=tl.float32)
    kept_weights = ()
    for k in tl.static_range(0, TOPK):
        best_choice_score = tl.max(choice_scores, axis=1)
        best_expert = tl.min(
            tl.where(
                choice_scores == best_choice_score[:, None], offs_e[None, :], BLOCK_E
            ),
            axis=1,
        )
        best_expert = tl.minimum(best_expert, num_experts - 1)
        chosen = offs_e[None, :] == best_expert[:, None]
        best_weight = tl.max(tl.where(chosen, scores, 0.0), axis=1)
        weights_sum += best_weight
        kept_weights = kept_weights + (best_weight,)
        tl.store(
            topk_ids_ptr + offs_t * stride_im + k * stride_ik,
            best_expert.to(tl.int32),
            mask=token_mask,
        )
        choice_scores = tl.where(chosen, -float("inf"), choice_scores)
    # moe_sigmoid_bias_topk scales unnormalized weights after its kernel; one rounding either way.
    if renormalize:
        denom = tl.where(weights_sum != 0.0, weights_sum, 1.0)
    else:
        denom = 1.0
    for k in tl.static_range(0, TOPK):
        weight = kept_weights[k] / denom * routed_scaling_factor
        tl.store(
            topk_weights_ptr + offs_t * stride_wm + k * stride_wk,
            weight,
            mask=token_mask,
        )
    # The ids just stored are read back flat; the barrier publishes them within the program.
    tl.debug_barrier()

    offs = tl.arange(0, BLOCK_T * TOPK)
    token = offs // TOPK
    valid = token < num_tokens
    ids = tl.load(
        topk_ids_ptr + token * stride_im + (offs % TOPK) * stride_ik,
        mask=valid,
        other=-1,
    )
    local_id = ids - local_offset
    is_local = valid & (local_id >= 0) & (local_id < num_local)
    experts = tl.arange(0, BLOCK_L)
    expert_valid = experts < num_local
    # Routes off this rank count into bin 0 and are taken out again: no masked-bin semantics.
    counts = tl.histogram(tl.where(is_local, local_id, 0), BLOCK_L)
    off_rank = BLOCK_T * TOPK - tl.sum(is_local.to(tl.int32), axis=0)
    counts = counts - tl.where(experts == 0, off_rank, 0)
    num_cta = (counts + tile - 1) // tile
    cta_offset = tl.cumsum(num_cta, axis=0) - num_cta
    padded_counts = num_cta * tile
    row_offset = tl.cumsum(padded_counts, axis=0) - padded_counts
    total_ctas = tl.sum(num_cta, axis=0)

    # A route's row within its expert follows token order; top-k never repeats an expert per token.
    earlier_same = (ids[:, None] == ids[None, :]) & (token[None, :] < token[:, None])
    rank = tl.sum(earlier_same.to(tl.int32), axis=1)
    base = tl.gather(row_offset, tl.minimum(tl.maximum(local_id, 0), BLOCK_L - 1), 0)
    permuted = tl.where(is_local, base + rank, -1)
    tl.store(expanded_ptr + offs, permuted, mask=valid)
    tl.store(permuted_ptr + permuted, token, mask=is_local)
    # Each live tile's padding rows read -1, as flashinfer's populate writes them.
    pad = tl.arange(0, tile)
    pad_rows = row_offset[:, None] + counts[:, None] + pad[None, :]
    pad_mask = expert_valid[:, None] & (
        pad[None, :] < (padded_counts - counts)[:, None]
    )
    tl.store(
        permuted_ptr + pad_rows, tl.full([BLOCK_L, tile], -1, tl.int32), mask=pad_mask
    )

    # Local experts take their counts, the rest zero; the two stores never overlap.
    tl.store(counts_ptr + local_offset + experts, counts, mask=expert_valid)
    global_local = offs_e - local_offset
    remote = expert_mask & ((global_local < 0) | (global_local >= num_local))
    tl.store(counts_ptr + offs_e, tl.zeros([BLOCK_E], dtype=tl.int32), mask=remote)
    for cta in tl.static_range(MAX_CTA):
        slot = cta_offset + cta
        live = expert_valid & (cta < num_cta)
        tl.store(batch_ptr + slot, experts, mask=live)
        limit = tl.minimum((slot + 1) * tile, cta_offset * tile + counts)
        tl.store(limit_ptr + slot, limit, mask=live)
    tl.store(padded_ptr, total_ctas * tile)
    tl.store(num_ctas_ptr, total_ctas)


# Batches the one program beats the separate top-k and routing kernels for.
MAX_TOKENS = 4
# Warps per program by padded batch: a warp alone is fastest until the tiles outgrow it.
_WARPS = {1: 1, 2: 1, 4: 4}


def triton_sigmoid_bias_topk_routing(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    local_expert_offset: int,
    num_local_experts: int,
    routing: object,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the biased-sigmoid top-k experts and fill one trtllm-gen routing slot.

    The selection and weights are those of ``moe_sigmoid_bias_topk``. One
    program does both for up to ``MAX_TOKENS`` rows, so the routing never
    waits for a second launch.

    Args:
        router_logits: ``[num_tokens, num_experts]`` router outputs.
        correction_bias: ``[num_experts]`` bias added for the selection only.
        topk: Experts per token, a power of two.
        routed_scaling_factor: Factor applied to the route weights.
        normalize_topk_weights: Whether the weights sum to the factor.
        local_expert_offset: First expert this rank serves.
        num_local_experts: Number of experts this rank serves.
        routing: A flashinfer routing metadata slot, written in place: its
            ``tile_n`` pads each expert's rows; ``total_num_padded_tokens``,
            ``expanded_idx_to_permuted_idx``, ``permuted_idx_to_token_idx``,
            ``num_tokens_per_expert``, ``cta_idx_xy_to_batch_idx``,
            ``cta_idx_xy_to_mn_limit`` and ``num_non_exiting_ctas`` receive
            what flashinfer's own routing kernel writes.
        enable_pdl: Enable NVIDIA programmatic dependent launch.

    Returns:
        ``(topk_weights, topk_ids)``: FP32 and int32 ``[num_tokens, topk]``.
    """
    num_tokens, num_experts = router_logits.shape
    if num_tokens > MAX_TOKENS or num_tokens == 0:
        raise ValueError(f"the fused routing serves 1 to {MAX_TOKENS} tokens")
    if topk & (topk - 1) or topk > 32:
        raise ValueError("topk must be a power of two up to 32")
    if correction_bias.shape != (num_experts,):
        raise ValueError("correction_bias must hold one entry per expert")
    if routing.expanded_idx_to_permuted_idx.numel() < num_tokens * topk:
        raise ValueError("expanded_idx_to_permuted_idx is too short for the routes")
    if routing.num_tokens_per_expert.numel() != num_experts:
        raise ValueError("num_tokens_per_expert must hold every expert")
    if num_local_experts <= 0 or routing.tile_n <= 0:
        raise ValueError("experts and tile must be positive")
    if local_expert_offset < 0 or local_expert_offset + num_local_experts > num_experts:
        raise ValueError("the local experts must lie within num_experts")
    topk_weights = torch.empty(
        num_tokens, topk, dtype=torch.float32, device=router_logits.device
    )
    topk_ids = torch.empty(
        num_tokens, topk, dtype=torch.int32, device=router_logits.device
    )
    block_t = triton.next_power_of_2(num_tokens)
    use_pdl = enable_pdl and current_platform().is_hopper_plus
    launch = {"launch_pdl": True} if use_pdl else {}
    _sigmoid_bias_topk_routing_kernel[(1,)](
        router_logits,
        correction_bias,
        topk_weights,
        topk_ids,
        router_logits.stride(0),
        router_logits.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        routing.total_num_padded_tokens,
        routing.expanded_idx_to_permuted_idx,
        routing.permuted_idx_to_token_idx,
        routing.num_tokens_per_expert,
        routing.cta_idx_xy_to_batch_idx,
        routing.cta_idx_xy_to_mn_limit,
        routing.num_non_exiting_ctas,
        num_tokens,
        num_experts=num_experts,
        routed_scaling_factor=routed_scaling_factor,
        renormalize=normalize_topk_weights,
        local_offset=local_expert_offset,
        num_local=num_local_experts,
        tile=routing.tile_n,
        TOPK=topk,
        BLOCK_T=block_t,
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK_L=triton.next_power_of_2(num_local_experts),
        MAX_CTA=-(-block_t // routing.tile_n),
        ENABLE_PDL=use_pdl,
        num_warps=_WARPS[block_t],
        **launch,
    )
    return topk_weights, topk_ids
