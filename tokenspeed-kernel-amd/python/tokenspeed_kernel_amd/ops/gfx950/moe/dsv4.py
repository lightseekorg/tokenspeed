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

"""DeepSeek V4 expert selection optimized for AMD GFX950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

__all__ = ["gluon_dsv4_select_experts_gfx950"]

cdna4 = gl.amd.cdna4


@gluon.jit
def _dsv4_select_experts_kernel(
    router_logits,
    correction_bias,
    hash_indices_table,
    input_ids,
    topk_weights,
    topk_ids,
    output_scores,
    stride_lm,
    stride_le,
    stride_be,
    stride_hm,
    stride_hk,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    stride_sm,
    stride_se,
    E: gl.constexpr,
    EP: gl.constexpr,
    TOPK: gl.constexpr,
    TKP: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    HASH_ROUTING: gl.constexpr,
    RENORMALIZE: gl.constexpr,
    NEED_SCORES: gl.constexpr,
):
    token = gl.program_id(0)
    expert_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    topk_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    expert = gl.arange(0, EP, layout=expert_layout)
    expert_mask = expert < E

    logits = cdna4.buffer_load(
        router_logits,
        (token * stride_lm + expert * stride_le).to(gl.int32),
        mask=expert_mask,
        other=0.0,
    ).to(gl.float32)
    exp_neg_abs = gl.exp(-gl.abs(logits))
    # Gluon has no log1p; use its convergent series where 1 + x would round to 1.
    log1p_series = exp_neg_abs * (
        1.0
        + exp_neg_abs
        * (-0.5 + exp_neg_abs * (1.0 / 3.0 + exp_neg_abs * (-0.25 + exp_neg_abs * 0.2)))
    )
    log1p_exp = gl.where(
        exp_neg_abs < 0.01831563888873418,
        log1p_series,
        gl.log(1.0 + exp_neg_abs),
    )
    stable_softplus = gl.maximum(logits, 0.0) + log1p_exp
    softplus = gl.where(logits > 20.0, logits, stable_softplus)
    scores = gl.sqrt(softplus)

    if NEED_SCORES:
        cdna4.buffer_store(
            scores,
            output_scores,
            (token * stride_sm + expert * stride_se).to(gl.int32),
            mask=expert_mask,
        )

    topk_lane = gl.arange(0, TKP, layout=topk_layout)
    selected_ids = gl.zeros([TKP], gl.int32, layout=topk_layout)
    selected_weights = gl.zeros([TKP], gl.float32, layout=topk_layout)
    if HASH_ROUTING:
        input_id = gl.load(input_ids + token).to(gl.int64)
        for rank in gl.static_range(TOPK):
            selected_id = gl.load(
                hash_indices_table + input_id * stride_hm + rank * stride_hk
            ).to(gl.int32)
            weight = gl.sum(gl.where(expert == selected_id, scores, 0.0), axis=0)
            selected_ids = gl.where(topk_lane == rank, selected_id, selected_ids)
            selected_weights = gl.where(topk_lane == rank, weight, selected_weights)
    else:
        choice = scores
        if HAS_BIAS:
            bias = cdna4.buffer_load(
                correction_bias,
                (expert * stride_be).to(gl.int32),
                mask=expert_mask,
                other=0.0,
            ).to(gl.float32)
            choice += bias
        choice = gl.where(expert_mask, choice, -float("inf"))
        sentinel = gl.full([EP], E, gl.int32, expert_layout)
        for rank in gl.static_range(TOPK):
            maximum = gl.max(choice, axis=0)
            selected_id = gl.min(
                gl.where((choice == maximum) & expert_mask, expert, sentinel),
                axis=0,
            )
            weight = gl.sum(gl.where(expert == selected_id, scores, 0.0), axis=0)
            selected_ids = gl.where(topk_lane == rank, selected_id, selected_ids)
            selected_weights = gl.where(topk_lane == rank, weight, selected_weights)
            choice = gl.where(expert == selected_id, -float("inf"), choice)

    if RENORMALIZE:
        denominator = gl.sum(selected_weights, axis=0)
        denominator = gl.maximum(denominator, 1.1754943508222875e-38)
        selected_weights = selected_weights / denominator

    topk_mask = topk_lane < TOPK
    cdna4.buffer_store(
        selected_weights,
        topk_weights,
        (token * stride_wm + topk_lane * stride_wk).to(gl.int32),
        mask=topk_mask,
    )
    cdna4.buffer_store(
        selected_ids,
        topk_ids,
        (token * stride_im + topk_lane * stride_ik).to(gl.int32),
        mask=topk_mask,
    )


def _validate_tensor_device(
    tensor: torch.Tensor, name: str, device: torch.device
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on the same device as router_logits")


def gluon_dsv4_select_experts_gfx950(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    need_scores: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select DeepSeek V4 experts with one GFX950 Gluon kernel launch.

    Args:
        router_logits: FP16, BF16, or FP32 logits shaped ``[tokens, experts]``.
        top_k: Number of experts to select. This specialization requires six.
        renormalize: Whether to normalize selected unbiased scores per token.
        correction_bias: Optional selection-only bias shaped ``[experts]``.
        hash_indices_table: Optional token-id to expert-id table shaped
            ``[vocabulary, top_k]``.
        input_ids: Token ids corresponding to rows of ``router_logits``.
        need_scores: Whether to materialize all FP32 sqrt-softplus scores.

    Returns:
        FP32 weights, INT32 expert ids, and FP32 scores. When ``need_scores`` is
        false, ``router_logits`` is returned as the ignored third output.
    """
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")
    if not router_logits.is_cuda:
        raise ValueError("router_logits must be a CUDA/HIP tensor")
    if router_logits.layout != torch.strided:
        raise ValueError("router_logits must use a strided layout")
    if router_logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("router_logits must have dtype float16, bfloat16, or float32")

    tokens, experts = router_logits.shape
    if experts not in (256, 384):
        raise ValueError(
            f"GFX950 DeepSeek V4 routing requires 256 or 384 experts, got {experts}"
        )
    if top_k != 6:
        raise ValueError(f"GFX950 DeepSeek V4 routing requires top_k=6, got {top_k}")
    if not isinstance(renormalize, bool):
        raise TypeError("renormalize must be a bool")
    if not isinstance(need_scores, bool):
        raise TypeError("need_scores must be a bool")

    hash_routing = hash_indices_table is not None
    if correction_bias is not None:
        if correction_bias.shape != (experts,):
            raise ValueError(f"correction_bias must have shape [{experts}]")
        if not hash_routing:
            if correction_bias.layout != torch.strided:
                raise ValueError("correction_bias must use a strided layout")
            if correction_bias.dtype not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ):
                raise ValueError(
                    "correction_bias must have dtype float16, bfloat16, or float32"
                )
            _validate_tensor_device(
                correction_bias, "correction_bias", router_logits.device
            )

    if hash_routing:
        if input_ids is None:
            raise ValueError("hash-routed DeepSeek V4 MoE requires input_ids")
        if hash_indices_table.ndim != 2 or hash_indices_table.shape[1] != top_k:
            raise ValueError("hash_indices_table must have shape [vocabulary, top_k]")
        if hash_indices_table.layout != torch.strided:
            raise ValueError("hash_indices_table must use a strided layout")
        if input_ids.layout != torch.strided:
            raise ValueError("input_ids must use a strided layout")
        if hash_indices_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("hash_indices_table must have dtype int32 or int64")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must have dtype int32 or int64")
        if input_ids.numel() != tokens:
            raise ValueError(f"input_ids must contain {tokens} token ids")
        if not input_ids.is_contiguous():
            raise ValueError("input_ids must be contiguous")
        _validate_tensor_device(
            hash_indices_table, "hash_indices_table", router_logits.device
        )
        _validate_tensor_device(input_ids, "input_ids", router_logits.device)

    topk_weights = torch.empty(
        (tokens, top_k), dtype=torch.float32, device=router_logits.device
    )
    topk_ids = torch.empty(
        (tokens, top_k), dtype=torch.int32, device=router_logits.device
    )
    scores = (
        torch.empty((tokens, experts), dtype=torch.float32, device=router_logits.device)
        if need_scores
        else router_logits
    )
    if tokens == 0:
        return topk_weights, topk_ids, scores

    bias_routing = correction_bias is not None and not hash_routing
    bias = correction_bias if bias_routing else topk_weights
    hash_table = hash_indices_table if hash_indices_table is not None else topk_ids
    token_ids = input_ids if hash_routing else topk_ids
    _dsv4_select_experts_kernel[(tokens,)](
        router_logits,
        bias,
        hash_table,
        token_ids,
        topk_weights,
        topk_ids,
        scores,
        router_logits.stride(0),
        router_logits.stride(1),
        bias.stride(0),
        hash_table.stride(0),
        hash_table.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        scores.stride(0),
        scores.stride(1),
        E=experts,
        EP=256 if experts == 256 else 512,
        TOPK=top_k,
        TKP=8,
        HAS_BIAS=bias_routing,
        HASH_ROUTING=hash_routing,
        RENORMALIZE=renormalize,
        NEED_SCORES=need_scores,
        num_warps=1,
    )
    return topk_weights, topk_ids, scores
