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

"""Ascend BF16 and W8A8 INT8 MoE primitives."""

from __future__ import annotations

from functools import partial

import torch
import torch_npu


def _format_grouped_weight_nz(weight: torch.Tensor) -> torch.Tensor:
    config = torch.npu.config
    previous = getattr(config, "allow_internal_format", None)
    try:
        config.allow_internal_format = True
        weight = torch_npu.npu_format_cast(weight, 29)
        if torch_npu.get_npu_format(weight) != 29:
            raise RuntimeError("Ascend MoE weight conversion did not produce NZ")
        return weight
    finally:
        config.allow_internal_format = previous


def ascend_softmax_bias_topk(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Lite's selection-only biased softmax TopK."""
    del enable_pdl
    weights, ids, _ = torch_npu.npu_moe_gating_top_k(
        router_logits,
        topk,
        bias=correction_bias,
        k_group=1,
        group_count=1,
        group_select_mode=0,
        renorm=0,
        norm_type=0,
        out_flag=False,
        routed_scaling_factor=routed_scaling_factor,
        eps=1e-20,
    )
    return weights.float(), ids.to(torch.int32)


def ascend_bf16_process_moe_weights(*, plan: dict, w: torch.nn.Module) -> None:
    """Prepare GMM weights and load later HF updates into that layout in place."""
    if not getattr(w, "_ascend_bf16_moe_weights_processed", False):
        expected = (
            (w.num_local_experts, 2 * w.intermediate_size, w.hidden_size),
            (w.num_local_experts, w.hidden_size, w.intermediate_size),
        )
        if (tuple(w.w13_weight.shape), tuple(w.w2_weight.shape)) != expected:
            raise ValueError("Ascend MoE weights must use canonical W13/W2 layout")
        w.w13_weight.data = w.w13_weight.data.transpose(-1, -2).contiguous()
        w.w2_weight.data = w.w2_weight.data.transpose(-1, -2).contiguous()
        # The checkpoint stays in HF layout after the resident weights transpose.
        # Future loads must transpose sources and slice gate/up along the new axis.
        w.w13_weight.weight_loader = partial(
            w.w13_weight.weight_loader, do_transpose=True
        )
        w.w2_weight.weight_loader = partial(
            w.w2_weight.weight_loader, do_transpose=True
        )
        w._ascend_bf16_moe_weights_processed = True


def ascend_int8_process_moe_weights(*, plan: dict, w: torch.nn.Module) -> None:
    """Transpose canonical INT8 expert weights and persist them in NZ."""
    del plan
    if getattr(w, "_ascend_int8_moe_weights_processed", False):
        return
    expected = (
        (w.num_local_experts, 2 * w.intermediate_size, w.hidden_size),
        (w.num_local_experts, w.hidden_size, w.intermediate_size),
        (w.num_local_experts, 2 * w.intermediate_size),
        (w.num_local_experts, w.hidden_size),
    )
    actual = (
        tuple(w.w13_weight.shape),
        tuple(w.w2_weight.shape),
        tuple(w.w13_weight_scale.shape),
        tuple(w.w2_weight_scale.shape),
    )
    if actual != expected:
        raise ValueError("Ascend W8A8 MoE weights must use canonical layouts")
    if w.w13_weight.dtype != torch.int8 or w.w2_weight.dtype != torch.int8:
        raise TypeError("Ascend W8A8 MoE weights must be INT8")

    w.w13_weight.data = _format_grouped_weight_nz(
        w.w13_weight.data.transpose(-1, -2).contiguous()
    )
    w.w2_weight.data = _format_grouped_weight_nz(
        w.w2_weight.data.transpose(-1, -2).contiguous()
    )
    w.w13_weight_scale.data = w.w13_weight_scale.data.float().contiguous()
    w.w2_weight_scale.data = w.w2_weight_scale.data.contiguous()
    for name in ("w13_smooth_scale", "w2_smooth_scale"):
        smooth = getattr(w, name, None)
        if smooth is not None:
            data = smooth.data.float()
            if data.ndim == 1:
                data = data.unsqueeze(0)
            # Official DSQ requires one row per group, even for a shared scale.
            # Materialize once during preparation, never in forward/capture.
            if name == "w2_smooth_scale" and data.shape[0] == 1:
                data = data.expand(int(w.num_local_experts), -1)
            smooth.data = data.contiguous()
    w._ascend_int8_moe_weights_processed = True


def _ascend_bf16_expert_gmm(
    expanded: torch.Tensor,
    expert_counts: torch.Tensor,
    w: torch.nn.Module,
) -> torch.Tensor:
    """Apply the two local expert GMMs to dispatch-ordered real routes."""
    hidden = torch_npu.npu_grouped_matmul(
        [expanded],
        [w.w13_weight],
        bias=None,
        group_list=expert_counts,
        split_item=3,
        output_dtype=torch.bfloat16,
        group_type=0,
        group_list_type=1,
    )[0]
    hidden = torch_npu.mlp_split_swiglu(
        hidden,
        expert_counts.to(torch.int32),
        local_exp_start=0,
        local_exp_end=int(w.num_local_experts),
    )
    return torch_npu.npu_grouped_matmul(
        [hidden],
        [w.w2_weight],
        bias=None,
        group_list=expert_counts,
        split_item=3,
        output_dtype=torch.bfloat16,
        group_type=0,
        group_list_type=1,
    )[0]


def ascend_bf16_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Apply the graph-stable Ascend local-expert chain."""
    del router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
    if not do_finalize:
        raise ValueError("Ascend BF16 MoE does not support deferred finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Ascend BF16 MoE requires precomputed top-k tensors")
    if x.ndim != 2 or topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("x and top-k tensors must be rank-2")
    if topk_ids.shape[0] != x.shape[0] or topk_ids.shape[1] == 0:
        raise ValueError("top-k tensors must have shape [tokens, top_k > 0]")
    if x.dtype != torch.bfloat16 or x.device.type != "npu":
        raise TypeError("Ascend BF16 MoE requires BF16 NPU hidden states")
    if topk_ids.device != x.device or topk_weights.device != x.device:
        raise ValueError("top-k tensors and hidden states must share a device")
    if plan.get("activation") not in {"silu", "swiglu"}:
        raise ValueError("Ascend BF16 MoE supports standard SwiGLU only")
    if not getattr(w, "_ascend_bf16_moe_weights_processed", False):
        raise RuntimeError("Ascend BF16 MoE weights were not processed after loading")
    if x.shape[0] == 0:
        return torch.empty_like(x)

    local_experts = int(w.num_local_experts)
    hidden_size = int(w.hidden_size)
    intermediate_size = int(w.intermediate_size)
    if tuple(w.w13_weight.shape) != (
        local_experts,
        hidden_size,
        2 * intermediate_size,
    ) or tuple(w.w2_weight.shape) != (
        local_experts,
        intermediate_size,
        hidden_size,
    ):
        raise ValueError("Ascend MoE weights have an incompatible GMM layout")
    if w.w13_weight.dtype != x.dtype or w.w2_weight.dtype != x.dtype:
        raise TypeError("hidden states and Ascend MoE weights must share a dtype")

    num_experts = int(w.num_experts)
    expert_lo = int(w.ep_rank) * local_experts
    expert_hi = expert_lo + local_experts
    ep_size = int(w.ep_size)
    if expert_hi > num_experts or ep_size <= 0:
        raise ValueError("Ascend local-expert routing requires contiguous EP ownership")
    ids = topk_ids.to(torch.int32)
    routing_num_experts = num_experts
    if ep_size == 1:
        routing_num_experts += int(plan.get("num_zero_experts", 0))
    expanded, row_indices, expert_counts, _ = torch_npu.npu_moe_init_routing_v2(
        x,
        ids,
        scale=None,
        active_num=ids.numel(),
        expert_num=routing_num_experts,
        quant_mode=-1,
        active_expert_range=[expert_lo, expert_hi],
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
        drop_pad_mode=0,
    )
    hidden = _ascend_bf16_expert_gmm(expanded, expert_counts, w)
    return torch_npu.npu_moe_finalize_routing(
        hidden.unsqueeze(1),
        None,
        None,
        None,
        topk_weights,
        row_indices,
        ids,
        3,
    )


def ascend_int8_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Apply dynamic-token/per-channel W8A8 experts with NZ weights."""
    del router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
    if not do_finalize:
        raise ValueError("Ascend W8A8 MoE does not support deferred finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Ascend W8A8 MoE requires precomputed top-k tensors")
    if x.ndim != 2 or topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("x and top-k tensors must be rank-2")
    if topk_ids.shape[0] != x.shape[0] or topk_ids.shape[1] == 0:
        raise ValueError("top-k tensors must have shape [tokens, top_k > 0]")
    if x.dtype != torch.bfloat16 or x.device.type != "npu":
        raise TypeError("Ascend W8A8 MoE requires BF16 NPU hidden states")
    if topk_ids.device != x.device or topk_weights.device != x.device:
        raise ValueError("top-k tensors and hidden states must share a device")
    if plan.get("activation") not in {"silu", "swiglu"}:
        raise ValueError("Ascend W8A8 MoE supports standard SwiGLU only")
    if not getattr(w, "_ascend_int8_moe_weights_processed", False):
        raise RuntimeError("Ascend W8A8 MoE weights were not processed after loading")
    if x.shape[0] == 0:
        return torch.empty_like(x)

    local_experts = int(w.num_local_experts)
    hidden_size = int(w.hidden_size)
    intermediate_size = int(w.intermediate_size)
    if tuple(w.w13_weight.shape) != (
        local_experts,
        hidden_size,
        2 * intermediate_size,
    ) or tuple(w.w2_weight.shape) != (
        local_experts,
        intermediate_size,
        hidden_size,
    ):
        raise ValueError("Ascend W8A8 MoE weights have an incompatible GMM layout")
    if w.w13_weight.dtype != torch.int8 or w.w2_weight.dtype != torch.int8:
        raise TypeError("Ascend W8A8 MoE GMM weights must be INT8")
    if (
        torch_npu.get_npu_format(w.w13_weight) != 29
        or torch_npu.get_npu_format(w.w2_weight) != 29
    ):
        raise RuntimeError("Ascend W8A8 MoE GMM weights must use NZ format")

    num_experts = int(w.num_experts)
    expert_lo = int(w.ep_rank) * local_experts
    expert_hi = expert_lo + local_experts
    if expert_hi > num_experts or int(w.ep_size) <= 0:
        raise ValueError("Ascend local-expert routing requires contiguous EP ownership")
    ids = topk_ids.to(torch.int32)
    routing_num_experts = num_experts
    if int(w.ep_size) == 1:
        routing_num_experts += int(plan.get("num_zero_experts", 0))
    smooth13 = getattr(w, "w13_smooth_scale", None)
    expanded, row_indices, expert_counts, activation_scale = (
        torch_npu.npu_moe_init_routing_v2(
            x,
            ids,
            scale=smooth13,
            offset=None,
            active_num=ids.numel(),
            expert_num=routing_num_experts,
            quant_mode=1,
            active_expert_range=[expert_lo, expert_hi],
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            row_idx_type=0,
            drop_pad_mode=0,
        )
    )
    hidden = torch_npu.npu_grouped_matmul(
        [expanded],
        [w.w13_weight],
        bias=None,
        group_list=expert_counts,
        split_item=3,
        output_dtype=torch.int32,
        group_type=0,
        group_list_type=1,
    )[0]
    hidden, hidden_scale = torch_npu.npu_dequant_swiglu_quant(
        hidden,
        weight_scale=w.w13_weight_scale,
        activation_scale=activation_scale,
        bias=None,
        quant_scale=getattr(w, "w2_smooth_scale", None),
        quant_offset=None,
        group_index=expert_counts,
        activate_left=True,
        quant_mode=1,
    )
    return _ascend_int8_gmm2_finalize(
        hidden, hidden_scale, expert_counts, row_indices, ids, topk_weights, w
    )


def _ascend_int8_gmm2_finalize(hidden, hidden_scale, counts, rows, ids, weights, w):
    """Unchanged shared tail for composed and input-fused W8A8 paths."""
    hidden = torch_npu.npu_grouped_matmul(
        [hidden],
        [w.w2_weight],
        scale=[w.w2_weight_scale],
        per_token_scale=[hidden_scale.float()],
        bias=None,
        group_list=counts,
        split_item=3,
        output_dtype=torch.bfloat16,
        group_type=0,
        group_list_type=1,
    )[0]
    return torch_npu.npu_moe_finalize_routing(
        hidden.unsqueeze(1), None, None, None, weights.to(hidden.dtype), rows, ids, 3
    )


__all__ = [
    "ascend_bf16_precomputed_moe_apply",
    "ascend_bf16_process_moe_weights",
    "ascend_int8_precomputed_moe_apply",
    "ascend_int8_process_moe_weights",
    "ascend_softmax_bias_topk",
]
