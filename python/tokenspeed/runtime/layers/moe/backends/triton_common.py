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

import functools
from functools import partial

import tokenspeed_kernel
import torch
import triton.language as tl
from torch import nn

from tokenspeed.runtime.layers.moe.backends.triton_config import (
    try_get_optimal_moe_config,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.utils.env import envs

__all__ = [
    "support_tensor_descriptor",
    "triton_forward",
]

padding_size = 128 if envs.TOKENSPEED_MOE_PADDING.get() else 0


def build_triton_gemms(
    layer: nn.Module,
    spec: MoELayerSpec,
    *,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    block_shape=None,
    dtype_tag: str = "bf16",
    gate_up_B_scale=None,
    down_B_scale=None,
):
    num_local_experts, intermediate_size_x2, hidden_size = layer.w13_weight.shape
    intermediate_size = intermediate_size_x2 // 2

    common = dict(
        compute_type=tl.bfloat16,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        filter_expert=True,
    )
    _experts_common = dict(
        **common,
        dtype=torch.bfloat16,
        features={"dispatch_sorted"},
        expected_kernel_name="triton_moe_fused_experts",
    )
    gemm = partial(tokenspeed_kernel.moe_experts, **_experts_common)
    gate_up_gemm = partial(
        gemm,
        A_scale=None,
        B_scale=gate_up_B_scale,
        mul_routed_weight=False,
        top_k=spec.top_k,
    )
    down_gemm = partial(
        gemm,
        A_scale=None,
        B_scale=down_B_scale,
        mul_routed_weight=True,
        top_k=1,
    )
    get_config_func = partial(
        try_get_optimal_moe_config,
        (num_local_experts, intermediate_size * 2, hidden_size),
        (num_local_experts, hidden_size, intermediate_size),
        spec.top_k,
        dtype_tag,
        block_shape=None,
        return_down_config=True,
    )
    return gate_up_gemm, down_gemm, get_config_func


def triton_forward(
    gate_up_gemm,
    down_gemm,
    get_config_func,
    activation: str,
    layer: nn.Module,
    hidden_states: torch.Tensor,
    topk_output: object,
) -> torch.Tensor:
    from tokenspeed.runtime.layers.activation import silu_and_mul

    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    ep_size = getattr(layer, "ep_size", 1)
    if ep_size > 1:
        num_local_experts_for_ep = getattr(
            layer, "num_local_experts", layer.w13_weight.shape[0]
        )
        local_expert_start = getattr(layer, "ep_rank", 0) * num_local_experts_for_ep
        local_expert_end = local_expert_start + num_local_experts_for_ep
        local_expert_mask = (topk_ids >= local_expert_start) & (
            topk_ids < local_expert_end
        )
        topk_ids = torch.where(
            local_expert_mask,
            topk_ids - local_expert_start,
            torch.zeros_like(topk_ids),
        )
        topk_weights = torch.where(
            local_expert_mask,
            topk_weights,
            torch.zeros_like(topk_weights),
        )
    m_tokens = hidden_states.shape[0]
    num_experts, intermediate_size_x2, hidden_size = layer.w13_weight.shape
    top_k = topk_ids.shape[1]
    dtype = hidden_states.dtype
    device = hidden_states.device

    config, (down_config, _max_block_m) = get_config_func(M=m_tokens)

    gate_up_moe_use_tma = config is not None and config.pop("USE_TMA", False)
    down_moe_use_tma = down_config is not None and down_config.pop("USE_TMA", False)

    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        tokenspeed_kernel.moe_dispatch(
            topk_ids,
            config["BLOCK_SIZE_M"],
            num_experts,
            dtype=torch.int32,
            expected_kernel_name="triton_moe_align_block_size",
        )
    )

    max_num_active_experts = min(m_tokens * top_k, num_experts + 1)
    padded_tokens = (
        max_num_active_experts * (config["BLOCK_SIZE_M"] - 1) if down_moe_use_tma else 0
    )
    intermediate_cache1 = torch.empty(
        (m_tokens * top_k + padded_tokens, intermediate_size_x2),
        device=device,
        dtype=dtype,
    )
    intermediate_cache2 = torch.empty(
        (m_tokens * top_k + padded_tokens, intermediate_size_x2 // 2),
        device=device,
        dtype=dtype,
    )
    intermediate_cache3 = torch.empty(
        (m_tokens, top_k, hidden_size),
        device=device,
        dtype=dtype,
    )

    gate_up_gemm(
        A=hidden_states,
        B=layer.w13_weight,
        bias=None,
        C=intermediate_cache1,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        config=config,
        a_use_tma=False,
        b_use_tma=gate_up_moe_use_tma,
        c_sorted=down_moe_use_tma,
    )

    if activation == "silu":
        silu_and_mul(
            intermediate_cache1.view(-1, intermediate_size_x2),
            intermediate_cache2,
        )
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    down_gemm(
        A=intermediate_cache2,
        B=layer.w2_weight,
        bias=None,
        C=intermediate_cache3,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        config=down_config,
        a_use_tma=down_moe_use_tma,
        b_use_tma=down_moe_use_tma,
    )

    out_hidden_states = torch.empty_like(hidden_states)
    # Current limitation: Should avoid using runtime shapes as traits
    expected_combine_kernel = (
        "torch_compile_moe_sum_reduce" if m_tokens <= 32 else "triton_moe_sum_reduce"
    )
    routed_scaling_factor = 1.0
    tokenspeed_kernel.moe_combine(
        intermediate_cache3,
        out_hidden_states,
        routed_scaling_factor,
        dtype=dtype,
        traits={"num_tokens": m_tokens, "comm_strategy": None},
        expected_kernel_name=expected_combine_kernel,
    )
    return out_hidden_states
