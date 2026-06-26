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
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


def _scale_attr(w: torch.nn.Module, base: str) -> torch.Tensor:
    scale_inv = getattr(w, f"{base}_scale_inv", None)
    if scale_inv is not None:
        return scale_inv
    scale = getattr(w, f"{base}_scale", None)
    if scale is not None:
        return scale
    raise RuntimeError(f"FP8 MoE weight {base!r} is missing block scales")


@triton.jit
def _quantize_fp8_1x128(x, FP8_DTYPE: tl.constexpr):
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1.0e-10)
    scale = amax / 448.0
    q = tl.clamp(x / scale, -448.0, 448.0).to(FP8_DTYPE)
    return q, scale


@triton.jit
def _fp8_moe_w13_kernel(
    x_ptr,
    w13_ptr,
    w13_scale_ptr,
    topk_ids_ptr,
    hidden_ptr,
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    top_k: tl.constexpr,
    num_local_experts: tl.constexpr,
    expert_offset: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    w13_stride_e: tl.constexpr,
    w13_stride_n: tl.constexpr,
    w13_stride_k: tl.constexpr,
    w13_scale_stride_e: tl.constexpr,
    w13_scale_stride_n: tl.constexpr,
    w13_scale_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_r: tl.constexpr,
    hidden_stride_m: tl.constexpr,
    hidden_stride_r: tl.constexpr,
    hidden_stride_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    global_expert = tl.load(
        topk_ids_ptr + pid_m * topk_ids_stride_m + pid_r * topk_ids_stride_r
    )
    expert = global_expert - expert_offset
    valid_expert = (expert >= 0) & (expert < num_local_experts)

    gate_acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, hidden_size, BLOCK_K):
        k_idx = k_start + offs_k
        x = tl.load(
            x_ptr + pid_m * x_stride_m + k_idx * x_stride_k,
            mask=k_idx < hidden_size,
            other=0.0,
        ).to(tl.float32)
        x_q, x_scale = _quantize_fp8_1x128(x, FP8_DTYPE)

        gate_rows = offs_n
        up_rows = intermediate_size + offs_n
        gate = tl.load(
            w13_ptr
            + expert * w13_stride_e
            + gate_rows[None, :] * w13_stride_n
            + k_idx[:, None] * w13_stride_k,
            mask=valid_expert
            & (gate_rows[None, :] < intermediate_size)
            & (k_idx[:, None] < hidden_size),
            other=0.0,
        )
        up = tl.load(
            w13_ptr
            + expert * w13_stride_e
            + up_rows[None, :] * w13_stride_n
            + k_idx[:, None] * w13_stride_k,
            mask=valid_expert
            & (offs_n[None, :] < intermediate_size)
            & (k_idx[:, None] < hidden_size),
            other=0.0,
        )

        k_group = k_start // 128
        gate_scale = tl.load(
            w13_scale_ptr
            + expert * w13_scale_stride_e
            + (gate_rows // 128) * w13_scale_stride_n
            + k_group * w13_scale_stride_k,
            mask=valid_expert & (gate_rows < intermediate_size),
            other=0.0,
        )
        up_scale = tl.load(
            w13_scale_ptr
            + expert * w13_scale_stride_e
            + (up_rows // 128) * w13_scale_stride_n
            + k_group * w13_scale_stride_k,
            mask=valid_expert & (offs_n < intermediate_size),
            other=0.0,
        )

        x_q = tl.reshape(x_q, (1, BLOCK_K))
        gate_acc += tl.dot(x_q, gate) * x_scale * gate_scale[None, :]
        up_acc += tl.dot(x_q, up) * x_scale * up_scale[None, :]

    gate = gate_acc / (1.0 + tl.exp(-gate_acc))
    hidden = gate * up_acc
    tl.store(
        hidden_ptr
        + pid_m * hidden_stride_m
        + pid_r * hidden_stride_r
        + offs_n * hidden_stride_n,
        tl.reshape(hidden, (BLOCK_N,)).to(hidden_ptr.dtype.element_ty),
        mask=offs_n < intermediate_size,
    )


@triton.jit
def _fp8_moe_w2_kernel(
    hidden_ptr,
    w2_ptr,
    w2_scale_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    output_ptr,
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    top_k: tl.constexpr,
    num_local_experts: tl.constexpr,
    expert_offset: tl.constexpr,
    hidden_stride_m: tl.constexpr,
    hidden_stride_r: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    w2_stride_e: tl.constexpr,
    w2_stride_n: tl.constexpr,
    w2_stride_k: tl.constexpr,
    w2_scale_stride_e: tl.constexpr,
    w2_scale_stride_n: tl.constexpr,
    w2_scale_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_r: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_r: tl.constexpr,
    output_stride_m: tl.constexpr,
    output_stride_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)

    for route in range(0, top_k):
        global_expert = tl.load(
            topk_ids_ptr + pid_m * topk_ids_stride_m + route * topk_ids_stride_r
        )
        expert = global_expert - expert_offset
        valid_expert = (expert >= 0) & (expert < num_local_experts)
        route_weight = tl.load(
            topk_weights_ptr
            + pid_m * topk_weights_stride_m
            + route * topk_weights_stride_r
        ).to(tl.float32)
        route_weight = tl.where(valid_expert, route_weight, 0.0)

        route_acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, intermediate_size, BLOCK_K):
            k_idx = k_start + offs_k
            hidden = tl.load(
                hidden_ptr
                + pid_m * hidden_stride_m
                + route * hidden_stride_r
                + k_idx * hidden_stride_k,
                mask=k_idx < intermediate_size,
                other=0.0,
            ).to(tl.float32)
            hidden_q, hidden_scale = _quantize_fp8_1x128(hidden, FP8_DTYPE)
            weight = tl.load(
                w2_ptr
                + expert * w2_stride_e
                + offs_n[None, :] * w2_stride_n
                + k_idx[:, None] * w2_stride_k,
                mask=valid_expert
                & (offs_n[None, :] < hidden_size)
                & (k_idx[:, None] < intermediate_size),
                other=0.0,
            )
            weight_scale = tl.load(
                w2_scale_ptr
                + expert * w2_scale_stride_e
                + (offs_n // 128) * w2_scale_stride_n
                + (k_start // 128) * w2_scale_stride_k,
                mask=valid_expert & (offs_n < hidden_size),
                other=0.0,
            )
            hidden_q = tl.reshape(hidden_q, (1, BLOCK_K))
            route_acc += tl.dot(hidden_q, weight) * hidden_scale * weight_scale[None, :]
        acc += route_acc * route_weight

    tl.store(
        output_ptr + pid_m * output_stride_m + offs_n * output_stride_n,
        tl.reshape(acc, (BLOCK_N,)).to(output_ptr.dtype.element_ty),
        mask=offs_n < hidden_size,
    )


def _triton_fp8_moe_apply(
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    if topk_weights is None or topk_ids is None:
        raise RuntimeError("triton FP8 MoE requires precomputed topk_weights/topk_ids")
    if topk_weights.shape != topk_ids.shape:
        raise RuntimeError(
            "topk_weights and topk_ids must have the same shape, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )

    w13_scale = _scale_attr(w, "w13_weight")
    w2_scale = _scale_attr(w, "w2_weight")
    num_tokens = x.shape[0]
    top_k = topk_ids.shape[1]
    hidden_size = x.shape[1]
    intermediate_size = w.w2_weight.shape[2]
    num_local_experts = int(getattr(w, "num_local_experts", getattr(w, "num_experts")))
    expert_offset = int(getattr(w, "ep_rank", 0)) * num_local_experts
    fp8_dtype = tl.float8e4nv
    hidden = torch.empty(
        (num_tokens, top_k, intermediate_size),
        dtype=x.dtype,
        device=x.device,
    )
    output = torch.empty(
        (x.shape[0], w.w2_weight.shape[1]),
        dtype=x.dtype,
        device=x.device,
    )

    _fp8_moe_w13_kernel[(num_tokens, top_k, triton.cdiv(intermediate_size, 64))](
        x,
        w.w13_weight,
        w13_scale,
        topk_ids,
        hidden,
        num_tokens,
        hidden_size,
        intermediate_size,
        top_k,
        num_local_experts,
        expert_offset,
        x.stride(0),
        x.stride(1),
        w.w13_weight.stride(0),
        w.w13_weight.stride(1),
        w.w13_weight.stride(2),
        w13_scale.stride(0),
        w13_scale.stride(1),
        w13_scale.stride(2),
        topk_ids.stride(0),
        topk_ids.stride(1),
        hidden.stride(0),
        hidden.stride(1),
        hidden.stride(2),
        BLOCK_N=64,
        BLOCK_K=128,
        FP8_DTYPE=fp8_dtype,
        num_warps=4,
        num_stages=1,
    )
    _fp8_moe_w2_kernel[(num_tokens, triton.cdiv(output.shape[1], 64))](
        hidden,
        w.w2_weight,
        w2_scale,
        topk_ids,
        topk_weights,
        output,
        num_tokens,
        output.shape[1],
        intermediate_size,
        top_k,
        num_local_experts,
        expert_offset,
        hidden.stride(0),
        hidden.stride(1),
        hidden.stride(2),
        w.w2_weight.stride(0),
        w.w2_weight.stride(1),
        w.w2_weight.stride(2),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_N=64,
        BLOCK_K=128,
        FP8_DTYPE=fp8_dtype,
        num_warps=4,
        num_stages=1,
    )
    return output


@register_kernel(
    "moe",
    "apply",
    name="triton_fp8_precomputed_moe_apply",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures(
        "x",
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={
        "weight_dtype": frozenset({"fp8"}),
        "activation": frozenset({"silu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"input"}),
        "fp8_scale_block_shape": frozenset({(128, 128)}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.SPECIALIZED + 2,
)
@register_kernel(
    "moe",
    "apply",
    name="triton_fp8_ep_precomputed_moe_apply",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures(
        "x",
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={
        "weight_dtype": frozenset({"fp8"}),
        "activation": frozenset({"silu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"input"}),
        "fp8_scale_block_shape": frozenset({(128, 128)}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.SPECIALIZED + 1,
)
def triton_fp8_moe_apply(
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
    return _triton_fp8_moe_apply(x, w, topk_weights, topk_ids)


__all__ = ["triton_fp8_moe_apply"]
