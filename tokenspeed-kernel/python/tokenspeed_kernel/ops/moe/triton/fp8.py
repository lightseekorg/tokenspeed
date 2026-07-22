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
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

with redirect_triton_to_tokenspeed_triton():
    import triton_kernels  # noqa: F401
    import triton_kernels.matmul  # noqa: F401
    import triton_kernels.matmul_details  # noqa: F401
    import triton_kernels.matmul_details.opt_flags  # noqa: F401
    import triton_kernels.numerics  # noqa: F401
    import triton_kernels.numerics_details.mxfp  # noqa: F401
    import triton_kernels.swiglu  # noqa: F401
    import triton_kernels.tensor  # noqa: F401
    import triton_kernels.tensor_details  # noqa: F401
    import triton_kernels.tensor_details.layout  # noqa: F401
    import triton_kernels.topk  # noqa: F401

from tokenspeed_kernel.ops.moe.triton.mxfp4 import (
    _local_topk_for_ep,
    _release_parameter,
    _routing_from_topk,
    _silu_gate_up,
)
from triton_kernels.matmul import (
    FlexCtx,
    FnSpecs,
    FusedActivation,
    PrecisionConfig,
    matmul,
)
from triton_kernels.numerics import InFlexData
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.swiglu import swiglu_fn


def _scale_attr(w: torch.nn.Module, base: str) -> torch.Tensor:
    scale_inv = getattr(w, f"{base}_scale_inv", None)
    if scale_inv is not None:
        return scale_inv
    scale = getattr(w, f"{base}_scale", None)
    if scale is not None:
        return scale
    raise RuntimeError(f"FP8 MoE weight {base!r} is missing block scales")


def _block_dequant_transpose_for_matmul(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_shape: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    block_n, block_k = block_shape
    num_experts, n, k = weight.shape
    if n % block_n == 0 and k % block_k == 0:
        n_tiles = n // block_n
        k_tiles = k // block_k
        dequant = weight.to(torch.float32).view(
            num_experts,
            n_tiles,
            block_n,
            k_tiles,
            block_k,
        )
        dequant = dequant * scale[:, :, None, :, None]
        return dequant.permute(0, 3, 4, 1, 2).reshape(num_experts, k, n)

    scale_expanded = scale.repeat_interleave(block_n, dim=1).repeat_interleave(
        block_k, dim=2
    )
    return (weight.to(torch.float32) * scale_expanded[:, :n, :k]).transpose(-2, -1)


def _downcast_block_fp8_weight_to_mxfp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_dequant = _block_dequant_transpose_for_matmul(weight, scale).contiguous()
    weight_mxfp8, weight_scale = downcast_to_mxfp(
        weight_dequant,
        torch.float8_e4m3fn,
        axis=-2,
        scale_dtype=torch.uint8,
        microblock_size=32,
    )
    return weight_mxfp8, weight_scale


def triton_fp8_moe_weights(plan: dict, w: torch.nn.Module):
    w13_scale = _scale_attr(w, "w13_weight")
    w2_scale = _scale_attr(w, "w2_weight")

    if tuple(w.quant_config.weight_block_size) == (1, 32):
        if w13_scale.dtype != torch.uint8 or w2_scale.dtype != torch.uint8:
            raise RuntimeError("MXFP8 (1, 32) MoE scales must use UE8M0 uint8")

        if w.w13_input_layout == "concatenated":
            num_experts, two_intermediate, hidden_size = w.w13_weight.shape
            intermediate = two_intermediate // 2
            w13_weight = (
                w.w13_weight.view(num_experts, 2, intermediate, hidden_size)
                .transpose(1, 2)
                .reshape(num_experts, two_intermediate, hidden_size)
            )
            num_scale_groups = w13_scale.shape[-1]
            w13_scale = (
                w13_scale.view(num_experts, 2, intermediate, num_scale_groups)
                .transpose(1, 2)
                .reshape(num_experts, two_intermediate, num_scale_groups)
            )
        elif w.w13_input_layout == "interleaved":
            w13_weight = w.w13_weight
        else:
            raise RuntimeError(f"Unsupported w13 layout: {w.w13_input_layout!r}")

        w.w13_weight_triton_tensor = w13_weight.transpose(-2, -1)
        w.w2_weight_triton_tensor = w.w2_weight.transpose(-2, -1)
        w.w13_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(
                rhs_data=InFlexData(dtype=current_platform().fp8e4m3fn.dtype)
            ),
            b_mx_scale=w13_scale.transpose(-2, -1),
            b_microblock_size=32,
            out_dtype=torch.bfloat16,
        )
        w.w2_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(
                rhs_data=InFlexData(dtype=current_platform().fp8e4m3fn.dtype)
            ),
            b_mx_scale=w2_scale.transpose(-2, -1),
            b_microblock_size=32,
            out_dtype=torch.bfloat16,
        )

        _release_parameter(w, "w13_weight")
        _release_parameter(w, "w2_weight")
        _release_parameter(w, "w13_weight_scale_inv")
        _release_parameter(w, "w2_weight_scale_inv")
        return

    w13_weight, w13_mx_scale = _downcast_block_fp8_weight_to_mxfp8(
        w.w13_weight,
        w13_scale,
    )
    w2_weight, w2_mx_scale = _downcast_block_fp8_weight_to_mxfp8(
        w.w2_weight,
        w2_scale,
    )

    w.w13_weight_triton_tensor = w13_weight
    w.w2_weight_triton_tensor = w2_weight
    w.w13_precision_config = PrecisionConfig(
        flex_ctx=FlexCtx(rhs_data=InFlexData(dtype=current_platform().fp8e4m3fn.dtype)),
        b_mx_scale=w13_mx_scale,
        b_microblock_size=32,
        out_dtype=torch.bfloat16,
    )
    w.w2_precision_config = PrecisionConfig(
        flex_ctx=FlexCtx(rhs_data=InFlexData(dtype=current_platform().fp8e4m3fn.dtype)),
        b_mx_scale=w2_mx_scale,
        b_microblock_size=32,
        out_dtype=torch.bfloat16,
    )

    _release_parameter(w, "w13_weight")
    _release_parameter(w, "w2_weight")
    _release_parameter(w, "w13_weight_scale_inv")
    _release_parameter(w, "w2_weight_scale_inv")
    _release_parameter(w, "w13_weight_scale")
    _release_parameter(w, "w2_weight_scale")
    torch.cuda.empty_cache()


@register_kernel(
    "moe",
    "apply",
    name="triton_mxfp8_precomputed_moe_apply",
    solution="triton",
    weight_preprocessor=triton_fp8_moe_weights,
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures(
        "x",
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={
        "weight_dtype": frozenset({"fp8"}),
        "activation": frozenset({"swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"input"}),
        "fp8_scale_block_shape": frozenset({(1, 32)}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
@register_kernel(
    "moe",
    "apply",
    name="triton_fp8_precomputed_moe_apply",
    solution="triton",
    weight_preprocessor=triton_fp8_moe_weights,
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
    priority=Priority.PORTABLE,
)
@register_kernel(
    "moe",
    "apply",
    name="triton_fp8_ep_precomputed_moe_apply",
    solution="triton",
    weight_preprocessor=triton_fp8_moe_weights,
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
    priority=Priority.PORTABLE,
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
    if topk_weights is None or topk_ids is None:
        raise RuntimeError("triton FP8 MoE requires precomputed topk_weights/topk_ids")
    if topk_weights.shape != topk_ids.shape:
        raise RuntimeError(
            "topk_weights and topk_ids must have the same shape, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )

    top_k = getattr(w, "top_k", topk_ids.shape[1])
    n_tokens = x.shape[0]
    topk_weights, topk_ids, num_experts = _local_topk_for_ep(
        topk_weights,
        topk_ids,
        w,
    )
    ragged_metadata, gather_indx, scatter_indx, gate_scal = _routing_from_topk(
        topk_weights,
        topk_ids,
        num_experts=num_experts,
        dtype=router_logits.dtype,
    )

    w13_bias = getattr(w, "w13_weight_bias", None)
    w2_bias = getattr(w, "w2_weight_bias", None)
    if w13_bias is not None or w2_bias is not None:
        raise RuntimeError("triton FP8 MoE does not support bias")

    swiglu_arg = getattr(w, "swiglu_arg", None)
    act = None
    if swiglu_arg is not None:
        act = FusedActivation(
            FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
            (swiglu_arg.alpha, swiglu_arg.limit),
        )

    intermediate_cache = matmul(
        x,
        w.w13_weight_triton_tensor,
        None,
        a_ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        precision_config=w.w13_precision_config,
        fused_activation=act,
    )
    if act is None:
        intermediate_cache = _silu_gate_up(
            intermediate_cache,
            output_dtype=x.dtype,
        )

    output = matmul(
        intermediate_cache,
        w.w2_weight_triton_tensor,
        None,
        a_ragged_metadata=ragged_metadata,
        precision_config=w.w2_precision_config,
        scatter_indx=scatter_indx,
        gammas=gate_scal,
    )
    if top_k > 1:
        return output.view(n_tokens, top_k, output.shape[-1]).sum(dim=1)
    return output
