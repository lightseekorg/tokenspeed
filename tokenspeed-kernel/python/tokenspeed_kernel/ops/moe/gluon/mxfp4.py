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
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signature, format_signatures

with redirect_triton_to_tokenspeed_triton():
    import triton_kernels  # noqa: F401
    import triton_kernels.matmul  # noqa: F401
    import triton_kernels.matmul_details  # noqa: F401
    import triton_kernels.matmul_details.opt_flags  # noqa: F401
    import triton_kernels.numerics  # noqa: F401
    import triton_kernels.tensor  # noqa: F401
    import triton_kernels.tensor_details  # noqa: F401
    import triton_kernels.tensor_details.layout  # noqa: F401

import triton_kernels.matmul_details.opt_flags as opt_flags
from triton_kernels.matmul import (
    FlexCtx,
    PrecisionConfig,
)
from triton_kernels.numerics import InFlexData
from triton_kernels.tensor import (
    FP4,
    convert_layout,
    wrap_torch_tensor,
)
from triton_kernels.tensor_details import layout

platform = current_platform()


def _swizzle_mxfp4(quant_tensor, scale, num_warps):
    """Weight swizzle for mxfp4 MoE, used for OAI mxfp4 kernel."""

    value_layout = layout.make_default_matmul_mxfp4_w_layout(mx_axis=-2)
    scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=-2, num_warps=num_warps
    )
    if platform.is_blackwell:
        constraints = {
            "is_persistent": True,
            "epilogue_subtile": 1,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    elif platform.is_hopper:
        constraints = {
            "split_k": 1,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    elif platform.is_amd:
        # Fix block_k=256 to support scale swizzling.
        constraints = {
            "block_k": 256,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    # transpose the tensor so that the quantization axis is on dim1
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4), value_layout
    )
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout)
    return quant_tensor, InFlexData(), scale


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import gluon_mxfp_fused_moe

    @register_kernel(
        "moe",
        "process_weights",
        name="gluon_mxfp4_moe_process_weights",
        solution="gluon",
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=frozenset({format_signature()}),
        traits={"weight_dtype": frozenset({"mxfp4"})},
        # Lower priority than the triton implementation due to performance issue
        priority=Priority.PORTABLE - 1,
    )
    def gluon_mxfp4_moe_process_weights(plan: dict, w: torch.nn.Module):
        MXFP_BLOCK_SIZE = 32

        w13_weight_bias = w.w13_weight_bias.to(torch.float32)
        w2_weight_bias = w.w2_weight_bias.to(torch.float32)
        w.w13_weight_bias = torch.nn.Parameter(w13_weight_bias, requires_grad=False)
        w.w2_weight_bias = torch.nn.Parameter(w2_weight_bias, requires_grad=False)

        num_warps = 8
        w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
            w.w13_weight, w.w13_weight_scale, num_warps
        )
        w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(
            w.w2_weight, w.w2_weight_scale, num_warps
        )

        # Collapse per-expert input scales to a single per-tensor scale
        # per GEMM. Quark exports a constant value across experts for
        # static ``per_tensor`` quantisation; ``max`` is a safe reduction
        # in case individual experts reach slightly different values.
        w13_in_scale = (
            w.w13_input_scale.data.to(torch.float32)
            .max()
            .reshape(1)
            .to(w.w13_input_scale.device)
            .contiguous()
        )
        w2_in_scale = (
            w.w2_input_scale.data.to(torch.float32)
            .max()
            .reshape(1)
            .to(w.w2_input_scale.device)
            .contiguous()
        )
        w.w13_act_scale = w13_in_scale
        w.w2_act_scale = w2_in_scale

        fp8_dtype = current_platform().fp8e4m3fn.dtype
        w13_lhs = InFlexData(dtype=fp8_dtype, scale=w13_in_scale)
        w2_lhs = InFlexData(dtype=fp8_dtype, scale=w2_in_scale)
        # Force bf16 output so the swiglu / down-proj results stay in a
        # standard floating dtype; without this, ``triton_kernels.matmul``
        # defaults ``out_dtype`` to the input dtype (fp8) which would
        # make the subsequent reductions / re-quantisation blow up.
        out_dtype = torch.bfloat16

        w.w13_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(lhs_data=w13_lhs, rhs_data=w13_flex),
            b_mx_scale=w13_scale,
            b_microblock_size=MXFP_BLOCK_SIZE,
            out_dtype=out_dtype,
        )
        w.w2_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(lhs_data=w2_lhs, rhs_data=w2_flex),
            b_mx_scale=w2_scale,
            b_microblock_size=MXFP_BLOCK_SIZE,
            out_dtype=out_dtype,
        )

        w.w13_weight_triton_tensor = w13_weight
        w.w2_weight_triton_tensor = w2_weight
        del w.w13_weight
        del w.w2_weight

        torch.cuda.empty_cache()

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_moe_apply",
        solution="gluon",
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"kernel_routing"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"fp8"}),
            "supports_bias": frozenset({True}),
        },
        # Lower priority than the triton implementation due to performance issue
        priority=Priority.PORTABLE - 1,
    )
    def gluon_mxfp4_moe_apply(
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
    ):
        swiglu_arg = getattr(w, "swiglu_arg", None)

        router_logits = router_logits
        top_k = getattr(w, "top_k")

        swiglu_alpha = swiglu_arg.alpha if swiglu_arg else 1.702
        swiglu_limit = swiglu_arg.limit if swiglu_arg else 7.0

        return gluon_mxfp_fused_moe(
            x,
            router_logits,
            w.w13_weight_triton_tensor,
            w.w2_weight_triton_tensor,
            w13_bias=getattr(w, "w13_weight_bias", None),
            w2_bias=getattr(w, "w2_weight_bias", None),
            w13_precision_config=getattr(w, "w13_precision_config", None),
            w2_precision_config=getattr(w, "w2_precision_config", None),
            w13_act_scale=w.w13_act_scale,
            w2_act_scale=w.w2_act_scale,
            top_k=top_k,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
        )
