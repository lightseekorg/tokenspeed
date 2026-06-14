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

import copy
from contextlib import contextmanager

import tokenspeed_kernel
import torch
import torch.nn.functional as F
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signature, format_signatures

with redirect_triton_to_tokenspeed_triton():
    import triton_kernels  # noqa: F401
    import triton_kernels.matmul  # noqa: F401
    import triton_kernels.matmul_details  # noqa: F401
    import triton_kernels.matmul_details.opt_flags  # noqa: F401
    import triton_kernels.numerics  # noqa: F401
    import triton_kernels.swiglu  # noqa: F401
    import triton_kernels.tensor  # noqa: F401
    import triton_kernels.tensor_details  # noqa: F401
    import triton_kernels.tensor_details.layout  # noqa: F401
    import triton_kernels.topk  # noqa: F401

import triton_kernels.matmul_details.opt_flags as opt_flags
from triton_kernels.matmul import (
    FlexCtx,
    FnSpecs,
    FusedActivation,
    PrecisionConfig,
    matmul,
)
from triton_kernels.matmul_details.opt_flags import scoped_opt_flags_constraints
from triton_kernels.numerics import InFlexData
from triton_kernels.swiglu import swiglu_fn
from triton_kernels.tensor import (
    FP4,
    RaggedTensorMetadata,
    convert_layout,
    make_ragged_tensor_metadata,
    wrap_torch_tensor,
)
from triton_kernels.tensor_details import layout
from triton_kernels.topk import topk

# isort: off
from tokenspeed_kernel.ops.quantization.triton import fp8_quantize

platform = current_platform()

MXFP4_BLOCK = 32
MXFP4_ACTIVATION_SCALE_LAYOUT = "linear"


def _uses_dynamic_mxfp4_activations(w: torch.nn.Module) -> bool:
    quant_config = getattr(w, "quant_config", None)
    return current_platform().is_amd and bool(
        getattr(quant_config, "use_dynamic_mxfp4_activations", False)
    )


def _quantize_mxfp4_activation(
    activations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return tokenspeed_kernel.quantize_mxfp4(
        activations.contiguous(),
        scale_size=MXFP4_BLOCK,
        scale_layout=MXFP4_ACTIVATION_SCALE_LAYOUT,
        solution="triton",
        enable_pdl=False,
    )


def _with_activation_mx_scale(
    precision_config: PrecisionConfig | None,
    activation_scale: torch.Tensor,
) -> PrecisionConfig:
    if precision_config is None:
        precision_config = PrecisionConfig()
    precision_config = copy.copy(precision_config)
    precision_config.a_mx_scale = activation_scale
    precision_config.a_microblock_size = MXFP4_BLOCK
    return precision_config


def _release_parameter(module: torch.nn.Module, name: str) -> None:
    if name in module._parameters:
        module.register_parameter(name, None)
    elif hasattr(module, name):
        delattr(module, name)


def _silu_gate_up(
    gate_up: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    gate, up = gate_up.float().chunk(2, dim=-1)
    return (F.silu(gate) * up).to(output_dtype)


def _is_bf16_mxfp4(x, w, precision_config):
    if precision_config is None:
        return False
    if getattr(precision_config, "b_mx_scale", None) is None:
        return False
    x_dtype = getattr(x, "dtype", None)
    if x_dtype not in (torch.float16, torch.bfloat16):
        return False
    w_bw = getattr(getattr(w, "dtype", None), "bitwidth", None)
    return w_bw == 4


def _lds_guard_should_apply(x, w, precision_config):
    if scoped_opt_flags_constraints is None:
        return False
    if not current_platform().is_cdna4:
        return False
    return _is_bf16_mxfp4(x, w, precision_config)


@contextmanager
def _maybe_lds_guard(x, w, precision_config):
    if not _lds_guard_should_apply(x, w, precision_config):
        yield
        return
    with scoped_opt_flags_constraints({"block_m": 64, "block_n": 128, "block_k": 256}):
        yield


def _matmul(
    x,
    w,
    bias=None,
    a_ragged_metadata=None,
    gather_indx=None,
    scatter_indx=None,
    precision_config=None,
    fused_activation=None,
    epilogue=None,
    betas=None,
    gammas=None,
    out_alpha=None,
    y=None,
    n_tokens=None,
    n_expts_act=None,
    **_ignored,
):
    with _maybe_lds_guard(x, w, precision_config):
        out = matmul(
            x,
            w,
            bias,
            a_ragged_metadata=a_ragged_metadata,
            gather_indx=gather_indx,
            scatter_indx=scatter_indx,
            precision_config=precision_config,
            fused_activation=fused_activation,
            epilogue=epilogue,
            betas=betas,
            gammas=gammas,
            out_alpha=out_alpha,
            c=y,
        )
    if scatter_indx is not None and n_expts_act is not None and n_expts_act > 1:
        assert (
            n_tokens is not None
        ), "n_tokens required when n_expts_act > 1 for top-k reduction"
        return out.view(n_tokens, n_expts_act, out.shape[-1]).sum(dim=1)
    return out


def _routing(
    logits: torch.Tensor,
    n_expts_act: int,
    sm_first: bool = False,
    dtype: torch.dtype | None = None,
) -> tuple[RaggedTensorMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dtype is None:
        dtype = logits.dtype

    assert logits.ndim == 2, "router_logits must be (n_tokens, n_expts_tot)"
    n_tokens, _ = logits.shape

    assert sm_first is False, "sm_first=True not supported for triton_kernels routing"
    sparse = topk(logits, n_expts_act, apply_softmax=not sm_first)
    mask_metadata = sparse.mask_metadata

    col_sorted = mask_metadata.col_sorted_indx
    gather_indx = col_sorted // n_expts_act
    scatter_indx = col_sorted

    vals_flat = sparse.vals.reshape(-1)
    if dtype is not None and vals_flat.dtype != dtype:
        vals_flat = vals_flat.to(dtype)
    gate_scal = vals_flat[scatter_indx]

    n_total_rows = n_tokens * n_expts_act
    ragged_metadata = make_ragged_tensor_metadata(mask_metadata.col_sum, n_total_rows)

    return ragged_metadata, gather_indx, scatter_indx, gate_scal


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


def _routing_from_topk(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype | None = None,
) -> tuple[RaggedTensorMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be rank-2, got {tuple(topk_ids.shape)}")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights and topk_ids must have the same shape, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")

    flat_ids = topk_ids.reshape(-1).to(torch.long)
    valid = flat_ids >= 0
    safe_ids = torch.where(valid, flat_ids, flat_ids.new_zeros(()))
    sort_order = torch.argsort(safe_ids, stable=True)

    top_k = topk_ids.shape[1]
    gather_indx = (sort_order // top_k).to(torch.int32)
    scatter_indx = sort_order.to(torch.int32)
    gate_scal = topk_weights.reshape(-1)[sort_order]
    gate_scal = torch.where(valid[sort_order], gate_scal, torch.zeros_like(gate_scal))
    if dtype is not None and gate_scal.dtype != dtype:
        gate_scal = gate_scal.to(dtype)

    col_sum = torch.zeros((num_experts,), dtype=torch.int32, device=safe_ids.device)
    col_sum.scatter_add_(
        0,
        safe_ids,
        torch.ones_like(safe_ids, dtype=torch.int32),
    )
    n_total_rows = int(sort_order.numel())
    ragged_metadata = make_ragged_tensor_metadata(col_sum, n_total_rows)

    return ragged_metadata, gather_indx, scatter_indx, gate_scal


def _local_topk_for_ep(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    ep_size = int(getattr(w, "ep_size", 1))
    if ep_size <= 1:
        return topk_weights, topk_ids, int(getattr(w, "num_experts"))

    num_local_experts = int(getattr(w, "num_local_experts"))
    expert_offset = int(getattr(w, "ep_rank", 0)) * num_local_experts
    local_ids = topk_ids - expert_offset
    local_mask = (local_ids >= 0) & (local_ids < num_local_experts)
    local_weights = torch.where(local_mask, topk_weights, torch.zeros_like(topk_weights))
    local_ids = torch.where(local_mask, local_ids, topk_ids.new_full((), -1))
    return local_weights, local_ids, num_local_experts


@register_kernel(
    "moe",
    "process_weights",
    name="triton_mxfp4_moe_process_weights",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=frozenset({format_signature()}),
    traits={"weight_dtype": frozenset({"mxfp4"})},
    priority=Priority.PORTABLE,
)
def triton_mxfp4_moe_process_weights(plan: dict, w: torch.nn.Module):
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

    if plan.get("internal_activation_dtype", None) == "fp8":
        assert hasattr(w, "w13_input_scale") and hasattr(w, "w2_input_scale")
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
    else:
        w13_lhs = InFlexData()
        w2_lhs = InFlexData()
        out_dtype = None

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
    # Free original weights (replaced by shuffled versions)
    del w.w13_weight
    del w.w2_weight
    torch.cuda.empty_cache()


@register_kernel(
    "moe",
    "process_weights",
    name="triton_kernels_mxfp4_moe_process_weights",
    solution="triton_kernels",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=frozenset({format_signature()}),
    traits={"weight_dtype": frozenset({"mxfp4"})},
    priority=Priority.PERFORMANT,
)
def triton_kernels_mxfp4_moe_process_weights(plan: dict, w: torch.nn.Module):
    num_warps = 8
    if layout is None:
        raise RuntimeError("triton_kernels backend unavailable")

    value_layout = layout.make_default_matmul_mxfp4_w_layout(mx_axis=-2)
    scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=-2, num_warps=num_warps
    )
    if platform.is_amd:
        opt_flags.update_opt_flags_constraints({"block_k": 256})

    if hasattr(w, "w13_weight_bias"):
        w.w13_weight_bias = torch.nn.Parameter(
            w.w13_weight_bias.to(torch.float32), requires_grad=False
        )
    if hasattr(w, "w2_weight_bias"):
        w.w2_weight_bias = torch.nn.Parameter(
            w.w2_weight_bias.to(torch.float32), requires_grad=False
        )

    w13_quant = w.w13_weight.transpose(-2, -1)
    w13_scale_tensor = w.w13_weight_scale.transpose(-2, -1)
    w13_weight = convert_layout(wrap_torch_tensor(w13_quant, dtype=FP4), value_layout)
    w13_flex = InFlexData()
    w13_scale = convert_layout(wrap_torch_tensor(w13_scale_tensor), scale_layout)

    w2_quant = w.w2_weight.transpose(-2, -1)
    w2_scale_tensor = w.w2_weight_scale.transpose(-2, -1)
    w2_weight = convert_layout(wrap_torch_tensor(w2_quant, dtype=FP4), value_layout)
    w2_flex = InFlexData()
    w2_scale = convert_layout(wrap_torch_tensor(w2_scale_tensor), scale_layout)

    if hasattr(w, "w13_input_scale") and hasattr(w, "w2_input_scale"):
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
        out_dtype = torch.bfloat16
    else:
        w13_lhs = InFlexData()
        w2_lhs = InFlexData()
        out_dtype = torch.bfloat16 if _uses_dynamic_mxfp4_activations(w) else None

    w.w13_precision_config = PrecisionConfig(
        flex_ctx=FlexCtx(lhs_data=w13_lhs, rhs_data=w13_flex),
        b_mx_scale=w13_scale,
        b_microblock_size=MXFP4_BLOCK,
        out_dtype=out_dtype,
    )
    w.w2_precision_config = PrecisionConfig(
        flex_ctx=FlexCtx(lhs_data=w2_lhs, rhs_data=w2_flex),
        b_mx_scale=w2_scale,
        b_microblock_size=MXFP4_BLOCK,
        out_dtype=out_dtype,
    )
    w.w13_weight_triton_tensor = w13_weight
    w.w2_weight_triton_tensor = w2_weight
    _release_parameter(w, "w13_weight")
    _release_parameter(w, "w13_weight_scale")
    _release_parameter(w, "w2_weight")
    _release_parameter(w, "w2_weight_scale")
    return None


@register_kernel(
    "moe",
    "apply",
    name="triton_kernels_mxfp4_moe_apply",
    solution="triton_kernels",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
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
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({True}),
    },
    priority=Priority.PERFORMANT,
)
@register_kernel(
    "moe",
    "apply",
    name="triton_kernels_mxfp4_precomputed_moe_apply",
    solution="triton_kernels",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures(
        "x",
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"silu"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"fp8", "input"}),
        "supports_bias": frozenset({True}),
    },
    priority=Priority.SPECIALIZED + 2,
)
@register_kernel(
    "moe",
    "apply",
    name="triton_kernels_mxfp4_ep_precomputed_moe_apply",
    solution="triton_kernels",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures(
        "x",
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"silu"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"fp8", "input"}),
        "supports_bias": frozenset({True}),
    },
    priority=Priority.SPECIALIZED + 1,
)
@register_kernel(
    "moe",
    "apply",
    name="triton_kernels_mxfp4_fp8_activation_moe_apply",
    solution="triton_kernels",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
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
    priority=Priority.PERFORMANT - 1,
)
def triton_kernels_mxfp4_moe_apply(
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
    del enable_pdl
    top_k = getattr(w, "top_k")
    if topk_weights is not None or topk_ids is not None:
        if topk_weights is None or topk_ids is None:
            raise ValueError("topk_weights and topk_ids must be provided together")
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
    else:
        ragged_metadata, gather_indx, scatter_indx, gate_scal = _routing(
            router_logits,
            top_k,
            sm_first=False,
            dtype=router_logits.dtype,
        )

    swiglu_arg = getattr(w, "swiglu_arg", None)
    activation = None
    if swiglu_arg is not None:
        activation = FusedActivation(
            FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
            (swiglu_arg.alpha, swiglu_arg.limit),
        )

    use_dynamic_mxfp4 = _uses_dynamic_mxfp4_activations(w)
    w13_precision_config = getattr(w, "w13_precision_config", None)
    w2_precision_config = getattr(w, "w2_precision_config", None)
    if hasattr(w, "w13_act_scale"):
        gemm1_input = tokenspeed_kernel.quantize_fp8(
            x,
            scale=w.w13_act_scale,
            solution="triton",
        )
    elif use_dynamic_mxfp4:
        gemm1_input, gemm1_scale = _quantize_mxfp4_activation(x)
        w13_precision_config = _with_activation_mx_scale(
            w13_precision_config,
            gemm1_scale,
        )
    else:
        gemm1_input = x

    intermediate_cache = _matmul(
        gemm1_input,
        w.w13_weight_triton_tensor,
        getattr(w, "w13_weight_bias", None),
        a_ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        precision_config=w13_precision_config,
        fused_activation=activation,
    )
    if activation is None:
        intermediate_cache = _silu_gate_up(
            intermediate_cache,
            output_dtype=x.dtype,
        )

    if hasattr(w, "w2_act_scale"):
        gemm2_input = tokenspeed_kernel.quantize_fp8(
            intermediate_cache,
            scale=w.w2_act_scale,
            solution="triton",
        )
    elif use_dynamic_mxfp4:
        gemm2_input, gemm2_scale = _quantize_mxfp4_activation(intermediate_cache)
        w2_precision_config = _with_activation_mx_scale(
            w2_precision_config,
            gemm2_scale,
        )
    else:
        gemm2_input = intermediate_cache

    return _matmul(
        gemm2_input,
        w.w2_weight_triton_tensor,
        getattr(w, "w2_weight_bias", None),
        a_ragged_metadata=ragged_metadata,
        precision_config=w2_precision_config,
        scatter_indx=scatter_indx,
        betas=None,
        gammas=gate_scal,
        n_tokens=router_logits.shape[0],
        n_expts_act=top_k,
    )


@register_kernel(
    "moe",
    "apply",
    name="triton_mxfp4_moe_apply",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
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
        "internal_activation_dtype": frozenset({"input", "fp8"}),
        "supports_bias": frozenset({True}),
    },
    priority=Priority.PORTABLE,
)
def triton_mxfp4_moe_apply(
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
    n_tokens = router_logits.shape[0]

    ragged_metadata, gather_indx, scatter_indx, gate_scal = _routing(
        router_logits,
        top_k,
        sm_first=False,
        dtype=router_logits.dtype,
    )

    w13_weight = w.w13_weight_triton_tensor
    w2_weight = w.w2_weight_triton_tensor
    w13_bias = getattr(w, "w13_weight_bias", None)
    w2_bias = getattr(w, "w2_weight_bias", None)
    w13_pc = getattr(w, "w13_precision_config", None)
    w2_pc = getattr(w, "w2_precision_config", None)

    gemm1_alpha = swiglu_arg.alpha if swiglu_arg else 1.702
    gemm1_clamp = swiglu_arg.limit if swiglu_arg else 7.0

    act = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (gemm1_alpha, gemm1_clamp),
    )

    if plan.get("internal_activation_dtype", None) == "fp8":
        w13_act_scale = getattr(w, "w13_act_scale", None)
        gemm1_input = fp8_quantize(
            x,
            scale=w13_act_scale,
        )
    else:
        gemm1_input = x

    # First GEMM: gate_up projection with fused activation
    with _maybe_lds_guard(gemm1_input, w13_weight, w13_pc):
        intermediate_cache = matmul(
            gemm1_input,
            w13_weight,
            w13_bias,
            a_ragged_metadata=ragged_metadata,
            gather_indx=gather_indx,
            precision_config=w13_pc,
            fused_activation=act,
        )

    if plan.get("internal_activation_dtype", None) == "fp8":
        w2_act_scale = getattr(w, "w2_act_scale", None)
        gemm2_input = fp8_quantize(
            intermediate_cache,
            scale=w2_act_scale,
        )
    else:
        gemm2_input = intermediate_cache

    # Second GEMM: down projection with scatter (combine)
    # gammas applies the routing weights (expert contribution weights)
    with _maybe_lds_guard(gemm2_input, w2_weight, w2_pc):
        out = matmul(
            gemm2_input,
            w2_weight,
            w2_bias,
            a_ragged_metadata=ragged_metadata,
            scatter_indx=scatter_indx,
            precision_config=w2_pc,
            gammas=gate_scal,
        )
    return out.view(n_tokens, top_k, out.shape[-1]).sum(dim=1)
