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

from contextlib import contextmanager

import tokenspeed_kernel
import torch
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton
from tokenspeed_kernel.platform import current_platform
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

platform = current_platform()
process_weight_signature = frozenset({format_signature()})
apply_signatures = format_signatures(
    "x",
    "dense",
    {torch.float16, torch.bfloat16},
)


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
        if out.ndim == 3:
            out = out.sum(dim=0)
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


@register_kernel(
    "moe_v2",
    "process_weights",
    name="triton_kernels_mxfp4_moe_v2_process_weights",
    solution="triton_kernels",
    signatures=process_weight_signature,
    traits={"weight_dtype": frozenset({"mxfp4"})},
    priority=Priority.PERFORMANT,
)
def triton_kernels_mxfp4_moe_process_weights(plan: dict, w: torch.nn.Module):
    block_size = 32
    num_warps = 8
    if layout is None:
        raise RuntimeError("triton_kernels backend unavailable")

    value_layout = layout.make_default_matmul_mxfp4_w_layout(mx_axis=-2)
    scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=-2, num_warps=num_warps
    )
    if platform.is_blackwell:
        opt_flags.update_opt_flags_constraints(
            {
                "is_persistent": True,
                "epilogue_subtile": 1,
            }
        )
    elif platform.is_hopper:
        opt_flags.update_opt_flags_constraints({"split_k": 1})
    elif platform.is_amd:
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
        out_dtype = None

    w.w13_precision_config = PrecisionConfig(
        flex_ctx=FlexCtx(lhs_data=w13_lhs, rhs_data=w13_flex),
        b_mx_scale=w13_scale,
        b_microblock_size=block_size,
        out_dtype=out_dtype,
    )
    w.w2_precision_config = PrecisionConfig(
        flex_ctx=FlexCtx(lhs_data=w2_lhs, rhs_data=w2_flex),
        b_mx_scale=w2_scale,
        b_microblock_size=block_size,
        out_dtype=out_dtype,
    )
    w.w13_weight_triton_tensor = w13_weight
    w.w2_weight_triton_tensor = w2_weight
    return None


@register_kernel(
    "moe_v2",
    "apply",
    name="triton_kernels_mxfp4_moe_v2_apply",
    solution="triton_kernels",
    signatures=apply_signatures,
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "support_routing": frozenset({True}),
        "supports_deferred_finalize": frozenset({False}),
    },
    priority=Priority.PERFORMANT,
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
):
    top_k = getattr(w, "top_k")
    ragged_metadata, gather_indx, scatter_indx, gate_scal = _routing(
        router_logits,
        top_k,
        sm_first=False,
        dtype=router_logits.dtype,
    )

    swiglu_arg = getattr(w, "swiglu_arg", None)
    swiglu_alpha = swiglu_arg.alpha if swiglu_arg else 1.702
    swiglu_limit = swiglu_arg.limit if swiglu_arg else 7.0
    activation = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (swiglu_alpha, swiglu_limit),
    )

    if hasattr(w, "w13_act_scale"):
        gemm1_input = tokenspeed_kernel.quantize_fp8(
            x,
            scale=w.w13_act_scale,
            solution="triton",
        )
    else:
        gemm1_input = x

    intermediate_cache = _matmul(
        gemm1_input,
        w.w13_weight_triton_tensor,
        getattr(w, "w13_weight_bias", None),
        a_ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        precision_config=getattr(w, "w13_precision_config", None),
        fused_activation=activation,
    )

    if hasattr(w, "w2_act_scale"):
        gemm2_input = tokenspeed_kernel.quantize_fp8(
            intermediate_cache,
            scale=w.w2_act_scale,
            solution="triton",
        )
    else:
        gemm2_input = intermediate_cache

    return _matmul(
        gemm2_input,
        w.w2_weight_triton_tensor,
        getattr(w, "w2_weight_bias", None),
        precision_config=getattr(w, "w2_precision_config", None),
        scatter_indx=scatter_indx,
        betas=None,
        gammas=gate_scal,
        n_tokens=router_logits.shape[0],
        n_expts_act=top_k,
    )
