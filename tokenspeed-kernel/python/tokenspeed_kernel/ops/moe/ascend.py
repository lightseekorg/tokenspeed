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

"""Ascend MoE kernel registrations."""

import logging
import time

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.moe import (
        ascend_bf16_distributed_precomputed_moe_apply as _distributed_moe_apply,
    )
    from tokenspeed_kernel_npu.ops.moe import (
        ascend_bf16_precomputed_moe_apply as _moe_apply,
    )
    from tokenspeed_kernel_npu.ops.moe import (
        ascend_bf16_process_moe_weights as _process_weights,
    )
    from tokenspeed_kernel_npu.ops.moe import (
        ascend_int8_precomputed_moe_apply as _int8_moe_apply,
    )
    from tokenspeed_kernel_npu.ops.moe import (
        ascend_int8_process_moe_weights as _int8_process_weights,
    )
    from tokenspeed_kernel_npu.ops.moe import (
        ascend_softmax_bias_topk as _softmax_bias_topk,
    )
    from tokenspeed_kernel_npu.ops.w4a4 import (
        ascend_int4_precomputed_moe_apply as _int4_moe_apply,
    )
    from tokenspeed_kernel_npu.ops.w4a4 import (
        ascend_int4_process_moe_weights as _int4_process_weights,
    )

    _CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))

    @register_kernel(
        "moe",
        "softmax_bias_topk",
        name="ascend_softmax_bias_topk",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(
            "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
        ),
        priority=Priority.SPECIALIZED,
        features={"ascend", "latency"},
    )
    def softmax_bias_topk(**kwargs):
        return _softmax_bias_topk(**kwargs)

    @register_kernel(
        "moe",
        "apply",
        name="ascend_bf16_precomputed_moe_apply",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"unquant"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({32}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
        features={"ascend", "cuda_graph"},
        weight_preprocessor=_process_weights,
    )
    def moe_apply(**kwargs):
        return _moe_apply(**kwargs)

    @register_kernel(
        "moe",
        "apply",
        name="ascend_int8_precomputed_moe_apply",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"int8"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({32}),
            "internal_activation_dtype": frozenset({"int8"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
        features={"ascend", "npu_graph", "w8a8", "nz"},
        weight_preprocessor=_int8_process_weights,
    )
    def int8_moe_apply(**kwargs):
        return _int8_moe_apply(**kwargs)

    @register_kernel(
        "moe",
        "apply",
        name="ascend_int4_precomputed_moe_apply",
        solution="flash_npu_chain",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"int4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({64}),
            "internal_activation_dtype": frozenset({"int4"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
        features={"ascend", "npu_graph", "w4a4", "flatquant", "nz"},
        weight_preprocessor=_int4_process_weights,
    )
    def int4_moe_apply(**kwargs):
        return _int4_moe_apply(**kwargs)

    @register_kernel(
        "moe",
        "apply",
        name="ascend_bf16_distributed_precomputed_moe_apply",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"unquant"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({True}),
            "ispp_alignment": frozenset({32}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
        features={"ascend", "npu_graph", "all_to_all"},
        weight_preprocessor=_process_weights,
    )
    def distributed_moe_apply(**kwargs):
        return _distributed_moe_apply(**kwargs)


__all__ = ["moe_apply", "softmax_bias_topk"]


def ascend_v41_prepare_weights(plan: dict, w: torch.nn.Module):
    import torch_npu
    from tokenspeed_kernel_npu.ops._native import require_v41_libraries
    from tokenspeed_kernel_npu.ops.dsv41_checkpoint import dequant_checkpoint
    from tokenspeed_kernel_npu.ops.dsv41_moe import (
        AscendGroupedLinear,
        AscendV41GroupedMoE,
    )

    if w.tp_size != 1 or w.hidden_size != 5120 or w.intermediate_size != 2304:
        raise ValueError("Ascend V4.1 requires MoE TP1, hidden5120, intermediate2304")
    if w.w13_input_layout != "concatenated" or w.swiglu_beta not in (None, 0.0):
        raise ValueError("Ascend V4.1 requires concatenated standard SwiGLU")
    if (
        w.swiglu_arg is None
        or w.swiglu_arg.alpha not in (None, 1.0)
        or w.swiglu_arg.limit != 10.0
    ):
        raise ValueError("Ascend V4.1 requires the checkpoint's SwiGLU clamp=10")
    mode = {"ascend_w4a4": "w4a4", "ascend_bf16": "bf16"}[plan["solution"]]
    require_v41_libraries(w4a4=mode == "w4a4")
    started = time.monotonic()
    if mode == "w4a4" and w.npu_flatquant_config is None:
        raise ValueError(
            "ascend_w4a4 requires a preconverted --npu-flatquant-config manifest"
        )
    if mode == "bf16" and w.npu_flatquant_config is not None:
        raise ValueError(
            "BF16 must not silently ignore a supplied FlatQuant configuration"
        )
    device = w.w13_weight.device
    linears = []
    for projection, (packed, scales) in zip(
        ("w13", "w2"),
        ((w.w13_weight, w.w13_weight_scale), (w.w2_weight, w.w2_weight_scale)),
        strict=True,
    ):
        if mode == "w4a4":
            from tokenspeed_kernel_npu.ops.dsv41_w4a4_weights import load_projection

            linear = load_projection(
                w.npu_flatquant_config,
                layer_index=w.layer_index,
                ep_rank=w.ep_rank,
                ep_size=w.ep_size,
                projection=projection,
                shape=(packed.shape[0], packed.shape[1], packed.shape[2] * 2),
                device=device,
            )
        else:
            decoded = torch.stack(
                [
                    dequant_checkpoint(p.cpu(), s.cpu(), "mxfp4").bfloat16()
                    for p, s in zip(packed, scales, strict=True)
                ]
            )
            weight = torch_npu.npu_format_cast(decoded.to(device), 29).transpose(1, 2)
            linear = AscendGroupedLinear(
                "bf16", weight, None, None, decoded.shape[-1], decoded.shape[-2]
            )
            del decoded
        linears.append(linear)
    w._moe_backend_state = AscendV41GroupedMoE(linears[0], linears[1], 10.0)
    # Replace checkpoint storage with the prepared tensors; do not retain a
    # second full-model copy of the MXFP4 source on device.
    w.w13_weight = torch.nn.Parameter(linears[0].weight, requires_grad=False)
    w.w2_weight = torch.nn.Parameter(linears[1].weight, requires_grad=False)
    for name, linear in zip(
        ("w13_weight_scale", "w2_weight_scale"), linears, strict=True
    ):
        scale = linear.weight_scale
        if scale is None:
            scale = torch.empty(0, dtype=torch.uint8, device=device)
        w.register_parameter(name, torch.nn.Parameter(scale, requires_grad=False))
    logging.getLogger(__name__).info(
        f"Ascend {mode} weights ready: layer={w.layer_index}, EP rank={w.ep_rank}, "
        f"experts={w.num_local_experts}, seconds={time.monotonic() - started:.2f}"
    )


def _ascend_v41_apply(
    plan,
    x,
    w,
    router_logits,
    topk_weights,
    topk_ids,
    num_tokens_global,
    max_num_tokens_per_gpu,
    do_finalize,
    enable_pdl,
):
    if not do_finalize or enable_pdl:
        raise ValueError("Ascend MoE requires finalized output and disabled CUDA PDL")
    if not isinstance(x, torch.Tensor) or topk_weights is None or topk_ids is None:
        raise ValueError("Ascend MoE requires dense input and top-k IDs/weights")
    if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != x.shape[0]:
        raise ValueError("MoE top-k shape mismatch")
    if w._moe_backend_state is None:
        raise RuntimeError("Ascend MoE weights were not prepared")
    from tokenspeed_kernel_npu.ops.dsv41_moe import routed_moe_native

    return routed_moe_native(
        x,
        topk_ids,
        topk_weights,
        w._moe_backend_state,
        expert_start=w.ep_rank * w.num_local_experts,
        num_experts=w.num_experts,
    )


def _register(solution, activation_dtype):
    return register_kernel(
        "moe",
        "apply",
        name=solution + "_v41_apply",
        solution=solution,
        weight_preprocessor=ascend_v41_prepare_weights,
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"swiglu"}),
            "swiglu_form": frozenset({"standard"}),
            "activation_clamped": frozenset({True}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "a2a_backend": frozenset({None, "none"}),
            "combine_order": frozenset({"rank"}),
            "expert_id_repeats": frozenset({True}),
            "supports_bias": frozenset({False}),
            "internal_activation_dtype": frozenset({activation_dtype}),
            "hidden": frozenset({5120}),
            "ispp": frozenset({2304}),
            "persistent_workspace": frozenset({False}),
        },
        priority=Priority.PORTABLE,
    )


if current_platform().is_npu:
    _register("ascend_w4a4", "int4")(_ascend_v41_apply)
    _register("ascend_bf16", "input")(_ascend_v41_apply)


@register_kernel(
    "moe",
    "topk",
    name="ascend_sqrt_softplus_topk",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
    signatures=format_signatures(
        "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    traits={
        "experts": frozenset({384}),
        "top_k": frozenset({6}),
        "routing_kind": frozenset({"plain", "bias", "hash"}),
        "score_function": frozenset({"sqrt_softplus"}),
    },
    priority=Priority.PERFORMANT,
)
def ascend_sqrt_softplus_topk(
    router_logits,
    top_k,
    renormalize,
    correction_bias,
    hash_indices_table,
    input_ids,
    need_scores,
    routed_scaling_factor,
    weights_dtype,
):
    from tokenspeed_kernel_npu.ops.dsv41_router import ascend_sqrt_softplus_topk as impl

    return impl(
        router_logits,
        top_k,
        renormalize,
        correction_bias,
        hash_indices_table,
        input_ids,
        need_scores,
        routed_scaling_factor,
        weights_dtype,
    )
