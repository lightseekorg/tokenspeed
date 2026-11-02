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

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.moe import packed_projection_weight_view
from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

platform = current_platform()

GLUON_FP8_CHANNEL_APPLY = "gluon_fp8_channel_bf16_precomputed_moe_apply"


@register_kernel(
    "moe",
    "apply",
    name=GLUON_FP8_CHANNEL_APPLY,
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"fp8_channel"}),
        "activation": frozenset({"silu", "swiglu"}),
        "swiglu_form": frozenset({"standard"}),
        "activation_clamped": frozenset({False}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ep_size": frozenset({1}),
        "ispp": frozenset({320, 640}),
        "hidden": frozenset({5120}),
        "num_experts": frozenset({256}),
        "top_k": frozenset({8}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
        "expert_id_repeats": frozenset({False}),
    },
    priority=Priority.SPECIALIZED,
)
def gluon_fp8_channel_bf16_precomputed_moe_apply(
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
    """TP4/TP8 EP1 experts: FP8 storage, BF16 MFMA, FP32 accumulation.

    Accepts only the default precision policy.
    Uses pre-existing grouped Gluon stages, not native-FP8 projection kernels.
    The resident codes/scales and unweighted-BF16-before-FP32-combine boundary
    match the channel expert contract; FP32 accumulation order can differ from
    native FP8.
    """
    if plan.get("precision_policy", "default") != "default":
        raise ValueError(
            "Gluon BF16-compute channel experts require default precision_policy"
        )
    if enable_pdl:
        raise ValueError("Gluon channel experts do not support enable_pdl=True")
    if (
        plan.get("combine_order", "rank") != "rank"
        or plan.get("output_scale", 1.0) != 1.0
        or getattr(w, "ep_size", 1) != 1
        or getattr(w, "ep_rank", 0) != 0
    ):
        raise ValueError(
            "Gluon channel experts require EP1 rank-local output and unit output scale"
        )
    intermediate = w.w2_weight.shape[-1]
    if (
        x.ndim != 2
        or x.shape[1] != 5120
        or intermediate not in (320, 640)
        or w.w13_weight.shape != (256, 2 * intermediate, 5120)
        or w.w2_weight.shape != (256, 5120, intermediate)
        or topk_ids is None
        or topk_ids.shape != (x.shape[0], 8)
    ):
        raise ValueError("Gluon channel experts require E256 H5120 I320/I640 top8")
    from tokenspeed_kernel.ops.moe._triton.common import _combine
    from tokenspeed_kernel.ops.moe._triton.fp8_channel import _validate_fp8_channel_moe
    from tokenspeed_kernel.ops.moe._triton.fp8_channel_fused import _quantize_routes
    from tokenspeed_kernel.ops.quantization import quantize_fp8
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.exact_mfma import (
        prepare_gluon_fp8_channel_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.stage2_kernel import (
        launch_gluon_fp8_channel_stage2_gfx950,
    )

    _validate_fp8_channel_moe(plan, x, w, topk_weights, topk_ids, do_finalize)
    if x.shape[0] == 0:
        return torch.empty_like(x)
    codes, scales = quantize_fp8(
        x, granularity="token", scale_encoding="float32", enable_pdl=False
    )
    activated, routing, block_m = prepare_gluon_fp8_channel_gfx950(
        codes,
        scales,
        w.w13_weight,
        w.w13_weight_scale,
        topk_ids,
        topk_weights,
    )
    route_codes, route_scales = _quantize_routes(activated)
    partials = torch.zeros((x.shape[0], 8, 5120), dtype=torch.bfloat16, device=x.device)
    sorted_ids, sorted_experts, sorted_weights, num_valid = routing
    launch_gluon_fp8_channel_stage2_gfx950(
        route_codes,
        w.w2_weight,
        route_scales.view(-1, 1),
        w.w2_weight_scale,
        sorted_ids,
        sorted_experts,
        sorted_weights,
        num_valid,
        partials,
        block_m=block_m,
    )
    output = torch.empty_like(x)
    _combine(partials.view(-1, 5120), topk_weights, output, None, 0)
    return output


# ===-----------------------------------------------------------------------===#
# Routing: softmax top-k (AMD gfx950)
# ===-----------------------------------------------------------------------===#


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.moe import (
        launch_gluon_softmax_topk_gfx950 as _softmax_topk_impl,
    )

    @register_kernel(
        "moe",
        "softmax_topk",
        name="gluon_softmax_topk_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
        ),
        traits={
            "experts": frozenset(range(1, 1025)),
            "topk": frozenset(range(1, 33)),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_softmax_topk_gfx950(**kwargs):
        return _softmax_topk_impl(**kwargs)


# ===-----------------------------------------------------------------------===#
# Routing: sqrt-softplus top-k (AMD gfx950)
# ===-----------------------------------------------------------------------===#


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.moe import (
        launch_gluon_router_sqrt_softplus_topk_gemv_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe import (
        launch_gluon_sqrt_softplus_topk_gfx950 as _topk_impl,
    )

    @register_kernel(
        "moe",
        "topk",
        name="gluon_sqrt_softplus_topk_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            format_signature(router_logits=dense_tensor_format(dtype))
            for dtype in (torch.float16, torch.bfloat16, torch.float32)
        ),
        traits={
            "tokens": frozenset({1, 2, 3, 4, 5, 6}),
            "experts": frozenset({256, 384}),
            "top_k": frozenset({6}),
            "renormalize": frozenset({False, True}),
            "routing_kind": frozenset({"plain", "bias", "hash"}),
            "score_function": frozenset({"sqrt_softplus"}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_sqrt_softplus_topk_gfx950(
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        correction_bias: torch.Tensor | None,
        hash_indices_table: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        need_scores: bool,
        routed_scaling_factor: float,
        weights_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights, ids, scores = _topk_impl(
            router_logits,
            top_k,
            renormalize,
            correction_bias,
            hash_indices_table,
            input_ids,
            need_scores,
            routed_scaling_factor,
        )
        return weights.to(weights_dtype), ids, scores

    @register_kernel(
        "moe",
        "router_topk",
        name="gluon_router_sqrt_softplus_topk_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    hidden_states=dense_tensor_format(torch.bfloat16),
                    gate_weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        traits={
            # Measured to beat the decode GEMM plus top-k up to 128 rows.
            "tokens": frozenset(range(1, 129)),
            "experts": frozenset({256, 384}),
            "hidden_alignment": frozenset({512}),
            "top_k": frozenset({6}),
            "routing_kind": frozenset({"plain", "bias", "hash"}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_router_sqrt_softplus_topk_gfx950(
        hidden_states: torch.Tensor,
        gate_weight: torch.Tensor,
        top_k: int,
        renormalize: bool,
        correction_bias: torch.Tensor | None,
        hash_indices_table: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Split-K GEMV partials; the top-k kernel sums them in order.
        partials = launch_gluon_router_sqrt_softplus_topk_gemv_gfx950(
            hidden_states, gate_weight
        )
        weights, ids, _ = _topk_impl(
            partials,
            top_k,
            renormalize,
            correction_bias,
            hash_indices_table,
            input_ids,
            False,
            routed_scaling_factor,
        )
        return weights, ids


# ===-----------------------------------------------------------------------===#
# Routing: sqrt-softplus top-k (AMD gfx1250)
# ===-----------------------------------------------------------------------===#


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.gfx1250.moe.router_gemv import (
        launch_gluon_router_sqrt_softplus_topk_gemv_gfx1250,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.moe.sqrt_softplus_topk import (
        launch_gluon_sqrt_softplus_topk_gfx1250 as _topk_gfx1250_impl,
    )

    @register_kernel(
        "moe",
        "topk",
        name="gluon_sqrt_softplus_topk_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            format_signature(router_logits=dense_tensor_format(dtype))
            for dtype in (torch.float16, torch.bfloat16, torch.float32)
        ),
        # Measured to beat the Triton top-k at every batch from 1 to 16384 rows,
        # so unlike gfx950 it is not limited to decode batches.
        traits={
            "experts": frozenset({256, 384}),
            "top_k": frozenset({6}),
            "renormalize": frozenset({False, True}),
            "routing_kind": frozenset({"plain", "bias", "hash"}),
            "score_function": frozenset({"sqrt_softplus"}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_sqrt_softplus_topk_gfx1250(
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        correction_bias: torch.Tensor | None,
        hash_indices_table: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        need_scores: bool,
        routed_scaling_factor: float,
        weights_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights, ids, scores = _topk_gfx1250_impl(
            router_logits,
            top_k,
            renormalize,
            correction_bias,
            hash_indices_table,
            input_ids,
            need_scores,
            routed_scaling_factor,
        )
        return weights.to(weights_dtype), ids, scores

    @register_kernel(
        "moe",
        "router_topk",
        name="gluon_router_sqrt_softplus_topk_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    hidden_states=dense_tensor_format(torch.bfloat16),
                    gate_weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        traits={
            # Measured to beat the FP32 router GEMM plus top-k up to 256 rows;
            # the GEMV's serial row tiles lose badly beyond that.
            "tokens": frozenset(range(1, 257)),
            "experts": frozenset({256, 384}),
            "hidden_alignment": frozenset({512}),
            "top_k": frozenset({6}),
            "routing_kind": frozenset({"plain", "bias", "hash"}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_router_sqrt_softplus_topk_gfx1250(
        hidden_states: torch.Tensor,
        gate_weight: torch.Tensor,
        top_k: int,
        renormalize: bool,
        correction_bias: torch.Tensor | None,
        hash_indices_table: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Split-K GEMV partials; the top-k kernel sums them in order.
        partials = launch_gluon_router_sqrt_softplus_topk_gemv_gfx1250(
            hidden_states, gate_weight
        )
        weights, ids, _ = _topk_gfx1250_impl(
            partials,
            top_k,
            renormalize,
            correction_bias,
            hash_indices_table,
            input_ids,
            False,
            routed_scaling_factor,
        )
        return weights, ids


# ===-----------------------------------------------------------------------===#
# Routing: sigmoid-bias top-k (AMD gfx950, gfx1250)
# ===-----------------------------------------------------------------------===#


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.routing import (
        invoke_sigmoid_bias_topk_route_gluon,
        invoke_sigmoid_bias_topk_route_prefill_gluon,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.routing import (
        invoke_sigmoid_bias_topk_route_prefill_gluon as invoke_sigmoid_bias_topk_route_prefill_gfx1250,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.moe.sigmoid_bias_topk import (
        launch_gluon_sigmoid_bias_topk_route_gfx1250,
    )

    @register_kernel(
        "moe",
        "sigmoid_bias_topk",
        name="gluon_sigmoid_bias_topk_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
        ),
        priority=Priority.SPECIALIZED,
    )
    def gluon_sigmoid_bias_topk_gfx950(
        *,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor,
        topk: int,
        routed_scaling_factor: float,
        normalize_topk_weights: bool,
        weights_dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decode_supported = router_logits.shape[0] * topk <= 128
        route = (
            invoke_sigmoid_bias_topk_route_gluon
            if decode_supported
            and (
                router_logits.shape[0] == 1
                or router_logits.dtype != torch.float32
                or topk > 16
            )
            else invoke_sigmoid_bias_topk_route_prefill_gluon
        )
        topk_ids, topk_weights = route(
            router_logits,
            correction_bias,
            topk,
            routed_scaling_factor=routed_scaling_factor,
            normalize_topk_weights=normalize_topk_weights,
        )
        # The gluon route has no output-dtype argument to store through.
        return topk_weights.to(weights_dtype), topk_ids

    @register_kernel(
        "moe",
        "sigmoid_bias_topk",
        name="gluon_sigmoid_bias_topk_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures("router_logits", "dense", {torch.float32}),
        priority=Priority.SPECIALIZED,
        traits={
            # Keep the Gluon path available for explicit selection below 513.
            "tokens": range(2, 1 << 31),
            "experts": frozenset({896}),
            "topk": frozenset({16}),
        },
    )
    def gluon_sigmoid_bias_topk_gfx1250(
        *,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor,
        topk: int,
        routed_scaling_factor: float,
        normalize_topk_weights: bool,
        weights_dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_ids, topk_weights = invoke_sigmoid_bias_topk_route_prefill_gfx1250(
            router_logits,
            correction_bias,
            topk,
            routed_scaling_factor=routed_scaling_factor,
            normalize_topk_weights=normalize_topk_weights,
        )
        return topk_weights.to(weights_dtype), topk_ids

    @register_kernel(
        "moe",
        "sigmoid_bias_topk",
        name="gluon_sigmoid_bias_topk_route_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
        ),
        priority=Priority.PERFORMANT,
        traits={
            "experts": frozenset(range(1, 1025)),
            "topk": frozenset(range(1, 17)),
        },
    )
    def gluon_sigmoid_bias_topk_route_gfx1250(
        *,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor,
        topk: int,
        routed_scaling_factor: float,
        normalize_topk_weights: bool,
        weights_dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids = launch_gluon_sigmoid_bias_topk_route_gfx1250(
            router_logits=router_logits,
            correction_bias=correction_bias,
            topk=topk,
            routed_scaling_factor=routed_scaling_factor,
            normalize_topk_weights=normalize_topk_weights,
            weights_dtype=weights_dtype,
        )
        return topk_weights, topk_ids


# ===-----------------------------------------------------------------------===#
# BF16 experts (AMD gfx950)
# ===-----------------------------------------------------------------------===#


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16 import gluon_bf16_moe

    @register_kernel(
        "moe",
        "apply",
        name="gluon_bf16_precomputed_moe_apply",
        solution="gluon",
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"unquant"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False}),
            "supports_all_to_all_ep": frozenset({False}),
            # Stage 1 tiles I_r at BLOCK_N=64 or 128. The optional warp-decode
            # shortcut still needs I_r % 256 == 0 and is gated inside the AMD
            # entry point.
            "ispp_alignment": frozenset({64}),
            # Stage 2 tiles the hidden size at BLOCK_N=256.
            "hidden_alignment": frozenset({256}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        # gluon is narrowly gated to gfx950.
        priority=Priority.SPECIALIZED,
    )
    def gluon_bf16_precomputed_moe_apply(
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
        """Unquantized bf16 two-stage fused MoE (gfx950, precomputed top-k).

        Args:
            plan: MoE execution plan from ``moe_plan`` (unused here).
            x: Hidden states ``[tokens, hidden]`` bf16.
            w: Weight module exposing ``w13_weight`` ``[E, 2*I, D]`` (gate rows
                ``[0:I]``, up rows ``[I:2I]``) and ``w2_weight`` ``[E, D, I]``,
                both bf16, plus ``top_k``.
            router_logits: ``[tokens, num_experts]``; only used to derive top-k
                when ``topk_ids`` / ``topk_weights`` are not supplied.
            topk_weights: Precomputed expert weights ``[tokens, top_k]``.
            topk_ids: Precomputed expert ids ``[tokens, top_k]``.
            num_tokens_global, max_num_tokens_per_gpu, do_finalize, enable_pdl:
                Unused (no EP / deferred finalize support).

        Returns:
            MoE output ``[tokens, hidden]`` bf16.
        """
        del num_tokens_global, max_num_tokens_per_gpu, do_finalize, enable_pdl
        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return gluon_bf16_moe(
            x,
            w.w13_weight,
            w.w2_weight,
            topk_ids,
            topk_weights.to(torch.float32),
            expert_start=(
                int(getattr(w, "ep_rank", 0))
                * int(getattr(w, "num_local_experts", w.w13_weight.shape[0]))
            ),
            expert_parallel=int(getattr(w, "ep_size", 1)) > 1,
        )


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.routes_decode import (
        MAX_TOKENS as _BF16_ROUTES_DECODE_MAX_TOKENS,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.routes_decode import (
        launch_gluon_bf16_moe_routes_decode_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.routes_prefill import (
        launch_gluon_bf16_moe_routes_prefill_gfx950,
    )

    @register_kernel(
        "moe",
        "routes",
        name="gluon_bf16_moe_routes_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            # The kernels address buffers with 32-bit byte offsets. The
            # zero kernel clears 512-column hidden blocks; the decode and
            # prefill GEMMs tile the intermediate size by 256.
            "buffer_offsets_fit_int32": frozenset({True}),
            "hidden_alignment": frozenset({512}),
            "ispp_alignment": frozenset({256}),
            "topk": frozenset({1, 2, 4, 8, 16}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_bf16_moe_routes_gfx950(
        x: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_start: int,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        """Gluon kernel of :func:`tokenspeed_kernel.ops.moe.bf16_moe_routes`.

        Direct per-expert decode GEMMs through 16 tokens, block-aligned grouped
        GEMMs above; both write zeros for non-local routes, so ``output`` needs
        no clearing. The facade validates the tensors before selection.
        """
        num_tokens, top_k = topk_ids.shape
        if output is None:
            output = torch.empty(
                (num_tokens * top_k, x.shape[1]), dtype=x.dtype, device=x.device
            )
        if num_tokens:
            launch = (
                launch_gluon_bf16_moe_routes_decode_gfx950
                if num_tokens <= _BF16_ROUTES_DECODE_MAX_TOKENS
                else launch_gluon_bf16_moe_routes_prefill_gfx950
            )
            launch(
                x,
                w13,
                w2,
                topk_ids.to(torch.int32).contiguous(),
                expert_start,
                output.view(num_tokens * top_k, x.shape[1]),
            )
        return output.view(num_tokens, top_k, x.shape[1])


# ===-----------------------------------------------------------------------===#
# FP8 block-scaled experts (AMD gfx950, gfx1250)
# ===-----------------------------------------------------------------------===#


_FP8_BLOCK = 128
_DIRECT_DECODE_MAX_TOKENS = 32
_EXACT_MFMA_MAX_TOKENS = 4096


def _quantize_packed_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each activation row in groups of 128 values."""
    return quantize_fp8_with_scale(
        x,
        granularity="token_group",
        group_size=128,
        solution="triton",
    )


def _validate_fp8_moe(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    do_finalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    if not do_finalize:
        raise ValueError("Gluon block-FP8 MoE does not support deferred finalization")
    ep_size = int(getattr(w, "ep_size", 1))
    ep_rank = int(getattr(w, "ep_rank", 0))
    if ep_size < 1 or not 0 <= ep_rank < ep_size:
        raise ValueError("invalid expert-parallel rank or size")
    if any(
        getattr(w, name, None) is not None
        for name in ("w13_weight_bias", "w2_weight_bias")
    ):
        raise ValueError("Gluon block-FP8 MoE does not support expert bias")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Gluon block-FP8 MoE requires precomputed top-k routing")

    activation = plan.get("activation") or getattr(w, "activation", "silu")
    if activation not in {"silu", "swiglu"}:
        raise ValueError(
            f"Gluon block-FP8 MoE does not support activation {activation!r}"
        )
    swiglu_arg = getattr(w, "swiglu_arg", None)
    if swiglu_arg is not None and getattr(swiglu_arg, "alpha", None) not in {
        None,
        1.0,
    }:
        raise ValueError("Gluon block-FP8 MoE requires SwiGLU alpha=1")
    if getattr(w, "swiglu_beta", None) not in {None, 0.0}:
        raise ValueError("Gluon block-FP8 MoE supports only standard SwiGLU")
    if getattr(w, "w13_input_layout", "concatenated") != "concatenated":
        raise ValueError("Gluon block-FP8 MoE requires concatenated gate/up weights")

    w13 = w.w13_weight
    w2 = w.w2_weight
    s13 = w.w13_weight_scale_inv
    s2 = w.w2_weight_scale_inv
    if x.ndim != 2 or w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("x and block-FP8 MoE weights must be rank-2/rank-3")
    if x.dtype != torch.bfloat16:
        raise TypeError("x must use torch.bfloat16")
    fp8_dtypes = {torch.float8_e4m3fn, torch.float8_e4m3fnuz}
    if w13.dtype not in fp8_dtypes or w2.dtype not in fp8_dtypes:
        raise TypeError("w13_weight and w2_weight must use E4M3 FP8")
    if s13.dtype != torch.float32 or s2.dtype != torch.float32:
        raise TypeError("block-FP8 inverse scales must use torch.float32")
    if not all(t.is_cuda and t.is_contiguous() for t in (x, w13, w2, s13, s2)):
        raise ValueError("x, weights, and scales must be contiguous GPU tensors")
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("top-k tensors must have shape [num_tokens, top_k]")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must use torch.int32 or torch.int64")
    if not topk_weights.is_floating_point():
        raise TypeError("topk_weights must use a floating-point dtype")
    if any(t.device != x.device for t in (w13, w2, s13, s2, topk_ids, topk_weights)):
        raise ValueError("weights, scales, routing tensors, and x must share a device")

    num_tokens, hidden_size = x.shape
    num_experts, twice_intermediate_size, weight_hidden_size = w13.shape
    intermediate_size = twice_intermediate_size // 2
    top_k = topk_ids.shape[1]
    if num_experts == 0:
        raise ValueError("block-FP8 MoE requires at least one expert")
    if topk_ids.shape[0] != num_tokens or top_k == 0:
        raise ValueError("top-k tensors must have shape [num_tokens, top_k > 0]")
    if twice_intermediate_size % 2 or weight_hidden_size != hidden_size:
        raise ValueError("w13_weight has an incompatible shape")
    if w2.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError("w2_weight has an incompatible shape")
    expected_s13 = (
        num_experts,
        (twice_intermediate_size + _FP8_BLOCK - 1) // _FP8_BLOCK,
        (hidden_size + _FP8_BLOCK - 1) // _FP8_BLOCK,
    )
    expected_s2 = (
        num_experts,
        (hidden_size + _FP8_BLOCK - 1) // _FP8_BLOCK,
        (intermediate_size + _FP8_BLOCK - 1) // _FP8_BLOCK,
    )
    if tuple(s13.shape) != expected_s13 or tuple(s2.shape) != expected_s2:
        raise ValueError(
            "block-FP8 scale tensors have incompatible shapes: "
            f"expected {expected_s13} and {expected_s2}, got "
            f"{tuple(s13.shape)} and {tuple(s2.shape)}"
        )
    if hidden_size % 512 or intermediate_size % 512:
        raise ValueError("hidden and intermediate sizes must be multiples of 512")
    swiglu_limit = getattr(swiglu_arg, "limit", None)
    if swiglu_limit is not None and swiglu_limit <= 0:
        raise ValueError("Gluon block-FP8 MoE requires a positive SwiGLU limit")
    return topk_weights, topk_ids, swiglu_limit


if platform.is_amd:
    from tokenspeed_kernel.ops.moe._triton.common import (
        _combine,
        _routing,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8 import (
        gluon_fp8_block_dequantize,
        gluon_fp8_block_exact_mfma_moe,
        gluon_fp8_block_warp_decode_moe,
        launch_gluon_fp8_packed_moe_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess import (
        preprocess_gluon_fp8_packed_gfx950_moe_weights as gluon_fp8_packed_moe_weights,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.moe.fp8 import (
        launch_gluon_fp8_block_experts_gfx1250,
    )

    def gluon_fp8_moe_weights(plan: dict, w: torch.nn.Module) -> None:
        """Retain compact experts for decode and materialize BF16 prefill copies.

        Args:
            plan: MoE execution plan; unused beyond the preprocessor contract.
            w: Module containing block-scaled FP8 expert weights and inverse scales.
        """
        del plan
        w.w13_weight_prefill_bf16 = gluon_fp8_block_dequantize(
            w.w13_weight, w.w13_weight_scale_inv
        )
        w.w2_weight_prefill_bf16 = gluon_fp8_block_dequantize(
            w.w2_weight, w.w2_weight_scale_inv
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_fp8_block_precomputed_moe_apply",
        solution="gluon",
        weight_preprocessor=gluon_fp8_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"fp8"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False, True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ep_size": frozenset({1, 4}),
            "ispp_alignment": frozenset({512}),
            "internal_activation_dtype": frozenset({"input"}),
            "fp8_scale_block_shape": frozenset({(_FP8_BLOCK, _FP8_BLOCK)}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_fp8_block_precomputed_moe_apply(
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
        """Apply block-E4M3 experts with gfx950 decode and prefill kernels.

        Compact FP8 warp GEMVs handle batches through 32 tokens. Medium batches
        dequantize compact weights into LDS for exact BF16 MFMA; large prefill
        batches use materialized BF16 expert copies.

        Args:
            plan: MoE execution plan selecting SiLU/SwiGLU activation.
            x: Contiguous BF16 hidden states ``[tokens, hidden]``.
            w: Module containing block-E4M3 experts, scales, and BF16 copies.
            router_logits: Unused for precomputed routing.
            topk_weights: Route weights ``[tokens, top_k]``.
            topk_ids: Expert ids ``[tokens, top_k]``.
            num_tokens_global: Unused EP token count.
            max_num_tokens_per_gpu: Unused token-capacity hint.
            do_finalize: Must be true.
            enable_pdl: Unused launch hint.

        Returns:
            Finalized BF16 states ``[tokens, hidden]``.
        """
        validated_weights, validated_ids, swiglu_limit = _validate_fp8_moe(
            plan, x, w, topk_weights, topk_ids, do_finalize
        )
        if x.shape[0] == 0:
            return torch.empty_like(x)
        if x.shape[0] <= _DIRECT_DECODE_MAX_TOKENS:
            num_local_experts = int(
                getattr(w, "num_local_experts", w.w13_weight.shape[0])
            )
            expert_start = int(getattr(w, "ep_rank", 0)) * num_local_experts
            return gluon_fp8_block_warp_decode_moe(
                x,
                w.w13_weight,
                w.w2_weight,
                w.w13_weight_scale_inv,
                w.w2_weight_scale_inv,
                validated_ids,
                validated_weights,
                swiglu_limit,
                expert_start,
                int(getattr(w, "ep_size", 1)) > 1,
                packed_weights=False,
            )
        if x.shape[0] <= _EXACT_MFMA_MAX_TOKENS:
            num_local_experts = int(
                getattr(w, "num_local_experts", w.w13_weight.shape[0])
            )
            return gluon_fp8_block_exact_mfma_moe(
                x,
                w.w13_weight,
                w.w2_weight,
                w.w13_weight_scale_inv,
                w.w2_weight_scale_inv,
                validated_ids,
                validated_weights,
                int(getattr(w, "ep_rank", 0)) * num_local_experts,
                int(getattr(w, "ep_size", 1)) > 1,
            )

        prefill_weights = SimpleNamespace(
            w13_weight=w.w13_weight_prefill_bf16,
            w2_weight=w.w2_weight_prefill_bf16,
            ep_rank=int(getattr(w, "ep_rank", 0)),
            ep_size=int(getattr(w, "ep_size", 1)),
            num_local_experts=int(
                getattr(w, "num_local_experts", w.w13_weight.shape[0])
            ),
        )
        return gluon_bf16_precomputed_moe_apply(
            plan,
            x,
            prefill_weights,
            router_logits,
            topk_weights=validated_weights,
            topk_ids=validated_ids,
            num_tokens_global=num_tokens_global,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            do_finalize=do_finalize,
            enable_pdl=enable_pdl,
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_fp8_block_precomputed_moe_apply_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"fp8"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False, True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ep_size": frozenset({1, 4}),
            "ispp_alignment": frozenset({512}),
            "internal_activation_dtype": frozenset({"input"}),
            "fp8_scale_block_shape": frozenset({(_FP8_BLOCK, _FP8_BLOCK)}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_fp8_block_precomputed_moe_apply_gfx1250(
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
        """Apply block-E4M3 experts to BF16 activations with gfx1250 WMMA.

        Weight tiles are upcast to BF16 in registers and each 128x128 block's
        FP32 inverse scale multiplies its FP32 partial product, so neither the
        weights nor the activations are requantized.

        Args:
            plan: MoE execution plan selecting SiLU/SwiGLU activation.
            x: Contiguous BF16 hidden states ``[tokens, hidden]``.
            w: Module containing block-E4M3 experts and their inverse scales.
            router_logits: Unused for precomputed routing.
            topk_weights: Route weights ``[tokens, top_k]``.
            topk_ids: Global expert ids ``[tokens, top_k]``; ids outside this
                rank's experts contribute zero.
            num_tokens_global: Unused EP token count.
            max_num_tokens_per_gpu: Unused token-capacity hint.
            do_finalize: Must be true.
            enable_pdl: Unused launch hint.

        Returns:
            Finalized BF16 states ``[tokens, hidden]``.
        """
        validated_weights, validated_ids, swiglu_limit = _validate_fp8_moe(
            plan, x, w, topk_weights, topk_ids, do_finalize
        )
        if x.shape[0] == 0:
            return torch.empty_like(x)
        num_experts = w.w13_weight.shape[0]
        if int(getattr(w, "ep_size", 1)) > 1:
            num_local_experts = int(getattr(w, "num_local_experts", num_experts))
            validated_ids = (
                validated_ids - int(getattr(w, "ep_rank", 0)) * num_local_experts
            )
        # The ids above are already relative to this rank, so experts start at 0.
        expert_route_ids, expert_counts = _routing(validated_ids, num_experts, 0)
        route_output = torch.empty(
            (validated_ids.numel(), x.shape[1]), device=x.device, dtype=x.dtype
        )
        output = torch.empty_like(x)
        intermediate = torch.empty(
            (validated_ids.numel(), w.w13_weight.shape[1] // 2),
            device=x.device,
            dtype=x.dtype,
        )
        launch_gluon_fp8_block_experts_gfx1250(
            x,
            w.w13_weight,
            w.w2_weight,
            w.w13_weight_scale_inv,
            w.w2_weight_scale_inv,
            expert_route_ids,
            expert_counts,
            validated_ids.shape[1],
            swiglu_limit,
            intermediate,
            route_output,
        )
        # Expert programs skip ids outside this rank and leave those rows unset.
        _combine(route_output, validated_weights, output, validated_ids, num_experts)
        return output

    _PACKED_FP8_TRAITS = {
        "weight_dtype": frozenset({"fp8"}),
        "activation": frozenset({"swiglu"}),
        "swiglu_form": frozenset({"standard"}),
        "activation_clamped": frozenset({True}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False, True}),
        "supports_all_to_all_ep": frozenset({False}),
        "hidden": frozenset({4096}),
        "internal_activation_dtype": frozenset({"input"}),
        "fp8_scale_block_shape": frozenset({(_FP8_BLOCK, _FP8_BLOCK)}),
        "supports_bias": frozenset({False}),
        "expert_id_repeats": frozenset({False}),
        "combine_order": frozenset({"rank"}),
    }

    @register_kernel(
        "moe",
        "apply",
        name="gluon_fp8_packed_ep1_precomputed_moe_apply",
        solution="gluon",
        weight_preprocessor=gluon_fp8_packed_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            **_PACKED_FP8_TRAITS,
            "ep_size": frozenset({1}),
            "ispp": frozenset({512}),
        },
        priority=Priority.SPECIALIZED + 1,
    )
    def gluon_fp8_packed_ep1_precomputed_moe_apply(
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
        """Apply packed block-FP8 experts without expert parallelism.

        Args:
            plan: MoE execution plan selecting standard clamped SwiGLU.
            x: Contiguous BF16 hidden states ``[tokens, 4096]``.
            w: Module holding prepared packed E4M3 expert weights and FP32
                inverse scales per 128-by-128 block.
            router_logits: Accepted but unused; routing must be precomputed.
            topk_weights: Required global route weights ``[tokens, 8]``.
            topk_ids: Required global expert ids ``[tokens, 8]``.
            num_tokens_global: Accepted but unused EP token count.
            max_num_tokens_per_gpu: Accepted but unused token-capacity hint.
            do_finalize: Must be true; deferred finalization is unsupported.
            enable_pdl: Accepted but unused launch hint.

        Returns:
            Finalized BF16 expert contribution ``[tokens, 4096]``.
        """
        return _gluon_fp8_packed_precomputed_moe_apply(
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
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_fp8_packed_ep4_precomputed_moe_apply",
        solution="gluon",
        weight_preprocessor=gluon_fp8_packed_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            **_PACKED_FP8_TRAITS,
            "ep_size": frozenset({4}),
            "ispp": frozenset({2048}),
        },
        priority=Priority.SPECIALIZED + 1,
    )
    def gluon_fp8_packed_ep4_precomputed_moe_apply(
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
        """Apply packed block-FP8 experts with four expert-parallel ranks.

        Args:
            plan: MoE execution plan selecting standard clamped SwiGLU.
            x: Contiguous BF16 hidden states ``[tokens, 4096]``.
            w: Module holding this rank's prepared packed E4M3 expert weights
                and FP32 inverse scales per 128-by-128 block.
            router_logits: Accepted but unused; routing must be precomputed.
            topk_weights: Required global route weights ``[tokens, 8]``.
            topk_ids: Required global expert ids ``[tokens, 8]``.
            num_tokens_global: Accepted but unused EP token count.
            max_num_tokens_per_gpu: Accepted but unused token-capacity hint.
            do_finalize: Must be true; deferred finalization is unsupported.
            enable_pdl: Accepted but unused launch hint.

        Returns:
            Finalized BF16 local expert contribution ``[tokens, 4096]``;
            the caller performs any cross-rank reduction.
        """
        return _gluon_fp8_packed_precomputed_moe_apply(
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
        )

    def _gluon_fp8_packed_precomputed_moe_apply(
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
        """Apply packed block-FP8 experts and return this rank's full partial.

        Args:
            plan: MoE execution plan selecting standard clamped SwiGLU.
            x: Contiguous BF16 hidden states ``[tokens, 4096]``.
            w: Module holding canonical packed FP8 expert parameters.
            router_logits: Unused for precomputed routing.
            topk_weights: Global route weights ``[tokens, top_k]``.
            topk_ids: Global expert ids ``[tokens, top_k]``.
            num_tokens_global: Unused EP token count.
            max_num_tokens_per_gpu: Unused token-capacity hint.
            do_finalize: Must be true.
            enable_pdl: Unused launch hint.

        Returns:
            Finalized BF16 local expert contribution ``[tokens, 4096]``.
        """
        validated_weights, validated_ids, swiglu_limit = _validate_fp8_moe(
            plan, x, w, topk_weights, topk_ids, do_finalize
        )
        if swiglu_limit is None or validated_ids.shape[1] != 8:
            raise ValueError("packed FP8 MoE requires clamped SwiGLU and eight routes")
        if x.shape[0] == 0:
            return torch.empty_like(x)

        num_local_experts = int(w.num_local_experts)
        return launch_gluon_fp8_packed_moe_gfx950(
            x,
            w.w13_weight,
            w.w2_weight,
            w.w13_weight_scale_inv,
            w.w2_weight_scale_inv,
            validated_ids.to(dtype=torch.int32).contiguous(),
            validated_weights.to(dtype=torch.float32).contiguous(),
            float(swiglu_limit),
            int(w.ep_rank) * num_local_experts,
            num_experts_total=w.num_experts,
            quantize_activation=_quantize_packed_activation,
        )


# ===-----------------------------------------------------------------------===#
# MXFP4 experts (AMD gfx950, gfx1250)
# ===-----------------------------------------------------------------------===#


_ROUTE_DIRECT_DECODE_MAX_TOKENS = 16
_GFX1250_DECODE_MAX_AVERAGE_BPE = 16
_warmed_a4w4_ep_families: set[tuple] = set()


def _select_gfx950_grouped_block_m(
    num_tokens: int,
    top_k: int,
    num_experts: int,
) -> int:
    """Select the MI350X-tuned grouped-MoE row block."""

    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")

    routed_rows_x7 = 7 * num_tokens * top_k
    if routed_rows_x7 < 65 * num_experts:
        return 16
    if routed_rows_x7 < 215 * num_experts:
        return 32
    if routed_rows_x7 < 425 * num_experts:
        return 64
    return 128


def _use_gfx1250_moe_decode(num_routed_rows: int, num_experts: int) -> bool:
    # Compare the average batch per expert without a division or device sync.
    return (
        0 < num_routed_rows
        and num_routed_rows <= _GFX1250_DECODE_MAX_AVERAGE_BPE * num_experts
    )


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (
        gluon_mxfp4_fp8_precomputed_situ,
        gluon_mxfp_dynamic_mxfp4_fused_moe,
        gluon_mxfp_fused_moe,
        gluon_mxfp_precomputed_mxfp4_fused_moe,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.mxfp4_gemm import (
        launch_gluon_mxfp4_a4w4_moe_apply_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.n16_weights import (
        n16_mxfp4_shape,
        preprocess_n16_mxfp4_weights,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.prefill_mxfp8 import (
        mxfp8_situ_prefill,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.situ_decode import (
        _supports_a16w4_warp_decode_ep_gfx950,
        gluon_a16w4_situ_warp_decode_ep_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.situ_grouped import (
        gluon_a16w4_grouped_ep_gfx950,
        gluon_a16w4_situ_grouped_ep_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (
        preprocess_gluon_mxfp4_gfx950_moe_weights,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4 import fused as fused_mxfp_gfx1250
    from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.weight_preprocess import (
        preprocess_gluon_mxfp4_gfx1250_moe_weights,
    )

    def gluon_mxfp4_gfx950_moe_weights(plan: dict, w: torch.nn.Module):
        return preprocess_gluon_mxfp4_gfx950_moe_weights(plan, w, preshuffle=True)

    def gluon_mxfp4_gfx950_a8w4_situ_ep_weights(plan: dict, w: torch.nn.Module) -> None:
        if hasattr(w, "w13_weight") and w.w13_weight.ndim == 6:
            raise ValueError("N16 MXFP4 weights were already prepared")
        if getattr(w, "activation_situ_linear_beta", None) is None:
            if plan.get("internal_activation_dtype") == "fp8":
                raise ValueError("explicit FP8 SiTU requires a linear clamp")
            validate_linear_mxfp4_moe_weights(plan, w)
            return
        if w.w13_weight.shape[-1] * 2 % 256 or w.w2_weight.shape[-1] * 2 % 256:
            gluon_mxfp4_gfx950_moe_weights(plan, w)
            return
        preprocess_n16_mxfp4_weights(w)

    def gluon_mxfp4_gfx1250_moe_weights(plan: dict, w: torch.nn.Module):
        return preprocess_gluon_mxfp4_gfx1250_moe_weights(plan, w)

    def validate_linear_mxfp4_moe_weights(plan: dict, w: torch.nn.Module) -> None:
        del plan
        names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
        if any(not hasattr(w, name) for name in names):
            raise ValueError("linear MXFP4 MoE weights are incomplete")
        tensors = [getattr(w, name) for name in names]
        if any(t.dtype != torch.uint8 for t in tensors):
            raise TypeError("linear MXFP4 weights must use uint8 storage")
        if any(not t.is_contiguous() for t in tensors):
            raise ValueError("linear MXFP4 weights must be contiguous")
        if any(t.ndim != 3 for t in tensors) or len({t.shape[0] for t in tensors}) != 1:
            raise ValueError("linear MXFP4 weights must share a rank-3 expert axis")
        expected = int(getattr(w, "num_local_experts", tensors[0].shape[0]))
        if tensors[0].shape[0] != expected:
            raise ValueError("linear MXFP4 weights have the wrong local expert count")

    def _swiglu_args(w: torch.nn.Module) -> tuple[float, float, float]:
        swiglu_arg = getattr(w, "swiglu_arg", None)
        if swiglu_arg is None:
            # alpha=1, limit=0, beta=0 makes the reducer use an unclamped
            # SiLU gate multiplied by the linear branch.
            return 1.0, 0.0, 0.0
        swiglu_beta = getattr(w, "swiglu_beta", None)
        return (
            1.0 if swiglu_arg.alpha is None else swiglu_arg.alpha,
            0.0 if swiglu_arg.limit is None else swiglu_arg.limit,
            0.0 if swiglu_beta is None else swiglu_beta,
        )

    def _gluon_mxfp4_a16w4_ep_precomputed_moe_apply(
        x: torch.Tensor,
        w: torch.nn.Module,
        topk_weights: torch.Tensor | None,
        topk_ids: torch.Tensor | None,
        *,
        activation: str,
        do_finalize: bool,
    ) -> torch.Tensor:
        if not do_finalize:
            raise ValueError("gfx950 A16W4 Gluon MoE cannot defer finalization")
        if topk_weights is None or topk_ids is None:
            raise ValueError("gfx950 A16W4 Gluon MoE requires precomputed top-k")
        if x.ndim != 2 or x.shape[1] % 256:
            raise ValueError(
                "gfx950 A16W4 Gluon MoE requires rank-2 activations with "
                "hidden size divisible by 256"
            )

        activation_kwargs: dict[str, object]
        if activation == "situ":
            activation_kwargs = {
                "situ_beta": float(getattr(w, "activation_situ_beta", 1.0)),
                "situ_linear_beta": getattr(w, "activation_situ_linear_beta", None),
            }
        elif activation == "swiglu":
            swiglu_alpha, swiglu_limit, swiglu_beta = _swiglu_args(w)
            if swiglu_alpha != 1.0 or swiglu_beta != 0.0:
                raise ValueError(
                    "gfx950 A16W4 EP supports SwiGLU alpha=1 and beta=0 only"
                )
            activation_kwargs = {
                "situ_beta": 1.0,
                "situ_linear_beta": None,
                "activation": "swiglu",
                "swiglu_limit": swiglu_limit if swiglu_limit > 0.0 else None,
            }
        else:
            raise ValueError(f"unsupported gfx950 A16W4 activation: {activation}")

        num_local_experts = int(getattr(w, "num_local_experts", w.w13_weight.shape[0]))
        expert_start = int(getattr(w, "ep_rank", 0)) * num_local_experts
        output = getattr(w, "_situ_output_buffer", None)
        from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.situ_decode import (
            gluon_a16w4_situ_warp_decode_ep_gfx950 as route_direct_decode,
        )

        use_route_direct_decode = _supports_a16w4_warp_decode_ep_gfx950(
            x,
            w.w13_weight,
            w.w13_weight_scale,
            w.w2_weight,
            w.w2_weight_scale,
        ) and (
            activation != "situ" or int(x.shape[0]) < _ROUTE_DIRECT_DECODE_MAX_TOKENS
        )
        if use_route_direct_decode:
            # Both stages localize global expert IDs while consuming the linear
            # checkpoint layout, avoiding four pointwise localization kernels.
            return route_direct_decode(
                x,
                w.w13_weight,
                w.w13_weight_scale,
                w.w2_weight,
                w.w2_weight_scale,
                topk_weights,
                topk_ids,
                expert_start=expert_start,
                linear_weights=True,
                w13_interleaved=(
                    getattr(w, "w13_input_layout", "concatenated") == "interleaved"
                ),
                routed_out=output,
                **activation_kwargs,
            )
        global_num_experts = int(
            getattr(
                w,
                "num_experts",
                num_local_experts * int(getattr(w, "ep_size", 1)),
            )
        )
        block_m = (
            _select_gfx950_grouped_block_m(
                int(x.shape[0]),
                int(topk_ids.shape[1]),
                global_num_experts,
            )
            if activation == "situ"
            else (128 if int(x.shape[0]) >= 3584 else 64)
        )
        grouped_kernel = (
            gluon_a16w4_situ_grouped_ep_gfx950
            if activation == "situ"
            else gluon_a16w4_grouped_ep_gfx950
        )
        return grouped_kernel(
            x,
            w.w13_weight,
            w.w13_weight_scale,
            w.w2_weight,
            w.w2_weight_scale,
            topk_weights,
            topk_ids,
            block_m=block_m,
            expert_start=expert_start,
            out=output,
            **activation_kwargs,
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_a4w4_ep_deferred_precomputed_moe_apply",
        solution="gluon",
        weight_preprocessor=gluon_mxfp4_gfx950_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.uint8}),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"silu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({True}),
            "deferred_output_is_token_order": frozenset({True}),
            "supports_deferred_out": frozenset({True}),
            "supports_prequantized_input": frozenset({True}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ep_size": frozenset({1, 4, 8}),
            "ispp": frozenset({512, 4096}),
            "internal_activation_dtype": frozenset({"mxfp4"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED + 1,
    )
    def gluon_mxfp4_a4w4_ep_deferred_precomputed_moe_apply(
        plan: dict,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        w: torch.nn.Module,
        router_logits: torch.Tensor | None,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        num_tokens_global: int | None = None,
        max_num_tokens_per_gpu: int | None = None,
        do_finalize: bool = True,
        enable_pdl: bool = False,
        x_quantized: tuple[torch.Tensor, torch.Tensor] | None = None,
        deferred_out: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, None, torch.Tensor]:
        """Run gfx950 dynamic-A4W4 experts with optional finalization.

        Args:
            plan: Kernel plan selected by the MoE registry.
            x: Normalized BF16 ``[tokens, hidden]`` input or packed E2M1
                rows with CDNA4-swizzled E8M0 scales.
            w: Processed local expert weights and quantization metadata.
            router_logits: Unused because this path receives precomputed routes.
            topk_weights: Route weights with shape ``[tokens, top_k]``.
            topk_ids: Global expert IDs with shape ``[tokens, top_k]``.
            num_tokens_global: Unused global token-count hint.
            max_num_tokens_per_gpu: Unused per-rank capacity hint.
            do_finalize: Whether to combine weighted routes into hidden states.
            enable_pdl: Unused launch hint.
            x_quantized: Optional packed E2M1 input and CDNA4 E8M0 scales.
            deferred_out: Optional contiguous BF16 ``[tokens * top_k, hidden]``
                destination for unweighted routes with ``do_finalize=False``.
                It must not overlap inputs; the route map marks defined rows.

        Returns:
            Final hidden states when ``do_finalize`` is true; otherwise the
            canonical route outputs, no bias, and the local route map.
        """

        del plan, router_logits, num_tokens_global, max_num_tokens_per_gpu
        del enable_pdl
        if do_finalize and deferred_out is not None:
            raise ValueError("deferred_out requires do_finalize=False")
        if topk_weights is None or topk_ids is None:
            raise ValueError("gfx950 dynamic A4W4 EP requires precomputed top-k")
        num_local_experts = int(getattr(w, "num_local_experts"))
        expert_start = int(getattr(w, "ep_rank")) * num_local_experts
        global_num_experts = int(getattr(w, "num_experts"))
        swiglu_alpha, swiglu_limit, swiglu_beta = _swiglu_args(w)
        w13_precision_config = w.w13_precision_config
        w2_precision_config = w.w2_precision_config
        return gluon_mxfp_precomputed_mxfp4_fused_moe(
            x,
            topk_weights,
            topk_ids,
            w.w13_weight_triton_tensor,
            w.w2_weight_triton_tensor,
            w13_mx_scale=w13_precision_config.b_mx_scale,
            w2_mx_scale=w2_precision_config.b_mx_scale,
            expert_start=expert_start,
            global_num_experts=global_num_experts,
            do_finalize=do_finalize,
            x_quantized=x_quantized,
            w13_bias=None,
            w2_bias=None,
            out_dtype=w2_precision_config.out_dtype or torch.bfloat16,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_beta=swiglu_beta,
            out=getattr(w, "_situ_output_buffer", None) if do_finalize else None,
            deferred_out=deferred_out,
        )

    def _warmup_a4w4_ep_deferred(plan: dict, w: torch.nn.Module) -> None:
        """Compile Muse's bounded row families using the processed expert layout."""
        from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (
            _extract_gluon_raw_s,
            _extract_gluon_raw_w,
        )
        from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
            empty_swizzled_cdna4_mxfp4_scale,
        )

        if (
            w.hidden_size != 4096
            or w.intermediate_size != 4096
            or w.num_experts != 256
            or w.top_k != 8
            or _swiglu_args(w) != (1.0, 0.0, 0.0)
        ):
            return
        tensors = (
            _extract_gluon_raw_w(w.w13_weight_triton_tensor),
            _extract_gluon_raw_s(w.w13_precision_config.b_mx_scale),
            _extract_gluon_raw_w(w.w2_weight_triton_tensor),
            _extract_gluon_raw_s(w.w2_precision_config.b_mx_scale),
        )
        # The processed W13 bank holds the local intermediate shard.
        if (w.num_local_experts, tensors[0].shape[-1] // 2) not in (
            (32, 4096),
            (64, 4096),
            (256, 512),
        ):
            return
        device = tensors[0].device
        if device.type != "cuda":
            return
        layout = tuple(
            (
                tensor.device,
                tensor.dtype,
                tuple(tensor.shape),
                tensor.stride(),
                getattr(tensor, "is_shuffled_for_gluon_dot", False),
                getattr(tensor, "gluon_dot_block_k_pk", None),
                getattr(tensor, "gluon_dot_block_n", None),
                getattr(tensor, "original_k_pk", None),
                getattr(tensor, "original_n", None),
            )
            for tensor in tensors
        )
        geometry = (
            layout,
            w.ep_rank * w.num_local_experts,
            w.num_experts,
            w.top_k,
            _swiglu_args(w),
            w.w2_precision_config.out_dtype,
        )
        # Runtime M has two integer-alignment classes. Sort widths grow only
        # at these bounds; the direct/compact decode kernels have two row tiles.
        representatives = (
            1,
            2,
            4,
            16,
            17,
            32,
            49,
            64,
            65,
            80,
            129,
            144,
            192,
            193,
            257,
            272,
            513,
            528,
            1025,
            1040,
            2049,
            2064,
            4097,
            4112,
        )
        max_tokens = min(int(plan["warmup_max_num_tokens"]), 8192)
        pending = [
            rows
            for rows in representatives
            if rows <= max_tokens and (geometry, rows) not in _warmed_a4w4_ep_families
        ]
        if not pending:
            return
        with torch.cuda.device(device), torch.no_grad():
            for rows in pending:
                packed = torch.zeros((rows, 2048), device=device, dtype=torch.uint8)
                scales = empty_swizzled_cdna4_mxfp4_scale(rows, 128, device=device)
                scales.fill_(127)
                # Routes are replicated; EP filters by expert, TP computes
                # each route's local intermediate shard.
                ids = torch.arange(rows * 8, device=device, dtype=torch.int32)
                ids = ids.remainder(256).view(rows, 8)
                weights = torch.full(
                    (rows, 8), 0.125, device=device, dtype=torch.float32
                )
                result = gluon_mxfp4_a4w4_ep_deferred_precomputed_moe_apply(
                    plan,
                    (packed, scales),
                    w,
                    None,
                    topk_weights=weights,
                    topk_ids=ids,
                    do_finalize=False,
                )
                del result, packed, scales, ids, weights
            torch.cuda.synchronize(device)
        _warmed_a4w4_ep_families.update((geometry, rows) for rows in pending)

    gluon_mxfp4_a4w4_ep_deferred_precomputed_moe_apply._tokenspeed_warmup = (  # type: ignore[attr-defined]
        _warmup_a4w4_ep_deferred
    )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_a8w4_situ_precomputed_moe_apply",
        solution="gluon",
        weight_preprocessor=gluon_mxfp4_gfx950_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"situ"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False}),
            "supports_all_to_all_ep": frozenset({False}),
            # K3 TP8 shards the 3072-wide expert intermediate to 384 columns.
            "ispp": frozenset({384}),
            "ispp_alignment": frozenset({128}),
            "internal_activation_dtype": frozenset({"input"}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_mxfp4_a8w4_situ_precomputed_moe_apply(
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
        shared_input: torch.Tensor | None = None,
        shared_weight: torch.Tensor | None = None,
        shared_out: torch.Tensor | None = None,
    ):
        del plan, router_logits, num_tokens_global, max_num_tokens_per_gpu
        del enable_pdl
        if not do_finalize:
            raise ValueError("gfx950 A8W4 SiTU MoE cannot defer finalization")
        if topk_weights is None or topk_ids is None:
            raise ValueError("gfx950 A8W4 SiTU MoE requires precomputed top-k")

        situ_beta = float(getattr(w, "activation_situ_beta", 1.0))
        situ_linear_beta = getattr(w, "activation_situ_linear_beta", None)
        if situ_linear_beta is not None:
            out = gluon_mxfp4_fp8_precomputed_situ(
                x,
                topk_weights,
                topk_ids,
                w.w13_weight_triton_tensor,
                w.w2_weight_triton_tensor,
                w13_mx_scale=w.w13_precision_config.b_mx_scale,
                w2_mx_scale=w.w2_precision_config.b_mx_scale,
                situ_beta=situ_beta,
                situ_linear_beta=float(situ_linear_beta),
                out=getattr(w, "_situ_output_buffer", None),
                shared_input=shared_input,
                shared_weight=shared_weight,
                shared_out=shared_out,
            )
            if out is not None:
                return out
        raise ValueError(
            "gfx950 A8W4 SiTU MoE does not support this activation or weight shape"
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_a8w4_situ_ep_precomputed_moe_apply",
        solution="gluon",
        weight_preprocessor=gluon_mxfp4_gfx950_a8w4_situ_ep_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"situ"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ep_size": frozenset({8}),
            "ispp": frozenset({3072}),
            "internal_activation_dtype": frozenset({"input", "fp8"}),
            "supports_bias": frozenset({False}),
        },
        # One N16 bank serves MXFP8 prefill and the existing A16 joint decoder.
        priority=Priority.SPECIALIZED + 1,
    )
    def gluon_mxfp4_a8w4_situ_ep_precomputed_moe_apply(
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
        require_fp8 = plan.get("internal_activation_dtype") == "fp8"
        del plan, router_logits, num_tokens_global, max_num_tokens_per_gpu
        del enable_pdl
        if not do_finalize:
            raise ValueError("gfx950 A8W4 SiTU EP MoE cannot defer finalization")
        if topk_weights is None or topk_ids is None:
            raise ValueError("gfx950 A8W4 SiTU EP MoE requires precomputed top-k")
        if x.ndim == 2 and x.shape[0] == 0:
            if topk_weights.ndim != 2 or topk_ids.shape != topk_weights.shape:
                raise ValueError("gfx950 A8W4 SiTU EP MoE requires matching top-k")
            if topk_ids.shape[0] != 0:
                raise ValueError("empty EP input requires empty top-k tensors")
            output = getattr(w, "_situ_output_buffer", None)
            if output is None:
                return torch.empty_like(x)
            if (
                output.shape != x.shape
                or output.dtype != x.dtype
                or output.device != x.device
            ):
                raise ValueError("empty EP output must match the input tensor")
            return output

        situ_linear_beta = getattr(w, "activation_situ_linear_beta", None)
        if situ_linear_beta is None:
            if require_fp8:
                raise ValueError("explicit FP8 SiTU requires a linear clamp")
            if w.w13_weight.ndim == 6:
                raise ValueError("SiTU linear clamp changed after weight preparation")
            return _gluon_mxfp4_a16w4_ep_precomputed_moe_apply(
                x,
                w,
                topk_weights,
                topk_ids,
                activation="situ",
                do_finalize=do_finalize,
            )
        num_local_experts = int(getattr(w, "num_local_experts"))
        global_num_experts = int(getattr(w, "num_experts"))
        if hasattr(w, "w13_weight_triton_tensor"):
            # Widths outside full K256/N128 cells retain the existing bank and
            # A8 implementation; they do not materialize a second layout.
            out = gluon_mxfp4_fp8_precomputed_situ(
                x,
                topk_weights,
                topk_ids,
                w.w13_weight_triton_tensor,
                w.w2_weight_triton_tensor,
                w13_mx_scale=w.w13_precision_config.b_mx_scale,
                w2_mx_scale=w.w2_precision_config.b_mx_scale,
                situ_beta=float(getattr(w, "activation_situ_beta", 1.0)),
                situ_linear_beta=float(situ_linear_beta),
                out=getattr(w, "_situ_output_buffer", None),
                expert_start=int(getattr(w, "ep_rank", 0)) * num_local_experts,
                global_num_experts=global_num_experts,
                prefill_activation_format="e4m3",
            )
            if out is None:
                raise ValueError("gfx950 A8W4 SiTU EP MoE does not support this shape")
            return out
        n16_mxfp4_shape(
            w.w13_weight, w.w13_weight_scale, w.w2_weight, w.w2_weight_scale
        )
        if (
            not require_fp8
            and x.shape[0] <= 4
            and _supports_a16w4_warp_decode_ep_gfx950(
                x,
                w.w13_weight,
                w.w13_weight_scale,
                w.w2_weight,
                w.w2_weight_scale,
                linear_weights=True,
            )
        ):
            return gluon_a16w4_situ_warp_decode_ep_gfx950(
                x,
                w.w13_weight,
                w.w13_weight_scale,
                w.w2_weight,
                w.w2_weight_scale,
                topk_weights,
                topk_ids,
                situ_beta=float(getattr(w, "activation_situ_beta", 1.0)),
                situ_linear_beta=float(situ_linear_beta),
                expert_start=int(getattr(w, "ep_rank", 0)) * num_local_experts,
                linear_weights=True,
                w13_interleaved=getattr(w, "w13_input_layout", "concatenated")
                == "interleaved",
                routed_out=getattr(w, "_situ_output_buffer", None),
            )
        return mxfp8_situ_prefill(
            x,
            w.w13_weight,
            w.w13_weight_scale,
            w.w2_weight,
            w.w2_weight_scale,
            topk_ids,
            topk_weights,
            situ_beta=float(getattr(w, "activation_situ_beta", 1.0)),
            situ_linear_beta=float(situ_linear_beta),
            out=getattr(w, "_situ_output_buffer", None),
            expert_start=int(getattr(w, "ep_rank", 0)) * num_local_experts,
            global_experts=global_num_experts,
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_a16w4_situ_ep_precomputed_moe_apply",
        solution="gluon",
        weight_preprocessor=validate_linear_mxfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"situ"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            # This grouped kernel was validated for K3's TP1/EP8 placement.
            # Keep auto-selection scoped to that exact EP degree; Triton
            # remains the fallback for other AMD EP layouts.
            "ep_size": frozenset({8}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
        },
        # Capability and ep_size traits restrict this measured K3 fast path to
        # gfx950 EP8, so it needs no intra-band priority offset.
        priority=Priority.SPECIALIZED,
    )
    def gluon_mxfp4_a16w4_situ_ep_precomputed_moe_apply(
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
        del plan, router_logits, num_tokens_global, max_num_tokens_per_gpu
        del enable_pdl
        return _gluon_mxfp4_a16w4_ep_precomputed_moe_apply(
            x,
            w,
            topk_weights,
            topk_ids,
            activation="situ",
            do_finalize=do_finalize,
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_a16w4_swiglu_ep_precomputed_moe_apply",
        solution="gluon",
        weight_preprocessor=validate_linear_mxfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ep_size": frozenset({2, 4, 8}),
            "ispp_alignment": frozenset({128}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_mxfp4_a16w4_swiglu_ep_precomputed_moe_apply(
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
        return _gluon_mxfp4_a16w4_ep_precomputed_moe_apply(
            x,
            w,
            topk_weights,
            topk_ids,
            activation="swiglu",
            do_finalize=do_finalize,
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_a8w4_situ_gfx1250_precomputed_moe_apply",
        solution="gluon",
        weight_preprocessor=gluon_mxfp4_gfx1250_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"situ"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False}),
            "supports_all_to_all_ep": frozenset({False}),
            # K3 TP8 shards the 3072-wide expert intermediate to 384 columns.
            "ispp": frozenset({384}),
            "ispp_alignment": frozenset({128}),
            "internal_activation_dtype": frozenset({"input"}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_mxfp4_a8w4_situ_gfx1250_precomputed_moe_apply(
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
        shared_input: torch.Tensor | None = None,
        shared_weight: torch.Tensor | None = None,
        shared_out: torch.Tensor | None = None,
    ):
        del plan, router_logits, num_tokens_global, max_num_tokens_per_gpu
        del enable_pdl
        if not do_finalize:
            raise ValueError("gfx1250 A8W4 SiTU MoE cannot defer finalization")
        if topk_weights is None or topk_ids is None:
            raise ValueError("gfx1250 A8W4 SiTU MoE requires precomputed top-k")
        if (
            shared_input is not None
            or shared_weight is not None
            or shared_out is not None
        ):
            raise ValueError(
                "gfx1250 A8W4 SiTU MoE does not support shared projection fusion"
            )

        situ_beta = float(getattr(w, "activation_situ_beta", 1.0))
        situ_linear_beta = getattr(w, "activation_situ_linear_beta", None)
        if situ_linear_beta is None:
            raise ValueError("gfx1250 A8W4 SiTU MoE requires linear_beta")
        w13_pc = w.w13_precision_config
        w2_pc = w.w2_precision_config
        decode = _use_gfx1250_moe_decode(
            topk_ids.numel(), w.w13_weight_triton_tensor.shape[0]
        )

        return fused_mxfp_gfx1250.gluon_mxfp_precomputed_mxfp4_fused_moe(
            x,
            topk_weights,
            topk_ids,
            w.w13_weight_triton_tensor,
            w.w2_weight_triton_tensor,
            w13_bias=(
                None
                if getattr(w, "_gluon_w13_bias_is_zero", False)
                else getattr(w, "w13_weight_bias", None)
            ),
            w2_bias=(
                None
                if getattr(w, "_gluon_w2_bias_is_zero", False)
                else getattr(w, "w2_weight_bias", None)
            ),
            w13_mx_scale=w13_pc.b_mx_scale,
            w2_mx_scale=w2_pc.b_mx_scale,
            activation_dtype="fp8",
            out_dtype=w2_pc.out_dtype or torch.bfloat16,
            activation="situ",
            situ_beta=situ_beta,
            situ_linear_beta=float(situ_linear_beta),
            decode=decode,
            out=getattr(w, "_situ_output_buffer", None),
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_gfx1250_precomputed_moe_apply",
        solution="gluon",
        weight_preprocessor=gluon_mxfp4_gfx1250_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            # "input" and "mxfp4" both run dynamic MXFP4 activations.
            "internal_activation_dtype": frozenset({"fp8", "input", "mxfp4"}),
            "supports_bias": frozenset({True}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_mxfp4_gfx1250_precomputed_moe_apply(
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
        del router_logits, num_tokens_global, max_num_tokens_per_gpu
        del do_finalize, enable_pdl
        if topk_weights is None or topk_ids is None:
            raise ValueError(
                "gluon_mxfp4_gfx1250_precomputed_moe_apply requires "
                "topk_weights and topk_ids"
            )
        activation_dtype = (
            "fp8" if plan.get("internal_activation_dtype") == "fp8" else "mxfp4"
        )

        swiglu_alpha, swiglu_limit, swiglu_beta = _swiglu_args(w)
        w13_pc = w.w13_precision_config
        w2_pc = w.w2_precision_config
        decode = _use_gfx1250_moe_decode(
            topk_ids.numel(), w.w13_weight_triton_tensor.shape[0]
        )

        return fused_mxfp_gfx1250.gluon_mxfp_precomputed_mxfp4_fused_moe(
            x,
            topk_weights,
            topk_ids,
            w.w13_weight_triton_tensor,
            w.w2_weight_triton_tensor,
            w13_bias=(
                None
                if getattr(w, "_gluon_w13_bias_is_zero", False)
                else getattr(w, "w13_weight_bias", None)
            ),
            w2_bias=(
                None
                if getattr(w, "_gluon_w2_bias_is_zero", False)
                else getattr(w, "w2_weight_bias", None)
            ),
            w13_mx_scale=w13_pc.b_mx_scale,
            w2_mx_scale=w2_pc.b_mx_scale,
            out_dtype=w2_pc.out_dtype or torch.bfloat16,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_beta=swiglu_beta,
            decode=decode,
            activation_dtype=activation_dtype,
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_dynamic_moe_apply",
        solution="gluon",
        weight_preprocessor=gluon_mxfp4_gfx950_moe_weights,
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
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({True}),
        },
        # ``routing_mode=None`` deliberately leaves moe_plan unconstrained, so
        # this and the precomputed sibling both match. Prefer kernel routing;
        # an explicit precomputed_topk request still filters this entry out.
        priority=Priority.SPECIALIZED + 1,
    )
    def gluon_mxfp4_dynamic_moe_apply(
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
        del num_tokens_global, max_num_tokens_per_gpu
        del do_finalize, enable_pdl
        top_k = getattr(w, "top_k")
        swiglu_alpha, swiglu_limit, swiglu_beta = _swiglu_args(w)
        w13_precision_config = w.w13_precision_config
        w2_precision_config = w.w2_precision_config
        # Forward the caller's precomputed top-k whenever it is supplied. The
        # downstream dispatch picks the tuned kernel by batch size (direct
        # decode owns M <= _DIRECT_DECODE_MAX_M, precomputed-MFMA decode owns
        # M >= _PRECOMPUTED_MFMA_MIN_M) and otherwise builds ragged metadata
        # directly from the forwarded top-k. Dropping it for any M in between
        # (e.g. M == 3) would silently recompute routing from router_logits,
        # so we always forward and let the dispatch choose.
        forward_precomputed_topk = topk_weights is not None and topk_ids is not None

        return gluon_mxfp_dynamic_mxfp4_fused_moe(
            x,
            router_logits,
            w.w13_weight_triton_tensor,
            w.w2_weight_triton_tensor,
            w13_bias=(
                None
                if getattr(w, "_gluon_w13_bias_is_zero", False)
                else getattr(w, "w13_weight_bias", None)
            ),
            w2_bias=(
                None
                if getattr(w, "_gluon_w2_bias_is_zero", False)
                else getattr(w, "w2_weight_bias", None)
            ),
            w13_mx_scale=w13_precision_config.b_mx_scale,
            w2_mx_scale=w2_precision_config.b_mx_scale,
            out_dtype=w2_precision_config.out_dtype or torch.bfloat16,
            top_k=top_k,
            correction_bias=getattr(w, "_correction_bias", None),
            n_group=int(getattr(w, "_n_group", 0) or 0),
            topk_group=int(getattr(w, "_topk_group", 0) or 0),
            routed_scaling_factor=float(
                getattr(w, "_routed_scaling_factor", 1.0) or 1.0
            ),
            normalize_topk_weights=bool(getattr(w, "_normalize_topk_weights", True)),
            routing_method_type=int(getattr(w, "_routing_method_type", 0)),
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_beta=swiglu_beta,
            precomputed_topk_weights=topk_weights if forward_precomputed_topk else None,
            precomputed_topk_ids=topk_ids if forward_precomputed_topk else None,
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_precomputed_moe_apply",
        solution="gluon",
        weight_preprocessor=gluon_mxfp4_gfx950_moe_weights,
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
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({True}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_mxfp4_precomputed_moe_apply(
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
        del plan, router_logits, num_tokens_global, max_num_tokens_per_gpu
        del do_finalize, enable_pdl
        if topk_weights is None or topk_ids is None:
            raise ValueError(
                "gluon_mxfp4_precomputed_moe_apply requires topk_weights and topk_ids"
            )
        swiglu_alpha, swiglu_limit, swiglu_beta = _swiglu_args(w)
        w13_precision_config = w.w13_precision_config
        w2_precision_config = w.w2_precision_config

        return gluon_mxfp_precomputed_mxfp4_fused_moe(
            x,
            topk_weights,
            topk_ids,
            w.w13_weight_triton_tensor,
            w.w2_weight_triton_tensor,
            w13_bias=(
                None
                if getattr(w, "_gluon_w13_bias_is_zero", False)
                else getattr(w, "w13_weight_bias", None)
            ),
            w2_bias=(
                None
                if getattr(w, "_gluon_w2_bias_is_zero", False)
                else getattr(w, "w2_weight_bias", None)
            ),
            w13_mx_scale=w13_precision_config.b_mx_scale,
            w2_mx_scale=w2_precision_config.b_mx_scale,
            expert_start=0,
            global_num_experts=int(getattr(w, "num_experts")),
            do_finalize=True,
            x_quantized=None,
            out_dtype=w2_precision_config.out_dtype or torch.bfloat16,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_beta=swiglu_beta,
            out=getattr(w, "_situ_output_buffer", None),
        )

    def gluon_mxfp4_a4w4_moe_weights_gfx950(plan: dict, w: torch.nn.Module) -> None:
        del plan
        preprocess_n16_mxfp4_weights(w)

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_a4w4_moe_apply_gfx950",
        solution="gluon",
        weight_preprocessor=gluon_mxfp4_a4w4_moe_weights_gfx950,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "swiglu_form": frozenset({"standard"}),
            "activation_clamped": frozenset({False}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({64}),
            "hidden_alignment": frozenset({512}),
            "internal_activation_dtype": frozenset({"mxfp4"}),
            "supports_bias": frozenset({False}),
            "combine_order": frozenset({"rank"}),
        },
        # Beats the portable Triton dynamic-MXFP4 path on gfx950.
        priority=Priority.SPECIALIZED,
    )
    def gluon_mxfp4_a4w4_moe_apply_gfx950(
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
        """Apply SiLU experts with dynamic MXFP4 activations and MXFP4 weights.

        Both GEMMs are E2M1 x E2M1 scaled MFMAs; activations and the SiLU
        intermediate are quantized per 32 values like the portable Triton
        path, so results match it up to FP32 accumulation order.

        Args:
            plan: MoE plan from ``moe_plan``.
            x: BF16 hidden states ``[tokens, hidden]``.
            w: Module holding the N16 banks written by the weight
                preprocessor.
            router_logits: Unused; routing must be precomputed.
            topk_weights: Route weights ``[tokens, top_k]``.
            topk_ids: Expert ids ``[tokens, top_k]``; out-of-range ids add 0.
            num_tokens_global: Unused; expert parallelism is unsupported.
            max_num_tokens_per_gpu: Unused token-capacity hint.
            do_finalize: Must be true.
            enable_pdl: Unused launch hint.

        Returns:
            BF16 output ``[tokens, hidden]``.
        """
        del plan, router_logits, num_tokens_global, max_num_tokens_per_gpu
        del enable_pdl
        if not do_finalize:
            raise ValueError("gfx950 A4W4 Gluon MoE cannot defer finalization")
        if topk_weights is None or topk_ids is None:
            raise ValueError("gfx950 A4W4 Gluon MoE requires precomputed top-k")
        # The kernels compute SiLU(gate) * up: alpha 1, no clamp (limit 0), beta 0.
        if _swiglu_args(w) != (1.0, 0.0, 0.0):
            raise ValueError("gfx950 A4W4 Gluon MoE supports only standard SiLU")
        return launch_gluon_mxfp4_a4w4_moe_apply_gfx950(
            # Quantization reads rows by stride; only the inner stride must be 1.
            x if x.stride(1) == 1 else x.contiguous(),
            w.w13_weight,
            w.w13_weight_scale,
            w.w2_weight,
            w.w2_weight_scale,
            topk_weights,
            topk_ids,
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_moe_apply",
        solution="gluon",
        weight_preprocessor=gluon_mxfp4_gfx950_moe_weights,
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
        # gluon is narrowly gated to gfx950
        priority=Priority.SPECIALIZED,
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
        del topk_weights, topk_ids, num_tokens_global, max_num_tokens_per_gpu
        del do_finalize, enable_pdl
        top_k = getattr(w, "top_k")
        swiglu_alpha, swiglu_limit, swiglu_beta = _swiglu_args(w)
        w13_precision_config = w.w13_precision_config
        w2_precision_config = w.w2_precision_config

        return gluon_mxfp_fused_moe(
            x,
            router_logits,
            w.w13_weight_triton_tensor,
            w.w2_weight_triton_tensor,
            w13_bias=(
                None
                if getattr(w, "_gluon_w13_bias_is_zero", False)
                else getattr(w, "w13_weight_bias", None)
            ),
            w2_bias=(
                None
                if getattr(w, "_gluon_w2_bias_is_zero", False)
                else getattr(w, "w2_weight_bias", None)
            ),
            w13_mx_scale=w13_precision_config.b_mx_scale,
            w2_mx_scale=w2_precision_config.b_mx_scale,
            w13_act_scale=w.w13_act_scale,
            w2_act_scale=w.w2_act_scale,
            out_dtype=w2_precision_config.out_dtype or torch.bfloat16,
            top_k=top_k,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_beta=swiglu_beta,
        )


# ===-----------------------------------------------------------------------===#
# MXFP4 Petit MegaMoE (AMD gfx950)
# ===-----------------------------------------------------------------------===#


# TokenSpeed adapter for the vendored Gluon Petit MegaMoE kernels.

_WORLD_SIZE = 8
_MAX_TOKENS_PER_RANK = 1024
_GPT_OSS_SWIGLU_ALPHA = 1.702
_GPT_OSS_SWIGLU_LIMIT = 7.0
_GPT_OSS_SWIGLU_BETA = 1.0


def _import_petit_kernel() -> Any:
    """Import the vendored implementation only after Petit is selected."""
    from tokenspeed_kernel.thirdparty.gluon_petit import load_petit_kernel

    return load_petit_kernel()


@dataclass(frozen=True)
class _Profile:
    name: str
    num_experts: int
    top_k: int
    model_dim: int
    logical_intermediate: int
    inter_dim: int
    has_bias: bool
    activation: str


_GPT_OSS_120B_PROFILE = _Profile(
    name="gpt_oss_120b",
    num_experts=128,
    top_k=4,
    model_dim=2880,
    logical_intermediate=2880,
    inter_dim=3072,
    has_bias=True,
    activation="swiglu",
)
_DSV4_PROFILE = _Profile(
    name="dsv4",
    num_experts=384,
    top_k=6,
    model_dim=7168,
    logical_intermediate=3072,
    inter_dim=3072,
    has_bias=False,
    activation="silu",
)
_KIMI_K3_PROFILE = _Profile(
    name="kimi_k3",
    num_experts=896,
    top_k=16,
    model_dim=3584,
    logical_intermediate=3072,
    inter_dim=3072,
    has_bias=False,
    activation="situ",
)
_DSV41_FLASH_PROFILE = _Profile(
    name="dsv41_flash",
    num_experts=384,
    top_k=6,
    model_dim=5120,
    logical_intermediate=2304,
    inter_dim=2560,
    has_bias=False,
    activation="silu_clamped",
)
_DSV41_FLASH_DSPARK_PROFILE = _Profile(
    name="dsv41_flash_dspark",
    num_experts=128,
    top_k=3,
    model_dim=5120,
    logical_intermediate=2304,
    inter_dim=2560,
    has_bias=False,
    activation="silu_clamped",
)
_PROFILES = (
    _GPT_OSS_120B_PROFILE,
    _DSV4_PROFILE,
    _KIMI_K3_PROFILE,
    _DSV41_FLASH_PROFILE,
    _DSV41_FLASH_DSPARK_PROFILE,
)


@dataclass
class _GluonPetitState:
    """Repacked expert weights kept on the layer's ``_moe_backend_state``."""

    profile: _Profile
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    w13_bias: torch.Tensor | None
    w2_bias: torch.Tensor | None


def _release_parameter(module: torch.nn.Module, name: str) -> None:
    if name in module._parameters:
        module.register_parameter(name, None)
    elif hasattr(module, name):
        delattr(module, name)


def _deinterleave_gate_up(tensor: torch.Tensor) -> torch.Tensor:
    shape = tensor.shape
    return (
        tensor.reshape(shape[0], shape[1] // 2, 2, *shape[2:])
        .permute(0, 2, 1, *range(3, tensor.dim() + 1))
        .contiguous()
        .reshape(shape)
    )


def _pad_mxfp4_pair(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    padded_weight = weight.new_zeros((weight.shape[0], rows, columns // 2))
    padded_scale = scale.new_zeros((scale.shape[0], rows, columns // 32))
    padded_weight[:, : weight.shape[1], : weight.shape[2]].copy_(weight)
    padded_scale[:, : scale.shape[1], : scale.shape[2]].copy_(scale)
    return padded_weight, padded_scale


def _pad_gate_up_pair(
    weight: torch.Tensor,
    scale: torch.Tensor,
    profile: _Profile,
) -> tuple[torch.Tensor, torch.Tensor]:
    experts = weight.shape[0]
    logical_intermediate = weight.shape[1] // 2
    compute_model_dim = (profile.model_dim + 511) // 512 * 512
    padded_weight = weight.new_zeros(
        (experts, 2, profile.inter_dim, compute_model_dim // 2)
    )
    padded_scale = scale.new_zeros(
        (experts, 2, profile.inter_dim, compute_model_dim // 32)
    )
    weight = weight.reshape(experts, 2, logical_intermediate, weight.shape[2])
    scale = scale.reshape(experts, 2, logical_intermediate, scale.shape[2])
    padded_weight[:, :, :logical_intermediate, : weight.shape[3]].copy_(weight)
    padded_scale[:, :, :logical_intermediate, : scale.shape[3]].copy_(scale)
    return padded_weight.reshape(
        experts, 2 * profile.inter_dim, -1
    ), padded_scale.reshape(experts, 2 * profile.inter_dim, -1)


def _validate_layer(w: torch.nn.Module) -> _Profile:
    geometry = (w.num_experts, w.top_k, w.hidden_size, w.intermediate_size)
    profile = next(
        (
            candidate
            for candidate in _PROFILES
            if geometry
            == (
                candidate.num_experts,
                candidate.top_k,
                candidate.model_dim,
                candidate.logical_intermediate,
            )
        ),
        None,
    )
    if profile is None:
        raise ValueError(
            f"Unsupported Gluon Petit MegaMoE geometry: "
            f"num_experts={w.num_experts}, top_k={w.top_k}, "
            f"hidden_size={w.hidden_size}, intermediate_size={w.intermediate_size}"
        )
    if (
        w.ep_size != _WORLD_SIZE
        or w.tp_size != 1
        or w.num_local_experts != profile.num_experts // _WORLD_SIZE
    ):
        raise ValueError(
            "Gluon Petit MegaMoE requires EP8, TP1 and contiguous expert shards"
        )

    if profile is _GPT_OSS_120B_PROFILE:
        swiglu_arg = getattr(w, "swiglu_arg", None)
        if (
            w.activation != "swiglu"
            or swiglu_arg is None
            or swiglu_arg.alpha != _GPT_OSS_SWIGLU_ALPHA
            or swiglu_arg.limit != _GPT_OSS_SWIGLU_LIMIT
            or w.swiglu_beta != _GPT_OSS_SWIGLU_BETA
        ):
            raise ValueError(
                "Gluon Petit GPT-OSS MegaMoE requires OpenAI SwiGLU "
                "alpha=1.702, limit=7.0, beta=1.0"
            )
        if w.w13_input_layout != "interleaved":
            raise ValueError(
                "Gluon Petit GPT-OSS MegaMoE requires interleaved W13 input"
            )
        if (
            getattr(w, "w13_weight_bias", None) is None
            or getattr(w, "w2_weight_bias", None) is None
        ):
            raise ValueError("Gluon Petit GPT-OSS MegaMoE requires expert biases")
        return profile

    if profile is _KIMI_K3_PROFILE:
        if (
            w.activation != "situ"
            or w.activation_situ_beta != 4.0
            or w.activation_situ_linear_beta != 25.0
            or w.swiglu_beta is not None
            or w.swiglu_arg is not None
        ):
            raise ValueError("Gluon Petit Kimi K3 requires SiTU beta=4, linear_beta=25")
        if w.w13_input_layout != "concatenated":
            raise ValueError("Gluon Petit Kimi K3 requires concatenated W13 input")
        if w.w13_weight_bias is not None or w.w2_weight_bias is not None:
            raise ValueError("Gluon Petit Kimi K3 requires bias-free experts")
        return profile

    if w.activation not in {"silu", "swiglu"} or w.swiglu_beta is not None:
        raise ValueError("Gluon Petit DeepSeek MegaMoE requires standard SiLU")
    if w.w13_input_layout != "concatenated":
        raise ValueError("Gluon Petit DeepSeek MegaMoE requires concatenated W13 input")
    swiglu_arg = getattr(w, "swiglu_arg", None)
    if swiglu_arg is not None and swiglu_arg.alpha is not None:
        raise ValueError(
            "Gluon Petit DeepSeek MegaMoE does not support nonstandard SiLU alpha"
        )
    if (
        getattr(w, "w13_weight_bias", None) is not None
        or getattr(w, "w2_weight_bias", None) is not None
    ):
        raise ValueError("Gluon Petit DeepSeek MegaMoE requires bias-free experts")

    if profile.activation == "silu_clamped":
        if swiglu_arg is None or swiglu_arg.limit != 10.0:
            raise ValueError(
                "Gluon Petit V4.1 requires the checkpoint SiLU clamp of 10"
            )
        return profile
    if swiglu_arg is not None and swiglu_arg.limit is not None:
        raise ValueError(
            "Gluon Petit DSV4 MegaMoE does not support an activation clamp; "
            "select a backend that preserves the checkpoint activation"
        )
    return profile


@dataclass
class _Workspace:
    config: Any
    heap: Any
    inputs: Any


_workspace_cache: dict[tuple[int, _Profile], _Workspace] = {}


def _workspace_key(device: torch.device, profile: _Profile) -> tuple[int, _Profile]:
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    return index, profile


def _get_workspace(device: torch.device, profile: _Profile) -> _Workspace:
    key = _workspace_key(device, profile)
    workspace = _workspace_cache.get(key)
    if workspace is not None:
        return workspace
    if not dist.is_initialized():
        raise RuntimeError(
            "Gluon Petit MegaMoE requires torch.distributed initialization"
        )
    if dist.get_world_size() != _WORLD_SIZE:
        raise RuntimeError(
            "Gluon Petit MegaMoE requires the default distributed world size 8"
        )

    petit_kernel = _import_petit_kernel()
    config = petit_kernel.MegaMoeConfig(
        world_size=_WORLD_SIZE,
        num_experts=profile.num_experts,
        topk=profile.top_k,
        model_dim=profile.model_dim,
        activation=petit_kernel.MegaMoeActivation.mxfp4,
        # Petit's SiTU variant hard-codes Kimi K3's beta=4 and linear_beta=25.
        activation_function=(
            petit_kernel.MegaMoeActivationFunction.kimi_situ
            if profile.name == _KIMI_K3_PROFILE.name
            else petit_kernel.MegaMoeActivationFunction(profile.activation)
        ),
        stages=petit_kernel.MegaMoeStages.two_stage,
        inter_dim=profile.inter_dim,
        has_bias=profile.has_bias,
    )
    heap = petit_kernel.create_vmm_symmetric_heap(_WORLD_SIZE)
    inputs = config.input_views(heap, _MAX_TOKENS_PER_RANK)
    workspace = _Workspace(config=config, heap=heap, inputs=inputs)
    _workspace_cache[key] = workspace
    return workspace


def _slice_inputs(inputs: Any, num_tokens: int) -> Any:
    """Return exact-row views required by Petit's quantize and run APIs."""
    return type(inputs)(
        tokens=inputs.tokens[:num_tokens],
        scales=None if inputs.scales is None else inputs.scales[:num_tokens],
        expert_ids=inputs.expert_ids[:num_tokens],
        expert_weights=inputs.expert_weights[:num_tokens],
    )


def gluon_petit_mxfp4_megamoe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Repack loaded serialized MXFP4 expert weights for Gluon Petit.

    Args:
        plan: Selected TokenSpeed MoE plan. Petit needs no extra plan fields.
        w: MoE layer whose loaded weights are replaced with Petit layouts.
    """
    del plan
    profile = _validate_layer(w)
    _get_workspace(w.w13_weight.device, profile)
    petit_kernel = _import_petit_kernel()

    w13 = w.w13_weight.detach().contiguous()
    s13 = w.w13_weight_scale.detach().contiguous()
    if profile.has_bias:
        w13 = _deinterleave_gate_up(w13)
        s13 = _deinterleave_gate_up(s13)
    w13, s13 = _pad_gate_up_pair(w13, s13, profile)
    compute_model_dim = (profile.model_dim + 511) // 512 * 512
    w2, s2 = _pad_mxfp4_pair(
        w.w2_weight.detach().contiguous(),
        w.w2_weight_scale.detach().contiguous(),
        rows=compute_model_dim,
        columns=profile.inter_dim,
    )
    layout = petit_kernel.MoeKernelLayout.native_mxfp4
    w13, s13 = petit_kernel.repack_moe_kernel_layout(
        w13,
        s13,
        layout=layout,
        petit_format=True,
    )
    w2, s2 = petit_kernel.repack_moe_kernel_layout(
        w2,
        s2,
        layout=layout,
        petit_format=True,
    )

    b13 = b2 = None
    if profile.has_bias:
        b13 = _deinterleave_gate_up(w.w13_weight_bias.detach().contiguous())
        logical_intermediate = b13.shape[1] // 2
        padded_b13 = b13.new_zeros((b13.shape[0], 2, profile.inter_dim))
        padded_b13[:, :, :logical_intermediate].copy_(
            b13.reshape(b13.shape[0], 2, logical_intermediate)
        )
        b13 = petit_kernel.repack_moe_kernel_layout(
            padded_b13.reshape(-1, profile.inter_dim),
            None,
            layout=layout,
            petit_format=True,
        ).reshape(b13.shape[0], -1)
        padded_b2 = w.w2_weight_bias.new_zeros(
            (w.w2_weight_bias.shape[0], profile.inter_dim)
        )
        padded_b2[:, : profile.model_dim].copy_(w.w2_weight_bias.detach())
        b2 = petit_kernel.repack_moe_kernel_layout(
            padded_b2.contiguous(),
            None,
            layout=layout,
            petit_format=True,
        )

    w._moe_backend_state = _GluonPetitState(
        profile=profile,
        w13_weight=w13.contiguous(),
        w2_weight=w2.contiguous(),
        w13_scale=s13.contiguous(),
        w2_scale=s2.contiguous(),
        w13_bias=None if b13 is None else b13.contiguous(),
        w2_bias=None if b2 is None else b2.contiguous(),
    )
    for name in (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_weight_bias",
        "w2_weight_bias",
    ):
        _release_parameter(w, name)
    torch.cuda.empty_cache()


@register_kernel(
    "moe",
    "apply",
    name="gluon_petit_mxfp4_megamoe_apply",
    solution="gluon",
    weight_preprocessor=gluon_petit_mxfp4_megamoe_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"amd"}),
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"swiglu", "silu", "situ"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({True}),
        "a2a_backend": frozenset({"gluon_petit"}),
        "ep_size": frozenset({_WORLD_SIZE}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"mxfp4"}),
        "supports_bias": frozenset({True}),
    },
    priority=Priority.SPECIALIZED + 3,
)
def gluon_petit_mxfp4_megamoe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    num_tokens_global: int | None,
    max_num_tokens_per_gpu: int | None,
    do_finalize: bool,
    enable_pdl: bool,
    low_latency: bool | None,
    overlap_fn: Callable[[], None] | None,
) -> torch.Tensor:
    """Run the fused Gluon Petit dispatch, experts, return, and combine path.

    Args:
        plan: Selected TokenSpeed MoE plan.
        x: Local BF16 token activations.
        w: MoE layer containing preprocessed Petit weights.
        router_logits: Unused raw routing logits.
        topk_weights: Precomputed route weights for every local token.
        topk_ids: Precomputed global expert ids for every local token.
        num_tokens_global: Unused global token-count hint.
        max_num_tokens_per_gpu: Largest local token count in this collective.
        do_finalize: Whether to produce the combined output.
        enable_pdl: Unused programmatic dependent launch setting.
        low_latency: Unused all-to-all mode hint.
        overlap_fn: Optional work to enqueue before the Petit collective.

    Returns:
        Final local BF16 output with shape ``[local_tokens, model_dim]``.
    """
    del plan, router_logits, num_tokens_global, enable_pdl, low_latency
    if not do_finalize:
        raise ValueError("Gluon Petit MegaMoE does not support deferred finalize")
    if topk_weights is None or topk_ids is None:
        raise ValueError(
            "Gluon Petit MegaMoE requires precomputed top-k ids and weights"
        )
    if (
        max_num_tokens_per_gpu is not None
        and max_num_tokens_per_gpu > _MAX_TOKENS_PER_RANK
    ):
        raise ValueError(
            "Gluon Petit MegaMoE per-rank token count exceeds its 1024-token capacity"
        )
    num_tokens = int(x.shape[0])
    if num_tokens > _MAX_TOKENS_PER_RANK:
        raise ValueError(
            "Gluon Petit MegaMoE received more than 1024 tokens on one rank"
        )

    state = w._moe_backend_state
    profile = state.profile
    workspace = _get_workspace(x.device, profile)
    inputs = _slice_inputs(workspace.inputs, num_tokens)
    if num_tokens:
        workspace.config.quantize(x.contiguous(), out=inputs)
        inputs.expert_ids.copy_(topk_ids.to(torch.int32))
        inputs.expert_weights.copy_(topk_weights.to(torch.float32))
    if overlap_fn is not None:
        overlap_fn()
    output = x.new_empty((num_tokens, profile.model_dim))
    return workspace.config.run(
        workspace.heap,
        state.w13_weight,
        state.w2_weight,
        state.w13_scale,
        state.w2_scale,
        num_tokens,
        w13_bias=state.w13_bias,
        w2_bias=state.w2_bias,
        out=output,
        inputs=inputs,
    )


# ===-----------------------------------------------------------------------===#
# BF16 latent-MoE input projections (AMD gfx950, gfx1250)
# ===-----------------------------------------------------------------------===#


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_decode import (
        launch_gluon_latent_input_decode_gfx950 as _decode_gfx950_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_largem import (
        launch_gluon_latent_input_largem_gfx950 as _largem_gfx950_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_mediumm import (
        launch_gluon_latent_input_mediumm_gfx950 as _mediumm_gfx950_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_small_batch import (
        launch_gluon_latent_input_small_batch_gfx950 as _small_batch_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.moe.fp16.latent_input_decode import (
        launch_gluon_latent_input_decode_gfx1250 as _decode_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.moe.fp16.latent_input_largem import (
        launch_gluon_latent_input_largem_gfx1250 as _largem_gfx1250_impl,
    )

    _SIGNATURES = frozenset(
        {
            format_signature(
                hidden_states=dense_tensor_format(torch.bfloat16),
                router_weight=dense_tensor_format(torch.bfloat16),
                routed_weight=dense_tensor_format(torch.bfloat16),
                shared_gate_up_weight=dense_tensor_format(torch.bfloat16),
            )
        }
    )
    _GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )
    _GFX1250 = CapabilityRequirement(
        min_arch_version=ArchVersion(12, 5),
        max_arch_version=ArchVersion(12, 5),
        vendors=frozenset({"amd"}),
    )

    @register_kernel(
        "moe",
        "latent_input",
        name="gluon_latent_input_decode_gfx1250",
        solution="gluon",
        capability=_GFX1250,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "weights_packed": frozenset({True}),
            "hidden_size": frozenset({7168}),
            "num_experts": frozenset({896}),
            "latent_size": frozenset({3584}),
            "shared_size": frozenset({768}),
            "inputs_contiguous": frozenset({True}),
            "tokens": frozenset(range(1, 65)),
        },
    )
    def gluon_latent_input_decode_gfx1250(**kwargs):
        weights = (
            kwargs["router_weight"],
            kwargs["routed_weight"],
            kwargs["shared_gate_up_weight"],
        )
        packed_weight = packed_projection_weight_view(*weights)
        if packed_weight is None:
            raise ValueError("Kimi K3 projection weights must be packed")
        return _decode_gfx1250_impl(
            kwargs["hidden_states"],
            *weights,
            packed_weight,
            beta=kwargs["gate_clamp"],
            linear_beta=kwargs["up_clamp"],
        )

    @register_kernel(
        "moe",
        "latent_input",
        name="gluon_latent_input_largem_gfx1250",
        solution="gluon",
        capability=_GFX1250,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "tokens_min": frozenset({1025}),
            "hidden_size": frozenset({7168}),
            "num_experts": frozenset({896}),
            "latent_size": frozenset({3584}),
            "shared_size": frozenset({768}),
            "inputs_contiguous": frozenset({True}),
            "weights_packed": frozenset({True}),
        },
    )
    def gluon_latent_input_largem_gfx1250(**kwargs):
        weights = (
            kwargs["router_weight"],
            kwargs["routed_weight"],
            kwargs["shared_gate_up_weight"],
        )
        packed_weight = packed_projection_weight_view(*weights)
        if packed_weight is None:
            raise ValueError("Kimi K3 projection weights must be packed")
        return _largem_gfx1250_impl(
            kwargs["hidden_states"],
            *weights,
            packed_weight,
            beta=kwargs["gate_clamp"],
            linear_beta=kwargs["up_clamp"],
        )

    @register_kernel(
        "moe",
        "latent_input",
        name="gluon_latent_input_decode_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "tokens": frozenset({1}),
            "hidden_size": frozenset({7168}),
            "num_experts": frozenset({896}),
            "latent_size": frozenset({3584}),
            "shared_size": frozenset({768}),
            "inputs_contiguous": frozenset({True}),
        },
    )
    def gluon_latent_input_decode_gfx950(**kwargs):
        return _decode_gfx950_impl(
            kwargs["hidden_states"],
            kwargs["router_weight"],
            kwargs["routed_weight"],
            kwargs["shared_gate_up_weight"],
            beta=kwargs["gate_clamp"],
            linear_beta=kwargs["up_clamp"],
        )

    @register_kernel(
        "moe",
        "latent_input",
        name="gluon_latent_input_small_batch_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        # Below the single-token specialist and above the portable Triton
        # kernel. Its split-K tiles win until 320 tokens.
        priority=Priority.SPECIALIZED - 1,
        traits={
            "weights_packed": frozenset({True}),
            "hidden_size_multiple_64": frozenset({True}),
            "inputs_contiguous": frozenset({True}),
            "tokens": frozenset(range(1, 321)),
        },
    )
    def gluon_latent_input_small_batch_gfx950(**kwargs):
        weights = (
            kwargs["router_weight"],
            kwargs["routed_weight"],
            kwargs["shared_gate_up_weight"],
        )
        return _small_batch_impl(
            kwargs["hidden_states"],
            *weights,
            packed_projection_weight_view(*weights),
            beta=kwargs["gate_clamp"],
            linear_beta=kwargs["up_clamp"],
        )

    @register_kernel(
        "moe",
        "latent_input",
        name="gluon_latent_input_mediumm_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "tokens": frozenset(range(321, 1281)),
            "hidden_size": frozenset({7168}),
            "num_experts": frozenset({896}),
            "latent_size": frozenset({3584}),
            "shared_size": frozenset({768}),
            "inputs_contiguous": frozenset({True}),
            "weights_packed": frozenset({True}),
        },
    )
    def gluon_latent_input_mediumm_gfx950(**kwargs):
        weights = (
            kwargs["router_weight"],
            kwargs["routed_weight"],
            kwargs["shared_gate_up_weight"],
        )
        packed_weight = packed_projection_weight_view(*weights)
        if packed_weight is None:
            raise ValueError("Kimi K3 latent input weights must be packed")
        return _mediumm_gfx950_impl(
            kwargs["hidden_states"],
            *weights,
            packed_weight,
            beta=kwargs["gate_clamp"],
            linear_beta=kwargs["up_clamp"],
        )

    @register_kernel(
        "moe",
        "latent_input",
        name="gluon_latent_input_largem_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            # A partial final row tile is masked, so no alignment is required.
            "tokens_min": frozenset({1281}),
            "hidden_size": frozenset({7168}),
            "inputs_contiguous": frozenset({True}),
            "latent_size": frozenset({3584}),
            "num_experts": frozenset({896}),
            "shared_size": frozenset({768}),
            "weights_packed": frozenset({True}),
        },
    )
    def gluon_latent_input_largem_gfx950(**kwargs):
        weights = (
            kwargs["router_weight"],
            kwargs["routed_weight"],
            kwargs["shared_gate_up_weight"],
        )
        packed_weight = packed_projection_weight_view(*weights)
        if packed_weight is None:
            raise ValueError("Kimi K3 prefill projection weights must be packed")
        return _largem_gfx950_impl(
            kwargs["hidden_states"],
            *weights,
            packed_weight,
            beta=kwargs["gate_clamp"],
            linear_beta=kwargs["up_clamp"],
        )


# ===-----------------------------------------------------------------------===#
# MXFP4 latent-MoE joint routed/shared decode (AMD gfx950)
# ===-----------------------------------------------------------------------===#


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.latent_shared_decode import (
        gluon_latent_expert_shared_decode_gfx950 as _joint_decode_impl,
    )

    @register_kernel(
        "moe",
        "latent_expert_shared",
        name="gluon_latent_expert_shared_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            format_signature(
                hidden_states=dense_tensor_format(torch.bfloat16),
                w13_weight=dense_tensor_format(torch.uint8),
                w13_scale=dense_tensor_format(torch.uint8),
                w2_weight=dense_tensor_format(torch.uint8),
                w2_scale=dense_tensor_format(torch.uint8),
                topk_weights=dense_tensor_format(topk_weight_dtype),
                topk_ids=dense_tensor_format(torch.int32),
                shared_input=dense_tensor_format(torch.bfloat16),
                shared_weight=dense_tensor_format(torch.bfloat16),
                routed_out=dense_tensor_format(torch.bfloat16),
                shared_out=dense_tensor_format(torch.bfloat16),
            )
            for topk_weight_dtype in (torch.bfloat16, torch.float32)
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "tokens": frozenset({1, 2, 3, 4}),
            "latent_size": frozenset({3584}),
            "topk": frozenset({16}),
            "num_local_experts": frozenset({112}),
            "intermediate_size": frozenset({3072}),
            "shared_size": frozenset({768}),
            "output_size": frozenset({7168}),
            "linear_weights": frozenset({True, False}),
            "inputs_contiguous": frozenset({True}),
        },
    )
    def gluon_latent_expert_shared_gfx950(**kwargs):
        return _joint_decode_impl(
            kwargs["hidden_states"],
            kwargs["w13_weight"],
            kwargs["w13_scale"],
            kwargs["w2_weight"],
            kwargs["w2_scale"],
            kwargs["topk_weights"],
            kwargs["topk_ids"],
            situ_beta=kwargs["activation_clamp"],
            situ_linear_beta=kwargs["linear_clamp"],
            expert_start=kwargs["expert_start"],
            w13_interleaved=kwargs["w13_interleaved"],
            shared_input=kwargs["shared_input"],
            shared_weight=kwargs["shared_weight"],
            routed_out=kwargs["routed_out"],
            shared_out=kwargs["shared_out"],
        )
