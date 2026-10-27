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

import tokenspeed_kernel.ops.moe._triton.routed_norm  # noqa: F401
import torch
from tokenspeed_kernel.ops.moe._triton.bf16 import (
    _check_route_tensors as _check_bf16_route_tensors,
)
from tokenspeed_kernel.ops.moe._triton.bf16 import _moe as _triton_bf16_moe_impl
from tokenspeed_kernel.ops.moe._triton.bf16 import _route_outputs as _bf16_route_outputs
from tokenspeed_kernel.ops.moe._triton.bf16 import _validate as _validate_bf16_moe
from tokenspeed_kernel.ops.moe._triton.decode_sigmoid_topk import (
    decode_sigmoid_bias_topk,
)
from tokenspeed_kernel.ops.moe._triton.fp8 import (
    _triton_fp8_moe_impl,
    _validate_fp8_moe,
)
from tokenspeed_kernel.ops.moe._triton.fp8_channel import (
    _triton_fp8_channel_moe_impl,
    _validate_fp8_channel_moe,
)
from tokenspeed_kernel.ops.moe._triton.fp8_channel_fused import (
    _triton_fp8_channel_fused_moe_impl,
)
from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (
    kimi3_sigmoid_bias_topk,
)
from tokenspeed_kernel.ops.moe._triton.latent_input import (
    triton_latent_input_packed,
)
from tokenspeed_kernel.ops.moe._triton.minimax_topk import (
    minimax_biased_grouped_topk,
)
from tokenspeed_kernel.ops.moe._triton.mxfp4 import _moe as _triton_mxfp4_moe_impl
from tokenspeed_kernel.ops.moe._triton.sigmoid_topk_routing import (
    triton_sigmoid_bias_topk_routing,
)
from tokenspeed_kernel.ops.moe._triton.softmax_topk import triton_softmax_topk
from tokenspeed_kernel.ops.moe._triton.sqrt_softplus_topk import (
    triton_sqrt_softplus_topk,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

# ===-----------------------------------------------------------------------===#
# Routing: softmax top-k (NVIDIA, AMD gfx1250)
# ===-----------------------------------------------------------------------===#


register_kernel(
    "moe",
    "softmax_topk",
    name="triton_softmax_topk",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures(
        "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    priority=Priority.PERFORMANT,
)(triton_softmax_topk)

register_kernel(
    "moe",
    "softmax_topk",
    name="triton_softmax_topk_gfx1250",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(12, 5),
        max_arch_version=ArchVersion(12, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=format_signatures(
        "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    priority=Priority.PERFORMANT,
)(triton_softmax_topk)

# ===-----------------------------------------------------------------------===#
# Routing: sqrt-softplus top-k (NVIDIA, AMD)
# ===-----------------------------------------------------------------------===#


register_kernel(
    "moe",
    "topk",
    name="triton_sqrt_softplus_topk",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        format_signature(router_logits=dense_tensor_format(dtype))
        for dtype in (torch.float16, torch.bfloat16, torch.float32)
    ),
    traits={
        "routing_kind": frozenset({"plain", "bias", "hash"}),
        "score_function": frozenset({"sqrt_softplus"}),
    },
    priority=Priority.PORTABLE,
)(triton_sqrt_softplus_topk)

# ===-----------------------------------------------------------------------===#
# Routing: sigmoid-bias top-k (NVIDIA)
# ===-----------------------------------------------------------------------===#


@register_kernel(
    "moe",
    "sigmoid_bias_topk",
    name="triton_minimax_sigmoid_bias_topk",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures(
        "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    priority=Priority.PERFORMANT,
    traits={"supports_output_buffers": frozenset({True})},
)
def triton_minimax_sigmoid_bias_topk(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    weights_dtype: torch.dtype = torch.float32,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused single-kernel routing via the minimax biased-topk Triton kernel.

    The ungrouped biased-sigmoid case is exactly minimax routing with one
    expert group; the multi-launch torch reference costs ~3x on NVIDIA
    (40us vs 13us at [1, 896] topk=16). ``hidden_states`` is only shape-
    validated by the kernel wrapper, so the logits stand in for it.
    """
    if out is not None and not normalize_topk_weights and out[0].dtype != torch.float32:
        raise ValueError(
            "Unnormalized routing output buffers require FP32 weight storage"
        )
    # The kernel scales only when it renormalizes, so keep fp32 through mul_.
    topk_weights, topk_ids = minimax_biased_grouped_topk(
        router_logits,
        router_logits,
        correction_bias,
        topk=topk,
        renormalize=normalize_topk_weights,
        num_expert_group=1,
        topk_group=1,
        num_fused_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        weights_dtype=weights_dtype if normalize_topk_weights else torch.float32,
        out=out,
    )
    if not normalize_topk_weights:
        topk_weights.mul_(routed_scaling_factor)
        if out is None:
            topk_weights = topk_weights.to(weights_dtype)
    return topk_weights, topk_ids if out is not None else topk_ids.to(torch.int32)


# ===-----------------------------------------------------------------------===#
# Routing: Kimi K3 packed sigmoid-bias top-k (NVIDIA, AMD gfx1250)
# ===-----------------------------------------------------------------------===#


#: Row ranges where the packed sort beats the kernel otherwise selected.
#: NVIDIA measured on GB200, cold L2: ahead to 256 rows, tying at 320.
#: CDNA5 measured on MI455X: 1.54x from two rows, still ahead at 512 and
#: behind by 1024. One row stays with the decode specialist, and the
#: gfx1250 Gluon remains available on these rows for explicit selection;
#: the packed registration wins automatic selection by priority.
#: CDNA4 one-token K3 stays on the decode specialist: on MI355X that kernel
#: is ahead of packed, and it already covers this shape.
_PACKED_ROWS_NVIDIA = range(1, 257)
_PACKED_ROWS_CDNA5 = range(2, 513)
_PACKED_CONTRACT = {
    "experts": frozenset({896}),
    "topk": frozenset({16}),
}


def _invoke_packed(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    weights_dtype: torch.dtype,
    logical_to_physical_map: torch.Tensor | None = None,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the packed K3 specialist after layout/dtype adapters.

    Selection only matches ``topk=16``; an override can still pass another
    value, which this wrapper rejects rather than returning 16 routes. The
    registry signature cannot see ``correction_bias`` dtype, so a non-FP32
    bias is widened here (exact) instead of raising inside the kernel.
    """
    if topk != 16:
        raise ValueError(f"packed Kimi K3 top-k supports only topk=16, got {topk}")
    return kimi3_sigmoid_bias_topk(
        router_logits.contiguous(),
        correction_bias.float().contiguous(),
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        logical_to_physical_map=(
            None
            if logical_to_physical_map is None
            else logical_to_physical_map.contiguous()
        ),
        weights_dtype=weights_dtype,
        enable_pdl=pdl_enabled(),
        out=out,
    )


_PACKED_SIGNATURES = format_signatures("router_logits", "dense", {torch.float32})


@register_kernel(
    "moe",
    "sigmoid_bias_topk",
    name="triton_kimi3_packed_sigmoid_bias_topk_nvidia",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_PACKED_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        **_PACKED_CONTRACT,
        "tokens": _PACKED_ROWS_NVIDIA,
        "supports_output_buffers": frozenset({True}),
    },
)
def triton_kimi3_packed_sigmoid_bias_topk_nvidia(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    weights_dtype: torch.dtype,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed top-16 weights and ids for FP32 ``[tokens, 896]`` logits.

    Arguments follow ``kimi3_sigmoid_bias_topk`` with explicit ``topk=16``.
    Layouts are made contiguous and correction bias is widened to FP32.
    Returns ``[tokens, 16]`` weights in ``weights_dtype`` and INT32 ids;
    mapped variants translate ids through ``logical_to_physical_map``.
    """
    return _invoke_packed(
        router_logits=router_logits,
        correction_bias=correction_bias,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        weights_dtype=weights_dtype,
        out=out,
    )


@register_kernel(
    "moe",
    "sigmoid_bias_topk_mapped",
    name="triton_kimi3_packed_sigmoid_bias_topk_nvidia_mapped",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_PACKED_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        **_PACKED_CONTRACT,
        "tokens": _PACKED_ROWS_NVIDIA,
        "supports_output_buffers": frozenset({True}),
    },
)
def triton_kimi3_packed_sigmoid_bias_topk_nvidia_mapped(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    logical_to_physical_map: torch.Tensor,
    weights_dtype: torch.dtype,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed top-16 weights and ids for FP32 ``[tokens, 896]`` logits.

    Arguments follow ``kimi3_sigmoid_bias_topk`` with explicit ``topk=16``.
    Layouts are made contiguous and correction bias is widened to FP32.
    Returns ``[tokens, 16]`` weights in ``weights_dtype`` and INT32 ids;
    mapped variants translate ids through ``logical_to_physical_map``.
    """
    return _invoke_packed(
        router_logits=router_logits,
        correction_bias=correction_bias,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        logical_to_physical_map=logical_to_physical_map,
        weights_dtype=weights_dtype,
        out=out,
    )


@register_kernel(
    "moe",
    "sigmoid_bias_topk",
    name="triton_kimi3_packed_sigmoid_bias_topk_gfx1250",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(12, 5),
        max_arch_version=ArchVersion(12, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=_PACKED_SIGNATURES,
    priority=Priority.SPECIALIZED + 1,
    traits={**_PACKED_CONTRACT, "tokens": _PACKED_ROWS_CDNA5},
)
def triton_kimi3_packed_sigmoid_bias_topk_gfx1250(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    weights_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed top-16 weights and ids for FP32 ``[tokens, 896]`` logits.

    Arguments follow ``kimi3_sigmoid_bias_topk`` with explicit ``topk=16``.
    Layouts are made contiguous and correction bias is widened to FP32.
    Returns ``[tokens, 16]`` weights in ``weights_dtype`` and INT32 ids;
    mapped variants translate ids through ``logical_to_physical_map``.
    """
    return _invoke_packed(
        router_logits=router_logits,
        correction_bias=correction_bias,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        weights_dtype=weights_dtype,
    )


@register_kernel(
    "moe",
    "sigmoid_bias_topk_mapped",
    name="triton_kimi3_packed_sigmoid_bias_topk_gfx1250_mapped",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(12, 5),
        max_arch_version=ArchVersion(12, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=_PACKED_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={**_PACKED_CONTRACT, "tokens": _PACKED_ROWS_CDNA5},
)
def triton_kimi3_packed_sigmoid_bias_topk_gfx1250_mapped(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    logical_to_physical_map: torch.Tensor,
    weights_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed top-16 weights and ids for FP32 ``[tokens, 896]`` logits.

    Arguments follow ``kimi3_sigmoid_bias_topk`` with explicit ``topk=16``.
    Layouts are made contiguous and correction bias is widened to FP32.
    Returns ``[tokens, 16]`` weights in ``weights_dtype`` and INT32 ids;
    mapped variants translate ids through ``logical_to_physical_map``.
    """
    return _invoke_packed(
        router_logits=router_logits,
        correction_bias=correction_bias,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        logical_to_physical_map=logical_to_physical_map,
        weights_dtype=weights_dtype,
    )


# ===-----------------------------------------------------------------------===#
# Routing: sigmoid-bias top-k, one decode token (AMD)
# ===-----------------------------------------------------------------------===#


@register_kernel(
    "moe",
    "sigmoid_bias_topk",
    name="triton_decode_sigmoid_bias_topk",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures("router_logits", "dense", {torch.float32}),
    # The broader gfx950 Gluon kernel declares no shape traits, so the caller's
    # single-token traits cannot filter it out. Prefer this decode specialization.
    priority=Priority.SPECIALIZED + 1,
    traits={
        "tokens": frozenset({1}),
        "experts": frozenset(range(1, 1025)),
        "topk": frozenset(range(1, 17)),
    },
)
def triton_decode_sigmoid_bias_topk(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    return decode_sigmoid_bias_topk(
        router_logits,
        correction_bias,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        weights_dtype=weights_dtype,
    )


@register_kernel(
    "moe",
    "sigmoid_bias_topk_mapped",
    name="triton_decode_sigmoid_bias_topk_mapped",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures("router_logits", "dense", {torch.float32}),
    priority=Priority.SPECIALIZED,
    traits={
        "tokens": frozenset({1}),
        "experts": frozenset(range(1, 1025)),
        "topk": frozenset(range(1, 17)),
    },
)
def triton_decode_sigmoid_bias_topk_mapped(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    logical_to_physical_map: torch.Tensor,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    return decode_sigmoid_bias_topk(
        router_logits,
        correction_bias,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        logical_to_physical_map=logical_to_physical_map,
        weights_dtype=weights_dtype,
    )


# ===-----------------------------------------------------------------------===#
# BF16/FP16 experts (NVIDIA, AMD)
# ===-----------------------------------------------------------------------===#


@register_kernel(
    "moe",
    "routes",
    name="triton_bf16_moe_routes",
    solution="triton",
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    priority=Priority.PORTABLE,
)
def triton_bf16_moe_routes(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_start: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton kernel of :func:`tokenspeed_kernel.ops.moe.bf16_moe_routes`."""
    _check_bf16_route_tensors(x, w13, w2, topk_ids)
    if x.dtype != torch.bfloat16:
        raise TypeError("x must use torch.bfloat16")
    if expert_start < 0:
        raise ValueError("expert_start must be non-negative")
    return _bf16_route_outputs(
        x,
        w13,
        w2,
        topk_ids,
        "swiglu",
        1.0,
        None,
        expert_start,
        False,
        output,
    )


@register_kernel(
    "moe",
    "apply",
    name="triton_bf16_precomputed_moe_apply",
    solution="triton",
    signatures=format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"unquant"}),
        "activation": frozenset({"silu", "situ", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({32}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def triton_bf16_precomputed_moe_apply(
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
    """Apply an FP16/BF16 Triton MoE using precomputed top-k routing.

    Args:
        plan: MoE plan selecting standard SiLU/SwiGLU or SiTU activation.
            Opt-in ``precision_policy="bf16_route_accum_fp32"`` rounds GEMM1
            gate/up outputs to BF16 before FP32 SiLU, rounds every weighted
            BF16 GEMM2 output to BF16, accumulates those routes in FP32, and
            applies the required ``output_scale`` before the final BF16 store.
            Route weights must already be normalized, unscaled FP32 values.
        x: Contiguous hidden states `[tokens, hidden]` in FP16 or BF16.
        w: Module with contiguous `w13_weight` `[E, 2I, H]` and
            `w2_weight` `[E, H, I]` tensors matching `x.dtype`.
        router_logits: Unused because routing must be precomputed.
        topk_weights: Route weights `[tokens, top_k]`.
        topk_ids: Expert ids `[tokens, top_k]`. Out-of-range ids contribute zero.
        num_tokens_global: Unused; distributed expert parallelism is unsupported.
        max_num_tokens_per_gpu: Unused token-capacity hint.
        do_finalize: Must be true.
        enable_pdl: Unused launch hint.

    Returns:
        Finalized hidden states `[tokens, hidden]` with dtype matching `x`.
    """
    topk_weights, topk_ids, activation = _validate_bf16_moe(
        plan, x, w, topk_weights, topk_ids, do_finalize
    )
    return _triton_bf16_moe_impl(
        x,
        w.w13_weight,
        w.w2_weight,
        topk_weights,
        topk_ids,
        activation,
        float(getattr(w, "activation_situ_beta", 1.0) or 1.0),
        getattr(w, "activation_situ_linear_beta", None),
        plan.get("precision_policy", "default") == "bf16_route_accum_fp32",
        float(plan.get("output_scale", 1.0)),
    )


# ===-----------------------------------------------------------------------===#
# FP8 block-scaled experts (NVIDIA SM89+, AMD gfx950+)
# ===-----------------------------------------------------------------------===#


@register_kernel(
    "moe",
    "apply",
    name="triton_fp8_block_precomputed_moe_apply",
    solution="triton",
    capability=CapabilityRequirement(
        vendors=frozenset({"amd", "nvidia"}),
        vendor_min_arch_versions={
            "amd": ArchVersion(9, 5),
            "nvidia": ArchVersion(8, 9),
        },
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"fp8"}),
        "activation": frozenset({"silu", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({128}),
        "hidden_alignment": frozenset({128}),
        "internal_activation_dtype": frozenset({"input"}),
        "fp8_scale_block_shape": frozenset({(128, 128)}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def triton_fp8_block_precomputed_moe_apply(
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
    """Apply 128x128 block-scaled E4M3 experts to BF16 activations.

    Weight tiles are upcast to BF16 exactly and each block's FP32 inverse scale
    multiplies its FP32 partial product, so neither the weights nor the
    activations are requantized.

    Args:
        plan: MoE plan selecting SiLU/SwiGLU activation. A `swiglu_arg` limit,
            alpha, or `swiglu_beta` on `w` applies to either name.
        x: Contiguous BF16 hidden states `[tokens, hidden]`.
        w: Module with contiguous E4M3 `w13_weight` `[E, 2I, H]` and
            `w2_weight` `[E, H, I]`, plus FP32 `w13_weight_scale_inv`
            `[E, 2I/128, H/128]` and `w2_weight_scale_inv` `[E, H/128, I/128]`.
        router_logits: Unused because routing must be precomputed.
        topk_weights: Route weights `[tokens, top_k]`.
        topk_ids: Expert ids `[tokens, top_k]`. Out-of-range ids contribute zero.
        num_tokens_global: Unused; distributed expert parallelism is unsupported.
        max_num_tokens_per_gpu: Unused token-capacity hint.
        do_finalize: Must be true.
        enable_pdl: Unused launch hint.

    Returns:
        Finalized BF16 hidden states `[tokens, hidden]`.
    """
    topk_weights, topk_ids = _validate_fp8_moe(
        plan, x, w, topk_weights, topk_ids, do_finalize
    )
    return _triton_fp8_moe_impl(x, w, topk_weights, topk_ids)


# ===-----------------------------------------------------------------------===#
# FP8 per-channel experts (NVIDIA SM90-SM103)
# ===-----------------------------------------------------------------------===#


@register_kernel(
    "moe",
    "apply",
    name="triton_fp8_channel_precomputed_moe_apply",
    solution="triton",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(10, 3),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"fp8_channel"}),
        "activation": frozenset({"silu", "swiglu"}),
        "swiglu_form": frozenset({"standard"}),
        "activation_clamped": frozenset({False}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False, True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({32}),
        "hidden_alignment": frozenset({128}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def triton_fp8_channel_precomputed_moe_apply(
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
    """Apply FP8 per-output-channel experts with dynamic per-token activations.

    Args:
        plan: MoE plan for standard SiLU/SwiGLU and the default precision
            policy. The rounding points are those of the
            ``_triton/fp8_channel.py`` docstring; fold any routed scale into
            ``topk_weights``.
        x: Contiguous BF16 hidden states ``[tokens, hidden]``.
        w: Module with contiguous FP8 E4M3 ``w13_weight`` ``[E, 2I, H]`` (gate
            rows, then up rows), FP32 ``w13_weight_scale`` ``[E, 2I]``, FP8
            E4M3 ``w2_weight`` ``[E, H, I]`` and FP32 ``w2_weight_scale``
            ``[E, H]``; ``E`` is the local expert count. Under expert
            parallelism ``ep_rank`` places the local experts at global ids
            ``[ep_rank * E, (ep_rank + 1) * E)``.
        router_logits: Unused because routing must be precomputed.
        topk_weights: FP32 route weights ``[tokens, top_k]``.
        topk_ids: Global expert ids ``[tokens, top_k]``. Ids outside this
            rank's experts contribute zero, so each rank's output sums its
            local routes only.
        num_tokens_global: Unused; there is no all-to-all.
        max_num_tokens_per_gpu: Unused token-capacity hint.
        do_finalize: Must be true.
        enable_pdl: Unused launch hint.

    Returns:
        Finalized BF16 hidden states ``[tokens, hidden]``.
    """
    expert_offset = _validate_fp8_channel_moe(
        plan, x, w, topk_weights, topk_ids, do_finalize
    )
    return _triton_fp8_channel_moe_impl(
        x,
        w.w13_weight,
        w.w13_weight_scale,
        w.w2_weight,
        w.w2_weight_scale,
        topk_weights,
        topk_ids,
        expert_offset,
    )


# ===-----------------------------------------------------------------------===#
# FP8 per-channel experts with fused FP32 activation
# ===-----------------------------------------------------------------------===#


@register_kernel(
    "moe",
    "apply",
    name="triton_fp8_channel_fused_precomputed_moe_apply",
    solution="triton",
    capability=CapabilityRequirement(
        vendors=frozenset({"amd"}),
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"fp8_channel"}),
        "activation": frozenset({"silu", "swiglu"}),
        "swiglu_form": frozenset({"standard"}),
        "activation_clamped": frozenset({False}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False, True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({32}),
        "hidden_alignment": frozenset({128}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def triton_fp8_channel_fused_precomputed_moe_apply(
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
    """Apply standard FP8 channel experts, retaining FP32 gate/up until SiLU.

    Arguments and resident weight layout match
    ``triton_fp8_channel_precomputed_moe_apply``. The selected kernel explicitly
    chooses fused FP32 activation and BF16-before-E4M3 route quantization;
    it preserves original FP32 route weights and finalizes to BF16. Expert
    parallelism filters global expert IDs to this rank's contiguous range.
    Input rows use the public token quantizer. No vendor weight shuffle is used.
    Returns finalized BF16 hidden states with the shape of ``x``.
    """
    if enable_pdl:
        raise ValueError("Portable FP8 channel MoE does not support enable_pdl=True")
    expert_offset = _validate_fp8_channel_moe(
        plan, x, w, topk_weights, topk_ids, do_finalize
    )
    return _triton_fp8_channel_fused_moe_impl(
        x,
        w.w13_weight,
        w.w13_weight_scale,
        w.w2_weight,
        w.w2_weight_scale,
        topk_weights,
        topk_ids,
        expert_offset,
    )


# ===-----------------------------------------------------------------------===#
# MXFP4 experts (NVIDIA, AMD)
# ===-----------------------------------------------------------------------===#


def triton_mxfp4_moe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Validate that the module retains the linear OCP MXFP4 weight layout."""
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    if any(not hasattr(w, name) for name in names):
        raise ValueError("linear MXFP4 MoE weights are incomplete")
    tensors = tuple(getattr(w, name) for name in names)
    if any(t.dtype != torch.uint8 for t in tensors):
        raise TypeError("MXFP4 packed values and scales must use torch.uint8")
    if any(not t.is_cuda or not t.is_contiguous() for t in tensors):
        raise ValueError("linear MXFP4 weights must be contiguous GPU tensors")
    if any(t.ndim != 3 for t in tensors):
        raise ValueError("linear MXFP4 weights must be rank-3")
    if len({t.shape[0] for t in tensors}) != 1:
        raise ValueError("linear MXFP4 weights must share an expert axis")
    layout = getattr(w, "w13_input_layout", "concatenated")
    if layout not in {
        "concatenated",
        "interleaved",
    }:
        raise ValueError("Triton MXFP4 MoE received an invalid gate/up weight layout")
    if layout == "interleaved" and plan.get("internal_activation_dtype") != "input":
        raise ValueError("dynamic MXFP4 MoE requires concatenated gate/up weights")
    activation = plan.get("activation") or getattr(w, "activation", "silu")
    if activation not in {"silu", "situ", "swiglu"}:
        raise ValueError(f"Triton MXFP4 MoE does not support {activation!r}")


@register_kernel(
    "moe",
    "apply",
    name="triton_mxfp4_precomputed_moe_apply",
    solution="triton",
    weight_preprocessor=triton_mxfp4_moe_weights,
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"silu", "situ", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False, True}),
        "supports_ep": frozenset({False, True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({32, 128}),
        "internal_activation_dtype": frozenset({"input", "mxfp4"}),
        "supports_bias": frozenset({False, True}),
    },
    priority=Priority.PORTABLE,
)
def triton_mxfp4_precomputed_moe_apply(
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
) -> torch.Tensor | tuple[torch.Tensor, None, torch.Tensor]:
    """Apply experts with dynamic MXFP4 activations or weight-only MXFP4.

    Only the default precision policy and unit ``output_scale`` are supported.
    Expert outputs are stored in the input dtype, then weighted and summed
    in FP32 before the final store. Fold any routed scale into ``topk_weights``.

    Args:
        plan: MoE plan selecting standard SiLU/SwiGLU or SiTU activation.
        x: Contiguous FP16/BF16 hidden states `[tokens, hidden]`.
        w: Module holding linear packed E2M1 `w13_weight`/`w2_weight` and
            E8M0 `w13_weight_scale`/`w2_weight_scale` tensors.
        router_logits: Unused because routing must be precomputed.
        topk_weights: Route weights `[tokens, top_k]`.
        topk_ids: Global expert ids `[tokens, top_k]`. Experts outside this
            rank's contiguous shard contribute zero.
        num_tokens_global: Unused; EP reduction stays with the caller.
        max_num_tokens_per_gpu: Unused token-capacity hint.
        do_finalize: Whether to reduce the weighted routes into hidden states.
        enable_pdl: Unused launch hint.

    Returns:
        Finalized hidden states `[tokens, hidden]` with dtype matching `x`, or
        `(route_outputs, None, expanded_idx)` for deferred finalization. The
        map holds canonical route rows for local experts and `-1` for remote
        experts.
    """
    del router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
    return _triton_mxfp4_moe_impl(
        plan,
        x,
        w,
        topk_weights,
        topk_ids,
        do_finalize,
    )


# ===-----------------------------------------------------------------------===#
# BF16 latent-MoE input projections
# ===-----------------------------------------------------------------------===#


register_kernel(
    "moe",
    "latent_input",
    name="triton_latent_input_packed",
    solution="triton",
    signatures=frozenset(
        {
            format_signature(
                hidden_states=dense_tensor_format(torch.bfloat16),
                router_weight=dense_tensor_format(torch.bfloat16),
                routed_weight=dense_tensor_format(torch.bfloat16),
                shared_gate_up_weight=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    # Below the hand-written decode kernels, which stay ahead at one token.
    priority=Priority.PERFORMANT,
    traits={
        "weights_packed": frozenset({True}),
        "inputs_contiguous": frozenset({True}),
    },
)(triton_latent_input_packed)


register_kernel(
    "moe",
    "sigmoid_bias_topk_routing",
    name="triton_sigmoid_bias_topk_routing",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures(
        "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    priority=Priority.PERFORMANT,
)(triton_sigmoid_bias_topk_routing)
