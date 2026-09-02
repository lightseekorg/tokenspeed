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

from types import SimpleNamespace

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_FP8_BLOCK = 128
_DIRECT_DECODE_MAX_TOKENS = 32
_EXACT_MFMA_MAX_TOKENS = 4096

platform = current_platform()


def _validate(
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
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8 import (
        gluon_fp8_block_dequantize,
        gluon_fp8_block_exact_mfma_moe,
        gluon_fp8_block_warp_decode_moe,
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
        validated_weights, validated_ids, swiglu_limit = _validate(
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

        from tokenspeed_kernel.ops.moe.gluon.bf16 import (
            gluon_bf16_precomputed_moe_apply,
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
