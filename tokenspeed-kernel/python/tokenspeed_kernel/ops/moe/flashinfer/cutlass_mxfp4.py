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

"""FlashInfer CUTLASS mixed-input MoE for MXFP4 experts on Hopper (SM90).

Two registrations of the same grouped GEMM: W4A16 keeps BF16 activations and
dequantizes the E2M1 weights in the mainloop; W4A8 ("Humming") also quantizes
the activations to FP8 and folds the FP4->FP8 exponent bias into per-expert
residual scales. Both read the loader's ``[gate; up]`` MXFP4 rows and rewrite
them once into FlashInfer's SM90 interleaved layout. The E8M0 group-32 scales
travel as int32 views, which is why hidden and intermediate sizes must be
multiples of 128.

Marlin remains the Hopper solution for SiTU experts (Kimi-K3) and for DeepEP
all-to-all layouts; this one serves SiLU/SwiGLU experts with dense EP.

Routing carries global expert ids and the kernel drops the ids outside its EP
shard. A token's ids must be distinct: FlashInfer's permutation assumes one
row per (token, expert) and returns wrong sums when an expert repeats, so a
caller must not mask unwanted slots by pointing them at a placeholder expert
with zero weight. Top-k routing never repeats an expert.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.tuning import get_autotune_max_num_tokens
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

MXFP4_GROUP = 32
# quant_scales are E8M0 bytes viewed as int32 along K, so K/32 must be a
# multiple of 4 for both GEMMs: hidden (fc1) and intermediate (fc2).
_ALIGNMENT = 4 * MXFP4_GROUP
_W4A16 = "w4a16"
_W4A8 = "w4a8"

platform = current_platform()

if platform.is_nvidia:
    from flashinfer import ActivationType, cutlass_fused_moe
    from flashinfer.fused_moe import (
        interleave_moe_scales_for_sm90_mixed_gemm,
        interleave_moe_weights_for_sm90_mixed_gemm,
        preprocess_moe_weights_for_sm90_mixed_gemm_humming,
    )

    def _swiglu_limit(w: torch.nn.Module) -> float | None:
        """The checkpoint's SwiGLU clamp, or None for a plain ``silu(gate) * up``.

        The CUTLASS epilogue clips ``gate`` from above and ``up`` on both sides
        when a limit is given. A sigmoid multiplier (``alpha``) or an up-branch
        offset (``swiglu_beta``) is rejected instead of silently dropped.
        """
        swiglu_arg = getattr(w, "swiglu_arg", None)
        alpha = None if swiglu_arg is None else getattr(swiglu_arg, "alpha", None)
        beta = getattr(w, "swiglu_beta", None)
        if alpha not in (None, 1.0) or beta not in (None, 0.0):
            raise ValueError(
                "FlashInfer cutlass MXFP4 MoE supports only standard SwiGLU with "
                f"an optional clamp limit; got alpha={alpha!r}, swiglu_beta={beta!r}"
            )
        if swiglu_arg is None:
            return None
        limit = getattr(swiglu_arg, "limit", None)
        return None if limit is None else float(limit)

    def _up_gate(t: torch.Tensor) -> torch.Tensor:
        """Reorder ``[gate; up]`` row halves to FlashInfer's ``[up; gate]``."""
        half = t.shape[1] // 2
        return torch.cat((t[:, half:], t[:, :half]), dim=1).contiguous()

    def _loader_layout(
        w: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate the loader's MXFP4 tensors and return them in kernel order.

        Returns ``(w13, w13_scale, w2, w2_scale)`` with the w13 halves swapped
        to ``[up; gate]``; the w2 tensors are returned contiguous.
        """
        names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
        if any(not hasattr(w, name) for name in names):
            raise ValueError("MXFP4 MoE weights are incomplete for FlashInfer cutlass")
        if getattr(w, "w13_input_layout", "concatenated") != "concatenated":
            raise ValueError(
                "FlashInfer cutlass MXFP4 MoE needs the concatenated [w1 | w3] row "
                "layout; interleaved gate/up rows are not supported"
            )
        w13, w13_scale = w.w13_weight.data, w.w13_weight_scale.data
        w2, w2_scale = w.w2_weight.data, w.w2_weight_scale.data
        for name, tensor in zip(names, (w13, w13_scale, w2, w2_scale), strict=True):
            if tensor.dtype != torch.uint8 or tensor.ndim != 3:
                raise ValueError(f"{name} must be a rank-3 uint8 MXFP4 tensor")
        num_experts, two_ispp, packed_hidden = w13.shape
        hidden = packed_hidden * 2
        ispp = w2.shape[2] * 2
        if two_ispp != 2 * ispp or w2.shape[0] != num_experts or w2.shape[1] != hidden:
            raise ValueError(
                f"w13 {tuple(w13.shape)} and w2 {tuple(w2.shape)} disagree on "
                "experts, hidden or intermediate size"
            )
        if w13_scale.shape != (num_experts, two_ispp, hidden // MXFP4_GROUP):
            raise ValueError(f"w13_weight_scale has shape {tuple(w13_scale.shape)}")
        if w2_scale.shape != (num_experts, hidden, ispp // MXFP4_GROUP):
            raise ValueError(f"w2_weight_scale has shape {tuple(w2_scale.shape)}")
        if hidden % _ALIGNMENT != 0 or ispp % _ALIGNMENT != 0:
            raise ValueError(
                f"FlashInfer cutlass MXFP4 needs hidden%{_ALIGNMENT}==0 and "
                f"ispp%{_ALIGNMENT}==0, got hidden={hidden}, ispp={ispp}"
            )
        return (
            _up_gate(w13),
            _up_gate(w13_scale),
            w2.contiguous(),
            w2_scale.contiguous(),
        )

    def _begin_preprocess(w: torch.nn.Module, layout: str) -> bool:
        """Return False when ``w`` already carries ``layout``; reject the other one."""
        existing = getattr(w, "_flashinfer_cutlass_mxfp4_layout", None)
        if existing == layout:
            return False
        if existing is not None:
            raise ValueError(
                f"MXFP4 experts are already interleaved for {existing}; they "
                f"cannot be re-laid out for {layout}"
            )
        return True

    def _finish_preprocess(
        w: torch.nn.Module,
        layout: str,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        num_experts = w13.shape[0]
        limit = _swiglu_limit(w)
        w.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        w.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        w.w13_weight_scale = torch.nn.Parameter(
            w13_scale.view(torch.int32), requires_grad=False
        )
        w.w2_weight_scale = torch.nn.Parameter(
            w2_scale.view(torch.int32), requires_grad=False
        )
        w.swiglu_limit_t = (
            None
            if limit is None
            else torch.nn.Parameter(
                torch.full(
                    (num_experts,), limit, dtype=torch.float32, device=w13.device
                ),
                requires_grad=False,
            )
        )
        w._flashinfer_cutlass_mxfp4_layout = layout

    def flashinfer_cutlass_mxfp4_w4a16_moe_weights(plan: dict, w: torch.nn.Module):
        """Interleave loader-format MXFP4 experts for the SM90 W4A16 GEMM, once."""
        del plan
        if not _begin_preprocess(w, _W4A16):
            return
        w13, w13_scale, w2, w2_scale = _loader_layout(w)
        _finish_preprocess(
            w,
            _W4A16,
            interleave_moe_weights_for_sm90_mixed_gemm(w13, "fp4"),
            interleave_moe_scales_for_sm90_mixed_gemm(
                w13_scale, group_size=MXFP4_GROUP
            ),
            interleave_moe_weights_for_sm90_mixed_gemm(w2, "fp4"),
            interleave_moe_scales_for_sm90_mixed_gemm(w2_scale, group_size=MXFP4_GROUP),
        )

    def flashinfer_cutlass_mxfp4_w4a8_moe_weights(plan: dict, w: torch.nn.Module):
        """Interleave for the SM90 W4A8 GEMM and attach its residual scales, once.

        Humming keeps the FP4->FP8 exponent-bias compensation (2^6) in the
        epilogue through per-expert residual scales; the FC2 activation scale
        is a fixed 1.0 because the SwiGLU clamp already bounds that input.
        """
        del plan
        if _swiglu_limit(w) is None:
            # Selection already requires activation_clamped; a forced solution
            # must not get past it either (fc2_act_scale below assumes the bound).
            raise ValueError(
                "FlashInfer cutlass W4A8 needs the checkpoint's SwiGLU clamp to "
                "bound the FC2 input; this layer's activation is unclamped"
            )
        if not _begin_preprocess(w, _W4A8):
            return
        w13, w13_scale, w2, w2_scale = _loader_layout(w)
        w13_il, w13_scale_il, w13_residual = (
            preprocess_moe_weights_for_sm90_mixed_gemm_humming(w13, w13_scale)
        )
        w2_il, w2_scale_il, w2_residual = (
            preprocess_moe_weights_for_sm90_mixed_gemm_humming(w2, w2_scale)
        )
        _finish_preprocess(w, _W4A8, w13_il, w13_scale_il, w2_il, w2_scale_il)
        w.w13_weight_residual = torch.nn.Parameter(
            (w13_residual * 64.0).contiguous(), requires_grad=False
        )
        w.w2_weight_residual = torch.nn.Parameter(
            (w2_residual * 64.0).contiguous(), requires_grad=False
        )
        w.fc2_act_scale = torch.nn.Parameter(
            torch.ones((), dtype=torch.float32, device=w13_il.device),
            requires_grad=False,
        )

    def _cutlass_mxfp4_apply(
        x: torch.Tensor,
        w: torch.nn.Module,
        topk_weights: torch.Tensor | None,
        topk_ids: torch.Tensor | None,
        do_finalize: bool,
        enable_pdl: bool,
        layout: str,
        quant_scales: list[torch.Tensor],
        humming: bool,
    ) -> torch.Tensor:
        if not do_finalize:
            raise ValueError("FlashInfer cutlass MXFP4 MoE cannot defer finalization")
        if topk_weights is None or topk_ids is None:
            raise ValueError("FlashInfer cutlass MXFP4 MoE requires precomputed top-k")
        if x.dtype != torch.bfloat16:
            raise TypeError(
                f"FlashInfer cutlass MXFP4 MoE requires bf16 activations, got {x.dtype}"
            )
        if getattr(w, "_flashinfer_cutlass_mxfp4_layout", None) != layout:
            raise ValueError(f"MXFP4 experts were not preprocessed for {layout}")
        output = torch.empty(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
        # Expert ids stay global: FlashInfer keeps the ``ep_rank`` slice and
        # contributes zero for the others, like the marlin EP mask.
        return cutlass_fused_moe(
            output=output,
            input=x,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights.to(torch.float32),
            fc1_expert_weights=w.w13_weight,
            fc2_expert_weights=w.w2_weight,
            output_dtype=x.dtype,
            quant_scales=quant_scales,
            input_sf=None,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=w.swiglu_limit_t,
            tp_size=getattr(w, "tp_size", 1),
            tp_rank=getattr(w, "tp_rank", 0),
            ep_size=getattr(w, "ep_size", 1),
            ep_rank=getattr(w, "ep_rank", 0),
            use_w4_group_scaling=True,
            use_mxfp8_act_scaling=False,
            use_wfp4afp8_humming=humming,
            activation_type=ActivationType.Swiglu,
            tune_max_num_tokens=get_autotune_max_num_tokens(),
            enable_pdl=enable_pdl,
        )[0]

    _CAPABILITY = CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        # The *_for_sm90_mixed_gemm layouts feed FlashInfer's fused_moe_90
        # module only; Blackwell consumes MXFP4 through other layouts.
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(9, 0),
    )

    def _traits(internal_activation_dtype: str) -> dict:
        traits = {
            "weight_dtype": frozenset({"mxfp4"}),
            # The CUTLASS epilogue is gated SiLU (== SwiGLU); SiTU stays on marlin.
            "activation": frozenset({"silu", "swiglu"}),
            # No alpha/beta in the epilogue (rejected by _swiglu_limit), and
            # FlashInfer's permutation needs distinct expert ids per token, so
            # generalized SwiGLU (MiniMax-M3) and zero-expert routing
            # (LongCat) stay on other kernels at plan time.
            "swiglu_form": frozenset({"standard"}),
            "expert_id_repeats": frozenset({False}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            # Both GEMM K dims carry int32-viewed E8M0 scales (see _ALIGNMENT);
            # declaring them here keeps other widths on marlin at plan time.
            "ispp_alignment": frozenset({_ALIGNMENT}),
            "hidden_alignment": frozenset({_ALIGNMENT}),
            "internal_activation_dtype": frozenset({internal_activation_dtype}),
            "supports_bias": frozenset({False}),
        }
        if internal_activation_dtype == "fp8":
            # Humming feeds FC2 through a fixed activation scale of 1.0, which
            # is sound only when the SwiGLU clamp bounds silu(gate) * up
            # (10 * 10 for DeepSeek-V4.1); an unbounded activation would
            # saturate the FP8 conversion, so unclamped layers never plan here.
            traits["activation_clamped"] = frozenset({True})
        return traits

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_cutlass_mxfp4_w4a16_moe_apply",
        solution="flashinfer_cutlass",
        weight_preprocessor=flashinfer_cutlass_mxfp4_w4a16_moe_weights,
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits=_traits("input"),
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutlass_mxfp4_w4a16_moe_apply(
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
        """Apply MXFP4 experts with BF16 activations (SM90 mixed-input GEMM).

        Args:
            plan: MoE plan; routing is precomputed, so only the kernel identity
                is read from it.
            x: bf16 hidden states ``[tokens, hidden]``.
            w: Module holding the SM90-interleaved ``w13_weight``/``w2_weight``
                (uint8) and ``w13_weight_scale``/``w2_weight_scale`` (int32 views
                of E8M0 bytes) written by the preprocessor, ``swiglu_limit_t``,
                and ``ep_size``/``ep_rank``/``tp_size``/``tp_rank``.
            router_logits: Unused; routing is precomputed.
            topk_weights: Route weights ``[tokens, top_k]``.
            topk_ids: Global expert ids ``[tokens, top_k]``.
            num_tokens_global: Unused; distributed EP dispatch is not owned here.
            max_num_tokens_per_gpu: Unused capacity hint.
            do_finalize: Must be true (no deferred finalize).
            enable_pdl: Programmatic dependent launch hint for the kernel.

        Returns:
            Finalized hidden states ``[tokens, hidden]`` in the dtype of ``x``.
        """
        del plan, router_logits, num_tokens_global, max_num_tokens_per_gpu
        return _cutlass_mxfp4_apply(
            x,
            w,
            topk_weights,
            topk_ids,
            do_finalize,
            enable_pdl,
            _W4A16,
            [w.w13_weight_scale, w.w2_weight_scale],
            False,
        )

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_cutlass_mxfp4_w4a8_moe_apply",
        solution="flashinfer_cutlass",
        weight_preprocessor=flashinfer_cutlass_mxfp4_w4a8_moe_weights,
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits=_traits("fp8"),
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutlass_mxfp4_w4a8_moe_apply(
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
        """Apply MXFP4 experts with FP8 activations (SM90 Humming GEMM).

        Same contract as :func:`flashinfer_cutlass_mxfp4_w4a16_moe_apply`; the
        module additionally carries ``w13_weight_residual``,
        ``w2_weight_residual`` and ``fc2_act_scale`` from its preprocessor. The
        FP8 activation rounding costs a few percent of relative error on the
        expert outputs, so this variant is an explicit opt-in.
        """
        del plan, router_logits, num_tokens_global, max_num_tokens_per_gpu
        return _cutlass_mxfp4_apply(
            x,
            w,
            topk_weights,
            topk_ids,
            do_finalize,
            enable_pdl,
            _W4A8,
            [
                w.w13_weight_scale,
                w.w13_weight_residual,
                w.fc2_act_scale,
                w.w2_weight_scale,
                w.w2_weight_residual,
            ],
            True,
        )
