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

import dataclasses
import functools
import math
import warnings

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.tuning import get_autotune_max_num_tokens, is_autotuning
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()


# ===-----------------------------------------------------------------------===#
# BF16 experts: CUTLASS (NVIDIA SM89+)
# ===-----------------------------------------------------------------------===#


if platform.is_nvidia:
    from flashinfer import ActivationType, cutlass_fused_moe
    from flashinfer.fused_moe import cutlass_fused_moe_workspace_size

    def flashinfer_cutlass_unquant_moe_weights(plan: dict, w: torch.nn.Module):
        half_w = w.w13_weight.shape[1] // 2
        first_half = w.w13_weight.data[:, :half_w, :].clone()
        w.w13_weight.data[:, :half_w, :] = w.w13_weight.data[:, half_w:, :]
        w.w13_weight.data[:, half_w:, :] = first_half
        return None

    @functools.cache
    def _warn_pdl_forced_off() -> None:
        warnings.warn(
            "flashinfer_cutlass_unquant_moe_apply forces enable_pdl off: PDL "
            "races inside this fused-MoE chain on SM90 (transient NaN GEMM "
            "rows at decode-sized batches).",
            stacklevel=3,
        )

    # One persistent scratch buffer per device, handed to every call as
    # ``workspace_buffer``. Left to itself the runner allocates its scratch
    # uninitialized on every call, and on SM90 the chain then reads bytes it
    # never wrote: for some (EP rank, routing) combinations the rank's whole
    # routed output came back NaN for finite inputs, reproducibly for that
    # call, while the identical call with any caller-provided buffer -- or
    # with fresh allocator state -- matched the fp32 reference. The buffer
    # is zero-filled when (re)allocated and only ever holds this chain's own
    # writes afterwards. Consecutive MoE layers are stream-ordered through
    # their activations, so one buffer per device is never in use twice.
    _workspaces: dict[torch.device, torch.Tensor] = {}

    def cutlass_unquant_moe_workspace(
        *,
        num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts_total: int,
        top_k: int,
        x_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
        device: torch.device,
    ) -> torch.Tensor:
        """The device's persistent workspace, grown (and re-zeroed) on demand.

        Args:
            num_tokens: Rows of this call; the buffer is sized for the largest
                row count seen so far on the device.
            hidden_size: Model hidden size (``w2.shape[1]``).
            intermediate_size: Per-partition intermediate size (``w2.shape[2]``).
            num_experts_total: Experts across the EP group.
            top_k: Routed experts per token.
            x_dtype: Activation dtype.
            weight_dtype: Expert weight dtype.
            tp_size: MoE tensor-parallel width.
            tp_rank: This rank's MoE tensor-parallel coordinate.
            ep_size: Expert-parallel width.
            ep_rank: This rank's expert-parallel coordinate.
            device: CUDA device the call runs on.

        Returns:
            A 1-D ``uint8`` CUDA tensor of at least the bytes
            :func:`flashinfer.fused_moe.cutlass_fused_moe_workspace_size`
            asks for ``num_tokens``.
        """
        needed = cutlass_fused_moe_workspace_size(
            max(int(num_tokens), 1),
            hidden_size,
            intermediate_size,
            num_experts_total,
            top_k,
            x_dtype=x_dtype,
            weight_dtype=weight_dtype,
            output_dtype=x_dtype,
            activation_type=ActivationType.Swiglu,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            device=device,
        )
        workspace = _workspaces.get(device)
        if workspace is None or workspace.numel() < needed:
            workspace = torch.zeros(needed, dtype=torch.uint8, device=device)
            _workspaces[device] = workspace
        return workspace

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_cutlass_unquant_moe_apply",
        solution="flashinfer_cutlass",
        weight_preprocessor=flashinfer_cutlass_unquant_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(8, 9),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"unquant"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutlass_unquant_moe_apply(
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
        if enable_pdl:
            # Not silently: a caller enabling PDL globally would otherwise
            # misread its perf measurements with nothing in the logs.
            _warn_pdl_forced_off()
        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)
        ep_size = getattr(w, "ep_size", 1)
        ep_rank = getattr(w, "ep_rank", 0)
        tp_size = getattr(w, "tp_size", 1)
        tp_rank = getattr(w, "tp_rank", 0)
        workspace = cutlass_unquant_moe_workspace(
            num_tokens=x.shape[0],
            hidden_size=w.w2_weight.shape[1],
            intermediate_size=w.w2_weight.shape[2],
            num_experts_total=w.w13_weight.shape[0] * ep_size,
            top_k=topk_ids.shape[1],
            x_dtype=x.dtype,
            weight_dtype=w.w13_weight.dtype,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            device=x.device,
        )
        return cutlass_fused_moe(
            input=x,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=w.w13_weight,
            fc2_expert_weights=w.w2_weight,
            output_dtype=x.dtype,
            quant_scales=None,
            ep_size=ep_size,
            ep_rank=ep_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tune_max_num_tokens=get_autotune_max_num_tokens(),
            activation_type=ActivationType.Swiglu,
            # PDL races inside this fused-MoE chain on SM90 at decode-sized
            # batches: a routed GEMM row transiently reads NaN (rerunning the
            # identical call is clean). Keep the chain fully serialized until
            # the flashinfer kernels are fixed.
            enable_pdl=False,
            workspace_buffer=workspace,
        )[0]


# ===-----------------------------------------------------------------------===#
# BF16 experts: TRT-LLM (NVIDIA SM100)
# ===-----------------------------------------------------------------------===#


# Intermediate-size multiple the SiLU/SwiGLU kernels accept; ReLU2 keeps 128.
TRTLLM_UNQUANT_ISPP_ALIGNMENT = 128

_RELU2_TRAITS = {
    "weight_dtype": frozenset({"unquant"}),
    "activation": frozenset({"relu2"}),
    "supports_deferred_finalize": frozenset({True}),
    "supports_ep": frozenset({True}),
    "supports_all_to_all_ep": frozenset({False}),
    "ispp_alignment": frozenset({128}),
    "internal_activation_dtype": frozenset({"input"}),
    "supports_bias": frozenset({False}),
}


if platform.is_nvidia:
    from flashinfer import trtllm_bf16_moe
    from flashinfer.fused_moe import ActivationType, trtllm_bf16_routed_moe
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices as maybe_get_cached_w3_w1_permute_indices,
    )
    from flashinfer.fused_moe.core import (
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )
    from tokenspeed_kernel.thirdparty.flashinfer import (
        trtllm_bf16_moe as ispp64_launcher,
    )

    # Only the SM100-SM103 kernels below use the private launcher; other GPUs
    # keep 128 without checking it or FlashInfer's JIT.
    if ArchVersion(10, 0) <= platform.arch_version <= ArchVersion(10, 3):
        TRTLLM_UNQUANT_ISPP_ALIGNMENT = ispp64_launcher.gated_ispp_alignment()

    def _flashinfer_trtllm_unquant_moe_weights(w: torch.nn.Module, *, gated: bool):
        cache_permute_indices = {}
        num_experts = w.w13_weight.shape[0]
        epilogue_tile_m = 128
        block_k = 128

        if gated:
            # The fused gated activation expects [W3(up), W1(gate)].
            half_w = w.w13_weight.shape[1] // 2
            w1_weight = w.w13_weight.data[:, :half_w, :].clone()
            w.w13_weight.data[:, :half_w, :] = w.w13_weight.data[:, half_w:, :]
            w.w13_weight.data[:, half_w:, :] = w1_weight

        old_shape_w13 = w.w13_weight.data[0].shape
        old_shape_w2 = w.w2_weight.data[0].shape
        new_shape_w13 = old_shape_w13
        new_shape_w2 = old_shape_w2

        for idx in range(num_experts):
            permute_indices = maybe_get_cached_w3_w1_permute_indices(
                cache_permute_indices,
                w.w13_weight.data[idx].view(torch.uint8),
                epilogue_tile_m,
                is_gated_act_gemm=gated,
            )
            tmp_weights1 = (
                w.w13_weight.data[idx]
                .clone()
                .view(torch.uint8)[permute_indices.to(w.w13_weight.data.device)]
                .contiguous()
            )
            permute_indices = get_w2_permute_indices_with_cache(
                cache_permute_indices,
                w.w2_weight.data[idx].view(torch.uint8),
                epilogue_tile_m,
            )
            tmp_weights2 = (
                w.w2_weight.data[idx]
                .clone()
                .view(torch.uint8)[permute_indices.to(w.w2_weight.data.device)]
                .contiguous()
            )
            tmp_weights1 = convert_to_block_layout(
                tmp_weights1.view(torch.uint8), block_k
            )
            tmp_weights2 = convert_to_block_layout(
                tmp_weights2.view(torch.uint8), block_k
            )
            new_shape_w13 = tmp_weights1.view(torch.bfloat16).shape
            new_shape_w2 = tmp_weights2.view(torch.bfloat16).shape
            w.w13_weight.data[idx] = (
                tmp_weights1.view(torch.bfloat16).contiguous().reshape(old_shape_w13)
            )
            w.w2_weight.data[idx] = (
                tmp_weights2.view(torch.bfloat16).contiguous().reshape(old_shape_w2)
            )

        w.w13_weight.data = w.w13_weight.data.reshape(num_experts, *new_shape_w13)
        w.w2_weight.data = w.w2_weight.data.reshape(num_experts, *new_shape_w2)
        return None

    def flashinfer_trtllm_unquant_moe_weights(plan: dict, w: torch.nn.Module):
        return _flashinfer_trtllm_unquant_moe_weights(w, gated=True)

    def flashinfer_trtllm_unquant_relu2_moe_weights(plan: dict, w: torch.nn.Module):
        return _flashinfer_trtllm_unquant_moe_weights(w, gated=False)

    def _flashinfer_trtllm_unquant_moe_apply(
        x: torch.Tensor,
        w: torch.nn.Module,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor | None,
        topk_ids: torch.Tensor | None,
        do_finalize: bool,
        enable_pdl: bool,
        routed: bool,
        activation_type: ActivationType,
        fp32_correction_bias: bool,
    ):
        """Shared body for the in-kernel-routing and precomputed-topk variants.

        ``routed`` selects between ``trtllm_bf16_moe`` (in-kernel routing from
        ``router_logits``) and ``trtllm_bf16_routed_moe`` (precomputed
        ``topk_ids``/``topk_weights``). Precomputed routes use FlashInfer's
        unpacked (int32 IDs, BF16 weights) ABI, preserving the packed path's
        weight rounding; everything else is identical.
        ``activation_type`` is SwiGLU for gated experts and Relu2 for
        non-gated ones.
        ``fp32_correction_bias`` routes on a module that adds the DeepSeekV3
        correction bias in FP32.
        """
        if x.shape[0] == 0:
            # Idle DP ranks run a dummy forward with 0 tokens; the fused kernel
            # divides by the token count on the host. Skip the experts entirely.
            if do_finalize:
                return x
            return (
                x,
                x.new_empty((0, getattr(w, "top_k")), dtype=torch.bfloat16),
                x.new_empty((0,), dtype=torch.int32),
            )

        local_experts = getattr(w, "num_local_experts", w.w13_weight.shape[0])
        intermediate_size = getattr(w, "intermediate_size") // getattr(w, "tp_size", 1)
        # Sizes the stock launcher rejects run on the 64-aligned private one.
        bf16_moe, bf16_routed_moe = (
            (ispp64_launcher.trtllm_bf16_moe, ispp64_launcher.trtllm_bf16_routed_moe)
            if intermediate_size % ispp64_launcher.STOCK_ISPP_ALIGNMENT
            else (trtllm_bf16_moe, trtllm_bf16_routed_moe)
        )
        if fp32_correction_bias and not ispp64_launcher.stock_routing_keeps_fp32_bias():
            bf16_moe = ispp64_launcher.trtllm_bf16_fp32_routing_bias_moe
        # GEMM and sizing arguments shared by both kernel entry points.
        common_kwargs = dict(
            hidden_states=x,
            gemm1_weights=w.w13_weight,
            gemm2_weights=w.w2_weight,
            num_experts=getattr(w, "num_experts"),
            top_k=getattr(w, "top_k"),
            intermediate_size=intermediate_size,
            local_expert_offset=getattr(w, "ep_rank", 0) * local_experts,
            local_num_experts=local_experts,
            do_finalize=do_finalize,
            enable_pdl=enable_pdl,
            tune_max_num_tokens=get_autotune_max_num_tokens(),
            activation_type=int(activation_type),
        )

        if routed:
            # Preserve the packed path's BF16 rounding without constructing
            # score/index words. FlashInfer accepts these tensors directly.
            topk = (
                topk_ids.to(torch.int32).contiguous(),
                topk_weights.to(torch.bfloat16).contiguous(),
            )
            result = bf16_routed_moe(
                topk_ids=topk,
                n_group=None,
                topk_group=None,
                routed_scaling_factor=None,
                **common_kwargs,
            )
        else:
            routing_config = getattr(w, "routing_config", {})
            if not isinstance(routing_config, dict):
                routing_config = {}
            routing_value = lambda name, default: (
                routing_config[name]
                if name in routing_config
                else getattr(w, name, default)
            )
            routing_method_type = routing_value("routing_method_type", 1)
            routing_logits_dtype = (
                torch.float32 if int(routing_method_type) in {2, 7} else torch.bfloat16
            )
            routing_bias = routing_value("correction_bias", None)
            if routing_bias is not None:
                routing_bias = routing_bias.to(routing_logits_dtype)
            result = bf16_moe(
                routing_logits=router_logits.to(routing_logits_dtype),
                routing_bias=routing_bias,
                n_group=routing_value("n_group", None),
                topk_group=routing_value("topk_group", None),
                routed_scaling_factor=routing_value("routed_scaling_factor", None),
                routing_method_type=routing_method_type,
                **common_kwargs,
            )

        if do_finalize:
            if isinstance(result, (list, tuple)):
                return result[0]
            return result
        # Deferred: [gemm2_out, expert_weights, expanded_idx_to_permuted_idx].
        gemm2_out, expert_weights, expanded_idx = result
        if routed:
            # Shared-sink callers use their original route weights at finalize.
            return (gemm2_out, expert_weights, expanded_idx)
        # In-kernel routing may hand back an fp32-typed buffer that actually
        # holds bf16 data (see the NVFP4 TRT-LLM section); reinterpret to bf16
        # and keep the live prefix.
        if expert_weights.dtype == torch.float32:
            n, k = expert_weights.size()
            expert_weights = expert_weights.view(torch.bfloat16).view(-1, k)[:n]
        return (gemm2_out, expert_weights, expanded_idx)

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_unquant_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_unquant_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"unquant"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"kernel_routing"}),
            "supports_deferred_finalize": frozenset({True}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({TRTLLM_UNQUANT_ISPP_ALIGNMENT}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_unquant_moe_apply(
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
        return _flashinfer_trtllm_unquant_moe_apply(
            x,
            w,
            router_logits,
            topk_weights,
            topk_ids,
            do_finalize,
            enable_pdl,
            routed=False,
            activation_type=ActivationType.Swiglu,
            fp32_correction_bias=plan["fp32_correction_bias"],
        )

    # moe_plan keeps in-kernel routing for an FP32 correction bias only if
    # this returns True.
    flashinfer_trtllm_unquant_moe_apply._tokenspeed_fp32_correction_bias = (  # type: ignore[attr-defined]
        ispp64_launcher.fp32_routing_bias_ready
    )

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_unquant_routed_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_unquant_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
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
            "supports_deferred_finalize": frozenset({True}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({TRTLLM_UNQUANT_ISPP_ALIGNMENT}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        # Priority rationale: see the NVFP4 TRT-LLM routed registration.
        priority=Priority.PERFORMANT + 3,
    )
    def flashinfer_trtllm_unquant_routed_moe_apply(
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
        assert (
            topk_weights is not None and topk_ids is not None
        ), "precomputed_topk plan requires topk_weights and topk_ids"
        return _flashinfer_trtllm_unquant_moe_apply(
            x,
            w,
            router_logits,
            topk_weights,
            topk_ids,
            do_finalize,
            enable_pdl,
            routed=True,
            activation_type=ActivationType.Swiglu,
            fp32_correction_bias=False,
        )

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_unquant_relu2_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_unquant_relu2_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={**_RELU2_TRAITS, "routing_mode": frozenset({"kernel_routing"})},
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_unquant_relu2_moe_apply(
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
        return _flashinfer_trtllm_unquant_moe_apply(
            x,
            w,
            router_logits,
            topk_weights,
            topk_ids,
            do_finalize,
            enable_pdl,
            routed=False,
            activation_type=ActivationType.Relu2,
            # Without the FP32 correction-bias hook, moe_plan plans
            # precomputed top-k for layers that need that bias.
            fp32_correction_bias=False,
        )

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_unquant_relu2_routed_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_unquant_relu2_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={**_RELU2_TRAITS, "routing_mode": frozenset({"precomputed_topk"})},
        priority=Priority.PERFORMANT + 3,
    )
    def flashinfer_trtllm_unquant_relu2_routed_moe_apply(
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
        assert (
            topk_weights is not None and topk_ids is not None
        ), "precomputed_topk plan requires topk_weights and topk_ids"
        return _flashinfer_trtllm_unquant_moe_apply(
            x,
            w,
            router_logits,
            topk_weights,
            topk_ids,
            do_finalize,
            enable_pdl,
            routed=True,
            activation_type=ActivationType.Relu2,
            fp32_correction_bias=False,
        )


# ===-----------------------------------------------------------------------===#
# FP8 block-scaled experts: CUTLASS (NVIDIA SM90+)
# ===-----------------------------------------------------------------------===#


if platform.is_nvidia:
    from flashinfer import ActivationType, cutlass_fused_moe

    def flashinfer_cutlass_fp8_moe_weights(plan: dict, w: torch.nn.Module):
        half_w = w.w13_weight.shape[1] // 2
        first_half = w.w13_weight.data[:, :half_w, :].clone()
        w.w13_weight.data[:, :half_w, :] = w.w13_weight.data[:, half_w:, :]
        w.w13_weight.data[:, half_w:, :] = first_half

        half_s = w.w13_weight_scale_inv.shape[1] // 2
        first_scale = w.w13_weight_scale_inv.data[:, :half_s, :].clone()
        w.w13_weight_scale_inv.data[:, :half_s, :] = w.w13_weight_scale_inv.data[
            :, half_s:, :
        ]
        w.w13_weight_scale_inv.data[:, half_s:, :] = first_scale
        w.w13_weight_scale_inv.data.clamp_(min=1e-10)
        w.w2_weight_scale_inv.data.clamp_(min=1e-10)
        swiglu_arg = getattr(w, "swiglu_arg", None)
        if swiglu_arg is not None:
            # The cutlass gated activation runs post-dequant, so alpha/beta/
            # limit are passed in the model's actual-value domain; default
            # parameters stay None to keep the plain Swiglu kernel path.
            num_experts = w.w13_weight.shape[0]
            device = w.w13_weight.device

            def _per_expert(value: float) -> torch.nn.Parameter:
                return torch.nn.Parameter(
                    torch.full(
                        (num_experts,), float(value), dtype=torch.float32, device=device
                    ),
                    requires_grad=False,
                )

            alpha = swiglu_arg.alpha
            w.swiglu_alpha_t = (
                _per_expert(alpha)
                if alpha is not None and float(alpha) != 1.0
                else None
            )
            beta = getattr(w, "swiglu_beta", None)
            w.swiglu_beta_t = (
                _per_expert(beta) if beta is not None and float(beta) != 0.0 else None
            )
            w.swiglu_limit_t = (
                _per_expert(swiglu_arg.limit) if swiglu_arg.limit is not None else None
            )
        return None

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_cutlass_fp8_moe_apply",
        solution="flashinfer_cutlass",
        weight_preprocessor=flashinfer_cutlass_fp8_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(9, 0),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"fp8"}),
            # The kernel invokes ActivationType.Swiglu (gated SiLU); accept both
            # names so DeepSeek-V4 (activation="swiglu") selects this path.
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "fp8_scale_block_shape": frozenset({(128, 128)}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutlass_fp8_moe_apply(
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
        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(torch.float32)
        output = torch.empty(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
        swiglu_alpha = getattr(w, "swiglu_alpha_t", None)
        swiglu_beta = getattr(w, "swiglu_beta_t", None)
        swiglu_limit = getattr(w, "swiglu_limit_t", None)
        if swiglu_alpha is None and swiglu_beta is None and swiglu_limit is None:
            activation_type = ActivationType.Swiglu
        else:
            activation_type = ActivationType.SwigluBias
        return cutlass_fused_moe(
            output=output,
            input=x,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=w.w13_weight,
            fc2_expert_weights=w.w2_weight,
            output_dtype=x.dtype,
            input_sf=None,
            quant_scales=[w.w13_weight_scale_inv, w.w2_weight_scale_inv],
            ep_size=getattr(w, "ep_size", 1),
            ep_rank=getattr(w, "ep_rank", 0),
            tp_size=getattr(w, "tp_size", 1),
            tp_rank=getattr(w, "tp_rank", 0),
            tune_max_num_tokens=get_autotune_max_num_tokens(),
            activation_type=activation_type,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            use_deepseek_fp8_block_scale=True,
            enable_pdl=enable_pdl,
        )[0]


# ===-----------------------------------------------------------------------===#
# FP8 block-scaled experts: TRT-LLM (NVIDIA SM100+)
# ===-----------------------------------------------------------------------===#


# FP8 block-scale MoE via the TensorRT-LLM-Gen fused kernel.
#
# Checkpoint weights are converted once at load time to TRT-LLM's shuffled
# ``BlockMajorK`` layout.  Keeping that layout resident avoids the slower
# unshuffled MajorK load path on Blackwell without changing the block scales or
# the logical expert weights.

if platform.is_nvidia:
    from flashinfer import shuffle_matrix_a
    from flashinfer.fused_moe import (
        RoutingMethodType,
        convert_to_block_layout,
        trtllm_fp8_block_scale_moe,
        trtllm_fp8_block_scale_routed_moe,
    )
    from flashinfer.tllm_enums import WeightLayout
    from tokenspeed_kernel.ops.quantization import quantize_fp8

    _FP8_BLOCK = 128
    # DeepSeek block-FP8 kernels use a 64-row epilogue tile when weights are
    # shuffled.  This differs from the 128-row MXFP8/BF16 preparation path.
    _FP8_EPILOGUE_TILE_M = 64

    def _shuffle_block_major_(weight: torch.nn.Parameter) -> None:
        """Shuffle an expert weight in place, then expose BlockMajorK shape."""
        num_experts = weight.shape[0]
        old_expert_shape = weight.shape[1:]
        new_expert_shape = None
        for expert_idx in range(num_experts):
            shuffled = shuffle_matrix_a(
                weight.data[expert_idx].view(torch.uint8),
                _FP8_EPILOGUE_TILE_M,
            )
            blocked = convert_to_block_layout(shuffled, _FP8_BLOCK)
            new_expert_shape = blocked.shape
            # Reuse the checkpoint allocation while processing experts.  The
            # physical element count is unchanged; only the final view differs.
            weight.data[expert_idx] = (
                blocked.view(weight.dtype).contiguous().reshape(old_expert_shape)
            )
        if new_expert_shape is not None:
            weight.data = weight.data.reshape(num_experts, *new_expert_shape)

    def _routing_value(w: torch.nn.Module, name: str, default):
        routing_config = getattr(w, "routing_config", {})
        if not isinstance(routing_config, dict):
            routing_config = {}
        if name in routing_config:
            return routing_config[name]
        return getattr(w, name, default)

    def flashinfer_trtllm_fp8_moe_process_weights(plan: dict, w: torch.nn.Module):
        # The shared MoE checkpoint loader stores w13 as a concatenated
        # ``[w1(gate) | w3(up)]`` block; the TRT-LLM-Gen gated kernel consumes
        # ``[w3 | w1]`` ordering (same swap flashinfer_cutlass applies). Swap the
        # gate/up halves of both the weight and its block-scale in place.
        half_w = w.w13_weight.shape[1] // 2
        first_w = w.w13_weight.data[:, :half_w, :].clone()
        w.w13_weight.data[:, :half_w, :] = w.w13_weight.data[:, half_w:, :]
        w.w13_weight.data[:, half_w:, :] = first_w

        half_s = w.w13_weight_scale_inv.shape[1] // 2
        first_s = w.w13_weight_scale_inv.data[:, :half_s, :].clone()
        w.w13_weight_scale_inv.data[:, :half_s, :] = w.w13_weight_scale_inv.data[
            :, half_s:, :
        ]
        w.w13_weight_scale_inv.data[:, half_s:, :] = first_s

        w.w13_weight_scale_inv.data.clamp_(min=1e-10)
        w.w2_weight_scale_inv.data.clamp_(min=1e-10)

        # Preserve the logical size before BlockMajorK changes the final tensor
        # dimension from intermediate_size to block_k.
        w.intermediate_size_per_partition = w.w2_weight.shape[-1]
        _shuffle_block_major_(w.w13_weight)
        _shuffle_block_major_(w.w2_weight)

        # DeepSeek block-FP8 applies SwiGLU after GEMM1 dequantization, so
        # activation parameters stay in the model's actual-value domain.
        swiglu_arg = getattr(w, "swiglu_arg", None)
        if swiglu_arg is not None:
            num_experts = w.w13_weight.shape[0]
            device = w.w13_weight.device

            def _per_expert(value: float) -> torch.nn.Parameter:
                return torch.nn.Parameter(
                    torch.full(
                        (num_experts,), float(value), dtype=torch.float32, device=device
                    ),
                    requires_grad=False,
                )

            if swiglu_arg.alpha is not None and float(swiglu_arg.alpha) != 1.0:
                w.gemm1_alpha = _per_expert(swiglu_arg.alpha)
            beta = getattr(w, "swiglu_beta", None)
            if beta is not None and float(beta) != 0.0:
                w.gemm1_beta = _per_expert(beta)
            if swiglu_arg.limit is not None:
                w.gemm1_clamp_limit = _per_expert(swiglu_arg.limit)
        return None

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_fp8_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_fp8_moe_process_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"fp8"}),
            "activation": frozenset({"silu"}),
            "routing_mode": frozenset({"kernel_routing"}),
            "supports_deferred_finalize": frozenset({True}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "fp8_scale_block_shape": frozenset({(128, 128)}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_fp8_moe_apply(
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
        hidden_size = x.shape[1]
        if x.shape[0] == 0:
            empty_output = x.new_empty(0, hidden_size, dtype=torch.bfloat16)
            if do_finalize:
                return empty_output
            expert_weights = (
                topk_weights
                if topk_weights is not None
                else x.new_empty((0, getattr(w, "top_k")), dtype=torch.bfloat16)
            )
            return (
                empty_output,
                expert_weights,
                x.new_empty((0,), dtype=torch.int32),
            )

        # Per-token group (block=128) FP8 quantization of activations.
        # Convert canonical scales to the fused MoE kernel's group-major layout.
        x_fp8, x_scale = quantize_fp8(
            x,
            granularity="token_group",
            group_size=_FP8_BLOCK,
            solution="trtllm",
        )
        x_scale = x_scale.t().to(torch.float32).contiguous()
        hidden_blocks = hidden_size // _FP8_BLOCK
        if x_scale.shape != (hidden_blocks, x_fp8.shape[0]):
            raise RuntimeError(
                "unexpected hidden_states_scale shape "
                f"{tuple(x_scale.shape)}; expected "
                f"{(hidden_blocks, x_fp8.shape[0])}"
            )

        local_experts = getattr(w, "num_local_experts", w.w13_weight.shape[0])
        num_experts = getattr(w, "num_experts", local_experts)
        correction_bias = _routing_value(w, "correction_bias", None)
        routing_bias = (
            correction_bias.to(x.dtype)
            if isinstance(correction_bias, torch.Tensor)
            else None
        )
        n_group = _routing_value(w, "n_group", 0) or 1
        topk_group = _routing_value(w, "topk_group", 0) or 1
        routed_scaling_factor = _routing_value(w, "routed_scaling_factor", None)
        routing_method_type = _routing_value(w, "routing_method_type", 1)

        intermediate_size = w.intermediate_size_per_partition

        common_kwargs = dict(
            hidden_states=x_fp8,
            hidden_states_scale=x_scale,
            gemm1_weights=w.w13_weight,
            gemm1_weights_scale=w.w13_weight_scale_inv,
            gemm2_weights=w.w2_weight,
            gemm2_weights_scale=w.w2_weight_scale_inv,
            num_experts=num_experts,
            top_k=getattr(w, "top_k"),
            intermediate_size=intermediate_size,
            local_expert_offset=getattr(w, "ep_rank", 0) * local_experts,
            local_num_experts=local_experts,
            do_finalize=do_finalize,
            enable_pdl=enable_pdl,
            use_shuffled_weight=True,
            weight_layout=int(WeightLayout.BlockMajorK),
            tune_max_num_tokens=get_autotune_max_num_tokens(),
            gemm1_alpha=getattr(w, "gemm1_alpha", None),
            gemm1_beta=getattr(w, "gemm1_beta", None),
            gemm1_clamp_limit=getattr(w, "gemm1_clamp_limit", None),
        )
        if topk_weights is not None and topk_ids is not None:
            result = trtllm_fp8_block_scale_routed_moe(
                topk_ids=(
                    topk_ids.to(torch.int32).contiguous(),
                    topk_weights.contiguous(),
                ),
                routing_bias=None,
                n_group=None,
                topk_group=None,
                routed_scaling_factor=None,
                # Routing is already complete. Use the generic permutation
                # pipeline instead of DeepSeekV3's <=8 groups / <=32 experts
                # per group logits router, which cannot represent GLM's E=288,
                # n_group=1 topology.
                routing_method_type=int(RoutingMethodType.Renormalize),
                **common_kwargs,
            )
        else:
            result = trtllm_fp8_block_scale_moe(
                routing_logits=router_logits.to(torch.float32),
                routing_bias=routing_bias,
                n_group=n_group,
                topk_group=topk_group,
                routed_scaling_factor=routed_scaling_factor,
                routing_method_type=int(routing_method_type),
                **common_kwargs,
            )
        if do_finalize:
            if isinstance(result, (list, tuple)):
                result = result[0]
            return result

        if not isinstance(result, (list, tuple)) or len(result) < 3:
            raise RuntimeError(
                "FlashInfer deferred FP8 MoE must return gemm2 output, "
                "expert weights, and the expanded-to-permuted index"
            )
        gemm2_out, expert_weights, expanded_idx = result[:3]
        if topk_weights is None and expert_weights.dtype == torch.float32:
            # In-kernel DeepSeek routing allocates this carrier with the fp32
            # logits dtype, while the TRT-LLM launcher writes bf16 route
            # weights into it. Recover the live bf16 prefix before handing it
            # to the external fused finalize kernel. The pre-routed path
            # borrows the caller's real tensor and does not need this fixup.
            num_tokens, top_k = expert_weights.shape
            expert_weights = expert_weights.view(torch.bfloat16).view(-1, top_k)[
                :num_tokens
            ]
        return gemm2_out, expert_weights, expanded_idx

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_fp8_routed_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_fp8_moe_process_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"fp8"}),
            "activation": frozenset({"swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({True}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "fp8_scale_block_shape": frozenset({(128, 128)}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT + 3,
    )
    def flashinfer_trtllm_fp8_routed_moe_apply(
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
        if topk_weights is None or topk_ids is None:
            raise RuntimeError(
                "precomputed_topk plan requires topk_weights and topk_ids"
            )
        return flashinfer_trtllm_fp8_moe_apply(
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


# ===-----------------------------------------------------------------------===#
# FP8 per-channel experts: TRT-LLM (NVIDIA SM100-SM103)
# ===-----------------------------------------------------------------------===#


# FP8 per-channel MoE via FlashInfer's TensorRT-LLM-Gen fused kernel.
#
# The ``fp8_channel`` experts (compressed-tensors FP8 dynamic W8A8: FP8 E4M3
# weights with one FP32 scale per output row, activations quantized per token at
# run time) run on ``trtllm_fp8_per_channel_scale_moe``. Its rounding points:
#
# - the input is quantized per token, one FP32 scale per token, by the kernel
#   that ``quantize_fp8_with_scale(granularity="token")`` selects;
# - FC1 accumulates in FP32, applies the token and row scales, runs SiLU(gate) *
#   up in its epilogue and writes BF16;
# - the FC2 input is quantized per routed row, over this rank's intermediate
#   partition, to FP8 with one FP32 scale by TRT-LLM's perTokenQuantization
#   (fp8RowWise): ``amax`` is the row's largest magnitude in BF16 (at least
#   1e-6), the scale ``max(amax / 448, 1 / (448 * 512))`` and the codes
#   ``sat(E4M3(BF16(x * min(448 / amax, 448 * 512))))``: the product is FP32
#   and is rounded to BF16 before E4M3 (``cuda_cast<__nv_fp8x2_e4m3, float2>``
#   goes through ``__nv_bfloat162``); the quotients are FlashInfer's
#   ``-use_fast_math`` divisions, within an ulp or two of IEEE;
# - FC2 accumulates in FP32, applies the row and channel scales and writes BF16
#   per route;
# - the combine multiplies each route output by its route weight and sums in
#   FP32: the kernel's own finalize for BF16 route weights or kernel routing,
#   ``moe_finalize_fuse_shared`` for FP32 route weights (the kernel takes route
#   weights only as BF16).
#
# The weight layout is prepared once at load (:func:`prepare_fp8_channel_weights`):
# each half of w13 is padded to a multiple of
# :data:`TRTLLM_FP8_CHANNEL_ISPP_ALIGNMENT` rows (zero weights, scale 1) on its
# own rank, so the channels a rank owns do not move, and w2's columns likewise
# (partitions that are already a multiple need none);
# gate and up are swapped to the kernel's ``[up | gate]`` order; the rows are
# interleaved for the gated activation and shuffled for the epilogue tile. Every
# row permutation moves the row's scale with it.

# The preprocessor pads each rank's intermediate partition to this multiple.
# On GB200 the kernel takes a multiple of 64 that is not a multiple of 128
# unpadded, bit-identical to the same experts padded to the next multiple of
# 128 (1, 64 and 2,048 tokens), so such partitions run as they are; other
# sizes pad.
TRTLLM_FP8_CHANNEL_ISPP_ALIGNMENT = 64
# The epilogue tile of the per-channel FP8 cubins (FlashInfer 0.7.1rc2).
_FP8_CHANNEL_EPILOGUE_TILE_M = 128


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def pad_fp8_channel_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad the intermediate size per partition and swap gate and up.

    Args:
        w13: FP8 ``[E, 2 * I, H]``, gate rows then up rows.
        w13_scale: FP32 ``[E, 2 * I]``, the scale of each w13 row.
        w2: FP8 ``[E, H, I]``.
        w2_scale: FP32 ``[E, H]``, the scale of each w2 row.
        alignment: The padded intermediate size is a multiple of it.

    Returns:
        ``(w13 [E, 2 * Ip, H] up rows then gate rows, w13_scale [E, 2 * Ip],
        w2 [E, H, Ip], w2_scale)``. Padded rows and columns are zero and the
        padded rows' scales 1: they add exactly zero.
    """
    experts, rows, hidden = w13.shape
    ispp = rows // 2
    padded = _round_up(ispp, alignment)
    out13 = w13.new_zeros(experts, 2 * padded, hidden)
    out13_scale = w13_scale.new_ones(experts, 2 * padded)
    # Up rows first, then gate rows: the kernel's [w3 | w1] order.
    out13[:, :ispp] = w13[:, ispp:]
    out13[:, padded : padded + ispp] = w13[:, :ispp]
    out13_scale[:, :ispp] = w13_scale[:, ispp:]
    out13_scale[:, padded : padded + ispp] = w13_scale[:, :ispp]
    out2 = w2.new_zeros(experts, w2.shape[1], padded)
    out2[:, :, :ispp] = w2
    return out13, out13_scale, out2, w2_scale


def shuffle_fp8_channel_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_rows: torch.Tensor,
    w2_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder the rows of every expert, each row's scale with it.

    ``w13_rows`` / ``w2_rows`` give, for each destination row, its source row
    (one permutation for all experts: it depends only on the shape).
    """
    return (
        w13.index_select(1, w13_rows),
        w13_scale.index_select(1, w13_rows),
        w2.index_select(1, w2_rows),
        w2_scale.index_select(1, w2_rows),
    )


if platform.is_nvidia:
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices as maybe_get_cached_w3_w1_permute_indices,
    )
    from flashinfer.fused_moe.core import (
        get_w2_permute_indices_with_cache,
        trtllm_fp8_per_channel_scale_moe,
        trtllm_fp8_per_channel_scale_routed_moe,
    )
    from tokenspeed_kernel.ops.moe.cuda import moe_finalize_fuse_shared
    from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale

    def fp8_channel_row_permutations(
        w13: torch.Tensor, w2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FlashInfer's row permutations of one expert's padded ``[up | gate]``
        w13 ``[2 * Ip, H]`` (gated-activation interleave, then the epilogue
        shuffle) and w2 ``[H, Ip]`` (the epilogue shuffle)."""
        cache: dict = {}
        w13_rows = maybe_get_cached_w3_w1_permute_indices(
            cache,
            w13.view(torch.uint8),
            _FP8_CHANNEL_EPILOGUE_TILE_M,
            is_gated_act_gemm=True,
        )
        w2_rows = get_w2_permute_indices_with_cache(
            cache, w2.view(torch.uint8), _FP8_CHANNEL_EPILOGUE_TILE_M
        )
        return w13_rows.to(w13.device), w2_rows.to(w2.device)

    def prepare_fp8_channel_weights(
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The kernel's resident layout of the ``fp8_channel`` weights: padded,
        ``[up | gate]``, interleaved and shuffled, scales permuted alike."""
        padded = pad_fp8_channel_weights(
            w13, w13_scale, w2, w2_scale, TRTLLM_FP8_CHANNEL_ISPP_ALIGNMENT
        )
        rows13, rows2 = fp8_channel_row_permutations(padded[0][0], padded[2][0])
        return shuffle_fp8_channel_weights(*padded, rows13, rows2)

    def flashinfer_trtllm_fp8_channel_moe_process_weights(
        plan: dict, w: torch.nn.Module
    ):
        w13, w13_scale, w2, w2_scale = prepare_fp8_channel_weights(
            w.w13_weight.data,
            w.w13_weight_scale.data,
            w.w2_weight.data,
            w.w2_weight_scale.data,
        )
        w.w13_weight.data = w13.contiguous()
        w.w13_weight_scale.data = w13_scale.contiguous()
        w.w2_weight.data = w2.contiguous()
        w.w2_weight_scale.data = w2_scale.contiguous()
        w.intermediate_size_per_partition = w2.shape[-1]
        # The per-expert output scalars stay 1: the scales above carry the
        # whole dequantization.
        w.fp8_channel_output_scales = torch.nn.Parameter(
            torch.ones(w13.shape[0], dtype=torch.float32, device=w13.device),
            requires_grad=False,
        )
        return None

    def _pack_routes(topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        """``(expert_id << 16) | bf16 bits of the route weight`` per route."""
        bits = topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        return (topk_ids.to(torch.int32) << 16) | (bits & 0xFFFF)

    def _flashinfer_trtllm_fp8_channel_moe_apply(
        x: torch.Tensor,
        w: torch.nn.Module,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor | None,
        topk_ids: torch.Tensor | None,
        do_finalize: bool,
        enable_pdl: bool,
        routed: bool,
    ):
        spec = w._spec
        if x.shape[0] == 0:
            output = x.new_empty((0, x.shape[1]), dtype=torch.bfloat16)
            if do_finalize:
                return output
            # The caller's route weights keep their dtype, as on non-empty
            # batches of the routed path.
            expert_weights = (
                topk_weights
                if topk_weights is not None
                else x.new_empty((0, spec.top_k), dtype=torch.bfloat16)
            )
            return output, expert_weights, x.new_empty((0,), dtype=torch.int32)
        x_fp8, x_scale = quantize_fp8_with_scale(x.contiguous(), granularity="token")
        scalars = w.fp8_channel_output_scales.data
        common = dict(
            hidden_states=x_fp8,
            hidden_states_scale=x_scale.reshape(-1, 1).contiguous(),
            gemm1_weights=w.w13_weight.data,
            gemm1_per_channel_weight_scale=w.w13_weight_scale.data,
            output1_scale_scalar=scalars,
            output1_scale_gate_scalar=scalars,
            gemm2_weights=w.w2_weight.data,
            gemm2_per_channel_weight_scale=w.w2_weight_scale.data,
            output2_scale_scalar=scalars,
            num_experts=spec.num_experts,
            top_k=spec.top_k,
            intermediate_size=w.intermediate_size_per_partition,
            local_expert_offset=spec.ep_rank * spec.num_local_experts,
            local_num_experts=spec.num_local_experts,
            enable_pdl=enable_pdl,
            tune_max_num_tokens=get_autotune_max_num_tokens(),
        )
        if not routed:
            result = trtllm_fp8_per_channel_scale_moe(
                routing_logits=router_logits.to(w._routing_logits_dtype),
                routing_bias=w._correction_bias,
                n_group=w._n_group or None,
                topk_group=w._topk_group or None,
                routed_scaling_factor=w._routed_scaling_factor,
                use_routing_scales_on_input=False,
                routing_method_type=int(w._routing_method_type),
                do_finalize=do_finalize,
                **common,
            )
            if do_finalize:
                return result
            gemm2_out, expert_weights, expanded_idx = result[:3]
            if expert_weights.dtype == torch.float32:
                # FP32 routing logits allocate an FP32 carrier the kernel
                # fills with BF16 weights; keep its live BF16 prefix.
                tokens, top_k = expert_weights.shape
                expert_weights = expert_weights.view(torch.bfloat16).view(-1, top_k)[
                    :tokens
                ]
            return gemm2_out, expert_weights, expanded_idx
        # The kernel takes packed routes with BF16 weights; FP32 route weights
        # are applied by the FP32-accumulating finalize below instead.
        fp32_routes = topk_weights.dtype == torch.float32
        result = trtllm_fp8_per_channel_scale_routed_moe(
            topk_ids=_pack_routes(topk_ids, topk_weights),
            routing_bias=None,
            n_group=None,
            topk_group=None,
            routed_scaling_factor=None,
            do_finalize=do_finalize and not fp32_routes,
            **common,
        )
        if do_finalize and not fp32_routes:
            return result
        gemm2_out, _, expanded_idx = result[:3]
        if not do_finalize:
            return gemm2_out, topk_weights, expanded_idx
        return moe_finalize_fuse_shared(
            gemm2_out,
            expanded_idx,
            topk_weights.contiguous(),
            None,
            spec.top_k,
            enable_pdl=enable_pdl,
            hidden_dim=gemm2_out.shape[1],
        )

    _FP8_CHANNEL_TRAITS = {
        "weight_dtype": frozenset({"fp8_channel"}),
        "activation": frozenset({"silu", "swiglu"}),
        "swiglu_form": frozenset({"standard"}),
        "activation_clamped": frozenset({False}),
        "supports_deferred_finalize": frozenset({True}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        # Selection admits partitions that are a multiple of 32; the
        # preprocessor then pads each to TRTLLM_FP8_CHANNEL_ISPP_ALIGNMENT (64).
        "ispp_alignment": frozenset({32}),
        "hidden_alignment": frozenset({128}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    }
    _FP8_CHANNEL_CAPABILITY = CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
    )

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_fp8_channel_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_fp8_channel_moe_process_weights,
        capability=_FP8_CHANNEL_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={**_FP8_CHANNEL_TRAITS, "routing_mode": frozenset({"kernel_routing"})},
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_fp8_channel_moe_apply(
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
        """FP8 per-channel experts routed in the kernel from ``router_logits``
        (the layer's routing method, correction bias and routed scale); BF16
        route weights. Explicitly supplied ``topk_weights`` and ``topk_ids``
        run on those routes instead, as the routed variant does. Rounding
        points as in the section comment above."""
        if (topk_weights is None) != (topk_ids is None):
            raise ValueError("pass both topk_weights and topk_ids, or neither")
        return _flashinfer_trtllm_fp8_channel_moe_apply(
            x,
            w,
            router_logits,
            topk_weights,
            topk_ids,
            do_finalize,
            enable_pdl,
            routed=topk_ids is not None,
        )

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_fp8_channel_routed_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_fp8_channel_moe_process_weights,
        capability=_FP8_CHANNEL_CAPABILITY,
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={**_FP8_CHANNEL_TRAITS, "routing_mode": frozenset({"precomputed_topk"})},
        # One below in-kernel routing: this wins only for plans with
        # routing_mode="precomputed_topk".
        priority=Priority.PERFORMANT + 3,
    )
    def flashinfer_trtllm_fp8_channel_routed_moe_apply(
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
        """FP8 per-channel experts on precomputed routes.

        Args:
            plan: MoE plan for standard SiLU/SwiGLU; fold any routed scale
                into ``topk_weights``.
            x: BF16 hidden states ``[tokens, hidden]``.
            w: Module with the layout :func:`prepare_fp8_channel_weights` made.
            router_logits: Unused; routing is precomputed.
            topk_weights: Route weights ``[tokens, top_k]``, FP32 or BF16; each
                is applied at its dtype and the routes summed in FP32.
            topk_ids: Global expert ids ``[tokens, top_k]``; routes to other
                ranks' experts contribute zero.
            num_tokens_global: Unused; there is no all-to-all.
            max_num_tokens_per_gpu: Unused token-capacity hint.
            do_finalize: False returns ``(per-route BF16 outputs, the caller's
                route weights, expanded-to-permuted index)``.
            enable_pdl: Programmatic dependent launch.

        Returns:
            Finalized BF16 hidden states ``[tokens, hidden]``.
        """
        if topk_weights is None or topk_ids is None:
            raise ValueError("precomputed_topk plan requires topk_weights and topk_ids")
        return _flashinfer_trtllm_fp8_channel_moe_apply(
            x,
            w,
            router_logits,
            topk_weights,
            topk_ids,
            do_finalize,
            enable_pdl,
            routed=True,
        )


# ===-----------------------------------------------------------------------===#
# MXFP4 experts: CUTLASS mixed input (NVIDIA SM90)
# ===-----------------------------------------------------------------------===#


# FlashInfer CUTLASS mixed-input MoE for MXFP4 experts on Hopper (SM90).
#
# Two registrations of the same grouped GEMM: W4A16 keeps BF16 activations and
# dequantizes the E2M1 weights in the mainloop; W4A8 ("Humming") also quantizes
# the activations to FP8 and folds the FP4->FP8 exponent bias into per-expert
# residual scales. Both read the loader's ``[gate; up]`` MXFP4 rows and rewrite
# them once into FlashInfer's SM90 interleaved layout. The E8M0 group-32 scales
# travel as int32 views, which is why hidden and intermediate sizes must be
# multiples of 128.
#
# Marlin remains the Hopper solution for SiTU experts (Kimi-K3) and for DeepEP
# all-to-all layouts; this one serves SiLU/SwiGLU experts with dense EP.
#
# Routing carries global expert ids and the kernel drops the ids outside its EP
# shard. A token's ids must be distinct: FlashInfer's permutation assumes one
# row per (token, expert) and returns wrong sums when an expert repeats, so a
# caller must not mask unwanted slots by pointing them at a placeholder expert
# with zero weight. Top-k routing never repeats an expert.

MXFP4_GROUP = 32
# quant_scales are E8M0 bytes viewed as int32 along K, so K/32 must be a
# multiple of 4 for both GEMMs: hidden (fc1) and intermediate (fc2).
_ALIGNMENT = 4 * MXFP4_GROUP
_W4A16 = "w4a16"
_W4A8 = "w4a8"


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


# ===-----------------------------------------------------------------------===#
# MXFP4 experts: TRT-LLM (NVIDIA SM100)
# ===-----------------------------------------------------------------------===#


_permute_indices_cache: dict[tuple[str, torch.Size], torch.Tensor] = {}
_permute_indices_device_cache: dict[
    tuple[str, tuple[int, ...], int, int, str, int], torch.Tensor
] = {}


def _float8_view_dtype(x: torch.Tensor) -> torch.dtype | None:
    float8_dtypes = tuple(
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e8m0fnu", None),
        )
        if dtype is not None
    )
    return x.dtype if x.dtype in float8_dtypes else None


def _pair_swap_rows(x: torch.Tensor, dim: int = -2) -> torch.Tensor:
    view_dtype = _float8_view_dtype(x)
    if view_dtype is not None:
        x = x.view(torch.uint8)
    if dim < 0:
        dim += x.dim()
    shape = list(x.shape)
    if shape[dim] % 2 != 0:
        raise ValueError(f"expected even size in dim {dim}, got {shape[dim]}")
    new_shape = shape[:dim] + [shape[dim] // 2, 2] + shape[dim + 1 :]
    out = x.reshape(new_shape).flip(dim + 1).reshape(shape).contiguous()
    return out.view(view_dtype) if view_dtype is not None else out


def _reorder_w1w3_to_w3w1(x: torch.Tensor, dim: int = -2) -> torch.Tensor:
    view_dtype = _float8_view_dtype(x)
    if view_dtype is not None:
        x = x.view(torch.uint8)
    if dim < 0:
        dim += x.dim()
    size = x.shape[dim]
    if size % 2 != 0:
        raise ValueError(f"expected even size in dim {dim}, got {size}")
    first, second = x.split(size // 2, dim=dim)
    out = torch.cat([second, first], dim=dim).contiguous()
    return out.view(view_dtype) if view_dtype is not None else out


if platform.is_nvidia:
    from flashinfer import (
        mxfp8_quantize,
        nvfp4_block_scale_interleave,
        trtllm_fp4_block_scale_moe,
    )
    from flashinfer.cute_dsl import is_cute_dsl_available
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices as maybe_get_cached_w3_w1_permute_indices,
    )
    from flashinfer.fused_moe.core import (
        get_w2_permute_indices_with_cache,
    )
    from flashinfer.tllm_enums import ActivationType

    def _get_device_permute_indices(
        x: torch.Tensor,
        epilogue_tile_m: int,
        num_elts_per_sf: int | None = None,
        *,
        kind: str = "w2",
    ) -> torch.Tensor:
        extra_args = (
            {} if num_elts_per_sf is None else {"num_elts_per_sf": num_elts_per_sf}
        )
        if kind == "w13":
            permute_indices = maybe_get_cached_w3_w1_permute_indices(
                _permute_indices_cache,
                x,
                epilogue_tile_m,
                **extra_args,
            )
        elif kind == "w2":
            permute_indices = get_w2_permute_indices_with_cache(
                _permute_indices_cache,
                x,
                epilogue_tile_m,
                **extra_args,
            )
        else:
            raise ValueError(f"unknown FlashInfer MXFP4 permute kind: {kind}")

        device_index = -1 if x.device.index is None else x.device.index
        num_elts_per_sf_key = -1 if num_elts_per_sf is None else num_elts_per_sf
        cache_key = (
            kind,
            tuple(x.shape),
            epilogue_tile_m,
            num_elts_per_sf_key,
            x.device.type,
            device_index,
        )
        cached_device_indices = _permute_indices_device_cache.get(cache_key)
        if cached_device_indices is None:
            cached_device_indices = permute_indices.to(x.device)
            _permute_indices_device_cache[cache_key] = cached_device_indices
        return cached_device_indices

    def _param_like_weight(
        w: torch.nn.Module,
        value: float | None,
    ) -> torch.nn.Parameter | None:
        if value is None:
            return None
        return torch.nn.Parameter(
            torch.full(
                (w.w13_weight.shape[0],),
                float(value),
                dtype=torch.float32,
                device=w.w13_weight.device,
            ),
            requires_grad=False,
        )

    def _positive_situ_value(w: torch.nn.Module, *names: str) -> float:
        routing_config = getattr(w, "routing_config", {})
        if not isinstance(routing_config, dict):
            routing_config = {}

        value = None
        for name in names:
            value = routing_config.get(name)
            if value is not None:
                break
        if value is None:
            for name in names:
                value = getattr(w, name, None)
                if value is not None:
                    break
        if value is None:
            joined_names = ", ".join(names)
            raise ValueError(f"SiTU requires one of: {joined_names}")

        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"SiTU parameter {names[0]} must be finite and positive, got {value}"
            )
        return value

    def _flashinfer_trtllm_mxfp4_moe_weights(
        plan: dict,
        w: torch.nn.Module,
        *,
        situ: bool,
    ):
        sf_block_size = 32
        num_experts = w.w13_weight.shape[0]
        ispp_padded = w.w13_weight.shape[1] // 2
        hidden_padded = w.w2_weight.shape[1]

        if situ:
            alpha = _positive_situ_value(w, "activation_situ_beta", "situ_beta")
            beta = _positive_situ_value(
                w, "activation_situ_linear_beta", "situ_linear_beta"
            )
            limit = None
        elif (swiglu_arg := getattr(w, "swiglu_arg", None)) is None:
            alpha = 1.702
            limit = 7.0
            beta = 1.0
        else:
            alpha = swiglu_arg.alpha
            limit = swiglu_arg.limit
            beta = getattr(w, "swiglu_beta", None)
        w.gemm1_alpha = _param_like_weight(w, alpha)
        w.gemm1_beta = _param_like_weight(w, beta)
        w.gemm1_clamp_limit = _param_like_weight(w, limit)

        w13_weight_scale = w.w13_weight_scale.data
        w2_weight_scale = w.w2_weight_scale.data
        w13_weight = w.w13_weight.data
        w2_weight = w.w2_weight.data
        has_bias = hasattr(w, "w13_weight_bias") and hasattr(w, "w2_weight_bias")
        w13_bias = w.w13_weight_bias.data.to(torch.float32) if has_bias else None
        w2_bias = w.w2_weight_bias.data.to(torch.float32) if has_bias else None

        w13_layout = getattr(w, "w13_input_layout", "concatenated")
        if w13_layout == "interleaved":
            w13_weight_scale = _pair_swap_rows(w13_weight_scale, -2)
            w13_weight = _pair_swap_rows(w13_weight, -2)
            if w13_bias is not None:
                w13_bias = _pair_swap_rows(w13_bias, -1)
            w13_permute_kind = "w2"
        elif w13_layout == "concatenated":
            w13_weight_scale = _reorder_w1w3_to_w3w1(w13_weight_scale, -2)
            w13_weight = _reorder_w1w3_to_w3w1(w13_weight, -2)
            if w13_bias is not None:
                w13_bias = _reorder_w1w3_to_w3w1(w13_bias, -1)
            w13_permute_kind = "w13"
        else:
            raise ValueError(f"unknown w13_input_layout: {w13_layout!r}")

        epilogue_tile_m = 128
        w13_weight_perm = _get_device_permute_indices(
            w13_weight[0].view(torch.uint8), epilogue_tile_m, kind=w13_permute_kind
        )
        w13_scale_perm = _get_device_permute_indices(
            w13_weight_scale[0].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            kind=w13_permute_kind,
        )
        w2_weight_perm = _get_device_permute_indices(
            w2_weight[0].view(torch.uint8), epilogue_tile_m, kind="w2"
        )
        w2_scale_perm = _get_device_permute_indices(
            w2_weight_scale[0].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            kind="w2",
        )
        if has_bias:
            w13_bias_perm = _get_device_permute_indices(
                w13_bias[0].reshape(-1, 1), epilogue_tile_m
            )
            w2_bias_perm = _get_device_permute_indices(
                w2_bias[0].reshape(-1, 1), epilogue_tile_m
            )

        gemm1_weights_shuffled = []
        gemm1_scales_shuffled = []
        gemm2_weights_shuffled = []
        gemm2_scales_shuffled = []
        gemm1_bias_shuffled = []
        gemm2_bias_shuffled = []
        for idx in range(num_experts):
            gemm1_weights_shuffled.append(
                w13_weight[idx].view(torch.uint8)[w13_weight_perm].contiguous()
            )
            gemm1_scales_shuffled.append(
                nvfp4_block_scale_interleave(
                    w13_weight_scale[idx].view(torch.uint8)[w13_scale_perm].contiguous()
                )
            )
            gemm2_weights_shuffled.append(
                w2_weight[idx].view(torch.uint8)[w2_weight_perm].contiguous()
            )
            gemm2_scales_shuffled.append(
                nvfp4_block_scale_interleave(
                    w2_weight_scale[idx].view(torch.uint8)[w2_scale_perm].contiguous()
                )
            )
            if has_bias:
                gemm1_bias_shuffled.append(
                    w13_bias[idx].reshape(-1, 1)[w13_bias_perm].contiguous()
                )
                gemm2_bias_shuffled.append(
                    w2_bias[idx].reshape(-1, 1)[w2_bias_perm].contiguous()
                )

        w.w13_weight = torch.nn.Parameter(
            torch.stack(gemm1_weights_shuffled), requires_grad=False
        )
        w.w13_weight_scale = torch.nn.Parameter(
            torch.stack(gemm1_scales_shuffled)
            .reshape(num_experts, 2 * ispp_padded, hidden_padded // sf_block_size)
            .view(torch.float8_e4m3fn),
            requires_grad=False,
        )
        w.w2_weight = torch.nn.Parameter(
            torch.stack(gemm2_weights_shuffled), requires_grad=False
        )
        w.w2_weight_scale = torch.nn.Parameter(
            torch.stack(gemm2_scales_shuffled)
            .reshape(num_experts, hidden_padded, ispp_padded // sf_block_size)
            .view(torch.float8_e4m3fn),
            requires_grad=False,
        )
        if has_bias:
            w.w13_weight_bias = torch.nn.Parameter(
                torch.stack(gemm1_bias_shuffled).reshape(num_experts, -1),
                requires_grad=False,
            )
            w.w2_weight_bias = torch.nn.Parameter(
                torch.stack(gemm2_bias_shuffled).reshape(num_experts, -1),
                requires_grad=False,
            )
        w.intermediate_size_per_partition = ispp_padded
        w.hidden_size_padded = hidden_padded
        w.hidden_size_original = getattr(w, "hidden_size", hidden_padded)
        w._flashinfer_trtllm_autotuned = False
        return None

    def flashinfer_trtllm_mxfp4_moe_weights(plan: dict, w: torch.nn.Module):
        return _flashinfer_trtllm_mxfp4_moe_weights(plan, w, situ=False)

    def flashinfer_trtllm_mxfp4_situ_moe_weights(
        plan: dict,
        w: torch.nn.Module,
    ):
        # The public TRTLLM-Gen SiTU entry point uses the same standard
        # TRT-LLM [up|gate] physical layout as SwiGLU; keep the shared
        # reorder/shuffle above.
        return _flashinfer_trtllm_mxfp4_moe_weights(plan, w, situ=True)

    def _local_expert_range(w: torch.nn.Module) -> tuple[int, int, int]:
        local_experts = getattr(w, "num_local_experts", w.w13_weight.shape[0])
        if local_experts != w.w13_weight.shape[0]:
            raise RuntimeError(
                f"expected {local_experts} local experts, "
                f"got {w.w13_weight.shape[0]} weight batches"
            )
        local_expert_offset = getattr(w, "ep_rank", 0) * local_experts
        num_experts = getattr(w, "num_experts")
        if local_expert_offset < 0 or local_expert_offset + local_experts > num_experts:
            raise RuntimeError(
                f"invalid local expert range [{local_expert_offset}, "
                f"{local_expert_offset + local_experts}) for {num_experts} experts"
            )
        return num_experts, local_experts, local_expert_offset

    def _call_mxfp4_moe(
        w: torch.nn.Module,
        router_logits: torch.Tensor,
        x_quant: torch.Tensor,
        x_scale: torch.Tensor | None,
        output: torch.Tensor | None,
        enable_pdl: bool,
        do_finalize: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        routing_logits = router_logits.to(torch.float32)
        num_experts, local_experts, local_expert_offset = _local_expert_range(w)
        result = trtllm_fp4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=None,
            hidden_states=x_quant,
            hidden_states_scale=(
                None if x_scale is None else x_scale.view(torch.float8_e4m3fn)
            ),
            gemm1_weights=w.w13_weight,
            gemm1_weights_scale=w.w13_weight_scale.view(torch.float8_e4m3fn),
            gemm1_bias=getattr(w, "w13_weight_bias", None),
            gemm1_alpha=getattr(w, "gemm1_alpha", None),
            gemm1_beta=getattr(w, "gemm1_beta", None),
            gemm1_clamp_limit=getattr(w, "gemm1_clamp_limit", None),
            gemm2_weights=w.w2_weight,
            gemm2_weights_scale=w.w2_weight_scale.view(torch.float8_e4m3fn),
            gemm2_bias=getattr(w, "w2_weight_bias", None),
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=num_experts,
            top_k=getattr(w, "top_k"),
            n_group=None,
            topk_group=None,
            intermediate_size=getattr(w, "intermediate_size_per_partition"),
            local_expert_offset=local_expert_offset,
            local_num_experts=local_experts,
            routed_scaling_factor=None,
            routing_method_type=1,
            do_finalize=do_finalize,
            tune_max_num_tokens=get_autotune_max_num_tokens(),
            output=output,
            enable_pdl=enable_pdl,
        )
        return (
            result[0]
            if do_finalize
            else _normalize_kernel_routing_deferred_result(result)
        )

    def _call_mxfp4_routed_moe(
        w: torch.nn.Module,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        x: torch.Tensor,
        output: torch.Tensor | None,
        enable_pdl: bool,
        hidden_states_scale: torch.Tensor | None,
        do_finalize: bool,
        activation_type: int,
    ):
        num_experts, local_experts, local_expert_offset = _local_expert_range(w)
        # FlashInfer's precomputed route is an unpacked (ids, weights) tuple.
        # IDs remain global; the kernel filters them to this rank's expert range.
        topk = (
            topk_ids.to(torch.int32).contiguous(),
            topk_weights.to(torch.bfloat16).contiguous(),
        )
        result = trtllm_fp4_block_scale_routed_moe(
            topk_ids=topk,
            routing_bias=None,
            hidden_states=x,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=w.w13_weight,
            gemm1_weights_scale=w.w13_weight_scale.view(torch.float8_e4m3fn),
            gemm1_bias=getattr(w, "w13_weight_bias", None),
            gemm1_alpha=getattr(w, "gemm1_alpha", None),
            gemm1_beta=getattr(w, "gemm1_beta", None),
            gemm1_clamp_limit=getattr(w, "gemm1_clamp_limit", None),
            gemm2_weights=w.w2_weight,
            gemm2_weights_scale=w.w2_weight_scale.view(torch.float8_e4m3fn),
            gemm2_bias=getattr(w, "w2_weight_bias", None),
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=num_experts,
            top_k=getattr(w, "top_k"),
            n_group=None,
            topk_group=None,
            intermediate_size=getattr(w, "intermediate_size_per_partition"),
            local_expert_offset=local_expert_offset,
            local_num_experts=local_experts,
            routed_scaling_factor=None,
            routing_method_type=1,
            do_finalize=do_finalize,
            enable_pdl=enable_pdl,
            activation_type=activation_type,
            tune_max_num_tokens=get_autotune_max_num_tokens(),
            output=output if do_finalize else None,
        )
        return output if do_finalize else result

    # FP16 shared-expert output cannot use the BF16-only common finalizer.
    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_mxfp4_fp16_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_mxfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=format_signatures("x", "dense", {torch.float16}),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"kernel_routing", "precomputed_topk"}),
            "topk_weights_dtype": frozenset({torch.bfloat16}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({True}),
        },
        priority=Priority.SPECIALIZED,
    )
    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_mxfp4_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_mxfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"kernel_routing", "precomputed_topk"}),
            "topk_weights_dtype": frozenset({torch.bfloat16}),
            "supports_deferred_finalize": frozenset({True, False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({True}),
        },
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_mxfp4_moe_apply(
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
        hidden_padded = getattr(w, "hidden_size_padded", w.w2_weight_scale.shape[1])
        hidden_original = getattr(w, "hidden_size_original", hidden_padded)
        if x.shape[0] == 0:
            if do_finalize:
                return x.new_empty(0, hidden_original)
            return (
                x.new_empty((0, hidden_padded), dtype=torch.bfloat16),
                x.new_empty((0, w.top_k), dtype=torch.bfloat16),
                x.new_empty((0,), dtype=torch.int32),
            )

        precision = plan.get(
            "flashinfer_trtllm_moe_precision",
            getattr(w, "flashinfer_trtllm_moe_precision", "default"),
        )
        if precision == "bf16":
            if x.dtype != torch.bfloat16:
                raise TypeError("FlashInfer MXFP4 bf16 precision requires bf16 input")
            x_quant = x
            x_scale = None
            if hidden_padded != x_quant.shape[-1]:
                x_quant = torch.nn.functional.pad(
                    x_quant,
                    (0, hidden_padded - x_quant.shape[-1]),
                    mode="constant",
                    value=0.0,
                )
        elif precision == "default":
            # Use CuTe across batch sizes. Its padding path adds a zero-fill
            # and copy, so keep native padding.
            quant_backend = (
                "cute-dsl"
                if x.shape[-1] == hidden_padded
                and x.is_contiguous()
                and is_cute_dsl_available()
                else "cuda"
            )
            x_quant, x_scale = mxfp8_quantize(
                x,
                False,
                enable_pdl=enable_pdl,
                alignment=hidden_padded,
                backend=quant_backend,
            )
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(*x.shape[:-1], -1)
        else:
            raise NotImplementedError(
                f"Unknown flashinfer_trtllm_moe_precision: {precision}"
            )

        if x_quant.shape[-1] != hidden_padded:
            raise RuntimeError(
                f"expected hidden size {hidden_padded}, got {x_quant.shape[-1]}"
            )

        h_dim = (
            x_quant.shape[-1] * 2 if x_quant.dtype == torch.uint8 else x_quant.shape[-1]
        )
        output = (
            torch.empty(
                x_quant.shape[0], h_dim, dtype=torch.bfloat16, device=x_quant.device
            )
            if do_finalize
            else None
        )

        if (topk_weights is None) != (topk_ids is None):
            raise ValueError("topk_weights and topk_ids must be supplied together")
        if topk_ids is not None:
            result = _call_mxfp4_routed_moe(
                w,
                topk_weights,
                topk_ids,
                x_quant,
                output,
                enable_pdl,
                x_scale,
                do_finalize,
                ActivationType.Swiglu,
            )
        else:
            result = _call_mxfp4_moe(
                w, router_logits, x_quant, x_scale, output, enable_pdl, do_finalize
            )
        if not do_finalize:
            return result
        if hidden_original != hidden_padded:
            result = result[:, :hidden_original].contiguous()
        return result

    def _normalize_kernel_routing_deferred_result(result):
        gemm2_output, expert_weights, expanded_idx = result
        if expert_weights.dtype == torch.float32:
            # FlashInfer allocates this buffer using routing_logits.dtype, but
            # TRT-LLM's in-kernel routers write packed BF16 values into its
            # live prefix. Recover the BF16 view deferred consumers expect.
            num_tokens, top_k = expert_weights.shape
            expert_weights = expert_weights.view(torch.bfloat16).view(-1, top_k)[
                :num_tokens
            ]
        return gemm2_output, expert_weights, expanded_idx

    def _call_mxfp4_situ_moe(
        w: torch.nn.Module,
        router_logits: torch.Tensor | None,
        topk_weights: torch.Tensor | None,
        topk_ids: torch.Tensor | None,
        x: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        output: torch.Tensor | None,
        enable_pdl: bool,
        do_finalize: bool,
    ):
        if topk_weights is not None:
            return _call_mxfp4_routed_moe(
                w,
                topk_weights,
                topk_ids,
                x,
                output,
                enable_pdl,
                hidden_states_scale=hidden_states_scale,
                do_finalize=do_finalize,
                activation_type=ActivationType.Situ,
            )
        num_experts, local_experts, local_expert_offset = _local_expert_range(w)

        result = trtllm_fp4_block_scale_moe(
            routing_logits=router_logits.to(torch.float32),
            routing_bias=_routing_value(w, "correction_bias", None),
            hidden_states=x,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=w.w13_weight,
            gemm1_weights_scale=w.w13_weight_scale.view(torch.float8_e4m3fn),
            gemm1_bias=None,
            gemm1_alpha=w.gemm1_alpha,
            gemm1_beta=w.gemm1_beta,
            gemm1_clamp_limit=getattr(w, "gemm1_clamp_limit", None),
            gemm2_weights=w.w2_weight,
            gemm2_weights_scale=w.w2_weight_scale.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=num_experts,
            top_k=getattr(w, "top_k"),
            n_group=_routing_value(w, "n_group", 0),
            topk_group=_routing_value(w, "topk_group", 0),
            intermediate_size=getattr(w, "intermediate_size_per_partition"),
            local_expert_offset=local_expert_offset,
            local_num_experts=local_experts,
            routed_scaling_factor=_routing_value(w, "routed_scaling_factor", 1.0),
            routing_method_type=int(_routing_value(w, "routing_method_type", 2)),
            do_finalize=do_finalize,
            enable_pdl=enable_pdl,
            activation_type=ActivationType.Situ,
            tune_max_num_tokens=get_autotune_max_num_tokens(),
            norm_topk_prob=_routing_value(w, "normalize_topk_weights", True),
            output=output if do_finalize else None,
        )

        if not do_finalize:
            return _normalize_kernel_routing_deferred_result(result)
        return output

    def _autotune_mxfp4_situ_moe(
        *,
        x: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        w: torch.nn.Module,
        router_logits: torch.Tensor | None,
        topk_weights: torch.Tensor | None,
        topk_ids: torch.Tensor | None,
        do_finalize: bool,
        enable_pdl: bool,
    ) -> None:
        current_routing = "kernel_routing" if topk_ids is None else "precomputed_topk"
        variants = tuple(
            (routing, finalize)
            for routing in ("kernel_routing", "precomputed_topk")
            for finalize in (False, True)
            if (routing, finalize) != (current_routing, do_finalize)
            and (routing != "kernel_routing" or router_logits is not None)
        )
        if not variants:
            return
        output = (
            torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
            if any(finalize for _, finalize in variants)
            else None
        )
        if topk_ids is None and any(
            routing == "precomputed_topk" for routing, _ in variants
        ):
            # These inputs are only for profiling, never for the model result.
            # Cycle global expert IDs to seed a balanced, rank-independent
            # workload. FI's own initializers generate the native-bucket
            # profile tensors from this valid (tokens, top_k) contract.
            tokens, top_k = x.shape[0], w.top_k
            topk_ids = (
                torch.arange(
                    tokens * top_k, dtype=torch.int32, device=x.device
                ).reshape(tokens, top_k)
                % w.num_experts
            )
            topk_weights = torch.full(
                (tokens, top_k),
                1.0 / top_k,
                dtype=torch.bfloat16,
                device=x.device,
            )
        for routing, finalize in variants:
            precomputed = routing == "precomputed_topk"
            _call_mxfp4_situ_moe(
                w=w,
                router_logits=router_logits,
                topk_weights=topk_weights if precomputed else None,
                topk_ids=topk_ids if precomputed else None,
                x=x,
                hidden_states_scale=hidden_states_scale,
                output=output if finalize else None,
                enable_pdl=enable_pdl,
                do_finalize=finalize,
            )

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_mxfp4_situ_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_mxfp4_situ_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"situ"}),
            "routing_mode": frozenset({"kernel_routing", "precomputed_topk"}),
            "topk_weights_dtype": frozenset({torch.bfloat16}),
            "supports_deferred_finalize": frozenset({True, False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            # FlashInfer's SiTU cubins are MxFP4 x MxFP8 (w4a8) only.
            "internal_activation_dtype": frozenset({"fp8"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_mxfp4_situ_moe_apply(
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
        has_topk_weights = topk_weights is not None
        has_topk_ids = topk_ids is not None
        if has_topk_weights != has_topk_ids:
            raise ValueError(
                "topk_weights and topk_ids must either both be set or both be None"
            )
        use_precomputed_topk = has_topk_weights
        if not use_precomputed_topk and router_logits is None:
            raise ValueError("kernel_routing plan requires router_logits")
        if x.dtype != torch.bfloat16:
            raise TypeError("FlashInfer MXFP4 SiTU requires bf16 input")

        hidden_padded = getattr(w, "hidden_size_padded", w.w2_weight_scale.shape[1])
        hidden_original = getattr(w, "hidden_size_original", hidden_padded)
        if x.shape[0] == 0:
            if do_finalize:
                return x.new_empty((0, hidden_original))
            return (
                x.new_empty((0, hidden_padded)),
                x.new_empty((0, getattr(w, "top_k")), dtype=torch.bfloat16),
                x.new_empty((0,), dtype=torch.int32),
            )
        if x.shape[-1] > hidden_padded:
            raise RuntimeError(
                f"expected hidden size at most {hidden_padded}, got {x.shape[-1]}"
            )
        if x.shape[-1] != hidden_padded:
            x = torch.nn.functional.pad(
                x,
                (0, hidden_padded - x.shape[-1]),
                mode="constant",
                value=0.0,
            )

        # cute-dsl wins under CUDA-graph replay.
        x, x_scale = mxfp8_quantize(
            x,
            False,
            enable_pdl=enable_pdl,
            alignment=hidden_padded,
            backend="cute-dsl",
        )
        hidden_states_scale = x_scale.view(torch.float8_e4m3fn).reshape(x.shape[0], -1)

        if is_autotuning():
            _autotune_mxfp4_situ_moe(
                x=x,
                hidden_states_scale=hidden_states_scale,
                w=w,
                router_logits=router_logits,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                do_finalize=do_finalize,
                enable_pdl=enable_pdl,
            )

        # Deferred finalize returns the raw triple and needs no destination.
        output = None
        if do_finalize:
            out_buf = getattr(w, "_situ_output_buffer", None)
            if (
                out_buf is not None
                and out_buf.shape == (x.shape[0], hidden_padded)
                and out_buf.is_contiguous()
            ):
                output = out_buf
            else:
                output = torch.empty(
                    x.shape[0], hidden_padded, dtype=torch.bfloat16, device=x.device
                )

        result = _call_mxfp4_situ_moe(
            w=w,
            router_logits=router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            x=x,
            hidden_states_scale=hidden_states_scale,
            output=output,
            enable_pdl=enable_pdl,
            do_finalize=do_finalize,
        )
        if not do_finalize:
            return result
        assert result is not None
        if hidden_original != hidden_padded:
            return result[:, :hidden_original].contiguous()
        return result


# ===-----------------------------------------------------------------------===#
# MXINT4 experts: TRT-LLM (NVIDIA SM100)
# ===-----------------------------------------------------------------------===#


# Weight-only INT4 group-32 MoE using the fused FlashInfer block-scale kernel.
#
# Serves ``compressed-tensors`` ``pack-quantized`` checkpoints whose routed
# experts are INT4 with a group size of 32 and ``bfloat16`` group scales (e.g.
# Kimi-K2.5 / K2.6 / K2.7). Blackwell-only, weight-only (activations stay
# ``bfloat16``, no activation quant); routing runs inside the kernel from raw logits.
#
# The checkpoint stores ``weight_packed`` as ``int32`` (eight INT4 per word,
# ``+8`` zero-point) with per-group ``bfloat16`` scales; the kernel wants signed
# ``uint8`` weights (two INT4 per byte) in ``BlockMajorK`` layout with
# permuted/interleaved scales. Weight preprocessing repacks them once.

# FlashInfer's block-scale MoE kernel is tiled with a fixed 128-row epilogue and
# a 128-element K block; the weight/scale permutations are computed against
# these constants.
_EPILOGUE_TILE_M = 128
_BLOCK_K = 128


if platform.is_nvidia:
    from flashinfer import block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
        trtllm_mxint4_block_scale_moe,
    )

    def _repack_int4(packed: torch.Tensor) -> torch.Tensor:
        """Convert one expert's ``int32`` ``(w/s)+8`` words to ``uint8`` ``(w/s)``.

        The checkpoint packs eight unsigned 4-bit values per ``int32`` with the
        ``+8`` zero-point convention; the kernel wants signed two's-complement
        INT4, two per byte, so the offset is removed and the nibbles re-packed.
        """
        assert packed.dim() == 2 and packed.dtype == torch.int32
        shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
        nibbles = (packed.unsqueeze(2) >> shifts) & 0x0F
        nibbles = (nibbles - 8).to(torch.int8).reshape(packed.shape[0], -1, 2)
        out = (nibbles[..., 0] & 0x0F) | ((nibbles[..., 1] & 0x0F) << 4)
        return out.to(torch.uint8)

    def _prepare_mxint4_weights_for_kernel(
        w13_weight_packed: torch.Tensor,
        w2_weight_packed: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Repack + permute + block-layout every expert into the kernel format."""
        cache: dict = {}
        w13_weights_shuffled = []
        w13_scales_shuffled = []
        w2_weights_shuffled = []
        w2_scales_shuffled = []

        for i in range(num_experts):
            gemm1_weight = _repack_int4(w13_weight_packed[i])
            gemm2_weight = _repack_int4(w2_weight_packed[i])

            # gate_up (gemm1): permute rows for the transposed-MMA epilogue, then
            # interleave the group scales the same way.
            perm = _maybe_get_cached_w3_w1_permute_indices(
                cache, gemm1_weight, _EPILOGUE_TILE_M
            )
            w13_weights_shuffled.append(
                gemm1_weight[perm.to(gemm1_weight.device)].contiguous()
            )
            perm_sf = _maybe_get_cached_w3_w1_permute_indices(
                cache,
                w13_weight_scale[i].to(torch.bfloat16),
                _EPILOGUE_TILE_M,
                num_elts_per_sf=32,
            )
            w13_scales_shuffled.append(
                block_scale_interleave(
                    w13_weight_scale[i]
                    .to(torch.bfloat16)[perm_sf.to(w13_weight_scale.device)]
                    .contiguous()
                )
            )

            # down (gemm2): same treatment with the w2 permutation helper.
            perm2 = get_w2_permute_indices_with_cache(
                cache, gemm2_weight, _EPILOGUE_TILE_M
            )
            w2_weights_shuffled.append(
                gemm2_weight[perm2.to(gemm2_weight.device)].contiguous()
            )
            perm2_sf = get_w2_permute_indices_with_cache(
                cache,
                w2_weight_scale[i].to(torch.bfloat16),
                _EPILOGUE_TILE_M,
                num_elts_per_sf=16,
            )
            w2_scales_shuffled.append(
                block_scale_interleave(
                    w2_weight_scale[i]
                    .to(torch.bfloat16)[perm2_sf.to(w2_weight_scale.device)]
                    .contiguous()
                )
            )

        w13_weights = torch.stack(
            [
                convert_to_block_layout(w.view(torch.uint8), _BLOCK_K)
                for w in w13_weights_shuffled
            ]
        )
        w2_weights = torch.stack(
            [
                convert_to_block_layout(w.view(torch.uint8), _BLOCK_K)
                for w in w2_weights_shuffled
            ]
        )
        w13_scales = torch.stack(w13_scales_shuffled)
        w2_scales = torch.stack(w2_scales_shuffled)
        return w13_weights, w13_scales, w2_weights, w2_scales

    def flashinfer_trtllm_mxint4_moe_weights(plan: dict, w: torch.nn.Module):
        num_experts = w.w13_weight_packed.shape[0]

        # Swap [W1(Gate), W3(Up)] -> [W3(Up), W1(Gate)]. The flashinfer fused
        # gated-act epilogue expects the up-proj rows first; the shared loader
        # fills the natural [gate|up] order, so swap the two w13 halves (weight +
        # group scale) here, mirroring the NVFP4 TRT-LLM weight preprocessing.
        # Without this every expert computes silu(W_up x) * (W_gate x) instead of
        # silu(W_gate x) * (W_up x).
        half_w = w.w13_weight_packed.shape[1] // 2
        w1_weight = w.w13_weight_packed.data[:, :half_w, :].clone()
        w.w13_weight_packed.data[:, :half_w, :] = w.w13_weight_packed.data[
            :, half_w:, :
        ]
        w.w13_weight_packed.data[:, half_w:, :] = w1_weight
        del w1_weight

        half_s = w.w13_weight_scale.shape[1] // 2
        w1_scale = w.w13_weight_scale.data[:, :half_s, :].clone()
        w.w13_weight_scale.data[:, :half_s, :] = w.w13_weight_scale.data[:, half_s:, :]
        w.w13_weight_scale.data[:, half_s:, :] = w1_scale
        del w1_scale

        w13_weights, w13_scales, w2_weights, w2_scales = (
            _prepare_mxint4_weights_for_kernel(
                w.w13_weight_packed.data,
                w.w2_weight_packed.data,
                w.w13_weight_scale.data,
                w.w2_weight_scale.data,
                num_experts=num_experts,
            )
        )
        w.w13_weight_packed = torch.nn.Parameter(w13_weights, requires_grad=False)
        w.w2_weight_packed = torch.nn.Parameter(w2_weights, requires_grad=False)
        w.w13_weight_scale = torch.nn.Parameter(w13_scales, requires_grad=False)
        w.w2_weight_scale = torch.nn.Parameter(w2_scales, requires_grad=False)

        # Drop the unused compressed-tensors ``weight_shape`` metadata absorbed
        # during loading.
        for shape_name in ("w13_weight_shape", "w2_weight_shape"):
            if hasattr(w, shape_name):
                delattr(w, shape_name)

        # The kernel reads the routing bias in bfloat16. _correction_bias is a
        # registered nn.Parameter on the layer, so mutate its data in place;
        # rebinding the attribute to a plain Tensor would raise.
        correction_bias = getattr(w, "_correction_bias", None)
        if correction_bias is not None and correction_bias.dtype != torch.bfloat16:
            correction_bias.data = correction_bias.data.to(torch.bfloat16)
        return None

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_mxint4_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_mxint4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"mxint4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"kernel_routing"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            # w2 stores two INT4 per byte, so its packed K dim is ispp // 2 and
            # the 128-element block layout needs ispp divisible by 256.
            "ispp_alignment": frozenset({256}),
            "internal_activation_dtype": frozenset({"input"}),
        },
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_mxint4_moe_apply(
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
        # Idle DP ranks / dummy warmup issue a 0-token forward; the fused kernel
        # divides by token count on the host and SIGFPEs on empty input. This
        # path is do_finalize-only, and x is already [0, hidden].
        if x.shape[0] == 0:
            return x

        # DeepSeekV3 / MiniMax2 routing reads the logits in float32; the layer
        # records the right dtype for its routing method.
        routing_logits = router_logits.to(
            getattr(w, "_routing_logits_dtype", torch.float32)
        )
        local_experts = w.num_local_experts
        result = trtllm_mxint4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=getattr(w, "_correction_bias", None),
            hidden_states=x,
            gemm1_weights=w.w13_weight_packed,
            gemm1_weights_scale=w.w13_weight_scale,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=w.w2_weight_packed,
            gemm2_weights_scale=w.w2_weight_scale,
            num_experts=w.num_experts,
            top_k=w.top_k,
            n_group=w._n_group,
            topk_group=w._topk_group,
            intermediate_size=w.intermediate_size // w.tp_size,
            local_expert_offset=w.ep_rank * local_experts,
            local_num_experts=local_experts,
            routed_scaling_factor=w._routed_scaling_factor,
            routing_method_type=w._routing_method_type,
            enable_pdl=enable_pdl,
            tune_max_num_tokens=get_autotune_max_num_tokens(),
        )
        return result[0]


# ===-----------------------------------------------------------------------===#
# NVFP4 experts: CUTLASS (NVIDIA SM100+)
# ===-----------------------------------------------------------------------===#


if platform.is_nvidia:
    from flashinfer import ActivationType, cutlass_fused_moe

    def flashinfer_cutlass_nvfp4_moe_weights(plan: dict, w: torch.nn.Module):
        half_w = w.w13_weight.shape[1] // 2
        first_half = w.w13_weight.data[:, :half_w, :].clone()
        w.w13_weight.data[:, :half_w, :] = w.w13_weight.data[:, half_w:, :]
        w.w13_weight.data[:, half_w:, :] = first_half

        half_s = w.w13_weight_scale.shape[1] // 2
        first_scale = w.w13_weight_scale.data[:, :half_s, :].clone()
        w.w13_weight_scale.data[:, :half_s, :] = w.w13_weight_scale.data[:, half_s:, :]
        w.w13_weight_scale.data[:, half_s:, :] = first_scale

        w13_ws2 = w.w13_weight_scale_2[:, 0]
        w13_input_scale = w.w13_input_scale.to(torch.float32)
        w2_input_scale = w.w2_input_scale.to(torch.float32)
        w.w13_weight_scale_2 = torch.nn.Parameter(w13_ws2, requires_grad=False)
        w.w13_input_scale_quant = torch.nn.Parameter(
            (1.0 / w13_input_scale).to(torch.float32), requires_grad=False
        )
        w.w2_input_scale_quant = torch.nn.Parameter(
            (1.0 / w2_input_scale).to(torch.float32), requires_grad=False
        )
        w.g1_alphas = torch.nn.Parameter(
            (w13_input_scale * w13_ws2).to(torch.float32), requires_grad=False
        )
        w.g2_alphas = torch.nn.Parameter(
            (w2_input_scale * w.w2_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )

        swiglu_arg = getattr(w, "swiglu_arg", None)
        if swiglu_arg is not None:
            # The cutlass gated activation runs post-dequant, so alpha/beta/
            # limit are passed in the actual-value domain (SwigluBias adaptor).
            num_experts = w.w13_weight.shape[0]
            device = w.w13_weight.device

            def _per_expert(value: float) -> torch.nn.Parameter:
                return torch.nn.Parameter(
                    torch.full(
                        (num_experts,), float(value), dtype=torch.float32, device=device
                    ),
                    requires_grad=False,
                )

            alpha = swiglu_arg.alpha if swiglu_arg.alpha is not None else 1.0
            beta = getattr(w, "swiglu_beta", None)
            w.swiglu_alpha_t = _per_expert(alpha)
            if beta is not None:
                w.swiglu_beta_t = _per_expert(beta)
            if swiglu_arg.limit is not None:
                w.swiglu_limit_t = _per_expert(swiglu_arg.limit)

        scales = w.w13_weight_scale
        scale_ndim = scales.ndim
        if scale_ndim == 2:
            scales = scales.unsqueeze(0)
        batches, rows, cols = scales.shape
        rows_padded = (rows + 127) // 128 * 128
        cols_padded = (cols + 3) // 4 * 4
        padded = torch.zeros(
            (batches, rows_padded, cols_padded),
            dtype=scales.dtype,
            device=scales.device,
        )
        padded[:batches, :rows, :cols] = scales
        padded = padded.reshape(batches, rows_padded // 128, 4, 32, cols_padded // 4, 4)
        padded = padded.permute((0, 1, 4, 3, 2, 5)).contiguous()
        if scale_ndim == 2:
            swizzled = padded.reshape(rows_padded, cols_padded)
        else:
            swizzled = padded.reshape(batches, rows_padded, cols_padded)
        w.w13_blockscale_swizzled = torch.nn.Parameter(swizzled, requires_grad=False)

        scales = w.w2_weight_scale
        scale_ndim = scales.ndim
        if scale_ndim == 2:
            scales = scales.unsqueeze(0)
        batches, rows, cols = scales.shape
        rows_padded = (rows + 127) // 128 * 128
        cols_padded = (cols + 3) // 4 * 4
        padded = torch.zeros(
            (batches, rows_padded, cols_padded),
            dtype=scales.dtype,
            device=scales.device,
        )
        padded[:batches, :rows, :cols] = scales
        padded = padded.reshape(batches, rows_padded // 128, 4, 32, cols_padded // 4, 4)
        padded = padded.permute((0, 1, 4, 3, 2, 5)).contiguous()
        if scale_ndim == 2:
            swizzled = padded.reshape(rows_padded, cols_padded)
        else:
            swizzled = padded.reshape(batches, rows_padded, cols_padded)
        w.w2_blockscale_swizzled = torch.nn.Parameter(swizzled, requires_grad=False)
        return None

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_cutlass_nvfp4_moe_apply",
        solution="flashinfer_cutlass",
        weight_preprocessor=flashinfer_cutlass_nvfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"nvfp4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutlass_nvfp4_moe_apply(
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
        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)
        output = torch.empty(
            x.shape[0], x.shape[1], dtype=torch.bfloat16, device=x.device
        )
        swiglu_alpha = getattr(w, "swiglu_alpha_t", None)
        activation_type = (
            ActivationType.Swiglu if swiglu_alpha is None else ActivationType.SwigluBias
        )
        return cutlass_fused_moe(
            output=output,
            input=x,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=w.w13_weight.view(torch.long),
            fc2_expert_weights=w.w2_weight.view(torch.long),
            output_dtype=torch.bfloat16,
            input_sf=None,
            quant_scales=[
                w.w13_input_scale_quant,
                w.w13_blockscale_swizzled.view(torch.int32),
                w.g1_alphas,
                w.w2_input_scale_quant,
                w.w2_blockscale_swizzled.view(torch.int32),
                w.g2_alphas,
            ],
            ep_size=getattr(w, "ep_size", 1),
            ep_rank=getattr(w, "ep_rank", 0),
            tp_size=getattr(w, "tp_size", 1),
            tp_rank=getattr(w, "tp_rank", 0),
            tune_max_num_tokens=get_autotune_max_num_tokens(),
            activation_type=activation_type,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=getattr(w, "swiglu_beta_t", None),
            swiglu_limit=getattr(w, "swiglu_limit_t", None),
            enable_pdl=enable_pdl,
        )[0]


# ===-----------------------------------------------------------------------===#
# NVFP4 experts: TRT-LLM (NVIDIA SM100)
# ===-----------------------------------------------------------------------===#


TRTLLM_NVFP4_ISPP_ALIGNMENT = 64
# Non-gated GEMM1 tiles its rows by 128.
TRTLLM_NVFP4_RELU2_ISPP_ALIGNMENT = 128
# The fused kernels miscompute a hidden size off this multiple, so it is zero-padded to one.
TRTLLM_NVFP4_HIDDEN_ALIGNMENT = 256


def _zero_pad(t: torch.Tensor, pad: tuple[int, ...]) -> torch.Tensor:
    """Zero-pad byte-sized FP4 codes or FP8 block scales as raw bytes."""
    return F.pad(t.view(torch.uint8), pad).view(t.dtype)


@dataclasses.dataclass(frozen=True)
class PreparedNvfp4Routing:
    """Split routing state of the precomputed-top-k NVFP4 kernel for one token count.

    The top-k kernel fills ``routing`` on any stream; the apply then runs the
    expert GEMMs from it into ``workspace``.
    """

    num_tokens: int
    activation_type: int
    routing: object
    body: object
    workspace: tuple
    body_runner: object
    output: torch.Tensor


if platform.is_nvidia:
    from flashinfer import (
        fp4_quantize,
        nvfp4_block_scale_interleave,
    )

    # The split routing API: metadata filled apart from the expert GEMMs. An
    # older wheel fails this import rather than silently keeping the fused launch.
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe import core as _fi_core
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices as maybe_get_cached_w3_w1_permute_indices,
    )
    from flashinfer.fused_moe.core import (
        get_w2_permute_indices_with_cache,
    )
    from flashinfer.fused_moe.da_moe import DABody
    from flashinfer.tllm_enums import ActivationType
    from tokenspeed_kernel.thirdparty.flashinfer.trtllm_moe import (
        trtllm_fp4_block_scale_moe as tokenspeed_trtllm_fp4_block_scale_moe,
    )
    from tokenspeed_kernel.thirdparty.flashinfer.trtllm_moe import (
        trtllm_fp4_block_scale_routed_moe as tokenspeed_trtllm_fp4_block_scale_routed_moe,
    )

    def _flashinfer_trtllm_nvfp4_moe_weights(
        plan: dict, w: torch.nn.Module, *, activation: str
    ):
        if activation not in ("swiglu", "situ", "relu2"):
            raise ValueError(f"unsupported NVFP4 MoE activation {activation!r}")
        # relu2 is non-gated: w13 holds only the up projection.
        gated = activation != "relu2"
        _group_size = 16
        _correction_bias = getattr(w, "_correction_bias", None)
        _routing_logits_dtype = getattr(w, "_routing_logits_dtype", torch.bfloat16)

        num_experts = w.w13_weight.shape[0]
        w13_rows = w.w13_weight.shape[1]
        # intermediate_size_per_partition: gated w13 stacks gate + up.
        intermediate_size = w13_rows // 2 if gated else w13_rows
        hidden_size = w.w13_weight.shape[2] * 2
        hidden_pad = -hidden_size % TRTLLM_NVFP4_HIDDEN_ALIGNMENT
        padded_hidden = hidden_size + hidden_pad

        if gated:
            # Fix 1: Swap [W1(Gate), W3(Up)] -> [W3(Up), W1(Gate)].
            # The fused gated-act reorder interleaves [first_half, second_half] as
            # [row0_first, row0_second, row1_first, row1_second, ...].
            # It expects [W3(Up), W1(Gate)] so that the interleaved result pairs
            # each up-proj row with its corresponding gate-proj row correctly.
            half_w = w13_rows // 2
            w1_weight = w.w13_weight.data[:, :half_w, :].clone()
            w.w13_weight.data[:, :half_w, :] = w.w13_weight.data[:, half_w:, :]
            w.w13_weight.data[:, half_w:, :] = w1_weight
            del w1_weight

            half_s = w.w13_weight_scale.shape[1] // 2
            w1_scale = w.w13_weight_scale.data[:, :half_s, :].clone()
            w.w13_weight_scale.data[:, :half_s, :] = w.w13_weight_scale.data[
                :, half_s:, :
            ]
            w.w13_weight_scale.data[:, half_s:, :] = w1_scale
            del w1_scale

        # Shuffle weights and scales using fused-kernel permute indices.
        cache: dict = {}
        epilogue_tile_m = 128

        # View as fp8 for permutation (uint8 and fp8_e4m3fn are both 1 byte)
        w13_fp4 = w.w13_weight.data.view(torch.float8_e4m3fn).reshape(
            num_experts, w13_rows, hidden_size // 2
        )
        w13_scales = w.w13_weight_scale.data.view(torch.float8_e4m3fn).reshape(
            num_experts, w13_rows, hidden_size // _group_size
        )
        w2_fp4 = w.w2_weight.data.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // 2
        )
        w2_scales = w.w2_weight_scale.data.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // _group_size
        )
        if hidden_pad:
            # Zero FP4 codes and zero block scales: the padded columns and rows contribute nothing.
            w13_fp4 = _zero_pad(w13_fp4, (0, hidden_pad // 2))
            w13_scales = _zero_pad(w13_scales, (0, hidden_pad // _group_size))
            w2_fp4 = _zero_pad(w2_fp4, (0, 0, 0, hidden_pad))
            w2_scales = _zero_pad(w2_scales, (0, 0, 0, hidden_pad))

        w13_weights_shuffled = []
        w13_scales_shuffled = []
        w2_weights_shuffled = []
        w2_scales_shuffled = []

        for idx in range(num_experts):
            # W1/W3 (gemm1) weight permutation
            perm = maybe_get_cached_w3_w1_permute_indices(
                cache,
                w13_fp4[idx].view(torch.uint8),
                epilogue_tile_m,
                is_gated_act_gemm=gated,
            )
            w13_weights_shuffled.append(
                w13_fp4[idx].view(torch.uint8)[perm.to(w13_fp4.device)].contiguous()
            )
            # W1/W3 scale permutation + interleave
            perm_sf = maybe_get_cached_w3_w1_permute_indices(
                cache,
                w13_scales[idx].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
                is_gated_act_gemm=gated,
            )
            w13_scales_shuffled.append(
                nvfp4_block_scale_interleave(
                    w13_scales[idx]
                    .view(torch.uint8)[perm_sf.to(w13_scales.device)]
                    .contiguous()
                )
            )
            # W2 (gemm2) weight permutation
            perm2 = get_w2_permute_indices_with_cache(
                cache, w2_fp4[idx].view(torch.uint8), epilogue_tile_m
            )
            w2_weights_shuffled.append(
                w2_fp4[idx].view(torch.uint8)[perm2.to(w2_fp4.device)].contiguous()
            )
            # W2 scale permutation + interleave
            perm2_sf = get_w2_permute_indices_with_cache(
                cache,
                w2_scales[idx].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            w2_scales_shuffled.append(
                nvfp4_block_scale_interleave(
                    w2_scales[idx]
                    .view(torch.uint8)[perm2_sf.to(w2_scales.device)]
                    .contiguous()
                )
            )

        # Stack and store shuffled weights (uint8)
        w.gemm1_weights_fp4_shuffled = torch.nn.Parameter(
            torch.stack(w13_weights_shuffled), requires_grad=False
        )
        w.gemm1_scales_fp4_shuffled = torch.nn.Parameter(
            torch.stack(w13_scales_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, w13_rows, padded_hidden // _group_size),
            requires_grad=False,
        )
        w.gemm2_weights_fp4_shuffled = torch.nn.Parameter(
            torch.stack(w2_weights_shuffled), requires_grad=False
        )
        w.gemm2_scales_fp4_shuffled = torch.nn.Parameter(
            torch.stack(w2_scales_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, padded_hidden, intermediate_size // _group_size),
            requires_grad=False,
        )

        # Free original weights (replaced by shuffled versions)
        del w.w13_weight
        del w.w2_weight
        del w.w13_weight_scale
        del w.w2_weight_scale

        # Compute fused-kernel scales. The trtllm SwiGLU MoE kernel dequantizes the GEMM1 gate
        # (W1) and up (W3) halves with separate scalars, so feed each its own global scale_2
        # (else non-uniform-W1/W3 checkpoints mis-scale the up-proj by up_s2/gate_s2).
        ws2 = w.w13_weight_scale_2
        if ws2.dim() == 2 and ws2.shape[1] == 2:
            gate_ws2, up_ws2 = ws2[:, 0], ws2[:, 1]
        else:
            gate_ws2 = ws2.reshape(ws2.shape[0])
            up_ws2 = gate_ws2
        w13_input_scale = w.w13_input_scale.to(torch.float32)
        w2_input_scale = w.w2_input_scale.to(torch.float32)

        # Store input_scale_quant for runtime fp4_quantize
        w13_input_scale_quant = (1.0 / w13_input_scale).to(torch.float32)
        w2_input_scale_quant = (1.0 / w2_input_scale).to(torch.float32)

        w.w13_input_scale_quant = torch.nn.Parameter(
            w13_input_scale_quant, requires_grad=False
        )
        # gate (W1) dequant alpha -> output1_scale_gate_scalar
        w.g1_alphas = torch.nn.Parameter(
            (w13_input_scale * gate_ws2).to(torch.float32), requires_grad=False
        )
        w.g2_alphas = torch.nn.Parameter(
            (w2_input_scale * w.w2_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )
        if activation == "relu2":
            # Dequant is applied before squaring, so scale_c is only the GEMM2-input requant.
            w.g1_scale_c = torch.nn.Parameter(
                (w2_input_scale_quant * torch.ones_like(up_ws2)).to(torch.float32),
                requires_grad=False,
            )
        elif activation == "situ":
            # SiTU is nonlinear in BOTH GEMM1 halves: the kernel dequantizes
            # the raw accumulators with output1_scale_gate_scalar inside the
            # activation, so (a) gate and up must share one per-expert global
            # scale and (b) output1_scale_scalar must carry ONLY the
            # GEMM2-input requant factor -- the SwiGLU recipe below folds the
            # up-half dequant into it, which for SiTU would move that factor
            # inside tanh and change the activation.
            if not torch.equal(gate_ws2, up_ws2):
                raise RuntimeError(
                    "NVFP4 SiTU requires equal gate/up weight_scale_2 per expert."
                )
            w.g1_scale_c = torch.nn.Parameter(
                (w2_input_scale_quant * torch.ones_like(up_ws2)).to(torch.float32),
                requires_grad=False,
            )
            # Actual-domain per-expert SiTU constants: dequant happens before
            # the activation, so alpha/beta pass through unchanged (no
            # raw-domain folding as in SwiGLU).
            alpha = _positive_situ_value(w, "activation_situ_beta", "situ_beta")
            beta = _positive_situ_value(
                w, "activation_situ_linear_beta", "situ_linear_beta"
            )
            w.gemm1_alpha = torch.nn.Parameter(
                torch.full_like(w.g1_alphas.data, alpha), requires_grad=False
            )
            w.gemm1_beta = torch.nn.Parameter(
                torch.full_like(w.g1_alphas.data, beta), requires_grad=False
            )
        else:
            # up (W3) dequant alpha folded with the GEMM2-input requant
            # -> output1_scale_scalar
            w.g1_scale_c = torch.nn.Parameter(
                w2_input_scale_quant * (w13_input_scale * up_ws2).to(torch.float32),
                requires_grad=False,
            )

            swiglu_arg = getattr(w, "swiglu_arg", None)
            if swiglu_arg is not None:
                # The fused gated activation runs on pre-dequant accumulators
                # (actual = raw * g1_alphas), so beta and the clamp limit must be
                # expressed in the raw domain; alpha passes through unchanged.
                # Folding the gate dequant scale is only exact when the gate and
                # up halves share one global scale.
                if not torch.equal(gate_ws2, up_ws2):
                    raise RuntimeError(
                        "NVFP4 swiglu alpha/limit requires equal gate/up "
                        "weight_scale_2 per expert."
                    )
                alpha = swiglu_arg.alpha if swiglu_arg.alpha is not None else 1.0
                beta = getattr(w, "swiglu_beta", None)
                w.gemm1_alpha = torch.nn.Parameter(
                    torch.full_like(w.g1_alphas.data, float(alpha)),
                    requires_grad=False,
                )
                if beta is not None:
                    w.gemm1_beta = torch.nn.Parameter(
                        float(beta) / w.g1_alphas.data, requires_grad=False
                    )
                if swiglu_arg.limit is not None:
                    w.gemm1_clamp_limit = torch.nn.Parameter(
                        float(swiglu_arg.limit) / w.g1_alphas.data,
                        requires_grad=False,
                    )

        # Store intermediate_size_per_partition for the executor
        w.intermediate_size_per_partition = intermediate_size

        # Free per-shard scales that are no longer needed
        del w.w13_weight_scale_2
        del w.w2_weight_scale_2
        del w.w13_input_scale
        del w.w2_input_scale

        # The fused MoE kernel requires routing bias dtype to match
        # routing logits dtype. Cast here (post weight-load) so the captured
        # bias reflects the loaded values, not the empty Parameter.
        if _correction_bias is not None:
            _correction_bias = _correction_bias.to(_routing_logits_dtype)

    _FP4_TILES = (8, 16, 32, 64, 128)

    def _default_tile(num_tokens: int, top_k: int, num_local_experts: int) -> int:
        """The launcher's fallback routing tile: the neighbour below its occupancy heuristic."""
        rows = math.ceil(num_tokens * top_k / num_local_experts)
        rows = 1 if rows <= 1 else 1 << (rows - 1).bit_length()
        center = min(max(rows, _FP4_TILES[0]), _FP4_TILES[-1])
        index = _FP4_TILES.index(center)
        return _FP4_TILES[index - 1] if index else center

    def _nvfp4_runner_kwargs(
        w: torch.nn.Module,
        spec: object,
        activation_type: int,
        enable_pdl: bool,
    ) -> dict:
        """Keyword arguments the flashinfer runner takes beside its tensor inputs."""
        return dict(
            routing_input_mode=_fi_core.RoutingInputMode.UnpackedPrecomputed,
            num_experts=spec.num_experts,
            routing_bias=None,
            gemm1_weights=w.gemm1_weights_fp4_shuffled.data,
            gemm1_weights_scale=w.gemm1_scales_fp4_shuffled.data.view(
                torch.float8_e4m3fn
            ),
            gemm1_bias=None,
            gemm1_alpha=getattr(w, "gemm1_alpha", None),
            gemm1_beta=getattr(w, "gemm1_beta", None),
            gemm1_clamp_limit=getattr(w, "gemm1_clamp_limit", None),
            gemm2_weights=w.gemm2_weights_fp4_shuffled.data,
            gemm2_weights_scale=w.gemm2_scales_fp4_shuffled.data.view(
                torch.float8_e4m3fn
            ),
            gemm2_bias=None,
            output1_scale_scalar=w.g1_scale_c.data,
            output1_scale_gate_scalar=w.g1_alphas.data,
            output2_scale_scalar=w.g2_alphas.data,
            # flashinfer's runner reads this key unconditionally; there is no per-token scale.
            per_token_scale=None,
            n_group=None,
            topk_group=None,
            local_expert_offset=spec.ep_rank * spec.num_local_experts,
            routed_scaling_factor=None,
            routing_method_type=0,
            enable_pdl=enable_pdl
            and _fi_core._device_support_moe_pdl(w.g1_alphas.device),
            do_finalize=False,
            activation_type=activation_type,
            num_fused_shared_experts=0,
            norm_topk_prob=True,
            routing_replay_out=None,
        )

    def _nvfp4_runner_inputs(
        output: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        hs_fp4: torch.Tensor,
        hs_scale: torch.Tensor,
    ) -> list:
        return _fi_core.MoeRunnerInputs(
            output=output,
            routing_logits=None,
            topk_ids=topk_ids,
            expert_weights=topk_weights,
            hidden_states=hs_fp4,
            hidden_states_scale=hs_scale.view(torch.float8_e4m3fn),
            gemm1_lora_delta=None,
            per_token_scale=None,
        ).to_list()

    def prepare_nvfp4_routing(
        w: torch.nn.Module,
        x_quantized: tuple[torch.Tensor, torch.Tensor],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        activation_type: int | None,
        enable_pdl: bool,
    ) -> PreparedNvfp4Routing:
        """Prepare split routing for the precomputed-top-k NVFP4 kernel at one token count.

        Resolves the same tactic the fused launch would take, then allocates the
        routing metadata and the expert-GEMM workspace for it, so a later
        the top-k kernel fills ``routing`` and the apply launches nothing but the
        GEMMs. Call outside graph capture, once per token count and per
        layer: the top-k of one layer fills its slot while another layer's
        GEMMs may still be reading theirs.

        Args:
            w: Processed NVFP4 expert weights, as the apply kernel receives them.
            x_quantized: The apply's ``x_quantized`` pair at this token count.
            topk_ids: Selected expert ids ``[tokens, top_k]``.
            topk_weights: FP32 or BF16 route weights ``[tokens, top_k]``.
            activation_type: flashinfer activation, or None for SwiGLU.
            enable_pdl: Enable NVIDIA programmatic dependent launch.

        Returns:
            The prepared routing for this token count.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare the routing outside graph capture")
        if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
            raise ValueError("the routing takes int32 ids and FP32 weights")
        spec = w._spec
        hs_fp4, hs_scale = x_quantized
        num_tokens, packed_width = hs_fp4.shape
        ids, weights = topk_ids.contiguous(), topk_weights.contiguous()
        activation = (
            _fi_core.ActivationType.Swiglu.value
            if activation_type is None
            else activation_type
        )
        runtime = _fi_core.get_trtllm_moe_sm100_module()
        runner = runtime.MoERunner(
            runtime.moe_op,
            top_k=spec.top_k,
            num_local_experts=spec.num_local_experts,
            dtype_act=_fi_core.deduce_trtllm_gen_tensor_dtype(
                hs_fp4, hs_scale.view(torch.float8_e4m3fn)
            ),
            dtype_weights=_fi_core.deduce_trtllm_gen_tensor_dtype(
                w.gemm1_weights_fp4_shuffled.data,
                w.gemm1_scales_fp4_shuffled.data.view(torch.float8_e4m3fn),
            ),
            fp8_quantization_type=_fi_core.Fp8QuantizationType.NoneFp8,
            hidden_size=packed_width * 2,
            intermediate_size=w.intermediate_size_per_partition,
            activation_type=activation,
            weight_layout=_fi_core.WeightLayout.MajorK,
            use_shuffled_weight=True,
            num_experts=spec.num_experts,
            num_fused_shared_experts=0,
        )
        output = _fi_core._alloc_trtllm_moe_output(
            num_tokens, packed_width * 2, False, hs_fp4.device
        )
        inputs = _nvfp4_runner_inputs(output, ids, weights, hs_fp4, hs_scale)
        kwargs = _nvfp4_runner_kwargs(w, spec, activation, enable_pdl)
        tuning = runner._make_tuning_config(
            _fi_core.MoeRunnerInputs.from_list(inputs),
            tune_max_num_tokens=get_autotune_max_num_tokens(),
            routing_input_mode=_fi_core.RoutingInputMode.UnpackedPrecomputed,
            use_cold_l2_cache=True,
            use_cuda_graph=True,
        )
        _, tactic = AutoTuner.get().choose_one(
            "flashinfer::trtllm_fp4_block_scale_moe", [runner], tuning, inputs, **kwargs
        )
        if tactic == -1:
            tile, config = (
                _default_tile(num_tokens, spec.top_k, spec.num_local_experts),
                -1,
            )
        else:
            tile, config = int(tactic[0]), int(tactic[1])
        metadata = _fi_core.trtllm_moe_allocate_routing_metadata_multi_tile(
            ids,
            num_experts=spec.num_experts,
            top_k=spec.top_k,
            local_expert_offset=spec.ep_rank * spec.num_local_experts,
            num_local_experts=spec.num_local_experts,
            tile_ns=(tile,),
            routing_input_mode=_fi_core.RoutingInputMode.UnpackedPrecomputed,
            topk_weights=weights,
        )
        body = DABody(tactic=config, tile_n=tile)
        body_runner = runtime.DABodyRunner(runner)
        workspace = tuple(
            body_runner.prepare_body(inputs, body, metadata.slots[0], **kwargs)
        )
        return PreparedNvfp4Routing(
            num_tokens,
            activation,
            metadata.slots[0],
            body,
            workspace,
            body_runner,
            output,
        )

    def _apply_prepared_routing(
        prepared: PreparedNvfp4Routing,
        w: torch.nn.Module,
        spec: object,
        hs_fp4: torch.Tensor,
        hs_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        enable_pdl: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Launch only the expert GEMMs from populated routing metadata."""
        num_tokens = hs_fp4.shape[0]
        if prepared.num_tokens != num_tokens:
            raise ValueError(
                f"prepared routing serves {prepared.num_tokens} tokens, got {num_tokens}"
            )
        prepared.body_runner.forward_from_metadata(
            _nvfp4_runner_inputs(
                prepared.output,
                topk_ids,
                topk_weights,
                hs_fp4,
                hs_scale,
            ),
            prepared.body,
            prepared.routing,
            prepared.workspace,
            **_nvfp4_runner_kwargs(w, spec, prepared.activation_type, enable_pdl),
        )
        # FP4 body workspace order: gemm1 out, its scale, activation, FC2 token scales, gemm2 out, two scratches.
        gemm2_out = prepared.workspace[4]
        if (
            gemm2_out.dtype != torch.bfloat16
            or gemm2_out.shape[-1] != hs_fp4.shape[1] * 2
        ):
            raise RuntimeError("the DA body workspace no longer holds gemm2_out fifth")
        permute_map = prepared.routing.expanded_idx_to_permuted_idx.view(-1)[
            : num_tokens * spec.top_k
        ]
        return gemm2_out, topk_weights, permute_map

    def flashinfer_trtllm_nvfp4_moe_weights(plan: dict, w: torch.nn.Module):
        return _flashinfer_trtllm_nvfp4_moe_weights(plan, w, activation="swiglu")

    def flashinfer_trtllm_nvfp4_situ_moe_weights(plan: dict, w: torch.nn.Module):
        # SiTU shares the standard TRT-LLM [up|gate] shuffled layout with
        # SwiGLU; only the GEMM1 output scales and act constants differ.
        return _flashinfer_trtllm_nvfp4_moe_weights(plan, w, activation="situ")

    def flashinfer_trtllm_nvfp4_relu2_moe_weights(plan: dict, w: torch.nn.Module):
        return _flashinfer_trtllm_nvfp4_moe_weights(plan, w, activation="relu2")

    def _flashinfer_trtllm_nvfp4_moe_apply(
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        w: torch.nn.Module,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor | None,
        topk_ids: torch.Tensor | None,
        do_finalize: bool,
        enable_pdl: bool,
        routed: bool,
        activation_type: int | None = None,
        output: torch.Tensor | None = None,
        x_quantized: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None
        ) = None,
        routing_prepared: PreparedNvfp4Routing | None = None,
    ):
        """Shared body for the in-kernel-routing and precomputed-topk variants.

        ``routed`` selects between ``trtllm_fp4_block_scale_moe`` (in-kernel
        routing from ``router_logits``) and ``trtllm_fp4_block_scale_routed_moe``
        (precomputed ``topk_ids``/``topk_weights``); everything else is
        identical. ``routing_prepared`` (precomputed top-k, deferred finalize
        only) skips the routing launch: its metadata was populated already.
        """
        _spec = getattr(w, "_spec", None)

        prequantized = isinstance(x, tuple)
        data = x[0] if prequantized else x
        num_tokens = data.shape[0]
        # Idle DP ranks pass 0 tokens and the fused kernel divides by token count on host; skip experts.
        if num_tokens == 0:
            output = (
                data.new_empty((0, data.shape[1] * 2), dtype=torch.bfloat16)
                if prequantized
                else data
            )
            if do_finalize:
                return output
            return (
                output,
                data.new_empty((0, _spec.top_k), dtype=torch.bfloat16),
                # moe_finalize_fuse_shared expects a 1-D [num_tokens * top_k] permute map.
                data.new_empty((0,), dtype=torch.int32),
            )

        hidden = data.shape[1] * 2 if prequantized else data.shape[1]
        hidden_pad = w.gemm2_weights_fp4_shuffled.shape[1] - hidden
        if hidden_pad and (prequantized or output is not None or not do_finalize):
            raise NotImplementedError(
                f"NVFP4 MoE at hidden size {hidden} (padded for the fused kernels) "
                "takes bf16 input and returns a finalized output of its own"
            )
        if prequantized:
            hs_fp4, hs_scale = x
        elif x_quantized is None:
            # Quantize input to FP4 using the fused-kernel scale layout.
            hs_fp4, hs_scale = fp4_quantize(
                F.pad(x, (0, hidden_pad)) if hidden_pad else x,
                w.w13_input_scale_quant,
                is_sf_swizzled_layout=False,
                enable_pdl=enable_pdl,
            )
        else:
            hs_fp4, hs_scale = x_quantized
            if hs_fp4.shape != (num_tokens, data.shape[1] // 2) or hs_scale.shape != (
                num_tokens,
                data.shape[1] // 16,
            ):
                raise ValueError("x_quantized does not match x's [tokens, hidden]")
            if (
                hs_fp4.dtype != torch.uint8
                or hs_scale.dtype not in (torch.uint8, torch.float8_e4m3fn)
                or not hs_fp4.is_contiguous()
                or not hs_scale.is_contiguous()
                or hs_fp4.device != x.device
                or hs_scale.device != x.device
            ):
                raise ValueError(
                    "x_quantized must be contiguous packed e2m1 bytes and e4m3 "
                    "block scales in the linear layout, on x's device"
                )

        if routing_prepared is not None:
            if not routed or do_finalize:
                raise ValueError(
                    "prepared routing serves the precomputed-top-k kernel with a deferred finalize"
                )
            return _apply_prepared_routing(
                routing_prepared,
                w,
                _spec,
                hs_fp4,
                hs_scale,
                topk_ids,
                topk_weights,
                enable_pdl,
            )

        # GEMM and scale arguments shared by both kernel entry points.
        common_kwargs = dict(
            hidden_states=hs_fp4,
            hidden_states_scale=hs_scale.view(torch.float8_e4m3fn),
            gemm1_weights=w.gemm1_weights_fp4_shuffled.data,
            gemm1_weights_scale=w.gemm1_scales_fp4_shuffled.data.view(
                torch.float8_e4m3fn
            ),
            gemm1_bias=None,
            gemm1_alpha=getattr(w, "gemm1_alpha", None),
            gemm1_beta=getattr(w, "gemm1_beta", None),
            gemm1_clamp_limit=getattr(w, "gemm1_clamp_limit", None),
            gemm2_weights=w.gemm2_weights_fp4_shuffled.data,
            gemm2_weights_scale=w.gemm2_scales_fp4_shuffled.data.view(
                torch.float8_e4m3fn
            ),
            gemm2_bias=None,
            output1_scale_scalar=w.g1_scale_c.data,
            output1_scale_gate_scalar=w.g1_alphas.data,
            output2_scale_scalar=w.g2_alphas.data,
            num_experts=_spec.num_experts,
            top_k=_spec.top_k,
            intermediate_size=w.intermediate_size_per_partition,
            local_expert_offset=_spec.ep_rank * _spec.num_local_experts,
            local_num_experts=_spec.num_local_experts,
            do_finalize=do_finalize,
            enable_pdl=enable_pdl,
            tune_max_num_tokens=get_autotune_max_num_tokens(),
        )

        if activation_type is not None:
            common_kwargs["activation_type"] = activation_type
        if output is not None:
            # Caller-owned destination (e.g. a fused all-reduce lane slice);
            # the kernel writes the finalized rows in place.
            common_kwargs["output"] = output

        if routed:
            # UnpackedPrecomputed route weights pass at their native dtype
            # (fp32 or bf16), no cast needed since flashinfer 0.6.16.
            topk = (
                topk_ids.to(torch.int32),
                topk_weights,
            )
            result = tokenspeed_trtllm_fp4_block_scale_routed_moe(
                topk_ids=topk,
                routing_bias=None,
                n_group=None,
                topk_group=None,
                routed_scaling_factor=None,
                **common_kwargs,
            )
        else:
            _routing_logits_dtype = getattr(w, "_routing_logits_dtype", torch.bfloat16)
            result = tokenspeed_trtllm_fp4_block_scale_moe(
                routing_logits=router_logits.to(_routing_logits_dtype),
                routing_bias=getattr(w, "_correction_bias", None),
                n_group=getattr(w, "_n_group", 0),
                topk_group=getattr(w, "_topk_group", 0),
                routed_scaling_factor=getattr(w, "_routed_scaling_factor", 1.0),
                routing_method_type=getattr(w, "_routing_method_type", 0),
                **common_kwargs,
            )
        if do_finalize:
            return result[0][:, :hidden].contiguous() if hidden_pad else result[0]
        # Deferred: [gemm2_out, expert_weights, expanded_idx_to_permuted_idx]
        gemm2_out, expert_weights, expanded_idx = result
        if routed:
            # expert_weights just echoes the caller's input; shared-sink callers drop it and pass their own.
            return (gemm2_out, expert_weights, expanded_idx)
        # Flashinfer's Python wrapper allocates expert_weights with
        # ``routing_logits.dtype`` (fp32 for DSv3), but the C++ routing
        # kernel writes bf16 contiguously for DeepSeekV3 routing
        # into the buffer. Only the first half holds valid data; reading
        # as fp32 interprets two adjacent bf16s as one fp32. Reinterpret
        # to bf16 and keep the live prefix.
        if expert_weights.dtype == torch.float32:
            n, k = expert_weights.size()
            expert_weights = expert_weights.view(torch.bfloat16).view(-1, k)[:n]
        return (gemm2_out, expert_weights, expanded_idx)

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_nvfp4_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_nvfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 7),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"nvfp4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"kernel_routing"}),
            "supports_deferred_finalize": frozenset({True}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({TRTLLM_NVFP4_ISPP_ALIGNMENT}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_nvfp4_moe_apply(
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
        x_quantized: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None
        ) = None,
    ):
        return _flashinfer_trtllm_nvfp4_moe_apply(
            x,
            w,
            router_logits,
            topk_weights,
            topk_ids,
            do_finalize,
            enable_pdl,
            routed=False,
            x_quantized=x_quantized,
        )

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_nvfp4_routed_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_nvfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 7),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"nvfp4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({True}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({TRTLLM_NVFP4_ISPP_ALIGNMENT}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        # One below in-kernel routing: this wins only for plans with routing_mode="precomputed_topk".
        priority=Priority.PERFORMANT + 3,
    )
    def flashinfer_trtllm_nvfp4_routed_moe_apply(
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
        x_quantized: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None
        ) = None,
        routing_prepared: PreparedNvfp4Routing | None = None,
    ):
        assert (
            topk_weights is not None and topk_ids is not None
        ), "precomputed_topk plan requires topk_weights and topk_ids"
        return _flashinfer_trtllm_nvfp4_moe_apply(
            x,
            w,
            router_logits,
            topk_weights,
            topk_ids,
            do_finalize,
            enable_pdl,
            routed=True,
            x_quantized=x_quantized,
            routing_prepared=routing_prepared,
        )

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_nvfp4_situ_routed_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_nvfp4_situ_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 7),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.bfloat16, torch.uint8},
        ),
        traits={
            "weight_dtype": frozenset({"nvfp4"}),
            "activation": frozenset({"situ"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({True, False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({TRTLLM_NVFP4_ISPP_ALIGNMENT}),
            # NVFP4 SiTU runs w4a4, accepting BF16 or an already-quantized pair.
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_nvfp4_situ_routed_moe_apply(
        plan: dict,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        w: torch.nn.Module,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        num_tokens_global: int | None = None,
        max_num_tokens_per_gpu: int | None = None,
        do_finalize: bool = True,
        enable_pdl: bool = False,
    ):
        if topk_weights is None or topk_ids is None:
            raise ValueError("precomputed_topk plan requires topk_weights and topk_ids")
        if isinstance(x, tuple):
            data, scales = x
            if data.dtype != torch.uint8 or scales.dtype not in (
                torch.uint8,
                torch.float8_e4m3fn,
            ):
                raise TypeError(
                    "Prequantized NVFP4 requires uint8 data and byte-sized block scales"
                )
            if (
                data.ndim != 2
                or data.shape[1] % 8 != 0
                or scales.shape != (data.shape[0], data.shape[1] // 8)
            ):
                raise ValueError(
                    "Prequantized NVFP4 requires linear per-token block scales"
                )
            output_shape = (data.shape[0], data.shape[1] * 2)
        else:
            if x.dtype != torch.bfloat16:
                raise TypeError(
                    "FlashInfer NVFP4 SiTU requires bf16 input or an NVFP4 pair"
                )
            output_shape = x.shape
        # Caller-owned destination (e.g. K3's fused all-reduce lane slice):
        # writing the finalized rows in place keeps the join zero-copy, same
        # contract as the MXFP4 SiTU kernel. Only meaningful when finalizing;
        # deferred mode returns the permuted triple (gemm2 rows, the caller's
        # bf16 expert weights echoed back, expanded_idx) instead of finalized
        # rows, so no destination exists to write into.
        out_buf = getattr(w, "_situ_output_buffer", None) if do_finalize else None
        if not (
            out_buf is not None
            and out_buf.shape == output_shape
            and out_buf.is_contiguous()
        ):
            out_buf = None
        return _flashinfer_trtllm_nvfp4_moe_apply(
            x,
            w,
            router_logits,
            topk_weights,
            topk_ids,
            do_finalize,
            enable_pdl,
            routed=True,
            activation_type=ActivationType.Situ,
            output=out_buf,
        )

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_nvfp4_relu2_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_nvfp4_relu2_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"nvfp4"}),
            "activation": frozenset({"relu2"}),
            "routing_mode": frozenset({"kernel_routing"}),
            "supports_deferred_finalize": frozenset({True}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({TRTLLM_NVFP4_RELU2_ISPP_ALIGNMENT}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_nvfp4_relu2_moe_apply(
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
        return _flashinfer_trtllm_nvfp4_moe_apply(
            x,
            w,
            router_logits,
            topk_weights,
            topk_ids,
            do_finalize,
            enable_pdl,
            routed=False,
            activation_type=ActivationType.Relu2,
        )

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_nvfp4_relu2_routed_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_nvfp4_relu2_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"nvfp4"}),
            "activation": frozenset({"relu2"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({True}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({TRTLLM_NVFP4_RELU2_ISPP_ALIGNMENT}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        # One below in-kernel routing: this wins only for plans with routing_mode="precomputed_topk".
        priority=Priority.PERFORMANT + 3,
    )
    def flashinfer_trtllm_nvfp4_relu2_routed_moe_apply(
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
        if topk_weights is None or topk_ids is None:
            raise ValueError("precomputed_topk plan requires topk_weights and topk_ids")
        return _flashinfer_trtllm_nvfp4_moe_apply(
            x,
            w,
            router_logits,
            topk_weights,
            topk_ids,
            do_finalize,
            enable_pdl,
            routed=True,
            activation_type=ActivationType.Relu2,
        )


if not platform.is_nvidia:

    def prepare_nvfp4_routing(
        w: torch.nn.Module,
        x_quantized: tuple[torch.Tensor, torch.Tensor],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        activation_type: int | None,
        enable_pdl: bool,
    ) -> PreparedNvfp4Routing:
        """Split routing exists only for the NVIDIA flashinfer kernel."""
        raise NotImplementedError("split routing needs the NVIDIA flashinfer NVFP4 MoE")
