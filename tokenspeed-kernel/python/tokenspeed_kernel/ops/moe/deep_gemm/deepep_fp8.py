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

"""Block-scale FP8 MoE over DeepEP all-to-all with DeepGEMM grouped GEMMs.

Unlike the nvfp4 cutedsl DeepEP path, the activations stay FP8 end to end:
DeepEP casts the tokens to FP8 on the wire (halving dispatch traffic versus
bf16) and hands back the matching 1x128 block scales, which feed DeepGEMM's
grouped GEMMs directly. Only the combine leg travels in bf16.

Both DeepEP modes are wired, because either one alone only suits one batch
shape:

* Low latency, for decode-shaped batches. Dispatch returns a padded
  ``[num_local_experts, capacity, hidden]`` tensor plus per-expert counts, which
  masked grouped GEMMs consume in place. The capacity is a preallocated buffer
  size, so an extend-sized batch does not fit.
* Normal, for extend-shaped batches. Dispatch returns rank-ordered tokens with
  the top-k slots that landed here, which are permuted into per-expert
  contiguous row blocks for contiguous grouped GEMMs. There is no capacity
  ceiling, and the routing weights are applied before combine because the normal
  combine leg reduces without weights.

Which one runs is the caller's ``low_latency`` decision (see ``moe_apply``); it
must be identical on every rank of the EP group because the two modes are
different collectives.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace

import torch
from tokenspeed_kernel.ops.communication.deep_ep import DeepEPDispatcher, DeepEPMode
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()
logger = logging.getLogger(__name__)
_warned_about_requantization = False

try:
    from tokenspeed_kernel.thirdparty.deep_gemm import (
        get_mn_major_tma_aligned_tensor,
        get_pdl,
        m_grouped_fp8_gemm_nt_contiguous,
        m_grouped_fp8_gemm_nt_masked,
        set_pdl,
        transform_sf_into_required_layout,
    )
except ImportError:  # pragma: no cover - DeepGEMM is an optional dependency
    get_mn_major_tma_aligned_tensor = None
    get_pdl = None
    m_grouped_fp8_gemm_nt_contiguous = None
    m_grouped_fp8_gemm_nt_masked = None
    set_pdl = None
    transform_sf_into_required_layout = None


if platform.is_nvidia and m_grouped_fp8_gemm_nt_masked is not None:
    from tokenspeed_kernel.ops.activation.triton import (
        fused_swiglu_fp8_ue8m0,
        fused_swiglu_fp8_ue8m0_masked_packed,
    )
    from tokenspeed_kernel.ops.moe.deep_gemm.ue8m0 import (
        deep_gemm_requires_ue8m0,
        is_ue8m0,
        requantize_to_ue8m0_,
    )
    from tokenspeed_kernel.ops.moe.triton.deepep_permute import (
        deepep_gather,
        deepep_scatter,
    )
    from tokenspeed_kernel.thirdparty.cuda import silu_and_mul_fuse_block_quant

    _FP8_BLOCK = 128
    _UE8M0_WEIGHT_RECIPE = (1, _FP8_BLOCK, _FP8_BLOCK)
    _UE8M0_PACKED_RECIPE = (1, 1, _FP8_BLOCK)

    def _configure_deep_gemm_pdl(enable_pdl: bool) -> None:
        """Keep DeepGEMM's process-wide launch mode aligned with this MoE call."""
        if get_pdl() != enable_pdl:
            set_pdl(enable_pdl)

    def _prepare_routing_tensors(
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Canonicalize DeepEP routing metadata once for both collective legs."""
        topk_ids = topk_ids.to(dtype=torch.int64, memory_format=torch.contiguous_format)
        topk_weights = topk_weights.to(
            dtype=torch.float32, memory_format=torch.contiguous_format
        )
        return topk_weights, topk_ids

    def deep_gemm_deepep_fp8_moe_weights(plan: dict, w: torch.nn.Module):
        # DeepGEMM's ``nt`` grouped GEMM consumes B as [E, N, K] with block
        # scales [E, ceil(N/128), ceil(K/128)] and computes ``a @ b.T``. That is
        # exactly what the shared MoE checkpoint loader already produces
        # (w13 = [w1(gate) | w3(up)] concatenated, w2 = [hidden, ispp]), and the
        # gate-first half order is what silu_and_mul consumes, so unlike the
        # cutlass/TRT-LLM FP8 paths no gate/up swap is required here.
        #
        # Clamp the inverse scales so experts whose shards were never populated
        # cannot dequantize into zeros and produce NaNs downstream.
        w.w13_weight_scale_inv.data.clamp_(min=1e-10)
        w.w2_weight_scale_inv.data.clamp_(min=1e-10)

        if deep_gemm_requires_ue8m0():
            # This device's DeepGEMM reads block scales as UE8M0, so a
            # checkpoint quantized with arbitrary scales must be converted or
            # every GEMM silently returns garbage. Doing it here means it happens
            # once, at load, on the already-sharded local experts.
            already_ue8m0 = is_ue8m0(w.w13_weight_scale_inv.data) and is_ue8m0(
                w.w2_weight_scale_inv.data
            )
            if not already_ue8m0:
                global _warned_about_requantization
                if not _warned_about_requantization:
                    _warned_about_requantization = True
                    logger.warning(
                        "Requantizing FP8 MoE weights to UE8M0 (power-of-two) "
                        "block scales: this device's DeepGEMM only implements "
                        "the 1d1d kernel, which reads scales as UE8M0, and this "
                        "checkpoint's scales are not. Expect a small accuracy "
                        "change versus a natively UE8M0-quantized checkpoint. "
                        "Set TOKENSPEED_DISABLE_DEEP_GEMM_UE8M0=1 to skip."
                    )
                requantize_to_ue8m0_(
                    w.w13_weight.data,
                    w.w13_weight_scale_inv.data,
                    (_FP8_BLOCK, _FP8_BLOCK),
                )
                requantize_to_ue8m0_(
                    w.w2_weight.data,
                    w.w2_weight_scale_inv.data,
                    (_FP8_BLOCK, _FP8_BLOCK),
                )

            # FP32 checkpoint scales are block-granular in N. DeepGEMM expands
            # them to one scale row per N and packs four UE8M0 values per int32
            # on every GEMM unless the required B layout is prepared up front.
            # Replace the now-loaded scale parameters so both normal and
            # low-latency paths reuse the packed tensors for every forward.
            w.w13_weight_scale_inv.data = transform_sf_into_required_layout(
                sf=w.w13_weight_scale_inv.data,
                mn=w.w13_weight.shape[1],
                k=w.w13_weight.shape[2],
                recipe=_UE8M0_WEIGHT_RECIPE,
                num_groups=w.w13_weight.shape[0],
                is_sfa=False,
            )
            w.w2_weight_scale_inv.data = transform_sf_into_required_layout(
                sf=w.w2_weight_scale_inv.data,
                mn=w.w2_weight.shape[1],
                k=w.w2_weight.shape[2],
                recipe=_UE8M0_WEIGHT_RECIPE,
                num_groups=w.w2_weight.shape[0],
                is_sfa=False,
            )
        return None

    def _get_dispatcher(
        plan: dict,
        w: torch.nn.Module,
        x: torch.Tensor,
    ) -> DeepEPDispatcher:
        """Build (once per plan) the dispatcher owning this layer's DeepEP legs.

        Sizing comes from the plan rather than the live batch: the DeepEP buffer
        is allocated on first use and reused for every later forward, so a
        low-latency capacity taken from whichever batch happened to arrive first
        would leave decode either over-provisioned or too small.
        """
        dispatcher = plan.get("_deepep_dispatcher")
        if dispatcher is not None:
            return dispatcher

        group = plan.get("deepep_group")
        if group is None:
            raise ValueError("DeepEP MoE plan is missing deepep_group")
        deepep_mode = DeepEPMode(plan.get("deepep_mode") or DeepEPMode.auto.value)
        capacity = plan.get("deepep_low_latency_max_num_tokens_per_gpu")
        if deepep_mode.enable_low_latency() and not capacity:
            raise ValueError(
                f"DeepEP plan with mode {deepep_mode.value} is missing "
                "deepep_low_latency_max_num_tokens_per_gpu"
            )
        config = SimpleNamespace(
            top_k=getattr(w, "top_k"),
            num_experts=getattr(w, "num_experts"),
            low_latency_max_num_tokens_per_gpu=capacity,
            hidden_size=x.shape[1],
            world_size=getattr(w, "ep_size", group.size()),
            group=group,
            params_dtype=torch.bfloat16,
        )
        dispatcher = DeepEPDispatcher(
            config,
            deepep_mode=deepep_mode,
            async_finish=False,
            return_recv_hook=True,
            # FP8 on the wire: dispatch returns (fp8 tokens, 1x128 block
            # scales) rather than a single bf16 tensor.
            use_fp8=True,
            ue8m0_scales=deep_gemm_requires_ue8m0(),
        )
        plan["_deepep_dispatcher"] = dispatcher
        return dispatcher

    def _expected_m(
        x: torch.Tensor,
        w: torch.nn.Module,
        top_k: int,
        num_tokens_global: int | None,
        recv_m: int,
    ) -> int:
        """Average tokens per local expert, used by DeepGEMM for tile scheduling.

        This is only a scheduling hint (``masked_m`` bounds the real work), so a
        host-side estimate is enough and avoids a device sync on ``masked_m``.
        """
        num_experts = int(getattr(w, "num_experts", 0) or 0)
        if num_experts <= 0:
            return recv_m
        ep_size = int(getattr(w, "ep_size", 1) or 1)
        total_tokens = int(num_tokens_global or x.shape[0] * ep_size)
        expected = (total_tokens * int(top_k) + num_experts - 1) // num_experts
        return max(1, min(expected, recv_m))

    def _apply_low_latency(
        dispatcher: DeepEPDispatcher,
        x: torch.Tensor,
        w: torch.nn.Module,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_tokens_global: int | None,
        enable_pdl: bool,
        overlap_fn: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """Decode-shaped path: padded per-expert buffers + masked grouped GEMMs."""
        _configure_deep_gemm_pdl(enable_pdl)
        dispatcher.dispatch_a(x, topk_ids, topk_weights, low_latency=True)
        # dispatch_a only launched the send phase, so the RDMA transfer is in
        # flight on the NIC: any work queued here runs while it lands instead of
        # leaving the GPU spinning in the recv phase.
        if overlap_fn is not None:
            overlap_fn()
        recv_hidden, _, _, _, _, _, masked_m = dispatcher.dispatch_b()
        # ``use_fp8=True`` makes DeepEP return the tokens already quantized.
        recv_x, recv_scales = recv_hidden

        num_local_experts, recv_m, _ = recv_x.shape
        expected_m = _expected_m(x, w, topk_ids.shape[1], num_tokens_global, recv_m)

        # GEMM1: [E, recv_m, hidden] @ [E, 2*ispp, hidden]^T -> [E, recv_m, 2*ispp]
        gateup = torch.empty(
            (num_local_experts, recv_m, w.w13_weight.shape[1]),
            dtype=torch.bfloat16,
            device=x.device,
        )
        requires_ue8m0 = deep_gemm_requires_ue8m0()
        recv_gemm_scales = (
            recv_scales
            if requires_ue8m0
            else get_mn_major_tma_aligned_tensor(recv_scales)
        )
        m_grouped_fp8_gemm_nt_masked(
            (recv_x, recv_gemm_scales),
            (w.w13_weight, w.w13_weight_scale_inv),
            gateup,
            masked_m,
            expected_m,
            recipe=_UE8M0_PACKED_RECIPE if requires_ue8m0 else None,
        )

        # Fused SiLU(gate)*up followed by a 1x128 block FP8 quantize. The EP
        # variant honors ``masked_m`` so padded rows are skipped.
        ispp = w.w2_weight.shape[-1]
        if requires_ue8m0:
            # Produce DeepGEMM's packed MN-major scales in the activation
            # kernel itself. Masked GEMM never reads padded rows, so neither
            # output needs a separate zero-fill pass.
            down_in, down_scales = fused_swiglu_fp8_ue8m0_masked_packed(
                gateup,
                masked_m,
                expected_m=expected_m,
                enable_pdl=enable_pdl,
            )
        else:
            down_in = torch.empty(
                (num_local_experts, recv_m, ispp),
                dtype=torch.float8_e4m3fn,
                device=x.device,
            )
            # The CUDA fallback does not emit packed UE8M0 and retains its
            # historical zero-padding contract for masked rows.
            down_scales = torch.zeros(
                (num_local_experts, ispp // _FP8_BLOCK, recv_m),
                dtype=torch.float32,
                device=x.device,
            ).permute(0, 2, 1)
            silu_and_mul_fuse_block_quant(
                gateup,
                down_scales,
                down_in,
                enable_pdl=enable_pdl,
                num_tokens_per_expert=masked_m,
                num_tokens_hint=recv_m,
                num_experts=num_local_experts,
            )

        # GEMM2: [E, recv_m, ispp] @ [E, hidden, ispp]^T -> [E, recv_m, hidden]
        out = torch.empty(
            (num_local_experts, recv_m, x.shape[1]),
            dtype=torch.bfloat16,
            device=x.device,
        )
        m_grouped_fp8_gemm_nt_masked(
            (
                down_in,
                (
                    down_scales
                    if requires_ue8m0
                    else get_mn_major_tma_aligned_tensor(down_scales)
                ),
            ),
            (w.w2_weight, w.w2_weight_scale_inv),
            out,
            masked_m,
            expected_m,
            recipe=_UE8M0_PACKED_RECIPE if requires_ue8m0 else None,
        )

        # Combine travels in bf16 and applies the routing weights.
        dispatcher.combine_a(out, topk_ids, topk_weights, low_latency=True)
        return dispatcher.combine_b()

    def _apply_normal(
        dispatcher: DeepEPDispatcher,
        x: torch.Tensor,
        w: torch.nn.Module,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        enable_pdl: bool,
        overlap_fn: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """Extend-shaped path: permuted expert blocks + contiguous grouped GEMMs."""
        _configure_deep_gemm_pdl(enable_pdl)
        dispatcher.dispatch_a(x, topk_ids, topk_weights, low_latency=False)
        # Normal-mode dispatch finishes asynchronously, so work queued here
        # overlaps the transfer rather than waiting behind it.
        if overlap_fn is not None:
            overlap_fn()
        (
            recv_hidden,
            recv_topk_ids,
            recv_topk_weights,
            _,
            num_recv_tokens_per_expert,
            _,
            _,
        ) = dispatcher.dispatch_b()
        recv_x, recv_scales = recv_hidden

        # One row per (received token, surviving top-k slot), grouped by expert.
        requires_ue8m0 = deep_gemm_requires_ue8m0()
        gemm_x, gemm_scales, m_indices, dest_index = deepep_scatter(
            recv_x,
            recv_scales,
            recv_topk_ids,
            num_recv_tokens_per_expert,
            expert_alignment=_FP8_BLOCK,
            pack_ue8m0_scales=requires_ue8m0,
        )
        hidden_size = x.shape[1]
        total_rows = gemm_x.shape[0]
        combine_in = torch.zeros(
            (recv_x.shape[0], hidden_size),
            dtype=torch.bfloat16,
            device=x.device,
        )

        if total_rows:
            # GEMM1: [total_rows, hidden] @ [E, 2*ispp, hidden]^T
            gateup = torch.empty(
                (total_rows, w.w13_weight.shape[1]),
                dtype=torch.bfloat16,
                device=x.device,
            )
            gemm1_scales = (
                gemm_scales
                if requires_ue8m0
                else get_mn_major_tma_aligned_tensor(gemm_scales)
            )
            m_grouped_fp8_gemm_nt_contiguous(
                (gemm_x, gemm1_scales),
                (w.w13_weight, w.w13_weight_scale_inv),
                gateup,
                m_indices,
                recipe=_UE8M0_PACKED_RECIPE if requires_ue8m0 else None,
            )

            # Fused SiLU(gate)*up + FP8 block quantize. Padding rows of an expert
            # block are quantized like any other row; they are simply never
            # gathered back.
            ispp = w.w2_weight.shape[-1]
            if requires_ue8m0:
                # Power-of-two scales, packed in the column-major TMA layout the
                # grouped GEMM consumes directly (no realignment needed).
                down_in, down_scales = fused_swiglu_fp8_ue8m0(
                    gateup, enable_pdl=enable_pdl
                )
                gemm2_scales = down_scales
            else:
                down_in = torch.empty(
                    (total_rows, ispp),
                    dtype=torch.float8_e4m3fn,
                    device=x.device,
                )
                # DeepGEMM wants mn-major scales, hence the [blocks, M]
                # allocation viewed as [M, blocks].
                down_scales = torch.zeros(
                    (ispp // _FP8_BLOCK, total_rows),
                    dtype=torch.float32,
                    device=x.device,
                ).permute(1, 0)
                silu_and_mul_fuse_block_quant(
                    gateup,
                    down_scales,
                    down_in,
                    enable_pdl=enable_pdl,
                )
                gemm2_scales = get_mn_major_tma_aligned_tensor(down_scales)

            # GEMM2: [total_rows, ispp] @ [E, hidden, ispp]^T
            expert_out = torch.empty(
                (total_rows, hidden_size),
                dtype=torch.bfloat16,
                device=x.device,
            )
            m_grouped_fp8_gemm_nt_contiguous(
                (down_in, gemm2_scales),
                (w.w2_weight, w.w2_weight_scale_inv),
                expert_out,
                m_indices,
                recipe=_UE8M0_PACKED_RECIPE if requires_ue8m0 else None,
            )

            # The normal combine leg reduces without weights, so fold the
            # routing weights in here.
            deepep_gather(
                expert_out,
                recv_topk_ids,
                recv_topk_weights,
                dest_index,
                out=combine_in,
            )

        dispatcher.combine_a(
            combine_in, recv_topk_ids, recv_topk_weights, low_latency=False
        )
        return dispatcher.combine_b()

    @register_kernel(
        "moe",
        "apply",
        name="deep_gemm_deepep_fp8_moe_apply",
        solution="deep_gemm",
        weight_preprocessor=deep_gemm_deepep_fp8_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(9, 0),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"fp8"}),
            # The fused activation computes SiLU(gate)*up, i.e. gated SiLU ==
            # SwiGLU; accept both names so DeepSeek-V4 (activation="swiglu")
            # selects this kernel too.
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({True}),
            "deepep_modes": frozenset({"normal", "low_latency"}),
            # DeepGEMM tiles both GEMMs on 128-element K/N blocks, so the
            # per-partition intermediate size must be a multiple of the block.
            "ispp_alignment": frozenset({_FP8_BLOCK}),
            "internal_activation_dtype": frozenset({"input"}),
            "fp8_scale_block_shape": frozenset({(_FP8_BLOCK, _FP8_BLOCK)}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def deep_gemm_deepep_fp8_moe_apply(
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
        low_latency: bool | None = None,
        overlap_fn: Callable[[], None] | None = None,
    ):
        """Run one FP8 MoE layer over DeepEP.

        Args:
            plan: Execution plan from ``moe_plan``; owns the DeepEP mode, the
                low-latency capacity, and the lazily built dispatcher.
            x: ``[tokens, hidden]`` BF16 local hidden states.
            w: Module holding the processed FP8 expert weights.
            router_logits: ``[tokens, num_experts]`` logits, used only when the
                caller passes no precomputed top-k.
            topk_weights: ``[tokens, top_k]`` routing weights.
            topk_ids: ``[tokens, top_k]`` selected (global) expert ids.
            num_tokens_global: Global token count, a tile-scheduling hint for the
                masked GEMMs.
            max_num_tokens_per_gpu: Unused; the low-latency capacity is pinned by
                the plan so the DeepEP buffer does not depend on batch order.
            do_finalize: Must be True; this kernel cannot defer finalize.
            enable_pdl: Enable PDL for both DeepGEMM launches and the fused
                activation joining them.
            low_latency: Which DeepEP mode to run when the plan mode is ``auto``.
                Every rank of the EP group must pass the same value.
            overlap_fn: Optional work to queue inside the dispatch window, i.e.
                after the tokens are sent but before they are awaited. Must not
                read the dispatch result or write ``x``; anything else (a shared
                expert on the same input) runs while the transfer lands.

        Returns:
            ``[tokens, hidden]`` bf16 combined MoE output.
        """
        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        dispatcher = _get_dispatcher(plan, w, x)
        # The router emits int32 ids while DeepEP's C++ API needs int64, and both
        # the dispatch and combine legs convert on entry. Converting once here
        # turns those into no-ops instead of two cast kernels per layer.
        topk_weights, topk_ids = _prepare_routing_tensors(topk_weights, topk_ids)
        if dispatcher.deepep_mode.resolve(low_latency) == DeepEPMode.normal:
            return _apply_normal(
                dispatcher,
                x,
                w,
                topk_weights,
                topk_ids,
                enable_pdl,
                overlap_fn=overlap_fn,
            )
        return _apply_low_latency(
            dispatcher,
            x,
            w,
            topk_weights,
            topk_ids,
            num_tokens_global,
            enable_pdl,
            overlap_fn=overlap_fn,
        )
