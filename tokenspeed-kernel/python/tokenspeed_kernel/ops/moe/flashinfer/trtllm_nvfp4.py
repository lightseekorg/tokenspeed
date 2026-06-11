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
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signature, format_signatures

platform = current_platform()
next_power_of_2 = lambda value: 1 if value <= 1 else 1 << (value - 1).bit_length()


if platform.is_nvidia:
    from flashinfer import (
        fp4_quantize,
        nvfp4_block_scale_interleave,
        trtllm_fp4_block_scale_moe,
    )
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices as maybe_get_cached_w3_w1_permute_indices,
    )
    from flashinfer.fused_moe.core import (
        get_w2_permute_indices_with_cache,
    )

    @register_kernel(
        "moe",
        "process_weights",
        name="flashinfer_trtllm_nvfp4_moe_process_weights",
        solution="flashinfer_trtllm",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=frozenset({format_signature()}),
        traits={"weight_dtype": frozenset({"nvfp4"})},
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_nvfp4_moe_process_weights(plan: dict, w: torch.nn.Module):
        _group_size = 16
        _correction_bias = getattr(w, "_correction_bias", None)
        _routing_logits_dtype = getattr(w, "_routing_logits_dtype", torch.bfloat16)

        num_experts = w.w13_weight.shape[0]
        # intermediate_size_per_partition = half of w13 rows (gate + up)
        intermediate_size = w.w13_weight.shape[1] // 2
        hidden_size = w.w13_weight.shape[2] * 2

        # Fix 1: Swap [W1(Gate), W3(Up)] -> [W3(Up), W1(Gate)].
        # The fused gated-act reorder interleaves [first_half, second_half] as
        # [row0_first, row0_second, row1_first, row1_second, ...].
        # It expects [W3(Up), W1(Gate)] so that the interleaved result pairs
        # each up-proj row with its corresponding gate-proj row correctly.
        half_w = w.w13_weight.shape[1] // 2
        w1_weight = w.w13_weight.data[:, :half_w, :].clone()
        w.w13_weight.data[:, :half_w, :] = w.w13_weight.data[:, half_w:, :]
        w.w13_weight.data[:, half_w:, :] = w1_weight
        del w1_weight

        half_s = w.w13_weight_scale.shape[1] // 2
        w1_scale = w.w13_weight_scale.data[:, :half_s, :].clone()
        w.w13_weight_scale.data[:, :half_s, :] = w.w13_weight_scale.data[:, half_s:, :]
        w.w13_weight_scale.data[:, half_s:, :] = w1_scale
        del w1_scale

        # Shuffle weights and scales using fused-kernel permute indices.
        cache: dict = {}
        epilogue_tile_m = 128

        # View as fp8 for permutation (uint8 and fp8_e4m3fn are both 1 byte)
        w13_fp4 = w.w13_weight.data.view(torch.float8_e4m3fn).reshape(
            num_experts, 2 * intermediate_size, hidden_size // 2
        )
        w13_scales = w.w13_weight_scale.data.view(torch.float8_e4m3fn).reshape(
            num_experts, 2 * intermediate_size, hidden_size // _group_size
        )
        w2_fp4 = w.w2_weight.data.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // 2
        )
        w2_scales = w.w2_weight_scale.data.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // _group_size
        )

        w13_weights_shuffled = []
        w13_scales_shuffled = []
        w2_weights_shuffled = []
        w2_scales_shuffled = []

        for idx in range(num_experts):
            # W1/W3 (gemm1) weight permutation
            perm = maybe_get_cached_w3_w1_permute_indices(
                cache, w13_fp4[idx].view(torch.uint8), epilogue_tile_m
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
            .reshape(num_experts, 2 * intermediate_size, hidden_size // _group_size),
            requires_grad=False,
        )
        w.gemm2_weights_fp4_shuffled = torch.nn.Parameter(
            torch.stack(w2_weights_shuffled), requires_grad=False
        )
        w.gemm2_scales_fp4_shuffled = torch.nn.Parameter(
            torch.stack(w2_scales_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, hidden_size, intermediate_size // _group_size),
            requires_grad=False,
        )

        # Free original weights (replaced by shuffled versions)
        del w.w13_weight
        del w.w2_weight
        del w.w13_weight_scale
        del w.w2_weight_scale

        # Compute fused-kernel scales.
        # Reduce w13_weight_scale_2: take per-expert value.
        w13_ws2 = w.w13_weight_scale_2[:, 0]
        # Input scales (max across shards) for alpha computation
        w13_input_scale = w.w13_input_scale.max().to(torch.float32)
        w2_input_scale = w.w2_input_scale.max().to(torch.float32)

        # Store input_scale_quant for runtime fp4_quantize
        w13_input_scale_quant = (1.0 / w13_input_scale).to(torch.float32)
        w2_input_scale_quant = (1.0 / w2_input_scale).to(torch.float32)

        w.w13_input_scale_quant = torch.nn.Parameter(
            w13_input_scale_quant, requires_grad=False
        )
        # Fused-kernel alphas: input_scale * weight_scale_2
        w.g1_alphas = torch.nn.Parameter(
            (w13_input_scale * w13_ws2).to(torch.float32), requires_grad=False
        )
        w.g2_alphas = torch.nn.Parameter(
            (w2_input_scale * w.w2_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )
        w.g1_scale_c = torch.nn.Parameter(
            (w2_input_scale_quant * w.g1_alphas).to(torch.float32),
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

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_nvfp4_moe_apply",
        solution="flashinfer_trtllm",
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
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"kernel_routing"}),
            "supports_deferred_finalize": frozenset({True}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
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
    ):
        _spec = getattr(w, "_spec", None)
        _correction_bias = getattr(w, "_correction_bias", None)
        _routing_logits_dtype = getattr(w, "_routing_logits_dtype", torch.bfloat16)
        _n_group = getattr(w, "_n_group", 0)
        _topk_group = getattr(w, "_topk_group", 0)
        _routed_scaling_factor = getattr(w, "_routed_scaling_factor", 1.0)
        _routing_method_type = getattr(w, "_routing_method_type", 0)

        num_tokens = x.shape[0]
        # Idle DP ranks run a dummy forward with 0 tokens. The fused
        # kernel divides by the token count on the host and SIGFPEs on
        # empty input, so skip the experts entirely.
        if num_tokens == 0:
            if do_finalize:
                return x
            return (
                x,
                x.new_empty((0, _spec.top_k), dtype=torch.bfloat16),
                # moe_finalize_fuse_shared expects a 1-D
                # [num_tokens * top_k] permute map.
                x.new_empty((0,), dtype=torch.int32),
            )

        # Quantize input to FP4 using the fused-kernel scale layout.
        hs_fp4, hs_scale = fp4_quantize(
            x,
            w.w13_input_scale_quant,
            is_sf_swizzled_layout=False,
            enable_pdl=enable_pdl,
        )

        routing_bias = _correction_bias
        routing_logits = router_logits.to(_routing_logits_dtype)

        result = trtllm_fp4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            hidden_states=hs_fp4,
            hidden_states_scale=hs_scale.view(torch.float8_e4m3fn),
            gemm1_weights=w.gemm1_weights_fp4_shuffled.data,
            gemm1_weights_scale=w.gemm1_scales_fp4_shuffled.data.view(
                torch.float8_e4m3fn
            ),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
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
            n_group=_n_group,
            topk_group=_topk_group,
            intermediate_size=w.intermediate_size_per_partition,
            local_expert_offset=_spec.ep_rank * _spec.num_local_experts,
            local_num_experts=_spec.num_local_experts,
            routed_scaling_factor=_routed_scaling_factor,
            routing_method_type=_routing_method_type,
            do_finalize=do_finalize,
            tune_max_num_tokens=next_power_of_2(num_tokens),
        )
        if do_finalize:
            return result[0]
        # Deferred: [gemm2_out, expert_weights, expanded_idx_to_permuted_idx]
        gemm2_out, expert_weights, expanded_idx = result
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
