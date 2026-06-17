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

"""Weight-only INT4 group-32 MoE using the fused FlashInfer block-scale kernel.

Serves ``compressed-tensors`` ``pack-quantized`` checkpoints whose routed
experts are INT4 with a group size of 32 and ``bfloat16`` group scales (e.g.
Kimi-K2.5 / K2.6 / K2.7). Blackwell-only, weight-only (activations stay
``bfloat16``, no activation quant); routing runs inside the kernel from raw logits.

The checkpoint stores ``weight_packed`` as ``int32`` (eight INT4 per word,
``+8`` zero-point) with per-group ``bfloat16`` scales; the kernel wants signed
``uint8`` weights (two INT4 per byte) in ``BlockMajorK`` layout with
permuted/interleaved scales. process_weights repacks them once.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.preprocessing import register_weight_preprocessor
from tokenspeed_kernel.registry import Priority, WeightPreprocessorRef, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()
next_power_of_2 = lambda value: 1 if value <= 1 else 1 << (value - 1).bit_length()

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

    _FLASHINFER_TRTLLM_MXINT4_MOE_TRAITS = {
        "weight_dtype": frozenset({"mxint4"}),
        "ispp_alignment": frozenset({256}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "activation": frozenset({"silu", "swiglu"}),
        "routing_mode": frozenset({"kernel_routing"}),
        "supports_deferred_finalize": frozenset({False}),
        "internal_activation_dtype": frozenset({"input"}),
    }

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

    @register_weight_preprocessor(
        "moe",
        name="flashinfer_trtllm_mxint4_moe_weights",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        traits=_FLASHINFER_TRTLLM_MXINT4_MOE_TRAITS,
    )
    def flashinfer_trtllm_mxint4_moe_weights(plan: dict, w: torch.nn.Module):
        num_experts = w.w13_weight_packed.shape[0]

        # Swap [W1(Gate), W3(Up)] -> [W3(Up), W1(Gate)]. The flashinfer fused
        # gated-act epilogue expects the up-proj rows first; the shared loader
        # fills the natural [gate|up] order, so swap the two w13 halves (weight +
        # group scale) here, mirroring the nvfp4 path (trtllm_nvfp4.py:73-88).
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
        weight_preprocessor=WeightPreprocessorRef(
            "flashinfer_trtllm_mxint4_moe_weights", required=True
        ),
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
        traits=_FLASHINFER_TRTLLM_MXINT4_MOE_TRAITS,
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
            tune_max_num_tokens=next_power_of_2(x.shape[0]),
        )
        return result[0]
