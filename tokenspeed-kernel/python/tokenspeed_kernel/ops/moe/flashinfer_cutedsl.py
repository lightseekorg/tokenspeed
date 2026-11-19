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

"""FlashInfer CuTe DSL NVFP4 routed MoE with the SiTU activation (precomputed top-k).

The preprocessor turns NVFP4 checkpoint experts (``[gate | up]`` W13 rows,
linear E4M3 block scales) into the CuTe DSL layout: W13 rows become
``[up | gate]`` interleaved in 64-row groups and every block-scale matrix is
swizzled. The kernels read the scales through an MMA-layout view of that
expert-major storage, so the peer-weight store can stream experts as it does
the TRT-LLM layout.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()

_SF_VEC = 16
_W13_ROW_GROUP = 64
# One fixed tactic for both GEMMs (256-row routing tiles, 2-CTA clusters), so
# each kernel compiles once, during the startup forward, for any row count.
# It is in FlashInfer's Blackwell format, which builds the SM100 kernels; SM107
# runs separate kernels that take Rubin-format tactics.
_TACTIC = (256, ((256, 256), (2, 1), False), ((256, 256), (2, 1), False))

_STREAMED = (
    "cutedsl_w1_weight",
    "cutedsl_w1_weight_sf",
    "cutedsl_w2_weight",
    "cutedsl_w2_weight_sf",
)
_REPLICATED = ("cutedsl_w1_alpha", "cutedsl_w2_alpha")


if platform.is_nvidia:
    from tokenspeed_kernel.ops.moe.flashinfer import _positive_situ_value

    def _w13_to_cute_rows(t: torch.Tensor) -> torch.Tensor:
        """``[E, 2I, X]`` rows ``[gate | up]`` -> ``[up | gate]`` interleaved by 64 rows."""
        experts, rows, cols = t.shape
        halves = t.view(experts, 2, rows // (2 * _W13_ROW_GROUP), _W13_ROW_GROUP, cols)
        return halves.flip(1).transpose(1, 2).reshape(experts, rows, cols).contiguous()

    def _swizzle_block_scales(linear: torch.Tensor) -> torch.Tensor:
        """Per-expert 128x4 swizzle of ``[E, rows, cols]`` E4M3 block scales (bytes)."""
        from flashinfer import nvfp4_block_scale_interleave

        experts, rows, cols = linear.shape
        return torch.stack(
            [
                nvfp4_block_scale_interleave(linear[e].contiguous()).view(rows, cols)
                for e in range(experts)
            ]
        )

    def flashinfer_cutedsl_nvfp4_situ_moe_weights(plan: dict, w: torch.nn.Module):
        """Convert NVFP4 checkpoint experts to the CuTe DSL layout.

        Args:
            plan: The selected MoE plan.
            w: Expert module holding ``w13_*`` / ``w2_*`` NVFP4 parameters.

        Returns:
            The expert-major processed weights for the peer-weight store.
        """
        from tokenspeed_kernel.ops.communication.peer_weights import (
            ExpertWeightShard,
            PreparedExpertWeights,
        )

        gate_ws2, up_ws2 = w.w13_weight_scale_2.unbind(1)
        if not torch.equal(gate_ws2, up_ws2):
            raise RuntimeError(
                "NVFP4 SiTU requires equal gate/up weight_scale_2 per expert."
            )
        w2_input_scale = w.w2_input_scale.to(torch.float32)
        if not torch.all(w2_input_scale == w2_input_scale[0]):
            raise RuntimeError(
                "CuTe DSL NVFP4 MoE quantizes the GEMM2 input with one global "
                "scale; the checkpoint has per-expert values."
            )
        w13_input_scale = w.w13_input_scale.to(torch.float32)

        w1 = _w13_to_cute_rows(w.w13_weight.data)
        w1_sf = _swizzle_block_scales(
            _w13_to_cute_rows(w.w13_weight_scale.data.view(torch.uint8))
        )
        w2 = w.w2_weight.data.contiguous()
        w2_sf = _swizzle_block_scales(w.w2_weight_scale.data.view(torch.uint8))

        def param(t: torch.Tensor) -> torch.nn.Parameter:
            return torch.nn.Parameter(t, requires_grad=False)

        w.cutedsl_w1_weight = param(w1)
        w.cutedsl_w1_weight_sf = param(w1_sf)
        w.cutedsl_w2_weight = param(w2)
        w.cutedsl_w2_weight_sf = param(w2_sf)
        w.cutedsl_w1_alpha = param((w13_input_scale * gate_ws2).contiguous())
        w.cutedsl_w2_alpha = param(
            (w2_input_scale * w.w2_weight_scale_2.to(torch.float32)).contiguous()
        )
        w.cutedsl_fc2_input_scale = param((1.0 / w2_input_scale[:1]).contiguous())
        w.w13_input_scale_quant = param((1.0 / w13_input_scale).contiguous())
        w.cutedsl_situ_beta = _positive_situ_value(
            w, "activation_situ_beta", "situ_beta"
        )
        w.cutedsl_situ_linear_beta = _positive_situ_value(
            w, "activation_situ_linear_beta", "situ_linear_beta"
        )

        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "w13_input_scale",
            "w2_input_scale",
        ):
            delattr(w, name)

        parameters = dict(w.named_parameters(recurse=False))
        return PreparedExpertWeights(
            layout="flashinfer-cutedsl-nvfp4-situ-v1",
            shards=tuple(
                ExpertWeightShard(name, parameters[name], name in _REPLICATED)
                for name in (*_STREAMED, *_REPLICATED)
            ),
        )

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_cutedsl_nvfp4_situ_routed_moe_apply",
        solution="flashinfer_cutedsl",
        weight_preprocessor=flashinfer_cutedsl_nvfp4_situ_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.uint8}),
        traits={
            "weight_dtype": frozenset({"nvfp4"}),
            "activation": frozenset({"situ"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({128}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED - 1,
    )
    def flashinfer_cutedsl_nvfp4_situ_routed_moe_apply(
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
    ) -> torch.Tensor:
        """Routed NVFP4 SiTU experts on precomputed top-k, finalized in the kernel.

        Args:
            plan: The selected MoE plan.
            x: BF16 ``[tokens, hidden]`` activations, or an NVFP4 pair of packed
                ``[tokens, hidden / 2]`` codes and linear ``[tokens, hidden / 16]``
                block scales quantized with ``w.w13_input_scale_quant``.
            w: Expert module prepared by the preprocessor.
            router_logits: Unused; routing is precomputed.
            topk_weights: ``[tokens, top_k]`` FP32 route weights.
            topk_ids: ``[tokens, top_k]`` global expert ids.
            num_tokens_global: Unused.
            max_num_tokens_per_gpu: Unused.
            do_finalize: Must be True.
            enable_pdl: Whether the kernels use programmatic dependent launch.

        Returns:
            The BF16 ``[tokens, hidden]`` weighted sum over this rank's local
            experts. The caller combines contributions across EP ranks.
        """
        from flashinfer import fp4_quantize
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        from flashinfer.fused_moe.cute_dsl.fused_moe import cute_dsl_fused_moe
        from flashinfer.tllm_enums import ActivationType

        if topk_weights is None or topk_ids is None:
            raise ValueError("precomputed_topk plan requires topk_weights and topk_ids")
        if not do_finalize:
            raise ValueError("CuTe DSL NVFP4 SiTU MoE always finalizes")
        # Idle attention-DP ranks route no tokens, which FlashInfer's routing rejects.
        if topk_ids.shape[0] == 0:
            hidden = w.cutedsl_w2_weight.shape[1]
            return topk_weights.new_empty((0, hidden), dtype=torch.bfloat16)
        if isinstance(x, tuple):
            x_fp4, x_sf = x
        else:
            x_fp4, x_sf = fp4_quantize(
                x,
                w.w13_input_scale_quant,
                sf_vec_size=_SF_VEC,
                is_sf_swizzled_layout=False,
                enable_pdl=enable_pdl,
            )
        num_local_experts, w13_rows, half_hidden = w.cutedsl_w1_weight.shape
        hidden = half_hidden * 2
        w1_sf = convert_sf_to_mma_layout(
            w.cutedsl_w1_weight_sf.view(num_local_experts * w13_rows, -1),
            m=w13_rows,
            k=hidden,
            num_groups=num_local_experts,
            sf_vec_size=_SF_VEC,
        )
        w2_sf = convert_sf_to_mma_layout(
            w.cutedsl_w2_weight_sf.view(num_local_experts * hidden, -1),
            m=hidden,
            k=w13_rows // 2,
            num_groups=num_local_experts,
            sf_vec_size=_SF_VEC,
        )
        return cute_dsl_fused_moe(
            x_fp4,
            x_sf.view(torch.uint8).unsqueeze(-1),
            topk_ids.to(torch.int32),
            topk_weights,
            w.cutedsl_w1_weight,
            w1_sf,
            w.cutedsl_w1_alpha,
            w.cutedsl_fc2_input_scale,
            w.cutedsl_w2_weight,
            w2_sf,
            w.cutedsl_w2_alpha,
            num_experts=w._spec.num_experts,
            top_k=topk_ids.shape[1],
            num_local_experts=num_local_experts,
            local_expert_offset=w._spec.ep_rank * num_local_experts,
            use_fused_finalize=True,
            enable_pdl=enable_pdl,
            activation_type=ActivationType.Swiglu.value,
            situ_beta=w.cutedsl_situ_beta,
            situ_linear_beta=w.cutedsl_situ_linear_beta,
            tactic=_TACTIC,
        )
