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

"""MXFP4 SiTU routed experts on FlashInfer's Cake dense grouped-GEMM kernels.

The kernels take MXFP8 activations and MXFP4 weights (W4A8), the format the
TRT-LLM-Gen SiTU kernels use, and are faster on prefill-sized batches. The
preprocessor stores the experts once in the kernels' expert-major layout, so
the peer-weight store streams them like the TRT-LLM layout.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.tuning import get_autotune_max_num_tokens, is_warming_up
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()

_STREAMED = ("cake_w1", "cake_w1_sf", "cake_w2", "cake_w2_sf")


if platform.is_nvidia:
    from flashinfer import mxfp8_quantize
    from tokenspeed_kernel.ops.moe.flashinfer import _positive_situ_value
    from tokenspeed_kernel.thirdparty.flashinfer import cake_mxfp4_moe

    def flashinfer_cake_mxfp4_situ_moe_weights(plan: dict, w: torch.nn.Module):
        """Convert MXFP4 checkpoint experts to the Cake dense layout.

        Args:
            plan: The selected MoE plan.
            w: Expert module holding ``w13_*`` / ``w2_*`` MXFP4 parameters with
                ``[gate | up]`` W13 rows.

        Returns:
            The expert-major processed weights for the peer-weight store.
        """
        from tokenspeed_kernel.ops.communication.peer_weights import (
            ExpertWeightShard,
            PreparedExpertWeights,
        )

        layout = getattr(w, "w13_input_layout", "concatenated")
        if layout != "concatenated":
            raise ValueError(
                f"Cake MXFP4 SiTU MoE needs [gate | up] W13 rows, got {layout!r}"
            )
        dense = cake_mxfp4_moe.dense_weights(
            w.w13_weight.data,
            w.w13_weight_scale.data,
            w.w2_weight.data,
            w.w2_weight_scale.data,
        )
        for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
            delattr(w, name)
        for name, tensor in dense.items():
            setattr(w, f"cake_{name}", torch.nn.Parameter(tensor, requires_grad=False))
        w.cake_situ_beta = _positive_situ_value(w, "activation_situ_beta", "situ_beta")
        w.cake_situ_linear_beta = _positive_situ_value(
            w, "activation_situ_linear_beta", "situ_linear_beta"
        )
        parameters = dict(w.named_parameters(recurse=False))
        return PreparedExpertWeights(
            layout="flashinfer-cake-mxfp4-situ-dense-v1",
            shards=tuple(
                ExpertWeightShard(name, parameters[name], False) for name in _STREAMED
            ),
        )

    def _geometry(w: torch.nn.Module, device: torch.device, top_k: int) -> dict:
        local_experts, w13_rows, half_hidden = w.cake_w1.shape
        return dict(
            device_index=device.index,
            num_experts=w.num_experts,
            top_k=top_k,
            hidden=2 * half_hidden,
            intermediate=w13_rows // 2,
            num_local_experts=local_experts,
            local_expert_offset=getattr(w, "ep_rank", 0) * local_experts,
        )

    def _routed(
        w: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        output: torch.Tensor | None,
        enable_pdl: bool,
    ) -> torch.Tensor:
        tokens, hidden = x.shape
        top_k = topk_ids.shape[1]
        geometry = _geometry(w, x.device, top_k)
        local_experts = geometry["num_local_experts"]
        local_expert_offset = geometry["local_expert_offset"]
        padded = cake_mxfp4_moe.dense_tokens(tokens, **geometry)
        topk_ids = topk_ids.to(torch.int32)
        if padded != tokens:
            # The dense chain skips some token counts. Padding rows are routed
            # to this rank's first experts and their output is dropped.
            if local_experts < top_k:
                raise ValueError(
                    f"padding needs {top_k} local experts, this rank has {local_experts}"
                )
            extra = padded - tokens
            x = F.pad(x, (0, 0, 0, extra))
            topk_weights = F.pad(topk_weights, (0, 0, 0, extra))
            pad_ids = local_expert_offset + torch.arange(
                top_k, dtype=torch.int32, device=x.device
            )
            topk_ids = torch.cat([topk_ids, pad_ids.expand(extra, top_k)])
            output = None
        x_fp8, x_scale = mxfp8_quantize(
            x, False, enable_pdl=enable_pdl, alignment=hidden, backend="cute-dsl"
        )
        if output is None:
            output = torch.empty(padded, hidden, dtype=torch.bfloat16, device=x.device)
        cake_mxfp4_moe.situ_routed_moe(
            x_fp8.view(torch.float8_e4m3fn).reshape(padded, hidden),
            x_scale.view(torch.uint8).reshape(padded, -1),
            topk_ids.contiguous(),
            topk_weights.contiguous(),
            {name.removeprefix("cake_"): getattr(w, name) for name in _STREAMED},
            beta=x.new_full((local_experts,), w.cake_situ_beta, dtype=torch.float32),
            linear_beta=x.new_full(
                (local_experts,), w.cake_situ_linear_beta, dtype=torch.float32
            ),
            num_experts=w.num_experts,
            local_expert_offset=local_expert_offset,
            output=output,
        )
        return output[:tokens]

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_cake_mxfp4_situ_routed_moe_apply",
        solution="flashinfer_cake",
        weight_preprocessor=flashinfer_cake_mxfp4_situ_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"situ"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({128}),
            "internal_activation_dtype": frozenset({"fp8"}),
            "supports_bias": frozenset({False}),
            "supports_peer_weights": frozenset({True}),
        },
        priority=Priority.SPECIALIZED - 1,
    )
    def flashinfer_cake_mxfp4_situ_routed_moe_apply(
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
        """Routed MXFP4 SiTU experts on precomputed top-k, finalized in the kernel.

        Args:
            plan: The selected MoE plan.
            x: BF16 ``[tokens, hidden]`` activations.
            w: Expert module prepared by the preprocessor.
            router_logits: Unused; routing is precomputed.
            topk_weights: ``[tokens, top_k]`` FP32 or BF16 route weights.
            topk_ids: ``[tokens, top_k]`` global expert ids.
            num_tokens_global: Unused.
            max_num_tokens_per_gpu: Unused.
            do_finalize: Must be True.
            enable_pdl: Whether the input quantization uses programmatic
                dependent launch; the expert kernels are built without it.

        Returns:
            The BF16 ``[tokens, hidden]`` weighted sum over the experts.
        """
        if topk_weights is None or topk_ids is None:
            raise ValueError("precomputed_topk plan requires topk_weights and topk_ids")
        if not do_finalize:
            raise ValueError("Cake MXFP4 SiTU MoE always finalizes")
        if x.dtype != torch.bfloat16:
            raise TypeError("Cake MXFP4 SiTU MoE requires bf16 input")
        hidden = w.cake_w2.shape[1]
        if x.shape[1] != hidden:
            raise ValueError(f"expected hidden size {hidden}, got {x.shape[1]}")
        tokens = x.shape[0]
        # Idle attention-DP ranks route no tokens, which FlashInfer's routing rejects.
        if tokens == 0:
            return x.new_empty((0, hidden))
        if is_warming_up():
            # Load the kernels of every batch size up to the engine's token
            # ceiling during the startup forward.
            top_k = topk_ids.shape[1]
            geometry = _geometry(w, x.device, top_k)
            first = geometry["local_expert_offset"]
            for count in cake_mxfp4_moe.warmup_token_counts(
                max(tokens, get_autotune_max_num_tokens()), **geometry
            ):
                _routed(
                    w,
                    x.new_zeros((count, hidden)),
                    topk_weights.new_zeros((count, top_k)),
                    (first + torch.arange(top_k, device=x.device)).expand(count, top_k),
                    None,
                    enable_pdl,
                )
        output = getattr(w, "_situ_output_buffer", None)
        if output is not None and (
            output.shape != (tokens, hidden) or not output.is_contiguous()
        ):
            output = None
        return _routed(w, x, topk_weights, topk_ids, output, enable_pdl)
