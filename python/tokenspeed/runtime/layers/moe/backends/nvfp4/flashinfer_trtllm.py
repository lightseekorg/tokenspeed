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

"""NVFP4 MoE backend using fused FlashInfer kernels.

This backend uses the standalone flashinfer.fp4_quantize to pre-quantize
activations before calling the fused FP4 block-scale MoE kernel. This standalone
fp4_quantize has a bug (as of flashinfer 0.6.6) that causes illegal memory
access (IMA) under high-concurrency serving loads, particularly when the token
count is large after all_gather in attention-data-parallelism (attn_dp)
configs. The corruption is silent during the fp4_quantize kernel itself and
manifests as IMA in subsequent operations.

WAR: Use --moe-backend flashinfer_cutlass instead.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.moe.flashinfer import (
    ActivationType,
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
    trtllm_fp4_block_scale_moe_raw,
)
from tokenspeed_kernel.ops.quantization.flashinfer import nvfp4_block_scale_interleave
from tokenspeed_kernel.platform import current_platform
from torch import nn
from torch.nn import functional as F

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.nvfp4.weights import create_fp4_weights
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.layers.quantization import Nvfp4Config
from tokenspeed.runtime.utils import next_power_of_2


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _align_hidden_dim_for_trtllm_fp4(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    group_size: int,
    min_alignment: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    num_experts, gate_up_rows, packed_hidden = w13.shape
    hidden_size = packed_hidden * 2
    padded_hidden_size = _round_up(hidden_size, min_alignment)

    if padded_hidden_size == hidden_size:
        return w13, w13_scale, w2, w2_scale, hidden_size, hidden_size

    padded_w13 = w13.new_zeros((num_experts, gate_up_rows, padded_hidden_size // 2))
    padded_w13[:, :, :packed_hidden] = w13

    padded_w13_scale = w13_scale.new_zeros(
        (num_experts, gate_up_rows, padded_hidden_size // group_size)
    )
    padded_w13_scale[:, :, : w13_scale.shape[2]] = w13_scale

    padded_w2 = w2.new_zeros((num_experts, padded_hidden_size, w2.shape[2]))
    padded_w2[:, : w2.shape[1], :] = w2

    padded_w2_scale = w2_scale.new_zeros(
        (num_experts, padded_hidden_size, w2_scale.shape[2])
    )
    padded_w2_scale[:, : w2_scale.shape[1], :] = w2_scale

    return (
        padded_w13,
        padded_w13_scale,
        padded_w2,
        padded_w2_scale,
        hidden_size,
        padded_hidden_size,
    )


def _align_intermediate_for_trtllm_fp4(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    group_size: int,
    *,
    is_gated: bool,
    min_alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    num_experts, hidden_size, packed_intermediate = w2.shape
    intermediate_size = packed_intermediate * 2
    padded_intermediate_size = _round_up(intermediate_size, min_alignment)

    if padded_intermediate_size == intermediate_size:
        return w13, w13_scale, w2, w2_scale, intermediate_size

    w13_rows = 2 * padded_intermediate_size if is_gated else padded_intermediate_size
    padded_w13 = w13.new_zeros((num_experts, w13_rows, w13.shape[2]))
    padded_w13[:, : w13.shape[1], :] = w13

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate_size // 2))
    padded_w2[:, :, :packed_intermediate] = w2

    padded_w13_scale = w13_scale.new_zeros((num_experts, w13_rows, w13_scale.shape[2]))
    padded_w13_scale[:, : w13_scale.shape[1], :] = w13_scale

    padded_w2_scale = w2_scale.new_zeros(
        (num_experts, hidden_size, padded_intermediate_size // group_size)
    )
    padded_w2_scale[:, :, : w2_scale.shape[2]] = w2_scale

    return (
        padded_w13,
        padded_w13_scale,
        padded_w2,
        padded_w2_scale,
        padded_intermediate_size,
    )


def _get_w13_permute_indices(
    cache: dict,
    tensor: torch.Tensor,
    epilogue_tile_m: int,
    *,
    is_gated: bool,
    num_elts_per_sf: int | None = None,
) -> torch.Tensor:
    kwargs = {"is_gated_act_gemm": is_gated}
    if num_elts_per_sf is not None:
        kwargs["num_elts_per_sf"] = num_elts_per_sf
    try:
        return _maybe_get_cached_w3_w1_permute_indices(
            cache,
            tensor,
            epilogue_tile_m,
            **kwargs,
        )
    except TypeError:
        if not is_gated:
            raise RuntimeError(
                "FlashInfer TRT-LLM FP4 MoE must support "
                "is_gated_act_gemm=False for relu2 Nemotron-H experts"
            ) from None
        kwargs.pop("is_gated_act_gemm")
        return _maybe_get_cached_w3_w1_permute_indices(
            cache,
            tensor,
            epilogue_tile_m,
            **kwargs,
        )


class Nvfp4FlashinferTrtllmBackend(MoEBackend):
    supported_arches = frozenset({"sm100"})

    def __init__(
        self,
        key,
        spec: MoELayerSpec,
        quant_config: object,
        routing_config: dict | None = None,
    ):
        self.key = key
        self.spec = spec
        self.quant_config = quant_config
        self._group_size = quant_config.group_size

        routing_config = routing_config or {}
        self._n_group = routing_config.get("n_group", 0)
        self._topk_group = routing_config.get("topk_group", 0)
        self._routed_scaling_factor = routing_config.get("routed_scaling_factor", None)
        self._correction_bias = routing_config.get("correction_bias", None)
        self._routing_method_type = routing_config.get(
            "routing_method_type", RoutingMethodType.DeepSeekV3
        )
        # Routing precision.
        self._routing_logits_dtype = torch.bfloat16
        if self._routing_method_type in (
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.MiniMax2,
        ):
            self._routing_logits_dtype = torch.float32
        self._is_gated = spec.activation != "relu2"

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        return (
            current_platform().is_nvidia
            and isinstance(quant_config, Nvfp4Config)
            and spec.activation in {"relu2", "silu", "swiglu"}
            and not spec.use_deepep
        )

    @property
    def topk_output_format(self) -> TopKOutputFormat:
        return TopKOutputFormat.BYPASSED

    def create_layer_weights(
        self, layer: nn.Module, *, with_bias: bool = False
    ) -> None:
        del with_bias
        ispp = self.spec.intermediate_size // self.spec.tp_size
        create_fp4_weights(
            self,
            layer,
            self.spec.num_local_experts,
            self.spec.hidden_size,
            ispp,
            self._group_size,
            is_gated=self._is_gated,
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        num_experts = layer.w13_weight.shape[0]
        if self._is_gated:
            # intermediate_size_per_partition = half of w13 rows (gate + up)
            intermediate_size = layer.w13_weight.shape[1] // 2
        else:
            intermediate_size = layer.w13_weight.shape[1]
        hidden_size = layer.w13_weight.shape[2] * 2

        if self._is_gated:
            # Fix 1: Swap [W1(Gate), W3(Up)] -> [W3(Up), W1(Gate)].
            # The fused gated-act reorder interleaves [first_half, second_half]
            # as [row0_first, row0_second, row1_first, row1_second, ...].
            # It expects [W3(Up), W1(Gate)] so paired rows line up correctly.
            half_w = layer.w13_weight.shape[1] // 2
            w1_weight = layer.w13_weight.data[:, :half_w, :].clone()
            layer.w13_weight.data[:, :half_w, :] = layer.w13_weight.data[:, half_w:, :]
            layer.w13_weight.data[:, half_w:, :] = w1_weight
            del w1_weight

            half_s = layer.w13_weight_scale.shape[1] // 2
            w1_scale = layer.w13_weight_scale.data[:, :half_s, :].clone()
            layer.w13_weight_scale.data[:, :half_s, :] = layer.w13_weight_scale.data[
                :, half_s:, :
            ]
            layer.w13_weight_scale.data[:, half_s:, :] = w1_scale
            del w1_scale

        (
            w13_weight,
            w13_weight_scale,
            w2_weight,
            w2_weight_scale,
            hidden_size_unpadded,
            hidden_size,
        ) = _align_hidden_dim_for_trtllm_fp4(
            layer.w13_weight.data,
            layer.w13_weight_scale.data,
            layer.w2_weight.data,
            layer.w2_weight_scale.data,
            self._group_size,
        )
        min_intermediate_alignment = 16 if self._is_gated else 128
        (
            w13_weight,
            w13_weight_scale,
            w2_weight,
            w2_weight_scale,
            intermediate_size,
        ) = _align_intermediate_for_trtllm_fp4(
            w13_weight,
            w13_weight_scale,
            w2_weight,
            w2_weight_scale,
            self._group_size,
            is_gated=self._is_gated,
            min_alignment=min_intermediate_alignment,
        )

        # Shuffle weights and scales using fused-kernel permute indices.
        cache: dict = {}
        epilogue_tile_m = 128
        gemm1_rows = 2 * intermediate_size if self._is_gated else intermediate_size

        # View as fp8 for permutation (uint8 and fp8_e4m3fn are both 1 byte)
        w13_fp4 = w13_weight.view(torch.float8_e4m3fn).reshape(
            num_experts, gemm1_rows, hidden_size // 2
        )
        w13_scales = w13_weight_scale.view(torch.float8_e4m3fn).reshape(
            num_experts, gemm1_rows, hidden_size // self._group_size
        )
        w2_fp4 = w2_weight.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // 2
        )
        w2_scales = w2_weight_scale.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // self._group_size
        )

        w13_weights_shuffled = []
        w13_scales_shuffled = []
        w2_weights_shuffled = []
        w2_scales_shuffled = []

        for idx in range(num_experts):
            # W1/W3 (gemm1) weight permutation
            perm = _get_w13_permute_indices(
                cache,
                w13_fp4[idx].view(torch.uint8),
                epilogue_tile_m,
                is_gated=self._is_gated,
            )
            w13_weights_shuffled.append(
                w13_fp4[idx].view(torch.uint8)[perm.to(w13_fp4.device)].contiguous()
            )
            # W1/W3 scale permutation + interleave
            perm_sf = _get_w13_permute_indices(
                cache,
                w13_scales[idx].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
                is_gated=self._is_gated,
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
        layer.gemm1_weights_fp4_shuffled = torch.nn.Parameter(
            torch.stack(w13_weights_shuffled), requires_grad=False
        )
        layer.gemm1_scales_fp4_shuffled = torch.nn.Parameter(
            torch.stack(w13_scales_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, gemm1_rows, hidden_size // self._group_size),
            requires_grad=False,
        )
        layer.gemm2_weights_fp4_shuffled = torch.nn.Parameter(
            torch.stack(w2_weights_shuffled), requires_grad=False
        )
        layer.gemm2_scales_fp4_shuffled = torch.nn.Parameter(
            torch.stack(w2_scales_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, hidden_size, intermediate_size // self._group_size),
            requires_grad=False,
        )

        # Free original weights (replaced by shuffled versions)
        del layer.w13_weight
        del layer.w2_weight
        del layer.w13_weight_scale
        del layer.w2_weight_scale

        # Compute fused-kernel scales. Gated W1/W3 scales should match, so
        # keep the first shard value. Non-gated relu2 has one up-proj scale.
        w13_ws2 = (
            layer.w13_weight_scale_2[:, 0]
            if self._is_gated
            else layer.w13_weight_scale_2
        )
        # Input scales (max across shards) for alpha computation
        w13_input_scale = layer.w13_input_scale.max().to(torch.float32)
        w2_input_scale = layer.w2_input_scale.max().to(torch.float32)

        # Store input_scale_quant for runtime fp4_quantize
        w13_input_scale_quant = (1.0 / w13_input_scale).to(torch.float32)
        w2_input_scale_quant = (1.0 / w2_input_scale).to(torch.float32)

        layer.w13_input_scale_quant = torch.nn.Parameter(
            w13_input_scale_quant, requires_grad=False
        )
        # Fused-kernel alphas: input_scale * weight_scale_2
        layer.g1_alphas = torch.nn.Parameter(
            (w13_input_scale * w13_ws2).to(torch.float32), requires_grad=False
        )
        layer.g2_alphas = torch.nn.Parameter(
            (w2_input_scale * layer.w2_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )
        if self._is_gated:
            g1_scale_c = w2_input_scale_quant * layer.g1_alphas
        else:
            g1_scale_c = (
                w2_input_scale_quant.to(torch.float32).expand(num_experts).contiguous()
            )
        layer.g1_scale_c = torch.nn.Parameter(
            g1_scale_c.to(torch.float32),
            requires_grad=False,
        )
        # Store intermediate_size_per_partition for the executor
        layer.intermediate_size_per_partition = intermediate_size
        layer.hidden_size_unpadded = hidden_size_unpadded
        layer.hidden_size_padded = hidden_size

        # Free per-shard scales that are no longer needed
        del layer.w13_weight_scale_2
        del layer.w2_weight_scale_2
        del layer.w13_input_scale
        del layer.w2_input_scale

    def _activation_type_value(self) -> int:
        if self.spec.activation == "relu2":
            return ActivationType.Relu2.value
        return ActivationType.Swiglu.value

    @property
    def supports_deferred_finalize(self) -> bool:
        return True

    def forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        topk_output: object,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        do_finalize: bool = True,
    ) -> torch.Tensor:
        del num_global_tokens, max_num_tokens_per_gpu
        from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize

        x = hidden_states
        hidden_size_unpadded = getattr(layer, "hidden_size_unpadded", x.shape[-1])
        hidden_size_padded = getattr(layer, "hidden_size_padded", x.shape[-1])
        if hidden_size_padded != x.shape[-1]:
            x = F.pad(x, (0, hidden_size_padded - x.shape[-1]))
        num_tokens = x.shape[0]

        # Quantize input to FP4 using the fused-kernel scale layout.
        hs_fp4, hs_scale = fp4_quantize(
            x,
            layer.w13_input_scale_quant,
            is_sf_swizzled_layout=False,
            backend="cuda",
        )
        hs_scale = hs_scale.view(torch.float8_e4m3fn).reshape(
            num_tokens, hidden_size_padded // 16
        )
        per_token_scale = None

        routing_logits = topk_output.router_logits.to(self._routing_logits_dtype)
        topk_config = topk_output.topk_config
        correction_bias = topk_output.topk_config.correction_bias
        if correction_bias is None:
            correction_bias = self._correction_bias
        routing_bias = None if correction_bias is None else correction_bias.to(x.dtype)
        output = torch.empty(
            num_tokens,
            hidden_size_padded,
            dtype=x.dtype,
            device=x.device,
        )

        moe_kwargs = dict(
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            hidden_states=hs_fp4,
            hidden_states_scale=hs_scale,
            gemm1_weights=layer.gemm1_weights_fp4_shuffled.data,
            gemm1_weights_scale=layer.gemm1_scales_fp4_shuffled.data.view(
                torch.float8_e4m3fn
            ),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=layer.gemm2_weights_fp4_shuffled.data,
            gemm2_weights_scale=layer.gemm2_scales_fp4_shuffled.data.view(
                torch.float8_e4m3fn
            ),
            gemm2_bias=None,
            output1_scale_scalar=layer.g1_scale_c.data,
            output1_scale_gate_scalar=layer.g1_alphas.data,
            output2_scale_scalar=layer.g2_alphas.data,
            per_token_scale=per_token_scale,
            num_experts=int(self.spec.num_experts),
            top_k=int(topk_config.top_k),
            n_group=int(topk_config.num_expert_group or 0),
            topk_group=int(topk_config.topk_group or 0),
            intermediate_size=int(layer.intermediate_size_per_partition),
            local_expert_offset=int(self.spec.ep_rank * self.spec.num_local_experts),
            local_num_experts=int(self.spec.num_local_experts),
            routed_scaling_factor=self._routed_scaling_factor,
            routing_method_type=int(
                self._routing_method_type
                if self._routing_method_type is not None
                else RoutingMethodType.Default
            ),
            do_finalize=do_finalize,
            activation_type=self._activation_type_value(),
            output=output,
            tune_max_num_tokens=next_power_of_2(num_tokens),
        )
        result = trtllm_fp4_block_scale_moe_raw(**moe_kwargs)
        if do_finalize:
            output = result[0]
            if output.shape[-1] != hidden_size_unpadded:
                output = output[:, :hidden_size_unpadded]
            return output
        # Deferred: [gemm2_out, expert_weights, expanded_idx_to_permuted_idx]
        gemm2_out, expert_weights, expanded_idx = result
        if gemm2_out.shape[-1] != hidden_size_unpadded:
            gemm2_out = gemm2_out[:, :hidden_size_unpadded]
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


__all__ = ["Nvfp4FlashinferTrtllmBackend"]
