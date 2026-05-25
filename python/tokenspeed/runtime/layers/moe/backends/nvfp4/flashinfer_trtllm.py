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

import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.moe.flashinfer import (
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)
from tokenspeed_kernel.ops.quantization.flashinfer import nvfp4_block_scale_interleave
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.nvfp4.weights import create_fp4_weights
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.layers.quantization import Nvfp4Config
from tokenspeed.runtime.utils import next_power_of_2
from tokenspeed.runtime.utils.pdl import pdl_enabled


def _expand_self_routing_logits_for_eplb(
    routing_logits: torch.Tensor,
    expert_location_dispatch_info: object | None,
    *,
    num_experts: int,
) -> torch.Tensor:
    if expert_location_dispatch_info is None or routing_logits.shape[-1] == num_experts:
        return routing_logits
    logical_to_physical = getattr(
        expert_location_dispatch_info, "partial_logical_to_all_physical_map", None
    )
    if (
        logical_to_physical is None
        or logical_to_physical.shape[0] != routing_logits.shape[-1]
    ):
        return routing_logits

    fill = torch.finfo(routing_logits.dtype).min
    out = torch.full(
        (*routing_logits.shape[:-1], int(num_experts)),
        fill,
        dtype=routing_logits.dtype,
        device=routing_logits.device,
    )
    physical_ids = logical_to_physical[:, 0].to(
        device=routing_logits.device, dtype=torch.long
    )
    return out.index_copy(1, physical_ids, routing_logits)


def _record_self_routing_topk_for_eplb(
    routing_logits: torch.Tensor,
    topk_config: object,
    *,
    num_experts: int,
) -> None:
    from tokenspeed.runtime.moe.distribution_recorder import (
        get_global_expert_distribution_recorder,
    )

    recorder = get_global_expert_distribution_recorder()
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    if not (getattr(recorder, "recording", False) or capturing):
        return
    top_k = int(getattr(topk_config, "top_k", 0))
    top_k -= int(getattr(topk_config, "num_fused_shared_experts", 0) or 0)
    top_k = min(top_k, int(routing_logits.shape[-1]))
    if top_k <= 0:
        return
    topk_ids = torch.topk(routing_logits, k=top_k, dim=-1).indices.to(torch.int32)
    recorder.on_select_experts(topk_ids=topk_ids, num_experts=num_experts)


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
        self._routed_scaling_factor = routing_config.get("routed_scaling_factor", 1.0)
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

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        return (
            current_platform().is_nvidia
            and isinstance(quant_config, Nvfp4Config)
            and spec.activation in {"silu", "swiglu"}
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
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        num_experts = layer.w13_weight.shape[0]
        # intermediate_size_per_partition = half of w13 rows (gate + up)
        intermediate_size = layer.w13_weight.shape[1] // 2
        hidden_size = layer.w13_weight.shape[2] * 2

        # Fix 1: Swap [W1(Gate), W3(Up)] -> [W3(Up), W1(Gate)].
        # The fused gated-act reorder interleaves [first_half, second_half] as
        # [row0_first, row0_second, row1_first, row1_second, ...].
        # It expects [W3(Up), W1(Gate)] so that the interleaved result pairs
        # each up-proj row with its corresponding gate-proj row correctly.
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

        # Shuffle weights and scales using fused-kernel permute indices.
        cache: dict = {}
        epilogue_tile_m = 128

        # View as fp8 for permutation (uint8 and fp8_e4m3fn are both 1 byte)
        w13_fp4 = layer.w13_weight.data.view(torch.float8_e4m3fn).reshape(
            num_experts, 2 * intermediate_size, hidden_size // 2
        )
        w13_scales = layer.w13_weight_scale.data.view(torch.float8_e4m3fn).reshape(
            num_experts, 2 * intermediate_size, hidden_size // self._group_size
        )
        w2_fp4 = layer.w2_weight.data.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // 2
        )
        w2_scales = layer.w2_weight_scale.data.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // self._group_size
        )

        w13_weights_shuffled = []
        w13_scales_shuffled = []
        w2_weights_shuffled = []
        w2_scales_shuffled = []

        for idx in range(num_experts):
            # W1/W3 (gemm1) weight permutation
            perm = _maybe_get_cached_w3_w1_permute_indices(
                cache, w13_fp4[idx].view(torch.uint8), epilogue_tile_m
            )
            w13_weights_shuffled.append(
                w13_fp4[idx].view(torch.uint8)[perm.to(w13_fp4.device)].contiguous()
            )
            # W1/W3 scale permutation + interleave
            perm_sf = _maybe_get_cached_w3_w1_permute_indices(
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
        layer.gemm1_weights_fp4_shuffled = torch.nn.Parameter(
            torch.stack(w13_weights_shuffled), requires_grad=False
        )
        layer.gemm1_scales_fp4_shuffled = torch.nn.Parameter(
            torch.stack(w13_scales_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(
                num_experts, 2 * intermediate_size, hidden_size // self._group_size
            ),
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

        # Compute fused-kernel scales.
        # Reduce w13_weight_scale_2: take per-expert value.
        w13_ws2 = layer.w13_weight_scale_2[:, 0]
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
        layer.g1_scale_c = torch.nn.Parameter(
            (w2_input_scale_quant * layer.g1_alphas).to(torch.float32),
            requires_grad=False,
        )
        # Store intermediate_size_per_partition for the executor
        layer.intermediate_size_per_partition = intermediate_size

        # Free per-shard scales that are no longer needed
        del layer.w13_weight_scale_2
        del layer.w2_weight_scale_2
        del layer.w13_input_scale
        del layer.w2_input_scale

        # The fused MoE kernel requires routing bias dtype to match
        # routing logits dtype. Cast here (post weight-load) so the captured
        # bias reflects the loaded values, not the empty Parameter.
        if self._correction_bias is not None:
            self._correction_bias = self._correction_bias.to(self._routing_logits_dtype)

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
        num_tokens = x.shape[0]

        # Quantize input to FP4 using the fused-kernel scale layout.
        hs_fp4, hs_scale = fp4_quantize(
            x,
            layer.w13_input_scale_quant,
            is_sf_swizzled_layout=False,
            enable_pdl=pdl_enabled(),
        )

        routing_logits = topk_output.router_logits.to(self._routing_logits_dtype)
        routing_bias = self._correction_bias
        routing_logits = _expand_self_routing_logits_for_eplb(
            routing_logits,
            getattr(topk_output, "expert_location_dispatch_info", None),
            num_experts=self.spec.num_experts,
        )
        _record_self_routing_topk_for_eplb(
            routing_logits,
            topk_output.topk_config,
            num_experts=self.spec.num_experts,
        )

        result = tokenspeed_kernel.moe_fused(
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            hidden_states=hs_fp4,
            hidden_states_scale=hs_scale.view(torch.float8_e4m3fn),
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
            num_experts=self.spec.num_experts,
            top_k=self.spec.top_k,
            n_group=self._n_group,
            topk_group=self._topk_group,
            intermediate_size=layer.intermediate_size_per_partition,
            local_expert_offset=self.spec.ep_rank * self.spec.num_local_experts,
            local_num_experts=self.spec.num_local_experts,
            routed_scaling_factor=self._routed_scaling_factor,
            routing_method_type=self._routing_method_type,
            do_finalize=do_finalize,
            tune_max_num_tokens=next_power_of_2(num_tokens),
            dtype=x.dtype,
            features={"self_routing"},
            traits={"weight_dtype": "nvfp4"},
            expected_kernel_name="flashinfer_trtllm_fp4_fused_moe",
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


__all__ = ["Nvfp4FlashinferTrtllmBackend"]
