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

import tokenspeed_kernel
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn
from torch import nn

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.weights import create_moe_weight_pair
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.utils import next_power_of_2


class Bf16FlashinferTrtllmBackend(MoEBackend):
    supported_arches = frozenset({"sm100"})

    def __init__(
        self,
        key,
        spec: MoELayerSpec,
        quant_config: object,
        routing_config: dict | None = None,
    ):
        del quant_config
        self.key = key
        self.spec = spec
        self._autotuned = False
        routing_config = routing_config or {}
        self._n_group = routing_config.get("n_group", None)
        self._topk_group = routing_config.get("topk_group", None)
        self._routed_scaling_factor = routing_config.get("routed_scaling_factor", None)
        self._correction_bias = routing_config.get("correction_bias", None)
        self._routing_method_type = routing_config.get(
            "routing_method_type", RoutingMethodType.Renormalize
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
        ispp = spec.intermediate_size // spec.tp_size
        return (
            current_platform().is_nvidia
            and quant_config is None
            and spec.activation in {"silu", "swiglu"}
            and ispp % 128 == 0
        )

    def create_layer_weights(
        self, layer: nn.Module, *, with_bias: bool = False
    ) -> None:
        ispp = self.spec.intermediate_size // self.spec.tp_size
        create_moe_weight_pair(
            self,
            layer,
            self.spec.num_local_experts,
            self.spec.hidden_size,
            ispp,
            torch.get_default_dtype(),
            with_bias=with_bias,
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        from tokenspeed_kernel.ops.moe.flashinfer import (
            _maybe_get_cached_w3_w1_permute_indices,
            convert_to_block_layout,
            get_w2_permute_indices_with_cache,
        )

        cache_permute_indices: dict = {}
        num_experts = layer.w13_weight.shape[0]
        epilogue_tile_m = 128
        block_k = 128

        # The shared MoE loader stores w13 as [W1(gate), W3(up)]. The fused
        # gated activation layout expects [W3(up), W1(gate)] before applying
        # its interleaved row permutation.
        half_w = layer.w13_weight.shape[1] // 2
        w1_weight = layer.w13_weight.data[:, :half_w, :].clone()
        layer.w13_weight.data[:, :half_w, :] = layer.w13_weight.data[:, half_w:, :]
        layer.w13_weight.data[:, half_w:, :] = w1_weight
        del w1_weight

        old_shape_w13 = layer.w13_weight.data[0].shape
        old_shape_w2 = layer.w2_weight.data[0].shape
        new_shape_w13 = None
        new_shape_w2 = None

        for idx in range(num_experts):
            # Process w13 (gate_up_proj)
            permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                cache_permute_indices,
                layer.w13_weight.data[idx].view(torch.uint8),
                epilogue_tile_m,
            )
            tmp_weights1 = (
                layer.w13_weight.data[idx]
                .clone()
                .view(torch.uint8)[permute_indices.to(layer.w13_weight.data.device)]
                .contiguous()
            )

            # Process w2 (down_proj)
            permute_indices = get_w2_permute_indices_with_cache(
                cache_permute_indices,
                layer.w2_weight.data[idx].view(torch.uint8),
                epilogue_tile_m,
            )
            tmp_weights2 = (
                layer.w2_weight.data[idx]
                .clone()
                .view(torch.uint8)[permute_indices.to(layer.w2_weight.data.device)]
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

            layer.w13_weight.data[idx] = (
                tmp_weights1.view(torch.bfloat16).contiguous().reshape(old_shape_w13)
            )
            layer.w2_weight.data[idx] = (
                tmp_weights2.view(torch.bfloat16).contiguous().reshape(old_shape_w2)
            )

        layer.w13_weight.data = layer.w13_weight.data.reshape(
            num_experts, *new_shape_w13
        )
        layer.w2_weight.data = layer.w2_weight.data.reshape(num_experts, *new_shape_w2)

        # The fused MoE kernel requires routing bias to be bf16. Cast here
        # (post weight-load) so the captured bias reflects the loaded values,
        # not the empty Parameter.
        if self._correction_bias is not None:
            self._correction_bias = self._correction_bias.to(torch.bfloat16)

    @property
    def topk_output_format(self) -> TopKOutputFormat:
        return TopKOutputFormat.BYPASSED

    @property
    def supports_deferred_finalize(self) -> bool:
        return True

    def _call_trtllm_kernel(self, router_logits, x, layer, top_k, do_finalize):
        ispp = self.spec.intermediate_size // self.spec.tp_size
        num_local = self.spec.num_local_experts
        local_offset = self.spec.ep_rank * num_local

        routing_logits = router_logits.to(self._routing_logits_dtype)
        routing_bias = self._correction_bias

        output = tokenspeed_kernel.moe_fused(
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            hidden_states=x,
            gemm1_weights=layer.w13_weight,
            gemm2_weights=layer.w2_weight,
            num_experts=self.spec.num_experts,
            top_k=top_k,
            n_group=self._n_group,
            topk_group=self._topk_group,
            intermediate_size=ispp,
            local_expert_offset=local_offset,
            local_num_experts=num_local,
            routed_scaling_factor=self._routed_scaling_factor,
            routing_method_type=self._routing_method_type,
            do_finalize=do_finalize,
            tune_max_num_tokens=next_power_of_2(x.shape[0]),
            dtype=x.dtype,
            features={"self_routing"},
            weight_format="bf16",
            expected_kernel_name="flashinfer_trtllm_bf16_fused_moe",
        )
        if do_finalize:
            return output[0] if isinstance(output, (list, tuple)) else output
        # Deferred: [gemm2_out, expert_weights, expanded_idx_to_permuted_idx]
        gemm2_out, expert_weights, expanded_idx = output
        # The fused DSv3 routing kernel writes bf16 into a buffer Python
        # allocated as fp32 (``routing_logits.dtype``).
        if expert_weights.dtype == torch.float32:
            n, k = expert_weights.size()
            expert_weights = expert_weights.view(torch.bfloat16).view(-1, k)[:n]
        return (gemm2_out, expert_weights, expanded_idx)

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
        x = hidden_states
        # After dispatch, some ranks may receive 0 tokens. The
        # The fused kernel cannot handle empty input, so return
        # an empty tensor directly.
        if x.shape[0] == 0:
            return x.new_empty(0, self.spec.hidden_size)
        # Unquantized FlashInfer path expects BF16 activations.
        assert x.dtype == torch.bfloat16

        # BypassedTopKOutput provides router_logits and topk_config
        top_k = topk_output.topk_config.top_k
        router_logits = topk_output.router_logits

        try:
            from tokenspeed_kernel.ops.moe.flashinfer import (
                autotune as flashinfer_autotune,
            )
        except ImportError:
            flashinfer_autotune = None

        # Autotune on first call to pre-compile all kernel variants.
        # Equivalent to tokenspeed's _flashinfer_autotune() which runs a dummy
        # forward inside autotune() context. Without this, calls with new
        # token counts trigger JIT compilation that desyncs TP ranks.
        if not self._autotuned and flashinfer_autotune not in (None, error_fn):
            with flashinfer_autotune():
                # Autotune with do_finalize=True (matches the default path the
                # kernel heuristic is tuned for). The deferred path reuses
                # the same underlying kernel up to the last finalize step.
                self._call_trtllm_kernel(
                    router_logits, x, layer, top_k, do_finalize=True
                )
            self._autotuned = True

        return self._call_trtllm_kernel(router_logits, x, layer, top_k, do_finalize)


__all__ = ["Bf16FlashinferTrtllmBackend"]
