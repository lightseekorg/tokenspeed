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
from tokenspeed.runtime.layers.moe.backends.nvfp4.weights import (
    create_fp4_weights,
    finalize_common_flashinfer_weights,
    interleave_gate_up_chunks,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.quantization import Nvfp4Config
from tokenspeed.runtime.utils.pdl import pdl_enabled


def quantize_cutedsl_input(
    x: torch.Tensor, input_global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from tokenspeed_kernel.ops.quantization.flashinfer import (
        fp4_quantize as _fp4_quantize,
    )

    x_fp4, x_scale = _fp4_quantize(
        x,
        global_scale=input_global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=False,
        enable_pdl=pdl_enabled(),
    )
    return x_fp4, x_scale.unsqueeze(-1)


def get_cutedsl_graph_wrapper_capacity_hint() -> int:
    from tokenspeed.runtime.utils.env import global_server_args_dict

    if global_server_args_dict.get("enforce_eager", False):
        return 0

    capture_bs = global_server_args_dict.get("cudagraph_capture_sizes")
    if capture_bs:
        capacity = max(int(bs) for bs in capture_bs)
    else:
        capacity = int(global_server_args_dict.get("max_cudagraph_capture_size") or 0)

    if not global_server_args_dict.get("disable_prefill_graph", False):
        capacity = max(
            capacity,
            int(global_server_args_dict.get("prefill_graph_max_tokens") or 0),
        )

    return max(capacity, 0)


class Nvfp4FlashinferCuteDslBackend(MoEBackend):
    supported_arches = frozenset({"sm100", "sm110"})

    def __init__(
        self,
        key,
        spec: MoELayerSpec,
        quant_config: object,
        routing_config: dict | None = None,
    ):
        del routing_config
        self.key = key
        self.spec = spec
        self.quant_config = quant_config
        self._autotuned = False
        self._cuda_graph_wrapper = None
        self._cuda_graph_wrapper_capacity = get_cutedsl_graph_wrapper_capacity_hint()
        self._deepep_executor = None

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        ispp = spec.intermediate_size // spec.tp_size
        return (
            current_platform().is_nvidia
            and isinstance(quant_config, Nvfp4Config)
            and spec.activation in {"silu", "swiglu"}
            and ispp % 64 == 0
            and spec.a2a_backend in {"none", "deepep"}
        )

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
            self.quant_config.group_size,
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if not self.spec.use_deepep:
            layer.w13_weight = torch.nn.Parameter(
                interleave_gate_up_chunks(layer.w13_weight.data), requires_grad=False
            )
            layer.w13_weight_scale = torch.nn.Parameter(
                interleave_gate_up_chunks(layer.w13_weight_scale.data),
                requires_grad=False,
            )
        finalize_common_flashinfer_weights(layer, swap_gate_up=False)

    def _call_kernel(
        self,
        layer,
        x_fp4,
        x_scale,
        topk_ids,
        topk_weights,
        output_dtype,
        use_cuda_graph=False,
        capacity=None,
    ):
        return tokenspeed_kernel.moe_fused(
            x_fp4,
            x_scale,
            topk_ids,
            topk_weights,
            layer.w13_weight.data,
            layer.w13_blockscale_swizzled.data,
            layer.g1_alphas.data,
            layer.w2_input_scale_quant.data,
            layer.w2_weight.data,
            layer.w2_blockscale_swizzled.data,
            layer.g2_alphas.data,
            num_experts=self.spec.num_experts,
            top_k=self.spec.top_k,
            num_local_experts=self.spec.num_local_experts,
            local_expert_offset=self.spec.ep_rank * self.spec.num_local_experts,
            output_dtype=output_dtype,
            use_cuda_graph=use_cuda_graph,
            capacity=capacity,
            dtype=x_fp4.dtype,
            features={"pre_routed"},
            weight_format="nvfp4",
            traits={
                "tp": False,
                "ep": True,
                "cuda_graph": True,
            },
            expected_kernel_name="flashinfer_cutedsl_nvfp4_fused_moe",
        )

    def _get_deepep_executor(self, layer: nn.Module):
        if self._deepep_executor is None:
            from tokenspeed.runtime.layers.moe.backends.nvfp4.deepep_cutedsl_fp4_executor import (
                DeepEPCuteDslFp4Executor,
            )

            self._deepep_executor = DeepEPCuteDslFp4Executor(
                top_k=self.spec.top_k,
                num_experts=self.spec.num_experts,
                ep_rank=self.spec.ep_rank,
                ep_size=self.spec.ep_size,
                hidden_size=layer.w2_weight.shape[1],
            )
        return self._deepep_executor

    def forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        topk_output: object,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        x = hidden_states
        if self.spec.use_deepep:
            return self._get_deepep_executor(layer).forward(
                layer,
                hidden_states,
                topk_output,
                num_global_tokens,
                max_num_tokens_per_gpu,
            )

        from tokenspeed_kernel.ops.moe.flashinfer import autotune as flashinfer_autotune

        # After dispatch, some ranks may receive 0 tokens. The
        # CuteDSL wrapper cannot handle empty input, so return an empty
        # tensor directly.
        if x.shape[0] == 0:
            return torch.empty(0, x.shape[1], dtype=x.dtype, device=x.device)

        if topk_output.format.is_standard():
            topk_ids = topk_output.topk_ids.to(torch.int32)
            topk_weights = topk_output.topk_weights
        else:
            from tokenspeed.runtime.layers.moe.topk import select_experts

            result = select_experts(
                hidden_states=hidden_states,
                router_logits=topk_output.router_logits,
                topk_config=topk_output.topk_config,
                num_token_non_padded=topk_output.num_token_non_padded,
                expert_location_dispatch_info=topk_output.expert_location_dispatch_info,
            )
            topk_weights = result.topk_weights
            topk_ids = result.topk_ids.to(torch.int32)
        x_fp4, x_scale = quantize_cutedsl_input(x, layer.w13_input_scale_quant)
        wrapper_capacity = max(int(max_num_tokens_per_gpu or 0), x.shape[0])
        capacity = max(1, int(wrapper_capacity))

        if not self._autotuned and flashinfer_autotune is not error_fn:
            # Avoid profiling through the persistent CUDA-graph wrapper. Its
            # preallocated buffers are sized for the current graph capacity,
            # while autotune may profile larger internal token buckets.
            with flashinfer_autotune():
                output = self._call_kernel(
                    layer,
                    x_fp4,
                    x_scale,
                    topk_ids,
                    topk_weights,
                    x.dtype,
                    use_cuda_graph=False,
                )
            self._autotuned = True
            return output

        # Large runtime prefills should not mutate the persistent graph wrapper.
        # Build a one-shot non-graph wrapper instead so its workspace does not
        # stay resident across requests.
        use_graph = capacity <= self._cuda_graph_wrapper_capacity
        return self._call_kernel(
            layer,
            x_fp4,
            x_scale,
            topk_ids,
            topk_weights,
            x.dtype,
            use_cuda_graph=use_graph,
            capacity=self._cuda_graph_wrapper_capacity if use_graph else None,
        )


__all__ = ["Nvfp4FlashinferCuteDslBackend"]
