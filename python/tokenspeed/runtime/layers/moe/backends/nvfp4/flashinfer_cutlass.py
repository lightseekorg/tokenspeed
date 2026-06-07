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
from torch import nn

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.nvfp4.weights import (
    create_fp4_weights,
    finalize_common_flashinfer_weights,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.quantization import Nvfp4Config
from tokenspeed.runtime.utils import next_power_of_2


class Nvfp4FlashinferCutlassBackend(MoEBackend):
    supported_arches = frozenset({"sm100", "sm110", "sm120"})

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

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        return (
            current_platform().is_nvidia
            and isinstance(quant_config, Nvfp4Config)
            and spec.activation in {"silu", "swiglu"}
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
        finalize_common_flashinfer_weights(layer, swap_gate_up=True)

    def forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        topk_output: object,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        del num_global_tokens, max_num_tokens_per_gpu
        from tokenspeed_kernel.ops.moe.flashinfer import (
            ActivationType,
        )

        x = hidden_states
        output_dtype = torch.bfloat16
        output_col = x.shape[1]

        # After dispatch, some ranks may receive 0 tokens. The
        # flashinfer CUTLASS kernel cannot handle empty input, so return
        # a zero tensor directly.
        if x.shape[0] == 0:
            return x.new_zeros(0, output_col, dtype=output_dtype)

        # Allocate output
        symm_output = torch.empty(
            x.shape[0], output_col, dtype=output_dtype, device=x.device
        )

        return tokenspeed_kernel.moe_fused(
            output=symm_output,
            input=x,
            token_selected_experts=topk_output.topk_ids.to(torch.int),
            token_final_scales=topk_output.topk_weights,
            fc1_expert_weights=layer.w13_weight.view(torch.long),
            fc2_expert_weights=layer.w2_weight.view(torch.long),
            output_dtype=output_dtype,
            input_sf=None,
            quant_scales=[
                layer.w13_input_scale_quant,
                layer.w13_blockscale_swizzled.view(torch.int32),
                layer.g1_alphas,
                layer.w2_input_scale_quant,
                layer.w2_blockscale_swizzled.view(torch.int32),
                layer.g2_alphas,
            ],
            ep_size=self.spec.ep_size,
            ep_rank=self.spec.ep_rank,
            tp_size=self.spec.tp_size,
            tp_rank=self.spec.tp_rank,
            tune_max_num_tokens=next_power_of_2(x.shape[0]),
            activation_type=ActivationType.Swiglu,
            dtype=x.dtype,
            features={"pre_routed"},
            weight_format="nvfp4",
            traits={
                "tp": True,
                "ep": True,
                "cuda_graph": False,
            },
            expected_kernel_name="flashinfer_cutlass_fused_moe",
        )[0]


__all__ = ["Nvfp4FlashinferCutlassBackend"]
