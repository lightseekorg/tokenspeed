# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
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
from tokenspeed.runtime.layers.moe.backends.triton_weights import (
    attach_dense_weight_pair,
    register_block_scale_inverses,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.quantization import Fp8Config
from tokenspeed.runtime.utils import next_power_of_2

_FI_CUTLASS_MIN_BLOCK_SCALE = 1e-10
_FI_CUTLASS_TUNE_MIN_TOKENS = 8192


def _swap_w13_to_w31(tensor: torch.Tensor) -> torch.Tensor:
    gate, up = tensor.chunk(2, dim=1)
    return torch.cat((up, gate), dim=1).contiguous()


class Fp8FlashinferCutlassBackend(MoEBackend):
    supported_arches = frozenset({"sm90"})

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        platform = current_platform()
        return (
            platform.is_nvidia
            and platform.arch_version.major >= 9
            and isinstance(quant_config, Fp8Config)
            and tuple(quant_config.weight_block_size or ()) == (128, 128)
            and spec.ep_size > 1
            and spec.activation == "silu"
        )

    def create_layer_weights(
        self, layer: nn.Module, *, with_bias: bool = False
    ) -> None:
        ispp = attach_dense_weight_pair(
            self,
            layer,
            with_bias=with_bias,
            params_dtype=torch.float8_e4m3fn,
        )
        register_block_scale_inverses(
            self,
            layer,
            num_local_experts=self.spec.num_local_experts,
            hidden_size=self.spec.hidden_size,
            intermediate_size_per_partition=ispp,
            block_shape=self.quant_config.weight_block_size,
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # FlashInfer CUTLASS SwiGLU consumes W31 ([up; gate]) while the
        # checkpoint loader stores W13 ([gate; up]) for the Triton backend.
        layer.w13_weight.data = _swap_w13_to_w31(layer.w13_weight.data)
        layer.w13_weight_scale_inv.data = _swap_w13_to_w31(
            layer.w13_weight_scale_inv.data
        )
        layer.w13_weight_scale_inv.data.clamp_(min=_FI_CUTLASS_MIN_BLOCK_SCALE)
        layer.w2_weight_scale_inv.data.clamp_(min=_FI_CUTLASS_MIN_BLOCK_SCALE)

    def forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        topk_output: object,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        del num_global_tokens, max_num_tokens_per_gpu
        from tokenspeed_kernel.ops.moe.flashinfer import ActivationType

        x = hidden_states
        output_dtype = x.dtype
        output_col = x.shape[1]

        if x.shape[0] == 0:
            return x.new_zeros(0, output_col, dtype=output_dtype)

        output = torch.empty(
            x.shape[0], output_col, dtype=output_dtype, device=x.device
        )

        return tokenspeed_kernel.moe_fused(
            output=output,
            input=x,
            token_selected_experts=topk_output.topk_ids.to(torch.int),
            token_final_scales=topk_output.topk_weights,
            fc1_expert_weights=layer.w13_weight,
            fc2_expert_weights=layer.w2_weight,
            output_dtype=output_dtype,
            input_sf=None,
            quant_scales=[
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
            ],
            ep_size=self.spec.ep_size,
            ep_rank=self.spec.ep_rank,
            tp_size=self.spec.tp_size,
            tp_rank=self.spec.tp_rank,
            tune_max_num_tokens=max(
                _FI_CUTLASS_TUNE_MIN_TOKENS, next_power_of_2(x.shape[0])
            ),
            activation_type=ActivationType.Swiglu,
            dtype=x.dtype,
            features={"pre_routed"},
            traits={
                "weight_dtype": "fp8",
                "tp": self.spec.tp_size > 1,
                "ep": self.spec.ep_size > 1,
                "cuda_graph": False,
            },
            expected_kernel_name="flashinfer_cutlass_fused_moe",
            use_deepseek_fp8_block_scale=True,
        )[0]


__all__ = ["Fp8FlashinferCutlassBackend"]
