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
from tokenspeed.runtime.layers.moe.backends.weights import create_moe_weight_pair
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.utils import next_power_of_2


class Bf16FlashinferCutlassBackend(MoEBackend):
    supported_arches = frozenset({"sm89", "sm90", "sm100", "sm110", "sm120"})

    def __init__(
        self,
        key,
        spec: MoELayerSpec,
        quant_config: object,
        routing_config: dict | None = None,
    ):
        from tokenspeed.runtime.utils.env import global_server_args_dict

        del quant_config, routing_config
        self.key = key
        self.spec = spec
        self._autotuned = False

        mapping = global_server_args_dict["mapping"]
        self._tp_size = mapping.moe.tp_size
        self._tp_rank = mapping.moe.tp_rank

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        return (
            current_platform().is_nvidia
            and quant_config is None
            and spec.activation in {"silu", "swiglu"}
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
        if hasattr(layer, "w13_weight") and layer.w13_weight is not None:
            # Swap w1 and w3 as the definition of
            # SwiGLU uses the fused-kernel layout.
            half_w = layer.w13_weight.shape[1] // 2
            temp_w = layer.w13_weight.data[:, :half_w, :].clone()
            layer.w13_weight.data[:, :half_w, :] = layer.w13_weight.data[:, half_w:, :]
            layer.w13_weight.data[:, half_w:, :] = temp_w
            del temp_w

    def _call_cutlass_kernel(self, x, layer, topk_output):
        from tokenspeed_kernel.ops.moe.flashinfer import (
            ActivationType,
        )

        return tokenspeed_kernel.moe_fused(
            input=x,
            token_selected_experts=topk_output.topk_ids.to(torch.int),
            token_final_scales=topk_output.topk_weights,
            fc1_expert_weights=layer.w13_weight,
            fc2_expert_weights=layer.w2_weight,
            output_dtype=x.dtype,
            quant_scales=None,
            ep_size=self.spec.ep_size,
            ep_rank=self.spec.ep_rank,
            tp_size=self._tp_size,
            tp_rank=self._tp_rank,
            tune_max_num_tokens=max(8192, next_power_of_2(x.shape[0])),
            activation_type=ActivationType.Swiglu,
            dtype=x.dtype,
            features={"pre_routed"},
            traits={
                "weight_dtype": "bf16",
                "tp": True,
                "ep": True,
                "cuda_graph": False,
            },
            expected_kernel_name="flashinfer_cutlass_fused_moe",
        )[0]

    def forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        topk_output: object,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        del num_global_tokens, max_num_tokens_per_gpu
        x = hidden_states
        # After dispatch, some ranks may receive 0 tokens. The
        # flashinfer CUTLASS kernel cannot handle empty input, so return
        # an empty tensor directly.
        if x.shape[0] == 0:
            return x.new_empty(0, self.spec.hidden_size)
        # Unquantized FlashInfer path expects BF16 activations.
        assert x.dtype == torch.bfloat16

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
        if not self._autotuned and flashinfer_autotune is not None:
            with flashinfer_autotune():
                self._call_cutlass_kernel(x, layer, topk_output)
            self._autotuned = True
        return self._call_cutlass_kernel(x, layer, topk_output)


__all__ = ["Bf16FlashinferCutlassBackend"]
