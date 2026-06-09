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


import torch

from tokenspeed.runtime.layers.activation import SwigluArg
from tokenspeed.runtime.layers.moe.core import MoELayerSpec, select_backend
from tokenspeed.runtime.layers.moe.utils import get_all2all_backend
from tokenspeed.runtime.layers.quantization.utils import should_ignore_quant_layer
from tokenspeed.runtime.utils.env import global_server_args_dict


class MoELayer(torch.nn.Module):
    def __init__(
        self,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        quant_config,
        layer_index: int,
        prefix: str = "",
        tp_rank: int | None = None,
        tp_size: int | None = None,
        ep_rank: int | None = None,
        ep_size: int | None = None,
        zero_expert_type: str = "",
        activation: str = "silu",
        activation_alpha=None,
        swiglu_limit=None,
        swiglu_beta: float | None = None,
        w13_input_layout: str = "concatenated",
        with_bias=False,
        routing_config: dict = None,
    ):
        super().__init__()
        self.layer_index = layer_index
        self.prefix = prefix
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.quant_config = quant_config
        self.ep_num_redundant_experts = global_server_args_dict[
            "ep_num_redundant_experts"
        ]
        self.zero_expert_type = zero_expert_type
        self.activation = activation
        self.swiglu_arg = None
        if self.activation == "swiglu":
            self.swiglu_arg = SwigluArg(alpha=activation_alpha, limit=swiglu_limit)
        # Per-model knobs the MoE backend reads in process_weights_after_loading.
        # ``swiglu_beta``: gpt-oss uses silu(α·gate)·(up + 1) and sets 1.0;
        # standard SwiGLU (e.g. deepseek-v4) leaves it None.
        # ``w13_input_layout``: "interleaved" for HF gpt-oss-style row layout
        # ([w1_0, w3_0, w1_1, w3_1, ...]); "concatenated" (default) for the
        # shared MoE checkpoint loader's [w1_all | w3_all] block layout.
        self.swiglu_beta = swiglu_beta
        if w13_input_layout not in {"interleaved", "concatenated"}:
            raise ValueError(
                f"w13_input_layout must be 'interleaved' or 'concatenated', "
                f"got {w13_input_layout!r}"
            )
        self.w13_input_layout = w13_input_layout

        if tp_rank is None:
            assert tp_size is None
            tp_rank, tp_size = 0, 1
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.moe_tp_size = self.tp_size
        if ep_rank is None:
            assert ep_size is None
            ep_rank, ep_size = 0, 1
        self.ep_rank, self.ep_size = ep_rank, ep_size

        if tp_size > 1 and ep_size > 1:
            raise ValueError("Mixed TP and EP is not supported yet.")

        num_local_experts = num_experts // self.ep_size
        a2a_backend = get_all2all_backend()
        self.num_local_experts = num_local_experts
        self._spec = MoELayerSpec(
            top_k=top_k,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            prefix=prefix,
            a2a_backend=a2a_backend.value,
        )
        self.backend = select_backend(
            self._spec,
            quant_config,
            routing_config=routing_config,
        )
        self.backend.create_layer_weights(self, with_bias=with_bias)

    def process_weights_after_loading(self, module) -> None:
        if getattr(self, "_weights_processed_after_loading", False):
            return
        self.backend.process_weights_after_loading(module)
        self._weights_processed_after_loading = True

    @property
    def topk_output_format(self):
        return self.backend.topk_output_format

    @property
    def apply_routed_scaling_factor_on_output(self):
        return self.backend.apply_routed_scaling_factor_on_output

    @property
    def supports_deferred_finalize(self) -> bool:
        return self.backend.supports_deferred_finalize

    def forward_zero_experts(self, topk_output):
        zero_expert_limit = self.num_experts
        if self.ep_num_redundant_experts is not None:
            zero_expert_limit = zero_expert_limit - self.ep_num_redundant_experts

        normal_expert_mask = topk_output.topk_ids >= zero_expert_limit
        topk_output.topk_ids[normal_expert_mask] = -1
        if self.zero_expert_type == "copy":
            topk_output.topk_weights[normal_expert_mask] = 1.0
        if self.zero_expert_type == "drop":
            topk_output.topk_weights[normal_expert_mask] = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        do_finalize: bool = True,
        prequantized_input: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        # Only pass ``do_finalize`` through when the caller actually wants
        # the deferred path. Other backends do not accept this kwarg;
        # their default (always-finalized) behavior is preserved when we
        # omit it.
        kwargs = {}
        if not do_finalize:
            assert (
                self.backend.supports_deferred_finalize
            ), f"{type(self.backend).__name__} does not support do_finalize=False"
            kwargs["do_finalize"] = False
        if prequantized_input is not None:
            kwargs["prequantized_input"] = prequantized_input
        return self.backend.forward(
            self,
            hidden_states,
            topk_output,
            num_global_tokens,
            max_num_tokens_per_gpu,
            **kwargs,
        )

    def construct_executor(self):
        from tokenspeed.runtime.layers.moe.dispatcher.deep_ep_local_compute import (
            DeepExecutor,
        )

        not_quant = self.quant_config is None or should_ignore_quant_layer(
            prefix=self.prefix,
            ignored_layers=getattr(self.quant_config, "ignored_layers", []),
        )
        w13_weight_scale = None
        w2_weight_scale = None
        if not not_quant:
            w13_weight_scale = getattr(
                self, "w13_weight_scale_inv", getattr(self, "w13_weight_scale", None)
            )
            w2_weight_scale = getattr(
                self, "w2_weight_scale_inv", getattr(self, "w2_weight_scale", None)
            )
        return DeepExecutor(
            self.w13_weight,
            w13_weight_scale,
            self.w2_weight,
            w2_weight_scale,
        )
