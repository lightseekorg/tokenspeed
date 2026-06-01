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


"""Qwen3.5 MoE blocks shared by dense and MoE model variants."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.activation.triton import fused_gate_sigmoid_mul_add
from torch import nn

from tokenspeed.runtime.configs.qwen3_5_text_base_config import Qwen3_5BaseTextConfig
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_capture_mode
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.moe.layer import MoELayer
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.layers.moe.utils import (
    RoutingMethodType,
    get_all2all_backend,
    get_moe_backend,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict


def _is_moe_layer(layer_id: int, config) -> bool:
    """Return whether the given decoder layer should use the MoE block."""
    if layer_id < 0:
        return False
    mlp_only_layers = getattr(config, "mlp_only_layers", [])
    if layer_id in mlp_only_layers:
        return False
    return config.num_experts > 0 and (layer_id + 1) % config.decoder_sparse_step == 0


class Qwen3_5MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        if mapping.dense.has_tp:
            tp_size = mapping.dense.tp_size
            tp_rank = mapping.dense.tp_rank
            tp_group = mapping.dense.tp_group
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2,
                bias=False,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_group=tp_group,
                quant_config=quant_config,
                prefix=add_prefix("gate_up_proj", prefix),
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_group=tp_group,
                reduce_results=reduce_results,
                quant_config=quant_config,
                prefix=add_prefix("down_proj", prefix),
            )
        else:
            self.gate_up_proj = ReplicatedLinear(
                hidden_size,
                intermediate_size * 2,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("gate_up_proj", prefix),
            )
            self.down_proj = ReplicatedLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("down_proj", prefix),
            )

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        if x.shape[0] == 0:
            return x
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: Qwen3_5BaseTextConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        layer_index: int = -1,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.mapping = mapping
        self.layer_index = layer_index
        self.tp_size = mapping.world_size
        self.stream_fork = StreamFork(alt_stream)
        # DeepEP is only supported with the nvfp4 cutedsl MoE backend.
        # Draft models (non-quantized) must fall back to the TP path even
        # when the target model has deep_ep configured globally.
        self.use_deepep = (
            get_all2all_backend().is_deepep()
            and get_moe_backend().is_flashinfer_cutedsl()
        )
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_index,
            is_moe=True,
            prev_is_moe=_is_moe_layer(layer_index - 1, config),
        )

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.experts = MoELayer(
            top_k=config.num_experts_per_tok,
            num_experts=config.num_experts
            + global_server_args_dict["ep_num_redundant_experts"],
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=self.mapping.moe.tp_rank,
            tp_size=self.mapping.moe.tp_size,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
            routing_config={"routing_method_type": RoutingMethodType.RenormalizeNaive},
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=False,
            output_format=self.experts.topk_output_format,
            apply_routed_scaling_factor_on_output=(
                self.experts.apply_routed_scaling_factor_on_output
            ),
        )
        if getattr(config, "shared_expert_intermediate_size", 0) > 0:
            self.shared_expert = Qwen3_5MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                mapping=self.mapping,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_expert", prefix),
            )
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)
        else:
            self.shared_expert = None
            self.shared_expert_gate = None

    def get_moe_routed_weights(self):
        """Return routed expert weights excluding auxiliary shared parameters."""
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"] and "shared_experts" not in name
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        if self.use_deepep:
            return self._forward_deepep(
                hidden_states, num_global_tokens, max_num_tokens_per_gpu, ctx
            )
        return self._forward_tp(
            hidden_states, num_global_tokens, max_num_tokens_per_gpu, ctx
        )

    def _forward_tp(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Gate on local (pre-comm) tokens
        router_logits, _ = self.gate(hidden_states)

        # All-gather hidden_states and router_logits for topk + experts
        hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
        router_logits = self.comm_manager.pre_mlp_comm(router_logits, ctx)

        shared_output = None
        with self.stream_fork.scope(
            enable=(
                self.shared_expert is not None
                and hidden_states.shape[0] > 0
                and get_is_capture_mode()
            )
        ) as fork:
            with fork.branch():
                if self.shared_expert is not None:
                    shared_output = self.shared_expert(hidden_states)

            if hidden_states.shape[0] > 0:
                topk_output = self.topk(hidden_states, router_logits)
            else:
                topk_output = self.topk.empty_topk_output(
                    hidden_states.device,
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                )

            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                topk_output=topk_output,
                num_global_tokens=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            )

        if shared_output is not None:
            if self.shared_expert_gate is not None and hidden_states.shape[0] > 0:
                fused_gate_sigmoid_mul_add(
                    hidden_states,
                    self.shared_expert_gate.weight.squeeze(0),
                    shared_output,
                    final_hidden_states,
                )
            else:
                final_hidden_states = final_hidden_states + shared_output

        # Reduce-scatter / all-reduce expert output back to local token count
        final_hidden_states, _ = self.comm_manager.post_mlp_fused(
            final_hidden_states, None, ctx
        )

        return final_hidden_states.view(num_tokens, hidden_dim)

    def _forward_deepep(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """DeepEP path: routing on local tokens, dispatch/combine handled by executor."""
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Gate on local tokens (no all-gather needed)
        router_logits, _ = self.gate(hidden_states)

        # Shared expert on local tokens (TP-parallel, needs explicit reduce)
        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.mapping.dense.has_tp:
                from tokenspeed.runtime.distributed.comm_ops import all_reduce

                shared_output = all_reduce(
                    shared_output,
                    self.mapping.dense.tp_group,
                )

        # TopK on local tokens
        if hidden_states.shape[0] > 0:
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        # DeepEP executor handles dispatch -> MoE GEMM -> combine internally
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        )

        if shared_output is not None:
            if self.shared_expert_gate is not None and hidden_states.shape[0] > 0:
                fused_gate_sigmoid_mul_add(
                    hidden_states,
                    self.shared_expert_gate.weight.squeeze(0),
                    shared_output,
                    final_hidden_states,
                )
            else:
                final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states.view(num_tokens, hidden_dim)
