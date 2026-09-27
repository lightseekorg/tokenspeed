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

"""Kimi-K3 DeepEP composition: token ownership, weights and collectives."""

from __future__ import annotations

import torch
from torch import nn

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.comm_ops import all_reduce, token_all_gather
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.latent import (
    Kimi3LatentProjection,
    Kimi3MoEExecutionPlan,
)
from tokenspeed.runtime.layers.moe.topk import TopK, TopKOutputFormat
from tokenspeed.runtime.layers.moe.utils import (
    RoutingMethodType,
    get_moe_backend,
    use_deepep_low_latency,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.models.kimi_k3 import (
    KimiLinearMLP,
    KimiLinearMoEGate,
    _situ_betas,
)
from tokenspeed.runtime.utils import add_prefix


class KimiLinearMoEDeepEP(nn.Module):
    """Dispatch unique TP token slices and combine within their attention replica.

    The down projection is replicated for disjoint source rows. Shared experts
    and the up projection shard over attention TP; DeepEP owns only routed
    experts. Parameter names retain the ordinary K3 checkpoint layout.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        layer_index: int,
        model_scope: str,
        moe_block_count: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        alt_stream: torch.cuda.Stream | None,
    ) -> None:
        super().__init__()
        del model_scope, moe_block_count
        plan = Kimi3MoEExecutionPlan.build(mapping, get_moe_backend(), alt_stream)
        if (
            not plan.use_marlin
            or mapping.attn.cp_size != 1
            or mapping.moe.tp_size != 1
            or mapping.moe.dp_size != 1
            or mapping.moe.ep_size != mapping.attn.tp_size * mapping.attn.dp_size
        ):
            raise ValueError(
                "Kimi-K3 DeepEP requires Marlin, attention CP=1, MoE TP=1/DP=1, "
                "and expert EP == attention TP*DP inside each pipeline stage."
            )
        self.mapping = mapping
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.routed_scaling_factor = config.routed_scaling_factor
        self.routed_hidden = config.routed_expert_hidden_size or config.hidden_size
        situ_beta, situ_linear_beta = _situ_betas(config)
        self.gate = KimiLinearMoEGate(config.hidden_size, config.num_experts)
        self.experts = MoELayer(
            top_k=self.top_k,
            num_experts=self.num_experts,
            hidden_size=self.routed_hidden,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=mapping.moe.tp_rank,
            tp_size=mapping.moe.tp_size,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
            activation="situ",
            activation_situ_beta=situ_beta,
            activation_situ_linear_beta=situ_linear_beta,
            routing_config={
                "n_group": config.num_expert_group,
                "topk_group": config.topk_group,
                "routed_scaling_factor": self.routed_scaling_factor,
                "normalize_topk_weights": config.moe_renormalize,
                "correction_bias": self.gate.e_score_correction_bias,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "activation_situ_beta": situ_beta,
                "activation_situ_linear_beta": situ_linear_beta,
            },
            routing_mode="precomputed_topk",
            internal_activation_dtype_override="input",
        )

        self.topk = TopK(
            top_k=self.top_k,
            renormalize=config.moe_renormalize,
            use_grouped_topk=config.use_grouped_topk,
            num_expert_group=config.num_expert_group,
            num_fused_shared_experts=0,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            output_format=TopKOutputFormat.STANDARD,
            topk_weights_dtype=torch.float32,
        )
        self.routed_expert_down_proj = Kimi3LatentProjection(
            config.hidden_size,
            self.routed_hidden,
            params_dtype=None,
            prefix=add_prefix("routed_expert_down_proj", prefix),
            solution="auto",
            shard_group=None,
            column_group=None,
            shard_rank=0,
            shard_size=1,
            multicast_down=None,
        )
        tp = mapping.attn
        self._shard_up_projection = (
            tp.tp_size > 1 and config.hidden_size % tp.tp_size == 0
        )
        self.routed_expert_up_proj = Kimi3LatentProjection(
            self.routed_hidden,
            config.hidden_size,
            params_dtype=None,
            prefix=add_prefix("routed_expert_up_proj", prefix),
            solution="auto",
            shard_group=tp.tp_group if self._shard_up_projection else None,
            column_group=None,
            shard_rank=tp.tp_rank,
            shard_size=tp.tp_size,
            multicast_down=None,
        )
        self.routed_expert_norm = (
            RMSNorm(self.routed_hidden, eps=config.rms_norm_eps)
            if config.latent_moe_use_norm
            else None
        )
        self.shared_experts = KimiLinearMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
            tp_rank=tp.tp_rank,
            tp_size=tp.tp_size,
            tp_group=tp.tp_group,
            shared_parallel=None,
            quant_config=quant_config,
            prefix=add_prefix("shared_experts", prefix),
            reduce_results=False,
            is_shared_expert=True,
            activation_situ_beta=situ_beta,
            activation_situ_linear_beta=situ_linear_beta,
        )

    def pack_input_projection_weights(self) -> None:
        """Keep routed and shared producers separate: their token rows differ."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext | None,
    ) -> torch.Tensor:
        if ctx is None:
            raise RuntimeError("Kimi-K3 DeepEP requires a ForwardContext")
        tp = self.mapping.attn
        num_tokens = hidden_states.shape[0]
        counts = CommManager._scatter_count(num_tokens, tp.tp_size)
        start = sum(counts[: tp.tp_rank])
        source = hidden_states.narrow(0, start, counts[tp.tp_rank])
        logits = (
            self.gate(source)
            if source.shape[0]
            else hidden_states.new_empty((0, self.num_experts), dtype=torch.float32)
        )
        routing = (
            self.topk(source, logits, output_format=TopKOutputFormat.STANDARD)
            if source.shape[0]
            else self.topk.empty_topk_output(
                hidden_states.device, hidden_states=source, router_logits=logits
            )
        )
        if source.shape[0] and ctx.forward_mode.is_decode_or_idle():
            live = (
                ctx.attn_backend.decode_window_locations().narrow(
                    0, start, source.shape[0]
                )
                > 0
            )
            routing.topk_ids.masked_fill_(~live[:, None], -1)
            routing.topk_weights.masked_fill_(~live[:, None], 0)
        routed_in = (
            self.routed_expert_down_proj(source)[0]
            if source.shape[0]
            else hidden_states.new_empty((0, self.routed_hidden))
        )
        shared = None

        def compute_shared() -> None:
            nonlocal shared
            shared = self.shared_experts(hidden_states, down_out=None)

        # Empty DP ranks still join dispatch/combine, including graph padding.
        routed = self.experts(
            hidden_states=routed_in,
            topk_output=routing,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            do_finalize=True,
            low_latency=use_deepep_low_latency(ctx, tp.dp_size),
            overlap_fn=compute_shared,
        )
        if shared is None:
            raise RuntimeError("MoE did not execute its shared-expert callback")
        if num_tokens == 0:
            return prefix_sum
        if tp.tp_size > 1:
            routed = token_all_gather(
                routed, group=tp.tp_group, scattered_num_tokens=counts
            )
        if self.routed_expert_norm is not None:
            routed = self.routed_expert_norm(routed)
        if self._shard_up_projection:
            projected = self.routed_expert_up_proj.project_shard(routed)
            column, width = self.routed_expert_up_proj.shard_slice
            shared[:, column : column + width].add_(projected)
        else:
            projected, _ = self.routed_expert_up_proj(routed)
            if tp.tp_rank == 0:
                shared.add_(projected)
        if tp.tp_size > 1:
            shared = all_reduce(shared, tp.tp_group)
        return prefix_sum + shared
