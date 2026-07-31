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

"""Inference-only Qwen2.5 MoE model compatible with HuggingFace weights."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from tokenspeed.runtime.configs.qwen2_5_moe_config import Qwen2_5MoeConfig
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_capture_mode
from tokenspeed.runtime.layers.linear import (
    ReplicatedLinear,
)
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    MoELayer,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.layers.moe.utils import (
    RoutingMethodType,
    get_all2all_backend,
    get_moe_backend,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.utils import get_layer_id
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
)
from tokenspeed.runtime.models.qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from tokenspeed.runtime.models.qwen3_5_moe import (
    Qwen3_5MoeMLP,
)
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.cuda_stream import StreamFork


def _is_moe_layer(layer_id: int, config) -> bool:
    """Return whether the given decoder layer should use the MoE block."""
    if layer_id < 0:
        return False
    mlp_only_layers = getattr(config, "mlp_only_layers", [])
    if layer_id in mlp_only_layers:
        return False
    return config.num_experts > 0 and (layer_id + 1) % config.decoder_sparse_step == 0


class Qwen2_5MoeSparseMoeBlock(nn.Module):
    """MoE block for Qwen2.5-MoE."""

    def __init__(
        self,
        config: Qwen2_5MoeConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        layer_index: int = -1,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.mapping = mapping
        self.layer_index = layer_index
        self.stream_fork = StreamFork(alt_stream)

        self.use_deepep = (
            get_all2all_backend().is_deepep()
            and get_moe_backend().is_flashinfer_cutedsl()
        )

        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_index,
            is_moe=True,
            prev_is_moe=_is_moe_layer(layer_index - 1, config) if layer_index > 0 else False,
        )

        if mapping.moe.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {mapping.moe.tp_size} is greater than "
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
            tp_rank=mapping.moe.tp_rank,
            tp_size=mapping.moe.tp_size,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
            routing_config={
                "routing_method_type": RoutingMethodType.RenormalizeNaive,
                "normalize_topk_weights": config.norm_topk_prob,
            },
        )

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=False,
            output_format=self.experts.topk_output_format,
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

        router_logits, _ = self.gate(hidden_states)

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
                from tokenspeed_kernel.ops.activation.triton import fused_gate_sigmoid_mul_add

                fused_gate_sigmoid_mul_add(
                    hidden_states,
                    self.shared_expert_gate.weight.squeeze(0),
                    shared_output,
                    final_hidden_states,
                )
            else:
                final_hidden_states = final_hidden_states + shared_output

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

        router_logits, _ = self.gate(hidden_states)

        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.mapping.dense.has_tp:
                from tokenspeed.runtime.distributed.comm_ops import all_reduce

                shared_output = all_reduce(
                    shared_output,
                    self.mapping.dense.tp_group,
                )

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
                from tokenspeed_kernel.ops.activation.triton import fused_gate_sigmoid_mul_add

                fused_gate_sigmoid_mul_add(
                    hidden_states,
                    self.shared_expert_gate.weight.squeeze(0),
                    shared_output,
                    final_hidden_states,
                )
            else:
                final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states.view(num_tokens, hidden_dim)

    def get_moe_routed_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"] and "shared_experts" not in name
        ]


class Qwen2MoeDecoderLayer(Qwen2DecoderLayer):
    """Qwen2.5-MoE decoder layer with conditional MoE or dense MLP."""

    def __init__(
        self,
        config: Qwen2_5MoeConfig,
        mapping: Mapping,
        layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            mapping=mapping,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
        )

        if _is_moe_layer(layer_id, config):
            self.mlp = Qwen2_5MoeSparseMoeBlock(
                config=config,
                mapping=mapping,
                quant_config=quant_config,
                layer_index=layer_id,
                prefix=add_prefix("mlp", prefix),
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        residual: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif ctx.input_num_tokens > global_server_args_dict["comm_fusion_max_num_tokens"]:
            hidden_states = all_reduce(hidden_states, self.mapping.dense.tp_group)
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            hidden_states, residual, *_ = (
                self.input_layernorm.forward_with_allreduce_fusion(
                    self.mapping.dense.tp_rank,
                    self.mapping.dense.tp_group,
                    hidden_states,
                    residual,
                )
            )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
            cos_sin=cos_sin,
        )

        if ctx.input_num_tokens > global_server_args_dict["comm_fusion_max_num_tokens"]:
            hidden_states = all_reduce(hidden_states, self.mapping.attn.tp_group)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        else:
            hidden_states, residual, *_ = (
                self.post_attention_layernorm.forward_with_allreduce_fusion(
                    self.mapping.attn.tp_rank,
                    self.mapping.attn.tp_group,
                    hidden_states,
                    residual,
                )
            )

        if isinstance(self.mlp, Qwen2_5MoeSparseMoeBlock):
            num_global_tokens, max_num_tokens_per_gpu = (
                self.mlp.comm_manager.get_num_tokens(ctx)
            )
            hidden_states = self.mlp(
                hidden_states,
                num_global_tokens,
                max_num_tokens_per_gpu,
                ctx,
            )
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen2MoeModel(Qwen2Model):
    """Qwen2.5-MoE model using MoE decoder layers."""

    def __init__(
        self,
        config: Qwen2_5MoeConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen2MoeDecoderLayer,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                ctx,
                out_cache_loc,
                residual,
                cos_sin=None,
            )

        if ctx.input_num_tokens > global_server_args_dict["comm_fusion_max_num_tokens"]:
            hidden_states = all_reduce(hidden_states, self.mapping.dense.tp_group)
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states, *_ = self.norm.forward_with_allreduce_fusion(
                self.mapping.dense.tp_rank,
                self.mapping.dense.tp_group,
                hidden_states,
                residual,
            )
        return hidden_states, None

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = self.mapping.attn.tp_size
        tp_rank = self.mapping.attn.tp_rank
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn
            if hasattr(layer_self_attn.attn, "k_scale"):
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling factor attribute!"
                )


class Qwen2MoeForCausalLM(Qwen2ForCausalLM):
    """Qwen2.5-MoE causal language model head."""

    model_cls = Qwen2MoeModel

    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
            ),
            fused_schema=ExpertCheckpointSchema(
                gate_up_fused_name="gate_up_proj",
                down_proj_name="down_proj",
            ),
            num_experts=self.config.num_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )

        for name, loaded_weight in weights:
            if "Embedding" in self.config.name_or_path:
                name = add_prefix(name, "model")
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith((".bias", "_bias")) and name not in params_dict:
                    continue
                if moe_loader.matches(name):
                    moe_loader.load(name, loaded_weight)
                    continue
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = Qwen2MoeForCausalLM