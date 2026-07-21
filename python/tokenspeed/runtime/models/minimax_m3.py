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

"""Inference-only MiniMax-M3 model for the M3-VL checkpoint."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import torch
from tokenspeed_kernel.ops.activation.triton import swiglu_oai
from tokenspeed_kernel.ops.layernorm.triton import qk_rmsnorm
from torch import nn
from transformers import MiniMaxM3VLTextConfig

from tokenspeed.runtime.configs.minimax_m3_config import (
    MiniMaxM3Config,
    MiniMaxM3VisionConfig,
)
from tokenspeed.runtime.configs.paged_cache_spec import FULL_ATTENTION
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.mm_encoder_attention import VisionAttention
from tokenspeed.runtime.layers.conv import Conv3dLayer
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import BaseCausalLM, BaseTransformerModel
from tokenspeed.runtime.models.base.comm_ops import FinalNormOp
from tokenspeed.runtime.models.base.placement import ParallelGroup
from tokenspeed.runtime.models.utils import validate_attention_partition
from tokenspeed.runtime.moe.expert_location import ModelConfigForExpertLocation
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    MultimodalEmbedder,
    pad_input_tokens,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)


class MiniMaxM3MLP(nn.Module):
    """Dense MiniMax-M3 MLP using the SwiGLU-OAI activation."""

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        intermediate_size: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_shared_expert: bool = False,
    ) -> None:
        super().__init__()
        if is_shared_expert:
            # The MoE block's combined routed+shared output is reduced once
            # over the MoE tp_ep group, so shard the shared expert over that
            # same group (mapping.dense may differ under DP/EP layouts).
            tp_rank = mapping.moe.tp_ep_rank
            tp_size = mapping.moe.tp_ep_size
            tp_group = mapping.moe.tp_ep_group
        else:
            tp_rank = mapping.dense.tp_rank
            tp_size = mapping.dense.tp_size
            tp_group = mapping.dense.tp_group
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("down_proj", prefix),
        )
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        gate_up, _ = self.gate_up_proj(hidden_states)
        activated = swiglu_oai(
            gate_up,
            alpha=self.swiglu_alpha,
            limit=self.swiglu_limit,
        )
        output, _ = self.down_proj(activated)
        return output


class MiniMaxM3SparseMoeBlock(nn.Module):
    """MiniMax-M3 routed experts plus one unconditional shared expert."""

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        layer_index: int = -1,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if mapping.moe.tp_ep_size > config.num_local_experts:
            raise ValueError(
                f"MoE parallel size {mapping.moe.tp_ep_size} exceeds "
                f"{config.num_local_experts} experts."
            )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            quant_config=None,
            params_dtype=torch.float32,
            prefix=add_prefix("gate", prefix),
        )
        self.routing_bias = nn.Parameter(
            torch.zeros(config.num_local_experts, dtype=torch.float32)
        )

        routing_config = {
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": config.routed_scaling_factor,
            "normalize_topk_weights": True,
            "correction_bias": self.routing_bias,
            "routing_method_type": RoutingMethodType.MiniMax2,
        }
        self.experts = MoELayer(
            top_k=config.num_experts_per_tok,
            num_experts=(
                config.num_local_experts
                + global_server_args_dict["ep_num_redundant_experts"]
            ),
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=add_prefix("experts", prefix),
            tp_rank=mapping.moe.tp_rank,
            tp_size=mapping.moe.tp_size,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
            activation="swiglu",
            activation_alpha=config.swiglu_alpha,
            swiglu_limit=config.swiglu_limit,
            swiglu_beta=1.0,
            w13_input_layout="concatenated",
            routing_config=routing_config,
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=True,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            correction_bias=self.routing_bias,
            routed_scaling_factor=config.routed_scaling_factor,
            output_format=self.experts.topk_output_format,
        )
        self.shared_experts = MiniMaxM3MLP(
            config=config,
            intermediate_size=config.shared_intermediate_size,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("shared_experts", prefix),
            is_shared_expert=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        router_logits, _ = self.gate(hidden_states.to(torch.float32))

        if hidden_states.shape[0] == 0:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        else:
            topk_output = self.topk(hidden_states, router_logits)

        routed_output = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        )
        shared_output = self.shared_experts(hidden_states)
        return (routed_output + shared_output).view(num_tokens, hidden_size)


class MiniMaxM3Indexer(nn.Module):
    """MiniMax-M3's lightweight per-GQA-group block indexer."""

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        total_index_heads = config.index_n_heads
        if total_index_heads != config.num_key_value_heads:
            raise ValueError(
                "MiniMax-M3 requires one index-query head per GQA group: "
                f"index_heads={total_index_heads}, kv_heads={config.num_key_value_heads}."
            )
        if mapping.attn.tp_size > total_index_heads:
            raise ValueError(
                f"TP={mapping.attn.tp_size} exceeds {total_index_heads} index "
                "heads; index-head replication is not supported."
            )
        self.num_index_heads = total_index_heads // mapping.attn.tp_size
        self.head_dim = config.index_head_dim
        self.q_size = self.num_index_heads * self.head_dim
        self.index_q_proj = ColumnParallelLinear(
            config.hidden_size,
            total_index_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("index_q_proj", prefix),
        )
        self.index_k_proj = ReplicatedLinear(
            config.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("index_k_proj", prefix),
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rope_type = (config.rope_parameters or {}).get("rope_type", "default")
        assert rope_type == "default", f"RoPE type {rope_type} is not supported."
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=config.rotary_dim,
            max_position=config.max_position_embeddings,
            base=int(config.rope_parameters["rope_theta"]),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index_q, _ = self.index_q_proj(hidden_states)
        index_k, _ = self.index_k_proj(hidden_states)
        index_q, index_k = qk_rmsnorm(
            index_q,
            index_k,
            self.q_norm.gemma_weight,
            self.k_norm.gemma_weight,
            self.q_norm.variance_epsilon,
        )
        index_q, index_k = self.rotary_emb(positions, index_q, index_k)
        return (
            index_q.view(-1, self.num_index_heads, self.head_dim),
            index_k.view(-1, self.head_dim),
        )


class MiniMaxM3Attention(nn.Module):
    """Project attention inputs and delegate dense/MSA execution to the backend."""

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        validate_attention_partition(
            config.num_attention_heads,
            config.num_key_value_heads,
            mapping.attn.tp_size,
        )
        self.num_heads = config.num_attention_heads // mapping.attn.tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // mapping.attn.tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.is_sparse = config.layer_types[layer_id] == "minimax_m3_sparse"

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rope_type = (config.rope_parameters or {}).get("rope_type", "default")
        assert rope_type == "default", f"RoPE type {rope_type} is not supported."
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=config.rotary_dim,
            max_position=config.max_position_embeddings,
            base=int(config.rope_parameters["rope_theta"]),
        )
        self.indexer = (
            MiniMaxM3Indexer(
                config=config,
                mapping=mapping,
                quant_config=quant_config,
                prefix=prefix,
            )
            if self.is_sparse
            else None
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            group_id=FULL_ATTENTION,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = qk_rmsnorm(
            q,
            k,
            self.q_norm.gemma_weight,
            self.k_norm.gemma_weight,
            self.q_norm.variance_epsilon,
        )
        q, k = self.rotary_emb(positions, q, k)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        attn_kwargs = {}
        if self.is_sparse:
            index_q, index_k = self.indexer(positions, hidden_states)
            attn_kwargs = {"index_q": index_q, "index_k": index_k}
        attn_output = self.attn(
            q,
            k,
            v,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
            **attn_kwargs,
        )
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM3DecoderLayer(nn.Module):
    """Decoder layer selected as dense or MoE by ``mlp_layer_types``."""

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.layer_id = layer_id
        self.is_moe_layer = config.mlp_layer_types[layer_id] == "sparse"
        previous_is_moe_layer = (
            layer_id > 0 and config.mlp_layer_types[layer_id - 1] == "sparse"
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = MiniMaxM3Attention(
            config=config,
            mapping=mapping,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        if self.is_moe_layer:
            self.mlp = MiniMaxM3SparseMoeBlock(
                config=config,
                mapping=mapping,
                quant_config=quant_config,
                layer_index=layer_id,
                prefix=add_prefix("block_sparse_moe", prefix),
            )
        else:
            self.mlp = MiniMaxM3MLP(
                config=config,
                intermediate_size=config.dense_intermediate_size,
                mapping=mapping,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=self.is_moe_layer,
            prev_is_moe=previous_is_moe_layer,
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        residual: torch.Tensor | None,
        aux_hidden_states: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(
            ctx
        )

        if not ctx.forward_mode.is_idle():
            hidden_states, residual = self.comm_manager.input_reduce_norm(
                hidden_states, residual
            )
            if aux_hidden_states is not None:
                aux_hidden_states.append(
                    self.comm_manager.gather_residual(residual, ctx).clone()
                )
            hidden_states = self.comm_manager.pre_attn_comm(hidden_states, ctx)
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
            )
            hidden_states, residual = self.comm_manager.post_attn_reduce_norm(
                hidden_states, residual, ctx
            )

        hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
        if self.is_moe_layer:
            hidden_states = self.mlp(
                hidden_states, num_global_tokens, max_num_tokens_per_gpu
            )
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.comm_manager.post_mlp_fused(
            hidden_states, residual, ctx
        )
        return hidden_states, residual


class MiniMaxM3Model(BaseTransformerModel):
    """MiniMax-M3 decoder-only text backbone."""

    layer_cls = MiniMaxM3DecoderLayer

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, mapping, quant_config, prefix)
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        last_layer = self.layers[-1]
        group_type = (
            ParallelGroup.MOE_TP_EP
            if last_layer.is_moe_layer
            else ParallelGroup.DENSE_TP
        )
        self._final_norm_op = FinalNormOp(
            mapping=mapping,
            group_type=group_type,
            norm_module=self.norm,
            use_all_reduce_mode=last_layer.comm_manager.use_all_reduce(
                last_layer.is_moe_layer
            ),
            lm_head_group_type=ParallelGroup.ATTN_TP,
        )


class MiniMaxM3SparseForCausalLM(BaseCausalLM):
    """MiniMax-M3 text model with checkpoint-compatible weight loading."""

    model_cls = MiniMaxM3Model
    fall_back_to_pt_during_load = False

    def _load_non_language_weight(
        self,
        checkpoint_name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> str | None:
        del checkpoint_name, loaded_weight, params_dict
        return None

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        **kwargs,
    ) -> set[str]:
        del kwargs
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="w1",
                down_proj_name="w2",
                up_proj_name="w3",
            ),
            num_experts=self.config.num_local_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )

        loaded_params: set[str] = set()
        for checkpoint_name, loaded_weight in weights:
            if checkpoint_name.startswith("language_model."):
                name = checkpoint_name.removeprefix("language_model.")
            elif checkpoint_name.startswith(
                ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")
            ):
                loaded_name = self._load_non_language_weight(
                    checkpoint_name,
                    loaded_weight,
                    params_dict,
                )
                if loaded_name is not None:
                    loaded_params.add(loaded_name)
                continue
            else:
                name = checkpoint_name

            if name.startswith("model.mtp."):
                continue
            if "rotary_emb.inv_freq" in name:
                continue

            name = name.replace(".block_sparse_moe", ".mlp")
            name = name.replace(".e_score_correction_bias", ".routing_bias")
            name = name.replace(
                ".self_attn.index_q_proj", ".self_attn.indexer.index_q_proj"
            )
            name = name.replace(
                ".self_attn.index_k_proj", ".self_attn.indexer.index_k_proj"
            )
            name = name.replace(".self_attn.index_q_norm", ".self_attn.indexer.q_norm")
            name = name.replace(".self_attn.index_k_norm", ".self_attn.indexer.k_norm")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or ".mlp.experts." in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                param = params_dict.get(mapped_name)
                if param is None:
                    continue
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                break
            else:
                if moe_loader.matches(name):
                    loaded_params.add(moe_loader.load(name, loaded_weight))
                    continue
                if moe_loader.is_expert_checkpoint_weight(name):
                    continue

                param = params_dict.get(name)
                if param is None:
                    raise KeyError(
                        f"MiniMax-M3 checkpoint parameter {checkpoint_name!r} "
                        "has no runtime parameter mapping."
                    )
                weight_loader = getattr(
                    param,
                    "weight_loader",
                    default_weight_loader,
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(
        cls,
        config: MiniMaxM3Config,
    ) -> ModelConfigForExpertLocation:
        return ModelConfigForExpertLocation(
            num_layers=config.text_config.num_hidden_layers,
            num_logical_experts=config.text_config.num_local_experts,
            num_groups=None,
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_vision_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    _input_shape,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MiniMax-M3's partial 3D RoPE to packed vision queries and keys."""

    cos, sin = position_embeddings
    rotary_dim = cos.shape[-1]
    query_rot, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    key_rot, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]

    query_rot = query_rot.float()
    key_rot = key_rot.float()
    cos = cos.float()
    sin = sin.float()
    query_rot = query_rot * cos + _rotate_half(query_rot) * sin
    key_rot = key_rot * cos + _rotate_half(key_rot) * sin
    query = torch.cat((query_rot.to(query_pass.dtype), query_pass), dim=-1)
    key = torch.cat((key_rot.to(key_pass.dtype), key_pass), dim=-1)
    return query, key


class MiniMaxM3VisionEmbeddings(nn.Module):
    """Convert flattened image/video patches into vision hidden states."""

    def __init__(self, config: MiniMaxM3VisionConfig) -> None:
        super().__init__()
        self.num_channels = config.num_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        kernel_size = (
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        self.patch_embedding = Conv3dLayer(
            self.num_channels,
            self.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() != 2:
            raise ValueError(
                f"MiniMax-M3 pixel_values must be 2D, got {pixel_values.dim()}D."
            )
        expected_patch_width = (
            self.num_channels * self.temporal_patch_size * self.patch_size**2
        )
        if pixel_values.shape[1] != expected_patch_width:
            raise ValueError(
                "MiniMax-M3 flattened patch width must be "
                f"{expected_patch_width}, got {pixel_values.shape[1]}."
            )
        pixel_values = pixel_values.reshape(
            -1,
            self.num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.patch_embedding(pixel_values)
        return hidden_states.reshape(-1, self.hidden_size)


class MiniMaxM3VisionMLP(nn.Module):
    """Tensor-parallel GELU feed-forward block used by the vision tower."""

    def __init__(
        self,
        config: MiniMaxM3VisionConfig,
        mapping: Mapping,
        prefix: str,
    ) -> None:
        super().__init__()
        vision = mapping.vision
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=None,
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            prefix=add_prefix("fc1", prefix),
        )
        self.activation = nn.GELU()
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=None,
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            prefix=add_prefix("fc2", prefix),
            reduce_results=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class MiniMaxM3VisionEncoderLayer(nn.Module):
    """One CLIP-style vision layer with MiniMax 3D rotary attention."""

    def __init__(
        self,
        config: MiniMaxM3VisionConfig,
        mapping: Mapping,
        prefix: str,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.self_attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            mapping=mapping,
            head_size=config.hidden_size // config.num_attention_heads,
            quant_config=None,
            prefix=add_prefix("self_attn", prefix),
            proj_bias=True,
            qkv_bias=True,
            customized_position_embedding_applier=_apply_vision_rotary,
            mm_attention_backend=mm_attention_backend,
        )
        self.layer_norm2 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.mlp = MiniMaxM3VisionMLP(
            config,
            mapping,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int,
    ) -> torch.Tensor:
        residual = hidden_states
        attention_output = self.self_attn(
            self.layer_norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
        )
        if attention_output.dim() == 3:
            attention_output = attention_output.squeeze(0)
        hidden_states = residual + attention_output
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class MiniMaxM3VisionEncoder(nn.Module):
    """Stack of MiniMax-M3 vision encoder layers."""

    def __init__(
        self,
        config: MiniMaxM3VisionConfig,
        mapping: Mapping,
        prefix: str,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MiniMaxM3VisionEncoderLayer(
                    config,
                    mapping,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    mm_attention_backend=mm_attention_backend,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cu_seqlens,
                position_embeddings,
                max_seqlen,
            )
        return hidden_states


class MiniMaxM3VisionTransformer(nn.Module):
    """MiniMax-M3 Conv3d vision transformer with packed variable-length attention."""

    def __init__(
        self,
        config: MiniMaxM3VisionConfig,
        mapping: Mapping,
        prefix: str,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError(
                "MiniMax-M3 vision hidden_size must be divisible by "
                "num_attention_heads."
            )

        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.vision_segment_max_frames = getattr(
            config, "vision_segment_max_frames", None
        )
        self.embeddings = MiniMaxM3VisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.encoder = MiniMaxM3VisionEncoder(
            config,
            mapping,
            prefix=add_prefix("encoder", prefix),
            mm_attention_backend=mm_attention_backend,
        )

        head_dim = config.hidden_size // config.num_attention_heads
        rotary_dims = 2 * (head_dim // 2)
        self.axis_dim = 2 * ((rotary_dims // 3) // 2)
        inv_freq = 1.0 / (
            config.rope_parameters["rope_theta"]
            ** (torch.arange(0, self.axis_dim, 2, dtype=torch.float32) / self.axis_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embeddings.patch_embedding.weight.device

    def _split_video_segments(
        self,
        grid_thw: Sequence[Sequence[int]],
    ) -> list[tuple[int, int, int]]:
        segments: list[tuple[int, int, int]] = []
        max_frames = self.vision_segment_max_frames
        for raw_t, raw_h, raw_w in grid_thw:
            grid_t, grid_h, grid_w = int(raw_t), int(raw_h), int(raw_w)
            if min(grid_t, grid_h, grid_w) <= 0:
                raise ValueError(
                    f"Invalid MiniMax-M3 vision grid {(grid_t, grid_h, grid_w)}."
                )
            if grid_h % self.spatial_merge_size or grid_w % self.spatial_merge_size:
                raise ValueError(
                    "MiniMax-M3 vision grid height and width must be divisible "
                    f"by {self.spatial_merge_size}."
                )
            if max_frames is None:
                segments.append((grid_t, grid_h, grid_w))
                continue
            for start in range(0, grid_t, max_frames):
                segments.append((min(max_frames, grid_t - start), grid_h, grid_w))
        return segments

    def _rope_for_grid(self, grid_t: int, grid_h: int, grid_w: int) -> torch.Tensor:
        merge = self.spatial_merge_size
        tokens_per_frame = grid_h * grid_w
        temporal = (
            torch.arange(grid_t, device=self.device)
            .unsqueeze(1)
            .expand(-1, tokens_per_frame)
            .reshape(-1)
        )

        height = (
            torch.arange(grid_h, device=self.device).unsqueeze(1).expand(-1, grid_w)
        )
        width = torch.arange(grid_w, device=self.device).unsqueeze(0).expand(grid_h, -1)
        reordered_shape = (
            grid_h // merge,
            merge,
            grid_w // merge,
            merge,
        )
        height = height.reshape(reordered_shape).permute(0, 2, 1, 3)
        width = width.reshape(reordered_shape).permute(0, 2, 1, 3)
        height = height.unsqueeze(0).expand(grid_t, -1, -1, -1, -1).reshape(-1)
        width = width.unsqueeze(0).expand(grid_t, -1, -1, -1, -1).reshape(-1)

        coordinates = torch.stack((temporal, height, width), dim=-1).float()
        frequencies = (coordinates.unsqueeze(-1) * self.inv_freq).reshape(
            coordinates.shape[0],
            -1,
        )
        frequencies = torch.cat((frequencies, frequencies), dim=-1)
        return frequencies

    def _prepare_metadata(
        self,
        grid_thw: torch.Tensor | Sequence[Sequence[int]],
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        int,
        list[tuple[int, int, int]],
    ]:
        raw_grid = grid_thw.tolist() if isinstance(grid_thw, torch.Tensor) else grid_thw
        segments = self._split_video_segments(raw_grid)
        sequence_lengths = [
            grid_t * grid_h * grid_w for grid_t, grid_h, grid_w in segments
        ]
        cu_seqlens = torch.tensor(
            [0, *sequence_lengths],
            dtype=torch.int32,
            device=self.device,
        ).cumsum(0)
        frequencies = torch.cat(
            [self._rope_for_grid(*segment) for segment in segments],
            dim=0,
        )
        cos = frequencies.cos().unsqueeze(-2).to(self.dtype)
        sin = frequencies.sin().unsqueeze(-2).to(self.dtype)
        return cu_seqlens, (cos, sin), max(sequence_lengths), segments

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | Sequence[Sequence[int]],
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        hidden_states = self.pre_layrnorm(self.embeddings(pixel_values))
        cu_seqlens, position_embeddings, max_seqlen, segments = self._prepare_metadata(
            grid_thw
        )
        expected_tokens = sum(t * h * w for t, h, w in segments)
        if hidden_states.shape[0] != expected_tokens:
            raise ValueError(
                "MiniMax-M3 vision grid describes "
                f"{expected_tokens} patches, but received {hidden_states.shape[0]}."
            )
        return self.encoder(
            hidden_states,
            cu_seqlens,
            position_embeddings,
            max_seqlen,
        )


class MiniMaxM3VisionTower(nn.Module):
    """Checkpoint-compatible wrapper around the MiniMax-M3 vision transformer."""

    def __init__(
        self,
        config: MiniMaxM3VisionConfig,
        mapping: Mapping,
        prefix: str,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        self.vision_model = MiniMaxM3VisionTransformer(
            config,
            mapping,
            prefix=add_prefix("vision_model", prefix),
            mm_attention_backend=mm_attention_backend,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_model.dtype

    @property
    def device(self) -> torch.device:
        return self.vision_model.device

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Encode flattened patches and return one hidden state per input patch."""

        return self.vision_model(pixel_values, grid_thw)


class MiniMaxM3MultiModalProjector(nn.Module):
    """Project per-patch vision states into the language hidden size."""

    def __init__(
        self,
        vision_hidden_size: int,
        projector_hidden_size: int,
        text_hidden_size: int,
        mapping: Mapping,
        prefix: str,
        bias: bool = True,
    ) -> None:
        super().__init__()
        vision = mapping.vision
        self.linear_1 = ColumnParallelLinear(
            vision_hidden_size,
            projector_hidden_size,
            bias=bias,
            quant_config=None,
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            prefix=add_prefix("linear_1", prefix),
        )
        self.activation = nn.GELU()
        self.linear_2 = RowParallelLinear(
            projector_hidden_size,
            text_hidden_size,
            bias=bias,
            quant_config=None,
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            prefix=add_prefix("linear_2", prefix),
            reduce_results=True,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.activation(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class MiniMaxM3PatchMergeMLP(nn.Module):
    """Merge each spatial patch group into one language-model token."""

    def __init__(
        self,
        spatial_merge_size: int,
        text_hidden_size: int,
        projector_hidden_size: int,
        mapping: Mapping,
        prefix: str,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.group_size = spatial_merge_size**2
        vision = mapping.vision
        self.linear_1 = ColumnParallelLinear(
            text_hidden_size * self.group_size,
            projector_hidden_size,
            bias=bias,
            quant_config=None,
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            prefix=add_prefix("linear_1", prefix),
        )
        self.activation = nn.GELU()
        self.linear_2 = RowParallelLinear(
            projector_hidden_size,
            text_hidden_size,
            bias=bias,
            quant_config=None,
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            prefix=add_prefix("linear_2", prefix),
            reduce_results=True,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        if image_features.shape[0] % self.group_size:
            raise ValueError(
                "MiniMax-M3 projected patch count must be divisible by "
                f"{self.group_size}, got {image_features.shape[0]}."
            )
        image_features = image_features.reshape(
            image_features.shape[0] // self.group_size,
            -1,
        )
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.activation(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class MiniMaxM3SparseForConditionalGeneration(MiniMaxM3SparseForCausalLM):
    """MiniMax-M3 multimodal entry point."""

    def __init__(
        self,
        config: MiniMaxM3Config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
    ) -> None:
        self.vl_config = config
        self.is_multimodal_active = is_multimodal_active
        super().__init__(
            config=config.text_config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )

        if not is_multimodal_active:
            self.vision_tower = None
            self.multi_modal_projector = None
            self.patch_merge_mlp = None
            self.multimodal_embedder = None
            self.image_encoder = None
            self.video_encoder = None
            return

        self.vision_tower = MiniMaxM3VisionTower(
            config.vision_config,
            mapping,
            prefix=add_prefix("vision_tower", prefix),
            mm_attention_backend=mm_attention_backend,
        )
        self.multi_modal_projector = MiniMaxM3MultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            projector_hidden_size=config.projector_hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            mapping=mapping,
            prefix=add_prefix("multi_modal_projector", prefix),
            bias=True,
        )
        self.patch_merge_mlp = MiniMaxM3PatchMergeMLP(
            spatial_merge_size=config.vision_config.spatial_merge_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_size=config.projector_hidden_size,
            mapping=mapping,
            prefix=add_prefix("patch_merge_mlp", prefix),
            bias=True,
        )
        self.multimodal_embedder = MultimodalEmbedder()
        self.image_encoder = self.get_image_feature
        self.video_encoder = self.get_video_feature

    def pad_input_ids(
        self,
        input_ids: list[int],
        mm_inputs: MultimodalInputs,
    ) -> list[int]:
        """Replace media placeholders with content-derived prefix-cache IDs."""

        return pad_input_tokens(input_ids, mm_inputs)

    def _get_vision_feature(
        self,
        items: list[MultimodalDataItem],
        modality: Modality,
    ) -> torch.Tensor:
        if self.vision_tower is None:
            raise RuntimeError("MiniMax-M3 vision tower is disabled.")
        if not items:
            return torch.empty(
                (0, self.config.hidden_size),
                dtype=self.vision_tower.dtype,
                device=self.vision_tower.device,
            )
        grid_name = "video_grid_thw" if modality == Modality.VIDEO else "image_grid_thw"
        pixel_values = torch.cat(
            [item.feature.to(self.vision_tower.device) for item in items],
            dim=0,
        ).to(self.vision_tower.dtype)
        grids = torch.cat([getattr(item, grid_name) for item in items], dim=0)
        hidden_states = self.vision_tower(pixel_values, grids)
        hidden_states = self.multi_modal_projector(hidden_states)
        return self.patch_merge_mlp(hidden_states)

    def get_image_feature(
        self,
        items: list[MultimodalDataItem],
    ) -> torch.Tensor:
        """Encode image items into language-model embeddings."""

        return self._get_vision_feature(items, Modality.IMAGE)

    def get_video_feature(
        self,
        items: list[MultimodalDataItem],
    ) -> torch.Tensor:
        """Encode video items into language-model embeddings."""

        return self._get_vision_feature(items, Modality.VIDEO)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        multimodal_context = kwargs.pop("multimodal_context", None)
        has_multimodal_prefill = (
            multimodal_context is not None
            and multimodal_context.has_extend_inputs()
            and not ctx.forward_mode.is_decode_or_idle()
        )
        if not has_multimodal_prefill:
            return super().forward(
                ctx,
                input_ids,
                positions,
                out_cache_loc,
                **kwargs,
            )
        if self.multimodal_embedder is None:
            raise RuntimeError(
                "MiniMax-M3 received image or video input while running with "
                "--language-model-only."
            )

        input_embeds, model_kwargs = self.multimodal_embedder.apply(
            input_ids=input_ids,
            text_embedding=self.model.embed_tokens,
            ctx=multimodal_context,
            encoders={
                Modality.IMAGE: EncoderSpec(self.image_encoder),
                Modality.VIDEO: EncoderSpec(self.video_encoder),
            },
            multimodal_model=self,
            is_decode_or_idle=ctx.forward_mode.is_decode_or_idle(),
        )
        hidden_states, aux_hidden_states = self.model(
            input_ids,
            positions,
            ctx,
            out_cache_loc,
            input_embeds=input_embeds,
            **model_kwargs,
        )
        logits_metadata = LogitsMetadata.from_forward_context(ctx)
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            logits_metadata,
            aux_hidden_states,
        )

    def _load_non_language_weight(
        self,
        checkpoint_name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> str | None:
        if not self.is_multimodal_active:
            return None

        name = checkpoint_name
        shard_id = None
        for projection, candidate_shard in (
            ("q_proj", "q"),
            ("k_proj", "k"),
            ("v_proj", "v"),
        ):
            marker = f".self_attn.{projection}."
            if marker in name:
                name = name.replace(marker, ".self_attn.qkv_proj.")
                shard_id = candidate_shard
                break
        name = name.replace(".self_attn.out_proj.", ".self_attn.proj.")

        param = params_dict.get(name)
        if param is None:
            raise KeyError(
                f"MiniMax-M3 vision checkpoint parameter {checkpoint_name!r} "
                "has no runtime parameter mapping."
            )
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        if shard_id is None:
            weight_loader(param, loaded_weight)
        else:
            weight_loader(param, loaded_weight, shard_id)
        return name


EntryClass = [MiniMaxM3SparseForConditionalGeneration]
