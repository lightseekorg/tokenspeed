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

"""Inference-only MiniMax-M3 language model for the M3-VL checkpoint."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
from tokenspeed_kernel import (
    minimax_m3_msa_indexer,
    minimax_m3_msa_sparse_attention,
)
from tokenspeed_kernel.ops.activation.triton import swiglu_oai
from tokenspeed_kernel.ops.layernorm.triton import qk_rmsnorm
from torch import nn

from tokenspeed.runtime.configs.minimax_m3_config import (
    MiniMaxM3TextConfig,
    MiniMaxM3VLConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
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
from tokenspeed.runtime.models.base import (
    BaseCausalLM,
    BaseDecoderLayer,
    BaseTransformerModel,
)
from tokenspeed.runtime.models.utils import validate_attention_partition
from tokenspeed.runtime.moe.expert_location import ModelConfigForExpertLocation
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)

_MSA_BLOCK_SIZE = 128


class MiniMaxM3MLP(nn.Module):
    """Dense MiniMax-M3 MLP using the SwiGLU-OAI activation."""

    def __init__(
        self,
        config: MiniMaxM3TextConfig,
        intermediate_size: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        dense = mapping.dense
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            tp_rank=dense.tp_rank,
            tp_size=dense.tp_size,
            tp_group=dense.tp_group,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=dense.tp_rank,
            tp_size=dense.tp_size,
            tp_group=dense.tp_group,
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
        config: MiniMaxM3TextConfig,
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
        if config.n_shared_experts != 1:
            raise ValueError(
                "MiniMax-M3 currently requires exactly one shared expert, got "
                f"{config.n_shared_experts}."
            )
        if not config.use_routing_bias:
            raise ValueError("MiniMax-M3 requires correction-bias routing.")

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
            solution="triton" if quant_config is not None else None,
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
        config: MiniMaxM3TextConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        sparse_config = config.sparse_attention_config
        total_index_heads = int(sparse_config["sparse_num_index_heads"])
        if total_index_heads != config.num_key_value_heads:
            raise ValueError(
                "MiniMax-M3 requires one index-query head per GQA group: "
                f"index_heads={total_index_heads}, kv_heads={config.num_key_value_heads}."
            )
        if total_index_heads % mapping.attn.tp_size:
            raise ValueError(
                f"{total_index_heads} index heads cannot be sharded over "
                f"TP={mapping.attn.tp_size}."
            )
        self.num_index_heads = total_index_heads // mapping.attn.tp_size
        self.head_dim = int(sparse_config["sparse_index_dim"])
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
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=config.rotary_dim,
            max_position=config.max_position_embeddings,
            base=int(config.rope_theta),
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
    """Dense or native MSA attention, selected from the layer-frequency config."""

    def __init__(
        self,
        config: MiniMaxM3TextConfig,
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
        self.num_kv_heads = max(
            1, config.num_key_value_heads // mapping.attn.tp_size
        )
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        sparse_config = config.sparse_attention_config
        self.is_sparse = bool(sparse_config["sparse_attention_freq"][layer_id])
        self.sparse_topk = int(sparse_config["sparse_topk_blocks"])
        self.sparse_block_size = int(sparse_config["sparse_block_size"])
        self.sparse_init_blocks = int(sparse_config["sparse_init_block"])
        self.sparse_local_blocks = int(sparse_config["sparse_local_block"])
        if self.is_sparse and self.sparse_block_size != _MSA_BLOCK_SIZE:
            raise ValueError(
                f"MiniMax-M3 MSA requires block size {_MSA_BLOCK_SIZE}, got "
                f"{self.sparse_block_size}."
            )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=config.rotary_dim,
            max_position=config.max_position_embeddings,
            base=int(config.rope_theta),
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
        if self.is_sparse:
            return self._forward_sparse(
                positions,
                hidden_states,
                q,
                k,
                v,
                ctx,
                out_cache_loc,
            )
        attn_output = self.attn(
            q,
            k,
            v,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
        )
        output, _ = self.o_proj(attn_output)
        return output

    def _forward_sparse(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        if self.indexer is None:
            raise RuntimeError("Sparse MiniMax-M3 attention requires an indexer.")
        index_q, index_k = self.indexer(positions, hidden_states)
        pool = ctx.token_to_kv_pool
        pool.set_kv_buffer(
            self.attn,
            out_cache_loc,
            k,
            v,
            self.attn.k_scale,
            self.attn.v_scale,
        )
        key_cache = pool.get_key_buffer(self.attn.layer_id).view(
            -1,
            self.sparse_block_size,
            self.num_kv_heads,
            self.head_dim,
        )
        value_cache = pool.get_value_buffer(self.attn.layer_id).view_as(key_cache)
        key_cache = key_cache.permute(0, 2, 1, 3)
        value_cache = value_cache.permute(0, 2, 1, 3)
        index_k_cache = pool.get_index_k_buffer(self.attn.layer_id)

        if ctx.forward_mode.is_decode():
            metadata = ctx.attn_backend.forward_decode_metadata
            if metadata is None or metadata.page_table is None:
                raise RuntimeError("MiniMax-M3 MSA decode requires paged metadata.")
            decode_query_len = q.shape[0] // metadata.seq_lens.shape[0]
            selected_blocks = minimax_m3_msa_indexer(
                index_q,
                index_k,
                index_k_cache,
                out_cache_loc,
                metadata.page_table,
                metadata.seq_lens,
                topk=self.sparse_topk,
                scale=self.indexer.head_dim**-0.5,
                init_blocks=self.sparse_init_blocks,
                local_blocks=self.sparse_local_blocks,
                decode_query_len=decode_query_len,
                max_blocks=metadata.page_table.shape[1],
                solution="triton",
            )
            attn_output = minimax_m3_msa_sparse_attention(
                q,
                key_cache,
                value_cache,
                selected_blocks,
                metadata.page_table,
                metadata.seq_lens,
                scale=self.head_dim**-0.5,
                decode_query_len=decode_query_len,
                solution="triton",
            )
        elif ctx.forward_mode.is_extend_or_mixed():
            metadata = ctx.attn_backend.forward_extend_metadata
            if metadata is None or metadata.page_table is None:
                raise RuntimeError("MiniMax-M3 MSA prefill requires paged metadata.")
            max_seq_len = (
                metadata.max_extend_prefix_len + metadata.max_extend_seq_len
            )
            max_blocks = (max_seq_len + self.sparse_block_size - 1) // self.sparse_block_size
            selected_blocks = minimax_m3_msa_indexer(
                index_q,
                index_k,
                index_k_cache,
                out_cache_loc,
                metadata.page_table,
                metadata.seq_lens,
                topk=self.sparse_topk,
                scale=self.indexer.head_dim**-0.5,
                init_blocks=self.sparse_init_blocks,
                local_blocks=self.sparse_local_blocks,
                cu_seqlens_q=metadata.cu_extend_seq_lens,
                prefix_lens=metadata.extend_prefix_lens,
                max_query_len=metadata.max_extend_seq_len,
                max_blocks=max_blocks,
                solution="triton",
            )
            attn_output = minimax_m3_msa_sparse_attention(
                q,
                key_cache,
                value_cache,
                selected_blocks,
                metadata.page_table,
                metadata.seq_lens,
                scale=self.head_dim**-0.5,
                cu_seqlens_q=metadata.cu_extend_seq_lens,
                prefix_lens=metadata.extend_prefix_lens,
                max_query_len=metadata.max_extend_seq_len,
                solution="triton",
            )
        else:
            raise RuntimeError(
                f"MiniMax-M3 MSA does not support forward mode {ctx.forward_mode}."
            )

        output, _ = self.o_proj(attn_output.flatten(1))
        return output


class MiniMaxM3DecoderLayer(BaseDecoderLayer[MiniMaxM3TextConfig]):
    """Decoder layer selected as dense or MoE by ``moe_layer_freq``."""

    @property
    def is_moe_layer(self) -> bool:
        return bool(self.config.moe_layer_freq[self.layer_id])

    @property
    def previous_is_moe_layer(self) -> bool:
        if self.layer_id == 0:
            return False
        return bool(self.config.moe_layer_freq[self.layer_id - 1])

    def resolve_norm(self) -> nn.Module:
        return GemmaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def resolve_attn(self, prefix: str) -> nn.Module:
        return MiniMaxM3Attention(
            config=self.config,
            mapping=self.mapping,
            layer_id=self.layer_id,
            quant_config=self.quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

    def resolve_mlp(self, prefix: str) -> nn.Module:
        if self.is_moe_layer:
            return MiniMaxM3SparseMoeBlock(
                config=self.config,
                mapping=self.mapping,
                quant_config=self.quant_config,
                layer_index=self.layer_id,
                prefix=add_prefix("block_sparse_moe", prefix),
            )
        return MiniMaxM3MLP(
            config=self.config,
            intermediate_size=self.config.dense_intermediate_size,
            mapping=self.mapping,
            quant_config=self.quant_config,
            prefix=add_prefix("mlp", prefix),
        )


class MiniMaxM3Model(BaseTransformerModel):
    """MiniMax-M3 decoder-only text backbone."""

    layer_cls = MiniMaxM3DecoderLayer

    def resolve_norm(self, config: MiniMaxM3TextConfig) -> nn.Module:
        return GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MiniMaxM3SparseForCausalLM(BaseCausalLM):
    """MiniMax-M3 text model with checkpoint-compatible weight loading."""

    model_cls = MiniMaxM3Model
    fall_back_to_pt_during_load = False

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
                continue
            else:
                name = checkpoint_name

            if name.startswith("model.mtp."):
                continue
            if "rotary_emb.inv_freq" in name:
                continue

            name = name.replace(".block_sparse_moe", ".mlp")
            name = name.replace(".e_score_correction_bias", ".routing_bias")
            name = name.replace(".self_attn.index_q_proj", ".self_attn.indexer.index_q_proj")
            name = name.replace(".self_attn.index_k_proj", ".self_attn.indexer.index_k_proj")
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
        config: MiniMaxM3VLConfig,
    ) -> ModelConfigForExpertLocation:
        return ModelConfigForExpertLocation(
            num_layers=config.text_config.num_hidden_layers,
            num_logical_experts=config.text_config.num_local_experts,
            num_groups=None,
        )


class MiniMaxM3SparseForConditionalGeneration(MiniMaxM3SparseForCausalLM):
    """Language-only MiniMax-M3 entry point with native MSA."""

    def __init__(
        self,
        config: MiniMaxM3VLConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
    ) -> None:
        del mm_attention_backend
        if is_multimodal_active:
            raise ValueError(
                "MiniMax-M3 is currently language-only. Start TokenSpeed with "
                "--language-model-only."
            )
        self.is_multimodal_active = False
        super().__init__(
            config=config.text_config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )


EntryClass = [MiniMaxM3SparseForConditionalGeneration]
