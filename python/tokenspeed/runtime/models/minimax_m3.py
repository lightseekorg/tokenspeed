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

from collections.abc import Iterable

import torch
from tokenspeed_kernel import (
    minimax_m3_msa_indexer,
    minimax_m3_msa_sparse_attention,
    minimax_m3_topk,
)
from tokenspeed_kernel.ops.activation.triton import swiglu_oai
from tokenspeed_kernel.ops.layernorm.triton import qk_rmsnorm
from torch import nn

from tokenspeed.runtime.configs.minimax_m3_config import (
    MiniMaxM3TextConfig,
    MiniMaxM3VLConfig,
)
from tokenspeed.runtime.configs.paged_cache_spec import FULL_ATTENTION
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_forward_ctx,
    slice_to_real_tokens,
)
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
from tokenspeed.runtime.models.minimax_m3_vision import (
    MiniMaxM3MultiModalProjector,
    MiniMaxM3PatchMergeMLP,
    MiniMaxM3VisionTower,
)
from tokenspeed.runtime.models.utils import validate_attention_partition
from tokenspeed.runtime.moe.expert_location import ModelConfigForExpertLocation
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    MultimodalEmbedder,
    pad_input_tokens,
)
from tokenspeed.runtime.multimodal.encoder_cudagraph import (
    EncoderCudaGraphWrapper,
    VisionEncoderCudaGraphAdapter,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import global_server_args_dict

_MSA_BLOCK_SIZE = 128


def _msa_score_block_upper_bound(
    max_prefix_len: int,
    max_extend_len: int,
    block_size: int,
    page_table_cols: int,
) -> int:
    """Bound batched MSA scoring without exceeding the page-table width."""
    max_seq_len = int(max_prefix_len) + int(max_extend_len)
    estimated_blocks = (max_seq_len + int(block_size) - 1) // int(block_size)
    return min(estimated_blocks, int(page_table_cols))


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
            "routing_method_type": RoutingMethodType.FP32SigmoidBias,
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
            solution="triton",
        )
        self.routed_scaling_factor = config.routed_scaling_factor
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=True,
            custom_routing_function=self._route_topk,
            output_format=self.experts.topk_output_format,
        )
        self.shared_experts = MiniMaxM3MLP(
            config=config,
            intermediate_size=config.shared_intermediate_size,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("shared_experts", prefix),
        )

    def _route_topk(
        self,
        *,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return minimax_m3_topk(
            hidden_states,
            gating_output,
            self.routing_bias,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
            solution="triton",
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
        self.num_index_heads = total_index_heads
        self.head_dim = int(sparse_config["sparse_index_dim"])
        self.index_q_proj = ColumnParallelLinear(
            config.hidden_size,
            total_index_heads * self.head_dim,
            bias=False,
            gather_output=True,
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
        self.num_kv_heads = max(1, config.num_key_value_heads // mapping.attn.tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        sparse_config = config.sparse_attention_config
        self.is_sparse = bool(sparse_config["sparse_attention_freq"][layer_id])
        self.sparse_topk = int(sparse_config["sparse_topk_blocks"])
        self.sparse_block_size = int(sparse_config["sparse_block_size"])
        self.sparse_init_blocks = int(sparse_config["sparse_init_block"])
        self.sparse_local_blocks = int(sparse_config["sparse_local_block"])
        if self.is_sparse and sparse_config["sparse_score_type"] != "max":
            raise ValueError(
                "MiniMax-M3 MSA requires sparse_score_type='max', got "
                f"{sparse_config['sparse_score_type']!r}."
            )
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
        attn_output = self._sparse_attention(
            q,
            k,
            v,
            index_q,
            index_k,
            ctx,
            out_cache_loc,
        )
        output, _ = self.o_proj(attn_output.flatten(1))
        return output

    @staticmethod
    def _select_cache_io(metadata, out_cache_loc):
        """Return one strict full-attention page table and its write slots."""
        if metadata.page_tables is None:
            if metadata.page_table is None:
                raise RuntimeError("MiniMax-M3 MSA requires paged metadata.")
            return metadata.page_table, out_cache_loc
        if metadata.out_cache_locs is None:
            raise RuntimeError("MiniMax-M3 flat paged metadata has no write slots.")
        if FULL_ATTENTION not in metadata.page_tables:
            raise KeyError(
                "MiniMax-M3 flat page tables do not contain "
                f"{FULL_ATTENTION!r}: {sorted(metadata.page_tables)}"
            )
        if FULL_ATTENTION not in metadata.out_cache_locs:
            raise KeyError(
                "MiniMax-M3 flat write slots do not contain "
                f"{FULL_ATTENTION!r}: {sorted(metadata.out_cache_locs)}"
            )
        return (
            metadata.page_tables[FULL_ATTENTION],
            metadata.out_cache_locs[FULL_ATTENTION],
        )

    @break_point
    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Write caches and run MSA eagerly inside a breakable prefill graph."""
        if self.indexer is None:
            raise RuntimeError("Sparse MiniMax-M3 attention requires an indexer.")
        if current_forward_ctx() is not None:
            metadata = ctx.attn_backend.forward_extend_metadata
            if metadata is None:
                raise RuntimeError("MiniMax-M3 graph replay requires prefill metadata.")
            num_real_tokens = metadata.cu_extend_seq_lens_cpu[-1]
            q, k, v, index_q, index_k, out_cache_loc = slice_to_real_tokens(
                num_real_tokens,
                q,
                k,
                v,
                index_q,
                index_k,
                out_cache_loc,
            )

        pool = ctx.token_to_kv_pool
        if ctx.forward_mode.is_decode():
            metadata = ctx.attn_backend.forward_decode_metadata
            if metadata is None:
                raise RuntimeError("MiniMax-M3 MSA decode requires paged metadata.")
            page_table, cache_locs = self._select_cache_io(metadata, out_cache_loc)
            decode_query_len = q.shape[0] // metadata.seq_lens.shape[0]
        elif ctx.forward_mode.is_extend_or_mixed():
            metadata = ctx.attn_backend.forward_extend_metadata
            if metadata is None:
                raise RuntimeError("MiniMax-M3 MSA prefill requires paged metadata.")
            page_table, cache_locs = self._select_cache_io(metadata, out_cache_loc)
            decode_query_len = 0
        else:
            raise RuntimeError(
                f"MiniMax-M3 MSA does not support forward mode {ctx.forward_mode}."
            )

        pool.set_kv_buffer(
            self.attn,
            cache_locs,
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

        if decode_query_len:
            selected_blocks = minimax_m3_msa_indexer(
                index_q,
                index_k,
                index_k_cache,
                cache_locs,
                page_table,
                metadata.seq_lens,
                topk=self.sparse_topk,
                scale=self.indexer.head_dim**-0.5,
                init_blocks=self.sparse_init_blocks,
                local_blocks=self.sparse_local_blocks,
                decode_query_len=decode_query_len,
                max_blocks=page_table.shape[1],
                solution="triton",
            )
            return minimax_m3_msa_sparse_attention(
                q,
                key_cache,
                value_cache,
                selected_blocks,
                page_table,
                metadata.seq_lens,
                scale=self.head_dim**-0.5,
                decode_query_len=decode_query_len,
                solution="triton",
            )

        if ctx.forward_mode.is_extend_or_mixed():
            max_blocks = _msa_score_block_upper_bound(
                metadata.max_extend_prefix_len,
                metadata.max_extend_seq_len,
                self.sparse_block_size,
                page_table.shape[1],
            )
            selected_blocks = minimax_m3_msa_indexer(
                index_q,
                index_k,
                index_k_cache,
                cache_locs,
                page_table,
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
            return minimax_m3_msa_sparse_attention(
                q,
                key_cache,
                value_cache,
                selected_blocks,
                page_table,
                metadata.seq_lens,
                scale=self.head_dim**-0.5,
                cu_seqlens_q=metadata.cu_extend_seq_lens,
                prefix_lens=metadata.extend_prefix_lens,
                max_query_len=metadata.max_extend_seq_len,
                solution="triton",
            )
        raise AssertionError("unreachable MiniMax-M3 MSA mode")


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

    def _load_vision_weight(
        self,
        checkpoint_name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> str | None:
        """Load one visual checkpoint tensor, if this model has a vision path."""

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
                loaded_name = self._load_vision_weight(
                    checkpoint_name,
                    loaded_weight,
                    params_dict,
                )
                if loaded_name is not None:
                    loaded_params.add(loaded_name)
                continue
            else:
                name = checkpoint_name

            if self.encoder_only:
                continue

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
        config: MiniMaxM3VLConfig,
    ) -> ModelConfigForExpertLocation:
        return ModelConfigForExpertLocation(
            num_layers=config.text_config.num_hidden_layers,
            num_logical_experts=config.text_config.num_local_experts,
            num_groups=None,
        )


class MiniMaxM3SparseForConditionalGeneration(MiniMaxM3SparseForCausalLM):
    """MiniMax-M3 vision-language model with native MSA text decoding."""

    def __init__(
        self,
        config: MiniMaxM3VLConfig,
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
            encoder_only=bool(config.encoder_only),
        )

        if not is_multimodal_active:
            self.vision_tower = None
            self.multi_modal_projector = None
            self.patch_merge_mlp = None
            self.vision_embedder = None
            self.image_encoder = None
            return

        # The released MXFP8 checkpoint keeps the vision path unquantized
        # (FP32 patch convolution, BF16 blocks/projectors). Do not pass the text
        # quantization config through this boundary.
        self.vision_tower = MiniMaxM3VisionTower(
            config.vision_config,
            mapping=mapping,
            prefix=add_prefix("vision_tower", prefix),
            mm_attention_backend=mm_attention_backend,
        )
        self.multi_modal_projector = MiniMaxM3MultiModalProjector(
            config,
            mapping=mapping,
            prefix=add_prefix("multi_modal_projector", prefix),
        )
        self.patch_merge_mlp = MiniMaxM3PatchMergeMLP(
            config,
            mapping=mapping,
            prefix=add_prefix("patch_merge_mlp", prefix),
        )
        self.vision_embedder = MultimodalEmbedder()
        self.image_encoder = self.get_image_feature

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        """Replace image placeholders with content-derived prefix-cache IDs."""

        return pad_input_tokens(input_ids, mm_inputs)

    @staticmethod
    def _image_grid(item: MultimodalDataItem) -> torch.Tensor:
        """Read and validate the explicit SMG grid contract for one image."""

        try:
            grid = item.model_specific_data["image_grid_thw"]
        except KeyError as error:
            raise ValueError(
                "MiniMax-M3 image input is missing image_grid_thw."
            ) from error
        if not isinstance(grid, torch.Tensor):
            raise TypeError(
                "MiniMax-M3 image_grid_thw must be a torch.Tensor, got "
                f"{type(grid).__name__}."
            )
        if grid.dtype == torch.bool or grid.is_floating_point() or grid.is_complex():
            raise TypeError(
                "MiniMax-M3 image_grid_thw must use an integer dtype, got "
                f"{grid.dtype}."
            )
        if grid.ndim != 2 or grid.shape[1] != 3:
            raise ValueError(
                "MiniMax-M3 image_grid_thw must have shape [num_images, 3], "
                f"got {tuple(grid.shape)}."
            )
        if bool((grid <= 0).any()):
            raise ValueError("MiniMax-M3 image_grid_thw values must be positive.")
        return grid

    def _validate_image_item(
        self, item: MultimodalDataItem, grid: torch.Tensor
    ) -> None:
        if not isinstance(item.feature, torch.Tensor):
            raise TypeError(
                "MiniMax-M3 image features must be materialized torch tensors "
                f"before encoding, got {type(item.feature).__name__}."
            )
        if item.feature.ndim != 2:
            raise ValueError(
                "MiniMax-M3 pixel_values must be 2D, got shape "
                f"{tuple(item.feature.shape)}."
            )

        patch_counts = grid.to(dtype=torch.int64).prod(dim=1)
        expected_patches = int(patch_counts.sum().item())
        if item.feature.shape[0] != expected_patches:
            raise ValueError(
                "MiniMax-M3 pixel row count must equal image_grid_thw product, "
                f"got {item.feature.shape[0]} and {expected_patches}."
            )

        merge = self.vl_config.vision_config.spatial_merge_size
        if bool((grid[:, 1:] % merge != 0).any()):
            raise ValueError(
                "MiniMax-M3 image grid height and width must be divisible by "
                f"spatial_merge_size={merge}."
            )
        if not item.offsets or len(item.offsets) != grid.shape[0]:
            raise ValueError(
                "MiniMax-M3 requires one placeholder offset range per image grid."
            )
        expected_tokens = (patch_counts // (merge**2)).tolist()
        actual_tokens = [end - start + 1 for start, end in item.offsets]
        if actual_tokens != expected_tokens:
            raise ValueError(
                "MiniMax-M3 image placeholder lengths do not match merged grid "
                f"tokens: got {actual_tokens}, expected {expected_tokens}."
            )

    def pre_encode(
        self, items: list[MultimodalDataItem]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate image payloads and apply the Conv3d patch embedding."""

        if self.vision_tower is None:
            raise RuntimeError("MiniMax-M3 vision tower is not initialized.")
        if not items:
            raise ValueError("MiniMax-M3 image encoder received an empty batch.")

        grids = []
        pixels = []
        for item in items:
            if item.modality is not Modality.IMAGE:
                raise ValueError(
                    "MiniMax-M3 vision tower currently accepts image items only."
                )
            grid = self._image_grid(item)
            self._validate_image_item(item, grid)
            grids.append(grid)
            pixels.append(item.feature)

        device = self.vision_tower.device
        pixel_values = torch.cat(pixels, dim=0).to(
            device=device,
            non_blocking=True,
        )
        grid_thw = torch.cat(grids, dim=0).to(
            device=device, dtype=torch.int64, non_blocking=True
        )
        return self.vision_tower.prepare_patch_embed(pixel_values, grid_thw), grid_thw

    def post_encode(
        self, encoder_outs: list[torch.Tensor], grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Project patches, then merge each processor-grouped 2x2 block."""

        del grid_thw
        if self.multi_modal_projector is None or self.patch_merge_mlp is None:
            raise RuntimeError("MiniMax-M3 multimodal projectors are not initialized.")
        vision_features = torch.cat(encoder_outs, dim=0)
        if vision_features.ndim == 3 and vision_features.shape[1] == 1:
            vision_features = vision_features.squeeze(1)
        projected = self.multi_modal_projector(vision_features)
        return self.patch_merge_mlp(projected)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        """Encode a batch of dynamically sized images into LM embeddings."""

        tokens, grid_thw = self.pre_encode(items)
        metadata = self.vision_tower.prepare_metadata(grid_thw)
        encoded = self.vision_tower.forward_blocks(tokens, metadata)
        output = self.post_encode([encoded], grid_thw)
        expected_tokens = sum(
            sum(end - start + 1 for start, end in item.offsets or []) for item in items
        )
        if output.shape[0] != expected_tokens:
            raise RuntimeError(
                "MiniMax-M3 vision output token count does not match placeholders: "
                f"got {output.shape[0]}, expected {expected_tokens}."
            )
        return output

    def make_encoder_cudagraph_wrappers(
        self, mapping: Mapping
    ) -> dict[str, EncoderCudaGraphWrapper]:
        """Build the standard token-preserving ViT CUDA-graph wrapper."""

        if self.vision_tower is None:
            return {}
        return {
            "image_encoder": EncoderCudaGraphWrapper(
                adapter=VisionEncoderCudaGraphAdapter(
                    tower=self.vision_tower,
                    pre_encode=self.pre_encode,
                    post_encode=self.post_encode,
                    out_div=1,
                    merge=self.vision_tower.spatial_merge_size,
                    input_feature_shape=(1, self.vision_tower.hidden_size),
                    modality_name="image",
                    capture_tp_size=mapping.vision.tp_size,
                    capture_tp_group=mapping.vision.tp_group,
                    # The patch Conv3d is FP32, but pre_encode applies the
                    # BF16 pre-LayerNorm before entering the captured blocks.
                    input_dtype=self.vision_tower.dtype,
                ),
                # The processor emits 16 patches at its minimum resolution and
                # at most 2304 patches per image at 672 x 672.
                budget_range=(16, 2304),
                max_batch_size=10,
            )
        }

    def get_input_embeddings(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError(
                "MiniMax-M3 input embeddings are unavailable in encoder-only mode."
            )
        return self.model.embed_tokens

    @torch.no_grad()
    def multimodal_input_embeds(
        self,
        input_ids: torch.Tensor,
        ctx: ForwardContext,
        multimodal_context,
    ) -> torch.Tensor | None:
        """Assemble text and vision embeddings for a multimodal prefill."""

        if (
            multimodal_context is None
            or self.vision_embedder is None
            or not multimodal_context.has_extend_inputs()
            or ctx.forward_mode.is_decode_or_idle()
        ):
            return None
        input_embeds, model_kwargs = self.vision_embedder.apply(
            input_ids=input_ids,
            text_embedding=self.get_input_embeddings(),
            ctx=multimodal_context,
            encoders={Modality.IMAGE: EncoderSpec(self.image_encoder)},
            multimodal_model=self,
            is_decode_or_idle=ctx.forward_mode.is_decode_or_idle(),
        )
        if model_kwargs:
            raise RuntimeError("MiniMax-M3 multimodal path must remain embeds-only.")
        return input_embeds

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.encoder_only:
            raise RuntimeError("MiniMax-M3 encoder-only workers cannot run LM forward.")
        multimodal_context = kwargs.pop("multimodal_context", None)
        input_embeds = self.multimodal_input_embeds(input_ids, ctx, multimodal_context)
        if input_embeds is not None:
            kwargs["input_embeds"] = input_embeds
        return super().forward(ctx, input_ids, positions, out_cache_loc, **kwargs)

    def _load_vision_weight(
        self,
        checkpoint_name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> str | None:
        """Load one unquantized visual tensor without buffering the checkpoint."""

        if not self.is_multimodal_active:
            return None

        stacked_attention = {
            ".self_attn.q_proj": (".self_attn.qkv_proj", "q"),
            ".self_attn.k_proj": (".self_attn.qkv_proj", "k"),
            ".self_attn.v_proj": (".self_attn.qkv_proj", "v"),
        }
        name = checkpoint_name.replace("vision_tower.vision_model.", "vision_tower.", 1)
        name = name.replace(".encoder.layers.", ".layers.")
        name = name.replace(".self_attn.out_proj", ".self_attn.proj")

        for weight_name, (param_name, shard_id) in stacked_attention.items():
            if weight_name not in name:
                continue
            mapped_name = name.replace(weight_name, param_name)
            param = params_dict.get(mapped_name)
            if param is None:
                raise KeyError(
                    f"MiniMax-M3 vision parameter {checkpoint_name!r} maps to "
                    f"missing runtime parameter {mapped_name!r}."
                )
            param.weight_loader(param, loaded_weight, shard_id)
            return mapped_name

        param = params_dict.get(name)
        if param is None:
            raise KeyError(
                f"MiniMax-M3 vision checkpoint parameter {checkpoint_name!r} "
                "has no runtime mapping."
            )
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        return name


EntryClass = [MiniMaxM3SparseForConditionalGeneration]
