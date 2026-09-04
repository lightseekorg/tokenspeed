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

"""Kimi-K3 MTP (NextN) draft model.

The MTP checkpoint ships one extra layer (index ``num_hidden_layers``) with
the DeepSeek NextN structure — ``embed_tokens`` / ``enorm`` / ``hnorm`` /
``eh_proj`` / ``shared_head`` — around a full K3 decoder layer whose
attention is NoPE-MLA (no KDA state in the draft) and whose FFN is the K3
latent MoE. The layer also ships AttnRes mix weights; with zero residual
snapshots at draft depth the mix softmax has a single candidate, so those
weights are mathematically inert and the draft runs as a plain pre-norm
layer.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
from torch import nn

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.layers.moe.loader import build_moe_checkpoint_loader
from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3DraftAttentionMLA,
    _prepare_mla_kv_b_proj_weights,
)
from tokenspeed.runtime.models.kimi_k3 import (
    KimiLinearMLAAttention,
    KimiLinearMoE,
    sigmoid_mul,
)
from tokenspeed.runtime.utils import add_prefix

logger = logging.getLogger(__name__)


class KimiK3DraftAttentionMLA(KimiLinearMLAAttention, DeepseekV3DraftAttentionMLA):
    """K3 gated NoPE-MLA with the draft row-narrowing ``_attn``.

    MRO: Kimi's projections/gate wrap DeepseekV3DraftAttentionMLA's ``_attn``
    (which narrows the attention output to the live rows on the first draft
    step); the gate must be narrowed the same way before the fused
    sigmoid-mul.
    """

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        comm_manager,
        block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        if self.use_output_gate:
            q, latent_cache, gate, absorbed_query = self._project_q_latent_gated(
                hidden_states, ctx, comm_manager, block_scale
            )
        else:
            q, latent_cache = self._project_q_latent(
                hidden_states, ctx, comm_manager, block_scale
            )
            gate = None
            absorbed_query = None
        attn_output = self._attn(
            positions,
            q,
            latent_cache,
            ctx,
            absorbed_query=absorbed_query,
        )
        if gate is not None:
            if attn_output.shape[0] != gate.shape[0]:
                gate = gate.index_select(0, ctx.gather_ids)
            attn_output = sigmoid_mul(attn_output, gate)
        output, _ = self.o_proj(attn_output)
        return output


class KimiK3DraftDecoderLayer(nn.Module):
    """The single MTP decoder layer: plain pre-norm MLA + K3 MoE.

    The checkpoint's AttnRes mix weights for this layer are registered so
    loading is 1:1, but the forward is a plain pre-norm residual block (see
    module docstring).
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        model_scope: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping

        self.self_attn = KimiK3DraftAttentionMLA(
            config=config,
            mapping=mapping,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            layer_id=0,
            prefix=add_prefix("self_attn", prefix),
            reduce_attn_results=True,
            alt_stream=alt_stream,
        )
        self.block_sparse_moe = KimiLinearMoE(
            config=config,
            mapping=mapping,
            layer_index=0,
            model_scope=model_scope,
            quant_config=quant_config,
            prefix=add_prefix("block_sparse_moe", prefix),
            alt_stream=alt_stream,
        )
        # The old underscored flag writes were dead; making that intent effective needs a spec-decode run.
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Shipped but inert at draft depth (zero snapshots).
        from tokenspeed.runtime.layers.linear import ReplicatedLinear

        self.self_attention_res_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention_res_proj = ReplicatedLinear(
            config.hidden_size, 1, bias=False
        )
        self.mlp_res_proj = ReplicatedLinear(config.hidden_size, 1, bias=False)

        # pre_attn_comm (no-op in AllReduce mode) + token-count queries only;
        # attention reduces in o_proj and the MoE reduces on its lane.
        self.comm_manager = CommManager(
            mapping,
            0,
            is_moe=True,
            prev_is_moe=False,
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(
            ctx
        )
        residual = hidden_states
        if not ctx.forward_mode.is_idle():
            attn_out = self.self_attn(
                positions=positions,
                hidden_states=self.input_layernorm(hidden_states),
                ctx=ctx,
                comm_manager=self.comm_manager,
            )
            if (
                ctx.draft_narrowing is not None
                and attn_out.shape[0] != residual.shape[0]
            ):
                residual = residual.index_select(0, ctx.gather_ids)
            residual = residual + attn_out
        prefix = self.block_sparse_moe(
            self.post_attention_layernorm(residual),
            residual,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            ctx=ctx,
        )
        return prefix.view(residual.shape)


class KimiK3ModelNextN(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.vocab_size = config.vocab_size

        from tokenspeed.runtime.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        self.alt_stream = torch.cuda.Stream()
        decoder_scope = add_prefix("decoder", prefix)
        self.decoder = KimiK3DraftDecoderLayer(
            config=config,
            mapping=mapping,
            model_scope=decoder_scope,
            quant_config=quant_config,
            prefix=decoder_scope,
            alt_stream=self.alt_stream,
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        input_embeds: torch.Tensor | None = None,
        captured_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if captured_hidden_states is None:
            raise ValueError("Kimi-K3 NextN requires captured_hidden_states.")
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        hidden_states = self.eh_proj(
            torch.cat(
                (self.enorm(hidden_states), self.hnorm(captured_hidden_states)),
                dim=-1,
            )
        )
        hidden_states = self.decoder(positions, hidden_states, ctx)
        return self.shared_head.norm(hidden_states), None


class KimiK3NextNForCausalLM(nn.Module):
    """Text-side NextN causal LM (draft worker)."""

    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.model = KimiK3ModelNextN(
            config, mapping, quant_config, prefix=add_prefix("model", prefix)
        )
        # The checkpoint ships a dedicated draft head (shared_head.head).
        from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
        from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        # Distributed argmax over the vocab-parallel head: every TP rank must
        # emit the SAME draft token ids (per-rank shard-local argmax silently
        # diverges and caps chain acceptance at whatever rank 0's vocab shard
        # happens to contain).
        self.logits_processor = LogitsProcessor(
            config,
            skip_all_gather=mapping.attn.has_dp,
            do_argmax=True,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_hot_token_id(self):
        return None

    def set_embed_and_head(self, embed, head):
        # DeepSeek MTP convention: the draft shares the target's embedding
        # and lm head (the checkpoint's per-layer copies are skipped).
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        captured_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states, _ = self.model(
            input_ids,
            positions,
            ctx,
            captured_hidden_states=captured_hidden_states,
        )
        logits_metadata = LogitsMetadata.from_forward_context(ctx)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, logits_metadata
        )

    def checkpoint_weight_name_filter(self, name: str) -> bool:
        """Shard preselection for ``load_weights`` (see DefaultModelLoader).

        Accepts a superset of the checkpoint names ``load_weights`` consumes:
        everything under the NextN layer prefix.
        """
        return name.startswith(f"model.layers.{self.config.num_hidden_layers}.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        config = self.config
        nextn_prefix = f"model.layers.{config.num_hidden_layers}."
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="w1", up_proj_name="w3", down_proj_name="w2"
            ),
            num_experts=config.num_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )

        for name, loaded_weight in weights:
            if not name.startswith(nextn_prefix):
                continue
            name = name[len(nextn_prefix) :]

            # NextN scaffolding around the decoder layer.
            if name.startswith(("enorm", "hnorm", "eh_proj", "embed_tokens")):
                name = "model." + name
            elif name == "shared_head.head.weight":
                name = "lm_head.weight"
            elif name.startswith("shared_head.norm"):
                name = "model." + name
            else:
                name = "model.decoder." + name

            if "experts." in name and name.endswith(".weight_packed"):
                name = name[: -len(".weight_packed")] + ".weight"

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ".experts." in name and name not in params_dict:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if moe_loader.matches(name):
                    moe_loader.load(name, loaded_weight)
                    continue

                if ".g_proj" in name:
                    mapped = name.replace("g_proj", "fused_qkv_a_proj_with_mqa")
                    param = params_dict.get(mapped)
                    if param is not None:
                        begin = (
                            config.q_lora_rank
                            + config.kv_lora_rank
                            + config.qk_rope_head_dim
                        )
                        shard = loaded_weight.shape[0] // self.mapping.attn.tp_size
                        rank = self.mapping.attn.tp_rank
                        param.weight_loader(
                            param,
                            loaded_weight[rank * shard : (rank + 1) * shard],
                            begin_size=begin,
                        )
                        continue

                if "q_a_proj" in name or "kv_a_proj_with_mqa" in name:
                    if "q_a_proj" in name:
                        begin_size = 0
                        mapped = name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                    else:
                        begin_size = config.q_lora_rank
                        mapped = name.replace(
                            "kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa"
                        )
                    param = params_dict.get(mapped)
                    if param is None:
                        continue
                    param.weight_loader(param, loaded_weight, begin_size=begin_size)
                    continue

                param = params_dict.get(name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        self.post_load_weights()

    def post_load_weights(self) -> None:
        attn = self.model.decoder.self_attn
        attn.w_kc, attn.w_vc = _prepare_mla_kv_b_proj_weights(
            attn.kv_b_proj.weight, attn
        )


class KimiK3ForConditionalGenerationNextN(nn.Module):
    """Draft-worker entry: strips the ``language_model.`` checkpoint prefix."""

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        text_config = getattr(config, "text_config", None) or config
        self.config = config
        self.language_model = KimiK3NextNForCausalLM(
            text_config, mapping, quant_config, prefix=prefix
        )
        self.logits_processor = self.language_model.logits_processor
        self.lm_head = self.language_model.lm_head

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def get_hot_token_id(self):
        return None

    def set_embed_and_head(self, embed, head):
        self.language_model.set_embed_and_head(embed, head)

    def forward(self, *args, **kwargs):
        return self.language_model.forward(*args, **kwargs)

    def checkpoint_weight_name_filter(self, name: str) -> bool:
        """Shard preselection for ``load_weights`` (see DefaultModelLoader)."""
        lm_prefix = "language_model."
        if not name.startswith(lm_prefix):
            return False
        return self.language_model.checkpoint_weight_name_filter(name[len(lm_prefix) :])

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        lm_prefix = "language_model."
        self.language_model.load_weights(
            (name[len(lm_prefix) :], w)
            for name, w in weights
            if name.startswith(lm_prefix)
        )


EntryClass = [KimiK3ForConditionalGenerationNextN]
