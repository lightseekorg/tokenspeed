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

"""Inference-only GLM5 NextN (MTP) Speculative Decoding.

GLM5.1 ships a single MTP / NextN predict layer in the base checkpoint
(``model.layers.{num_hidden_layers}`` with ``enorm``/``hnorm``/``eh_proj``/
``shared_head.norm`` plus a full decoder layer). This module exposes that layer
as a draft model for ``--speculative-algorithm MTP``.

It mirrors ``deepseek_nextn.py`` (GLM5 fully inherits the DeepSeek V3 stack),
with two GLM5-specific differences:

1. The single nextn decoder is a :class:`GlmMoeDsaDecoderLayer` (with the GLM
   DSA lightning indexer), not a plain ``DeepseekV3DecoderLayer``.
2. ``load_weights`` additionally routes the indexer projection weights through
   the fused-indexer loaders inherited from :class:`GlmMoeDsaForCausalLM`
   (FP8 ``wk`` dequant + ``wk_weights_proj`` fusion).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.utils import block_dequant
from tokenspeed.runtime.layers.utils import (
    CP_METADATA,
    ENABLE_CP,
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
)
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.glm5 import (
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaForCausalLM,
    pad_fused_qkv_a_proj_weight_for_fp8_blockscale,
)

logger = logging.getLogger(__name__)


def _mask_glm5_nextn_position_zero_embeddings(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    # The first MTP slot is seeded by target hidden states; its token embedding
    # must not contribute to the draft input.
    return torch.where(positions.unsqueeze(-1) == 0, 0, hidden_states)


class GlmMoeDsaModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.vocab_size = config.vocab_size

        # Shared with the target model via set_embed_and_head(); the local
        # weight is a placeholder that gets replaced after construction.
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        # GLM5-specific: the nextn decoder is a DSA layer (lightning indexer),
        # not a plain DeepseekV3DecoderLayer.
        self.alt_stream = torch.cuda.Stream()
        self.decoder = GlmMoeDsaDecoderLayer(
            config,
            0,
            mapping=self.mapping,
            quant_config=quant_config,
            is_nextn=True,
            alt_stream=self.alt_stream,
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        captured_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        hidden_states = _mask_glm5_nextn_position_zero_embeddings(
            hidden_states,
            positions,
        )
        if captured_hidden_states is None:
            if hidden_states.shape[0] == 0:
                # DP idle ranks pair the active ranks' collectives with a
                # zero-token draft forward that carries no target hidden
                # states; run the layer with an empty hidden tensor so the
                # MoE collectives still execute.
                captured_hidden_states = hidden_states
            else:
                raise ValueError("GLM5 NextN requires captured_hidden_states.")

        hidden_states = self.eh_proj(
            torch.cat(
                (
                    self.enorm(hidden_states),
                    self.hnorm(captured_hidden_states),
                ),
                dim=-1,
            )
        )

        residual = None
        if CP_METADATA:
            hidden_states = cp_split_and_rebuild_data(
                hidden_states,
                CP_METADATA.value.split_list,
                CP_METADATA.value.zigzag_index,
            )
            positions = cp_split_and_rebuild_data(
                positions, CP_METADATA.value.split_list, CP_METADATA.value.zigzag_index
            )
        hidden_states, residual = self.decoder(
            positions,
            hidden_states,
            ctx,
            out_cache_loc,
            residual,
        )

        if not ctx.forward_mode.is_idle():
            if not ENABLE_CP:
                hidden_states = self.decoder.comm_manager.final_norm(
                    hidden_states, residual, ctx, self.shared_head.norm
                )
            else:
                hidden_states, _ = self.shared_head.norm(hidden_states, residual)
        if CP_METADATA:
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                CP_METADATA.value,
                self.mapping.attn.tp_rank,
                self.mapping.attn.tp_group,
            )
        return hidden_states, None


class GlmMoeDsaForCausalLMNextN(GlmMoeDsaForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.mapping = mapping

        # FP4 quantization is not used for the NextN draft model; the NextN MoE
        # weights are stored in BF16, so the draft runs entirely in BF16.
        if quant_config is not None and quant_config.get_name() == "nvfp4":
            logger.warning(
                "Overriding GlmMoeDsaForCausalLMNextN quant config: "
                "FP4 quantization not used for NextN draft model."
            )
            quant_config = None

        self.quant_config = quant_config

        self.model = GlmMoeDsaModelNextN(
            config, mapping=self.mapping, quant_config=quant_config
        )

        if self.mapping.attn.has_dp:
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                tp_rank=self.mapping.attn.tp_rank,
                tp_size=self.mapping.attn.tp_size,
                tp_group=self.mapping.attn.tp_group,
            )
            self.logits_processor = LogitsProcessor(
                config,
                tp_rank=self.mapping.attn.tp_rank,
                tp_size=self.mapping.attn.tp_size,
                tp_group=self.mapping.attn.tp_group,
            )

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        captured_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states, _ = self.model(
            input_ids,
            positions,
            ctx,
            out_cache_loc,
            captured_hidden_states=captured_hidden_states,
        )
        logits_metadata = LogitsMetadata.from_forward_context(ctx)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, logits_metadata
        )

    def get_hot_token_id(self):
        # MTP drafts every vocab token; the hot-token-id mechanism is an
        # EAGLE3-only optimization.
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dim when q_lora_rank
        # is set.
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None

        nextn_spec_weight_names = [
            "shared_head.norm",
            "eh_proj",
            "enorm",
            "hnorm",
        ]

        params_dict = dict(self.named_parameters())
        modules_dict = dict(self.named_modules())
        # GLM DSA fused-indexer loader state (see GlmMoeDsaForCausalLM).
        pending_fp8_wk: dict[str, dict[str, torch.Tensor]] = {}
        loaded_fused_indexer_shards: dict[str, set[int]] = {}

        # MoE expert weights/scales are handled by the checkpoint loader.
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
            ),
            num_experts=self.config.n_routed_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )
        for name, loaded_weight in weights:
            # --- Locate + remap the single nextn layer (model.layers.{N}) ---
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                nextn_layer_prefix = "model.layers.0"
                if num_nextn_layers != self.config.num_hidden_layers:
                    if name.startswith("model.layers"):
                        name_list = name.split(".")
                        if (
                            len(name_list) >= 3
                            and int(name_list[2]) >= self.config.num_hidden_layers
                        ):
                            nextn_layer_prefix = "model.layers." + str(name_list[2])
                        else:
                            continue
                if not name.startswith(nextn_layer_prefix):
                    continue
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

            # Embed + final head are shared from the target via
            # set_embed_and_head(); skip loading them here.
            if "shared_head.head" in name or "embed_tokens" in name:
                continue

            is_decoder = True
            for weight_name in nextn_spec_weight_names:
                if weight_name in name:
                    name = name.replace(nextn_layer_prefix, "model")
                    is_decoder = False
                    break
            if is_decoder:
                name = name.replace(nextn_layer_prefix, "model.decoder")

            if "rotary_emb.inv_freq" in name:
                continue

            # --- GLM5-specific: DSA indexer projection weights ---
            # After remap these live under model.decoder.self_attn.indexer.*;
            # route them through the fused-indexer loaders (FP8 wk dequant +
            # wk_weights_proj fusion), mirroring GlmMoeDsaForCausalLM.
            if ".indexer." in name:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = self.get_param(params_dict, name)
                if param is not None:
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                self._try_load_fused_indexer_projection(
                    name=name,
                    loaded_weight=loaded_weight,
                    params_dict=params_dict,
                    modules_dict=modules_dict,
                    pending_fp8_wk=pending_fp8_wk,
                    loaded_shards=loaded_fused_indexer_shards,
                )
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Experts are handled by moe_loader below; skip the stacked
                # rewrite for them (otherwise the name gets double-mangled).
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if moe_loader.matches(name):
                    moe_loader.load(name, loaded_weight)
                    continue

                if fuse_qkv_a_proj and (
                    "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                ):
                    cached_a_proj[name] = loaded_weight
                    q_a_proj_name = (
                        name
                        if "q_a_proj" in name
                        else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                    )
                    kv_a_proj_name = (
                        name
                        if "kv_a_proj_with_mqa" in name
                        else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                    )

                    if (
                        q_a_proj_name in cached_a_proj
                        and kv_a_proj_name in cached_a_proj
                    ):
                        q_a_proj_weight = cached_a_proj[q_a_proj_name]
                        kv_a_proj_weight = cached_a_proj[kv_a_proj_name]
                        fused_weight = torch.cat(
                            [q_a_proj_weight, kv_a_proj_weight], dim=0
                        )

                        if "q_a_proj" in name:
                            param_name = name.replace(
                                "q_a_proj", "fused_qkv_a_proj_with_mqa"
                            )
                        else:
                            param_name = name.replace(
                                "kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa"
                            )
                        param = params_dict[param_name]

                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, fused_weight)
                        cached_a_proj.pop(q_a_proj_name)
                        cached_a_proj.pop(kv_a_proj_name)
                else:
                    # Owned-expert weights were already consumed by
                    # moe_loader.load() above (matches() == True). Anything that
                    # still looks like an expert weight here belongs to an
                    # expert this rank does not own under ep_size > 1 — skip it
                    # instead of KeyError'ing on params_dict.
                    if ".mlp.experts." in name:
                        continue
                    param = self.get_param(params_dict, name)
                    if param is None:
                        continue
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
        self.post_load_weights()

    def post_load_weights(self):
        self_attn = self.model.decoder.self_attn
        # The NextN draft decoder carries the same fused QKV-A projection as the
        # main model, so it hits the same FP8 block-scale GEMM NaN when N is not
        # 128-aligned. Without this the draft attention is NaN and the drafter
        # accepts zero tokens (avg_accept_len stays at 1.0).
        pad_fused_qkv_a_proj_weight_for_fp8_blockscale(self_attn)
        if (
            hasattr(self.quant_config, "weight_block_size")
            and (self.quant_config.weight_block_size is not None)
            and self_attn.kv_b_proj.weight.dtype
            in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
            )
        ):
            weight_block_size = self.quant_config.weight_block_size
            dtype = torch.get_default_dtype()
            w = block_dequant(
                self_attn.kv_b_proj.weight,
                self_attn.kv_b_proj.weight_scale_inv,
                weight_block_size,
            ).to(dtype)
        else:
            w = self_attn.kv_b_proj.weight

        w_kc, w_vc = w.unflatten(
            0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
        ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
        self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        self_attn.w_vc = w_vc.contiguous().transpose(1, 2)


EntryClass = [GlmMoeDsaForCausalLMNextN]
