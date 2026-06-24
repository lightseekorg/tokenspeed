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

"""Inference-only GLM5 NextN speculative decoding."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.dsa.utils import workspace_indices_to_kv_slots
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

_NEXTN_SPEC_WEIGHT_NAMES = (
    "shared_head.norm",
    "eh_proj",
    "enorm",
    "hnorm",
)

_STACKED_PARAMS_MAPPING = (
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)


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

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

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
        hidden_states = torch.where(positions.unsqueeze(-1) == 0, 0, hidden_states)
        if captured_hidden_states is None:
            if not ctx.forward_mode.is_idle():
                raise ValueError("GLM5 NextN requires captured_hidden_states.")
            captured_hidden_states = hidden_states

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
                positions,
                CP_METADATA.value.split_list,
                CP_METADATA.value.zigzag_index,
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
    compute_dsa_topk_first_step = True
    draft_first_step_reduce_for_catchup = True

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.mapping = mapping

        if quant_config is not None and quant_config.get_name() == "nvfp4":
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
            skip_all_gather=self.mapping.attn.has_dp,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )

    @staticmethod
    def _apply_first_step_correction(ctx: ForwardContext) -> None:
        seq_lens_buf = ctx.draft_seq_lens_buf
        accept_lengths = ctx.accept_lengths
        if (
            not ctx.draft_first_step_reduce
            or seq_lens_buf is None
            or accept_lengths is None
        ):
            return
        num_extends = ctx.num_extends
        if num_extends >= ctx.bs:
            return
        correction = (
            ctx.attn_backend.spec_num_tokens - accept_lengths[num_extends:]
        ).to(seq_lens_buf.dtype)
        seq_lens_buf[num_extends : ctx.bs].sub_(correction).clamp_(min=1)
        ctx.attn_backend.advance_draft_forward_metadata(seq_lens_buf[: ctx.bs])

    @staticmethod
    def prepare_dsa_topk_for_mtp_decode(
        dsa_topk: tuple[Any | None, Any | None],
        gather_ids: torch.Tensor,
        *,
        num_prefill_rows: int = 0,
    ) -> tuple[Any | None, Any | None]:
        prefill_topk, decode_topk = dsa_topk
        if decode_topk is None:
            return dsa_topk
        topk_indices = decode_topk.topk_indices
        topk_lens = decode_topk.topk_lens
        if topk_indices.shape[0] == 0:
            return dsa_topk
        if num_prefill_rows <= 0 and topk_indices.shape[0] <= gather_ids.numel():
            return dsa_topk
        if num_prefill_rows <= 0:
            selected_indices = topk_indices.index_select(0, gather_ids)
            selected_lens = topk_lens.index_select(0, gather_ids)
        else:
            if prefill_topk is None:
                return dsa_topk
            num_prefill_rows = min(int(num_prefill_rows), gather_ids.numel())
            prefill_row_ids = gather_ids[:num_prefill_rows]
            decode_row_ids = gather_ids[num_prefill_rows:]
            selected_prefill_indices = workspace_indices_to_kv_slots(
                prefill_topk.workspace_indices.index_select(0, prefill_row_ids),
                prefill_topk.kv_workspace_slots,
            ).to(device=topk_indices.device, dtype=topk_indices.dtype)
            selected_prefill_lens = prefill_topk.topk_lens.index_select(
                0,
                prefill_row_ids,
            ).to(
                device=topk_lens.device,
                dtype=topk_lens.dtype,
            )
            if decode_row_ids.numel() > 0:
                selected_decode_indices = topk_indices.index_select(0, decode_row_ids)
                selected_decode_lens = topk_lens.index_select(0, decode_row_ids)
                selected_indices = torch.cat(
                    [selected_prefill_indices, selected_decode_indices],
                    dim=0,
                )
                selected_lens = torch.cat(
                    [selected_prefill_lens, selected_decode_lens],
                    dim=0,
                )
            else:
                selected_indices = selected_prefill_indices
                selected_lens = selected_prefill_lens
        selected_decode_topk = replace(
            decode_topk,
            topk_indices=selected_indices,
            topk_lens=selected_lens,
        )
        return prefill_topk, selected_decode_topk

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
        self._apply_first_step_correction(ctx)
        logits_metadata = LogitsMetadata.from_forward_context(ctx)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, logits_metadata
        )

    def get_hot_token_id(self) -> None:
        return None

    def _nextn_layer_prefix(self, name: str) -> str | None:
        if not hasattr(self.config, "num_nextn_predict_layers"):
            raise ValueError("num_nextn_predict_layers is not in the config")
        if self.config.num_nextn_predict_layers != 1:
            raise ValueError("Only 1 nextn layer is supported")

        if self.config.num_nextn_predict_layers == self.config.num_hidden_layers:
            prefix = "model.layers.0"
            return prefix if name.startswith(prefix) else None

        if not name.startswith("model.layers."):
            return None
        name_parts = name.split(".")
        if len(name_parts) < 3:
            return None
        try:
            layer_id = int(name_parts[2])
        except ValueError:
            return None
        if layer_id < self.config.num_hidden_layers:
            return None
        return f"model.layers.{layer_id}"

    def _map_checkpoint_name(self, raw_name: str) -> str | None:
        nextn_layer_prefix = self._nextn_layer_prefix(raw_name)
        if nextn_layer_prefix is None:
            return None
        if "shared_head.head" in raw_name or "embed_tokens" in raw_name:
            return None
        if "rotary_emb.inv_freq" in raw_name:
            return None

        if any(weight_name in raw_name for weight_name in _NEXTN_SPEC_WEIGHT_NAMES):
            return raw_name.replace(nextn_layer_prefix, "model")
        return raw_name.replace(nextn_layer_prefix, "model.decoder")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj: dict[str, torch.Tensor] | None = {} if fuse_qkv_a_proj else None

        params_dict = dict(self.named_parameters())
        modules_dict = dict(self.named_modules())
        pending_fp8_wk: dict[str, dict[str, torch.Tensor]] = {}
        loaded_fused_indexer_shards: dict[str, set[int]] = {}

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
        for raw_name, loaded_weight in weights:
            name = self._map_checkpoint_name(raw_name)
            if name is None:
                continue

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

            for param_name, weight_name, shard_id in _STACKED_PARAMS_MAPPING:
                if weight_name not in name:
                    continue
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

                if cached_a_proj is not None and (
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

    def post_load_weights(self) -> None:
        self_attn = self.model.decoder.self_attn
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
