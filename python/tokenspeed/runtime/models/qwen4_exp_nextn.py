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

"""Qwen4-Exp multi-token-prediction draft model."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
from torch import nn

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import (
    ForwardContext,
    report_collective_sizing,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.utils import should_exclude_quant_module
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.models.qwen4_exp import (
    Qwen4ExpAttentionDecoderLayer,
    Qwen4ExpModel,
    load_qwen4_exp_weights,
)
from tokenspeed.runtime.utils import add_prefix


def _resolve_mtp_quant_config(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    if quant_config is not None and should_exclude_quant_module(
        "mtp.layers.0", quant_config.exclude_modules
    ):
        return None
    return quant_config


def _mtp_index_sharing_enabled(config) -> bool:
    text_config = getattr(config, "text_config", config)
    return bool(getattr(text_config, "index_share_for_mtp_iteration", False))


def _build_mtp_lm_head(config, mapping: Mapping, quant_config, prefix: str):
    if mapping.attn.has_dp:
        return ReplicatedLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
        )
    return ParallelLMHead(
        config.vocab_size,
        config.hidden_size,
        quant_config=quant_config,
        prefix=add_prefix("model.shared_head.head", prefix),
        tp_rank=mapping.attn.tp_rank,
        tp_size=mapping.attn.tp_size,
        tp_group=mapping.attn.tp_group,
    )


class Qwen4ExpDraftAttentionDecoderLayer(Qwen4ExpAttentionDecoderLayer):
    """Qwen4-Exp draft attention that removes dead catch-up rows on step zero."""

    def _apply_correction(self, ctx: ForwardContext) -> None:
        seq_lens = ctx.draft_seq_lens_buf
        if seq_lens is None or ctx.accept_lengths is None:
            return
        if ctx.num_extends >= ctx.bs:
            return
        correction = (
            ctx.attn_backend.spec_num_tokens - ctx.accept_lengths[ctx.num_extends :]
        ).to(seq_lens.dtype)
        seq_lens[ctx.num_extends : ctx.bs].sub_(correction).clamp_(min=1)
        ctx.attn_backend.advance_draft_forward_metadata(seq_lens[: ctx.bs])

    def _attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor | None,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        if ctx.accept_lengths is None:
            return super()._attn(q, k, v, gate, ctx, out_cache_loc)
        from tokenspeed_kernel.ops.activation.triton import sigmoid_mul

        self._apply_correction(ctx)
        q = q.index_select(0, ctx.gather_ids)
        if gate is not None:
            gate = gate.index_select(0, ctx.gather_ids)
        decode_ctx = replace(ctx, forward_mode=ForwardMode.DECODE)
        output = self.attn(
            q,
            k,
            v,
            decode_ctx,
            out_cache_loc,
            record_kv_cache=not ctx.forward_mode.is_decode_or_idle(),
        )
        if gate is not None:
            sigmoid_mul(output, gate)
        return output

    def _qsa_attention(self, **kwargs) -> torch.Tensor:
        ctx = kwargs["ctx"]
        if ctx.accept_lengths is None:
            return super()._qsa_attention(**kwargs)
        self._apply_correction(ctx)
        kwargs["q"] = kwargs["q"].index_select(0, ctx.gather_ids)
        kwargs["topk_indices"] = kwargs["topk_indices"].index_select(0, ctx.gather_ids)
        if kwargs["gate"] is not None:
            kwargs["gate"] = kwargs["gate"].index_select(0, ctx.gather_ids)
        return super()._qsa_attention(**kwargs)

    def forward(self, *args, **kwargs):
        ctx = kwargs["ctx"]
        hidden_states = kwargs["hidden_states"]
        input_ids = kwargs["input_ids"]
        mixed, residuals = self._prepare_attention(hidden_states, input_ids, ctx)
        attention_output = (
            mixed
            if ctx.forward_mode.is_idle()
            else self.self_attention(
                kwargs["positions"], mixed, ctx, kwargs["out_cache_loc"]
            )
        )
        if ctx.accept_lengths is not None and not ctx.forward_mode.is_idle():
            residuals = tuple(
                value.index_select(0, ctx.gather_ids) for value in residuals
            )
        hidden_states = self._finish_attention(attention_output, residuals, ctx)
        return self._run_mlp(hidden_states, ctx), None


class Qwen4ExpDraftModel(Qwen4ExpModel):
    ATTENTION_LAYER_CLS = Qwen4ExpDraftAttentionDecoderLayer

    def __init__(self, config, mapping, quant_config=None, prefix: str = ""):
        if config.num_hidden_layers != 1:
            raise ValueError("Qwen4-Exp MTP requires exactly one decoder layer")
        super().__init__(config, mapping, quant_config, prefix)


class Qwen4ExpForCausalLMNextN(nn.Module):
    """One-layer Qwen4-Exp MTP head used by TokenSpeed's MTP drafter."""

    compute_dsa_topk_first_step = True

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = copy.deepcopy(getattr(config, "text_config", config))
        quant_config = _resolve_mtp_quant_config(quant_config)
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]
        config.ple_layer_ids = []
        self.config = config
        self.index_share_for_mtp_iteration = _mtp_index_sharing_enabled(config)
        self.mapping = mapping
        self.quant_config = quant_config
        self.hidden_size = int(config.hidden_size)
        self.hc_count = int(config.hc_count)

        self.pre_fc_norm_embedding = GemmaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            self.hidden_size * self.hc_count, eps=config.rms_norm_eps
        )
        self.fc_embedding = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("mtp.fc_embedding", prefix),
        )
        self.fc_hidden = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("mtp.fc_hidden", prefix),
        )
        self.model = Qwen4ExpDraftModel(
            config,
            mapping,
            quant_config,
            prefix=add_prefix("mtp", prefix),
        )
        for indexer in self.model.qsa_indexers:
            indexer.share_topk_for_mtp_iteration = self.index_share_for_mtp_iteration
        self.lm_head = _build_mtp_lm_head(config, mapping, quant_config, prefix)
        self.logits_processor = LogitsProcessor(
            config,
            skip_all_gather=mapping.attn.has_dp,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )

    def get_hot_token_id(self):
        return None

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @staticmethod
    def prepare_dsa_topk_for_mtp_decode(
        dsa_topk: tuple[Any | None, Any | None],
        gather_ids: torch.Tensor,
        *,
        num_prefill_rows: int = 0,
    ) -> tuple[Any | None, Any | None]:
        """Select one target-aligned QSA top-k row per draft request."""

        del num_prefill_rows
        prefill_topk, decode_topk = dsa_topk
        if decode_topk is None or decode_topk.shape[0] == 0:
            return dsa_topk
        return prefill_topk, decode_topk.index_select(0, gather_ids)

    def _fuse_inputs(
        self, input_embeds: torch.Tensor, captured_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.pre_fc_norm_embedding(input_embeds)
        embedded, _ = self.fc_embedding(embedded)
        hidden = self.pre_fc_norm_hidden(captured_hidden_states).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        hidden, _ = self.fc_hidden(hidden)
        return (hidden + embedded.unsqueeze(-2)).flatten(-2)

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        captured_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ):
        del kwargs
        if captured_hidden_states is None and not ctx.forward_mode.is_idle():
            raise ValueError("Qwen4-Exp MTP requires captured HC hidden states")
        if ctx.forward_mode.is_idle():
            fused = self.model.embed_tokens.weight.new_empty(
                (0, self.hc_count * self.hidden_size)
            )
        else:
            if input_embeds is not None:
                raise ValueError("Qwen4-Exp MTP does not accept input_embeds")
            fused = self._fuse_inputs(
                self.model.embed_tokens(input_ids), captured_hidden_states
            )
        with report_collective_sizing(ctx, ctx.bs, ctx.global_bs):
            hidden_states, aux_hidden_states = self.model(
                input_ids,
                positions,
                ctx,
                out_cache_loc,
                input_embeds=fused,
            )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            LogitsMetadata.from_forward_context(ctx),
            aux_hidden_states,
        )

    def checkpoint_weight_name_filter(self, name: str) -> bool:
        return "mtp" in name or "shared_head" in name

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        del is_mtp

        def remap():
            for name, weight in weights:
                if "mtp" not in name and "shared_head" not in name:
                    continue
                if name.startswith(
                    (
                        "mtp.fc_embedding.",
                        "mtp.fc_hidden.",
                        "mtp.pre_fc_norm_embedding.",
                        "mtp.pre_fc_norm_hidden.",
                    )
                ):
                    name = name.replace("mtp.", "", 1)
                elif "shared_head" in name:
                    name = "lm_head.weight"
                else:
                    name = name.replace("mtp.", "model.", 1)
                yield name, weight

        return load_qwen4_exp_weights(
            self,
            self.config,
            self.mapping,
            remap(),
            include_visual=False,
        )


EntryClass = [Qwen4ExpForCausalLMNextN]
