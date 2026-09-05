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

"""GLM-5.3-Flash single-layer MTP drafter."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from tokenspeed.runtime.configs.glm53_flash_config import Glm53FlashConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import (
    ForwardContext,
    report_collective_sizing,
)
from tokenspeed.runtime.layers.attention.dsa.utils import (
    _prepare_dsa_topk_for_mtp_decode,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.utils import block_dequant
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.models.deepseek_v3 import _prepare_mla_kv_b_proj_weights
from tokenspeed.runtime.models.glm53_flash import (
    Glm53FlashAttention,
    Glm53FlashDecoderLayer,
    Glm53FlashMoE,
    load_glm53_flash_text_weights,
)


class Glm53FlashModelNextN(nn.Module):
    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        decoder_prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.decoder = Glm53FlashDecoderLayer(
            config=config,
            layer_id=0,
            mapping=mapping,
            quant_config=quant_config,
            prefix=decoder_prefix,
            alt_stream=alt_stream,
            is_nextn=True,
        )
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        captured_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        hidden_states = self.embed_tokens(input_ids)
        if captured_hidden_states is None:
            if not ctx.forward_mode.is_idle():
                raise ValueError(
                    "GLM-5.3-Flash MTP requires captured target hidden states"
                )
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
        hidden_states, residual = self.decoder(
            positions,
            hidden_states,
            ctx,
            None,
        )
        if not ctx.forward_mode.is_idle():
            hidden_states, _ = self.decoder.comm_manager.final_norm(
                hidden_states,
                residual,
                ctx,
                self.shared_head.norm,
            )
        return hidden_states, None


class Glm53FlashForConditionalGenerationNextN(nn.Module):
    """Single trained GLM-5.3-Flash draft layer stored after the base stack."""

    compute_dsa_topk_first_step = True

    def __init__(
        self,
        config: Glm53FlashConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        del prefix
        self.vl_config = config
        self.config = config.text_config
        self.mapping = mapping
        self.quant_config = quant_config
        self.model = Glm53FlashModelNextN(
            self.config,
            mapping,
            quant_config=quant_config,
            decoder_prefix=f"model.layers.{self.config.num_hidden_layers}",
        )
        if mapping.attn.has_dp:
            self.lm_head = ReplicatedLinear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=False,
            )
        else:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                tp_rank=mapping.attn.tp_rank,
                tp_size=mapping.attn.tp_size,
                tp_group=mapping.attn.tp_group,
            )
        self.logits_processor = LogitsProcessor(
            self.config,
            skip_all_gather=mapping.attn.has_dp,
            do_argmax=True,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )

    def get_embed_and_head(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(
        self,
        embed: torch.Tensor,
        head: torch.Tensor,
    ) -> None:
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

    prepare_dsa_topk_for_mtp_decode = staticmethod(_prepare_dsa_topk_for_mtp_decode)

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        captured_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        # The first NextN pass is the draft-extend/catch-up pass over the full
        # target verify window and keeps the target's post-verify sequence
        # lengths for it; shrinking them to the accepted prefix would shift the
        # attention window whenever not every draft was accepted. The accepted
        # prefix is never published here — the drafter's step loop publishes it
        # for the subsequent one-token draft steps.
        with report_collective_sizing(ctx, ctx.bs, ctx.global_bs):
            hidden_states, _ = self.model(
                input_ids,
                positions,
                ctx,
                captured_hidden_states=captured_hidden_states,
            )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            LogitsMetadata.from_forward_context(ctx),
        )

    def get_hot_token_id(self) -> None:
        return None

    def _normalize_checkpoint_name(self, name: str) -> str:
        if name.startswith("model.language_model."):
            return "model." + name[len("model.language_model.") :]
        return name

    def _nextn_layer_prefix(self, raw_name: str) -> str | None:
        name = self._normalize_checkpoint_name(raw_name)
        if not name.startswith("model.layers."):
            return None
        parts = name.split(".")
        if len(parts) < 3 or not parts[2].isdigit():
            return None
        layer_id = int(parts[2])
        if not (
            self.config.num_hidden_layers
            <= layer_id
            < self.config.num_hidden_layers + self.config.num_nextn_predict_layers
        ):
            return None
        return f"model.layers.{layer_id}"

    def checkpoint_weight_name_filter(self, name: str) -> bool:
        return self._nextn_layer_prefix(name) is not None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        def nextn_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for raw_name, loaded_weight in weights:
                nextn_prefix = self._nextn_layer_prefix(raw_name)
                if nextn_prefix is None:
                    continue
                name = self._normalize_checkpoint_name(raw_name)
                if "shared_head.head" in name or "embed_tokens" in name:
                    continue
                if any(
                    marker in name
                    for marker in (
                        "shared_head.norm",
                        ".eh_proj.",
                        ".enorm.",
                        ".hnorm.",
                    )
                ):
                    name = name.replace(nextn_prefix, "model")
                else:
                    name = name.replace(nextn_prefix, "model.decoder")
                yield name, loaded_weight

        load_glm53_flash_text_weights(self, nextn_weights())
        self.post_load_weights()

    def post_load_weights(self) -> None:
        attention = self.model.decoder.self_attn
        if not isinstance(attention, Glm53FlashAttention):
            raise RuntimeError("GLM-5.3-Flash MTP decoder must use sparse MLA")
        weight = attention.kv_b_proj.weight
        weight_block_size = getattr(self.quant_config, "weight_block_size", None)
        if weight_block_size is not None and weight.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            if not hasattr(attention.kv_b_proj, "weight_scale_inv"):
                raise RuntimeError(
                    "kv_b_proj.weight_scale_inv is required for block FP8 dequant."
                )
            weight = block_dequant(
                weight,
                attention.kv_b_proj.weight_scale_inv,
                weight_block_size,
            ).to(torch.get_default_dtype())
        attention.w_kc, attention.w_vc = _prepare_mla_kv_b_proj_weights(
            weight, attention
        )
        attention._absorbed_kv_b_version = attention.kv_b_proj.weight._version
        if isinstance(self.model.decoder.mlp, Glm53FlashMoE):
            experts = self.model.decoder.mlp.experts
            experts.process_weights_after_loading(experts)


EntryClass = [Glm53FlashForConditionalGenerationNextN]
