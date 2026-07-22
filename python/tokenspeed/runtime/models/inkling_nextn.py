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

"""Inference-only Inkling MTP / NextN draft model.

The checkpoint ships an 8-depth MTP chain (``mtp_config``) under
``model.mtp.*``: per depth a ``hidden_norm``/``embed_norm`` pair, an
``input_proj`` fusing ``concat[hidden_norm(prev_hidden),
embed_norm(token_embed)]`` (hidden FIRST — the reverse of DeepSeek's
``eh_proj``), and one full Inkling transformer block (full attention +
dense MLP, both with sconv), plus ONE ``chain_norm`` shared by all depths.
The (possibly chain-normalized) block output is both the logits input and
the hidden chained to the next depth; ``chain_hidden_post_norm`` in the
model config decides whether the post-norm applies when the checkpoint
ships no ``chain_norm`` weight. Embedding and LM head are shared with the
target model (``set_embed_and_head``).

Depth blocks reuse ``InklingDecoderLayer`` unmodified: a copied text config
with ``local_layer_ids`` from ``mtp_config`` (checkpoints since 4d71c3ea mix
SWA and full-attention depths) and ``dense_mlp_idx = num depths`` (dense MLP)
steers the base constructor onto the right branches, and local layer ids
0..N-1 index the draft worker's own KV/conv pools. Depths beyond
``--speculative-num-steps`` never run and are pruned from construction,
weight loading, and pool sizing (see ``inkling_mtp_text_config``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

import torch
from torch import nn

from tokenspeed.runtime.configs.inkling_config import (
    InklingMMConfig,
    InklingModelConfig,
    inkling_mtp_text_config,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.models.inkling import (
    _KV_REPLICATED_SUFFIXES,
    InklingDecoderLayer,
    InklingForConditionalGeneration,
    _load_block_param,
    _make_param_loader,
)
from tokenspeed.runtime.utils import add_prefix

logger = logging.getLogger(__name__)


def _draft_text_config(text_config: InklingModelConfig) -> InklingModelConfig:
    """Text config specialized for the MTP depth blocks.

    Depths use their own local/full attention pattern from ``mtp_config``
    and dense MLPs throughout; local layer ids 0..N-1 index the draft
    worker's own KV and conv pools. ModelConfig already swaps this in (with
    steps-pruning) for the draft worker; the call here is an idempotent
    defense for direct construction (unit tests).
    """
    return inkling_mtp_text_config(text_config)


class InklingMultiTokenPredictorLayer(nn.Module):
    """One MTP depth: norm-concat-project fusion + one Inkling block."""

    def __init__(
        self,
        config: InklingModelConfig,
        mapping: Mapping,
        layer_id: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.hidden_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.embed_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.input_proj = ReplicatedLinear(
            2 * hidden_size,
            hidden_size,
            bias=False,
            prefix=add_prefix("input_proj", prefix),
        )
        self.transformer_block = InklingDecoderLayer(
            config,
            mapping,
            layer_id=layer_id,
            quant_config=None,
            prefix=add_prefix("transformer_block", prefix),
        )

    def forward(
        self,
        token_embeds: torch.Tensor,
        previous_hidden: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Checkpoint order: [hidden, embed] (mtp_model.py reference).
        fused, _ = self.input_proj(
            torch.cat(
                (self.hidden_norm(previous_hidden), self.embed_norm(token_embeds)),
                dim=-1,
            )
        )
        return self.transformer_block(fused, None, ctx, out_cache_loc)


class InklingMultiTokenPredictor(nn.Module):
    def __init__(
        self,
        config: InklingModelConfig,
        mapping: Mapping,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.num_mtp_layers = config.num_hidden_layers
        self.embed_tokens = VocabParallelEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.padded_vocab_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            InklingMultiTokenPredictorLayer(
                config,
                mapping,
                layer_id=idx,
                prefix=add_prefix(f"layers.{idx}", prefix),
            )
            for idx in range(self.num_mtp_layers)
        )
        # Post-depth chain norm, per the ``chain_hidden_post_norm`` model
        # config (weight from the checkpoint's ``chain_norm`` if shipped,
        # else its init weight of 1).
        self.chain_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.chain_hidden_post_norm
            else None
        )
        # Base-model embed_norm before the depth's own embed_norm (weight
        # loaded from the base checkpoint's llm.embed_norm): the head is
        # trained on the normalized embeddings the base decoder consumes.
        self.base_embed_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_embed_norm
            else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        captured_hidden_states: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if input_embeds is None:
            # Lookup + base embed_norm (matching what the base decoder
            # consumes), then each depth applies its own embed_norm.
            input_embeds = self.embed_tokens(input_ids)
            if self.base_embed_norm is not None:
                input_embeds = self.base_embed_norm(input_embeds)
        layer = self.layers[spec_step_idx % self.num_mtp_layers]
        hidden, residual = layer(
            input_embeds, captured_hidden_states, ctx, out_cache_loc
        )
        if ctx.forward_mode.is_idle():
            return hidden
        if residual is None:
            if self.chain_norm is None:
                return hidden
            return self.chain_norm(hidden)
        if self.chain_norm is None:
            return hidden + residual
        hidden, _ = self.chain_norm(hidden, residual)
        return hidden


class InklingForConditionalGenerationNextN(nn.Module):
    # Catch-up runs the full padded window; ctx.gather_ids narrows to one row per request.
    draft_first_step_reduce_for_catchup = True
    # Full-sequence depths: re-run each depth over the whole verify window
    # (mtp.py window mode); the trained dataflow, not optional.
    draft_multi_depth_windows = True
    # Prompt catch-up: at EXTEND rounds run depths
    # 1..steps-1 over the prompt rows too (inputs shifted d+1, chaining the
    # previous depth's full-row hidden), so every used depth gets dense
    # prompt KV and sconv prompt state.
    draft_extend_depth_catchup = True
    # Provisional-tail recompute: decode windows carry D=steps-1
    # lookback rows so the previous round's provisional tail entries (written
    # from then-unverified drafts) are recomputed from committed tokens.
    # Kept as an A/B knob for further experiments:
    # INKLING_MTP_DECODE_LOOKBACK=0 reverts (depth-d KV/conv then keeps
    # ~d/<a>*(1-p) stale slots).
    draft_decode_lookback = os.environ.get("INKLING_MTP_DECODE_LOOKBACK", "1") != "0"

    def __init__(
        self,
        config: InklingMMConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
    ) -> None:
        del is_multimodal_active, mm_attention_backend  # draft is text-only
        super().__init__()
        self.config = config
        self.mapping = mapping
        text_config = config.get_text_config()
        if text_config.num_nextn_predict_layers <= 0:
            raise ValueError(
                "Inkling NextN requires mtp_config.num_nextn_predict_layers > 0"
            )
        # The MTP head is BF16 in both the BF16 and NVFP4 checkpoints.
        if quant_config is not None:
            logger.warning(
                "Overriding InklingForConditionalGenerationNextN quant config: "
                "the Inkling MTP head is unquantized (BF16) in all checkpoints."
            )
        self.quant_config = None
        self.text_config = _draft_text_config(text_config)

        self.model = InklingMultiTokenPredictor(
            self.text_config, mapping, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            self.text_config.padded_vocab_size,
            self.text_config.hidden_size,
            org_num_embeddings=self.text_config.padded_vocab_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("lm_head", prefix),
        )
        # do_argmax off (as base model): the distributed-argmax path skips the padded-vocab masking below.
        self.logits_processor = LogitsProcessor(
            self.text_config,
            skip_all_gather=mapping.attn.has_dp,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )

    def get_hot_token_id(self):
        # MTP drafts over the full vocab (EAGLE3-only optimization).
        return None

    def get_embed_and_head(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        captured_hidden_states: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        **kwargs,
    ):
        del positions, kwargs  # rel attention needs no positions; tau is off
        if captured_hidden_states is None:
            if not ctx.forward_mode.is_idle():
                raise ValueError("Inkling NextN requires captured_hidden_states.")
            captured_hidden_states = torch.zeros(
                input_ids.shape[0],
                self.text_config.hidden_size,
                device=input_ids.device,
                dtype=self.model.embed_tokens.weight.dtype,
            )
        hidden_states = self.model(
            input_ids,
            ctx,
            out_cache_loc,
            captured_hidden_states,
            input_embeds=input_embeds,
            spec_step_idx=spec_step_idx,
        )
        # Base-model muP convention: lm_head consumes hidden/mup; next depth's RMSNorm is invariant to it.
        return self._compute_logits(input_ids, hidden_states, ctx)

    _compute_logits = InklingForConditionalGeneration._compute_logits

    # Base impl works verbatim: the depth-specialized text config's
    # local_layer_ids resolve each depth's ckpt/served head counts.
    _replicate_kv_heads = InklingForConditionalGeneration._replicate_kv_heads

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``model.mtp.*`` tensors (all BF16).

        Remap ``model.mtp.layers.{i}.`` -> ``model.layers.{i}.`` and
        ``model.mtp.chain_norm.`` -> ``model.chain_norm.``, then apply the
        same block transforms as the base loader: qkvr fusion, KV-head
        replication to the uniform served count, w13 gate/up de-interleave.
        Embedding and lm_head are shared from the target (not in the file).
        """
        cfg = self.text_config
        interleaved = cfg.inference_moe_w13_interleaved
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        dropped: list[str] = []
        load_param = _make_param_loader(params_dict, loaded, dropped)

        for name, w in weights:
            name = name.removeprefix("model.")
            if name == "llm.embed_norm.weight":
                # The base embedding norm, applied before each depth's own
                # embed_norm.
                load_param("model.base_embed_norm.weight", w)
                continue
            if not name.startswith("mtp."):
                continue  # base-model tensor; the target worker owns it
            name = "model." + name.removeprefix("mtp.")

            if name.startswith("model.layers."):
                depth = int(name.split(".")[2])
                if depth >= self.model.num_mtp_layers:
                    continue  # depth pruned by --speculative-num-steps

            if name.endswith(_KV_REPLICATED_SUFFIXES):
                w = self._replicate_kv_heads(name, w)

            if _load_block_param(load_param, name, w, interleaved):
                continue

            load_param(name, w)

        missing = [
            idx
            for idx in range(self.model.num_mtp_layers)
            if not any(n.startswith(f"model.layers.{idx}.") for n in loaded)
        ]
        if missing:
            raise ValueError(
                f"Inkling MTP weights missing for depth layer(s) {missing}. "
                "Use a checkpoint that ships `model.mtp.*` weights or disable "
                "MTP speculative decoding."
            )
        if dropped:
            logger.warning(
                f"Inkling NextN load_weights dropped {len(dropped)} tensors (first: {dropped[:8]})"
            )
        return loaded


EntryClass = [InklingForConditionalGenerationNextN]
