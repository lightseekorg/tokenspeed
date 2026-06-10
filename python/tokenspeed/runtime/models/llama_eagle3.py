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

"""LLaMA Eagle3 draft model for speculative decoding.

Extends base classes. Preserves the low-latency fused allreduce+norm
path from the original implementation.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from transformers import LlamaConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.common import concat
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import (
    BaseCausalLM,
    BaseDecoderLayer,
    BaseTransformerModel,
)
from tokenspeed.runtime.models.llama import LlamaAttention as BaseLlamaAttention
from tokenspeed.runtime.utils import add_prefix, get_colorful_logger

logger = get_colorful_logger(__name__)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class LlamaAttention(BaseLlamaAttention):
    """Eagle3 draft head attention.

    Inherits ``__init__`` (with ``qkv_input_size=2*hidden_size`` for the
    [embed || hidden] concat) and ``forward`` (= qkv_proj + o_proj scaffolding)
    from base. Overrides ``_attn`` so the draft's first step skips dead
    catch-up rows: on backends that support fused KV pre-write, q is sliced
    to one live row per request and dispatched as DECODE; otherwise the
    fallback runs the full N-row attn and post-slices the output. Inactive
    draft steps delegate to base.
    """

    def _attn(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        # Active draft first step (drafter set up gather_ids + accept_lengths).
        # Covers both decode catch-up and prefill catch-up; multi-step decode
        # delegates to base.
        if ctx.accept_lengths is None:
            return super()._attn(positions, q, k, v, ctx, out_cache_loc)

        if ctx.attn_backend.support_kv_cache_prewrite(ctx.forward_mode):
            fused_kv_arg = self._build_fused_kv_arg(v, ctx, out_cache_loc)
            if fused_kv_arg is not None:
                # Trim only on the sliced single-token decode path; the
                # post-slice fallback below still runs full N-row attn and
                # needs the original seq_lens.
                self._apply_correction(ctx)
                q_rope = self._fused_rope_kv_write(
                    positions, q, k, fused_kv_arg
                ).index_select(0, ctx.gather_ids)
                return ctx.attn_backend.forward(
                    q_rope,
                    None,
                    None,
                    self.attn,
                    out_cache_loc,
                    ctx.token_to_kv_pool,
                    ForwardMode.DECODE,
                    ctx.bs,
                    save_kv_cache=False,
                )
        q, k = self.rotary_emb(positions, q, k)
        return self.attn(q, k, v, ctx=ctx, out_cache_loc=out_cache_loc).index_select(
            0, ctx.gather_ids
        )

    def _apply_correction(self, ctx: ForwardContext) -> None:
        """Trim decode rows' cache_seqlens by ``spec_num_tokens - accept_lengths``."""
        seq_lens_buf = ctx.draft_seq_lens_buf
        if seq_lens_buf is None or ctx.accept_lengths is None:
            return
        num_extends = ctx.num_extends
        if num_extends >= ctx.bs:
            return
        correction = (
            ctx.attn_backend.spec_num_tokens - ctx.accept_lengths[num_extends:]
        ).to(seq_lens_buf.dtype)
        seq_lens_buf[num_extends : ctx.bs].sub_(correction)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:

        super().__init__()
        tp_rank = mapping.dense.tp_rank
        tp_size = mapping.dense.tp_size
        tp_group = mapping.dense.tp_group

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()
        self.gateup_unquanted = quant_config is None

    def forward(self, x, block_scale=None):

        if x.shape[0] == 0:
            return x
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Eagle3DecoderLayer(BaseDecoderLayer):
    """Eagle3 decoder layer with low-latency fused allreduce+norm path.

    Inherits norm/attn/mlp/comm_manager init from BaseDecoderLayer.
    Overrides forward with eagle3-specific embed+hidden concat logic.
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:

        self._eagle3_config = config
        self._eagle3_mapping = mapping
        self._eagle3_quant_config = quant_config
        self._eagle3_prefix = prefix

        super().__init__(
            config=config,
            layer_id=layer_id,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def resolve_attn(self, prefix: str) -> nn.Module:

        config = self._eagle3_config
        return LlamaAttention(
            config,
            self._eagle3_mapping,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=self.layer_id,
            quant_config=self._eagle3_quant_config,
            prefix=add_prefix("self_attn", prefix),
            qkv_input_size=2 * config.hidden_size,
        )

    def resolve_mlp(self, prefix: str) -> nn.Module:

        config = self._eagle3_config
        inter_size = (
            config.intermediate_size_mlp
            if config.model_type == "llama4_text"
            else config.intermediate_size
        )

        return LlamaMLP(
            config.hidden_size,
            inter_size,
            config.hidden_act,
            self._eagle3_mapping,
            self._eagle3_quant_config,
            prefix=f"{prefix}.mlp",
        )

    def _maybe_narrow_residual(
        self,
        residual: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """Align residual with attn output narrowed to [bs, H]."""
        if ctx.accept_lengths is not None and not ctx.forward_mode.is_idle():
            return residual.index_select(0, ctx.gather_ids)
        return residual

    def forward_low_latency(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        residual: torch.Tensor | None,
        final_norm: RMSNorm = None,
        fuse_embed_reduce: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states

        if fuse_embed_reduce:
            # Fuse embedding allreduce with input_layernorm.
            embeds, _, *_ = self.input_layernorm.forward_with_allreduce_fusion(
                self.mapping.attn.tp_rank,
                self.mapping.attn.tp_group,
                embeds,
                torch.zeros_like(embeds),
            )
        else:
            embeds = self.input_layernorm(embeds)

        hidden_states = self.hidden_norm(hidden_states)
        hidden_states = concat(embeds, hidden_states)

        # Attention
        hidden_states = self.comm_manager.pre_attn_comm(hidden_states, ctx)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
        )
        residual = self._maybe_narrow_residual(residual, ctx)

        # Fused post-attn allreduce + norm (uses attn tp group)
        block_scale = None
        hidden_states, residual, block_scale, *_ = (
            self.post_attention_layernorm.forward_with_allreduce_fusion(
                self.mapping.attn.tp_rank,
                self.mapping.attn.tp_group,
                hidden_states,
                residual,
                fuse_block_quant_fp8=not self.mlp.gateup_unquanted,
            )
        )

        hidden_states = self.mlp(hidden_states, block_scale)

        # Fused final allreduce + norm (uses dense tp group)
        hidden_states, residual, *_ = final_norm.forward_with_allreduce_fusion(
            self.mapping.dense.tp_rank,
            self.mapping.dense.tp_group,
            hidden_states,
            residual,
            fuse_block_quant_fp8=False,
        )

        return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        residual: torch.Tensor | None,
        final_norm: RMSNorm = None,
        fuse_embed_reduce: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.comm_manager.should_fuse(hidden_states.shape[0]):
            return self.forward_low_latency(
                positions,
                embeds,
                hidden_states,
                ctx,
                out_cache_loc,
                residual,
                final_norm,
                fuse_embed_reduce=fuse_embed_reduce,
            )

        # Non-fused path: fuse_embed_reduce is always False here because
        # the model only sets it when should_fuse() is True.
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        # Attention
        hidden_states = self.comm_manager.pre_attn_comm(hidden_states, ctx)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
        )
        residual = self._maybe_narrow_residual(residual, ctx)
        hidden_states, residual = self.comm_manager.post_attn_comm(
            hidden_states, residual, ctx
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MLP
        hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
        hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.comm_manager.post_mlp_comm(
            hidden_states, residual, ctx
        )

        return hidden_states, residual


# ---------------------------------------------------------------------------
# Model and CausalLM
# ---------------------------------------------------------------------------


class Eagle3LlamaModel(BaseTransformerModel):

    layer_cls = Eagle3DecoderLayer

    def __init__(
        self,
        config: LlamaConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:

        super().__init__(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )

        # Eagle3 uses "midlayer" (not "layers.0") in checkpoint weights.
        # Re-register the single layer under the correct name.
        self.midlayer = self.layers[0]
        del self.layers

        self.num_fc_input_dim = (
            len(config.eagle_aux_hidden_state_layer_ids)
            if hasattr(config, "eagle_aux_hidden_state_layer_ids")
            else 3
        )

        self.fc = ColumnParallelLinear(
            config.hidden_size * self.num_fc_input_dim,
            config.hidden_size,
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix("fc", prefix),
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor = None,
        hidden_states: torch.Tensor = None,
    ) -> torch.Tensor:

        if input_embeds is None:
            # When TP > 1 and fused allreduce+norm is available, skip the
            # NCCL allreduce in the embedding and let the midlayer fuse it
            # with the input_layernorm.
            midlayer = self.midlayer
            num_tokens = input_ids.shape[0]
            fuse_embed_reduce = (
                self.mapping.attn.tp_size > 1
                and midlayer.comm_manager.should_fuse(num_tokens)
            )
            embeds = self.embed_tokens(input_ids, reduce_results=not fuse_embed_reduce)
        else:
            embeds = input_embeds
            fuse_embed_reduce = False

        if hidden_states is None:
            raise ValueError("Eagle3 forward requires hidden_states")

        if hidden_states.size(-1) != embeds.size(-1):
            hidden_states, _ = self.fc(hidden_states)

        residual = None
        midlayer = self.midlayer
        hidden_states, residual = midlayer(
            positions,
            embeds,
            hidden_states,
            ctx,
            out_cache_loc,
            residual,
            self.norm,
            fuse_embed_reduce=fuse_embed_reduce,
        )

        # Decide on pre-slice token count so this matches the path midlayer
        # actually took; under draft reduce, hidden_states.shape[0] shrinks.
        if midlayer.comm_manager.should_fuse(input_ids.shape[0]):
            hidden_states_to_logits, hidden_states_to_aux = hidden_states, residual
        else:
            hidden_states_to_logits, hidden_states_to_aux = self.norm(
                hidden_states, residual
            )
            hidden_states_to_logits, _ = midlayer.comm_manager.post_final_norm_comm(
                hidden_states_to_logits, None, ctx
            )
            hidden_states_to_aux, _ = midlayer.comm_manager.post_final_norm_comm(
                hidden_states_to_aux, None, ctx
            )

        return hidden_states_to_logits, [hidden_states_to_aux]


class LlamaForCausalLMEagle3(BaseCausalLM):

    model_cls = Eagle3LlamaModel

    def __init__(
        self,
        config: LlamaConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:

        nn.Module.__init__(self)
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config

        if self.config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")

        self.model = self.resolve_model(config, mapping, quant_config, prefix)

        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            if getattr(config, "draft_vocab_size", None) is None:
                self.load_lm_head_from_target = True
            self.lm_head = ParallelLMHead(
                getattr(config, "draft_vocab_size", None) or config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                tp_rank=mapping.attn.tp_rank,
                tp_size=mapping.attn.tp_size,
                tp_group=mapping.attn.tp_group,
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = self.resolve_logits_processor(config)
        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    def prepare_model_kwargs(
        self, ctx: ForwardContext, input_ids: torch.Tensor, kwargs: dict
    ) -> dict:
        model_kwargs = super().prepare_model_kwargs(ctx, input_ids, kwargs)
        captured_hidden_states = kwargs.get("captured_hidden_states")
        if captured_hidden_states is not None:
            model_kwargs["hidden_states"] = captured_hidden_states
        else:
            # During CUDA graph capture warmup, provide dummy hidden states.
            num_tokens = input_ids.shape[0]
            hidden_size = self.config.hidden_size
            num_fc = self.model.num_fc_input_dim
            model_kwargs["hidden_states"] = torch.zeros(
                num_tokens,
                hidden_size * num_fc,
                dtype=torch.bfloat16,
                device=input_ids.device,
            )
        return model_kwargs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:

        params_dict = dict(self.named_parameters())
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            if "d2t" in name:
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue

            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name = f"model.{name}" if name not in params_dict else name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param_name = name if name in params_dict else f"model.{name}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

    def get_hot_token_id(self):
        return self.hot_token_id

    def get_embed(self):
        return self.model.embed_tokens.weight

    def set_embed_and_head(self, embed, head):
        # If draft hidden size != target hidden size, embed cannot be shared
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        if head is not None and self.load_lm_head_from_target:
            del self.lm_head.weight
            self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = [LlamaForCausalLMEagle3]
