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

"""Inference-only dense Llama model compatible with HuggingFace weights.

Covers Llama-2 / Llama-3 / Llama-3.1 / Llama-3.2 dense checkpoints whose
``config.architectures`` is ``["LlamaForCausalLM"]``. MoE and Eagle3 draft
variants have their own modules (``longcat_large.py``, ``llama_eagle3.py``).
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import LlamaConfig

from tokenspeed.runtime.configs.utils import get_rope_theta
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.linear import QKVParallelLinear, RowParallelLinear
from tokenspeed.runtime.layers.mlp import GatedMLP, gated_activation
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.models.base import (
    BaseCausalLM,
    BaseDecoderLayer,
    BaseTransformerModel,
)
from tokenspeed.runtime.models.utils import (
    validate_attention_partition,
)
from tokenspeed.runtime.utils import add_prefix


class LlamaAttention(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        mapping: Mapping,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        qkv_input_size: int | None = None,
        *,
        sliding_window_size: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.attn_tp_size = mapping.attn.tp_size
        self.attn_tp_rank = mapping.attn.tp_rank
        attn_tp_group = mapping.attn.tp_group

        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        validate_attention_partition(
            self.total_num_heads,
            self.total_num_kv_heads,
            self.attn_tp_size,
        )
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Dense Llama is consistently bias-free (`attention_bias=False` in every
        # upstream release). Still read it off the config so forks that flip
        # the flag load without surprises.
        attention_bias = getattr(config, "attention_bias", False)

        self.qkv_proj = QKVParallelLinear(
            qkv_input_size or hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            tp_group=attn_tp_group,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            tp_group=attn_tp_group,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = self.resolve_rotary_emb(config)
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            rotary_emb=self.rotary_emb,
            qk_norm=None,
        )

    def resolve_rotary_emb(self, config: LlamaConfig) -> nn.Module | None:
        """The rotary embedding of q and k, or None for attention without one."""
        return get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            base=get_rope_theta(config),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        # Skip the QKV projection, RoPE, attention, and o_proj kernels when
        # the batch row is empty (e.g. idle ranks under DP attention). Matches
        # the short-circuit ``GatedMLP.forward`` already has.
        if hidden_states.shape[0] == 0:
            return hidden_states.new_zeros(
                (0, self.hidden_size), dtype=hidden_states.dtype
            )
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self._attn(positions, q, k, v, ctx)
        output, _ = self.o_proj(attn_output)
        return output

    def _attn(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """Attention before ``o_proj``; the Eagle3 draft head overrides it."""
        return self.attn(q, k, v, positions, ctx)


class LlamaDecoderLayer(BaseDecoderLayer):

    def resolve_attn(self, prefix: str) -> nn.Module:
        return LlamaAttention(
            config=self.config,
            mapping=self.mapping,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            layer_id=self.layer_id,
            quant_config=self.quant_config,
            prefix=add_prefix("self_attn", prefix),
            sliding_window_size=-1,
        )

    def resolve_mlp(self, prefix: str) -> nn.Module:
        return GatedMLP.tensor_parallel(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            activation=gated_activation(self.config.hidden_act, swiglu_limit=None),
            tp_rank=self.mapping.dense.tp_rank,
            tp_size=self.mapping.dense.tp_size,
            tp_group=self.mapping.dense.tp_group,
            reduce_results=False,
            fusions=(),
            quant_config=self.quant_config,
            prefix=add_prefix("mlp", prefix),
        )


class LlamaModel(BaseTransformerModel):
    layer_cls = LlamaDecoderLayer


class LlamaForCausalLM(BaseCausalLM):
    model_cls = LlamaModel
    supports_aux_capture = True
    supports_acl_graph_discard = True

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_stacked_params_mapping(self) -> list[tuple[str, str, int | str]]:
        return [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]


EntryClass = LlamaForCausalLM
