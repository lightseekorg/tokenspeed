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

"""MiniMax-M3 DSpark draft model.

``DSparkDraftModel`` with MiniMax-M3 dense layers: Gemma norms, per-head Gemma
QK norms with partial NeoX RoPE, and the SwiGLU-OAI MLP. Target-hidden
projection, KV injection, the block forward and the vanilla Markov head are
inherited. The checkpoint's confidence head is not loaded; the full block is
verified every step.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm
from tokenspeed.runtime.layers.linear import QKVParallelLinear, RowParallelLinear
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.models.dflash import (
    DFlashAttention,
    DFlashDecoderLayer,
    _get_dflash_layer_sliding_window,
)
from tokenspeed.runtime.models.dspark import DSparkDraftModel
from tokenspeed.runtime.models.minimax_m3 import MiniMaxM3MLP
from tokenspeed.runtime.models.utils import validate_attention_partition
from tokenspeed.runtime.utils import add_prefix


class MiniMaxM3DSparkAttention(DFlashAttention):
    """DFlash draft attention with MiniMax-M3's dense-attention modules."""

    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # DFlashAttention.__init__ builds plain RMS QK norms and full RoPE;
        # these modules mirror MiniMaxM3Attention's dense layers instead.
        nn.Module.__init__(self)
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

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
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
            base=int(config.rope_parameters["rope_theta"]),
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=_get_dflash_layer_sliding_window(config, layer_id),
            rotary_emb=self.rotary_emb,
            qk_norm=(self.q_norm, self.k_norm),
        )


class MiniMaxM3DSparkDecoderLayer(DFlashDecoderLayer):
    """DFlash block-draft layer with MiniMax-M3's dense-layer modules."""

    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # DFlashDecoderLayer.forward owns the residual stream and the TP
        # reductions; only the modules differ from its Qwen3-shaped ones.
        nn.Module.__init__(self)
        self.mapping = mapping
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MiniMaxM3DSparkAttention(
            config=config,
            mapping=mapping,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = MiniMaxM3MLP(
            config=config,
            intermediate_size=config.dense_intermediate_size,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )


class DSparkMiniMaxDraftModel(DSparkDraftModel):
    """MiniMax-M3 DSpark checkpoint entry class."""

    decoder_layer_cls = MiniMaxM3DSparkDecoderLayer

    # The checkpoint nests the draft inside the target's module tree; the
    # runtime mounts it at the root.
    checkpoint_name_replacements = (
        ("language_model.", ""),
        ("model.dspark.", ""),
        (".decoder_layer.", "."),
        ("final_norm.", "norm."),
    )

    def __init__(
        self,
        config,
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
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def final_norm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        """The last layer hands back its MLP's partial; reduce it here."""
        if residual is None:
            return self.norm(hidden_states)
        hidden_states = all_reduce(hidden_states, self.mapping.dense.tp_group)
        return self.norm(hidden_states, residual)[0]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        def runtime_named(weights):
            for name, loaded_weight in weights:
                for old, new in self.checkpoint_name_replacements:
                    name = name.replace(old, new)
                yield name, loaded_weight

        super().load_weights(runtime_named(weights))


EntryClass = [DSparkMiniMaxDraftModel]
